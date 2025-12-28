"""
Trial Management API Routes.

Endpoints for creating, managing, and executing analysis trials.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_session
from database.crud import TrialCRUD, DatasetCRUD, FitResultCRUD, MasterCurveCRUD, ScalingCRUD
from database.models import TrialStatus, DatasetStatus


router = APIRouter()


# Pydantic models
class TrialCreate(BaseModel):
    """Request model for creating a trial."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    dataset_id: Optional[int] = None
    dataset_ids: Optional[List[int]] = None
    config: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "models": ["stretched_exponential", "power_law", "weibull", "double_exponential"],
        "selection_criterion": "aic",
        "engagement_norm": "max",
        "outlier_method": "iqr",
        "cross_validation": True,
        "n_folds": 5
    })


class TrialResponse(BaseModel):
    """Response model for trial."""
    id: int
    name: str
    description: Optional[str]
    status: str
    dataset_id: Optional[int]
    dataset_ids: Optional[List[int]]
    config: Optional[Dict[str, Any]]
    n_users_processed: int
    best_model: Optional[str]
    collapse_quality: Optional[float]
    master_curve_params: Optional[Dict[str, Any]]
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_seconds: Optional[float]
    error_message: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class TrialListResponse(BaseModel):
    """Response model for trial list."""
    trials: List[TrialResponse]
    total: int
    limit: int
    offset: int


class TrialUpdate(BaseModel):
    """Request model for updating a trial."""
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


# Helper function to convert trial to response
def trial_to_response(trial) -> TrialResponse:
    return TrialResponse(
        id=trial.id,
        name=trial.name,
        description=trial.description,
        status=trial.status.value,
        dataset_id=trial.dataset_id,
        dataset_ids=trial.dataset_ids,
        config=trial.config,
        n_users_processed=trial.n_users_processed,
        best_model=trial.best_model,
        collapse_quality=trial.collapse_quality,
        master_curve_params=trial.master_curve_params,
        started_at=trial.started_at.isoformat() if trial.started_at else None,
        completed_at=trial.completed_at.isoformat() if trial.completed_at else None,
        duration_seconds=trial.duration_seconds,
        error_message=trial.error_message,
        created_at=trial.created_at.isoformat()
    )


# Endpoints
@router.post("/", response_model=TrialResponse)
async def create_trial(
    data: TrialCreate,
    session: AsyncSession = Depends(get_session)
):
    """Create a new analysis trial."""
    # Validate dataset(s)
    if data.dataset_id:
        dataset = await DatasetCRUD.get(session, data.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        if dataset.status != DatasetStatus.READY:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset is not ready. Current status: {dataset.status.value}"
            )

    trial = await TrialCRUD.create(
        session,
        name=data.name,
        description=data.description,
        dataset_id=data.dataset_id,
        dataset_ids=data.dataset_ids,
        config=data.config
    )

    return trial_to_response(trial)


@router.get("/", response_model=TrialListResponse)
async def list_trials(
    status: Optional[str] = Query(None),
    dataset_id: Optional[int] = Query(None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session)
):
    """List trials with optional filters."""
    status_enum = TrialStatus(status) if status else None

    trials = await TrialCRUD.list(
        session,
        status=status_enum,
        dataset_id=dataset_id,
        limit=limit,
        offset=offset
    )

    return TrialListResponse(
        trials=[trial_to_response(t) for t in trials],
        total=len(trials),
        limit=limit,
        offset=offset
    )


@router.get("/{trial_id}", response_model=TrialResponse)
async def get_trial(
    trial_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Get trial by ID."""
    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    return trial_to_response(trial)


@router.patch("/{trial_id}", response_model=TrialResponse)
async def update_trial(
    trial_id: int,
    data: TrialUpdate,
    session: AsyncSession = Depends(get_session)
):
    """Update trial configuration."""
    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    if trial.status not in [TrialStatus.CREATED, TrialStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail="Can only update trials that haven't started or have failed"
        )

    update_data = data.dict(exclude_unset=True)
    if update_data:
        trial = await TrialCRUD.update(session, trial_id, **update_data)

    return trial_to_response(trial)


@router.delete("/{trial_id}")
async def delete_trial(
    trial_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Delete trial and all related results."""
    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    await TrialCRUD.delete(session, trial_id)
    return {"message": "Trial deleted successfully"}


@router.post("/{trial_id}/run")
async def run_trial(
    trial_id: int,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session)
):
    """Start running a trial."""
    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    if trial.status not in [TrialStatus.CREATED, TrialStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot run trial. Current status: {trial.status.value}"
        )

    await TrialCRUD.start(session, trial_id)

    # Add background task
    background_tasks.add_task(_run_trial, trial_id)

    return {"message": "Trial started", "status": "running"}


@router.post("/{trial_id}/cancel")
async def cancel_trial(
    trial_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Cancel a running trial."""
    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    if trial.status != TrialStatus.RUNNING:
        raise HTTPException(
            status_code=400,
            detail="Can only cancel running trials"
        )

    await TrialCRUD.update(session, trial_id, status=TrialStatus.CANCELLED)
    return {"message": "Trial cancelled"}


@router.get("/{trial_id}/results")
async def get_trial_results(
    trial_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Get detailed results for a completed trial."""
    trial = await TrialCRUD.get_with_results(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    if trial.status != TrialStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Trial not completed. Current status: {trial.status.value}"
        )

    # Get master curve
    master_curve = await MasterCurveCRUD.get_by_trial(session, trial_id)

    # Get scaling relationship
    scaling = await ScalingCRUD.get_by_trial(session, trial_id)

    return {
        "trial": trial_to_response(trial),
        "results": trial.results,
        "validation": trial.validation_results,
        "statistics": trial.statistics,
        "master_curve": {
            "model_name": master_curve.model_name if master_curve else None,
            "parameters": master_curve.parameters if master_curve else None,
            "collapse_quality": master_curve.collapse_quality if master_curve else None,
            "confidence_intervals": master_curve.confidence_intervals if master_curve else None
        } if master_curve else None,
        "scaling_relationship": {
            "tau0": scaling.tau0 if scaling else None,
            "beta": scaling.beta if scaling else None,
            "r_squared": scaling.r_squared if scaling else None,
            "p_value": scaling.p_value if scaling else None
        } if scaling else None
    }


@router.get("/{trial_id}/fit-results")
async def get_fit_results(
    trial_id: int,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session)
):
    """Get individual user fit results for a trial."""
    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    fit_results = await FitResultCRUD.list_by_trial(session, trial_id, limit, offset)

    return {
        "fit_results": [
            {
                "id": fr.id,
                "user_id": fr.user_id,
                "best_model": fr.best_model,
                "estimated_tau": fr.estimated_tau,
                "estimated_alpha": fr.estimated_alpha,
                "r_squared": fr.r_squared,
                "deviation_score": fr.deviation_score,
                "is_deviant": fr.is_deviant
            }
            for fr in fit_results
        ],
        "limit": limit,
        "offset": offset
    }


@router.get("/{trial_id}/deviants")
async def get_deviants(
    trial_id: int,
    threshold: float = Query(default=2.0, ge=0),
    session: AsyncSession = Depends(get_session)
):
    """Get users that deviate from universal behavior."""
    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    deviants = await FitResultCRUD.get_deviants(session, trial_id, threshold)

    return {
        "deviants": [
            {
                "id": d.id,
                "user_id": d.user_id,
                "deviation_score": d.deviation_score,
                "best_model": d.best_model,
                "estimated_tau": d.estimated_tau
            }
            for d in deviants
        ],
        "threshold": threshold,
        "count": len(deviants)
    }


async def _run_trial(trial_id: int):
    """Background task to run the analysis trial."""
    from database.connection import get_session_context

    async with get_session_context() as session:
        try:
            trial = await TrialCRUD.get(session, trial_id)
            if not trial:
                return

            config = trial.config or {}

            # Load dataset(s)
            from src.data.loader import UnifiedDataset
            from database.crud import UserCRUD, DatasetCRUD

            dataset = UnifiedDataset(f"trial_{trial_id}")

            if trial.dataset_id:
                db_dataset = await DatasetCRUD.get(session, trial.dataset_id)
                users = await UserCRUD.list_by_dataset(session, trial.dataset_id, limit=10000)

                from src.data.base import UserEngagementData, Activity, EngagementType
                from src.data.loader import ProcessedUser
                import numpy as np

                processed_users = []
                for u in users:
                    if u.time_array and u.engagement_array:
                        processed_users.append(ProcessedUser(
                            user_id=u.external_id,
                            platform=db_dataset.platform,
                            time=np.array(u.time_array),
                            engagement=np.array(u.engagement_array),
                            external_mask=np.array(u.external_trigger_array or [False] * len(u.time_array)),
                            metadata=u.metadata or {}
                        ))

                dataset._users = {f"{db_dataset.platform}_{u.user_id}": u for u in processed_users}
                dataset._platforms = {db_dataset.platform: list(dataset._users.keys())}

            # Run analysis pipeline
            from src.analysis.fitting import FittingPipeline, PipelineConfig
            from src.analysis.universality import UniversalityAnalyzer

            pipeline_config = PipelineConfig(
                models=config.get("models"),
                selection_criterion=config.get("selection_criterion", "aic"),
                engagement_norm=config.get("engagement_norm", "max"),
                outlier_method=config.get("outlier_method", "iqr")
            )

            pipeline = FittingPipeline(pipeline_config)

            # Run fitting
            fit_results = pipeline.run(dataset)

            # Universality analysis
            analyzer = UniversalityAnalyzer()
            collapse = analyzer.analyze(fit_results)
            master_curve = analyzer.fit_master_curve(collapse)
            quality = analyzer.compute_collapse_quality(collapse, master_curve)

            # Store results
            summary = pipeline.summarize(fit_results)

            # Store individual fit results
            fit_data = []
            for i, fr in enumerate(fit_results):
                deviation = 0.0
                if i < len(collapse.rescaled_times):
                    y_pred = master_curve.evaluate(collapse.rescaled_times[i])
                    residuals = collapse.rescaled_engagements[i] - y_pred
                    deviation = float(np.sqrt(np.mean(residuals**2)) / collapse.residual_std) if collapse.residual_std > 0 else 0

                fit_data.append({
                    "trial_id": trial_id,
                    "user_id": (await session.execute(
                        select(User.id).where(User.external_id == fr.user_id)
                    )).scalar() or 0,
                    "best_model": fr.best_model,
                    "estimated_tau": fr.estimated_tau,
                    "model_results": {
                        name: {
                            "parameters": result.parameters,
                            "aic": result.aic,
                            "bic": result.bic,
                            "r_squared": result.r_squared
                        }
                        for name, result in fr.fit_results.items()
                        if result.converged
                    },
                    "deviation_score": deviation,
                    "is_deviant": deviation > 2.0
                })

            from sqlalchemy import select
            from database.models import User
            await FitResultCRUD.bulk_create(session, fit_data)

            # Store master curve
            await MasterCurveCRUD.create(
                session, trial_id,
                model_name=master_curve.model_name,
                parameters=master_curve.parameters,
                collapse_quality=quality,
                ks_statistic=collapse.ks_statistic,
                residual_std=collapse.residual_std
            )

            # Store scaling relationship
            if summary.get("tau_mean"):
                await ScalingCRUD.create(
                    session, trial_id,
                    tau0=summary["tau_mean"],
                    beta=0.5,  # Would be fitted in full implementation
                    r_squared=summary.get("tau_std", 0) / summary["tau_mean"] if summary["tau_mean"] else 0
                )

            # Complete trial
            await TrialCRUD.complete(
                session, trial_id,
                results=summary,
                best_model=summary.get("model_selection", {}).get(
                    max(summary.get("model_selection", {"": 0}), key=summary.get("model_selection", {}).get)
                ),
                collapse_quality=quality,
                master_curve_params=master_curve.parameters,
                n_users_processed=len(fit_results)
            )

        except Exception as e:
            import traceback
            await TrialCRUD.fail(session, trial_id, str(e) + "\n" + traceback.format_exc())
