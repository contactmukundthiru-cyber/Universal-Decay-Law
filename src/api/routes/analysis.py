"""
Analysis API Routes.

Endpoints for running analyses and retrieving statistical results.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np

from database.connection import get_session
from database.crud import TrialCRUD, DatasetCRUD, UserCRUD


router = APIRouter()


# Pydantic models
class FitModelRequest(BaseModel):
    """Request model for fitting a decay model."""
    time: List[float]
    engagement: List[float]
    model_name: str = "stretched_exponential"


class FitModelResponse(BaseModel):
    """Response model for model fit."""
    model_name: str
    parameters: Dict[str, float]
    aic: float
    bic: float
    r_squared: float
    converged: bool


class CompareModelsRequest(BaseModel):
    """Request for comparing multiple models."""
    time: List[float]
    engagement: List[float]
    models: Optional[List[str]] = None


class CompareModelsResponse(BaseModel):
    """Response for model comparison."""
    best_model: str
    comparisons: List[Dict[str, Any]]


class UniversalityTestRequest(BaseModel):
    """Request for universality test."""
    trial_id: int
    test_type: str = "platform_comparison"  # platform_comparison, k_fold, bootstrap


class PredictionRequest(BaseModel):
    """Request for engagement prediction."""
    time: List[float]
    engagement: List[float]
    prediction_horizon: List[float]
    model_name: str = "stretched_exponential"


class StatisticalTestRequest(BaseModel):
    """Request for statistical tests."""
    trial_id: int
    test_name: str  # normality, homoscedasticity, universality, scaling


# Endpoints
@router.post("/fit", response_model=FitModelResponse)
async def fit_model(data: FitModelRequest):
    """Fit a decay model to provided data."""
    from src.models.base import DecayModelRegistry

    try:
        model = DecayModelRegistry.get(data.model_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    t = np.array(data.time)
    E = np.array(data.engagement)

    if len(t) < 5:
        raise HTTPException(status_code=400, detail="Insufficient data points (min 5)")

    try:
        result = model.fit(t, E)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fitting failed: {str(e)}")

    return FitModelResponse(
        model_name=data.model_name,
        parameters=result.parameters,
        aic=result.aic,
        bic=result.bic,
        r_squared=result.r_squared,
        converged=result.converged
    )


@router.post("/compare", response_model=CompareModelsResponse)
async def compare_models(data: CompareModelsRequest):
    """Compare multiple decay models."""
    from src.models.base import DecayModelRegistry

    t = np.array(data.time)
    E = np.array(data.engagement)

    if len(t) < 5:
        raise HTTPException(status_code=400, detail="Insufficient data points (min 5)")

    comparisons = DecayModelRegistry.compare_all(t, E)

    return CompareModelsResponse(
        best_model=comparisons[0].model_name if comparisons else "",
        comparisons=[
            {
                "model_name": c.model_name,
                "aic": c.fit_result.aic,
                "bic": c.fit_result.bic,
                "delta_aic": c.delta_aic,
                "akaike_weight": c.akaike_weight,
                "ranking": c.ranking,
                "converged": c.fit_result.converged,
                "parameters": c.fit_result.parameters
            }
            for c in comparisons
        ]
    )


@router.get("/models")
async def list_models():
    """List available decay models."""
    from src.models.base import DecayModelRegistry

    models = []
    for name in DecayModelRegistry.list_models():
        try:
            model = DecayModelRegistry.get(name)
            models.append({
                "name": name,
                "description": model.description,
                "parameters": model.parameter_names,
                "n_parameters": model.n_parameters
            })
        except Exception:
            continue

    return {"models": models}


@router.post("/predict")
async def predict_engagement(data: PredictionRequest):
    """Predict future engagement."""
    from src.analysis.validation import Predictor

    t = np.array(data.time)
    E = np.array(data.engagement)
    t_future = np.array(data.prediction_horizon)

    predictor = Predictor(model_name=data.model_name)
    predictions, info = predictor.predict_from_early_data(t, E, t_future)

    return {
        "predictions": predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
        "confidence_intervals": {
            "lower": info.get("ci_lower", []).tolist() if isinstance(info.get("ci_lower"), np.ndarray) else info.get("ci_lower"),
            "upper": info.get("ci_upper", []).tolist() if isinstance(info.get("ci_upper"), np.ndarray) else info.get("ci_upper")
        } if info.get("converged") else None,
        "parameters": info.get("parameters"),
        "converged": info.get("converged", False)
    }


@router.post("/statistical-test")
async def run_statistical_test(
    data: StatisticalTestRequest,
    session: AsyncSession = Depends(get_session)
):
    """Run a statistical test on trial results."""
    from src.analysis.statistics import StatisticalTests

    trial = await TrialCRUD.get(session, data.trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    if trial.results is None:
        raise HTTPException(status_code=400, detail="Trial has no results")

    # Get appropriate data from trial
    # This is a simplified example - real implementation would extract
    # the relevant data from the trial's fit results

    if data.test_name == "normality":
        # Get residuals from trial
        residuals = trial.results.get("residuals", [])
        if not residuals:
            raise HTTPException(status_code=400, detail="No residuals available")

        result = StatisticalTests.test_residual_normality(np.array(residuals))

    elif data.test_name == "universality":
        # Get gamma values by platform
        platform_gammas = trial.results.get("platform_gammas", {})
        if not platform_gammas:
            raise HTTPException(status_code=400, detail="No gamma values available")

        result = StatisticalTests.test_universality(platform_gammas)

    else:
        raise HTTPException(status_code=400, detail=f"Unknown test: {data.test_name}")

    return {
        "test_name": result.test_name,
        "statistic": result.statistic,
        "p_value": result.p_value,
        "null_hypothesis": result.null_hypothesis,
        "alternative_hypothesis": result.alternative_hypothesis,
        "reject_null": result.reject_null,
        "alpha": result.alpha,
        "details": result.details
    }


@router.get("/universality-summary/{trial_id}")
async def get_universality_summary(
    trial_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Get comprehensive universality analysis summary."""
    from database.crud import MasterCurveCRUD, ScalingCRUD, FitResultCRUD

    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    master_curve = await MasterCurveCRUD.get_by_trial(session, trial_id)
    scaling = await ScalingCRUD.get_by_trial(session, trial_id)
    fit_results = await FitResultCRUD.list_by_trial(session, trial_id)

    # Compute summary statistics
    taus = [fr.estimated_tau for fr in fit_results if fr.estimated_tau and fr.estimated_tau > 0]
    deviations = [fr.deviation_score for fr in fit_results if fr.deviation_score is not None]

    return {
        "trial_id": trial_id,
        "n_users": len(fit_results),
        "collapse_quality": master_curve.collapse_quality if master_curve else None,
        "master_curve": {
            "model": master_curve.model_name if master_curve else None,
            "parameters": master_curve.parameters if master_curve else None,
            "residual_std": master_curve.residual_std if master_curve else None
        },
        "scaling_relationship": {
            "tau0": scaling.tau0 if scaling else None,
            "beta": scaling.beta if scaling else None,
            "r_squared": scaling.r_squared if scaling else None
        },
        "tau_distribution": {
            "mean": float(np.mean(taus)) if taus else None,
            "std": float(np.std(taus)) if taus else None,
            "median": float(np.median(taus)) if taus else None,
            "min": float(np.min(taus)) if taus else None,
            "max": float(np.max(taus)) if taus else None
        },
        "deviation_statistics": {
            "mean": float(np.mean(deviations)) if deviations else None,
            "std": float(np.std(deviations)) if deviations else None,
            "n_deviants": sum(1 for d in deviations if d > 2.0)
        },
        "universality_supported": master_curve.collapse_quality > 0.7 if master_curve and master_curve.collapse_quality else None
    }


@router.get("/platform-comparison/{trial_id}")
async def get_platform_comparison(
    trial_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Compare results across platforms."""
    from database.crud import FitResultCRUD

    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    fit_results = await FitResultCRUD.list_by_trial(session, trial_id)

    # Group by platform (would need to join with users table in real implementation)
    # Simplified version using trial results
    platform_stats = trial.results.get("platform_stats", {}) if trial.results else {}

    return {
        "trial_id": trial_id,
        "platforms": platform_stats,
        "overall_comparison": {
            "tau_variance_ratio": None,  # Would compute F-test
            "supports_universality": trial.collapse_quality > 0.7 if trial.collapse_quality else None
        }
    }
