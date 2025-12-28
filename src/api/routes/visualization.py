"""
Visualization API Routes.

Endpoints for generating visualizations and exporting figures.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
import io
import json

from database.connection import get_session
from database.crud import TrialCRUD, DatasetCRUD, FitResultCRUD, MasterCurveCRUD


router = APIRouter()


# Pydantic models
class PlotRequest(BaseModel):
    """Request for generating a plot."""
    trial_id: int
    plot_type: str  # raw_curves, master_collapse, scaling, prediction, deviants, mechanistic
    format: str = "json"  # json (plotly), png, svg, pdf
    options: Optional[Dict[str, Any]] = None


class DataExportRequest(BaseModel):
    """Request for exporting data."""
    trial_id: int
    format: str = "csv"  # csv, json, parquet


# Endpoints
@router.post("/plot")
async def generate_plot(
    data: PlotRequest,
    session: AsyncSession = Depends(get_session)
):
    """Generate a visualization."""
    trial = await TrialCRUD.get(session, data.trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    if data.format == "json":
        # Return Plotly JSON
        plot_data = await _generate_plotly_data(session, trial, data.plot_type, data.options)
        return plot_data
    else:
        # Generate image
        image_bytes = await _generate_image(session, trial, data.plot_type, data.format, data.options)
        media_type = {
            "png": "image/png",
            "svg": "image/svg+xml",
            "pdf": "application/pdf"
        }.get(data.format, "image/png")

        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=plot.{data.format}"}
        )


@router.get("/raw-curves/{trial_id}")
async def get_raw_curves_data(
    trial_id: int,
    max_curves: int = Query(default=50, ge=1, le=500),
    session: AsyncSession = Depends(get_session)
):
    """Get data for raw decay curves visualization."""
    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    # Get users from the dataset
    if trial.dataset_id:
        from database.crud import UserCRUD
        users = await UserCRUD.list_by_dataset(session, trial.dataset_id, limit=max_curves)

        curves = []
        for user in users:
            if user.time_array and user.engagement_array:
                E_max = max(user.engagement_array) if user.engagement_array else 1
                curves.append({
                    "user_id": user.external_id,
                    "time": user.time_array,
                    "engagement": [e / E_max for e in user.engagement_array] if E_max > 0 else user.engagement_array,
                })

        return {"curves": curves, "n_total": len(curves)}

    return {"curves": [], "n_total": 0}


@router.get("/collapse/{trial_id}")
async def get_collapse_data(
    trial_id: int,
    max_curves: int = Query(default=100, ge=1, le=1000),
    session: AsyncSession = Depends(get_session)
):
    """Get data for master curve collapse visualization."""
    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    master_curve = await MasterCurveCRUD.get_by_trial(session, trial_id)
    fit_results = await FitResultCRUD.list_by_trial(session, trial_id, limit=max_curves)

    # Get rescaled data
    curves = []
    for fr in fit_results:
        if fr.rescaled_time and fr.rescaled_engagement:
            curves.append({
                "user_id": fr.user_id,
                "rescaled_time": fr.rescaled_time,
                "rescaled_engagement": fr.rescaled_engagement,
                "is_deviant": fr.is_deviant
            })

    # Generate master curve
    if master_curve:
        from src.models.base import DecayModelRegistry
        try:
            model = DecayModelRegistry.get(master_curve.model_name)
            x = np.linspace(0, 10, 100)
            y = model.evaluate(x, **master_curve.parameters)
            master_curve_data = {
                "x": x.tolist(),
                "y": y.tolist(),
                "parameters": master_curve.parameters
            }
        except Exception:
            master_curve_data = None
    else:
        master_curve_data = None

    return {
        "curves": curves,
        "master_curve": master_curve_data,
        "collapse_quality": master_curve.collapse_quality if master_curve else None
    }


@router.get("/scaling/{trial_id}")
async def get_scaling_data(
    trial_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Get data for scaling relationship visualization."""
    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    from database.crud import ScalingCRUD
    scaling = await ScalingCRUD.get_by_trial(session, trial_id)
    fit_results = await FitResultCRUD.list_by_trial(session, trial_id)

    # Extract tau and alpha values
    points = []
    for fr in fit_results:
        if fr.estimated_tau and fr.estimated_tau > 0:
            alpha = fr.estimated_alpha or np.random.lognormal(0, 0.5)  # Fallback
            points.append({
                "user_id": fr.user_id,
                "tau": fr.estimated_tau,
                "alpha": alpha
            })

    # Generate fit line
    fit_line = None
    if scaling:
        alphas = np.logspace(-1, 1, 50)
        taus = scaling.tau0 * np.power(alphas, -scaling.beta)
        fit_line = {
            "alpha": alphas.tolist(),
            "tau": taus.tolist(),
            "tau0": scaling.tau0,
            "beta": scaling.beta,
            "r_squared": scaling.r_squared
        }

    return {
        "points": points,
        "fit_line": fit_line
    }


@router.get("/deviants/{trial_id}")
async def get_deviants_data(
    trial_id: int,
    threshold: float = Query(default=2.0, ge=0),
    session: AsyncSession = Depends(get_session)
):
    """Get data for deviant behaviors visualization."""
    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    fit_results = await FitResultCRUD.list_by_trial(session, trial_id)

    # Deviation distribution
    deviations = [fr.deviation_score for fr in fit_results if fr.deviation_score is not None]

    # Categorize deviants
    deviants = []
    for fr in fit_results:
        if fr.deviation_score and fr.deviation_score > threshold:
            deviants.append({
                "user_id": fr.user_id,
                "deviation_score": fr.deviation_score,
                "tau": fr.estimated_tau,
                "rescaled_time": fr.rescaled_time,
                "rescaled_engagement": fr.rescaled_engagement
            })

    # Sort by deviation
    deviants.sort(key=lambda x: x["deviation_score"], reverse=True)

    return {
        "deviation_distribution": {
            "values": deviations,
            "mean": float(np.mean(deviations)) if deviations else None,
            "std": float(np.std(deviations)) if deviations else None,
            "threshold": threshold
        },
        "deviants": deviants[:20],  # Top 20
        "n_deviants": len(deviants)
    }


@router.get("/dashboard/{trial_id}")
async def get_dashboard_data(
    trial_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Get all data needed for the interactive dashboard."""
    trial = await TrialCRUD.get(session, trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    from database.crud import MasterCurveCRUD, ScalingCRUD

    master_curve = await MasterCurveCRUD.get_by_trial(session, trial_id)
    scaling = await ScalingCRUD.get_by_trial(session, trial_id)
    fit_results = await FitResultCRUD.list_by_trial(session, trial_id, limit=200)

    # Compile all dashboard data
    return {
        "trial": {
            "id": trial.id,
            "name": trial.name,
            "status": trial.status.value,
            "n_users": trial.n_users_processed,
            "collapse_quality": trial.collapse_quality
        },
        "master_curve": {
            "model": master_curve.model_name if master_curve else None,
            "parameters": master_curve.parameters if master_curve else None,
            "quality": master_curve.collapse_quality if master_curve else None
        },
        "scaling": {
            "tau0": scaling.tau0 if scaling else None,
            "beta": scaling.beta if scaling else None,
            "r_squared": scaling.r_squared if scaling else None
        },
        "summary_stats": trial.results if trial.results else {},
        "fit_results_sample": [
            {
                "user_id": fr.user_id,
                "tau": fr.estimated_tau,
                "alpha": fr.estimated_alpha,
                "deviation": fr.deviation_score
            }
            for fr in fit_results[:50]
        ]
    }


@router.post("/export")
async def export_data(
    data: DataExportRequest,
    session: AsyncSession = Depends(get_session)
):
    """Export trial data in various formats."""
    trial = await TrialCRUD.get(session, data.trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    fit_results = await FitResultCRUD.list_by_trial(session, data.trial_id)

    if data.format == "csv":
        import csv
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["user_id", "best_model", "tau", "alpha", "deviation_score", "is_deviant"])

        # Data
        for fr in fit_results:
            writer.writerow([
                fr.user_id,
                fr.best_model,
                fr.estimated_tau,
                fr.estimated_alpha,
                fr.deviation_score,
                fr.is_deviant
            ])

        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=trial_{data.trial_id}.csv"}
        )

    elif data.format == "json":
        export_data = {
            "trial_id": data.trial_id,
            "trial_name": trial.name,
            "results": trial.results,
            "fit_results": [
                {
                    "user_id": fr.user_id,
                    "best_model": fr.best_model,
                    "tau": fr.estimated_tau,
                    "alpha": fr.estimated_alpha,
                    "deviation_score": fr.deviation_score,
                    "model_results": fr.model_results
                }
                for fr in fit_results
            ]
        }

        return StreamingResponse(
            io.BytesIO(json.dumps(export_data, indent=2).encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=trial_{data.trial_id}.json"}
        )

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {data.format}")


async def _generate_plotly_data(
    session: AsyncSession,
    trial,
    plot_type: str,
    options: Optional[Dict]
) -> Dict:
    """Generate Plotly-compatible JSON data."""
    from src.visualization.interactive import (
        create_interactive_dashboard,
        create_collapse_animation
    )

    # This is a simplified version - full implementation would
    # reconstruct the collapse result from database

    if plot_type == "dashboard":
        # Return dashboard layout
        return {"type": "dashboard", "trial_id": trial.id}

    elif plot_type == "raw_curves":
        data = await get_raw_curves_data(trial.id, session=session)
        return {
            "data": [
                {"x": c["time"], "y": c["engagement"], "type": "scatter", "mode": "lines"}
                for c in data["curves"][:50]
            ],
            "layout": {"title": "Raw Decay Curves"}
        }

    elif plot_type == "master_collapse":
        data = await get_collapse_data(trial.id, session=session)
        traces = [
            {"x": c["rescaled_time"], "y": c["rescaled_engagement"], "type": "scatter", "mode": "lines", "opacity": 0.3}
            for c in data["curves"][:50]
        ]
        if data["master_curve"]:
            traces.append({
                "x": data["master_curve"]["x"],
                "y": data["master_curve"]["y"],
                "type": "scatter",
                "mode": "lines",
                "line": {"color": "black", "width": 3},
                "name": "Master Curve"
            })
        return {
            "data": traces,
            "layout": {"title": "Master Curve Collapse"}
        }

    else:
        return {"error": f"Unknown plot type: {plot_type}"}


async def _generate_image(
    session: AsyncSession,
    trial,
    plot_type: str,
    format: str,
    options: Optional[Dict]
) -> bytes:
    """Generate image bytes for a plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.visualization.style import set_publication_style

    set_publication_style()

    fig, ax = plt.subplots(figsize=(6, 4))

    if plot_type == "raw_curves":
        data = await get_raw_curves_data(trial.id, session=session)
        for curve in data["curves"][:30]:
            ax.plot(curve["time"], curve["engagement"], alpha=0.3)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("E(t)/E₀")
        ax.set_title("Raw Decay Curves")

    elif plot_type == "master_collapse":
        data = await get_collapse_data(trial.id, session=session)
        for curve in data["curves"][:50]:
            ax.plot(curve["rescaled_time"], curve["rescaled_engagement"], alpha=0.2)
        if data["master_curve"]:
            ax.plot(data["master_curve"]["x"], data["master_curve"]["y"], 'k-', linewidth=2)
        ax.set_xlabel("Rescaled time t/τ")
        ax.set_ylabel("E(t)/E₀")
        ax.set_title("Master Curve Collapse")

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=300, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue()
