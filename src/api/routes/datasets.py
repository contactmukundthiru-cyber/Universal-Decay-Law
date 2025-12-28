"""
Dataset Management API Routes.

Endpoints for creating, managing, and querying datasets.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_session
from database.crud import DatasetCRUD, UserCRUD
from database.models import DatasetStatus


router = APIRouter()


# Pydantic models for request/response
class DatasetCreate(BaseModel):
    """Request model for creating a dataset."""
    name: str = Field(..., min_length=1, max_length=255)
    platform: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    target_users: int = Field(default=100, ge=1, le=100000)
    min_activities: int = Field(default=5, ge=1)
    config: Optional[Dict[str, Any]] = None


class DatasetResponse(BaseModel):
    """Response model for dataset."""
    id: int
    name: str
    platform: str
    description: Optional[str]
    status: str
    target_users: int
    n_users: int
    n_activities: int
    avg_duration_days: Optional[float]
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class DatasetListResponse(BaseModel):
    """Response model for dataset list."""
    datasets: List[DatasetResponse]
    total: int
    limit: int
    offset: int


class DatasetUpdate(BaseModel):
    """Request model for updating a dataset."""
    name: Optional[str] = None
    description: Optional[str] = None
    target_users: Optional[int] = None
    config: Optional[Dict[str, Any]] = None


class DataCollectionRequest(BaseModel):
    """Request model for triggering data collection."""
    api_credentials: Optional[Dict[str, str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# Endpoints
@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    data: DatasetCreate,
    session: AsyncSession = Depends(get_session)
):
    """Create a new dataset."""
    # Check if name already exists
    existing = await DatasetCRUD.get_by_name(session, data.name)
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Dataset with name '{data.name}' already exists"
        )

    dataset = await DatasetCRUD.create(
        session,
        name=data.name,
        platform=data.platform,
        description=data.description,
        target_users=data.target_users,
        config=data.config
    )

    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        platform=dataset.platform,
        description=dataset.description,
        status=dataset.status.value,
        target_users=dataset.target_users,
        n_users=dataset.n_users,
        n_activities=dataset.n_activities,
        avg_duration_days=dataset.avg_duration_days,
        created_at=dataset.created_at.isoformat(),
        updated_at=dataset.updated_at.isoformat()
    )


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    platform: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session)
):
    """List datasets with optional filters."""
    status_enum = DatasetStatus(status) if status else None

    datasets = await DatasetCRUD.list(
        session,
        platform=platform,
        status=status_enum,
        limit=limit,
        offset=offset
    )

    return DatasetListResponse(
        datasets=[
            DatasetResponse(
                id=d.id,
                name=d.name,
                platform=d.platform,
                description=d.description,
                status=d.status.value,
                target_users=d.target_users,
                n_users=d.n_users,
                n_activities=d.n_activities,
                avg_duration_days=d.avg_duration_days,
                created_at=d.created_at.isoformat(),
                updated_at=d.updated_at.isoformat()
            )
            for d in datasets
        ],
        total=len(datasets),
        limit=limit,
        offset=offset
    )


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Get dataset by ID."""
    dataset = await DatasetCRUD.get(session, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        platform=dataset.platform,
        description=dataset.description,
        status=dataset.status.value,
        target_users=dataset.target_users,
        n_users=dataset.n_users,
        n_activities=dataset.n_activities,
        avg_duration_days=dataset.avg_duration_days,
        created_at=dataset.created_at.isoformat(),
        updated_at=dataset.updated_at.isoformat()
    )


@router.patch("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: int,
    data: DatasetUpdate,
    session: AsyncSession = Depends(get_session)
):
    """Update dataset fields."""
    dataset = await DatasetCRUD.get(session, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    update_data = data.dict(exclude_unset=True)
    if update_data:
        dataset = await DatasetCRUD.update(session, dataset_id, **update_data)

    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        platform=dataset.platform,
        description=dataset.description,
        status=dataset.status.value,
        target_users=dataset.target_users,
        n_users=dataset.n_users,
        n_activities=dataset.n_activities,
        avg_duration_days=dataset.avg_duration_days,
        created_at=dataset.created_at.isoformat(),
        updated_at=dataset.updated_at.isoformat()
    )


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    session: AsyncSession = Depends(get_session)
):
    """Delete dataset and all related data."""
    dataset = await DatasetCRUD.get(session, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    await DatasetCRUD.delete(session, dataset_id)
    return {"message": "Dataset deleted successfully"}


@router.post("/{dataset_id}/collect")
async def start_collection(
    dataset_id: int,
    data: DataCollectionRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session)
):
    """Start data collection for a dataset."""
    dataset = await DatasetCRUD.get(session, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.status not in [DatasetStatus.PENDING, DatasetStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start collection. Current status: {dataset.status.value}"
        )

    await DatasetCRUD.update_status(session, dataset_id, DatasetStatus.COLLECTING)

    # Add background task for actual collection
    background_tasks.add_task(
        _run_collection,
        dataset_id,
        dataset.platform,
        dataset.target_users,
        data.api_credentials
    )

    return {"message": "Data collection started", "status": "collecting"}


@router.get("/{dataset_id}/users")
async def list_dataset_users(
    dataset_id: int,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session)
):
    """List users in a dataset."""
    dataset = await DatasetCRUD.get(session, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    users = await UserCRUD.list_by_dataset(session, dataset_id, limit, offset)

    return {
        "users": [
            {
                "id": u.id,
                "external_id": u.external_id,
                "n_activities": u.n_activities,
                "duration_days": u.duration_days,
            }
            for u in users
        ],
        "total": await UserCRUD.count_by_dataset(session, dataset_id),
        "limit": limit,
        "offset": offset
    }


async def _run_collection(
    dataset_id: int,
    platform: str,
    target_users: int,
    credentials: Optional[Dict[str, str]]
):
    """Background task for data collection."""
    from database.connection import get_session_context

    async with get_session_context() as session:
        try:
            # Get appropriate connector
            connector = _get_connector(platform)
            if not connector:
                await DatasetCRUD.update_status(session, dataset_id, DatasetStatus.FAILED)
                return

            # Authenticate
            if credentials:
                connector.authenticate(**credentials)
            else:
                connector.authenticate()

            # Collect data
            users_data, metadata = connector.collect_dataset(target_users)

            # Store in database
            import numpy as np
            db_users = []
            total_activities = 0

            for user in users_data:
                time_arr, eng_arr = user.to_timeseries()
                db_users.append({
                    "dataset_id": dataset_id,
                    "external_id": user.user_id,
                    "adoption_timestamp": user.adoption_timestamp,
                    "time_array": time_arr.tolist(),
                    "engagement_array": eng_arr.tolist(),
                    "n_activities": user.total_activities,
                    "duration_days": user.duration_days,
                    "metadata": user.metadata
                })
                total_activities += user.total_activities

            await UserCRUD.bulk_create(session, db_users)

            durations = [u["duration_days"] for u in db_users if u["duration_days"]]
            await DatasetCRUD.update_statistics(
                session, dataset_id,
                n_users=len(db_users),
                n_activities=total_activities,
                avg_duration_days=np.mean(durations) if durations else None
            )
            await DatasetCRUD.update_status(session, dataset_id, DatasetStatus.READY)

        except Exception as e:
            await DatasetCRUD.update_status(session, dataset_id, DatasetStatus.FAILED)
            print(f"Collection failed: {e}")


def _get_connector(platform: str):
    """Get appropriate data connector for platform."""
    from src.data import (
        RedditConnector, GitHubConnector, WikipediaConnector,
        StravaConnector, LastFMConnector
    )

    connectors = {
        "reddit": RedditConnector,
        "github": GitHubConnector,
        "wikipedia": WikipediaConnector,
        "strava": StravaConnector,
        "lastfm": LastFMConnector,
    }

    connector_class = connectors.get(platform.lower())
    return connector_class() if connector_class else None
