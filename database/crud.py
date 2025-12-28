"""
CRUD Operations for Database Models.

Provides create, read, update, delete operations for all entities.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from database.models import (
    Dataset, User, Trial, UserFitResult, MasterCurve, ScalingRelationship,
    DatasetStatus, TrialStatus
)


class DatasetCRUD:
    """CRUD operations for Dataset model."""

    @staticmethod
    async def create(
        session: AsyncSession,
        name: str,
        platform: str,
        description: Optional[str] = None,
        target_users: int = 100,
        config: Optional[Dict] = None
    ) -> Dataset:
        """Create a new dataset."""
        dataset = Dataset(
            name=name,
            platform=platform,
            description=description,
            target_users=target_users,
            config=config or {}
        )
        session.add(dataset)
        await session.flush()
        return dataset

    @staticmethod
    async def get(session: AsyncSession, dataset_id: int) -> Optional[Dataset]:
        """Get dataset by ID."""
        result = await session.execute(
            select(Dataset).where(Dataset.id == dataset_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_name(session: AsyncSession, name: str) -> Optional[Dataset]:
        """Get dataset by name."""
        result = await session.execute(
            select(Dataset).where(Dataset.name == name)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list(
        session: AsyncSession,
        platform: Optional[str] = None,
        status: Optional[DatasetStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Sequence[Dataset]:
        """List datasets with optional filters."""
        query = select(Dataset)

        if platform:
            query = query.where(Dataset.platform == platform)
        if status:
            query = query.where(Dataset.status == status)

        query = query.order_by(Dataset.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def update(
        session: AsyncSession,
        dataset_id: int,
        **kwargs
    ) -> Optional[Dataset]:
        """Update dataset fields."""
        await session.execute(
            update(Dataset)
            .where(Dataset.id == dataset_id)
            .values(**kwargs, updated_at=datetime.utcnow())
        )
        return await DatasetCRUD.get(session, dataset_id)

    @staticmethod
    async def update_status(
        session: AsyncSession,
        dataset_id: int,
        status: DatasetStatus
    ) -> Optional[Dataset]:
        """Update dataset status."""
        return await DatasetCRUD.update(session, dataset_id, status=status)

    @staticmethod
    async def update_statistics(
        session: AsyncSession,
        dataset_id: int,
        n_users: int,
        n_activities: int,
        avg_duration_days: Optional[float] = None
    ) -> Optional[Dataset]:
        """Update dataset statistics after processing."""
        return await DatasetCRUD.update(
            session, dataset_id,
            n_users=n_users,
            n_activities=n_activities,
            avg_duration_days=avg_duration_days
        )

    @staticmethod
    async def delete(session: AsyncSession, dataset_id: int) -> bool:
        """Delete dataset and all related data."""
        await session.execute(
            delete(Dataset).where(Dataset.id == dataset_id)
        )
        return True


class UserCRUD:
    """CRUD operations for User model."""

    @staticmethod
    async def create(
        session: AsyncSession,
        dataset_id: int,
        external_id: str,
        adoption_timestamp: float,
        time_array: Optional[List[float]] = None,
        engagement_array: Optional[List[float]] = None,
        n_activities: int = 0,
        metadata: Optional[Dict] = None
    ) -> User:
        """Create a new user."""
        user = User(
            dataset_id=dataset_id,
            external_id=external_id,
            adoption_timestamp=adoption_timestamp,
            time_array=time_array,
            engagement_array=engagement_array,
            n_activities=n_activities,
            metadata=metadata or {}
        )
        session.add(user)
        await session.flush()
        return user

    @staticmethod
    async def bulk_create(
        session: AsyncSession,
        users_data: List[Dict]
    ) -> List[User]:
        """Bulk create users for efficiency."""
        users = [User(**data) for data in users_data]
        session.add_all(users)
        await session.flush()
        return users

    @staticmethod
    async def get(session: AsyncSession, user_id: int) -> Optional[User]:
        """Get user by ID."""
        result = await session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list_by_dataset(
        session: AsyncSession,
        dataset_id: int,
        limit: int = 1000,
        offset: int = 0
    ) -> Sequence[User]:
        """List users for a dataset."""
        result = await session.execute(
            select(User)
            .where(User.dataset_id == dataset_id)
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    @staticmethod
    async def count_by_dataset(session: AsyncSession, dataset_id: int) -> int:
        """Count users in a dataset."""
        result = await session.execute(
            select(func.count(User.id)).where(User.dataset_id == dataset_id)
        )
        return result.scalar() or 0


class TrialCRUD:
    """CRUD operations for Trial model."""

    @staticmethod
    async def create(
        session: AsyncSession,
        name: str,
        dataset_id: Optional[int] = None,
        dataset_ids: Optional[List[int]] = None,
        description: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> Trial:
        """Create a new trial."""
        trial = Trial(
            name=name,
            dataset_id=dataset_id,
            dataset_ids=dataset_ids,
            description=description,
            config=config or {},
            status=TrialStatus.CREATED
        )
        session.add(trial)
        await session.flush()
        return trial

    @staticmethod
    async def get(session: AsyncSession, trial_id: int) -> Optional[Trial]:
        """Get trial by ID."""
        result = await session.execute(
            select(Trial).where(Trial.id == trial_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def get_with_results(
        session: AsyncSession,
        trial_id: int
    ) -> Optional[Trial]:
        """Get trial with all related results."""
        result = await session.execute(
            select(Trial)
            .options(selectinload(Trial.fit_results))
            .where(Trial.id == trial_id)
        )
        return result.scalar_one_or_none()

    @staticmethod
    async def list(
        session: AsyncSession,
        status: Optional[TrialStatus] = None,
        dataset_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Sequence[Trial]:
        """List trials with optional filters."""
        query = select(Trial)

        if status:
            query = query.where(Trial.status == status)
        if dataset_id:
            query = query.where(Trial.dataset_id == dataset_id)

        query = query.order_by(Trial.created_at.desc()).limit(limit).offset(offset)
        result = await session.execute(query)
        return result.scalars().all()

    @staticmethod
    async def update(
        session: AsyncSession,
        trial_id: int,
        **kwargs
    ) -> Optional[Trial]:
        """Update trial fields."""
        await session.execute(
            update(Trial)
            .where(Trial.id == trial_id)
            .values(**kwargs, updated_at=datetime.utcnow())
        )
        return await TrialCRUD.get(session, trial_id)

    @staticmethod
    async def start(session: AsyncSession, trial_id: int) -> Optional[Trial]:
        """Mark trial as started."""
        return await TrialCRUD.update(
            session, trial_id,
            status=TrialStatus.RUNNING,
            started_at=datetime.utcnow()
        )

    @staticmethod
    async def complete(
        session: AsyncSession,
        trial_id: int,
        results: Dict,
        best_model: Optional[str] = None,
        collapse_quality: Optional[float] = None,
        master_curve_params: Optional[Dict] = None,
        n_users_processed: int = 0
    ) -> Optional[Trial]:
        """Mark trial as completed with results."""
        started_at = (await TrialCRUD.get(session, trial_id)).started_at
        duration = (datetime.utcnow() - started_at).total_seconds() if started_at else None

        return await TrialCRUD.update(
            session, trial_id,
            status=TrialStatus.COMPLETED,
            completed_at=datetime.utcnow(),
            duration_seconds=duration,
            results=results,
            best_model=best_model,
            collapse_quality=collapse_quality,
            master_curve_params=master_curve_params,
            n_users_processed=n_users_processed
        )

    @staticmethod
    async def fail(
        session: AsyncSession,
        trial_id: int,
        error_message: str
    ) -> Optional[Trial]:
        """Mark trial as failed."""
        return await TrialCRUD.update(
            session, trial_id,
            status=TrialStatus.FAILED,
            completed_at=datetime.utcnow(),
            error_message=error_message
        )

    @staticmethod
    async def delete(session: AsyncSession, trial_id: int) -> bool:
        """Delete trial and all related data."""
        await session.execute(
            delete(Trial).where(Trial.id == trial_id)
        )
        return True


class FitResultCRUD:
    """CRUD operations for UserFitResult model."""

    @staticmethod
    async def bulk_create(
        session: AsyncSession,
        results_data: List[Dict]
    ) -> List[UserFitResult]:
        """Bulk create fit results."""
        results = [UserFitResult(**data) for data in results_data]
        session.add_all(results)
        await session.flush()
        return results

    @staticmethod
    async def list_by_trial(
        session: AsyncSession,
        trial_id: int,
        limit: int = 1000,
        offset: int = 0
    ) -> Sequence[UserFitResult]:
        """List fit results for a trial."""
        result = await session.execute(
            select(UserFitResult)
            .where(UserFitResult.trial_id == trial_id)
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    @staticmethod
    async def get_deviants(
        session: AsyncSession,
        trial_id: int,
        threshold: float = 2.0
    ) -> Sequence[UserFitResult]:
        """Get users that deviate from universal behavior."""
        result = await session.execute(
            select(UserFitResult)
            .where(UserFitResult.trial_id == trial_id)
            .where(UserFitResult.deviation_score > threshold)
            .order_by(UserFitResult.deviation_score.desc())
        )
        return result.scalars().all()


class MasterCurveCRUD:
    """CRUD operations for MasterCurve model."""

    @staticmethod
    async def create(
        session: AsyncSession,
        trial_id: int,
        model_name: str,
        parameters: Dict,
        collapse_quality: Optional[float] = None,
        **kwargs
    ) -> MasterCurve:
        """Create master curve record."""
        curve = MasterCurve(
            trial_id=trial_id,
            model_name=model_name,
            parameters=parameters,
            collapse_quality=collapse_quality,
            **kwargs
        )
        session.add(curve)
        await session.flush()
        return curve

    @staticmethod
    async def get_by_trial(
        session: AsyncSession,
        trial_id: int
    ) -> Optional[MasterCurve]:
        """Get master curve for a trial."""
        result = await session.execute(
            select(MasterCurve).where(MasterCurve.trial_id == trial_id)
        )
        return result.scalar_one_or_none()


class ScalingCRUD:
    """CRUD operations for ScalingRelationship model."""

    @staticmethod
    async def create(
        session: AsyncSession,
        trial_id: int,
        tau0: float,
        beta: float,
        **kwargs
    ) -> ScalingRelationship:
        """Create scaling relationship record."""
        scaling = ScalingRelationship(
            trial_id=trial_id,
            tau0=tau0,
            beta=beta,
            **kwargs
        )
        session.add(scaling)
        await session.flush()
        return scaling

    @staticmethod
    async def get_by_trial(
        session: AsyncSession,
        trial_id: int
    ) -> Optional[ScalingRelationship]:
        """Get scaling relationship for a trial."""
        result = await session.execute(
            select(ScalingRelationship).where(ScalingRelationship.trial_id == trial_id)
        )
        return result.scalar_one_or_none()
