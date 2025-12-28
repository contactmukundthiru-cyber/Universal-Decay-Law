"""
Database Models (SQLAlchemy ORM).

Defines the database schema for storing:
    - Datasets and metadata
    - User engagement data
    - Analysis results
    - Trial configurations
"""

from datetime import datetime
from typing import Optional, List, Any
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Index, UniqueConstraint, Enum as SQLEnum
)
from sqlalchemy.orm import relationship, DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import ARRAY
import enum


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class DatasetStatus(enum.Enum):
    """Dataset processing status."""
    PENDING = "pending"
    COLLECTING = "collecting"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class TrialStatus(enum.Enum):
    """Analysis trial status."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Dataset(Base):
    """
    Dataset metadata and configuration.
    """
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    platform: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[DatasetStatus] = mapped_column(
        SQLEnum(DatasetStatus),
        default=DatasetStatus.PENDING
    )

    # Collection parameters
    collection_start: Mapped[Optional[datetime]] = mapped_column(DateTime)
    collection_end: Mapped[Optional[datetime]] = mapped_column(DateTime)
    target_users: Mapped[int] = mapped_column(Integer, default=100)
    min_activities: Mapped[int] = mapped_column(Integer, default=5)

    # Statistics (populated after collection)
    n_users: Mapped[int] = mapped_column(Integer, default=0)
    n_activities: Mapped[int] = mapped_column(Integer, default=0)
    avg_duration_days: Mapped[Optional[float]] = mapped_column(Float)

    # Metadata
    config: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="dataset", cascade="all, delete-orphan")
    trials: Mapped[List["Trial"]] = relationship("Trial", back_populates="dataset")

    __table_args__ = (
        Index("idx_dataset_platform", "platform"),
        Index("idx_dataset_status", "status"),
    )


class User(Base):
    """
    User engagement data.
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[int] = mapped_column(Integer, ForeignKey("datasets.id"), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Adoption info
    adoption_timestamp: Mapped[float] = mapped_column(Float)
    first_activity: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_activity: Mapped[Optional[datetime]] = mapped_column(DateTime)

    # Aggregated data (stored as JSON arrays for efficiency)
    time_array: Mapped[Optional[list]] = mapped_column(JSON)  # Days since adoption
    engagement_array: Mapped[Optional[list]] = mapped_column(JSON)  # Normalized engagement
    external_trigger_array: Mapped[Optional[list]] = mapped_column(JSON)  # Boolean mask

    # Statistics
    n_activities: Mapped[int] = mapped_column(Integer, default=0)
    duration_days: Mapped[Optional[float]] = mapped_column(Float)
    total_engagement: Mapped[Optional[float]] = mapped_column(Float)

    # Extra data
    extra_data: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="users")
    fit_results: Mapped[List["UserFitResult"]] = relationship(
        "UserFitResult", back_populates="user", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("dataset_id", "external_id", name="uq_user_dataset"),
        Index("idx_user_dataset", "dataset_id"),
    )


class Trial(Base):
    """
    Analysis trial configuration and results.
    """
    __tablename__ = "trials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[TrialStatus] = mapped_column(SQLEnum(TrialStatus), default=TrialStatus.CREATED)

    # Dataset(s) used
    dataset_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("datasets.id"))
    dataset_ids: Mapped[Optional[list]] = mapped_column(JSON)  # For multi-dataset trials

    # Configuration
    config: Mapped[Optional[dict]] = mapped_column(JSON)
    # Includes: models to fit, normalization method, cross-validation settings, etc.

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float)

    # Results summary
    n_users_processed: Mapped[int] = mapped_column(Integer, default=0)
    best_model: Mapped[Optional[str]] = mapped_column(String(100))
    collapse_quality: Mapped[Optional[float]] = mapped_column(Float)
    master_curve_params: Mapped[Optional[dict]] = mapped_column(JSON)

    # Detailed results
    results: Mapped[Optional[dict]] = mapped_column(JSON)
    validation_results: Mapped[Optional[dict]] = mapped_column(JSON)
    statistics: Mapped[Optional[dict]] = mapped_column(JSON)

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    dataset: Mapped[Optional["Dataset"]] = relationship("Dataset", back_populates="trials")
    fit_results: Mapped[List["UserFitResult"]] = relationship(
        "UserFitResult", back_populates="trial", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_trial_status", "status"),
        Index("idx_trial_dataset", "dataset_id"),
    )


class UserFitResult(Base):
    """
    Model fitting results for individual users.
    """
    __tablename__ = "user_fit_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trial_id: Mapped[int] = mapped_column(Integer, ForeignKey("trials.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    # Best model
    best_model: Mapped[Optional[str]] = mapped_column(String(100))
    estimated_tau: Mapped[Optional[float]] = mapped_column(Float)
    estimated_alpha: Mapped[Optional[float]] = mapped_column(Float)

    # Model-specific results (JSON dict of model_name -> results)
    model_results: Mapped[Optional[dict]] = mapped_column(JSON)

    # Collapse data
    rescaled_time: Mapped[Optional[list]] = mapped_column(JSON)
    rescaled_engagement: Mapped[Optional[list]] = mapped_column(JSON)

    # Quality metrics
    residual_mean: Mapped[Optional[float]] = mapped_column(Float)
    residual_std: Mapped[Optional[float]] = mapped_column(Float)
    r_squared: Mapped[Optional[float]] = mapped_column(Float)

    # Deviation from universal
    deviation_score: Mapped[Optional[float]] = mapped_column(Float)
    is_deviant: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    trial: Mapped["Trial"] = relationship("Trial", back_populates="fit_results")
    user: Mapped["User"] = relationship("User", back_populates="fit_results")

    __table_args__ = (
        UniqueConstraint("trial_id", "user_id", name="uq_fit_result"),
        Index("idx_fit_trial", "trial_id"),
        Index("idx_fit_user", "user_id"),
    )


class MasterCurve(Base):
    """
    Fitted master curve parameters.
    """
    __tablename__ = "master_curves"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trial_id: Mapped[int] = mapped_column(Integer, ForeignKey("trials.id"), nullable=False)

    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    parameter_errors: Mapped[Optional[dict]] = mapped_column(JSON)

    # Quality metrics
    collapse_quality: Mapped[Optional[float]] = mapped_column(Float)
    ks_statistic: Mapped[Optional[float]] = mapped_column(Float)
    residual_std: Mapped[Optional[float]] = mapped_column(Float)

    # Confidence intervals
    confidence_intervals: Mapped[Optional[dict]] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_master_trial", "trial_id"),
    )


class ScalingRelationship(Base):
    """
    τ(α) = τ₀ · α^(-β) scaling relationship fits.
    """
    __tablename__ = "scaling_relationships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trial_id: Mapped[int] = mapped_column(Integer, ForeignKey("trials.id"), nullable=False)

    tau0: Mapped[float] = mapped_column(Float, nullable=False)
    beta: Mapped[float] = mapped_column(Float, nullable=False)
    tau0_error: Mapped[Optional[float]] = mapped_column(Float)
    beta_error: Mapped[Optional[float]] = mapped_column(Float)

    r_squared: Mapped[Optional[float]] = mapped_column(Float)
    correlation: Mapped[Optional[float]] = mapped_column(Float)
    p_value: Mapped[Optional[float]] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_scaling_trial", "trial_id"),
    )
