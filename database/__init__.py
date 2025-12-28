"""
Database Module.

Provides database connection, session management, and CRUD operations.
"""

from database.models import (
    Base,
    Dataset,
    User,
    Trial,
    UserFitResult,
    MasterCurve,
    ScalingRelationship,
    DatasetStatus,
    TrialStatus,
)
from database.connection import (
    get_engine,
    get_session,
    init_db,
    AsyncSessionLocal,
)
from database.crud import (
    DatasetCRUD,
    UserCRUD,
    TrialCRUD,
)

__all__ = [
    # Models
    "Base",
    "Dataset",
    "User",
    "Trial",
    "UserFitResult",
    "MasterCurve",
    "ScalingRelationship",
    "DatasetStatus",
    "TrialStatus",
    # Connection
    "get_engine",
    "get_session",
    "init_db",
    "AsyncSessionLocal",
    # CRUD
    "DatasetCRUD",
    "UserCRUD",
    "TrialCRUD",
]
