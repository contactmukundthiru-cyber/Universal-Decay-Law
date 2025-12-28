"""
Database Connection and Session Management.

Provides async SQLAlchemy engine and session handling.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy import event

from database.models import Base
from config.settings import settings


def get_engine():
    """Create async SQLAlchemy engine."""
    # Ensure database directory exists for SQLite
    if "sqlite" in settings.database.driver:
        db_path = Path(settings.database.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(
        settings.database.url,
        echo=settings.server.debug,
        pool_pre_ping=True,
    )

    return engine


# Global engine instance
_engine = None


def get_global_engine():
    """Get or create global engine instance."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


# Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=get_global_engine(),
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database sessions.

    Usage in FastAPI:
        @app.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_session_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.

    Usage:
        async with get_session_context() as session:
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database tables.

    Creates all tables defined in models.
    """
    engine = get_global_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """
    Drop all database tables.

    WARNING: This will delete all data!
    """
    engine = get_global_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def reset_db() -> None:
    """
    Reset database (drop and recreate all tables).

    WARNING: This will delete all data!
    """
    await drop_db()
    await init_db()
