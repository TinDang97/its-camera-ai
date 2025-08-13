"""Database connection and session management.

Provides centralized database connection management using SQLAlchemy
with async support and connection pooling.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from .config import get_settings
from .logging import get_logger

logger = get_logger(__name__)

# Create declarative base for models
Base = declarative_base()

# Global variables for engine and session factory
_engine = None
_async_session_factory = None


def get_engine():
    """Get or create the async database engine.

    Returns:
        AsyncEngine: The database engine
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database.url,
            echo=settings.database.echo,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_timeout=settings.database.pool_timeout,
            pool_pre_ping=True,  # Verify connections before using
        )
        logger.info("Database engine created", url=settings.database.url.split("@")[0] + "@...")
    return _engine


def get_session_factory():
    """Get or create the async session factory.

    Returns:
        async_sessionmaker: The session factory
    """
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_engine()
        _async_session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        logger.info("Session factory created")
    return _async_session_factory


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session for FastAPI dependency injection.

    Yields:
        AsyncSession: A database session
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session as async context manager.

    Usage:
        async with get_db_session() as session:
            result = await session.execute(query)

    Yields:
        AsyncSession: A database session
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_database():
    """Initialize the database (create tables, etc.).

    This should be called during application startup.
    """
    engine = get_engine()

    # Import all models to ensure they're registered with Base
    from ..models import analytics, user  # noqa: F401

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("Database initialized")


async def close_database():
    """Close the database connections.

    This should be called during application shutdown.
    """
    global _engine, _async_session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _async_session_factory = None
        logger.info("Database connections closed")
