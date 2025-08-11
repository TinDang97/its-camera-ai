"""Database configuration and session management.

Provides SQLAlchemy engine and session factory for the application.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from ..core.config import Settings
from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from .base import BaseModel


logger = get_logger(__name__)


class DatabaseManager:
    """Database connection and session manager."""
    
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.settings.get_database_url(async_driver=True),
                echo=self.settings.database.echo,
                pool_size=self.settings.database.pool_size,
                max_overflow=self.settings.database.max_overflow,
                pool_timeout=self.settings.database.pool_timeout,
                poolclass=NullPool if self.settings.is_development() else None,
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            
            logger.info(
                "Database initialized",
                database_url=self.settings.database.url.split('@')[-1],  # Hide credentials
                pool_size=self.settings.database.pool_size,
            )
            
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise DatabaseError("Database initialization failed", cause=e) from e
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self.engine:
            raise DatabaseError("Database not initialized")
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(BaseModel.metadata.create_all)
            
            logger.info("Database tables created")
            
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise DatabaseError("Table creation failed", cause=e) from e
    
    async def close(self) -> None:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.session_factory = None
            logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session context manager.
        
        Yields:
            AsyncSession: Database session
        """
        if not self.session_factory:
            raise DatabaseError("Database not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()


# Global database manager instance
_db_manager: DatabaseManager | None = None


async def create_database_engine(settings: Settings) -> DatabaseManager:
    """Create and initialize database manager.
    
    Args:
        settings: Application settings
        
    Returns:
        DatabaseManager: Initialized database manager
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager(settings)
        await _db_manager.initialize()
    
    return _db_manager


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency.
    
    Yields:
        AsyncSession: Database session
    """
    if _db_manager is None:
        raise DatabaseError("Database not initialized")
    
    async with _db_manager.get_session() as session:
        yield session


async def close_database() -> None:
    """Close database connections."""
    global _db_manager
    
    if _db_manager:
        await _db_manager.close()
        _db_manager = None
