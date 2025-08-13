"""Database configuration and session management.

Provides SQLAlchemy engine and session factory for the application.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

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
from .base import BaseTableModel

logger = get_logger(__name__)


class DatabaseManager:
    """Database connection and session manager."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None

    async def initialize(self) -> None:
        """Initialize database engine and session factory with high-throughput optimizations."""
        try:
            # High-throughput engine configuration
            engine_kwargs = {
                "echo": self.settings.database.echo,
                "pool_size": self.settings.database.pool_size,
                "max_overflow": self.settings.database.max_overflow,
                "pool_timeout": self.settings.database.pool_timeout,
                "pool_recycle": 3600,  # Recycle connections every hour
                "pool_pre_ping": True,  # Validate connections before use
            }

            # Production optimizations
            if self.settings.is_production():
                engine_kwargs.update(
                    {
                        # Increase connection limits for high throughput
                        "pool_size": min(self.settings.database.pool_size * 2, 50),
                        "max_overflow": min(
                            self.settings.database.max_overflow * 2, 100
                        ),
                        # Optimize for bulk operations
                        "connect_args": {
                            "server_settings": {
                                "jit": "off",  # Disable JIT for consistent performance
                                "application_name": "ITS-Camera-AI",
                            }
                        },
                    }
                )
            else:
                # Development configuration
                engine_kwargs["poolclass"] = NullPool

            # Create async engine
            self.engine = create_async_engine(
                self.settings.get_database_url(async_driver=True), **engine_kwargs
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            logger.info(
                "Database initialized",
                database_url=self.settings.database.url.split("@")[
                    -1
                ],  # Hide credentials
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
                await conn.run_sync(BaseTableModel.metadata.create_all)

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
