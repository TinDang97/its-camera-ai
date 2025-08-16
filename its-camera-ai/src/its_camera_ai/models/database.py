"""Optimized database management with dependency injection and production features.

This module provides a unified DatabaseManager class that combines production-optimized
database configurations with proper dependency injection patterns using the Resource
lifecycle management approach.

Key Features:
- Production vs development configuration optimization
- High-throughput connection pooling with environment-specific settings
- PostgreSQL-specific optimizations (JIT, application naming, search paths)
- Proper async session management with automatic transaction handling
- Resource lifecycle management for dependency injection containers
- Settings-based configuration with secure URL handling
- Comprehensive error handling with custom exceptions
- Health monitoring and connection pool statistics
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import quote_plus

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, QueuePool

from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from .base import BaseTableModel

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from ..core.config import Settings


logger = get_logger(__name__)


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(
            message,
            details={"original_error": str(original_error) if original_error else None},
        )
        self.original_error = original_error


class DatabaseTransactionError(DatabaseError):
    """Raised when database transaction fails."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(
            message,
            details={"original_error": str(original_error) if original_error else None},
        )
        self.original_error = original_error


class DatabaseConfigurationError(DatabaseError):
    """Raised when database configuration is invalid."""

    def __init__(self, message: str):
        super().__init__(message, details={"configuration_issue": message})


class DatabaseManager:
    """Unified database manager with production optimizations and dependency injection support.

    This class combines high-performance database connection management with proper
    Resource lifecycle patterns for dependency injection containers.

    Features:
    - Environment-specific configuration (production vs development vs testing)
    - Optimized connection pooling for high-throughput scenarios (20+40 for prod, 5+10 for dev)
    - PostgreSQL-specific performance optimizations (JIT off, search paths, cache settings)
    - Proper async session management with automatic transaction handling
    - Resource lifecycle management for dependency injection containers
    - Comprehensive error handling with custom exceptions and detailed logging
    - Health monitoring with connection pool statistics

    Usage:
        # As a dependency injection resource
        container.database.provided.get_session()

        # Direct usage
        async with database_manager.get_session() as session:
            result = await session.execute(text("SELECT 1"))
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize DatabaseManager with settings-based configuration.

        Args:
            settings: Application settings containing database configuration

        Raises:
            DatabaseConfigurationError: If database configuration is invalid
        """
        self.settings = settings
        self._engine: AsyncEngine | None = None
        self._sessionmaker: async_sessionmaker[AsyncSession] | None = None
        self._is_initialized = False

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate database configuration.

        Raises:
            DatabaseConfigurationError: If configuration is invalid
        """
        if not self.settings.database.url:
            raise DatabaseConfigurationError("Database URL is required")

        db_url = self.settings.database.url
        if not db_url.startswith(("postgresql://", "postgresql+asyncpg://")):
            raise DatabaseConfigurationError(
                "Database URL must be a PostgreSQL connection string"
            )

    def _build_database_url(self) -> str:
        """Build optimized database URL with proper encoding and connection parameters.

        Returns:
            Properly formatted database URL with PostgreSQL optimizations
        """
        base_url = self.settings.database.url

        # Ensure we're using asyncpg driver for async support
        if base_url.startswith("postgresql://"):
            base_url = base_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        # Add PostgreSQL-specific optimizations via connection parameters
        optimizations = {
            "server_side_cursors": "false",  # Better for small result sets in our use case
            "jit": "off",  # Disable JIT for stable performance in high-throughput scenarios
            "application_name": f"its_camera_ai_{self.settings.environment}",
            "connect_timeout": "30",  # 30-second connection timeout
        }

        # Build query string
        separator = "&" if "?" in base_url else "?"
        query_params = "&".join(
            f"{k}={quote_plus(str(v))}" for k, v in optimizations.items()
        )

        return f"{base_url}{separator}{query_params}"

    def _get_engine_config(self) -> dict[str, Any]:
        """Get environment-specific engine configuration.

        Returns:
            Dictionary of SQLAlchemy engine configuration parameters optimized per environment
        """
        base_config = {
            "echo": self.settings.database.echo,
            "echo_pool": self.settings.debug
            and self.settings.environment == "development",
            "future": True,
            "connect_args": {
                "server_settings": {
                    "jit": "off",  # Disable JIT for consistent performance
                    "application_name": f"its_camera_ai_{self.settings.environment}",
                }
            },
        }

        if self.settings.environment == "production":
            # Production optimizations for high-throughput scenarios (10TB/day capacity)
            base_config.update(
                {
                    "poolclass": QueuePool,
                    "pool_size": min(
                        self.settings.database.pool_size * 2, 50
                    ),  # Increased for high throughput
                    "max_overflow": min(self.settings.database.max_overflow * 2, 100),
                    "pool_pre_ping": True,  # Validate connections before use
                    "pool_recycle": 3600,  # Recycle connections every hour
                    "pool_reset_on_return": "commit",  # Ensure clean state
                    "pool_timeout": self.settings.database.pool_timeout,
                    "connect_args": {
                        **base_config["connect_args"],
                        "server_settings": {
                            **base_config["connect_args"]["server_settings"],
                            # PostgreSQL performance optimizations for production
                            "random_page_cost": "1.1",  # SSD optimization
                            "effective_cache_size": "1GB",  # Cache size hint
                            "work_mem": "16MB",  # Memory for operations
                            "maintenance_work_mem": "256MB",  # Maintenance operations
                        },
                    },
                }
            )
        elif self.settings.environment == "testing":
            # Testing configuration with no connection pooling for isolation
            base_config.update(
                {
                    "poolclass": NullPool,  # No connection pooling for tests
                    "echo": False,  # Reduce test noise
                    "connect_args": {
                        **base_config["connect_args"],
                        "isolation_level": "AUTOCOMMIT",  # Fast test execution
                    },
                }
            )
        else:
            # Development configuration with moderate pooling
            base_config.update(
                {
                    "poolclass": QueuePool,
                    "pool_size": self.settings.database.pool_size,
                    "max_overflow": self.settings.database.max_overflow,
                    "pool_pre_ping": True,
                    "pool_recycle": 1800,  # 30 minutes for development
                    "pool_timeout": self.settings.database.pool_timeout,
                }
            )

        return base_config

    async def _create_engine(self) -> AsyncEngine:
        """Create and configure the async database engine.

        Returns:
            Configured AsyncEngine instance with environment-specific optimizations

        Raises:
            DatabaseConnectionError: If engine creation fails
        """
        try:
            database_url = self._build_database_url()
            engine_config = self._get_engine_config()

            engine = create_async_engine(database_url, **engine_config)

            # Add event listeners for connection management
            @event.listens_for(engine.sync_engine, "connect")
            def set_postgresql_optimizations(dbapi_connection, connection_record):
                """Set PostgreSQL optimizations on new connections."""
                try:
                    with dbapi_connection.cursor() as cursor:
                        # Set search path for security
                        cursor.execute("SET search_path TO public")

                        # Performance optimizations for ITS Camera AI workload
                        if self.settings.environment == "production":
                            # High-throughput optimizations
                            cursor.execute(
                                "SET random_page_cost = 1.1"
                            )  # SSD optimization
                            cursor.execute("SET effective_cache_size = '1GB'")
                            cursor.execute("SET work_mem = '16MB'")
                            cursor.execute("SET maintenance_work_mem = '256MB'")
                            cursor.execute("SET checkpoint_completion_target = 0.9")
                            cursor.execute("SET wal_buffers = '16MB'")

                except Exception as e:
                    logger.warning(
                        "Failed to set PostgreSQL connection optimizations",
                        error=str(e),
                        connection=connection_record.info.get("pid"),
                    )

            pool_info = engine_config.get("pool_size", "default")
            max_overflow = engine_config.get("max_overflow", "default")

            logger.info(
                "Database engine created successfully",
                environment=self.settings.environment,
                pool_size=pool_info,
                max_overflow=max_overflow,
                url_host=(
                    database_url.split("@")[1].split("/")[0]
                    if "@" in database_url
                    else "localhost"
                ),
            )

            return engine

        except Exception as e:
            logger.error(
                "Failed to create database engine",
                error=str(e),
                environment=self.settings.environment,
                url_configured=bool(self.settings.database.url),
            )
            raise DatabaseConnectionError(
                f"Failed to create database engine: {str(e)}", e
            ) from e

    async def _test_connection(self) -> None:
        """Test database connectivity and basic functionality.

        Raises:
            DatabaseConnectionError: If connection test fails
        """
        if not self._sessionmaker:
            raise DatabaseConnectionError("Session factory not initialized")

        try:
            async with self._sessionmaker() as session:
                # Test basic connectivity
                result = await session.execute(text("SELECT 1 as connectivity_test"))
                test_value = result.scalar()
                if test_value != 1:
                    raise DatabaseConnectionError("Database connectivity test failed")

                # Test PostgreSQL-specific functionality
                result = await session.execute(text("SELECT version() as pg_version"))
                pg_version = result.scalar()

                logger.debug(
                    "Database connectivity test passed",
                    postgres_version=(
                        pg_version[:50] + "..." if len(pg_version) > 50 else pg_version
                    ),
                )

        except Exception as e:
            raise DatabaseConnectionError(
                f"Database connectivity test failed: {str(e)}", e
            ) from e

    async def _ensure_tables_exist(self) -> None:
        """Ensure database tables exist (non-production environments only)."""
        if not self._engine:
            raise DatabaseConnectionError("Engine not initialized")

        try:
            async with self._engine.begin() as conn:
                await conn.run_sync(BaseTableModel.metadata.create_all)

            logger.info("Database tables verified/created")

        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise DatabaseConnectionError("Table creation failed", e) from e

    @asynccontextmanager
    async def initialize(self) -> None:
        """Initialize the database manager.

        This must be called before using the database manager.
        Creates the engine and session factory.

        Raises:
            DatabaseConnectionError: If initialization fails
        """
        if self._is_initialized:
            logger.debug("Database manager already initialized")
            return

        try:
            self._engine = await self._create_engine()
            self._sessionmaker = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )

            # Test connection
            await self._test_connection()

            # Create tables in non-production environments
            if self.settings.environment != "production":
                await self._ensure_tables_exist()

            self._is_initialized = True

            logger.info(
                "Database manager initialized successfully",
                environment=self.settings.environment,
                pool_size=(
                    getattr(self._engine.pool, "size", lambda: "unknown")()
                    if hasattr(self._engine, "pool")
                    else "unknown"
                ),
            )

        except Exception as e:
            logger.error("Failed to initialize database manager", error=str(e))
            await self.cleanup()
            raise DatabaseConnectionError(
                f"Database initialization failed: {str(e)}", e
            ) from e

    async def cleanup(self) -> None:
        """Clean up database resources.

        Disposes of the engine and resets initialization state.
        """
        if self._engine:
            try:
                await self._engine.dispose()
                logger.debug("Database engine disposed successfully")
            except Exception as e:
                logger.error("Error disposing database engine", error=str(e))

        self._engine = None
        self._sessionmaker = None
        self._is_initialized = False

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session with proper transaction management.

        This context manager provides a database session that automatically
        handles transactions. It uses SQLAlchemy's built-in transaction handling
        which automatically commits on success and rolls back on exceptions.

        Yields:
            AsyncSession: Database session for query execution

        Raises:
            DatabaseConnectionError: If session creation fails
            DatabaseTransactionError: If transaction handling fails

        Example:
            async with database_manager.get_session() as session:
                result = await session.execute(text("SELECT * FROM cameras"))
                cameras = result.fetchall()
                # Automatic commit happens here if no exception
        """
        if not self._is_initialized:
            await self.initialize()

        if not self._sessionmaker:
            raise DatabaseConnectionError("Session factory not available")

        async with self._sessionmaker() as session:
            try:
                yield session
                # SQLAlchemy's async context manager automatically handles commit/rollback
            except Exception as e:
                # Log the error for debugging
                logger.error(
                    "Database session error", error=str(e), error_type=type(e).__name__
                )
                # Re-raise the exception (rollback is handled by SQLAlchemy)
                raise DatabaseTransactionError(
                    f"Database operation failed: {str(e)}", e
                ) from e

    async def get_raw_session(self) -> AsyncSession:
        """Get a raw database session without automatic transaction management.

        This method provides a session without automatic transaction handling.
        The caller is responsible for managing transactions and closing the session.

        Returns:
            AsyncSession: Raw database session

        Raises:
            DatabaseConnectionError: If session creation fails

        Warning:
            The caller must ensure proper session cleanup and transaction management.
        """
        if not self._is_initialized:
            await self.initialize()

        if not self._sessionmaker:
            raise DatabaseConnectionError("Session factory not available")

        try:
            return self._sessionmaker()
        except Exception as e:
            raise DatabaseConnectionError(
                f"Failed to create database session: {str(e)}", e
            ) from e

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def health_check(self) -> dict[str, Any]:
        """Perform a comprehensive health check of the database.

        Returns:
            Dictionary containing health check results including connection pool statistics
        """
        health_status = {
            "status": "unknown",
            "initialized": self._is_initialized,
            "environment": self.settings.environment,
            "engine_disposed": self._engine is None,
            "error": None,
            "connection_pool": {},
            "database_info": {},
        }

        try:
            if not self._is_initialized:
                health_status.update(
                    {
                        "status": "unhealthy",
                        "error": "Database manager not initialized",
                    }
                )
                return health_status

            # Test connectivity
            await self._test_connection()

            # Get connection pool statistics if available
            if self._engine and hasattr(self._engine.pool, "status"):
                pool = self._engine.pool
                health_status["connection_pool"] = {
                    "size": getattr(pool, "size", lambda: None)(),
                    "checked_in": getattr(pool, "checkedin", lambda: None)(),
                    "checked_out": getattr(pool, "checkedout", lambda: None)(),
                    "overflow": getattr(pool, "overflow", lambda: None)(),
                    "invalid": getattr(pool, "invalid", lambda: None)(),
                }

            # Get database information
            if self._sessionmaker:
                try:
                    async with self._sessionmaker() as session:
                        result = await session.execute(
                            text(
                                "SELECT current_database(), current_user, inet_server_addr(), inet_server_port()"
                            )
                        )
                        db_info = result.fetchone()
                        if db_info:
                            health_status["database_info"] = {
                                "database": db_info[0],
                                "user": db_info[1],
                                "host": db_info[2],
                                "port": db_info[3],
                            }
                except Exception as info_error:
                    logger.debug(
                        "Could not retrieve database info", error=str(info_error)
                    )

            health_status["status"] = "healthy"

        except Exception as e:
            health_status.update(
                {
                    "status": "unhealthy",
                    "error": str(e),
                }
            )

        return health_status

    @property
    def is_initialized(self) -> bool:
        """Check if the database manager is initialized and ready for use."""
        return self._is_initialized

    @property
    def engine(self) -> AsyncEngine | None:
        """Get the database engine (read-only access)."""
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession] | None:
        """Get the session factory (read-only access)."""
        return self._sessionmaker

    def __repr__(self) -> str:
        """String representation of DatabaseManager."""
        return (
            f"DatabaseManager("
            f"environment={self.settings.environment}, "
            f"initialized={self._is_initialized}, "
            f"pool_size={getattr(self._engine.pool, 'size', lambda: 'unknown')() if self._engine else 'none'}"
            f")"
        )
