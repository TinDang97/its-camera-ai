"""Database factory for managing multiple database connections.

Provides a centralized factory for creating and managing database connections
with singleton pattern, environment-based configuration, and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import atexit
from typing import Any

from ...core.config import get_settings
from ...core.exceptions import ConfigurationError, DatabaseError
from ...core.logging import get_logger
from .models import (
    DatabaseConfig,
    DatabaseType,
    RedisConfig,
    TimescaleConfig,
)
from .postgresql_service import PostgreSQLService
from .redis_manager import RedisManager
from .timescale_service import TimescaleService

logger = get_logger(__name__)


class DatabaseFactory:
    """Database connection factory with singleton pattern.
    
    Manages multiple database connections with automatic configuration,
    connection pooling, and graceful shutdown handling.
    """

    _instance: DatabaseFactory | None = None
    _lock = asyncio.Lock()

    def __new__(cls) -> DatabaseFactory:
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize database factory."""
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._services: dict[str, Any] = {}
        self._connections_initialized = False

        # Register cleanup on exit
        atexit.register(self._cleanup_sync)

    def _convert_database_config(self, core_config: Any) -> DatabaseConfig:
        """Convert core database config to infrastructure config.
        
        Args:
            core_config: Core database configuration
            
        Returns:
            Infrastructure database configuration
        """
        return DatabaseConfig(
            host=getattr(core_config, 'host', 'localhost'),
            port=getattr(core_config, 'port', 5432),
            database=getattr(core_config, 'database', 'its_camera_ai'),
            username=getattr(core_config, 'username', 'postgres'),
            password=getattr(core_config, 'password', 'password'),
        )

    def _convert_redis_config(self, core_config: Any) -> RedisConfig:
        """Convert core Redis config to infrastructure config.
        
        Args:
            core_config: Core Redis configuration
            
        Returns:
            Infrastructure Redis configuration
        """
        return RedisConfig(
            host=getattr(core_config, 'host', 'localhost'),
            port=getattr(core_config, 'port', 6379),
            database=getattr(core_config, 'database', 0),
            password=getattr(core_config, 'password', None),
        )

    async def initialize(self) -> None:
        """Initialize all configured database connections."""
        if self._connections_initialized:
            return

        async with self._lock:
            if self._connections_initialized:
                return

            logger.info("Initializing database connections")
            settings = get_settings()

            try:
                # Convert core configs to infrastructure configs
                db_config = self._convert_database_config(settings.database)
                redis_config = self._convert_redis_config(settings.redis)

                # Initialize PostgreSQL connection
                await self._initialize_postgresql(db_config)

                # Initialize Redis connection
                await self._initialize_redis(redis_config)

                self._connections_initialized = True
                logger.info("Database connections initialized successfully")

            except Exception as e:
                logger.error("Failed to initialize database connections", error=str(e))
                await self.cleanup()
                raise DatabaseError(
                    "Database initialization failed",
                    operation="initialize",
                    cause=e,
                ) from e

    async def _initialize_postgresql(self, config: DatabaseConfig) -> None:
        """Initialize PostgreSQL connection.
        
        Args:
            config: PostgreSQL configuration
        """
        try:
            service = PostgreSQLService(config)
            await service.connect()
            self._services["postgresql"] = service
            logger.info("PostgreSQL service initialized")

        except Exception as e:
            logger.error("Failed to initialize PostgreSQL", error=str(e))
            raise DatabaseError(
                "PostgreSQL initialization failed",
                operation="initialize_postgresql",
                cause=e,
            ) from e

    async def _initialize_timescale(self, config: TimescaleConfig) -> None:
        """Initialize TimescaleDB connection.
        
        Args:
            config: TimescaleDB configuration
        """
        try:
            service = TimescaleService(config)
            await service.connect()
            self._services["timescale"] = service
            logger.info("TimescaleDB service initialized")

        except Exception as e:
            logger.error("Failed to initialize TimescaleDB", error=str(e))
            raise DatabaseError(
                "TimescaleDB initialization failed",
                operation="initialize_timescale",
                cause=e,
            ) from e

    async def _initialize_redis(self, config: RedisConfig) -> None:
        """Initialize Redis connection.
        
        Args:
            config: Redis configuration
        """
        try:
            manager = RedisManager(config)
            await manager.connect()
            self._services["redis"] = manager
            logger.info("Redis manager initialized")

        except Exception as e:
            logger.error("Failed to initialize Redis", error=str(e))
            raise DatabaseError(
                "Redis initialization failed",
                operation="initialize_redis",
                cause=e,
            ) from e

    async def get_postgresql(self) -> PostgreSQLService:
        """Get PostgreSQL service instance.
        
        Returns:
            PostgreSQL service
            
        Raises:
            DatabaseError: If PostgreSQL is not initialized
        """
        if not self._connections_initialized:
            await self.initialize()

        service = self._services.get("postgresql")
        if not isinstance(service, PostgreSQLService):
            raise DatabaseError(
                "PostgreSQL service not initialized",
                operation="get_postgresql",
            )

        return service

    async def get_timescale(self) -> TimescaleService:
        """Get TimescaleDB service instance.
        
        Returns:
            TimescaleDB service
            
        Raises:
            DatabaseError: If TimescaleDB is not initialized
        """
        if not self._connections_initialized:
            await self.initialize()

        service = self._services.get("timescale")
        if service is None:
            # Fall back to PostgreSQL service if TimescaleDB not separately configured
            postgresql_service = self._services.get("postgresql")
            if isinstance(postgresql_service, TimescaleService):
                return postgresql_service
            else:
                raise DatabaseError(
                    "TimescaleDB service not initialized",
                    operation="get_timescale",
                )

        if not isinstance(service, TimescaleService):
            raise DatabaseError(
                "TimescaleDB service not properly configured",
                operation="get_timescale",
            )

        return service

    async def get_redis(self) -> RedisManager:
        """Get Redis manager instance.
        
        Returns:
            Redis manager
            
        Raises:
            DatabaseError: If Redis is not initialized
        """
        if not self._connections_initialized:
            await self.initialize()

        manager = self._services.get("redis")
        if not isinstance(manager, RedisManager):
            raise DatabaseError(
                "Redis manager not initialized",
                operation="get_redis",
            )

        return manager

    async def get_service(self, database_type: DatabaseType) -> Any:
        """Get service by database type.
        
        Args:
            database_type: Type of database service
            
        Returns:
            Database service instance
            
        Raises:
            ConfigurationError: If database type is not supported
            DatabaseError: If service is not initialized
        """
        if database_type == DatabaseType.POSTGRESQL:
            return await self.get_postgresql()
        elif database_type == DatabaseType.TIMESCALE:
            return await self.get_timescale()
        elif database_type == DatabaseType.REDIS:
            return await self.get_redis()
        else:
            raise ConfigurationError(
                f"Unsupported database type: {database_type}",
                config_key="database_type",
            )

    async def health_check_all(self) -> dict[str, bool]:
        """Perform health check on all initialized services.
        
        Returns:
            Dictionary mapping service names to health status
        """
        if not self._connections_initialized:
            return {}

        health_results = {}

        for service_name, service in self._services.items():
            try:
                health = await service.health_check()
                health_results[service_name] = health.status.value == "healthy"
            except Exception as e:
                logger.error(f"Health check failed for {service_name}", error=str(e))
                health_results[service_name] = False

        return health_results

    async def reconnect_all(self) -> None:
        """Reconnect all database services."""
        logger.info("Reconnecting all database services")

        reconnection_tasks = []

        for service_name, service in self._services.items():
            try:
                # Disconnect first
                if hasattr(service, "disconnect"):
                    await service.disconnect()

                # Reconnect
                if hasattr(service, "connect"):
                    reconnection_tasks.append(
                        asyncio.create_task(
                            service.connect(),
                            name=f"reconnect_{service_name}"
                        )
                    )

            except Exception as e:
                logger.error(f"Failed to reconnect {service_name}", error=str(e))

        # Wait for all reconnections to complete
        if reconnection_tasks:
            results = await asyncio.gather(*reconnection_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                service_name = reconnection_tasks[i].get_name().replace("reconnect_", "")
                if isinstance(result, Exception):
                    logger.error(f"Reconnection failed for {service_name}", error=str(result))
                else:
                    logger.info(f"Reconnection successful for {service_name}")

    async def cleanup(self) -> None:
        """Clean up all database connections."""
        if not self._connections_initialized:
            return

        logger.info("Cleaning up database connections")

        cleanup_tasks = []

        for service_name, service in self._services.items():
            if hasattr(service, "disconnect"):
                cleanup_tasks.append(
                    asyncio.create_task(
                        service.disconnect(),
                        name=f"cleanup_{service_name}"
                    )
                )

        # Wait for all cleanup tasks to complete
        if cleanup_tasks:
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)

            for i, result in enumerate(results):
                service_name = cleanup_tasks[i].get_name().replace("cleanup_", "")
                if isinstance(result, Exception):
                    logger.error(f"Cleanup failed for {service_name}", error=str(result))
                else:
                    logger.debug(f"Cleanup completed for {service_name}")

        self._services.clear()
        self._connections_initialized = False
        logger.info("Database cleanup completed")

    def _cleanup_sync(self) -> None:
        """Synchronous cleanup for atexit handler."""
        try:
            if self._connections_initialized and self._services:
                # Run cleanup in event loop if available
                try:
                    loop = asyncio.get_running_loop()
                    # Schedule cleanup as a task
                    loop.create_task(self.cleanup())
                except RuntimeError:
                    # No event loop running, create a new one
                    asyncio.run(self.cleanup())
        except Exception as e:
            logger.error("Error during synchronous cleanup", error=str(e))

    @property
    def is_initialized(self) -> bool:
        """Check if factory is initialized."""
        return self._connections_initialized

    @property
    def service_count(self) -> int:
        """Get number of initialized services."""
        return len(self._services)

    @property
    def service_names(self) -> list[str]:
        """Get list of initialized service names."""
        return list(self._services.keys())


# Global factory instance
_database_factory: DatabaseFactory | None = None


async def get_database_factory() -> DatabaseFactory:
    """Get global database factory instance.
    
    Returns:
        Database factory instance
    """
    global _database_factory

    if _database_factory is None:
        _database_factory = DatabaseFactory()
        await _database_factory.initialize()

    return _database_factory


async def get_postgresql() -> PostgreSQLService:
    """Get PostgreSQL service from global factory.
    
    Returns:
        PostgreSQL service instance
    """
    factory = await get_database_factory()
    return await factory.get_postgresql()


async def get_timescale() -> TimescaleService:
    """Get TimescaleDB service from global factory.
    
    Returns:
        TimescaleDB service instance
    """
    factory = await get_database_factory()
    return await factory.get_timescale()


async def get_redis() -> RedisManager:
    """Get Redis manager from global factory.
    
    Returns:
        Redis manager instance
    """
    factory = await get_database_factory()
    return await factory.get_redis()

