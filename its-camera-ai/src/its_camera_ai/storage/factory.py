"""Storage service factory for ITS Camera AI.

Provides centralized factory methods for creating and configuring
storage services with proper dependency injection and configuration.
"""

import logging

from ..core.config import Settings
from ..core.exceptions import ConfigurationError
from .minio_service import MinIOService
from .model_registry import MinIOModelRegistry
from .models import StorageConfig

logger = logging.getLogger(__name__)


class StorageServiceFactory:
    """Factory for creating storage services."""

    _storage_service: MinIOService | None = None
    _model_registry: MinIOModelRegistry | None = None
    _initialized = False

    @classmethod
    def create_storage_service(
        cls,
        settings: Settings,
        initialize: bool = True
    ) -> MinIOService:
        """Create MinIO storage service.

        Args:
            settings: Application settings
            initialize: Whether to initialize the service

        Returns:
            Configured MinIO storage service

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Create storage configuration from settings
            storage_config = StorageConfig(
                endpoint=settings.minio.endpoint,
                access_key=settings.minio.access_key,
                secret_key=settings.minio.secret_key,
                secure=settings.minio.secure,
                region=settings.minio.region,
                default_bucket=settings.minio.default_bucket,
                create_buckets=settings.minio.create_buckets,
                multipart_threshold=settings.minio.multipart_threshold,
                multipart_chunksize=settings.minio.multipart_chunksize,
                max_concurrent_uploads=settings.minio.max_concurrent_uploads,
                max_pool_connections=settings.minio.max_pool_connections,
                connection_timeout=settings.minio.connection_timeout,
                read_timeout=settings.minio.read_timeout,
                enable_caching=settings.minio.enable_caching,
                cache_ttl=settings.minio.cache_ttl,
                cache_max_size=settings.minio.cache_max_size,
                enable_versioning=settings.minio.enable_versioning,
                enable_compression=settings.minio.enable_compression,
                enable_metrics=settings.minio.enable_metrics,
                metrics_interval=settings.minio.metrics_interval,
            )

            service = MinIOService(storage_config)

            if initialize:
                # Note: This is sync, actual initialization should be done async
                logger.info("MinIO storage service created (initialization pending)")

            return service

        except Exception as e:
            raise ConfigurationError(
                f"Failed to create storage service: {e}",
                config_key="minio"
            ) from e

    @classmethod
    def create_model_registry(
        cls,
        settings: Settings,
        storage_service: MinIOService | None = None,
        initialize: bool = True
    ) -> MinIOModelRegistry:
        """Create MinIO model registry.

        Args:
            settings: Application settings
            storage_service: Optional existing storage service
            initialize: Whether to initialize the registry

        Returns:
            Configured model registry

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Create storage configuration if service not provided
            if storage_service is None:
                storage_config = StorageConfig(
                    endpoint=settings.minio.endpoint,
                    access_key=settings.minio.access_key,
                    secret_key=settings.minio.secret_key,
                    secure=settings.minio.secure,
                    region=settings.minio.region,
                )
            else:
                storage_config = storage_service.config

            # Create registry configuration
            registry_config = {
                "models_bucket": settings.minio.models_bucket,
                "metadata_bucket": "its-metadata",
                "enable_compression": settings.minio.enable_compression,
                "enable_versioning": settings.minio.enable_versioning,
                "cache_ttl": settings.minio.cache_ttl,
                "allow_overwrites": False,
            }

            registry = MinIOModelRegistry(storage_config, registry_config)

            if initialize:
                logger.info("MinIO model registry created (initialization pending)")

            return registry

        except Exception as e:
            raise ConfigurationError(
                f"Failed to create model registry: {e}",
                config_key="minio"
            ) from e

    @classmethod
    async def get_storage_service(
        cls,
        settings: Settings,
        force_recreate: bool = False
    ) -> MinIOService:
        """Get singleton storage service instance.

        Args:
            settings: Application settings
            force_recreate: Force recreation of service

        Returns:
            MinIO storage service instance
        """
        if cls._storage_service is None or force_recreate:
            cls._storage_service = cls.create_storage_service(
                settings, initialize=False
            )
            await cls._storage_service.initialize()
            logger.info("Singleton storage service initialized")

        return cls._storage_service

    @classmethod
    async def get_model_registry(
        cls,
        settings: Settings,
        force_recreate: bool = False
    ) -> MinIOModelRegistry:
        """Get singleton model registry instance.

        Args:
            settings: Application settings
            force_recreate: Force recreation of registry

        Returns:
            Model registry instance
        """
        if cls._model_registry is None or force_recreate:
            # Ensure storage service exists
            storage_service = await cls.get_storage_service(settings)

            cls._model_registry = cls.create_model_registry(
                settings, storage_service, initialize=False
            )
            await cls._model_registry.initialize()
            logger.info("Singleton model registry initialized")

        return cls._model_registry

    @classmethod
    async def initialize_all_services(
        cls,
        settings: Settings
    ) -> tuple[MinIOService, MinIOModelRegistry]:
        """Initialize all storage services.

        Args:
            settings: Application settings

        Returns:
            Tuple of (storage_service, model_registry)
        """
        if cls._initialized:
            return cls._storage_service, cls._model_registry

        logger.info("Initializing all storage services...")

        try:
            # Initialize storage service
            storage_service = await cls.get_storage_service(settings)

            # Initialize model registry
            model_registry = await cls.get_model_registry(settings)

            cls._initialized = True
            logger.info("All storage services initialized successfully")

            return storage_service, model_registry

        except Exception as e:
            logger.error(f"Failed to initialize storage services: {e}")
            raise

    @classmethod
    async def cleanup_all_services(cls) -> None:
        """Cleanup all storage services."""
        logger.info("Cleaning up storage services...")

        try:
            if cls._model_registry:
                await cls._model_registry.cleanup()
                cls._model_registry = None

            if cls._storage_service:
                await cls._storage_service.cleanup()
                cls._storage_service = None

            cls._initialized = False
            logger.info("Storage services cleanup completed")

        except Exception as e:
            logger.error(f"Error during storage services cleanup: {e}")

    @classmethod
    def get_service_health(cls) -> dict:
        """Get health status of all services."""
        health_info = {
            "initialized": cls._initialized,
            "storage_service": None,
            "model_registry": None,
        }

        if cls._storage_service:
            try:
                health_info["storage_service"] = cls._storage_service.get_health_status()
            except Exception as e:
                health_info["storage_service"] = {"status": "error", "error": str(e)}

        if cls._model_registry:
            try:
                health_info["model_registry"] = cls._model_registry.get_health_status()
            except Exception as e:
                health_info["model_registry"] = {"status": "error", "error": str(e)}

        return health_info


# Convenience functions for easy access

async def get_storage_service(settings: Settings | None = None) -> MinIOService:
    """Get storage service instance.

    Args:
        settings: Optional settings, will get from config if not provided

    Returns:
        MinIO storage service
    """
    if settings is None:
        from ..core.config import get_settings
        settings = get_settings()

    return await StorageServiceFactory.get_storage_service(settings)


async def get_model_registry(settings: Settings | None = None) -> MinIOModelRegistry:
    """Get model registry instance.

    Args:
        settings: Optional settings, will get from config if not provided

    Returns:
        Model registry
    """
    if settings is None:
        from ..core.config import get_settings
        settings = get_settings()

    return await StorageServiceFactory.get_model_registry(settings)
