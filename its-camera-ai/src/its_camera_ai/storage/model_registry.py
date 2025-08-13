"""Enhanced Model Registry with MinIO integration.

Extends the base model registry to store model artifacts in MinIO
object storage with versioning, metadata tracking, and efficient retrieval.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from ..core.exceptions import ModelError, ResourceNotFoundError, ValidationError
from ..ml.model_pipeline import ModelVersion
from .minio_service import MinIOService
from .models import (
    DownloadRequest,
    ObjectType,
    StorageConfig,
    UploadRequest,
)

logger = logging.getLogger(__name__)


class MinIOModelRegistry:
    """Model registry with MinIO object storage backend.

    Provides centralized model artifact storage with:
    - Version-controlled model storage
    - Efficient model retrieval and caching
    - Metadata tracking and search
    - Model artifact compression and checksums
    - Integration with ML training pipeline
    """

    def __init__(self, storage_config: StorageConfig, registry_config: dict[str, Any]):
        """Initialize MinIO model registry.

        Args:
            storage_config: MinIO storage configuration
            registry_config: Registry-specific configuration
        """
        self.storage_config = storage_config
        self.registry_config = registry_config

        # MinIO storage service
        self.storage_service: MinIOService

        # Registry configuration
        self.models_bucket = registry_config.get("models_bucket", "its-models")
        self.metadata_bucket = registry_config.get("metadata_bucket", "its-metadata")
        self.enable_compression = registry_config.get("enable_compression", True)
        self.enable_versioning = registry_config.get("enable_versioning", True)

        # Local cache for metadata
        self.metadata_cache: dict[str, dict[str, Any]] = {}
        self.cache_ttl = registry_config.get("cache_ttl", 300)  # 5 minutes
        self.last_cache_update = 0.0

        # Model statistics
        self.stats = {
            "total_models": 0,
            "total_storage_size": 0,
            "successful_uploads": 0,
            "failed_uploads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info("MinIO Model Registry initialized")

    async def initialize(self) -> None:
        """Initialize the model registry and MinIO connection."""
        try:
            # Initialize MinIO storage service
            self.storage_service = MinIOService(self.storage_config)
            await self.storage_service.initialize()

            # Ensure required buckets exist
            await self.storage_service.ensure_bucket(self.models_bucket)
            await self.storage_service.ensure_bucket(self.metadata_bucket)

            # Load existing registry metadata
            await self._load_registry_metadata()

            # Start background tasks
            asyncio.create_task(self._periodic_cache_refresh())

            logger.info(
                f"Model registry initialized with buckets: "
                f"{self.models_bucket}, {self.metadata_bucket}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize model registry: {e}")
            raise ModelError(
                f"Registry initialization failed: {e}", operation="initialize"
            ) from e

    async def register_model(
        self,
        model_path: Path,
        model_name: str,
        version: str,
        metrics: dict[str, float],
        training_config: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
    ) -> ModelVersion:
        """Register a new model version in MinIO storage.

        Args:
            model_path: Local path to model file
            model_name: Model name/identifier
            version: Model version string
            metrics: Model performance metrics
            training_config: Training configuration used
            tags: Optional tags for organization

        Returns:
            Registered model version information

        Raises:
            ModelError: If registration fails
        """
        start_time = time.time()

        try:
            if not model_path.exists():
                raise ValidationError(f"Model file not found: {model_path}")

            # Generate storage key
            model_key = self._generate_model_key(model_name, version)
            metadata_key = self._generate_metadata_key(model_name, version)

            # Check if version already exists
            if await self.storage_service.object_exists(self.models_bucket, model_key) and not self.registry_config.get("allow_overwrites", False):
                raise ValidationError(
                    f"Model version {model_name}:{version} already exists"
                )

            # Prepare model metadata
            model_metadata = {
                "model_name": model_name,
                "version": version,
                "created_at": time.time(),
                "metrics": metrics,
                "training_config": training_config or {},
                "tags": tags or {},
                "file_size": model_path.stat().st_size,
                "file_name": model_path.name,
                "storage_key": model_key,
                "registration_time": time.time() - start_time,
            }

            # Upload model artifact
            model_upload_request = UploadRequest(
                bucket=self.models_bucket,
                key=model_key,
                file_path=model_path,
                object_type=ObjectType.MODEL_ARTIFACT,
                metadata={
                    "model-name": model_name,
                    "version": version,
                    "content-type": "application/octet-stream",
                },
                tags={
                    "model-name": model_name,
                    "version": version,
                    "type": "model-artifact",
                    **(tags or {}),
                },
                compression=(
                    self.storage_config.default_compression
                    if self.enable_compression
                    else None
                ),
                storage_class="STANDARD",
            )

            model_response = await self.storage_service.upload_object(
                model_upload_request
            )

            if not model_response.success:
                raise ModelError(
                    f"Failed to upload model artifact: {model_response.error_message}",
                    model_name=model_name,
                    operation="register_model",
                )

            # Update metadata with upload results
            model_metadata.update(
                {
                    "etag": model_response.etag,
                    "version_id": model_response.version_id,
                    "upload_time": model_response.upload_time_seconds,
                    "compressed_size": model_response.total_size,
                    "compression_ratio": (
                        model_response.compression_ratio
                        if hasattr(model_response, "compression_ratio")
                        else None
                    ),
                }
            )

            # Upload metadata
            metadata_json = json.dumps(model_metadata, indent=2, default=str)
            metadata_upload_request = UploadRequest(
                bucket=self.metadata_bucket,
                key=metadata_key,
                data=metadata_json.encode("utf-8"),
                content_type="application/json",
                object_type=ObjectType.METADATA,
                metadata={
                    "model-name": model_name,
                    "version": version,
                    "metadata-type": "model-registration",
                },
                tags={
                    "model-name": model_name,
                    "version": version,
                    "type": "metadata",
                },
            )

            metadata_response = await self.storage_service.upload_object(
                metadata_upload_request
            )

            if not metadata_response.success:
                logger.warning(
                    f"Failed to upload metadata for {model_name}:{version}: "
                    f"{metadata_response.error_message}"
                )

            # Update local cache and statistics
            cache_key = f"{model_name}:{version}"
            self.metadata_cache[cache_key] = model_metadata
            self.stats["successful_uploads"] += 1
            self.stats["total_models"] += 1
            self.stats["total_storage_size"] += model_response.total_size

            # Create ModelVersion object
            model_version = ModelVersion(
                model_id=f"{model_name}_{version}",
                version=version,
                model_path=model_path,  # Keep original path for compatibility
                config_path=model_path.with_suffix(".json"),
                metadata_path=model_path.with_suffix(".metadata.json"),
                accuracy_score=metrics.get("accuracy", 0.0),
                latency_p95_ms=metrics.get("latency_p95_ms", 0.0),
                throughput_fps=metrics.get("throughput_fps", 0.0),
                model_size_mb=model_response.total_size / (1024 * 1024),
                storage_key=model_key,
                storage_bucket=self.models_bucket,
            )

            registration_time = time.time() - start_time

            logger.info(
                f"Successfully registered model {model_name}:{version} "
                f"({model_response.total_size / 1024 / 1024:.1f}MB) "
                f"in {registration_time:.3f}s"
            )

            return model_version

        except Exception as e:
            self.stats["failed_uploads"] += 1
            logger.error(f"Failed to register model {model_name}:{version}: {e}")
            raise ModelError(
                f"Model registration failed: {e}",
                model_name=model_name,
                operation="register_model",
            ) from e

    async def get_model(
        self, model_name: str, version: str, download_path: Path | None = None
    ) -> ModelVersion:
        """Retrieve model from storage.

        Args:
            model_name: Model name
            version: Model version
            download_path: Optional local path to download model

        Returns:
            Model version information

        Raises:
            ModelError: If model not found or retrieval fails
        """
        try:
            # Get model metadata
            metadata = await self._get_model_metadata(model_name, version)

            if not metadata:
                raise ResourceNotFoundError("model", f"{model_name}:{version}")

            # Download model if path specified
            local_model_path = download_path
            if download_path:
                model_key = metadata["storage_key"]

                download_request = DownloadRequest(
                    bucket=self.models_bucket,
                    key=model_key,
                    file_path=download_path,
                    validate_checksum=True,
                    decompress=True,
                )

                await self.storage_service.download_object(download_request)

                logger.info(
                    f"Downloaded model {model_name}:{version} to {download_path}"
                )
            else:
                # Use a temporary path reference
                local_model_path = Path(f"temp/{model_name}_{version}.pt")

            # Create ModelVersion object
            model_version = ModelVersion(
                model_id=f"{model_name}_{version}",
                version=version,
                model_path=local_model_path,
                config_path=local_model_path.with_suffix(".json"),
                metadata_path=local_model_path.with_suffix(".metadata.json"),
                accuracy_score=metadata.get("metrics", {}).get("accuracy", 0.0),
                latency_p95_ms=metadata.get("metrics", {}).get("latency_p95_ms", 0.0),
                throughput_fps=metadata.get("metrics", {}).get("throughput_fps", 0.0),
                model_size_mb=metadata.get("compressed_size", 0) / (1024 * 1024),
                storage_key=metadata["storage_key"],
                storage_bucket=self.models_bucket,
            )

            return model_version

        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ModelError(
                f"Failed to retrieve model {model_name}:{version}: {e}",
                model_name=model_name,
                operation="get_model",
            ) from e

    async def list_models(
        self,
        model_name: str | None = None,
        limit: int = 50,
        tags: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """List available models.

        Args:
            model_name: Filter by model name
            limit: Maximum number of results
            tags: Filter by tags

        Returns:
            List of model information dictionaries
        """
        try:
            from .models import ListObjectsRequest

            # List metadata objects
            list_request = ListObjectsRequest(
                bucket=self.metadata_bucket,
                prefix=f"models/{model_name}/" if model_name else "models/",
                max_keys=limit,
                include_metadata=True,
            )

            objects = await self.storage_service.list_objects(list_request)

            models = []
            for obj in objects:
                try:
                    # Parse metadata from object
                    if obj.metadata:
                        metadata = json.loads(
                            await self.storage_service.download_object(
                                DownloadRequest(
                                    bucket=self.metadata_bucket, key=obj.key
                                )
                            )
                        )

                        # Apply tag filters if specified
                        if tags:
                            obj_tags = metadata.get("tags", {})
                            if not all(obj_tags.get(k) == v for k, v in tags.items()):
                                continue

                        models.append(
                            {
                                "model_name": metadata.get("model_name"),
                                "version": metadata.get("version"),
                                "created_at": metadata.get("created_at"),
                                "metrics": metadata.get("metrics", {}),
                                "file_size": metadata.get("file_size", 0),
                                "compressed_size": metadata.get("compressed_size", 0),
                                "tags": metadata.get("tags", {}),
                                "storage_key": metadata.get("storage_key"),
                            }
                        )

                except Exception as e:
                    logger.warning(f"Failed to parse metadata for {obj.key}: {e}")
                    continue

            # Sort by creation time (newest first)
            models.sort(key=lambda x: x.get("created_at", 0), reverse=True)

            return models[:limit]

        except Exception as e:
            raise ModelError(
                f"Failed to list models: {e}", operation="list_models"
            ) from e

    async def delete_model(
        self, model_name: str, version: str, _force: bool = False
    ) -> bool:
        """Delete model and its metadata.

        Args:
            model_name: Model name
            version: Model version
            force: Force deletion even if in use

        Returns:
            True if successful

        Raises:
            ModelError: If deletion fails
        """
        try:
            model_key = self._generate_model_key(model_name, version)
            metadata_key = self._generate_metadata_key(model_name, version)

            # Check if model exists
            if not await self.storage_service.object_exists(
                self.models_bucket, model_key
            ):
                raise ResourceNotFoundError("model", f"{model_name}:{version}")

            # Delete model artifact
            await self.storage_service.delete_object(self.models_bucket, model_key)

            # Delete metadata
            try:
                await self.storage_service.delete_object(
                    self.metadata_bucket, metadata_key
                )
            except Exception as e:
                logger.warning(
                    f"Failed to delete metadata for {model_name}:{version}: {e}"
                )

            # Remove from cache
            cache_key = f"{model_name}:{version}"
            if cache_key in self.metadata_cache:
                del self.metadata_cache[cache_key]

            # Update statistics
            self.stats["total_models"] -= 1

            logger.info(f"Deleted model {model_name}:{version}")

            return True

        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ModelError(
                f"Failed to delete model {model_name}:{version}: {e}",
                model_name=model_name,
                operation="delete_model",
            ) from e

    async def get_model_metadata(self, model_name: str, version: str) -> dict[str, Any]:
        """Get detailed metadata for a model.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            Model metadata dictionary

        Raises:
            ModelError: If metadata not found
        """
        metadata = await self._get_model_metadata(model_name, version)

        if not metadata:
            raise ResourceNotFoundError("model_metadata", f"{model_name}:{version}")

        return metadata

    async def generate_download_url(
        self, model_name: str, version: str, expires_in_seconds: int = 3600
    ) -> str:
        """Generate presigned download URL for model.

        Args:
            model_name: Model name
            version: Model version
            expires_in_seconds: URL expiration time

        Returns:
            Presigned download URL

        Raises:
            ModelError: If URL generation fails
        """
        try:
            model_key = self._generate_model_key(model_name, version)

            # Check if model exists
            if not await self.storage_service.object_exists(
                self.models_bucket, model_key
            ):
                raise ResourceNotFoundError("model", f"{model_name}:{version}")

            url = await self.storage_service.generate_presigned_url(
                self.models_bucket, model_key, expires_in_seconds, method="GET"
            )

            logger.info(
                f"Generated download URL for {model_name}:{version} "
                f"(expires in {expires_in_seconds}s)"
            )

            return url

        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise ModelError(
                f"Failed to generate download URL for {model_name}:{version}: {e}",
                model_name=model_name,
                operation="generate_download_url",
            ) from e

    # Private Helper Methods

    def _generate_model_key(self, model_name: str, version: str) -> str:
        """Generate storage key for model artifact."""
        return f"models/{model_name}/{version}/model.pt"

    def _generate_metadata_key(self, model_name: str, version: str) -> str:
        """Generate storage key for model metadata."""
        return f"models/{model_name}/{version}/metadata.json"

    async def _get_model_metadata(
        self, model_name: str, version: str
    ) -> dict[str, Any] | None:
        """Get model metadata with caching."""
        cache_key = f"{model_name}:{version}"

        # Check local cache first
        if cache_key in self.metadata_cache:
            cached_time = self.metadata_cache[cache_key].get("_cached_at", 0)
            if time.time() - cached_time < self.cache_ttl:
                self.stats["cache_hits"] += 1
                return self.metadata_cache[cache_key]

        self.stats["cache_misses"] += 1

        # Load from storage
        try:
            metadata_key = self._generate_metadata_key(model_name, version)

            from .models import DownloadRequest

            download_request = DownloadRequest(
                bucket=self.metadata_bucket, key=metadata_key
            )

            metadata_json = await self.storage_service.download_object(download_request)

            if isinstance(metadata_json, str):
                metadata_json = metadata_json.encode("utf-8")

            metadata = json.loads(metadata_json.decode("utf-8"))

            # Cache with timestamp
            metadata["_cached_at"] = time.time()
            self.metadata_cache[cache_key] = metadata

            return metadata

        except Exception as e:
            logger.warning(f"Failed to load metadata for {model_name}:{version}: {e}")
            return None

    async def _load_registry_metadata(self) -> None:
        """Load registry-wide metadata and statistics."""
        try:
            # List all models to update statistics
            models = await self.list_models(limit=1000)

            self.stats["total_models"] = len(models)
            self.stats["total_storage_size"] = sum(
                model.get("compressed_size", 0) for model in models
            )

            logger.info(
                f"Loaded registry metadata: {len(models)} models, "
                f"{self.stats['total_storage_size'] / 1024 / 1024:.1f}MB total"
            )

        except Exception as e:
            logger.warning(f"Failed to load registry metadata: {e}")

    async def _periodic_cache_refresh(self) -> None:
        """Background task to refresh metadata cache."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Remove expired cache entries
                current_time = time.time()
                expired_keys = [
                    key
                    for key, metadata in self.metadata_cache.items()
                    if current_time - metadata.get("_cached_at", 0) > self.cache_ttl
                ]

                for key in expired_keys:
                    del self.metadata_cache[key]

                if expired_keys:
                    logger.info(f"Removed {len(expired_keys)} expired cache entries")

            except Exception as e:
                logger.error(f"Cache refresh error: {e}")
                await asyncio.sleep(300)

    # Statistics and Health

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        storage_stats = (
            self.storage_service.get_stats() if self.storage_service else None
        )

        return {
            "registry_stats": self.stats.copy(),
            "cache_size": len(self.metadata_cache),
            "cache_hit_rate": (
                self.stats["cache_hits"]
                / max(1, self.stats["cache_hits"] + self.stats["cache_misses"])
            )
            * 100,
            "storage_stats": storage_stats.model_dump() if storage_stats else None,
            "buckets": {
                "models": self.models_bucket,
                "metadata": self.metadata_bucket,
            },
        }

    def get_health_status(self) -> dict[str, Any]:
        """Get health status."""
        storage_health = (
            self.storage_service.get_health_status()
            if self.storage_service
            else {"status": "unknown"}
        )

        return {
            "status": (
                "healthy" if storage_health.get("status") == "healthy" else "degraded"
            ),
            "total_models": self.stats["total_models"],
            "cache_entries": len(self.metadata_cache),
            "storage_health": storage_health,
            "buckets_accessible": storage_health.get("status") == "healthy",
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.storage_service:
                await self.storage_service.cleanup()

            logger.info("Model registry cleanup completed")

        except Exception as e:
            logger.error(f"Registry cleanup error: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
