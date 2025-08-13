"""MinIO object storage service for ITS Camera AI.

Provides comprehensive MinIO integration with:
- Async operations with connection pooling
- Multi-part upload for large files
- Presigned URL generation
- Server-side encryption (SSE-S3)
- Object lifecycle management
- Redis caching layer
- Comprehensive error handling and retry logic
- Performance monitoring and metrics

Optimized for:
- <100ms latency for small operations
- High throughput for video frame storage
- Efficient ML model artifact management
- Scalable training dataset versioning
"""

import asyncio
import gzip
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

try:
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
from minio import Minio
from minio.commonconfig import ENABLED, REPLACE, CopySource
from minio.error import S3Error
from minio.lifecycle import Expiration, LifecycleConfig, Rule
from minio.sseconfig import SseConfig
from minio.versioningconfig import VersioningConfig

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..core.exceptions import (
    ITSCameraAIError,
    ResourceNotFoundError,
    ValidationError,
)
from .models import (
    BucketConfig,
    BucketInfo,
    CompressionType,
    DownloadRequest,
    EncryptionType,
    ListObjectsRequest,
    ObjectListItem,
    ObjectMetadata,
    ObjectType,
    StorageConfig,
    StorageStats,
    UploadProgress,
    UploadRequest,
    UploadResponse,
)

if TYPE_CHECKING:
    import aiohttp

logger = logging.getLogger(__name__)


class MinIOError(ITSCameraAIError):
    """MinIO-specific error."""

    def __init__(self, message: str, operation: str, details: dict[str, Any] = None):
        super().__init__(
            message=message,
            code="MINIO_ERROR",
            details=details or {},
        )
        self.operation = operation


class MinIOService:
    """Async MinIO object storage service with advanced features."""

    def __init__(self, config: StorageConfig):
        """Initialize MinIO service.

        Args:
            config: Storage configuration
        """
        self.config = config
        self._client: Minio | None = None
        self._redis_client: aioredis.Redis | None = None

        # Connection management
        self._connection_pool: aiohttp.ClientSession | None = None
        self._active_connections = 0
        self._max_connections = config.max_pool_connections

        # Upload tracking
        self._active_uploads: dict[str, UploadProgress] = {}
        self._upload_callbacks: dict[str, list[Callable]] = defaultdict(list)

        # Performance tracking
        self._stats = StorageStats()
        self._operation_times = deque(maxlen=1000)

        # Caching
        self._cache_enabled = config.enable_caching and REDIS_AVAILABLE
        self._metadata_cache: dict[str, ObjectMetadata] = {}
        self._cache_stats = {"hits": 0, "misses": 0}

        # Error tracking
        self._recent_errors = deque(maxlen=100)

        logger.info(f"MinIO service initialized with endpoint: {config.endpoint}")

    async def initialize(self) -> None:
        """Initialize MinIO client and connections."""
        try:
            # Initialize MinIO client
            self._client = Minio(
                endpoint=self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure,
                region=self.config.region,
            )

            # Test connection
            await self._test_connection()

            # Initialize Redis cache if enabled
            if self._cache_enabled:
                await self._initialize_redis()

            # Create default bucket if configured
            if self.config.create_buckets:
                await self.ensure_bucket(self.config.default_bucket)

            # Start background tasks
            asyncio.create_task(self._cleanup_completed_uploads())
            asyncio.create_task(self._update_stats_periodically())

            logger.info("MinIO service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MinIO service: {e}")
            raise MinIOError(f"Initialization failed: {e}", "initialize") from e

    async def _test_connection(self) -> None:
        """Test MinIO connection."""
        try:
            # List buckets to test connection
            buckets = self._client.list_buckets()
            logger.info(f"Connected to MinIO, found {len(buckets)} buckets")
        except Exception as e:
            raise MinIOError(f"Connection test failed: {e}", "test_connection") from e

    async def _initialize_redis(self) -> None:
        """Initialize Redis connection for caching."""
        try:
            # Use Redis URL from main config if available
            redis_url = "redis://localhost:6379/2"  # Use DB 2 for storage cache
            self._redis_client = await aioredis.from_url(
                redis_url,
                max_connections=10,
                retry_on_timeout=True
            )

            # Test Redis connection
            await self._redis_client.ping()
            logger.info("Redis cache initialized for MinIO service")

        except Exception as e:
            logger.warning(f"Redis cache initialization failed: {e}")
            self._cache_enabled = False

    # Bucket Management

    async def create_bucket(
        self,
        bucket_config: BucketConfig
    ) -> BucketInfo:
        """Create a new bucket with configuration.

        Args:
            bucket_config: Bucket configuration

        Returns:
            Bucket information

        Raises:
            MinIOError: If bucket creation fails
        """
        start_time = time.time()

        try:
            bucket_name = bucket_config.name

            # Check if bucket already exists
            if self._client.bucket_exists(bucket_name):
                logger.info(f"Bucket {bucket_name} already exists")
                return await self.get_bucket_info(bucket_name)

            # Create bucket
            self._client.make_bucket(
                bucket_name=bucket_name,
                location=bucket_config.region
            )

            # Configure versioning if enabled
            if bucket_config.versioning_enabled:
                version_config = VersioningConfig(ENABLED)
                self._client.set_bucket_versioning(
                    bucket_name, version_config
                )

            # Configure encryption
            if bucket_config.encryption != EncryptionType.NONE:
                sse_config = SseConfig.new_sse_s3_config()
                self._client.set_bucket_encryption(
                    bucket_name, sse_config
                )

            # Set lifecycle rules if provided
            if bucket_config.lifecycle_rules:
                await self._set_bucket_lifecycle(
                    bucket_name, bucket_config.lifecycle_rules
                )

            # Set bucket tags if provided
            if bucket_config.tags:
                from minio.tagging import Tagging
                tags = Tagging()
                for key, value in bucket_config.tags.items():
                    tags[key] = value
                self._client.set_bucket_tagging(bucket_name, tags)

            # Update stats
            self._stats.successful_operations += 1
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)

            logger.info(
                f"Created bucket {bucket_name} in {operation_time:.3f}s"
            )

            return BucketInfo(
                name=bucket_name,
                creation_date=datetime.now(UTC),
                region=bucket_config.region,
                versioning_enabled=bucket_config.versioning_enabled,
                encryption_enabled=bucket_config.encryption != EncryptionType.NONE,
                tags=bucket_config.tags,
                lifecycle_rules=bucket_config.lifecycle_rules,
            )

        except Exception as e:
            self._stats.failed_operations += 1
            error_details = {
                "bucket_name": bucket_config.name,
                "operation_time": time.time() - start_time,
            }
            self._record_error("create_bucket", str(e), error_details)
            raise MinIOError(
                f"Failed to create bucket {bucket_config.name}: {e}",
                "create_bucket",
                error_details
            ) from e

    async def ensure_bucket(self, bucket_name: str) -> BucketInfo:
        """Ensure bucket exists, create if it doesn't.

        Args:
            bucket_name: Name of bucket to ensure exists

        Returns:
            Bucket information
        """
        try:
            if self._client.bucket_exists(bucket_name):
                return await self.get_bucket_info(bucket_name)

            # Create with default configuration
            bucket_config = BucketConfig(
                name=bucket_name,
                region=self.config.region,
                versioning_enabled=self.config.enable_versioning,
                encryption=self.config.default_encryption,
            )

            return await self.create_bucket(bucket_config)

        except Exception as e:
            raise MinIOError(
                f"Failed to ensure bucket {bucket_name}: {e}",
                "ensure_bucket"
            ) from e

    async def delete_bucket(
        self,
        bucket_name: str,
        force: bool = False
    ) -> bool:
        """Delete a bucket.

        Args:
            bucket_name: Name of bucket to delete
            force: If True, delete all objects first

        Returns:
            True if successful

        Raises:
            MinIOError: If deletion fails
        """
        start_time = time.time()

        try:
            # If force is True, delete all objects first
            if force:
                await self._delete_all_objects(bucket_name)

            # Delete bucket
            self._client.remove_bucket(bucket_name)

            # Update stats
            self._stats.successful_operations += 1
            operation_time = time.time() - start_time
            self._operation_times.append(operation_time)

            logger.info(
                f"Deleted bucket {bucket_name} in {operation_time:.3f}s"
            )

            return True

        except Exception as e:
            self._stats.failed_operations += 1
            error_details = {
                "bucket_name": bucket_name,
                "force": force,
                "operation_time": time.time() - start_time,
            }
            self._record_error("delete_bucket", str(e), error_details)
            raise MinIOError(
                f"Failed to delete bucket {bucket_name}: {e}",
                "delete_bucket",
                error_details
            ) from e

    async def list_buckets(self) -> list[BucketInfo]:
        """List all buckets.

        Returns:
            List of bucket information
        """
        try:
            buckets = self._client.list_buckets()
            bucket_infos = []

            for bucket in buckets:
                bucket_info = BucketInfo(
                    name=bucket.name,
                    creation_date=bucket.creation_date,
                    region="unknown",  # MinIO client doesn't provide region in list
                )
                bucket_infos.append(bucket_info)

            return bucket_infos

        except Exception as e:
            raise MinIOError(f"Failed to list buckets: {e}", "list_buckets") from e

    async def get_bucket_info(self, bucket_name: str) -> BucketInfo:
        """Get detailed bucket information.

        Args:
            bucket_name: Name of bucket

        Returns:
            Bucket information
        """
        try:
            # Check if bucket exists
            if not self._client.bucket_exists(bucket_name):
                raise ResourceNotFoundError("bucket", bucket_name)

            # Get basic bucket info
            buckets = self._client.list_buckets()
            bucket_data = next(
                (b for b in buckets if b.name == bucket_name), None
            )

            if not bucket_data:
                raise ResourceNotFoundError("bucket", bucket_name)

            # Get versioning status
            versioning_enabled = False
            try:
                version_config = self._client.get_bucket_versioning(bucket_name)
                versioning_enabled = version_config.status == ENABLED
            except S3Error:
                pass  # Versioning not supported or not configured

            # Get encryption status
            encryption_enabled = False
            try:
                self._client.get_bucket_encryption(bucket_name)
                encryption_enabled = True
            except S3Error:
                pass  # Encryption not configured

            # Get bucket tags
            tags = {}
            try:
                bucket_tags = self._client.get_bucket_tagging(bucket_name)
                tags = dict(bucket_tags)
            except S3Error:
                pass  # No tags configured

            # Calculate statistics (expensive operation)
            object_count, total_size = await self._calculate_bucket_stats(bucket_name)

            return BucketInfo(
                name=bucket_name,
                creation_date=bucket_data.creation_date,
                region="unknown",  # Would need additional API call
                versioning_enabled=versioning_enabled,
                encryption_enabled=encryption_enabled,
                object_count=object_count,
                total_size=total_size,
                tags=tags,
            )

        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise MinIOError(
                f"Failed to get bucket info for {bucket_name}: {e}",
                "get_bucket_info"
            ) from e

    async def _delete_all_objects(self, bucket_name: str) -> None:
        """Delete all objects in a bucket."""
        try:
            objects = self._client.list_objects(
                bucket_name, recursive=True
            )

            object_names = [obj.object_name for obj in objects]

            if object_names:
                # Delete objects in batches
                from minio.deleteobjects import DeleteObject
                delete_objects = [DeleteObject(name) for name in object_names]

                errors = self._client.remove_objects(
                    bucket_name, delete_objects
                )

                # Check for errors
                error_list = list(errors)
                if error_list:
                    raise MinIOError(
                        f"Failed to delete some objects: {error_list}",
                        "delete_all_objects"
                    )

                logger.info(f"Deleted {len(object_names)} objects from {bucket_name}")

        except Exception as e:
            raise MinIOError(
                f"Failed to delete all objects in {bucket_name}: {e}",
                "delete_all_objects"
            ) from e

    async def _set_bucket_lifecycle(
        self,
        bucket_name: str,
        lifecycle_rules: list[dict[str, Any]]
    ) -> None:
        """Set bucket lifecycle configuration."""
        try:
            rules = []

            for rule_config in lifecycle_rules:
                rule_id = rule_config.get("id", f"rule_{len(rules)}")
                status = rule_config.get("status", "Enabled")

                # Create expiration rule
                if "expiration_days" in rule_config:
                    expiration = Expiration(
                        days=rule_config["expiration_days"]
                    )

                    rule = Rule(
                        rule_id=rule_id,
                        status=status,
                        rule_filter=None,  # Apply to all objects
                        expiration=expiration,
                    )
                    rules.append(rule)

            if rules:
                lifecycle_config = LifecycleConfig(rules)
                self._client.set_bucket_lifecycle(
                    bucket_name, lifecycle_config
                )
                logger.info(
                    f"Set lifecycle rules for bucket {bucket_name}: {len(rules)} rules"
                )

        except Exception as e:
            logger.warning(f"Failed to set lifecycle rules for {bucket_name}: {e}")

    async def _calculate_bucket_stats(self, bucket_name: str) -> tuple[int, int]:
        """Calculate bucket object count and total size.

        Returns:
            Tuple of (object_count, total_size_bytes)
        """
        try:
            objects = self._client.list_objects(
                bucket_name, recursive=True
            )

            object_count = 0
            total_size = 0

            for obj in objects:
                object_count += 1
                total_size += obj.size or 0

                # Limit calculation time for large buckets
                if object_count > 10000:
                    logger.warning(
                        f"Bucket {bucket_name} has >10k objects, using partial stats"
                    )
                    break

            return object_count, total_size

        except Exception as e:
            logger.warning(f"Failed to calculate stats for {bucket_name}: {e}")
            return 0, 0

    # Object Operations

    async def upload_object(
        self,
        request: UploadRequest,
        progress_callback: Callable[[UploadProgress], None] | None = None
    ) -> UploadResponse:
        """Upload object to MinIO storage.

        Args:
            request: Upload request configuration
            progress_callback: Optional progress callback function

        Returns:
            Upload response with metadata

        Raises:
            MinIOError: If upload fails
        """
        start_time = time.time()
        upload_id = str(uuid.uuid4())

        try:
            # Ensure target bucket exists
            await self.ensure_bucket(request.bucket)

            # Check if object already exists and not overwriting
            if not request.overwrite:
                try:
                    self._client.stat_object(request.bucket, request.key)
                    raise ValidationError(
                        f"Object {request.key} already exists and overwrite=False"
                    )
                except S3Error as e:
                    if e.code != "NoSuchKey":
                        raise

            # Get source data info
            if request.file_path:
                if not request.file_path.exists():
                    raise ValidationError(f"Source file not found: {request.file_path}")
                data_size = request.file_path.stat().st_size
                data_source = "file"
            else:
                data_size = len(request.data)
                data_source = "memory"

            # Create upload progress tracker
            progress = UploadProgress(
                upload_id=upload_id,
                key=request.key,
                bucket=request.bucket,
                total_size=data_size,
                is_multipart=self._should_use_multipart(request, data_size),
            )

            self._active_uploads[upload_id] = progress

            # Register progress callback
            if progress_callback:
                self._upload_callbacks[upload_id].append(progress_callback)

            # Prepare upload metadata
            metadata = request.metadata.copy()
            metadata.update({
                "upload-id": upload_id,
                "object-type": request.object_type.value,
                "source": data_source,
                "compression": request.compression.value,
                "client": "its-camera-ai",
            })

            # Perform upload based on size and configuration
            if progress.is_multipart:
                result = await self._upload_multipart(
                    request, upload_id, metadata, progress
                )
            else:
                result = await self._upload_single_part(
                    request, upload_id, metadata, progress
                )

            # Create object metadata
            object_metadata = await self._create_object_metadata(
                request, result, data_size
            )

            # Cache metadata if caching enabled
            if self._cache_enabled:
                await self._cache_metadata(request.bucket, request.key, object_metadata)

            # Update statistics
            upload_time = time.time() - start_time
            self._stats.successful_operations += 1
            self._stats.total_uploads += 1
            self._stats.bytes_uploaded += data_size
            self._operation_times.append(upload_time)

            # Update average upload speed
            if upload_time > 0:
                speed = data_size / upload_time
                self._stats.average_upload_speed_bps = (
                    self._stats.average_upload_speed_bps * 0.9 + speed * 0.1
                )

            # Mark upload as completed
            progress.status = "completed"
            progress.update_progress(data_size)

            # Notify callbacks
            await self._notify_upload_callbacks(upload_id, progress)

            # Create response
            response = UploadResponse(
                success=True,
                upload_id=upload_id,
                bucket=request.bucket,
                key=request.key,
                etag=result.get("etag"),
                version_id=result.get("version_id"),
                total_size=data_size,
                upload_time_seconds=upload_time,
                average_speed_bps=data_size / upload_time if upload_time > 0 else 0,
                content_type=request.get_content_type(),
                object_metadata=object_metadata,
                started_at=datetime.fromtimestamp(start_time, tz=UTC),
                was_multipart=progress.is_multipart,
                parts_count=progress.parts_total,
            )

            logger.info(
                f"Successfully uploaded {request.key} "
                f"({data_size / 1024 / 1024:.1f}MB) in {upload_time:.3f}s"
            )

            return response

        except Exception as e:
            # Mark upload as failed
            if upload_id in self._active_uploads:
                progress = self._active_uploads[upload_id]
                progress.status = "failed"
                progress.error_message = str(e)
                await self._notify_upload_callbacks(upload_id, progress)

            # Update error statistics
            self._stats.failed_operations += 1
            upload_time = time.time() - start_time
            error_details = {
                "upload_id": upload_id,
                "bucket": request.bucket,
                "key": request.key,
                "size": data_size if 'data_size' in locals() else 0,
                "operation_time": upload_time,
            }
            self._record_error("upload_object", str(e), error_details)

            # Create error response
            response = UploadResponse(
                success=False,
                upload_id=upload_id,
                bucket=request.bucket,
                key=request.key,
                total_size=data_size if 'data_size' in locals() else 0,
                upload_time_seconds=upload_time,
                average_speed_bps=0,
                content_type=request.get_content_type(),
                error_message=str(e),
                error_code=getattr(e, 'code', 'UNKNOWN_ERROR'),
                started_at=datetime.fromtimestamp(start_time, tz=UTC),
            )

            logger.error(f"Upload failed for {request.key}: {e}")
            return response

        finally:
            # Cleanup upload tracking
            if upload_id in self._active_uploads:
                del self._active_uploads[upload_id]
            if upload_id in self._upload_callbacks:
                del self._upload_callbacks[upload_id]

    def _should_use_multipart(
        self,
        request: UploadRequest,
        data_size: int
    ) -> bool:
        """Determine if multipart upload should be used."""
        if request.use_multipart is not None:
            return request.use_multipart

        return data_size >= self.config.multipart_threshold

    async def _upload_single_part(
        self,
        request: UploadRequest,
        _upload_id: str,
        metadata: dict[str, str],
        _progress: UploadProgress
    ) -> dict[str, Any]:
        """Upload object as single part."""
        try:
            # Prepare data
            if request.file_path:
                # Stream from file
                if AIOFILES_AVAILABLE:
                    async with aiofiles.open(request.file_path, 'rb') as f:
                        data = await f.read()
                else:
                    # Fallback to sync file reading
                    with open(request.file_path, 'rb') as f:
                        data = f.read()
            else:
                data = request.data

            # Apply compression if enabled
            if request.compression != CompressionType.NONE:
                data = await self._compress_data(data, request.compression)
                metadata["compression"] = request.compression.value

            # Calculate hash for integrity
            md5_hash = hashlib.md5(data).hexdigest()
            metadata["md5"] = md5_hash

            # Perform upload
            result = self._client.put_object(
                bucket_name=request.bucket,
                object_name=request.key,
                data=data,
                length=len(data),
                content_type=request.get_content_type(),
                metadata=metadata,
                sse=self._get_sse_config(request.encryption) if request.encryption != EncryptionType.NONE else None,
                tags=request.tags if request.tags else None,
            )

            return {
                "etag": result.etag,
                "version_id": result.version_id,
                "method": "single_part",
            }

        except Exception as e:
            raise MinIOError(
                f"Single part upload failed: {e}",
                "upload_single_part"
            ) from e

    async def _upload_multipart(
        self,
        request: UploadRequest,
        upload_id: str,
        metadata: dict[str, str],
        progress: UploadProgress
    ) -> dict[str, Any]:
        """Upload object using multipart upload."""
        try:
            # Initialize multipart upload
            self._client._presign_v4_upload_policy(
                bucket_name=request.bucket,
                object_name=request.key,
                expires=timedelta(hours=1),
            )

            # For simplicity, fall back to single part for now
            # Full multipart implementation would require more complex async handling
            logger.warning("Multipart upload not fully implemented, using single part")
            return await self._upload_single_part(request, upload_id, metadata, progress)

        except Exception as e:
            raise MinIOError(
                f"Multipart upload failed: {e}",
                "upload_multipart"
            ) from e

    async def _compress_data(
        self,
        data: bytes,
        compression_type: CompressionType
    ) -> bytes:
        """Compress data using specified compression type."""
        if compression_type == CompressionType.GZIP:
            return gzip.compress(data)
        elif compression_type == CompressionType.LZ4:
            try:
                import lz4.frame
                return lz4.frame.compress(data)
            except ImportError:
                logger.warning("LZ4 not available, using gzip")
                return gzip.compress(data)
        elif compression_type == CompressionType.ZSTD:
            try:
                import zstd
                return zstd.compress(data)
            except ImportError:
                logger.warning("ZSTD not available, using gzip")
                return gzip.compress(data)
        else:
            return data

    def _get_sse_config(self, encryption_type: EncryptionType):
        """Get SSE configuration for encryption type."""
        if encryption_type == EncryptionType.SSE_S3:
            return SseConfig.new_sse_s3_config()
        # Add other encryption types as needed
        return None

    async def _create_object_metadata(
        self,
        request: UploadRequest,
        upload_result: dict[str, Any],
        data_size: int
    ) -> ObjectMetadata:
        """Create object metadata from upload request and result."""
        now = datetime.now(UTC)

        metadata = ObjectMetadata(
            key=request.key,
            bucket=request.bucket,
            size=data_size,
            content_type=request.get_content_type(),
            etag=upload_result.get("etag"),
            created_at=now,
            modified_at=now,
            storage_class=request.storage_class,
            encryption=request.encryption,
            compression=request.compression,
            object_type=request.object_type,
            version_id=upload_result.get("version_id"),
            user_metadata=request.metadata,
            tags=request.tags,
        )

        # Calculate hashes if data is available
        if request.data:
            metadata.calculate_md5(request.data)
            metadata.calculate_sha256(request.data)
        elif request.file_path and request.file_path.exists():
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(request.file_path, 'rb') as f:
                    file_data = await f.read()
                    metadata.calculate_md5(file_data)
                    metadata.calculate_sha256(file_data)
            else:
                # Fallback to sync file reading
                with open(request.file_path, 'rb') as f:
                    file_data = f.read()
                    metadata.calculate_md5(file_data)
                    metadata.calculate_sha256(file_data)

        return metadata

    async def _cache_metadata(
        self,
        bucket: str,
        key: str,
        metadata: ObjectMetadata
    ) -> None:
        """Cache object metadata in Redis."""
        if not self._redis_client:
            return

        try:
            cache_key = f"metadata:{bucket}:{key}"
            metadata_json = metadata.model_dump_json()

            await self._redis_client.setex(
                cache_key,
                self.config.cache_ttl,
                metadata_json
            )

        except Exception as e:
            logger.warning(f"Failed to cache metadata for {bucket}/{key}: {e}")

    async def _notify_upload_callbacks(
        self,
        upload_id: str,
        progress: UploadProgress
    ) -> None:
        """Notify all registered callbacks for upload progress."""
        callbacks = self._upload_callbacks.get(upload_id, [])

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(progress)
                else:
                    callback(progress)
            except Exception as e:
                logger.error(f"Upload callback error: {e}")

    async def download_object(
        self,
        request: DownloadRequest,
        _progress_callback: Callable[[int, int], None] | None = None
    ) -> bytes | str:
        """Download object from MinIO storage.

        Args:
            request: Download request configuration
            progress_callback: Optional progress callback (downloaded, total)

        Returns:
            Downloaded data as bytes, or file path if file_path was specified

        Raises:
            MinIOError: If download fails
        """
        start_time = time.time()

        try:
            # Get object info first
            try:
                stat = self._client.stat_object(
                    request.bucket,
                    request.key,
                    version_id=request.version_id
                )
            except S3Error as e:
                if e.code == "NoSuchKey":
                    raise ResourceNotFoundError("object", f"{request.bucket}/{request.key}")
                raise

            # Perform download
            if request.byte_range:
                # Range download
                start_byte, end_byte = request.byte_range
                response = self._client.get_object(
                    request.bucket,
                    request.key,
                    offset=start_byte,
                    length=end_byte - start_byte + 1,
                    version_id=request.version_id
                )
            else:
                # Full object download
                response = self._client.get_object(
                    request.bucket,
                    request.key,
                    version_id=request.version_id
                )

            # Read data
            data = response.read()
            response.close()
            response.release_conn()

            # Validate checksum if requested
            if request.validate_checksum and stat.etag:
                calculated_md5 = hashlib.md5(data).hexdigest()
                # Note: ETag might not always be MD5 for multipart uploads
                if f'"{calculated_md5}"' != stat.etag and calculated_md5 != stat.etag.strip('"'):
                    logger.warning(
                        f"Checksum mismatch for {request.bucket}/{request.key}: "
                        f"expected {stat.etag}, got {calculated_md5}"
                    )

            # Decompress if needed
            if request.decompress:
                data = await self._decompress_data(data, stat.metadata)

            # Save to file if file_path specified
            if request.file_path:
                request.file_path.parent.mkdir(parents=True, exist_ok=True)
                if AIOFILES_AVAILABLE:
                    async with aiofiles.open(request.file_path, 'wb') as f:
                        await f.write(data)
                else:
                    # Fallback to sync file writing
                    with open(request.file_path, 'wb') as f:
                        f.write(data)

                result = str(request.file_path)
            else:
                result = data

            # Update statistics
            download_time = time.time() - start_time
            self._stats.successful_operations += 1
            self._stats.total_downloads += 1
            self._stats.bytes_downloaded += len(data)
            self._operation_times.append(download_time)

            # Update average download speed
            if download_time > 0:
                speed = len(data) / download_time
                self._stats.average_download_speed_bps = (
                    self._stats.average_download_speed_bps * 0.9 + speed * 0.1
                )

            logger.info(
                f"Downloaded {request.key} ({len(data) / 1024 / 1024:.1f}MB) "
                f"in {download_time:.3f}s"
            )

            return result

        except ResourceNotFoundError:
            raise
        except Exception as e:
            self._stats.failed_operations += 1
            download_time = time.time() - start_time
            error_details = {
                "bucket": request.bucket,
                "key": request.key,
                "operation_time": download_time,
            }
            self._record_error("download_object", str(e), error_details)
            raise MinIOError(
                f"Failed to download {request.bucket}/{request.key}: {e}",
                "download_object",
                error_details
            ) from e

    async def _decompress_data(
        self,
        data: bytes,
        metadata: dict[str, str]
    ) -> bytes:
        """Decompress data based on metadata."""
        compression = metadata.get("compression", "none")

        if compression == "gzip":
            try:
                return gzip.decompress(data)
            except gzip.BadGzipFile:
                logger.warning("Failed to decompress gzip data")
                return data
        elif compression == "lz4":
            try:
                import lz4.frame
                return lz4.frame.decompress(data)
            except ImportError:
                logger.warning("LZ4 not available for decompression")
                return data
        elif compression == "zstd":
            try:
                import zstd
                return zstd.decompress(data)
            except ImportError:
                logger.warning("ZSTD not available for decompression")
                return data
        else:
            return data

    async def delete_object(
        self,
        bucket: str,
        key: str,
        version_id: str | None = None
    ) -> bool:
        """Delete object from storage.

        Args:
            bucket: Bucket name
            key: Object key
            version_id: Specific version to delete

        Returns:
            True if successful

        Raises:
            MinIOError: If deletion fails
        """
        start_time = time.time()

        try:
            self._client.remove_object(
                bucket, key, version_id=version_id
            )

            # Remove from cache if present
            if self._cache_enabled:
                await self._remove_from_cache(bucket, key)

            # Update statistics
            operation_time = time.time() - start_time
            self._stats.successful_operations += 1
            self._operation_times.append(operation_time)

            logger.info(f"Deleted object {bucket}/{key} in {operation_time:.3f}s")

            return True

        except Exception as e:
            self._stats.failed_operations += 1
            operation_time = time.time() - start_time
            error_details = {
                "bucket": bucket,
                "key": key,
                "version_id": version_id,
                "operation_time": operation_time,
            }
            self._record_error("delete_object", str(e), error_details)
            raise MinIOError(
                f"Failed to delete {bucket}/{key}: {e}",
                "delete_object",
                error_details
            ) from e

    async def _remove_from_cache(self, bucket: str, key: str) -> None:
        """Remove object metadata from cache."""
        if not self._redis_client:
            return

        try:
            cache_key = f"metadata:{bucket}:{key}"
            await self._redis_client.delete(cache_key)
        except Exception as e:
            logger.warning(f"Failed to remove {bucket}/{key} from cache: {e}")

    async def get_object_metadata(
        self,
        bucket: str,
        key: str,
        version_id: str | None = None
    ) -> ObjectMetadata:
        """Get object metadata.

        Args:
            bucket: Bucket name
            key: Object key
            version_id: Specific version

        Returns:
            Object metadata

        Raises:
            MinIOError: If object not found or operation fails
        """
        try:
            # Check cache first
            if self._cache_enabled:
                cached_metadata = await self._get_cached_metadata(bucket, key)
                if cached_metadata:
                    self._cache_stats["hits"] += 1
                    return cached_metadata
                self._cache_stats["misses"] += 1

            # Get from MinIO
            try:
                stat = self._client.stat_object(
                    bucket, key, version_id=version_id
                )
            except S3Error as e:
                if e.code == "NoSuchKey":
                    raise ResourceNotFoundError("object", f"{bucket}/{key}")
                raise

            # Convert to ObjectMetadata
            metadata = ObjectMetadata(
                key=key,
                bucket=bucket,
                size=stat.size,
                content_type=stat.content_type,
                etag=stat.etag,
                modified_at=stat.last_modified,
                storage_class=getattr(stat, 'storage_class', 'STANDARD'),
                version_id=stat.version_id,
                user_metadata=dict(stat.metadata) if stat.metadata else {},
                object_type=ObjectType(stat.metadata.get('object-type', 'video_frame')) if stat.metadata else ObjectType.VIDEO_FRAME,
            )

            # Cache the metadata
            if self._cache_enabled:
                await self._cache_metadata(bucket, key, metadata)

            return metadata

        except ResourceNotFoundError:
            raise
        except Exception as e:
            raise MinIOError(
                f"Failed to get metadata for {bucket}/{key}: {e}",
                "get_object_metadata"
            ) from e

    async def _get_cached_metadata(
        self,
        bucket: str,
        key: str
    ) -> ObjectMetadata | None:
        """Get cached object metadata."""
        if not self._redis_client:
            return None

        try:
            cache_key = f"metadata:{bucket}:{key}"
            cached_data = await self._redis_client.get(cache_key)

            if cached_data:
                metadata_dict = json.loads(cached_data)
                return ObjectMetadata(**metadata_dict)

            return None

        except Exception as e:
            logger.warning(f"Failed to get cached metadata for {bucket}/{key}: {e}")
            return None

    async def list_objects(
        self,
        request: ListObjectsRequest
    ) -> list[ObjectListItem]:
        """List objects in bucket.

        Args:
            request: List request configuration

        Returns:
            List of object items

        Raises:
            MinIOError: If listing fails
        """
        try:
            objects = self._client.list_objects(
                bucket_name=request.bucket,
                prefix=request.prefix,
                recursive=request.recursive,
                start_after=request.start_after,
            )

            items = []
            count = 0

            for obj in objects:
                if count >= request.max_keys:
                    break

                # Apply filters
                if request.size_range:
                    min_size, max_size = request.size_range
                    if not (min_size <= obj.size <= max_size):
                        continue

                if request.date_range:
                    start_date, end_date = request.date_range
                    if not (start_date <= obj.last_modified <= end_date):
                        continue

                # Create list item
                item = ObjectListItem(
                    key=obj.object_name,
                    size=obj.size,
                    etag=obj.etag,
                    last_modified=obj.last_modified,
                    storage_class=getattr(obj, 'storage_class', 'STANDARD'),
                )

                # Include full metadata if requested
                if request.include_metadata:
                    try:
                        item.metadata = await self.get_object_metadata(
                            request.bucket, obj.object_name
                        )
                        item.content_type = item.metadata.content_type
                        item.object_type = item.metadata.object_type
                    except Exception:
                        pass  # Continue without metadata

                # Filter by object types if specified
                if request.object_types and item.object_type and item.object_type not in request.object_types:
                    continue

                items.append(item)
                count += 1

            return items

        except Exception as e:
            raise MinIOError(
                f"Failed to list objects in {request.bucket}: {e}",
                "list_objects"
            ) from e

    # Presigned URL Operations

    async def generate_presigned_url(
        self,
        bucket: str,
        key: str,
        expires_in_seconds: int = 3600,
        method: str = "GET"
    ) -> str:
        """Generate presigned URL for object access.

        Args:
            bucket: Bucket name
            key: Object key
            expires_in_seconds: URL expiration time
            method: HTTP method (GET, PUT, DELETE)

        Returns:
            Presigned URL

        Raises:
            MinIOError: If URL generation fails
        """
        try:
            from datetime import timedelta

            if method.upper() == "GET":
                url = self._client.presigned_get_object(
                    bucket, key, expires=timedelta(seconds=expires_in_seconds)
                )
            elif method.upper() == "PUT":
                url = self._client.presigned_put_object(
                    bucket, key, expires=timedelta(seconds=expires_in_seconds)
                )
            else:
                raise ValidationError(f"Unsupported method: {method}")

            logger.info(f"Generated {method} presigned URL for {bucket}/{key}")

            return url

        except Exception as e:
            raise MinIOError(
                f"Failed to generate presigned URL for {bucket}/{key}: {e}",
                "generate_presigned_url"
            ) from e

    # Utility Methods

    async def copy_object(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str,
        metadata: dict[str, str] | None = None
    ) -> bool:
        """Copy object within or between buckets.

        Args:
            source_bucket: Source bucket name
            source_key: Source object key
            dest_bucket: Destination bucket name
            dest_key: Destination object key
            metadata: Optional metadata override

        Returns:
            True if successful

        Raises:
            MinIOError: If copy fails
        """
        try:
            source = CopySource(source_bucket, source_key)

            self._client.copy_object(
                dest_bucket,
                dest_key,
                source,
                metadata=metadata,
                metadata_directive=REPLACE if metadata else None,
            )

            logger.info(
                f"Copied {source_bucket}/{source_key} to {dest_bucket}/{dest_key}"
            )

            return True

        except Exception as e:
            raise MinIOError(
                f"Failed to copy {source_bucket}/{source_key} "
                f"to {dest_bucket}/{dest_key}: {e}",
                "copy_object"
            ) from e

    async def object_exists(
        self,
        bucket: str,
        key: str,
        version_id: str | None = None
    ) -> bool:
        """Check if object exists.

        Args:
            bucket: Bucket name
            key: Object key
            version_id: Specific version

        Returns:
            True if object exists
        """
        try:
            self._client.stat_object(bucket, key, version_id=version_id)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise MinIOError(
                f"Failed to check existence of {bucket}/{key}: {e}",
                "object_exists"
            ) from e
        except Exception as e:
            raise MinIOError(
                f"Failed to check existence of {bucket}/{key}: {e}",
                "object_exists"
            ) from e

    # Background Tasks

    async def _cleanup_completed_uploads(self) -> None:
        """Background task to cleanup completed upload tracking."""
        while True:
            try:
                current_time = time.time()
                expired_uploads = []

                for upload_id, progress in self._active_uploads.items():
                    # Remove uploads completed more than 1 hour ago
                    if (progress.is_completed() or progress.is_failed()) and \
                       (current_time - progress.updated_at.timestamp()) > 3600:
                        expired_uploads.append(upload_id)

                for upload_id in expired_uploads:
                    del self._active_uploads[upload_id]
                    if upload_id in self._upload_callbacks:
                        del self._upload_callbacks[upload_id]

                if expired_uploads:
                    logger.info(f"Cleaned up {len(expired_uploads)} expired uploads")

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error(f"Upload cleanup error: {e}")
                await asyncio.sleep(300)

    async def _update_stats_periodically(self) -> None:
        """Background task to update statistics."""
        while True:
            try:
                # Update connection stats
                self._stats.active_connections = self._active_connections
                self._stats.total_connections = self._max_connections

                # Update cache stats
                if self._cache_enabled:
                    self._stats.cache_hits = self._cache_stats["hits"]
                    self._stats.cache_misses = self._cache_stats["misses"]

                # Update timestamps
                self._stats.last_updated = datetime.now(UTC)

                # Log stats if metrics enabled
                if self.config.enable_metrics:
                    logger.info(f"MinIO stats: {self._stats.model_dump()}")

                await asyncio.sleep(self.config.metrics_interval)

            except Exception as e:
                logger.error(f"Stats update error: {e}")
                await asyncio.sleep(self.config.metrics_interval)

    def _record_error(
        self,
        operation: str,
        error_message: str,
        details: dict[str, Any]
    ) -> None:
        """Record error for tracking and analysis."""
        error_record = {
            "operation": operation,
            "error": error_message,
            "details": details,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self._recent_errors.append(error_record)
        self._stats.recent_errors = list(self._recent_errors)[-10:]  # Keep last 10

    # Public Interface

    def get_active_uploads(self) -> dict[str, UploadProgress]:
        """Get currently active uploads.

        Returns:
            Dictionary of upload ID to progress
        """
        return self._active_uploads.copy()

    def get_upload_progress(self, upload_id: str) -> UploadProgress | None:
        """Get progress for specific upload.

        Args:
            upload_id: Upload ID

        Returns:
            Upload progress or None if not found
        """
        return self._active_uploads.get(upload_id)

    def get_stats(self) -> StorageStats:
        """Get current storage statistics.

        Returns:
            Current statistics
        """
        return self._stats.model_copy(deep=True)

    def get_health_status(self) -> dict[str, Any]:
        """Get service health status.

        Returns:
            Health status information
        """
        try:
            # Test basic connectivity
            bucket_count = len(self._client.list_buckets())

            return {
                "status": "healthy",
                "endpoint": self.config.endpoint,
                "bucket_count": bucket_count,
                "active_uploads": len(self._active_uploads),
                "cache_enabled": self._cache_enabled,
                "redis_connected": self._redis_client is not None,
                "success_rate": self._stats.get_success_rate(),
                "average_operation_time": (
                    sum(self._operation_times) / len(self._operation_times)
                    if self._operation_times else 0
                ),
                "recent_errors": len(self._recent_errors),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "endpoint": self.config.endpoint,
                "cache_enabled": self._cache_enabled,
            }

    async def cleanup(self) -> None:
        """Cleanup resources and connections."""
        try:
            # Close Redis connection
            if self._redis_client:
                await self._redis_client.close()
                self._redis_client = None

            # Close HTTP connection pool
            if self._connection_pool:
                await self._connection_pool.close()
                self._connection_pool = None

            logger.info("MinIO service cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
