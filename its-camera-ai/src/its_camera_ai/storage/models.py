"""Storage data models for MinIO integration.

Defines Pydantic models for storage operations, configurations,
and metadata tracking with comprehensive validation.
"""

import hashlib
import mimetypes
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class StorageBackend(str, Enum):
    """Storage backend types."""

    MINIO = "minio"
    AWS_S3 = "aws_s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    LOCAL_FS = "local_fs"


class ObjectType(str, Enum):
    """Object storage types for categorization."""

    VIDEO_FRAME = "video_frame"
    MODEL_ARTIFACT = "model_artifact"
    TRAINING_DATA = "training_data"
    INFERENCE_RESULT = "inference_result"
    LOG_FILE = "log_file"
    BACKUP = "backup"
    CHECKPOINT = "checkpoint"
    DATASET = "dataset"
    ANNOTATION = "annotation"
    METADATA = "metadata"


class CompressionType(str, Enum):
    """Compression types for storage optimization."""

    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"


class EncryptionType(str, Enum):
    """Encryption types for security."""

    NONE = "none"
    SSE_S3 = "sse_s3"
    SSE_KMS = "sse_kms"
    SSE_C = "sse_c"
    CLIENT_SIDE = "client_side"


class LifecycleAction(str, Enum):
    """Lifecycle management actions."""

    DELETE = "delete"
    TRANSITION_IA = "transition_ia"  # Infrequent Access
    TRANSITION_GLACIER = "transition_glacier"
    TRANSITION_DEEP_ARCHIVE = "transition_deep_archive"
    ABORT_MULTIPART = "abort_multipart"


class StorageConfig(BaseModel):
    """Storage service configuration."""

    # MinIO/S3 Connection
    endpoint: str = Field(..., description="MinIO server endpoint")
    access_key: str = Field(..., description="Access key ID")
    secret_key: str = Field(..., description="Secret access key", repr=False)
    secure: bool = Field(default=False, description="Use HTTPS")
    region: str = Field(default="us-east-1", description="Storage region")

    # Connection pooling
    max_pool_connections: int = Field(default=20, description="Max pool connections")
    connection_timeout: int = Field(default=60, description="Connection timeout (seconds)")
    read_timeout: int = Field(default=300, description="Read timeout (seconds)")

    # Default bucket settings
    default_bucket: str = Field(default="its-camera-ai", description="Default bucket name")
    create_buckets: bool = Field(default=True, description="Auto-create buckets")

    # Upload settings
    multipart_threshold: int = Field(
        default=64 * 1024 * 1024,  # 64MB
        description="Multipart upload threshold (bytes)"
    )
    multipart_chunksize: int = Field(
        default=16 * 1024 * 1024,  # 16MB
        description="Multipart chunk size (bytes)"
    )
    max_concurrent_uploads: int = Field(
        default=4,
        description="Max concurrent multipart uploads"
    )

    # Caching settings
    enable_caching: bool = Field(default=True, description="Enable Redis caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL (seconds)")
    cache_max_size: int = Field(default=1000, description="Max cached objects")

    # Security settings
    default_encryption: EncryptionType = Field(
        default=EncryptionType.SSE_S3,
        description="Default encryption type"
    )
    enable_versioning: bool = Field(default=True, description="Enable object versioning")

    # Performance settings
    enable_compression: bool = Field(default=True, description="Enable compression")
    default_compression: CompressionType = Field(
        default=CompressionType.GZIP,
        description="Default compression type"
    )

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval: int = Field(default=60, description="Metrics interval (seconds)")

    @field_validator('multipart_threshold', 'multipart_chunksize')
    @classmethod
    def validate_sizes(cls, v: int) -> int:
        """Validate size parameters."""
        if v <= 0:
            raise ValueError("Size must be positive")
        if v < 5 * 1024 * 1024:  # 5MB minimum for S3 compatibility
            raise ValueError("Size must be at least 5MB for multipart uploads")
        return v

    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Validate endpoint URL."""
        if not v:
            raise ValueError("Endpoint cannot be empty")
        if v.startswith('http://'):
            v = v[7:]
        elif v.startswith('https://'):
            v = v[8:]
        return v


class BucketConfig(BaseModel):
    """Bucket configuration and metadata."""

    name: str = Field(..., description="Bucket name")
    region: str = Field(default="us-east-1", description="Bucket region")
    versioning_enabled: bool = Field(default=True, description="Enable versioning")
    encryption: EncryptionType = Field(
        default=EncryptionType.SSE_S3,
        description="Encryption type"
    )

    # Lifecycle rules
    lifecycle_rules: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Lifecycle management rules"
    )

    # Access control
    public_read: bool = Field(default=False, description="Public read access")
    public_write: bool = Field(default=False, description="Public write access")

    # Metadata
    description: str | None = Field(None, description="Bucket description")
    tags: dict[str, str] = Field(default_factory=dict, description="Bucket tags")

    # Storage class
    default_storage_class: str = Field(
        default="STANDARD",
        description="Default storage class"
    )

    @field_validator('name')
    @classmethod
    def validate_bucket_name(cls, v: str) -> str:
        """Validate bucket name according to S3 rules."""
        if not v:
            raise ValueError("Bucket name cannot be empty")
        if len(v) < 3 or len(v) > 63:
            raise ValueError("Bucket name must be 3-63 characters long")
        if not v.islower():
            raise ValueError("Bucket name must be lowercase")
        if not v.replace('-', '').replace('.', '').isalnum():
            raise ValueError("Bucket name can only contain letters, numbers, hyphens, and periods")
        return v


class ObjectMetadata(BaseModel):
    """Object metadata for storage tracking."""

    # Basic information
    key: str = Field(..., description="Object key/path")
    bucket: str = Field(..., description="Bucket name")
    size: int = Field(..., description="Object size in bytes")
    content_type: str = Field(..., description="MIME content type")

    # Hashing and integrity
    etag: str | None = Field(None, description="ETag from storage")
    md5_hash: str | None = Field(None, description="MD5 hash")
    sha256_hash: str | None = Field(None, description="SHA256 hash")

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp"
    )
    modified_at: datetime | None = Field(None, description="Last modification time")
    accessed_at: datetime | None = Field(None, description="Last access time")

    # Storage metadata
    storage_class: str = Field(default="STANDARD", description="Storage class")
    encryption: EncryptionType = Field(
        default=EncryptionType.SSE_S3,
        description="Encryption type"
    )
    compression: CompressionType = Field(
        default=CompressionType.NONE,
        description="Compression type"
    )

    # Classification
    object_type: ObjectType = Field(..., description="Object type classification")
    version_id: str | None = Field(None, description="Object version ID")

    # Custom metadata
    user_metadata: dict[str, str] = Field(
        default_factory=dict,
        description="User-defined metadata"
    )
    system_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="System metadata"
    )

    # ML-specific metadata
    camera_id: str | None = Field(None, description="Source camera ID")
    frame_timestamp: datetime | None = Field(None, description="Frame timestamp")
    model_version: str | None = Field(None, description="Model version")
    accuracy_score: float | None = Field(None, description="Accuracy score")
    processing_time_ms: float | None = Field(None, description="Processing time")

    # Tags for organization
    tags: dict[str, str] = Field(default_factory=dict, description="Object tags")

    @field_validator('size')
    @classmethod
    def validate_size(cls, v: int) -> int:
        """Validate object size."""
        if v < 0:
            raise ValueError("Size cannot be negative")
        return v

    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """Validate and normalize content type."""
        if not v:
            return "application/octet-stream"  # Default binary type
        return v.lower()

    def calculate_md5(self, data: bytes) -> str:
        """Calculate MD5 hash from data."""
        md5_hash = hashlib.md5(data).hexdigest()
        self.md5_hash = md5_hash
        return md5_hash

    def calculate_sha256(self, data: bytes) -> str:
        """Calculate SHA256 hash from data."""
        sha256_hash = hashlib.sha256(data).hexdigest()
        self.sha256_hash = sha256_hash
        return sha256_hash

    def get_file_extension(self) -> str:
        """Get file extension from key."""
        return Path(self.key).suffix.lower()

    def is_image(self) -> bool:
        """Check if object is an image."""
        image_types = {'image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'}
        return self.content_type.lower() in image_types

    def is_video(self) -> bool:
        """Check if object is a video."""
        video_types = {'video/mp4', 'video/avi', 'video/mov', 'video/mkv'}
        return self.content_type.lower() in video_types

    def is_model_file(self) -> bool:
        """Check if object is a model file."""
        model_extensions = {'.pt', '.pth', '.onnx', '.pb', '.h5', '.pkl'}
        return self.get_file_extension() in model_extensions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.model_dump(mode='json')

    @classmethod
    def from_path(cls, file_path: Path, bucket: str, key: str,
                  object_type: ObjectType) -> 'ObjectMetadata':
        """Create metadata from file path."""
        stat = file_path.stat()
        content_type, _ = mimetypes.guess_type(str(file_path))

        return cls(
            key=key,
            bucket=bucket,
            size=stat.st_size,
            content_type=content_type or "application/octet-stream",
            object_type=object_type,
            created_at=datetime.fromtimestamp(stat.st_ctime, tz=UTC),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
        )


class UploadRequest(BaseModel):
    """Upload request configuration."""

    # Target location
    bucket: str = Field(..., description="Target bucket")
    key: str = Field(..., description="Object key/path")

    # Source data
    file_path: Path | None = Field(None, description="Source file path")
    data: bytes | None = Field(None, description="Raw data bytes", repr=False)

    # Upload options
    content_type: str | None = Field(None, description="Content type override")
    metadata: dict[str, str] = Field(default_factory=dict, description="User metadata")
    tags: dict[str, str] = Field(default_factory=dict, description="Object tags")

    # Storage options
    storage_class: str = Field(default="STANDARD", description="Storage class")
    encryption: EncryptionType = Field(
        default=EncryptionType.SSE_S3,
        description="Encryption type"
    )
    compression: CompressionType = Field(
        default=CompressionType.NONE,
        description="Compression type"
    )

    # Upload behavior
    overwrite: bool = Field(default=False, description="Overwrite existing object")
    use_multipart: bool | None = Field(None, description="Force multipart upload")
    part_size: int | None = Field(None, description="Multipart chunk size")

    # Progress tracking
    track_progress: bool = Field(default=True, description="Track upload progress")
    progress_callback: Any | None = Field(
        None, description="Progress callback function", exclude=True
    )

    # Classification
    object_type: ObjectType = Field(
        default=ObjectType.VIDEO_FRAME,
        description="Object type classification"
    )

    @model_validator(mode='after')
    def validate_source_data(self) -> 'UploadRequest':
        """Validate that either file_path or data is provided."""
        if not self.file_path and not self.data:
            raise ValueError("Either file_path or data must be provided")
        if self.file_path and self.data:
            raise ValueError("Only one of file_path or data should be provided")
        return self

    @field_validator('key')
    @classmethod
    def validate_key(cls, v: str) -> str:
        """Validate object key."""
        if not v:
            raise ValueError("Key cannot be empty")
        if v.startswith('/'):
            v = v[1:]  # Remove leading slash
        return v

    def get_content_type(self) -> str:
        """Get content type for upload."""
        if self.content_type:
            return self.content_type

        if self.file_path:
            content_type, _ = mimetypes.guess_type(str(self.file_path))
            return content_type or "application/octet-stream"

        return "application/octet-stream"

    def get_size(self) -> int:
        """Get data size."""
        if self.file_path and self.file_path.exists():
            return self.file_path.stat().st_size
        elif self.data:
            return len(self.data)
        return 0


class UploadProgress(BaseModel):
    """Upload progress tracking."""

    # Upload identification
    upload_id: str = Field(..., description="Unique upload ID")
    key: str = Field(..., description="Object key")
    bucket: str = Field(..., description="Bucket name")

    # Progress metrics
    total_size: int = Field(..., description="Total upload size (bytes)")
    uploaded_size: int = Field(default=0, description="Uploaded size (bytes)")
    progress_percent: float = Field(default=0.0, description="Progress percentage")

    # Timing
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Upload start time"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update time"
    )
    estimated_completion: datetime | None = Field(
        None, description="Estimated completion time"
    )

    # Performance metrics
    upload_speed_bps: float = Field(default=0.0, description="Upload speed (bytes/sec)")
    average_speed_bps: float = Field(default=0.0, description="Average speed (bytes/sec)")

    # Status
    status: str = Field(default="uploading", description="Upload status")
    error_message: str | None = Field(None, description="Error message if failed")

    # Multipart upload tracking
    is_multipart: bool = Field(default=False, description="Is multipart upload")
    parts_total: int = Field(default=1, description="Total parts")
    parts_completed: int = Field(default=0, description="Completed parts")
    active_parts: list[int] = Field(
        default_factory=list,
        description="Currently uploading parts"
    )

    def update_progress(self, uploaded_bytes: int) -> None:
        """Update progress metrics."""
        self.uploaded_size = uploaded_bytes
        self.progress_percent = min(100.0, (uploaded_bytes / self.total_size) * 100)

        now = datetime.now(UTC)
        self.updated_at = now

        # Calculate speed
        elapsed = (now - self.started_at).total_seconds()
        if elapsed > 0:
            self.average_speed_bps = uploaded_bytes / elapsed

            # Estimate completion
            if self.average_speed_bps > 0:
                remaining_bytes = self.total_size - uploaded_bytes
                remaining_seconds = remaining_bytes / self.average_speed_bps
                self.estimated_completion = now + datetime.timedelta(
                    seconds=remaining_seconds
                )

    def is_completed(self) -> bool:
        """Check if upload is completed."""
        return self.status == "completed" or self.progress_percent >= 100.0

    def is_failed(self) -> bool:
        """Check if upload failed."""
        return self.status == "failed"

    def get_eta_seconds(self) -> float | None:
        """Get estimated time to completion in seconds."""
        if self.estimated_completion:
            remaining = self.estimated_completion - datetime.now(UTC)
            return max(0.0, remaining.total_seconds())
        return None


class UploadResponse(BaseModel):
    """Upload operation response."""

    # Upload result
    success: bool = Field(..., description="Upload success status")
    upload_id: str = Field(..., description="Upload ID")

    # Object information
    bucket: str = Field(..., description="Bucket name")
    key: str = Field(..., description="Object key")
    etag: str | None = Field(None, description="Object ETag")
    version_id: str | None = Field(None, description="Object version ID")

    # Performance metrics
    total_size: int = Field(..., description="Total uploaded size")
    upload_time_seconds: float = Field(..., description="Upload duration")
    average_speed_bps: float = Field(..., description="Average upload speed")

    # Metadata
    content_type: str = Field(..., description="Content type")
    object_metadata: ObjectMetadata | None = Field(
        None, description="Object metadata"
    )

    # Error information
    error_message: str | None = Field(None, description="Error message if failed")
    error_code: str | None = Field(None, description="Error code")

    # Timestamps
    started_at: datetime = Field(..., description="Upload start time")
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Upload completion time"
    )

    # Additional info
    was_multipart: bool = Field(default=False, description="Was multipart upload")
    parts_count: int = Field(default=1, description="Number of parts")
    compression_ratio: float | None = Field(
        None, description="Compression ratio if compressed"
    )

    def get_upload_throughput_mbps(self) -> float:
        """Get upload throughput in MB/s."""
        if self.upload_time_seconds <= 0:
            return 0.0
        return (self.total_size / (1024 * 1024)) / self.upload_time_seconds

    def get_success_summary(self) -> str:
        """Get human-readable success summary."""
        if not self.success:
            return f"Upload failed: {self.error_message or 'Unknown error'}"

        size_mb = self.total_size / (1024 * 1024)
        throughput = self.get_upload_throughput_mbps()

        return (
            f"Successfully uploaded {size_mb:.1f}MB to {self.bucket}/{self.key} "
            f"in {self.upload_time_seconds:.1f}s ({throughput:.1f}MB/s)"
        )


class DownloadRequest(BaseModel):
    """Download request configuration."""

    # Source location
    bucket: str = Field(..., description="Source bucket")
    key: str = Field(..., description="Object key")
    version_id: str | None = Field(None, description="Specific version ID")

    # Download target
    file_path: Path | None = Field(None, description="Target file path")

    # Range download
    byte_range: tuple[int, int] | None = Field(
        None, description="Byte range (start, end)"
    )

    # Options
    validate_checksum: bool = Field(default=True, description="Validate checksums")
    decompress: bool = Field(default=True, description="Auto-decompress if compressed")

    # Progress tracking
    track_progress: bool = Field(default=True, description="Track download progress")
    progress_callback: Any | None = Field(
        None, description="Progress callback", exclude=True
    )


class ListObjectsRequest(BaseModel):
    """List objects request configuration."""

    bucket: str = Field(..., description="Bucket name")
    prefix: str | None = Field(None, description="Key prefix filter")
    delimiter: str | None = Field(None, description="Delimiter for grouping")
    max_keys: int = Field(default=1000, description="Maximum keys to return")
    start_after: str | None = Field(None, description="Start after key")

    # Filters
    object_types: list[ObjectType] | None = Field(
        None, description="Filter by object types"
    )
    size_range: tuple[int, int] | None = Field(
        None, description="Size range filter (min, max)"
    )
    date_range: tuple[datetime, datetime] | None = Field(
        None, description="Date range filter"
    )

    # Options
    include_metadata: bool = Field(default=False, description="Include full metadata")
    recursive: bool = Field(default=True, description="Recursive listing")


class ObjectListItem(BaseModel):
    """Object list item response."""

    key: str = Field(..., description="Object key")
    size: int = Field(..., description="Object size")
    etag: str = Field(..., description="Object ETag")
    last_modified: datetime = Field(..., description="Last modification time")
    storage_class: str = Field(default="STANDARD", description="Storage class")

    # Optional metadata
    content_type: str | None = Field(None, description="Content type")
    object_type: ObjectType | None = Field(None, description="Object type")
    metadata: ObjectMetadata | None = Field(None, description="Full metadata")


class BucketInfo(BaseModel):
    """Bucket information response."""

    name: str = Field(..., description="Bucket name")
    creation_date: datetime = Field(..., description="Bucket creation date")
    region: str = Field(..., description="Bucket region")
    versioning_enabled: bool = Field(default=False, description="Versioning status")
    encryption_enabled: bool = Field(default=False, description="Encryption status")

    # Statistics
    object_count: int | None = Field(None, description="Total objects")
    total_size: int | None = Field(None, description="Total size in bytes")

    # Configuration
    lifecycle_rules: list[dict[str, Any]] = Field(
        default_factory=list, description="Lifecycle rules"
    )
    tags: dict[str, str] = Field(default_factory=dict, description="Bucket tags")


class StorageStats(BaseModel):
    """Storage service statistics."""

    # Connection stats
    total_connections: int = Field(default=0, description="Total connections")
    active_connections: int = Field(default=0, description="Active connections")

    # Operation stats
    total_uploads: int = Field(default=0, description="Total uploads")
    total_downloads: int = Field(default=0, description="Total downloads")
    successful_operations: int = Field(default=0, description="Successful operations")
    failed_operations: int = Field(default=0, description="Failed operations")

    # Data transfer stats
    bytes_uploaded: int = Field(default=0, description="Total bytes uploaded")
    bytes_downloaded: int = Field(default=0, description="Total bytes downloaded")

    # Performance stats
    average_upload_speed_bps: float = Field(
        default=0.0, description="Average upload speed"
    )
    average_download_speed_bps: float = Field(
        default=0.0, description="Average download speed"
    )

    # Cache stats
    cache_hits: int = Field(default=0, description="Cache hits")
    cache_misses: int = Field(default=0, description="Cache misses")

    # Error tracking
    recent_errors: list[dict[str, Any]] = Field(
        default_factory=list, description="Recent errors"
    )

    # Timestamps
    stats_start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Stats collection start time"
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last update time"
    )

    def get_success_rate(self) -> float:
        """Calculate operation success rate."""
        total_ops = self.successful_operations + self.failed_operations
        if total_ops == 0:
            return 0.0
        return (self.successful_operations / total_ops) * 100.0

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return (self.cache_hits / total_requests) * 100.0

    def get_uptime_hours(self) -> float:
        """Get uptime in hours."""
        return (self.last_updated - self.stats_start_time).total_seconds() / 3600
