"""Storage configuration settings."""

from pydantic import BaseModel, Field


class MinIOConfig(BaseModel):
    """MinIO object storage configuration."""

    # Connection settings
    endpoint: str = Field(default="localhost:9000", description="MinIO server endpoint")
    access_key: str = Field(default="minioadmin", description="MinIO access key")
    secret_key: str = Field(
        default="minioadmin123", description="MinIO secret key", repr=False
    )
    secure: bool = Field(default=False, description="Use HTTPS connection")
    region: str = Field(default="us-east-1", description="MinIO region")

    # Bucket settings
    default_bucket: str = Field(
        default="its-camera-ai", description="Default storage bucket"
    )
    models_bucket: str = Field(
        default="its-models", description="ML model artifacts bucket"
    )
    video_bucket: str = Field(default="its-video", description="Video frames bucket")
    logs_bucket: str = Field(default="its-logs", description="Logs and metadata bucket")
    create_buckets: bool = Field(
        default=True, description="Auto-create buckets on startup"
    )

    # Upload/Download settings
    multipart_threshold: int = Field(
        default=64 * 1024 * 1024,  # 64MB
        description="Multipart upload threshold (bytes)",
    )
    multipart_chunksize: int = Field(
        default=16 * 1024 * 1024,  # 16MB
        description="Multipart chunk size (bytes)",
    )
    max_concurrent_uploads: int = Field(
        default=4, description="Maximum concurrent uploads"
    )

    # Connection pooling
    max_pool_connections: int = Field(
        default=20, description="Maximum pool connections"
    )
    connection_timeout: int = Field(
        default=60, description="Connection timeout (seconds)"
    )
    read_timeout: int = Field(default=300, description="Read timeout (seconds)")

    # Caching
    enable_caching: bool = Field(
        default=True, description="Enable Redis caching for metadata"
    )
    cache_ttl: int = Field(default=3600, description="Cache TTL (seconds)")
    cache_max_size: int = Field(default=1000, description="Max cached objects")

    # Security and performance
    enable_versioning: bool = Field(
        default=True, description="Enable object versioning"
    )
    enable_encryption: bool = Field(
        default=True, description="Enable server-side encryption"
    )
    enable_compression: bool = Field(
        default=True, description="Enable data compression"
    )
    compression_level: int = Field(
        default=6, ge=1, le=9, description="Compression level (1-9)"
    )

    # Monitoring
    enable_metrics: bool = Field(
        default=True, description="Enable MinIO metrics collection"
    )
    metrics_interval: int = Field(
        default=60, description="Metrics collection interval (seconds)"
    )

    # Lifecycle management
    enable_lifecycle: bool = Field(
        default=True, description="Enable object lifecycle management"
    )
    video_retention_days: int = Field(
        default=30, description="Video frame retention (days)"
    )
    retention_days: int = Field(
        default=365, description="Model artifact retention (days)"
    )
    log_retention_days: int = Field(default=90, description="Log retention (days)")
