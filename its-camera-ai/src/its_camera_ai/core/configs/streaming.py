"""Streaming configuration settings."""

from pydantic import BaseModel, Field


class StreamingConfig(BaseModel):
    """Streaming service configuration."""

    # Core streaming settings
    min_resolution: tuple[int, int] = Field(
        default=(640, 480), description="Minimum frame resolution (width, height)"
    )
    min_quality_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum quality score threshold"
    )
    max_blur_threshold: float = Field(
        default=100.0, ge=0.0, description="Maximum blur threshold"
    )
    max_concurrent_streams: int = Field(
        default=100, ge=1, description="Maximum concurrent camera streams"
    )
    frame_processing_timeout: float = Field(
        default=0.01, ge=0.001, description="Frame processing timeout in seconds (10ms)"
    )

    # gRPC server settings
    grpc_host: str = Field(
        default="127.0.0.1", description="gRPC server host"
    )
    grpc_port: int = Field(
        default=50051, ge=1024, le=65535, description="gRPC server port"
    )

    # Performance settings
    batch_size: int = Field(
        default=10, ge=1, description="Frame batch size for processing"
    )
    queue_max_length: int = Field(
        default=10000, ge=100, description="Maximum queue length"
    )
    worker_threads: int = Field(
        default=4, ge=1, description="Number of worker threads"
    )

    # Health monitoring
    health_check_interval: int = Field(
        default=30, ge=5, description="Health check interval in seconds"
    )
    metrics_collection_interval: int = Field(
        default=10, ge=1, description="Metrics collection interval in seconds"
    )


class SSEStreamingConfig(BaseModel):
    """SSE (Server-Sent Events) streaming configuration."""

    # Connection management
    max_concurrent_connections: int = Field(
        default=100, ge=1, description="Maximum concurrent SSE connections"
    )
    fragment_duration_ms: int = Field(
        default=2000, ge=500, description="MP4 fragment duration in milliseconds"
    )
    heartbeat_interval: int = Field(
        default=30, ge=5, description="Heartbeat interval in seconds"
    )
    connection_timeout: int = Field(
        default=300, ge=60, description="Connection timeout in seconds"
    )

    # Quality settings
    quality_presets: dict[str, dict[str, int]] = Field(
        default={
            "low": {"bitrate": 500, "jpeg_quality": 60},
            "medium": {"bitrate": 2000, "jpeg_quality": 85},
            "high": {"bitrate": 5000, "jpeg_quality": 95},
        },
        description="Quality presets for different stream qualities"
    )

    # Buffer management
    stream_buffer_size: int = Field(
        default=100, ge=10, description="Stream buffer size per connection"
    )
    fragment_buffer_size: int = Field(
        default=50, ge=5, description="Fragment buffer size"
    )

    # Performance optimization
    enable_compression: bool = Field(
        default=True, description="Enable fragment compression"
    )
    compression_level: int = Field(
        default=6, ge=1, le=9, description="Compression level (1-9)"
    )

    # Monitoring
    enable_metrics: bool = Field(
        default=True, description="Enable SSE metrics collection"
    )
    metrics_interval: int = Field(
        default=10, ge=1, description="Metrics collection interval in seconds"
    )

    # Dual-channel settings
    sync_tolerance_ms: float = Field(
        default=50.0, ge=1.0, description="Channel synchronization tolerance in milliseconds"
    )
    sync_check_interval: float = Field(
        default=1.0, ge=0.1, description="Synchronization check interval in seconds"
    )
    max_sync_violations: int = Field(
        default=10, ge=1, description="Maximum sync violations before marking stream degraded"
    )
    enable_dual_channel: bool = Field(
        default=True, description="Enable dual-channel streaming support"
    )
    channel_switch_timeout: float = Field(
        default=0.1, ge=0.01, description="Channel switching timeout in seconds"
    )
