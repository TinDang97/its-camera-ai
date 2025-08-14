"""Performance optimization configuration."""

from pydantic import BaseModel, Field


class PerformanceConfig(BaseModel):
    """Performance optimization configuration."""

    # Performance optimization strategy
    enabled: bool = Field(
        default=True, description="Enable performance optimization"
    )
    strategy: str = Field(
        default="latency_optimized",
        description="Optimization strategy: latency_optimized, memory_optimized, balanced"
    )
    target_latency_ms: float = Field(
        default=100.0, ge=10.0, le=1000.0, description="Target end-to-end latency in ms"
    )
    max_concurrent_streams: int = Field(
        default=100, ge=1, le=1000, description="Maximum concurrent streams to optimize for"
    )

    # GPU optimization
    gpu_optimization_enabled: bool = Field(
        default=True, description="Enable GPU memory optimization"
    )
    gpu_memory_pool_size_gb: float = Field(
        default=8.0, ge=1.0, description="GPU memory pool size in GB"
    )
    enable_tensorrt: bool = Field(
        default=True, description="Enable TensorRT optimization"
    )
    tensorrt_precision: str = Field(
        default="fp16", description="TensorRT precision: fp32, fp16, int8"
    )

    # Caching configuration
    caching_enabled: bool = Field(
        default=True, description="Enable multi-level caching"
    )
    l1_cache_size_mb: int = Field(
        default=512, ge=64, description="L1 cache size in MB"
    )
    l2_cache_enabled: bool = Field(
        default=True, description="Enable L2 Redis cache"
    )
    cache_ttl_seconds: int = Field(
        default=300, ge=60, description="Cache TTL in seconds"
    )

    # Connection pooling
    redis_pool_size: int = Field(
        default=50, ge=5, description="Redis connection pool size"
    )
    database_pool_size: int = Field(
        default=20, ge=5, description="Database connection pool size"
    )
    ffmpeg_pool_size: int = Field(
        default=10, ge=2, description="FFMPEG process pool size"
    )

    # Latency monitoring
    latency_monitoring_enabled: bool = Field(
        default=True, description="Enable latency SLA monitoring"
    )
    latency_sla_ms: float = Field(
        default=100.0, ge=10.0, description="Latency SLA in milliseconds"
    )
    enable_sla_alerts: bool = Field(
        default=True, description="Enable SLA violation alerts"
    )

    # Adaptive quality management
    adaptive_quality_enabled: bool = Field(
        default=True, description="Enable adaptive quality management"
    )
    default_quality: str = Field(
        default="high", description="Default quality level: low, medium, high"
    )
    quality_adjustment_interval: int = Field(
        default=30, ge=5, description="Quality adjustment interval in seconds"
    )
    priority_camera_ids: list[str] = Field(
        default_factory=list, description="Priority camera IDs for quality protection"
    )
