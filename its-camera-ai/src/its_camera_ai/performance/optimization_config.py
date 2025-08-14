"""Performance optimization configuration for ITS Camera AI streaming system.

This module provides comprehensive configuration classes for GPU optimization,
caching strategies, connection pooling, and latency monitoring to achieve
sub-100ms end-to-end latency with 100+ concurrent streams.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class OptimizationStrategy(str, Enum):
    """Performance optimization strategy levels."""

    MINIMAL = "minimal"  # Basic optimizations
    AGGRESSIVE = "aggressive"  # Maximum performance
    BALANCED = "balanced"  # Performance vs stability balance
    MEMORY_OPTIMIZED = "memory_optimized"  # Prioritize memory efficiency
    LATENCY_OPTIMIZED = "latency_optimized"  # Prioritize sub-100ms latency


class CacheEvictionPolicy(str, Enum):
    """Cache eviction policies for different cache levels."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    PREDICTIVE = "predictive"  # ML-based predictive eviction


@dataclass
class GPUOptimizationConfig:
    """GPU inference optimization configuration.
    
    Targets >85% GPU utilization with optimized batch processing
    for YOLO11 models while maintaining <100ms latency.
    """

    # TensorRT Optimization
    enable_tensorrt: bool = True
    tensorrt_precision: str = "fp16"  # fp16, int8, fp32
    tensorrt_workspace_size_gb: float = 4.0
    tensorrt_max_batch_size: int = 32
    tensorrt_min_batch_size: int = 1
    tensorrt_opt_batch_size: int = 8

    # Dynamic Batching
    enable_dynamic_batching: bool = True
    batch_timeout_ms: float = 10.0  # Max wait for batch formation
    batch_size_range: tuple[int, int] = (1, 32)
    adaptive_batching: bool = True  # Adjust batch size based on load

    # Memory Management
    gpu_memory_pool_size_gb: float = 8.0
    enable_memory_pinning: bool = True
    preallocate_output_tensors: bool = True
    enable_cuda_graphs: bool = True  # Static computation graphs

    # Model Optimization
    enable_quantization: bool = True
    quantization_mode: str = "dynamic"  # dynamic, static
    enable_pruning: bool = False  # Model pruning for edge deployment
    pruning_ratio: float = 0.4  # 40% parameter reduction

    # Inference Optimization
    enable_mixed_precision: bool = True
    use_channels_last: bool = True  # Memory format optimization
    compile_model: bool = True  # PyTorch 2.0+ compilation
    enable_jit_fusion: bool = True  # Kernel fusion


@dataclass
class CachingConfig:
    """Multi-level caching configuration for streaming system.
    
    Implements L1 (in-memory), L2 (Redis), L3 (CDN-ready) caching
    with >85% cache hit ratio target.
    """

    # L1 Cache - In-Memory
    l1_cache_size_mb: int = 512
    l1_cache_ttl_seconds: int = 5  # 5-second TTL for fragments
    l1_eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU

    # L2 Cache - Redis Distributed
    l2_cache_enabled: bool = True
    l2_cache_ttl_seconds: int = 300  # 5-minute TTL
    l2_eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.TTL
    l2_max_memory_mb: int = 2048

    # L3 Cache - CDN Ready
    l3_cache_enabled: bool = True
    l3_cache_ttl_seconds: int = 3600  # 1-hour TTL
    l3_compression_enabled: bool = True
    l3_compression_level: int = 6  # gzip compression level

    # Predictive Caching
    enable_predictive_caching: bool = True
    prediction_window_minutes: int = 15
    popular_stream_threshold: int = 10  # Requests per window

    # Cache Warming
    enable_cache_warming: bool = True
    warm_cache_on_startup: bool = True
    warm_popular_streams_interval_minutes: int = 30


@dataclass
class ConnectionPoolConfig:
    """Connection pool optimization configuration.
    
    Optimizes Redis, database, and FFMPEG process pools
    for maximum throughput and minimum latency.
    """

    # Redis Connection Pool
    redis_pool_size: int = 50
    redis_max_connections: int = 100
    redis_socket_keepalive: bool = True
    redis_socket_keepalive_options: dict[str, int] = field(
        default_factory=lambda: {
            "TCP_KEEPIDLE": 1,
            "TCP_KEEPINTVL": 3,
            "TCP_KEEPCNT": 5
        }
    )
    redis_retry_on_timeout: bool = True
    redis_health_check_interval: float = 30.0

    # Database Connection Pool
    db_pool_size: int = 20
    db_max_overflow: int = 30
    db_pool_recycle: int = 3600  # Recycle connections every hour
    db_pool_pre_ping: bool = True
    db_pool_timeout: float = 30.0

    # FFMPEG Process Pool
    ffmpeg_pool_size: int = 10
    ffmpeg_max_concurrent: int = 20
    ffmpeg_process_timeout: float = 60.0
    ffmpeg_restart_threshold: int = 100  # Restart after N jobs


@dataclass
class LatencyMonitoringConfig:
    """Latency monitoring and SLA configuration.
    
    Monitors end-to-end latency with <100ms SLA target
    and implements performance regression detection.
    """

    # SLA Targets
    latency_sla_ms: float = 100.0  # <100ms end-to-end latency
    latency_p50_target_ms: float = 50.0
    latency_p95_target_ms: float = 80.0
    latency_p99_target_ms: float = 95.0

    # Monitoring Windows
    monitoring_window_seconds: int = 60  # 1-minute rolling window
    alert_window_seconds: int = 300  # 5-minute alert window
    trend_analysis_minutes: int = 60  # 1-hour trend analysis

    # Performance Alerts
    enable_sla_alerts: bool = True
    sla_violation_threshold: float = 0.05  # 5% of requests exceed SLA
    regression_detection_enabled: bool = True
    regression_threshold_percent: float = 20.0  # 20% performance degradation

    # Metrics Collection
    collect_detailed_metrics: bool = True
    track_pipeline_segments: bool = True  # Track each pipeline stage
    enable_distributed_tracing: bool = True
    trace_sampling_rate: float = 0.1  # 10% sampling rate


@dataclass
class AdaptiveQualityConfig:
    """Adaptive quality management configuration.
    
    Implements dynamic quality adjustment based on system performance
    to maintain SLA compliance under high load conditions.
    """

    # Quality Levels
    default_quality: str = "high"  # high, medium, low
    enable_adaptive_quality: bool = True
    quality_adjustment_interval_seconds: int = 30

    # Load Thresholds
    cpu_threshold_percent: float = 70.0  # Reduce quality above 70% CPU
    memory_threshold_percent: float = 80.0
    gpu_memory_threshold_percent: float = 85.0
    latency_threshold_ms: float = 90.0  # Reduce quality if latency > 90ms

    # Quality Adjustment Strategy
    gradual_quality_reduction: bool = True
    quality_recovery_enabled: bool = True
    recovery_stability_period_minutes: int = 10

    # Priority Streaming
    enable_priority_streaming: bool = True
    priority_camera_ids: list[str] = field(default_factory=list)
    priority_quality_protection: bool = True  # Maintain quality for priority cameras


class OptimizationConfig(BaseModel):
    """Comprehensive performance optimization configuration.
    
    Master configuration class that coordinates all optimization strategies
    to achieve production-ready performance with sub-100ms latency.
    """

    # Optimization Strategy
    strategy: OptimizationStrategy = OptimizationStrategy.LATENCY_OPTIMIZED

    # Component Configurations
    gpu: GPUOptimizationConfig = Field(default_factory=GPUOptimizationConfig)
    caching: CachingConfig = Field(default_factory=CachingConfig)
    connection_pool: ConnectionPoolConfig = Field(default_factory=ConnectionPoolConfig)
    latency_monitoring: LatencyMonitoringConfig = Field(
        default_factory=LatencyMonitoringConfig
    )
    adaptive_quality: AdaptiveQualityConfig = Field(
        default_factory=AdaptiveQualityConfig
    )

    # System-Wide Settings
    max_concurrent_streams: int = Field(100, ge=1, le=1000)
    target_latency_ms: float = Field(100.0, gt=0)
    memory_limit_gb: float = Field(8.0, gt=0)
    enable_performance_profiling: bool = True

    # Monitoring and Alerting
    enable_monitoring: bool = True
    monitoring_interval_seconds: int = Field(30, ge=1)
    enable_alerts: bool = True
    alert_webhook_url: str | None = None

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        validate_assignment = True
        extra = "forbid"

    @validator("strategy")
    def validate_optimization_strategy(cls, v: OptimizationStrategy) -> OptimizationStrategy:
        """Validate optimization strategy and adjust dependent configs."""
        return v

    def get_optimized_config_for_strategy(self) -> "OptimizationConfig":
        """Get configuration optimized for the selected strategy.
        
        Returns:
            OptimizationConfig: Strategy-optimized configuration
        """
        config = self.copy(deep=True)

        if self.strategy == OptimizationStrategy.LATENCY_OPTIMIZED:
            # Optimize for sub-100ms latency
            config.gpu.batch_timeout_ms = 5.0
            config.gpu.tensorrt_opt_batch_size = 4
            config.caching.l1_cache_ttl_seconds = 3
            config.latency_monitoring.latency_sla_ms = 80.0

        elif self.strategy == OptimizationStrategy.MEMORY_OPTIMIZED:
            # Optimize for memory efficiency
            config.gpu.gpu_memory_pool_size_gb = 4.0
            config.gpu.tensorrt_max_batch_size = 16
            config.caching.l1_cache_size_mb = 256
            config.connection_pool.redis_pool_size = 25

        elif self.strategy == OptimizationStrategy.AGGRESSIVE:
            # Maximum performance regardless of resource usage
            config.gpu.tensorrt_max_batch_size = 64
            config.gpu.batch_timeout_ms = 15.0
            config.caching.l1_cache_size_mb = 1024
            config.connection_pool.redis_pool_size = 100

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return self.dict()


def create_production_optimization_config(
    max_concurrent_streams: int = 100,
    target_latency_ms: float = 100.0,
    strategy: OptimizationStrategy = OptimizationStrategy.LATENCY_OPTIMIZED,
) -> OptimizationConfig:
    """Create production-ready optimization configuration.
    
    Args:
        max_concurrent_streams: Maximum concurrent streams to support
        target_latency_ms: Target end-to-end latency in milliseconds
        strategy: Optimization strategy to apply
        
    Returns:
        OptimizationConfig: Production-optimized configuration
    """
    config = OptimizationConfig(
        strategy=strategy,
        max_concurrent_streams=max_concurrent_streams,
        target_latency_ms=target_latency_ms,
        enable_performance_profiling=True,
        enable_monitoring=True,
        enable_alerts=True,
    )

    return config.get_optimized_config_for_strategy()
