"""Performance optimization module for ITS Camera AI streaming system.

This module provides comprehensive performance optimization including GPU memory
optimization, multi-level caching, connection pooling, latency monitoring,
and adaptive quality management to achieve sub-100ms end-to-end latency
with 100+ concurrent streams.

Key Components:
- PerformanceOptimizer: Main coordinator for all optimization strategies
- GPUMemoryOptimizer: GPU optimization with TensorRT and memory pooling
- StreamingCacheManager: Multi-level caching (L1/L2/L3) with predictive caching
- ConnectionPoolOptimizer: Redis, database, and FFMPEG process pool optimization
- LatencyMonitor: Real-time latency tracking with SLA monitoring
- AdaptiveQualityManager: Dynamic quality adjustment based on system load
"""

from .adaptive_quality_manager import (
    AdaptiveQualityManager,
    QualityLevel,
    QualityProfile,
    SystemMetrics,
    create_adaptive_quality_manager,
)
from .connection_pool_optimizer import (
    ConnectionPoolOptimizer,
    create_connection_pool_optimizer,
)
from .gpu_memory_optimizer import (
    GPUMemoryOptimizer,
    create_gpu_optimizer,
)
from .latency_monitor import (
    LatencyMeasurement,
    LatencyMetrics,
    LatencyMonitor,
    PipelineStage,
    create_latency_monitor,
)
from .optimization_config import (
    AdaptiveQualityConfig,
    CachingConfig,
    ConnectionPoolConfig,
    GPUOptimizationConfig,
    LatencyMonitoringConfig,
    OptimizationConfig,
    OptimizationStrategy,
    create_production_optimization_config,
)
from .performance_optimizer import (
    PerformanceOptimizer,
    create_performance_optimizer,
)
from .streaming_cache_manager import (
    StreamingCacheManager,
    create_streaming_cache_manager,
)

__all__ = [
    # Main Performance Optimizer
    "PerformanceOptimizer",
    "create_performance_optimizer",

    # Configuration Classes
    "OptimizationConfig",
    "OptimizationStrategy",
    "GPUOptimizationConfig",
    "CachingConfig",
    "ConnectionPoolConfig",
    "LatencyMonitoringConfig",
    "AdaptiveQualityConfig",
    "create_production_optimization_config",

    # GPU Optimization
    "GPUMemoryOptimizer",
    "create_gpu_optimizer",

    # Caching System
    "StreamingCacheManager",
    "create_streaming_cache_manager",

    # Connection Pool Optimization
    "ConnectionPoolOptimizer",
    "create_connection_pool_optimizer",

    # Latency Monitoring
    "LatencyMonitor",
    "LatencyMeasurement",
    "LatencyMetrics",
    "PipelineStage",
    "create_latency_monitor",

    # Adaptive Quality Management
    "AdaptiveQualityManager",
    "QualityLevel",
    "QualityProfile",
    "SystemMetrics",
    "create_adaptive_quality_manager",
]


# Version information
__version__ = "1.0.0"
__author__ = "ITS Camera AI Performance Team"
__description__ = "Production-ready performance optimization for streaming systems"


# Performance optimization factory functions for common use cases
async def create_latency_optimized_system(
    max_concurrent_streams: int = 100,
    target_latency_ms: float = 80.0,
    redis_url: str = "redis://localhost:6379",
    database_url: str = "postgresql://localhost/its_camera_ai"
) -> PerformanceOptimizer:
    """Create performance optimizer optimized for latency.
    
    Args:
        max_concurrent_streams: Maximum concurrent streams
        target_latency_ms: Target latency in milliseconds
        redis_url: Redis connection URL
        database_url: Database connection URL
        
    Returns:
        PerformanceOptimizer: Latency-optimized performance system
    """
    config = create_production_optimization_config(
        max_concurrent_streams=max_concurrent_streams,
        target_latency_ms=target_latency_ms,
        strategy=OptimizationStrategy.LATENCY_OPTIMIZED
    )

    return await create_performance_optimizer(
        config=config,
        redis_url=redis_url,
        database_url=database_url
    )


async def create_memory_optimized_system(
    max_concurrent_streams: int = 200,
    memory_limit_gb: float = 4.0,
    redis_url: str = "redis://localhost:6379"
) -> PerformanceOptimizer:
    """Create performance optimizer optimized for memory efficiency.
    
    Args:
        max_concurrent_streams: Maximum concurrent streams
        memory_limit_gb: Memory limit in gigabytes
        redis_url: Redis connection URL
        
    Returns:
        PerformanceOptimizer: Memory-optimized performance system
    """
    config = create_production_optimization_config(
        max_concurrent_streams=max_concurrent_streams,
        target_latency_ms=150.0,  # Relaxed latency for memory efficiency
        strategy=OptimizationStrategy.MEMORY_OPTIMIZED
    )

    # Override memory settings
    config.memory_limit_gb = memory_limit_gb
    config.gpu.gpu_memory_pool_size_gb = min(4.0, memory_limit_gb * 0.5)
    config.caching.l1_cache_size_mb = min(256, int(memory_limit_gb * 64))

    return await create_performance_optimizer(
        config=config,
        redis_url=redis_url
    )


async def create_balanced_performance_system(
    max_concurrent_streams: int = 150,
    target_latency_ms: float = 100.0,
    redis_url: str = "redis://localhost:6379",
    database_url: str = "postgresql://localhost/its_camera_ai",
    enable_quality_adaptation: bool = True
) -> PerformanceOptimizer:
    """Create balanced performance optimizer for production deployment.
    
    Args:
        max_concurrent_streams: Maximum concurrent streams
        target_latency_ms: Target latency in milliseconds
        redis_url: Redis connection URL
        database_url: Database connection URL
        enable_quality_adaptation: Enable adaptive quality management
        
    Returns:
        PerformanceOptimizer: Balanced performance system
    """
    config = create_production_optimization_config(
        max_concurrent_streams=max_concurrent_streams,
        target_latency_ms=target_latency_ms,
        strategy=OptimizationStrategy.BALANCED
    )

    # Configure quality adaptation
    config.adaptive_quality.enable_adaptive_quality = enable_quality_adaptation

    return await create_performance_optimizer(
        config=config,
        redis_url=redis_url,
        database_url=database_url
    )
