"""Comprehensive performance optimization coordinator for ITS Camera AI.

This module coordinates all performance optimization components including
GPU optimization, caching, connection pooling, latency monitoring, and
adaptive quality management to achieve sub-100ms end-to-end latency
with 100+ concurrent streams.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog

from ..core.exceptions import PerformanceOptimizationError, StreamProcessingError
from ..flow.redis_queue_manager import RedisQueueManager
from .adaptive_quality_manager import AdaptiveQualityManager, SystemMetrics
from .connection_pool_optimizer import ConnectionPoolOptimizer
from .gpu_memory_optimizer import GPUMemoryOptimizer
from .latency_monitor import LatencyMonitor, PipelineStage
from .optimization_config import OptimizationConfig
from .streaming_cache_manager import StreamingCacheManager

logger = structlog.get_logger(__name__)


class PerformanceMetricsCollector:
    """Collects and aggregates performance metrics from all optimization components."""

    def __init__(self):
        """Initialize metrics collector."""
        self.collection_start_time = time.time()
        self.metrics_history: list[dict[str, Any]] = []
        self.max_history = 1000  # Keep last 1000 metric snapshots

    def collect_metrics(
        self,
        gpu_optimizer: GPUMemoryOptimizer | None,
        cache_manager: StreamingCacheManager | None,
        connection_optimizer: ConnectionPoolOptimizer | None,
        latency_monitor: LatencyMonitor | None,
        quality_manager: AdaptiveQualityManager | None,
    ) -> dict[str, Any]:
        """Collect comprehensive performance metrics.

        Args:
            gpu_optimizer: GPU memory optimizer
            cache_manager: Streaming cache manager
            connection_optimizer: Connection pool optimizer
            latency_monitor: Latency monitor
            quality_manager: Adaptive quality manager

        Returns:
            Dict[str, Any]: Comprehensive performance metrics
        """
        timestamp = time.time()
        uptime_seconds = timestamp - self.collection_start_time

        metrics = {
            "timestamp": timestamp,
            "uptime_seconds": uptime_seconds,
            "collection_count": len(self.metrics_history),
        }

        # GPU optimization metrics
        if gpu_optimizer:
            try:
                metrics["gpu_optimization"] = gpu_optimizer.get_optimization_metrics()
            except Exception as e:
                logger.warning(f"GPU metrics collection failed: {e}")
                metrics["gpu_optimization"] = {"error": str(e)}

        # Cache performance metrics
        if cache_manager:
            try:
                metrics["caching"] = cache_manager.get_cache_metrics()
            except Exception as e:
                logger.warning(f"Cache metrics collection failed: {e}")
                metrics["caching"] = {"error": str(e)}

        # Connection pool metrics
        if connection_optimizer:
            try:
                metrics["connection_pools"] = (
                    connection_optimizer.get_comprehensive_stats()
                )
            except Exception as e:
                logger.warning(f"Connection pool metrics collection failed: {e}")
                metrics["connection_pools"] = {"error": str(e)}

        # Latency monitoring metrics
        if latency_monitor:
            try:
                metrics["latency_monitoring"] = (
                    latency_monitor.get_comprehensive_metrics()
                )
            except Exception as e:
                logger.warning(f"Latency metrics collection failed: {e}")
                metrics["latency_monitoring"] = {"error": str(e)}

        # Quality management metrics
        if quality_manager:
            try:
                metrics["quality_management"] = (
                    quality_manager.get_comprehensive_metrics()
                )
            except Exception as e:
                logger.warning(f"Quality management metrics collection failed: {e}")
                metrics["quality_management"] = {"error": str(e)}

        # Add to history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)

        return metrics

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary from recent metrics.

        Returns:
            Dict[str, Any]: Performance summary
        """
        if not self.metrics_history:
            return {"error": "No metrics available"}

        latest_metrics = self.metrics_history[-1]

        summary = {
            "timestamp": latest_metrics["timestamp"],
            "uptime_hours": latest_metrics["uptime_seconds"] / 3600,
            "metrics_collected": len(self.metrics_history),
        }

        # Extract key performance indicators
        try:
            # GPU utilization
            gpu_metrics = latest_metrics.get("gpu_optimization", {})
            if "gpu_utilization_percent" in gpu_metrics:
                summary["gpu_utilization_percent"] = gpu_metrics[
                    "gpu_utilization_percent"
                ]

            # Cache hit rate
            cache_metrics = latest_metrics.get("caching", {})
            if "performance" in cache_metrics:
                summary["cache_hit_rate_percent"] = cache_metrics["performance"].get(
                    "overall_hit_rate_percent", 0.0
                )

            # Latency metrics
            latency_metrics = latest_metrics.get("latency_monitoring", {})
            if "stages" in latency_metrics:
                end_to_end = latency_metrics["stages"].get("end_to_end", {})
                if "short_window" in end_to_end and end_to_end["short_window"]:
                    summary["end_to_end_latency_p95_ms"] = end_to_end[
                        "short_window"
                    ].get("p95_latency_ms", 0.0)

            # Quality management
            quality_metrics = latest_metrics.get("quality_management", {})
            if "global_quality_level" in quality_metrics:
                summary["global_quality_level"] = quality_metrics[
                    "global_quality_level"
                ]
                summary["total_streams"] = quality_metrics.get("total_cameras", 0)

            # SLA compliance
            if "overall_sla_violation_rate" in latency_metrics:
                summary["sla_compliance_percent"] = (
                    100.0 - latency_metrics["overall_sla_violation_rate"]
                )

        except Exception as e:
            logger.warning(f"Performance summary generation failed: {e}")
            summary["summary_error"] = str(e)

        return summary


class PerformanceOptimizer:
    """Comprehensive performance optimizer coordinating all optimization strategies.

    Manages GPU optimization, caching, connection pooling, latency monitoring,
    and adaptive quality management to achieve production-ready performance
    with sub-100ms end-to-end latency for 100+ concurrent streams.
    """

    def __init__(
        self, config: OptimizationConfig, redis_manager: RedisQueueManager | None = None
    ):
        """Initialize performance optimizer.

        Args:
            config: Optimization configuration
            redis_manager: Redis queue manager for caching
        """
        self.config = config
        self.redis_manager = redis_manager

        # Optimization components
        self.gpu_optimizer: GPUMemoryOptimizer | None = None
        self.cache_manager: StreamingCacheManager | None = None
        self.connection_optimizer: ConnectionPoolOptimizer | None = None
        self.latency_monitor: LatencyMonitor | None = None
        self.quality_manager: AdaptiveQualityManager | None = None

        # Performance monitoring
        self.metrics_collector = PerformanceMetricsCollector()
        self.monitoring_task: asyncio.Task | None = None

        # Initialization state
        self.is_initialized = False
        self.is_running = False
        self.initialization_time: float | None = None

        logger.info(
            f"PerformanceOptimizer created with {config.strategy.value} strategy"
        )

    async def initialize(
        self, redis_url: str | None = None, database_url: str | None = None
    ) -> None:
        """Initialize all optimization components.

        Args:
            redis_url: Redis connection URL for L2 caching
            database_url: Database connection URL
        """
        if self.is_initialized:
            logger.warning("PerformanceOptimizer already initialized")
            return

        init_start_time = time.perf_counter()

        try:
            logger.info("Initializing performance optimization components...")

            # Initialize GPU optimizer
            if self.config.gpu.gpu_memory_pool_size_gb > 0:
                from .gpu_memory_optimizer import create_gpu_optimizer

                self.gpu_optimizer = await create_gpu_optimizer(self.config.gpu)
                logger.info("GPU optimizer initialized")

            # Initialize cache manager
            if self.config.caching.l1_cache_size_mb > 0:
                from .streaming_cache_manager import create_streaming_cache_manager

                self.cache_manager = await create_streaming_cache_manager(
                    self.config.caching, self.redis_manager
                )
                logger.info("Cache manager initialized")

            # Initialize connection pool optimizer
            if redis_url or database_url:
                from .connection_pool_optimizer import create_connection_pool_optimizer

                self.connection_optimizer = await create_connection_pool_optimizer(
                    self.config.connection_pool, redis_url, database_url
                )
                logger.info("Connection pool optimizer initialized")

            # Initialize latency monitor
            if self.config.enable_monitoring:
                from .latency_monitor import create_latency_monitor

                self.latency_monitor = await create_latency_monitor(
                    self.config.latency_monitoring
                )
                logger.info("Latency monitor initialized")

            # Initialize adaptive quality manager
            if self.config.adaptive_quality.enable_adaptive_quality:
                from .adaptive_quality_manager import create_adaptive_quality_manager

                self.quality_manager = await create_adaptive_quality_manager(
                    self.config.adaptive_quality
                )
                logger.info("Adaptive quality manager initialized")

            # Record initialization time
            self.initialization_time = (time.perf_counter() - init_start_time) * 1000
            self.is_initialized = True

            logger.info(
                f"Performance optimizer initialized in {self.initialization_time:.2f}ms"
            )

        except Exception as e:
            logger.error(f"Performance optimizer initialization failed: {e}")
            raise PerformanceOptimizationError(f"Initialization failed: {e}") from e

    async def start(self) -> None:
        """Start performance optimization with monitoring."""
        if not self.is_initialized:
            raise PerformanceOptimizationError("Optimizer not initialized")

        if self.is_running:
            logger.warning("Performance optimizer already running")
            return

        try:
            self.is_running = True

            # Start monitoring task
            if self.config.enable_monitoring:
                self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            logger.info("Performance optimizer started")

        except Exception as e:
            logger.error(f"Performance optimizer start failed: {e}")
            raise PerformanceOptimizationError(f"Start failed: {e}") from e

    async def stop(self) -> None:
        """Stop performance optimization and cleanup."""
        if not self.is_running:
            return

        try:
            self.is_running = False

            # Stop monitoring task
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass

            # Cleanup optimization components
            if self.cache_manager:
                await self.cache_manager.stop()

            if self.connection_optimizer:
                await self.connection_optimizer.cleanup()

            if self.latency_monitor:
                await self.latency_monitor.stop_monitoring()

            if self.quality_manager:
                await self.quality_manager.stop()

            if self.gpu_optimizer:
                await self.gpu_optimizer.cleanup()

            logger.info("Performance optimizer stopped and cleaned up")

        except Exception as e:
            logger.error(f"Performance optimizer cleanup error: {e}")

    @asynccontextmanager
    async def optimize_stream_processing(
        self,
        camera_id: str,
        model_path: str | None = None,
        input_shape: tuple = (1, 3, 640, 640),
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Context manager for optimized stream processing.

        Args:
            camera_id: Camera identifier
            model_path: Path to ML model for GPU optimization
            input_shape: Model input shape for GPU optimization

        Yields:
            Dict[str, Any]: Optimization context with optimized components
        """
        if not self.is_initialized:
            raise PerformanceOptimizationError("Optimizer not initialized")

        optimization_context = {}
        request_id = f"{camera_id}_{int(time.time() * 1000000)}"

        try:
            # Track end-to-end latency
            latency_tracker = None
            if self.latency_monitor:
                latency_tracker = self.latency_monitor.track_end_to_end_latency(
                    camera_id, request_id
                )
                optimization_context["latency_tracker"] = latency_tracker
                await latency_tracker.__aenter__()

            # Get optimized ML model if available
            if self.gpu_optimizer and model_path:
                try:
                    import torch

                    model = (
                        torch.jit.load(model_path)
                        if model_path.endswith(".pt")
                        else None
                    )
                    if model:
                        model_optimizer = (
                            self.gpu_optimizer.optimize_inference_pipeline(
                                model, input_shape, f"model_{camera_id}"
                            )
                        )
                        optimization_context["model_optimizer"] = model_optimizer
                        optimization_context["optimized_model"] = (
                            await model_optimizer.__aenter__()
                        )
                except Exception as e:
                    logger.warning(f"ML model optimization failed: {e}")

            # Register camera with quality manager
            if self.quality_manager:
                is_priority = (
                    camera_id in self.config.adaptive_quality.priority_camera_ids
                )
                camera_state = await self.quality_manager.register_camera(
                    camera_id, is_priority
                )
                optimization_context["camera_quality_state"] = camera_state
                optimization_context["quality_profile"] = (
                    self.quality_manager.get_quality_profile(camera_id)
                )

            # Provide cache manager access
            if self.cache_manager:
                optimization_context["cache_manager"] = self.cache_manager

            # Provide connection pools
            if self.connection_optimizer:
                optimization_context["redis_pool"] = (
                    self.connection_optimizer.redis_optimizer
                )
                optimization_context["database_pool"] = (
                    self.connection_optimizer.database_optimizer
                )
                optimization_context["ffmpeg_pool"] = (
                    self.connection_optimizer.ffmpeg_pool
                )

            yield optimization_context

        except Exception as e:
            logger.error(f"Stream processing optimization error: {e}")
            raise StreamProcessingError(f"Optimization failed: {e}") from e

        finally:
            # Cleanup optimization context
            try:
                # Exit model optimizer context
                if "model_optimizer" in optimization_context:
                    await optimization_context["model_optimizer"].__aexit__(
                        None, None, None
                    )

                # Exit latency tracker context
                if latency_tracker:
                    await latency_tracker.__aexit__(None, None, None)

                # Unregister camera from quality manager
                if self.quality_manager:
                    await self.quality_manager.unregister_camera(camera_id)

            except Exception as e:
                logger.warning(f"Optimization context cleanup error: {e}")

    async def record_pipeline_measurement(
        self,
        camera_id: str,
        stage: PipelineStage,
        latency_ms: float,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record pipeline stage measurement for monitoring.

        Args:
            camera_id: Camera identifier
            stage: Pipeline stage
            latency_ms: Latency in milliseconds
            request_id: Optional request identifier
            metadata: Optional measurement metadata
        """
        if not self.latency_monitor:
            return

        try:
            from .latency_monitor import LatencyMeasurement

            measurement = LatencyMeasurement(
                timestamp=time.time(),
                latency_ms=latency_ms,
                stage=stage,
                camera_id=camera_id,
                request_id=request_id or f"{camera_id}_{int(time.time() * 1000)}",
                metadata=metadata or {},
            )

            await self.latency_monitor.record_measurement(measurement)

            # Update GPU optimizer metrics if available
            if self.gpu_optimizer and stage == PipelineStage.ML_INFERENCE:
                batch_size = metadata.get("batch_size", 1) if metadata else 1
                await self.gpu_optimizer.update_metrics(batch_size, latency_ms)

        except Exception as e:
            logger.warning(f"Pipeline measurement recording failed: {e}")

    async def cache_fragment(
        self, fragment_key: str, fragment_data: bytes, ttl_seconds: int | None = None
    ) -> bool:
        """Cache streaming fragment with optimization.

        Args:
            fragment_key: Fragment cache key
            fragment_data: Fragment data to cache
            ttl_seconds: Time to live in seconds

        Returns:
            bool: True if caching succeeded
        """
        if not self.cache_manager:
            return False

        try:
            await self.cache_manager.put(fragment_key, fragment_data, ttl_seconds)
            return True

        except Exception as e:
            logger.warning(f"Fragment caching failed: {e}")
            return False

    async def get_cached_fragment(self, fragment_key: str) -> bytes | None:
        """Get cached streaming fragment.

        Args:
            fragment_key: Fragment cache key

        Returns:
            Optional[bytes]: Cached fragment data or None
        """
        if not self.cache_manager:
            return None

        try:
            return await self.cache_manager.get(fragment_key)

        except Exception as e:
            logger.warning(f"Fragment cache retrieval failed: {e}")
            return None

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for performance metrics."""
        while self.is_running:
            try:
                # Collect comprehensive metrics
                metrics = self.metrics_collector.collect_metrics(
                    self.gpu_optimizer,
                    self.cache_manager,
                    self.connection_optimizer,
                    self.latency_monitor,
                    self.quality_manager,
                )

                # Log performance summary periodically
                if (
                    len(self.metrics_collector.metrics_history) % 10 == 0
                ):  # Every 10 collections
                    summary = self.metrics_collector.get_performance_summary()
                    logger.info(f"Performance summary: {summary}")

                # Check for performance issues
                await self._check_performance_issues(metrics)

                await asyncio.sleep(self.config.monitoring_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.config.monitoring_interval_seconds)

    async def _check_performance_issues(self, metrics: dict[str, Any]) -> None:
        """Check for performance issues and take corrective action.

        Args:
            metrics: Current performance metrics
        """
        try:
            # Check GPU utilization
            gpu_metrics = metrics.get("gpu_optimization", {})
            gpu_utilization = gpu_metrics.get("gpu_utilization_percent", 0.0)

            if gpu_utilization < 60.0 and self.gpu_optimizer:
                logger.warning(f"Low GPU utilization: {gpu_utilization:.1f}%")

            # Check cache hit rate
            cache_metrics = metrics.get("caching", {})
            if "performance" in cache_metrics:
                hit_rate = cache_metrics["performance"].get(
                    "overall_hit_rate_percent", 0.0
                )
                if hit_rate < 70.0:
                    logger.warning(f"Low cache hit rate: {hit_rate:.1f}%")

            # Check latency SLA compliance
            latency_metrics = metrics.get("latency_monitoring", {})
            violation_rate = latency_metrics.get("overall_sla_violation_rate", 0.0)

            if violation_rate > 5.0:  # More than 5% violations
                logger.warning(f"High SLA violation rate: {violation_rate:.1f}%")

                # Trigger quality reduction if available
                if self.quality_manager:
                    system_metrics = SystemMetrics(
                        cpu_percent=80.0,  # Simulate high load
                        memory_percent=75.0,
                        active_streams=len(self.quality_manager.camera_states),
                        average_latency_ms=120.0,  # Above SLA
                    )

                    adjustments = (
                        await self.quality_manager.adjust_quality_based_on_load(
                            system_metrics
                        )
                    )

                    if adjustments:
                        logger.info(
                            f"Applied emergency quality adjustments to {len(adjustments)} cameras"
                        )

        except Exception as e:
            logger.error(f"Performance issue checking failed: {e}")

    def get_optimization_status(self) -> dict[str, Any]:
        """Get current optimization status and metrics.

        Returns:
            Dict[str, Any]: Optimization status and metrics
        """
        status = {
            "initialized": self.is_initialized,
            "running": self.is_running,
            "strategy": self.config.strategy.value,
            "initialization_time_ms": self.initialization_time,
            "components": {
                "gpu_optimizer": self.gpu_optimizer is not None,
                "cache_manager": self.cache_manager is not None,
                "connection_optimizer": self.connection_optimizer is not None,
                "latency_monitor": self.latency_monitor is not None,
                "quality_manager": self.quality_manager is not None,
            },
        }

        # Add performance summary if available
        if self.metrics_collector.metrics_history:
            status["performance_summary"] = (
                self.metrics_collector.get_performance_summary()
            )

        return status

    def get_comprehensive_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics from all components.

        Returns:
            Dict[str, Any]: Comprehensive performance metrics
        """
        return self.metrics_collector.collect_metrics(
            self.gpu_optimizer,
            self.cache_manager,
            self.connection_optimizer,
            self.latency_monitor,
            self.quality_manager,
        )


async def create_performance_optimizer(
    config: OptimizationConfig,
    redis_manager: RedisQueueManager | None = None,
    redis_url: str | None = None,
    database_url: str | None = None,
) -> PerformanceOptimizer:
    """Create and initialize performance optimizer.

    Args:
        config: Optimization configuration
        redis_manager: Redis queue manager
        redis_url: Redis connection URL
        database_url: Database connection URL

    Returns:
        PerformanceOptimizer: Initialized performance optimizer
    """
    optimizer = PerformanceOptimizer(config, redis_manager)
    await optimizer.initialize(redis_url, database_url)
    await optimizer.start()
    return optimizer
