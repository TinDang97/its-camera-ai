"""
Advanced batch processing pipeline for ITS Camera AI Traffic Monitoring System.

This module implements intelligent batch processing with dynamic sizing, queue management,
priority handling, and load balancing across multiple GPUs. Designed to maximize throughput
while maintaining sub-100ms latency requirements.

Key Features:
1. Adaptive batch sizing based on GPU utilization and queue depth
2. Priority queuing for critical traffic events (accidents, violations)
3. Multi-GPU load balancing with device affinity optimization
4. Smart timeout management with queue age consideration
5. Memory-efficient tensor batching with zero-copy operations
6. Async processing with backpressure handling

Performance Optimizations:
- Dynamic batch size: 1-32 based on load and latency targets
- Intelligent timeout: 5-50ms based on queue depth and priority
- GPU affinity: Sticky sessions for temporal locality
- Memory pooling: Pre-allocated tensors to avoid allocation overhead
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch

from .inference_optimizer import DetectionResult, InferenceConfig

logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels for traffic monitoring."""

    EMERGENCY = 0  # Accidents, violations, emergency vehicles
    HIGH = 1  # Heavy traffic, congestion events
    NORMAL = 2  # Regular traffic monitoring
    LOW = 3  # Background analytics, statistics


@dataclass
class BatchRequest:
    """Individual inference request with metadata."""

    # Request data
    frame: np.ndarray[Any, np.dtype[Any]]
    frame_id: str
    camera_id: str
    timestamp: float

    # Processing metadata
    priority: RequestPriority = RequestPriority.NORMAL
    future: asyncio.Future[Any] | None = None
    retries: int = 0
    max_retries: int = 3

    # Performance tracking
    queue_entry_time: float | None = None
    processing_start_time: float | None = None

    def __post_init__(self) -> None:
        if self.queue_entry_time is None:
            self.queue_entry_time = time.time()

    def __lt__(self, other: "BatchRequest") -> bool:
        """Priority comparison for heapq."""
        return self.priority.value < other.priority.value

    @property
    def queue_age_ms(self) -> float:
        """Time spent in queue in milliseconds."""
        return (time.time() - self.queue_entry_time) * 1000

    @property
    def is_expired(self) -> bool:
        """Check if request has exceeded maximum age."""
        max_age_ms = {
            RequestPriority.EMERGENCY: 50,
            RequestPriority.HIGH: 100,
            RequestPriority.NORMAL: 200,
            RequestPriority.LOW: 500,
        }
        return self.queue_age_ms > max_age_ms[self.priority]


@dataclass
class BatchMetrics:
    """Performance metrics for batch processing."""

    # Throughput metrics
    requests_processed: int = 0
    total_processing_time_ms: float = 0
    avg_batch_size: float = 0

    # Latency metrics
    avg_queue_time_ms: float = 0
    p95_queue_time_ms: float = 0
    p99_queue_time_ms: float = 0

    # Quality metrics
    requests_dropped: int = 0
    requests_expired: int = 0
    retry_rate: float = 0

    # Resource utilization
    gpu_utilization_avg: float = 0
    memory_efficiency: float = 0
    queue_depth_avg: float = 0

    def update_from_batch(
        self, batch_size: int, processing_time_ms: float, queue_times: list[float]
    ):
        """Update metrics from processed batch."""
        self.requests_processed += batch_size
        self.total_processing_time_ms += processing_time_ms

        # Update running averages
        self.avg_batch_size = (
            self.avg_batch_size * (self.requests_processed - batch_size) + batch_size
        ) / self.requests_processed

        if queue_times:
            self.avg_queue_time_ms = np.mean(queue_times)
            self.p95_queue_time_ms = np.percentile(queue_times, 95)
            self.p99_queue_time_ms = np.percentile(queue_times, 99)


class AdaptiveBatchSizer:
    """Intelligent batch size adaptation based on system conditions."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.min_batch_size = 1
        self.max_batch_size = config.max_batch_size
        self.current_batch_size = config.batch_size

        # Adaptation parameters
        self.target_latency_ms = config.max_latency_ms
        self.target_gpu_utilization = 0.85
        self.adaptation_rate = 0.1

        # Performance history
        self.latency_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        self.gpu_util_history = deque(maxlen=50)

        # Adaptation state
        self.last_adaptation_time = time.time()
        self.adaptation_cooldown_s = 5.0

    def should_adapt(self) -> bool:
        """Check if it's time to adapt batch size."""
        return (time.time() - self.last_adaptation_time) > self.adaptation_cooldown_s

    def adapt_batch_size(
        self,
        current_latency_ms: float,
        current_throughput: float,
        gpu_utilization: float,
        queue_depth: int,
    ) -> int:
        """Adapt batch size based on current performance metrics."""

        if not self.should_adapt():
            return self.current_batch_size

        # Update history
        self.latency_history.append(current_latency_ms)
        self.throughput_history.append(current_throughput)
        self.gpu_util_history.append(gpu_utilization)

        # Calculate trend indicators
        self._calculate_trend(self.latency_history)
        gpu_util_avg = np.mean(list(self.gpu_util_history))

        # Adaptation logic
        new_batch_size = self.current_batch_size

        # Increase batch size if:
        # 1. Latency is below target and GPU utilization is low
        # 2. Queue depth is high (throughput pressure)
        if (
            current_latency_ms < self.target_latency_ms * 0.7
            and gpu_util_avg < self.target_gpu_utilization * 0.8
        ):
            new_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * (1 + self.adaptation_rate)),
            )
            logger.debug(
                f"Increasing batch size: {self.current_batch_size} -> {new_batch_size}"
            )

        # Decrease batch size if:
        # 1. Latency is above target
        # 2. GPU utilization is very high (resource pressure)
        elif (
            current_latency_ms > self.target_latency_ms
            or gpu_util_avg > self.target_gpu_utilization
        ):
            new_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * (1 - self.adaptation_rate)),
            )
            logger.debug(
                f"Decreasing batch size: {self.current_batch_size} -> {new_batch_size}"
            )

        # Special case: High queue depth with acceptable latency
        if (
            queue_depth > self.max_batch_size
            and current_latency_ms < self.target_latency_ms
        ):
            new_batch_size = min(self.max_batch_size, queue_depth // 2)
            logger.debug(
                f"Queue pressure adaptation: {self.current_batch_size} -> {new_batch_size}"
            )

        self.current_batch_size = new_batch_size
        self.last_adaptation_time = time.time()

        return new_batch_size

    def _calculate_trend(self, values: deque) -> float:
        """Calculate trend direction (-1 to 1) for recent values."""
        if len(values) < 10:
            return 0.0

        recent_values = list(values)[-10:]
        x = np.arange(len(recent_values))

        # Simple linear regression slope
        slope = np.polyfit(x, recent_values, 1)[0]
        return np.clip(slope / np.std(recent_values), -1, 1)


class GPULoadBalancer:
    """Intelligent load balancing across multiple GPUs."""

    def __init__(self, device_ids: list[int]):
        self.device_ids = device_ids
        self.device_loads = dict.fromkeys(device_ids, 0.0)
        self.device_queue_sizes = dict.fromkeys(device_ids, 0)
        self.device_affinities = {}  # camera_id -> preferred_device_id

        # Load balancing strategy
        self.use_affinity = True
        self.affinity_weight = 0.3
        self.load_update_interval = 1.0
        self.last_load_update = time.time()

    def select_device(
        self, camera_id: str, priority: RequestPriority = RequestPriority.NORMAL
    ) -> int:
        """Select optimal device for processing request."""

        # Update device loads if needed
        self._update_device_loads()

        # Emergency requests get the least loaded device immediately
        if priority == RequestPriority.EMERGENCY:
            return min(self.device_ids, key=lambda d: self.device_loads[d])

        # Check for camera affinity (temporal locality optimization)
        if self.use_affinity and camera_id in self.device_affinities:
            preferred_device = self.device_affinities[camera_id]

            # Use affinity device if not overloaded
            if (
                self.device_loads[preferred_device]
                < min(self.device_loads.values()) + 0.2
            ):
                return preferred_device

        # Select device with best composite score
        best_device = None
        best_score = float("inf")

        for device_id in self.device_ids:
            score = self._calculate_device_score(device_id, camera_id)

            if score < best_score:
                best_score = score
                best_device = device_id

        # Update affinity for future requests
        if self.use_affinity:
            self.device_affinities[camera_id] = best_device

        return best_device

    def _calculate_device_score(self, device_id: int, camera_id: str) -> float:
        """Calculate composite score for device selection."""

        # Base score from current load (0-1)
        load_score = self.device_loads[device_id]

        # Queue depth penalty (0-0.5)
        queue_score = min(0.5, self.device_queue_sizes[device_id] / 50.0)

        # Affinity bonus (-0.3 to 0)
        affinity_score = 0
        if (
            camera_id in self.device_affinities
            and self.device_affinities[camera_id] == device_id
        ):
            affinity_score = -self.affinity_weight

        # Memory utilization penalty (0-0.2)
        memory_score = self._get_memory_pressure_score(device_id)

        return load_score + queue_score + affinity_score + memory_score

    def _update_device_loads(self):
        """Update device load information."""
        current_time = time.time()

        if current_time - self.last_load_update < self.load_update_interval:
            return

        try:
            import pynvml

            pynvml.nvmlInit()

            for device_id in self.device_ids:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.device_loads[device_id] = util.gpu / 100.0
        except Exception:
            # Fallback to simple round-robin if pynvml unavailable
            logger.debug("Failed to query GPU utilization, using round-robin")

        self.last_load_update = current_time

    def _get_memory_pressure_score(self, device_id: int) -> float:
        """Get memory pressure score for device."""
        try:
            allocated = torch.cuda.memory_allocated(device_id)
            cached = torch.cuda.memory_reserved(device_id)

            # Simple heuristic: memory pressure increases exponentially
            memory_ratio = allocated / max(1, cached)
            return min(0.2, memory_ratio**2)
        except Exception:
            return 0.0

    def update_queue_size(self, device_id: int, queue_size: int):
        """Update queue size for device."""
        self.device_queue_sizes[device_id] = queue_size


class SmartBatchProcessor:
    """Advanced batch processor with intelligent optimization."""

    def __init__(
        self,
        config: InferenceConfig,
        inference_engine: Any,  # OptimizedInferenceEngine
        max_queue_size: int = 1000,
    ):
        self.config = config
        self.inference_engine = inference_engine
        self.max_queue_size = max_queue_size

        # Queue management
        self.request_queues = {
            priority: asyncio.Queue(maxsize=max_queue_size // 4)
            for priority in RequestPriority
        }

        # Processing components
        self.batch_sizer = AdaptiveBatchSizer(config)
        self.load_balancer = GPULoadBalancer(config.device_ids)

        # Processing state
        self.processing_tasks = {}
        self.is_running = False
        self.metrics = BatchMetrics()

        # Performance monitoring
        self.last_metrics_update = time.time()
        self.metrics_update_interval = 10.0

    async def start(self):
        """Start the batch processing system."""
        logger.info("Starting smart batch processor...")
        self.is_running = True

        # Start processing tasks for each GPU
        for device_id in self.config.device_ids:
            task = asyncio.create_task(self._process_batches_for_device(device_id))
            self.processing_tasks[device_id] = task

        # Start metrics collection task
        self.metrics_task = asyncio.create_task(self._collect_metrics())

        logger.info(
            f"Batch processor started with {len(self.processing_tasks)} GPU workers"
        )

    async def stop(self):
        """Stop the batch processing system."""
        logger.info("Stopping batch processor...")
        self.is_running = False

        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            task.cancel()

        if hasattr(self, "metrics_task"):
            self.metrics_task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)

        logger.info("Batch processor stopped")

    async def predict(
        self,
        frame: np.ndarray,
        frame_id: str,
        camera_id: str,
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> DetectionResult:
        """Submit prediction request with priority."""

        if not self.is_running:
            raise RuntimeError("Batch processor is not running")

        # Create request
        future = asyncio.Future()
        request = BatchRequest(
            frame=frame,
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=time.time(),
            priority=priority,
            future=future,
        )

        # Add to appropriate priority queue
        try:
            queue = self.request_queues[priority]
            await asyncio.wait_for(queue.put(request), timeout=1.0)
        except TimeoutError:
            # Queue is full, drop request
            self.metrics.requests_dropped += 1
            raise RuntimeError(
                f"Request queue full for priority {priority.name}"
            ) from None

        # Wait for result
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except TimeoutError:
            self.metrics.requests_expired += 1
            raise RuntimeError("Request timeout - system overloaded") from None

    async def _process_batches_for_device(self, device_id: int):
        """Process batches continuously for a specific device."""
        logger.info(f"Started batch processor for GPU {device_id}")

        while self.is_running:
            try:
                # Collect batch from priority queues
                batch = await self._collect_batch_for_device(device_id)

                if not batch:
                    await asyncio.sleep(0.001)  # Short sleep if no requests
                    continue

                # Process batch
                await self._process_single_batch(batch, device_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error on GPU {device_id}: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

        logger.info(f"Stopped batch processor for GPU {device_id}")

    async def _collect_batch_for_device(self, device_id: int) -> list[BatchRequest]:
        """Collect optimal batch for specific device."""
        batch = []

        # Determine target batch size
        current_metrics = self._get_current_performance_metrics()
        target_batch_size = self.batch_sizer.adapt_batch_size(
            current_latency_ms=current_metrics.get("avg_latency_ms", 50),
            current_throughput=current_metrics.get("throughput_fps", 30),
            gpu_utilization=current_metrics.get("gpu_utilization", 0.7),
            queue_depth=self._get_total_queue_depth(),
        )

        # Calculate timeout based on priority and queue age
        base_timeout_ms = self.config.batch_timeout_ms
        adaptive_timeout_ms = min(
            base_timeout_ms * 3, base_timeout_ms + self._get_total_queue_depth()
        )

        deadline = time.time() + (adaptive_timeout_ms / 1000.0)

        # Collect requests in priority order
        for priority in RequestPriority:
            queue = self.request_queues[priority]

            while len(batch) < target_batch_size and time.time() < deadline:
                try:
                    # Check if this device should handle requests from this camera
                    # Peek at queue without removing
                    if queue.empty():
                        break

                    # Get request with short timeout
                    timeout = max(0.001, deadline - time.time())
                    request = await asyncio.wait_for(queue.get(), timeout=timeout)

                    # Check if request has expired
                    if request.is_expired:
                        self.metrics.requests_expired += 1
                        request.future.set_exception(
                            RuntimeError("Request expired in queue")
                        )
                        continue

                    # Check device affinity
                    preferred_device = self.load_balancer.select_device(
                        request.camera_id, request.priority
                    )

                    if preferred_device != device_id and len(batch) == 0:
                        # Put request back if this is not the preferred device
                        # and we don't have any other requests yet
                        await queue.put(request)
                        break

                    # Add to batch
                    request.processing_start_time = time.time()
                    batch.append(request)

                    # Emergency requests get processed immediately
                    if priority == RequestPriority.EMERGENCY:
                        break

                except TimeoutError:
                    break

        # Update load balancer queue sizes
        self.load_balancer.update_queue_size(device_id, len(batch))

        return batch

    async def _process_single_batch(self, batch: list[BatchRequest], device_id: int):
        """Process a single batch of requests."""
        if not batch:
            return

        batch_start_time = time.time()

        try:
            # Extract frames and metadata
            frames = [request.frame for request in batch]
            frame_ids = [request.frame_id for request in batch]
            camera_ids = [request.camera_id for request in batch]

            # Process batch through inference engine
            results = await self.inference_engine.predict_batch(
                frames, frame_ids, camera_ids
            )

            # Calculate processing time and queue times
            processing_time_ms = (time.time() - batch_start_time) * 1000
            queue_times = [
                (batch_start_time - request.queue_entry_time) * 1000
                for request in batch
            ]

            # Return results to futures
            for request, result in zip(batch, results, strict=False):
                if not request.future.cancelled():
                    request.future.set_result(result)

            # Update metrics
            self.metrics.update_from_batch(len(batch), processing_time_ms, queue_times)

            logger.debug(
                f"Processed batch of {len(batch)} on GPU {device_id} "
                f"in {processing_time_ms:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")

            # Return error to all futures
            for request in batch:
                if not request.future.cancelled():
                    if request.retries < request.max_retries:
                        # Retry request
                        request.retries += 1
                        await self.request_queues[request.priority].put(request)
                        self.metrics.retry_rate += 1
                    else:
                        request.future.set_exception(e)

    def _get_total_queue_depth(self) -> int:
        """Get total number of requests in all queues."""
        return sum(queue.qsize() for queue in self.request_queues.values())

    def _get_current_performance_metrics(self) -> dict[str, float]:
        """Get current performance metrics from inference engine."""
        try:
            return self.inference_engine.get_performance_stats()
        except Exception:
            return {"avg_latency_ms": 50, "throughput_fps": 30, "gpu_utilization": 0.7}

    async def _collect_metrics(self):
        """Collect and log performance metrics periodically."""
        while self.is_running:
            try:
                await asyncio.sleep(self.metrics_update_interval)

                # Get current metrics
                self._get_current_performance_metrics()

                # Calculate batch processor specific metrics
                total_queue_depth = self._get_total_queue_depth()
                queue_by_priority = {
                    priority.name: queue.qsize()
                    for priority, queue in self.request_queues.items()
                }

                # Log comprehensive metrics
                logger.info(
                    f"Batch Processor Metrics - "
                    f"Processed: {self.metrics.requests_processed}, "
                    f"Avg Batch: {self.metrics.avg_batch_size:.1f}, "
                    f"Queue Depth: {total_queue_depth}, "
                    f"P95 Queue Time: {self.metrics.p95_queue_time_ms:.1f}ms, "
                    f"Dropped: {self.metrics.requests_dropped}, "
                    f"Current Batch Size: {self.batch_sizer.current_batch_size}"
                )

                # Log queue distribution
                if any(queue_by_priority.values()):
                    logger.debug(f"Queue distribution: {queue_by_priority}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive batch processor metrics."""
        engine_metrics = self._get_current_performance_metrics()

        return {
            # Batch processing metrics
            "batch_processor": {
                "requests_processed": self.metrics.requests_processed,
                "avg_batch_size": self.metrics.avg_batch_size,
                "current_batch_size": self.batch_sizer.current_batch_size,
                "requests_dropped": self.metrics.requests_dropped,
                "requests_expired": self.metrics.requests_expired,
                "retry_rate": self.metrics.retry_rate,
                "avg_queue_time_ms": self.metrics.avg_queue_time_ms,
                "p95_queue_time_ms": self.metrics.p95_queue_time_ms,
                "p99_queue_time_ms": self.metrics.p99_queue_time_ms,
            },
            # Queue status
            "queues": {
                priority.name: queue.qsize()
                for priority, queue in self.request_queues.items()
            },
            "total_queue_depth": self._get_total_queue_depth(),
            # Device utilization
            "devices": {
                f"gpu_{device_id}": {
                    "load": self.load_balancer.device_loads[device_id],
                    "queue_size": self.load_balancer.device_queue_sizes[device_id],
                }
                for device_id in self.config.device_ids
            },
            # Inference engine metrics
            "inference_engine": engine_metrics,
        }


# Utility functions for batch optimization


async def benchmark_batch_performance(
    processor: SmartBatchProcessor,
    num_requests: int = 1000,
    concurrent_cameras: int = 10,
) -> dict[str, float]:
    """Benchmark batch processor performance."""

    logger.info(f"Starting batch performance benchmark with {num_requests} requests")

    # Generate test data
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    camera_ids = [f"camera_{i:03d}" for i in range(concurrent_cameras)]

    # Submit requests concurrently
    start_time = time.time()
    tasks = []

    for i in range(num_requests):
        camera_id = camera_ids[i % len(camera_ids)]
        frame_id = f"frame_{i:06d}"

        # Vary priority distribution
        if i % 100 == 0:
            priority = RequestPriority.EMERGENCY
        elif i % 20 == 0:
            priority = RequestPriority.HIGH
        else:
            priority = RequestPriority.NORMAL

        task = processor.predict(test_frame, frame_id, camera_id, priority)
        tasks.append(task)

    # Wait for all requests to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()

    # Calculate metrics
    successful_results = [r for r in results if isinstance(r, DetectionResult)]
    failed_results = [r for r in results if isinstance(r, Exception)]

    total_time_s = end_time - start_time
    throughput = len(successful_results) / total_time_s

    if successful_results:
        latencies = [r.total_time_ms for r in successful_results]
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
    else:
        avg_latency = p95_latency = p99_latency = 0

    benchmark_results = {
        "total_requests": num_requests,
        "successful_requests": len(successful_results),
        "failed_requests": len(failed_results),
        "success_rate": len(successful_results) / num_requests,
        "total_time_s": total_time_s,
        "throughput_rps": throughput,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
    }

    logger.info(f"Benchmark completed: {benchmark_results}")
    return benchmark_results


def calculate_optimal_batch_configuration(
    target_throughput_rps: float,
    max_latency_ms: float,
    available_gpus: int,
    gpu_memory_gb: float,
) -> InferenceConfig:
    """Calculate optimal batch configuration for given requirements."""

    # Estimate processing capacity per GPU
    base_processing_time_ms = 15  # YOLO11s baseline
    max_batch_size_memory = int(gpu_memory_gb * 1024 / 50)  # ~50MB per batch item

    # Calculate required batch size for throughput
    required_batch_size = max(
        1,
        int(target_throughput_rps * base_processing_time_ms / (1000 * available_gpus)),
    )

    # Constrain by latency requirements
    max_batch_size_latency = max(1, int(max_latency_ms / base_processing_time_ms))

    # Final batch size considering all constraints
    optimal_batch_size = min(
        required_batch_size,
        max_batch_size_latency,
        max_batch_size_memory,
        32,  # Hard limit for stability
    )

    # Adaptive timeout based on batch size
    optimal_timeout_ms = min(max_latency_ms / 4, optimal_batch_size * 2 + 5)

    logger.info(
        f"Optimal batch config: size={optimal_batch_size}, "
        f"timeout={optimal_timeout_ms}ms for {target_throughput_rps} RPS target"
    )

    return InferenceConfig(
        batch_size=optimal_batch_size,
        max_batch_size=min(optimal_batch_size * 2, 32),
        batch_timeout_ms=int(optimal_timeout_ms),
        device_ids=list(range(available_gpus)),
        memory_fraction=0.8,
    )
