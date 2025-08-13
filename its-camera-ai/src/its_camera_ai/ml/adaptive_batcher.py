"""
Advanced adaptive batching system for YOLO11 production inference.

This module implements intelligent dynamic batching with priority queues,
adaptive timeout management, and workload-aware optimization for 100+
concurrent camera streams.
"""

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BatchPriority(Enum):
    """Priority levels for batch processing requests."""

    EMERGENCY = 0     # Emergency vehicles, accidents
    URGENT = 1        # Traffic violations, security alerts
    NORMAL = 2        # Standard traffic monitoring
    BACKGROUND = 3    # Analytics, reporting, training


@dataclass
class BatchRequest:
    """Individual request in the batching system."""

    frame: np.ndarray
    frame_id: str
    camera_id: str
    priority: BatchPriority
    future: asyncio.Future
    timestamp: float
    deadline: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0

    @property
    def age_ms(self) -> float:
        """Age of request in milliseconds."""
        return (time.time() - self.timestamp) * 1000

    @property
    def is_expired(self) -> bool:
        """Check if request has exceeded its deadline."""
        if self.deadline is None:
            return False
        return time.time() > self.deadline


@dataclass
class BatchMetrics:
    """Performance metrics for adaptive batching."""

    # Throughput metrics
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0

    # Latency metrics
    queue_times_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    processing_times_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    end_to_end_times_ms: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Batching efficiency
    batch_sizes: deque = field(default_factory=lambda: deque(maxlen=100))
    batch_utilization: deque = field(default_factory=lambda: deque(maxlen=100))

    # Priority queue metrics
    priority_counts: dict[BatchPriority, int] = field(
        default_factory=lambda: dict.fromkeys(BatchPriority, 0)
    )

    def add_latency_sample(self, queue_time_ms: float, processing_time_ms: float) -> None:
        """Add latency measurement sample."""
        self.queue_times_ms.append(queue_time_ms)
        self.processing_times_ms.append(processing_time_ms)
        self.end_to_end_times_ms.append(queue_time_ms + processing_time_ms)

    def add_batch_sample(self, batch_size: int, max_batch_size: int) -> None:
        """Add batch efficiency sample."""
        self.batch_sizes.append(batch_size)
        self.batch_utilization.append(batch_size / max_batch_size)

    @property
    def avg_queue_time_ms(self) -> float:
        """Average queue time in milliseconds."""
        return mean(self.queue_times_ms) if self.queue_times_ms else 0.0

    @property
    def p95_queue_time_ms(self) -> float:
        """95th percentile queue time."""
        if not self.queue_times_ms:
            return 0.0
        return np.percentile(list(self.queue_times_ms), 95)

    @property
    def avg_processing_time_ms(self) -> float:
        """Average processing time in milliseconds."""
        return mean(self.processing_times_ms) if self.processing_times_ms else 0.0

    @property
    def avg_batch_size(self) -> float:
        """Average batch size."""
        return mean(self.batch_sizes) if self.batch_sizes else 0.0

    @property
    def avg_batch_utilization(self) -> float:
        """Average batch utilization efficiency."""
        return mean(self.batch_utilization) if self.batch_utilization else 0.0

    @property
    def success_rate(self) -> float:
        """Request success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.completed_requests / self.total_requests


class AdaptiveTimeoutManager:
    """
    Manages adaptive timeout strategies based on system load and performance.

    Dynamically adjusts batch timeouts to balance latency vs throughput
    based on real-time performance metrics.
    """

    def __init__(
        self,
        base_timeout_ms: int = 10,
        min_timeout_ms: int = 2,
        max_timeout_ms: int = 50,
        adaptation_window: int = 100
    ):
        self.base_timeout_ms = base_timeout_ms
        self.min_timeout_ms = min_timeout_ms
        self.max_timeout_ms = max_timeout_ms
        self.adaptation_window = adaptation_window

        # Current adaptive timeout
        self.current_timeout_ms = base_timeout_ms

        # Performance tracking for adaptation
        self.recent_latencies: deque = deque(maxlen=adaptation_window)
        self.recent_batch_sizes: deque = deque(maxlen=adaptation_window)
        self.recent_gpu_utilization: deque = deque(maxlen=adaptation_window)

        # Adaptation state
        self.last_adaptation_time = time.time()
        self.adaptation_interval = 5.0  # Adapt every 5 seconds

        logger.info(f"Adaptive timeout manager initialized: {base_timeout_ms}ms base")

    def update_metrics(
        self,
        latency_ms: float,
        batch_size: int,
        gpu_utilization: float = 0.0
    ) -> None:
        """Update performance metrics for timeout adaptation."""
        self.recent_latencies.append(latency_ms)
        self.recent_batch_sizes.append(batch_size)
        self.recent_gpu_utilization.append(gpu_utilization)

        # Periodic adaptation
        if time.time() - self.last_adaptation_time > self.adaptation_interval:
            self._adapt_timeout()
            self.last_adaptation_time = time.time()

    def _adapt_timeout(self) -> None:
        """Adapt timeout based on recent performance."""
        if len(self.recent_latencies) < 10:
            return

        avg_latency = mean(self.recent_latencies)
        avg_batch_size = mean(self.recent_batch_sizes)
        avg_gpu_util = mean(self.recent_gpu_utilization) if self.recent_gpu_utilization else 0.0

        # Adaptation logic
        old_timeout = self.current_timeout_ms

        # Increase timeout if:
        # 1. Latency is high but batch sizes are small (underutilized)
        # 2. GPU utilization is low
        if avg_latency > 80 and avg_batch_size < 4:
            self.current_timeout_ms = min(
                self.max_timeout_ms,
                self.current_timeout_ms * 1.2
            )
        elif avg_gpu_util < 0.5 and avg_batch_size < 8:
            self.current_timeout_ms = min(
                self.max_timeout_ms,
                self.current_timeout_ms * 1.1
            )
        # Decrease timeout if:
        # 1. Latency is acceptable and batch sizes are good
        # 2. GPU utilization is high
        elif avg_latency < 40 and avg_batch_size > 8:
            self.current_timeout_ms = max(
                self.min_timeout_ms,
                self.current_timeout_ms * 0.9
            )
        elif avg_gpu_util > 0.8:
            self.current_timeout_ms = max(
                self.min_timeout_ms,
                self.current_timeout_ms * 0.95
            )

        if abs(old_timeout - self.current_timeout_ms) > 1:
            logger.debug(f"Adapted timeout: {old_timeout:.1f}ms -> {self.current_timeout_ms:.1f}ms "
                        f"(latency: {avg_latency:.1f}ms, batch: {avg_batch_size:.1f}, "
                        f"gpu: {avg_gpu_util:.2f})")

    def get_timeout_for_priority(self, priority: BatchPriority) -> float:
        """Get timeout value based on priority level."""
        base_timeout = self.current_timeout_ms / 1000.0  # Convert to seconds

        # Priority-based timeout scaling
        if priority == BatchPriority.EMERGENCY:
            return base_timeout * 0.1  # 10% of normal timeout
        elif priority == BatchPriority.URGENT:
            return base_timeout * 0.5  # 50% of normal timeout
        elif priority == BatchPriority.NORMAL:
            return base_timeout
        else:  # BACKGROUND
            return base_timeout * 2.0  # 200% of normal timeout


class WorkloadAnalyzer:
    """
    Analyzes incoming workload patterns to optimize batching strategy.

    Predicts optimal batch sizes and timing based on traffic patterns.
    """

    def __init__(self, analysis_window: int = 300):
        self.analysis_window = analysis_window  # 5 minutes

        # Request pattern tracking
        self.request_timestamps: deque = deque()
        self.camera_activity: dict[str, deque] = {}
        self.priority_patterns: dict[BatchPriority, deque] = {
            p: deque() for p in BatchPriority
        }

        # Pattern analysis results
        self.peak_hours: list[tuple[int, int]] = []
        self.camera_utilization: dict[str, float] = {}
        self.predicted_load: float = 1.0

    def record_request(
        self,
        camera_id: str,
        priority: BatchPriority,
        timestamp: float
    ) -> None:
        """Record incoming request for pattern analysis."""
        # Clean old data
        cutoff_time = timestamp - self.analysis_window

        # Remove old timestamps
        while self.request_timestamps and self.request_timestamps[0] < cutoff_time:
            self.request_timestamps.popleft()

        # Add new request
        self.request_timestamps.append(timestamp)

        # Track per-camera activity
        if camera_id not in self.camera_activity:
            self.camera_activity[camera_id] = deque()

        while (self.camera_activity[camera_id] and
               self.camera_activity[camera_id][0] < cutoff_time):
            self.camera_activity[camera_id].popleft()

        self.camera_activity[camera_id].append(timestamp)

        # Track priority patterns
        while (self.priority_patterns[priority] and
               self.priority_patterns[priority][0] < cutoff_time):
            self.priority_patterns[priority].popleft()

        self.priority_patterns[priority].append(timestamp)

    def analyze_workload(self) -> dict[str, Any]:
        """Analyze current workload patterns."""
        current_time = time.time()

        # Calculate request rate (requests per second)
        if len(self.request_timestamps) >= 2:
            time_span = self.request_timestamps[-1] - self.request_timestamps[0]
            request_rate = len(self.request_timestamps) / max(1, time_span)
        else:
            request_rate = 0.0

        # Calculate per-camera utilization
        camera_rates = {}
        for camera_id, timestamps in self.camera_activity.items():
            if len(timestamps) >= 2:
                time_span = timestamps[-1] - timestamps[0]
                camera_rates[camera_id] = len(timestamps) / max(1, time_span)
            else:
                camera_rates[camera_id] = 0.0

        # Calculate priority distribution
        priority_distribution = {}
        total_priority_requests = sum(len(queue) for queue in self.priority_patterns.values())

        for priority, queue in self.priority_patterns.items():
            if total_priority_requests > 0:
                priority_distribution[priority.name] = len(queue) / total_priority_requests
            else:
                priority_distribution[priority.name] = 0.0

        # Predict optimal batch size based on load
        if request_rate < 1.0:
            optimal_batch_size = 1
        elif request_rate < 5.0:
            optimal_batch_size = 4
        elif request_rate < 20.0:
            optimal_batch_size = 8
        elif request_rate < 50.0:
            optimal_batch_size = 16
        else:
            optimal_batch_size = 32

        return {
            'request_rate': request_rate,
            'camera_rates': camera_rates,
            'priority_distribution': priority_distribution,
            'optimal_batch_size': optimal_batch_size,
            'active_cameras': len([r for r in camera_rates.values() if r > 0.1]),
            'high_priority_ratio': (
                priority_distribution.get('EMERGENCY', 0) +
                priority_distribution.get('URGENT', 0)
            ),
            'analysis_timestamp': current_time,
        }

    def predict_load_spike(self) -> tuple[bool, float]:
        """Predict if a load spike is incoming."""
        if len(self.request_timestamps) < 20:
            return False, 1.0

        # Analyze recent trend
        recent_requests = list(self.request_timestamps)[-20:]
        time_intervals = [
            recent_requests[i] - recent_requests[i-1]
            for i in range(1, len(recent_requests))
        ]

        if len(time_intervals) < 5:
            return False, 1.0

        # Calculate acceleration (decreasing intervals = increasing rate)
        recent_avg = mean(time_intervals[-5:])
        older_avg = mean(time_intervals[-10:-5]) if len(time_intervals) >= 10 else recent_avg

        if recent_avg < older_avg * 0.7:  # 30% decrease in intervals
            predicted_multiplier = older_avg / recent_avg
            return True, min(predicted_multiplier, 3.0)  # Cap at 3x

        return False, 1.0


class AdaptiveBatchProcessor:
    """
    Advanced adaptive batching processor with intelligent workload management.

    Features:
    - Multi-priority queues with deadline awareness
    - Adaptive timeout based on system performance
    - Workload pattern analysis and prediction
    - GPU memory and compute optimization
    - Circuit breaker pattern for overload protection
    """

    def __init__(
        self,
        inference_func: Callable[[list[np.ndarray], list[str], list[str]], list[Any]],
        max_batch_size: int = 32,
        base_timeout_ms: int = 10,
        enable_workload_analysis: bool = True,
        circuit_breaker_threshold: int = 100
    ):
        self.inference_func = inference_func
        self.max_batch_size = max_batch_size
        self.circuit_breaker_threshold = circuit_breaker_threshold

        # Priority queues
        self.queues = {
            priority: asyncio.Queue(maxsize=max_batch_size * 4)
            for priority in BatchPriority
        }

        # Management components
        self.timeout_manager = AdaptiveTimeoutManager(base_timeout_ms)
        self.workload_analyzer = WorkloadAnalyzer() if enable_workload_analysis else None
        self.metrics = BatchMetrics()

        # Processing state
        self.running = False
        self.processor_task: asyncio.Task | None = None
        self.circuit_breaker_open = False
        self.circuit_breaker_reset_time = 0.0

        # Batch collection state
        self.current_batch: list[BatchRequest] = []
        self.batch_start_time = time.time()

        logger.info(f"Adaptive batch processor initialized: max_batch={max_batch_size}")

    async def start(self) -> None:
        """Start the adaptive batch processing system."""
        if self.running:
            return

        self.running = True
        self.processor_task = asyncio.create_task(self._process_batches())

        logger.info("Adaptive batch processor started")

    async def stop(self) -> None:
        """Stop the batch processing system gracefully."""
        self.running = False

        if self.processor_task:
            self.processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.processor_task

        # Complete any pending requests with errors
        for queue in self.queues.values():
            while not queue.empty():
                try:
                    request = queue.get_nowait()
                    if not request.future.done():
                        request.future.set_exception(
                            RuntimeError("Batch processor stopped")
                        )
                except asyncio.QueueEmpty:
                    break

        logger.info("Adaptive batch processor stopped")

    async def submit_request(
        self,
        frame: np.ndarray,
        frame_id: str,
        camera_id: str = "unknown",
        priority: BatchPriority = BatchPriority.NORMAL,
        deadline_ms: int | None = None
    ) -> Any:
        """
        Submit inference request to adaptive batching system.

        Args:
            frame: Input frame for inference
            frame_id: Unique frame identifier
            camera_id: Source camera identifier
            priority: Request priority level
            deadline_ms: Optional deadline for completion

        Returns:
            Inference result
        """
        if not self.running:
            raise RuntimeError("Batch processor not running")

        # Circuit breaker check
        if self._is_circuit_breaker_open():
            raise RuntimeError("System overloaded - circuit breaker open")

        timestamp = time.time()
        deadline = timestamp + (deadline_ms / 1000.0) if deadline_ms else None

        # Create request
        future = asyncio.Future()
        request = BatchRequest(
            frame=frame,
            frame_id=frame_id,
            camera_id=camera_id,
            priority=priority,
            future=future,
            timestamp=timestamp,
            deadline=deadline
        )

        # Update workload analysis
        if self.workload_analyzer:
            self.workload_analyzer.record_request(camera_id, priority, timestamp)

        # Add to appropriate queue
        try:
            await self.queues[priority].put(request)
            self.metrics.total_requests += 1
            self.metrics.priority_counts[priority] += 1

        except asyncio.QueueFull:
            self.metrics.failed_requests += 1
            raise RuntimeError(f"Priority queue {priority.name} is full")

        # Wait for result
        try:
            result = await future
            self.metrics.completed_requests += 1
            return result

        except TimeoutError:
            self.metrics.timeout_requests += 1
            raise RuntimeError(f"Request {frame_id} timed out")

    async def _process_batches(self) -> None:
        """Main batch processing loop."""
        logger.info("Starting adaptive batch processing loop")

        while self.running:
            try:
                # Collect batch with adaptive timing
                batch = await self._collect_adaptive_batch()

                if not batch:
                    await asyncio.sleep(0.001)  # Small sleep to prevent busy waiting
                    continue

                # Process batch
                await self._process_batch(batch)

                # Update circuit breaker
                self._update_circuit_breaker()

            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

    async def _collect_adaptive_batch(self) -> list[BatchRequest]:
        """
        Collect batch with adaptive sizing and timeout management.

        Returns:
            List of requests to process as a batch
        """
        batch: list[BatchRequest] = []
        time.time()

        # Get workload analysis for optimization
        workload_info = None
        if self.workload_analyzer:
            workload_info = self.workload_analyzer.analyze_workload()
            optimal_batch_size = min(
                workload_info['optimal_batch_size'],
                self.max_batch_size
            )
        else:
            optimal_batch_size = self.max_batch_size

        # Collect requests with priority-based strategy
        while len(batch) < optimal_batch_size and self.running:
            request = await self._get_next_priority_request()

            if request is None:
                break

            # Check if request is expired
            if request.is_expired:
                if not request.future.done():
                    request.future.set_exception(
                        RuntimeError(f"Request {request.frame_id} expired")
                    )
                self.metrics.timeout_requests += 1
                continue

            batch.append(request)

            # Adaptive timeout check
            if batch:
                oldest_request = min(batch, key=lambda r: r.timestamp)
                highest_priority = min(batch, key=lambda r: r.priority.value).priority

                timeout_seconds = self.timeout_manager.get_timeout_for_priority(highest_priority)
                age_seconds = time.time() - oldest_request.timestamp

                if age_seconds >= timeout_seconds:
                    break

        return batch

    async def _get_next_priority_request(self) -> BatchRequest | None:
        """Get next request based on priority ordering."""
        # Check queues in priority order
        for priority in BatchPriority:
            queue = self.queues[priority]

            if not queue.empty():
                try:
                    return queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue

        # If all queues are empty, wait for any request
        gathering_tasks = [
            asyncio.create_task(queue.get())
            for queue in self.queues.values()
            if not queue.empty()
        ]

        if not gathering_tasks:
            # All queues empty, wait briefly
            try:
                # Wait on all queues simultaneously with timeout
                done, pending = await asyncio.wait(
                    [asyncio.create_task(queue.get()) for queue in self.queues.values()],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.01  # 10ms timeout
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()

                if done:
                    return await done.pop()

            except TimeoutError:
                pass

        return None

    async def _process_batch(self, batch: list[BatchRequest]) -> None:
        """Process a collected batch of requests."""
        if not batch:
            return

        processing_start = time.time()

        try:
            # Prepare batch data
            frames = [req.frame for req in batch]
            frame_ids = [req.frame_id for req in batch]
            camera_ids = [req.camera_id for req in batch]

            # Run inference
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.inference_func, frames, frame_ids, camera_ids
            )

            processing_time = (time.time() - processing_start) * 1000

            # Distribute results
            for request, result in zip(batch, results, strict=False):
                if not request.future.done():
                    request.future.set_result(result)

                # Record metrics
                queue_time_ms = (processing_start - request.timestamp) * 1000
                self.metrics.add_latency_sample(queue_time_ms, processing_time / len(batch))

            # Update timeout manager
            self.timeout_manager.update_metrics(
                processing_time / len(batch),
                len(batch),
                0.0  # GPU utilization - would need to be measured
            )

            # Record batch metrics
            self.metrics.add_batch_sample(len(batch), self.max_batch_size)

            logger.debug(f"Processed batch of {len(batch)} requests in {processing_time:.1f}ms")

        except Exception as e:
            # Handle batch processing failure
            logger.error(f"Batch processing failed: {e}")

            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)

            self.metrics.failed_requests += len(batch)

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open due to system overload."""
        if not self.circuit_breaker_open:
            return False

        # Check if reset time has passed
        if time.time() > self.circuit_breaker_reset_time:
            self.circuit_breaker_open = False
            logger.info("Circuit breaker reset")
            return False

        return True

    def _update_circuit_breaker(self) -> None:
        """Update circuit breaker state based on system performance."""
        # Check queue depths
        total_pending = sum(queue.qsize() for queue in self.queues.values())

        if total_pending > self.circuit_breaker_threshold:
            if not self.circuit_breaker_open:
                self.circuit_breaker_open = True
                self.circuit_breaker_reset_time = time.time() + 30.0  # 30 second cooldown
                logger.warning(f"Circuit breaker opened - queue depth: {total_pending}")

        # Also check error rate
        if self.metrics.total_requests > 100:
            error_rate = self.metrics.failed_requests / self.metrics.total_requests
            if error_rate > 0.1:  # 10% error rate threshold
                if not self.circuit_breaker_open:
                    self.circuit_breaker_open = True
                    self.circuit_breaker_reset_time = time.time() + 60.0  # 60 second cooldown
                    logger.warning(f"Circuit breaker opened - error rate: {error_rate:.2%}")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        workload_info = {}
        if self.workload_analyzer:
            workload_info = self.workload_analyzer.analyze_workload()

        return {
            'success_rate': self.metrics.success_rate,
            'total_requests': self.metrics.total_requests,
            'avg_queue_time_ms': self.metrics.avg_queue_time_ms,
            'p95_queue_time_ms': self.metrics.p95_queue_time_ms,
            'avg_processing_time_ms': self.metrics.avg_processing_time_ms,
            'avg_batch_size': self.metrics.avg_batch_size,
            'avg_batch_utilization': self.metrics.avg_batch_utilization,
            'current_timeout_ms': self.timeout_manager.current_timeout_ms,
            'circuit_breaker_open': self.circuit_breaker_open,
            'priority_distribution': {
                p.name: count for p, count in self.metrics.priority_counts.items()
            },
            'queue_depths': {
                p.name: queue.qsize() for p, queue in self.queues.items()
            },
            'workload_analysis': workload_info,
        }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = BatchMetrics()
        logger.info("Performance metrics reset")


# Factory function for easy integration
async def create_adaptive_batcher(
    inference_func: Callable,
    max_batch_size: int = 32,
    base_timeout_ms: int = 10
) -> AdaptiveBatchProcessor:
    """
    Create and start an adaptive batch processor.

    Args:
        inference_func: Function to call for batch inference
        max_batch_size: Maximum batch size
        base_timeout_ms: Base timeout for batching

    Returns:
        Started AdaptiveBatchProcessor instance
    """
    batcher = AdaptiveBatchProcessor(
        inference_func=inference_func,
        max_batch_size=max_batch_size,
        base_timeout_ms=base_timeout_ms
    )

    await batcher.start()
    return batcher
