"""
Advanced dynamic batching system for ITS Camera AI.

Intelligent batching that optimizes for both latency and throughput,
with adaptive batch sizing, priority queuing, and deadline scheduling.
"""

import asyncio
import heapq
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BatchPriority(Enum):
    """Priority levels for batching requests."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class InferenceRequest:
    """Represents a single inference request."""

    request_id: str
    data: torch.Tensor
    camera_id: str
    frame_id: str
    timestamp: float
    deadline: float | None = None
    priority: BatchPriority = BatchPriority.NORMAL
    callback: Callable | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: 'InferenceRequest') -> bool:
        """For priority queue ordering - higher priority and earlier deadline first."""
        if self.priority != other.priority:
            return self.priority.value > other.priority.value

        if self.deadline is not None and other.deadline is not None:
            return self.deadline < other.deadline

        return self.timestamp < other.timestamp


@dataclass
class BatchConfiguration:
    """Configuration for dynamic batching."""

    # Batch size settings
    min_batch_size: int = 1
    max_batch_size: int = 32
    optimal_batch_size: int = 8

    # Timing constraints
    max_wait_time_ms: float = 10.0  # Maximum time to wait for batch formation
    deadline_buffer_ms: float = 5.0  # Buffer time before deadline

    # Adaptive settings
    enable_adaptive_batching: bool = True
    latency_target_ms: float = 75.0
    throughput_target_fps: float = 100.0

    # Queue settings
    max_queue_size: int = 1000
    enable_priority_queuing: bool = True

    # Performance monitoring
    stats_window_size: int = 100
    adaptation_interval_ms: float = 1000.0


@dataclass
class BatchStats:
    """Statistics for batch performance monitoring."""

    avg_batch_size: float = 0.0
    avg_wait_time_ms: float = 0.0
    avg_processing_time_ms: float = 0.0
    throughput_fps: float = 0.0
    queue_depth: int = 0
    deadline_miss_rate: float = 0.0
    gpu_utilization: float = 0.0


class AdaptiveBatchSizer:
    """Intelligent batch size adaptation based on performance metrics."""

    def __init__(self, config: BatchConfiguration):
        self.config = config
        self.current_batch_size = config.optimal_batch_size

        # Performance tracking
        self.latency_history: deque = deque(maxlen=config.stats_window_size)
        self.throughput_history: deque = deque(maxlen=config.stats_window_size)
        self.utilization_history: deque = deque(maxlen=config.stats_window_size)

        # Adaptation parameters
        self.adaptation_factor = 0.1
        self.last_adaptation_time = time.time()

        logger.info(f"Initialized adaptive batch sizer with target batch size {self.current_batch_size}")

    def update_metrics(self, latency_ms: float, throughput_fps: float, gpu_util: float) -> None:
        """Update performance metrics for adaptation."""
        self.latency_history.append(latency_ms)
        self.throughput_history.append(throughput_fps)
        self.utilization_history.append(gpu_util)

    def should_adapt(self) -> bool:
        """Check if adaptation should be performed."""
        current_time = time.time()
        return (current_time - self.last_adaptation_time) * 1000 >= self.config.adaptation_interval_ms

    def adapt_batch_size(self) -> int:
        """Adapt batch size based on performance metrics."""
        if not self.should_adapt() or len(self.latency_history) < 10:
            return self.current_batch_size

        # Calculate average metrics
        avg_latency = np.mean(list(self.latency_history))
        avg_throughput = np.mean(list(self.throughput_history))
        avg_utilization = np.mean(list(self.utilization_history))

        # Adaptation logic
        old_batch_size = self.current_batch_size

        # Primary constraint: latency target
        if avg_latency > self.config.latency_target_ms:
            # Latency too high - reduce batch size
            if avg_utilization > 0.9:  # GPU is saturated
                self.current_batch_size = max(
                    self.config.min_batch_size,
                    int(self.current_batch_size * (1 - self.adaptation_factor))
                )
            # If GPU not saturated, latency issue might be elsewhere
        elif avg_latency < self.config.latency_target_ms * 0.8:
            # Latency is good - can we increase throughput?
            if avg_throughput < self.config.throughput_target_fps and avg_utilization < 0.8:
                self.current_batch_size = min(
                    self.config.max_batch_size,
                    int(self.current_batch_size * (1 + self.adaptation_factor))
                )

        # Secondary constraint: GPU utilization
        if avg_utilization < 0.6 and avg_latency < self.config.latency_target_ms:
            # GPU underutilized - increase batch size
            self.current_batch_size = min(
                self.config.max_batch_size,
                self.current_batch_size + 1
            )
        elif avg_utilization > 0.95:
            # GPU overutilized - decrease batch size
            self.current_batch_size = max(
                self.config.min_batch_size,
                self.current_batch_size - 1
            )

        # Log adaptation
        if old_batch_size != self.current_batch_size:
            logger.info(
                f"Adapted batch size: {old_batch_size} -> {self.current_batch_size} "
                f"(latency: {avg_latency:.1f}ms, throughput: {avg_throughput:.1f}fps, "
                f"GPU util: {avg_utilization:.1f}%)"
            )

        self.last_adaptation_time = time.time()
        return self.current_batch_size

    def get_current_batch_size(self) -> int:
        """Get the current optimal batch size."""
        return self.current_batch_size


class PriorityBatchQueue:
    """Priority-based queue for batching inference requests."""

    def __init__(self, config: BatchConfiguration):
        self.config = config

        # Priority queues for different priority levels
        self.queues: dict[BatchPriority, list[InferenceRequest]] = {
            priority: [] for priority in BatchPriority
        }

        # Deadline-based queue (min-heap)
        self.deadline_queue: list[InferenceRequest] = []

        # Request tracking
        self.total_requests = 0
        self.processed_requests = 0
        self.deadline_misses = 0

        # Synchronization
        self._lock = asyncio.Lock()

    async def enqueue(self, request: InferenceRequest) -> bool:
        """Add a request to the appropriate queue."""
        async with self._lock:
            # Check queue capacity
            current_size = sum(len(q) for q in self.queues.values())
            if current_size >= self.config.max_queue_size:
                logger.warning("Queue full, dropping request")
                return False

            # Add to priority queue
            if self.config.enable_priority_queuing:
                heapq.heappush(self.queues[request.priority], request)
            else:
                # Use normal priority for all requests
                heapq.heappush(self.queues[BatchPriority.NORMAL], request)

            # Add to deadline queue if deadline is set
            if request.deadline is not None:
                heapq.heappush(self.deadline_queue, request)

            self.total_requests += 1

            return True

    async def dequeue_batch(self, target_batch_size: int) -> list[InferenceRequest]:
        """Dequeue a batch of requests based on priority and deadlines."""
        async with self._lock:
            batch = []
            current_time = time.time()

            # First, handle urgent deadline requests
            while (self.deadline_queue and
                   len(batch) < target_batch_size and
                   self.deadline_queue[0].deadline is not None and
                   self.deadline_queue[0].deadline - current_time <= self.config.deadline_buffer_ms / 1000):

                urgent_request = heapq.heappop(self.deadline_queue)
                batch.append(urgent_request)

                # Remove from priority queue as well
                try:
                    self.queues[urgent_request.priority].remove(urgent_request)
                except ValueError:
                    pass  # Already removed

            # Fill remaining batch slots with priority-based selection
            for priority in reversed(list(BatchPriority)):  # High to low priority
                while (self.queues[priority] and
                       len(batch) < target_batch_size):

                    request = heapq.heappop(self.queues[priority])

                    # Skip if already in batch (from deadline queue)
                    if request in batch:
                        continue

                    # Check if deadline is still valid
                    if (request.deadline is not None and
                        request.deadline - current_time <= 0):
                        self.deadline_misses += 1
                        logger.warning(f"Request {request.request_id} missed deadline")
                        continue

                    batch.append(request)

                    # Remove from deadline queue if present
                    try:
                        self.deadline_queue.remove(request)
                        heapq.heapify(self.deadline_queue)
                    except ValueError:
                        pass  # Not in deadline queue

            self.processed_requests += len(batch)

            return batch

    def get_queue_depth(self) -> int:
        """Get total number of requests in all queues."""
        return sum(len(q) for q in self.queues.values())

    def get_deadline_miss_rate(self) -> float:
        """Get the rate of deadline misses."""
        if self.processed_requests == 0:
            return 0.0
        return self.deadline_misses / self.processed_requests

    async def cleanup_expired_requests(self) -> int:
        """Remove expired requests from queues."""
        async with self._lock:
            current_time = time.time()
            expired_count = 0

            # Clean deadline queue
            while (self.deadline_queue and
                   self.deadline_queue[0].deadline is not None and
                   self.deadline_queue[0].deadline < current_time):

                expired_request = heapq.heappop(self.deadline_queue)

                # Remove from priority queue
                try:
                    self.queues[expired_request.priority].remove(expired_request)
                except ValueError:
                    pass

                expired_count += 1
                self.deadline_misses += 1

            return expired_count


class AdvancedDynamicBatcher:
    """Advanced dynamic batching system with intelligent optimization."""

    def __init__(self, config: BatchConfiguration, inference_fn: Callable | None = None):
        self.config = config
        self.inference_fn = inference_fn

        # Core components
        self.queue = PriorityBatchQueue(config)
        self.batch_sizer = AdaptiveBatchSizer(config) if config.enable_adaptive_batching else None

        # Performance tracking
        self.stats = BatchStats()
        self.batch_times: deque = deque(maxlen=config.stats_window_size)
        self.wait_times: deque = deque(maxlen=config.stats_window_size)
        self.processing_times: deque = deque(maxlen=config.stats_window_size)

        # Control flags
        self.running = False
        self.batch_task: asyncio.Task | None = None

        logger.info("Initialized advanced dynamic batcher")

    async def start(self) -> None:
        """Start the batching system."""
        if self.running:
            return

        self.running = True
        self.batch_task = asyncio.create_task(self._batch_processing_loop())

        logger.info("Started dynamic batching system")

    async def stop(self) -> None:
        """Stop the batching system."""
        if not self.running:
            return

        self.running = False

        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped dynamic batching system")

    async def submit_request(self, request: InferenceRequest) -> bool:
        """Submit an inference request for batching."""
        if not self.running:
            logger.error("Batcher not running")
            return False

        return await self.queue.enqueue(request)

    async def _batch_processing_loop(self) -> None:
        """Main batch processing loop."""
        logger.info("Started batch processing loop")

        while self.running:
            try:
                # Determine target batch size
                if self.batch_sizer:
                    target_batch_size = self.batch_sizer.adapt_batch_size()
                else:
                    target_batch_size = self.config.optimal_batch_size

                # Wait for requests or timeout
                batch_start_time = time.time()

                # Collect batch with timeout
                batch = await self._collect_batch(target_batch_size)

                if not batch:
                    # No requests available, short sleep
                    await asyncio.sleep(0.001)
                    continue

                # Calculate wait time
                wait_time = (time.time() - batch_start_time) * 1000
                self.wait_times.append(wait_time)

                # Process batch
                await self._process_batch(batch)

                # Update statistics
                self._update_stats()

                # Cleanup expired requests periodically
                if len(self.batch_times) % 100 == 0:
                    expired = await self.queue.cleanup_expired_requests()
                    if expired > 0:
                        logger.debug(f"Cleaned up {expired} expired requests")

            except asyncio.CancelledError:
                logger.info("Batch processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(0.1)  # Prevent tight error loop

    async def _collect_batch(self, target_batch_size: int) -> list[InferenceRequest]:
        """Collect a batch of requests with timeout."""
        batch = []
        start_time = time.time()
        max_wait_time = self.config.max_wait_time_ms / 1000

        while len(batch) < target_batch_size:
            # Check for available requests
            if self.queue.get_queue_depth() == 0:
                # No requests, wait a bit
                await asyncio.sleep(0.001)

                # Check timeout
                if time.time() - start_time >= max_wait_time:
                    break
                continue

            # Dequeue available requests
            new_requests = await self.queue.dequeue_batch(target_batch_size - len(batch))
            batch.extend(new_requests)

            # If we have enough requests, break
            if len(batch) >= self.config.min_batch_size:
                break

            # Check timeout for partial batch
            if time.time() - start_time >= max_wait_time:
                break

            # Small delay to allow more requests to arrive
            await asyncio.sleep(0.001)

        return batch

    async def _process_batch(self, batch: list[InferenceRequest]) -> None:
        """Process a batch of inference requests."""
        if not batch:
            return

        batch_start_time = time.time()

        try:
            # Stack tensors into batch
            batch_data = torch.stack([req.data for req in batch])

            # Perform inference
            if self.inference_fn:
                results = await self.inference_fn(batch_data)
            else:
                # Simulate inference for testing
                await asyncio.sleep(0.05)  # 50ms simulated inference
                results = [{'boxes': [], 'scores': [], 'classes': []} for _ in batch]

            # Process results and invoke callbacks
            for i, (request, result) in enumerate(zip(batch, results, strict=False)):
                if request.callback:
                    try:
                        if asyncio.iscoroutinefunction(request.callback):
                            await request.callback(result)
                        else:
                            request.callback(result)
                    except Exception as e:
                        logger.error(f"Error in callback for request {request.request_id}: {e}")

            # Record processing time
            processing_time = (time.time() - batch_start_time) * 1000
            self.processing_times.append(processing_time)

            # Record batch size
            self.batch_times.append(len(batch))

            logger.debug(f"Processed batch of {len(batch)} requests in {processing_time:.2f}ms")

        except Exception as e:
            logger.error(f"Error processing batch: {e}")

    def _update_stats(self) -> None:
        """Update performance statistics."""
        if not self.batch_times:
            return

        # Calculate averages
        self.stats.avg_batch_size = np.mean(list(self.batch_times))
        self.stats.avg_wait_time_ms = np.mean(list(self.wait_times)) if self.wait_times else 0.0
        self.stats.avg_processing_time_ms = np.mean(list(self.processing_times)) if self.processing_times else 0.0

        # Calculate throughput
        if len(self.processing_times) >= 2:
            total_requests = sum(self.batch_times)
            total_time = sum(self.processing_times) / 1000  # Convert to seconds
            self.stats.throughput_fps = total_requests / total_time if total_time > 0 else 0.0

        # Update queue depth and deadline miss rate
        self.stats.queue_depth = self.queue.get_queue_depth()
        self.stats.deadline_miss_rate = self.queue.get_deadline_miss_rate()

        # Update batch sizer metrics
        if self.batch_sizer:
            self.batch_sizer.update_metrics(
                self.stats.avg_processing_time_ms,
                self.stats.throughput_fps,
                self.stats.gpu_utilization
            )

    def get_stats(self) -> BatchStats:
        """Get current batching statistics."""
        self._update_stats()
        return self.stats

    def set_inference_function(self, inference_fn: Callable) -> None:
        """Set the inference function for batch processing."""
        self.inference_fn = inference_fn


# Example usage and testing
async def test_advanced_batcher():
    """Test the advanced dynamic batcher."""
    config = BatchConfiguration(
        min_batch_size=1,
        max_batch_size=16,
        optimal_batch_size=4,
        max_wait_time_ms=20.0,
        enable_adaptive_batching=True,
        latency_target_ms=100.0
    )

    # Mock inference function
    async def mock_inference(batch_data: torch.Tensor) -> list[dict[str, Any]]:
        # Simulate processing time based on batch size
        processing_time = 0.02 + (batch_data.shape[0] * 0.005)  # 20ms + 5ms per item
        await asyncio.sleep(processing_time)

        return [{'detection_count': i} for i in range(batch_data.shape[0])]

    # Create batcher
    batcher = AdvancedDynamicBatcher(config, mock_inference)
    await batcher.start()

    # Submit test requests
    results = []

    async def result_callback(result: dict[str, Any]) -> None:
        results.append(result)

    # Submit requests with different priorities
    for i in range(50):
        priority = BatchPriority.HIGH if i % 10 == 0 else BatchPriority.NORMAL
        deadline = time.time() + 0.5 if i % 5 == 0 else None

        request = InferenceRequest(
            request_id=f"req_{i}",
            data=torch.randn(3, 640, 640),
            camera_id=f"cam_{i % 5}",
            frame_id=f"frame_{i}",
            timestamp=time.time(),
            deadline=deadline,
            priority=priority,
            callback=result_callback
        )

        await batcher.submit_request(request)

        # Vary submission rate
        if i % 10 == 0:
            await asyncio.sleep(0.1)  # Burst
        else:
            await asyncio.sleep(0.02)  # Steady rate

    # Wait for processing to complete
    await asyncio.sleep(2.0)

    # Get statistics
    stats = batcher.get_stats()

    await batcher.stop()

    print("Advanced Batcher Test Results:")
    print(f"  Processed results: {len(results)}")
    print(f"  Average batch size: {stats.avg_batch_size:.1f}")
    print(f"  Average wait time: {stats.avg_wait_time_ms:.1f}ms")
    print(f"  Average processing time: {stats.avg_processing_time_ms:.1f}ms")
    print(f"  Throughput: {stats.throughput_fps:.1f} FPS")
    print(f"  Deadline miss rate: {stats.deadline_miss_rate:.2%}")
    print(f"  Final queue depth: {stats.queue_depth}")


if __name__ == "__main__":
    asyncio.run(test_advanced_batcher())
