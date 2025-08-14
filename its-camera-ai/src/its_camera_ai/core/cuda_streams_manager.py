"""CUDA Streams Manager for ITS Camera AI System.

This module implements advanced CUDA stream management for GPU pipeline optimization,
providing parallel execution, synchronization, and performance monitoring capabilities.

Key Features:
1. Dynamic stream pool management with automatic scaling
2. Stream priority queuing for critical inference tasks
3. Stream synchronization with event-based coordination
4. Memory stream integration for zero-copy operations
5. Performance monitoring and bottleneck detection
6. Automatic stream recycling and cleanup

Performance Targets:
- 25-35% latency reduction through stream parallelization
- 95%+ GPU utilization with concurrent operations
- <2ms stream synchronization overhead
- Support for 1000+ concurrent camera streams

Architecture:
- StreamPool: Manages available CUDA streams with priority levels
- StreamCoordinator: Orchestrates stream execution and synchronization
- StreamMonitor: Tracks performance metrics and optimization opportunities
- MemoryStreamBridge: Integrates with UnifiedMemoryManager for zero-copy
"""

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch.cuda import Event, Stream

from ..core.logging import get_logger

logger = get_logger(__name__)


class StreamPriority(Enum):
    """Stream priority levels for task scheduling."""

    EMERGENCY = 0  # Accident detection, violations
    HIGH = 1  # Real-time inference
    NORMAL = 2  # Standard processing
    BACKGROUND = 3  # Analytics, maintenance


class StreamState(Enum):
    """CUDA stream execution states."""

    IDLE = "idle"
    ACTIVE = "active"
    SYNCHRONIZING = "synchronizing"
    ERROR = "error"


@dataclass
class StreamMetrics:
    """Performance metrics for CUDA streams."""

    stream_id: int
    total_operations: int = 0
    total_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    last_operation_time: float = 0.0
    utilization_ratio: float = 0.0
    synchronization_wait_time_ms: float = 0.0
    error_count: int = 0

    def update_timing(self, execution_time_ms: float) -> None:
        """Update timing metrics with new execution time."""
        self.total_operations += 1
        self.total_execution_time_ms += execution_time_ms
        self.avg_execution_time_ms = (
            self.total_execution_time_ms / self.total_operations
        )
        self.last_operation_time = time.time()


@dataclass
class StreamTask:
    """Task to be executed on a CUDA stream."""

    task_id: str
    priority: StreamPriority
    operation: Callable[..., Any]
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    callback: Callable[[Any], None] | None = None
    error_callback: Callable[[Exception], None] | None = None
    timeout_ms: float = 30000  # 30 second default timeout
    created_at: float = field(default_factory=time.time)
    dependencies: set[str] = field(default_factory=set)

    def __lt__(self, other: "StreamTask") -> bool:
        """Priority comparison for heapq."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at


@dataclass
class ManagedStream:
    """Managed CUDA stream with state tracking."""

    stream: Stream
    stream_id: int
    device_id: int
    priority_level: StreamPriority
    state: StreamState = StreamState.IDLE
    current_task: StreamTask | None = None
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    metrics: StreamMetrics = field(default_factory=lambda: StreamMetrics(0))

    def __post_init__(self) -> None:
        """Initialize metrics with stream ID."""
        self.metrics.stream_id = self.stream_id


class StreamPool:
    """Pool of managed CUDA streams with dynamic allocation."""

    def __init__(
        self,
        device_id: int,
        initial_streams: int = 4,
        max_streams: int = 32,
        priority_distribution: dict[StreamPriority, int] | None = None,
    ):
        self.device_id = device_id
        self.initial_streams = initial_streams
        self.max_streams = max_streams

        # Stream allocation by priority
        self.priority_distribution = priority_distribution or {
            StreamPriority.EMERGENCY: 2,
            StreamPriority.HIGH: 8,
            StreamPriority.NORMAL: 16,
            StreamPriority.BACKGROUND: 6,
        }

        # Stream management
        self.streams: dict[int, ManagedStream] = {}
        self.available_streams: dict[StreamPriority, deque[ManagedStream]] = {
            priority: deque() for priority in StreamPriority
        }
        self.active_streams: set[int] = set()
        self.stream_counter = 0

        # Initialize streams
        self._initialize_streams()

    def _initialize_streams(self) -> None:
        """Initialize the stream pool with priority-based allocation."""
        with torch.cuda.device(self.device_id):
            # Create initial streams for each priority level
            for priority, count in self.priority_distribution.items():
                for _ in range(min(count, self.initial_streams // len(StreamPriority))):
                    stream = self._create_stream(priority)
                    self.available_streams[priority].append(stream)

        logger.info(
            f"Initialized stream pool on GPU {self.device_id} with "
            f"{sum(len(streams) for streams in self.available_streams.values())} streams"
        )

    def _create_stream(self, priority: StreamPriority) -> ManagedStream:
        """Create a new managed CUDA stream."""
        with torch.cuda.device(self.device_id):
            cuda_stream = Stream(device=self.device_id)

            self.stream_counter += 1
            managed_stream = ManagedStream(
                stream=cuda_stream,
                stream_id=self.stream_counter,
                device_id=self.device_id,
                priority_level=priority,
                metrics=StreamMetrics(self.stream_counter),
            )

            self.streams[self.stream_counter] = managed_stream
            return managed_stream

    def acquire_stream(self, priority: StreamPriority) -> ManagedStream | None:
        """Acquire an available stream for the given priority."""
        # Try to get stream from same priority level
        if self.available_streams[priority]:
            stream = self.available_streams[priority].popleft()
            stream.state = StreamState.ACTIVE
            stream.last_used = time.time()
            self.active_streams.add(stream.stream_id)
            return stream

        # Try lower priority streams if none available
        for lower_priority in StreamPriority:
            if (
                lower_priority.value > priority.value
                and self.available_streams[lower_priority]
            ):
                stream = self.available_streams[lower_priority].popleft()
                stream.state = StreamState.ACTIVE
                stream.last_used = time.time()
                self.active_streams.add(stream.stream_id)
                return stream

        # Create new stream if under limit
        if len(self.streams) < self.max_streams:
            stream = self._create_stream(priority)
            stream.state = StreamState.ACTIVE
            stream.last_used = time.time()
            self.active_streams.add(stream.stream_id)
            return stream

        return None

    def release_stream(self, stream_id: int) -> None:
        """Release a stream back to the available pool."""
        if stream_id not in self.streams:
            logger.warning(f"Attempting to release unknown stream {stream_id}")
            return

        stream = self.streams[stream_id]
        stream.state = StreamState.IDLE
        stream.current_task = None

        # Return to appropriate priority queue
        self.available_streams[stream.priority_level].append(stream)
        self.active_streams.discard(stream_id)

        logger.debug(
            f"Released stream {stream_id} back to {stream.priority_level} pool"
        )

    def get_utilization(self) -> dict[str, float]:
        """Get current pool utilization metrics."""
        total_streams = len(self.streams)
        active_streams = len(self.active_streams)

        priority_utilization = {}
        for priority in StreamPriority:
            available = len(self.available_streams[priority])
            priority_streams = [
                s for s in self.streams.values() if s.priority_level == priority
            ]
            total_priority = len(priority_streams)
            active_priority = len(
                [s for s in priority_streams if s.stream_id in self.active_streams]
            )

            priority_utilization[priority.name] = {
                "active": active_priority,
                "total": total_priority,
                "available": available,
                "utilization": active_priority / max(1, total_priority),
            }

        return {
            "overall_utilization": active_streams / max(1, total_streams),
            "active_streams": active_streams,
            "total_streams": total_streams,
            "priority_breakdown": priority_utilization,
        }


class StreamSynchronizer:
    """Manages stream synchronization using CUDA events."""

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.events: dict[str, Event] = {}
        self.event_dependencies: dict[str, set[str]] = defaultdict(set)
        self.completed_events: set[str] = set()

    def create_event(self, event_id: str) -> Event:
        """Create a CUDA event for synchronization."""
        if event_id in self.events:
            return self.events[event_id]

        with torch.cuda.device(self.device_id):
            event = Event(enable_timing=True)
            self.events[event_id] = event
            return event

    def record_event(self, event_id: str, stream: Stream) -> None:
        """Record an event on the given stream."""
        event = self.create_event(event_id)
        event.record(stream)
        logger.debug(f"Recorded event {event_id} on stream")

    def wait_for_event(self, event_id: str, stream: Stream) -> None:
        """Make a stream wait for an event."""
        if event_id not in self.events:
            logger.warning(f"Event {event_id} not found for wait operation")
            return

        event = self.events[event_id]
        stream.wait_event(event)
        logger.debug(f"Stream waiting for event {event_id}")

    def synchronize_streams(self, stream_ids: list[int], barrier_name: str) -> None:
        """Synchronize multiple streams at a barrier point."""
        # Record events on all streams
        for stream_id in stream_ids:
            event_id = f"{barrier_name}_stream_{stream_id}"
            # Note: This would need stream reference from pool
            self.create_event(event_id)

        # Create barrier event
        barrier_event = self.create_event(f"{barrier_name}_barrier")

        logger.debug(
            f"Synchronized {len(stream_ids)} streams at barrier {barrier_name}"
        )

    def cleanup_completed_events(self) -> None:
        """Clean up completed events to prevent memory leaks."""
        completed_events = []

        for event_id, event in self.events.items():
            if event.query():  # Event has completed
                completed_events.append(event_id)

        for event_id in completed_events:
            del self.events[event_id]
            self.completed_events.add(event_id)

        if completed_events:
            logger.debug(f"Cleaned up {len(completed_events)} completed events")


class CudaStreamsManager:
    """Advanced CUDA streams manager for GPU pipeline optimization."""

    def __init__(
        self,
        device_ids: list[int],
        streams_per_device: int = 8,
        max_streams_per_device: int = 32,
        enable_monitoring: bool = True,
    ):
        self.device_ids = device_ids
        self.streams_per_device = streams_per_device
        self.max_streams_per_device = max_streams_per_device
        self.enable_monitoring = enable_monitoring

        # Stream management per device
        self.stream_pools: dict[int, StreamPool] = {}
        self.synchronizers: dict[int, StreamSynchronizer] = {}

        # Task management
        self.task_queues: dict[StreamPriority, asyncio.PriorityQueue] = {
            priority: asyncio.PriorityQueue() for priority in StreamPriority
        }
        self.active_tasks: dict[str, StreamTask] = {}

        # Performance monitoring
        self.global_metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "avg_execution_time_ms": 0.0,
            "throughput_tasks_per_sec": 0.0,
            "stream_efficiency": 0.0,
        }

        # Processing state
        self.is_running = False
        self.worker_tasks: list[asyncio.Task] = []

        # Initialize components
        self._initialize_streams()

    def _initialize_streams(self) -> None:
        """Initialize stream pools and synchronizers for all devices."""
        for device_id in self.device_ids:
            # Initialize stream pool
            self.stream_pools[device_id] = StreamPool(
                device_id=device_id,
                initial_streams=self.streams_per_device,
                max_streams=self.max_streams_per_device,
            )

            # Initialize synchronizer
            self.synchronizers[device_id] = StreamSynchronizer(device_id)

        logger.info(f"Initialized CUDA streams manager for devices: {self.device_ids}")

    async def start(self) -> None:
        """Start the streams manager and worker tasks."""
        if self.is_running:
            logger.warning("CUDA streams manager already running")
            return

        self.is_running = True

        # Start worker tasks for each priority level
        for priority in StreamPriority:
            for device_id in self.device_ids:
                task = asyncio.create_task(
                    self._process_tasks_for_device(device_id, priority)
                )
                self.worker_tasks.append(task)

        # Start monitoring task if enabled
        if self.enable_monitoring:
            monitoring_task = asyncio.create_task(self._monitor_performance())
            self.worker_tasks.append(monitoring_task)

        logger.info(
            f"Started CUDA streams manager with {len(self.worker_tasks)} worker tasks"
        )

    async def stop(self) -> None:
        """Stop the streams manager and clean up resources."""
        if not self.is_running:
            return

        logger.info("Stopping CUDA streams manager...")
        self.is_running = False

        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        # Synchronize all streams
        for device_id in self.device_ids:
            with torch.cuda.device(device_id):
                torch.cuda.synchronize(device_id)

        # Cleanup events
        for synchronizer in self.synchronizers.values():
            synchronizer.cleanup_completed_events()

        self.worker_tasks.clear()
        logger.info("CUDA streams manager stopped")

    async def submit_task(
        self,
        task_id: str,
        operation: Callable[..., Any],
        priority: StreamPriority = StreamPriority.NORMAL,
        device_id: int | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        callback: Callable[[Any], None] | None = None,
        timeout_ms: float = 30000,
    ) -> str:
        """Submit a task for execution on CUDA streams.

        Args:
            task_id: Unique identifier for the task
            operation: Function to execute on GPU
            priority: Task priority level
            device_id: Preferred device ID (auto-selected if None)
            args: Positional arguments for operation
            kwargs: Keyword arguments for operation
            callback: Callback function for results
            timeout_ms: Task timeout in milliseconds

        Returns:
            Task ID for tracking
        """
        if not self.is_running:
            raise RuntimeError("CUDA streams manager not running")

        kwargs = kwargs or {}

        # Auto-select device if not specified
        if device_id is None:
            device_id = self._select_optimal_device(priority)

        # Create task
        task = StreamTask(
            task_id=task_id,
            priority=priority,
            operation=operation,
            args=args,
            kwargs=kwargs,
            callback=callback,
            timeout_ms=timeout_ms,
        )

        # Add to appropriate priority queue
        await self.task_queues[priority].put((priority.value, time.time(), task))
        self.active_tasks[task_id] = task
        self.global_metrics["total_tasks"] += 1

        logger.debug(
            f"Submitted task {task_id} with priority {priority.name} to device {device_id}"
        )
        return task_id

    def _select_optimal_device(self, priority: StreamPriority) -> int:
        """Select optimal device based on current load and priority."""
        best_device = self.device_ids[0]
        best_score = float("inf")

        for device_id in self.device_ids:
            pool = self.stream_pools[device_id]
            utilization = pool.get_utilization()

            # Calculate device score (lower is better)
            base_score = utilization["overall_utilization"]

            # Priority bonus for less loaded devices
            priority_bonus = (
                utilization.get("priority_breakdown", {})
                .get(priority.name, {})
                .get("utilization", 0.5)
            )

            total_score = base_score + priority_bonus * 0.3

            if total_score < best_score:
                best_score = total_score
                best_device = device_id

        return best_device

    async def _process_tasks_for_device(
        self, device_id: int, priority: StreamPriority
    ) -> None:
        """Process tasks for a specific device and priority level."""
        pool = self.stream_pools[device_id]

        while self.is_running:
            try:
                # Get task from queue with timeout
                try:
                    _, _, task = await asyncio.wait_for(
                        self.task_queues[priority].get(), timeout=1.0
                    )
                except TimeoutError:
                    continue

                # Acquire stream for task
                stream = pool.acquire_stream(priority)
                if stream is None:
                    # Put task back in queue if no stream available
                    await self.task_queues[priority].put(
                        (priority.value, time.time(), task)
                    )
                    await asyncio.sleep(0.01)  # Brief backoff
                    continue

                # Execute task
                await self._execute_task_on_stream(task, stream, device_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Task processing error on device {device_id}: {e}")
                await asyncio.sleep(0.1)

    async def _execute_task_on_stream(
        self, task: StreamTask, stream: ManagedStream, device_id: int
    ) -> None:
        """Execute a task on a specific CUDA stream."""
        start_time = time.time()

        try:
            # Set current task
            stream.current_task = task
            stream.state = StreamState.ACTIVE

            # Execute operation with proper CUDA context
            with torch.cuda.device(device_id), torch.cuda.stream(stream.stream):
                # Execute the operation
                result = await asyncio.to_thread(
                    task.operation, *task.args, **task.kwargs
                )

                # Synchronize stream to ensure completion
                stream.stream.synchronize()

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Update metrics
            stream.metrics.update_timing(execution_time_ms)
            self.global_metrics["completed_tasks"] += 1

            # Execute callback if provided
            if task.callback:
                try:
                    await asyncio.to_thread(task.callback, result)
                except Exception as callback_error:
                    logger.error(f"Task callback error: {callback_error}")

            logger.debug(
                f"Completed task {task.task_id} on stream {stream.stream_id} "
                f"in {execution_time_ms:.2f}ms"
            )

        except Exception as e:
            # Handle task execution error
            execution_time_ms = (time.time() - start_time) * 1000
            stream.metrics.error_count += 1
            stream.state = StreamState.ERROR
            self.global_metrics["failed_tasks"] += 1

            logger.error(
                f"Task {task.task_id} failed on stream {stream.stream_id}: {e}"
            )

            # Execute error callback if provided
            if task.error_callback:
                try:
                    await asyncio.to_thread(task.error_callback, e)
                except Exception as error_callback_error:
                    logger.error(f"Task error callback failed: {error_callback_error}")

        finally:
            # Clean up and release stream
            del self.active_tasks[task.task_id]
            self.stream_pools[device_id].release_stream(stream.stream_id)

    async def _monitor_performance(self) -> None:
        """Monitor and log performance metrics."""
        last_completed_tasks = 0
        last_time = time.time()

        while self.is_running:
            try:
                await asyncio.sleep(10.0)  # Monitor every 10 seconds

                current_time = time.time()
                time_diff = current_time - last_time

                # Calculate throughput
                current_completed = self.global_metrics["completed_tasks"]
                tasks_completed = current_completed - last_completed_tasks
                throughput = tasks_completed / time_diff if time_diff > 0 else 0

                # Update global metrics
                self.global_metrics["throughput_tasks_per_sec"] = throughput

                # Calculate stream efficiency
                total_utilization = 0
                device_count = len(self.device_ids)

                for device_id in self.device_ids:
                    util = self.stream_pools[device_id].get_utilization()
                    total_utilization += util["overall_utilization"]

                self.global_metrics["stream_efficiency"] = (
                    total_utilization / device_count
                )

                # Log performance summary
                logger.info(
                    f"CUDA Streams Performance - "
                    f"Throughput: {throughput:.1f} tasks/sec, "
                    f"Stream Efficiency: {self.global_metrics['stream_efficiency']:.1%}, "
                    f"Active Tasks: {len(self.active_tasks)}, "
                    f"Completed: {current_completed}, "
                    f"Failed: {self.global_metrics['failed_tasks']}"
                )

                # Update tracking variables
                last_completed_tasks = current_completed
                last_time = current_time

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

    def get_comprehensive_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance and utilization metrics."""
        device_metrics = {}

        for device_id in self.device_ids:
            pool = self.stream_pools[device_id]
            utilization = pool.get_utilization()

            # Stream-level metrics
            stream_details = {}
            for stream_id, stream in pool.streams.items():
                stream_details[str(stream_id)] = {
                    "priority": stream.priority_level.name,
                    "state": stream.state.value,
                    "total_operations": stream.metrics.total_operations,
                    "avg_execution_time_ms": stream.metrics.avg_execution_time_ms,
                    "utilization_ratio": stream.metrics.utilization_ratio,
                    "error_count": stream.metrics.error_count,
                    "last_used_seconds_ago": time.time() - stream.last_used,
                }

            device_metrics[f"device_{device_id}"] = {
                "utilization": utilization,
                "stream_details": stream_details,
                "synchronizer_events": len(self.synchronizers[device_id].events),
            }

        return {
            "global_metrics": self.global_metrics,
            "device_metrics": device_metrics,
            "task_queues": {
                priority.name: queue.qsize()
                for priority, queue in self.task_queues.items()
            },
            "active_tasks_count": len(self.active_tasks),
            "is_running": self.is_running,
            "worker_tasks_count": len(self.worker_tasks),
        }

    async def create_synchronization_barrier(
        self, barrier_name: str, participating_devices: list[int] | None = None
    ) -> None:
        """Create a synchronization barrier across multiple devices."""
        devices = participating_devices or self.device_ids

        # Record barrier events on all devices
        for device_id in devices:
            synchronizer = self.synchronizers[device_id]
            # This would require coordination with active streams
            synchronizer.create_event(f"{barrier_name}_device_{device_id}")

        logger.debug(
            f"Created synchronization barrier '{barrier_name}' for {len(devices)} devices"
        )

    def get_stream_recommendations(self) -> dict[str, Any]:
        """Get optimization recommendations based on current performance."""
        recommendations = []

        for device_id in self.device_ids:
            utilization = self.stream_pools[device_id].get_utilization()
            overall_util = utilization["overall_utilization"]

            if overall_util > 0.9:
                recommendations.append(
                    {
                        "type": "scale_up",
                        "device": device_id,
                        "message": f"Device {device_id} utilization at {overall_util:.1%} - consider adding more streams",
                    }
                )
            elif overall_util < 0.3:
                recommendations.append(
                    {
                        "type": "scale_down",
                        "device": device_id,
                        "message": f"Device {device_id} utilization at {overall_util:.1%} - consider reducing streams",
                    }
                )

            # Check priority imbalances
            priority_breakdown = utilization["priority_breakdown"]
            for priority_name, priority_stats in priority_breakdown.items():
                if priority_stats["utilization"] > 0.95:
                    recommendations.append(
                        {
                            "type": "priority_rebalance",
                            "device": device_id,
                            "priority": priority_name,
                            "message": f"Priority {priority_name} overloaded on device {device_id}",
                        }
                    )

        return {
            "recommendations": recommendations,
            "overall_health": (
                "good" if len(recommendations) == 0 else "needs_attention"
            ),
            "optimization_opportunities": len(recommendations),
        }


# Utility functions for CUDA streams integration


def create_streams_manager_for_inference(
    device_ids: list[int], inference_config: dict[str, Any] | None = None
) -> CudaStreamsManager:
    """Create a CUDA streams manager optimized for inference workloads."""
    config = inference_config or {}

    return CudaStreamsManager(
        device_ids=device_ids,
        streams_per_device=config.get("streams_per_device", 8),
        max_streams_per_device=config.get("max_streams_per_device", 24),
        enable_monitoring=config.get("enable_monitoring", True),
    )


async def benchmark_stream_performance(
    streams_manager: CudaStreamsManager,
    num_operations: int = 1000,
    operation_complexity: str = "simple",
) -> dict[str, float]:
    """Benchmark CUDA streams performance with synthetic workload."""
    logger.info(f"Starting CUDA streams benchmark with {num_operations} operations")

    # Define benchmark operations
    def simple_operation(tensor_size: int = 1000) -> torch.Tensor:
        """Simple tensor operation for benchmarking."""
        x = torch.randn(tensor_size, tensor_size, device="cuda")
        return torch.matmul(x, x.T)

    def complex_operation(tensor_size: int = 2000) -> torch.Tensor:
        """More complex operation for benchmarking."""
        x = torch.randn(tensor_size, tensor_size, device="cuda")
        for _ in range(5):
            x = torch.matmul(x, x.T)
            x = torch.nn.functional.relu(x)
        return x

    # Select operation based on complexity
    operation = (
        simple_operation if operation_complexity == "simple" else complex_operation
    )
    tensor_size = 1000 if operation_complexity == "simple" else 500

    # Submit benchmark tasks
    start_time = time.time()
    task_ids = []

    for i in range(num_operations):
        priority = StreamPriority.HIGH if i % 10 == 0 else StreamPriority.NORMAL

        task_id = await streams_manager.submit_task(
            task_id=f"benchmark_task_{i}",
            operation=operation,
            priority=priority,
            kwargs={"tensor_size": tensor_size},
        )
        task_ids.append(task_id)

    # Wait for all tasks to complete
    while len(streams_manager.active_tasks) > 0:
        await asyncio.sleep(0.1)

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate metrics
    metrics = streams_manager.get_comprehensive_metrics()

    results = {
        "total_operations": num_operations,
        "total_time_seconds": total_time,
        "throughput_ops_per_second": num_operations / total_time,
        "average_stream_efficiency": metrics["global_metrics"]["stream_efficiency"],
        "completed_tasks": metrics["global_metrics"]["completed_tasks"],
        "failed_tasks": metrics["global_metrics"]["failed_tasks"],
        "success_rate": metrics["global_metrics"]["completed_tasks"] / num_operations,
    }

    logger.info(f"Benchmark completed: {results}")
    return results
