"""
Enhanced GPU memory management for ITS Camera AI.

Advanced memory pool management with pre-allocation, CUDA streams,
and zero-copy operations for maximum inference performance.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    allocated_mb: float
    cached_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    utilization_percent: float
    fragmentation_percent: float


@dataclass
class TensorPoolConfig:
    """Configuration for tensor memory pools."""

    # Pool sizes
    max_tensors_per_shape: int = 8
    max_total_tensors: int = 200

    # Memory limits
    max_memory_fraction: float = 0.85
    emergency_cleanup_threshold: float = 0.95

    # Performance settings
    enable_pinned_memory: bool = True
    enable_memory_mapping: bool = True

    # Monitoring
    enable_memory_profiling: bool = True
    stats_update_interval: float = 5.0


class CUDAStreamPool:
    """Pool of CUDA streams for concurrent operations."""

    def __init__(self, device_id: int, num_streams: int = 8):
        self.device_id = device_id
        self.num_streams = num_streams
        self.streams: deque[torch.cuda.Stream] = deque()
        self.in_use: dict[int, torch.cuda.Stream] = {}
        self._lock = Lock()

        # Create streams
        with torch.cuda.device(device_id):
            for _ in range(num_streams):
                stream = torch.cuda.Stream(device=device_id)
                self.streams.append(stream)

        logger.debug(f"Created {num_streams} CUDA streams for device {device_id}")

    def get_stream(self) -> torch.cuda.Stream | None:
        """Get an available CUDA stream."""
        with self._lock:
            if self.streams:
                stream = self.streams.popleft()
                self.in_use[id(stream)] = stream
                return stream
        return None

    def return_stream(self, stream: torch.cuda.Stream) -> None:
        """Return a stream to the pool."""
        with self._lock:
            stream_id = id(stream)
            if stream_id in self.in_use:
                del self.in_use[stream_id]
                self.streams.append(stream)

    def synchronize_all(self) -> None:
        """Synchronize all streams."""
        with self._lock:
            for stream in list(self.in_use.values()) + list(self.streams):
                stream.synchronize()

    def cleanup(self) -> None:
        """Clean up all streams."""
        self.synchronize_all()
        with self._lock:
            self.streams.clear()
            self.in_use.clear()


class EnhancedTensorPool:
    """Enhanced tensor memory pool with advanced caching strategies."""

    def __init__(self, device_id: int, config: TensorPoolConfig):
        self.device_id = device_id
        self.config = config

        # Memory pools organized by shape and dtype
        self.pools: dict[tuple[tuple[int, ...], torch.dtype], deque] = defaultdict(deque)
        self.in_use: dict[int, tuple[torch.Tensor, float]] = {}  # tensor_id -> (tensor, timestamp)

        # Memory tracking
        self.total_tensors = 0
        self.allocated_memory = 0
        self.peak_memory = 0

        # Synchronization
        self._pool_lock = Lock()
        self._stats_lock = Lock()

        # Pinned memory for CPU-GPU transfers
        self.pinned_buffers: dict[tuple[int, ...], list[torch.Tensor]] = defaultdict(list)

        # Pre-allocate common tensor shapes
        self._pre_allocate_common_shapes()

        # Start memory monitoring
        if config.enable_memory_profiling:
            self._start_memory_monitoring()

    def _pre_allocate_common_shapes(self) -> None:
        """Pre-allocate tensors for common shapes used in inference."""
        torch.cuda.set_device(self.device_id)

        # Common shapes for different batch sizes and resolutions
        common_shapes = [
            # Input tensors (batch_size, channels, height, width)
            (1, 3, 640, 640),    # Single frame standard
            (1, 3, 1280, 1280),  # Single frame high-res
            (2, 3, 640, 640),    # Pair batch
            (4, 3, 640, 640),    # Small batch
            (8, 3, 640, 640),    # Standard batch
            (16, 3, 640, 640),   # Large batch
            (32, 3, 640, 640),   # Max batch

            # Output tensors (various detection formats)
            (1, 25200, 85),      # YOLO11 single output
            (4, 25200, 85),      # YOLO11 batch output
            (8, 25200, 85),      # YOLO11 large batch
            (1, 8400, 85),       # YOLO11 reduced anchors
            (4, 8400, 85),       # YOLO11 reduced batch

            # Intermediate tensors
            (1, 256, 80, 80),    # Feature maps
            (1, 512, 40, 40),
            (1, 1024, 20, 20),
            (4, 256, 80, 80),    # Batch feature maps
            (4, 512, 40, 40),
            (4, 1024, 20, 20),
        ]

        # Data types to pre-allocate
        dtypes = [torch.float32, torch.float16]

        try:
            for shape in common_shapes:
                for dtype in dtypes:
                    # Pre-allocate multiple tensors per shape/dtype
                    pool_key = (shape, dtype)

                    for _ in range(min(4, self.config.max_tensors_per_shape)):
                        if self.total_tensors >= self.config.max_total_tensors:
                            break

                        try:
                            tensor = torch.empty(
                                shape,
                                dtype=dtype,
                                device=f'cuda:{self.device_id}',
                                memory_format=torch.channels_last if len(shape) == 4 else torch.contiguous_format
                            )

                            self.pools[pool_key].append(tensor)
                            self.total_tensors += 1
                            self.allocated_memory += tensor.numel() * tensor.element_size()

                        except torch.cuda.OutOfMemoryError:
                            logger.warning(f"OOM while pre-allocating shape {shape} dtype {dtype}")
                            break

                    logger.debug(f"Pre-allocated {len(self.pools[pool_key])} tensors for shape {shape} dtype {dtype}")

            # Pre-allocate pinned memory buffers for common shapes
            if self.config.enable_pinned_memory:
                self._pre_allocate_pinned_memory(common_shapes[:10])  # Limit to avoid excessive CPU memory

            self.peak_memory = self.allocated_memory

            logger.info(f"Pre-allocated {self.total_tensors} tensors on device {self.device_id} "
                       f"({self.allocated_memory / 1024 / 1024:.1f} MB)")

        except Exception as e:
            logger.error(f"Error during pre-allocation: {e}")

    def _pre_allocate_pinned_memory(self, shapes: list[tuple[int, ...]]) -> None:
        """Pre-allocate pinned CPU memory for efficient transfers."""
        if not self.config.enable_pinned_memory:
            return

        for shape in shapes:
            try:
                # Allocate 2 pinned buffers per shape for double buffering
                for _ in range(2):
                    pinned_tensor = torch.empty(shape, dtype=torch.float32, pin_memory=True)
                    self.pinned_buffers[shape].append(pinned_tensor)

                logger.debug(f"Pre-allocated pinned memory for shape {shape}")

            except Exception as e:
                logger.warning(f"Failed to allocate pinned memory for shape {shape}: {e}")

    def get_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        memory_format: torch.memory_format = torch.contiguous_format
    ) -> torch.Tensor:
        """Get a tensor from the pool or allocate new one."""

        pool_key = (shape, dtype)

        with self._pool_lock:
            # Try to get from pool first
            if pool_key in self.pools and self.pools[pool_key]:
                tensor = self.pools[pool_key].popleft()

                # Zero out the tensor for clean slate
                tensor.zero_()

                # Track as in-use
                self.in_use[id(tensor)] = (tensor, time.time())

                return tensor

        # Allocate new tensor if pool is empty
        try:
            torch.cuda.set_device(self.device_id)

            # Use channels_last format for 4D tensors if not specified
            if len(shape) == 4 and memory_format == torch.contiguous_format:
                memory_format = torch.channels_last

            tensor = torch.empty(
                shape,
                dtype=dtype,
                device=f'cuda:{self.device_id}',
                memory_format=memory_format
            )

            # Track allocation
            with self._stats_lock:
                self.total_tensors += 1
                self.allocated_memory += tensor.numel() * tensor.element_size()
                self.peak_memory = max(self.peak_memory, self.allocated_memory)

            # Track as in-use
            self.in_use[id(tensor)] = (tensor, time.time())

            return tensor

        except torch.cuda.OutOfMemoryError:
            # Emergency cleanup and retry
            logger.warning("OOM detected, performing emergency cleanup")
            self._emergency_cleanup()

            # Retry allocation
            try:
                tensor = torch.empty(shape, dtype=dtype, device=f'cuda:{self.device_id}')
                self.in_use[id(tensor)] = (tensor, time.time())
                return tensor
            except torch.cuda.OutOfMemoryError:
                logger.error(f"Failed to allocate tensor {shape} {dtype} even after cleanup")
                raise

    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return a tensor to the pool."""
        tensor_id = id(tensor)

        with self._pool_lock:
            if tensor_id not in self.in_use:
                return  # Tensor not tracked

            # Remove from in-use tracking
            del self.in_use[tensor_id]

            # Get pool key
            shape = tuple(tensor.shape)
            dtype = tensor.dtype
            pool_key = (shape, dtype)

            # Return to pool if not full
            if len(self.pools[pool_key]) < self.config.max_tensors_per_shape:
                self.pools[pool_key].append(tensor)
            else:
                # Pool is full, let tensor be garbage collected
                with self._stats_lock:
                    self.total_tensors -= 1
                    self.allocated_memory -= tensor.numel() * tensor.element_size()

    def get_pinned_buffer(self, shape: tuple[int, ...]) -> torch.Tensor | None:
        """Get a pinned CPU buffer for efficient CPU-GPU transfers."""
        if not self.config.enable_pinned_memory:
            return None

        if shape in self.pinned_buffers and self.pinned_buffers[shape]:
            return self.pinned_buffers[shape].pop()

        # Allocate new pinned buffer
        try:
            return torch.empty(shape, dtype=torch.float32, pin_memory=True)
        except Exception as e:
            logger.warning(f"Failed to allocate pinned buffer: {e}")
            return None

    def return_pinned_buffer(self, tensor: torch.Tensor) -> None:
        """Return a pinned buffer to the pool."""
        if not tensor.is_pinned():
            return

        shape = tuple(tensor.shape)
        if len(self.pinned_buffers[shape]) < 4:  # Limit pinned buffer pool size
            self.pinned_buffers[shape].append(tensor)

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup when running out of memory."""
        logger.warning("Performing emergency memory cleanup")

        with self._pool_lock:
            # Clear all pools
            total_freed = 0
            for pool in self.pools.values():
                total_freed += len(pool)
                pool.clear()

            # Force garbage collection
            torch.cuda.empty_cache()

            with self._stats_lock:
                self.total_tensors -= total_freed
                # Recalculate allocated memory
                self.allocated_memory = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor, _ in self.in_use.values()
                )

            logger.info(f"Emergency cleanup freed {total_freed} tensors")

    def _start_memory_monitoring(self) -> None:
        """Start background memory monitoring."""
        async def monitor():
            while True:
                try:
                    await asyncio.sleep(self.config.stats_update_interval)
                    stats = self.get_memory_stats()

                    if stats.utilization_percent > self.config.emergency_cleanup_threshold * 100:
                        logger.warning(f"High memory usage detected: {stats.utilization_percent:.1f}%")
                        self._emergency_cleanup()

                except Exception as e:
                    logger.error(f"Error in memory monitoring: {e}")

        # Start monitoring task
        asyncio.create_task(monitor())

    def get_memory_stats(self) -> MemoryStats:
        """Get detailed memory statistics."""
        torch.cuda.set_device(self.device_id)

        # PyTorch memory stats
        allocated = torch.cuda.memory_allocated(self.device_id)
        cached = torch.cuda.memory_reserved(self.device_id)

        # GPU device memory info
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                total_mb = info.total / 1024 / 1024
                free_mb = info.free / 1024 / 1024

            except Exception:
                # Fallback to PyTorch estimates
                total_mb = torch.cuda.get_device_properties(self.device_id).total_memory / 1024 / 1024
                free_mb = total_mb - (allocated / 1024 / 1024)
        else:
            total_mb = torch.cuda.get_device_properties(self.device_id).total_memory / 1024 / 1024
            free_mb = total_mb - (allocated / 1024 / 1024)

        allocated_mb = allocated / 1024 / 1024
        cached_mb = cached / 1024 / 1024
        reserved_mb = cached_mb

        utilization_percent = (allocated_mb / total_mb) * 100
        fragmentation_percent = ((cached_mb - allocated_mb) / total_mb) * 100 if cached_mb > allocated_mb else 0

        return MemoryStats(
            allocated_mb=allocated_mb,
            cached_mb=cached_mb,
            reserved_mb=reserved_mb,
            free_mb=free_mb,
            total_mb=total_mb,
            utilization_percent=utilization_percent,
            fragmentation_percent=fragmentation_percent
        )

    def get_pool_stats(self) -> dict[str, Any]:
        """Get tensor pool statistics."""
        with self._pool_lock, self._stats_lock:
            pool_sizes = {f"{shape}_{dtype}": len(pool) for (shape, dtype), pool in self.pools.items()}

            return {
                'total_tensors_pooled': sum(len(pool) for pool in self.pools.values()),
                'total_tensors_in_use': len(self.in_use),
                'total_tensors_managed': self.total_tensors,
                'allocated_memory_mb': self.allocated_memory / 1024 / 1024,
                'peak_memory_mb': self.peak_memory / 1024 / 1024,
                'pool_sizes': pool_sizes,
                'pinned_buffers': {str(shape): len(buffers) for shape, buffers in self.pinned_buffers.items()}
            }

    def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info(f"Cleaning up tensor pool for device {self.device_id}")

        with self._pool_lock:
            # Clear all pools
            for pool in self.pools.values():
                pool.clear()

            # Clear pinned buffers
            for buffers in self.pinned_buffers.values():
                buffers.clear()

            self.in_use.clear()

        # Force CUDA cache cleanup
        torch.cuda.empty_cache()

        logger.info("Tensor pool cleanup completed")


class MultiGPUMemoryManager:
    """Unified memory manager for multi-GPU inference."""

    def __init__(self, device_ids: list[int], config: TensorPoolConfig | None = None):
        self.device_ids = device_ids
        self.config = config or TensorPoolConfig()

        # Per-device components
        self.tensor_pools: dict[int, EnhancedTensorPool] = {}
        self.stream_pools: dict[int, CUDAStreamPool] = {}

        # Load balancing
        self.device_load: dict[int, float] = defaultdict(float)
        self.last_selected_device = 0

        # Initialize per-device resources
        self._initialize_devices()

        logger.info(f"Initialized multi-GPU memory manager for devices {device_ids}")

    def _initialize_devices(self) -> None:
        """Initialize memory management for each GPU device."""
        for device_id in self.device_ids:
            try:
                # Initialize tensor pool
                self.tensor_pools[device_id] = EnhancedTensorPool(device_id, self.config)

                # Initialize stream pool
                self.stream_pools[device_id] = CUDAStreamPool(device_id, num_streams=8)

                logger.info(f"Initialized memory management for GPU {device_id}")

            except Exception as e:
                logger.error(f"Failed to initialize GPU {device_id}: {e}")
                continue

    def select_optimal_device(self) -> int:
        """Select the optimal GPU device based on current load."""
        if len(self.device_ids) == 1:
            return self.device_ids[0]

        # Get memory utilization for each device
        device_scores = []

        for device_id in self.device_ids:
            if device_id not in self.tensor_pools:
                continue

            try:
                stats = self.tensor_pools[device_id].get_memory_stats()
                # Lower utilization = better score
                score = 100 - stats.utilization_percent
                device_scores.append((score, device_id))

            except Exception:
                # Fallback to round-robin
                device_scores.append((50, device_id))

        if device_scores:
            # Sort by score (highest first) and return best device
            device_scores.sort(reverse=True)
            return device_scores[0][1]

        # Fallback to round-robin
        selected = self.device_ids[self.last_selected_device]
        self.last_selected_device = (self.last_selected_device + 1) % len(self.device_ids)
        return selected

    def get_tensor(
        self,
        shape: tuple[int, ...],
        device_id: int | None = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Get a tensor from the specified or optimal device."""
        if device_id is None:
            device_id = self.select_optimal_device()

        if device_id not in self.tensor_pools:
            raise ValueError(f"Device {device_id} not managed")

        return self.tensor_pools[device_id].get_tensor(shape, dtype)

    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return a tensor to its device pool."""
        device_id = tensor.device.index

        if device_id in self.tensor_pools:
            self.tensor_pools[device_id].return_tensor(tensor)

    def get_stream(self, device_id: int | None = None) -> torch.cuda.Stream | None:
        """Get a CUDA stream from the specified or optimal device."""
        if device_id is None:
            device_id = self.select_optimal_device()

        if device_id in self.stream_pools:
            return self.stream_pools[device_id].get_stream()

        return None

    def return_stream(self, stream: torch.cuda.Stream) -> None:
        """Return a CUDA stream to its device pool."""
        device_id = stream.device.index

        if device_id in self.stream_pools:
            self.stream_pools[device_id].return_stream(stream)

    def get_overall_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics across all devices."""
        overall_stats = {
            'devices': {},
            'total_memory_allocated_mb': 0,
            'total_memory_peak_mb': 0,
            'average_utilization_percent': 0,
        }

        utilizations = []

        for device_id in self.device_ids:
            if device_id not in self.tensor_pools:
                continue

            try:
                memory_stats = self.tensor_pools[device_id].get_memory_stats()
                pool_stats = self.tensor_pools[device_id].get_pool_stats()

                device_stats = {
                    'memory': memory_stats,
                    'pool': pool_stats
                }

                overall_stats['devices'][f'gpu_{device_id}'] = device_stats
                overall_stats['total_memory_allocated_mb'] += memory_stats.allocated_mb

                utilizations.append(memory_stats.utilization_percent)

            except Exception as e:
                logger.error(f"Error getting stats for device {device_id}: {e}")

        if utilizations:
            overall_stats['average_utilization_percent'] = sum(utilizations) / len(utilizations)

        return overall_stats

    def cleanup(self) -> None:
        """Clean up all managed resources."""
        logger.info("Cleaning up multi-GPU memory manager")

        for device_id in self.device_ids:
            if device_id in self.tensor_pools:
                self.tensor_pools[device_id].cleanup()

            if device_id in self.stream_pools:
                self.stream_pools[device_id].cleanup()

        # Clear all tracking
        self.tensor_pools.clear()
        self.stream_pools.clear()
        self.device_load.clear()

        logger.info("Multi-GPU memory manager cleanup completed")


# Context manager for automatic resource management
class ManagedTensor:
    """Context manager for automatic tensor lifecycle management."""

    def __init__(
        self,
        memory_manager: MultiGPUMemoryManager,
        shape: tuple[int, ...],
        device_id: int | None = None,
        dtype: torch.dtype = torch.float32
    ):
        self.memory_manager = memory_manager
        self.shape = shape
        self.device_id = device_id
        self.dtype = dtype
        self.tensor: torch.Tensor | None = None

    def __enter__(self) -> torch.Tensor:
        self.tensor = self.memory_manager.get_tensor(self.shape, self.device_id, self.dtype)
        return self.tensor

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.tensor is not None:
            self.memory_manager.return_tensor(self.tensor)
            self.tensor = None


# Example usage and testing
async def benchmark_memory_manager(device_ids: list[int], iterations: int = 1000) -> dict[str, float]:
    """Benchmark the enhanced memory manager performance."""
    config = TensorPoolConfig(
        max_tensors_per_shape=16,
        max_total_tensors=500,
        enable_memory_profiling=True
    )

    manager = MultiGPUMemoryManager(device_ids, config)

    # Test tensor allocation/deallocation performance
    start_time = time.time()

    for i in range(iterations):
        # Allocate tensors of various sizes
        shapes = [
            (1, 3, 640, 640),
            (4, 3, 640, 640),
            (8, 3, 640, 640),
            (1, 25200, 85)
        ]

        tensors = []
        for shape in shapes:
            tensor = manager.get_tensor(shape)
            tensors.append(tensor)

            # Simulate some computation
            tensor.fill_(i % 255)

        # Return tensors
        for tensor in tensors:
            manager.return_tensor(tensor)

    total_time = time.time() - start_time

    # Get final statistics
    stats = manager.get_overall_stats()

    manager.cleanup()

    return {
        'total_time_s': total_time,
        'avg_time_per_iteration_ms': (total_time / iterations) * 1000,
        'allocations_per_second': (iterations * len(shapes)) / total_time,
        'peak_memory_mb': stats['total_memory_allocated_mb']
    }


if __name__ == "__main__":
    # Example usage
    async def main():
        # Test with available GPUs
        if torch.cuda.is_available():
            device_ids = list(range(torch.cuda.device_count()))
            print(f"Testing with GPUs: {device_ids}")

            results = await benchmark_memory_manager(device_ids, 1000)

            print("Benchmark Results:")
            for key, value in results.items():
                print(f"  {key}: {value}")
        else:
            print("No CUDA devices available")

    asyncio.run(main())
