"""
Advanced GPU memory pooling and multi-stream architecture for YOLO11 inference.

This module implements production-grade memory management with zero-copy operations,
CUDA stream orchestration, and intelligent memory pooling for 100+ concurrent
camera streams with sub-100ms latency requirements.
"""

import logging
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.cuda

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    pynvml = None
    NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryPoolType(Enum):
    """Types of memory pools for different use cases."""

    INPUT_UINT8 = "input_uint8"        # Raw input frames (uint8)
    INPUT_FLOAT16 = "input_float16"    # Normalized input (float16)
    OUTPUT_BUFFER = "output_buffer"    # Model output buffers
    WORKSPACE = "workspace"            # Temporary workspace
    PINNED_HOST = "pinned_host"        # Pinned host memory


@dataclass
class TensorDescriptor:
    """Descriptor for tensor allocation and pooling."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: int | str
    memory_format: torch.memory_format = torch.contiguous_format
    requires_grad: bool = False

    @property
    def size_bytes(self) -> int:
        """Calculate tensor size in bytes."""
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        return int(np.prod(self.shape)) * element_size

    def __hash__(self) -> int:
        return hash((self.shape, self.dtype, str(self.device), self.memory_format))

    def __eq__(self, other) -> bool:
        if not isinstance(other, TensorDescriptor):
            return False
        return (
            self.shape == other.shape and
            self.dtype == other.dtype and
            str(self.device) == str(other.device) and
            self.memory_format == other.memory_format
        )


class GPUMemoryPool:
    """
    High-performance GPU memory pool with intelligent allocation strategies.

    Features:
    - Size-based allocation buckets for efficient memory reuse
    - Lazy allocation with pre-warming for hot paths
    - Memory fragmentation prevention
    - CUDA memory stream awareness
    - Automatic cleanup and garbage collection
    """

    def __init__(
        self,
        device_id: int,
        pool_type: MemoryPoolType,
        initial_size_mb: int = 512,
        max_size_mb: int = 4096,
        enable_preallocation: bool = True
    ):
        self.device_id = device_id
        self.pool_type = pool_type
        self.initial_size_mb = initial_size_mb
        self.max_size_mb = max_size_mb
        self.device = torch.device(f"cuda:{device_id}")

        # Memory pool storage - organized by tensor descriptor
        self.pools: dict[TensorDescriptor, deque] = defaultdict(lambda: deque(maxlen=16))
        self.allocated_tensors: set[int] = set()  # Track allocated tensor IDs
        self.pool_lock = threading.RLock()

        # Statistics
        self.stats = {
            'total_allocations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'current_allocated_mb': 0,
            'peak_allocated_mb': 0,
            'fragmentation_ratio': 0.0,
        }

        # Pre-allocation for common sizes
        if enable_preallocation:
            self._preallocate_common_tensors()

        logger.info(f"GPU memory pool initialized on device {device_id} for {pool_type.value}")

    def _preallocate_common_tensors(self) -> None:
        """Pre-allocate tensors for common inference scenarios."""
        with torch.cuda.device(self.device_id):
            common_descriptors = []

            if self.pool_type == MemoryPoolType.INPUT_UINT8:
                # Common input sizes for traffic cameras
                sizes = [
                    (1, 3, 640, 640),   # Single frame standard
                    (4, 3, 640, 640),   # Small batch
                    (8, 3, 640, 640),   # Standard batch
                    (16, 3, 640, 640),  # Large batch
                    (32, 3, 640, 640),  # Max batch
                    (1, 3, 1280, 1280), # High-res single
                    (4, 3, 1280, 1280), # High-res batch
                ]

                for shape in sizes:
                    desc = TensorDescriptor(
                        shape=shape,
                        dtype=torch.uint8,
                        device=self.device_id,
                        memory_format=torch.channels_last
                    )
                    common_descriptors.append(desc)

            elif self.pool_type == MemoryPoolType.INPUT_FLOAT16:
                # FP16 inference tensors
                sizes = [
                    (1, 3, 640, 640),
                    (4, 3, 640, 640),
                    (8, 3, 640, 640),
                    (16, 3, 640, 640),
                    (32, 3, 640, 640),
                ]

                for shape in sizes:
                    desc = TensorDescriptor(
                        shape=shape,
                        dtype=torch.half,
                        device=self.device_id,
                        memory_format=torch.channels_last
                    )
                    common_descriptors.append(desc)

            elif self.pool_type == MemoryPoolType.OUTPUT_BUFFER:
                # YOLO11 output tensor sizes
                # Assuming 3 detection heads: 80x80, 40x40, 20x20
                output_sizes = [
                    (1, 144, 80, 80),   # Head 1: (4+1+num_classes) * anchors
                    (1, 144, 40, 40),   # Head 2
                    (1, 144, 20, 20),   # Head 3
                    (8, 144, 80, 80),   # Batch versions
                    (8, 144, 40, 40),
                    (8, 144, 20, 20),
                    (16, 144, 80, 80),
                    (16, 144, 40, 40),
                    (16, 144, 20, 20),
                ]

                for shape in output_sizes:
                    desc = TensorDescriptor(
                        shape=shape,
                        dtype=torch.half,
                        device=self.device_id
                    )
                    common_descriptors.append(desc)

            # Pre-allocate tensors
            for desc in common_descriptors:
                for _ in range(4):  # 4 tensors per descriptor
                    try:
                        tensor = self._create_tensor(desc)
                        self.pools[desc].append(tensor)
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"OOM during preallocation for {desc.shape}")
                        break

            logger.info(f"Pre-allocated {len(common_descriptors)} tensor types")

    def get_tensor(self, descriptor: TensorDescriptor) -> torch.Tensor:
        """
        Get tensor from pool or allocate new one.

        Args:
            descriptor: Tensor descriptor specifying requirements

        Returns:
            Allocated tensor ready for use
        """
        with self.pool_lock:
            # Try to get from pool first
            if descriptor in self.pools and self.pools[descriptor]:
                tensor = self.pools[descriptor].popleft()
                self.stats['cache_hits'] += 1

                # Clear tensor data for reuse
                tensor.zero_()
                return tensor

            # Cache miss - allocate new tensor
            self.stats['cache_misses'] += 1
            tensor = self._create_tensor(descriptor)
            self.allocated_tensors.add(id(tensor))
            self.stats['total_allocations'] += 1

            return tensor

    def return_tensor(self, tensor: torch.Tensor) -> None:
        """
        Return tensor to pool for reuse.

        Args:
            tensor: Tensor to return to pool
        """
        if tensor.device.index != self.device_id:
            logger.warning(f"Tensor on wrong device {tensor.device.index}, expected {self.device_id}")
            return

        # Create descriptor for this tensor
        descriptor = TensorDescriptor(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=self.device_id,
            memory_format=tensor.memory_format if hasattr(tensor, 'memory_format') else torch.contiguous_format
        )

        with self.pool_lock:
            # Only return to pool if we have space
            if len(self.pools[descriptor]) < 16:  # Max 16 tensors per descriptor
                self.pools[descriptor].append(tensor)
            # Otherwise let it be garbage collected

    def _create_tensor(self, descriptor: TensorDescriptor) -> torch.Tensor:
        """Create new tensor according to descriptor."""
        with torch.cuda.device(self.device_id):
            if descriptor.memory_format == torch.channels_last:
                tensor = torch.empty(
                    descriptor.shape,
                    dtype=descriptor.dtype,
                    device=self.device,
                    memory_format=torch.channels_last,
                    requires_grad=descriptor.requires_grad
                )
            else:
                tensor = torch.empty(
                    descriptor.shape,
                    dtype=descriptor.dtype,
                    device=self.device,
                    requires_grad=descriptor.requires_grad
                )

            return tensor

    def prewarm_pool(self, descriptors: list[TensorDescriptor], count_per_desc: int = 4) -> None:
        """Pre-warm pool with specified tensor descriptors."""
        with torch.cuda.device(self.device_id):
            for desc in descriptors:
                for _ in range(count_per_desc):
                    try:
                        tensor = self._create_tensor(desc)
                        with self.pool_lock:
                            self.pools[desc].append(tensor)
                    except torch.cuda.OutOfMemoryError:
                        logger.warning(f"OOM during pool prewarming for {desc.shape}")
                        break

    def cleanup_pool(self, aggressive: bool = False) -> None:
        """Clean up pool memory."""
        with self.pool_lock:
            if aggressive:
                # Clear all pools
                for pool in self.pools.values():
                    pool.clear()
                self.pools.clear()
            else:
                # Keep only small number of tensors per descriptor
                for _desc, pool in self.pools.items():
                    while len(pool) > 4:
                        pool.popleft()

        # Force garbage collection
        torch.cuda.empty_cache()

        logger.info(f"Memory pool cleanup completed (aggressive={aggressive})")

    def get_memory_stats(self) -> dict[str, Any]:
        """Get detailed memory pool statistics."""
        with torch.cuda.device(self.device_id):
            allocated_mb = torch.cuda.memory_allocated(self.device_id) / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved(self.device_id) / (1024 ** 2)

        with self.pool_lock:
            pool_sizes = {str(desc.shape): len(pool) for desc, pool in self.pools.items()}
            total_pooled_tensors = sum(len(pool) for pool in self.pools.values())

        self.stats.update({
            'current_allocated_mb': allocated_mb,
            'reserved_mb': reserved_mb,
            'total_pooled_tensors': total_pooled_tensors,
            'active_descriptors': len(self.pools),
            'pool_sizes': pool_sizes,
        })

        return self.stats.copy()


class CUDAStreamManager:
    """
    Advanced CUDA stream manager for parallel processing across multiple streams.

    Manages stream allocation, synchronization, and work distribution for
    optimal GPU utilization with minimal latency.
    """

    def __init__(
        self,
        device_ids: list[int],
        streams_per_device: int = 8,
        enable_priority_streams: bool = True
    ):
        self.device_ids = device_ids
        self.streams_per_device = streams_per_device
        self.enable_priority_streams = enable_priority_streams

        # Stream allocation
        self.streams: dict[int, list[torch.cuda.Stream]] = {}
        self.priority_streams: dict[int, torch.cuda.Stream] = {}
        self.stream_locks: dict[int, list[threading.Lock]] = {}
        self.stream_usage: dict[int, list[int]] = {}  # Usage counters per stream

        # Stream assignment state
        self.current_stream_idx: dict[int, int] = {}
        self.stream_load_balancer = {}

        self._initialize_streams()

        logger.info(f"CUDA stream manager initialized for devices {device_ids}")

    def _initialize_streams(self) -> None:
        """Initialize CUDA streams for all devices."""
        for device_id in self.device_ids:
            with torch.cuda.device(device_id):
                # Create regular streams
                device_streams = []
                device_locks = []
                device_usage = []

                for _i in range(self.streams_per_device):
                    stream = torch.cuda.Stream(device=device_id, priority=0)
                    device_streams.append(stream)
                    device_locks.append(threading.Lock())
                    device_usage.append(0)

                self.streams[device_id] = device_streams
                self.stream_locks[device_id] = device_locks
                self.stream_usage[device_id] = device_usage
                self.current_stream_idx[device_id] = 0

                # Create priority stream if enabled
                if self.enable_priority_streams:
                    priority_stream = torch.cuda.Stream(device=device_id, priority=-1)  # Higher priority
                    self.priority_streams[device_id] = priority_stream

                logger.debug(f"Created {self.streams_per_device} streams for GPU {device_id}")

    @contextmanager
    def get_stream(self, device_id: int, high_priority: bool = False):
        """
        Get CUDA stream with automatic synchronization management.

        Args:
            device_id: Target GPU device ID
            high_priority: Use high-priority stream if available

        Yields:
            CUDA stream context
        """
        if device_id not in self.streams:
            raise ValueError(f"Device {device_id} not managed by this stream manager")

        if high_priority and device_id in self.priority_streams:
            # Use dedicated priority stream
            stream = self.priority_streams[device_id]

            try:
                with torch.cuda.stream(stream):
                    yield stream
            finally:
                pass  # Priority streams don't need special cleanup

        else:
            # Use load-balanced regular stream
            stream_idx = self._get_next_stream_index(device_id)
            stream = self.streams[device_id][stream_idx]
            lock = self.stream_locks[device_id][stream_idx]

            try:
                with lock:
                    self.stream_usage[device_id][stream_idx] += 1
                    with torch.cuda.stream(stream):
                        yield stream
            finally:
                with lock:
                    self.stream_usage[device_id][stream_idx] -= 1

    def _get_next_stream_index(self, device_id: int) -> int:
        """Get next stream index using load balancing."""
        # Simple round-robin for now, could be enhanced with load-aware selection
        current_idx = self.current_stream_idx[device_id]
        next_idx = (current_idx + 1) % self.streams_per_device
        self.current_stream_idx[device_id] = next_idx
        return current_idx

    def synchronize_device(self, device_id: int) -> None:
        """Synchronize all streams on specified device."""
        if device_id in self.streams:
            for stream in self.streams[device_id]:
                stream.synchronize()

        if device_id in self.priority_streams:
            self.priority_streams[device_id].synchronize()

    def synchronize_all(self) -> None:
        """Synchronize all streams on all devices."""
        for device_id in self.device_ids:
            self.synchronize_device(device_id)

    def get_stream_utilization(self) -> dict[int, list[float]]:
        """Get utilization metrics for all streams."""
        utilization = {}

        for device_id in self.device_ids:
            device_util = []
            for stream_idx in range(self.streams_per_device):
                usage = self.stream_usage[device_id][stream_idx]
                # Normalize usage (this is a simplified metric)
                util_pct = min(1.0, usage / 10.0)  # Assuming 10+ concurrent uses = 100%
                device_util.append(util_pct)

            utilization[device_id] = device_util

        return utilization

    def cleanup(self) -> None:
        """Clean up all streams and synchronize."""
        self.synchronize_all()

        # Streams will be cleaned up automatically by PyTorch
        self.streams.clear()
        self.priority_streams.clear()
        self.stream_locks.clear()
        self.stream_usage.clear()


class MultiGPUMemoryManager:
    """
    Comprehensive multi-GPU memory management system.

    Coordinates memory pools and streams across multiple GPUs for optimal
    performance in multi-camera inference scenarios.
    """

    def __init__(
        self,
        device_ids: list[int],
        memory_fraction_per_gpu: float = 0.8,
        enable_peer_access: bool = True
    ):
        self.device_ids = device_ids
        self.memory_fraction_per_gpu = memory_fraction_per_gpu

        # Memory pools per device and type
        self.memory_pools: dict[int, dict[MemoryPoolType, GPUMemoryPool]] = {}

        # Stream management
        self.stream_manager = CUDAStreamManager(device_ids)

        # Device selection and load balancing
        self.device_selector = DeviceLoadBalancer(device_ids)

        self._initialize_memory_management()

        if enable_peer_access:
            self._enable_peer_to_peer_access()

        logger.info(f"Multi-GPU memory manager initialized for devices {device_ids}")

    def _initialize_memory_management(self) -> None:
        """Initialize memory pools for all devices."""
        for device_id in self.device_ids:
            # Set memory fraction for this device
            torch.cuda.set_per_process_memory_fraction(
                self.memory_fraction_per_gpu,
                device_id
            )

            # Create memory pools for different types
            device_pools = {}

            for pool_type in MemoryPoolType:
                if pool_type != MemoryPoolType.PINNED_HOST:  # Host memory handled separately
                    pool = GPUMemoryPool(
                        device_id=device_id,
                        pool_type=pool_type,
                        initial_size_mb=256,
                        max_size_mb=2048
                    )
                    device_pools[pool_type] = pool

            self.memory_pools[device_id] = device_pools

    def _enable_peer_to_peer_access(self) -> None:
        """Enable P2P memory access between GPUs if supported."""
        if len(self.device_ids) < 2:
            return

        try:
            for i, dev1 in enumerate(self.device_ids):
                for dev2 in self.device_ids[i+1:]:
                    if torch.cuda.can_device_access_peer(dev1, dev2):
                        torch.cuda.set_device(dev1)
                        torch.cuda.device_enable_peer_access(dev2)
                        torch.cuda.set_device(dev2)
                        torch.cuda.device_enable_peer_access(dev1)

                        logger.info(f"P2P access enabled between GPU {dev1} and GPU {dev2}")

        except Exception as e:
            logger.warning(f"Failed to enable P2P access: {e}")

    def get_tensor(
        self,
        descriptor: TensorDescriptor,
        pool_type: MemoryPoolType,
        preferred_device: int | None = None
    ) -> torch.Tensor:
        """
        Get tensor from appropriate memory pool.

        Args:
            descriptor: Tensor requirements
            pool_type: Type of memory pool to use
            preferred_device: Preferred GPU device (None for auto-selection)

        Returns:
            Allocated tensor
        """
        # Select device
        device_id = preferred_device or self.device_selector.select_device()

        # Update descriptor to match selected device
        descriptor.device = device_id

        # Get tensor from appropriate pool
        pool = self.memory_pools[device_id][pool_type]
        return pool.get_tensor(descriptor)

    def return_tensor(
        self,
        tensor: torch.Tensor,
        pool_type: MemoryPoolType
    ) -> None:
        """Return tensor to appropriate memory pool."""
        device_id = tensor.device.index

        if device_id in self.memory_pools and pool_type in self.memory_pools[device_id]:
            pool = self.memory_pools[device_id][pool_type]
            pool.return_tensor(tensor)

    @contextmanager
    def get_inference_context(
        self,
        device_id: int | None = None,
        high_priority: bool = False
    ):
        """
        Get complete inference context with memory and stream management.

        Args:
            device_id: Target device (None for auto-selection)
            high_priority: Use high-priority resources

        Yields:
            Tuple of (device_id, stream, memory_pools)
        """
        selected_device = device_id or self.device_selector.select_device()

        with self.stream_manager.get_stream(selected_device, high_priority) as stream:
            device_pools = self.memory_pools[selected_device]
            yield selected_device, stream, device_pools

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get comprehensive memory and performance statistics."""
        stats = {
            'devices': {},
            'stream_utilization': self.stream_manager.get_stream_utilization(),
            'device_load_balance': self.device_selector.get_load_balance_stats(),
        }

        # Per-device memory stats
        for device_id in self.device_ids:
            device_stats = {}

            # Memory pool stats for each type
            for pool_type, pool in self.memory_pools[device_id].items():
                device_stats[pool_type.value] = pool.get_memory_stats()

            # Device memory info
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    device_stats['total_memory_mb'] = mem_info.total // (1024 ** 2)
                    device_stats['free_memory_mb'] = mem_info.free // (1024 ** 2)
                    device_stats['used_memory_mb'] = mem_info.used // (1024 ** 2)
                except:
                    pass

            stats['devices'][device_id] = device_stats

        return stats

    def cleanup_all_pools(self, aggressive: bool = False) -> None:
        """Clean up all memory pools across all devices."""
        for device_pools in self.memory_pools.values():
            for pool in device_pools.values():
                pool.cleanup_pool(aggressive=aggressive)

        # Clean up streams
        self.stream_manager.cleanup()

        logger.info("All memory pools cleaned up")


class DeviceLoadBalancer:
    """Load balancer for distributing work across multiple GPUs."""

    def __init__(self, device_ids: list[int]):
        self.device_ids = device_ids
        self.device_loads: dict[int, int] = dict.fromkeys(device_ids, 0)
        self.device_capabilities: dict[int, float] = {}
        self.load_lock = threading.Lock()

        self._assess_device_capabilities()

    def _assess_device_capabilities(self) -> None:
        """Assess relative capabilities of each GPU."""
        for device_id in self.device_ids:
            # Get device properties
            props = torch.cuda.get_device_properties(device_id)

            # Simple capability score based on compute capability and memory
            compute_score = props.major * 10 + props.minor
            memory_score = props.total_memory / (1024 ** 3)  # GB

            capability = compute_score * memory_score
            self.device_capabilities[device_id] = capability

            logger.debug(f"GPU {device_id}: capability score {capability:.1f}")

    def select_device(self) -> int:
        """Select optimal device based on current load and capabilities."""
        with self.load_lock:
            # Calculate weighted load (load / capability)
            weighted_loads = {}
            for device_id in self.device_ids:
                load = self.device_loads[device_id]
                capability = self.device_capabilities[device_id]
                weighted_loads[device_id] = load / capability

            # Select device with lowest weighted load
            selected_device = min(weighted_loads, key=weighted_loads.get)

            # Increment load for selected device
            self.device_loads[selected_device] += 1

            return selected_device

    def release_device(self, device_id: int) -> None:
        """Release device after work completion."""
        with self.load_lock:
            if self.device_loads[device_id] > 0:
                self.device_loads[device_id] -= 1

    def get_load_balance_stats(self) -> dict[str, Any]:
        """Get load balancing statistics."""
        with self.load_lock:
            total_load = sum(self.device_loads.values())

            return {
                'device_loads': self.device_loads.copy(),
                'device_capabilities': self.device_capabilities.copy(),
                'total_active_load': total_load,
                'load_distribution': {
                    dev_id: (load / max(1, total_load))
                    for dev_id, load in self.device_loads.items()
                }
            }


# Utility functions for easy integration
def create_memory_manager(
    device_ids: list[int],
    memory_fraction: float = 0.8
) -> MultiGPUMemoryManager:
    """Create multi-GPU memory manager with optimal settings."""
    return MultiGPUMemoryManager(
        device_ids=device_ids,
        memory_fraction_per_gpu=memory_fraction,
        enable_peer_access=True
    )


def get_optimal_tensor_descriptor(
    batch_size: int,
    channels: int = 3,
    height: int = 640,
    width: int = 640,
    dtype: torch.dtype = torch.half,
    device_id: int = 0
) -> TensorDescriptor:
    """Create optimal tensor descriptor for inference."""
    return TensorDescriptor(
        shape=(batch_size, channels, height, width),
        dtype=dtype,
        device=device_id,
        memory_format=torch.channels_last,  # Optimal for CNN models
        requires_grad=False
    )
