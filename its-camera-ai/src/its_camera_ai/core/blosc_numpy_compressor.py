"""Blosc-based NumPy array compression for high-performance inter-service communication.

This module provides optimized compression/decompression for NumPy arrays using
Blosc compression library, specifically designed for reducing memory usage and 
network bandwidth in ML service communications.

Key Features:
- Ultra-fast compression/decompression with blosc
- Automatic compression level selection based on array characteristics
- Memory-efficient streaming compression for large arrays
- Detailed performance metrics and monitoring
- Thread-safe operations for concurrent access

Performance Targets:
- 60%+ size reduction for typical computer vision arrays
- <10ms compression/decompression overhead
- 30% memory usage reduction during transfers
- 50%+ network bandwidth improvement
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import blosc
import numpy as np

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Compression level settings optimized for different use cases."""

    FASTEST = 1      # Fastest compression, lower ratio
    BALANCED = 5     # Balanced speed/ratio for typical use
    MAXIMUM = 9      # Maximum compression, slower speed
    AUTO = -1        # Auto-select based on array characteristics


class CompressionAlgorithm(Enum):
    """Supported compression algorithms."""

    BLOSCLZ = "blosclz"     # Fast, good for mixed data
    LZ4 = "lz4"             # Very fast decompression
    LZ4HC = "lz4hc"         # Higher compression ratio
    SNAPPY = "snappy"       # Google's fast compressor
    ZLIB = "zlib"           # Good compression ratio
    ZSTD = "zstd"           # Excellent ratio/speed balance


@dataclass
class CompressionMetrics:
    """Metrics for compression operations."""

    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0
    algorithm: str = ""
    level: int = 0
    array_shape: tuple[int, ...] = field(default_factory=tuple)
    array_dtype: str = ""
    threads_used: int = 1
    shuffle_enabled: bool = True

    @property
    def size_reduction_percent(self) -> float:
        """Calculate size reduction percentage."""
        if self.original_size == 0:
            return 0.0
        return (1.0 - self.compression_ratio) * 100.0

    @property
    def compression_speed_mbps(self) -> float:
        """Calculate compression speed in MB/s."""
        if self.compression_time_ms == 0:
            return 0.0
        mb_per_second = (self.original_size / 1024 / 1024) / (self.compression_time_ms / 1000)
        return mb_per_second


@dataclass
class ArrayCharacteristics:
    """Characteristics used for auto-selecting compression parameters."""

    size_bytes: int
    shape: tuple[int, ...]
    dtype: np.dtype
    is_image_like: bool = False
    has_patterns: bool = False
    sparsity_ratio: float = 0.0
    entropy_estimate: float = 0.0


class BloscNumpyCompressor:
    """High-performance blosc-based NumPy array compressor.
    
    Optimized for computer vision and ML workloads with automatic
    parameter selection and detailed performance monitoring.
    """

    def __init__(
        self,
        compression_level: CompressionLevel | int = CompressionLevel.BALANCED,
        algorithm: CompressionAlgorithm = CompressionAlgorithm.ZSTD,
        shuffle: bool = True,
        threads: int | None = None,
        enable_auto_tuning: bool = True,
        cache_size_mb: int = 64
    ):
        """Initialize blosc numpy compressor.
        
        Args:
            compression_level: Compression level (1-9) or auto
            algorithm: Compression algorithm to use
            shuffle: Enable byte shuffling for better compression
            threads: Number of threads (None for auto-detect)
            enable_auto_tuning: Enable automatic parameter tuning
            cache_size_mb: Cache size for repeated compression patterns
        """
        self.compression_level = compression_level
        self.algorithm = algorithm
        self.shuffle = shuffle
        self.enable_auto_tuning = enable_auto_tuning

        # Initialize blosc
        if threads is not None:
            blosc.set_nthreads(threads)
        else:
            # Auto-detect optimal thread count
            import os
            cpu_count = os.cpu_count() or 4
            blosc.set_nthreads(min(cpu_count, 8))  # Cap at 8 for efficiency

        self.threads = blosc.nthreads

        # Performance tracking
        self._compression_stats: dict[str, list[float]] = {
            "compression_times": [],
            "decompression_times": [],
            "compression_ratios": [],
            "throughput_mbps": []
        }
        self._total_compressions = 0
        self._total_decompressions = 0
        self._lock = threading.RLock()

        # Compression cache for repeated patterns
        self._compression_cache: dict[str, bytes] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_size_limit = cache_size_mb * 1024 * 1024

        logger.info(
            f"BloscNumpyCompressor initialized: algorithm={algorithm.value}, "
            f"level={compression_level}, threads={self.threads}, "
            f"shuffle={shuffle}, auto_tuning={enable_auto_tuning}"
        )

    def compress_array(
        self,
        array: np.ndarray,
        force_algorithm: CompressionAlgorithm | None = None,
        force_level: int | None = None
    ) -> bytes:
        """Compress numpy array using blosc with optimal settings.
        
        Args:
            array: NumPy array to compress
            force_algorithm: Override automatic algorithm selection
            force_level: Override automatic level selection
            
        Returns:
            bytes: Compressed array data
        """
        if array is None or array.size == 0:
            return b""

        start_time = time.perf_counter()

        try:
            # Ensure array is contiguous
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)

            # Auto-select parameters if enabled
            algorithm = force_algorithm or self.algorithm
            level = force_level

            if level is None:
                if isinstance(self.compression_level, CompressionLevel):
                    if self.compression_level == CompressionLevel.AUTO:
                        level = self._auto_select_level(array)
                    else:
                        level = self.compression_level.value
                else:
                    level = int(self.compression_level)

            if self.enable_auto_tuning and force_algorithm is None:
                algorithm = self._auto_select_algorithm(array)

            # Check cache first
            cache_key = self._generate_cache_key(array, algorithm, level)
            if cache_key in self._compression_cache:
                self._cache_hits += 1
                return self._compression_cache[cache_key]

            # Perform compression
            compressed_data = blosc.compress(
                array.tobytes(),
                typesize=array.dtype.itemsize,
                clevel=level,
                shuffle=blosc.SHUFFLE if self.shuffle else blosc.NOSHUFFLE,
                cname=algorithm.value
            )

            # Cache result if beneficial
            if len(compressed_data) < self._cache_size_limit // 100:  # Cache if <1% of limit
                self._update_cache(cache_key, compressed_data)

            # Track performance
            compression_time = (time.perf_counter() - start_time) * 1000
            self._record_compression_performance(
                array, compressed_data, compression_time, algorithm.value, level
            )

            self._cache_misses += 1
            return compressed_data

        except Exception as e:
            logger.error(f"Array compression failed: {e}")
            raise

    def decompress_array(
        self,
        compressed_data: bytes,
        dtype: np.dtype | None = None,
        shape: tuple[int, ...] | None = None
    ) -> np.ndarray:
        """Decompress blosc data back to numpy array.
        
        Args:
            compressed_data: Compressed data bytes
            dtype: Expected array dtype (for validation)
            shape: Expected array shape (for validation)
            
        Returns:
            np.ndarray: Decompressed numpy array
        """
        if not compressed_data:
            return np.array([])

        start_time = time.perf_counter()

        try:
            # Decompress data
            decompressed_bytes = blosc.decompress(compressed_data)

            # Convert back to numpy array
            if dtype is None or shape is None:
                # Try to infer from blosc metadata
                info = self.get_compression_info(compressed_data)
                if dtype is None:
                    # Default to uint8 for unknown dtype
                    dtype = np.uint8
                if shape is None:
                    # Calculate shape from size and dtype
                    expected_size = len(decompressed_bytes) // np.dtype(dtype).itemsize
                    shape = (expected_size,)

            # Create array
            array = np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)

            # Track performance
            decompression_time = (time.perf_counter() - start_time) * 1000
            self._record_decompression_performance(decompression_time)

            return array

        except Exception as e:
            logger.error(f"Array decompression failed: {e}")
            raise

    def compress_with_metadata(
        self,
        array: np.ndarray,
        include_shape: bool = True,
        include_dtype: bool = True
    ) -> bytes:
        """Compress array with embedded metadata for self-contained decompression.
        
        Args:
            array: NumPy array to compress
            include_shape: Include shape information
            include_dtype: Include dtype information
            
        Returns:
            bytes: Compressed data with metadata header
        """
        if array is None or array.size == 0:
            return b""

        try:
            # Compress array data
            compressed_data = self.compress_array(array)

            # Create metadata header
            metadata = {
                "shape": array.shape if include_shape else None,
                "dtype": str(array.dtype) if include_dtype else None,
                "compressed_size": len(compressed_data),
                "original_size": array.nbytes
            }

            # Serialize metadata (simple format)
            metadata_str = f"{metadata['shape']};{metadata['dtype']};{metadata['compressed_size']};{metadata['original_size']}"
            metadata_bytes = metadata_str.encode('utf-8')
            metadata_size = len(metadata_bytes)

            # Pack: [metadata_size:4][metadata][compressed_data]
            result = metadata_size.to_bytes(4, 'little') + metadata_bytes + compressed_data

            return result

        except Exception as e:
            logger.error(f"Compression with metadata failed: {e}")
            raise

    def decompress_with_metadata(self, data_with_metadata: bytes) -> np.ndarray:
        """Decompress data with embedded metadata.
        
        Args:
            data_with_metadata: Data with metadata header
            
        Returns:
            np.ndarray: Decompressed array
        """
        if not data_with_metadata or len(data_with_metadata) < 4:
            return np.array([])

        try:
            # Read metadata size
            metadata_size = int.from_bytes(data_with_metadata[:4], 'little')

            # Read metadata
            metadata_bytes = data_with_metadata[4:4+metadata_size]
            metadata_str = metadata_bytes.decode('utf-8')
            parts = metadata_str.split(';')

            # Parse metadata
            shape_str, dtype_str = parts[0], parts[1]
            shape = eval(shape_str) if shape_str != 'None' else None
            dtype = np.dtype(dtype_str) if dtype_str != 'None' else None

            # Extract compressed data
            compressed_data = data_with_metadata[4+metadata_size:]

            # Decompress
            return self.decompress_array(compressed_data, dtype, shape)

        except Exception as e:
            logger.error(f"Decompression with metadata failed: {e}")
            raise

    def get_compression_ratio(self, original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio.
        
        Args:
            original_size: Original data size in bytes
            compressed_size: Compressed data size in bytes
            
        Returns:
            float: Compression ratio (compressed/original)
        """
        if original_size == 0:
            return 0.0
        return compressed_size / original_size

    def get_compression_info(self, compressed_data: bytes) -> dict[str, Any]:
        """Get information about compressed data.
        
        Args:
            compressed_data: Compressed data bytes
            
        Returns:
            Dict[str, Any]: Compression information
        """
        try:
            info_tuple = blosc.get_clib(compressed_data)
            return {
                "compressor": info_tuple[0],
                "version": info_tuple[1],
                "compressed_size": len(compressed_data),
                "uncompressed_size": blosc.get_nbytes(compressed_data),
                "blocksize": blosc.get_blocksize(compressed_data)
            }
        except Exception as e:
            logger.error(f"Failed to get compression info: {e}")
            return {}

    def benchmark_algorithms(
        self,
        test_array: np.ndarray,
        algorithms: list[CompressionAlgorithm] | None = None
    ) -> dict[str, CompressionMetrics]:
        """Benchmark different compression algorithms on test array.
        
        Args:
            test_array: Array to use for benchmarking
            algorithms: List of algorithms to test (None for all)
            
        Returns:
            Dict[str, CompressionMetrics]: Benchmark results
        """
        if algorithms is None:
            algorithms = list(CompressionAlgorithm)

        results = {}
        original_algorithm = self.algorithm

        try:
            for algo in algorithms:
                self.algorithm = algo

                # Compression benchmark
                start_time = time.perf_counter()
                compressed_data = self.compress_array(test_array)
                compression_time = (time.perf_counter() - start_time) * 1000

                # Decompression benchmark
                start_time = time.perf_counter()
                decompressed = self.decompress_array(
                    compressed_data, test_array.dtype, test_array.shape
                )
                decompression_time = (time.perf_counter() - start_time) * 1000

                # Verify correctness
                if not np.array_equal(test_array, decompressed):
                    logger.warning(f"Algorithm {algo.value} produced incorrect result")

                # Create metrics
                metrics = CompressionMetrics(
                    original_size=test_array.nbytes,
                    compressed_size=len(compressed_data),
                    compression_ratio=len(compressed_data) / test_array.nbytes,
                    compression_time_ms=compression_time,
                    decompression_time_ms=decompression_time,
                    algorithm=algo.value,
                    level=self.compression_level.value if hasattr(self.compression_level, 'value') else int(self.compression_level),
                    array_shape=test_array.shape,
                    array_dtype=str(test_array.dtype),
                    threads_used=self.threads,
                    shuffle_enabled=self.shuffle
                )

                results[algo.value] = metrics

                logger.info(
                    f"Benchmark {algo.value}: {metrics.size_reduction_percent:.1f}% reduction, "
                    f"{metrics.compression_speed_mbps:.1f} MB/s"
                )

        finally:
            self.algorithm = original_algorithm

        return results

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        with self._lock:
            stats = self._compression_stats

            if not stats["compression_times"]:
                return {"status": "no_data"}

            return {
                "total_compressions": self._total_compressions,
                "total_decompressions": self._total_decompressions,
                "avg_compression_time_ms": sum(stats["compression_times"]) / len(stats["compression_times"]),
                "avg_decompression_time_ms": sum(stats["decompression_times"]) / len(stats["decompression_times"]) if stats["decompression_times"] else 0,
                "avg_compression_ratio": sum(stats["compression_ratios"]) / len(stats["compression_ratios"]),
                "avg_throughput_mbps": sum(stats["throughput_mbps"]) / len(stats["throughput_mbps"]) if stats["throughput_mbps"] else 0,
                "cache_hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0,
                "cache_size": len(self._compression_cache),
                "threads_used": self.threads,
                "algorithm": self.algorithm.value,
                "compression_level": self.compression_level.value if hasattr(self.compression_level, 'value') else int(self.compression_level),
                "shuffle_enabled": self.shuffle,
                "auto_tuning_enabled": self.enable_auto_tuning
            }

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        with self._lock:
            self._compression_stats = {
                "compression_times": [],
                "decompression_times": [],
                "compression_ratios": [],
                "throughput_mbps": []
            }
            self._total_compressions = 0
            self._total_decompressions = 0
            self._cache_hits = 0
            self._cache_misses = 0

    def clear_cache(self) -> None:
        """Clear compression cache."""
        with self._lock:
            self._compression_cache.clear()
            logger.info("Compression cache cleared")

    @contextmanager
    def temporary_settings(
        self,
        algorithm: CompressionAlgorithm | None = None,
        level: int | None = None,
        shuffle: bool | None = None
    ) -> Iterator[None]:
        """Context manager for temporary compression settings.
        
        Args:
            algorithm: Temporary algorithm
            level: Temporary compression level  
            shuffle: Temporary shuffle setting
        """
        # Store original settings
        orig_algorithm = self.algorithm
        orig_level = self.compression_level
        orig_shuffle = self.shuffle

        try:
            # Apply temporary settings
            if algorithm is not None:
                self.algorithm = algorithm
            if level is not None:
                self.compression_level = level
            if shuffle is not None:
                self.shuffle = shuffle

            yield

        finally:
            # Restore original settings
            self.algorithm = orig_algorithm
            self.compression_level = orig_level
            self.shuffle = orig_shuffle

    def _auto_select_level(self, array: np.ndarray) -> int:
        """Auto-select compression level based on array characteristics."""
        characteristics = self._analyze_array_characteristics(array)

        # Heuristics for level selection
        if characteristics.size_bytes < 1024 * 1024:  # < 1MB
            return 3  # Fast for small arrays
        elif characteristics.size_bytes > 100 * 1024 * 1024:  # > 100MB
            return 1  # Fastest for very large arrays
        elif characteristics.has_patterns or characteristics.is_image_like:
            return 7  # Higher compression for structured data
        else:
            return 5  # Balanced default

    def _auto_select_algorithm(self, array: np.ndarray) -> CompressionAlgorithm:
        """Auto-select compression algorithm based on array characteristics."""
        characteristics = self._analyze_array_characteristics(array)

        # Algorithm selection heuristics
        if characteristics.size_bytes > 50 * 1024 * 1024:  # > 50MB
            return CompressionAlgorithm.LZ4  # Fast for large arrays
        elif characteristics.is_image_like:
            return CompressionAlgorithm.ZSTD  # Good for image data
        elif characteristics.has_patterns:
            return CompressionAlgorithm.ZLIB  # Good compression for patterns
        else:
            return CompressionAlgorithm.ZSTD  # Best overall balance

    def _analyze_array_characteristics(self, array: np.ndarray) -> ArrayCharacteristics:
        """Analyze array to determine optimal compression parameters."""
        characteristics = ArrayCharacteristics(
            size_bytes=array.nbytes,
            shape=array.shape,
            dtype=array.dtype
        )

        # Detect image-like arrays (2D/3D with typical image dimensions)
        if len(array.shape) in [2, 3] and min(array.shape[:2]) > 32:
            characteristics.is_image_like = True

        # Simple pattern detection (check for repeated values)
        if array.size > 1000:  # Only for reasonably sized arrays
            sample = array.flat[:1000]  # Sample first 1000 elements
            unique_ratio = len(np.unique(sample)) / len(sample)
            characteristics.has_patterns = unique_ratio < 0.7

        return characteristics

    def _generate_cache_key(
        self,
        array: np.ndarray,
        algorithm: CompressionAlgorithm,
        level: int
    ) -> str:
        """Generate cache key for array compression."""
        # Use array hash, shape, dtype, and compression settings
        array_hash = hash(array.tobytes())
        return f"{array_hash}_{array.shape}_{array.dtype}_{algorithm.value}_{level}_{self.shuffle}"

    def _update_cache(self, cache_key: str, compressed_data: bytes) -> None:
        """Update compression cache with size limits."""
        with self._lock:
            # Check if we need to clear old entries
            current_size = sum(len(data) for data in self._compression_cache.values())
            if current_size + len(compressed_data) > self._cache_size_limit:
                # Clear cache to make room
                self._compression_cache.clear()

            self._compression_cache[cache_key] = compressed_data

    def _record_compression_performance(
        self,
        original_array: np.ndarray,
        compressed_data: bytes,
        compression_time_ms: float,
        algorithm: str,
        level: int
    ) -> None:
        """Record compression performance metrics."""
        with self._lock:
            self._total_compressions += 1
            self._compression_stats["compression_times"].append(compression_time_ms)

            ratio = len(compressed_data) / original_array.nbytes
            self._compression_stats["compression_ratios"].append(ratio)

            # Calculate throughput
            if compression_time_ms > 0:
                throughput = (original_array.nbytes / 1024 / 1024) / (compression_time_ms / 1000)
                self._compression_stats["throughput_mbps"].append(throughput)

            # Keep only recent metrics (last 1000)
            for key in self._compression_stats:
                if len(self._compression_stats[key]) > 1000:
                    self._compression_stats[key] = self._compression_stats[key][-1000:]

    def _record_decompression_performance(self, decompression_time_ms: float) -> None:
        """Record decompression performance metrics."""
        with self._lock:
            self._total_decompressions += 1
            self._compression_stats["decompression_times"].append(decompression_time_ms)

            # Keep only recent metrics (last 1000)
            if len(self._compression_stats["decompression_times"]) > 1000:
                self._compression_stats["decompression_times"] = self._compression_stats["decompression_times"][-1000:]


# Global compressor instance for shared use across services
_global_compressor: BloscNumpyCompressor | None = None
_compressor_lock = threading.Lock()


def get_global_compressor() -> BloscNumpyCompressor:
    """Get global blosc numpy compressor instance.
    
    Returns:
        BloscNumpyCompressor: Global compressor instance
    """
    global _global_compressor

    if _global_compressor is None:
        with _compressor_lock:
            if _global_compressor is None:
                _global_compressor = BloscNumpyCompressor(
                    compression_level=CompressionLevel.BALANCED,
                    algorithm=CompressionAlgorithm.ZSTD,
                    enable_auto_tuning=True,
                    cache_size_mb=128  # Larger cache for global use
                )
                logger.info("Global BloscNumpyCompressor initialized")

    return _global_compressor


def compress_numpy_array(array: np.ndarray, **kwargs: Any) -> bytes:
    """Convenience function to compress numpy array with global compressor.
    
    Args:
        array: NumPy array to compress
        **kwargs: Additional arguments for compression
        
    Returns:
        bytes: Compressed array data
    """
    return get_global_compressor().compress_array(array, **kwargs)


def decompress_numpy_array(
    compressed_data: bytes,
    dtype: np.dtype,
    shape: tuple[int, ...]
) -> np.ndarray:
    """Convenience function to decompress array with global compressor.
    
    Args:
        compressed_data: Compressed data bytes
        dtype: Array dtype
        shape: Array shape
        
    Returns:
        np.ndarray: Decompressed numpy array
    """
    return get_global_compressor().decompress_array(compressed_data, dtype, shape)
