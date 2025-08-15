"""Tests for Blosc NumPy compression integration with ML pipeline.

This module tests the high-performance blosc compression system
for numpy arrays in ML service communications, focusing on:

- Compression/decompression performance
- Memory usage optimization
- Cross-service communication efficiency
- Integration with ProcessedFrameSerializer
- ML model inference data handling
"""

import time
from typing import Any, Dict

import numpy as np
import pytest

from its_camera_ai.core.blosc_numpy_compressor import (
    BloscNumpyCompressor,
    CompressionAlgorithm,
    CompressionLevel,
    compress_numpy_array,
    decompress_numpy_array,
    get_global_compressor,
)
from its_camera_ai.flow.grpc_serialization import ProcessedFrameSerializer


@pytest.fixture
def sample_vision_arrays():
    """Create sample arrays typical of computer vision workloads."""
    return {
        "hd_frame": np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8),
        "feature_map": np.random.randn(512, 512, 64).astype(np.float32),
        "detection_masks": np.random.randint(0, 2, (10, 224, 224), dtype=np.bool_),
        "bbox_coordinates": np.random.randn(100, 4).astype(np.float32),
        "confidence_scores": np.random.rand(100).astype(np.float32),
        "small_thumbnail": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
    }


@pytest.fixture
def compressor():
    """Create test compressor instance."""
    return BloscNumpyCompressor(
        compression_level=CompressionLevel.BALANCED,
        algorithm=CompressionAlgorithm.ZSTD,
        enable_auto_tuning=True,
        cache_size_mb=32
    )


class TestBloscNumpyCompressor:
    """Test core blosc compression functionality."""
    
    def test_basic_compression_decompression(self, compressor, sample_vision_arrays):
        """Test basic compression and decompression cycle."""
        for array_name, array in sample_vision_arrays.items():
            # Compress
            compressed = compressor.compress_array(array)
            assert len(compressed) > 0
            
            # Decompress
            decompressed = compressor.decompress_array(
                compressed, array.dtype, array.shape
            )
            
            # Verify correctness
            np.testing.assert_array_equal(array, decompressed)
            
            # Check compression efficiency
            compression_ratio = len(compressed) / array.nbytes
            assert compression_ratio < 1.0, f"{array_name} should be compressed"
    
    def test_compression_with_metadata(self, compressor, sample_vision_arrays):
        """Test compression with embedded metadata."""
        hd_frame = sample_vision_arrays["hd_frame"]
        
        # Compress with metadata
        compressed_with_meta = compressor.compress_with_metadata(hd_frame)
        assert len(compressed_with_meta) > len(compressor.compress_array(hd_frame))
        
        # Decompress with metadata
        decompressed = compressor.decompress_with_metadata(compressed_with_meta)
        
        # Verify
        np.testing.assert_array_equal(hd_frame, decompressed)
        assert decompressed.shape == hd_frame.shape
        assert decompressed.dtype == hd_frame.dtype
    
    def test_auto_algorithm_selection(self, sample_vision_arrays):
        """Test automatic algorithm selection based on data characteristics."""
        auto_compressor = BloscNumpyCompressor(
            compression_level=CompressionLevel.AUTO,
            algorithm=CompressionAlgorithm.ZSTD,  # Default fallback
            enable_auto_tuning=True
        )
        
        results = {}
        for array_name, array in sample_vision_arrays.items():
            compressed = auto_compressor.compress_array(array)
            compression_ratio = len(compressed) / array.nbytes
            results[array_name] = compression_ratio
        
        # Different array types should achieve different compression ratios
        assert results["hd_frame"] < 0.8  # Images compress well
        assert results["detection_masks"] < 0.5  # Boolean masks compress very well
    
    def test_performance_benchmarking(self, compressor, sample_vision_arrays):
        """Test performance benchmarking across algorithms."""
        test_array = sample_vision_arrays["hd_frame"]
        
        # Run benchmark
        benchmark_results = compressor.benchmark_algorithms(
            test_array, 
            [CompressionAlgorithm.ZSTD, CompressionAlgorithm.LZ4, CompressionAlgorithm.ZLIB]
        )
        
        assert len(benchmark_results) == 3
        
        for algo_name, metrics in benchmark_results.items():
            assert metrics.original_size == test_array.nbytes
            assert metrics.compressed_size > 0
            assert metrics.compression_ratio < 1.0
            assert metrics.compression_time_ms > 0
            assert metrics.compression_speed_mbps > 0
    
    @pytest.mark.performance
    def test_compression_performance_targets(self, compressor, sample_vision_arrays):
        """Test performance targets for compression."""
        hd_frame = sample_vision_arrays["hd_frame"]
        
        # Performance test
        start_time = time.perf_counter()
        compressed = compressor.compress_array(hd_frame)
        compression_time = (time.perf_counter() - start_time) * 1000
        
        start_time = time.perf_counter()
        decompressed = compressor.decompress_array(
            compressed, hd_frame.dtype, hd_frame.shape
        )
        decompression_time = (time.perf_counter() - start_time) * 1000
        
        # Performance targets
        assert compression_time < 50, f"Compression too slow: {compression_time:.2f}ms"
        assert decompression_time < 30, f"Decompression too slow: {decompression_time:.2f}ms"
        
        # Size reduction target
        compression_ratio = len(compressed) / hd_frame.nbytes
        size_reduction = (1 - compression_ratio) * 100
        assert size_reduction > 60, f"Size reduction too low: {size_reduction:.1f}%"
    
    def test_global_compressor_thread_safety(self, sample_vision_arrays):
        """Test thread safety of global compressor."""
        import threading
        import time
        
        test_array = sample_vision_arrays["feature_map"]
        results = []
        errors = []
        
        def compress_worker():
            try:
                compressor = get_global_compressor()
                for _ in range(10):
                    compressed = compressor.compress_array(test_array)
                    decompressed = compressor.decompress_array(
                        compressed, test_array.dtype, test_array.shape
                    )
                    results.append(len(compressed))
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(str(e))
        
        # Run multiple threads
        threads = [threading.Thread(target=compress_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Check results
        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 40  # 4 threads * 10 compressions each
        assert all(size > 0 for size in results)
    
    def test_memory_efficiency(self, compressor):
        """Test memory efficiency during compression."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and compress large arrays
        large_arrays = [
            np.random.randn(1000, 1000, 3).astype(np.float32) for _ in range(10)
        ]
        
        compressed_arrays = []
        for array in large_arrays:
            compressed = compressor.compress_array(array)
            compressed_arrays.append(compressed)
            # Clear original to test memory usage of compressed data
            del array
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # Decompress to verify
        for i, compressed in enumerate(compressed_arrays):
            decompressed = compressor.decompress_array(
                compressed, np.float32, (1000, 1000, 3)
            )
            assert decompressed.shape == (1000, 1000, 3)
        
        # Memory increase should be reasonable
        expected_uncompressed_size = 10 * 1000 * 1000 * 3 * 4  # 10 arrays * size * bytes
        memory_efficiency = memory_increase / expected_uncompressed_size
        assert memory_efficiency < 0.5, f"Memory efficiency too low: {memory_efficiency:.3f}"


class TestProcessedFrameSerializerIntegration:
    """Test integration with ProcessedFrameSerializer."""
    
    def test_blosc_enabled_serializer(self, sample_vision_arrays):
        """Test ProcessedFrameSerializer with blosc compression enabled."""
        from its_camera_ai.flow.streaming_processor import ProcessedFrame
        
        serializer = ProcessedFrameSerializer(
            enable_blosc_compression=True,
            blosc_algorithm=CompressionAlgorithm.ZSTD,
            blosc_level=CompressionLevel.BALANCED
        )
        
        # Create test frame
        frame = ProcessedFrame(
            frame_id="test_blosc_001",
            camera_id="camera_001", 
            timestamp=time.time(),
            original_image=sample_vision_arrays["hd_frame"]
        )
        
        # Serialize with blosc
        serialized_data = serializer.serialize_processed_frame(frame)
        assert len(serialized_data) > 0
        
        # Deserialize
        deserialized_frame = serializer.deserialize_processed_frame(serialized_data)
        
        # Verify
        assert deserialized_frame.frame_id == frame.frame_id
        assert deserialized_frame.camera_id == frame.camera_id
        np.testing.assert_array_equal(
            deserialized_frame.original_image, frame.original_image
        )
    
    def test_compression_comparison(self, sample_vision_arrays):
        """Compare blosc vs traditional compression."""
        from its_camera_ai.flow.streaming_processor import ProcessedFrame
        
        # Traditional serializer
        traditional_serializer = ProcessedFrameSerializer(
            enable_blosc_compression=False,
            enable_compression=True
        )
        
        # Blosc serializer
        blosc_serializer = ProcessedFrameSerializer(
            enable_blosc_compression=True,
            enable_compression=False  # Use only blosc
        )
        
        frame = ProcessedFrame(
            frame_id="test_comparison_001",
            camera_id="camera_001",
            timestamp=time.time(),
            original_image=sample_vision_arrays["hd_frame"]
        )
        
        # Serialize with both methods
        traditional_data = traditional_serializer.serialize_processed_frame(frame)
        blosc_data = blosc_serializer.serialize_processed_frame(frame)
        
        # Compare sizes
        traditional_size = len(traditional_data)
        blosc_size = len(blosc_data)
        
        print(f"Traditional compression: {traditional_size} bytes")
        print(f"Blosc compression: {blosc_size} bytes")
        print(f"Blosc efficiency: {(traditional_size - blosc_size) / traditional_size * 100:.1f}% smaller")
        
        # Blosc should be competitive or better for numpy arrays
        assert blosc_size > 0
        assert traditional_size > 0
    
    def test_serializer_performance_metrics(self, sample_vision_arrays):
        """Test performance metrics collection in serializer."""
        from its_camera_ai.flow.streaming_processor import ProcessedFrame
        
        serializer = ProcessedFrameSerializer(
            enable_blosc_compression=True,
            use_global_blosc_compressor=False  # Use dedicated instance for testing
        )
        
        # Create and serialize multiple frames
        frames = []
        for i in range(5):
            frame = ProcessedFrame(
                frame_id=f"test_metrics_{i:03d}",
                camera_id="camera_001",
                timestamp=time.time() + i,
                original_image=sample_vision_arrays["hd_frame"].copy()
            )
            frames.append(frame)
        
        # Serialize frames
        for frame in frames:
            serializer.serialize_processed_frame(frame)
        
        # Get performance metrics
        metrics = serializer.get_performance_metrics()
        
        # Verify metrics structure
        assert "blosc_compression_enabled" in metrics
        assert metrics["blosc_compression_enabled"] is True
        assert "blosc_compressions" in metrics
        assert metrics["blosc_compressions"] == 5
        assert "blosc_avg_compression_ratio" in metrics
        assert 0 < metrics["blosc_avg_compression_ratio"] < 1
        assert "blosc_size_reduction_percent" in metrics
        assert metrics["blosc_size_reduction_percent"] > 50  # At least 50% reduction
        
        print(f"Blosc compression metrics: {metrics}")
    
    def test_blosc_benchmark_integration(self, sample_vision_arrays):
        """Test blosc benchmarking integration."""
        serializer = ProcessedFrameSerializer(
            enable_blosc_compression=True,
            use_global_blosc_compressor=False
        )
        
        # Run benchmark
        test_array = sample_vision_arrays["hd_frame"]
        benchmark_results = serializer.get_blosc_benchmark_results(test_array)
        
        assert "benchmark_results" in benchmark_results
        assert "recommended_algorithm" in benchmark_results
        assert "test_array_info" in benchmark_results
        
        # Verify benchmark ran for multiple algorithms
        assert len(benchmark_results["benchmark_results"]) >= 3
        
        for algo_name, metrics in benchmark_results["benchmark_results"].items():
            assert metrics.compression_ratio < 1.0
            assert metrics.compression_time_ms > 0
            assert metrics.size_reduction_percent > 0


class TestConvenienceFunctions:
    """Test convenience functions for quick compression."""
    
    def test_global_convenience_functions(self, sample_vision_arrays):
        """Test global compress/decompress convenience functions."""
        test_array = sample_vision_arrays["feature_map"]
        
        # Compress using convenience function
        compressed = compress_numpy_array(test_array)
        assert len(compressed) > 0
        
        # Decompress using convenience function
        decompressed = decompress_numpy_array(
            compressed, test_array.dtype, test_array.shape
        )
        
        # Verify
        np.testing.assert_array_equal(test_array, decompressed)
        
        # Check compression ratio
        compression_ratio = len(compressed) / test_array.nbytes
        assert compression_ratio < 0.8, "Should achieve good compression on float32 data"


@pytest.mark.integration
class TestMLPipelineIntegration:
    """Integration tests with ML pipeline components."""
    
    def test_prediction_service_array_compression(self):
        """Test compression of prediction service arrays."""
        # Simulate prediction arrays
        batch_predictions = np.random.rand(32, 1000).astype(np.float32)
        feature_embeddings = np.random.randn(100, 512).astype(np.float32)
        attention_weights = np.random.rand(12, 64, 64).astype(np.float32)
        
        compressor = get_global_compressor()
        
        # Test compression of ML arrays
        arrays = {
            "predictions": batch_predictions,
            "features": feature_embeddings,
            "attention": attention_weights
        }
        
        total_original = 0
        total_compressed = 0
        
        for name, array in arrays.items():
            compressed = compressor.compress_array(array)
            decompressed = compressor.decompress_array(
                compressed, array.dtype, array.shape
            )
            
            # Verify correctness for ML data
            np.testing.assert_allclose(array, decompressed, rtol=1e-6)
            
            total_original += array.nbytes
            total_compressed += len(compressed)
            
            print(f"{name}: {array.nbytes} -> {len(compressed)} bytes "
                  f"({len(compressed)/array.nbytes:.3f} ratio)")
        
        overall_ratio = total_compressed / total_original
        print(f"Overall compression ratio: {overall_ratio:.3f}")
        assert overall_ratio < 0.7, "Should achieve good compression on ML arrays"
    
    def test_memory_transfer_optimization(self, sample_vision_arrays):
        """Test memory transfer optimization for inter-service communication."""
        # Simulate cross-service data transfer
        large_batch = {
            "frames": [sample_vision_arrays["hd_frame"] for _ in range(10)],
            "features": [sample_vision_arrays["feature_map"] for _ in range(5)],
            "detections": [sample_vision_arrays["bbox_coordinates"] for _ in range(20)]
        }
        
        compressor = get_global_compressor()
        
        # Compress entire batch
        compressed_batch = {}
        total_original_size = 0
        total_compressed_size = 0
        
        for data_type, arrays in large_batch.items():
            compressed_arrays = []
            for array in arrays:
                compressed = compressor.compress_with_metadata(array)
                compressed_arrays.append(compressed)
                total_original_size += array.nbytes
                total_compressed_size += len(compressed)
            
            compressed_batch[data_type] = compressed_arrays
        
        # Calculate bandwidth savings
        bandwidth_reduction = (total_original_size - total_compressed_size) / total_original_size * 100
        memory_efficiency = total_compressed_size / total_original_size
        
        print(f"Bandwidth reduction: {bandwidth_reduction:.1f}%")
        print(f"Memory efficiency: {memory_efficiency:.3f}")
        
        # Performance targets
        assert bandwidth_reduction > 50, f"Should achieve >50% bandwidth reduction, got {bandwidth_reduction:.1f}%"
        assert memory_efficiency < 0.4, f"Memory usage should be <40% of original, got {memory_efficiency:.3f}"
        
        # Verify decompression works
        for data_type, compressed_arrays in compressed_batch.items():
            original_arrays = large_batch[data_type]
            for i, compressed in enumerate(compressed_arrays):
                decompressed = compressor.decompress_with_metadata(compressed)
                np.testing.assert_array_equal(original_arrays[i], decompressed)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])