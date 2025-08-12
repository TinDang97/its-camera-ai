"""Tests for gRPC Serialization.

This module tests the high-performance gRPC serialization system
for ProcessedFrame data optimized for video stream processing.
"""

import time

import numpy as np
import pytest

from its_camera_ai.data.grpc_serialization import (
    ImageCompressor,
    ProcessedFrameSerializer,
)
from its_camera_ai.data.streaming_processor import (
    ProcessedFrame,
    ProcessingStage,
)


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create a 100x100 RGB image with random data
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_processed_frame(sample_image):
    """Create a sample ProcessedFrame for testing."""
    # Create a smaller thumbnail
    thumbnail = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

    frame = ProcessedFrame(
        frame_id="test_frame_001",
        camera_id="camera_001",
        timestamp=time.time(),
        original_image=sample_image,
        processed_image=sample_image.copy(),
        thumbnail=thumbnail,
    )

    # Set quality metrics
    frame.quality_score = 0.85
    frame.blur_score = 0.90
    frame.brightness_score = 0.80
    frame.contrast_score = 0.75
    frame.noise_level = 0.95

    # Set traffic features
    frame.vehicle_density = 0.35
    frame.congestion_level = "medium"
    frame.weather_conditions = "clear"
    frame.lighting_conditions = "normal"

    # Set ROI features
    frame.roi_features = {
        "roi_0": {"density": 0.4, "brightness": 120.5, "congestion": "low"},
        "roi_1": {"density": 0.6, "brightness": 95.2, "congestion": "high"},
    }

    # Set processing metadata
    frame.processing_time_ms = 45.5
    frame.processing_stage = ProcessingStage.OUTPUT
    frame.validation_passed = True
    frame.source_hash = "abc123def456"
    frame.version = "1.0"

    return frame


class TestImageCompressor:
    """Test ImageCompressor functionality."""

    def test_compress_decompress_jpeg(self, sample_image):
        """Test JPEG compression and decompression."""
        compressor = ImageCompressor(default_format="jpeg", default_quality=85)

        # Compress
        compressed_data, metadata = compressor.compress_image(sample_image)

        assert len(compressed_data) > 0
        assert metadata["format"] == "jpeg"
        assert metadata["quality"] == 85
        assert metadata["width"] == sample_image.shape[1]
        assert metadata["height"] == sample_image.shape[0]
        assert metadata["compression_ratio"] < 1.0  # Should be compressed

        # Decompress
        decompressed = compressor.decompress_image(compressed_data)

        assert decompressed.shape == sample_image.shape
        assert decompressed.dtype == np.uint8

    def test_compress_different_formats(self, sample_image):
        """Test different compression formats."""
        compressor = ImageCompressor()

        formats = ["jpeg", "png", "webp"]

        for fmt in formats:
            compressed_data, metadata = compressor.compress_image(
                sample_image, format_type=fmt
            )

            assert len(compressed_data) > 0
            assert metadata["format"] == fmt

            # Verify decompression works
            decompressed = compressor.decompress_image(compressed_data)
            assert decompressed.shape == sample_image.shape

    def test_compression_quality_levels(self, sample_image):
        """Test different compression quality levels."""
        compressor = ImageCompressor(default_format="jpeg")

        qualities = [30, 60, 90]
        sizes = []

        for quality in qualities:
            compressed_data, metadata = compressor.compress_image(
                sample_image, quality=quality
            )

            sizes.append(len(compressed_data))
            assert metadata["quality"] == quality

        # Higher quality should generally result in larger files
        assert sizes[0] <= sizes[1] <= sizes[2]

    def test_create_thumbnail(self, sample_image):
        """Test thumbnail creation."""
        compressor = ImageCompressor()

        thumbnail_data, metadata = compressor.create_thumbnail(sample_image)

        assert len(thumbnail_data) > 0
        assert metadata["width"] == 128  # Default thumbnail size
        assert metadata["height"] == 128

        # Decompress to verify
        thumbnail = compressor.decompress_image(thumbnail_data)
        assert thumbnail.shape == (128, 128, 3)

    def test_grayscale_image(self):
        """Test compression of grayscale images."""
        # Create grayscale image
        gray_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

        compressor = ImageCompressor()
        compressed_data, metadata = compressor.compress_image(gray_image)

        assert len(compressed_data) > 0
        assert metadata["channels"] == 1

        # Decompress
        decompressed = compressor.decompress_image(compressed_data)
        assert len(decompressed.shape) == 2 or decompressed.shape[2] == 1

    def test_empty_image_handling(self):
        """Test handling of empty/invalid images."""
        compressor = ImageCompressor()

        # Test None input
        compressed_data, metadata = compressor.compress_image(None)
        assert compressed_data == b""
        assert metadata == {}

        # Test empty array
        empty_array = np.array([])
        compressed_data, metadata = compressor.compress_image(empty_array)
        assert compressed_data == b""

    def test_compression_performance(self, sample_image):
        """Test compression performance benchmarks."""
        compressor = ImageCompressor(default_format="jpeg", default_quality=85)

        # Benchmark compression
        iterations = 10
        start_time = time.time()

        for _ in range(iterations):
            compressed_data, _ = compressor.compress_image(sample_image)

        compression_time = (time.time() - start_time) / iterations

        # Should compress 100x100 image in reasonable time
        assert compression_time < 0.1  # Less than 100ms per image

        print(f"Average compression time: {compression_time * 1000:.2f}ms")


class TestProcessedFrameSerializer:
    """Test ProcessedFrameSerializer functionality."""

    def test_serialize_deserialize_complete_frame(self, sample_processed_frame):
        """Test complete serialization and deserialization."""
        serializer = ProcessedFrameSerializer(
            compression_format="jpeg", compression_quality=85, enable_compression=True
        )

        # Serialize
        serialized_data = serializer.serialize_processed_frame(sample_processed_frame)

        assert isinstance(serialized_data, bytes)
        assert len(serialized_data) > 0

        # Should be significantly smaller than raw data due to compression
        original_size = sample_processed_frame.original_image.nbytes
        compression_ratio = len(serialized_data) / original_size
        assert compression_ratio < 0.5  # At least 50% compression

        # Deserialize
        deserialized_frame = serializer.deserialize_processed_frame(serialized_data)

        # Verify core fields
        assert deserialized_frame.frame_id == sample_processed_frame.frame_id
        assert deserialized_frame.camera_id == sample_processed_frame.camera_id
        assert (
            abs(deserialized_frame.timestamp - sample_processed_frame.timestamp) < 1.0
        )

        # Verify quality metrics
        assert (
            abs(deserialized_frame.quality_score - sample_processed_frame.quality_score)
            < 0.01
        )
        assert (
            abs(deserialized_frame.blur_score - sample_processed_frame.blur_score)
            < 0.01
        )

        # Verify traffic features
        assert (
            abs(
                deserialized_frame.vehicle_density
                - sample_processed_frame.vehicle_density
            )
            < 0.01
        )
        assert (
            deserialized_frame.congestion_level
            == sample_processed_frame.congestion_level
        )
        assert (
            deserialized_frame.weather_conditions
            == sample_processed_frame.weather_conditions
        )

        # Verify ROI features
        assert len(deserialized_frame.roi_features) == len(
            sample_processed_frame.roi_features
        )

        # Verify processing metadata
        assert (
            abs(
                deserialized_frame.processing_time_ms
                - sample_processed_frame.processing_time_ms
            )
            < 0.1
        )
        assert (
            deserialized_frame.validation_passed
            == sample_processed_frame.validation_passed
        )
        assert deserialized_frame.source_hash == sample_processed_frame.source_hash

    def test_image_serialization_fidelity(self, sample_processed_frame):
        """Test image data preservation through serialization."""
        serializer = ProcessedFrameSerializer(enable_compression=True)

        # Serialize and deserialize
        serialized_data = serializer.serialize_processed_frame(sample_processed_frame)
        deserialized_frame = serializer.deserialize_processed_frame(serialized_data)

        # Images should have same dimensions (compression may change pixel values slightly)
        original_img = sample_processed_frame.original_image
        deserialized_img = deserialized_frame.original_image

        assert deserialized_img.shape == original_img.shape
        assert deserialized_img.dtype == original_img.dtype

        # For JPEG compression, allow some tolerance due to lossy compression
        # Calculate mean absolute difference
        diff = np.mean(
            np.abs(original_img.astype(float) - deserialized_img.astype(float))
        )
        assert diff < 10.0  # Should be reasonably close

    def test_serialization_without_compression(self, sample_processed_frame):
        """Test serialization with compression disabled."""
        serializer = ProcessedFrameSerializer(enable_compression=False)

        # Serialize
        serialized_data = serializer.serialize_processed_frame(sample_processed_frame)

        # Should be larger than compressed version
        compressed_serializer = ProcessedFrameSerializer(enable_compression=True)
        compressed_data = compressed_serializer.serialize_processed_frame(
            sample_processed_frame
        )

        assert len(serialized_data) > len(compressed_data)

        # Deserialize and verify exact image match
        deserialized_frame = serializer.deserialize_processed_frame(serialized_data)

        # Without compression, images should be exactly the same
        np.testing.assert_array_equal(
            sample_processed_frame.original_image, deserialized_frame.original_image
        )

    def test_partial_frame_serialization(self, sample_image):
        """Test serialization of frame with missing optional fields."""
        # Create minimal frame
        minimal_frame = ProcessedFrame(
            frame_id="minimal_001",
            camera_id="camera_001",
            timestamp=time.time(),
            original_image=sample_image,
        )

        serializer = ProcessedFrameSerializer()

        # Should handle minimal frame without errors
        serialized_data = serializer.serialize_processed_frame(minimal_frame)
        assert len(serialized_data) > 0

        # Deserialize
        deserialized_frame = serializer.deserialize_processed_frame(serialized_data)
        assert deserialized_frame.frame_id == minimal_frame.frame_id
        assert deserialized_frame.camera_id == minimal_frame.camera_id

    def test_batch_serialization(self, sample_processed_frame):
        """Test batch serialization functionality."""
        serializer = ProcessedFrameSerializer()

        # Create batch of frames
        frames = []
        for i in range(3):
            frame = ProcessedFrame(
                frame_id=f"batch_frame_{i}",
                camera_id="camera_001",
                timestamp=time.time() + i,
                original_image=sample_processed_frame.original_image.copy(),
            )
            frames.append(frame)

        # Serialize batch
        batch_data = serializer.serialize_batch(frames, batch_id="test_batch_001")

        assert isinstance(batch_data, bytes)
        assert len(batch_data) > 0

        # Batch should be more efficient than individual serializations
        individual_size = sum(
            len(serializer.serialize_processed_frame(frame)) for frame in frames
        )

        # Batch might have some overhead, but should be reasonably efficient
        efficiency_ratio = len(batch_data) / individual_size
        assert 0.8 <= efficiency_ratio <= 1.2  # Within 20% of individual serializations

    def test_performance_metrics(self, sample_processed_frame):
        """Test serialization performance tracking."""
        serializer = ProcessedFrameSerializer()

        # Perform multiple serializations
        for _ in range(5):
            serializer.serialize_processed_frame(sample_processed_frame)

        # Get performance metrics
        metrics = serializer.get_performance_metrics()

        assert "avg_serialization_time_ms" in metrics
        assert "max_serialization_time_ms" in metrics
        assert "total_serializations" in metrics
        assert metrics["total_serializations"] == 5

        # Serialization should be fast
        assert metrics["avg_serialization_time_ms"] < 100  # Less than 100ms average

    def test_different_compression_formats(self, sample_processed_frame):
        """Test serialization with different compression formats."""
        formats = ["jpeg", "png", "webp"]
        results = {}

        for fmt in formats:
            serializer = ProcessedFrameSerializer(
                compression_format=fmt, enable_compression=True
            )

            serialized_data = serializer.serialize_processed_frame(
                sample_processed_frame
            )
            results[fmt] = len(serialized_data)

            # Verify deserialization works
            deserialized = serializer.deserialize_processed_frame(serialized_data)
            assert deserialized.frame_id == sample_processed_frame.frame_id

        # All formats should produce valid results
        assert all(size > 0 for size in results.values())

        print(f"Compression sizes: {results}")

    @pytest.mark.benchmark
    def test_serialization_performance_benchmark(self, sample_processed_frame):
        """Benchmark serialization performance."""
        serializer = ProcessedFrameSerializer(
            compression_format="jpeg", compression_quality=85, enable_compression=True
        )

        iterations = 50

        # Benchmark serialization
        start_time = time.time()
        serialized_results = []

        for _ in range(iterations):
            data = serializer.serialize_processed_frame(sample_processed_frame)
            serialized_results.append(data)

        serialization_time = time.time() - start_time

        # Benchmark deserialization
        start_time = time.time()

        for data in serialized_results:
            serializer.deserialize_processed_frame(data)

        deserialization_time = time.time() - start_time

        # Calculate rates
        serialization_rate = iterations / serialization_time
        deserialization_rate = iterations / deserialization_time

        print(f"Serialization rate: {serialization_rate:.1f} frames/second")
        print(f"Deserialization rate: {deserialization_rate:.1f} frames/second")

        # Performance requirements
        assert serialization_rate > 50  # At least 50 frames/second
        assert deserialization_rate > 50  # At least 50 frames/second

        # Calculate compression effectiveness
        original_size = sample_processed_frame.original_image.nbytes
        avg_compressed_size = sum(len(data) for data in serialized_results) / len(
            serialized_results
        )
        compression_ratio = avg_compressed_size / original_size

        print(f"Average compression ratio: {compression_ratio:.3f}")
        assert compression_ratio < 0.3  # At least 70% compression

    def test_error_handling(self):
        """Test error handling in serialization."""
        serializer = ProcessedFrameSerializer()

        # Test with invalid data
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            serializer.deserialize_processed_frame(b"invalid protobuf data")

        # Test with empty data
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            serializer.deserialize_processed_frame(b"")

    def test_version_compatibility(self, sample_processed_frame):
        """Test version compatibility of serialized data."""
        serializer = ProcessedFrameSerializer()

        # Set specific version
        sample_processed_frame.version = "2.0"

        # Serialize
        serialized_data = serializer.serialize_processed_frame(sample_processed_frame)

        # Deserialize
        deserialized_frame = serializer.deserialize_processed_frame(serialized_data)

        # Version should be preserved
        assert deserialized_frame.version == "2.0"


@pytest.mark.integration
class TestSerializationIntegration:
    """Integration tests for gRPC serialization with real data."""

    def test_real_world_frame_sizes(self):
        """Test with realistic frame sizes."""
        # Create HD frame (1920x1080)
        hd_image = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

        frame = ProcessedFrame(
            frame_id="hd_frame_001",
            camera_id="hd_camera_001",
            timestamp=time.time(),
            original_image=hd_image,
        )

        serializer = ProcessedFrameSerializer(enable_compression=True)

        # Serialize HD frame
        start_time = time.time()
        serialized_data = serializer.serialize_processed_frame(frame)
        serialization_time = (time.time() - start_time) * 1000

        # Should handle HD frames efficiently
        assert serialization_time < 200  # Less than 200ms for HD frame

        # Should achieve good compression
        compression_ratio = len(serialized_data) / hd_image.nbytes
        assert compression_ratio < 0.1  # At least 90% compression for HD

        print(
            f"HD frame compression: {compression_ratio:.4f} ratio, {serialization_time:.1f}ms"
        )
