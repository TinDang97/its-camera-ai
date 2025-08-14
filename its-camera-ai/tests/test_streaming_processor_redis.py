"""Tests for Redis-based StreamProcessor.

This module tests the updated StreamProcessor that uses Redis queues
instead of Kafka, with gRPC serialization optimization.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from its_camera_ai.data.streaming_processor import (
    ProcessedFrame,
    StreamProcessor,
    StreamStatus,
)


@pytest.fixture
def redis_config():
    """Create Redis-based configuration for testing."""
    return {
        "redis_url": "redis://localhost:6379/15",  # Test database
        "input_queue": "test_camera_frames",
        "output_queue": "test_processed_frames",
        "queue_pool_size": 5,
        "max_concurrent_streams": 10,
        "quality_threshold": 0.7,
        "processing_timeout": 5.0,
        "enable_compression": True,
        "compression_format": "jpeg",
        "compression_quality": 85,
    }


@pytest.fixture
def sample_frame_data():
    """Create sample frame data for testing."""
    # Create a small test image
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    return {
        "frame_id": "test_frame_001",
        "camera_id": "test_camera_001",
        "timestamp": time.time(),
        "image_array": image.tolist(),  # Convert to list for JSON serialization
    }


@pytest.fixture
def sample_camera_stream():
    """Create sample camera stream configuration."""
    return {
        "camera_id": "test_camera_001",
        "location": "Test Location",
        "coordinates": [40.7128, -74.0060],  # NYC coordinates
        "resolution": [640, 480],
        "fps": 30,
        "quality_threshold": 0.7,
    }


@pytest.mark.asyncio
class TestRedisStreamProcessor:
    """Test Redis-based StreamProcessor functionality."""

    async def test_processor_initialization(self, redis_config):
        """Test processor initialization with Redis configuration."""
        processor = StreamProcessor(redis_config)

        # Check configuration is set correctly
        assert processor.redis_url == redis_config["redis_url"]
        assert processor.input_queue == redis_config["input_queue"]
        assert processor.output_queue == redis_config["output_queue"]
        assert processor.enable_compression == redis_config["enable_compression"]

        # Initial state should be correct
        assert processor.queue_manager is None
        assert processor.serializer is None

    @patch("its_camera_ai.data.streaming_processor.REDIS_AVAILABLE", True)
    async def test_processor_startup(self, redis_config):
        """Test processor startup sequence."""
        with (
            patch(
                "its_camera_ai.flow.redis_queue_manager.RedisQueueManager"
            ) as mock_manager_class,
            patch(
                "its_camera_ai.data.grpc_serialization.ProcessedFrameSerializer"
            ) as mock_serializer_class,
        ):
            # Setup mocks
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            mock_serializer = MagicMock()
            mock_serializer_class.return_value = mock_serializer

            processor = StreamProcessor(redis_config)

            # Mock Redis cache connection
            with patch("redis.asyncio.from_url") as mock_redis:
                mock_redis.return_value = AsyncMock()

                # Start processor
                await processor.start()

                # Verify queue manager was created and connected
                mock_manager_class.assert_called_once()
                mock_manager.connect.assert_called_once()
                mock_manager.create_queue.assert_called()

                # Verify serializer was created
                mock_serializer_class.assert_called_once_with(
                    compression_format=redis_config["compression_format"],
                    compression_quality=redis_config["compression_quality"],
                    enable_compression=redis_config["enable_compression"],
                )

                # Cleanup
                await processor.stop()

    @patch("its_camera_ai.data.streaming_processor.REDIS_AVAILABLE", False)
    async def test_processor_startup_without_redis(self, redis_config):
        """Test processor startup fails gracefully without Redis."""
        processor = StreamProcessor(redis_config)

        with pytest.raises(RuntimeError, match="Redis is required"):
            await processor.start()

    async def test_frame_enqueuing(self, redis_config, sample_frame_data):
        """Test enqueueing frames for processing."""
        with patch(
            "its_camera_ai.flow.redis_queue_manager.RedisQueueManager"
        ) as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager
            mock_manager.enqueue.return_value = "message_123"

            processor = StreamProcessor(redis_config)
            processor.queue_manager = mock_manager

            # Enqueue frame
            message_id = await processor.enqueue_frame_for_processing(
                sample_frame_data, priority=1
            )

            assert message_id == "message_123"

            # Verify enqueue was called with correct parameters
            mock_manager.enqueue.assert_called_once()
            call_args = mock_manager.enqueue.call_args

            assert call_args[0][0] == redis_config["input_queue"]  # queue name
            assert isinstance(call_args[0][1], bytes)  # message data
            assert call_args[1]["priority"] == 1
            assert "frame_id" in call_args[1]["metadata"]

    async def test_single_message_processing(self, redis_config, sample_frame_data):
        """Test processing of a single message."""
        processor = StreamProcessor(redis_config)

        # Mock dependencies
        processor.serializer = MagicMock()
        processor.quality_analyzer = MagicMock()
        processor.feature_extractor = MagicMock()

        # Setup mock responses
        processor.quality_analyzer.analyze_quality.return_value = {
            "quality_score": 0.85,
            "blur_score": 0.90,
            "brightness_score": 0.80,
            "contrast_score": 0.75,
            "noise_level": 0.95,
        }

        processor.feature_extractor.extract_features.return_value = {
            "vehicle_density": 0.35,
            "congestion_level": "medium",
            "weather_conditions": "clear",
            "lighting_conditions": "normal",
        }

        # Test message processing
        message_data = json.dumps(sample_frame_data).encode("utf-8")
        result = await processor._process_single_message("msg_001", message_data)

        assert result is True  # Should succeed with good quality

        # Verify quality analysis was called
        processor.quality_analyzer.analyze_quality.assert_called_once()
        processor.feature_extractor.extract_features.assert_called_once()

    async def test_batch_message_processing(self, redis_config):
        """Test batch processing of multiple messages."""
        processor = StreamProcessor(redis_config)

        # Mock queue manager
        processor.queue_manager = AsyncMock()

        # Create test messages
        messages = []
        for i in range(3):
            frame_data = {
                "frame_id": f"frame_{i}",
                "camera_id": "test_camera",
                "timestamp": time.time(),
                "image_array": np.random.randint(0, 256, (32, 32, 3)).tolist(),
            }
            message_data = json.dumps(frame_data).encode("utf-8")
            messages.append((f"msg_{i}", message_data))

        # Mock single message processing to always succeed
        async def mock_process_single(msg_id, data):
            return True

        processor._process_single_message = mock_process_single

        # Process batch
        await processor._process_frame_batch(messages)

        # Verify all messages were acknowledged
        assert processor.queue_manager.acknowledge.call_count == 3
        assert processor.performance_metrics["frames_processed"] == 3

    async def test_frame_emission(self, redis_config):
        """Test emission of processed frames."""
        processor = StreamProcessor(redis_config)

        # Mock dependencies
        processor.queue_manager = AsyncMock()
        processor.serializer = MagicMock()
        processor.redis_client = AsyncMock()

        # Create test frame
        image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        frame = ProcessedFrame(
            frame_id="test_frame",
            camera_id="test_camera",
            timestamp=time.time(),
            original_image=image,
        )
        frame.quality_score = 0.85
        frame.validation_passed = True

        # Mock serialization
        processor.serializer.serialize_processed_frame.return_value = b"serialized_data"

        # Emit frame
        await processor._emit_processed_frame(frame)

        # Verify serialization and enqueueing
        processor.serializer.serialize_processed_frame.assert_called_once_with(frame)
        processor.queue_manager.enqueue.assert_called_once()

        # Verify Redis caching
        processor.redis_client.setex.assert_called_once()

    async def test_stream_registration(self, redis_config, sample_camera_stream):
        """Test camera stream registration."""
        processor = StreamProcessor(redis_config)

        # Register stream
        success = await processor.register_stream(sample_camera_stream)

        assert success is True
        assert sample_camera_stream["camera_id"] in processor.active_streams
        assert sample_camera_stream["camera_id"] in processor.processing_queues

        # Verify stream configuration
        stream = processor.active_streams[sample_camera_stream["camera_id"]]
        assert stream.camera_id == sample_camera_stream["camera_id"]
        assert stream.location == sample_camera_stream["location"]
        assert stream.resolution == tuple(sample_camera_stream["resolution"])

    async def test_stream_status_retrieval(self, redis_config, sample_camera_stream):
        """Test getting stream status."""
        processor = StreamProcessor(redis_config)

        # Register stream first
        await processor.register_stream(sample_camera_stream)

        # Get status
        status = processor.get_stream_status(sample_camera_stream["camera_id"])

        assert status is not None
        assert status["camera_id"] == sample_camera_stream["camera_id"]
        assert status["status"] == StreamStatus.ACTIVE.value
        assert "total_frames_processed" in status
        assert "avg_processing_latency_ms" in status

    async def test_processing_statistics(self, redis_config):
        """Test getting processing statistics."""
        processor = StreamProcessor(redis_config)

        # Mock serializer for performance metrics
        processor.serializer = MagicMock()
        processor.serializer.get_performance_metrics.return_value = {
            "avg_serialization_time_ms": 15.5,
            "total_serializations": 100,
            "compression_enabled": True,
        }

        # Get stats
        stats = processor.get_processing_stats()

        assert "active_streams" in stats
        assert "performance_metrics" in stats
        assert "queue_manager_available" in stats
        assert "serialization_enabled" in stats
        assert "serialization_metrics" in stats

    async def test_queue_health_check(self, redis_config):
        """Test queue health checking."""
        processor = StreamProcessor(redis_config)

        # Mock queue manager
        processor.queue_manager = AsyncMock()
        processor.queue_manager.health_check.return_value = {
            "status": "healthy",
            "redis_connected": True,
            "queue_count": 2,
            "total_pending": 5,
        }

        # Get health
        health = await processor.get_queue_health()

        assert health["status"] == "healthy"
        assert health["redis_connected"] is True

        # Test when queue manager is not available
        processor.queue_manager = None
        health = await processor.get_queue_health()
        assert health["status"] == "unavailable"

    async def test_error_handling_in_batch_processing(self, redis_config):
        """Test error handling during batch processing."""
        processor = StreamProcessor(redis_config)
        processor.queue_manager = AsyncMock()

        # Create messages with one that will fail
        good_message = json.dumps(
            {
                "frame_id": "good_frame",
                "camera_id": "test_camera",
                "timestamp": time.time(),
                "image_array": np.random.randint(0, 256, (32, 32, 3)).tolist(),
            }
        ).encode("utf-8")

        bad_message = b"invalid json data"

        messages = [
            ("msg_good", good_message),
            ("msg_bad", bad_message),
        ]

        # Mock processing methods
        async def mock_process_single(msg_id, data):
            if msg_id == "msg_bad":
                raise ValueError("Invalid data")
            return True

        processor._process_single_message = mock_process_single

        # Process batch - should handle errors gracefully
        await processor._process_frame_batch(messages)

        # Good message should be acknowledged, bad message should be rejected
        processor.queue_manager.acknowledge.assert_called_once_with(
            redis_config["input_queue"], "msg_good", processing_time_ms=None
        )
        processor.queue_manager.reject.assert_called_once_with(
            redis_config["input_queue"], "msg_bad", reason="Invalid data"
        )

    @pytest.mark.benchmark
    async def test_processing_performance(self, redis_config):
        """Benchmark processing performance."""
        processor = StreamProcessor(redis_config)

        # Setup minimal mocks for performance test
        processor.quality_analyzer = MagicMock()
        processor.feature_extractor = MagicMock()
        processor.serializer = MagicMock()

        processor.quality_analyzer.analyze_quality.return_value = {
            "quality_score": 0.85,
            "blur_score": 0.90,
            "brightness_score": 0.80,
            "contrast_score": 0.75,
            "noise_level": 0.95,
        }

        processor.feature_extractor.extract_features.return_value = {
            "vehicle_density": 0.35,
            "congestion_level": "medium",
        }

        processor.serializer.serialize_processed_frame.return_value = b"test_data"

        # Create test messages
        message_count = 20
        messages = []

        for i in range(message_count):
            frame_data = {
                "frame_id": f"perf_frame_{i}",
                "camera_id": "perf_camera",
                "timestamp": time.time(),
                "image_array": np.random.randint(0, 256, (64, 64, 3)).tolist(),
            }
            message_data = json.dumps(frame_data).encode("utf-8")
            messages.append((f"msg_{i}", message_data))

        # Benchmark processing
        start_time = time.time()

        # Process messages individually
        for msg_id, msg_data in messages:
            await processor._process_single_message(msg_id, msg_data)

        processing_time = time.time() - start_time
        processing_rate = message_count / processing_time

        print(f"Processing rate: {processing_rate:.1f} frames/second")

        # Performance requirement: should process at least 10 frames/second
        assert processing_rate > 10

        # Average processing time should be under 100ms
        avg_time_per_frame = (processing_time / message_count) * 1000
        print(f"Average processing time: {avg_time_per_frame:.2f}ms")
        assert avg_time_per_frame < 100


@pytest.mark.integration
class TestStreamProcessorIntegration:
    """Integration tests requiring actual Redis instance."""

    @pytest.mark.skipif(
        not pytest.redis_available, reason="Redis not available for integration tests"
    )
    async def test_end_to_end_processing(self, redis_config, sample_frame_data):
        """Test complete end-to-end processing with real Redis."""
        # Use real Redis for integration test
        processor = StreamProcessor(redis_config)

        try:
            # Start processor
            await processor.start()

            # Register a camera stream
            stream_config = {
                "camera_id": "integration_camera",
                "location": "Integration Test",
                "coordinates": [0.0, 0.0],
            }
            await processor.register_stream(stream_config)

            # Enqueue a frame for processing
            message_id = await processor.enqueue_frame_for_processing(
                sample_frame_data, priority=1
            )

            assert message_id is not None

            # Allow some time for processing
            await asyncio.sleep(2)

            # Check processing stats
            stats = processor.get_processing_stats()
            assert stats["queue_manager_available"] is True
            assert stats["serialization_enabled"] is True

            # Check queue health
            health = await processor.get_queue_health()
            assert health["status"] == "healthy"

        finally:
            await processor.stop()
