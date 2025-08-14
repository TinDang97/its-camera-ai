"""Comprehensive tests for SSE Streaming Service (STRM-001).

Tests for the core SSE streaming service functionality including:
- SSE connection establishment and management
- MP4 fragment generation and streaming
- Authentication and authorization
- Performance benchmarks
- Error handling and resilience
- Integration with existing streaming infrastructure
"""

import asyncio
import json
import time
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import numpy as np

from src.its_camera_ai.services.streaming_service import (
    SSEStreamingService,
    SSEStream,
    MP4Fragment,
    StreamingService,
    ProcessedFrame,
    StreamProcessingError,
)
from src.its_camera_ai.flow.redis_queue_manager import RedisQueueManager


class MockUser:
    """Mock user for testing."""

    def __init__(self, user_id: str = "test_user", username: str = "testuser"):
        self.id = user_id
        self.username = username


class MockRequest:
    """Mock FastAPI request for testing."""

    def __init__(self, disconnected: bool = False):
        self._disconnected = disconnected
        self.state = MagicMock()
        self.state.user = MockUser()

    async def is_disconnected(self) -> bool:
        return self._disconnected


@pytest.fixture
def mock_redis_manager():
    """Mock Redis queue manager."""
    redis_manager = MagicMock(spec=RedisQueueManager)
    redis_manager.health_check = AsyncMock(return_value={"status": "healthy"})
    return redis_manager


@pytest.fixture
def mock_streaming_processor():
    """Mock streaming data processor."""
    processor = MagicMock()

    # Mock process_stream method
    async def mock_process_stream(camera_id: str):
        for i in range(5):  # Generate 5 test frames
            frame = ProcessedFrame(
                frame_id=f"frame_{i}",
                camera_id=camera_id,
                timestamp=time.time(),
                original_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                quality_score=0.85,
                processing_time_ms=5.0,
            )
            yield frame
            await asyncio.sleep(0.01)  # Small delay

    processor.process_stream = mock_process_stream
    return processor


@pytest.fixture
def mock_base_streaming_service(mock_streaming_processor, mock_redis_manager):
    """Mock base streaming service."""
    service = MagicMock(spec=StreamingService)
    service.streaming_processor = mock_streaming_processor
    service.redis_manager = mock_redis_manager
    service.health_check = AsyncMock(
        return_value={"healthy": True, "status": "running"}
    )
    return service


@pytest.fixture
def sse_config():
    """SSE streaming service configuration."""
    return {
        "max_concurrent_connections": 10,
        "fragment_duration_ms": 1000,
        "heartbeat_interval": 10,
        "connection_timeout": 60,
    }


@pytest.fixture
def sse_streaming_service(mock_base_streaming_service, mock_redis_manager, sse_config):
    """SSE streaming service instance."""
    return SSEStreamingService(
        base_streaming_service=mock_base_streaming_service,
        redis_manager=mock_redis_manager,
        config=sse_config,
    )


class TestSSEStreamingService:
    """Test suite for SSE streaming service core functionality."""

    @pytest.mark.asyncio
    async def test_create_sse_stream_success(self, sse_streaming_service):
        """Test successful SSE stream creation."""
        user = MockUser()
        camera_id = "test_camera_001"

        stream = await sse_streaming_service.create_sse_stream(
            camera_id=camera_id, user=user, stream_type="raw", quality="medium"
        )

        assert isinstance(stream, SSEStream)
        assert stream.camera_id == camera_id
        assert stream.user_id == user.id
        assert stream.stream_type == "raw"
        assert stream.quality == "medium"
        assert stream.is_active is True

        # Check that stream is registered
        assert stream.stream_id in sse_streaming_service.active_streams
        assert stream.stream_id in sse_streaming_service.stream_queues

        # Check metrics update
        assert sse_streaming_service.connection_metrics.active_connections == 1
        assert sse_streaming_service.connection_metrics.total_connections_created == 1

    @pytest.mark.asyncio
    async def test_create_sse_stream_concurrent_limit(self, sse_streaming_service):
        """Test concurrent connection limit enforcement."""
        user = MockUser()

        # Fill up to the limit
        for i in range(sse_streaming_service.max_concurrent_connections):
            await sse_streaming_service.create_sse_stream(
                camera_id=f"camera_{i}", user=user, stream_type="raw", quality="medium"
            )

        # Next connection should fail
        with pytest.raises(
            StreamProcessingError, match="Maximum concurrent connections"
        ):
            await sse_streaming_service.create_sse_stream(
                camera_id="overflow_camera",
                user=user,
                stream_type="raw",
                quality="medium",
            )

    @pytest.mark.asyncio
    async def test_stream_mp4_fragments(self, sse_streaming_service):
        """Test MP4 fragment generation from camera stream."""
        camera_id = "test_camera_001"
        quality = "medium"

        fragments = []
        async for fragment in sse_streaming_service.stream_mp4_fragments(
            camera_id, quality
        ):
            fragments.append(fragment)
            if len(fragments) >= 3:  # Collect first 3 fragments
                break

        assert len(fragments) == 3

        for i, fragment in enumerate(fragments):
            assert isinstance(fragment, MP4Fragment)
            assert fragment.camera_id == camera_id
            assert fragment.sequence_number == i
            assert fragment.quality == quality
            assert fragment.size_bytes > 0
            assert len(fragment.data) > 0
            assert fragment.content_type == "image/jpeg"  # Temporary format

    @pytest.mark.asyncio
    async def test_handle_sse_connection_success(self, sse_streaming_service):
        """Test successful SSE connection handling."""
        request = MockRequest()
        camera_id = "test_camera_001"

        # Mock StreamingResponse
        with patch(
            "src.its_camera_ai.services.streaming_service.StreamingResponse"
        ) as mock_response:
            response = await sse_streaming_service.handle_sse_connection(
                request=request,
                camera_id=camera_id,
                stream_type="raw",
                quality="medium",
            )

            # Verify StreamingResponse was called with correct parameters
            mock_response.assert_called_once()
            call_args = mock_response.call_args

            # Check media type and headers
            assert call_args[1]["media_type"] == "text/event-stream"
            headers = call_args[1]["headers"]
            assert headers["Cache-Control"] == "no-cache"
            assert headers["Connection"] == "keep-alive"
            assert headers["X-Accel-Buffering"] == "no"

    @pytest.mark.asyncio
    async def test_sse_event_formatting(self, sse_streaming_service):
        """Test SSE event formatting."""
        event_type = "test_event"
        data = {"key": "value", "number": 42}

        formatted_event = sse_streaming_service._format_sse_event(event_type, data)

        assert formatted_event.startswith(f"event: {event_type}\n")
        assert "data: " in formatted_event
        assert formatted_event.endswith("\n\n")

        # Parse the JSON data
        data_line = [
            line for line in formatted_event.split("\n") if line.startswith("data: ")
        ][0]
        json_data = json.loads(data_line[6:])  # Remove "data: " prefix

        assert json_data["type"] == event_type
        assert json_data["key"] == "value"
        assert json_data["number"] == 42
        assert "timestamp" in json_data

    @pytest.mark.asyncio
    async def test_connection_cleanup(self, sse_streaming_service):
        """Test proper cleanup of SSE connections."""
        user = MockUser()
        camera_id = "test_camera_001"

        # Create stream
        stream = await sse_streaming_service.create_sse_stream(
            camera_id=camera_id, user=user, stream_type="raw", quality="medium"
        )

        stream_id = stream.stream_id

        # Verify stream exists
        assert stream_id in sse_streaming_service.active_streams
        assert stream_id in sse_streaming_service.stream_queues

        # Cleanup stream
        await sse_streaming_service._cleanup_sse_stream(stream_id)

        # Verify cleanup
        assert stream_id not in sse_streaming_service.active_streams
        assert stream_id not in sse_streaming_service.stream_queues
        assert sse_streaming_service.connection_metrics.active_connections == 0
        assert sse_streaming_service.connection_metrics.total_disconnections == 1

    @pytest.mark.asyncio
    async def test_quality_settings(self, sse_streaming_service):
        """Test different quality settings."""
        test_cases = [
            ("low", 500, 60),
            ("medium", 2000, 85),
            ("high", 5000, 95),
        ]

        for quality, expected_bitrate, expected_jpeg_quality in test_cases:
            bitrate = sse_streaming_service._get_quality_bitrate(quality)
            jpeg_quality = sse_streaming_service._get_quality_jpeg_param(quality)

            assert bitrate == expected_bitrate
            assert jpeg_quality == expected_jpeg_quality

    @pytest.mark.asyncio
    async def test_connection_statistics(self, sse_streaming_service):
        """Test connection statistics collection."""
        # Initial stats
        stats = await sse_streaming_service.get_connection_stats()

        assert stats["active_connections"] == 0
        assert stats["total_connections_created"] == 0
        assert stats["total_disconnections"] == 0

        # Create some connections
        user = MockUser()
        streams = []

        for i in range(3):
            stream = await sse_streaming_service.create_sse_stream(
                camera_id=f"camera_{i}", user=user, stream_type="raw", quality="medium"
            )
            streams.append(stream)

        # Check updated stats
        stats = await sse_streaming_service.get_connection_stats()

        assert stats["active_connections"] == 3
        assert stats["total_connections_created"] == 3
        assert stats["total_disconnections"] == 0

        # Disconnect one
        await sse_streaming_service._cleanup_sse_stream(streams[0].stream_id)

        stats = await sse_streaming_service.get_connection_stats()
        assert stats["active_connections"] == 2
        assert stats["total_disconnections"] == 1

    @pytest.mark.asyncio
    async def test_health_check(self, sse_streaming_service):
        """Test SSE service health check."""
        health = await sse_streaming_service.health_check()

        assert "healthy" in health
        assert health["healthy"] is True
        assert health["sse_service_status"] == "healthy"
        assert "base_service_health" in health
        assert "connection_stats" in health
        assert "timestamp" in health

    @pytest.mark.asyncio
    async def test_fragment_streaming_to_queue(self, sse_streaming_service):
        """Test streaming fragments to SSE queue."""
        user = MockUser()
        camera_id = "test_camera_001"

        # Create stream
        stream = await sse_streaming_service.create_sse_stream(
            camera_id=camera_id, user=user, stream_type="raw", quality="medium"
        )

        # Start fragment streaming task
        task = asyncio.create_task(
            sse_streaming_service._stream_fragments_to_queue(stream)
        )

        # Wait a bit for fragments to be queued
        await asyncio.sleep(0.1)

        # Check that messages were queued
        queue = sse_streaming_service.stream_queues[stream.stream_id]
        assert not queue.empty()

        # Get a message
        message = await queue.get()
        assert message["type"] == "fragment"
        assert "data" in message

        fragment_data = message["data"]
        assert "fragment_id" in fragment_data
        assert "sequence_number" in fragment_data
        assert "timestamp" in fragment_data
        assert "data" in fragment_data
        assert fragment_data["content_type"] == "image/jpeg"

        # Cancel task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestSSEPerformance:
    """Performance tests for SSE streaming service."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_startup_time(
        self, mock_base_streaming_service, mock_redis_manager, sse_config
    ):
        """Test SSE service startup time requirement (<10ms)."""
        start_time = time.perf_counter()

        sse_service = SSEStreamingService(
            base_streaming_service=mock_base_streaming_service,
            redis_manager=mock_redis_manager,
            config=sse_config,
        )

        startup_time = (time.perf_counter() - start_time) * 1000

        # Should initialize in under 10ms
        assert (
            startup_time < 10.0
        ), f"Startup time {startup_time:.2f}ms exceeds 10ms requirement"
        assert sse_service is not None

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_fragment_processing_latency(self, sse_streaming_service):
        """Test fragment processing latency requirement (<50ms)."""
        camera_id = "test_camera_001"
        quality = "medium"

        latencies = []

        async for fragment in sse_streaming_service.stream_mp4_fragments(
            camera_id, quality
        ):
            # Extract processing time from metadata
            processing_time = fragment.metadata.get("processing_time_ms", 0)
            latencies.append(processing_time)

            if len(latencies) >= 10:  # Test 10 fragments
                break

        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)

        # Average should be well under 50ms
        assert avg_latency < 25.0, f"Average latency {avg_latency:.2f}ms too high"

        # Max should be under 50ms
        assert (
            max_latency < 50.0
        ), f"Max latency {max_latency:.2f}ms exceeds 50ms requirement"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_stream_performance(self, sse_streaming_service):
        """Test performance with multiple concurrent streams."""
        user = MockUser()
        num_streams = 10

        # Create multiple streams concurrently
        start_time = time.perf_counter()

        tasks = []
        for i in range(num_streams):
            task = sse_streaming_service.create_sse_stream(
                camera_id=f"camera_{i}", user=user, stream_type="raw", quality="medium"
            )
            tasks.append(task)

        streams = await asyncio.gather(*tasks)

        creation_time = (time.perf_counter() - start_time) * 1000

        # All streams should be created quickly
        assert len(streams) == num_streams
        assert (
            creation_time < 100.0
        ), f"Stream creation took {creation_time:.2f}ms for {num_streams} streams"

        # Check metrics
        stats = await sse_streaming_service.get_connection_stats()
        assert stats["active_connections"] == num_streams
        assert stats["peak_concurrent_connections"] == num_streams


class TestSSEIntegration:
    """Integration tests for SSE streaming service."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_integration_with_base_streaming_service(self, sse_streaming_service):
        """Test integration with base streaming service."""
        # Test that health check properly integrates
        health = await sse_streaming_service.health_check()

        assert health["healthy"] is True
        assert "base_service_health" in health
        assert health["base_service_health"]["healthy"] is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_integration_with_redis_queue_manager(self, sse_streaming_service):
        """Test integration with Redis queue manager."""
        # Health check should include Redis status
        health = await sse_streaming_service.health_check()

        # Base service health should include Redis information
        base_health = health["base_service_health"]
        assert base_health is not None

        # Redis manager should be accessible through base service
        assert sse_streaming_service.redis_manager is not None


class TestSSEErrorHandling:
    """Error handling and resilience tests."""

    @pytest.mark.asyncio
    async def test_fragment_creation_error_handling(self, sse_streaming_service):
        """Test handling of fragment creation errors."""
        # Create a processed frame with None image
        bad_frame = ProcessedFrame(
            frame_id="bad_frame",
            camera_id="test_camera",
            timestamp=time.time(),
            original_image=None,  # This should cause an error
            quality_score=0.0,
        )

        fragment = await sse_streaming_service._create_mp4_fragment(
            bad_frame, 0, "medium"
        )

        # Should return None for bad frame
        assert fragment is None

    @pytest.mark.asyncio
    async def test_health_check_error_handling(
        self, mock_base_streaming_service, mock_redis_manager, sse_config
    ):
        """Test health check error handling."""
        # Make base service health check fail
        mock_base_streaming_service.health_check.side_effect = Exception(
            "Base service error"
        )

        sse_service = SSEStreamingService(
            base_streaming_service=mock_base_streaming_service,
            redis_manager=mock_redis_manager,
            config=sse_config,
        )

        health = await sse_service.health_check()

        assert health["healthy"] is False
        assert "error" in health
        assert "Base service error" in health["error"]

    @pytest.mark.asyncio
    async def test_stream_queue_full_handling(self, sse_streaming_service):
        """Test handling of full stream queues."""
        user = MockUser()
        camera_id = "test_camera_001"

        # Create stream
        stream = await sse_streaming_service.create_sse_stream(
            camera_id=camera_id, user=user, stream_type="raw", quality="medium"
        )

        # Fill the queue manually
        queue = sse_streaming_service.stream_queues[stream.stream_id]

        # Fill beyond capacity (queue maxsize is 100)
        for i in range(105):
            try:
                queue.put_nowait({"type": "test", "data": i})
            except asyncio.QueueFull:
                break

        # Queue should be full
        assert queue.full()

        # Start fragment streaming (should handle full queue gracefully)
        task = asyncio.create_task(
            sse_streaming_service._stream_fragments_to_queue(stream)
        )

        # Wait briefly
        await asyncio.sleep(0.05)

        # Cancel task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Service should still be healthy despite queue issues
        health = await sse_streaming_service.health_check()
        assert health["healthy"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
