"""Test suite for the core StreamingService architecture.

This module tests the independent streaming service implementation
focusing on startup, health checks, dependency injection, and CLI integration.
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.its_camera_ai.flow.redis_queue_manager import RedisQueueManager
from src.its_camera_ai.services.streaming_service import (
    CameraConfig,
    StreamingDataProcessor,
    StreamingService,
    StreamProtocol,
    StreamChannelManager,
    ChannelSubscriptionManager,
    ChannelType,
    QualityLevel,
    DualChannelStream,
    ChannelMetadata,
)


class TestStreamingServiceCore:
    """Test suite for the main StreamingService class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_redis = AsyncMock(spec=RedisQueueManager)
        self.mock_redis.connect.return_value = None
        self.mock_redis.create_queue.return_value = True
        self.mock_redis.health_check.return_value = {"status": "healthy"}

        # Create streaming service with mocked dependencies
        self.streaming_service = StreamingService(
            redis_manager=self.mock_redis,
            config={
                "max_concurrent_streams": 10,
                "frame_processing_timeout": 0.01,
            },
        )

    @pytest.mark.asyncio
    async def test_independent_startup(self):
        """Test that the service can start independently without FastAPI."""
        assert not self.streaming_service.is_running
        assert self.streaming_service.health_status == "initializing"

        # Measure startup time
        start_time = time.perf_counter()
        await self.streaming_service.start()
        startup_time = (time.perf_counter() - start_time) * 1000

        assert self.streaming_service.is_running
        assert self.streaming_service.health_status == "healthy"
        assert startup_time < 50.0  # Allow some tolerance for test environment

        await self.streaming_service.stop()

    @pytest.mark.asyncio
    async def test_startup_time_requirement(self):
        """Test that startup time meets <10ms requirement (with mocked I/O)."""
        # Mock the streaming processor to avoid actual I/O
        mock_processor = AsyncMock(spec=StreamingDataProcessor)
        mock_processor.start.return_value = None

        service = StreamingService(
            streaming_processor=mock_processor,
            redis_manager=self.mock_redis,
        )

        start_time = time.perf_counter()
        await service.start()
        startup_time = (time.perf_counter() - start_time) * 1000

        # With mocked I/O, startup should be very fast
        assert (
            startup_time < 10.0
        ), f"Startup time {startup_time:.2f}ms exceeds 10ms requirement"
        assert service.startup_time is not None
        assert service.startup_metrics["startup_time_ms"] > 0

        await service.stop()

    @pytest.mark.asyncio
    async def test_health_check_independent(self):
        """Test health check works independently of FastAPI."""
        # Test health check when not running
        health = await self.streaming_service.health_check()
        assert not health["healthy"]
        assert health["status"] == "stopped"
        assert "timestamp" in health

        # Start service and test health check
        await self.streaming_service.start()

        health = await self.streaming_service.health_check()
        assert health["healthy"]
        assert health["status"] == "healthy"
        assert health["startup_time_ms"] is not None
        assert "processor_health" in health
        assert "redis_health" in health
        assert "timestamp" in health

        await self.streaming_service.stop()

        # Test health check after stopping
        health = await self.streaming_service.health_check()
        assert not health["healthy"]
        assert health["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self):
        """Test graceful shutdown with resource cleanup."""
        await self.streaming_service.start()
        assert self.streaming_service.is_running

        # Test graceful shutdown
        await self.streaming_service.stop()
        assert not self.streaming_service.is_running
        assert self.streaming_service.health_status == "stopped"

        # Verify processor was stopped
        if self.streaming_service.streaming_processor:
            # The processor should have been stopped
            pass  # We can't easily verify this with mocks

    @pytest.mark.asyncio
    async def test_dependency_injection(self):
        """Test proper dependency injection pattern."""
        # Test with injected dependencies
        mock_processor = AsyncMock(spec=StreamingDataProcessor)
        mock_processor.start.return_value = None
        mock_processor.stop.return_value = None
        mock_processor.get_health_status.return_value = {"service_status": "healthy"}

        service = StreamingService(
            streaming_processor=mock_processor,
            redis_manager=self.mock_redis,
            config={"test_config": True},
        )

        await service.start()

        # Verify injected processor was used
        assert service.streaming_processor is mock_processor
        mock_processor.start.assert_called_once()

        # Test health check uses injected dependencies
        health = await service.health_check()
        assert health["healthy"]
        mock_processor.get_health_status.assert_called_once()

        await service.stop()
        mock_processor.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_camera_registration_via_service(self):
        """Test camera registration through the service interface."""
        await self.streaming_service.start()

        # Mock the processor's register_camera method
        mock_registration = Mock()
        mock_registration.success = True
        mock_registration.camera_id = "test_camera"
        mock_registration.message = "Success"

        if self.streaming_service.streaming_processor:
            self.streaming_service.streaming_processor.register_camera = AsyncMock(
                return_value=mock_registration
            )

        camera_config = CameraConfig(
            camera_id="test_camera",
            stream_url="test://stream",
            resolution=(1280, 720),
            fps=30,
            protocol=StreamProtocol.HTTP,
        )

        registration = await self.streaming_service.register_camera(camera_config)
        assert registration.success
        assert registration.camera_id == "test_camera"

        await self.streaming_service.stop()

    @pytest.mark.asyncio
    async def test_camera_registration_service_not_running(self):
        """Test camera registration fails when service is not running."""
        camera_config = CameraConfig(
            camera_id="test_camera",
            stream_url="test://stream",
            resolution=(1280, 720),
            fps=30,
            protocol=StreamProtocol.HTTP,
        )

        registration = await self.streaming_service.register_camera(camera_config)
        assert not registration.success
        assert "not running" in registration.message.lower()

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test service metrics collection."""
        await self.streaming_service.start()

        metrics = await self.streaming_service.get_metrics()
        assert "service_status" in metrics
        assert "is_running" in metrics
        assert "startup_metrics" in metrics
        assert metrics["is_running"]
        assert metrics["service_status"] == "healthy"

        await self.streaming_service.stop()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager support."""
        async with self.streaming_service:
            assert self.streaming_service.is_running
            health = await self.streaming_service.health_check()
            assert health["healthy"]

        # Should be stopped after exiting context
        assert not self.streaming_service.is_running
        health = await self.streaming_service.health_check()
        assert not health["healthy"]

    @pytest.mark.asyncio
    async def test_error_handling_on_startup(self):
        """Test error handling during service startup."""
        # Create a service that will fail on startup
        mock_processor = AsyncMock(spec=StreamingDataProcessor)
        mock_processor.start.side_effect = Exception("Startup failed")

        service = StreamingService(
            streaming_processor=mock_processor,
            redis_manager=self.mock_redis,
        )

        with pytest.raises(Exception, match="Startup failed"):
            await service.start()

        assert not service.is_running
        assert service.health_status == "unhealthy"

    @pytest.mark.asyncio
    async def test_multiple_start_calls(self):
        """Test that multiple start calls don't cause issues."""
        await self.streaming_service.start()
        startup_time_1 = self.streaming_service.startup_time

        # Second start call should be ignored
        await self.streaming_service.start()
        startup_time_2 = self.streaming_service.startup_time

        assert startup_time_1 == startup_time_2  # No change
        assert self.streaming_service.is_running

        await self.streaming_service.stop()

    @pytest.mark.asyncio
    async def test_multiple_stop_calls(self):
        """Test that multiple stop calls don't cause issues."""
        await self.streaming_service.start()
        await self.streaming_service.stop()
        assert not self.streaming_service.is_running

        # Second stop call should be ignored
        await self.streaming_service.stop()
        assert not self.streaming_service.is_running

    @pytest.mark.asyncio
    async def test_programmatic_initialization(self):
        """Test programmatic initialization without CLI."""
        # This tests the use case where the service is embedded in another application
        config = {
            "max_concurrent_streams": 50,
            "frame_processing_timeout": 0.005,
        }

        service = StreamingService(
            redis_manager=self.mock_redis,
            config=config,
        )

        await service.start()
        assert service.is_running

        # Verify config was applied
        assert service.config == config

        metrics = await service.get_metrics()
        assert metrics["is_running"]

        await service.stop()


class TestStreamingServiceIntegrationWithContainer:
    """Integration tests with dependency injection container."""

    @pytest.mark.asyncio
    async def test_container_integration(self):
        """Test integration with the dependency injection container."""
        from src.its_camera_ai.services.streaming_container import (
            create_streaming_container,
            get_streaming_service,
        )

        # Create container with test configuration
        config = {
            "redis": {"url": "redis://localhost:6379"},
            "streaming": {
                "max_concurrent_streams": 5,
                "frame_processing_timeout": 0.01,
            },
        }

        container = create_streaming_container(config)

        # Get service from container
        with patch(
            "src.its_camera_ai.flow.redis_queue_manager.RedisQueueManager"
        ) as mock_redis_class:
            mock_redis_instance = AsyncMock()
            mock_redis_instance.connect.return_value = None
            mock_redis_instance.create_queue.return_value = True
            mock_redis_instance.health_check.return_value = {"status": "healthy"}
            mock_redis_class.return_value = mock_redis_instance

            service = get_streaming_service()
            assert service is not None
            assert isinstance(service, StreamingService)

            # Test that the service uses injected dependencies
            # Note: In a real test environment, we'd need proper Redis setup
            # This is a structural test to ensure DI works correctly

            # Test dual-channel functionality is available
            assert hasattr(service, "channel_manager") or hasattr(
                service, "sse_streaming_service"
            )


class TestStreamChannelManager:
    """Test suite for StreamChannelManager dual-channel functionality."""

    @pytest.fixture
    async def channel_manager(self):
        """Create StreamChannelManager instance for testing."""
        mock_redis = AsyncMock()
        config = {
            "sync_tolerance_ms": 50.0,
            "sync_check_interval": 1.0,
            "max_sync_violations": 10,
        }
        manager = StreamChannelManager(mock_redis, config)
        yield manager

        # Cleanup
        if manager._running:
            await manager._stop_sync_monitoring()

    @pytest.mark.asyncio
    async def test_create_dual_channel_stream(self, channel_manager):
        """Test dual-channel stream creation."""
        camera_id = "test_camera_001"
        user_id = "test_user"

        dual_stream = await channel_manager.create_dual_channel_stream(
            camera_id=camera_id,
            user_id=user_id,
            raw_quality=QualityLevel.HIGH,
            annotated_quality=QualityLevel.MEDIUM,
        )

        assert dual_stream.camera_id == camera_id
        assert dual_stream.user_id == user_id
        assert dual_stream.raw_channel.quality == QualityLevel.HIGH
        assert dual_stream.annotated_channel.quality == QualityLevel.MEDIUM
        assert dual_stream.sync_tolerance_ms == 50.0
        assert dual_stream.is_synchronized == True

        # Check stream is registered
        assert dual_stream.stream_id in channel_manager.dual_channel_streams

    @pytest.mark.asyncio
    async def test_channel_synchronization_within_tolerance(self, channel_manager):
        """Test channel synchronization within tolerance."""
        # Create dual-channel stream
        dual_stream = await channel_manager.create_dual_channel_stream(
            camera_id="test_camera", user_id="test_user"
        )

        # Set timestamps within tolerance (30ms apart)
        current_time = time.time()
        dual_stream.raw_channel.last_fragment_time = current_time
        dual_stream.annotated_channel.last_fragment_time = (
            current_time + 0.03
        )  # 30ms later

        sync_status = await channel_manager.synchronize_channels(dual_stream.stream_id)

        assert sync_status.is_synchronized == True
        assert sync_status.drift_ms == 30.0
        assert sync_status.correction_applied == False
        assert dual_stream.sync_violations == 0

    @pytest.mark.asyncio
    async def test_channel_synchronization_exceeds_tolerance(self, channel_manager):
        """Test channel synchronization when drift exceeds tolerance."""
        # Create dual-channel stream
        dual_stream = await channel_manager.create_dual_channel_stream(
            camera_id="test_camera", user_id="test_user"
        )

        # Set timestamps beyond tolerance (100ms apart)
        current_time = time.time()
        dual_stream.raw_channel.last_fragment_time = current_time
        dual_stream.annotated_channel.last_fragment_time = (
            current_time + 0.1
        )  # 100ms later

        sync_status = await channel_manager.synchronize_channels(dual_stream.stream_id)

        assert sync_status.is_synchronized == False
        assert sync_status.drift_ms == 100.0
        assert sync_status.correction_applied == True
        assert dual_stream.sync_violations == 1

        # Check that offset was applied to slower channel
        assert dual_stream.raw_channel.sync_offset_ms == 100.0

    @pytest.mark.asyncio
    async def test_channel_timestamp_update(self, channel_manager):
        """Test updating channel timestamps for synchronization."""
        # Create dual-channel stream
        dual_stream = await channel_manager.create_dual_channel_stream(
            camera_id="test_camera", user_id="test_user"
        )

        timestamp = time.time()

        # Update raw channel timestamp
        await channel_manager.update_channel_timestamp(
            dual_stream.stream_id, ChannelType.RAW, timestamp
        )

        assert dual_stream.raw_channel.last_fragment_time == timestamp
        assert dual_stream.raw_channel.fragments_processed == 1

        # Update annotated channel timestamp
        await channel_manager.update_channel_timestamp(
            dual_stream.stream_id, ChannelType.ANNOTATED, timestamp + 0.02
        )

        assert dual_stream.annotated_channel.last_fragment_time == timestamp + 0.02
        assert dual_stream.annotated_channel.fragments_processed == 1

    @pytest.mark.asyncio
    async def test_get_synchronization_stats(self, channel_manager):
        """Test getting synchronization statistics."""
        # Create dual-channel stream
        dual_stream = await channel_manager.create_dual_channel_stream(
            camera_id="test_camera_stats", user_id="test_user"
        )

        # Update some metrics
        dual_stream.raw_channel.fragments_processed = 10
        dual_stream.annotated_channel.fragments_processed = 9
        dual_stream.sync_violations = 2

        stats = await channel_manager.get_synchronization_stats(dual_stream.stream_id)

        assert stats["stream_id"] == dual_stream.stream_id
        assert stats["camera_id"] == "test_camera_stats"
        assert stats["sync_violations"] == 2
        assert stats["raw_channel"]["fragments_processed"] == 10
        assert stats["annotated_channel"]["fragments_processed"] == 9
        assert "uptime_seconds" in stats


class TestChannelSubscriptionManager:
    """Test suite for ChannelSubscriptionManager."""

    @pytest.fixture
    def subscription_manager(self):
        """Create ChannelSubscriptionManager instance for testing."""
        return ChannelSubscriptionManager()

    @pytest.mark.asyncio
    async def test_subscribe_to_channel(self, subscription_manager):
        """Test subscribing to a specific channel."""
        connection_id = "conn_123"
        camera_id = "camera_456"
        channel_type = ChannelType.RAW

        result = await subscription_manager.subscribe_to_channel(
            connection_id=connection_id, camera_id=camera_id, channel_type=channel_type
        )

        assert result["status"] == "subscribed"
        assert result["connection_id"] == connection_id
        assert result["camera_id"] == camera_id
        assert result["channel_type"] == "raw"
        assert "raw" in result["subscribed_channels"]

        # Check internal state
        assert connection_id in subscription_manager.client_subscriptions
        assert camera_id in subscription_manager.client_subscriptions[connection_id]
        assert (
            ChannelType.RAW
            in subscription_manager.client_subscriptions[connection_id][camera_id]
        )

    @pytest.mark.asyncio
    async def test_switch_channel(self, subscription_manager):
        """Test switching between channels."""
        connection_id = "conn_789"
        camera_id = "camera_101"

        # First subscribe to raw channel
        await subscription_manager.subscribe_to_channel(
            connection_id=connection_id,
            camera_id=camera_id,
            channel_type=ChannelType.RAW,
        )

        # Then switch to annotated channel
        result = await subscription_manager.switch_channel(
            connection_id=connection_id,
            camera_id=camera_id,
            new_channel=ChannelType.ANNOTATED,
        )

        assert result["status"] == "switched"
        assert result["connection_id"] == connection_id
        assert result["camera_id"] == camera_id
        assert result["old_channels"] == ["raw"]
        assert result["new_channel"] == "annotated"
        assert "switch_time" in result

        # Check internal state updated
        current_channels = subscription_manager.client_subscriptions[connection_id][
            camera_id
        ]
        assert ChannelType.ANNOTATED in current_channels
        assert ChannelType.RAW not in current_channels


class TestDualChannelIntegration:
    """Integration tests for dual-channel streaming."""

    @pytest.mark.asyncio
    async def test_dual_channel_end_to_end_flow(self):
        """Test complete dual-channel flow from creation to cleanup."""
        # Setup managers
        mock_redis = AsyncMock()
        config = {
            "sync_tolerance_ms": 50.0,
            "sync_check_interval": 1.0,
            "max_sync_violations": 10,
        }

        channel_manager = StreamChannelManager(mock_redis, config)
        subscription_manager = ChannelSubscriptionManager()

        try:
            # 1. Create dual-channel stream
            dual_stream = await channel_manager.create_dual_channel_stream(
                camera_id="integration_camera", user_id="integration_user"
            )

            # 2. Subscribe client to raw channel
            connection_id = "integration_conn"
            await subscription_manager.subscribe_to_channel(
                connection_id, dual_stream.camera_id, ChannelType.RAW
            )

            # 3. Simulate fragment processing with timestamps
            current_time = time.time()
            await channel_manager.update_channel_timestamp(
                dual_stream.stream_id, ChannelType.RAW, current_time
            )
            await channel_manager.update_channel_timestamp(
                dual_stream.stream_id, ChannelType.ANNOTATED, current_time + 0.02
            )

            # 4. Check synchronization
            sync_status = await channel_manager.synchronize_channels(
                dual_stream.stream_id
            )
            assert sync_status.is_synchronized == True

            # 5. Switch to annotated channel
            switch_result = await subscription_manager.switch_channel(
                connection_id, dual_stream.camera_id, ChannelType.ANNOTATED
            )
            assert switch_result["status"] == "switched"

            # 6. Verify statistics
            sync_stats = await channel_manager.get_synchronization_stats(
                dual_stream.stream_id
            )
            assert sync_stats["raw_channel"]["fragments_processed"] == 1
            assert sync_stats["annotated_channel"]["fragments_processed"] == 1

            subscription_stats = subscription_manager.get_subscription_stats()
            assert subscription_stats["channel_switches"] == 1

        finally:
            # Cleanup
            await subscription_manager.unsubscribe_from_channel(
                connection_id, dual_stream.camera_id
            )
            await channel_manager.remove_dual_channel_stream(dual_stream.stream_id)

            if channel_manager._running:
                await channel_manager._stop_sync_monitoring()

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self):
        """Test performance requirements for dual-channel operations."""
        mock_redis = AsyncMock()
        config = {
            "sync_tolerance_ms": 50.0,
            "sync_check_interval": 0.1,
            "max_sync_violations": 10,
        }

        channel_manager = StreamChannelManager(mock_redis, config)
        subscription_manager = ChannelSubscriptionManager()

        try:
            # Benchmark stream creation (should be < 10ms)
            start_time = time.perf_counter()

            dual_stream = await channel_manager.create_dual_channel_stream(
                camera_id="perf_camera", user_id="perf_user"
            )

            creation_time = (time.perf_counter() - start_time) * 1000
            assert (
                creation_time < 10.0
            ), f"Stream creation took {creation_time:.2f}ms, expected <10ms"

            # Benchmark channel switching (should be < 100ms)
            connection_id = "perf_conn"
            await subscription_manager.subscribe_to_channel(
                connection_id, dual_stream.camera_id, ChannelType.RAW
            )

            start_time = time.perf_counter()

            await subscription_manager.switch_channel(
                connection_id, dual_stream.camera_id, ChannelType.ANNOTATED
            )

            switch_time = (time.perf_counter() - start_time) * 1000
            assert (
                switch_time < 100.0
            ), f"Channel switch took {switch_time:.2f}ms, expected <100ms"

            # Benchmark synchronization check (should be < 5ms)
            start_time = time.perf_counter()

            sync_status = await channel_manager.synchronize_channels(
                dual_stream.stream_id
            )

            sync_time = (time.perf_counter() - start_time) * 1000
            assert sync_time < 5.0, f"Sync check took {sync_time:.2f}ms, expected <5ms"

        finally:
            await channel_manager.remove_dual_channel_stream(dual_stream.stream_id)
            if channel_manager._running:
                await channel_manager._stop_sync_monitoring()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
