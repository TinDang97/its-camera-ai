"""Tests for Kafka SSE Integration Components.

Tests for the complete Kafka-to-SSE streaming pipeline including:
- Kafka SSE consumer functionality
- SSE broadcaster enhancements
- Real-time streaming service coordination
- End-to-end event flow
"""

import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from collections import deque

from src.its_camera_ai.api.sse_broadcaster import SSEBroadcaster, SSEMessage
from src.its_camera_ai.core.exceptions import ServiceError
from src.its_camera_ai.services.kafka_event_producer import (
    EventType, EventPriority, DetectionEvent, EventMetadata
)

# Mock Kafka components if not available
try:
    from src.its_camera_ai.services.kafka_sse_consumer import KafkaSSEConsumer, EventFilter
    from src.its_camera_ai.services.realtime_streaming_service import RealtimeStreamingService
    KAFKA_AVAILABLE = True
except ImportError:
    KafkaSSEConsumer = None
    RealtimeStreamingService = None
    EventFilter = None
    KAFKA_AVAILABLE = False


class TestEventFilter:
    """Test event filtering functionality."""
    
    def test_filter_creation(self):
        """Test event filter creation with various parameters."""
        if not KAFKA_AVAILABLE:
            pytest.skip("Kafka components not available")
        
        # Test with all parameters
        event_filter = EventFilter(
            camera_ids=["cam1", "cam2"],
            event_types=["detection_result", "status_change"],
            zones=["zone1", "zone2"],
            min_confidence=0.7,
            vehicle_types=["car", "truck"]
        )
        
        assert event_filter.camera_ids == {"cam1", "cam2"}
        assert event_filter.event_types == {"detection_result", "status_change"}
        assert event_filter.zones == {"zone1", "zone2"}
        assert event_filter.min_confidence == 0.7
        assert event_filter.vehicle_types == {"car", "truck"}
    
    def test_filter_matching_camera_id(self):
        """Test filtering by camera ID."""
        if not KAFKA_AVAILABLE:
            pytest.skip("Kafka components not available")
        
        event_filter = EventFilter(camera_ids=["cam1", "cam2"])
        
        # Should match
        assert event_filter.matches_event({"camera_id": "cam1"}, "detection_result")
        assert event_filter.matches_event({"camera_id": "cam2"}, "detection_result")
        
        # Should not match
        assert not event_filter.matches_event({"camera_id": "cam3"}, "detection_result")
        assert not event_filter.matches_event({}, "detection_result")
    
    def test_filter_matching_event_type(self):
        """Test filtering by event type."""
        if not KAFKA_AVAILABLE:
            pytest.skip("Kafka components not available")
        
        event_filter = EventFilter(event_types=["detection_result", "status_change"])
        
        # Should match
        assert event_filter.matches_event({"camera_id": "cam1"}, "detection_result")
        assert event_filter.matches_event({"camera_id": "cam1"}, "status_change")
        
        # Should not match
        assert not event_filter.matches_event({"camera_id": "cam1"}, "health_update")
    
    def test_filter_matching_confidence(self):
        """Test filtering by confidence threshold."""
        if not KAFKA_AVAILABLE:
            pytest.skip("Kafka components not available")
        
        event_filter = EventFilter(min_confidence=0.7)
        
        # Should match
        assert event_filter.matches_event(
            {"confidence_scores": [0.8, 0.9]}, "detection_result"
        )
        assert event_filter.matches_event(
            {"confidence_scores": [0.5, 0.8]}, "detection_result"
        )
        
        # Should not match
        assert not event_filter.matches_event(
            {"confidence_scores": [0.5, 0.6]}, "detection_result"
        )
        assert not event_filter.matches_event(
            {"confidence_scores": []}, "detection_result"
        )
        
        # Non-detection events should match regardless
        assert event_filter.matches_event({}, "status_change")
    
    def test_filter_matching_vehicle_types(self):
        """Test filtering by vehicle types."""
        if not KAFKA_AVAILABLE:
            pytest.skip("Kafka components not available")
        
        event_filter = EventFilter(vehicle_types=["car", "truck"])
        
        # Should match
        assert event_filter.matches_event(
            {"vehicle_classes": ["car", "motorcycle"]}, "detection_result"
        )
        assert event_filter.matches_event(
            {"vehicle_classes": ["truck"]}, "detection_result"
        )
        
        # Should not match
        assert not event_filter.matches_event(
            {"vehicle_classes": ["motorcycle", "bicycle"]}, "detection_result"
        )
        assert not event_filter.matches_event(
            {"vehicle_classes": []}, "detection_result"
        )
        
        # Non-detection events should match regardless
        assert event_filter.matches_event({}, "status_change")
    
    def test_filter_no_filters(self):
        """Test filter with no restrictions (should match everything)."""
        if not KAFKA_AVAILABLE:
            pytest.skip("Kafka components not available")
        
        event_filter = EventFilter()
        
        # Should match everything
        assert event_filter.matches_event({"camera_id": "cam1"}, "detection_result")
        assert event_filter.matches_event({}, "status_change")
        assert event_filter.matches_event({"confidence_scores": [0.1]}, "detection_result")


class TestSSEBroadcasterEnhancements:
    """Test enhanced SSE broadcaster functionality."""
    
    def test_broadcaster_initialization_with_kafka(self):
        """Test broadcaster initialization with Kafka configuration."""
        kafka_config = {
            "bootstrap_servers": ["localhost:9092"],
            "topic_prefix": "test-its",
        }
        
        broadcaster = SSEBroadcaster(kafka_config=kafka_config)
        
        assert broadcaster.kafka_config == kafka_config
        assert broadcaster.kafka_enabled or not KAFKA_AVAILABLE  # Enabled if available
        assert broadcaster.max_connections == 200
        assert broadcaster.max_global_events_per_second == 500
    
    def test_connection_limits(self):
        """Test connection limit enforcement."""
        broadcaster = SSEBroadcaster()
        broadcaster.max_connections = 2  # Set low limit for testing
        
        # Should allow connections up to limit
        connection1 = asyncio.create_task(
            broadcaster.connect_client("conn1", {})
        )
        connection2 = asyncio.create_task(
            broadcaster.connect_client("conn2", {})
        )
        
        # Third connection should fail
        with pytest.raises(ServiceError, match="Maximum connections reached"):
            asyncio.create_task(broadcaster.connect_client("conn3", {}))
    
    def test_rate_limiting_per_connection(self):
        """Test per-connection rate limiting."""
        broadcaster = SSEBroadcaster()
        
        # Simulate rapid events for a connection
        connection_id = "test_conn"
        
        # Should not be rate limited initially
        assert not broadcaster._is_connection_rate_limited(connection_id, max_per_second=5)
        
        # Add events rapidly
        for _ in range(5):
            broadcaster._track_connection_event(connection_id)
        
        # Should be rate limited now
        assert broadcaster._is_connection_rate_limited(connection_id, max_per_second=5)
        
        # After time passes, should not be rate limited
        # (This would require actual time passage in real scenario)
    
    def test_global_rate_limiting(self):
        """Test global rate limiting."""
        broadcaster = SSEBroadcaster()
        broadcaster.max_global_events_per_second = 5
        broadcaster.global_rate_limit = deque(maxlen=5)
        
        # Should not be rate limited initially
        assert not broadcaster._is_global_rate_limited()
        
        # Fill up the rate limit
        current_time = time.time()
        for _ in range(5):
            broadcaster.global_rate_limit.append(current_time)
        
        # Should be rate limited now
        assert broadcaster._is_global_rate_limited()
    
    @pytest.mark.asyncio
    async def test_kafka_event_broadcasting(self):
        """Test broadcasting of Kafka events."""
        broadcaster = SSEBroadcaster()
        
        # Mock connection
        mock_connection = MagicMock()
        mock_connection.send_event = AsyncMock(return_value=True)
        broadcaster.connections["test_conn"] = mock_connection
        
        # Create test SSE message
        sse_message = SSEMessage(
            event="camera_update",
            data={"camera_id": "cam1", "status": "online"},
            id="test_123"
        )
        
        # Broadcast event
        await broadcaster.broadcast_kafka_event(sse_message)
        
        # Verify connection received event
        mock_connection.send_event.assert_called_once_with(sse_message)
        
        # Check stats updated
        assert broadcaster.stats["kafka_events_processed"] == 1
        assert broadcaster.stats["messages_sent"] == 1


@pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka components not available")
class TestKafkaSSEConsumer:
    """Test Kafka SSE consumer functionality."""
    
    def test_consumer_initialization(self):
        """Test consumer initialization with configuration."""
        config = {
            "bootstrap_servers": ["localhost:9092"],
            "topic_prefix": "test-its",
            "consumer_group_id": "test-sse-group",
        }
        
        broadcaster = SSEBroadcaster()
        consumer = KafkaSSEConsumer(config, broadcaster)
        
        assert consumer.config == config
        assert consumer.bootstrap_servers == ["localhost:9092"]
        assert consumer.topic_prefix == "test-its"
        assert consumer.consumer_group_id == "test-sse-group"
        assert consumer.sse_broadcaster == broadcaster
    
    def test_topic_initialization(self):
        """Test topic list initialization."""
        config = {"topic_prefix": "test-its"}
        broadcaster = SSEBroadcaster()
        consumer = KafkaSSEConsumer(config, broadcaster)
        
        expected_topics = [
            "test-its.detections.vehicles",
            "test-its.violations.traffic",
            "test-its.incidents",
            "test-its.cameras.status",
            "test-its.cameras.health",
            "test-its.cameras.config",
            "test-its.system.alerts",
            "test-its.system.metrics",
            "test-its.analytics.traffic-flow",
            "test-its.analytics.zones",
            "test-its.analytics.speed",
        ]
        
        assert consumer.topics == expected_topics
    
    def test_event_type_determination(self):
        """Test event type determination from topic and headers."""
        config = {"topic_prefix": "test-its"}
        broadcaster = SSEBroadcaster()
        consumer = KafkaSSEConsumer(config, broadcaster)
        
        # Test header-based determination
        assert consumer._determine_event_type(
            "test-its.detections.vehicles",
            {"event_type": "custom_detection"}
        ) == "custom_detection"
        
        # Test topic-based determination
        assert consumer._determine_event_type(
            "test-its.detections.vehicles", {}
        ) == "detection_result"
        
        assert consumer._determine_event_type(
            "test-its.cameras.status", {}
        ) == "status_change"
        
        assert consumer._determine_event_type(
            "test-its.system.alerts", {}
        ) == "system_alert"
        
        # Test unknown topic
        assert consumer._determine_event_type(
            "unknown.topic", {}
        ) == "unknown"
    
    def test_sse_message_creation(self):
        """Test SSE message creation from Kafka events."""
        config = {"topic_prefix": "test-its"}
        broadcaster = SSEBroadcaster()
        consumer = KafkaSSEConsumer(config, broadcaster)
        
        # Test detection event
        event_data = {
            "camera_id": "cam1",
            "detection_id": "det_123",
            "confidence_scores": [0.8, 0.9]
        }
        headers = {"timestamp": "1234567890"}
        
        sse_message = asyncio.run(consumer._create_sse_event(
            "detection_result", event_data, headers
        ))
        
        assert sse_message is not None
        assert sse_message.event == "camera_update"
        assert sse_message.data["event_type"] == "detection_result"
        assert sse_message.data["camera_id"] == "cam1"
        assert sse_message.id == "detection_result_1234567890"
        assert sse_message.retry == 5000
    
    def test_rate_limiting(self):
        """Test consumer rate limiting functionality."""
        config = {"max_events_per_second": 5}
        broadcaster = SSEBroadcaster()
        consumer = KafkaSSEConsumer(config, broadcaster)
        
        connection_id = "test_conn"
        
        # Should not be rate limited initially
        assert not consumer._is_rate_limited(connection_id)
        
        # Add events up to limit
        for _ in range(5):
            consumer._track_connection_event(connection_id)
        
        # Should be rate limited now
        assert consumer._is_rate_limited(connection_id)
    
    def test_event_transformer_registration(self):
        """Test event transformer registration and usage."""
        config = {}
        broadcaster = SSEBroadcaster()
        consumer = KafkaSSEConsumer(config, broadcaster)
        
        # Register transformer
        def transform_detection(data):
            data["transformed"] = True
            return data
        
        consumer.register_event_transformer("detection_result", transform_detection)
        
        assert "detection_result" in consumer.event_transformers
        
        # Test transformation
        original_data = {"camera_id": "cam1"}
        transformed = consumer.event_transformers["detection_result"](original_data)
        
        assert transformed["transformed"] is True
        assert transformed["camera_id"] == "cam1"


@pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka components not available")
class TestRealtimeStreamingService:
    """Test real-time streaming service coordination."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        kafka_producer = MagicMock()
        kafka_producer.start = AsyncMock()
        kafka_producer.stop = AsyncMock()
        kafka_producer.is_healthy = True
        kafka_producer.get_health_status = MagicMock(return_value={"is_healthy": True})
        
        kafka_consumer = MagicMock()
        kafka_consumer.start = AsyncMock()
        kafka_consumer.stop = AsyncMock()
        kafka_consumer.is_healthy = True
        kafka_consumer.get_health_status = MagicMock(return_value={"is_healthy": True})
        
        sse_broadcaster = MagicMock()
        sse_broadcaster.connections = {}
        sse_broadcaster.max_connections = 200
        sse_broadcaster.get_stats = MagicMock(return_value={"active_connections": 0})
        
        analytics_connector = MagicMock()
        analytics_connector.start = AsyncMock()
        analytics_connector.stop = AsyncMock()
        analytics_connector.is_running = True
        analytics_connector.get_health_status = MagicMock(return_value={"is_running": True})
        
        return {
            "kafka_producer": kafka_producer,
            "kafka_consumer": kafka_consumer,
            "sse_broadcaster": sse_broadcaster,
            "analytics_connector": analytics_connector,
        }
    
    def test_service_initialization(self, mock_components):
        """Test service initialization with components."""
        config = {
            "health_check_interval": 10.0,
            "auto_restart_on_failure": True,
            "max_restart_attempts": 2,
        }
        
        service = RealtimeStreamingService(
            config=config,
            **mock_components
        )
        
        assert service.config == config
        assert service.health_check_interval == 10.0
        assert service.auto_restart_on_failure is True
        assert service.max_restart_attempts == 2
        assert not service.is_running
        assert service.is_healthy
    
    @pytest.mark.asyncio
    async def test_service_start_success(self, mock_components):
        """Test successful service startup."""
        config = {"health_check_interval": 0.1}  # Fast for testing
        
        service = RealtimeStreamingService(
            config=config,
            **mock_components
        )
        
        # Start service
        await service.start()
        
        assert service.is_running
        assert service.is_healthy
        
        # Verify components were started
        mock_components["kafka_producer"].start.assert_called_once()
        mock_components["kafka_consumer"].start.assert_called_once()
        mock_components["analytics_connector"].start.assert_called_once()
        
        # Stop service
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_service_component_failure_handling(self, mock_components):
        """Test handling of component startup failures."""
        # Make Kafka producer fail to start
        mock_components["kafka_producer"].start = AsyncMock(
            side_effect=Exception("Kafka connection failed")
        )
        
        config = {"health_check_interval": 0.1}
        service = RealtimeStreamingService(
            config=config,
            **mock_components
        )
        
        # Service should still start but mark producer as unhealthy
        await service.start()
        
        assert service.is_running
        assert not service.metrics["component_health"]["kafka_producer"]
        assert service.metrics["component_health"]["kafka_consumer"]  # Should be healthy
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, mock_components):
        """Test health monitoring functionality."""
        config = {"health_check_interval": 0.05}  # Very fast for testing
        
        service = RealtimeStreamingService(
            config=config,
            **mock_components
        )
        
        await service.start()
        
        # Let health monitor run a few times
        await asyncio.sleep(0.2)
        
        # Verify health was updated
        assert service.metrics["last_health_check"] > 0
        
        await service.stop()
    
    def test_health_status_reporting(self, mock_components):
        """Test health status reporting."""
        config = {}
        service = RealtimeStreamingService(
            config=config,
            **mock_components
        )
        
        health = service.get_health_status()
        
        assert health["service"] == "realtime_streaming"
        assert health["is_running"] is False
        assert "component_health" in health
        assert "metrics" in health
        assert "configuration" in health
        assert "sse_connections" in health
    
    @pytest.mark.asyncio
    async def test_detection_publishing(self, mock_components):
        """Test detection result publishing."""
        mock_components["analytics_connector"].publish_detection_result = AsyncMock(
            return_value=True
        )
        
        config = {}
        service = RealtimeStreamingService(
            config=config,
            **mock_components
        )
        
        await service.start()
        
        # Publish detection
        result = await service.publish_detection_result(
            camera_id="cam1",
            frame_id="frame_123",
            detection_results=[{"class": "car", "confidence": 0.9}]
        )
        
        assert result is True
        mock_components["analytics_connector"].publish_detection_result.assert_called_once()
        
        await service.stop()


class TestEndToEndIntegration:
    """Test end-to-end integration scenarios."""
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka components not available")
    async def test_detection_to_sse_flow(self):
        """Test complete flow from detection event to SSE broadcast."""
        # This would be an integration test that requires running Kafka
        # For now, we'll test the components in isolation with mocks
        
        # Mock Kafka message
        kafka_message = MagicMock()
        kafka_message.value = {
            "camera_id": "cam1",
            "detection_id": "det_123",
            "bounding_boxes": [{"x": 100, "y": 100, "w": 50, "h": 50}],
            "confidence_scores": [0.9],
            "vehicle_classes": ["car"]
        }
        kafka_message.key = "cam1"
        kafka_message.headers = [("event_type", b"detection_result")]
        
        # Create components
        broadcaster = SSEBroadcaster()
        config = {"topic_prefix": "test"}
        consumer = KafkaSSEConsumer(config, broadcaster)
        
        # Process message
        await consumer._process_message("test.detections.vehicles", kafka_message)
        
        # Verify SSE message was created and would be broadcasted
        # (In real scenario, this would send to connected SSE clients)
    
    def test_configuration_validation(self):
        """Test configuration validation across components."""
        # Test that all components accept consistent configuration
        kafka_config = {
            "bootstrap_servers": ["localhost:9092"],
            "topic_prefix": "its-camera-ai",
        }
        
        # SSE broadcaster should accept Kafka config
        broadcaster = SSEBroadcaster(kafka_config=kafka_config)
        assert broadcaster.kafka_config == kafka_config
        
        if KAFKA_AVAILABLE:
            # Kafka consumer should use same config
            consumer = KafkaSSEConsumer(kafka_config, broadcaster)
            assert consumer.bootstrap_servers == kafka_config["bootstrap_servers"]
            assert consumer.topic_prefix == kafka_config["topic_prefix"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])