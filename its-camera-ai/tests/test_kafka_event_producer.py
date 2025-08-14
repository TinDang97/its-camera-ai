"""Tests for Kafka Event Producer with Topic Partitioning.

This module contains comprehensive tests for the enhanced Kafka event producer,
including partitioning logic, event serialization, and performance metrics.
"""

import asyncio
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.its_camera_ai.services.kafka_event_producer import (
    KAFKA_AVAILABLE,
    CameraEvent,
    CompressionType,
    DetectionEvent,
    EventMetadata,
    EventPriority,
    EventType,
    KafkaEventProducer,
    SystemEvent,
    TopicPartitioner,
    EventSchemaManager,
    create_kafka_producer,
)

# Skip all tests if Kafka dependencies are not available
pytestmark = pytest.mark.skipif(not KAFKA_AVAILABLE, reason="Kafka dependencies not available")


class TestTopicPartitioner:
    """Test topic partitioning logic."""
    
    def test_partition_initialization(self):
        """Test partitioner initialization."""
        partitioner = TopicPartitioner(partition_count=8)
        assert partitioner.partition_count == 8
        assert partitioner.camera_partition_map == {}
    
    def test_camera_event_partitioning(self):
        """Test camera event partitioning consistency."""
        partitioner = TopicPartitioner(partition_count=12)
        
        camera_id = "camera-001"
        partition1 = partitioner.get_partition(EventType.CAMERA_STATUS, camera_id)
        partition2 = partitioner.get_partition(EventType.CAMERA_STATUS, camera_id)
        
        # Same camera should always go to same partition
        assert partition1 == partition2
        assert 0 <= partition1 < 12
    
    def test_detection_event_partitioning(self):
        """Test detection event partitioning."""
        partitioner = TopicPartitioner(partition_count=12)
        
        camera_id = "camera-002"
        partition = partitioner.get_partition(EventType.VEHICLE_DETECTION, camera_id)
        
        assert 0 <= partition < 12
    
    def test_system_event_partitioning(self):
        """Test system event round-robin partitioning."""
        partitioner = TopicPartitioner(partition_count=4)
        
        partitions = []
        for i in range(10):
            service_name = f"service-{i}"
            partition = partitioner.get_partition(EventType.SYSTEM_ALERT, service_name)
            partitions.append(partition)
            assert 0 <= partition < 4
        
        # Should use all partitions over multiple services
        unique_partitions = set(partitions)
        assert len(unique_partitions) > 1
    
    def test_analytics_event_partitioning(self):
        """Test analytics event zone-based partitioning."""
        partitioner = TopicPartitioner(partition_count=6)
        
        zone_id = "zone-north-001"
        partition1 = partitioner.get_partition(EventType.TRAFFIC_FLOW, zone_id)
        partition2 = partitioner.get_partition(EventType.ZONE_ANALYTICS, zone_id)
        
        # Same zone should consistently route to same partition
        assert partition1 == partition2
        assert 0 <= partition1 < 6


class TestEventSchemaManager:
    """Test event schema validation."""
    
    def test_schema_initialization(self):
        """Test schema manager initialization."""
        manager = EventSchemaManager()
        
        assert EventType.VEHICLE_DETECTION in manager.schemas
        assert EventType.CAMERA_STATUS in manager.schemas
    
    def test_vehicle_detection_validation(self):
        """Test vehicle detection event validation."""
        manager = EventSchemaManager()
        
        # Valid event
        valid_data = {
            "detection_id": "det-123",
            "camera_id": "cam-001", 
            "timestamp": time.time(),
            "bounding_boxes": ["[100,100,200,200]"],
            "confidence_scores": [0.95],
        }
        
        assert manager.validate_event(EventType.VEHICLE_DETECTION, valid_data)
        
        # Invalid event (missing fields)
        invalid_data = {
            "detection_id": "det-123",
            # Missing required fields
        }
        
        assert not manager.validate_event(EventType.VEHICLE_DETECTION, invalid_data)
    
    def test_camera_status_validation(self):
        """Test camera status event validation."""
        manager = EventSchemaManager()
        
        # Valid event
        valid_data = {
            "camera_id": "cam-001",
            "status": "online",
            "timestamp": time.time(),
        }
        
        assert manager.validate_event(EventType.CAMERA_STATUS, valid_data)
    
    def test_undefined_event_type_validation(self):
        """Test validation for undefined event types."""
        manager = EventSchemaManager()
        
        # Should pass validation for undefined types
        assert manager.validate_event(EventType.SYSTEM_ALERT, {"any": "data"})


class TestKafkaEventProducer:
    """Test Kafka event producer functionality."""
    
    @pytest.fixture
    def producer_config(self):
        """Producer configuration for testing."""
        return {
            "bootstrap_servers": ["localhost:9092"],
            "topic_prefix": "test-its",
            "partition_count": 4,
            "batch_size": 1024,
            "linger_ms": 1,
            "compression_type": CompressionType.NONE.value,
            "acks": 1,
            "retries": 1,
        }
    
    @pytest.fixture
    def mock_producer(self):
        """Mock AIOKafkaProducer for testing."""
        with patch("src.its_camera_ai.services.kafka_event_producer.AIOKafkaProducer") as mock:
            producer_instance = AsyncMock()
            mock.return_value = producer_instance
            
            # Mock send method to return a future with metadata
            send_future = asyncio.Future()
            record_metadata = MagicMock()
            record_metadata.offset = 123
            send_future.set_result(record_metadata)
            producer_instance.send.return_value = send_future
            
            yield producer_instance
    
    @pytest.mark.asyncio
    async def test_producer_initialization(self, producer_config):
        """Test producer initialization."""
        producer = KafkaEventProducer(producer_config)
        
        assert producer.bootstrap_servers == ["localhost:9092"] 
        assert producer.topic_prefix == "test-its"
        assert producer.partition_count == 4
        assert not producer.is_healthy
        assert producer.connection_errors == 0
    
    @pytest.mark.asyncio
    async def test_topic_initialization(self, producer_config):
        """Test topic name initialization."""
        producer = KafkaEventProducer(producer_config)
        
        expected_topics = {
            EventType.VEHICLE_DETECTION: "test-its.detections.vehicles",
            EventType.CAMERA_STATUS: "test-its.cameras.status",
            EventType.SYSTEM_ALERT: "test-its.system.alerts",
            EventType.TRAFFIC_FLOW: "test-its.analytics.traffic-flow",
        }
        
        for event_type, expected_topic in expected_topics.items():
            assert producer.topics[event_type] == expected_topic
    
    @pytest.mark.asyncio
    async def test_producer_start_success(self, producer_config, mock_producer):
        """Test successful producer startup."""
        producer = KafkaEventProducer(producer_config)
        
        await producer.start()
        
        assert producer.is_healthy
        assert producer.connection_errors == 0
        mock_producer.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_producer_start_failure(self, producer_config):
        """Test producer startup failure."""
        with patch("src.its_camera_ai.services.kafka_event_producer.AIOKafkaProducer") as mock:
            mock.side_effect = Exception("Connection failed")
            
            producer = KafkaEventProducer(producer_config)
            
            with pytest.raises(Exception):
                await producer.start()
            
            assert not producer.is_healthy
            assert producer.connection_errors == 1
    
    @pytest.mark.asyncio
    async def test_send_detection_event(self, producer_config, mock_producer):
        """Test sending detection event."""
        producer = KafkaEventProducer(producer_config)
        await producer.start()
        
        # Create detection event
        detection = DetectionEvent(
            detection_id="det-123",
            camera_id="cam-001",
            frame_id="frame-456", 
            timestamp=time.time(),
            bounding_boxes=[{"x": 100, "y": 100, "width": 50, "height": 50}],
            vehicle_classes=["car"],
            confidence_scores=[0.95],
            tracking_ids=["track-789"],
        )
        
        # Send event
        result = await producer.send_detection_event(detection, EventPriority.NORMAL)
        
        assert result is True
        mock_producer.send.assert_called_once()
        
        # Check call arguments
        call_args = mock_producer.send.call_args
        assert call_args[1]["topic"] == "test-its.detections.vehicles"
        assert call_args[1]["key"] == "cam-001"
        assert 0 <= call_args[1]["partition"] < 4
    
    @pytest.mark.asyncio
    async def test_send_camera_event(self, producer_config, mock_producer):
        """Test sending camera event."""
        producer = KafkaEventProducer(producer_config)
        await producer.start()
        
        # Create camera event
        metadata = EventMetadata(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            source_service="camera_service"
        )
        
        camera_event = CameraEvent(
            camera_id="cam-002",
            event_type=EventType.CAMERA_STATUS,
            priority=EventPriority.HIGH,
            data={"status": "online", "fps": 30},
            metadata=metadata,
            zone_id="zone-001"
        )
        
        # Send event
        result = await producer.send_camera_event(camera_event, EventPriority.HIGH)
        
        assert result is True
        mock_producer.send.assert_called_once()
        
        # Check call arguments
        call_args = mock_producer.send.call_args
        assert call_args[1]["topic"] == "test-its.cameras.status"
        assert call_args[1]["key"] == "cam-002"
    
    @pytest.mark.asyncio
    async def test_send_system_event(self, producer_config, mock_producer):
        """Test sending system event."""
        producer = KafkaEventProducer(producer_config)
        await producer.start()
        
        # Create system event
        metadata = EventMetadata(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            source_service="system_monitor"
        )
        
        system_event = SystemEvent(
            event_type=EventType.SYSTEM_ALERT,
            priority=EventPriority.CRITICAL,
            service_name="vision_engine",
            data={"alert_type": "high_cpu", "cpu_usage": 95.5},
            metadata=metadata,
            node_id="node-001"
        )
        
        # Send event
        result = await producer.send_system_event(system_event, EventPriority.CRITICAL)
        
        assert result is True
        mock_producer.send.assert_called_once()
        
        # Check call arguments  
        call_args = mock_producer.send.call_args
        assert call_args[1]["topic"] == "test-its.system.alerts"
        assert call_args[1]["key"] == "vision_engine"
    
    @pytest.mark.asyncio
    async def test_send_analytics_event(self, producer_config, mock_producer):
        """Test sending analytics event."""
        producer = KafkaEventProducer(producer_config)
        await producer.start()
        
        # Analytics data
        analytics_data = {
            "vehicle_count": 15,
            "avg_speed_kmh": 45.2,
            "flow_rate_vph": 120,
        }
        
        # Send event
        result = await producer.send_analytics_event(
            event_type=EventType.TRAFFIC_FLOW,
            data=analytics_data,
            zone_id="zone-north-001", 
            priority=EventPriority.NORMAL
        )
        
        assert result is True
        mock_producer.send.assert_called_once()
        
        # Check call arguments
        call_args = mock_producer.send.call_args
        assert call_args[1]["topic"] == "test-its.analytics.traffic-flow"
        assert call_args[1]["key"] == "zone-north-001"
    
    @pytest.mark.asyncio
    async def test_send_event_failure(self, producer_config, mock_producer):
        """Test handling of send failures."""
        producer = KafkaEventProducer(producer_config)
        await producer.start()
        
        # Mock send to raise exception
        from aiokafka.errors import KafkaTimeoutError
        mock_producer.send.side_effect = KafkaTimeoutError("Timeout")
        
        # Create detection event
        detection = DetectionEvent(
            detection_id="det-123",
            camera_id="cam-001",
            frame_id="frame-456",
            timestamp=time.time(),
            bounding_boxes=[],
            vehicle_classes=[],
            confidence_scores=[],
            tracking_ids=[],
        )
        
        # Send should fail gracefully
        result = await producer.send_detection_event(detection)
        
        assert result is False
        assert producer.metrics["events_failed"][EventType.VEHICLE_DETECTION] == 1
    
    @pytest.mark.asyncio
    async def test_value_serialization(self, producer_config):
        """Test event value serialization."""
        producer = KafkaEventProducer(producer_config)
        
        # Test dictionary serialization
        dict_data = {"key": "value", "number": 123}
        serialized = producer._serialize_value(dict_data)
        assert isinstance(serialized, bytes)
        
        deserialized = json.loads(serialized.decode('utf-8'))
        assert deserialized == dict_data
        
        # Test object serialization
        metadata = EventMetadata(
            event_id="test-123",
            timestamp=time.time(),
            source_service="test"
        )
        serialized = producer._serialize_value(metadata)
        assert isinstance(serialized, bytes)
        
        # Should be able to deserialize
        deserialized = json.loads(serialized.decode('utf-8'))
        assert deserialized["event_id"] == "test-123"
    
    def test_health_status(self, producer_config):
        """Test health status reporting."""
        producer = KafkaEventProducer(producer_config)
        
        health = producer.get_health_status()
        
        assert "is_healthy" in health
        assert "connection_errors" in health
        assert "metrics" in health
        assert "configuration" in health
        
        # Check metrics structure
        metrics = health["metrics"]
        assert "throughput_events_per_sec" in metrics
        assert "total_events_sent" in metrics
        assert "total_events_failed" in metrics
        assert "events_by_type" in metrics
        
        # Check configuration
        config = health["configuration"]
        assert config["bootstrap_servers"] == ["localhost:9092"]
        assert config["topic_prefix"] == "test-its"
        assert config["partition_count"] == 4
    
    @pytest.mark.asyncio 
    async def test_context_manager(self, producer_config, mock_producer):
        """Test async context manager usage."""
        producer = KafkaEventProducer(producer_config)
        
        async with producer:
            assert producer.is_healthy
            mock_producer.start.assert_called_once()
        
        mock_producer.stop.assert_called_once()


class TestFactoryFunction:
    """Test factory function for producer creation."""
    
    def test_create_kafka_producer_default_config(self):
        """Test factory function with default configuration."""
        producer = create_kafka_producer()
        
        assert producer.bootstrap_servers == ["localhost:9092"]
        assert producer.topic_prefix == "its-camera-ai"
        assert producer.partition_count == 12
    
    def test_create_kafka_producer_custom_config(self):
        """Test factory function with custom configuration."""
        custom_config = {
            "bootstrap_servers": ["kafka1:9092", "kafka2:9092"],
            "topic_prefix": "custom-prefix",
            "partition_count": 8,
        }
        
        producer = create_kafka_producer(custom_config)
        
        assert producer.bootstrap_servers == ["kafka1:9092", "kafka2:9092"]
        assert producer.topic_prefix == "custom-prefix"
        assert producer.partition_count == 8


@pytest.mark.asyncio
@pytest.mark.integration
class TestKafkaProducerIntegration:
    """Integration tests requiring actual Kafka instance."""
    
    async def test_real_kafka_connection(self):
        """Test connection to real Kafka instance."""
        # Skip if no real Kafka available
        pytest.skip("Requires actual Kafka instance for integration testing")
        
        config = {
            "bootstrap_servers": ["localhost:9092"],
            "topic_prefix": "test-integration",
        }
        
        producer = KafkaEventProducer(config)
        
        try:
            await producer.start()
            assert producer.is_healthy
            
            # Send a test event
            detection = DetectionEvent(
                detection_id="integration-test",
                camera_id="cam-test",
                frame_id="frame-test",
                timestamp=time.time(),
                bounding_boxes=[],
                vehicle_classes=[],
                confidence_scores=[],
                tracking_ids=[],
            )
            
            result = await producer.send_detection_event(detection)
            assert result is True
            
        finally:
            await producer.stop()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])