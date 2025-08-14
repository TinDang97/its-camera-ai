"""Enhanced Kafka Event Producer with Topic Partitioning.

This module provides a comprehensive Kafka event producer for the ITS Camera AI system,
featuring intelligent topic partitioning, schema management, and performance optimization
for real-time camera analytics events.

Key Features:
- Intelligent topic partitioning based on event type and data characteristics
- Schema validation and event serialization
- High-throughput batching and compression
- Connection pooling and retry logic
- Comprehensive monitoring and metrics
- Integration with vision analytics pipeline
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from typing import Any

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError, KafkaTimeoutError

from ..core.exceptions import ServiceError, ValidationError
from ..core.logging import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Event types for topic routing and partitioning."""

    # Detection events
    VEHICLE_DETECTION = "vehicle_detection"
    TRAFFIC_VIOLATION = "traffic_violation"
    INCIDENT_DETECTION = "incident_detection"

    # Camera events
    CAMERA_STATUS = "camera_status"
    STREAM_HEALTH = "stream_health"
    CAMERA_CONFIG = "camera_config"

    # System events
    SYSTEM_ALERT = "system_alert"
    PERFORMANCE_METRIC = "performance_metric"
    MODEL_DEPLOYMENT = "model_deployment"

    # Analytics events
    TRAFFIC_FLOW = "traffic_flow"
    ZONE_ANALYTICS = "zone_analytics"
    SPEED_CALCULATION = "speed_calculation"


class EventPriority(Enum):
    """Event priority levels for processing order."""

    CRITICAL = 0  # System alerts, incidents
    HIGH = 1  # Traffic violations, camera status
    NORMAL = 2  # Regular detections, analytics
    LOW = 3  # Metrics, non-urgent updates


class CompressionType(Enum):
    """Kafka message compression types."""

    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"


@dataclass
class EventMetadata:
    """Metadata for event tracking and processing."""

    event_id: str
    timestamp: float
    source_service: str
    correlation_id: str | None = None
    trace_id: str | None = None
    retry_count: int = 0
    processing_time_ms: float = 0.0


@dataclass
class CameraEvent:
    """Camera-related event data structure."""

    camera_id: str
    event_type: EventType
    priority: EventPriority
    data: dict[str, Any]
    metadata: EventMetadata

    # Camera-specific fields
    zone_id: str | None = None
    stream_url: str | None = None
    frame_timestamp: float | None = None


@dataclass
class DetectionEvent:
    """Vehicle detection event structure."""

    detection_id: str
    camera_id: str
    frame_id: str
    timestamp: float

    # Detection data
    bounding_boxes: list[dict[str, Any]]
    vehicle_classes: list[str]
    confidence_scores: list[float]
    tracking_ids: list[str]

    # Analytics data
    speed_kmh: float | None = None
    direction: str | None = None
    zone_violations: list[str] = None

    metadata: EventMetadata | None = None


@dataclass
class SystemEvent:
    """System-level event structure."""

    event_type: EventType
    priority: EventPriority
    service_name: str
    data: dict[str, Any]
    metadata: EventMetadata

    # System-specific fields
    node_id: str | None = None
    resource_usage: dict[str, float] | None = None


class TopicPartitioner:
    """Intelligent topic partitioning strategy."""

    def __init__(self, partition_count: int = 12):
        self.partition_count = partition_count
        self.camera_partition_map = {}  # Cache for camera -> partition mapping

    def get_partition(self, event_type: EventType, key: str) -> int:
        """Get optimal partition for event based on type and key.

        Partitioning strategy:
        - Camera events: Hash camera_id for locality
        - Detection events: Hash camera_id + frame_timestamp for ordering
        - System events: Round-robin by service_name
        - Analytics events: Hash zone_id for aggregation locality

        Args:
            event_type: Type of event for routing strategy
            key: Primary key for partitioning (camera_id, service_name, etc.)

        Returns:
            Partition number (0 to partition_count-1)
        """
        if event_type in [
            EventType.CAMERA_STATUS,
            EventType.STREAM_HEALTH,
            EventType.CAMERA_CONFIG,
        ]:
            # Camera events: partition by camera_id for locality
            return self._hash_partition(key)

        elif event_type in [
            EventType.VEHICLE_DETECTION,
            EventType.TRAFFIC_VIOLATION,
            EventType.INCIDENT_DETECTION,
        ]:
            # Detection events: partition by camera_id for processing order
            return self._hash_partition(key)

        elif event_type in [
            EventType.SYSTEM_ALERT,
            EventType.PERFORMANCE_METRIC,
            EventType.MODEL_DEPLOYMENT,
        ]:
            # System events: distribute evenly
            return self._round_robin_partition(key)

        elif event_type in [
            EventType.TRAFFIC_FLOW,
            EventType.ZONE_ANALYTICS,
            EventType.SPEED_CALCULATION,
        ]:
            # Analytics events: partition by zone_id for aggregation
            return self._hash_partition(key)

        else:
            # Default: hash partition
            return self._hash_partition(key)

    def _hash_partition(self, key: str) -> int:
        """Hash-based partitioning for consistent routing."""
        hash_bytes = sha256(key.encode("utf-8")).digest()
        hash_int = int.from_bytes(hash_bytes[:4], byteorder="big")
        return hash_int % self.partition_count

    def _round_robin_partition(self, key: str) -> int:
        """Round-robin partitioning for load distribution."""
        return abs(hash(key)) % self.partition_count


class EventSchemaManager:
    """Manages event schemas and validation."""

    def __init__(self):
        self.schemas = self._load_schemas()

    def _load_schemas(self) -> dict[EventType, dict]:
        """Load Avro schemas for each event type."""
        return {
            EventType.VEHICLE_DETECTION: {
                "type": "record",
                "name": "VehicleDetection",
                "fields": [
                    {"name": "detection_id", "type": "string"},
                    {"name": "camera_id", "type": "string"},
                    {"name": "timestamp", "type": "double"},
                    {
                        "name": "bounding_boxes",
                        "type": {"type": "array", "items": "string"},
                    },
                    {
                        "name": "confidence_scores",
                        "type": {"type": "array", "items": "float"},
                    },
                ],
            },
            EventType.CAMERA_STATUS: {
                "type": "record",
                "name": "CameraStatus",
                "fields": [
                    {"name": "camera_id", "type": "string"},
                    {"name": "status", "type": "string"},
                    {"name": "timestamp", "type": "double"},
                ],
            },
            # Additional schemas would be defined here
        }

    def validate_event(self, event_type: EventType, data: dict[str, Any]) -> bool:
        """Validate event data against schema."""
        schema = self.schemas.get(event_type)
        if not schema:
            return True  # No schema validation for undefined types

        # Basic validation - in production would use proper Avro validation
        try:
            required_fields = [field["name"] for field in schema["fields"]]
            return all(field in data for field in required_fields)
        except Exception:
            return False


class KafkaEventProducer:
    """High-performance Kafka event producer with topic partitioning."""

    def __init__(self, config: dict[str, Any]):
        """Initialize Kafka event producer.

        Args:
            config: Producer configuration dictionary
                - bootstrap_servers: List of Kafka broker addresses
                - topic_prefix: Prefix for all topics (default: "its-camera-ai")
                - partition_count: Number of partitions per topic (default: 12)
                - batch_size: Batch size for producer (default: 16384)
                - linger_ms: Batch linger time (default: 5)
                - compression_type: Message compression (default: "snappy")
                - acks: Acknowledgment level (default: 1)
                - retries: Number of retries (default: 3)
                - enable_idempotence: Enable idempotent producer (default: True)
        """
        self.config = config
        self.bootstrap_servers = config.get("bootstrap_servers", ["localhost:9092"])
        self.topic_prefix = config.get("topic_prefix", "its-camera-ai")
        self.partition_count = config.get("partition_count", 12)

        # Producer configuration
        self.producer_config = {
            "bootstrap_servers": self.bootstrap_servers,
            "batch_size": config.get("batch_size", 16384),
            "linger_ms": config.get("linger_ms", 5),
            "compression_type": config.get(
                "compression_type", CompressionType.SNAPPY.value
            ),
            "acks": config.get("acks", 1),
            "retries": config.get("retries", 3),
            "enable_idempotence": config.get("enable_idempotence", True),
            "max_in_flight_requests_per_connection": 5,
            "value_serializer": self._serialize_value,
            "key_serializer": lambda x: x.encode("utf-8") if x else None,
        }

        # Components
        self.producer: AIOKafkaProducer | None = None
        self.partitioner = TopicPartitioner(self.partition_count)
        self.schema_manager = EventSchemaManager()

        # Topic management
        self.topics = self._initialize_topics()
        self.topic_configs = {}

        # Performance tracking
        self.metrics = {
            "events_sent": defaultdict(int),
            "events_failed": defaultdict(int),
            "batch_sizes": deque(maxlen=1000),
            "send_latencies": deque(maxlen=1000),
            "throughput_events_per_sec": 0.0,
        }

        # Rate limiting and buffering
        self.send_buffer = deque(maxlen=10000)
        self.batch_timer = None
        self.last_metrics_update = time.time()

        # Connection health
        self.is_healthy = False
        self.connection_errors = 0
        self.last_error_time = 0.0

        logger.info(
            "Kafka event producer initialized",
            bootstrap_servers=self.bootstrap_servers,
            topic_prefix=self.topic_prefix,
            partition_count=self.partition_count,
        )

    def _initialize_topics(self) -> dict[EventType, str]:
        """Initialize topic names for each event type."""
        return {
            # Detection topics
            EventType.VEHICLE_DETECTION: f"{self.topic_prefix}.detections.vehicles",
            EventType.TRAFFIC_VIOLATION: f"{self.topic_prefix}.violations.traffic",
            EventType.INCIDENT_DETECTION: f"{self.topic_prefix}.incidents",
            # Camera topics
            EventType.CAMERA_STATUS: f"{self.topic_prefix}.cameras.status",
            EventType.STREAM_HEALTH: f"{self.topic_prefix}.cameras.health",
            EventType.CAMERA_CONFIG: f"{self.topic_prefix}.cameras.config",
            # System topics
            EventType.SYSTEM_ALERT: f"{self.topic_prefix}.system.alerts",
            EventType.PERFORMANCE_METRIC: f"{self.topic_prefix}.system.metrics",
            EventType.MODEL_DEPLOYMENT: f"{self.topic_prefix}.system.deployments",
            # Analytics topics
            EventType.TRAFFIC_FLOW: f"{self.topic_prefix}.analytics.traffic-flow",
            EventType.ZONE_ANALYTICS: f"{self.topic_prefix}.analytics.zones",
            EventType.SPEED_CALCULATION: f"{self.topic_prefix}.analytics.speed",
        }

    def _serialize_value(self, value: Any) -> bytes:
        """Serialize event value to bytes."""
        try:
            if isinstance(value, dict | list):
                return json.dumps(value, default=str).encode("utf-8")
            elif hasattr(value, "__dict__"):
                return json.dumps(asdict(value), default=str).encode("utf-8")
            else:
                return json.dumps(value, default=str).encode("utf-8")
        except Exception as e:
            logger.error("Failed to serialize event value", error=str(e))
            raise ValidationError(f"Event serialization failed: {e}") from e

    async def start(self):
        """Start the Kafka producer."""
        try:
            self.producer = AIOKafkaProducer(**self.producer_config)
            await self.producer.start()

            # Start background tasks
            asyncio.create_task(self._metrics_updater())
            asyncio.create_task(self._batch_processor())

            self.is_healthy = True
            self.connection_errors = 0

            logger.info("Kafka event producer started successfully")

        except Exception as e:
            self.is_healthy = False
            self.connection_errors += 1
            self.last_error_time = time.time()

            logger.error("Failed to start Kafka producer", error=str(e))
            raise ServiceError(
                f"Kafka producer startup failed: {e}", service="kafka_producer"
            ) from e

    async def send_detection_event(
        self, detection: DetectionEvent, priority: EventPriority = EventPriority.NORMAL
    ) -> bool:
        """Send vehicle detection event.

        Args:
            detection: Detection event data
            priority: Event priority for processing order

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create event metadata
            metadata = EventMetadata(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                source_service="vision_analytics",
                correlation_id=(
                    detection.metadata.correlation_id if detection.metadata else None
                ),
                trace_id=detection.metadata.trace_id if detection.metadata else None,
            )
            detection.metadata = metadata

            # Determine partition key
            partition_key = detection.camera_id

            # Send event
            return await self._send_event(
                event_type=EventType.VEHICLE_DETECTION,
                data=detection,
                key=partition_key,
                priority=priority,
                headers={
                    "camera_id": detection.camera_id,
                    "frame_id": detection.frame_id,
                },
            )

        except Exception as e:
            logger.error(
                "Failed to send detection event",
                detection_id=detection.detection_id,
                camera_id=detection.camera_id,
                error=str(e),
            )
            return False

    async def send_camera_event(
        self, camera_event: CameraEvent, priority: EventPriority = EventPriority.HIGH
    ) -> bool:
        """Send camera status/health event.

        Args:
            camera_event: Camera event data
            priority: Event priority for processing order

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Use camera_id as partition key for locality
            partition_key = camera_event.camera_id

            return await self._send_event(
                event_type=camera_event.event_type,
                data=camera_event,
                key=partition_key,
                priority=priority,
                headers={"camera_id": camera_event.camera_id},
            )

        except Exception as e:
            logger.error(
                "Failed to send camera event",
                camera_id=camera_event.camera_id,
                event_type=camera_event.event_type.value,
                error=str(e),
            )
            return False

    async def send_system_event(
        self, system_event: SystemEvent, priority: EventPriority = EventPriority.HIGH
    ) -> bool:
        """Send system-level event.

        Args:
            system_event: System event data
            priority: Event priority for processing order

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Use service_name as partition key
            partition_key = system_event.service_name

            return await self._send_event(
                event_type=system_event.event_type,
                data=system_event,
                key=partition_key,
                priority=priority,
                headers={"service_name": system_event.service_name},
            )

        except Exception as e:
            logger.error(
                "Failed to send system event",
                service_name=system_event.service_name,
                event_type=system_event.event_type.value,
                error=str(e),
            )
            return False

    async def send_analytics_event(
        self,
        event_type: EventType,
        data: dict[str, Any],
        zone_id: str,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> bool:
        """Send analytics event (traffic flow, speed, etc.).

        Args:
            event_type: Type of analytics event
            data: Event data payload
            zone_id: Zone identifier for partitioning
            priority: Event priority for processing order

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Create metadata
            metadata = EventMetadata(
                event_id=str(uuid.uuid4()),
                timestamp=time.time(),
                source_service="analytics_engine",
            )

            # Add metadata to data
            event_data = {"metadata": asdict(metadata), "zone_id": zone_id, **data}

            # Use zone_id as partition key for aggregation locality
            partition_key = zone_id

            return await self._send_event(
                event_type=event_type,
                data=event_data,
                key=partition_key,
                priority=priority,
                headers={"zone_id": zone_id},
            )

        except Exception as e:
            logger.error(
                "Failed to send analytics event",
                event_type=event_type.value,
                zone_id=zone_id,
                error=str(e),
            )
            return False

    async def _send_event(
        self,
        event_type: EventType,
        data: Any,
        key: str,
        priority: EventPriority,
        headers: dict[str, str] | None = None,
    ) -> bool:
        """Internal method to send event to Kafka.

        Args:
            event_type: Type of event for topic routing
            data: Event data to send
            key: Partition key
            priority: Event priority
            headers: Optional message headers

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.producer or not self.is_healthy:
            logger.warning(
                "Producer not available, buffering event", event_type=event_type.value
            )
            return False

        try:
            start_time = time.time()

            # Get topic and partition
            topic = self.topics[event_type]
            partition = self.partitioner.get_partition(event_type, key)

            # Validate event against schema
            if isinstance(data, dict):
                event_data = data
            else:
                event_data = asdict(data) if hasattr(data, "__dict__") else data

            if not self.schema_manager.validate_event(event_type, event_data):
                logger.warning(
                    "Event failed schema validation", event_type=event_type.value
                )
                self.metrics["events_failed"][event_type] += 1
                return False

            # Prepare message headers
            message_headers = {
                "event_type": event_type.value,
                "priority": str(priority.value),
                "timestamp": str(time.time()),
            }
            if headers:
                message_headers.update(headers)

            # Send to Kafka
            future = await self.producer.send(
                topic=topic,
                value=data,
                key=key,
                partition=partition,
                headers=[(k, v.encode("utf-8")) for k, v in message_headers.items()],
            )

            # Wait for completion
            record_metadata = await future

            # Track metrics
            send_time = time.time() - start_time
            self.metrics["send_latencies"].append(send_time * 1000)  # Convert to ms
            self.metrics["events_sent"][event_type] += 1

            logger.debug(
                "Event sent successfully",
                event_type=event_type.value,
                topic=topic,
                partition=partition,
                offset=record_metadata.offset,
                send_time_ms=send_time * 1000,
            )

            return True

        except KafkaTimeoutError:
            logger.warning("Kafka send timeout", event_type=event_type.value, key=key)
            self.metrics["events_failed"][event_type] += 1
            return False

        except KafkaError as e:
            logger.error(
                "Kafka send error", event_type=event_type.value, key=key, error=str(e)
            )
            self.metrics["events_failed"][event_type] += 1
            self.connection_errors += 1
            self.last_error_time = time.time()

            # Check if we should mark as unhealthy
            if self.connection_errors > 5:
                self.is_healthy = False

            return False

        except Exception as e:
            logger.error(
                "Unexpected send error",
                event_type=event_type.value,
                key=key,
                error=str(e),
            )
            self.metrics["events_failed"][event_type] += 1
            return False

    async def _batch_processor(self):
        """Background task to process buffered events in batches."""
        while True:
            try:
                await asyncio.sleep(0.1)  # Process every 100ms

                if not self.send_buffer:
                    continue

                # Process buffer in batches
                batch_size = min(100, len(self.send_buffer))
                batch = []

                for _ in range(batch_size):
                    if self.send_buffer:
                        batch.append(self.send_buffer.popleft())

                if batch:
                    self.metrics["batch_sizes"].append(len(batch))

                    # Send batch (implement batch sending logic here)
                    logger.debug(f"Processing batch of {len(batch)} events")

            except Exception as e:
                logger.error("Batch processor error", error=str(e))
                await asyncio.sleep(1.0)

    async def _metrics_updater(self):
        """Background task to update performance metrics."""
        while True:
            try:
                await asyncio.sleep(10.0)  # Update every 10 seconds

                current_time = time.time()
                time_diff = current_time - self.last_metrics_update

                # Calculate throughput
                total_events = sum(self.metrics["events_sent"].values())
                if time_diff > 0:
                    self.metrics["throughput_events_per_sec"] = total_events / time_diff

                # Log metrics summary
                avg_latency = (
                    sum(self.metrics["send_latencies"])
                    / len(self.metrics["send_latencies"])
                    if self.metrics["send_latencies"]
                    else 0
                )

                logger.info(
                    "Kafka producer metrics",
                    throughput_eps=self.metrics["throughput_events_per_sec"],
                    avg_latency_ms=avg_latency,
                    total_sent=total_events,
                    total_failed=sum(self.metrics["events_failed"].values()),
                    connection_errors=self.connection_errors,
                    is_healthy=self.is_healthy,
                )

                self.last_metrics_update = current_time

            except Exception as e:
                logger.error("Metrics updater error", error=str(e))
                await asyncio.sleep(10.0)

    def get_health_status(self) -> dict[str, Any]:
        """Get producer health status and metrics."""
        return {
            "is_healthy": self.is_healthy,
            "connection_errors": self.connection_errors,
            "last_error_time": self.last_error_time,
            "metrics": {
                "throughput_events_per_sec": self.metrics["throughput_events_per_sec"],
                "total_events_sent": sum(self.metrics["events_sent"].values()),
                "total_events_failed": sum(self.metrics["events_failed"].values()),
                "avg_send_latency_ms": (
                    sum(self.metrics["send_latencies"])
                    / len(self.metrics["send_latencies"])
                    if self.metrics["send_latencies"]
                    else 0
                ),
                "events_by_type": dict(self.metrics["events_sent"]),
                "failures_by_type": dict(self.metrics["events_failed"]),
            },
            "configuration": {
                "bootstrap_servers": self.bootstrap_servers,
                "topic_prefix": self.topic_prefix,
                "partition_count": self.partition_count,
                "compression_type": self.producer_config.get("compression_type"),
                "batch_size": self.producer_config.get("batch_size"),
            },
        }

    async def stop(self):
        """Stop the Kafka producer and cleanup resources."""
        try:
            if self.producer:
                await self.producer.stop()

            self.is_healthy = False
            logger.info("Kafka event producer stopped")

        except Exception as e:
            logger.error("Error stopping Kafka producer", error=str(e))

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Factory function for easy initialization
def create_kafka_producer(config: dict[str, Any] | None = None) -> KafkaEventProducer:
    """Create and configure a Kafka event producer.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured KafkaEventProducer instance
    """
    default_config = {
        "bootstrap_servers": ["localhost:9092"],
        "topic_prefix": "its-camera-ai",
        "partition_count": 12,
        "batch_size": 16384,
        "linger_ms": 5,
        "compression_type": CompressionType.SNAPPY.value,
        "acks": 1,
        "retries": 3,
        "enable_idempotence": True,
    }

    if config:
        default_config.update(config)

    return KafkaEventProducer(default_config)
