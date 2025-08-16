"""Kafka Event Streaming Processor for Real-time ML Detection Events.

This processor handles the real-time streaming of ML detection events to enable:
1. Real-time dashboard updates via SSE endpoints
2. Event-driven analytics processing  
3. Integration with external systems
4. Scalable event distribution across microservices

Critical for connecting ML detection results to streaming analytics endpoints.
"""

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from ..core.logging import get_logger

# Kafka imports with fallback
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

logger = get_logger(__name__)


class EventType(Enum):
    """Event types for ML streaming."""

    DETECTION = "detection"
    ANALYTICS = "analytics"
    METRICS = "metrics"
    HEALTH = "health"
    ALERT = "alert"


@dataclass
class StreamingEvent:
    """Streaming event for real-time processing."""

    event_type: EventType
    timestamp: datetime
    camera_id: str
    frame_id: str
    data: dict[str, Any]
    source: str = "ml_pipeline"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "data": self.data,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamingEvent":
        """Create from dictionary."""
        return cls(
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            camera_id=data["camera_id"],
            frame_id=data["frame_id"],
            data=data["data"],
            source=data.get("source", "unknown"),
        )


class KafkaEventProcessor:
    """Kafka-based event processor for real-time ML detection streaming.
    
    Handles bidirectional event streaming:
    - Produces detection/analytics events from ML pipeline
    - Consumes events for real-time processing and SSE distribution
    """

    def __init__(
        self,
        bootstrap_servers: list[str] = None,
        topics_config: dict[str, str] = None,
        consumer_group: str = "ml_event_processor",
        batch_size: int = 100,
        linger_ms: int = 10,
    ):
        """Initialize Kafka event processor."""
        self.bootstrap_servers = bootstrap_servers or ["localhost:9092"]
        self.consumer_group = consumer_group
        self.batch_size = batch_size
        self.linger_ms = linger_ms

        # Topic configuration
        self.topics = topics_config or {
            "detections": "detection_events",
            "analytics": "analytics_events",
            "metrics": "metrics_events",
            "alerts": "alert_events",
        }

        # Kafka clients
        self.producer = None
        self.consumer = None

        # Event handlers
        self.event_handlers: dict[EventType, list[Callable]] = {
            event_type: [] for event_type in EventType
        }

        # State
        self.is_running = False
        self.processing_task = None

        # Performance tracking
        self.events_produced = 0
        self.events_consumed = 0
        self.last_metrics_time = time.time()

        logger.info(f"Kafka Event Processor initialized with servers: {self.bootstrap_servers}")

    async def start(self) -> None:
        """Start Kafka event processor."""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available - event streaming disabled")
            return

        if self.is_running:
            logger.warning("Event processor already running")
            return

        logger.info("Starting Kafka event processor...")

        try:
            # Initialize producer
            await self._initialize_producer()

            # Initialize consumer
            await self._initialize_consumer()

            # Start processing task
            self.processing_task = asyncio.create_task(self._process_events())

            self.is_running = True
            logger.info("Kafka event processor started successfully")

        except Exception as e:
            logger.error(f"Failed to start Kafka event processor: {e}")
            raise

    async def stop(self) -> None:
        """Stop Kafka event processor."""
        if not self.is_running:
            return

        logger.info("Stopping Kafka event processor...")

        try:
            # Cancel processing task
            if self.processing_task:
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass

            # Stop consumer
            if self.consumer:
                await self.consumer.stop()

            # Stop producer
            if self.producer:
                await self.producer.stop()

            self.is_running = False
            logger.info("Kafka event processor stopped")

        except Exception as e:
            logger.error(f"Error stopping event processor: {e}")

    async def publish_detection_event(
        self,
        camera_id: str,
        frame_id: str,
        detections: list[dict[str, Any]],
        metadata: dict[str, Any] = None,
    ) -> None:
        """Publish detection event for real-time processing."""
        if not self.producer:
            return

        try:
            event = StreamingEvent(
                event_type=EventType.DETECTION,
                timestamp=datetime.now(UTC),
                camera_id=camera_id,
                frame_id=frame_id,
                data={
                    "detections": detections,
                    "vehicle_count": len([d for d in detections if d.get("is_vehicle", False)]),
                    "metadata": metadata or {},
                },
            )

            await self._publish_event(self.topics["detections"], event)
            self.events_produced += 1

            logger.debug(f"Published detection event for {camera_id}:{frame_id}")

        except Exception as e:
            logger.error(f"Failed to publish detection event: {e}")

    async def publish_analytics_event(
        self,
        camera_id: str,
        frame_id: str,
        analytics_result: dict[str, Any],
    ) -> None:
        """Publish analytics event for dashboard updates."""
        if not self.producer:
            return

        try:
            event = StreamingEvent(
                event_type=EventType.ANALYTICS,
                timestamp=datetime.now(UTC),
                camera_id=camera_id,
                frame_id=frame_id,
                data=analytics_result,
            )

            await self._publish_event(self.topics["analytics"], event)
            self.events_produced += 1

            logger.debug(f"Published analytics event for {camera_id}:{frame_id}")

        except Exception as e:
            logger.error(f"Failed to publish analytics event: {e}")

    async def publish_metrics_event(
        self,
        source: str,
        metrics: dict[str, Any],
    ) -> None:
        """Publish metrics event for monitoring."""
        if not self.producer:
            return

        try:
            event = StreamingEvent(
                event_type=EventType.METRICS,
                timestamp=datetime.now(UTC),
                camera_id="system",
                frame_id="metrics",
                data=metrics,
                source=source,
            )

            await self._publish_event(self.topics["metrics"], event)

        except Exception as e:
            logger.error(f"Failed to publish metrics event: {e}")

    async def publish_alert_event(
        self,
        camera_id: str,
        alert_type: str,
        message: str,
        severity: str = "info",
        data: dict[str, Any] = None,
    ) -> None:
        """Publish alert event for notifications."""
        if not self.producer:
            return

        try:
            event = StreamingEvent(
                event_type=EventType.ALERT,
                timestamp=datetime.now(UTC),
                camera_id=camera_id,
                frame_id="alert",
                data={
                    "alert_type": alert_type,
                    "message": message,
                    "severity": severity,
                    "data": data or {},
                },
            )

            await self._publish_event(self.topics["alerts"], event)

        except Exception as e:
            logger.error(f"Failed to publish alert event: {e}")

    def register_event_handler(
        self,
        event_type: EventType,
        handler: Callable[[StreamingEvent], None],
    ) -> None:
        """Register event handler for specific event type."""
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for {event_type.value} events")

    def remove_event_handler(
        self,
        event_type: EventType,
        handler: Callable[[StreamingEvent], None],
    ) -> None:
        """Remove event handler."""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get event processor performance metrics."""
        current_time = time.time()
        time_diff = current_time - self.last_metrics_time

        produce_rate = self.events_produced / time_diff if time_diff > 0 else 0
        consume_rate = self.events_consumed / time_diff if time_diff > 0 else 0

        metrics = {
            "timestamp": datetime.now(UTC).isoformat(),
            "is_running": self.is_running,
            "events_produced": self.events_produced,
            "events_consumed": self.events_consumed,
            "produce_rate_per_sec": produce_rate,
            "consume_rate_per_sec": consume_rate,
            "kafka_available": KAFKA_AVAILABLE,
            "topics": self.topics,
        }

        # Reset counters
        self.events_produced = 0
        self.events_consumed = 0
        self.last_metrics_time = current_time

        return metrics

    async def _initialize_producer(self) -> None:
        """Initialize Kafka producer."""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            compression_type='snappy',
            max_batch_size=self.batch_size,
            linger_ms=self.linger_ms,
            acks='all',  # Wait for all replicas
            retries=3,
            max_in_flight_requests_per_connection=5,
        )

        await self.producer.start()
        logger.info("Kafka producer initialized")

    async def _initialize_consumer(self) -> None:
        """Initialize Kafka consumer."""
        self.consumer = AIOKafkaConsumer(
            *self.topics.values(),
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',  # Start from latest messages
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            max_poll_records=self.batch_size,
        )

        await self.consumer.start()
        logger.info(f"Kafka consumer initialized for topics: {list(self.topics.values())}")

    async def _publish_event(self, topic: str, event: StreamingEvent) -> None:
        """Publish event to Kafka topic."""
        await self.producer.send_and_wait(
            topic,
            value=event.to_dict(),
            key=event.camera_id.encode('utf-8'),
        )

    async def _process_events(self) -> None:
        """Main event processing loop."""
        logger.info("Started event processing loop")

        try:
            async for message in self.consumer:
                try:
                    # Parse event
                    event_data = message.value
                    event = StreamingEvent.from_dict(event_data)

                    # Call registered handlers
                    handlers = self.event_handlers.get(event.event_type, [])
                    for handler in handlers:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(event)
                            else:
                                handler(event)
                        except Exception as e:
                            logger.error(f"Event handler error: {e}")

                    self.events_consumed += 1

                    # Log high-level events
                    if event.event_type in [EventType.ALERT, EventType.METRICS]:
                        logger.debug(f"Processed {event.event_type.value} event from {event.camera_id}")

                except Exception as e:
                    logger.error(f"Error processing event: {e}")

        except asyncio.CancelledError:
            logger.info("Event processing loop cancelled")
        except Exception as e:
            logger.error(f"Event processing loop error: {e}")


# Factory function for creating event processor
def create_kafka_event_processor(
    bootstrap_servers: list[str] = None,
    topics_config: dict[str, str] = None,
    consumer_group: str = "ml_event_processor",
) -> KafkaEventProcessor:
    """Create and configure Kafka event processor."""

    if not KAFKA_AVAILABLE:
        logger.warning("Kafka not available - returning no-op processor")

        # Return a no-op processor that doesn't actually use Kafka
        class NoOpEventProcessor:
            async def start(self): pass
            async def stop(self): pass
            async def publish_detection_event(self, *args, **kwargs): pass
            async def publish_analytics_event(self, *args, **kwargs): pass
            async def publish_metrics_event(self, *args, **kwargs): pass
            async def publish_alert_event(self, *args, **kwargs): pass
            def register_event_handler(self, *args, **kwargs): pass
            async def get_performance_metrics(self): return {}

        return NoOpEventProcessor()

    return KafkaEventProcessor(
        bootstrap_servers=bootstrap_servers,
        topics_config=topics_config,
        consumer_group=consumer_group,
    )
