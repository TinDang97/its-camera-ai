"""Kafka Analytics Connector for Vision System Integration.

This module provides seamless integration between the vision analytics pipeline
and the Kafka event producer, enabling real-time streaming of detection results,
camera events, and system analytics to Kafka topics.

Features:
- Automatic event publishing from vision analytics results
- Camera stream health monitoring and alerting
- System performance metrics streaming
- Error handling and retry logic
- Configurable event filtering and routing
"""

import asyncio
import time
from typing import Any

from ..core.exceptions import ServiceError
from ..core.logging import get_logger
from ..services.kafka_event_producer import (
    KAFKA_AVAILABLE,
    CameraEvent,
    DetectionEvent,
    EventMetadata,
    EventPriority,
    EventType,
    KafkaEventProducer,
    SystemEvent,
)

logger = get_logger(__name__)


class KafkaAnalyticsConnector:
    """Connector between vision analytics system and Kafka event streaming.

    This class acts as a bridge between the core vision analytics components
    and the Kafka event producer, automatically translating detection results,
    camera status updates, and system metrics into structured Kafka events.
    """

    def __init__(
        self,
        kafka_config: dict[str, Any],
        event_filters: dict[str, Any] | None = None,
        enable_system_events: bool = True,
        enable_detection_events: bool = True,
        enable_camera_events: bool = True,
    ):
        """Initialize Kafka analytics connector.

        Args:
            kafka_config: Kafka producer configuration
            event_filters: Optional filters for event publishing
            enable_system_events: Enable system event publishing
            enable_detection_events: Enable detection event publishing
            enable_camera_events: Enable camera event publishing
        """
        if not KAFKA_AVAILABLE:
            raise ServiceError("Kafka dependencies not available", service="kafka_connector")

        self.kafka_config = kafka_config
        self.event_filters = event_filters or {}
        self.enable_system_events = enable_system_events
        self.enable_detection_events = enable_detection_events
        self.enable_camera_events = enable_camera_events

        # Initialize Kafka producer
        self.kafka_producer: KafkaEventProducer | None = None

        # Event processing queues
        self.detection_queue = asyncio.Queue(maxsize=1000)
        self.camera_event_queue = asyncio.Queue(maxsize=500)
        self.system_event_queue = asyncio.Queue(maxsize=200)

        # Performance tracking
        self.stats = {
            "detections_processed": 0,
            "camera_events_processed": 0,
            "system_events_processed": 0,
            "events_filtered": 0,
            "events_failed": 0,
            "last_reset": time.time(),
        }

        # Background tasks
        self.processing_tasks: list[asyncio.Task] = []
        self.is_running = False

        logger.info("Kafka analytics connector initialized",
                   enable_detections=enable_detection_events,
                   enable_camera=enable_camera_events,
                   enable_system=enable_system_events)

    async def start(self):
        """Start the Kafka analytics connector."""
        try:
            # Initialize Kafka producer
            self.kafka_producer = KafkaEventProducer(self.kafka_config)
            await self.kafka_producer.start()

            # Start background processing tasks
            if self.enable_detection_events:
                self.processing_tasks.append(
                    asyncio.create_task(self._process_detection_events())
                )

            if self.enable_camera_events:
                self.processing_tasks.append(
                    asyncio.create_task(self._process_camera_events())
                )

            if self.enable_system_events:
                self.processing_tasks.append(
                    asyncio.create_task(self._process_system_events())
                )

            # Start metrics reporter
            self.processing_tasks.append(
                asyncio.create_task(self._metrics_reporter())
            )

            self.is_running = True
            logger.info("Kafka analytics connector started successfully")

        except Exception as e:
            logger.error("Failed to start Kafka analytics connector", error=str(e))
            raise ServiceError(f"Kafka connector startup failed: {e}", service="kafka_connector") from e

    async def publish_detection_result(
        self,
        camera_id: str,
        frame_id: str,
        detection_results: list[dict[str, Any]],
        frame_timestamp: float | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        """Publish vision detection results to Kafka.

        Args:
            camera_id: Source camera identifier
            frame_id: Frame identifier for tracking
            detection_results: List of detection results from vision engine
            frame_timestamp: Optional frame timestamp
            correlation_id: Optional correlation ID for request tracking

        Returns:
            True if successfully queued for publishing
        """
        if not self.enable_detection_events or not self.is_running:
            return False

        try:
            # Extract detection data
            bounding_boxes = []
            vehicle_classes = []
            confidence_scores = []
            tracking_ids = []

            for detection in detection_results:
                bounding_boxes.append(detection.get("bbox", {}))
                vehicle_classes.append(detection.get("class", "unknown"))
                confidence_scores.append(detection.get("confidence", 0.0))
                tracking_ids.append(detection.get("track_id", ""))

            # Apply confidence filter
            min_confidence = self.event_filters.get("min_confidence", 0.0)
            if confidence_scores and max(confidence_scores) < min_confidence:
                self.stats["events_filtered"] += 1
                return True  # Filtered but not an error

            # Create detection event
            detection_event = DetectionEvent(
                detection_id=f"{camera_id}_{frame_id}_{int(time.time() * 1000)}",
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp=frame_timestamp or time.time(),
                bounding_boxes=bounding_boxes,
                vehicle_classes=vehicle_classes,
                confidence_scores=confidence_scores,
                tracking_ids=tracking_ids,
                metadata=EventMetadata(
                    event_id=f"detection_{camera_id}_{int(time.time() * 1000000)}",
                    timestamp=time.time(),
                    source_service="vision_analytics",
                    correlation_id=correlation_id,
                )
            )

            # Queue for async processing
            try:
                self.detection_queue.put_nowait(detection_event)
                return True
            except asyncio.QueueFull:
                logger.warning("Detection queue full, dropping event", camera_id=camera_id)
                return False

        except Exception as e:
            logger.error("Failed to publish detection result",
                        camera_id=camera_id,
                        frame_id=frame_id,
                        error=str(e))
            self.stats["events_failed"] += 1
            return False

    async def publish_camera_status(
        self,
        camera_id: str,
        status: str,
        metadata: dict[str, Any],
        zone_id: str | None = None,
        priority: EventPriority = EventPriority.HIGH,
    ) -> bool:
        """Publish camera status update to Kafka.

        Args:
            camera_id: Camera identifier
            status: Camera status (online, offline, error, etc.)
            metadata: Additional status metadata
            zone_id: Optional zone identifier
            priority: Event priority level

        Returns:
            True if successfully queued for publishing
        """
        if not self.enable_camera_events or not self.is_running:
            return False

        try:
            # Create camera event
            camera_event = CameraEvent(
                camera_id=camera_id,
                event_type=EventType.CAMERA_STATUS,
                priority=priority,
                data={
                    "status": status,
                    "timestamp": time.time(),
                    **metadata
                },
                metadata=EventMetadata(
                    event_id=f"camera_status_{camera_id}_{int(time.time() * 1000000)}",
                    timestamp=time.time(),
                    source_service="camera_service",
                ),
                zone_id=zone_id,
            )

            # Queue for async processing
            try:
                self.camera_event_queue.put_nowait(camera_event)
                return True
            except asyncio.QueueFull:
                logger.warning("Camera event queue full, dropping event", camera_id=camera_id)
                return False

        except Exception as e:
            logger.error("Failed to publish camera status",
                        camera_id=camera_id,
                        status=status,
                        error=str(e))
            self.stats["events_failed"] += 1
            return False

    async def publish_stream_health(
        self,
        camera_id: str,
        health_metrics: dict[str, Any],
        zone_id: str | None = None,
    ) -> bool:
        """Publish camera stream health metrics to Kafka.

        Args:
            camera_id: Camera identifier
            health_metrics: Stream health metrics (fps, latency, errors, etc.)
            zone_id: Optional zone identifier

        Returns:
            True if successfully queued for publishing
        """
        if not self.enable_camera_events or not self.is_running:
            return False

        try:
            # Apply health threshold filters
            fps_threshold = self.event_filters.get("min_fps", 0)
            if health_metrics.get("fps", 0) < fps_threshold:
                # Only publish if below threshold (alerting)
                priority = EventPriority.HIGH
            else:
                priority = EventPriority.NORMAL

            # Create stream health event
            camera_event = CameraEvent(
                camera_id=camera_id,
                event_type=EventType.STREAM_HEALTH,
                priority=priority,
                data={
                    "health_metrics": health_metrics,
                    "timestamp": time.time(),
                },
                metadata=EventMetadata(
                    event_id=f"stream_health_{camera_id}_{int(time.time() * 1000000)}",
                    timestamp=time.time(),
                    source_service="stream_monitor",
                ),
                zone_id=zone_id,
            )

            # Queue for async processing
            try:
                self.camera_event_queue.put_nowait(camera_event)
                return True
            except asyncio.QueueFull:
                logger.warning("Camera event queue full, dropping health event", camera_id=camera_id)
                return False

        except Exception as e:
            logger.error("Failed to publish stream health",
                        camera_id=camera_id,
                        error=str(e))
            self.stats["events_failed"] += 1
            return False

    async def publish_system_alert(
        self,
        service_name: str,
        alert_type: str,
        alert_data: dict[str, Any],
        priority: EventPriority = EventPriority.HIGH,
        node_id: str | None = None,
    ) -> bool:
        """Publish system alert to Kafka.

        Args:
            service_name: Name of the service generating the alert
            alert_type: Type of alert (cpu_high, memory_low, etc.)
            alert_data: Alert details and metrics
            priority: Alert priority level
            node_id: Optional node identifier

        Returns:
            True if successfully queued for publishing
        """
        if not self.enable_system_events or not self.is_running:
            return False

        try:
            # Create system event
            system_event = SystemEvent(
                event_type=EventType.SYSTEM_ALERT,
                priority=priority,
                service_name=service_name,
                data={
                    "alert_type": alert_type,
                    "alert_data": alert_data,
                    "timestamp": time.time(),
                },
                metadata=EventMetadata(
                    event_id=f"system_alert_{service_name}_{int(time.time() * 1000000)}",
                    timestamp=time.time(),
                    source_service="system_monitor",
                ),
                node_id=node_id,
            )

            # Queue for async processing
            try:
                self.system_event_queue.put_nowait(system_event)
                return True
            except asyncio.QueueFull:
                logger.warning("System event queue full, dropping alert", service_name=service_name)
                return False

        except Exception as e:
            logger.error("Failed to publish system alert",
                        service_name=service_name,
                        alert_type=alert_type,
                        error=str(e))
            self.stats["events_failed"] += 1
            return False

    async def publish_performance_metrics(
        self,
        service_name: str,
        metrics: dict[str, Any],
        node_id: str | None = None,
    ) -> bool:
        """Publish system performance metrics to Kafka.

        Args:
            service_name: Name of the service
            metrics: Performance metrics dictionary
            node_id: Optional node identifier

        Returns:
            True if successfully queued for publishing
        """
        if not self.enable_system_events or not self.is_running:
            return False

        try:
            # Create performance metrics event
            system_event = SystemEvent(
                event_type=EventType.PERFORMANCE_METRIC,
                priority=EventPriority.LOW,
                service_name=service_name,
                data={
                    "metrics": metrics,
                    "timestamp": time.time(),
                },
                metadata=EventMetadata(
                    event_id=f"perf_metrics_{service_name}_{int(time.time() * 1000000)}",
                    timestamp=time.time(),
                    source_service="metrics_collector",
                ),
                node_id=node_id,
                resource_usage=metrics.get("resource_usage"),
            )

            # Queue for async processing
            try:
                self.system_event_queue.put_nowait(system_event)
                return True
            except asyncio.QueueFull:
                logger.debug("System event queue full, dropping metrics", service_name=service_name)
                return False

        except Exception as e:
            logger.error("Failed to publish performance metrics",
                        service_name=service_name,
                        error=str(e))
            self.stats["events_failed"] += 1
            return False

    async def publish_analytics_result(
        self,
        zone_id: str,
        analytics_type: str,
        analytics_data: dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
    ) -> bool:
        """Publish traffic analytics results to Kafka.

        Args:
            zone_id: Zone identifier for the analytics
            analytics_type: Type of analytics (traffic_flow, speed, violations)
            analytics_data: Analytics results and metrics
            priority: Event priority level

        Returns:
            True if successfully queued for publishing
        """
        if not self.is_running or not self.kafka_producer:
            return False

        try:
            # Map analytics type to event type
            event_type_map = {
                "traffic_flow": EventType.TRAFFIC_FLOW,
                "speed_calculation": EventType.SPEED_CALCULATION,
                "zone_analytics": EventType.ZONE_ANALYTICS,
            }

            event_type = event_type_map.get(analytics_type, EventType.ZONE_ANALYTICS)

            # Send analytics event directly (no queuing for analytics)
            return await self.kafka_producer.send_analytics_event(
                event_type=event_type,
                data={
                    "analytics_type": analytics_type,
                    "analytics_data": analytics_data,
                    "timestamp": time.time(),
                },
                zone_id=zone_id,
                priority=priority,
            )

        except Exception as e:
            logger.error("Failed to publish analytics result",
                        zone_id=zone_id,
                        analytics_type=analytics_type,
                        error=str(e))
            self.stats["events_failed"] += 1
            return False

    # Background processing tasks

    async def _process_detection_events(self):
        """Background task to process detection events queue."""
        logger.info("Detection events processor started")

        while self.is_running:
            try:
                # Get detection event from queue with timeout
                detection_event = await asyncio.wait_for(
                    self.detection_queue.get(),
                    timeout=1.0
                )

                # Send to Kafka
                success = await self.kafka_producer.send_detection_event(detection_event)

                if success:
                    self.stats["detections_processed"] += 1
                else:
                    self.stats["events_failed"] += 1

                # Mark task as done
                self.detection_queue.task_done()

            except TimeoutError:
                continue  # No events in queue
            except Exception as e:
                logger.error("Detection event processing error", error=str(e))
                await asyncio.sleep(0.1)

    async def _process_camera_events(self):
        """Background task to process camera events queue."""
        logger.info("Camera events processor started")

        while self.is_running:
            try:
                # Get camera event from queue with timeout
                camera_event = await asyncio.wait_for(
                    self.camera_event_queue.get(),
                    timeout=1.0
                )

                # Send to Kafka
                success = await self.kafka_producer.send_camera_event(camera_event)

                if success:
                    self.stats["camera_events_processed"] += 1
                else:
                    self.stats["events_failed"] += 1

                # Mark task as done
                self.camera_event_queue.task_done()

            except TimeoutError:
                continue  # No events in queue
            except Exception as e:
                logger.error("Camera event processing error", error=str(e))
                await asyncio.sleep(0.1)

    async def _process_system_events(self):
        """Background task to process system events queue."""
        logger.info("System events processor started")

        while self.is_running:
            try:
                # Get system event from queue with timeout
                system_event = await asyncio.wait_for(
                    self.system_event_queue.get(),
                    timeout=1.0
                )

                # Send to Kafka
                success = await self.kafka_producer.send_system_event(system_event)

                if success:
                    self.stats["system_events_processed"] += 1
                else:
                    self.stats["events_failed"] += 1

                # Mark task as done
                self.system_event_queue.task_done()

            except TimeoutError:
                continue  # No events in queue
            except Exception as e:
                logger.error("System event processing error", error=str(e))
                await asyncio.sleep(0.1)

    async def _metrics_reporter(self):
        """Background task to report connector metrics."""
        while self.is_running:
            try:
                await asyncio.sleep(60.0)  # Report every minute

                current_time = time.time()
                time_diff = current_time - self.stats["last_reset"]

                # Calculate rates
                detection_rate = self.stats["detections_processed"] / time_diff if time_diff > 0 else 0
                camera_rate = self.stats["camera_events_processed"] / time_diff if time_diff > 0 else 0
                system_rate = self.stats["system_events_processed"] / time_diff if time_diff > 0 else 0

                # Get Kafka producer health
                kafka_health = self.kafka_producer.get_health_status() if self.kafka_producer else {}

                logger.info("Kafka connector metrics",
                           detection_rate_per_min=detection_rate * 60,
                           camera_rate_per_min=camera_rate * 60,
                           system_rate_per_min=system_rate * 60,
                           events_failed=self.stats["events_failed"],
                           events_filtered=self.stats["events_filtered"],
                           kafka_healthy=kafka_health.get("is_healthy", False),
                           detection_queue_size=self.detection_queue.qsize(),
                           camera_queue_size=self.camera_event_queue.qsize(),
                           system_queue_size=self.system_event_queue.qsize())

                # Reset counters
                self.stats.update({
                    "detections_processed": 0,
                    "camera_events_processed": 0,
                    "system_events_processed": 0,
                    "events_filtered": 0,
                    "events_failed": 0,
                    "last_reset": current_time,
                })

            except Exception as e:
                logger.error("Metrics reporter error", error=str(e))
                await asyncio.sleep(60.0)

    def get_health_status(self) -> dict[str, Any]:
        """Get connector health status and statistics."""
        return {
            "is_running": self.is_running,
            "kafka_producer_healthy": (
                self.kafka_producer.is_healthy if self.kafka_producer else False
            ),
            "queue_sizes": {
                "detection_events": self.detection_queue.qsize(),
                "camera_events": self.camera_event_queue.qsize(),
                "system_events": self.system_event_queue.qsize(),
            },
            "processing_stats": self.stats.copy(),
            "event_types_enabled": {
                "detection_events": self.enable_detection_events,
                "camera_events": self.enable_camera_events,
                "system_events": self.enable_system_events,
            },
            "kafka_producer_health": (
                self.kafka_producer.get_health_status() if self.kafka_producer else None
            ),
        }

    async def stop(self):
        """Stop the Kafka analytics connector."""
        logger.info("Stopping Kafka analytics connector...")

        self.is_running = False

        # Cancel background tasks
        for task in self.processing_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)

        # Wait for queues to empty (with timeout)
        try:
            await asyncio.wait_for(self.detection_queue.join(), timeout=5.0)
            await asyncio.wait_for(self.camera_event_queue.join(), timeout=5.0)
            await asyncio.wait_for(self.system_event_queue.join(), timeout=5.0)
        except TimeoutError:
            logger.warning("Timeout waiting for event queues to empty")

        # Stop Kafka producer
        if self.kafka_producer:
            await self.kafka_producer.stop()

        logger.info("Kafka analytics connector stopped")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Factory function for easy initialization
def create_kafka_analytics_connector(
    kafka_config: dict[str, Any] | None = None,
    event_filters: dict[str, Any] | None = None,
    **kwargs
) -> KafkaAnalyticsConnector:
    """Create and configure a Kafka analytics connector.

    Args:
        kafka_config: Optional Kafka configuration
        event_filters: Optional event filters
        **kwargs: Additional connector configuration

    Returns:
        Configured KafkaAnalyticsConnector instance
    """
    default_kafka_config = {
        "bootstrap_servers": ["localhost:9092"],
        "topic_prefix": "its-camera-ai",
        "partition_count": 12,
    }

    if kafka_config:
        default_kafka_config.update(kafka_config)

    default_filters = {
        "min_confidence": 0.5,
        "min_fps": 10,
    }

    if event_filters:
        default_filters.update(event_filters)

    return KafkaAnalyticsConnector(
        kafka_config=default_kafka_config,
        event_filters=default_filters,
        **kwargs
    )
