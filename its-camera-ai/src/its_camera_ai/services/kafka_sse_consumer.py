"""Kafka SSE Consumer Service for Real-Time Event Streaming.

This module provides a Kafka consumer service that bridges Kafka topics with 
Server-Sent Events (SSE) for real-time streaming of camera analytics data to 
web clients.

Features:
- Multi-topic Kafka consumption with intelligent routing
- Real-time event filtering and transformation for SSE
- High-performance connection management for 100+ concurrent clients
- Backpressure handling and connection limits
- Comprehensive error handling and automatic reconnection
- Integration with existing SSE broadcaster
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

try:
    from aiokafka import AIOKafkaConsumer
    from aiokafka.errors import KafkaError, KafkaTimeoutError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

from ..api.sse_broadcaster import SSEBroadcaster, SSEMessage
from ..core.exceptions import ServiceError
from ..core.logging import get_logger

logger = get_logger(__name__)


class EventFilter:
    """Event filter for client-specific filtering."""

    def __init__(
        self,
        camera_ids: list[str] | None = None,
        event_types: list[str] | None = None,
        zones: list[str] | None = None,
        min_confidence: float = 0.0,
        vehicle_types: list[str] | None = None,
    ):
        self.camera_ids = set(camera_ids) if camera_ids else None
        self.event_types = set(event_types) if event_types else None
        self.zones = set(zones) if zones else None
        self.min_confidence = min_confidence
        self.vehicle_types = set(vehicle_types) if vehicle_types else None

    def matches_event(self, event_data: dict[str, Any], event_type: str) -> bool:
        """Check if event matches this filter."""
        # Event type filter
        if self.event_types and event_type not in self.event_types:
            return False

        # Camera ID filter
        if self.camera_ids:
            camera_id = event_data.get("camera_id")
            if not camera_id or camera_id not in self.camera_ids:
                return False

        # Zone filter
        if self.zones:
            zone_id = event_data.get("zone_id")
            if not zone_id or zone_id not in self.zones:
                return False

        # Confidence filter for detection events
        if self.min_confidence > 0 and event_type == "detection_result":
            confidence_scores = event_data.get("confidence_scores", [])
            if not confidence_scores or max(confidence_scores, default=0) < self.min_confidence:
                return False

        # Vehicle type filter for detection events
        if self.vehicle_types and event_type == "detection_result":
            vehicle_classes = event_data.get("vehicle_classes", [])
            if not vehicle_classes or not any(vc in self.vehicle_types for vc in vehicle_classes):
                return False

        return True


class KafkaSSEConsumer:
    """Kafka consumer service for SSE real-time streaming."""

    def __init__(self, config: dict[str, Any], sse_broadcaster: SSEBroadcaster):
        """Initialize Kafka SSE consumer.
        
        Args:
            config: Consumer configuration dictionary
                - bootstrap_servers: List of Kafka broker addresses
                - topic_prefix: Prefix for topics (default: "its-camera-ai")
                - consumer_group_id: Consumer group ID (default: "sse-streaming")
                - auto_offset_reset: Offset reset policy (default: "latest")
                - enable_auto_commit: Auto-commit offsets (default: True)
                - max_poll_records: Max records per poll (default: 500)
                - session_timeout_ms: Session timeout (default: 30000)
                - heartbeat_interval_ms: Heartbeat interval (default: 3000)
            sse_broadcaster: SSE broadcaster instance
        """
        if not KAFKA_AVAILABLE:
            raise ServiceError("Kafka dependencies not available", service="kafka_sse_consumer")

        self.config = config
        self.sse_broadcaster = sse_broadcaster

        # Kafka configuration
        self.bootstrap_servers = config.get("bootstrap_servers", ["localhost:9092"])
        self.topic_prefix = config.get("topic_prefix", "its-camera-ai")
        self.consumer_group_id = config.get("consumer_group_id", "sse-streaming")

        # Optimized consumer configuration for high throughput
        self.consumer_config = {
            "bootstrap_servers": self.bootstrap_servers,
            "group_id": self.consumer_group_id,
            "auto_offset_reset": config.get("auto_offset_reset", "latest"),
            "enable_auto_commit": config.get("enable_auto_commit", True),
            "auto_commit_interval_ms": config.get("auto_commit_interval_ms", 3000),  # Faster commits
            "max_poll_records": config.get("max_poll_records", 1000),  # Increased for better throughput
            "session_timeout_ms": config.get("session_timeout_ms", 20000),  # Shorter timeout
            "heartbeat_interval_ms": config.get("heartbeat_interval_ms", 2000),  # More frequent heartbeats
            "fetch_min_bytes": config.get("fetch_min_bytes", 1024),  # Minimum fetch size
            "fetch_max_wait_ms": config.get("fetch_max_wait_ms", 500),  # Faster fetch
            "max_partition_fetch_bytes": config.get("max_partition_fetch_bytes", 2 * 1024 * 1024),  # 2MB
            "connections_max_idle_ms": config.get("connections_max_idle_ms", 300000),  # 5 min idle
            "value_deserializer": self._deserialize_value,
            "key_deserializer": lambda x: x.decode('utf-8') if x else None,
        }

        # Topics to consume
        self.topics = self._initialize_topics()

        # Consumer and state
        self.consumer: AIOKafkaConsumer | None = None
        self.is_running = False
        self.processing_tasks: list[asyncio.Task] = []

        # Performance tracking
        self.metrics = {
            "events_processed": defaultdict(int),
            "events_filtered": defaultdict(int),
            "events_failed": defaultdict(int),
            "processing_latencies": deque(maxlen=1000),
            "throughput_events_per_sec": 0.0,
            "last_processed_time": time.time(),
        }

        # Connection health
        self.is_healthy = False
        self.connection_errors = 0
        self.last_error_time = 0.0

        # Enhanced rate limiting with adaptive throttling
        self.rate_limiter = defaultdict(lambda: deque(maxlen=200))  # Increased tracking
        self.max_events_per_second = config.get("max_events_per_second", 100)  # Increased limit
        self.adaptive_rate_limiting = config.get("adaptive_rate_limiting", True)
        self.current_rate_limits = defaultdict(lambda: self.max_events_per_second)

        # Blosc compression support for event deserialization
        self.enable_blosc_decompression = config.get("enable_blosc_decompression", True)

        # Connection pool optimization
        self.parallel_processing = config.get("parallel_processing", True)
        self.max_concurrent_messages = config.get("max_concurrent_messages", 50)
        self.message_processing_semaphore = asyncio.Semaphore(self.max_concurrent_messages)

        # Event transformation callbacks
        self.event_transformers: dict[str, Callable] = {}

        logger.info("Kafka SSE consumer initialized",
                   bootstrap_servers=self.bootstrap_servers,
                   topic_prefix=self.topic_prefix,
                   consumer_group=self.consumer_group_id)

    def _initialize_topics(self) -> list[str]:
        """Initialize list of topics to consume."""
        return [
            f"{self.topic_prefix}.detections.vehicles",
            f"{self.topic_prefix}.violations.traffic",
            f"{self.topic_prefix}.incidents",
            f"{self.topic_prefix}.cameras.status",
            f"{self.topic_prefix}.cameras.health",
            f"{self.topic_prefix}.cameras.config",
            f"{self.topic_prefix}.system.alerts",
            f"{self.topic_prefix}.system.metrics",
            f"{self.topic_prefix}.analytics.traffic-flow",
            f"{self.topic_prefix}.analytics.zones",
            f"{self.topic_prefix}.analytics.speed",
        ]

    def _deserialize_value(self, value: bytes) -> dict[str, Any]:
        """Deserialize Kafka message value with blosc decompression support."""
        try:
            # Check for blosc compression header
            if self.enable_blosc_decompression and value.startswith(b"BLOSC_COMPRESSED:"):
                try:
                    # Extract original size and compressed data
                    header_size = len(b"BLOSC_COMPRESSED:") + 4
                    original_size = int.from_bytes(value[len(b"BLOSC_COMPRESSED:"):header_size], 'little')
                    compressed_data = value[header_size:]

                    # Decompress using blosc
                    import blosc

                    # Decompress to numpy array then back to bytes
                    decompressed_array = blosc.decompress(compressed_data)
                    json_data = bytes(decompressed_array)

                    logger.debug(f"Decompressed event: {len(value)} -> {len(json_data)} bytes")

                except Exception as e:
                    logger.warning(f"Blosc decompression failed, using raw data: {e}")
                    json_data = value
            else:
                json_data = value

            return json.loads(json_data.decode('utf-8'))

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error("Failed to deserialize message value", error=str(e))
            return {}

    def register_event_transformer(self, event_type: str, transformer: Callable[[dict], dict]):
        """Register event transformation callback.
        
        Args:
            event_type: Event type to transform
            transformer: Function that takes event data and returns transformed data
        """
        self.event_transformers[event_type] = transformer
        logger.debug("Registered event transformer", event_type=event_type)

    async def start(self):
        """Start the Kafka SSE consumer."""
        try:
            # Initialize Kafka consumer
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                **self.consumer_config
            )

            await self.consumer.start()

            # Start background processing tasks
            self.processing_tasks = [
                asyncio.create_task(self._consume_events()),
                asyncio.create_task(self._metrics_updater()),
                asyncio.create_task(self._health_monitor()),
            ]

            self.is_running = True
            self.is_healthy = True
            self.connection_errors = 0

            logger.info("Kafka SSE consumer started successfully")

        except Exception as e:
            self.is_healthy = False
            self.connection_errors += 1
            self.last_error_time = time.time()

            logger.error("Failed to start Kafka SSE consumer", error=str(e))
            raise ServiceError(f"Kafka SSE consumer startup failed: {e}", service="kafka_sse_consumer") from e

    async def _consume_events(self):
        """Optimized main event consumption loop with parallel processing."""
        logger.info("Optimized Kafka event consumption started")

        while self.is_running:
            try:
                # Poll for more messages with optimized settings
                msg_pack = await self.consumer.getmany(
                    timeout_ms=500,  # Faster polling
                    max_records=self.consumer_config["max_poll_records"]
                )

                if not msg_pack:
                    continue

                # Process messages in parallel if enabled
                if self.parallel_processing:
                    await self._process_messages_parallel(msg_pack)
                else:
                    await self._process_messages_sequential(msg_pack)

            except KafkaTimeoutError:
                continue  # Normal timeout, continue polling
            except KafkaError as e:
                logger.error("Kafka consumption error", error=str(e))
                self.connection_errors += 1
                self.last_error_time = time.time()

                # Check if we should mark as unhealthy
                if self.connection_errors > 10:
                    self.is_healthy = False

                await asyncio.sleep(1.0)  # Back off on errors

            except Exception as e:
                logger.error("Unexpected consumption error", error=str(e))
                await asyncio.sleep(0.5)

    async def _process_messages_parallel(self, msg_pack):
        """Process messages in parallel for higher throughput."""
        tasks = []

        for topic_partition, messages in msg_pack.items():
            topic = topic_partition.topic

            for message in messages:
                # Use semaphore to limit concurrent processing
                task = asyncio.create_task(
                    self._process_message_with_semaphore(topic, message)
                )
                tasks.append(task)

        # Process all messages concurrently
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_messages_sequential(self, msg_pack):
        """Process messages sequentially (fallback method)."""
        for topic_partition, messages in msg_pack.items():
            topic = topic_partition.topic

            for message in messages:
                try:
                    await self._process_message(topic, message)
                except Exception as e:
                    logger.error("Error processing message",
                               topic=topic,
                               offset=message.offset,
                               error=str(e))
                    self.metrics["events_failed"][topic] += 1

    async def _process_message_with_semaphore(self, topic: str, message):
        """Process individual message with semaphore limiting."""
        async with self.message_processing_semaphore:
            try:
                await self._process_message(topic, message)
            except Exception as e:
                logger.error("Error processing message",
                           topic=topic,
                           offset=message.offset,
                           error=str(e))
                self.metrics["events_failed"][topic] += 1

    async def _process_message(self, topic: str, message):
        """Process individual Kafka message."""
        start_time = time.time()

        try:
            # Extract event data
            event_data = message.value
            event_key = message.key
            headers = dict(message.headers or [])

            # Determine event type from topic and headers
            event_type = self._determine_event_type(topic, headers)

            # Transform event if transformer is registered
            if event_type in self.event_transformers:
                event_data = self.event_transformers[event_type](event_data)

            # Create SSE event based on type
            sse_event = await self._create_sse_event(event_type, event_data, headers)

            if sse_event:
                # Broadcast to SSE clients with filtering
                await self._broadcast_filtered_event(sse_event, event_type, event_data)

                # Update metrics
                self.metrics["events_processed"][topic] += 1
                processing_time = (time.time() - start_time) * 1000  # ms
                self.metrics["processing_latencies"].append(processing_time)

        except Exception as e:
            logger.error("Message processing failed",
                        topic=topic,
                        offset=message.offset,
                        error=str(e))
            self.metrics["events_failed"][topic] += 1

    def _determine_event_type(self, topic: str, headers: dict) -> str:
        """Determine event type from topic and headers."""
        # Check headers first
        if "event_type" in headers:
            return headers["event_type"].decode('utf-8') if isinstance(headers["event_type"], bytes) else headers["event_type"]

        # Fallback to topic-based mapping
        topic_mapping = {
            "detections.vehicles": "detection_result",
            "violations.traffic": "traffic_violation",
            "incidents": "incident_detection",
            "cameras.status": "status_change",
            "cameras.health": "health_update",
            "cameras.config": "configuration_change",
            "system.alerts": "system_alert",
            "system.metrics": "performance_metric",
            "analytics.traffic-flow": "traffic_flow",
            "analytics.zones": "zone_analytics",
            "analytics.speed": "speed_calculation",
        }

        for suffix, event_type in topic_mapping.items():
            if topic.endswith(suffix):
                return event_type

        return "unknown"

    async def _create_sse_event(self, event_type: str, event_data: dict, headers: dict) -> SSEMessage | None:
        """Create SSE message from Kafka event."""
        try:
            # Generate message ID
            timestamp = headers.get("timestamp", str(time.time() * 1000))
            message_id = f"{event_type}_{timestamp}"

            # Determine SSE event category
            if event_type in ["detection_result", "traffic_violation", "incident_detection"] or event_type in ["status_change", "health_update", "configuration_change"]:
                sse_event_type = "camera_update"
            elif event_type in ["system_alert", "performance_metric"]:
                sse_event_type = "system_update"
            elif event_type in ["traffic_flow", "zone_analytics", "speed_calculation"]:
                sse_event_type = "analytics_update"
            else:
                sse_event_type = "general_update"

            # Create SSE message
            return SSEMessage(
                event=sse_event_type,
                data={
                    "event_type": event_type,
                    "timestamp": time.time(),
                    **event_data
                },
                id=message_id,
                retry=5000  # 5 second retry
            )

        except Exception as e:
            logger.error("Failed to create SSE event", event_type=event_type, error=str(e))
            return None

    async def _broadcast_filtered_event(self, sse_event: SSEMessage, event_type: str, event_data: dict):
        """Broadcast SSE event with client-specific filtering."""
        try:
            # Get all active connections
            connections = self.sse_broadcaster.connections

            if not connections:
                return

            successful_broadcasts = 0
            filtered_broadcasts = 0
            failed_broadcasts = 0

            # Broadcast to each connection with individual filtering
            for connection_id, connection in connections.items():
                try:
                    # Apply connection-specific filters
                    if await self._should_send_to_connection(connection, event_type, event_data):
                        # Check rate limiting
                        if self._is_rate_limited(connection_id):
                            continue

                        # Send event to connection
                        success = await connection.send_event(sse_event)
                        if success:
                            successful_broadcasts += 1
                        else:
                            failed_broadcasts += 1
                    else:
                        filtered_broadcasts += 1
                        self.metrics["events_filtered"][event_type] += 1

                except Exception as e:
                    logger.debug("Error broadcasting to connection",
                               connection_id=connection_id,
                               error=str(e))
                    failed_broadcasts += 1

            # Log broadcast summary
            if successful_broadcasts > 0 or failed_broadcasts > 0:
                logger.debug("Event broadcast completed",
                           event_type=event_type,
                           successful=successful_broadcasts,
                           filtered=filtered_broadcasts,
                           failed=failed_broadcasts)

        except Exception as e:
            logger.error("Broadcast failed", event_type=event_type, error=str(e))

    async def _should_send_to_connection(self, connection, event_type: str, event_data: dict) -> bool:
        """Check if event should be sent to specific connection."""
        try:
            # Get connection filters
            filters = connection.filters or {}

            # Create event filter from connection filters
            event_filter = EventFilter(
                camera_ids=filters.get("camera_ids"),
                event_types=filters.get("event_types"),
                zones=filters.get("zones"),
                min_confidence=filters.get("min_confidence", 0.0),
                vehicle_types=filters.get("vehicle_types")
            )

            return event_filter.matches_event(event_data, event_type)

        except Exception as e:
            logger.debug("Filter check failed", error=str(e))
            return True  # Default to sending if filter check fails

    def _is_rate_limited(self, connection_id: str) -> bool:
        """Check if connection is rate limited with adaptive throttling."""
        now = time.time()
        current_limit = self.current_rate_limits[connection_id]

        # Clean old entries
        rate_queue = self.rate_limiter[connection_id]
        while rate_queue and now - rate_queue[0] > 1.0:  # 1 second window
            rate_queue.popleft()

        # Check current rate limit
        if len(rate_queue) >= current_limit:
            # Adaptive rate limiting: decrease limit if consistently hitting it
            if self.adaptive_rate_limiting and len(rate_queue) >= current_limit * 0.95:
                self.current_rate_limits[connection_id] = max(
                    current_limit * 0.9, self.max_events_per_second * 0.1
                )
                logger.debug(f"Reduced rate limit for {connection_id} to {self.current_rate_limits[connection_id]}")
            return True

        # Adaptive rate limiting: increase limit if under-utilized
        if self.adaptive_rate_limiting and len(rate_queue) < current_limit * 0.5:
            self.current_rate_limits[connection_id] = min(
                current_limit * 1.1, self.max_events_per_second
            )

        # Add current event
        rate_queue.append(now)
        return False

    async def _metrics_updater(self):
        """Background task to update performance metrics."""
        while self.is_running:
            try:
                await asyncio.sleep(30.0)  # Update every 30 seconds

                current_time = time.time()
                time_diff = current_time - self.metrics["last_processed_time"]

                # Calculate throughput
                total_processed = sum(self.metrics["events_processed"].values())
                if time_diff > 0:
                    self.metrics["throughput_events_per_sec"] = total_processed / time_diff

                # Calculate average latency
                avg_latency = (
                    sum(self.metrics["processing_latencies"]) / len(self.metrics["processing_latencies"])
                    if self.metrics["processing_latencies"] else 0
                )

                # Log metrics
                logger.info("Kafka SSE consumer metrics",
                           throughput_eps=self.metrics["throughput_events_per_sec"],
                           avg_latency_ms=avg_latency,
                           total_processed=total_processed,
                           total_failed=sum(self.metrics["events_failed"].values()),
                           total_filtered=sum(self.metrics["events_filtered"].values()),
                           connection_errors=self.connection_errors,
                           is_healthy=self.is_healthy,
                           active_connections=len(self.sse_broadcaster.connections))

                # Reset counters for next period
                self.metrics["events_processed"].clear()
                self.metrics["events_failed"].clear()
                self.metrics["events_filtered"].clear()
                self.metrics["last_processed_time"] = current_time

            except Exception as e:
                logger.error("Metrics updater error", error=str(e))
                await asyncio.sleep(30.0)

    async def _health_monitor(self):
        """Background task to monitor consumer health."""
        while self.is_running:
            try:
                await asyncio.sleep(10.0)  # Check every 10 seconds

                # Check consumer health
                if self.consumer and not self.consumer._closed:
                    # Reset connection errors if consumer is healthy
                    if self.connection_errors > 0:
                        self.connection_errors = max(0, self.connection_errors - 1)

                    if not self.is_healthy and self.connection_errors < 5:
                        self.is_healthy = True
                        logger.info("Consumer health restored")

                # Check if we've been unhealthy too long
                if not self.is_healthy and time.time() - self.last_error_time > 300:  # 5 minutes
                    logger.warning("Consumer unhealthy for extended period, attempting restart")
                    # Could trigger restart logic here

            except Exception as e:
                logger.error("Health monitor error", error=str(e))
                await asyncio.sleep(10.0)

    def get_health_status(self) -> dict[str, Any]:
        """Get consumer health status and metrics."""
        return {
            "is_healthy": self.is_healthy,
            "is_running": self.is_running,
            "connection_errors": self.connection_errors,
            "last_error_time": self.last_error_time,
            "metrics": {
                "throughput_events_per_sec": self.metrics["throughput_events_per_sec"],
                "total_events_processed": sum(self.metrics["events_processed"].values()),
                "total_events_failed": sum(self.metrics["events_failed"].values()),
                "total_events_filtered": sum(self.metrics["events_filtered"].values()),
                "avg_processing_latency_ms": (
                    sum(self.metrics["processing_latencies"]) / len(self.metrics["processing_latencies"])
                    if self.metrics["processing_latencies"] else 0
                ),
                "events_by_topic": dict(self.metrics["events_processed"]),
                "failures_by_topic": dict(self.metrics["events_failed"]),
            },
            "configuration": {
                "bootstrap_servers": self.bootstrap_servers,
                "consumer_group": self.consumer_group_id,
                "topics": self.topics,
                "max_events_per_second": self.max_events_per_second,
                "parallel_processing_enabled": self.parallel_processing,
                "max_concurrent_messages": self.max_concurrent_messages,
                "adaptive_rate_limiting_enabled": self.adaptive_rate_limiting,
                "blosc_decompression_enabled": self.enable_blosc_decompression,
            },
            "sse_broadcaster_stats": self.sse_broadcaster.get_stats()
        }

    async def stop(self):
        """Stop the Kafka SSE consumer."""
        logger.info("Stopping Kafka SSE consumer...")

        self.is_running = False

        # Cancel background tasks
        for task in self.processing_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)

        # Stop Kafka consumer
        if self.consumer:
            try:
                await self.consumer.stop()
            except Exception as e:
                logger.error("Error stopping Kafka consumer", error=str(e))

        self.is_healthy = False
        logger.info("Kafka SSE consumer stopped")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Factory function for easy initialization
def create_kafka_sse_consumer(
    config: dict[str, Any] | None = None,
    sse_broadcaster: SSEBroadcaster | None = None
) -> KafkaSSEConsumer:
    """Create and configure an optimized Kafka SSE consumer for high throughput.
    
    Args:
        config: Optional configuration dictionary
        sse_broadcaster: SSE broadcaster instance
        
    Returns:
        Configured KafkaSSEConsumer instance optimized for real-time streaming
    """
    default_config = {
        "bootstrap_servers": ["localhost:9092"],
        "topic_prefix": "its-camera-ai",
        "consumer_group_id": "sse-streaming",
        "auto_offset_reset": "latest",
        "enable_auto_commit": True,
        "max_poll_records": 1000,  # Increased for better throughput
        "max_events_per_second": 100,  # Increased rate limit
        "enable_blosc_decompression": True,  # Enable blosc decompression
        "parallel_processing": True,  # Enable parallel message processing
        "max_concurrent_messages": 50,  # Concurrent message processing limit
        "adaptive_rate_limiting": True,  # Enable adaptive rate limiting
        "fetch_min_bytes": 1024,  # Minimum fetch size for efficiency
        "fetch_max_wait_ms": 500,  # Faster message fetching
    }

    if config:
        default_config.update(config)

    if not sse_broadcaster:
        from ..api.sse_broadcaster import get_broadcaster
        sse_broadcaster = get_broadcaster()

    return KafkaSSEConsumer(default_config, sse_broadcaster)
