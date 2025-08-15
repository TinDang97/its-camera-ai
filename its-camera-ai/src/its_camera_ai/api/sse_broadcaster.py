"""
Server-Sent Events (SSE) broadcaster for real-time camera updates.

Provides real-time streaming of camera status, detection results, and system metrics
to connected clients via HTTP SSE connections.
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core.exceptions import ServiceError
from ..core.logging import get_logger

if TYPE_CHECKING:
    from ..services.kafka_sse_consumer import KafkaSSEConsumer

logger = get_logger(__name__)


class SSEMessage(BaseModel):
    """SSE message structure."""

    event: str
    data: dict[str, Any]
    id: str | None = None
    retry: int | None = None


class CameraEvent(BaseModel):
    """Camera-specific event data."""

    camera_id: str
    event_type: str
    timestamp: datetime
    data: dict[str, Any]


class SystemEvent(BaseModel):
    """System-wide event data."""

    event_type: str
    timestamp: datetime
    data: dict[str, Any]


class SSEConnection:
    """Represents an active SSE connection."""

    def __init__(self, connection_id: str, filters: dict[str, Any] | None = None):
        self.connection_id = connection_id
        self.filters = filters or {}
        self.created_at = datetime.now()
        self.last_ping = time.time()
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.closed = False

    def matches_filter(self, event: CameraEvent) -> bool:
        """Check if event matches connection filters."""
        if not self.filters:
            return True

        # Camera ID filter
        if (
            "camera_ids" in self.filters
            and event.camera_id not in self.filters["camera_ids"]
        ):
            return False

        # Event type filter
        if (
            "event_types" in self.filters
            and event.event_type not in self.filters["event_types"]
        ):
            return False

        # Zone filter
        if "zones" in self.filters and "zone_id" in event.data:
            if event.data["zone_id"] not in self.filters["zones"]:
                return False

        return True

    async def send_event(self, message: SSEMessage) -> bool:
        """Send event to this connection."""
        if self.closed:
            return False

        try:
            await asyncio.wait_for(self.queue.put(message), timeout=0.1)
            return True
        except (TimeoutError, asyncio.QueueFull):
            logger.warning(
                "Queue full for connection", connection_id=self.connection_id
            )
            return False

    async def get_event(self) -> SSEMessage | None:
        """Get next event for this connection."""
        if self.closed:
            return None

        try:
            return await asyncio.wait_for(self.queue.get(), timeout=30.0)
        except TimeoutError:
            # Send ping to keep connection alive
            return SSEMessage(
                event="ping", data={"timestamp": datetime.now().isoformat()}
            )

    def close(self):
        """Close this connection."""
        self.closed = True


class SSEBroadcaster:
    """Server-Sent Events broadcaster for real-time updates with Kafka integration."""

    def __init__(self, kafka_config: dict[str, Any] | None = None):
        self.connections: dict[str, SSEConnection] = {}
        self.event_history: list[CameraEvent] = []
        self.max_history_size = 1000
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_dropped": 0,
            "kafka_events_processed": 0,
        }

        # Kafka integration
        self.kafka_consumer: KafkaSSEConsumer | None = None
        self.kafka_config = kafka_config

        # Connection limits and rate limiting
        self.max_connections = 200  # Increased limit for production
        self.connection_rate_limits = defaultdict(list)  # Per-connection rate limiting
        self.global_rate_limit = deque(maxlen=1000)  # Global rate limiting
        self.max_global_events_per_second = 500

        # Start background tasks
        self._cleanup_task = None
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background tasks including cleanup and Kafka integration."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_connections())

        # Start Kafka integration if enabled
        self._kafka_integration_task = asyncio.create_task(
            self._start_kafka_integration()
        )

    async def _cleanup_connections(self):
        """Background task to clean up stale connections."""
        while True:
            try:
                current_time = time.time()
                stale_connections = []

                for connection_id, connection in self.connections.items():
                    # Mark stale connections (no activity for 60 seconds)
                    if current_time - connection.last_ping > 60:
                        stale_connections.append(connection_id)

                # Remove stale connections
                for connection_id in stale_connections:
                    await self.disconnect_client(connection_id)
                    logger.info(
                        "Cleaned up stale connection", connection_id=connection_id
                    )

                # Clean up event history
                if len(self.event_history) > self.max_history_size:
                    self.event_history = self.event_history[-self.max_history_size :]

                await asyncio.sleep(30)  # Run cleanup every 30 seconds

            except Exception as e:
                logger.error("Error in cleanup task", error=str(e))
                await asyncio.sleep(30)

    async def _start_kafka_integration(self):
        """Start Kafka consumer integration."""
        try:
            logger.info("Starting Kafka SSE integration")

            # Create Kafka consumer with this broadcaster
            from ..services.kafka_sse_consumer import create_kafka_sse_consumer

            self.kafka_consumer = create_kafka_sse_consumer(
                config=self.kafka_config, sse_broadcaster=self
            )

            # Start the consumer
            await self.kafka_consumer.start()

            logger.info("Kafka SSE integration started successfully")

        except Exception as e:
            logger.error("Failed to start Kafka integration", error=str(e))
            self.kafka_consumer = None

    async def connect_client(
        self, connection_id: str, filters: dict[str, Any] | None = None
    ) -> SSEConnection:
        """Connect a new client for SSE updates."""
        # Check connection limits
        if len(self.connections) >= self.max_connections:
            logger.warning(
                "Maximum connections reached, rejecting new connection",
                connection_id=connection_id,
                max_connections=self.max_connections,
            )
            raise ServiceError("Maximum connections reached", service="sse_broadcaster")

        connection = SSEConnection(connection_id, filters)
        self.connections[connection_id] = connection

        self.stats["total_connections"] += 1
        self.stats["active_connections"] = len(self.connections)

        logger.info(
            "SSE client connected",
            connection_id=connection_id,
            filters=filters,
            total_connections=self.stats["active_connections"],
        )

        # Send recent history if requested
        if filters and filters.get("include_history", False):
            recent_events = self.event_history[-50:]  # Last 50 events
            for event in recent_events:
                if connection.matches_filter(event):
                    message = SSEMessage(
                        event="camera_update",
                        data=event.model_dump(),
                        id=str(int(event.timestamp.timestamp() * 1000)),
                    )
                    await connection.send_event(message)

        return connection

    async def disconnect_client(self, connection_id: str):
        """Disconnect a client."""
        if connection_id in self.connections:
            connection = self.connections[connection_id]
            connection.close()
            del self.connections[connection_id]

            # Clean up rate limiting data
            if connection_id in self.connection_rate_limits:
                del self.connection_rate_limits[connection_id]

            self.stats["active_connections"] = len(self.connections)

            logger.info(
                "SSE client disconnected",
                connection_id=connection_id,
                active_connections=self.stats["active_connections"],
            )

    async def broadcast_camera_event(self, event: CameraEvent):
        """Broadcast a camera event to all matching connections."""
        if not self.connections:
            return

        # Add to history
        self.event_history.append(event)

        message = SSEMessage(
            event="camera_update",
            data=event.model_dump(),
            id=str(int(event.timestamp.timestamp() * 1000)),
        )

        # Send to all matching connections with rate limiting
        successful_sends = 0
        failed_connections = []
        rate_limited_connections = 0

        for connection_id, connection in self.connections.items():
            if connection.matches_filter(event):
                # Check rate limits
                if self._is_connection_rate_limited(connection_id):
                    rate_limited_connections += 1
                    continue

                success = await connection.send_event(message)
                if success:
                    successful_sends += 1
                    self._track_connection_event(connection_id)
                else:
                    failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect_client(connection_id)

        self.stats["messages_sent"] += successful_sends
        self.stats["messages_dropped"] += len(failed_connections)

        # Track global rate limiting
        self._track_global_event()

        logger.debug(
            "Camera event broadcasted",
            camera_id=event.camera_id,
            event_type=event.event_type,
            successful_sends=successful_sends,
            dropped=len(failed_connections),
            rate_limited=rate_limited_connections,
        )

    async def broadcast_system_event(self, event: SystemEvent):
        """Broadcast a system-wide event to all connections."""
        if not self.connections:
            return

        message = SSEMessage(
            event="system_update",
            data=event.model_dump(),
            id=str(int(event.timestamp.timestamp() * 1000)),
        )

        # Send to all connections with rate limiting
        successful_sends = 0
        failed_connections = []
        rate_limited_connections = 0

        for connection_id, connection in self.connections.items():
            # Check rate limits for system events (less strict)
            if self._is_connection_rate_limited(connection_id, max_per_second=20):
                rate_limited_connections += 1
                continue

            success = await connection.send_event(message)
            if success:
                successful_sends += 1
                self._track_connection_event(connection_id)
            else:
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect_client(connection_id)

        self.stats["messages_sent"] += successful_sends
        self.stats["messages_dropped"] += len(failed_connections)

        # Track global rate limiting
        self._track_global_event()

        logger.debug(
            "System event broadcasted",
            event_type=event.event_type,
            successful_sends=successful_sends,
            dropped=len(failed_connections),
            rate_limited=rate_limited_connections,
        )

    async def get_connection_stream(self, connection_id: str):
        """Get SSE stream for a specific connection."""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        try:
            while not connection.closed:
                # Get next event
                event = await connection.get_event()
                if event is None:
                    break

                # Update ping time
                connection.last_ping = time.time()

                # Format as SSE
                yield self._format_sse_message(event)

        except Exception as e:
            logger.error(
                "Error in SSE stream", connection_id=connection_id, error=str(e)
            )
        finally:
            await self.disconnect_client(connection_id)

    def _format_sse_message(self, message: SSEMessage) -> str:
        """Format message as SSE string."""
        lines = []

        if message.id:
            lines.append(f"id: {message.id}")
        if message.retry:
            lines.append(f"retry: {message.retry}")

        lines.append(f"event: {message.event}")

        # Format data (can be multiline)
        data_str = json.dumps(message.data, default=str)
        for line in data_str.split("\n"):
            lines.append(f"data: {line}")

        lines.append("")  # Empty line to end message
        return "\n".join(lines) + "\n"

    def _is_connection_rate_limited(
        self, connection_id: str, max_per_second: int = 10
    ) -> bool:
        """Check if a connection is rate limited."""
        now = time.time()

        # Get connection events
        events = self.connection_rate_limits[connection_id]

        # Remove old events (older than 1 second)
        while events and now - events[0] > 1.0:
            events.pop(0)

        # Check rate limit
        return len(events) >= max_per_second

    def _track_connection_event(self, connection_id: str):
        """Track an event for connection rate limiting."""
        self.connection_rate_limits[connection_id].append(time.time())

    def _track_global_event(self):
        """Track a global event for rate limiting."""
        self.global_rate_limit.append(time.time())

    def _is_global_rate_limited(self) -> bool:
        """Check if global rate limit is exceeded."""
        if len(self.global_rate_limit) < self.max_global_events_per_second:
            return False

        now = time.time()
        oldest_event = self.global_rate_limit[0]

        # Check if we're within the rate limit window
        return now - oldest_event < 1.0

    async def broadcast_kafka_event(self, sse_message: SSEMessage):
        """Broadcast event received from Kafka consumer."""
        if not self.connections:
            return

        # Check global rate limiting
        if self._is_global_rate_limited():
            logger.debug("Global rate limit exceeded, dropping Kafka event")
            self.stats["messages_dropped"] += 1
            return

        successful_sends = 0
        failed_connections = []

        # Send to all connections (Kafka events are pre-filtered)
        for connection_id, connection in self.connections.items():
            try:
                success = await connection.send_event(sse_message)
                if success:
                    successful_sends += 1
                    self._track_connection_event(connection_id)
                else:
                    failed_connections.append(connection_id)
            except Exception as e:
                logger.debug(
                    "Error sending Kafka event to connection",
                    connection_id=connection_id,
                    error=str(e),
                )
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect_client(connection_id)

        # Update stats
        self.stats["messages_sent"] += successful_sends
        self.stats["messages_dropped"] += len(failed_connections)
        self.stats["kafka_events_processed"] += 1

        # Track global event
        self._track_global_event()

        logger.debug(
            "Kafka event broadcasted",
            event_type=sse_message.event,
            successful_sends=successful_sends,
            dropped=len(failed_connections),
        )

    def get_stats(self) -> dict[str, Any]:
        """Get broadcaster statistics."""
        kafka_stats = {}
        if self.kafka_consumer:
            kafka_stats = self.kafka_consumer.get_health_status()

        return {
            **self.stats,
            "connection_details": [
                {
                    "connection_id": conn_id,
                    "created_at": conn.created_at.isoformat(),
                    "filters": conn.filters,
                    "queue_size": conn.queue.qsize(),
                }
                for conn_id, conn in self.connections.items()
            ],
            "history_size": len(self.event_history),
            "connection_limits": {
                "max_connections": self.max_connections,
                "current_connections": len(self.connections),
                "max_global_events_per_second": self.max_global_events_per_second,
            },
            "kafka_integration": {
                "enabled": self.kafka_enabled,
                "consumer_healthy": (
                    kafka_stats.get("is_healthy", False) if kafka_stats else False
                ),
                "consumer_stats": kafka_stats,
            },
        }


# Global broadcaster instance
_broadcaster: SSEBroadcaster | None = None


def get_broadcaster(kafka_config: dict[str, Any] | None = None) -> SSEBroadcaster:
    """Get global SSE broadcaster instance with optional Kafka integration."""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = SSEBroadcaster(kafka_config=kafka_config)
    return _broadcaster


def initialize_broadcaster_with_kafka(kafka_config: dict[str, Any]) -> SSEBroadcaster:
    """Initialize broadcaster with Kafka configuration."""
    global _broadcaster
    if _broadcaster is not None:
        logger.warning("Broadcaster already initialized, ignoring Kafka config")
        return _broadcaster

    _broadcaster = SSEBroadcaster(kafka_config=kafka_config)
    return _broadcaster


async def shutdown_broadcaster():
    """Shutdown the global broadcaster and cleanup resources."""
    global _broadcaster
    if _broadcaster and _broadcaster.kafka_consumer:
        try:
            await _broadcaster.kafka_consumer.stop()
            logger.info("SSE broadcaster Kafka integration stopped")
        except Exception as e:
            logger.error("Error stopping broadcaster Kafka integration", error=str(e))

    _broadcaster = None


async def create_sse_response(
    request: Request, connection_id: str, filters: dict[str, Any] | None = None
) -> StreamingResponse:
    """Create SSE streaming response for a client."""
    broadcaster = get_broadcaster()

    # Check if client disconnected
    async def check_disconnect():
        while True:
            if await request.is_disconnected():
                await broadcaster.disconnect_client(connection_id)
                break
            await asyncio.sleep(1)

    # Start disconnect monitoring
    asyncio.create_task(check_disconnect())

    # Connect client and get stream
    await broadcaster.connect_client(connection_id, filters)

    return StreamingResponse(
        broadcaster.get_connection_stream(connection_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control",
        },
    )
