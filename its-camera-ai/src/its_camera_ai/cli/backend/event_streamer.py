"""Event streaming for real-time monitoring.

Provides Server-Sent Events (SSE) and WebSocket connectivity for real-time
event streaming and monitoring of backend services.
"""

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from aiohttp import ClientSession, ClientWebSocketResponse, WSMsgType

from ...core.config import Settings, get_settings
from ...core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StreamEvent:
    """Represents a streaming event."""
    event_type: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    id: str | None = None


class EventStreamer:
    """Real-time event streaming for monitoring.

    Features:
    - Server-Sent Events (SSE) streaming
    - WebSocket connections
    - Event filtering and routing
    - Connection management with reconnection
    - Event buffering and replay
    """

    def __init__(self, settings: Settings = None):
        """Initialize event streamer.

        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self.base_url = f"http://{self.settings.api_host}:{self.settings.api_port}"

        # Connection management
        self._session: ClientSession | None = None
        self._websocket: ClientWebSocketResponse | None = None
        self._connected = False

        # Event handlers
        self._event_handlers: dict[str, list[Callable]] = {}

        # Event buffering
        self._event_buffer: list[StreamEvent] = []
        self._buffer_size = 1000

        # Reconnection settings
        self._reconnect_delay = 5.0
        self._max_reconnect_attempts = 10
        self._reconnect_task: asyncio.Task | None = None

        logger.info("Event streamer initialized")

    async def connect(self, timeout: float = 10.0) -> None:
        """Establish connection for event streaming.

        Args:
            timeout: Connection timeout in seconds
        """
        if self._connected:
            return

        try:
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                keepalive_timeout=30
            )

            timeout_config = aiohttp.ClientTimeout(total=timeout)

            self._session = ClientSession(
                connector=connector,
                timeout=timeout_config,
                headers={"User-Agent": "ITS-Camera-AI-CLI"}
            )

            self._connected = True
            logger.info("Event streamer connected")

        except Exception as e:
            logger.error(f"Failed to connect event streamer: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect event streamer."""
        self._connected = False

        # Stop reconnection task
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconnect_task
            self._reconnect_task = None

        # Close WebSocket
        if self._websocket and not self._websocket.closed:
            await self._websocket.close()
            self._websocket = None

        # Close session
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        logger.info("Event streamer disconnected")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register event handler.

        Args:
            event_type: Type of event to handle
            handler: Async function to handle events
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []

        self._event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")

    def unregister_handler(self, event_type: str, handler: Callable) -> None:
        """Unregister event handler.

        Args:
            event_type: Type of event
            handler: Handler function to remove
        """
        if event_type in self._event_handlers:
            try:
                self._event_handlers[event_type].remove(handler)
                logger.info(f"Unregistered handler for event type: {event_type}")
            except ValueError:
                logger.warning(f"Handler not found for event type: {event_type}")

    async def _dispatch_event(self, event: StreamEvent) -> None:
        """Dispatch event to registered handlers.

        Args:
            event: Event to dispatch
        """
        # Add to buffer
        self._event_buffer.append(event)
        if len(self._event_buffer) > self._buffer_size:
            self._event_buffer.pop(0)  # Remove oldest event

        # Dispatch to handlers
        handlers = self._event_handlers.get(event.event_type, [])
        handlers.extend(self._event_handlers.get("*", []))  # Wildcard handlers

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler for {event.event_type}: {e}")

    async def stream_sse(
        self,
        endpoint: str = "/api/v1/events/stream",
        params: dict[str, Any] | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream Server-Sent Events.

        Args:
            endpoint: SSE endpoint to connect to
            params: Query parameters

        Yields:
            StreamEvent: Events from the stream
        """
        if not self._session:
            await self.connect()

        url = f"{self.base_url}{endpoint}"

        try:
            async with self._session.get(
                url,
                params=params,
                headers={"Accept": "text/event-stream"}
            ) as response:
                if response.status != 200:
                    raise Exception(f"SSE connection failed: {response.status}")

                logger.info(f"SSE stream connected to {url}")

                async for line in response.content:
                    line = line.decode('utf-8').strip()

                    if not line or line.startswith(':'):
                        continue  # Skip empty lines and comments

                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])  # Remove 'data: ' prefix
                            event = StreamEvent(
                                event_type=data.get('type', 'unknown'),
                                data=data.get('data', {}),
                                source=data.get('source', 'sse'),
                                id=data.get('id'),
                                timestamp=data.get('timestamp', time.time())
                            )

                            yield event
                            await self._dispatch_event(event)

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse SSE data: {e}")

        except Exception as e:
            logger.error(f"SSE streaming error: {e}")
            raise

    async def connect_websocket(
        self,
        endpoint: str = "/api/v1/events/ws",
        params: dict[str, Any] | None = None
    ) -> None:
        """Connect to WebSocket endpoint.

        Args:
            endpoint: WebSocket endpoint
            params: Query parameters
        """
        if not self._session:
            await self.connect()

        ws_url = f"ws://{self.settings.api_host}:{self.settings.api_port}{endpoint}"

        try:
            self._websocket = await self._session.ws_connect(
                ws_url,
                params=params,
                heartbeat=30  # Send ping every 30 seconds
            )

            logger.info(f"WebSocket connected to {ws_url}")

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def listen_websocket(self) -> AsyncGenerator[StreamEvent, None]:
        """Listen for WebSocket messages.

        Yields:
            StreamEvent: Events from WebSocket
        """
        if not self._websocket:
            raise Exception("WebSocket not connected")

        try:
            async for msg in self._websocket:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        event = StreamEvent(
                            event_type=data.get('type', 'unknown'),
                            data=data.get('data', {}),
                            source=data.get('source', 'websocket'),
                            id=data.get('id'),
                            timestamp=data.get('timestamp', time.time())
                        )

                        yield event
                        await self._dispatch_event(event)

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse WebSocket message: {e}")

                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._websocket.exception()}")
                    break

                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
                    logger.info("WebSocket connection closed")
                    break

        except Exception as e:
            logger.error(f"WebSocket listening error: {e}")
            raise

    async def send_websocket_message(self, message: dict[str, Any]) -> None:
        """Send message via WebSocket.

        Args:
            message: Message to send
        """
        if not self._websocket or self._websocket.closed:
            raise Exception("WebSocket not connected")

        try:
            await self._websocket.send_str(json.dumps(message))
            logger.debug(f"Sent WebSocket message: {message.get('type', 'unknown')}")

        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            raise

    async def stream_system_events(self) -> AsyncGenerator[StreamEvent, None]:
        """Stream system events (metrics, health, alerts).

        Yields:
            StreamEvent: System events
        """
        params = {
            "types": "system,health,metrics,alerts",
            "format": "json"
        }

        async for event in self.stream_sse(
            endpoint="/api/v1/system/events",
            params=params
        ):
            yield event

    async def stream_camera_events(self, camera_ids: list[str] | None = None) -> AsyncGenerator[StreamEvent, None]:
        """Stream camera-related events.

        Args:
            camera_ids: List of camera IDs to filter by

        Yields:
            StreamEvent: Camera events
        """
        params = {
            "types": "camera,detection,alert",
            "format": "json"
        }

        if camera_ids:
            params["camera_ids"] = ",".join(camera_ids)

        async for event in self.stream_sse(
            endpoint="/api/v1/cameras/events",
            params=params
        ):
            yield event

    async def stream_analytics_events(self) -> AsyncGenerator[StreamEvent, None]:
        """Stream analytics and ML events.

        Yields:
            StreamEvent: Analytics events
        """
        params = {
            "types": "analytics,ml,inference,training",
            "format": "json"
        }

        async for event in self.stream_sse(
            endpoint="/api/v1/analytics/events",
            params=params
        ):
            yield event

    async def start_monitoring(
        self,
        event_types: list[str] | None = None,
        auto_reconnect: bool = True
    ) -> None:
        """Start continuous event monitoring.

        Args:
            event_types: List of event types to monitor
            auto_reconnect: Whether to auto-reconnect on failure
        """
        logger.info("Starting event monitoring")

        params = {}
        if event_types:
            params["types"] = ",".join(event_types)

        while self._connected:
            try:
                async for _event in self.stream_sse(
                    endpoint="/api/v1/events/stream",
                    params=params
                ):
                    # Events are automatically dispatched in stream_sse
                    pass

            except Exception as e:
                logger.error(f"Event monitoring error: {e}")

                if auto_reconnect and self._connected:
                    logger.info(f"Reconnecting in {self._reconnect_delay} seconds...")
                    await asyncio.sleep(self._reconnect_delay)

                    # Exponential backoff (up to 60 seconds)
                    self._reconnect_delay = min(self._reconnect_delay * 1.5, 60)
                else:
                    break

        logger.info("Event monitoring stopped")

    def get_recent_events(
        self,
        event_type: str | None = None,
        limit: int = 100
    ) -> list[StreamEvent]:
        """Get recent events from buffer.

        Args:
            event_type: Filter by event type
            limit: Maximum number of events to return

        Returns:
            List of recent events
        """
        events = self._event_buffer.copy()

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Return most recent events first
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def clear_event_buffer(self) -> None:
        """Clear the event buffer."""
        self._event_buffer.clear()
        logger.info("Event buffer cleared")

    async def wait_for_event(
        self,
        event_type: str,
        timeout: float = 30.0,
        condition: Callable[[StreamEvent], bool] | None = None
    ) -> StreamEvent | None:
        """Wait for a specific event.

        Args:
            event_type: Event type to wait for
            timeout: Timeout in seconds
            condition: Optional condition function

        Returns:
            Matching event or None if timeout
        """
        event_received = asyncio.Event()
        received_event = None

        async def handler(event: StreamEvent) -> None:
            nonlocal received_event
            if condition is None or condition(event):
                received_event = event
                event_received.set()

        # Register temporary handler
        self.register_handler(event_type, handler)

        try:
            # Wait for event or timeout
            await asyncio.wait_for(event_received.wait(), timeout=timeout)
            return received_event

        except TimeoutError:
            logger.warning(f"Timeout waiting for event type: {event_type}")
            return None

        finally:
            # Clean up handler
            self.unregister_handler(event_type, handler)

    def get_connection_status(self) -> dict[str, Any]:
        """Get connection status information.

        Returns:
            Connection status dictionary
        """
        return {
            "connected": self._connected,
            "session_open": self._session is not None and not self._session.closed,
            "websocket_connected": (
                self._websocket is not None and not self._websocket.closed
            ),
            "event_handlers": {
                event_type: len(handlers)
                for event_type, handlers in self._event_handlers.items()
            },
            "buffer_size": len(self._event_buffer),
            "reconnect_delay": self._reconnect_delay,
            "base_url": self.base_url
        }
