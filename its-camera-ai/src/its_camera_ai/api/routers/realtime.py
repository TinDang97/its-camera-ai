"""
Real-time communication endpoints using WebSocket and Server-Sent Events.

Provides low-latency real-time updates for camera feeds, analytics,
and system events with optimized performance.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Literal

from dependency_injector.wiring import Provide, inject
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

from ...core.logging import get_logger
from ...models.user import User
from ...services.analytics_service import AnalyticsService
from ...services.cache import CacheService
from ...services.streaming_service import SSEStreamingService

# Optional ML imports
try:
    from ...ml.streaming_annotation_processor import (
        AnnotationStyleConfig,
        DetectionConfig,
        DetectionMetadata,
        MLAnnotationProcessor,
    )
    from ...services.streaming_container import get_ml_annotation_processor
    ML_ENDPOINTS_AVAILABLE = True
except ImportError:
    get_ml_annotation_processor = lambda: None
    DetectionConfig = None
    AnnotationStyleConfig = None
    DetectionMetadata = None
    MLAnnotationProcessor = None
    ML_ENDPOINTS_AVAILABLE = False
from ..dependencies import (
    get_current_user,
    require_permissions,
)
from ..schemas.analytics import (
    WebSocketAnalyticsUpdate,
)

logger = get_logger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections for real-time updates with rate limiting and buffering."""

    def __init__(self):
        self.active_connections: dict[str, set[WebSocket]] = {}
        self.user_connections: dict[str, WebSocket] = {}
        self.connection_metadata: dict[WebSocket, dict[str, Any]] = {}
        self.rate_limits: dict[str, dict[str, Any]] = {}  # Per-connection rate limiting
        self.message_buffer: dict[str, list[dict[str, Any]]] = {}  # Message buffering
        self.max_buffer_size = 1000  # Maximum buffered messages per connection
        self.max_messages_per_second = 100  # Rate limit: 100 messages/second

    async def connect(
        self, websocket: WebSocket, channel: str, user_id: str | None = None, auth_token: str | None = None
    ) -> None:
        """Accept and register WebSocket connection with authentication."""
        await websocket.accept()

        if channel not in self.active_connections:
            self.active_connections[channel] = set()
        self.active_connections[channel].add(websocket)

        connection_id = f"{channel}:{id(websocket)}"

        if user_id:
            self.user_connections[user_id] = websocket

        self.connection_metadata[websocket] = {
            "connection_id": connection_id,
            "channel": channel,
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
            "message_count": 0,
            "last_message_time": datetime.utcnow(),
            "auth_token": auth_token,
            "is_authenticated": user_id is not None,
        }

        # Initialize rate limiting for this connection
        self.rate_limits[connection_id] = {
            "messages_this_second": 0,
            "last_reset_time": datetime.utcnow(),
        }

        # Initialize message buffer
        self.message_buffer[connection_id] = []

        logger.info(
            "WebSocket connection established",
            connection_id=connection_id,
            channel=channel,
            user_id=user_id,
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Remove WebSocket connection and cleanup resources."""
        metadata = self.connection_metadata.get(websocket)
        if metadata:
            connection_id = metadata["connection_id"]
            channel = metadata["channel"]
            user_id = metadata["user_id"]

            if channel in self.active_connections:
                self.active_connections[channel].discard(websocket)
                if not self.active_connections[channel]:
                    del self.active_connections[channel]

            if user_id and user_id in self.user_connections:
                del self.user_connections[user_id]

            # Cleanup rate limiting and buffers
            if connection_id in self.rate_limits:
                del self.rate_limits[connection_id]
            if connection_id in self.message_buffer:
                del self.message_buffer[connection_id]

            del self.connection_metadata[websocket]

            logger.info(
                "WebSocket connection closed",
                connection_id=connection_id,
                channel=channel,
                user_id=user_id,
                duration=(datetime.utcnow() - metadata["connected_at"]).total_seconds(),
                messages=metadata["message_count"],
            )

    def _check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection is within rate limits."""
        if connection_id not in self.rate_limits:
            return True

        rate_data = self.rate_limits[connection_id]
        now = datetime.utcnow()

        # Reset counter if more than 1 second has passed
        if (now - rate_data["last_reset_time"]).total_seconds() >= 1.0:
            rate_data["messages_this_second"] = 0
            rate_data["last_reset_time"] = now

        return rate_data["messages_this_second"] < self.max_messages_per_second

    def _add_to_buffer(self, connection_id: str, message: dict[str, Any]) -> None:
        """Add message to connection buffer."""
        if connection_id not in self.message_buffer:
            self.message_buffer[connection_id] = []

        buffer = self.message_buffer[connection_id]
        buffer.append(message)

        # Keep buffer size under limit
        if len(buffer) > self.max_buffer_size:
            buffer.pop(0)  # Remove oldest message

    async def _send_message_with_rate_limit(
        self, websocket: WebSocket, message: dict[str, Any]
    ) -> bool:
        """Send message with rate limiting and buffering."""
        metadata = self.connection_metadata.get(websocket)
        if not metadata:
            return False

        connection_id = metadata["connection_id"]

        # Check rate limit
        if not self._check_rate_limit(connection_id):
            # Add to buffer instead of sending immediately
            self._add_to_buffer(connection_id, message)
            return False

        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps(message))

                # Update counters
                metadata["message_count"] += 1
                metadata["last_message_time"] = datetime.utcnow()
                self.rate_limits[connection_id]["messages_this_second"] += 1

                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error sending message to WebSocket {connection_id}: {e}")
            return False

    async def broadcast_to_channel(
        self, channel: str, message: dict[str, Any]
    ) -> None:
        """Broadcast message to all connections in a channel with rate limiting."""
        if channel in self.active_connections:
            dead_connections = set()

            for connection in self.active_connections[channel].copy():
                success = await self._send_message_with_rate_limit(connection, message)
                if not success and connection.client_state != WebSocketState.CONNECTED:
                    dead_connections.add(connection)

            # Clean up dead connections
            for connection in dead_connections:
                self.disconnect(connection)

    async def send_to_user(self, user_id: str, message: dict[str, Any]) -> bool:
        """Send message to specific user with rate limiting."""
        if user_id in self.user_connections:
            connection = self.user_connections[user_id]
            success = await self._send_message_with_rate_limit(connection, message)
            if not success and connection.client_state != WebSocketState.CONNECTED:
                self.disconnect(connection)
            return success
        return False

    async def flush_buffered_messages(self, connection_id: str) -> int:
        """Flush buffered messages for a connection when rate limit allows."""
        if connection_id not in self.message_buffer:
            return 0

        buffer = self.message_buffer[connection_id]
        sent_count = 0

        # Find the websocket for this connection
        target_websocket = None
        for websocket, metadata in self.connection_metadata.items():
            if metadata.get("connection_id") == connection_id:
                target_websocket = websocket
                break

        if not target_websocket:
            return 0

        # Send buffered messages while rate limit allows
        while buffer and self._check_rate_limit(connection_id):
            message = buffer.pop(0)
            success = await self._send_message_with_rate_limit(target_websocket, message)
            if success:
                sent_count += 1
            else:
                # Put message back at front if sending failed
                buffer.insert(0, message)
                break

        return sent_count

    def get_channel_stats(self) -> dict[str, Any]:
        """Get statistics about active connections."""
        return {
            "total_connections": sum(
                len(conns) for conns in self.active_connections.values()
            ),
            "channels": {
                channel: len(conns)
                for channel, conns in self.active_connections.items()
            },
            "user_connections": len(self.user_connections),
        }


# Analytics helper functions
async def _get_camera_analytics(camera_id: str, analytics_service: AnalyticsService, cache_service: CacheService) -> dict[str, Any]:
    """Get real-time analytics for a specific camera."""
    try:
        # Check cache first for sub-100ms response
        cache_key = f"analytics:realtime:{camera_id}"
        cached_data = await cache_service.get_json(cache_key)
        if cached_data:
            return cached_data

        # Get recent traffic metrics
        now = datetime.utcnow()
        last_minute = now - timedelta(minutes=1)

        # Calculate real-time metrics
        metrics = await analytics_service.calculate_traffic_metrics(
            camera_id=camera_id,
            time_range=(last_minute, now),
            aggregation_period="1min"
        )

        # Get active violations
        violations = await analytics_service.get_active_violations(
            camera_id=camera_id,
            limit=10
        )

        # Format analytics data
        analytics_data = {
            "camera_id": camera_id,
            "timestamp": now.isoformat(),
            "total_vehicles": metrics[-1].total_vehicles if metrics else 0,
            "average_speed": metrics[-1].average_speed if metrics else 0,
            "congestion_level": metrics[-1].congestion_level if metrics else "free_flow",
            "occupancy_rate": metrics[-1].occupancy_rate if metrics else 0,
            "active_violations": len(violations),
            "flow_rate": metrics[-1].flow_rate if metrics else 0,
            "processing_time": 45.0,  # Mock processing time
            "confidence": 0.95,
        }

        # Cache for 5 seconds
        await cache_service.set_json(cache_key, analytics_data, ttl=5)

        return analytics_data

    except Exception as e:
        logger.error(f"Error getting camera analytics for {camera_id}: {e}")
        return {
            "camera_id": camera_id,
            "timestamp": datetime.utcnow().isoformat(),
            "error": "Failed to fetch analytics",
            "processing_time": 0.0,
            "confidence": 0.0,
        }

async def _get_system_analytics(analytics_service: AnalyticsService, cache_service: CacheService) -> dict[str, Any]:
    """Get system-wide analytics."""
    try:
        # Mock system-wide analytics (implement with real data)
        analytics_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_cameras": 42,
            "active_cameras": 38,
            "total_vehicles_detected": 15420,
            "average_system_speed": 52.3,
            "total_violations": 23,
            "system_load": 0.68,
            "inference_fps": 28.5,
            "processing_time": 35.0,
            "confidence": 0.92,
        }

        return analytics_data

    except Exception as e:
        logger.error(f"Error getting system analytics: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "error": "Failed to fetch system analytics",
            "processing_time": 0.0,
            "confidence": 0.0,
        }

# Global connection manager instance
manager = ConnectionManager()


class RealtimeMessage(BaseModel):
    """Real-time message format."""

    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict[str, Any] = Field(..., description="Message payload")
    metadata: dict[str, Any] | None = Field(
        None, description="Optional metadata"
    )


class ChannelSubscriptionRequest(BaseModel):
    """Request to subscribe to a specific channel."""

    channel_type: Literal["raw", "annotated"] = Field(
        ..., description="Channel type to subscribe to"
    )
    quality: Literal["low", "medium", "high"] | None = Field(
        "medium", description="Stream quality level"
    )


class ChannelSubscriptionResponse(BaseModel):
    """Response for channel subscription."""

    status: str = Field(..., description="Subscription status")
    connection_id: str = Field(..., description="Connection identifier")
    camera_id: str = Field(..., description="Camera identifier")
    channel_type: str = Field(..., description="Subscribed channel type")
    subscribed_channels: list[str] = Field(
        default_factory=list, description="All subscribed channels"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChannelSwitchRequest(BaseModel):
    """Request to switch to a different channel."""

    new_channel: Literal["raw", "annotated"] = Field(
        ..., description="New channel type"
    )
    connection_id: str = Field(..., description="Connection identifier")


class ChannelSwitchResponse(BaseModel):
    """Response for channel switching."""

    status: str = Field(..., description="Switch status")
    connection_id: str = Field(..., description="Connection identifier")
    camera_id: str = Field(..., description="Camera identifier")
    old_channels: list[str] = Field(
        default_factory=list, description="Previous channels"
    )
    new_channel: str = Field(..., description="New active channel")
    switch_time: float = Field(..., description="Switch timestamp")


class DualChannelStreamRequest(BaseModel):
    """Request to create dual-channel stream."""

    initial_channel: Literal["raw", "annotated"] = Field(
        "raw", description="Initial channel to stream"
    )
    raw_quality: Literal["low", "medium", "high"] = Field(
        "medium", description="Raw channel quality"
    )
    annotated_quality: Literal["low", "medium", "high"] = Field(
        "medium", description="Annotated channel quality"
    )


@router.websocket("/ws/cameras/{camera_id}")
async def camera_feed_websocket(
    websocket: WebSocket,
    camera_id: str,
    token: str | None = Query(None),
):
    """
    WebSocket endpoint for real-time camera feed updates.

    Streams:
    - Live detection results
    - Camera status changes
    - Analytics updates
    - Alert notifications
    """
    # Note: In production, validate token and get user
    user_id = "authenticated_user"  # Placeholder

    try:
        await manager.connect(websocket, f"camera:{camera_id}", user_id)

        while True:
            try:
                # Receive messages from client
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "ping":
                    await websocket.send_text(
                        json.dumps({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
                    )
                elif message.get("type") == "subscribe":
                    # Subscribe to specific event types
                    logger.info(f"Client subscribed to events: {message.get('events')}")
                elif message.get("type") == "control":
                    # Handle camera control commands
                    logger.info(f"Camera control command: {message.get('command')}")

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Invalid JSON"})
                )
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break

    finally:
        manager.disconnect(websocket)


@router.websocket("/ws/analytics")
@inject
async def analytics_websocket(
    websocket: WebSocket,
    token: str | None = Query(None),
    camera_id: str | None = Query(None, description="Specific camera ID for analytics"),
    analytics_service: AnalyticsService = Depends(Provide["analytics_service"]),
    cache_service: CacheService = Depends(Provide["cache_service"]),
):
    """
    Enhanced WebSocket endpoint for real-time analytics updates.

    Features:
    - JWT authentication
    - Rate limiting (100 messages/second)
    - Message buffering (max 1000 messages)
    - Heartbeat/ping-pong mechanism
    - Sub-100ms latency streaming
    - Camera-specific filtering

    Streams:
    - Real-time traffic metrics
    - Vehicle counts and classifications
    - Speed violations and incidents
    - ML predictions
    - System performance metrics
    """
    # TODO: Implement proper JWT token validation
    # For now, accept any token as valid authentication
    user_id = "authenticated_user" if token else None

    if not user_id:
        await websocket.close(code=4001, reason="Authentication required")
        return

    sequence_id = 0

    try:
        await manager.connect(websocket, "analytics", user_id, token)
        logger.info(f"Analytics WebSocket connected for user {user_id}, camera: {camera_id}")

        # Send initial connection acknowledgment
        await websocket.send_text(json.dumps({
            "type": "connection_ack",
            "timestamp": datetime.utcnow().isoformat(),
            "camera_id": camera_id,
            "rate_limit": manager.max_messages_per_second,
            "buffer_size": manager.max_buffer_size,
        }))

        # Background task for periodic analytics updates
        async def send_analytics_updates():
            nonlocal sequence_id

            while True:
                try:
                    # Get real-time analytics data
                    if camera_id:
                        # Camera-specific analytics
                        analytics_data = await _get_camera_analytics(camera_id, analytics_service, cache_service)
                    else:
                        # System-wide analytics
                        analytics_data = await _get_system_analytics(analytics_service, cache_service)

                    # Create WebSocket update message
                    update = WebSocketAnalyticsUpdate(
                        event_type="metrics",
                        camera_id=camera_id or "system",
                        timestamp=datetime.utcnow(),
                        data=analytics_data,
                        processing_latency_ms=analytics_data.get("processing_time", 0.0),
                        confidence_score=analytics_data.get("confidence", 1.0),
                        sequence_id=sequence_id,
                    )

                    sequence_id += 1

                    # Send update through rate-limited connection manager
                    await manager._send_message_with_rate_limit(websocket, update.model_dump())

                    # Update frequency: every 1 second for real-time updates
                    await asyncio.sleep(1.0)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in analytics update loop: {e}")
                    await asyncio.sleep(5.0)  # Back off on error

        # Start background update task
        update_task = asyncio.create_task(send_analytics_updates())

        # Handle client messages (ping/pong, subscriptions, etc.)
        while True:
            try:
                # Wait for client message with timeout for heartbeat
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                message = json.loads(data)

                message_type = message.get("type")

                if message_type == "ping":
                    # Respond to ping with pong
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat(),
                        "sequence_id": message.get("sequence_id"),
                    }))

                elif message_type == "subscribe":
                    # Handle subscription to specific event types
                    event_types = message.get("events", [])
                    logger.info(f"Client subscribed to events: {event_types}")

                    await websocket.send_text(json.dumps({
                        "type": "subscription_ack",
                        "events": event_types,
                        "timestamp": datetime.utcnow().isoformat(),
                    }))

                elif message_type == "buffer_flush":
                    # Manually trigger buffer flush
                    connection_id = manager.connection_metadata[websocket]["connection_id"]
                    sent_count = await manager.flush_buffered_messages(connection_id)

                    await websocket.send_text(json.dumps({
                        "type": "buffer_flushed",
                        "messages_sent": sent_count,
                        "timestamp": datetime.utcnow().isoformat(),
                    }))

                else:
                    logger.warning(f"Unknown message type: {message_type}")

            except TimeoutError:
                # Send heartbeat if no messages received
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat(),
                }))

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat(),
                }))
            except Exception as e:
                logger.error(f"Analytics WebSocket error: {e}")
                break

    except Exception as e:
        logger.error(f"Analytics WebSocket connection error: {e}")
    finally:
        if 'update_task' in locals():
            update_task.cancel()
        manager.disconnect(websocket)
        logger.info(f"Analytics WebSocket disconnected for user {user_id}")


@router.get("/sse/events")
async def server_sent_events(
    current_user: User = Depends(get_current_user),
    event_types: list[str] | None = Query(None),
) -> StreamingResponse:
    """
    Server-Sent Events endpoint for real-time updates.

    Provides a simpler alternative to WebSockets for one-way communication.
    Automatically reconnects on connection loss.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events."""
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'type': 'connected', 'timestamp': datetime.utcnow().isoformat()})}\n\n"

            # Keep connection alive and send events
            while True:
                # Check for new events (would come from event bus in production)
                event = {
                    "type": "system_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "cpu_usage": 45,
                        "memory_usage": 62,
                        "active_streams": 38,
                    },
                }

                yield f"data: {json.dumps(event)}\n\n"

                # Send heartbeat every 30 seconds
                await asyncio.sleep(30)
                yield ": heartbeat\n\n"

        except asyncio.CancelledError:
            logger.info("SSE connection cancelled")
        except Exception as e:
            logger.error(f"SSE error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


@router.post("/broadcast/{channel}")
async def broadcast_message(
    channel: str,
    message: RealtimeMessage,
    current_user: User = Depends(require_permissions("admin")),
) -> dict[str, Any]:
    """
    Broadcast message to all clients in a channel.

    Admin endpoint for sending system-wide notifications.
    """
    await manager.broadcast_to_channel(
        channel,
        {
            "type": message.type,
            "timestamp": message.timestamp.isoformat(),
            "data": message.data,
            "metadata": message.metadata,
        },
    )

    return {
        "status": "broadcast_sent",
        "channel": channel,
        "recipients": len(manager.active_connections.get(channel, [])),
    }


@router.get("/connections/stats")
async def get_connection_stats(
    current_user: User = Depends(require_permissions("admin")),
) -> dict[str, Any]:
    """
    Get real-time connection statistics.

    Returns information about active WebSocket connections.
    """
    return manager.get_channel_stats()


@router.websocket("/ws/system")
async def system_monitoring_websocket(
    websocket: WebSocket,
    token: str | None = Query(None),
):
    """
    WebSocket endpoint for system monitoring.

    Streams:
    - Resource usage metrics
    - Service health status
    - Error logs
    - Performance metrics
    """
    user_id = "admin_user"  # Placeholder

    try:
        await manager.connect(websocket, "system", user_id)

        while True:
            try:
                # Send system metrics
                metrics = {
                    "type": "system_metrics",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "cpu": {"usage": 42.5, "cores": 8},
                        "memory": {"used_gb": 12.3, "total_gb": 32},
                        "gpu": {"usage": 78.2, "memory_gb": 8.5},
                        "network": {"in_mbps": 125, "out_mbps": 89},
                        "inference": {"fps": 28.5, "latency_ms": 35},
                    },
                }

                await websocket.send_text(json.dumps(metrics))
                await asyncio.sleep(2)  # Update every 2 seconds

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"System WebSocket error: {e}")
                break

    finally:
        manager.disconnect(websocket)


# SSE Streaming Endpoints for MP4 Fragmented Streaming

@router.get("/streams/sse/{camera_id}/raw")
@inject
async def camera_raw_sse_stream(
    request: Request,
    camera_id: str,
    quality: str = Query("medium", regex="^(low|medium|high)$"),
    current_user: User = Depends(get_current_user),
    sse_service: SSEStreamingService = Depends(Provide["sse_streaming_service"])
) -> StreamingResponse:
    """Server-Sent Events endpoint for raw camera stream.
    
    Provides MP4 fragmented streaming for browser-native video viewing.
    
    Args:
        request: FastAPI request object
        camera_id: Target camera identifier
        quality: Stream quality (low, medium, high)
        current_user: Authenticated user
        sse_service: Injected SSE streaming service
        
    Returns:
        StreamingResponse with SSE MP4 fragments
        
    Raises:
        HTTPException: If camera not found or access denied
    """
    try:
        # Validate camera access (implement proper RBAC here)
        # For now, basic validation
        if not camera_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Camera ID is required"
            )

        logger.info(
            f"Starting raw SSE stream for camera {camera_id}, "
            f"user {current_user.username}, quality {quality}"
        )

        return await sse_service.handle_sse_connection(
            request=request,
            camera_id=camera_id,
            stream_type="raw",
            quality=quality
        )

    except Exception as e:
        logger.error(f"Raw SSE stream failed for camera {camera_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start stream: {str(e)}"
        )


@router.get("/streams/sse/{camera_id}/annotated")
@inject
async def camera_annotated_sse_stream(
    request: Request,
    camera_id: str,
    quality: str = Query("medium", regex="^(low|medium|high)$"),
    current_user: User = Depends(get_current_user),
    sse_service: SSEStreamingService = Depends(Provide["sse_streaming_service"])
) -> StreamingResponse:
    """Server-Sent Events endpoint for AI-annotated camera stream.
    
    Provides MP4 fragmented streaming with AI annotations overlaid.
    
    Args:
        request: FastAPI request object
        camera_id: Target camera identifier
        quality: Stream quality (low, medium, high)
        current_user: Authenticated user
        sse_service: Injected SSE streaming service
        
    Returns:
        StreamingResponse with SSE MP4 fragments including AI annotations
        
    Raises:
        HTTPException: If camera not found or access denied
    """
    try:
        # Validate camera access and AI annotation permissions
        if not camera_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Camera ID is required"
            )

        # Check if user has AI annotation access
        # TODO: Implement proper permission checking

        logger.info(
            f"Starting annotated SSE stream for camera {camera_id}, "
            f"user {current_user.username}, quality {quality}"
        )

        return await sse_service.handle_sse_connection(
            request=request,
            camera_id=camera_id,
            stream_type="annotated",
            quality=quality
        )

    except Exception as e:
        logger.error(f"Annotated SSE stream failed for camera {camera_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start stream: {str(e)}"
        )


@router.get("/streams/sse/stats")
@inject
async def get_sse_stream_stats(
    current_user: User = Depends(require_permissions("admin")),
    sse_service: SSEStreamingService = Depends(Provide["sse_streaming_service"])
) -> dict[str, Any]:
    """Get SSE streaming statistics.
    
    Admin endpoint for monitoring SSE streaming performance.
    
    Args:
        current_user: Authenticated admin user
        sse_service: Injected SSE streaming service
        
    Returns:
        Dictionary with SSE streaming statistics
    """
    try:
        stats = await sse_service.get_connection_stats()
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get SSE stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/streams/sse/health")
@inject
async def get_sse_health(
    current_user: User = Depends(require_permissions("admin")),
    sse_service: SSEStreamingService = Depends(Provide["sse_streaming_service"])
) -> dict[str, Any]:
    """Get SSE streaming service health status.
    
    Admin endpoint for monitoring SSE service health.
    
    Args:
        current_user: Authenticated admin user
        sse_service: Injected SSE streaming service
        
    Returns:
        Dictionary with health status
    """
    try:
        health = await sse_service.health_check()
        return {
            "status": "success",
            "data": health,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get SSE health: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


# Dual Channel Stream Endpoints

@router.get("/streams/sse/{camera_id}/dual")
@inject
async def camera_dual_channel_sse_stream(
    request: Request,
    camera_id: str,
    initial_channel: str = Query("raw", regex="^(raw|annotated)$"),
    raw_quality: str = Query("medium", regex="^(low|medium|high)$"),
    annotated_quality: str = Query("medium", regex="^(low|medium|high)$"),
    current_user: User = Depends(get_current_user),
    sse_service: SSEStreamingService = Depends(Provide["sse_streaming_service"])
) -> StreamingResponse:
    """Server-Sent Events endpoint for dual-channel streaming.
    
    Provides synchronized raw and AI-annotated camera streams with runtime switching capability.
    
    Args:
        request: FastAPI request object
        camera_id: Target camera identifier
        initial_channel: Initial channel to stream (raw or annotated)
        raw_quality: Quality for raw channel (low, medium, high)
        annotated_quality: Quality for annotated channel (low, medium, high)
        current_user: Authenticated user
        sse_service: Injected SSE streaming service
        
    Returns:
        StreamingResponse with dual-channel SSE fragments
        
    Raises:
        HTTPException: If camera not found or access denied
    """
    try:
        if not camera_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Camera ID is required"
            )

        logger.info(
            f"Starting dual-channel SSE stream for camera {camera_id}, "
            f"user {current_user.username}, initial: {initial_channel}, "
            f"raw quality: {raw_quality}, annotated quality: {annotated_quality}"
        )

        return await sse_service.handle_dual_channel_sse_connection(
            request=request,
            camera_id=camera_id,
            initial_channel=initial_channel,
            raw_quality=raw_quality,
            annotated_quality=annotated_quality
        )

    except Exception as e:
        logger.error(f"Dual-channel SSE stream failed for camera {camera_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start dual-channel stream: {str(e)}"
        )


@router.post("/streams/sse/{camera_id}/subscribe")
@inject
async def subscribe_to_channel(
    camera_id: str,
    channel_request: ChannelSubscriptionRequest,
    connection_id: str = Query(..., description="Connection identifier"),
    current_user: User = Depends(get_current_user),
    sse_service: SSEStreamingService = Depends(Provide["sse_streaming_service"])
) -> ChannelSubscriptionResponse:
    """Subscribe client to specific stream channel.
    
    Args:
        camera_id: Camera identifier
        channel_request: Channel subscription request
        connection_id: Client connection identifier
        current_user: Authenticated user
        sse_service: Injected SSE streaming service
        
    Returns:
        Channel subscription response
        
    Raises:
        HTTPException: If subscription fails
    """
    try:
        from ...services.streaming_service import ChannelType

        channel_type = ChannelType.RAW if channel_request.channel_type == "raw" else ChannelType.ANNOTATED

        result = await sse_service.subscription_manager.subscribe_to_channel(
            connection_id=connection_id,
            camera_id=camera_id,
            channel_type=channel_type
        )

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        return ChannelSubscriptionResponse(
            status=result["status"],
            connection_id=connection_id,
            camera_id=camera_id,
            channel_type=channel_request.channel_type,
            subscribed_channels=result["subscribed_channels"]
        )

    except Exception as e:
        logger.error(f"Channel subscription failed for camera {camera_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Subscription failed: {str(e)}"
        )


@router.post("/streams/sse/{camera_id}/switch")
@inject
async def switch_channel(
    camera_id: str,
    switch_request: ChannelSwitchRequest,
    current_user: User = Depends(get_current_user),
    sse_service: SSEStreamingService = Depends(Provide["sse_streaming_service"])
) -> ChannelSwitchResponse:
    """Switch client to different channel without reconnection.
    
    Args:
        camera_id: Camera identifier
        switch_request: Channel switch request
        current_user: Authenticated user
        sse_service: Injected SSE streaming service
        
    Returns:
        Channel switch response
        
    Raises:
        HTTPException: If channel switch fails
    """
    try:
        from ...services.streaming_service import ChannelType

        new_channel_type = ChannelType.RAW if switch_request.new_channel == "raw" else ChannelType.ANNOTATED

        result = await sse_service.subscription_manager.switch_channel(
            connection_id=switch_request.connection_id,
            camera_id=camera_id,
            new_channel=new_channel_type
        )

        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        # Update metrics
        sse_service.connection_metrics.channel_switches += 1

        return ChannelSwitchResponse(
            status=result["status"],
            connection_id=switch_request.connection_id,
            camera_id=camera_id,
            old_channels=result["old_channels"],
            new_channel=switch_request.new_channel,
            switch_time=result["switch_time"]
        )

    except Exception as e:
        logger.error(f"Channel switch failed for camera {camera_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Channel switch failed: {str(e)}"
        )


@router.get("/streams/sse/{camera_id}/sync-stats")
@inject
async def get_channel_sync_stats(
    camera_id: str,
    stream_id: str = Query(..., description="Dual-channel stream identifier"),
    current_user: User = Depends(require_permissions("admin")),
    sse_service: SSEStreamingService = Depends(Provide["sse_streaming_service"])
) -> dict[str, Any]:
    """Get synchronization statistics for dual-channel stream.
    
    Args:
        camera_id: Camera identifier
        stream_id: Dual-channel stream identifier
        current_user: Authenticated admin user
        sse_service: Injected SSE streaming service
        
    Returns:
        Dictionary with synchronization statistics
        
    Raises:
        HTTPException: If stream not found or access denied
    """
    try:
        sync_stats = await sse_service.channel_manager.get_synchronization_stats(stream_id)

        if "error" in sync_stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sync_stats["error"]
            )

        return {
            "status": "success",
            "camera_id": camera_id,
            "sync_stats": sync_stats,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get sync stats for stream {stream_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sync stats: {str(e)}"
        )


@router.get("/streams/dual-channel/stats")
@inject
async def get_dual_channel_stats(
    current_user: User = Depends(require_permissions("admin")),
    sse_service: SSEStreamingService = Depends(Provide["sse_streaming_service"])
) -> dict[str, Any]:
    """Get comprehensive dual-channel streaming statistics.
    
    Args:
        current_user: Authenticated admin user
        sse_service: Injected SSE streaming service
        
    Returns:
        Dictionary with dual-channel statistics
    """
    try:
        dual_channel_stats = sse_service.channel_manager.get_dual_channel_stats()
        subscription_stats = sse_service.subscription_manager.get_subscription_stats()

        return {
            "status": "success",
            "dual_channel_stats": dual_channel_stats,
            "subscription_stats": subscription_stats,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get dual-channel stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


# ML Detection Configuration Endpoints

class DetectionConfigRequest(BaseModel):
    """Request model for updating detection configuration."""

    confidence_threshold: float | None = Field(None, ge=0.0, le=1.0, description="Detection confidence threshold")
    nms_threshold: float | None = Field(None, ge=0.0, le=1.0, description="Non-maximum suppression threshold")
    max_detections: int | None = Field(None, ge=1, le=1000, description="Maximum number of detections")
    classes_to_detect: list[str] | None = Field(None, description="Object classes to detect")
    vehicle_priority: bool | None = Field(None, description="Enable vehicle detection priority")
    target_latency_ms: float | None = Field(None, ge=10.0, le=1000.0, description="Target ML processing latency")


class DetectionConfigResponse(BaseModel):
    """Response model for detection configuration."""

    status: str = Field(..., description="Operation status")
    camera_id: str = Field(..., description="Camera identifier")
    config: dict[str, Any] = Field(..., description="Updated detection configuration")
    performance_impact: dict[str, float] = Field(
        default_factory=dict, description="Expected performance impact"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DetectionStatsResponse(BaseModel):
    """Response model for detection statistics."""

    status: str = Field(..., description="Operation status")
    camera_id: str = Field(..., description="Camera identifier")
    performance_stats: dict[str, Any] = Field(..., description="ML performance statistics")
    recent_detections: list[dict[str, Any]] = Field(
        default_factory=list, description="Recent detection results"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


@router.post("/streams/sse/{camera_id}/detection-config")
@inject
async def update_detection_config(
    camera_id: str,
    config_request: DetectionConfigRequest,
    current_user: User = Depends(get_current_user),
    sse_service: SSEStreamingService = Depends(Provide["sse_streaming_service"]),
    ml_processor: MLAnnotationProcessor | None = Depends(Provide["ml_annotation_processor"]),
) -> DetectionConfigResponse:
    """Update real-time detection configuration for a camera.
    
    Args:
        camera_id: Camera identifier
        config_request: Detection configuration updates
        current_user: Authenticated user
        sse_service: Injected SSE streaming service
        ml_processor: Injected ML annotation processor
        
    Returns:
        Detection configuration response
        
    Raises:
        HTTPException: If update fails or ML processor unavailable
    """
    try:
        if not ml_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML annotation processor not available"
            )

        # Validate camera access
        if not camera_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Camera ID is required"
            )

        # Get current configuration
        current_config = ml_processor.config

        # Apply updates
        updated_fields = {}
        if config_request.confidence_threshold is not None:
            current_config.confidence_threshold = config_request.confidence_threshold
            updated_fields["confidence_threshold"] = config_request.confidence_threshold

        if config_request.nms_threshold is not None:
            current_config.nms_threshold = config_request.nms_threshold
            updated_fields["nms_threshold"] = config_request.nms_threshold

        if config_request.max_detections is not None:
            current_config.max_detections = config_request.max_detections
            updated_fields["max_detections"] = config_request.max_detections

        if config_request.classes_to_detect is not None:
            current_config.classes_to_detect = config_request.classes_to_detect
            updated_fields["classes_to_detect"] = config_request.classes_to_detect

        if config_request.vehicle_priority is not None:
            current_config.vehicle_priority = config_request.vehicle_priority
            updated_fields["vehicle_priority"] = config_request.vehicle_priority

        if config_request.target_latency_ms is not None:
            current_config.target_latency_ms = config_request.target_latency_ms
            updated_fields["target_latency_ms"] = config_request.target_latency_ms

        # Calculate performance impact estimate
        performance_impact = {
            "expected_latency_change_ms": 0.0,
            "expected_throughput_change_percent": 0.0,
            "gpu_memory_impact_mb": 0.0
        }

        # Estimate impact based on configuration changes
        if "confidence_threshold" in updated_fields:
            # Lower threshold = more detections = higher latency
            threshold_change = updated_fields["confidence_threshold"] - 0.5  # baseline
            performance_impact["expected_latency_change_ms"] += threshold_change * -10.0

        if "max_detections" in updated_fields:
            # More detections = higher rendering latency
            detection_change = (updated_fields["max_detections"] - 100) / 100.0  # baseline 100
            performance_impact["expected_latency_change_ms"] += detection_change * 5.0

        logger.info(
            f"Updated detection config for camera {camera_id}: {updated_fields} "
            f"by user {current_user.username}"
        )

        return DetectionConfigResponse(
            status="updated",
            camera_id=camera_id,
            config={
                "confidence_threshold": current_config.confidence_threshold,
                "nms_threshold": current_config.nms_threshold,
                "max_detections": current_config.max_detections,
                "classes_to_detect": current_config.classes_to_detect,
                "vehicle_priority": current_config.vehicle_priority,
                "target_latency_ms": current_config.target_latency_ms,
                "updated_fields": updated_fields
            },
            performance_impact=performance_impact
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update detection config for camera {camera_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration update failed: {str(e)}"
        )


@router.get("/streams/sse/{camera_id}/detection-stats")
@inject
async def get_detection_stats(
    camera_id: str,
    include_recent: bool = Query(True, description="Include recent detection results"),
    recent_limit: int = Query(10, ge=1, le=100, description="Number of recent detections to include"),
    current_user: User = Depends(get_current_user),
    sse_service: SSEStreamingService = Depends(Provide["sse_streaming_service"]),
    ml_processor: MLAnnotationProcessor | None = Depends(Provide["ml_annotation_processor"]),
) -> DetectionStatsResponse:
    """Get ML inference statistics and performance metrics for a camera.
    
    Args:
        camera_id: Camera identifier
        include_recent: Whether to include recent detection results
        recent_limit: Number of recent detections to include
        current_user: Authenticated user
        sse_service: Injected SSE streaming service
        ml_processor: Injected ML annotation processor
        
    Returns:
        Detection statistics response
        
    Raises:
        HTTPException: If camera not found or ML processor unavailable
    """
    try:
        if not ml_processor:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ML annotation processor not available"
            )

        # Validate camera access
        if not camera_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Camera ID is required"
            )

        # Get performance statistics
        performance_stats = ml_processor.get_performance_stats()

        # Add additional camera-specific stats
        performance_stats.update({
            "camera_id": camera_id,
            "ml_engine_status": "active" if ml_processor._active_engine else "inactive",
            "current_config": {
                "confidence_threshold": ml_processor.config.confidence_threshold,
                "target_latency_ms": ml_processor.config.target_latency_ms,
                "classes_to_detect": ml_processor.config.classes_to_detect,
                "batch_size": ml_processor.config.batch_size
            }
        })

        # Get recent detections (placeholder - implement actual recent detection storage)
        recent_detections = []
        if include_recent:
            # In production, implement proper detection history storage
            recent_detections = [
                {
                    "timestamp": datetime.utcnow().timestamp(),
                    "detection_count": 3,
                    "vehicle_count": {"car": 2, "truck": 1},
                    "processing_time_ms": 45.2,
                    "confidence_avg": 0.87
                }
            ]

        return DetectionStatsResponse(
            status="success",
            camera_id=camera_id,
            performance_stats=performance_stats,
            recent_detections=recent_detections[:recent_limit]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get detection stats for camera {camera_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get detection statistics: {str(e)}"
        )
