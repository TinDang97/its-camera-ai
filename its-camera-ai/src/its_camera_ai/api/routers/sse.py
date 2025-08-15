"""
Server-Sent Events (SSE) endpoints for real-time camera updates.

Provides streaming endpoints for live camera status, detection results,
and system monitoring data.
"""

import time
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from ...core.exceptions import ServiceError
from ...core.logging import get_logger
from ...models.user import User
from ..dependencies import get_current_user, require_permissions
from ..sse_broadcaster import (
    CameraEvent,
    SystemEvent,
    create_sse_response,
    get_broadcaster,
)

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/cameras",
    summary="Camera events stream",
    description="Stream real-time camera status updates, detection results, and health changes.",
)
async def stream_camera_events(
    request: Request,
    camera_ids: list[str] | None = Query(None, description="Filter by specific camera IDs"),
    event_types: list[str] | None = Query(None, description="Filter by event types"),
    zones: list[str] | None = Query(None, description="Filter by zone IDs"),
    include_history: bool = Query(False, description="Include recent events"),
    max_events_per_second: int = Query(10, ge=1, le=50, description="Rate limit events per second"),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream real-time camera events via Server-Sent Events.

    Event types include:
    - `status_change`: Camera online/offline status changes
    - `detection_result`: New vehicle detection results
    - `health_update`: Stream health metric updates
    - `configuration_change`: Camera setting modifications
    - `error`: Camera error notifications
    - `maintenance`: Maintenance mode changes

    Args:
        request: FastAPI request object
        camera_ids: List of camera IDs to monitor
        event_types: List of event types to include
        zones: List of zone IDs to monitor
        include_history: Include recent event history
        current_user: Authenticated user

    Returns:
        StreamingResponse: SSE stream of camera events
    """
    connection_id = f"camera_stream_{uuid.uuid4().hex}"

    # Validate event types
    valid_event_types = {
        "status_change", "detection_result", "health_update",
        "configuration_change", "error", "maintenance",
        "traffic_violation", "incident_detection"
    }
    if event_types:
        invalid_types = set(event_types) - valid_event_types
        if invalid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid event types: {', '.join(invalid_types)}"
            )

    filters = {}
    if camera_ids:
        filters["camera_ids"] = camera_ids
    if event_types:
        filters["event_types"] = event_types
    if zones:
        filters["zones"] = zones
    filters["include_history"] = include_history
    filters["max_events_per_second"] = max_events_per_second

    logger.info(
        "Camera SSE stream requested",
        connection_id=connection_id,
        user_id=str(current_user.id),
        filters=filters,
    )

    try:
        return await create_sse_response(request, connection_id, filters)
    except ServiceError as e:
        if "Maximum connections reached" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Server at maximum capacity. Please try again later."
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/system",
    summary="System events stream",
    description="Stream real-time system status, performance metrics, and alerts.",
)
async def stream_system_events(
    request: Request,
    event_types: list[str] | None = Query(None, description="Filter by event types"),
    include_history: bool = Query(False, description="Include recent events"),
    max_events_per_second: int = Query(20, ge=1, le=100, description="Rate limit events per second"),
    current_user: User = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["system:monitor"])),
) -> StreamingResponse:
    """
    Stream real-time system events via Server-Sent Events.

    Event types include:
    - `performance_alert`: System performance alerts
    - `resource_usage`: CPU, memory, GPU utilization updates
    - `service_status`: Core service health updates
    - `system_error`: Critical system errors
    - `maintenance`: System maintenance notifications
    - `statistics`: Aggregate processing statistics

    Args:
        request: FastAPI request object
        event_types: List of event types to include
        include_history: Include recent event history
        current_user: Authenticated user

    Returns:
        StreamingResponse: SSE stream of system events

    Raises:
        HTTPException: If user lacks system monitoring permissions
    """
    connection_id = f"system_stream_{uuid.uuid4().hex}"

    # Validate event types
    valid_system_event_types = {
        "performance_alert", "resource_usage", "service_status",
        "system_error", "maintenance", "statistics"
    }
    if event_types:
        invalid_types = set(event_types) - valid_system_event_types
        if invalid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid system event types: {', '.join(invalid_types)}"
            )

    filters = {"stream_type": "system"}
    if event_types:
        filters["event_types"] = event_types
    filters["include_history"] = include_history
    filters["max_events_per_second"] = max_events_per_second

    logger.info(
        "System SSE stream requested",
        connection_id=connection_id,
        user_id=str(current_user.id),
        filters=filters,
    )

    try:
        return await create_sse_response(request, connection_id, filters)
    except ServiceError as e:
        if "Maximum connections reached" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Server at maximum capacity. Please try again later."
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/detections",
    summary="Detection events stream",
    description="Stream real-time vehicle detection results from all cameras.",
)
async def stream_detection_events(
    request: Request,
    camera_ids: list[str] | None = Query(None, description="Filter by specific camera IDs"),
    vehicle_types: list[str] | None = Query(None, description="Filter by vehicle types"),
    min_confidence: float = Query(0.5, ge=0.0, le=1.0, description="Minimum detection confidence"),
    zones: list[str] | None = Query(None, description="Filter by zone IDs"),
    max_events_per_second: int = Query(15, ge=1, le=100, description="Rate limit events per second"),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """
    Stream real-time vehicle detection events.

    Streams detection results from the Core Vision Engine as they are processed,
    including bounding boxes, classification confidence, and metadata.

    Args:
        request: FastAPI request object
        camera_ids: List of camera IDs to monitor
        vehicle_types: Filter by vehicle types (car, truck, bus, etc.)
        min_confidence: Minimum detection confidence threshold
        zones: List of zone IDs to monitor
        current_user: Authenticated user

    Returns:
        StreamingResponse: SSE stream of detection events
    """
    connection_id = f"detection_stream_{uuid.uuid4().hex}"

    # Validate vehicle types
    valid_vehicle_types = {
        "car", "truck", "bus", "motorcycle", "bicycle", "person", "van", "trailer"
    }
    if vehicle_types:
        invalid_types = set(vehicle_types) - valid_vehicle_types
        if invalid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid vehicle types: {', '.join(invalid_types)}"
            )

    filters = {
        "event_types": ["detection_result"],
        "min_confidence": min_confidence,
        "max_events_per_second": max_events_per_second,
    }
    if camera_ids:
        filters["camera_ids"] = camera_ids
    if vehicle_types:
        filters["vehicle_types"] = vehicle_types
    if zones:
        filters["zones"] = zones

    logger.info(
        "Detection SSE stream requested",
        connection_id=connection_id,
        user_id=str(current_user.id),
        filters=filters,
    )

    try:
        return await create_sse_response(request, connection_id, filters)
    except ServiceError as e:
        if "Maximum connections reached" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Server at maximum capacity. Please try again later."
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/broadcast/camera-event",
    summary="Broadcast camera event",
    description="Manually broadcast a camera event to all connected clients.",
)
async def broadcast_camera_event(
    camera_id: str,
    event_type: str,
    data: dict,
    current_user: User = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["system:admin"])),
) -> dict[str, str]:
    """
    Manually broadcast a camera event to all connected SSE clients.

    This endpoint allows administrators to send custom events to all
    connected dashboard clients.

    Args:
        camera_id: Camera identifier
        event_type: Type of event to broadcast
        data: Event data payload
        current_user: Authenticated user

    Returns:
        Dict: Broadcast confirmation

    Raises:
        HTTPException: If user lacks admin permissions
    """
    broadcaster = get_broadcaster()

    event = CameraEvent(
        camera_id=camera_id,
        event_type=event_type,
        timestamp=datetime.now(),
        data=data,
    )

    await broadcaster.broadcast_camera_event(event)

    logger.info(
        "Manual camera event broadcasted",
        camera_id=camera_id,
        event_type=event_type,
        user_id=str(current_user.id),
    )

    return {"status": "broadcasted", "event_type": event_type, "camera_id": camera_id}


@router.post(
    "/broadcast/system-event",
    summary="Broadcast system event",
    description="Manually broadcast a system event to all connected clients.",
)
async def broadcast_system_event(
    event_type: str,
    data: dict,
    current_user: User = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["system:admin"])),
) -> dict[str, str]:
    """
    Manually broadcast a system event to all connected SSE clients.

    Args:
        event_type: Type of event to broadcast
        data: Event data payload
        current_user: Authenticated user

    Returns:
        Dict: Broadcast confirmation

    Raises:
        HTTPException: If user lacks admin permissions
    """
    broadcaster = get_broadcaster()

    event = SystemEvent(
        event_type=event_type,
        timestamp=datetime.now(),
        data=data,
    )

    await broadcaster.broadcast_system_event(event)

    logger.info(
        "Manual system event broadcasted",
        event_type=event_type,
        user_id=str(current_user.id),
    )

    return {"status": "broadcasted", "event_type": event_type}


@router.get(
    "/analytics",
    summary="Traffic analytics stream",
    description="Stream real-time traffic analytics, flow data, and speed calculations.",
)
async def stream_analytics_events(
    request: Request,
    zones: list[str] | None = Query(None, description="Filter by zone IDs"),
    analytics_types: list[str] | None = Query(None, description="Filter by analytics types"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence for analytics"),
    max_events_per_second: int = Query(25, ge=1, le=100, description="Rate limit events per second"),
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Stream real-time traffic analytics events.
    
    Analytics types include:
    - `traffic_flow`: Real-time traffic flow measurements
    - `speed_calculation`: Vehicle speed calculations
    - `zone_analytics`: Zone-specific analytics and violations
    - `occupancy_analysis`: Parking/zone occupancy analysis
    - `congestion_detection`: Traffic congestion alerts
    
    Args:
        request: FastAPI request object
        zones: List of zone IDs to monitor
        analytics_types: List of analytics types to include
        min_confidence: Minimum confidence threshold for analytics
        max_events_per_second: Rate limit for events per second
        current_user: Authenticated user
        
    Returns:
        StreamingResponse: SSE stream of analytics events
    """
    connection_id = f"analytics_stream_{uuid.uuid4().hex}"

    # Validate analytics types
    valid_analytics_types = {
        "traffic_flow", "speed_calculation", "zone_analytics",
        "occupancy_analysis", "congestion_detection"
    }
    if analytics_types:
        invalid_types = set(analytics_types) - valid_analytics_types
        if invalid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analytics types: {', '.join(invalid_types)}"
            )

    filters = {
        "event_types": analytics_types or list(valid_analytics_types),
        "min_confidence": min_confidence,
        "max_events_per_second": max_events_per_second,
    }
    if zones:
        filters["zones"] = zones

    logger.info(
        "Analytics SSE stream requested",
        connection_id=connection_id,
        user_id=str(current_user.id),
        filters=filters,
    )

    try:
        return await create_sse_response(request, connection_id, filters)
    except ServiceError as e:
        if "Maximum connections reached" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Server at maximum capacity. Please try again later."
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/stats",
    summary="SSE broadcaster statistics",
    description="Get statistics about active SSE connections and message throughput.",
)
async def get_sse_stats(
    current_user: User = Depends(get_current_user),
    _permissions: None = Depends(require_permissions(["system:monitor"])),
) -> dict:
    """
    Get statistics about the SSE broadcaster.

    Returns information about active connections, message throughput,
    and connection details for monitoring purposes.

    Args:
        current_user: Authenticated user

    Returns:
        Dict: SSE broadcaster statistics

    Raises:
        HTTPException: If user lacks monitoring permissions
    """
    broadcaster = get_broadcaster()
    stats = broadcaster.get_stats()

    logger.debug("SSE statistics requested", user_id=str(current_user.id))

    return stats


@router.get(
    "/health",
    summary="SSE service health check",
    description="Get health status of SSE service and Kafka integration.",
)
async def get_sse_health(
    current_user: User = Depends(get_current_user),
) -> dict:
    """Get detailed health status of SSE service.
    
    Returns comprehensive health information including:
    - Connection statistics and limits
    - Kafka integration status
    - Rate limiting status
    - Performance metrics
    
    Args:
        current_user: Authenticated user
        
    Returns:
        Dict: Detailed health status
    """
    broadcaster = get_broadcaster()
    health_data = broadcaster.get_stats()

    # Add additional health checks
    health_data["service_status"] = "healthy" if len(broadcaster.connections) < broadcaster.max_connections else "degraded"
    health_data["timestamp"] = time.time()

    logger.debug("SSE health check requested", user_id=str(current_user.id))

    return health_data
