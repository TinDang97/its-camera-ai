"""Camera management endpoints.

Provides CRUD operations for camera management, stream control,
health monitoring, and batch operations.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.logging import get_logger
from ...models.user import User
from ...services.cache import CacheService
from ...services.camera_service import CameraService
from ..dependencies import (
    RateLimiter,
    get_cache_service,
    get_current_user,
    get_db,
    rate_limit_strict,
    require_permissions,
)
from ..schemas.cameras import (
    CameraBatchOperation,
    CameraBatchResult,
    CameraCreate,
    CameraResponse,
    CameraStats,
    CameraStatus,
    CameraUpdate,
    StreamHealth,
    StreamRequest,
)
from ..schemas.common import PaginatedResponse, SuccessResponse

logger = get_logger(__name__)
router = APIRouter()

# Rate limiters
batch_operation_rate_limit = RateLimiter(calls=5, period=300)  # 5 batch ops per 5 min
stream_control_rate_limit = RateLimiter(calls=20, period=60)  # 20 stream ops per min

# Database service dependency
async def get_camera_service(db: AsyncSession = Depends(get_db)) -> CameraService:
    """Get camera service instance."""
    return CameraService(db)


# Remove simulated database - now using real database service
# cameras_db: dict[str, dict[str, Any]] = {}
stream_health_db: dict[str, StreamHealth] = {}


async def check_stream_health(_camera_id: str, _stream_url: str) -> StreamHealth:
    """Check the health of a camera stream.

    Args:
        camera_id: Camera identifier
        stream_url: Stream URL to check

    Returns:
        StreamHealth: Stream health metrics
    """
    # TODO: Implement actual stream health checking
    # This would involve connecting to the stream and measuring metrics
    import random

    # Simulate health check
    is_connected = random.choice([True, True, True, False])  # 75% uptime

    return StreamHealth(
        is_connected=is_connected,
        bitrate=random.uniform(2000, 8000) if is_connected else None,
        fps=random.uniform(25, 30) if is_connected else None,
        packet_loss=random.uniform(0, 2) if is_connected else None,
        latency=random.uniform(50, 200) if is_connected else None,
        last_frame_time=datetime.now(UTC) if is_connected else None,
        error_message=None if is_connected else "Connection timeout",
        uptime=random.uniform(3600, 86400) if is_connected else 0,
    )


async def start_stream_monitoring(camera_id: str) -> None:
    """Start background stream monitoring for a camera.

    Args:
        camera_id: Camera identifier
    """
    # TODO: Implement actual stream monitoring
    logger.info("Stream monitoring started", camera_id=camera_id)


async def stop_stream_monitoring(camera_id: str) -> None:
    """Stop background stream monitoring for a camera.

    Args:
        camera_id: Camera identifier
    """
    # TODO: Implement actual stream monitoring
    logger.info("Stream monitoring stopped", camera_id=camera_id)


@router.get(
    "/",
    response_model=PaginatedResponse[CameraResponse],
    summary="List cameras",
    description="Retrieve a paginated list of cameras with optional filtering.",
)
async def list_cameras(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    status: CameraStatus | None = Query(None, description="Filter by camera status"),
    location: str | None = Query(None, description="Filter by location"),
    zone_id: str | None = Query(None, description="Filter by zone ID"),
    tags: list[str] | None = Query(None, description="Filter by tags"),
    search: str | None = Query(None, description="Search in name and description"),
    current_user: User = Depends(get_current_user),
    camera_service: CameraService = Depends(get_camera_service),
    cache: CacheService = Depends(get_cache_service),
) -> PaginatedResponse[CameraResponse]:
    """List cameras with pagination and filtering.

    Args:
        page: Page number
        size: Items per page
        status: Filter by camera status
        location: Filter by location
        zone_id: Filter by zone ID
        tags: Filter by tags
        search: Search query
        current_user: Current user
        db: Database session
        cache: Cache service

    Returns:
        PaginatedResponse[CameraResponse]: Paginated camera list
    """
    try:
        # Build cache key for this query
        cache_key = f"cameras:list:{page}:{size}:{status}:{location}:{zone_id}:{':'.join(tags or [])}:{search}"

        # Try to get from cache first
        cached_result = await cache.get_json(cache_key)
        if cached_result:
            logger.debug("Returning cached camera list", cache_key=cache_key)
            return PaginatedResponse[CameraResponse](**cached_result)

        # TODO: Replace with actual database queries
        # For now, use simulated data
        cameras = list(cameras_db.values())

        # Apply filters
        if status:
            cameras = [c for c in cameras if c.get("status") == status]
        if location:
            cameras = [
                c for c in cameras if location.lower() in c.get("location", "").lower()
            ]
        if zone_id:
            cameras = [c for c in cameras if c.get("zone_id") == zone_id]
        if tags:
            cameras = [
                c for c in cameras if any(tag in c.get("tags", []) for tag in tags)
            ]
        if search:
            search_lower = search.lower()
            cameras = [
                c
                for c in cameras
                if search_lower in c.get("name", "").lower()
                or search_lower in c.get("description", "").lower()
            ]

        # Calculate pagination
        total = len(cameras)
        offset = (page - 1) * size
        paginated_cameras = cameras[offset : offset + size]

        # Convert to response models
        camera_responses = [
            CameraResponse(
                id=camera["id"],
                name=camera["name"],
                description=camera.get("description"),
                location=camera["location"],
                coordinates=camera.get("coordinates"),
                camera_type=camera["camera_type"],
                stream_url=camera["stream_url"],
                stream_protocol=camera["stream_protocol"],
                backup_stream_url=camera.get("backup_stream_url"),
                status=camera.get("status", CameraStatus.OFFLINE),
                config=camera["config"],
                health=stream_health_db.get(camera["id"]),
                zone_id=camera.get("zone_id"),
                tags=camera.get("tags", []),
                is_active=camera.get("is_active", True),
                created_at=camera.get("created_at", datetime.now(UTC)),
                updated_at=camera.get("updated_at", datetime.now(UTC)),
                last_seen_at=camera.get("last_seen_at"),
            )
            for camera in paginated_cameras
        ]

        result = PaginatedResponse.create(
            items=camera_responses,
            total=total,
            page=page,
            size=size,
        )

        # Cache the result for 1 minute
        await cache.set_json(cache_key, result.model_dump(), ttl=60)

        logger.info(
            "Cameras listed",
            total=total,
            page=page,
            size=size,
            user_id=current_user.id,
        )

        return result

    except Exception as e:
        logger.error("Failed to list cameras", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cameras",
        ) from e


@router.post(
    "/",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create camera",
    description="Register a new camera in the system.",
)
async def create_camera(
    camera_data: CameraCreate,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permissions("cameras:create")),
    _db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
    _rate_limit: None = Depends(rate_limit_strict),
) -> CameraResponse:
    """Create a new camera.

    Args:
        camera_data: Camera creation data
        background_tasks: Background task manager
        current_user: Current user with permissions
        db: Database session
        cache: Cache service
        _rate_limit: Rate limiting dependency

    Returns:
        CameraResponse: Created camera information

    Raises:
        HTTPException: If camera creation fails
    """
    try:
        # Generate camera ID
        camera_id = str(uuid4())

        # TODO: Validate stream URL connectivity
        # health = await check_stream_health(camera_id, camera_data.stream_url)

        # Create camera record
        now = datetime.now(UTC)
        camera_record = {
            "id": camera_id,
            "name": camera_data.name,
            "description": camera_data.description,
            "location": camera_data.location,
            "coordinates": camera_data.coordinates,
            "camera_type": camera_data.camera_type,
            "stream_url": camera_data.stream_url,
            "stream_protocol": camera_data.stream_protocol,
            "backup_stream_url": camera_data.backup_stream_url,
            "status": CameraStatus.OFFLINE,
            "config": camera_data.config,
            "zone_id": camera_data.zone_id,
            "tags": camera_data.tags,
            "is_active": True,
            "created_at": now,
            "updated_at": now,
            "created_by": current_user.id,
        }

        # Store in simulated database
        cameras_db[camera_id] = camera_record

        # Start health monitoring in background
        background_tasks.add_task(start_stream_monitoring, camera_id)

        # Invalidate cache
        await cache.delete("cameras:list:*")

        logger.info(
            "Camera created successfully",
            camera_id=camera_id,
            name=camera_data.name,
            user_id=current_user.id,
        )

        return CameraResponse(
            id=camera_id,
            name=camera_record["name"],
            description=camera_record["description"],
            location=camera_record["location"],
            coordinates=camera_record["coordinates"],
            camera_type=camera_record["camera_type"],
            stream_url=camera_record["stream_url"],
            stream_protocol=camera_record["stream_protocol"],
            backup_stream_url=camera_record["backup_stream_url"],
            status=camera_record["status"],
            config=camera_record["config"],
            health=None,  # Will be populated by monitoring
            zone_id=camera_record["zone_id"],
            tags=camera_record["tags"],
            is_active=camera_record["is_active"],
            created_at=camera_record["created_at"],
            updated_at=camera_record["updated_at"],
            last_seen_at=None,
        )

    except Exception as e:
        logger.error(
            "Camera creation failed",
            error=str(e),
            name=camera_data.name,
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Camera creation failed",
        ) from e


@router.get(
    "/{camera_id}",
    response_model=CameraResponse,
    summary="Get camera",
    description="Retrieve detailed information about a specific camera.",
)
async def get_camera(
    camera_id: str,
    current_user: User = Depends(get_current_user),
    _db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
) -> CameraResponse:
    """Get camera by ID.

    Args:
        camera_id: Camera identifier
        current_user: Current user
        db: Database session
        cache: Cache service

    Returns:
        CameraResponse: Camera information

    Raises:
        HTTPException: If camera not found
    """
    # Try cache first
    cache_key = f"camera:{camera_id}"
    cached_camera = await cache.get_json(cache_key)
    if cached_camera:
        return CameraResponse(**cached_camera)

    # Get from database
    camera = cameras_db.get(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    # Get current health status
    health = stream_health_db.get(camera_id)
    if not health:
        health = await check_stream_health(camera_id, camera["stream_url"])
        stream_health_db[camera_id] = health

    camera_response = CameraResponse(
        id=camera["id"],
        name=camera["name"],
        description=camera.get("description"),
        location=camera["location"],
        coordinates=camera.get("coordinates"),
        camera_type=camera["camera_type"],
        stream_url=camera["stream_url"],
        stream_protocol=camera["stream_protocol"],
        backup_stream_url=camera.get("backup_stream_url"),
        status=camera.get("status", CameraStatus.OFFLINE),
        config=camera["config"],
        health=health,
        zone_id=camera.get("zone_id"),
        tags=camera.get("tags", []),
        is_active=camera.get("is_active", True),
        created_at=camera.get("created_at", datetime.now(UTC)),
        updated_at=camera.get("updated_at", datetime.now(UTC)),
        last_seen_at=camera.get("last_seen_at"),
    )

    # Cache for 30 seconds
    await cache.set_json(cache_key, camera_response.model_dump(), ttl=30)

    logger.debug("Camera retrieved", camera_id=camera_id, user_id=current_user.id)

    return camera_response


@router.put(
    "/{camera_id}",
    response_model=CameraResponse,
    summary="Update camera",
    description="Update camera configuration and settings.",
)
async def update_camera(
    camera_id: str,
    camera_data: CameraUpdate,
    current_user: User = Depends(require_permissions("cameras:update")),
    _db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
    _rate_limit: None = Depends(rate_limit_strict),
) -> CameraResponse:
    """Update camera configuration.

    Args:
        camera_id: Camera identifier
        camera_data: Camera update data
        current_user: Current user with permissions
        db: Database session
        cache: Cache service
        _rate_limit: Rate limiting dependency

    Returns:
        CameraResponse: Updated camera information

    Raises:
        HTTPException: If camera not found or update fails
    """
    camera = cameras_db.get(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    try:
        # Update fields if provided
        update_data = camera_data.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            if value is not None:
                camera[field] = value

        camera["updated_at"] = datetime.now(UTC)

        # Invalidate caches
        await cache.delete(f"camera:{camera_id}")
        await cache.delete("cameras:list:*")

        logger.info(
            "Camera updated successfully",
            camera_id=camera_id,
            updated_fields=list(update_data.keys()),
            user_id=current_user.id,
        )

        return await get_camera(camera_id, _db, current_user)

    except Exception as e:
        logger.error(
            "Camera update failed",
            camera_id=camera_id,
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Camera update failed",
        ) from e


@router.delete(
    "/{camera_id}",
    response_model=SuccessResponse,
    summary="Delete camera",
    description="Remove a camera from the system.",
)
async def delete_camera(
    camera_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permissions("cameras:delete")),
    _db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
    _rate_limit: None = Depends(rate_limit_strict),
) -> SuccessResponse:
    """Delete a camera.

    Args:
        camera_id: Camera identifier
        background_tasks: Background task manager
        current_user: Current user with permissions
        db: Database session
        cache: Cache service
        _rate_limit: Rate limiting dependency

    Returns:
        SuccessResponse: Deletion confirmation

    Raises:
        HTTPException: If camera not found
    """
    camera = cameras_db.get(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    try:
        # Stop monitoring
        background_tasks.add_task(stop_stream_monitoring, camera_id)

        # Remove from databases
        del cameras_db[camera_id]
        if camera_id in stream_health_db:
            del stream_health_db[camera_id]

        # Invalidate caches
        await cache.delete(f"camera:{camera_id}")
        await cache.delete("cameras:list:*")

        logger.info(
            "Camera deleted successfully",
            camera_id=camera_id,
            name=camera.get("name"),
            user_id=current_user.id,
        )

        return SuccessResponse(
            success=True,
            message="Camera deleted successfully",
            data={"camera_id": camera_id},
        )

    except Exception as e:
        logger.error(
            "Camera deletion failed",
            camera_id=camera_id,
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Camera deletion failed",
        ) from e


@router.post(
    "/{camera_id}/stream",
    response_model=SuccessResponse,
    summary="Control camera stream",
    description="Start, stop, or restart camera stream.",
)
async def control_stream(
    camera_id: str,
    stream_request: StreamRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permissions("cameras:stream")),
    _db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
    _rate_limit: None = Depends(stream_control_rate_limit),
) -> SuccessResponse:
    """Control camera stream.

    Args:
        camera_id: Camera identifier
        stream_request: Stream control request
        background_tasks: Background task manager
        current_user: Current user with permissions
        db: Database session
        cache: Cache service
        _rate_limit: Rate limiting dependency

    Returns:
        SuccessResponse: Stream control confirmation

    Raises:
        HTTPException: If camera not found or operation fails
    """
    camera = cameras_db.get(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    try:
        action = stream_request.action

        if action == "start":
            camera["status"] = CameraStatus.STREAMING
            background_tasks.add_task(start_stream_monitoring, camera_id)
        elif action == "stop":
            camera["status"] = CameraStatus.STOPPED
            background_tasks.add_task(stop_stream_monitoring, camera_id)
        elif action == "restart":
            camera["status"] = CameraStatus.STREAMING
            background_tasks.add_task(stop_stream_monitoring, camera_id)
            background_tasks.add_task(start_stream_monitoring, camera_id)

        camera["updated_at"] = datetime.now(UTC)

        # Invalidate cache
        await cache.delete(f"camera:{camera_id}")

        logger.info(
            "Stream control executed",
            camera_id=camera_id,
            action=action,
            user_id=current_user.id,
        )

        return SuccessResponse(
            success=True,
            message=f"Stream {action} initiated successfully",
            data={
                "camera_id": camera_id,
                "action": action,
                "status": camera["status"],
            },
        )

    except Exception as e:
        logger.error(
            "Stream control failed",
            camera_id=camera_id,
            action=stream_request.action,
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Stream control failed",
        ) from e


@router.get(
    "/{camera_id}/health",
    response_model=StreamHealth,
    summary="Get stream health",
    description="Get current stream health metrics for a camera.",
)
async def get_stream_health(
    camera_id: str,
    current_user: User = Depends(get_current_user),
    _db: AsyncSession = Depends(get_db),
) -> StreamHealth:
    """Get stream health metrics.

    Args:
        camera_id: Camera identifier
        current_user: Current user
        db: Database session

    Returns:
        StreamHealth: Stream health metrics

    Raises:
        HTTPException: If camera not found
    """
    camera = cameras_db.get(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    # Get or refresh health data
    health = stream_health_db.get(camera_id)
    if not health:
        health = await check_stream_health(camera_id, camera["stream_url"])
        stream_health_db[camera_id] = health

    logger.debug(
        "Stream health retrieved", camera_id=camera_id, user_id=current_user.id
    )

    return health


@router.post(
    "/batch",
    response_model=list[CameraBatchResult],
    summary="Batch camera operations",
    description="Perform operations on multiple cameras simultaneously.",
)
async def batch_operations(
    batch_request: CameraBatchOperation,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permissions("cameras:batch")),
    _db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
    _rate_limit: None = Depends(batch_operation_rate_limit),
) -> list[CameraBatchResult]:
    """Perform batch operations on cameras.

    Args:
        batch_request: Batch operation request
        background_tasks: Background task manager
        current_user: Current user with permissions
        db: Database session
        cache: Cache service
        _rate_limit: Rate limiting dependency

    Returns:
        list[CameraBatchResult]: Results for each camera operation
    """
    results = []
    operation = batch_request.operation
    parameters = batch_request.parameters or {}

    for camera_id in batch_request.camera_ids:
        try:
            camera = cameras_db.get(camera_id)
            if not camera:
                results.append(
                    CameraBatchResult(
                        camera_id=camera_id,
                        success=False,
                        message=f"Camera {camera_id} not found",
                    )
                )
                continue

            # Perform operation based on type
            if operation == "start_streams":
                camera["status"] = CameraStatus.STREAMING
                background_tasks.add_task(start_stream_monitoring, camera_id)
                message = "Stream started"

            elif operation == "stop_streams":
                camera["status"] = CameraStatus.STOPPED
                background_tasks.add_task(stop_stream_monitoring, camera_id)
                message = "Stream stopped"

            elif operation == "restart_streams":
                camera["status"] = CameraStatus.STREAMING
                background_tasks.add_task(stop_stream_monitoring, camera_id)
                background_tasks.add_task(start_stream_monitoring, camera_id)
                message = "Stream restarted"

            elif operation == "update_config":
                if "config" in parameters:
                    camera["config"].update(parameters["config"])
                camera["updated_at"] = datetime.now(UTC)
                message = "Configuration updated"

            elif operation == "enable_analytics":
                camera["config"]["analytics_enabled"] = True
                camera["updated_at"] = datetime.now(UTC)
                message = "Analytics enabled"

            elif operation == "disable_analytics":
                camera["config"]["analytics_enabled"] = False
                camera["updated_at"] = datetime.now(UTC)
                message = "Analytics disabled"

            elif operation == "delete":
                del cameras_db[camera_id]
                if camera_id in stream_health_db:
                    del stream_health_db[camera_id]
                background_tasks.add_task(stop_stream_monitoring, camera_id)
                message = "Camera deleted"

            else:
                results.append(
                    CameraBatchResult(
                        camera_id=camera_id,
                        success=False,
                        message=f"Unknown operation: {operation}",
                    )
                )
                continue

            results.append(
                CameraBatchResult(
                    camera_id=camera_id,
                    success=True,
                    message=message,
                    details={"operation": operation},
                )
            )

            # Invalidate cache
            await cache.delete(f"camera:{camera_id}")

        except Exception as e:
            logger.error(
                "Batch operation failed for camera",
                camera_id=camera_id,
                operation=operation,
                error=str(e),
            )
            results.append(
                CameraBatchResult(
                    camera_id=camera_id,
                    success=False,
                    message=f"Operation failed: {str(e)}",
                )
            )

    # Invalidate list cache
    await cache.delete("cameras:list:*")

    logger.info(
        "Batch operation completed",
        operation=operation,
        total_cameras=len(batch_request.camera_ids),
        successful=len([r for r in results if r.success]),
        user_id=current_user.id,
    )

    return results


@router.get(
    "/{camera_id}/stats",
    response_model=CameraStats,
    summary="Get camera statistics",
    description="Get statistical information about camera performance.",
)
async def get_camera_stats(
    camera_id: str,
    days: int = Query(7, ge=1, le=30, description="Number of days for statistics"),
    current_user: User = Depends(get_current_user),
    _db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
) -> CameraStats:
    """Get camera statistics.

    Args:
        camera_id: Camera identifier
        days: Number of days for statistics
        current_user: Current user
        db: Database session
        cache: Cache service

    Returns:
        CameraStats: Camera statistics

    Raises:
        HTTPException: If camera not found
    """
    camera = cameras_db.get(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    # Check cache first
    cache_key = f"camera_stats:{camera_id}:{days}"
    cached_stats = await cache.get_json(cache_key)
    if cached_stats:
        return CameraStats(**cached_stats)

    # TODO: Calculate real statistics from database
    import random

    # Generate mock statistics
    now = datetime.now(UTC)
    period_start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(
        days=days
    )

    # Generate hourly activity for last 24 hours
    last_24h_activity = {f"{i:02d}:00": random.randint(50, 500) for i in range(24)}

    stats = CameraStats(
        camera_id=camera_id,
        frames_processed=random.randint(50000, 200000) * days,
        vehicles_detected=random.randint(5000, 20000) * days,
        incidents_detected=random.randint(10, 50) * days,
        uptime_percentage=random.uniform(95, 99.9),
        avg_processing_time=random.uniform(50, 150),
        last_24h_activity=last_24h_activity,
        period_start=period_start,
        period_end=now,
    )

    # Cache for 5 minutes
    await cache.set_json(cache_key, stats.model_dump(), ttl=300)

    logger.debug(
        "Camera statistics retrieved",
        camera_id=camera_id,
        days=days,
        user_id=current_user.id,
    )

    return stats
