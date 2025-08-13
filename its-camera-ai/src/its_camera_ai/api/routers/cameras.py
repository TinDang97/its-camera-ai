"""Camera management endpoints.

Provides CRUD operations for camera management, stream control,
health monitoring, and batch operations.
"""

from datetime import UTC, datetime, timedelta

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
from ..schemas.database import CameraCreateSchema, CameraUpdateSchema

logger = get_logger(__name__)
router = APIRouter()

# Rate limiters
batch_operation_rate_limit = RateLimiter(calls=5, period=300)  # 5 batch ops per 5 min
stream_control_rate_limit = RateLimiter(calls=20, period=60)  # 20 stream ops per min

# Database service dependency
async def get_camera_service(db: AsyncSession = Depends(get_db)) -> CameraService:
    """Get camera service instance."""
    return CameraService(db)


async def check_stream_health(camera_id: str, stream_url: str, cache: CacheService) -> StreamHealth:
    """Check the health of a camera stream with Redis caching.

    Args:
        camera_id: Camera identifier
        stream_url: Stream URL to check
        cache: Redis cache service

    Returns:
        StreamHealth: Stream health status and metrics
    """
    # Build cache key for stream health
    cache_key = f"stream_health:{camera_id}"

    try:
        # Try to get cached health status
        cached_health = await cache.get_json(cache_key)
        if cached_health:
            # Check if cached data is recent enough (within 30 seconds)
            cached_time = datetime.fromisoformat(cached_health["last_checked"].replace("Z", "+00:00"))
            if (datetime.now(UTC) - cached_time).total_seconds() < 30:
                return StreamHealth(**cached_health)
    except Exception as e:
        logger.warning(f"Failed to retrieve cached stream health for {camera_id}: {e}")

    # Perform actual stream health check
    start_time = datetime.now(UTC)

    try:
        # Simulate stream connectivity check (in production, this would be actual stream validation)

        import aiohttp

        # Quick HTTP HEAD request to check if stream endpoint is reachable
        timeout = aiohttp.ClientTimeout(total=5.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.head(stream_url) as response:
                    is_healthy = response.status < 400
                    response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

                    if is_healthy:
                        status = "healthy"
                        error_message = None
                    else:
                        status = "error"
                        error_message = f"HTTP {response.status}"

            except TimeoutError:
                is_healthy = False
                status = "timeout"
                error_message = "Connection timeout"
                response_time = 5000.0

            except Exception as e:
                is_healthy = False
                status = "error"
                error_message = str(e)
                response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

    except Exception as e:
        # Fallback for any connection issues
        is_healthy = False
        status = "error"
        error_message = f"Connection failed: {str(e)}"
        response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

    # Create stream health object
    stream_health = StreamHealth(
        camera_id=camera_id,
        is_healthy=is_healthy,
        status=status,
        last_checked=datetime.now(UTC),
        response_time_ms=response_time,
        error_message=error_message,
        stream_url=stream_url,
        bitrate_kbps=None,  # Would need actual stream analysis
        frame_rate=None,    # Would need actual stream analysis
        resolution=None,    # Would need actual stream analysis
    )

    # Cache the health status for 30 seconds
    try:
        await cache.set_json(cache_key, stream_health.model_dump(mode="json"), ttl=30)
    except Exception as e:
        logger.warning(f"Failed to cache stream health for {camera_id}: {e}")

    return stream_health


async def get_stream_health_from_cache(camera_id: str, cache: CacheService) -> StreamHealth | None:
    """Get stream health from Redis cache only.

    Args:
        camera_id: Camera identifier
        cache: Redis cache service

    Returns:
        StreamHealth or None if not cached or expired
    """
    cache_key = f"stream_health:{camera_id}"

    try:
        cached_health = await cache.get_json(cache_key)
        if cached_health:
            return StreamHealth(**cached_health)
    except Exception as e:
        logger.warning(f"Failed to retrieve cached stream health for {camera_id}: {e}")

    return None


async def update_stream_health_cache(camera_id: str, health_data: dict, cache: CacheService, ttl: int = 30) -> bool:
    """Update stream health in Redis cache.

    Args:
        camera_id: Camera identifier
        health_data: Stream health data to cache
        cache: Redis cache service
        ttl: Time to live in seconds

    Returns:
        bool: True if successfully cached
    """
    cache_key = f"stream_health:{camera_id}"

    try:
        return await cache.set_json(cache_key, health_data, ttl=ttl)
    except Exception as e:
        logger.error(f"Failed to update stream health cache for {camera_id}: {e}")
        return False


async def clear_stream_health_cache(camera_id: str, cache: CacheService) -> bool:
    """Clear stream health from Redis cache.

    Args:
        camera_id: Camera identifier
        cache: Redis cache service

    Returns:
        bool: True if successfully cleared
    """
    cache_key = f"stream_health:{camera_id}"

    try:
        return await cache.delete(cache_key)
    except Exception as e:
        logger.error(f"Failed to clear stream health cache for {camera_id}: {e}")
        return False


async def get_multiple_stream_health(camera_ids: list[str], cache: CacheService) -> dict[str, StreamHealth]:
    """Get stream health for multiple cameras from cache.

    Args:
        camera_ids: List of camera identifiers
        cache: Redis cache service

    Returns:
        dict: Camera ID to StreamHealth mapping
    """
    health_data = {}

    for camera_id in camera_ids:
        health = await get_stream_health_from_cache(camera_id, cache)
        if health:
            health_data[camera_id] = health

    return health_data


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

        # Build filters for database query
        filters = {}
        if status:
            filters["status"] = status.value
        if zone_id:
            filters["zone_id"] = zone_id
        if search:
            filters["search"] = search
        if tags:
            filters["tags"] = tags
        # Add location filter (camera service handles this via search)
        if location and not search:
            filters["search"] = location

        # Get cameras from database service
        cameras, total = await camera_service.get_cameras_with_pagination(
            page=page,
            size=size,
            filters=filters,
            order_by="name",
            order_desc=False,
        )

        # Convert to response models
        camera_responses = [
            CameraResponse(
                id=camera.id,
                name=camera.name,
                description=camera.description,
                location=camera.location,
                coordinates=camera.coordinates,
                camera_type=camera.camera_type,
                stream_url=camera.stream_url,
                stream_protocol=camera.stream_protocol,
                backup_stream_url=camera.backup_stream_url,
                status=CameraStatus(camera.status),
                config=camera.config,
                health=await get_stream_health_from_cache(camera.id, cache),
                zone_id=camera.zone_id,
                tags=camera.tags,
                is_active=camera.is_active,
                created_at=camera.created_at,
                updated_at=camera.updated_at,
                last_seen_at=camera.last_seen_at,
            )
            for camera in cameras
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
    camera_service: CameraService = Depends(get_camera_service),
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
        # Convert API schema to database schema
        db_camera_data = CameraCreateSchema(
            name=camera_data.name,
            description=camera_data.description,
            location=camera_data.location,
            coordinates=camera_data.coordinates,
            camera_type=camera_data.camera_type,
            stream_url=camera_data.stream_url,
            stream_protocol=camera_data.stream_protocol,
            backup_stream_url=camera_data.backup_stream_url,
            username=camera_data.username,
            password=camera_data.password,
            config=camera_data.config.model_dump(),
            zone_id=camera_data.zone_id,
            tags=camera_data.tags,
        )

        # Create camera in database
        camera = await camera_service.create_camera(db_camera_data, current_user.id)

        # Start health monitoring in background
        background_tasks.add_task(start_stream_monitoring, camera.id)

        # Invalidate cache
        await cache.delete("cameras:list:*")

        logger.info(
            "Camera created successfully",
            camera_id=camera.id,
            name=camera_data.name,
            user_id=current_user.id,
        )

        return CameraResponse(
            id=camera.id,
            name=camera.name,
            description=camera.description,
            location=camera.location,
            coordinates=camera.coordinates,
            camera_type=camera.camera_type,
            stream_url=camera.stream_url,
            stream_protocol=camera.stream_protocol,
            backup_stream_url=camera.backup_stream_url,
            status=CameraStatus(camera.status),
            config=camera.config,
            health=None,  # Will be populated by monitoring
            zone_id=camera.zone_id,
            tags=camera.tags,
            is_active=camera.is_active,
            created_at=camera.created_at,
            updated_at=camera.updated_at,
            last_seen_at=camera.last_seen_at,
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
    camera_service: CameraService = Depends(get_camera_service),
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
    camera = await camera_service.get_camera_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    # Get current health status from cache or check stream
    health = await get_stream_health_from_cache(camera_id, cache)
    if not health:
        health = await check_stream_health(camera_id, camera.stream_url, cache)

    camera_response = CameraResponse(
        id=camera.id,
        name=camera.name,
        description=camera.description,
        location=camera.location,
        coordinates=camera.coordinates,
        camera_type=camera.camera_type,
        stream_url=camera.stream_url,
        stream_protocol=camera.stream_protocol,
        backup_stream_url=camera.backup_stream_url,
        status=CameraStatus(camera.status),
        config=camera.config,
        health=health,
        zone_id=camera.zone_id,
        tags=camera.tags,
        is_active=camera.is_active,
        created_at=camera.created_at,
        updated_at=camera.updated_at,
        last_seen_at=camera.last_seen_at,
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
    # Check if camera exists first
    existing_camera = await camera_service.get_camera_by_id(camera_id)
    if not existing_camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    try:
        # Convert API schema to database schema for update
        # Convert update data
        update_dict = camera_data.model_dump(exclude_unset=True, exclude_none=True)

        # Handle config update if provided
        if "config" in update_dict and update_dict["config"]:
            update_dict["config"] = update_dict["config"].model_dump()

        db_update_data = CameraUpdateSchema(**update_dict)

        # Update camera in database
        updated_camera = await camera_service.update_camera(camera_id, db_update_data)
        if not updated_camera:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera {camera_id} not found",
            )

        # Invalidate caches
        await cache.delete(f"camera:{camera_id}")
        await cache.delete("cameras:list:*")

        logger.info(
            "Camera updated successfully",
            camera_id=camera_id,
            updated_fields=list(update_dict.keys()),
            user_id=current_user.id,
        )

        return await get_camera(camera_id, current_user, camera_service, cache)

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
    camera_service: CameraService = Depends(get_camera_service),
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
    # Check if camera exists
    camera = await camera_service.get_camera_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    try:
        # Stop monitoring
        background_tasks.add_task(stop_stream_monitoring, camera_id)

        # Delete camera from database
        success = await camera_service.delete_camera(camera_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera {camera_id} not found",
            )

        # Remove from stream health cache
        # Clear stream health cache
        await clear_stream_health_cache(camera_id, cache)

        # Invalidate caches
        await cache.delete(f"camera:{camera_id}")
        await cache.delete("cameras:list:*")

        logger.info(
            "Camera deleted successfully",
            camera_id=camera_id,
            name=camera.name,
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
    camera_service: CameraService = Depends(get_camera_service),
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
    # Check if camera exists
    camera = await camera_service.get_camera_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    try:
        action = stream_request.action

        # Update camera status based on action
        if action == "start":
            new_status = CameraStatus.STREAMING
            background_tasks.add_task(start_stream_monitoring, camera_id)
        elif action == "stop":
            new_status = CameraStatus.STOPPED
            background_tasks.add_task(stop_stream_monitoring, camera_id)
        elif action == "restart":
            new_status = CameraStatus.STREAMING
            background_tasks.add_task(stop_stream_monitoring, camera_id)
            background_tasks.add_task(start_stream_monitoring, camera_id)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {action}",
            )

        # Update camera status in database
        await camera_service.update_camera_status(camera_id, new_status)

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
                "status": new_status.value,
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
    camera_service: CameraService = Depends(get_camera_service),
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
    camera = await camera_service.get_camera_by_id(camera_id)
    if not camera:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Camera {camera_id} not found",
        )

    # Get or refresh health data from cache
    health = await get_stream_health_from_cache(camera_id, cache)
    if not health:
        health = await check_stream_health(camera_id, camera.stream_url, cache)

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
    camera_service: CameraService = Depends(get_camera_service),
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
            # Check if camera exists
            camera = await camera_service.get_camera_by_id(camera_id)
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
                await camera_service.update_camera_status(camera_id, CameraStatus.STREAMING)
                background_tasks.add_task(start_stream_monitoring, camera_id)
                message = "Stream started"

            elif operation == "stop_streams":
                await camera_service.update_camera_status(camera_id, CameraStatus.STOPPED)
                background_tasks.add_task(stop_stream_monitoring, camera_id)
                message = "Stream stopped"

            elif operation == "restart_streams":
                await camera_service.update_camera_status(camera_id, CameraStatus.STREAMING)
                background_tasks.add_task(stop_stream_monitoring, camera_id)
                background_tasks.add_task(start_stream_monitoring, camera_id)
                message = "Stream restarted"

            elif operation == "update_config":
                if "config" in parameters:
                    # Create update schema with new config
                    update_data = CameraUpdateSchema(config=parameters["config"])
                    await camera_service.update_camera(camera_id, update_data)
                message = "Configuration updated"

            elif operation == "enable_analytics":
                # Update camera config to enable analytics
                new_config = camera.config.copy()
                new_config["analytics_enabled"] = True
                update_data = CameraUpdateSchema(config=new_config)
                await camera_service.update_camera(camera_id, update_data)
                message = "Analytics enabled"

            elif operation == "disable_analytics":
                # Update camera config to disable analytics
                new_config = camera.config.copy()
                new_config["analytics_enabled"] = False
                update_data = CameraUpdateSchema(config=new_config)
                await camera_service.update_camera(camera_id, update_data)
                message = "Analytics disabled"

            elif operation == "delete":
                success = await camera_service.delete_camera(camera_id)
                if not success:
                    results.append(
                        CameraBatchResult(
                            camera_id=camera_id,
                            success=False,
                            message=f"Failed to delete camera {camera_id}",
                        )
                    )
                    continue

                # Clear stream health cache
                await clear_stream_health_cache(camera_id, cache)
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
    camera_service: CameraService = Depends(get_camera_service),
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
    camera = await camera_service.get_camera_by_id(camera_id)
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

    # Calculate real statistics from database
    from sqlalchemy import and_, func, select

    from ...models.analytics import RuleViolation, TrafficAnomaly, TrafficMetrics
    from ...models.frame_metadata import FrameMetadata

    try:
        now = datetime.now(UTC)
        period_start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)

        # Query frame processing statistics
        frame_query = select(
            func.count(FrameMetadata.id).label("total_frames"),
            func.avg(FrameMetadata.processing_time_ms).label("avg_processing_time"),
            func.sum(FrameMetadata.vehicle_count).label("total_vehicles")
        ).where(
            and_(
                FrameMetadata.camera_id == camera_id,
                FrameMetadata.created_at >= period_start,
                FrameMetadata.created_at <= now
            )
        )

        frame_result = await db.execute(frame_query)
        frame_stats = frame_result.first()

        # Query violation statistics
        violation_query = select(
            func.count(RuleViolation.id).label("violation_count")
        ).where(
            and_(
                RuleViolation.camera_id == camera_id,
                RuleViolation.detection_time >= period_start,
                RuleViolation.detection_time <= now
            )
        )

        violation_result = await db.execute(violation_query)
        violation_stats = violation_result.scalar()

        # Query anomaly statistics
        anomaly_query = select(
            func.count(TrafficAnomaly.id).label("anomaly_count")
        ).where(
            and_(
                TrafficAnomaly.camera_id == camera_id,
                TrafficAnomaly.detection_time >= period_start,
                TrafficAnomaly.detection_time <= now
            )
        )

        anomaly_result = await db.execute(anomaly_query)
        anomaly_stats = anomaly_result.scalar()

        # Query hourly activity for last 24 hours
        last_24h_start = now - timedelta(hours=24)
        hourly_query = select(
            func.extract('hour', TrafficMetrics.timestamp).label("hour"),
            func.sum(TrafficMetrics.total_vehicles).label("vehicle_count")
        ).where(
            and_(
                TrafficMetrics.camera_id == camera_id,
                TrafficMetrics.timestamp >= last_24h_start,
                TrafficMetrics.timestamp <= now,
                TrafficMetrics.aggregation_period == "1hour"
            )
        ).group_by(func.extract('hour', TrafficMetrics.timestamp))

        hourly_result = await db.execute(hourly_query)
        hourly_data = {f"{int(row.hour):02d}:00": int(row.vehicle_count or 0) for row in hourly_result}

        # Fill missing hours with 0
        last_24h_activity = {}
        for i in range(24):
            hour_key = f"{i:02d}:00"
            last_24h_activity[hour_key] = hourly_data.get(hour_key, 0)

        # Calculate uptime percentage
        uptime_query = select(
            func.count(FrameMetadata.id).label("active_periods")
        ).where(
            and_(
                FrameMetadata.camera_id == camera_id,
                FrameMetadata.created_at >= period_start,
                FrameMetadata.created_at <= now
            )
        )

        uptime_result = await db.execute(uptime_query)
        active_periods = uptime_result.scalar() or 0

        # Calculate uptime as percentage of expected frames (assuming 1 frame per minute)
        expected_frames = int((now - period_start).total_seconds() / 60)
        uptime_percentage = min(100.0, (active_periods / max(1, expected_frames)) * 100) if expected_frames > 0 else 0.0

        stats = CameraStats(
            camera_id=camera_id,
            frames_processed=int(frame_stats.total_frames or 0),
            vehicles_detected=int(frame_stats.total_vehicles or 0),
            incidents_detected=int((violation_stats or 0) + (anomaly_stats or 0)),
            uptime_percentage=uptime_percentage,
            avg_processing_time=float(frame_stats.avg_processing_time or 0.0),
            last_24h_activity=last_24h_activity,
            period_start=period_start,
            period_end=now,
        )

    except Exception as e:
        logger.error(f"Failed to calculate camera statistics from database: {e}")
        # Fallback to basic stats from camera record
        stats = CameraStats(
            camera_id=camera_id,
            frames_processed=camera.total_frames_processed,
            vehicles_detected=0,  # Not available without DB query
            incidents_detected=0,  # Not available without DB query
            uptime_percentage=camera.uptime_percentage or 0.0,
            avg_processing_time=camera.avg_processing_time or 0.0,
            last_24h_activity={f"{i:02d}:00": 0 for i in range(24)},
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
