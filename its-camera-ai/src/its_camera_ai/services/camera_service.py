"""Async CRUD service for camera registry management.

Provides high-performance camera operations with optimized queries,
batch operations, and real-time status updates for 100+ concurrent cameras.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.exc import IntegrityError

from ..api.schemas.database import (
    CameraCreateSchema,
    CameraUpdateSchema,
)
from ..core.exceptions import DatabaseError, NotFoundError
from ..core.logging import get_logger
from ..models import Camera, CameraStatus
from ..repositories.camera_repository import CameraRepository

logger = get_logger(__name__)


class CameraService:
    """Async CRUD service for camera registry operations.

    Optimized for high-throughput operations with efficient querying,
    batch operations, and real-time status management.
    """

    def __init__(self, camera_repository: CameraRepository):
        self.camera_repository = camera_repository

    async def create_camera(
        self, camera_data: CameraCreateSchema, user_id: str | None = None
    ) -> Camera:
        """Create a new camera with validation and settings initialization.

        Args:
            camera_data: Camera creation data
            user_id: Optional user ID for audit logging

        Returns:
            Created camera instance

        Raises:
            DatabaseError: If camera creation fails
        """
        try:
            # Convert schema to dict and handle special fields
            camera_dict = camera_data.model_dump(exclude={"config"})

            # Handle coordinates conversion
            if camera_data.coordinates:
                camera_dict["coordinates"] = {
                    "latitude": camera_data.coordinates.latitude,
                    "longitude": camera_data.coordinates.longitude,
                    "altitude": camera_data.coordinates.altitude,
                }

            # Convert config to dict
            camera_dict["config"] = camera_data.config.model_dump()

            # Set initial status and metadata
            camera_dict["status"] = CameraStatus.OFFLINE
            camera_dict["total_frames_processed"] = 0

            # Create camera using repository
            camera = await self.camera_repository.create(**camera_dict)

            # TODO: Create default camera settings - requires CameraSettings repository
            # For now, we'll create the camera without settings

            logger.info(
                "Camera created successfully",
                camera_id=camera.id,
                name=camera.name,
                user_id=user_id,
            )

            return camera

        except IntegrityError as e:
            logger.error("Camera creation failed - integrity error", error=str(e))
            raise DatabaseError("Camera name or stream URL already exists") from e
        except Exception as e:
            logger.error("Camera creation failed", error=str(e))
            raise DatabaseError(f"Failed to create camera: {str(e)}") from e

    async def get_camera_by_id(
        self, camera_id: str, include_settings: bool = False
    ) -> Camera | None:
        """Get camera by ID with optional settings loading.

        Args:
            camera_id: Camera unique identifier
            include_settings: Whether to load camera settings (not yet implemented)

        Returns:
            Camera instance if found, None otherwise
        """
        try:
            return await self.camera_repository.get_by_id(camera_id)

        except Exception as e:
            logger.error(
                "Failed to get camera by ID", camera_id=camera_id, error=str(e)
            )
            raise DatabaseError(f"Failed to retrieve camera: {str(e)}") from e

    async def get_cameras_by_zone(
        self, zone_id: str, active_only: bool = True
    ) -> list[Camera]:
        """Get cameras in a specific zone.

        Args:
            zone_id: Traffic zone identifier
            active_only: Only return active cameras

        Returns:
            List of cameras in the zone
        """
        # TODO: This method needs a specialized repository method for zone filtering
        # For now, we'll get all cameras and filter in memory (not optimal for large datasets)
        try:
            all_cameras = await self.camera_repository.list_all(limit=1000)  # Increased limit
            filtered_cameras = [
                camera for camera in all_cameras
                if camera.zone_id == zone_id and (not active_only or camera.is_active)
            ]
            return sorted(filtered_cameras, key=lambda c: c.name)

        except Exception as e:
            logger.error("Failed to get cameras by zone", zone_id=zone_id, error=str(e))
            raise DatabaseError(f"Failed to retrieve cameras by zone: {str(e)}") from e

    async def get_cameras_by_status(
        self,
        status: CameraStatus,
        limit: int | None = None,
        include_settings: bool = False,
    ) -> list[Camera]:
        """Get cameras by status.

        Args:
            status: Camera status to filter by
            limit: Maximum number of cameras to return
            include_settings: Whether to load camera settings (not yet implemented)

        Returns:
            List of cameras with the specified status
        """
        try:
            return await self.camera_repository.get_by_status(
                status,
                limit=limit or 100,
                offset=0
            )

        except Exception as e:
            logger.error("Failed to get cameras by status", status=status, error=str(e))
            raise DatabaseError(
                f"Failed to retrieve cameras by status: {str(e)}"
            ) from e

    async def update_camera(
        self, camera_id: str, update_data: CameraUpdateSchema
    ) -> Camera | None:
        """Update camera information.

        Args:
            camera_id: Camera unique identifier
            update_data: Camera update data

        Returns:
            Updated camera instance if found, None otherwise

        Raises:
            DatabaseError: If update fails
        """
        try:
            # Convert schema to dict excluding None values
            update_dict = update_data.model_dump(exclude_none=True, exclude={"config"})

            # Handle coordinates conversion
            if update_data.coordinates:
                update_dict["coordinates"] = {
                    "latitude": update_data.coordinates.latitude,
                    "longitude": update_data.coordinates.longitude,
                    "altitude": update_data.coordinates.altitude,
                }

            # Handle config update
            if update_data.config:
                update_dict["config"] = update_data.config.model_dump()

            # Add updated_at timestamp
            update_dict["updated_at"] = datetime.now(UTC)

            # Update using repository
            try:
                updated_camera = await self.camera_repository.update(camera_id, **update_dict)

                logger.info(
                    "Camera updated successfully",
                    camera_id=camera_id,
                    updated_fields=list(update_dict.keys()),
                )

                return updated_camera

            except NotFoundError:
                return None

        except Exception as e:
            logger.error("Camera update failed", camera_id=camera_id, error=str(e))
            raise DatabaseError(f"Failed to update camera: {str(e)}") from e

    async def update_camera_status(
        self,
        camera_id: str,
        status: CameraStatus,
        error_message: str | None = None,
        performance_metrics: dict[str, Any] | None = None,
    ) -> bool:
        """Update camera status and related metrics.

        Args:
            camera_id: Camera unique identifier
            status: New camera status
            error_message: Optional error message
            performance_metrics: Optional performance metrics

        Returns:
            True if update successful, False if camera not found
        """
        try:
            # First check if camera exists
            camera = await self.get_camera_by_id(camera_id)
            if not camera:
                return False

            # Prepare update data
            update_dict = {"status": status}

            if status in (CameraStatus.ONLINE, CameraStatus.STREAMING):
                update_dict["last_seen_at"] = datetime.now(UTC)

            # Handle performance metrics update
            if performance_metrics:
                if "total_frames_processed" in performance_metrics:
                    update_dict["total_frames_processed"] = performance_metrics[
                        "total_frames_processed"
                    ]
                if "avg_processing_time" in performance_metrics:
                    update_dict["avg_processing_time"] = performance_metrics[
                        "avg_processing_time"
                    ]
                if "uptime_percentage" in performance_metrics:
                    update_dict["uptime_percentage"] = performance_metrics["uptime_percentage"]

            # Handle error message by updating config
            if error_message:
                config = camera.config.copy() if camera.config else {}
                if "errors" not in config:
                    config["errors"] = []
                config["errors"].append(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "message": error_message,
                        "status": status.value,
                    }
                )
                # Keep only last 10 errors
                config["errors"] = config["errors"][-10:]
                update_dict["config"] = config

            # Update using repository
            await self.camera_repository.update(camera_id, **update_dict)

            logger.info(
                "Camera status updated",
                camera_id=camera_id,
                old_status=camera.status,
                new_status=status.value,
            )

            return True

        except NotFoundError:
            return False
        except Exception as e:
            logger.error(
                "Camera status update failed", camera_id=camera_id, error=str(e)
            )
            raise DatabaseError(f"Failed to update camera status: {str(e)}") from e

    async def batch_update_camera_status(
        self, status_updates: dict[str, tuple[CameraStatus, dict[str, Any] | None]]
    ) -> dict[str, bool]:
        """Batch update multiple camera statuses.

        Args:
            status_updates: Dict mapping camera_id to (status, metrics)

        Returns:
            Dict mapping camera_id to success status
        """
        results = {}

        try:
            for camera_id, (status, metrics) in status_updates.items():
                try:
                    result = await self.update_camera_status(
                        camera_id, status, performance_metrics=metrics
                    )
                    results[camera_id] = result
                except Exception as e:
                    logger.error(
                        "Failed to update camera status in batch",
                        camera_id=camera_id,
                        error=str(e),
                    )
                    results[camera_id] = False

            return results

        except Exception as e:
            logger.error("Batch camera status update failed", error=str(e))
            raise DatabaseError(f"Batch status update failed: {str(e)}") from e

    async def get_cameras_with_pagination(
        self,
        page: int = 1,
        size: int = 20,
        filters: dict[str, Any] | None = None,
        order_by: str = "name",
        order_desc: bool = False,
    ) -> tuple[list[Camera], int]:
        """Get cameras with pagination and filtering.

        Args:
            page: Page number (1-based)
            size: Page size
            filters: Optional filters dict (limited filtering for now)
            order_by: Field to order by
            order_desc: Order descending if True

        Returns:
            Tuple of (cameras, total_count)
        """
        try:
            # TODO: This is a simplified implementation. For production,
            # we need to add filtering capabilities to the repository.

            # Calculate offset
            offset = (page - 1) * size

            # Get cameras with basic pagination from repository
            cameras = await self.camera_repository.list_all(
                limit=size,
                offset=offset,
                order_by=order_by,
                order_desc=order_desc
            )

            # Get total count
            total_count = await self.camera_repository.count()

            # Apply basic filtering in memory (not optimal but functional)
            if filters:
                filtered_cameras = []
                for camera in cameras:
                    if "status" in filters and camera.status != filters["status"]:
                        continue
                    if "is_active" in filters and camera.is_active != filters["is_active"]:
                        continue
                    if "search" in filters:
                        search_term = filters["search"].lower()
                        searchable_text = f"{camera.name} {camera.location or ''} {camera.description or ''}".lower()
                        if search_term not in searchable_text:
                            continue
                    filtered_cameras.append(camera)
                cameras = filtered_cameras

            return cameras, total_count

        except Exception as e:
            logger.error("Failed to get cameras with pagination", error=str(e))
            raise DatabaseError(f"Failed to retrieve cameras: {str(e)}") from e

    async def get_camera_health_summary(self, camera_id: str) -> dict[str, Any]:
        """Get comprehensive camera health summary.

        Args:
            camera_id: Camera unique identifier

        Returns:
            Camera health summary dict
        """
        try:
            camera = await self.get_camera_by_id(camera_id)
            if not camera:
                raise NotFoundError("Camera not found")

            # Calculate health metrics
            now = datetime.now(UTC)
            last_seen_minutes = None
            if camera.last_seen_at:
                last_seen_minutes = (now - camera.last_seen_at).total_seconds() / 60

            # Determine health status
            is_healthy = (
                camera.is_active
                and camera.status
                in [CameraStatus.ONLINE.value, CameraStatus.STREAMING.value]
                and (
                    not last_seen_minutes or last_seen_minutes < 5
                )  # Seen within 5 minutes
            )

            return {
                "camera_id": camera_id,
                "name": camera.name,
                "status": camera.status,
                "is_healthy": is_healthy,
                "is_active": camera.is_active,
                "last_seen_at": camera.last_seen_at,
                "last_seen_minutes_ago": last_seen_minutes,
                "total_frames_processed": camera.total_frames_processed,
                "avg_processing_time_ms": camera.avg_processing_time,
                "uptime_percentage": camera.uptime_percentage,
                "recent_errors": (camera.config or {}).get("errors", [])[-5:],  # Last 5 errors
            }

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(
                "Failed to get camera health summary", camera_id=camera_id, error=str(e)
            )
            raise DatabaseError(f"Failed to get camera health: {str(e)}") from e

    async def get_offline_cameras(self, threshold_minutes: int = 5) -> list[Camera]:
        """Get cameras that are offline or haven't been seen recently.

        Args:
            threshold_minutes: Minutes threshold for considering camera offline

        Returns:
            List of offline cameras
        """
        try:
            threshold_time = datetime.now(UTC) - timedelta(minutes=threshold_minutes)

            # Get offline cameras using repository method
            offline_cameras = await self.camera_repository.get_by_status(CameraStatus.OFFLINE)
            error_cameras = await self.camera_repository.get_by_status(CameraStatus.ERROR)

            # Get all active cameras to check for stale ones
            all_cameras = await self.camera_repository.list_all(limit=1000)  # TODO: implement proper filtering

            # Filter for stale cameras (haven't been seen recently)
            stale_cameras = [
                camera for camera in all_cameras
                if camera.is_active and (
                    camera.last_seen_at is None or
                    camera.last_seen_at < threshold_time
                ) and camera.status not in [CameraStatus.OFFLINE.value, CameraStatus.ERROR.value]
            ]

            # Combine and deduplicate
            offline_camera_ids = set()
            result_cameras = []

            for camera_list in [offline_cameras, error_cameras, stale_cameras]:
                for camera in camera_list:
                    if camera.id not in offline_camera_ids and camera.is_active:
                        offline_camera_ids.add(camera.id)
                        result_cameras.append(camera)

            return result_cameras

        except Exception as e:
            logger.error("Failed to get offline cameras", error=str(e))
            raise DatabaseError(f"Failed to get offline cameras: {str(e)}") from e

    async def delete_camera(self, camera_id: str) -> bool:
        """Delete camera and all related data.

        Args:
            camera_id: Camera unique identifier

        Returns:
            True if camera was deleted, False if not found
        """
        try:
            result = await self.camera_repository.delete(camera_id)

            if result:
                logger.info("Camera deleted successfully", camera_id=camera_id)

            return result

        except Exception as e:
            logger.error("Camera deletion failed", camera_id=camera_id, error=str(e))
            raise DatabaseError(f"Failed to delete camera: {str(e)}") from e

    async def get_system_overview(self) -> dict[str, Any]:
        """Get system-wide camera overview statistics.

        Returns:
            System overview dict with camera statistics
        """
        try:
            # Get status counts from repository
            status_counts = await self.camera_repository.get_status_counts()

            # Get total count
            total_cameras = await self.camera_repository.count()

            # Get all cameras to calculate active/inactive split
            # TODO: This is not optimal for large datasets - need repository method for this
            all_cameras = await self.camera_repository.list_all(limit=10000)
            active_cameras = sum(1 for camera in all_cameras if camera.is_active)

            return {
                "total_cameras": total_cameras,
                "active_cameras": active_cameras,
                "inactive_cameras": total_cameras - active_cameras,
                "status_breakdown": {
                    "online": status_counts.get(CameraStatus.ONLINE.value, 0),
                    "streaming": status_counts.get(CameraStatus.STREAMING.value, 0),
                    "offline": status_counts.get(CameraStatus.OFFLINE.value, 0),
                    "error": status_counts.get(CameraStatus.ERROR.value, 0),
                    "maintenance": status_counts.get(CameraStatus.MAINTENANCE.value, 0),
                },
                "healthy_cameras": (
                    status_counts.get(CameraStatus.ONLINE.value, 0)
                    + status_counts.get(CameraStatus.STREAMING.value, 0)
                ),
                "problematic_cameras": (
                    status_counts.get(CameraStatus.ERROR.value, 0)
                    + status_counts.get(CameraStatus.OFFLINE.value, 0)
                ),
            }

        except Exception as e:
            logger.error("Failed to get system overview", error=str(e))
            raise DatabaseError(f"Failed to get system overview: {str(e)}") from e
