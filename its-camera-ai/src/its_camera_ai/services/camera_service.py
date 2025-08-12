"""Async CRUD service for camera registry management.

Provides high-performance camera operations with optimized queries,
batch operations, and real-time status updates for 100+ concurrent cameras.
"""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..api.schemas.database import (
    CameraCreateSchema,
    CameraUpdateSchema,
)
from ..core.exceptions import DatabaseError, NotFoundError
from ..core.logging import get_logger
from ..models import Camera, CameraSettings, CameraStatus
from .base_service import BaseAsyncService

logger = get_logger(__name__)


class CameraService(BaseAsyncService[Camera]):
    """Async CRUD service for camera registry operations.

    Optimized for high-throughput operations with efficient querying,
    batch operations, and real-time status management.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, Camera)

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

            camera = Camera(**camera_dict)

            self.session.add(camera)
            await self.session.flush()

            # Create default camera settings
            settings = CameraSettings(
                camera_id=camera.id,
                detection_enabled=True,
                tracking_enabled=True,
                analytics_enabled=True,
                model_name="yolo11n",
                confidence_threshold=0.5,
                nms_threshold=0.4,
                max_batch_size=8,
                quality_threshold=0.7,
                max_processing_time=100,
                storage_retention_days=7,
                alert_thresholds={},
                notification_settings={},
                advanced_settings={},
            )

            self.session.add(settings)
            await self.session.commit()

            logger.info(
                "Camera created successfully",
                camera_id=camera.id,
                name=camera.name,
                user_id=user_id,
            )

            return camera

        except IntegrityError as e:
            await self.session.rollback()
            logger.error("Camera creation failed - integrity error", error=str(e))
            raise DatabaseError("Camera name or stream URL already exists") from e
        except Exception as e:
            await self.session.rollback()
            logger.error("Camera creation failed", error=str(e))
            raise DatabaseError(f"Failed to create camera: {str(e)}") from e

    async def get_camera_by_id(
        self, camera_id: str, include_settings: bool = False
    ) -> Camera | None:
        """Get camera by ID with optional settings loading.

        Args:
            camera_id: Camera unique identifier
            include_settings: Whether to load camera settings

        Returns:
            Camera instance if found, None otherwise
        """
        try:
            query = select(Camera).where(Camera.id == camera_id)

            if include_settings:
                query = query.options(selectinload(Camera.settings))

            result = await self.session.execute(query)
            return result.scalar_one_or_none()

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
        try:
            query = select(Camera).where(Camera.zone_id == zone_id)

            if active_only:
                query = query.where(Camera.is_active == True)  # noqa: E712

            query = query.order_by(Camera.name)

            result = await self.session.execute(query)
            return list(result.scalars().all())

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
            include_settings: Whether to load camera settings

        Returns:
            List of cameras with the specified status
        """
        try:
            query = select(Camera).where(Camera.status == status.value)

            if include_settings:
                query = query.options(selectinload(Camera.settings))

            query = query.order_by(Camera.last_seen_at.desc())

            if limit:
                query = query.limit(limit)

            result = await self.session.execute(query)
            return list(result.scalars().all())

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
            camera = await self.get_camera_by_id(camera_id)
            if not camera:
                return None

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

            # Update camera fields
            for key, value in update_dict.items():
                setattr(camera, key, value)

            camera.updated_at = datetime.utcnow()

            await self.session.commit()

            logger.info(
                "Camera updated successfully",
                camera_id=camera_id,
                updated_fields=list(update_dict.keys()),
            )

            return camera

        except Exception as e:
            await self.session.rollback()
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
            camera = await self.get_camera_by_id(camera_id)
            if not camera:
                return False

            # Update status and timestamp
            camera.status = status.value

            if status in (CameraStatus.ONLINE, CameraStatus.STREAMING):
                camera.last_seen_at = datetime.utcnow()

            # Handle performance metrics update
            if performance_metrics:
                if "total_frames_processed" in performance_metrics:
                    camera.total_frames_processed = performance_metrics[
                        "total_frames_processed"
                    ]
                if "avg_processing_time" in performance_metrics:
                    camera.avg_processing_time = performance_metrics[
                        "avg_processing_time"
                    ]
                if "uptime_percentage" in performance_metrics:
                    camera.uptime_percentage = performance_metrics["uptime_percentage"]

            # Handle error message
            if error_message:
                if "errors" not in camera.config:
                    camera.config["errors"] = []
                camera.config["errors"].append(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "message": error_message,
                        "status": status.value,
                    }
                )
                # Keep only last 10 errors
                camera.config["errors"] = camera.config["errors"][-10:]

            await self.session.commit()

            logger.info(
                "Camera status updated",
                camera_id=camera_id,
                old_status=camera.status,
                new_status=status.value,
            )

            return True

        except Exception as e:
            await self.session.rollback()
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
            filters: Optional filters dict
            order_by: Field to order by
            order_desc: Order descending if True

        Returns:
            Tuple of (cameras, total_count)
        """
        try:
            # Build base query
            query = select(Camera)
            count_query = select(func.count(Camera.id))

            # Apply filters
            if filters:
                conditions = []

                if "status" in filters:
                    conditions.append(Camera.status == filters["status"])

                if "is_active" in filters:
                    conditions.append(Camera.is_active == filters["is_active"])

                if "camera_type" in filters:
                    conditions.append(Camera.camera_type == filters["camera_type"])

                if "zone_id" in filters:
                    conditions.append(Camera.zone_id == filters["zone_id"])

                if "search" in filters:
                    search = f"%{filters['search']}%"
                    conditions.append(
                        or_(
                            Camera.name.ilike(search),
                            Camera.location.ilike(search),
                            Camera.description.ilike(search),
                        )
                    )

                if "tags" in filters and filters["tags"]:
                    # JSONB contains query for tags
                    conditions.append(Camera.tags.contains(filters["tags"]))

                if conditions:
                    where_clause = and_(*conditions)
                    query = query.where(where_clause)
                    count_query = count_query.where(where_clause)

            # Apply ordering
            order_column = getattr(Camera, order_by, Camera.name)
            if order_desc:
                query = query.order_by(order_column.desc())
            else:
                query = query.order_by(order_column)

            # Apply pagination
            offset = (page - 1) * size
            query = query.offset(offset).limit(size)

            # Execute queries
            cameras_result = await self.session.execute(query)
            count_result = await self.session.execute(count_query)

            cameras = list(cameras_result.scalars().all())
            total_count = count_result.scalar()

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
            camera = await self.get_camera_by_id(camera_id, include_settings=True)
            if not camera:
                raise NotFoundError("Camera not found")

            # Calculate health metrics
            now = datetime.utcnow()
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
                "recent_errors": camera.config.get("errors", [])[-5:],  # Last 5 errors
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
            threshold_time = datetime.utcnow() - timedelta(minutes=threshold_minutes)

            query = (
                select(Camera)
                .where(
                    or_(
                        Camera.status == CameraStatus.OFFLINE.value,
                        Camera.status == CameraStatus.ERROR.value,
                        and_(
                            Camera.last_seen_at.is_not(None),
                            Camera.last_seen_at < threshold_time,
                        ),
                        Camera.last_seen_at.is_(None),
                    )
                )
                .where(Camera.is_active == True)
            )  # noqa: E712

            result = await self.session.execute(query)
            return list(result.scalars().all())

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
            camera = await self.get_camera_by_id(camera_id)
            if not camera:
                return False

            await self.session.delete(camera)
            await self.session.commit()

            logger.info("Camera deleted successfully", camera_id=camera_id)
            return True

        except Exception as e:
            await self.session.rollback()
            logger.error("Camera deletion failed", camera_id=camera_id, error=str(e))
            raise DatabaseError(f"Failed to delete camera: {str(e)}") from e

    async def get_system_overview(self) -> dict[str, Any]:
        """Get system-wide camera overview statistics.

        Returns:
            System overview dict with camera statistics
        """
        try:
            # Count cameras by status
            status_query = (
                select(Camera.status, func.count(Camera.id).label("count"))
                .where(Camera.is_active == True)  # noqa: E712
                .group_by(Camera.status)
            )

            status_result = await self.session.execute(status_query)
            status_counts = {row.status: row.count for row in status_result}

            # Count total and active cameras
            total_query = select(func.count(Camera.id))
            active_query = select(func.count(Camera.id)).where(
                Camera.is_active == True
            )  # noqa: E712

            total_result = await self.session.execute(total_query)
            active_result = await self.session.execute(active_query)

            total_cameras = total_result.scalar()
            active_cameras = active_result.scalar()

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
