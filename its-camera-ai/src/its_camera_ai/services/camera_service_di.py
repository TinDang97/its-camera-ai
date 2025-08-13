"""Dependency injection version of Camera Service.

Refactored camera service that uses the repository pattern and
dependency injection for clean architecture separation.
"""

from datetime import datetime
from typing import Any

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
    """Camera service using repository pattern and dependency injection.
    
    Provides high-level camera operations while delegating data access
    to the camera repository for clean architecture separation.
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
            # Check if camera with same name already exists
            existing_camera = await self.camera_repository.get_by_name(camera_data.name)
            if existing_camera:
                raise DatabaseError(f"Camera with name '{camera_data.name}' already exists")

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

            # Create camera through repository
            camera = await self.camera_repository.create(**camera_dict)

            logger.info(
                "Camera created successfully",
                camera_id=camera.id,
                name=camera.name,
                location=camera.location,
                user_id=user_id
            )

            return camera

        except Exception as e:
            logger.error(
                "Failed to create camera",
                error=str(e),
                camera_name=camera_data.name,
                user_id=user_id
            )
            raise DatabaseError("Camera creation failed", cause=e) from e

    async def get_camera(self, camera_id: str) -> Camera:
        """Get camera by ID.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Camera instance
            
        Raises:
            NotFoundError: If camera not found
        """
        return await self.camera_repository.get_by_id_or_raise(camera_id)

    async def get_camera_by_name(self, name: str) -> Camera | None:
        """Get camera by name.
        
        Args:
            name: Camera name
            
        Returns:
            Camera instance or None if not found
        """
        return await self.camera_repository.get_by_name(name)

    async def update_camera(
        self,
        camera_id: str,
        camera_data: CameraUpdateSchema,
        user_id: str | None = None
    ) -> Camera:
        """Update camera information.
        
        Args:
            camera_id: Camera identifier
            camera_data: Update data
            user_id: Optional user ID for audit logging
            
        Returns:
            Updated camera instance
            
        Raises:
            NotFoundError: If camera not found
            DatabaseError: If update fails
        """
        try:
            # Prepare update data
            update_dict = camera_data.model_dump(exclude_unset=True, exclude={"config"})

            # Handle coordinates conversion if present
            if camera_data.coordinates:
                update_dict["coordinates"] = {
                    "latitude": camera_data.coordinates.latitude,
                    "longitude": camera_data.coordinates.longitude,
                    "altitude": camera_data.coordinates.altitude,
                }

            # Handle config update if present
            if camera_data.config:
                update_dict["config"] = camera_data.config.model_dump()

            # Update through repository
            camera = await self.camera_repository.update(camera_id, **update_dict)

            logger.info(
                "Camera updated successfully",
                camera_id=camera_id,
                updated_fields=list(update_dict.keys()),
                user_id=user_id
            )

            return camera

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(
                "Failed to update camera",
                camera_id=camera_id,
                error=str(e),
                user_id=user_id
            )
            raise DatabaseError("Camera update failed", cause=e) from e

    async def delete_camera(self, camera_id: str, user_id: str | None = None) -> bool:
        """Delete camera.
        
        Args:
            camera_id: Camera identifier
            user_id: Optional user ID for audit logging
            
        Returns:
            True if deleted successfully
            
        Raises:
            DatabaseError: If deletion fails
        """
        try:
            # Check if camera exists first
            camera = await self.camera_repository.get_by_id(camera_id)
            if not camera:
                return False

            # Delete through repository
            deleted = await self.camera_repository.delete(camera_id)

            if deleted:
                logger.info(
                    "Camera deleted successfully",
                    camera_id=camera_id,
                    camera_name=camera.name,
                    user_id=user_id
                )

            return deleted

        except Exception as e:
            logger.error(
                "Failed to delete camera",
                camera_id=camera_id,
                error=str(e),
                user_id=user_id
            )
            raise DatabaseError("Camera deletion failed", cause=e) from e

    async def list_cameras(
        self,
        limit: int = 100,
        offset: int = 0,
        status: CameraStatus | None = None,
        location: str | None = None
    ) -> list[Camera]:
        """List cameras with optional filtering.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            status: Optional status filter
            location: Optional location filter
            
        Returns:
            List of cameras
        """
        if status:
            return await self.camera_repository.get_by_status(status, limit, offset)
        elif location:
            cameras = await self.camera_repository.get_by_location(location)
            return cameras[offset:offset + limit]
        else:
            return await self.camera_repository.list_all(limit, offset)

    async def get_online_cameras(self) -> list[Camera]:
        """Get all online cameras.
        
        Returns:
            List of online cameras
        """
        return await self.camera_repository.get_online_cameras()

    async def get_streaming_cameras(self) -> list[Camera]:
        """Get all cameras currently streaming.
        
        Returns:
            List of streaming cameras
        """
        return await self.camera_repository.get_streaming_cameras()

    async def update_camera_status(
        self,
        camera_id: str,
        status: CameraStatus,
        status_message: str | None = None
    ) -> bool:
        """Update camera status.
        
        Args:
            camera_id: Camera identifier
            status: New status
            status_message: Optional status message
            
        Returns:
            True if status updated successfully
        """
        try:
            updated = await self.camera_repository.update_status(
                camera_id, status, status_message
            )

            if updated:
                logger.info(
                    "Camera status updated",
                    camera_id=camera_id,
                    status=status.value,
                    message=status_message
                )

            return updated

        except Exception as e:
            logger.error(
                "Failed to update camera status",
                camera_id=camera_id,
                status=status.value,
                error=str(e)
            )
            raise DatabaseError("Status update failed", cause=e) from e

    async def update_last_seen(self, camera_id: str) -> bool:
        """Update camera last seen timestamp.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            True if updated successfully
        """
        return await self.camera_repository.update_last_seen(camera_id)

    async def get_camera_statistics(self) -> dict[str, Any]:
        """Get camera statistics and status counts.
        
        Returns:
            Dictionary with camera statistics
        """
        try:
            total_count = await self.camera_repository.count()
            status_counts = await self.camera_repository.get_status_counts()

            # Calculate health metrics
            online_count = status_counts.get('online', 0)
            streaming_count = status_counts.get('streaming', 0)
            error_count = status_counts.get('error', 0)

            health_percentage = (
                (online_count + streaming_count) / total_count * 100
                if total_count > 0 else 0
            )

            return {
                'total_cameras': total_count,
                'status_counts': status_counts,
                'health_percentage': round(health_percentage, 2),
                'error_percentage': round(error_count / total_count * 100 if total_count > 0 else 0, 2),
                'generated_at': datetime.now()
            }

        except Exception as e:
            logger.error("Failed to get camera statistics", error=str(e))
            raise DatabaseError("Statistics retrieval failed", cause=e) from e

    async def search_cameras(
        self,
        query: str,
        limit: int = 50,
        offset: int = 0
    ) -> list[Camera]:
        """Search cameras by name, location, or description.
        
        Args:
            query: Search query
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of matching cameras
        """
        return await self.camera_repository.search_cameras(query, limit, offset)

    async def get_cameras_by_location(self, location: str) -> list[Camera]:
        """Get cameras by location.
        
        Args:
            location: Location to search for
            
        Returns:
            List of cameras at the location
        """
        return await self.camera_repository.get_by_location(location)

    async def get_maintenance_cameras(self) -> list[Camera]:
        """Get cameras that need maintenance.
        
        Returns:
            List of cameras needing maintenance
        """
        return await self.camera_repository.get_cameras_needing_maintenance()

    async def batch_update_status(
        self,
        camera_ids: list[str],
        status: CameraStatus,
        status_message: str | None = None
    ) -> dict[str, bool]:
        """Batch update status for multiple cameras.
        
        Args:
            camera_ids: List of camera identifiers
            status: New status
            status_message: Optional status message
            
        Returns:
            Dictionary mapping camera IDs to success status
        """
        results = {}

        for camera_id in camera_ids:
            try:
                result = await self.update_camera_status(
                    camera_id, status, status_message
                )
                results[camera_id] = result
            except Exception as e:
                logger.error(
                    "Failed to update camera status in batch",
                    camera_id=camera_id,
                    error=str(e)
                )
                results[camera_id] = False

        logger.info(
            "Batch status update completed",
            total_cameras=len(camera_ids),
            successful_updates=sum(results.values()),
            status=status.value
        )

        return results
