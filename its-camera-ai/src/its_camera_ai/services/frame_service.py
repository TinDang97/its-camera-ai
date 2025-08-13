"""Async service for frame metadata and detection results.

Optimized for high-throughput operations handling 3000+ inserts/second
with batch processing, efficient queries, and real-time updates.
"""

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..api.schemas.database import (
        BatchDetectionResultCreateSchema,
        BatchFrameMetadataCreateSchema,
        BatchOperationResultSchema,
        FrameMetadataCreateSchema,
        FrameMetadataUpdateSchema,
    )
from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from ..models import (
    DetectionResult,
    FrameMetadata,
    ProcessingStatus,
)
from ..repositories.detection_repository import DetectionRepository
from ..repositories.frame_repository import FrameRepository

logger = get_logger(__name__)


class FrameService:
    """High-throughput async service for frame metadata operations.

    Designed for processing 30+ FPS per camera across 100+ cameras
    with optimized batch operations and minimal latency.
    """

    def __init__(self, frame_repository: FrameRepository):
        self.frame_repository = frame_repository

    async def get_by_id(self, frame_id: str) -> FrameMetadata | None:
        """Get frame metadata by ID.
        
        Args:
            frame_id: Frame identifier
            
        Returns:
            Frame metadata if found, None otherwise
        """
        return await self.frame_repository.get_by_id(frame_id)

    async def create(self, **kwargs: Any) -> FrameMetadata:
        """Create frame metadata.
        
        Args:
            **kwargs: Frame metadata fields
            
        Returns:
            Created frame metadata
        """
        return await self.frame_repository.create(**kwargs)

    async def update(self, frame_id: str, **kwargs: Any) -> FrameMetadata:
        """Update frame metadata.
        
        Args:
            frame_id: Frame identifier
            **kwargs: Fields to update
            
        Returns:
            Updated frame metadata
        """
        return await self.frame_repository.update(frame_id, **kwargs)

    async def delete(self, frame_id: str) -> bool:
        """Delete frame metadata.
        
        Args:
            frame_id: Frame identifier
            
        Returns:
            True if deleted successfully
        """
        return await self.frame_repository.delete(frame_id)

    async def bulk_delete(self, frame_ids: list[str]) -> int:
        """Bulk delete frame metadata.
        
        Args:
            frame_ids: List of frame identifiers
            
        Returns:
            Number of deleted frames
        """
        deleted_count = 0
        for frame_id in frame_ids:
            if await self.frame_repository.delete(frame_id):
                deleted_count += 1
        return deleted_count

    async def create_frame_metadata(
        self, frame_data: "FrameMetadataCreateSchema"
    ) -> FrameMetadata:
        """Create single frame metadata record.

        Args:
            frame_data: Frame metadata creation data

        Returns:
            Created frame metadata instance

        Raises:
            DatabaseError: If creation fails
        """
        # Import schema at runtime to avoid circular import

        try:
            frame_dict = frame_data.model_dump()

            # Set timestamp if not provided
            if not frame_dict.get("timestamp"):
                frame_dict["timestamp"] = datetime.now(UTC)

            # Initialize processing fields
            frame_dict.update({
                "status": ProcessingStatus.PENDING,
                "has_detections": False,
                "detection_count": 0,
                "vehicle_count": 0,
                "is_stored": False,
                "retry_count": 0,
            })

            frame = await self.frame_repository.create(**frame_dict)

            logger.debug(
                "Frame metadata created",
                frame_id=frame.id,
                camera_id=frame.camera_id,
                frame_number=frame.frame_number,
            )

            return frame

        except Exception as e:
            logger.error(
                "Frame metadata creation failed",
                camera_id=frame_data.camera_id,
                frame_number=frame_data.frame_number,
                error=str(e),
            )
            raise DatabaseError(f"Failed to create frame metadata: {str(e)}") from e

    async def batch_create_frame_metadata(
        self, batch_data: "BatchFrameMetadataCreateSchema"
    ) -> "BatchOperationResultSchema":
        """Batch create frame metadata records for high throughput.

        Args:
            batch_data: Batch of frame metadata to create

        Returns:
            Batch operation result with performance metrics

        Raises:
            DatabaseError: If batch creation fails
        """
        # Import schema at runtime to avoid circular import
        from ..api.schemas.database import BatchOperationResultSchema

        start_time = datetime.now(UTC)
        successful_items = 0
        errors = []

        try:
            frame_instances = []

            for idx, frame_data in enumerate(batch_data.frames):
                try:
                    frame_dict = frame_data.model_dump()

                    # Set timestamp if not provided
                    if not frame_dict.get("timestamp"):
                        frame_dict["timestamp"] = datetime.now(UTC)

                    # Initialize processing fields
                    frame_dict.update({
                        "status": ProcessingStatus.PENDING,
                        "has_detections": False,
                        "detection_count": 0,
                        "vehicle_count": 0,
                        "is_stored": False,
                        "retry_count": 0,
                    })

                    frame = await self.frame_repository.create(**frame_dict)
                    frame_instances.append(frame)

                except Exception as e:
                    errors.append({
                        "index": idx,
                        "error": str(e),
                        "data": frame_data.model_dump() if frame_data else None,
                    })

            # All frames already committed individually by repository
            successful_items = len(frame_instances)

            # Calculate performance metrics
            processing_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            items_per_second = successful_items / max(processing_time_ms / 1000, 0.001)

            logger.info(
                "Batch frame metadata creation completed",
                total_items=len(batch_data.frames),
                successful_items=successful_items,
                failed_items=len(errors),
                processing_time_ms=processing_time_ms,
                items_per_second=items_per_second,
            )

            return BatchOperationResultSchema(
                total_items=len(batch_data.frames),
                successful_items=successful_items,
                failed_items=len(errors),
                errors=errors,
                processing_time_ms=processing_time_ms,
                items_per_second=items_per_second,
            )

        except Exception as e:
            logger.error(
                "Batch frame metadata creation failed",
                total_items=len(batch_data.frames),
                error=str(e),
            )
            raise DatabaseError(f"Batch frame creation failed: {str(e)}") from e

    async def update_frame_processing(
        self, frame_id: str, update_data: "FrameMetadataUpdateSchema"
    ) -> FrameMetadata | None:
        """Update frame metadata after processing completion.

        Args:
            frame_id: Frame metadata ID
            update_data: Processing update data

        Returns:
            Updated frame metadata if found, None otherwise
        """
        try:
            frame = await self.frame_repository.get_by_id(frame_id)
            if not frame:
                return None

            # Update fields from schema
            update_dict = update_data.model_dump(exclude_none=True)

            # Update processing timestamp
            if update_data.status == ProcessingStatus.COMPLETED:
                update_dict["processing_completed_at"] = datetime.now(UTC)
            elif update_data.status == ProcessingStatus.PROCESSING:
                update_dict["processing_started_at"] = datetime.now(UTC)

            frame = await self.frame_repository.update(frame_id, **update_dict)

            logger.debug(
                "Frame processing updated",
                frame_id=frame_id,
                status=update_data.status,
                processing_time=update_data.processing_time_ms,
            )

            return frame

        except Exception as e:
            logger.error(
                "Frame processing update failed",
                frame_id=frame_id,
                error=str(e),
            )
            raise DatabaseError(f"Failed to update frame processing: {str(e)}") from e

    async def get_pending_frames(
        self, camera_id: str | None = None, limit: int = 100
    ) -> list[FrameMetadata]:
        """Get pending frames for processing.

        Args:
            camera_id: Optional camera ID filter
            limit: Maximum number of frames to return

        Returns:
            List of pending frame metadata
        """
        try:
            # Use repository method for status-based queries
            pending_frames = await self.frame_repository.get_by_status(
                ProcessingStatus.PENDING, limit=limit
            )

            # Filter by camera_id if provided
            if camera_id:
                pending_frames = [
                    frame for frame in pending_frames
                    if frame.camera_id == camera_id
                ]

            return pending_frames

        except Exception as e:
            logger.error(
                "Failed to get pending frames",
                camera_id=camera_id,
                error=str(e),
            )
            raise DatabaseError(f"Failed to get pending frames: {str(e)}") from e

    async def get_frames_for_camera(
        self,
        camera_id: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        status_filter: ProcessingStatus | None = None,
        has_detections: bool | None = None,
        limit: int = 100,
        include_detections: bool = False,
    ) -> list[FrameMetadata]:
        """Get frames for a specific camera with filtering.

        Args:
            camera_id: Camera identifier
            start_time: Optional start time filter
            end_time: Optional end time filter
            status_filter: Optional processing status filter
            has_detections: Optional detection filter
            limit: Maximum number of frames
            include_detections: Whether to load detection results

        Returns:
            List of frame metadata
        """
        try:
            # Use repository method for camera-based queries
            frames = await self.frame_repository.get_by_camera_id(
                camera_id=camera_id, limit=limit
            )

            # Apply additional filters
            filtered_frames = []
            for frame in frames:
                # Time range filter
                if start_time and frame.timestamp < start_time:
                    continue
                if end_time and frame.timestamp > end_time:
                    continue

                # Status filter
                if status_filter and frame.status != status_filter:
                    continue

                # Detection filter
                if has_detections is not None and frame.has_detections != has_detections:
                    continue

                filtered_frames.append(frame)

            return filtered_frames

        except Exception as e:
            logger.error(
                "Failed to get frames for camera",
                camera_id=camera_id,
                error=str(e),
            )
            raise DatabaseError(f"Failed to get camera frames: {str(e)}") from e

    async def get_processing_statistics(
        self,
        camera_id: str | None = None,
        time_window_hours: int = 24,
    ) -> dict[str, Any]:
        """Get frame processing statistics.

        Args:
            camera_id: Optional camera ID filter
            time_window_hours: Time window for statistics (hours)

        Returns:
            Processing statistics dictionary
        """
        try:
            # Use repository method for processing statistics
            return await self.frame_repository.get_processing_statistics(
                camera_id=camera_id, hours=time_window_hours
            )

        except Exception as e:
            logger.error(
                "Failed to get processing statistics",
                camera_id=camera_id,
                error=str(e),
            )
            raise DatabaseError(f"Failed to get processing statistics: {str(e)}") from e

    async def cleanup_old_frames(
        self, retention_days: int = 7, batch_size: int = 1000
    ) -> int:
        """Clean up old frame metadata beyond retention period.

        Args:
            retention_days: Number of days to retain frames
            batch_size: Number of frames to delete per batch

        Returns:
            Number of frames deleted
        """
        try:
            # Use repository method for cleanup
            total_deleted = await self.frame_repository.cleanup_old_frames(
                older_than_days=retention_days, batch_size=batch_size
            )

            logger.info(
                "Frame metadata cleanup completed",
                retention_days=retention_days,
                total_deleted=total_deleted,
            )

            return total_deleted

        except Exception as e:
            logger.error(
                "Frame metadata cleanup failed",
                retention_days=retention_days,
                error=str(e),
            )
            raise DatabaseError(f"Failed to cleanup frames: {str(e)}") from e

    async def get_real_time_feed(
        self,
        camera_ids: list[str] | None = None,
        limit: int = 50,
    ) -> list[FrameMetadata]:
        """Get real-time frame feed for SSE broadcasting.

        Args:
            camera_ids: Optional list of camera IDs to filter by
            limit: Maximum number of recent frames

        Returns:
            List of recent frame metadata with detections
        """
        try:
            # Get recent frames for each camera and filter
            all_recent_frames = []

            if camera_ids:
                # Get recent frames for specific cameras
                for camera_id in camera_ids:
                    recent_frames = await self.frame_repository.get_recent_frames(
                        camera_id=camera_id, minutes=5, limit=limit // len(camera_ids)
                    )
                    # Filter for completed frames with detections
                    filtered_frames = [
                        frame for frame in recent_frames
                        if frame.status == ProcessingStatus.COMPLETED and frame.has_detections
                    ]
                    all_recent_frames.extend(filtered_frames)
            else:
                # Get recent frames from all cameras (need to use time range query)
                start_time = datetime.now(UTC) - timedelta(minutes=5)
                end_time = datetime.now(UTC)
                recent_frames = await self.frame_repository.get_by_time_range(
                    start_time=start_time, end_time=end_time, limit=limit
                )
                # Filter for completed frames with detections
                all_recent_frames = [
                    frame for frame in recent_frames
                    if frame.status == ProcessingStatus.COMPLETED and frame.has_detections
                ]

            # Sort by timestamp descending and limit
            all_recent_frames.sort(key=lambda x: x.timestamp, reverse=True)
            return all_recent_frames[:limit]

        except Exception as e:
            logger.error(
                "Failed to get real-time feed",
                camera_ids=camera_ids,
                error=str(e),
            )
            raise DatabaseError(f"Failed to get real-time feed: {str(e)}") from e

    async def get_frames_by_status(
        self,
        status: ProcessingStatus,
        limit: int = 100
    ) -> list[FrameMetadata]:
        """Get frames by processing status.
        
        Args:
            status: Processing status to filter by
            limit: Maximum number of frames
            
        Returns:
            List of frames with the specified status
        """
        return await self.frame_repository.get_by_status(status=status, limit=limit)

    async def get_frames_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        camera_id: str | None = None,
        limit: int = 1000
    ) -> list[FrameMetadata]:
        """Get frames within a time range.
        
        Args:
            start_time: Start time
            end_time: End time
            camera_id: Optional camera ID filter
            limit: Maximum number of frames
            
        Returns:
            List of frames within the time range
        """
        return await self.frame_repository.get_by_time_range(
            start_time=start_time,
            end_time=end_time,
            camera_id=camera_id,
            limit=limit
        )


class DetectionService:
    """High-throughput async service for detection result operations."""

    def __init__(self, detection_repository: DetectionRepository):
        self.detection_repository = detection_repository

    async def get_by_id(self, detection_id: str) -> DetectionResult | None:
        """Get detection result by ID.
        
        Args:
            detection_id: Detection identifier
            
        Returns:
            Detection result if found, None otherwise
        """
        return await self.detection_repository.get_by_id(detection_id)

    async def create(self, **kwargs: Any) -> DetectionResult:
        """Create detection result.
        
        Args:
            **kwargs: Detection result fields
            
        Returns:
            Created detection result
        """
        return await self.detection_repository.create(**kwargs)

    async def update(self, detection_id: str, **kwargs: Any) -> DetectionResult:
        """Update detection result.
        
        Args:
            detection_id: Detection identifier
            **kwargs: Fields to update
            
        Returns:
            Updated detection result
        """
        return await self.detection_repository.update(detection_id, **kwargs)

    async def delete(self, detection_id: str) -> bool:
        """Delete detection result.
        
        Args:
            detection_id: Detection identifier
            
        Returns:
            True if deleted successfully
        """
        return await self.detection_repository.delete(detection_id)

    async def batch_create_detections(
        self, batch_data: "BatchDetectionResultCreateSchema"
    ) -> "BatchOperationResultSchema":
        """Batch create detection results for high throughput.

        Args:
            batch_data: Batch of detection results to create

        Returns:
            Batch operation result with performance metrics
        """
        # Import schema at runtime to avoid circular import
        from ..api.schemas.database import BatchOperationResultSchema

        start_time = datetime.now(UTC)
        successful_items = 0
        errors = []

        try:
            detection_instances = []

            for idx, detection_data in enumerate(batch_data.detections):
                try:
                    detection_dict = detection_data.model_dump()

                    # Calculate bounding box derived properties
                    bbox_x1 = detection_dict["bbox_x1"]
                    bbox_y1 = detection_dict["bbox_y1"]
                    bbox_x2 = detection_dict["bbox_x2"]
                    bbox_y2 = detection_dict["bbox_y2"]

                    detection_dict.update({
                        "bbox_width": bbox_x2 - bbox_x1,
                        "bbox_height": bbox_y2 - bbox_y1,
                        "bbox_area": (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1),
                        "detection_quality": 1.0,  # Default quality
                        "is_false_positive": False,
                        "is_verified": False,
                        "is_anomaly": False,
                    })

                    detection = await self.detection_repository.create(**detection_dict)
                    detection_instances.append(detection)

                except Exception as e:
                    errors.append({
                        "index": idx,
                        "error": str(e),
                        "data": detection_data.model_dump() if detection_data else None,
                    })

            # All detections already committed individually by repository
            successful_items = len(detection_instances)

            # Calculate performance metrics
            processing_time_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
            items_per_second = successful_items / max(processing_time_ms / 1000, 0.001)

            logger.info(
                "Batch detection creation completed",
                total_items=len(batch_data.detections),
                successful_items=successful_items,
                failed_items=len(errors),
                processing_time_ms=processing_time_ms,
                items_per_second=items_per_second,
            )

            return BatchOperationResultSchema(
                total_items=len(batch_data.detections),
                successful_items=successful_items,
                failed_items=len(errors),
                errors=errors,
                processing_time_ms=processing_time_ms,
                items_per_second=items_per_second,
            )

        except Exception as e:
            logger.error(
                "Batch detection creation failed",
                total_items=len(batch_data.detections),
                error=str(e),
            )
            raise DatabaseError(f"Batch detection creation failed: {str(e)}") from e

    async def get_by_frame_id(self, frame_id: str) -> list[DetectionResult]:
        """Get detection results by frame ID.
        
        Args:
            frame_id: Frame identifier
            
        Returns:
            List of detection results for the frame
        """
        return await self.detection_repository.get_by_frame_id(frame_id)

    async def get_by_camera_id(
        self,
        camera_id: str,
        limit: int = 1000,
        offset: int = 0
    ) -> list[DetectionResult]:
        """Get detection results by camera ID.
        
        Args:
            camera_id: Camera identifier
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of detection results from the camera
        """
        return await self.detection_repository.get_by_camera_id(
            camera_id=camera_id, limit=limit, offset=offset
        )

    async def get_detection_statistics(
        self,
        camera_id: str | None = None,
        class_name: str | None = None,
        hours: int = 24
    ) -> dict[str, Any]:
        """Get detection statistics.
        
        Args:
            camera_id: Optional camera ID filter
            class_name: Optional class name filter
            hours: Number of hours to analyze
            
        Returns:
            Detection statistics dictionary
        """
        return await self.detection_repository.get_confidence_statistics(
            camera_id=camera_id, class_name=class_name, hours=hours
        )

    async def cleanup_old_detections(
        self, older_than_days: int = 90, batch_size: int = 1000
    ) -> int:
        """Clean up old detection results.
        
        Args:
            older_than_days: Delete detections older than this many days
            batch_size: Number of detections to delete per batch
            
        Returns:
            Number of detections deleted
        """
        return await self.detection_repository.cleanup_old_detections(
            older_than_days=older_than_days, batch_size=batch_size
        )
