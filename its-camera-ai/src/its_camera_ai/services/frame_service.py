"""Async service for frame metadata and detection results.

Optimized for high-throughput operations handling 3000+ inserts/second
with batch processing, efficient queries, and real-time updates.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

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
from .base_service import BaseAsyncService

logger = get_logger(__name__)


class FrameService(BaseAsyncService[FrameMetadata]):
    """High-throughput async service for frame metadata operations.
    
    Designed for processing 30+ FPS per camera across 100+ cameras
    with optimized batch operations and minimal latency.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, FrameMetadata)

    async def create_frame_metadata(
        self, frame_data: FrameMetadataCreateSchema
    ) -> FrameMetadata:
        """Create single frame metadata record.
        
        Args:
            frame_data: Frame metadata creation data
            
        Returns:
            Created frame metadata instance
            
        Raises:
            DatabaseError: If creation fails
        """
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

            frame = FrameMetadata(**frame_dict)

            self.session.add(frame)
            await self.session.commit()
            await self.session.refresh(frame)

            logger.debug(
                "Frame metadata created",
                frame_id=frame.id,
                camera_id=frame.camera_id,
                frame_number=frame.frame_number,
            )

            return frame

        except Exception as e:
            await self.session.rollback()
            logger.error(
                "Frame metadata creation failed",
                camera_id=frame_data.camera_id,
                frame_number=frame_data.frame_number,
                error=str(e),
            )
            raise DatabaseError(f"Failed to create frame metadata: {str(e)}") from e

    async def batch_create_frame_metadata(
        self, batch_data: BatchFrameMetadataCreateSchema
    ) -> BatchOperationResultSchema:
        """Batch create frame metadata records for high throughput.
        
        Args:
            batch_data: Batch of frame metadata to create
            
        Returns:
            Batch operation result with performance metrics
            
        Raises:
            DatabaseError: If batch creation fails
        """
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

                    frame = FrameMetadata(**frame_dict)
                    frame_instances.append(frame)
                    self.session.add(frame)

                except Exception as e:
                    errors.append({
                        "index": idx,
                        "error": str(e),
                        "data": frame_data.model_dump() if frame_data else None,
                    })

            # Commit all valid frames
            if frame_instances:
                await self.session.commit()
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
            await self.session.rollback()
            logger.error(
                "Batch frame metadata creation failed",
                total_items=len(batch_data.frames),
                error=str(e),
            )
            raise DatabaseError(f"Batch frame creation failed: {str(e)}") from e

    async def update_frame_processing(
        self, frame_id: str, update_data: FrameMetadataUpdateSchema
    ) -> FrameMetadata | None:
        """Update frame metadata after processing completion.
        
        Args:
            frame_id: Frame metadata ID
            update_data: Processing update data
            
        Returns:
            Updated frame metadata if found, None otherwise
        """
        try:
            frame = await self.get_by_id(frame_id)
            if not frame:
                return None

            # Update fields from schema
            update_dict = update_data.model_dump(exclude_none=True)

            for key, value in update_dict.items():
                if hasattr(frame, key):
                    setattr(frame, key, value)

            # Update processing timestamp
            if update_data.status == ProcessingStatus.COMPLETED:
                frame.processing_completed_at = datetime.now(UTC)
            elif update_data.status == ProcessingStatus.PROCESSING:
                frame.processing_started_at = datetime.now(UTC)

            await self.session.commit()

            logger.debug(
                "Frame processing updated",
                frame_id=frame_id,
                status=update_data.status,
                processing_time=update_data.processing_time_ms,
            )

            return frame

        except Exception as e:
            await self.session.rollback()
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
            query = (
                select(FrameMetadata)
                .where(FrameMetadata.status == ProcessingStatus.PENDING)
                .order_by(FrameMetadata.timestamp)
                .limit(limit)
            )

            if camera_id:
                query = query.where(FrameMetadata.camera_id == camera_id)

            result = await self.session.execute(query)
            return list(result.scalars().all())

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
            query = (
                select(FrameMetadata)
                .where(FrameMetadata.camera_id == camera_id)
                .order_by(desc(FrameMetadata.timestamp))
                .limit(limit)
            )

            if include_detections:
                query = query.options(selectinload(FrameMetadata.detection_results))

            # Apply time range filter
            if start_time:
                query = query.where(FrameMetadata.timestamp >= start_time)
            if end_time:
                query = query.where(FrameMetadata.timestamp <= end_time)

            # Apply status filter
            if status_filter:
                query = query.where(FrameMetadata.status == status_filter)

            # Apply detection filter
            if has_detections is not None:
                query = query.where(FrameMetadata.has_detections == has_detections)

            result = await self.session.execute(query)
            return list(result.scalars().all())

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
            start_time = datetime.now(UTC) - timedelta(hours=time_window_hours)

            # Base query with time filter
            base_query = select(FrameMetadata).where(
                FrameMetadata.timestamp >= start_time
            )

            if camera_id:
                base_query = base_query.where(FrameMetadata.camera_id == camera_id)

            # Count by status
            status_query = (
                select(
                    FrameMetadata.status,
                    func.count(FrameMetadata.id).label('count')
                )
                .where(FrameMetadata.timestamp >= start_time)
                .group_by(FrameMetadata.status)
            )

            if camera_id:
                status_query = status_query.where(FrameMetadata.camera_id == camera_id)

            # Average processing time for completed frames
            avg_processing_query = (
                select(func.avg(FrameMetadata.processing_time_ms))
                .where(
                    and_(
                        FrameMetadata.timestamp >= start_time,
                        FrameMetadata.status == ProcessingStatus.COMPLETED,
                        FrameMetadata.processing_time_ms.is_not(None),
                    )
                )
            )

            if camera_id:
                avg_processing_query = avg_processing_query.where(
                    FrameMetadata.camera_id == camera_id
                )

            # Detection statistics
            detection_query = (
                select(
                    func.count(FrameMetadata.id).label('total_frames'),
                    func.sum(
                        func.cast(FrameMetadata.has_detections, text('int'))
                    ).label('frames_with_detections'),
                    func.sum(FrameMetadata.detection_count).label('total_detections'),
                    func.sum(FrameMetadata.vehicle_count).label('total_vehicles'),
                )
                .where(FrameMetadata.timestamp >= start_time)
            )

            if camera_id:
                detection_query = detection_query.where(
                    FrameMetadata.camera_id == camera_id
                )

            # Execute queries
            status_result = await self.session.execute(status_query)
            avg_processing_result = await self.session.execute(avg_processing_query)
            detection_result = await self.session.execute(detection_query)

            # Process results
            status_counts = {row.status: row.count for row in status_result}
            avg_processing_time = avg_processing_result.scalar()
            detection_stats = detection_result.first()

            # Calculate rates and percentages
            total_frames = sum(status_counts.values())
            completed_frames = status_counts.get(ProcessingStatus.COMPLETED.value, 0)
            failed_frames = status_counts.get(ProcessingStatus.FAILED.value, 0)

            success_rate = (completed_frames / max(total_frames, 1)) * 100
            error_rate = (failed_frames / max(total_frames, 1)) * 100

            throughput_fps = total_frames / max(time_window_hours * 3600, 1)

            return {
                "time_window_hours": time_window_hours,
                "camera_id": camera_id,
                "total_frames": total_frames,
                "status_breakdown": {
                    "pending": status_counts.get(ProcessingStatus.PENDING.value, 0),
                    "processing": status_counts.get(ProcessingStatus.PROCESSING.value, 0),
                    "completed": completed_frames,
                    "failed": failed_frames,
                    "skipped": status_counts.get(ProcessingStatus.SKIPPED.value, 0),
                },
                "success_rate_percent": round(success_rate, 2),
                "error_rate_percent": round(error_rate, 2),
                "avg_processing_time_ms": round(avg_processing_time or 0, 2),
                "throughput_fps": round(throughput_fps, 2),
                "detection_stats": {
                    "total_detections": detection_stats.total_detections or 0,
                    "total_vehicles": detection_stats.total_vehicles or 0,
                    "frames_with_detections": detection_stats.frames_with_detections or 0,
                    "detection_rate_percent": round(
                        ((detection_stats.frames_with_detections or 0) / max(total_frames, 1)) * 100,
                        2,
                    ),
                },
            }

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
            cutoff_time = datetime.now(UTC) - timedelta(days=retention_days)
            total_deleted = 0

            while True:
                # Find frames to delete in batches
                delete_query = (
                    select(FrameMetadata.id)
                    .where(
                        and_(
                            FrameMetadata.timestamp < cutoff_time,
                            FrameMetadata.is_stored == False,  # Only delete unstored frames
                        )
                    )
                    .limit(batch_size)
                )

                result = await self.session.execute(delete_query)
                frame_ids = [row.id for row in result]

                if not frame_ids:
                    break

                # Delete batch
                deleted_count = await self.bulk_delete(frame_ids)
                total_deleted += deleted_count

                logger.debug(
                    "Deleted frame metadata batch",
                    batch_size=deleted_count,
                    total_deleted=total_deleted,
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
            # Get recent frames with detections
            query = (
                select(FrameMetadata)
                .where(
                    and_(
                        FrameMetadata.status == ProcessingStatus.COMPLETED,
                        FrameMetadata.has_detections == True,  # noqa: E712
                        FrameMetadata.timestamp >= datetime.now(UTC) - timedelta(minutes=5),
                    )
                )
                .options(selectinload(FrameMetadata.detection_results))
                .order_by(desc(FrameMetadata.timestamp))
                .limit(limit)
            )

            if camera_ids:
                query = query.where(FrameMetadata.camera_id.in_(camera_ids))

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(
                "Failed to get real-time feed",
                camera_ids=camera_ids,
                error=str(e),
            )
            raise DatabaseError(f"Failed to get real-time feed: {str(e)}") from e


class DetectionService(BaseAsyncService[DetectionResult]):
    """High-throughput async service for detection result operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, DetectionResult)

    async def batch_create_detections(
        self, batch_data: BatchDetectionResultCreateSchema
    ) -> BatchOperationResultSchema:
        """Batch create detection results for high throughput.
        
        Args:
            batch_data: Batch of detection results to create
            
        Returns:
            Batch operation result with performance metrics
        """
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

                    detection = DetectionResult(**detection_dict)
                    detection_instances.append(detection)
                    self.session.add(detection)

                except Exception as e:
                    errors.append({
                        "index": idx,
                        "error": str(e),
                        "data": detection_data.model_dump() if detection_data else None,
                    })

            # Commit all valid detections
            if detection_instances:
                await self.session.commit()
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
            await self.session.rollback()
            logger.error(
                "Batch detection creation failed",
                total_items=len(batch_data.detections),
                error=str(e),
            )
            raise DatabaseError(f"Batch detection creation failed: {str(e)}") from e
