"""Frame repository for frame management data access.

Provides specialized methods for frame operations, metadata management,
and performance monitoring with optimized queries.
"""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from ..models.frame_metadata import FrameMetadata, ProcessingStatus
from .base_repository import BaseRepository

logger = get_logger(__name__)


class FrameRepository(BaseRepository[FrameMetadata]):
    """Repository for frame data access operations.
    
    Specialized methods for frame management, metadata operations,
    and performance monitoring with optimized queries.
    """

    def __init__(self, session_factory: sessionmaker[AsyncSession]):
        super().__init__(session_factory, FrameMetadata)

    async def get_by_camera_id(
        self,
        camera_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> list[FrameMetadata]:
        """Get frames by camera ID.
        
        Args:
            camera_id: Camera identifier
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of frame metadata from the specified camera
            
        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(FrameMetadata).where(
                    FrameMetadata.camera_id == camera_id
                ).order_by(
                    FrameMetadata.timestamp.desc()
                ).limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get frames by camera ID",
                    camera_id=camera_id,
                    error=str(e)
                )
                raise DatabaseError("Frame retrieval failed", cause=e) from e

    async def get_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        camera_id: str | None = None,
        limit: int = 1000,
        offset: int = 0
    ) -> list[FrameMetadata]:
        """Get frames within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            camera_id: Optional camera ID to filter by
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of frames within the time range
            
        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(FrameMetadata).where(
                    and_(
                        FrameMetadata.timestamp >= start_time,
                        FrameMetadata.timestamp <= end_time
                    )
                )

                if camera_id:
                    query = query.where(FrameMetadata.camera_id == camera_id)

                query = query.order_by(
                    FrameMetadata.timestamp.desc()
                ).limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get frames by time range",
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    camera_id=camera_id,
                    error=str(e)
                )
                raise DatabaseError("Frame retrieval failed", cause=e) from e

    async def get_recent_frames(
        self,
        camera_id: str,
        minutes: int = 5,
        limit: int = 100
    ) -> list[FrameMetadata]:
        """Get recent frames from a camera.
        
        Args:
            camera_id: Camera identifier
            minutes: Number of minutes to look back
            limit: Maximum number of results
            
        Returns:
            List of recent frames
            
        Raises:
            DatabaseError: If query fails
        """
        start_time = datetime.now() - timedelta(minutes=minutes)
        return await self.get_by_time_range(
            start_time=start_time,
            end_time=datetime.now(),
            camera_id=camera_id,
            limit=limit
        )

    async def get_by_status(
        self,
        status: ProcessingStatus,
        limit: int = 100,
        offset: int = 0
    ) -> list[FrameMetadata]:
        """Get frames by processing status.
        
        Args:
            status: Frame processing status
            limit: Maximum number of results
            offset: Number of results to skip
            
        Returns:
            List of frames with the specified status
            
        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(FrameMetadata).where(
                    FrameMetadata.status == status
                ).order_by(
                    FrameMetadata.created_at.desc()
                ).limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get frames by status",
                    status=status.value,
                    error=str(e)
                )
                raise DatabaseError("Frame retrieval failed", cause=e) from e

    async def update_status(
        self,
        frame_id: str,
        status: ProcessingStatus,
        processing_time_ms: float | None = None,
        error_message: str | None = None
    ) -> bool:
        """Update frame processing status.
        
        Args:
            frame_id: Frame identifier
            status: New processing status
            processing_time_ms: Optional processing time in milliseconds
            error_message: Optional error message
            
        Returns:
            True if status updated successfully
            
        Raises:
            DatabaseError: If update fails
        """
        async with self._get_session() as session:
            try:
                frame = await session.get(Frame, frame_id)
                if frame is None:
                    return False

                frame.status = status
                frame.updated_at = datetime.now()

                if processing_time_ms is not None:
                    frame.processing_time_ms = processing_time_ms

                if error_message:
                    frame.error_message = error_message

                await session.commit()
                return True

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to update frame status",
                    frame_id=frame_id,
                    status=status.value,
                    error=str(e)
                )
                raise DatabaseError("Frame status update failed", cause=e) from e

    async def get_processing_statistics(
        self,
        camera_id: str | None = None,
        hours: int = 24
    ) -> dict[str, Any]:
        """Get frame processing statistics.
        
        Args:
            camera_id: Optional camera ID to filter by
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with processing statistics
            
        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                start_time = datetime.now() - timedelta(hours=hours)

                query = select(
                    FrameMetadata.status,
                    func.count(Frame.id).label('count'),
                    func.avg(Frame.processing_time_ms).label('avg_processing_time'),
                    func.min(Frame.processing_time_ms).label('min_processing_time'),
                    func.max(Frame.processing_time_ms).label('max_processing_time')
                ).where(FrameMetadata.created_at >= start_time)

                if camera_id:
                    query = query.where(FrameMetadata.camera_id == camera_id)

                query = query.group_by(FrameMetadata.status)

                result = await session.execute(query)

                stats = {}
                total_frames = 0
                total_processing_time = 0

                for row in result.all():
                    stats[row.status.value] = {
                        'count': row.count,
                        'avg_processing_time_ms': float(row.avg_processing_time or 0),
                        'min_processing_time_ms': float(row.min_processing_time or 0),
                        'max_processing_time_ms': float(row.max_processing_time or 0),
                    }
                    total_frames += row.count
                    if row.avg_processing_time:
                        total_processing_time += row.avg_processing_time * row.count

                stats['summary'] = {
                    'total_frames': total_frames,
                    'avg_processing_time_ms': total_processing_time / total_frames if total_frames > 0 else 0,
                    'time_range_hours': hours,
                    'camera_id': camera_id
                }

                return stats

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get processing statistics",
                    camera_id=camera_id,
                    hours=hours,
                    error=str(e)
                )
                raise DatabaseError("Statistics retrieval failed", cause=e) from e

    async def cleanup_old_frames(
        self,
        older_than_days: int = 30,
        batch_size: int = 1000
    ) -> int:
        """Clean up old frames to manage storage.
        
        Args:
            older_than_days: Delete frames older than this many days
            batch_size: Number of frames to delete per batch
            
        Returns:
            Number of frames deleted
            
        Raises:
            DatabaseError: If cleanup fails
        """
        async with self._get_session() as session:
            try:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)

                # Get frame IDs to delete
                query = select(Frame.id).where(
                    FrameMetadata.created_at < cutoff_date
                ).limit(batch_size)

                result = await session.execute(query)
                frame_ids = [row.id for row in result.all()]

                if not frame_ids:
                    return 0

                # Delete frames in batch
                from sqlalchemy import delete
                delete_query = delete(FrameMetadata).where(Frame.id.in_(frame_ids))
                result = await session.execute(delete_query)
                await session.commit()

                deleted_count = result.rowcount

                logger.info(
                    "Cleaned up old frames",
                    deleted_count=deleted_count,
                    cutoff_date=cutoff_date.isoformat()
                )

                return deleted_count

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to cleanup old frames",
                    older_than_days=older_than_days,
                    error=str(e)
                )
                raise DatabaseError("Frame cleanup failed", cause=e) from e
