"""Detection repository for detection result data access.

Provides specialized methods for detection operations, analytics,
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
from ..models.detection_result import DetectionResult
from .base_repository import BaseRepository

logger = get_logger(__name__)


class DetectionRepository(BaseRepository[DetectionResult]):
    """Repository for detection result data access operations.

    Specialized methods for detection analytics, confidence analysis,
    and performance monitoring with optimized queries.
    """

    def __init__(self, session_factory: sessionmaker[AsyncSession]):
        super().__init__(session_factory, DetectionResult)

    async def get_by_frame_id(self, frame_id: str) -> list[DetectionResult]:
        """Get detection results by frame ID.

        Args:
            frame_id: Frame identifier

        Returns:
            List of detection results for the frame

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(DetectionResult).where(
                    DetectionResult.frame_id == frame_id
                ).order_by(DetectionResult.confidence.desc())

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get detection results by frame ID",
                    frame_id=frame_id,
                    error=str(e)
                )
                raise DatabaseError("Detection retrieval failed", cause=e) from e

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

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(DetectionResult).where(
                    DetectionResult.camera_id == camera_id
                ).order_by(
                    DetectionResult.timestamp.desc()
                ).limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get detection results by camera ID",
                    camera_id=camera_id,
                    error=str(e)
                )
                raise DatabaseError("Detection retrieval failed", cause=e) from e

    async def get_by_class(
        self,
        class_name: str,
        camera_id: str | None = None,
        limit: int = 1000,
        offset: int = 0
    ) -> list[DetectionResult]:
        """Get detection results by object class.

        Args:
            class_name: Object class name (e.g., 'car', 'truck', 'person')
            camera_id: Optional camera ID to filter by
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of detection results for the specified class

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(DetectionResult).where(
                    DetectionResult.class_name == class_name
                )

                if camera_id:
                    query = query.where(DetectionResult.camera_id == camera_id)

                query = query.order_by(
                    DetectionResult.confidence.desc()
                ).limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get detection results by class",
                    class_name=class_name,
                    camera_id=camera_id,
                    error=str(e)
                )
                raise DatabaseError("Detection retrieval failed", cause=e) from e

    async def get_high_confidence_detections(
        self,
        min_confidence: float = 0.8,
        camera_id: str | None = None,
        hours: int = 24,
        limit: int = 1000
    ) -> list[DetectionResult]:
        """Get high-confidence detection results.

        Args:
            min_confidence: Minimum confidence threshold
            camera_id: Optional camera ID to filter by
            hours: Number of hours to look back
            limit: Maximum number of results

        Returns:
            List of high-confidence detection results

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                start_time = datetime.now() - timedelta(hours=hours)

                query = select(DetectionResult).where(
                    and_(
                        DetectionResult.confidence >= min_confidence,
                        DetectionResult.timestamp >= start_time
                    )
                )

                if camera_id:
                    query = query.where(DetectionResult.camera_id == camera_id)

                query = query.order_by(
                    DetectionResult.confidence.desc()
                ).limit(limit)

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get high confidence detections",
                    min_confidence=min_confidence,
                    camera_id=camera_id,
                    hours=hours,
                    error=str(e)
                )
                raise DatabaseError("Detection retrieval failed", cause=e) from e

    async def get_detection_counts_by_class(
        self,
        camera_id: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None
    ) -> dict[str, int]:
        """Get detection counts by object class.

        Args:
            camera_id: Optional camera ID to filter by
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Dictionary with class names and counts

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(
                    DetectionResult.class_name,
                    func.count(DetectionResult.id).label('count')
                )

                conditions = []
                if camera_id:
                    conditions.append(DetectionResult.camera_id == camera_id)
                if start_time:
                    conditions.append(DetectionResult.timestamp >= start_time)
                if end_time:
                    conditions.append(DetectionResult.timestamp <= end_time)

                if conditions:
                    query = query.where(and_(*conditions))

                query = query.group_by(DetectionResult.class_name).order_by(
                    func.count(DetectionResult.id).desc()
                )

                result = await session.execute(query)
                return {row.class_name: row.count for row in result.all()}

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get detection counts by class",
                    camera_id=camera_id,
                    error=str(e)
                )
                raise DatabaseError("Detection counting failed", cause=e) from e

    async def get_confidence_statistics(
        self,
        class_name: str | None = None,
        camera_id: str | None = None,
        hours: int = 24
    ) -> dict[str, float]:
        """Get confidence statistics for detections.

        Args:
            class_name: Optional class name to filter by
            camera_id: Optional camera ID to filter by
            hours: Number of hours to analyze

        Returns:
            Dictionary with confidence statistics

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                start_time = datetime.now() - timedelta(hours=hours)

                query = select(
                    func.count(DetectionResult.id).label('count'),
                    func.avg(DetectionResult.confidence).label('avg_confidence'),
                    func.min(DetectionResult.confidence).label('min_confidence'),
                    func.max(DetectionResult.confidence).label('max_confidence'),
                    func.stddev(DetectionResult.confidence).label('stddev_confidence')
                ).where(DetectionResult.timestamp >= start_time)

                if class_name:
                    query = query.where(DetectionResult.class_name == class_name)
                if camera_id:
                    query = query.where(DetectionResult.camera_id == camera_id)

                result = await session.execute(query)
                row = result.first()

                if row and row.count > 0:
                    return {
                        'count': row.count,
                        'avg_confidence': float(row.avg_confidence or 0),
                        'min_confidence': float(row.min_confidence or 0),
                        'max_confidence': float(row.max_confidence or 0),
                        'stddev_confidence': float(row.stddev_confidence or 0),
                        'time_range_hours': hours,
                        'class_name': class_name,
                        'camera_id': camera_id,
                    }
                else:
                    return {
                        'count': 0,
                        'avg_confidence': 0.0,
                        'min_confidence': 0.0,
                        'max_confidence': 0.0,
                        'stddev_confidence': 0.0,
                        'time_range_hours': hours,
                        'class_name': class_name,
                        'camera_id': camera_id,
                    }

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get confidence statistics",
                    class_name=class_name,
                    camera_id=camera_id,
                    hours=hours,
                    error=str(e)
                )
                raise DatabaseError("Statistics retrieval failed", cause=e) from e

    async def get_hourly_detection_counts(
        self,
        camera_id: str | None = None,
        days: int = 7
    ) -> list[dict[str, Any]]:
        """Get hourly detection counts for time-series analysis.

        Args:
            camera_id: Optional camera ID to filter by
            days: Number of days to analyze

        Returns:
            List of dictionaries with hourly counts

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                start_time = datetime.now() - timedelta(days=days)

                # PostgreSQL-specific date truncation
                query = select(
                    func.date_trunc('hour', DetectionResult.timestamp).label('hour'),
                    func.count(DetectionResult.id).label('count')
                ).where(DetectionResult.timestamp >= start_time)

                if camera_id:
                    query = query.where(DetectionResult.camera_id == camera_id)

                query = query.group_by('hour').order_by('hour')

                result = await session.execute(query)

                return [
                    {
                        'hour': row.hour,
                        'count': row.count,
                    }
                    for row in result.all()
                ]

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get hourly detection counts",
                    camera_id=camera_id,
                    days=days,
                    error=str(e)
                )
                raise DatabaseError("Hourly counts retrieval failed", cause=e) from e

    async def cleanup_old_detections(
        self,
        older_than_days: int = 90,
        batch_size: int = 1000
    ) -> int:
        """Clean up old detection results to manage storage.

        Args:
            older_than_days: Delete detections older than this many days
            batch_size: Number of detections to delete per batch

        Returns:
            Number of detections deleted

        Raises:
            DatabaseError: If cleanup fails
        """
        async with self._get_session() as session:
            try:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)

                # Get detection IDs to delete
                query = select(DetectionResult.id).where(
                    DetectionResult.timestamp < cutoff_date
                ).limit(batch_size)

                result = await session.execute(query)
                detection_ids = [row.id for row in result.all()]

                if not detection_ids:
                    return 0

                # Delete detections in batch
                from sqlalchemy import delete
                delete_query = delete(DetectionResult).where(
                    DetectionResult.id.in_(detection_ids)
                )
                result = await session.execute(delete_query)
                await session.commit()

                deleted_count = result.rowcount

                logger.info(
                    "Cleaned up old detections",
                    deleted_count=deleted_count,
                    cutoff_date=cutoff_date.isoformat()
                )

                return deleted_count

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to cleanup old detections",
                    older_than_days=older_than_days,
                    error=str(e)
                )
                raise DatabaseError("Detection cleanup failed", cause=e) from e
