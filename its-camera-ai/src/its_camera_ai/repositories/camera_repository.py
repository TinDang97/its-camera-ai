"""Camera repository for camera management data access.

Provides specialized methods for camera operations, status management,
and performance monitoring with optimized queries.
"""

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from ..models.camera import Camera, CameraStatus, CameraType
from .base_repository import BaseRepository

logger = get_logger(__name__)


class CameraRepository(BaseRepository[Camera]):
    """Repository for camera data access operations.

    Specialized methods for camera management, status tracking,
    and performance monitoring with optimized queries.
    """

    def __init__(self, session_factory: sessionmaker[AsyncSession]):
        super().__init__(session_factory, Camera)
        self._session_factory = session_factory

    def _get_session(self) -> AsyncSession:
        """Get async session.

        Returns:
            Async session
        """
        return self._session_factory()

    async def get_by_name(self, name: str) -> Camera | None:
        """Get camera by name.

        Args:
            name: Camera name to search for

        Returns:
            Camera instance or None if not found

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(Camera).where(Camera.name == name)
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get camera by name",
                    name=name,
                    error=str(e)
                )
                raise DatabaseError("Camera retrieval failed", cause=e) from e

    async def get_by_location(self, location: str) -> list[Camera]:
        """Get cameras by location.

        Args:
            location: Location to search for

        Returns:
            List of cameras at the location

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(Camera).where(Camera.location.ilike(f"%{location}%"))
                result = await session.execute(query)
                return list(result.scalars().all())
            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get cameras by location",
                    location=location,
                    error=str(e)
                )
                raise DatabaseError("Camera retrieval failed", cause=e) from e

    async def get_by_status(
        self,
        status: CameraStatus,
        limit: int = 100,
        offset: int = 0
    ) -> list[Camera]:
        """Get cameras by status.

        Args:
            status: Camera status to filter by
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of cameras with the specified status

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(Camera).where(
                    Camera.status == status
                ).limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get cameras by status",
                    status=status.value,
                    error=str(e)
                )
                raise DatabaseError("Camera retrieval failed", cause=e) from e

    async def get_by_type(
        self,
        camera_type: CameraType,
        limit: int = 100,
        offset: int = 0
    ) -> list[Camera]:
        """Get cameras by type.

        Args:
            camera_type: Camera type to filter by
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of cameras of the specified type

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(Camera).where(
                    Camera.camera_type == camera_type
                ).limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get cameras by type",
                    camera_type=camera_type.value,
                    error=str(e)
                )
                raise DatabaseError("Camera retrieval failed", cause=e) from e

    async def get_online_cameras(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> list[Camera]:
        """Get online cameras.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of online cameras

        Raises:
            DatabaseError: If query fails
        """
        return await self.get_by_status(CameraStatus.ONLINE, limit, offset)

    async def get_streaming_cameras(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> list[Camera]:
        """Get cameras currently streaming.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of streaming cameras

        Raises:
            DatabaseError: If query fails
        """
        return await self.get_by_status(CameraStatus.STREAMING, limit, offset)

    async def update_status(
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

        Raises:
            DatabaseError: If update fails
        """
        async with self._get_session() as session:
            try:
                camera = await session.get(Camera, camera_id)
                if camera is None:
                    return False

                camera.status = status
                camera.status_updated_at = datetime.now()

                if status_message:
                    camera.status_message = status_message

                await session.commit()
                return True

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to update camera status",
                    camera_id=camera_id,
                    status=status.value,
                    error=str(e)
                )
                raise DatabaseError("Camera status update failed", cause=e) from e

    async def update_last_seen(
        self,
        camera_id: str,
        last_seen: datetime | None = None
    ) -> bool:
        """Update camera last seen timestamp.

        Args:
            camera_id: Camera identifier
            last_seen: Last seen timestamp (defaults to current time)

        Returns:
            True if updated successfully

        Raises:
            DatabaseError: If update fails
        """
        async with self._get_session() as session:
            try:
                camera = await session.get(Camera, camera_id)
                if camera is None:
                    return False

                camera.last_seen = last_seen or datetime.now()
                await session.commit()
                return True

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to update camera last seen",
                    camera_id=camera_id,
                    error=str(e)
                )
                raise DatabaseError("Camera last seen update failed", cause=e) from e

    async def get_status_counts(self) -> dict[str, int]:
        """Get camera count by status.

        Returns:
            Dictionary with status counts

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(
                    Camera.status,
                    func.count(Camera.id).label('count')
                ).group_by(Camera.status)

                result = await session.execute(query)
                return {row.status.value: row.count for row in result.all()}

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get status counts",
                    error=str(e)
                )
                raise DatabaseError("Status counts retrieval failed", cause=e) from e

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

        Raises:
            DatabaseError: If search fails
        """
        async with self._get_session() as session:
            try:
                search_term = f"%{query}%"
                sql_query = select(Camera).where(
                    Camera.name.ilike(search_term) |
                    Camera.location.ilike(search_term) |
                    Camera.description.ilike(search_term)
                ).limit(limit).offset(offset)

                result = await session.execute(sql_query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to search cameras",
                    query=query,
                    error=str(e)
                )
                raise DatabaseError("Camera search failed", cause=e) from e

    async def get_cameras_needing_maintenance(self) -> list[Camera]:
        """Get cameras that need maintenance.

        Returns:
            List of cameras needing maintenance

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(Camera).where(
                    Camera.status == CameraStatus.MAINTENANCE
                )
                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get cameras needing maintenance",
                    error=str(e)
                )
                raise DatabaseError(
                    "Maintenance cameras retrieval failed",
                    cause=e
                ) from e
