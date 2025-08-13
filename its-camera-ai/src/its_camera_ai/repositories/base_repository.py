"""Base repository class for common data access patterns.

Provides generic async CRUD functionality with error handling,
transaction management, and query optimization for all repository classes.
"""

from typing import Any, AsyncContextManager, TypeVar
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import DatabaseError, NotFoundError
from ..core.logging import get_logger
from ..models.base import BaseTableModel

T = TypeVar("T", bound=BaseTableModel)
logger = get_logger(__name__)


class BaseRepository[T: BaseTableModel]:
    """Base repository class for database operations.

    Provides common CRUD operations with proper error handling,
    transaction management, and session lifecycle management.
    """

    def __init__(
        self, session_factory: AsyncContextManager[AsyncSession], model: type[T]
    ):
        self.session_factory = session_factory
        self.model = model

    def _get_session(self) -> AsyncContextManager[AsyncSession]:
        """Get async session context manager.

        Returns:
            Async session context manager
        """
        return self.session_factory()

    async def create(self, **kwargs: Any) -> T:
        """Create a new model instance.

        Args:
            **kwargs: Model field values

        Returns:
            Created model instance

        Raises:
            DatabaseError: If creation fails
        """
        async with self.session_factory() as session:
            try:
                instance = self.model(**kwargs)
                session.add(instance)
                await session.commit()
                await session.refresh(instance)
                return instance
            except SQLAlchemyError as e:
                logger.error(
                    "Failed to create entity", model=self.model.__name__, error=str(e)
                )
                raise DatabaseError("Entity creation failed", cause=e) from e

    async def get_by_id(self, entity_id: UUID | str) -> T | None:
        """Get model instance by ID.

        Args:
            entity_id: Entity identifier

        Returns:
            Model instance or None if not found

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(self.model).where(self.model.id == entity_id)
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get entity by ID",
                    model=self.model.__name__,
                    entity_id=str(entity_id),
                    error=str(e),
                )
                raise DatabaseError("Entity retrieval failed", cause=e) from e

    async def get_by_id_or_raise(self, entity_id: UUID | str) -> T:
        """Get model instance by ID or raise NotFoundError.

        Args:
            entity_id: Entity identifier

        Returns:
            Model instance

        Raises:
            NotFoundError: If entity not found
            DatabaseError: If query fails
        """
        entity = await self.get_by_id(entity_id)
        if entity is None:
            raise NotFoundError(f"{self.model.__name__} with ID {entity_id} not found")
        return entity

    async def update(self, entity_id: UUID | str, **kwargs: Any) -> T:
        """Update model instance.

        Args:
            entity_id: Entity identifier
            **kwargs: Fields to update

        Returns:
            Updated model instance

        Raises:
            NotFoundError: If entity not found
            DatabaseError: If update fails
        """
        async with self._get_session() as session:
            try:
                # First check if entity exists
                entity = await session.get(self.model, entity_id)
                if entity is None:
                    raise NotFoundError(
                        f"{self.model.__name__} with ID {entity_id} not found"
                    )

                # Update fields
                for key, value in kwargs.items():
                    if hasattr(entity, key):
                        setattr(entity, key, value)

                await session.commit()
                await session.refresh(entity)
                return entity

            except NotFoundError:
                raise
            except SQLAlchemyError as e:
                logger.error(
                    "Failed to update entity",
                    model=self.model.__name__,
                    entity_id=str(entity_id),
                    error=str(e),
                )
                raise DatabaseError("Entity update failed", cause=e) from e

    async def delete(self, entity_id: UUID | str) -> bool:
        """Delete model instance.

        Args:
            entity_id: Entity identifier

        Returns:
            True if deleted, False if not found

        Raises:
            DatabaseError: If deletion fails
        """
        async with self._get_session() as session:
            try:
                query = delete(self.model).where(self.model.id == entity_id)
                result = await session.execute(query)
                await session.commit()
                return result.rowcount > 0
            except SQLAlchemyError as e:
                logger.error(
                    "Failed to delete entity",
                    model=self.model.__name__,
                    entity_id=str(entity_id),
                    error=str(e),
                )
                raise DatabaseError("Entity deletion failed", cause=e) from e

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        order_desc: bool = True,
    ) -> list[T]:
        """List all model instances with pagination.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: Field to order by
            order_desc: Whether to order descending

        Returns:
            List of model instances

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(self.model)

                # Add ordering
                if hasattr(self.model, order_by):
                    order_field = getattr(self.model, order_by)
                    query = query.order_by(
                        order_field.desc() if order_desc else order_field
                    )

                # Add pagination
                query = query.limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to list entities", model=self.model.__name__, error=str(e)
                )
                raise DatabaseError("Entity listing failed", cause=e) from e

    async def count(self) -> int:
        """Count total number of model instances.

        Returns:
            Total count

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(self.model).with_only_columns(self.model.id)
                result = await session.execute(query)
                return len(result.all())
            except SQLAlchemyError as e:
                logger.error(
                    "Failed to count entities", model=self.model.__name__, error=str(e)
                )
                raise DatabaseError("Entity counting failed", cause=e) from e

    async def exists(self, entity_id: UUID | str) -> bool:
        """Check if model instance exists.

        Args:
            entity_id: Entity identifier

        Returns:
            True if exists, False otherwise

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(self.model.id).where(self.model.id == entity_id)
                result = await session.execute(query)
                return result.scalar_one_or_none() is not None
            except SQLAlchemyError as e:
                logger.error(
                    "Failed to check entity existence",
                    model=self.model.__name__,
                    entity_id=str(entity_id),
                    error=str(e),
                )
                raise DatabaseError("Entity existence check failed", cause=e) from e
