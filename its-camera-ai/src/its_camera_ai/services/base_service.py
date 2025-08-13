"""Base async service class for common CRUD operations.

Provides generic async CRUD functionality with error handling,
logging, and transaction management for all service classes.
"""

from typing import Any, TypeVar
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from ..models.base import BaseTableModel

T = TypeVar("T", bound=BaseTableModel)
logger = get_logger(__name__)


class BaseAsyncService[T: BaseTableModel]:
    """Base async service class for database operations.

    Provides common CRUD operations with proper error handling,
    logging, and transaction management.
    """

    def __init__(self, session: AsyncSession, model: type[T]):
        self.session = session
        self.model = model

    async def create(self, **kwargs: Any) -> T:
        """Create a new model instance.

        Args:
            **kwargs: Model field values

        Returns:
            Created model instance

        Raises:
            DatabaseError: If creation fails
        """
        try:
            instance = self.model(**kwargs)
            self.session.add(instance)
            await self.session.commit()
            await self.session.refresh(instance)

            logger.debug(
                "Model instance created",
                model=self.model.__name__,
                id=instance.id,
            )

            return instance

        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(
                "Failed to create model instance",
                model=self.model.__name__,
                error=str(e),
            )
            raise DatabaseError(
                f"Failed to create {self.model.__name__}: {str(e)}"
            ) from e

    async def get_by_id(self, id_value: str | UUID | int) -> T | None:
        """Get model instance by ID.

        Args:
            id_value: Model ID value

        Returns:
            Model instance if found, None otherwise

        Raises:
            DatabaseError: If query fails
        """
        try:
            query = select(self.model).where(self.model.id == id_value)
            result = await self.session.execute(query)
            return result.scalar_one_or_none()

        except SQLAlchemyError as e:
            logger.error(
                "Failed to get model by ID",
                model=self.model.__name__,
                id=id_value,
                error=str(e),
            )
            raise DatabaseError(f"Failed to get {self.model.__name__}: {str(e)}") from e

    async def get_all(
        self,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | None = None,
    ) -> list[T]:
        """Get all model instances.

        Args:
            limit: Maximum number of instances to return
            offset: Number of instances to skip
            order_by: Field name to order by

        Returns:
            List of model instances

        Raises:
            DatabaseError: If query fails
        """
        try:
            query = select(self.model)

            if order_by and hasattr(self.model, order_by):
                query = query.order_by(getattr(self.model, order_by))

            if offset:
                query = query.offset(offset)

            if limit:
                query = query.limit(limit)

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except SQLAlchemyError as e:
            logger.error(
                "Failed to get all models",
                model=self.model.__name__,
                error=str(e),
            )
            raise DatabaseError(
                f"Failed to get {self.model.__name__} list: {str(e)}"
            ) from e

    async def update_by_id(self, id_value: str | UUID | int, **kwargs: Any) -> T | None:
        """Update model instance by ID.

        Args:
            id_value: Model ID value
            **kwargs: Fields to update

        Returns:
            Updated model instance if found, None otherwise

        Raises:
            DatabaseError: If update fails
        """
        try:
            # First check if instance exists
            instance = await self.get_by_id(id_value)
            if not instance:
                return None

            # Update fields
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)

            await self.session.commit()
            await self.session.refresh(instance)

            logger.debug(
                "Model instance updated",
                model=self.model.__name__,
                id=id_value,
                updated_fields=list(kwargs.keys()),
            )

            return instance

        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(
                "Failed to update model",
                model=self.model.__name__,
                id=id_value,
                error=str(e),
            )
            raise DatabaseError(
                f"Failed to update {self.model.__name__}: {str(e)}"
            ) from e

    async def delete_by_id(self, id_value: str | UUID | int) -> bool:
        """Delete model instance by ID.

        Args:
            id_value: Model ID value

        Returns:
            True if instance was deleted, False if not found

        Raises:
            DatabaseError: If deletion fails
        """
        try:
            # Check if instance exists
            instance = await self.get_by_id(id_value)
            if not instance:
                return False

            await self.session.delete(instance)
            await self.session.commit()

            logger.debug(
                "Model instance deleted",
                model=self.model.__name__,
                id=id_value,
            )

            return True

        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(
                "Failed to delete model",
                model=self.model.__name__,
                id=id_value,
                error=str(e),
            )
            raise DatabaseError(
                f"Failed to delete {self.model.__name__}: {str(e)}"
            ) from e

    async def bulk_create(self, instances: list[dict[str, Any]]) -> list[T]:
        """Bulk create model instances.

        Args:
            instances: List of instance data dictionaries

        Returns:
            List of created model instances

        Raises:
            DatabaseError: If bulk creation fails
        """
        try:
            created_instances = []

            for instance_data in instances:
                instance = self.model(**instance_data)
                self.session.add(instance)
                created_instances.append(instance)

            await self.session.commit()

            # Refresh all instances to get generated IDs
            for instance in created_instances:
                await self.session.refresh(instance)

            logger.debug(
                "Bulk model creation completed",
                model=self.model.__name__,
                count=len(created_instances),
            )

            return created_instances

        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(
                "Failed to bulk create models",
                model=self.model.__name__,
                count=len(instances),
                error=str(e),
            )
            raise DatabaseError(
                f"Failed to bulk create {self.model.__name__}: {str(e)}"
            ) from e

    async def bulk_update(
        self, updates: list[dict[str, Any]], id_field: str = "id"
    ) -> int:
        """Bulk update model instances.

        Args:
            updates: List of update dictionaries (must include ID field)
            id_field: Name of the ID field (default: 'id')

        Returns:
            Number of updated instances

        Raises:
            DatabaseError: If bulk update fails
        """
        try:
            updated_count = 0

            for update_data in updates:
                if id_field not in update_data:
                    continue

                id_value = update_data.pop(id_field)

                query = (
                    update(self.model)
                    .where(self.model.id == id_value)
                    .values(**update_data)
                )

                result = await self.session.execute(query)
                updated_count += result.rowcount

            await self.session.commit()

            logger.debug(
                "Bulk model update completed",
                model=self.model.__name__,
                updated_count=updated_count,
                total_requested=len(updates),
            )

            return updated_count

        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(
                "Failed to bulk update models",
                model=self.model.__name__,
                count=len(updates),
                error=str(e),
            )
            raise DatabaseError(
                f"Failed to bulk update {self.model.__name__}: {str(e)}"
            ) from e

    async def bulk_delete(self, id_values: list[str | UUID | int]) -> int:
        """Bulk delete model instances by ID.

        Args:
            id_values: List of ID values to delete

        Returns:
            Number of deleted instances

        Raises:
            DatabaseError: If bulk deletion fails
        """
        try:
            query = delete(self.model).where(self.model.id.in_(id_values))
            result = await self.session.execute(query)
            deleted_count = result.rowcount

            await self.session.commit()

            logger.debug(
                "Bulk model deletion completed",
                model=self.model.__name__,
                deleted_count=deleted_count,
                requested_count=len(id_values),
            )

            return deleted_count

        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(
                "Failed to bulk delete models",
                model=self.model.__name__,
                count=len(id_values),
                error=str(e),
            )
            raise DatabaseError(
                f"Failed to bulk delete {self.model.__name__}: {str(e)}"
            ) from e

    async def count(self, **filters: Any) -> int:
        """Count model instances with optional filters.

        Args:
            **filters: Filter conditions

        Returns:
            Number of matching instances

        Raises:
            DatabaseError: If count query fails
        """
        try:
            from sqlalchemy import func

            query = select(func.count(self.model.id))

            # Apply filters
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.where(getattr(self.model, key) == value)

            result = await self.session.execute(query)
            return result.scalar()

        except SQLAlchemyError as e:
            logger.error(
                "Failed to count models",
                model=self.model.__name__,
                filters=filters,
                error=str(e),
            )
            raise DatabaseError(
                f"Failed to count {self.model.__name__}: {str(e)}"
            ) from e

    async def exists(self, **filters: Any) -> bool:
        """Check if model instance exists with given filters.

        Args:
            **filters: Filter conditions

        Returns:
            True if instance exists, False otherwise

        Raises:
            DatabaseError: If existence query fails
        """
        try:
            query = select(self.model.id)

            # Apply filters
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.where(getattr(self.model, key) == value)

            query = query.limit(1)

            result = await self.session.execute(query)
            return result.scalar_one_or_none() is not None

        except SQLAlchemyError as e:
            logger.error(
                "Failed to check model existence",
                model=self.model.__name__,
                filters=filters,
                error=str(e),
            )
            raise DatabaseError(
                f"Failed to check {self.model.__name__} existence: {str(e)}"
            ) from e
