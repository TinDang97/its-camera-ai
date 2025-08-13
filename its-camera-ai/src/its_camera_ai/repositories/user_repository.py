"""User repository for user management data access.

Provides specialized methods for user authentication, role management,
and security operations with optimized queries and proper error handling.
"""

from datetime import datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, sessionmaker

from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from ..models.user import User
from .base_repository import BaseRepository

logger = get_logger(__name__)


class UserRepository(BaseRepository[User]):
    """Repository for user data access operations.

    Specialized methods for user authentication, role management,
    and security operations with optimized queries.
    """

    def __init__(self, session_factory: sessionmaker[AsyncSession]):
        super().__init__(session_factory, User)

    async def get_by_username(self, username: str) -> User | None:
        """Get user by username.

        Args:
            username: Username to search for

        Returns:
            User instance or None if not found

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(User).where(User.username == username)
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get user by username",
                    username=username,
                    error=str(e)
                )
                raise DatabaseError("User retrieval failed", cause=e) from e

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email.

        Args:
            email: Email to search for

        Returns:
            User instance or None if not found

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(User).where(User.email == email)
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get user by email",
                    email=email,
                    error=str(e)
                )
                raise DatabaseError("User retrieval failed", cause=e) from e

    async def get_by_username_or_email(
        self,
        username_or_email: str
    ) -> User | None:
        """Get user by username or email.

        Args:
            username_or_email: Username or email to search for

        Returns:
            User instance or None if not found

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(User).where(
                    or_(
                        User.username == username_or_email,
                        User.email == username_or_email
                    )
                )
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get user by username or email",
                    username_or_email=username_or_email,
                    error=str(e)
                )
                raise DatabaseError("User retrieval failed", cause=e) from e

    async def get_with_roles(self, user_id: str) -> User | None:
        """Get user with roles loaded.

        Args:
            user_id: User identifier

        Returns:
            User instance with roles or None if not found

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(User).where(User.id == user_id).options(
                    selectinload(User.roles)
                )
                result = await session.execute(query)
                return result.scalar_one_or_none()
            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get user with roles",
                    user_id=user_id,
                    error=str(e)
                )
                raise DatabaseError("User retrieval failed", cause=e) from e

    async def get_active_users(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> list[User]:
        """Get active users.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of active users

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(User).where(
                    and_(User.is_active, User.is_verified)
                ).limit(limit).offset(offset)

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get active users",
                    error=str(e)
                )
                raise DatabaseError("Active users retrieval failed", cause=e) from e

    async def update_password(
        self,
        user_id: str,
        hashed_password: str
    ) -> bool:
        """Update user password.

        Args:
            user_id: User identifier
            hashed_password: New hashed password

        Returns:
            True if password updated successfully

        Raises:
            DatabaseError: If update fails
        """
        async with self._get_session() as session:
            try:
                user = await session.get(User, user_id)
                if user is None:
                    return False

                user.hashed_password = hashed_password
                user.password_changed_at = datetime.now()

                await session.commit()
                return True

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to update user password",
                    user_id=user_id,
                    error=str(e)
                )
                raise DatabaseError("Password update failed", cause=e) from e

    async def update_last_login(
        self,
        user_id: str,
        login_time: datetime | None = None
    ) -> bool:
        """Update user last login time.

        Args:
            user_id: User identifier
            login_time: Login time (defaults to current time)

        Returns:
            True if updated successfully

        Raises:
            DatabaseError: If update fails
        """
        async with self._get_session() as session:
            try:
                user = await session.get(User, user_id)
                if user is None:
                    return False

                user.last_login = login_time or datetime.now()
                await session.commit()
                return True

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to update last login",
                    user_id=user_id,
                    error=str(e)
                )
                raise DatabaseError("Last login update failed", cause=e) from e

    async def set_verified(self, user_id: str) -> bool:
        """Mark user as verified.

        Args:
            user_id: User identifier

        Returns:
            True if marked as verified

        Raises:
            DatabaseError: If update fails
        """
        async with self._get_session() as session:
            try:
                user = await session.get(User, user_id)
                if user is None:
                    return False

                user.is_verified = True
                user.verified_at = datetime.now()

                await session.commit()
                return True

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to set user as verified",
                    user_id=user_id,
                    error=str(e)
                )
                raise DatabaseError("User verification failed", cause=e) from e

    async def toggle_active_status(self, user_id: str) -> bool:
        """Toggle user active status.

        Args:
            user_id: User identifier

        Returns:
            New active status

        Raises:
            DatabaseError: If update fails
        """
        async with self._get_session() as session:
            try:
                user = await session.get(User, user_id)
                if user is None:
                    return False

                user.is_active = not user.is_active
                await session.commit()

                return user.is_active

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to toggle user active status",
                    user_id=user_id,
                    error=str(e)
                )
                raise DatabaseError("User status toggle failed", cause=e) from e

    async def search_users(
        self,
        query: str,
        limit: int = 50,
        offset: int = 0
    ) -> list[User]:
        """Search users by username, email, or full name.

        Args:
            query: Search query
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching users

        Raises:
            DatabaseError: If search fails
        """
        async with self._get_session() as session:
            try:
                search_term = f"%{query}%"
                sql_query = select(User).where(
                    or_(
                        User.username.ilike(search_term),
                        User.email.ilike(search_term),
                        User.full_name.ilike(search_term)
                    )
                ).limit(limit).offset(offset)

                result = await session.execute(sql_query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to search users",
                    query=query,
                    error=str(e)
                )
                raise DatabaseError("User search failed", cause=e) from e
