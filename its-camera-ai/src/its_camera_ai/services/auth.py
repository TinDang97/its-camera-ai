"""Authentication service for user management and JWT tokens."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..core.config import SecurityConfig
from ..models.user import User


class AuthService:
    """Authentication and authorization service."""

    def __init__(self, security_config: SecurityConfig) -> None:
        self.security_config = security_config

    async def get_user_by_id(self, db: AsyncSession, user_id: str) -> User | None:
        """Get user by ID with roles.

        Args:
            db: Database session
            user_id: User ID

        Returns:
            User or None if not found
        """
        query = select(User).options(selectinload(User.roles)).where(User.id == user_id)
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def get_user_by_username(
        self, db: AsyncSession, username: str
    ) -> User | None:
        """Get user by username with roles.

        Args:
            db: Database session
            username: Username

        Returns:
            User or None if not found
        """
        query = (
            select(User)
            .options(selectinload(User.roles))
            .where(User.username == username)
        )
        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def user_has_permissions(
        self, user: User, _permissions: tuple[str, ...]
    ) -> bool:
        """Check if user has required permissions.

        Args:
            user: User instance
            permissions: Required permissions

        Returns:
            True if user has all required permissions
        """
        # TODO: Implement permission checking logic
        return user.is_superuser or user.is_active

    async def user_has_roles(self, user: User, roles: tuple[str, ...]) -> bool:
        """Check if user has required roles.

        Args:
            user: User instance
            roles: Required roles

        Returns:
            True if user has any of the required roles
        """
        if user.is_superuser:
            return True

        user_role_names = {role.name for role in user.roles}
        required_roles = set(roles)

        return bool(user_role_names.intersection(required_roles))
