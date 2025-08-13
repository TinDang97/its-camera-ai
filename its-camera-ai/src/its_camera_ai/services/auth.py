"""Authentication service for user management and JWT tokens."""

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..core.config import SecurityConfig
from ..core.logging import get_logger
from ..models.user import Permission, SecurityAuditLog, User
from ..services.cache import CacheService

logger = get_logger(__name__)


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
        self, db: AsyncSession, user: User, permissions: tuple[str, ...]
    ) -> bool:
        """Check if user has required permissions.

        Args:
            db: Database session
            user: User instance
            permissions: Required permissions tuple

        Returns:
            True if user has all required permissions
        """
        if user.is_superuser:
            return True

        if not user.is_active:
            return False

        # Get user permissions through roles
        user_permissions = set()
        for role in user.roles:
            for permission in role.permissions:
                user_permissions.add(permission.name)

        # Check if user has all required permissions
        required_permissions = set(permissions)
        return required_permissions.issubset(user_permissions)

    async def get_user_permissions(self, db: AsyncSession, user: User) -> list[str]:
        """Get all permissions for a user.
        
        Args:
            db: Database session
            user: User instance
            
        Returns:
            List of permission names
        """
        if user.is_superuser:
            # Superuser has all permissions
            query = select(Permission.name)
            result = await db.execute(query)
            return [perm for perm in result.scalars().all()]

        permissions = set()
        for role in user.roles:
            for permission in role.permissions:
                permissions.add(permission.name)

        return list(permissions)

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

    async def create_security_audit_log(
        self,
        db: AsyncSession,
        event_type: str,
        user_id: str | None = None,
        username: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        session_id: str | None = None,
        resource: str | None = None,
        action: str | None = None,
        success: bool = True,
        error_message: str | None = None,
        details: dict[str, Any] | None = None,
        risk_score: int = 0
    ) -> SecurityAuditLog:
        """Create security audit log entry.
        
        Args:
            db: Database session
            event_type: Type of security event
            user_id: User ID if applicable
            username: Username if applicable
            ip_address: Client IP address
            user_agent: Client user agent
            session_id: Session ID if applicable
            resource: Resource being accessed
            action: Action being performed
            success: Whether the event was successful
            error_message: Error message if failed
            details: Additional event details
            risk_score: Risk score for the event (0-100)
            
        Returns:
            Created SecurityAuditLog entry
        """
        audit_log = SecurityAuditLog(
            event_type=event_type,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            resource=resource,
            action=action,
            success=success,
            error_message=error_message,
            details=json.dumps(details) if details else None,
            risk_score=risk_score,
            timestamp=datetime.now(UTC)
        )

        db.add(audit_log)
        await db.commit()

        # Log high-risk events
        if risk_score >= self.security_config.high_risk_alert_threshold:
            logger.warning(
                "High-risk security event detected",
                event_type=event_type,
                user_id=user_id,
                username=username,
                risk_score=risk_score,
                details=details
            )

        return audit_log

    async def check_account_lockout(
        self, db: AsyncSession, cache: CacheService, user: User, ip_address: str | None = None
    ) -> tuple[bool, datetime | None]:
        """Check if account is locked due to failed login attempts.
        
        Args:
            db: Database session
            cache: Cache service
            user: User to check
            ip_address: Client IP address
            
        Returns:
            Tuple of (is_locked, unlock_time)
        """
        now = datetime.now(UTC)

        # Check if account is manually locked
        if user.account_locked_until and user.account_locked_until > now:
            return True, user.account_locked_until

        # Check failed login attempts
        cache_key = f"failed_attempts:{user.id}"
        attempts_data = await cache.get_json(cache_key)

        if not attempts_data:
            return False, None

        attempts = attempts_data.get('count', 0)
        last_attempt = datetime.fromisoformat(attempts_data.get('last_attempt', now.isoformat()))

        # Check if within attempt window
        window_start = now - timedelta(minutes=self.security_config.attempt_window_minutes)
        if last_attempt < window_start:
            # Reset attempts if outside window
            await cache.delete(cache_key)
            return False, None

        # Check if max attempts exceeded
        if attempts >= self.security_config.max_login_attempts:
            lockout_until = last_attempt + timedelta(minutes=self.security_config.lockout_duration_minutes)

            if now < lockout_until:
                # Update database lockout status
                user.account_locked_until = lockout_until
                await db.commit()

                return True, lockout_until
            else:
                # Lockout expired, reset attempts
                await cache.delete(cache_key)
                user.account_locked_until = None
                await db.commit()

        return False, None

    async def record_failed_login(
        self, db: AsyncSession, cache: CacheService, user: User, ip_address: str | None = None
    ) -> None:
        """Record failed login attempt.
        
        Args:
            db: Database session
            cache: Cache service
            user: User who failed login
            ip_address: Client IP address
        """
        now = datetime.now(UTC)
        cache_key = f"failed_attempts:{user.id}"

        # Get current attempts
        attempts_data = await cache.get_json(cache_key) or {'count': 0}
        attempts_data['count'] += 1
        attempts_data['last_attempt'] = now.isoformat()

        # Store for attempt window duration
        ttl = self.security_config.attempt_window_minutes * 60
        await cache.set_json(cache_key, attempts_data, ttl)

        # Update database counter
        user.failed_login_attempts += 1
        await db.commit()

        # Create audit log
        await self.create_security_audit_log(
            db,
            event_type="failed_login",
            user_id=user.id,
            username=user.username,
            ip_address=ip_address,
            success=False,
            details={'attempt_count': attempts_data['count']},
            risk_score=min(100, attempts_data['count'] * 20)
        )

    async def clear_failed_login_attempts(
        self, db: AsyncSession, cache: CacheService, user: User
    ) -> None:
        """Clear failed login attempts for successful login.
        
        Args:
            db: Database session
            cache: Cache service
            user: User who successfully logged in
        """
        cache_key = f"failed_attempts:{user.id}"
        await cache.delete(cache_key)

        user.failed_login_attempts = 0
        user.account_locked_until = None
        await db.commit()
