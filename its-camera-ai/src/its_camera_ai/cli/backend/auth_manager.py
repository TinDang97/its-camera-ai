"""Authentication manager for CLI backend integration.

Provides authentication and authorization capabilities for CLI operations,
integrating with the existing security framework.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from passlib.context import CryptContext

from ...core.config import Settings, get_settings
from ...core.logging import get_logger

# from ...services.auth import AuthService  # Avoid circular import
from .database_manager import CLIDatabaseManager

logger = get_logger(__name__)


class AuthenticationError(Exception):
    """Authentication related errors."""
    pass


class AuthorizationError(Exception):
    """Authorization related errors."""
    pass


class CLIAuthManager:
    """Authentication manager for CLI operations.
    
    Features:
    - JWT token management
    - User authentication
    - Permission checking
    - Session management
    - Token refresh
    """

    def __init__(self, settings: Settings = None):
        """Initialize authentication manager.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self.db_manager = CLIDatabaseManager(self.settings)

        # Password hashing
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto"
        )

        # JWT settings
        self.secret_key = self.settings.security.secret_key
        self.algorithm = self.settings.security.algorithm
        self.access_token_expire_minutes = self.settings.security.access_token_expire_minutes

        # Current session
        self._current_user: dict[str, Any] | None = None
        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None

        # Auth service
        self._auth_service: AuthService | None = None

        logger.info("CLI authentication manager initialized")

    async def initialize(self) -> None:
        """Initialize authentication manager."""
        await self.db_manager.initialize()
        self._auth_service = AuthService(self.settings.security)
        logger.info("Authentication manager initialized")

    async def close(self) -> None:
        """Close authentication manager."""
        await self.db_manager.close()
        self._current_user = None
        self._access_token = None
        self._token_expires_at = None
        logger.info("Authentication manager closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def create_access_token(
        self,
        data: dict[str, Any],
        expires_delta: timedelta | None = None
    ) -> str:
        """Create JWT access token.
        
        Args:
            data: Token payload data
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.now(UTC) + expires_delta
        else:
            expire = datetime.now(UTC) + timedelta(
                minutes=self.access_token_expire_minutes
            )

        to_encode.update({"exp": expire})

        encoded_jwt = jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )

        return encoded_jwt

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            
        Returns:
            Decoded token payload
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return payload

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError("Invalid token")

    def hash_password(self, password: str) -> str:
        """Hash a password.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches
        """
        return self.pwd_context.verify(plain_password, hashed_password)

    async def authenticate_user(
        self,
        username: str,
        password: str
    ) -> dict[str, Any] | None:
        """Authenticate user with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User information if authentication successful
        """
        if not self._auth_service:
            await self.initialize()

        try:
            async with self.db_manager.get_session() as session:
                user = await self._auth_service.get_user_by_username(
                    session, username
                )

                if not user:
                    logger.warning(f"Authentication failed: user {username} not found")
                    return None

                if not user.is_active:
                    logger.warning(f"Authentication failed: user {username} is inactive")
                    return None

                if not self.verify_password(password, user.hashed_password):
                    logger.warning(f"Authentication failed: invalid password for {username}")
                    return None

                # Authentication successful
                user_data = {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "is_superuser": user.is_superuser,
                    "roles": [role.name for role in user.roles] if user.roles else [],
                    "permissions": user.permissions or []
                }

                logger.info(f"User {username} authenticated successfully")
                return user_data

        except Exception as e:
            logger.error(f"Authentication error for user {username}: {e}")
            return None

    async def login(
        self,
        username: str,
        password: str
    ) -> dict[str, Any]:
        """Login user and create session.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Login response with token information
            
        Raises:
            AuthenticationError: If authentication fails
        """
        user_data = await self.authenticate_user(username, password)

        if not user_data:
            raise AuthenticationError("Invalid username or password")

        # Create access token
        token_data = {
            "sub": user_data["username"],
            "user_id": user_data["id"],
            "roles": user_data["roles"],
            "is_superuser": user_data["is_superuser"]
        }

        access_token = self.create_access_token(token_data)
        expires_at = datetime.now(UTC) + timedelta(
            minutes=self.access_token_expire_minutes
        )

        # Store session information
        self._current_user = user_data
        self._access_token = access_token
        self._token_expires_at = expires_at

        logger.info(f"User {username} logged in successfully")

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_at": expires_at.isoformat(),
            "user": user_data
        }

    async def logout(self) -> None:
        """Logout current user and clear session."""
        if self._current_user:
            logger.info(f"User {self._current_user['username']} logged out")

        self._current_user = None
        self._access_token = None
        self._token_expires_at = None

    async def get_current_user(self) -> dict[str, Any] | None:
        """Get current authenticated user.
        
        Returns:
            Current user information or None
        """
        if not self._current_user or not self._access_token:
            return None

        # Check if token is still valid
        if self._token_expires_at and datetime.now(UTC) > self._token_expires_at:
            logger.warning("Access token expired")
            await self.logout()
            return None

        return self._current_user

    async def refresh_token(self) -> str | None:
        """Refresh current access token.
        
        Returns:
            New access token or None if refresh failed
        """
        if not self._current_user:
            return None

        try:
            # Create new token with same user data
            token_data = {
                "sub": self._current_user["username"],
                "user_id": self._current_user["id"],
                "roles": self._current_user["roles"],
                "is_superuser": self._current_user["is_superuser"]
            }

            new_token = self.create_access_token(token_data)
            new_expires_at = datetime.now(UTC) + timedelta(
                minutes=self.access_token_expire_minutes
            )

            self._access_token = new_token
            self._token_expires_at = new_expires_at

            logger.info(f"Token refreshed for user {self._current_user['username']}")
            return new_token

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return None

    async def check_permission(
        self,
        permission: str,
        user: dict[str, Any] | None = None
    ) -> bool:
        """Check if user has specific permission.
        
        Args:
            permission: Permission string to check
            user: User data (uses current user if not provided)
            
        Returns:
            True if user has permission
        """
        target_user = user or await self.get_current_user()

        if not target_user:
            return False

        # Superuser has all permissions
        if target_user.get("is_superuser", False):
            return True

        # Check explicit permissions
        user_permissions = target_user.get("permissions", [])
        if permission in user_permissions or "*" in user_permissions:
            return True

        # Check role-based permissions (simplified)
        # In a real implementation, this would query role permissions
        user_roles = target_user.get("roles", [])
        admin_permissions = [
            "system:read", "system:write", "cameras:read", "cameras:write",
            "analytics:read", "models:read", "models:write"
        ]

        if "admin" in user_roles and permission in admin_permissions:
            return True

        return False

    async def check_role(
        self,
        role: str,
        user: dict[str, Any] | None = None
    ) -> bool:
        """Check if user has specific role.
        
        Args:
            role: Role name to check
            user: User data (uses current user if not provided)
            
        Returns:
            True if user has role
        """
        target_user = user or await self.get_current_user()

        if not target_user:
            return False

        user_roles = target_user.get("roles", [])
        return role in user_roles

    def require_authentication(self, func):
        """Decorator to require authentication for a function."""
        async def wrapper(*args, **kwargs):
            current_user = await self.get_current_user()
            if not current_user:
                raise AuthenticationError("Authentication required")
            return await func(*args, **kwargs)
        return wrapper

    def require_permission(self, permission: str):
        """Decorator to require specific permission for a function."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if not await self.check_permission(permission):
                    raise AuthorizationError(f"Permission required: {permission}")
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def require_role(self, role: str):
        """Decorator to require specific role for a function."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if not await self.check_role(role):
                    raise AuthorizationError(f"Role required: {role}")
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    async def create_user(
        self,
        username: str,
        password: str,
        email: str,
        full_name: str,
        roles: list[str] | None = None,
        permissions: list[str] | None = None,
        is_superuser: bool = False
    ) -> dict[str, Any]:
        """Create new user account.
        
        Args:
            username: Username
            password: Plain text password
            email: Email address
            full_name: Full name
            roles: List of role names
            permissions: List of permission strings
            is_superuser: Whether user is superuser
            
        Returns:
            Created user information
            
        Raises:
            AuthorizationError: If current user lacks permission
        """
        # Check if current user can create users
        if not await self.check_permission("users:write"):
            raise AuthorizationError("Permission required to create users")

        try:
            from ...models.user import User

            # Hash password
            hashed_password = self.hash_password(password)

            async with self.db_manager.get_session() as session:
                # Check if username already exists
                existing_user = await self._auth_service.get_user_by_username(
                    session, username
                )
                if existing_user:
                    raise ValueError(f"Username {username} already exists")

                # Create user
                new_user = User(
                    username=username,
                    email=email,
                    full_name=full_name,
                    hashed_password=hashed_password,
                    is_active=True,
                    is_superuser=is_superuser,
                    roles=roles or [],
                    permissions=permissions or []
                )

                session.add(new_user)
                await session.commit()
                await session.refresh(new_user)

                user_data = {
                    "id": new_user.id,
                    "username": new_user.username,
                    "email": new_user.email,
                    "full_name": new_user.full_name,
                    "is_superuser": new_user.is_superuser,
                    "roles": new_user.roles,
                    "permissions": new_user.permissions
                }

                logger.info(f"Created user: {username}")
                return user_data

        except Exception as e:
            logger.error(f"Failed to create user {username}: {e}")
            raise

    async def list_users(self) -> list[dict[str, Any]]:
        """List all users.
        
        Returns:
            List of user information
            
        Raises:
            AuthorizationError: If current user lacks permission
        """
        if not await self.check_permission("users:read"):
            raise AuthorizationError("Permission required to list users")

        try:
            users_query = """
                SELECT id, username, email, full_name, is_active, is_superuser, 
                       roles, permissions, created_at, updated_at
                FROM users
                ORDER BY username
            """

            users = await self.db_manager.execute_query(users_query)
            return users

        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            raise

    def is_authenticated(self) -> bool:
        """Check if there is a current authenticated user.
        
        Returns:
            True if user is authenticated
        """
        return (
            self._current_user is not None
            and self._access_token is not None
            and self._token_expires_at is not None
            and datetime.now(UTC) < self._token_expires_at
        )

    def get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for API requests.
        
        Returns:
            Dictionary with authorization header
        """
        if self._access_token:
            return {"Authorization": f"Bearer {self._access_token}"}
        return {}

    def get_session_info(self) -> dict[str, Any]:
        """Get current session information.
        
        Returns:
            Session information dictionary
        """
        return {
            "authenticated": self.is_authenticated(),
            "current_user": self._current_user,
            "token_expires_at": self._token_expires_at.isoformat() if self._token_expires_at else None,
            "time_until_expiry": (
                (self._token_expires_at - datetime.now(UTC)).total_seconds()
                if self._token_expires_at and self._token_expires_at > datetime.now(UTC)
                else 0
            )
        }
