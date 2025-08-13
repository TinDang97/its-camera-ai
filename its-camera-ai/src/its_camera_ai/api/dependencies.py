"""FastAPI dependencies using dependency injection containers.

This module provides FastAPI-compatible dependencies that use the
dependency injection container system for clean architecture and
proper resource management.

The dependencies follow these patterns:
- Infrastructure dependencies (database, cache) use Resource providers
- Service dependencies use Factory or Singleton providers
- Authentication dependencies use the container's services
- Rate limiting and validation use container-managed services
"""

from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dependency_injector.wiring import Provide, inject
from fastapi import Depends, HTTPException, Request, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from ..containers import ApplicationContainer
from ..core.config import Settings, get_settings
from ..core.exceptions import AuthenticationError
from ..core.logging import get_logger
from ..models.user import User
from ..services.auth_service import AuthenticationService as AuthService
from ..services.cache import CacheService
from ..services.camera_service import CameraService
from ..services.frame_service import DetectionService, FrameService
from ..services.metrics_service import MetricsService

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)


# ============================
# Core Infrastructure Dependencies
# ============================


async def get_database_session(
    request: Request,
) -> AsyncGenerator[AsyncSession, None]:
    """Get database session from app-bound container.

    Args:
        request: FastAPI request object (contains app state)

    Yields:
        AsyncSession: Database session with proper lifecycle management

    Raises:
        DatabaseError: If session creation fails
    """
    container = request.app.state.container
    session_factory = container.infrastructure.session_factory()

    async with session_factory() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error("Database session error in DI", error=str(e))
            raise
        finally:
            await session.close()


def get_cache_service(
    request: Request,
) -> CacheService:
    """Get cache service from app-bound container.

    Args:
        request: FastAPI request object (contains app state)

    Returns:
        CacheService: Configured cache service instance
    """
    container = request.app.state.container
    return container.services.cache_service()


def get_redis_client(request: Request):
    """Get Redis client from app-bound container.

    Args:
        request: FastAPI request object (contains app state)

    Returns:
        Redis client instance
    """
    container = request.app.state.container
    return container.infrastructure.redis_client()


# ============================
# Service Dependencies
# ============================


def get_auth_service(
    request: Request,
) -> AuthService:
    """Get authentication service from app-bound container.

    Args:
        request: FastAPI request object (contains app state)

    Returns:
        AuthService: Configured authentication service instance
    """
    container = request.app.state.container
    return container.services.auth_service()


def get_camera_service(
    request: Request,
) -> CameraService:
    """Get camera service from app-bound container.

    Args:
        request: FastAPI request object (contains app state)

    Returns:
        CameraService: Camera service instance
    """
    container = request.app.state.container
    return container.services.camera_service()


def get_frame_service(request: Request) -> FrameService:
    """Get frame service from app-bound container.

    Args:
        request: FastAPI request object (contains app state)

    Returns:
        FrameService: Frame service instance
    """
    container = request.app.state.container
    return container.services.frame_service()


@inject
def get_detection_service(
    detection_service=Provide[ApplicationContainer.services.detection_service],
) -> DetectionService:
    """Get detection service from dependency injection container.

    Returns:
        DetectionService: Detection service instance
    """
    return detection_service


def get_metrics_service(request: Request) -> MetricsService:
    """Get metrics service from app-bound container.

    Args:
        request: FastAPI request object (contains app state)

    Returns:
        MetricsService: Metrics service instance
    """
    container = request.app.state.container
    return container.services.metrics_service()


@inject
def get_streaming_service(
    streaming_service=Provide[ApplicationContainer.services.streaming_service],
):
    """Get streaming service from dependency injection container.

    Returns:
        StreamingService: Streaming service instance
    """
    return streaming_service


@inject
def get_analytics_service(
    analytics_service=Provide[ApplicationContainer.services.analytics_service],
):
    """Get analytics service from dependency injection container.

    Returns:
        AnalyticsService: Analytics service instance
    """
    return analytics_service


@inject
def get_alert_service(
    alert_service=Provide[ApplicationContainer.services.alert_service],
):
    """Get alert service from dependency injection container.

    Returns:
        AlertService: Alert service instance
    """
    return alert_service


@inject
def get_token_service(
    token_service=Provide[ApplicationContainer.services.token_service],
):
    """Get token service from dependency injection container.

    Returns:
        TokenService: Token service instance
    """
    return token_service


@inject
def get_mfa_service(
    mfa_service=Provide[ApplicationContainer.services.mfa_service],
):
    """Get MFA service from dependency injection container.

    Returns:
        MFAService: MFA service instance
    """
    return mfa_service


@inject
def get_email_service(
    email_service=Provide[ApplicationContainer.services.email_service],
):
    """Get email service from dependency injection container.

    Returns:
        EmailService: Email service instance
    """
    return email_service


# ============================
# Authentication Dependencies
# ============================


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    settings: Settings = Depends(get_settings),
) -> User:
    """Get current authenticated user using app-bound container.

    Args:
        request: FastAPI request object (contains app state)
        credentials: HTTP bearer token credentials
        settings: Application settings

    Returns:
        User: Authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # Decode JWT token
        payload = jwt.decode(
            credentials.credentials,
            settings.security.secret_key,
            algorithms=[settings.security.algorithm],
        )

        user_id: str = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Invalid token payload")

    except JWTError as e:
        logger.warning("JWT decode error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    try:
        # Get auth service from container and get user
        container = request.app.state.container
        auth_service = container.services.auth_service()

        user = await auth_service.get_user_by_id(user_id)
        if user is None:
            raise AuthenticationError("User not found")

        return user

    except Exception as e:
        logger.error("Error getting current user", error=str(e), user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        ) from e


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
) -> User | None:
    """Get current user if authenticated, otherwise None.

    Args:
        credentials: HTTP bearer token credentials
        auth_service: Authentication service from DI container
        settings: Application settings

    Returns:
        Optional[User]: Authenticated user or None
    """
    try:
        return await get_current_user(credentials, auth_service, settings)
    except HTTPException:
        return None


def require_permissions(*permissions: str):
    """Dependency factory for permission-based access control using DI.

    Args:
        *permissions: Required permissions

    Returns:
        Dependency function that checks user permissions
    """

    async def permission_dependency(
        current_user: User = Depends(get_current_user),
        auth_service: AuthService = Depends(get_auth_service),
    ) -> User:
        """Check if current user has required permissions."""
        if not await auth_service.user_has_permissions(current_user, permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {', '.join(permissions)}",
            )
        return current_user

    return permission_dependency


def require_roles(*roles: str):
    """Dependency factory for role-based access control using DI.

    Args:
        *roles: Required roles

    Returns:
        Dependency function that checks user roles
    """

    async def role_dependency(
        current_user: User = Depends(get_current_user),
        auth_service: AuthService = Depends(get_auth_service),
    ) -> User:
        """Check if current user has required roles."""
        if not await auth_service.user_has_roles(current_user, roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {', '.join(roles)}",
            )
        return current_user

    return role_dependency


# ============================
# Rate Limiting with DI
# ============================


class RateLimiterDI:
    """Rate limiting dependency using dependency injection."""

    def __init__(self, calls: int, period: int) -> None:
        self.calls = calls
        self.period = period

    async def __call__(self, request: Request) -> None:
        """Check rate limit for the request using app-bound cache service."""
        # Get cache service from app-bound container
        container = request.app.state.container
        cache_service = container.services.cache_service()

        # Use client IP as identifier
        client_ip = request.client.host if request.client else "unknown"
        key = f"rate_limit:{client_ip}"

        current_calls = await cache_service.get_counter(key, self.period)

        if current_calls >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(self.period)},
            )

        await cache_service.increment_counter(key, self.period)


# ============================
# Service Factory with DI
# ============================


@inject
class ServiceFactory:
    """Factory for creating multiple services with DI container.

    Provides a unified interface for accessing multiple services
    while maintaining proper dependency injection patterns.
    """

    def __init__(
        self,
        camera_service=Provide[ApplicationContainer.services.camera_service],
        frame_service=Provide[ApplicationContainer.services.frame_service],
        detection_service=Provide[ApplicationContainer.services.detection_service],
        metrics_service=Provide[ApplicationContainer.services.metrics_service],
        streaming_service=Provide[ApplicationContainer.services.streaming_service],
        analytics_service=Provide[ApplicationContainer.services.analytics_service],
        alert_service=Provide[ApplicationContainer.services.alert_service],
    ):
        self._camera_service = camera_service
        self._frame_service = frame_service
        self._detection_service = detection_service
        self._metrics_service = metrics_service
        self._streaming_service = streaming_service
        self._analytics_service = analytics_service
        self._alert_service = alert_service

    @property
    def camera_service(self) -> CameraService:
        """Get camera service instance."""
        return self._camera_service

    @property
    def frame_service(self) -> FrameService:
        """Get frame service instance."""
        return self._frame_service

    @property
    def detection_service(self) -> DetectionService:
        """Get detection service instance."""
        return self._detection_service

    @property
    def metrics_service(self) -> MetricsService:
        """Get metrics service instance."""
        return self._metrics_service

    @property
    def streaming_service(self):
        """Get streaming service instance."""
        return self._streaming_service

    @property
    def analytics_service(self):
        """Get analytics service instance."""
        return self._analytics_service

    @property
    def alert_service(self):
        """Get alert service instance."""
        return self._alert_service


def get_service_factory() -> ServiceFactory:
    """Get service factory with dependency injection."""
    return ServiceFactory()


# ============================
# Pagination and Utility Dependencies
# ============================


class PaginationParams:
    """Pagination parameters for list endpoints."""

    def __init__(
        self,
        page: int = 1,
        size: int = 20,
        order_by: str = "created_at",
        order_desc: bool = True,
    ):
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page number must be >= 1",
            )
        if size < 1 or size > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page size must be between 1 and 100",
            )

        self.page = page
        self.size = size
        self.order_by = order_by
        self.order_desc = order_desc

    @property
    def offset(self) -> int:
        """Calculate offset from page and size."""
        return (self.page - 1) * self.size


def get_pagination_params(
    page: int = 1,
    size: int = 20,
    order_by: str = "created_at",
    order_desc: bool = True,
) -> PaginationParams:
    """Get pagination parameters dependency."""
    return PaginationParams(page, size, order_by, order_desc)


# ============================
# File Upload with DI
# ============================


async def validate_file_upload(
    file: UploadFile,
    max_size_mb: float = 500.0,
    allowed_extensions: set[str] | None = None,
) -> dict[str, Any]:
    """Validate uploaded file with DI support.

    Args:
        file: Uploaded file
        max_size_mb: Maximum file size in MB
        allowed_extensions: Set of allowed file extensions

    Returns:
        dict: Validation results

    Raises:
        HTTPException: If validation fails
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename",
        )

    # Check file extension
    file_path = Path(file.filename)
    extension = file_path.suffix.lower()

    if allowed_extensions and extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File extension {extension} not allowed. Allowed: {', '.join(allowed_extensions)}",
        )

    # Read and validate file content
    content = await file.read()
    file_size = len(content)

    # Check file size
    max_size_bytes = max_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {file_size / (1024 * 1024):.1f}MB exceeds maximum {max_size_mb}MB",
        )

    # Calculate checksum
    import hashlib

    checksum = hashlib.sha256(content).hexdigest()

    # Reset file position for future reads
    await file.seek(0)

    return {
        "filename": file.filename,
        "size": file_size,
        "extension": extension,
        "content_type": file.content_type,
        "checksum": checksum,
        "validation_passed": True,
    }


# ============================
# Background Task Management with DI
# ============================


@inject
class BackgroundTaskManager:
    """Background task tracking and management using DI."""

    def __init__(
        self,
        cache_service=Provide[ApplicationContainer.services.cache_service],
    ):
        self.cache_service = cache_service

    async def create_task(
        self, task_type: str, task_data: dict[str, Any], user_id: str
    ) -> str:
        """Create a new background task using cache service."""
        task_id = str(uuid4())
        task_info = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "pending",
            "progress": 0.0,
            "user_id": user_id,
            "data": task_data,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "error_message": None,
        }

        await self.cache_service.set(
            f"task:{task_id}", task_info, ttl=3600  # 1 hour TTL
        )

        return task_id

    async def update_task(
        self,
        task_id: str,
        status: str | None = None,
        progress: float | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Update task status using cache service."""
        task_info = await self.cache_service.get(f"task:{task_id}")
        if not task_info:
            return False

        if status:
            task_info["status"] = status
        if progress is not None:
            task_info["progress"] = min(100.0, max(0.0, progress))
        if error_message:
            task_info["error_message"] = error_message

        task_info["updated_at"] = datetime.now().isoformat()

        await self.cache_service.set(f"task:{task_id}", task_info, ttl=3600)
        return True

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get task by ID using cache service."""
        return await self.cache_service.get(f"task:{task_id}")

    async def cleanup_task(self, task_id: str) -> bool:
        """Clean up completed task using cache service."""
        return await self.cache_service.delete(f"task:{task_id}")


def get_background_task_manager() -> BackgroundTaskManager:
    """Get background task manager with dependency injection."""
    return BackgroundTaskManager()


# ============================
# Rate Limiter Instances
# ============================

# Pre-configured rate limiters using DI
rate_limit_strict = RateLimiterDI(calls=10, period=60)  # 10 calls per minute
rate_limit_normal = RateLimiterDI(calls=100, period=60)  # 100 calls per minute
rate_limit_relaxed = RateLimiterDI(calls=1000, period=60)  # 1000 calls per minute
rate_limit_upload = RateLimiterDI(calls=5, period=3600)  # 5 uploads per hour
rate_limit_batch = RateLimiterDI(calls=50, period=60)  # 50 batch operations per minute
