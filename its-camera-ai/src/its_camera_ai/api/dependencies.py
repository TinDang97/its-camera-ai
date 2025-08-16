"""FastAPI dependencies using dependency injection with proper wiring.

This module provides FastAPI-compatible dependencies that use the
dependency injection container system with consistent @inject wiring
for clean architecture and proper resource management.

All dependencies use the @inject decorator with Provide[] pattern for
consistent dependency injection throughout the application.
"""

from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from dependency_injector.wiring import Provide, inject
from fastapi import HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from ..containers import ApplicationContainer
from ..core.config import Settings
from ..core.exceptions import AuthenticationError
from ..core.logging import get_logger
from ..models.database import DatabaseManager
from ..models.user import User
from ..services.auth_service import AuthenticationService as AuthService
from ..services.cache import CacheService
from ..services.camera_service import CameraService
from ..services.frame_service import DetectionService, FrameService
from ..services.metrics_service import MetricsService

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)


# ============================
# Service Protocol Interfaces
# ============================


class RealtimeAnalyticsServiceProtocol(Protocol):
    """Protocol for real-time analytics service."""

    async def get_realtime_analytics(self, camera_id: str) -> Any:
        """Get real-time analytics for a camera."""
        ...


class HistoricalAnalyticsServiceProtocol(Protocol):
    """Protocol for historical analytics service."""

    async def query_historical_data(self, query: Any) -> Any:
        """Query historical analytics data."""
        ...


class IncidentManagementServiceProtocol(Protocol):
    """Protocol for incident management service."""

    async def list_incidents(self, **kwargs: Any) -> Any:
        """List incidents with filtering."""
        ...

    async def get_incident_by_id(self, incident_id: str) -> Any:
        """Get incident by ID."""
        ...


class AnalyticsServiceProtocol(Protocol):
    """Protocol for analytics service."""

    async def get_analytics(self, **kwargs: Any) -> Any:
        """Get analytics data."""
        ...


class StreamingServiceProtocol(Protocol):
    """Protocol for streaming service."""

    async def start_stream(self, **kwargs: Any) -> Any:
        """Start video stream."""
        ...


class PredictionServiceProtocol(Protocol):
    """Protocol for prediction service."""

    async def get_predictions(self, **kwargs: Any) -> Any:
        """Get predictions."""
        ...


class AlertServiceProtocol(Protocol):
    """Protocol for alert service."""

    async def create_alert(self, **kwargs: Any) -> Any:
        """Create an alert."""
        ...


class TokenServiceProtocol(Protocol):
    """Protocol for token service."""

    async def create_token(self, **kwargs: Any) -> Any:
        """Create a token."""
        ...


class MFAServiceProtocol(Protocol):
    """Protocol for MFA service."""

    async def verify_mfa(self, **kwargs: Any) -> Any:
        """Verify MFA token."""
        ...


class EmailServiceProtocol(Protocol):
    """Protocol for email service."""

    async def send_email(self, **kwargs: Any) -> Any:
        """Send an email."""
        ...


# ============================
# Core Infrastructure Dependencies
# ============================


@inject
async def get_database_session(
    database: DatabaseManager = Provide[ApplicationContainer.infrastructure.database],
) -> AsyncGenerator[AsyncSession, None]:
    """Get database session from dependency injection container.

    Args:
        database: Database manager from DI container

    Yields:
        AsyncSession: Database session with proper lifecycle management

    Raises:
        DatabaseError: If session creation fails
    """
    async with database.get_session() as session:
        try:
            yield session
        except Exception as e:
            logger.error("Database session error in DI", error=str(e))
            raise


@inject
async def get_database_manager(
    database: DatabaseManager = Provide[ApplicationContainer.infrastructure.database],
) -> DatabaseManager:
    """Get database manager from dependency injection container.

    Args:
        database: Database manager from DI container

    Returns:
        DatabaseManager: Database manager instance
    """
    return database


@inject
def get_cache_service(
    cache_service: CacheService = Provide[ApplicationContainer.services.cache_service],
) -> CacheService:
    """Get cache service from dependency injection container.

    Args:
        cache_service: Cache service from DI container

    Returns:
        CacheService: Configured cache service instance
    """
    return cache_service


@inject
def get_redis_client(
    redis_client: Any = Provide[ApplicationContainer.infrastructure.redis_client],
) -> Any:
    """Get Redis client from dependency injection container.

    Args:
        redis_client: Redis client from DI container

    Returns:
        Redis client instance
    """
    return redis_client


@inject
def get_settings(
    settings: Settings = Provide[ApplicationContainer.infrastructure.settings],
) -> Settings:
    """Get application settings from dependency injection container.

    Args:
        settings: Settings from DI container

    Returns:
        Settings: Application settings instance
    """
    return settings


# ============================
# Service Dependencies
# ============================


@inject
def get_auth_service(
    auth_service: AuthService = Provide[ApplicationContainer.services.auth_service],
) -> AuthService:
    """Get authentication service from dependency injection container.

    Args:
        auth_service: Authentication service from DI container

    Returns:
        AuthService: Configured authentication service instance
    """
    return auth_service


@inject
def get_camera_service(
    camera_service: CameraService = Provide[
        ApplicationContainer.services.camera_service
    ],
) -> CameraService:
    """Get camera service from dependency injection container.

    Args:
        camera_service: Camera service from DI container

    Returns:
        CameraService: Camera service instance
    """
    return camera_service


@inject
def get_frame_service(
    frame_service: FrameService = Provide[ApplicationContainer.services.frame_service],
) -> FrameService:
    """Get frame service from dependency injection container.

    Args:
        frame_service: Frame service from DI container

    Returns:
        FrameService: Frame service instance
    """
    return frame_service


@inject
def get_detection_service(
    detection_service: DetectionService = Provide[
        ApplicationContainer.services.detection_service
    ],
) -> DetectionService:
    """Get detection service from dependency injection container.

    Args:
        detection_service: Detection service from DI container

    Returns:
        DetectionService: Detection service instance
    """
    return detection_service


@inject
def get_metrics_service(
    metrics_service: MetricsService = Provide[
        ApplicationContainer.services.metrics_service
    ],
) -> MetricsService:
    """Get metrics service from dependency injection container.

    Args:
        metrics_service: Metrics service from DI container

    Returns:
        MetricsService: Metrics service instance
    """
    return metrics_service


@inject
def get_streaming_service(
    streaming_service: StreamingServiceProtocol = Provide[ApplicationContainer.services.streaming_service],
) -> StreamingServiceProtocol:
    """Get streaming service from dependency injection container.

    Args:
        streaming_service: Streaming service from DI container

    Returns:
        StreamingServiceProtocol: Streaming service instance
    """
    return streaming_service


@inject
def get_analytics_service(
    analytics_service: AnalyticsServiceProtocol = Provide[ApplicationContainer.services.analytics_service],
) -> AnalyticsServiceProtocol:
    """Get analytics service from dependency injection container.

    Args:
        analytics_service: Analytics service from DI container

    Returns:
        AnalyticsServiceProtocol: Analytics service instance
    """
    return analytics_service


@inject
def get_realtime_analytics_service(
    realtime_analytics_service: RealtimeAnalyticsServiceProtocol = Provide[ApplicationContainer.services.realtime_analytics_service],
) -> RealtimeAnalyticsServiceProtocol:
    """Get real-time analytics service from dependency injection container.

    Args:
        realtime_analytics_service: Real-time analytics service from DI container

    Returns:
        RealtimeAnalyticsServiceProtocol: Real-time analytics service instance
    """
    return realtime_analytics_service


@inject
def get_historical_analytics_service(
    historical_analytics_service: HistoricalAnalyticsServiceProtocol = Provide[ApplicationContainer.services.historical_analytics_service],
) -> HistoricalAnalyticsServiceProtocol:
    """Get historical analytics service from dependency injection container.

    Args:
        historical_analytics_service: Historical analytics service from DI container

    Returns:
        HistoricalAnalyticsServiceProtocol: Historical analytics service instance
    """
    return historical_analytics_service


@inject
def get_incident_management_service(
    incident_management_service: IncidentManagementServiceProtocol = Provide[ApplicationContainer.services.incident_management_service],
) -> IncidentManagementServiceProtocol:
    """Get incident management service from dependency injection container.

    Args:
        incident_management_service: Incident management service from DI container

    Returns:
        IncidentManagementServiceProtocol: Incident management service instance
    """
    return incident_management_service


@inject
def get_prediction_service(
    prediction_service: PredictionServiceProtocol = Provide[ApplicationContainer.services.prediction_service],
) -> PredictionServiceProtocol:
    """Get prediction service from dependency injection container.

    Args:
        prediction_service: Prediction service from DI container

    Returns:
        PredictionServiceProtocol: Prediction service instance
    """
    return prediction_service


@inject
def get_alert_service(
    alert_service: AlertServiceProtocol = Provide[ApplicationContainer.services.alert_service],
) -> AlertServiceProtocol:
    """Get alert service from dependency injection container.

    Args:
        alert_service: Alert service from DI container

    Returns:
        AlertServiceProtocol: Alert service instance
    """
    return alert_service


@inject
def get_token_service(
    token_service: TokenServiceProtocol = Provide[ApplicationContainer.services.token_service],
) -> TokenServiceProtocol:
    """Get token service from dependency injection container.

    Args:
        token_service: Token service from DI container

    Returns:
        TokenServiceProtocol: Token service instance
    """
    return token_service


@inject
def get_mfa_service(
    mfa_service: MFAServiceProtocol = Provide[ApplicationContainer.services.mfa_service],
) -> MFAServiceProtocol:
    """Get MFA service from dependency injection container.

    Args:
        mfa_service: MFA service from DI container

    Returns:
        MFAServiceProtocol: MFA service instance
    """
    return mfa_service


@inject
def get_email_service(
    email_service: EmailServiceProtocol = Provide[ApplicationContainer.services.email_service],
) -> EmailServiceProtocol:
    """Get email service from dependency injection container.

    Args:
        email_service: Email service from DI container

    Returns:
        EmailServiceProtocol: Email service instance
    """
    return email_service


# ============================
# Authentication Dependencies
# ============================


@inject
async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None,
    auth_service: AuthService = Provide[ApplicationContainer.services.auth_service],
    settings: Settings = Provide[ApplicationContainer.infrastructure.settings],
) -> User:
    """Get current authenticated user using dependency injection.

    Args:
        credentials: HTTP bearer token credentials
        auth_service: Authentication service from DI container
        settings: Application settings from DI container

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

        user_id = payload.get("sub")
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
        # Get user from auth service using internal method
        user = await auth_service._get_user_with_roles(str(user_id))
        if user is None:
            raise AuthenticationError("User not found")

        return user

    except Exception as e:
        logger.error("Error getting current user", error=str(e), user_id=user_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        ) from e


async def get_current_user_dependency(
    credentials: HTTPAuthorizationCredentials | None = None,
) -> User:
    """FastAPI dependency wrapper for get_current_user.

    Args:
        credentials: HTTP bearer token credentials

    Returns:
        User: Authenticated user
    """
    return await get_current_user(credentials)


@inject
async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None,
    auth_service: AuthService = Provide[ApplicationContainer.services.auth_service],
    settings: Settings = Provide[ApplicationContainer.infrastructure.settings],
) -> User | None:
    """Get current user if authenticated, otherwise None.

    Args:
        credentials: HTTP bearer token credentials
        auth_service: Authentication service from DI container
        settings: Application settings from DI container

    Returns:
        Optional[User]: Authenticated user or None
    """
    try:
        return await get_current_user(credentials, auth_service, settings)
    except HTTPException:
        return None


async def get_optional_user_dependency(
    credentials: HTTPAuthorizationCredentials | None = None,
) -> User | None:
    """FastAPI dependency wrapper for get_optional_user.

    Args:
        credentials: HTTP bearer token credentials

    Returns:
        Optional[User]: Authenticated user or None
    """
    return await get_optional_user(credentials)


def require_permissions(*permissions: str) -> Any:
    """Dependency factory for permission-based access control using DI.

    Args:
        *permissions: Required permissions

    Returns:
        Dependency function that checks user permissions
    """

    @inject
    async def permission_dependency(
        current_user: User,
        auth_service: AuthService = Provide[ApplicationContainer.services.auth_service],
    ) -> User:
        """Check if current user has required permissions."""
        user_permissions = await auth_service._get_user_permissions(current_user)
        if not any(perm in user_permissions for perm in permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {', '.join(permissions)}",
            )
        return current_user

    return permission_dependency


def require_roles(*roles: str) -> Any:
    """Dependency factory for role-based access control using DI.

    Args:
        *roles: Required roles

    Returns:
        Dependency function that checks user roles
    """

    @inject
    async def role_dependency(
        current_user: User,
        auth_service: AuthService = Provide[
            ApplicationContainer.services.auth_service
        ],  # noqa: ARG001
    ) -> User:
        """Check if current user has required roles."""
        user_roles = [role.name for role in current_user.roles]
        if not any(role in user_roles for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {', '.join(roles)}",
            )
        return current_user

    return role_dependency


# ============================
# File Upload with DI
# ============================


async def validate_file_upload(
    file: UploadFile,
    max_size_mb: float = 500.0,
    allowed_extensions: set[str] | None = None,
) -> dict[str, Any]:
    """Validate uploaded file with comprehensive checks.

    Args:
        file: Uploaded file
        max_size_mb: Maximum file size in MB
        allowed_extensions: Set of allowed file extensions

    Returns:
        dict: Validation results with file metadata

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
        cache_service: CacheService = Provide[
            ApplicationContainer.services.cache_service
        ],
    ):
        self.cache_service = cache_service

    async def create_task(
        self, task_type: str, task_data: dict[str, Any], user_id: str
    ) -> str:
        """Create a new background task using cache service.

        Args:
            task_type: Type of task to create
            task_data: Task-specific data
            user_id: ID of user creating the task

        Returns:
            str: Task ID
        """
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

        await self.cache_service.set_json(
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
        """Update task status using cache service.

        Args:
            task_id: Task ID to update
            status: New task status
            progress: Task progress percentage
            error_message: Error message if task failed

        Returns:
            bool: True if task was updated, False if not found
        """
        task_info = await self.cache_service.get_json(f"task:{task_id}")
        if not task_info:
            return False

        if status:
            task_info["status"] = status
        if progress is not None:
            task_info["progress"] = min(100.0, max(0.0, progress))
        if error_message:
            task_info["error_message"] = error_message

        task_info["updated_at"] = datetime.now().isoformat()

        await self.cache_service.set_json(f"task:{task_id}", task_info, ttl=3600)
        return True

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get task by ID using cache service.

        Args:
            task_id: Task ID to retrieve

        Returns:
            Task information or None if not found
        """
        return await self.cache_service.get_json(f"task:{task_id}")

    async def cleanup_task(self, task_id: str) -> bool:
        """Clean up completed task using cache service.

        Args:
            task_id: Task ID to clean up

        Returns:
            bool: True if task was deleted, False if not found
        """
        return await self.cache_service.delete(f"task:{task_id}")


@inject
def get_background_task_manager(
    cache_service: CacheService = Provide[ApplicationContainer.services.cache_service],
) -> BackgroundTaskManager:
    """Get background task manager with dependency injection.

    Args:
        cache_service: Cache service from DI container

    Returns:
        BackgroundTaskManager: Task manager instance
    """
    return BackgroundTaskManager(cache_service=cache_service)
