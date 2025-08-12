"""FastAPI dependencies for dependency injection.

Provides reusable dependency functions for database connections,
authentication, caching, and other shared resources.
"""

import hashlib
import tempfile
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import redis.asyncio as redis
from fastapi import Depends, HTTPException, Request, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import Settings, get_settings
from ..core.exceptions import AuthenticationError, DatabaseError
from ..core.logging import get_logger
from ..models.user import User
from ..services.auth import AuthService
from ..services.cache import CacheService
from ..services.camera_service import CameraService
from ..services.frame_service import DetectionService, FrameService
from ..services.metrics_service import MetricsService

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)


# Global service instances
_database_session_factory = None
_redis_client: redis.Redis | None = None
_cache_service: CacheService | None = None
_auth_service: AuthService | None = None


async def setup_dependencies(settings: Settings) -> None:
    """Initialize global dependencies.

    Args:
        settings: Application settings
    """
    global _database_session_factory, _redis_client, _cache_service, _auth_service

    try:
        # Setup database
        from ..models.database import create_database_engine

        engine = await create_database_engine(settings)
        _database_session_factory = engine.session_factory

        # Setup Redis
        _redis_client = redis.from_url(
            settings.redis.url,
            max_connections=settings.redis.max_connections,
            socket_timeout=settings.redis.timeout,
            retry_on_timeout=settings.redis.retry_on_timeout,
            decode_responses=True,
        )

        # Test Redis connection
        await _redis_client.ping()

        # Setup services
        _cache_service = CacheService(_redis_client)
        _auth_service = AuthService(settings.security)

        logger.info("Dependencies initialized successfully")

    except Exception as e:
        logger.error("Failed to initialize dependencies", error=str(e))
        raise


async def cleanup_dependencies() -> None:
    """Clean up global dependencies."""
    global _redis_client, _cache_service, _auth_service

    try:
        if _redis_client:
            await _redis_client.close()
            _redis_client = None

        _cache_service = None
        _auth_service = None

        logger.info("Dependencies cleaned up successfully")

    except Exception as e:
        logger.error("Error during dependency cleanup", error=str(e))


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency.

    Yields:
        AsyncSession: Database session

    Raises:
        DatabaseError: If database session creation fails
    """
    if _database_session_factory is None:
        raise DatabaseError("Database not initialized")

    async with _database_session_factory() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise DatabaseError("Database operation failed", cause=e) from e
        finally:
            await session.close()


async def get_redis() -> redis.Redis:
    """Get Redis client dependency.

    Returns:
        redis.Redis: Redis client instance

    Raises:
        RuntimeError: If Redis client is not initialized
    """
    if _redis_client is None:
        raise RuntimeError("Redis client not initialized")
    return _redis_client


async def get_cache_service() -> CacheService:
    """Get cache service dependency.

    Returns:
        CacheService: Cache service instance

    Raises:
        RuntimeError: If cache service is not initialized
    """
    if _cache_service is None:
        raise RuntimeError("Cache service not initialized")
    return _cache_service


async def get_auth_service() -> AuthService:
    """Get authentication service dependency.

    Returns:
        AuthService: Authentication service instance

    Raises:
        RuntimeError: If auth service is not initialized
    """
    if _auth_service is None:
        raise RuntimeError("Auth service not initialized")
    return _auth_service


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: AsyncSession = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
) -> User:
    """Get current authenticated user dependency.

    Args:
        credentials: HTTP bearer token credentials
        db: Database session
        auth_service: Authentication service
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
        # Get user from database
        user = await auth_service.get_user_by_id(db, user_id)
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
    db: AsyncSession = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
) -> User | None:
    """Get current user if authenticated, otherwise None.

    Args:
        credentials: HTTP bearer token credentials
        db: Database session
        auth_service: Authentication service
        settings: Application settings

    Returns:
        Optional[User]: Authenticated user or None
    """
    try:
        return await get_current_user(credentials, db, auth_service, settings)
    except HTTPException:
        return None


def require_permissions(*permissions: str):
    """Dependency factory for permission-based access control.

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
    """Dependency factory for role-based access control.

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


class RateLimiter:
    """Rate limiting dependency."""

    def __init__(self, calls: int, period: int) -> None:
        self.calls = calls
        self.period = period

    async def __call__(
        self,
        request: Request,
        cache_service: CacheService = Depends(get_cache_service),
    ) -> None:
        """Check rate limit for the request."""
        # Use client IP as identifier
        client_ip = request.client.host
        key = f"rate_limit:{client_ip}"

        current_calls = await cache_service.get_counter(key, self.period)

        if current_calls >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(self.period)},
            )

        await cache_service.increment_counter(key, self.period)


# Global upload tracking
_upload_sessions: dict[str, dict[str, Any]] = {}


class FileUploadManager:
    """File upload session manager."""

    def __init__(self):
        self.sessions = _upload_sessions

    def create_upload_session(self, user_id: str, model_name: str) -> str:
        """Create a new upload session.

        Args:
            user_id: User creating the upload
            model_name: Name of the model being uploaded

        Returns:
            str: Upload session ID
        """
        upload_id = str(uuid4())
        self.sessions[upload_id] = {
            "upload_id": upload_id,
            "user_id": user_id,
            "model_name": model_name,
            "status": "initialized",
            "progress": 0.0,
            "files": {},
            "validation_results": {},
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        return upload_id

    def get_upload_session(self, upload_id: str) -> dict[str, Any] | None:
        """Get upload session by ID.

        Args:
            upload_id: Upload session ID

        Returns:
            dict: Upload session data or None if not found
        """
        return self.sessions.get(upload_id)

    def update_upload_progress(
        self, upload_id: str, progress: float, status: str | None = None
    ) -> bool:
        """Update upload progress.

        Args:
            upload_id: Upload session ID
            progress: Progress percentage (0-100)
            status: Optional status update

        Returns:
            bool: True if update successful
        """
        session = self.sessions.get(upload_id)
        if not session:
            return False

        session["progress"] = min(100.0, max(0.0, progress))
        if status:
            session["status"] = status
        session["updated_at"] = datetime.now()
        return True

    def add_file_to_session(
        self, upload_id: str, file_type: str, file_info: dict[str, Any]
    ) -> bool:
        """Add file information to upload session.

        Args:
            upload_id: Upload session ID
            file_type: Type of file (model, config, requirements)
            file_info: File information dictionary

        Returns:
            bool: True if successful
        """
        session = self.sessions.get(upload_id)
        if not session:
            return False

        session["files"][file_type] = file_info
        session["updated_at"] = datetime.now()
        return True

    def cleanup_session(self, upload_id: str) -> bool:
        """Clean up upload session.

        Args:
            upload_id: Upload session ID

        Returns:
            bool: True if cleanup successful
        """
        return self.sessions.pop(upload_id, None) is not None


# Global file upload manager
_upload_manager: FileUploadManager | None = None


def get_upload_manager() -> FileUploadManager:
    """Get file upload manager dependency.

    Returns:
        FileUploadManager: Upload manager instance
    """
    global _upload_manager
    if _upload_manager is None:
        _upload_manager = FileUploadManager()
    return _upload_manager


async def validate_file_upload(
    file: UploadFile,
    max_size_mb: float = 500.0,
    allowed_extensions: set[str] | None = None,
) -> dict[str, Any]:
    """Validate uploaded file.

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


async def create_temp_file(upload_file: UploadFile, prefix: str = "upload_") -> str:
    """Create temporary file from upload.

    Args:
        upload_file: Uploaded file
        prefix: Filename prefix

    Returns:
        str: Path to temporary file

    Raises:
        HTTPException: If file creation fails
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, prefix=prefix) as temp_file:
            content = await upload_file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Reset upload file position
        await upload_file.seek(0)

        return temp_path

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create temporary file: {str(e)}",
        ) from e


class BackgroundTaskManager:
    """Background task tracking and management."""

    def __init__(self):
        self.tasks: dict[str, dict[str, Any]] = {}

    def create_task(
        self, task_type: str, task_data: dict[str, Any], user_id: str
    ) -> str:
        """Create a new background task.

        Args:
            task_type: Type of task (upload_processing, model_validation, etc.)
            task_data: Task-specific data
            user_id: User who initiated the task

        Returns:
            str: Task ID
        """
        task_id = str(uuid4())
        self.tasks[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "pending",
            "progress": 0.0,
            "user_id": user_id,
            "data": task_data,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "error_message": None,
        }
        return task_id

    def update_task(
        self,
        task_id: str,
        status: str | None = None,
        progress: float | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Update task status.

        Args:
            task_id: Task ID
            status: New status
            progress: Progress percentage
            error_message: Error message if failed

        Returns:
            bool: True if update successful
        """
        task = self.tasks.get(task_id)
        if not task:
            return False

        if status:
            task["status"] = status
        if progress is not None:
            task["progress"] = min(100.0, max(0.0, progress))
        if error_message:
            task["error_message"] = error_message

        task["updated_at"] = datetime.now()
        return True

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """Get task by ID.

        Args:
            task_id: Task ID

        Returns:
            dict: Task data or None if not found
        """
        return self.tasks.get(task_id)

    def cleanup_task(self, task_id: str) -> bool:
        """Clean up completed task.

        Args:
            task_id: Task ID

        Returns:
            bool: True if cleanup successful
        """
        return self.tasks.pop(task_id, None) is not None


# Global background task manager
_task_manager: BackgroundTaskManager | None = None


def get_task_manager() -> BackgroundTaskManager:
    """Get background task manager dependency.

    Returns:
        BackgroundTaskManager: Task manager instance
    """
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager


# Database Service Dependencies
async def get_camera_service(db: AsyncSession = Depends(get_db)) -> CameraService:
    """Get camera service dependency."""
    return CameraService(db)


async def get_frame_service(db: AsyncSession = Depends(get_db)) -> FrameService:
    """Get frame service dependency."""
    return FrameService(db)


async def get_detection_service(db: AsyncSession = Depends(get_db)) -> DetectionService:
    """Get detection service dependency."""
    return DetectionService(db)


async def get_metrics_service(db: AsyncSession = Depends(get_db)) -> MetricsService:
    """Get metrics service dependency."""
    return MetricsService(db)


# Service Factory for complex operations
class ServiceFactory:
    """Factory for creating multiple services with shared session."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self._camera_service: CameraService | None = None
        self._frame_service: FrameService | None = None
        self._detection_service: DetectionService | None = None
        self._metrics_service: MetricsService | None = None

    @property
    def camera_service(self) -> CameraService:
        """Get camera service instance."""
        if self._camera_service is None:
            self._camera_service = CameraService(self.session)
        return self._camera_service

    @property
    def frame_service(self) -> FrameService:
        """Get frame service instance."""
        if self._frame_service is None:
            self._frame_service = FrameService(self.session)
        return self._frame_service

    @property
    def detection_service(self) -> DetectionService:
        """Get detection service instance."""
        if self._detection_service is None:
            self._detection_service = DetectionService(self.session)
        return self._detection_service

    @property
    def metrics_service(self) -> MetricsService:
        """Get metrics service instance."""
        if self._metrics_service is None:
            self._metrics_service = MetricsService(self.session)
        return self._metrics_service


async def get_service_factory(db: AsyncSession = Depends(get_db)) -> ServiceFactory:
    """Get service factory dependency for complex operations."""
    return ServiceFactory(db)


# Pagination helpers
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
                detail="Page number must be >= 1"
            )
        if size < 1 or size > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page size must be between 1 and 100"
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


# Error handling helpers
def handle_database_error(error: Exception) -> HTTPException:
    """Convert database errors to HTTP exceptions."""
    if isinstance(error, DatabaseError):
        logger.error("Database operation error", error=str(error))
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database operation failed"
        )
    else:
        logger.error("Unexpected service error", error=str(error))
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Common rate limiters
rate_limit_strict = RateLimiter(calls=10, period=60)  # 10 calls per minute
rate_limit_normal = RateLimiter(calls=100, period=60)  # 100 calls per minute
rate_limit_relaxed = RateLimiter(calls=1000, period=60)  # 1000 calls per minute
rate_limit_upload = RateLimiter(calls=5, period=3600)  # 5 uploads per hour
rate_limit_batch = RateLimiter(calls=50, period=60)  # 50 batch operations per minute
