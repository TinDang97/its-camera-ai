"""FastAPI dependencies for dependency injection.

Provides reusable dependency functions for database connections,
authentication, caching, and other shared resources.
"""

from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ..core.config import Settings, get_settings
from ..core.exceptions import AuthenticationError, DatabaseError
from ..core.logging import get_logger
from ..models.database import get_database_session
from ..models.user import User
from ..services.auth import AuthService
from ..services.cache import CacheService


logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)


# Global service instances
_database_session_factory = None
_redis_client: Optional[redis.Redis] = None
_cache_service: Optional[CacheService] = None
_auth_service: Optional[AuthService] = None


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
            raise DatabaseError("Database operation failed", cause=e)
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
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
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
        )
    
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
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
) -> Optional[User]:
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


# Common rate limiters
rate_limit_strict = RateLimiter(calls=10, period=60)  # 10 calls per minute
rate_limit_normal = RateLimiter(calls=100, period=60)  # 100 calls per minute
rate_limit_relaxed = RateLimiter(calls=1000, period=60)  # 1000 calls per minute
