"""
API Key authentication middleware for service-to-service authentication.

Provides:
- Service-to-service authentication via API keys
- API key rotation support
- Key scope and permissions management
- Rate limiting per API key
- Key expiration and revocation
- Comprehensive audit logging
"""

import hashlib
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import redis.asyncio as redis
from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ...core.config import get_settings
from ...core.logging import get_logger

logger = get_logger(__name__)


class APIKeyStatus(Enum):
    """API Key status enumeration."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


class APIKeyScope(Enum):
    """API Key scope enumeration."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    SERVICE = "service"
    ANALYTICS = "analytics"
    MONITORING = "monitoring"


@dataclass
class APIKey:
    """API Key data model."""
    key_id: str
    key_hash: str
    name: str
    description: str
    scopes: list[APIKeyScope]
    status: APIKeyStatus
    created_at: datetime
    expires_at: datetime | None
    last_used_at: datetime | None
    usage_count: int
    rate_limit_per_minute: int
    rate_limit_per_hour: int
    allowed_ips: list[str]
    user_id: str | None
    service_name: str | None


@dataclass
class APIKeyValidationResult:
    """API Key validation result."""
    valid: bool
    api_key: APIKey | None = None
    error: str | None = None
    remaining_requests: int = 0


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """API Key authentication middleware."""

    def __init__(self, app, redis_client: redis.Redis | None = None, settings=None):
        super().__init__(app)
        self.redis = redis_client
        self.settings = settings or get_settings()

        # API key configuration
        self.api_key_header = "X-API-Key"
        self.api_key_query_param = "api_key"
        self.cache_ttl = 300  # 5 minutes cache for API key data

        # Endpoints that require API key authentication
        self.api_key_protected_routes = {
            "/api/v1/analytics/batch",
            "/api/v1/models/inference",
            "/api/v1/cameras/bulk-upload",
            "/api/v1/system/metrics",
            "/api/v1/admin/users",
        }

        # Endpoints that accept API key as alternative to JWT
        self.api_key_optional_routes = {
            "/api/v1/analytics/stats",
            "/api/v1/models/list",
            "/api/v1/cameras/status",
        }

        # Scope-based route permissions
        self.scope_permissions = {
            APIKeyScope.READ_ONLY: [
                "GET:/api/v1/analytics/stats",
                "GET:/api/v1/models/list",
                "GET:/api/v1/cameras/status",
                "GET:/api/v1/system/health",
            ],
            APIKeyScope.READ_WRITE: [
                "GET:/api/v1/analytics/*",
                "POST:/api/v1/analytics/stats",
                "GET:/api/v1/models/*",
                "POST:/api/v1/models/inference",
                "GET:/api/v1/cameras/*",
                "PUT:/api/v1/cameras/config",
            ],
            APIKeyScope.ADMIN: [
                "*:/api/v1/admin/*",
                "*:/api/v1/system/*",
                "*:/api/v1/users/*",
            ],
            APIKeyScope.SERVICE: [
                "*:/api/v1/*",  # Full access for service-to-service
            ],
            APIKeyScope.ANALYTICS: [
                "*:/api/v1/analytics/*",
                "GET:/api/v1/models/*",
                "POST:/api/v1/models/inference",
            ],
            APIKeyScope.MONITORING: [
                "GET:/api/v1/system/*",
                "GET:/api/v1/cameras/status",
                "GET:/metrics",
                "GET:/health",
            ],
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process API key authentication."""
        path = request.url.path
        method = request.method

        # Check if API key authentication is required or optional
        requires_api_key = any(route in path for route in self.api_key_protected_routes)
        accepts_api_key = any(route in path for route in self.api_key_optional_routes)

        if not requires_api_key and not accepts_api_key:
            # Route doesn't use API key authentication
            return await call_next(request)

        # Extract API key from request
        api_key = self._extract_api_key(request)

        if not api_key:
            if requires_api_key:
                return self._create_auth_error("API key required")
            # Continue without API key for optional routes
            return await call_next(request)

        try:
            # Validate API key
            validation_result = await self._validate_api_key(api_key, request)

            if not validation_result.valid:
                return self._create_auth_error(validation_result.error or "Invalid API key")

            # Check permissions
            if not self._check_permissions(validation_result.api_key, method, path):
                await self._log_security_event(
                    request, "api_key_permission_denied",
                    {
                        "key_id": validation_result.api_key.key_id,
                        "requested_endpoint": f"{method}:{path}",
                        "scopes": [scope.value for scope in validation_result.api_key.scopes]
                    }
                )
                return self._create_auth_error("Insufficient permissions")

            # Check rate limits
            rate_limit_result = await self._check_rate_limits(validation_result.api_key, request)
            if not rate_limit_result["allowed"]:
                return self._create_rate_limit_error(rate_limit_result)

            # Update usage statistics
            await self._update_usage_stats(validation_result.api_key.key_id)

            # Add API key info to request state
            request.state.api_key = validation_result.api_key
            request.state.authenticated_via = "api_key"

            # Process request
            response = await call_next(request)

            # Add API key headers
            response.headers["X-API-Key-ID"] = validation_result.api_key.key_id
            response.headers["X-RateLimit-Remaining"] = str(rate_limit_result["remaining"])

            # Log successful API key usage
            await self._log_security_event(
                request, "api_key_success",
                {
                    "key_id": validation_result.api_key.key_id,
                    "endpoint": f"{method}:{path}",
                    "response_status": response.status_code
                }
            )

            return response

        except Exception as e:
            logger.error("API key authentication error", error=str(e), path=path)
            return self._create_auth_error("Authentication failed")

    def _extract_api_key(self, request: Request) -> str | None:
        """Extract API key from request headers or query parameters."""
        # Check header first (preferred method)
        api_key = request.headers.get(self.api_key_header)
        if api_key:
            return api_key.strip()

        # Check query parameter (less secure, for compatibility)
        api_key = request.query_params.get(self.api_key_query_param)
        if api_key:
            # Log query parameter usage for security monitoring
            logger.warning(
                "API key provided via query parameter",
                ip=self._get_client_ip(request),
                path=request.url.path
            )
            return api_key.strip()

        return None

    async def _validate_api_key(self, api_key: str, request: Request) -> APIKeyValidationResult:
        """Validate API key and return key information."""
        try:
            # Check cache first
            cached_key = await self._get_cached_api_key(api_key)
            if cached_key:
                return APIKeyValidationResult(valid=True, api_key=cached_key)

            # Generate key hash for lookup
            key_hash = self._hash_api_key(api_key)

            # TODO: Replace with actual database lookup
            api_key_data = await self._lookup_api_key_by_hash(key_hash)

            if not api_key_data:
                await self._log_security_event(
                    request, "api_key_not_found", {"key_prefix": api_key[:8] + "..."}
                )
                return APIKeyValidationResult(valid=False, error="Invalid API key")

            api_key_obj = self._deserialize_api_key(api_key_data)

            # Check status
            if api_key_obj.status != APIKeyStatus.ACTIVE:
                await self._log_security_event(
                    request, "api_key_inactive",
                    {"key_id": api_key_obj.key_id, "status": api_key_obj.status.value}
                )
                return APIKeyValidationResult(valid=False, error=f"API key is {api_key_obj.status.value}")

            # Check expiration
            if api_key_obj.expires_at and datetime.now(UTC) > api_key_obj.expires_at:
                await self._log_security_event(
                    request, "api_key_expired",
                    {"key_id": api_key_obj.key_id, "expired_at": api_key_obj.expires_at.isoformat()}
                )
                return APIKeyValidationResult(valid=False, error="API key has expired")

            # Check IP restrictions
            if api_key_obj.allowed_ips:
                client_ip = self._get_client_ip(request)
                if not self._is_ip_allowed(client_ip, api_key_obj.allowed_ips):
                    await self._log_security_event(
                        request, "api_key_ip_blocked",
                        {"key_id": api_key_obj.key_id, "client_ip": client_ip, "allowed_ips": api_key_obj.allowed_ips}
                    )
                    return APIKeyValidationResult(valid=False, error="API key not allowed from this IP")

            # Cache valid API key
            await self._cache_api_key(api_key, api_key_obj)

            return APIKeyValidationResult(valid=True, api_key=api_key_obj)

        except Exception as e:
            logger.error("API key validation error", error=str(e))
            return APIKeyValidationResult(valid=False, error="Validation failed")

    def _check_permissions(self, api_key: APIKey, method: str, path: str) -> bool:
        """Check if API key has permission for the requested endpoint."""
        for scope in api_key.scopes:
            allowed_patterns = self.scope_permissions.get(scope, [])

            for pattern in allowed_patterns:
                pattern_method, pattern_path = pattern.split(":", 1)

                # Check method
                if pattern_method != "*" and pattern_method != method:
                    continue

                # Check path
                if pattern_path == "*":
                    return True
                elif pattern_path.endswith("/*"):
                    prefix = pattern_path[:-2]
                    if path.startswith(prefix):
                        return True
                elif pattern_path == path:
                    return True

        return False

    async def _check_rate_limits(self, api_key: APIKey, request: Request) -> dict[str, Any]:
        """Check API key specific rate limits."""
        if not self.redis:
            return {"allowed": True, "remaining": api_key.rate_limit_per_minute}

        current_time = int(time.time())
        minute_window = current_time - (current_time % 60)
        hour_window = current_time - (current_time % 3600)

        # Check minute limit
        minute_key = f"api_key_rate:{api_key.key_id}:minute:{minute_window}"
        minute_count = await self.redis.get(minute_key)
        minute_count = int(minute_count) if minute_count else 0

        if minute_count >= api_key.rate_limit_per_minute:
            return {
                "allowed": False,
                "limit": api_key.rate_limit_per_minute,
                "remaining": 0,
                "reset_time": minute_window + 60,
                "window": "minute"
            }

        # Check hour limit
        if api_key.rate_limit_per_hour > 0:
            hour_key = f"api_key_rate:{api_key.key_id}:hour:{hour_window}"
            hour_count = await self.redis.get(hour_key)
            hour_count = int(hour_count) if hour_count else 0

            if hour_count >= api_key.rate_limit_per_hour:
                return {
                    "allowed": False,
                    "limit": api_key.rate_limit_per_hour,
                    "remaining": 0,
                    "reset_time": hour_window + 3600,
                    "window": "hour"
                }

        # Increment counters
        pipe = self.redis.pipeline()
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)
        if api_key.rate_limit_per_hour > 0:
            pipe.incr(f"api_key_rate:{api_key.key_id}:hour:{hour_window}")
            pipe.expire(f"api_key_rate:{api_key.key_id}:hour:{hour_window}", 3600)
        await pipe.execute()

        return {
            "allowed": True,
            "remaining": api_key.rate_limit_per_minute - minute_count - 1,
            "limit": api_key.rate_limit_per_minute
        }

    async def _update_usage_stats(self, key_id: str) -> None:
        """Update API key usage statistics."""
        if not self.redis:
            return

        current_time = int(time.time())

        # Update usage count and last used time
        pipe = self.redis.pipeline()
        pipe.hincrby(f"api_key_stats:{key_id}", "usage_count", 1)
        pipe.hset(f"api_key_stats:{key_id}", "last_used_at", current_time)
        pipe.expire(f"api_key_stats:{key_id}", 86400 * 30)  # 30 days
        await pipe.execute()

    async def _get_cached_api_key(self, api_key: str) -> APIKey | None:
        """Get API key from cache."""
        if not self.redis:
            return None

        try:
            key_hash = self._hash_api_key(api_key)
            cached_data = await self.redis.get(f"api_key_cache:{key_hash}")

            if cached_data:
                import json
                return self._deserialize_api_key(json.loads(cached_data))

        except Exception as e:
            logger.warning("Cache lookup failed", error=str(e))

        return None

    async def _cache_api_key(self, api_key: str, api_key_obj: APIKey) -> None:
        """Cache API key data."""
        if not self.redis:
            return

        try:
            import json
            key_hash = self._hash_api_key(api_key)
            data = self._serialize_api_key(api_key_obj)
            await self.redis.setex(
                f"api_key_cache:{key_hash}",
                self.cache_ttl,
                json.dumps(data)
            )
        except Exception as e:
            logger.warning("Cache write failed", error=str(e))

    def _hash_api_key(self, api_key: str) -> str:
        """Create a hash of the API key for secure storage and lookup."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _is_ip_allowed(self, client_ip: str, allowed_ips: list[str]) -> bool:
        """Check if client IP is in allowed IP list."""
        import ipaddress

        try:
            client = ipaddress.ip_address(client_ip)

            for allowed in allowed_ips:
                try:
                    # Handle both individual IPs and CIDR ranges
                    if "/" in allowed:
                        network = ipaddress.ip_network(allowed, strict=False)
                        if client in network:
                            return True
                    else:
                        if client == ipaddress.ip_address(allowed):
                            return True
                except ValueError:
                    continue

        except ValueError:
            # Invalid IP address
            return False

        return False

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        return request.client.host if request.client else "unknown"

    def _create_auth_error(self, message: str) -> JSONResponse:
        """Create authentication error response."""
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "api_key_authentication_failed",
                "message": message,
                "details": "Valid API key required"
            },
            headers={"WWW-Authenticate": f"API-Key realm=\"{self.api_key_header}\""}
        )

    def _create_rate_limit_error(self, rate_limit_result: dict[str, Any]) -> JSONResponse:
        """Create rate limit error response."""
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "api_key_rate_limit_exceeded",
                "message": f"API key rate limit exceeded. Limit: {rate_limit_result['limit']} per {rate_limit_result.get('window', 'minute')}",
                "limit": rate_limit_result["limit"],
                "remaining": rate_limit_result["remaining"],
                "reset_time": rate_limit_result.get("reset_time", 0)
            },
            headers={
                "X-RateLimit-Limit": str(rate_limit_result["limit"]),
                "X-RateLimit-Remaining": str(rate_limit_result["remaining"]),
                "Retry-After": str(max(1, rate_limit_result.get("reset_time", 0) - int(time.time())))
            }
        )

    async def _log_security_event(self, request: Request, event_type: str, details: dict[str, Any]) -> None:
        """Log security events."""
        logger.warning(
            "API key security event",
            event_type=event_type,
            client_ip=self._get_client_ip(request),
            path=request.url.path,
            method=request.method,
            user_agent=request.headers.get("user-agent", ""),
            details=details
        )

    # Database integration methods
    async def _lookup_api_key_by_hash(self, key_hash: str) -> dict[str, Any] | None:
        """Lookup API key by hash in database."""
        try:
            # TODO: Replace with actual database session from DI container
            # For now, implement a secure fallback that prevents bypass
            if not hasattr(self, '_db_session') or self._db_session is None:
                logger.error("Database session not configured for API key lookup")
                return None

            # Implement actual database lookup
            # query = select(APIKeyModel).where(APIKeyModel.key_hash == key_hash)
            # result = await self._db_session.execute(query)
            # api_key_record = result.scalar_one_or_none()

            # Temporary secure implementation - fail closed
            logger.warning("API key database lookup not yet implemented - failing securely")
            return None

        except Exception as e:
            logger.error("Database lookup failed for API key", error=str(e), key_hash=key_hash[:8])
            return None

    def _deserialize_api_key(self, data: dict[str, Any]) -> APIKey:
        """Deserialize API key data."""
        return APIKey(
            key_id=data["key_id"],
            key_hash=data["key_hash"],
            name=data["name"],
            description=data["description"],
            scopes=[APIKeyScope(scope) for scope in data["scopes"]],
            status=APIKeyStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
            last_used_at=datetime.fromisoformat(data["last_used_at"]) if data["last_used_at"] else None,
            usage_count=data["usage_count"],
            rate_limit_per_minute=data["rate_limit_per_minute"],
            rate_limit_per_hour=data["rate_limit_per_hour"],
            allowed_ips=data["allowed_ips"],
            user_id=data.get("user_id"),
            service_name=data.get("service_name")
        )

    def _serialize_api_key(self, api_key: APIKey) -> dict[str, Any]:
        """Serialize API key data."""
        return {
            "key_id": api_key.key_id,
            "key_hash": api_key.key_hash,
            "name": api_key.name,
            "description": api_key.description,
            "scopes": [scope.value for scope in api_key.scopes],
            "status": api_key.status.value,
            "created_at": api_key.created_at.isoformat(),
            "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
            "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
            "usage_count": api_key.usage_count,
            "rate_limit_per_minute": api_key.rate_limit_per_minute,
            "rate_limit_per_hour": api_key.rate_limit_per_hour,
            "allowed_ips": api_key.allowed_ips,
            "user_id": api_key.user_id,
            "service_name": api_key.service_name
        }

    @staticmethod
    def generate_api_key() -> tuple[str, str]:
        """
        Generate a new API key and its hash.

        Returns:
            Tuple of (api_key, key_hash)
        """
        # Generate secure random API key
        api_key = f"its_{''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(32))}"
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        return api_key, key_hash

    @staticmethod
    def create_api_key(
        name: str,
        description: str,
        scopes: list[APIKeyScope],
        expires_in_days: int | None = None,
        rate_limit_per_minute: int = 60,
        rate_limit_per_hour: int = 1000,
        allowed_ips: list[str] | None = None,
        user_id: str | None = None,
        service_name: str | None = None
    ) -> tuple[str, APIKey]:
        """
        Create a new API key.

        Returns:
            Tuple of (api_key_string, api_key_object)
        """
        api_key_string, key_hash = APIKeyAuthMiddleware.generate_api_key()

        now = datetime.now(UTC)
        expires_at = None
        if expires_in_days:
            expires_at = now + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=secrets.token_urlsafe(16),
            key_hash=key_hash,
            name=name,
            description=description,
            scopes=scopes,
            status=APIKeyStatus.ACTIVE,
            created_at=now,
            expires_at=expires_at,
            last_used_at=None,
            usage_count=0,
            rate_limit_per_minute=rate_limit_per_minute,
            rate_limit_per_hour=rate_limit_per_hour,
            allowed_ips=allowed_ips or [],
            user_id=user_id,
            service_name=service_name
        )

        return api_key_string, api_key
