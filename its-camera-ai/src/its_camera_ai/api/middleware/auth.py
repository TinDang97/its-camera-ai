"""
Authentication middleware for FastAPI application.

Provides comprehensive authentication and authorization middleware:
- JWT token validation and parsing
- Role-based access control enforcement
- Rate limiting and brute force protection
- Security headers injection
- Request/response logging for security audit
- Session management and validation
"""

import time
from collections.abc import Callable
from uuid import uuid4

import redis.asyncio as redis
from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ...core.config import get_settings
from ...core.exceptions import AuthenticationError, AuthorizationError
from ...core.logging import get_logger
from ...services.auth_service import AuthenticationService, SecurityEventType

logger = get_logger(__name__)


# ============================
# Security Headers Middleware
# ============================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    def __init__(self, app, settings=None):
        super().__init__(app)
        self.settings = settings or get_settings()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        if self.settings.security.enable_security_headers:
            headers = self.settings.get_security_headers()
            for name, value in headers.items():
                response.headers[name] = value

        return response


# ============================
# Rate Limiting Middleware
# ============================


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware to prevent abuse."""

    def __init__(self, app, redis_client: redis.Redis | None = None, settings=None):
        super().__init__(app)
        self.redis = redis_client
        self.settings = settings or get_settings()
        self.general_limit = self.settings.security.rate_limit_per_minute
        self.auth_limit = self.settings.security.auth_rate_limit_per_minute

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply rate limiting based on IP and endpoint."""
        if not self.redis:
            # Skip rate limiting if Redis not available
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        endpoint = request.url.path

        # Different limits for auth endpoints
        is_auth_endpoint = any(
            auth_path in endpoint
            for auth_path in [
                "/auth/login",
                "/auth/register",
                "/auth/refresh",
                "/auth/mfa",
            ]
        )

        limit = self.auth_limit if is_auth_endpoint else self.general_limit
        window = 60  # 1 minute window

        # Rate limit key
        rate_limit_key = f"rate_limit:{client_ip}:{endpoint}"

        try:
            # Get current count
            current = await self.redis.get(rate_limit_key)
            current = int(current) if current else 0

            if current >= limit:
                # Rate limit exceeded
                logger.warning(
                    "Rate limit exceeded",
                    ip=client_ip,
                    endpoint=endpoint,
                    current=current,
                    limit=limit,
                )

                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. Limit: {limit} per minute",
                        "retry_after": 60,
                    },
                    headers={"Retry-After": "60"},
                )

            # Increment counter
            await self.redis.incr(rate_limit_key)
            if current == 0:
                # Set expiry on first request
                await self.redis.expire(rate_limit_key, window)

            # Add rate limit headers
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current - 1))
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + window)

            return response

        except Exception as e:
            logger.error("Rate limiting error", error=str(e))
            # Continue without rate limiting on Redis errors
            return await call_next(request)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded IP headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct connection IP
        return request.client.host if request.client else "unknown"


# ============================
# Authentication Middleware
# ============================


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for protected routes."""

    def __init__(self, app, auth_service: AuthenticationService, settings=None):
        super().__init__(app)
        self.auth_service = auth_service
        self.settings = settings or get_settings()

        # Routes that don't require authentication
        self.public_routes = {
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/auth/login",
            "/auth/register",
            "/auth/refresh",
            "/public",
        }

        # Routes that require specific permissions
        self.protected_routes = {
            "/admin": ["admin"],
            "/users": ["admin", "operator"],
            "/cameras/control": ["admin", "operator"],
            "/analytics/manage": ["admin", "operator", "analyst"],
            "/security": ["admin", "auditor"],
            "/system": ["admin"],
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Authenticate and authorize requests."""
        path = request.url.path

        # Skip authentication for public routes
        if self._is_public_route(path):
            return await call_next(request)

        try:
            # Extract and validate token
            token = self._extract_token(request)
            if not token:
                raise AuthenticationError("Authentication required")

            # Validate token
            validation = await self.auth_service.verify_token(token)
            if not validation.valid:
                raise AuthenticationError(validation.error_message or "Invalid token")

            # Add user context to request
            request.state.user_id = validation.user_id
            request.state.session_id = validation.session_id
            request.state.roles = validation.roles
            request.state.permissions = validation.permissions

            # Check route permissions
            if not self._check_route_permissions(
                path, validation.roles, validation.permissions
            ):
                raise AuthorizationError("Insufficient permissions")

            # Log successful authentication
            await self._log_auth_event(
                request, validation.user_id, True, "Authentication successful"
            )

            response = await call_next(request)

            # Add user info to response headers (for debugging)
            if self.settings.debug:
                response.headers["X-User-ID"] = validation.user_id or ""
                response.headers["X-User-Roles"] = ",".join(validation.roles)

            return response

        except (AuthenticationError, AuthorizationError) as e:
            # Log failed authentication
            await self._log_auth_event(request, None, False, str(e))

            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": "Authentication failed", "message": str(e)},
                headers={"WWW-Authenticate": "Bearer"},
            )

        except Exception as e:
            logger.error("Authentication middleware error", error=str(e))
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "message": "Authentication processing failed",
                },
            )

    def _is_public_route(self, path: str) -> bool:
        """Check if route is public (doesn't require authentication)."""
        return any(public_path in path for public_path in self.public_routes)

    def _extract_token(self, request: Request) -> str | None:
        """Extract JWT token from request."""
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix

        # Check cookie
        token = request.cookies.get("access_token")
        if token:
            return token

        # Check query parameter (less secure, only for development)
        if self.settings.debug:
            return request.query_params.get("token")

        return None

    def _check_route_permissions(
        self, path: str, user_roles: list[str], _user_permissions: list[str]
    ) -> bool:
        """Check if user has permissions for the route."""
        # Check specific route permissions
        for route_pattern, required_roles in self.protected_routes.items():
            if route_pattern in path:
                # Check if user has any of the required roles
                return any(role in user_roles for role in required_roles)

        # For unspecified routes, any authenticated user has access
        return True

    async def _log_auth_event(
        self, request: Request, user_id: str | None, success: bool, message: str
    ):
        """Log authentication event for audit."""
        try:
            from ...services.auth_service import SecurityAuditEvent

            event = SecurityAuditEvent(
                event_id=str(uuid4()),
                event_type=SecurityEventType.LOGIN_SUCCESS
                if success
                else SecurityEventType.LOGIN_FAILURE,
                user_id=user_id,
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("User-Agent"),
                session_id=getattr(request.state, "session_id", None),
                resource=request.url.path,
                action=request.method,
                success=success,
                details={"message": message},
                timestamp=time.time(),
                risk_score=10 if success else 50,
            )

            await self.auth_service.audit_logger.log_event(event)

        except Exception as e:
            logger.error("Failed to log auth event", error=str(e))

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"


# ============================
# CORS Middleware Enhancement
# ============================


class EnhancedCORSMiddleware(BaseHTTPMiddleware):
    """Enhanced CORS middleware with security considerations."""

    def __init__(self, app, settings=None):
        super().__init__(app)
        self.settings = settings or get_settings()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle CORS with enhanced security."""
        origin = request.headers.get("Origin")

        # Handle preflight requests
        if request.method == "OPTIONS":
            response = Response()
            self._add_cors_headers(response, origin)
            return response

        response = await call_next(request)
        self._add_cors_headers(response, origin)

        return response

    def _add_cors_headers(self, response: Response, origin: str | None):
        """Add CORS headers to response."""
        if not self.settings.security.enable_cors:
            return

        allowed_origins = self.settings.security.allowed_origins

        # Check if origin is allowed
        if origin and (
            "*" in allowed_origins
            or origin in allowed_origins
            or self._is_subdomain_allowed(origin, allowed_origins)
        ):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
        elif "*" in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"

        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        )
        response.headers["Access-Control-Allow-Headers"] = (
            "Authorization, Content-Type, X-Requested-With, X-CSRF-Token, "
            "X-User-Agent, X-Request-ID, X-Session-ID"
        )
        response.headers["Access-Control-Expose-Headers"] = (
            "X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset, "
            "X-Request-ID, X-Response-Time"
        )
        response.headers["Access-Control-Max-Age"] = "86400"  # 24 hours

    def _is_subdomain_allowed(self, origin: str, allowed_origins: list[str]) -> bool:
        """Check if origin is an allowed subdomain."""
        for allowed in allowed_origins:
            if allowed.startswith("*."):
                domain = allowed[2:]  # Remove *.
                if origin.endswith(f".{domain}") or origin == domain:
                    return True
        return False


# ============================
# Request/Response Logging Middleware
# ============================


class SecurityLoggingMiddleware(BaseHTTPMiddleware):
    """Log requests and responses for security monitoring."""

    def __init__(self, app, settings=None):
        super().__init__(app)
        self.settings = settings or get_settings()
        # Paths to exclude from logging (e.g., health checks)
        self.excluded_paths = {"/health", "/metrics"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response for security monitoring."""
        if request.url.path in self.excluded_paths:
            return await call_next(request)

        start_time = time.time()
        request_id = str(uuid4())[:8]

        # Log request
        logger.info(
            "HTTP Request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query=str(request.query_params) if request.query_params else None,
            ip=self._get_client_ip(request),
            user_agent=request.headers.get("User-Agent"),
            content_type=request.headers.get("Content-Type"),
            user_id=getattr(request.state, "user_id", None),
        )

        # Process request
        response = await call_next(request)

        # Calculate response time
        response_time = time.time() - start_time

        # Log response
        logger.info(
            "HTTP Response",
            request_id=request_id,
            status_code=response.status_code,
            response_time_ms=int(response_time * 1000),
            content_type=response.headers.get("Content-Type"),
        )

        # Add response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"

        # Log security events for suspicious activities
        if response.status_code >= 400:
            await self._log_security_event(request, response, request_id)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    async def _log_security_event(
        self, request: Request, response: Response, request_id: str
    ):
        """Log potential security events."""
        # Determine risk score based on response status
        risk_score = 0
        if response.status_code == 401:
            risk_score = 40  # Unauthorized access attempt
        elif response.status_code == 403:
            risk_score = 60  # Forbidden access attempt
        elif response.status_code == 429:
            risk_score = 70  # Rate limit exceeded
        elif response.status_code >= 500:
            risk_score = 30  # Server error

        if risk_score > 0:
            logger.warning(
                "Security event detected",
                request_id=request_id,
                status_code=response.status_code,
                risk_score=risk_score,
                path=request.url.path,
                method=request.method,
                ip=self._get_client_ip(request),
            )


# ============================
# Middleware Factory Functions
# ============================


def create_auth_middleware(app, auth_service: AuthenticationService, settings=None):
    """Create authentication middleware with dependencies."""
    return AuthenticationMiddleware(app, auth_service, settings)


def create_security_middleware_stack(
    app,
    auth_service: AuthenticationService,
    redis_client: redis.Redis | None = None,
    settings=None,
):
    """Create complete security middleware stack."""
    settings = settings or get_settings()

    # Add middlewares in reverse order (they're applied as a stack)
    middlewares = [
        SecurityLoggingMiddleware,
        lambda app: AuthenticationMiddleware(app, auth_service, settings),
        lambda app: RateLimitingMiddleware(app, redis_client, settings),
        lambda app: EnhancedCORSMiddleware(app, settings),
        lambda app: SecurityHeadersMiddleware(app, settings),
    ]

    # Apply middlewares
    for middleware in middlewares:
        app.add_middleware(middleware)

    return app
