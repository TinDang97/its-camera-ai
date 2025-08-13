"""
CSRF (Cross-Site Request Forgery) protection middleware.

Provides:
- Double-submit cookie pattern
- Custom header validation
- SameSite cookie configuration
- Token rotation on authentication
- Origin validation
- Referrer validation
"""

import hmac
import secrets
import time
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ...core.config import get_settings
from ...core.logging import get_logger

logger = get_logger(__name__)


class CSRFProtectionMiddleware(BaseHTTPMiddleware):
    """CSRF protection middleware using double-submit cookie pattern."""

    def __init__(self, app, settings=None):
        super().__init__(app)
        self.settings = settings or get_settings()

        # CSRF configuration
        self.csrf_cookie_name = "csrf_token"
        self.csrf_header_name = "X-CSRF-Token"
        self.csrf_form_field = "csrf_token"
        self.token_length = 32
        self.cookie_max_age = 3600  # 1 hour

        # Methods that require CSRF protection
        self.protected_methods = {"POST", "PUT", "PATCH", "DELETE"}

        # Routes that are exempt from CSRF protection
        self.exempt_routes = {
            "/api/v1/auth/login",
            "/api/v1/auth/register",
            "/api/v1/auth/refresh",
            "/api/v1/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }

        # API endpoints (typically use API keys or JWT, not cookies)
        self.api_routes_patterns = [
            "/api/v1/",
        ]

        # Origins that are always allowed (same-origin)
        self.allowed_origins = set()
        if self.settings.security.allowed_origins:
            self.allowed_origins.update(self.settings.security.allowed_origins)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply CSRF protection to requests."""
        # Skip CSRF for non-protected methods
        if request.method not in self.protected_methods:
            response = await call_next(request)
            # Always set CSRF token for subsequent requests
            self._set_csrf_token(response, request)
            return response

        # Skip CSRF for exempt routes
        if self._is_exempt_route(request.url.path):
            response = await call_next(request)
            self._set_csrf_token(response, request)
            return response

        # Skip CSRF for API routes using API key authentication
        if self._is_api_route(request) and self._has_api_key(request):
            response = await call_next(request)
            return response

        try:
            # Validate CSRF protection
            if not await self._validate_csrf(request):
                return self._create_csrf_error("CSRF validation failed")

            # Process request
            response = await call_next(request)

            # Rotate token after successful state-changing operations
            if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
                self._rotate_csrf_token(response, request)

            return response

        except Exception as e:
            logger.error("CSRF protection error", error=str(e), path=request.url.path)
            return self._create_csrf_error("CSRF protection failed")

    async def _validate_csrf(self, request: Request) -> bool:
        """Validate CSRF token using double-submit cookie pattern."""
        # Get CSRF token from cookie
        cookie_token = request.cookies.get(self.csrf_cookie_name)
        if not cookie_token:
            logger.warning("CSRF cookie missing", path=request.url.path, ip=self._get_client_ip(request))
            return False

        # Get CSRF token from header or form data
        header_token = request.headers.get(self.csrf_header_name)

        if not header_token:
            # Check form data for token
            if request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
                try:
                    form_data = await request.form()
                    header_token = form_data.get(self.csrf_form_field)
                except Exception as e:
                    logger.warning("Failed to read form data for CSRF token", error=str(e))
                    return False

        if not header_token:
            logger.warning("CSRF token missing from header/form", path=request.url.path, ip=self._get_client_ip(request))
            return False

        # Compare tokens (constant-time comparison)
        if not self._secure_compare(cookie_token, header_token):
            logger.warning("CSRF token mismatch", path=request.url.path, ip=self._get_client_ip(request))
            return False

        # Validate token structure and age
        if not self._validate_token_structure(cookie_token):
            logger.warning("Invalid CSRF token structure", path=request.url.path, ip=self._get_client_ip(request))
            return False

        # Additional validations
        if not self._validate_origin(request):
            return False

        return self._validate_referrer(request)

    def _validate_origin(self, request: Request) -> bool:
        """Validate Origin header."""
        origin = request.headers.get("origin")
        if not origin:
            # Origin header is not always present (e.g., same-origin requests)
            return True

        # Check if origin is in allowed list
        if self.allowed_origins and origin not in self.allowed_origins:
            logger.warning("Invalid origin", origin=origin, path=request.url.path, ip=self._get_client_ip(request))
            return False

        # Additional origin validation
        try:
            from urllib.parse import urlparse
            parsed_origin = urlparse(origin)

            # Check for suspicious origins
            if parsed_origin.scheme not in {"http", "https"}:
                logger.warning("Suspicious origin scheme", origin=origin, path=request.url.path)
                return False

            # In production, require HTTPS
            if self.settings.is_production() and parsed_origin.scheme != "https":
                logger.warning("Non-HTTPS origin in production", origin=origin, path=request.url.path)
                return False

        except Exception as e:
            logger.warning("Origin validation error", origin=origin, error=str(e))
            return False

        return True

    def _validate_referrer(self, request: Request) -> bool:
        """Validate Referer header."""
        referer = request.headers.get("referer")
        if not referer:
            # Referer is optional and may be blocked by privacy tools
            return True

        try:
            from urllib.parse import urlparse
            parsed_referer = urlparse(referer)
            request_host = request.url.hostname

            # Simple same-origin check
            if parsed_referer.hostname != request_host:
                # Allow configured origins
                if self.allowed_origins:
                    referer_origin = f"{parsed_referer.scheme}://{parsed_referer.netloc}"
                    if referer_origin not in self.allowed_origins:
                        logger.warning("Cross-origin referer", referer=referer, path=request.url.path)
                        return False
                else:
                    logger.warning("Cross-origin referer", referer=referer, path=request.url.path)
                    return False

        except Exception as e:
            logger.warning("Referer validation error", referer=referer, error=str(e))
            return False

        return True

    def _is_exempt_route(self, path: str) -> bool:
        """Check if route is exempt from CSRF protection."""
        return path in self.exempt_routes

    def _is_api_route(self, request: Request) -> bool:
        """Check if request is to an API route."""
        path = request.url.path
        return any(pattern in path for pattern in self.api_routes_patterns)

    def _has_api_key(self, request: Request) -> bool:
        """Check if request has API key authentication."""
        return (
            request.headers.get("x-api-key") is not None or
            request.query_params.get("api_key") is not None or
            (request.headers.get("authorization", "").startswith("Bearer ") and
             not request.cookies.get("session_id"))  # JWT without session cookies
        )

    def _generate_csrf_token(self) -> str:
        """Generate a new CSRF token with timestamp."""
        # Include timestamp for token age validation
        timestamp = str(int(time.time()))
        random_part = secrets.token_urlsafe(self.token_length)

        # Create token with timestamp
        token_data = f"{timestamp}:{random_part}"

        # Sign the token
        signature = self._sign_token(token_data)

        return f"{token_data}:{signature}"

    def _sign_token(self, token_data: str) -> str:
        """Sign token data with HMAC."""
        secret_key = self.settings.security.secret_key.encode()
        signature = hmac.new(secret_key, token_data.encode(), digestmod="sha256").hexdigest()
        return signature

    def _validate_token_structure(self, token: str) -> bool:
        """Validate CSRF token structure and age."""
        try:
            parts = token.split(":")
            if len(parts) != 3:
                return False

            timestamp_str, random_part, signature = parts

            # Verify signature
            token_data = f"{timestamp_str}:{random_part}"
            expected_signature = self._sign_token(token_data)

            if not self._secure_compare(signature, expected_signature):
                return False

            # Check token age
            timestamp = int(timestamp_str)
            current_time = int(time.time())
            age = current_time - timestamp

            # Token should not be older than cookie max age
            if age > self.cookie_max_age:
                return False

            # Token should not be from the future (allow small clock skew)
            if age < -60:  # 1 minute clock skew
                return False

            return True

        except (ValueError, IndexError) as e:
            logger.warning("Token structure validation failed", error=str(e))
            return False

    def _secure_compare(self, a: str, b: str) -> bool:
        """Constant-time string comparison."""
        return hmac.compare_digest(a, b)

    def _set_csrf_token(self, response: Response, request: Request) -> None:
        """Set CSRF token in cookie."""
        # Don't set CSRF cookie for API routes
        if self._is_api_route(request):
            return

        # Check if token already exists and is valid
        existing_token = request.cookies.get(self.csrf_cookie_name)
        if existing_token and self._validate_token_structure(existing_token):
            return

        # Generate new token
        csrf_token = self._generate_csrf_token()

        # Set cookie with security attributes
        response.set_cookie(
            key=self.csrf_cookie_name,
            value=csrf_token,
            max_age=self.cookie_max_age,
            httponly=True,  # Prevent XSS access
            secure=self.settings.is_production(),  # HTTPS only in production
            samesite="strict"  # CSRF protection
        )

        # Also provide token in response header for SPA applications
        response.headers["X-CSRF-Token"] = csrf_token

    def _rotate_csrf_token(self, response: Response, request: Request) -> None:
        """Rotate CSRF token after state-changing operations."""
        # Generate new token
        new_token = self._generate_csrf_token()

        # Update cookie
        response.set_cookie(
            key=self.csrf_cookie_name,
            value=new_token,
            max_age=self.cookie_max_age,
            httponly=True,
            secure=self.settings.is_production(),
            samesite="strict"
        )

        # Provide new token in response header
        response.headers["X-CSRF-Token"] = new_token

        logger.debug("CSRF token rotated", path=request.url.path)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        return request.client.host if request.client else "unknown"

    def _create_csrf_error(self, message: str) -> JSONResponse:
        """Create CSRF error response."""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "csrf_protection_failed",
                "message": message,
                "details": "CSRF token required for this operation"
            }
        )

    # Utility methods for CSRF token management
    @staticmethod
    def extract_token_from_response(response: Response) -> str | None:
        """Extract CSRF token from response headers."""
        return response.headers.get("X-CSRF-Token")

    def create_token_for_user(self, user_id: str) -> str:
        """Create CSRF token for a specific user (for testing/admin purposes)."""
        return self._generate_csrf_token()

    def validate_token_manually(self, token: str) -> bool:
        """Manually validate a CSRF token (for testing purposes)."""
        return self._validate_token_structure(token)

    def get_token_info(self, token: str) -> dict[str, Any]:
        """Get information about a CSRF token (for debugging)."""
        try:
            parts = token.split(":")
            if len(parts) != 3:
                return {"valid": False, "error": "Invalid token format"}

            timestamp_str, random_part, signature = parts

            # Verify signature
            token_data = f"{timestamp_str}:{random_part}"
            expected_signature = self._sign_token(token_data)
            signature_valid = self._secure_compare(signature, expected_signature)

            timestamp = int(timestamp_str)
            current_time = int(time.time())
            age = current_time - timestamp

            return {
                "valid": signature_valid and age <= self.cookie_max_age and age >= -60,
                "timestamp": timestamp,
                "age_seconds": age,
                "signature_valid": signature_valid,
                "expired": age > self.cookie_max_age,
                "from_future": age < -60
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}
