"""
Security Headers Middleware for comprehensive HTTP security headers.

Provides:
- Content Security Policy (CSP)
- X-Frame-Options
- X-Content-Type-Options
- Strict-Transport-Security (HSTS)
- X-XSS-Protection
- Referrer-Policy
- Permissions-Policy
- Feature-Policy (deprecated but still supported)
- Cross-Origin policies
"""

from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...core.config import get_settings
from ...core.logging import get_logger

logger = get_logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add comprehensive security headers to all responses."""

    def __init__(self, app, settings=None):
        super().__init__(app)
        self.settings = settings or get_settings()

        # Security headers configuration
        self.security_headers = self._build_security_headers()

        # Route-specific overrides
        self.route_overrides = {
            "/docs": {
                "Content-Security-Policy": self._build_docs_csp(),
                "X-Frame-Options": "SAMEORIGIN"  # Allow framing for docs
            },
            "/redoc": {
                "Content-Security-Policy": self._build_docs_csp(),
                "X-Frame-Options": "SAMEORIGIN"
            },
            "/api/v1/cameras/stream": {
                "Cross-Origin-Embedder-Policy": "unsafe-none",  # For video streaming
                "Cross-Origin-Opener-Policy": "unsafe-none"
            }
        }

        # Headers to remove for security
        self.headers_to_remove = {
            "Server",           # Hide server information
            "X-Powered-By",     # Hide technology stack
            "X-AspNet-Version", # Hide framework version
            "X-AspNetMvc-Version"
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        try:
            # Process request
            response = await call_next(request)

            # Remove potentially dangerous headers
            self._remove_insecure_headers(response)

            # Add security headers
            self._add_security_headers(response, request)

            # Apply route-specific overrides
            self._apply_route_overrides(response, request.url.path)

            # Add security monitoring headers
            self._add_monitoring_headers(response, request)

            return response

        except Exception as e:
            logger.error("Security headers middleware error", error=str(e), path=request.url.path)
            # Continue without headers on error
            response = await call_next(request)
            return response

    def _build_security_headers(self) -> dict[str, str]:
        """Build the default security headers."""
        headers = {}

        # Content Security Policy (CSP)
        headers["Content-Security-Policy"] = self._build_default_csp()

        # X-Frame-Options - Prevents clickjacking
        headers["X-Frame-Options"] = "DENY"

        # X-Content-Type-Options - Prevents MIME sniffing
        headers["X-Content-Type-Options"] = "nosniff"

        # X-XSS-Protection - Enables XSS filtering (legacy)
        headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer-Policy - Controls referrer information
        headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Strict-Transport-Security (HSTS) - HTTPS enforcement
        if self.settings.is_production():
            headers["Strict-Transport-Security"] = f"max-age={self.settings.security.hsts_max_age}; includeSubDomains; preload"

        # Permissions-Policy - Control browser features
        headers["Permissions-Policy"] = self._build_permissions_policy()

        # Feature-Policy - Deprecated but still supported
        headers["Feature-Policy"] = self._build_feature_policy()

        # Cross-Origin policies
        headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        headers["Cross-Origin-Opener-Policy"] = "same-origin"
        headers["Cross-Origin-Resource-Policy"] = "same-origin"

        # Cache control for sensitive responses
        headers["Cache-Control"] = "no-cache, no-store, must-revalidate, private"
        headers["Pragma"] = "no-cache"
        headers["Expires"] = "0"

        return headers

    def _build_default_csp(self) -> str:
        """Build default Content Security Policy."""
        # Base CSP for API responses
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # Relaxed for docs
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: https:",
            "connect-src 'self' ws: wss:",  # Allow WebSocket connections
            "media-src 'self' blob:",  # Allow video/audio from camera streams
            "object-src 'none'",
            "frame-src 'none'",
            "worker-src 'self'",
            "manifest-src 'self'",
            "form-action 'self'",
            "base-uri 'self'",
            "upgrade-insecure-requests"  # Upgrade HTTP to HTTPS
        ]

        # Add report URI if configured
        if hasattr(self.settings.security, 'csp_report_uri') and self.settings.security.csp_report_uri:
            csp_directives.append(f"report-uri {self.settings.security.csp_report_uri}")

        return "; ".join(csp_directives)

    def _build_docs_csp(self) -> str:
        """Build relaxed CSP for API documentation pages."""
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net",
            "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net",
            "img-src 'self' data: https: blob:",
            "connect-src 'self'",
            "object-src 'none'",
            "frame-src 'self'",
            "worker-src 'self' blob:",
            "manifest-src 'self'",
            "form-action 'self'",
            "base-uri 'self'"
        ]

        return "; ".join(csp_directives)

    def _build_permissions_policy(self) -> str:
        """Build Permissions-Policy header."""
        policies = [
            "accelerometer=()",          # Disable accelerometer
            "ambient-light-sensor=()",   # Disable ambient light sensor
            "autoplay=()",              # Disable autoplay
            "battery=()",               # Disable battery API
            "camera=(self)",            # Allow camera for our app only
            "cross-origin-isolated=()",  # Disable cross-origin isolation
            "display-capture=()",       # Disable display capture
            "document-domain=()",       # Disable document.domain
            "encrypted-media=()",       # Disable encrypted media
            "execution-while-not-rendered=()", # Disable execution while not rendered
            "execution-while-out-of-viewport=()", # Disable execution while out of viewport
            "fullscreen=(self)",        # Allow fullscreen for our app
            "geolocation=()",           # Disable geolocation
            "gyroscope=()",             # Disable gyroscope
            "magnetometer=()",          # Disable magnetometer
            "microphone=(self)",        # Allow microphone for our app only
            "navigation-override=()",   # Disable navigation override
            "payment=()",               # Disable payment API
            "picture-in-picture=()",    # Disable picture-in-picture
            "publickey-credentials-get=()", # Disable WebAuthn
            "screen-wake-lock=()",      # Disable screen wake lock
            "sync-xhr=()",              # Disable synchronous XHR
            "usb=()",                   # Disable USB API
            "web-share=()",             # Disable Web Share API
            "xr-spatial-tracking=()"    # Disable XR spatial tracking
        ]

        return ", ".join(policies)

    def _build_feature_policy(self) -> str:
        """Build Feature-Policy header (deprecated but still supported)."""
        policies = [
            "accelerometer 'none'",
            "ambient-light-sensor 'none'",
            "autoplay 'none'",
            "battery 'none'",
            "camera 'self'",
            "display-capture 'none'",
            "document-domain 'none'",
            "encrypted-media 'none'",
            "fullscreen 'self'",
            "geolocation 'none'",
            "gyroscope 'none'",
            "magnetometer 'none'",
            "microphone 'self'",
            "midi 'none'",
            "payment 'none'",
            "picture-in-picture 'none'",
            "publickey-credentials-get 'none'",
            "sync-xhr 'none'",
            "usb 'none'",
            "vr 'none'",
            "wake-lock 'none'",
            "xr-spatial-tracking 'none'"
        ]

        return "; ".join(policies)

    def _add_security_headers(self, response: Response, request: Request) -> None:
        """Add security headers to response."""
        for header_name, header_value in self.security_headers.items():
            # Don't override existing headers
            if header_name not in response.headers:
                response.headers[header_name] = header_value

    def _apply_route_overrides(self, response: Response, path: str) -> None:
        """Apply route-specific header overrides."""
        for route_pattern, overrides in self.route_overrides.items():
            if route_pattern in path:
                for header_name, header_value in overrides.items():
                    response.headers[header_name] = header_value
                break

    def _remove_insecure_headers(self, response: Response) -> None:
        """Remove headers that might leak sensitive information."""
        for header_name in self.headers_to_remove:
            if header_name in response.headers:
                del response.headers[header_name]

    def _add_monitoring_headers(self, response: Response, request: Request) -> None:
        """Add headers for security monitoring."""
        # Add security policy version for tracking
        response.headers["X-Security-Policy-Version"] = "1.0"

        # Add CSP nonce for script/style if needed
        if hasattr(request.state, "csp_nonce"):
            response.headers["X-CSP-Nonce"] = request.state.csp_nonce

        # Add frame options information for debugging
        if self.settings.debug:
            response.headers["X-Debug-Frame-Options"] = response.headers.get("X-Frame-Options", "not-set")

    def customize_csp_for_route(self, path: str, additional_sources: dict[str, list[str]]) -> str:
        """
        Customize CSP for specific routes.

        Args:
            path: Route path
            additional_sources: Additional sources to allow per directive

        Returns:
            Customized CSP string
        """
        base_csp = self._parse_csp(self._build_default_csp())

        # Merge additional sources
        for directive, sources in additional_sources.items():
            if directive in base_csp:
                base_csp[directive].extend(sources)
            else:
                base_csp[directive] = sources

        # Rebuild CSP string
        csp_parts = []
        for directive, sources in base_csp.items():
            csp_parts.append(f"{directive} {' '.join(sources)}")

        return "; ".join(csp_parts)

    def _parse_csp(self, csp_string: str) -> dict[str, list[str]]:
        """Parse CSP string into directive dictionary."""
        csp_dict = {}

        for directive in csp_string.split(";"):
            directive = directive.strip()
            if not directive:
                continue

            parts = directive.split()
            if len(parts) > 0:
                directive_name = parts[0]
                sources = parts[1:] if len(parts) > 1 else []
                csp_dict[directive_name] = sources

        return csp_dict

    def add_security_header(self, name: str, value: str) -> None:
        """Add a custom security header."""
        self.security_headers[name] = value

    def remove_security_header(self, name: str) -> None:
        """Remove a security header."""
        if name in self.security_headers:
            del self.security_headers[name]

    def get_security_headers_info(self) -> dict[str, Any]:
        """Get information about configured security headers."""
        return {
            "headers": dict(self.security_headers),
            "route_overrides": self.route_overrides,
            "removed_headers": list(self.headers_to_remove),
            "production_mode": self.settings.is_production()
        }

    def validate_csp(self, csp_string: str) -> dict[str, Any]:
        """
        Validate CSP string format and common issues.

        Args:
            csp_string: CSP string to validate

        Returns:
            Validation result with warnings and errors
        """
        result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }

        try:
            csp_dict = self._parse_csp(csp_string)

            # Check for common issues
            if "default-src" not in csp_dict:
                result["warnings"].append("Missing default-src directive")

            if "'unsafe-inline'" in csp_dict.get("script-src", []):
                result["warnings"].append("'unsafe-inline' in script-src reduces security")

            if "'unsafe-eval'" in csp_dict.get("script-src", []):
                result["warnings"].append("'unsafe-eval' in script-src reduces security")

            if "object-src" not in csp_dict:
                result["recommendations"].append("Consider adding 'object-src none' directive")

            if "base-uri" not in csp_dict:
                result["recommendations"].append("Consider adding 'base-uri self' directive")

            # Check for mixed content issues
            for _directive, sources in csp_dict.items():
                for source in sources:
                    if source.startswith("http://") and self.settings.is_production():
                        result["warnings"].append(f"HTTP source '{source}' in production CSP")

        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"CSP parsing error: {str(e)}")

        return result

    @staticmethod
    def generate_nonce() -> str:
        """Generate a cryptographically secure nonce for CSP."""
        import secrets
        return secrets.token_urlsafe(16)

    @staticmethod
    def create_nonce_csp(base_csp: str, nonce: str) -> str:
        """Add nonce to CSP directives."""
        # Replace 'unsafe-inline' with nonce where appropriate
        csp_with_nonce = base_csp.replace(
            "script-src 'self' 'unsafe-inline'",
            f"script-src 'self' 'nonce-{nonce}'"
        ).replace(
            "style-src 'self' 'unsafe-inline'",
            f"style-src 'self' 'nonce-{nonce}'"
        )

        return csp_with_nonce
