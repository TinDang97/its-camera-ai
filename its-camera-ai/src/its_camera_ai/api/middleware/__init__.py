"""
API middleware for ITS Camera AI system.

Provides comprehensive middleware for security, authentication, and monitoring:
- Authentication and authorization middleware
- Security headers middleware
- Rate limiting middleware
- CORS middleware with security features
- Request/response logging middleware
"""

# Import existing middleware from auth module
# Import new security middleware
from .api_key_auth import APIKeyAuthMiddleware
from .auth import (
    AuthenticationMiddleware,
    EnhancedCORSMiddleware,
    SecurityLoggingMiddleware,
    create_auth_middleware,
    create_security_middleware_stack,
)
from .csrf import CSRFProtectionMiddleware
from .rate_limiting import EnhancedRateLimitMiddleware
from .security_headers import SecurityHeadersMiddleware
from .security_validation import SecurityValidationMiddleware

# Import legacy middleware for backward compatibility
try:
    from .auth import RateLimitingMiddleware as LegacyRateLimitMiddleware
except ImportError:
    LegacyRateLimitMiddleware = None

# Add aliases for backward compatibility
LoggingMiddleware = SecurityLoggingMiddleware
MetricsMiddleware = SecurityLoggingMiddleware  # For now, combine with logging
RateLimitMiddleware = EnhancedRateLimitMiddleware  # Use enhanced version
SecurityMiddleware = SecurityHeadersMiddleware

# Legacy compatibility
RateLimitingMiddleware = LegacyRateLimitMiddleware or EnhancedRateLimitMiddleware

__all__ = [
    # Core authentication middleware
    "AuthenticationMiddleware",
    "EnhancedCORSMiddleware",
    "SecurityLoggingMiddleware",
    "create_auth_middleware",
    "create_security_middleware_stack",

    # New security middleware
    "SecurityValidationMiddleware",
    "EnhancedRateLimitMiddleware",
    "APIKeyAuthMiddleware",
    "CSRFProtectionMiddleware",
    "SecurityHeadersMiddleware",

    # Backward compatibility aliases
    "LoggingMiddleware",
    "MetricsMiddleware",
    "RateLimitMiddleware",
    "SecurityMiddleware",
    "RateLimitingMiddleware",
]
