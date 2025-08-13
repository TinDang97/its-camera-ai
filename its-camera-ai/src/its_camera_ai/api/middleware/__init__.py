"""
API middleware for ITS Camera AI system.

Provides comprehensive middleware for security, authentication, and monitoring:
- Authentication and authorization middleware
- Security headers middleware
- Rate limiting middleware
- CORS middleware with security features
- Request/response logging middleware
"""

from .auth import (
    AuthenticationMiddleware,
    EnhancedCORSMiddleware,
    RateLimitingMiddleware,
    SecurityHeadersMiddleware,
    SecurityLoggingMiddleware,
    create_auth_middleware,
    create_security_middleware_stack,
)

# Add aliases for backward compatibility
LoggingMiddleware = SecurityLoggingMiddleware
MetricsMiddleware = SecurityLoggingMiddleware  # For now, combine with logging
RateLimitMiddleware = RateLimitingMiddleware
SecurityMiddleware = SecurityHeadersMiddleware

__all__ = [
    "AuthenticationMiddleware",
    "EnhancedCORSMiddleware",
    "RateLimitingMiddleware",
    "SecurityHeadersMiddleware",
    "SecurityLoggingMiddleware",
    "create_auth_middleware",
    "create_security_middleware_stack",
    # Backward compatibility aliases
    "LoggingMiddleware",
    "MetricsMiddleware",
    "RateLimitMiddleware",
    "SecurityMiddleware",
]
