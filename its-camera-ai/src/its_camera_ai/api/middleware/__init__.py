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

__all__ = [
    "AuthenticationMiddleware",
    "EnhancedCORSMiddleware",
    "RateLimitingMiddleware",
    "SecurityHeadersMiddleware",
    "SecurityLoggingMiddleware",
    "create_auth_middleware",
    "create_security_middleware_stack",
]
