"""
API routes for ITS Camera AI system.

Provides comprehensive REST API endpoints for:
- Authentication and user management
- Camera control and monitoring
- Analytics and reporting
- System configuration
"""

from .auth import router as auth_router

__all__ = [
    "auth_router",
]
