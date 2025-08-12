"""Services for ITS Camera AI system.

Provides async CRUD services for all database models with optimized
queries, batch operations, and high-throughput processing capabilities.
"""

from .auth import AuthService
from .base_service import BaseAsyncService
from .cache import CacheService
from .camera_service import CameraService
from .frame_service import DetectionService, FrameService
from .metrics_service import MetricsService

__all__ = [
    # Base service
    "BaseAsyncService",

    # Authentication and caching
    "AuthService",
    "CacheService",

    # Database services
    "CameraService",
    "FrameService",
    "DetectionService",
    "MetricsService",
]
