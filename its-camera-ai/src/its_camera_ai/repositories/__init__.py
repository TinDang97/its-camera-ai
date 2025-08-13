"""Repository layer for data access abstraction.

This module provides the repository pattern implementation for clean
separation of data access logic from business logic. All repositories
follow consistent patterns for CRUD operations, error handling, and
async/await compatibility.

The repository layer abstracts:
- Database session management
- Query optimization
- Transaction handling
- Error propagation
- Pagination and filtering
"""

from .alert_repository import AlertRepository
from .analytics_repository import AnalyticsRepository
from .base_repository import BaseRepository
from .camera_repository import CameraRepository
from .detection_repository import DetectionRepository
from .frame_repository import FrameRepository
from .metrics_repository import MetricsRepository
from .user_repository import UserRepository

__all__ = [
    "BaseRepository",
    "AlertRepository",
    "AnalyticsRepository",
    "UserRepository",
    "CameraRepository",
    "FrameRepository",
    "DetectionRepository",
    "MetricsRepository",
]
