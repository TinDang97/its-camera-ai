"""Pydantic schemas for API request/response models.

Provides type-safe data validation and serialization for all API endpoints.
"""

from .analytics import (
    AnalyticsResponse,
    IncidentAlert,
    ReportRequest,
    TrafficMetrics,
    VehicleCount,
)
from .auth import (
    LoginRequest,
    LoginResponse,
    PasswordResetRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenRefreshResponse,
    UserProfile,
    UserResponse,
    VerificationRequest,
)
from .cameras import (
    CameraConfig,
    CameraCreate,
    CameraResponse,
    CameraStatus,
    CameraUpdate,
    StreamHealth,
    StreamRequest,
)
from .common import (
    ErrorResponse,
    PaginatedResponse,
    SuccessResponse,
)
from .models import (
    ABTestConfig,
    ModelDeployment,
    ModelMetrics,
    ModelResponse,
    ModelVersion,
)

__all__ = [
    # Analytics
    "AnalyticsResponse",
    "IncidentAlert",
    "ReportRequest",
    "TrafficMetrics",
    "VehicleCount",
    # Auth
    "LoginRequest",
    "LoginResponse",
    "PasswordResetRequest",
    "RefreshTokenRequest",
    "RegisterRequest",
    "TokenRefreshResponse",
    "UserProfile",
    "UserResponse",
    "VerificationRequest",
    # Cameras
    "CameraConfig",
    "CameraCreate",
    "CameraResponse",
    "CameraStatus",
    "CameraUpdate",
    "StreamHealth",
    "StreamRequest",
    # Common
    "ErrorResponse",
    "PaginatedResponse",
    "SuccessResponse",
    # Models
    "ABTestConfig",
    "ModelDeployment",
    "ModelMetrics",
    "ModelResponse",
    "ModelVersion",
]
