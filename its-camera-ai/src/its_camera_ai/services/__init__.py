"""Services for ITS Camera AI system.

Provides async CRUD services for all database models with optimized
queries, batch operations, and high-throughput processing capabilities.
"""

# Import safe services first
from .auth import AuthService
from .base_service import BaseAsyncService
from .cache import CacheService
from .metrics_service import MetricsService

# Import streaming service components (safe imports)
try:
    from .grpc_streaming_server import StreamingServer, StreamingServiceImpl
    from .streaming_container import (
        StreamingContainer,
        create_streaming_container,
        initialize_streaming_container,
    )
    from .streaming_service import (
        CameraConfig,
        CameraConnectionManager,
        FrameQualityValidator,
        StreamingDataProcessor,
        StreamingServiceInterface,
        StreamProtocol,
    )
except ImportError as e:
    # Handle import errors gracefully during development
    import warnings

    warnings.warn(f"Streaming service imports failed: {e}", stacklevel=2)


# Delayed imports to avoid circular dependencies
def _get_camera_service():
    """Lazy import of CameraService to avoid circular dependencies."""
    from .camera_service import CameraService

    return CameraService


def _get_frame_services():
    """Lazy import of frame services to avoid circular dependencies."""
    from .frame_service import DetectionService, FrameService

    return DetectionService, FrameService


# Provide access to services through module attributes
def __getattr__(name: str):
    """Dynamic attribute access for lazy-loaded services."""
    if name == "CameraService":
        return _get_camera_service()
    elif name == "FrameService":
        detection_service, frame_service = _get_frame_services()
        return frame_service
    elif name == "DetectionService":
        detection_service, frame_service = _get_frame_services()
        return detection_service
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Base service
    "BaseAsyncService",
    # Authentication and caching
    "AuthService",
    "CacheService",
    # Database services (lazy-loaded)
    "CameraService",
    "FrameService",
    "DetectionService",
    "MetricsService",
    # Streaming services
    "StreamingDataProcessor",
    "StreamingServiceImpl",
    "StreamingServer",
    "CameraConfig",
    "StreamProtocol",
    "FrameQualityValidator",
    "CameraConnectionManager",
    "StreamingServiceInterface",
    "StreamingContainer",
    "create_streaming_container",
    "initialize_streaming_container",
]
