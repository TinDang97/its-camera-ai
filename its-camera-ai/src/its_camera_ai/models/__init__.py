"""Database models for ITS Camera AI system.

Provides SQLAlchemy 2.0 models for high-throughput camera registry,
frame metadata processing, detection results, and system monitoring.
"""

from .analytics import (
    AlertNotification,
    AnomalyType,
    CongestionLevel,
    RuleViolation,
    TrafficAnomaly,
    TrafficMetrics,
    VehicleTrajectory,
    ViolationType,
)
from .base import BaseTableModel
from .camera import Camera, CameraSettings, CameraStatus, CameraType, StreamProtocol
from .database import DatabaseManager
from .detection_result import DetectionClass, DetectionResult, VehicleType
from .frame_metadata import FrameMetadata, FrameQuality, ProcessingStatus
from .system_metrics import AggregatedMetrics, MetricType, MetricUnit, SystemMetrics
from .user import User

__all__ = [
    # Base infrastructure
    "BaseTableModel",
    "DatabaseManager",
    # User management
    "User",
    # Camera registry
    "Camera",
    "CameraSettings",
    "CameraStatus",
    "CameraType",
    "StreamProtocol",
    # Frame processing
    "FrameMetadata",
    "FrameQuality",
    "ProcessingStatus",
    # Detection results
    "DetectionResult",
    "DetectionClass",
    "VehicleType",
    # System monitoring
    "SystemMetrics",
    "AggregatedMetrics",
    "MetricType",
    "MetricUnit",
    # Analytics models
    "TrafficMetrics",
    "RuleViolation",
    "VehicleTrajectory",
    "TrafficAnomaly",
    "AlertNotification",
    "ViolationType",
    "AnomalyType",
    "CongestionLevel",
]
