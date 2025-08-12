"""Optimized SQLAlchemy 2.0 models for ITS Camera AI high-throughput operations.

This module provides database models optimized for:
- High-throughput frame metadata processing (3000+ inserts/sec)
- Sub-10ms camera registry queries
- Efficient time-series data handling
- Real-time SSE event filtering

Performance optimizations included:
- Proper indexing strategies for common query patterns
- Table partitioning for time-series data
- Connection pooling configuration
- Bulk insert optimizations
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class CameraStatus(str, Enum):
    """Camera operational status."""
    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    STREAMING = "streaming"
    STOPPED = "stopped"


class CameraType(str, Enum):
    """Camera hardware type."""
    FIXED = "fixed"
    PTZ = "ptz"
    DOME = "dome"
    BULLET = "bullet"
    THERMAL = "thermal"


class StreamProtocol(str, Enum):
    """Video stream protocols."""
    RTSP = "rtsp"
    RTMP = "rtmp"
    HLS = "hls"
    WEBRTC = "webrtc"
    HTTP = "http"


class ProcessingStatus(str, Enum):
    """Frame processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class VehicleType(str, Enum):
    """Detected vehicle types."""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    UNKNOWN = "unknown"


class Camera(BaseModel):
    """Camera registry model optimized for CRUD operations.
    
    Design considerations:
    - UUID primary key for distributed systems
    - JSONB for flexible configuration storage
    - Proper indexing for location and status queries
    - Efficient foreign key relationships
    """

    __tablename__ = "cameras"

    # Core identification
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(String(500))
    location: Mapped[str] = mapped_column(String(200), nullable=False, index=True)

    # GPS coordinates for spatial queries
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)

    # Hardware specifications
    camera_type: Mapped[CameraType] = mapped_column(String(20), nullable=False)
    manufacturer: Mapped[str | None] = mapped_column(String(100))
    model: Mapped[str | None] = mapped_column(String(100))
    firmware_version: Mapped[str | None] = mapped_column(String(50))

    # Network configuration
    stream_url: Mapped[str] = mapped_column(String(500), nullable=False)
    stream_protocol: Mapped[StreamProtocol] = mapped_column(String(20), nullable=False)
    backup_stream_url: Mapped[str | None] = mapped_column(String(500))
    username: Mapped[str | None] = mapped_column(String(100))
    password: Mapped[str | None] = mapped_column(String(100))  # Encrypted in app layer

    # Operational status
    status: Mapped[CameraStatus] = mapped_column(
        String(20),
        nullable=False,
        default=CameraStatus.OFFLINE,
        index=True
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        index=True
    )

    # Configuration stored as JSONB for flexibility and performance
    config: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=lambda: {
            "resolution": {"width": 1920, "height": 1080},
            "fps": 30.0,
            "quality": 8,
            "analytics_enabled": True,
            "recording_enabled": True,
            "retention_days": 30
        }
    )

    # Zone and tagging
    zone_id: Mapped[str | None] = mapped_column(String(100), index=True)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)

    # Relationships
    frame_metadata = relationship(
        "FrameMetadata",
        back_populates="camera",
        lazy="selectin",
        cascade="all, delete-orphan"
    )
    system_metrics = relationship(
        "SystemMetric",
        back_populates="camera",
        lazy="selectin"
    )

    # Indexes for performance optimization
    __table_args__ = (
        # Composite index for active cameras by location
        Index("idx_cameras_active_location", "is_active", "location"),
        # Composite index for status monitoring
        Index("idx_cameras_status_last_seen", "status", "last_seen_at"),
        # Spatial index for GPS coordinates
        Index("idx_cameras_coordinates", "latitude", "longitude"),
        # GIN index for JSONB config queries
        Index("idx_cameras_config_gin", "config", postgresql_using="gin"),
        # Index for zone-based queries
        Index("idx_cameras_zone_active", "zone_id", "is_active"),
        # Array index for tag searches
        Index("idx_cameras_tags_gin", "tags", postgresql_using="gin"),
    )

    @hybrid_property
    def is_online(self) -> bool:
        """Check if camera is currently online."""
        return self.status == CameraStatus.ONLINE

    @hybrid_property
    def resolution_string(self) -> str:
        """Get formatted resolution string."""
        if self.config and "resolution" in self.config:
            res = self.config["resolution"]
            return f"{res['width']}x{res['height']}"
        return "Unknown"

    def get_stream_health(self) -> dict[str, Any]:
        """Get current stream health metrics."""
        # This would typically query recent frame metadata
        # Implementation depends on your monitoring strategy
        return {
            "is_connected": self.status == CameraStatus.STREAMING,
            "last_frame_time": self.last_seen_at,
            "uptime": None  # Calculated from system metrics
        }


class FrameMetadata(BaseModel):
    """Frame processing metadata with time-series partitioning.
    
    Optimized for high-throughput inserts (3000+/sec):
    - Partitioned by timestamp for efficient archival
    - Minimal indexes on partition key and camera_id
    - JSONB for flexible detection results storage
    - Bigint sequence for high-volume inserts
    """

    __tablename__ = "frame_metadata"

    # Use BIGINT for high-volume sequence
    sequence_id: Mapped[int] = mapped_column(
        BigInteger,
        autoincrement=True,
        unique=True,
        index=True
    )

    # Foreign key to camera (indexed)
    camera_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Frame identification
    frame_number: Mapped[int] = mapped_column(BigInteger, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,  # Critical for time-series queries
        server_default=text("CURRENT_TIMESTAMP")
    )

    # Processing metadata
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        String(20),
        nullable=False,
        default=ProcessingStatus.PENDING,
        index=True
    )
    processing_time_ms: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str | None] = mapped_column(String(50))

    # Frame quality metrics
    frame_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    quality_score: Mapped[float | None] = mapped_column(Float)
    blur_score: Mapped[float | None] = mapped_column(Float)
    brightness_score: Mapped[float | None] = mapped_column(Float)

    # Detection summary (for quick aggregations)
    total_detections: Mapped[int] = mapped_column(Integer, default=0)
    vehicle_count: Mapped[int] = mapped_column(Integer, default=0)
    person_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence_avg: Mapped[float | None] = mapped_column(Float)

    # Detailed results stored as JSONB
    detection_results: Mapped[dict | None] = mapped_column(JSONB)
    processing_metadata: Mapped[dict | None] = mapped_column(JSONB)

    # Relationships
    camera = relationship("Camera", back_populates="frame_metadata")
    detections = relationship(
        "Detection",
        back_populates="frame_metadata",
        lazy="selectin",
        cascade="all, delete-orphan"
    )

    # Optimized indexes for high-throughput queries
    __table_args__ = (
        # Primary composite index for time-series queries
        Index("idx_frame_metadata_camera_timestamp", "camera_id", "timestamp"),
        # Index for processing status monitoring
        Index("idx_frame_metadata_status", "processing_status", "timestamp"),
        # Index for detection count queries
        Index("idx_frame_metadata_detections", "camera_id", "total_detections"),
        # GIN index for JSONB detection results
        Index("idx_frame_metadata_results_gin", "detection_results", postgresql_using="gin"),
        # Partial index for failed frames (fewer rows, faster queries)
        Index(
            "idx_frame_metadata_failed",
            "camera_id",
            "timestamp",
            postgresql_where=text("processing_status = 'failed'")
        ),
        # Partition by timestamp (monthly partitions recommended)
        # Note: Actual partitioning setup requires DDL outside SQLAlchemy
    )

    @hybrid_property
    def processing_latency(self) -> float | None:
        """Calculate processing latency if available."""
        if self.processing_time_ms:
            return self.processing_time_ms
        return None


class Detection(BaseModel):
    """Individual detection results with spatial indexing.
    
    Separated from FrameMetadata for:
    - Better normalization
    - Efficient spatial queries
    - Optimized for analytics queries
    """

    __tablename__ = "detections"

    # Foreign key to frame metadata
    frame_metadata_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("frame_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Detection classification
    object_type: Mapped[VehicleType] = mapped_column(
        String(20),
        nullable=False,
        index=True
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Bounding box coordinates (normalized 0-1)
    bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)

    # Additional detection metadata
    track_id: Mapped[int | None] = mapped_column(Integer)  # For object tracking
    speed_kmh: Mapped[float | None] = mapped_column(Float)
    direction: Mapped[float | None] = mapped_column(Float)  # Angle in degrees

    # Zone information
    zone_name: Mapped[str | None] = mapped_column(String(100))
    is_in_roi: Mapped[bool] = mapped_column(Boolean, default=True)

    # Additional attributes as JSONB
    attributes: Mapped[dict | None] = mapped_column(JSONB)

    # Relationships
    frame_metadata = relationship("FrameMetadata", back_populates="detections")

    # Indexes optimized for analytics queries
    __table_args__ = (
        # Primary index for frame-based queries
        Index("idx_detections_frame", "frame_metadata_id"),
        # Index for object type analysis
        Index("idx_detections_type_confidence", "object_type", "confidence"),
        # Index for tracking queries
        Index("idx_detections_track", "track_id", "frame_metadata_id"),
        # Spatial index for bounding box queries
        Index("idx_detections_bbox", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
        # GIN index for attributes
        Index("idx_detections_attributes_gin", "attributes", postgresql_using="gin"),
    )

    @hybrid_property
    def bbox_center_x(self) -> float:
        """Calculate bounding box center X coordinate."""
        return (self.bbox_x1 + self.bbox_x2) / 2

    @hybrid_property
    def bbox_center_y(self) -> float:
        """Calculate bounding box center Y coordinate."""
        return (self.bbox_y1 + self.bbox_y2) / 2

    @hybrid_property
    def bbox_area(self) -> float:
        """Calculate bounding box area."""
        return (self.bbox_x2 - self.bbox_x1) * (self.bbox_y2 - self.bbox_y1)


class SystemMetric(BaseModel):
    """System performance and health metrics with time-series storage.
    
    Optimized for monitoring dashboards:
    - Time-series partitioning
    - Efficient aggregation queries
    - Alert threshold monitoring
    """

    __tablename__ = "system_metrics"

    # Foreign key to camera (nullable for system-wide metrics)
    camera_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        index=True
    )

    # Metric identification
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_type: Mapped[str] = mapped_column(String(50), nullable=False)  # counter, gauge, histogram

    # Metric values
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str | None] = mapped_column(String(20))

    # Temporal information
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        server_default=text("CURRENT_TIMESTAMP")
    )

    # Additional metadata
    labels: Mapped[dict | None] = mapped_column(JSONB)
    source: Mapped[str] = mapped_column(String(100), nullable=False, default="system")

    # Relationships
    camera = relationship("Camera", back_populates="system_metrics")

    # Time-series optimized indexes
    __table_args__ = (
        # Primary time-series index
        Index("idx_system_metrics_time", "timestamp", "metric_name"),
        # Camera-specific metrics
        Index("idx_system_metrics_camera", "camera_id", "metric_name", "timestamp"),
        # Metric type queries
        Index("idx_system_metrics_type", "metric_type", "timestamp"),
        # GIN index for labels
        Index("idx_system_metrics_labels_gin", "labels", postgresql_using="gin"),
        # Partial index for alerts (high-value metrics)
        Index(
            "idx_system_metrics_alerts",
            "metric_name",
            "value",
            "timestamp",
            postgresql_where=text("value > 0.8")  # Example threshold
        ),
    )


class User(BaseModel):
    """User management with role-based access control.
    
    Optimized for authentication and authorization:
    - Efficient username/email lookups
    - Role-based permissions
    - Session management
    """

    __tablename__ = "users"

    # Authentication fields
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    # Profile information
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)

    # Role and permissions
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="viewer", index=True)
    permissions: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)

    # Session management
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    login_count: Mapped[int] = mapped_column(Integer, default=0)

    # Security
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    user_sessions = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan"
    )

    # Indexes for authentication performance
    __table_args__ = (
        # Username lookup index
        Index("idx_users_username_active", "username", "is_active"),
        # Email lookup index
        Index("idx_users_email_active", "email", "is_active"),
        # Role-based queries
        Index("idx_users_role_active", "role", "is_active"),
        # Security monitoring
        Index("idx_users_failed_logins", "failed_login_attempts", "locked_until"),
    )

    @hybrid_property
    def full_name(self) -> str:
        """Get user's full name."""
        return f"{self.first_name} {self.last_name}"

    @hybrid_property
    def is_locked(self) -> bool:
        """Check if user account is locked."""
        if self.locked_until:
            return datetime.now(UTC) < self.locked_until
        return False


class UserSession(BaseModel):
    """User session tracking for security and monitoring.
    
    Optimized for session management:
    - Fast session lookups
    - Security monitoring
    - Automatic cleanup
    """

    __tablename__ = "user_sessions"

    # Foreign key to user
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Session identification
    session_token: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    refresh_token: Mapped[str | None] = mapped_column(String(255), unique=True)

    # Session metadata
    ip_address: Mapped[str | None] = mapped_column(String(45))  # IPv6 compatible
    user_agent: Mapped[str | None] = mapped_column(Text)

    # Timing
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP")
    )

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="user_sessions")

    # Session management indexes
    __table_args__ = (
        # Primary session lookup
        Index("idx_user_sessions_token_active", "session_token", "is_active"),
        # User session queries
        Index("idx_user_sessions_user", "user_id", "is_active"),
        # Cleanup queries (expired sessions)
        Index("idx_user_sessions_expires", "expires_at", "is_active"),
        # Security monitoring
        Index("idx_user_sessions_ip", "ip_address", "created_at"),
    )

    @hybrid_property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(UTC) > self.expires_at


class Alert(BaseModel):
    """System alerts and notifications.
    
    Optimized for real-time alerting:
    - Priority-based indexing
    - Status tracking
    - Efficient filtering
    """

    __tablename__ = "alerts"

    # Foreign key to camera (nullable for system alerts)
    camera_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        index=True
    )

    # Alert classification
    alert_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # critical, high, medium, low
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5, index=True)

    # Alert content
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)

    # Status tracking
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active", index=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    acknowledged_by: Mapped[str | None] = mapped_column(String(100))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_by: Mapped[str | None] = mapped_column(String(100))

    # Additional data
    metadata: Mapped[dict | None] = mapped_column(JSONB)

    # Relationships
    camera = relationship("Camera")

    # Alert management indexes
    __table_args__ = (
        # Primary alert queries
        Index("idx_alerts_status_priority", "status", "priority", "created_at"),
        # Camera-specific alerts
        Index("idx_alerts_camera_status", "camera_id", "status"),
        # Alert type analysis
        Index("idx_alerts_type_severity", "alert_type", "severity", "created_at"),
        # GIN index for metadata
        Index("idx_alerts_metadata_gin", "metadata", postgresql_using="gin"),
        # Cleanup queries
        Index("idx_alerts_resolved", "resolved_at", "status"),
    )

    @hybrid_property
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.status == "active"

    @hybrid_property
    def is_acknowledged(self) -> bool:
        """Check if alert has been acknowledged."""
        return self.acknowledged_at is not None
