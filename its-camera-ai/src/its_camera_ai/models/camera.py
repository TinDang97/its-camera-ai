"""Camera registry models for SQLAlchemy 2.0.

Provides comprehensive camera registry management with relationships,
indexing, and optimizations for high-throughput operations.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from its_camera_ai.core.types.geo_json import GeoJSONFeature

from .base import BaseTableModel


class CameraType(str, Enum):
    """Camera type enumeration."""

    FIXED = "fixed"
    PTZ = "ptz"  # Pan-Tilt-Zoom
    DOME = "dome"
    BULLET = "bullet"
    THERMAL = "thermal"


class CameraStatus(str, Enum):
    """Camera status enumeration."""

    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    STREAMING = "streaming"
    STOPPED = "stopped"


class StreamProtocol(str, Enum):
    """Stream protocol enumeration."""

    RTSP = "rtsp"
    RTMP = "rtmp"
    HLS = "hls"
    WEBRTC = "webrtc"
    HTTP = "http"


class Camera(BaseTableModel):
    """Camera registry model with comprehensive metadata and settings.

    Supports 100+ concurrent cameras with optimized indexing for
    high-performance queries and real-time status updates.
    """

    # Basic Information
    name: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, comment="Camera display name"
    )
    description: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Camera description"
    )
    location: Mapped[str] = mapped_column(
        String(200), nullable=False, index=True, comment="Camera location"
    )

    # GPS Coordinates (JSONB for flexible geo-queries)
    coordinates: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="GPS coordinates {lat, lng, altitude}"
    )

    # Camera Configuration
    camera_type: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True, comment="Camera hardware type"
    )

    # Stream Configuration
    stream_url: Mapped[str] = mapped_column(
        String(500), nullable=False, comment="Primary stream URL"
    )
    stream_protocol: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="Stream protocol"
    )
    backup_stream_url: Mapped[str | None] = mapped_column(
        String(500), nullable=True, comment="Backup stream URL"
    )

    # Authentication
    username: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Stream authentication username"
    )
    password: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="Encrypted stream password"
    )

    # Status and Health
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=CameraStatus.OFFLINE,
        index=True,
        comment="Current camera status",
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
        comment="Camera enabled status",
    )

    # Configuration (JSONB for flexible settings)
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, comment="Camera configuration settings"
    )

    # Zone and Tagging
    zone_id: Mapped[str | None] = mapped_column(
        String(50), nullable=True, index=True, comment="Traffic zone identifier"
    )
    tags: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, comment="Camera tags for filtering"
    )

    # Health Metrics
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="Last successful connection",
    )
    last_frame_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Last frame received timestamp"
    )

    # Performance Metrics (updated periodically)
    total_frames_processed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Total frames processed counter"
    )
    uptime_percentage: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Uptime percentage (last 24h)"
    )
    avg_processing_time: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Average processing time in ms"
    )

    # Calibration Data
    calibration: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Camera calibration parameters"
    )

    # Detection Zones
    detection_zones: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list, comment="Detection zone definitions"
    )

    # Relationships
    frame_metadata = relationship(
        "FrameMetadata",
        back_populates="camera",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select",
    )

    settings = relationship(
        "CameraSettings",
        back_populates="camera",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )

    # Indexes for performance optimization
    __table_args__ = (
        # Composite indexes for common query patterns
        Index("idx_camera_status_active", "status", "is_active"),
        Index("idx_camera_zone_active", "zone_id", "is_active"),
        Index("idx_camera_last_seen", "last_seen_at"),
        Index("idx_camera_location_type", "location", "camera_type"),
        # GIN index for JSONB columns (tags, config)
        Index("idx_camera_tags_gin", "tags", postgresql_using="gin"),
        Index("idx_camera_config_gin", "config", postgresql_using="gin"),
        Index("idx_camera_coordinates_gin", "coordinates", postgresql_using="gin"),
        {"comment": "Camera registry with optimizations for high-throughput queries"},
    )

    def update_health_metrics(
        self, frame_count: int, avg_time: float, uptime: float
    ) -> None:
        """Update camera health metrics.

        Args:
            frame_count: Total frames processed
            avg_time: Average processing time in ms
            uptime: Uptime percentage
        """
        self.total_frames_processed = frame_count
        self.avg_processing_time = avg_time
        self.uptime_percentage = uptime
        self.last_seen_at = datetime.now(UTC)

    def set_status(
        self, status: CameraStatus, error_message: str | None = None
    ) -> None:
        """Update camera status with optional error information.

        Args:
            status: New camera status
            error_message: Optional error message for troubleshooting
        """
        self.status = status.value
        if status in (CameraStatus.ONLINE, CameraStatus.STREAMING):
            self.last_seen_at = datetime.now(UTC)

        if error_message and "config" in self.__dict__:
            if "errors" not in self.config:
                self.config["errors"] = []
            self.config["errors"].append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "message": error_message,
                    "status": status.value,
                }
            )
            # Keep only last 10 errors
            self.config["errors"] = self.config["errors"][-10:]

    def to_geo_json(self) -> GeoJSONFeature:
        """Convert camera metadata to GeoJSON format."""
        coord_data = self.coordinates or {"lat": 0.0, "lng": 0.0, "altitude": 0.0}
        lat = float(coord_data.get("lat", 0.0))
        long = float(coord_data.get("lng", 0.0))
        altitude = float(coord_data.get("altitude", 0.0))

        return {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [long, lat, altitude]},
            "properties": {
                "id": self.id,
                "name": self.name,
                "status": self.status,
                "location": self.location,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "altitude": altitude,
            },
        }

    @property
    def is_streaming(self) -> bool:
        """Check if camera is actively streaming."""
        return self.status == CameraStatus.STREAMING.value

    @property
    def is_healthy(self) -> bool:
        """Check if camera is healthy and operational."""
        return (
            self.is_active
            and self.status in (CameraStatus.ONLINE.value, CameraStatus.STREAMING.value)
            and self.uptime_percentage is not None
            and self.uptime_percentage > 0.8  # 80% uptime threshold
        )

    def __repr__(self) -> str:
        return f"<Camera(id={self.id}, name={self.name}, status={self.status})>"


class CameraSettings(BaseTableModel):
    """Camera-specific processing and configuration settings.

    Separated from Camera model for better performance and flexibility
    in high-frequency updates during processing optimization.
    """

    # Foreign key to camera
    camera_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), nullable=False, index=True, comment="Reference to camera"
    )

    # Processing Settings
    detection_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, comment="Enable vehicle detection"
    )
    tracking_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, comment="Enable object tracking"
    )
    analytics_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, comment="Enable traffic analytics"
    )

    # Model Configuration
    model_name: Mapped[str] = mapped_column(
        String(100), nullable=False, default="yolo11n", comment="YOLO model variant"
    )
    confidence_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.5, comment="Detection confidence threshold"
    )
    nms_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.4, comment="Non-maximum suppression threshold"
    )

    # Processing Parameters
    max_batch_size: Mapped[int] = mapped_column(
        Integer, nullable=False, default=8, comment="Maximum batch size for inference"
    )
    frame_skip: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of frames to skip (0=process all)",
    )
    resize_resolution: Mapped[dict[str, int] | None] = mapped_column(
        JSONB, nullable=True, comment="Target resolution for processing {width, height}"
    )

    # Quality and Performance
    quality_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.7, comment="Minimum frame quality score"
    )
    max_processing_time: Mapped[int] = mapped_column(
        Integer, nullable=False, default=100, comment="Max processing time in ms"
    )

    # Recording and Storage
    record_detections_only: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Record only frames with detections",
    )
    storage_retention_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=7, comment="Frame storage retention in days"
    )

    # Alert Configuration
    alert_thresholds: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, comment="Alert threshold configuration"
    )
    notification_settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, comment="Notification settings"
    )

    # Advanced Settings (JSONB for flexibility)
    advanced_settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, comment="Advanced processing settings"
    )

    # Relationship
    camera = relationship("Camera", back_populates="settings")

    # Indexes for settings queries
    __table_args__ = (
        Index("idx_camera_settings_camera_id", "camera_id"),
        Index("idx_camera_settings_model", "model_name"),
        Index("idx_camera_settings_detection", "detection_enabled"),
        {"comment": "Camera-specific processing and configuration settings"},
    )

    def update_performance_settings(self, avg_processing_time: float) -> None:
        """Auto-adjust settings based on performance metrics.

        Args:
            avg_processing_time: Current average processing time in ms
        """
        # Auto-adjust batch size based on processing time
        if avg_processing_time > self.max_processing_time:
            if self.max_batch_size > 1:
                self.max_batch_size = max(1, self.max_batch_size - 1)
            elif self.frame_skip < 3:
                self.frame_skip += 1
        elif avg_processing_time < self.max_processing_time * 0.7:
            if self.frame_skip > 0:
                self.frame_skip = max(0, self.frame_skip - 1)
            elif self.max_batch_size < 16:
                self.max_batch_size = min(16, self.max_batch_size + 1)

    def __repr__(self) -> str:
        return f"<CameraSettings(camera_id={self.camera_id}, model={self.model_name})>"
