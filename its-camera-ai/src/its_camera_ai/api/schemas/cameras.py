"""Camera management schemas."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .common import Coordinates


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


class Resolution(BaseModel):
    """Video resolution schema."""

    width: int = Field(description="Frame width in pixels", gt=0)
    height: int = Field(description="Frame height in pixels", gt=0)

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


class CameraConfig(BaseModel):
    """Camera configuration schema."""

    resolution: Resolution = Field(description="Video resolution")
    fps: float = Field(description="Frames per second", gt=0, le=60)
    bitrate: int | None = Field(None, description="Video bitrate in kbps", gt=0)
    quality: int = Field(description="Video quality (1-10)", ge=1, le=10, default=8)
    night_vision: bool = Field(False, description="Enable night vision")
    motion_detection: bool = Field(True, description="Enable motion detection")
    recording_enabled: bool = Field(True, description="Enable video recording")
    retention_days: int = Field(
        description="Video retention period in days", ge=1, le=365, default=30
    )
    analytics_enabled: bool = Field(True, description="Enable AI analytics")
    alerts_enabled: bool = Field(True, description="Enable alert notifications")


class CameraCreate(BaseModel):
    """Camera creation request schema."""

    name: str = Field(description="Camera display name", max_length=100)
    description: str | None = Field(
        None, description="Camera description", max_length=500
    )
    location: str = Field(description="Camera location description", max_length=200)
    coordinates: Coordinates | None = Field(None, description="GPS coordinates")
    camera_type: CameraType = Field(description="Camera type")
    stream_url: str = Field(description="Primary stream URL")
    stream_protocol: StreamProtocol = Field(description="Stream protocol")
    backup_stream_url: str | None = Field(None, description="Backup stream URL")
    username: str | None = Field(None, description="Stream authentication username")
    password: str | None = Field(None, description="Stream authentication password")
    config: CameraConfig = Field(description="Camera configuration")
    zone_id: str | None = Field(None, description="Traffic zone ID")
    tags: list[str] = Field(default_factory=list, description="Camera tags")

    @field_validator("stream_url")
    @classmethod
    def validate_stream_url(cls, v: str) -> str:
        """Validate stream URL format."""
        if not v.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            raise ValueError("Stream URL must use a valid protocol")
        return v


class CameraUpdate(BaseModel):
    """Camera update request schema."""

    name: str | None = Field(None, description="Camera display name", max_length=100)
    description: str | None = Field(
        None, description="Camera description", max_length=500
    )
    location: str | None = Field(
        None, description="Camera location description", max_length=200
    )
    coordinates: Coordinates | None = Field(None, description="GPS coordinates")
    stream_url: str | None = Field(None, description="Primary stream URL")
    backup_stream_url: str | None = Field(None, description="Backup stream URL")
    username: str | None = Field(None, description="Stream authentication username")
    password: str | None = Field(None, description="Stream authentication password")
    config: CameraConfig | None = Field(None, description="Camera configuration")
    zone_id: str | None = Field(None, description="Traffic zone ID")
    tags: list[str] | None = Field(None, description="Camera tags")
    is_active: bool | None = Field(None, description="Camera active status")

    @field_validator("stream_url")
    @classmethod
    def validate_stream_url(cls, v: str | None) -> str | None:
        """Validate stream URL format."""
        if v and not v.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            raise ValueError("Stream URL must use a valid protocol")
        return v


class StreamHealth(BaseModel):
    """Stream health status schema."""

    is_connected: bool = Field(description="Stream connection status")
    bitrate: float | None = Field(None, description="Current bitrate in kbps")
    fps: float | None = Field(None, description="Current frames per second")
    packet_loss: float | None = Field(
        None, description="Packet loss percentage", ge=0, le=100
    )
    latency: float | None = Field(None, description="Stream latency in ms")
    last_frame_time: datetime | None = Field(
        None, description="Timestamp of last received frame"
    )
    error_message: str | None = Field(None, description="Last error message")
    uptime: float | None = Field(None, description="Stream uptime in seconds")


class CameraResponse(BaseModel):
    """Camera response schema."""

    id: str = Field(description="Camera ID")
    name: str = Field(description="Camera display name")
    description: str | None = Field(None, description="Camera description")
    location: str = Field(description="Camera location")
    coordinates: Coordinates | None = Field(None, description="GPS coordinates")
    camera_type: CameraType = Field(description="Camera type")
    stream_url: str = Field(description="Primary stream URL")
    stream_protocol: StreamProtocol = Field(description="Stream protocol")
    backup_stream_url: str | None = Field(None, description="Backup stream URL")
    status: CameraStatus = Field(description="Current camera status")
    config: CameraConfig = Field(description="Camera configuration")
    health: StreamHealth | None = Field(None, description="Stream health metrics")
    zone_id: str | None = Field(None, description="Traffic zone ID")
    tags: list[str] = Field(description="Camera tags")
    is_active: bool = Field(description="Camera active status")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    last_seen_at: datetime | None = Field(
        None, description="Last successful connection timestamp"
    )

    class Config:
        from_attributes = True


class StreamRequest(BaseModel):
    """Stream control request schema."""

    action: str = Field(description="Stream action (start/stop/restart)")
    quality: int | None = Field(
        None, description="Stream quality override", ge=1, le=10
    )
    record: bool | None = Field(None, description="Enable recording for this stream")

    @field_validator("action")
    @classmethod
    def validate_action(cls, v: str) -> str:
        """Validate stream action."""
        if v not in ["start", "stop", "restart"]:
            raise ValueError("Action must be 'start', 'stop', or 'restart'")
        return v


class CameraBatchOperation(BaseModel):
    """Batch operation request schema."""

    camera_ids: list[str] = Field(description="List of camera IDs")
    operation: str = Field(description="Operation to perform")
    parameters: dict[str, Any] | None = Field(None, description="Operation parameters")

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate batch operation."""
        valid_ops = [
            "start_streams",
            "stop_streams",
            "restart_streams",
            "update_config",
            "enable_analytics",
            "disable_analytics",
            "delete",
        ]
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {', '.join(valid_ops)}")
        return v


class CameraBatchResult(BaseModel):
    """Batch operation result schema."""

    camera_id: str = Field(description="Camera ID")
    success: bool = Field(description="Operation success status")
    message: str | None = Field(None, description="Success/error message")
    details: dict[str, Any] | None = Field(None, description="Additional details")


class CameraStats(BaseModel):
    """Camera statistics schema."""

    camera_id: str = Field(description="Camera ID")
    frames_processed: int = Field(description="Total frames processed")
    vehicles_detected: int = Field(description="Total vehicles detected")
    incidents_detected: int = Field(description="Total incidents detected")
    uptime_percentage: float = Field(description="Uptime percentage", ge=0, le=100)
    avg_processing_time: float = Field(description="Average processing time in ms")
    last_24h_activity: dict[str, int] = Field(
        description="Hourly activity for last 24 hours"
    )
    period_start: datetime = Field(description="Statistics period start")
    period_end: datetime = Field(description="Statistics period end")


class CameraCalibration(BaseModel):
    """Camera calibration data schema."""

    camera_matrix: list[list[float]] = Field(description="3x3 camera intrinsic matrix")
    distortion_coefficients: list[float] = Field(
        description="Lens distortion coefficients"
    )
    rotation_vector: list[float] | None = Field(
        None, description="Rotation vector for pose estimation"
    )
    translation_vector: list[float] | None = Field(
        None, description="Translation vector for pose estimation"
    )
    reference_points: list[dict[str, float]] | None = Field(
        None, description="Reference points for perspective correction"
    )


class CameraZone(BaseModel):
    """Camera detection zone schema."""

    name: str = Field(description="Zone name", max_length=100)
    polygon: list[dict[str, float]] = Field(
        description="Zone boundary as list of {x, y} points"
    )
    zone_type: str = Field(description="Zone type (detection/exclusion/roi)")
    is_active: bool = Field(True, description="Zone active status")
    detection_classes: list[str] | None = Field(
        None, description="Vehicle classes to detect in this zone"
    )
    sensitivity: float = Field(
        description="Detection sensitivity", ge=0.1, le=1.0, default=0.7
    )

    @field_validator("zone_type")
    @classmethod
    def validate_zone_type(cls, v: str) -> str:
        """Validate zone type."""
        if v not in ["detection", "exclusion", "roi"]:
            raise ValueError("Zone type must be 'detection', 'exclusion', or 'roi'")
        return v
