"""Database model Pydantic schemas for API serialization.

Provides comprehensive schemas for all database models with proper
validation, serialization, and integration with FastAPI endpoints.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...models import (
    CameraType,
    MetricType,
    ProcessingStatus,
    StreamProtocol,
)
from .common import Coordinates


# Camera Registry Schemas
class CameraConfigSchema(BaseModel):
    """Camera configuration schema."""

    model_config = ConfigDict(from_attributes=True)

    resolution: dict[str, int] = Field(description="Video resolution {width, height}")
    fps: float = Field(description="Frames per second", gt=0, le=120)
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


class CameraCreateSchema(BaseModel):
    """Schema for creating a new camera."""

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(description="Camera display name", max_length=100)
    description: str | None = Field(
        None, description="Camera description", max_length=500
    )
    location: str = Field(description="Camera location description", max_length=200)
    coordinates: Coordinates | None = Field(None, description="GPS coordinates")
    camera_type: CameraType = Field(description="Camera hardware type")
    stream_url: str = Field(description="Primary stream URL")
    stream_protocol: StreamProtocol = Field(description="Stream protocol")
    backup_stream_url: str | None = Field(None, description="Backup stream URL")
    username: str | None = Field(None, description="Stream authentication username")
    password: str | None = Field(None, description="Stream authentication password")
    config: CameraConfigSchema = Field(description="Camera configuration")
    zone_id: str | None = Field(None, description="Traffic zone ID")
    tags: list[str] = Field(default_factory=list, description="Camera tags")

    @field_validator("stream_url", "backup_stream_url")
    @classmethod
    def validate_stream_url(cls, v: str | None) -> str | None:
        """Validate stream URL format."""
        if v and not v.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            raise ValueError("Stream URL must use a valid protocol")
        return v


class CameraUpdateSchema(BaseModel):
    """Schema for updating camera information."""

    model_config = ConfigDict(from_attributes=True)

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
    config: CameraConfigSchema | None = Field(None, description="Camera configuration")
    zone_id: str | None = Field(None, description="Traffic zone ID")
    tags: list[str] | None = Field(None, description="Camera tags")
    is_active: bool | None = Field(None, description="Camera active status")

    @field_validator("stream_url", "backup_stream_url")
    @classmethod
    def validate_stream_url(cls, v: str | None) -> str | None:
        """Validate stream URL format."""
        if v and not v.startswith(("rtsp://", "rtmp://", "http://", "https://")):
            raise ValueError("Stream URL must use a valid protocol")
        return v


class CameraResponseSchema(BaseModel):
    """Schema for camera API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Camera unique identifier")
    name: str = Field(description="Camera display name")
    description: str | None = Field(None, description="Camera description")
    location: str = Field(description="Camera location")
    coordinates: dict[str, Any] | None = Field(None, description="GPS coordinates")
    camera_type: str = Field(description="Camera hardware type")
    stream_url: str = Field(description="Primary stream URL")
    stream_protocol: str = Field(description="Stream protocol")
    backup_stream_url: str | None = Field(None, description="Backup stream URL")
    status: str = Field(description="Current camera status")
    config: dict[str, Any] = Field(description="Camera configuration")
    zone_id: str | None = Field(None, description="Traffic zone ID")
    tags: list[str] = Field(description="Camera tags")
    is_active: bool = Field(description="Camera active status")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    last_seen_at: datetime | None = Field(
        None, description="Last successful connection timestamp"
    )
    last_frame_at: datetime | None = Field(
        None, description="Last frame received timestamp"
    )
    total_frames_processed: int = Field(description="Total frames processed counter")
    uptime_percentage: float | None = Field(
        None, description="Uptime percentage (last 24h)"
    )
    avg_processing_time: float | None = Field(
        None, description="Average processing time in ms"
    )


class CameraSettingsSchema(BaseModel):
    """Schema for camera processing settings."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Settings unique identifier")
    camera_id: str = Field(description="Reference to camera")
    detection_enabled: bool = Field(description="Enable vehicle detection")
    tracking_enabled: bool = Field(description="Enable object tracking")
    analytics_enabled: bool = Field(description="Enable traffic analytics")
    model_name: str = Field(description="YOLO model variant")
    confidence_threshold: float = Field(
        description="Detection confidence threshold", ge=0.0, le=1.0
    )
    nms_threshold: float = Field(
        description="Non-maximum suppression threshold", ge=0.0, le=1.0
    )
    max_batch_size: int = Field(
        description="Maximum batch size for inference", ge=1, le=128
    )
    frame_skip: int = Field(
        description="Number of frames to skip (0=process all)", ge=0
    )
    resize_resolution: dict[str, int] | None = Field(
        None, description="Target resolution for processing {width, height}"
    )
    quality_threshold: float = Field(
        description="Minimum frame quality score", ge=0.0, le=1.0
    )
    max_processing_time: int = Field(
        description="Max processing time in ms", gt=0
    )
    record_detections_only: bool = Field(
        description="Record only frames with detections"
    )
    storage_retention_days: int = Field(
        description="Frame storage retention in days", ge=1
    )
    alert_thresholds: dict[str, Any] = Field(
        description="Alert threshold configuration"
    )
    notification_settings: dict[str, Any] = Field(
        description="Notification settings"
    )
    advanced_settings: dict[str, Any] = Field(
        description="Advanced processing settings"
    )
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


# Frame Metadata Schemas
class FrameMetadataCreateSchema(BaseModel):
    """Schema for creating frame metadata."""

    model_config = ConfigDict(from_attributes=True)

    camera_id: str = Field(description="Reference to camera")
    frame_number: int = Field(description="Sequential frame number from camera")
    timestamp: datetime | None = Field(
        None, description="Frame capture timestamp"
    )
    width: int = Field(description="Frame width in pixels", gt=0)
    height: int = Field(description="Frame height in pixels", gt=0)
    format: str = Field(description="Frame image format", default="jpeg")
    quality_score: float | None = Field(
        None, description="Frame quality score (0.0-1.0)", ge=0.0, le=1.0
    )
    weather_conditions: dict[str, Any] | None = Field(
        None, description="Weather and lighting conditions"
    )
    lighting_conditions: str | None = Field(
        None, description="Lighting conditions (day/night/dawn/dusk)"
    )


class FrameMetadataUpdateSchema(BaseModel):
    """Schema for updating frame metadata after processing."""

    model_config = ConfigDict(from_attributes=True)

    status: ProcessingStatus | None = Field(
        None, description="Frame processing status"
    )
    processing_time_ms: float | None = Field(
        None, description="Total processing time in milliseconds", ge=0
    )
    quality_score: float | None = Field(
        None, description="Frame quality score (0.0-1.0)", ge=0.0, le=1.0
    )
    has_detections: bool | None = Field(
        None, description="Frame contains vehicle detections"
    )
    detection_count: int | None = Field(
        None, description="Total number of detections", ge=0
    )
    vehicle_count: int | None = Field(
        None, description="Number of vehicle detections", ge=0
    )
    traffic_density: float | None = Field(
        None, description="Traffic density score (0.0-1.0)", ge=0.0, le=1.0
    )
    storage_path: str | None = Field(
        None, description="Storage path or S3 key"
    )
    storage_bucket: str | None = Field(
        None, description="Storage bucket name"
    )
    is_stored: bool | None = Field(
        None, description="Frame stored to persistent storage"
    )
    error_message: str | None = Field(
        None, description="Error message if processing failed"
    )


class FrameMetadataResponseSchema(BaseModel):
    """Schema for frame metadata API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Frame metadata unique identifier")
    camera_id: str = Field(description="Reference to camera")
    frame_number: int = Field(description="Sequential frame number from camera")
    timestamp: datetime = Field(description="Frame capture timestamp")
    processing_started_at: datetime | None = Field(
        None, description="Processing start timestamp"
    )
    processing_completed_at: datetime | None = Field(
        None, description="Processing completion timestamp"
    )
    status: str = Field(description="Frame processing status")
    processing_time_ms: float | None = Field(
        None, description="Total processing time in milliseconds"
    )
    quality_score: float | None = Field(
        None, description="Frame quality score (0.0-1.0)"
    )
    quality_rating: str | None = Field(
        None, description="Quality rating category"
    )
    width: int = Field(description="Frame width in pixels")
    height: int = Field(description="Frame height in pixels")
    file_size: int | None = Field(
        None, description="Frame file size in bytes"
    )
    format: str = Field(description="Frame image format")
    has_detections: bool = Field(description="Frame contains vehicle detections")
    detection_count: int = Field(description="Total number of detections")
    vehicle_count: int = Field(description="Number of vehicle detections")
    traffic_density: float | None = Field(
        None, description="Traffic density score (0.0-1.0)"
    )
    is_stored: bool = Field(description="Frame stored to persistent storage")
    error_message: str | None = Field(
        None, description="Error message if processing failed"
    )
    retry_count: int = Field(description="Number of processing retries")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


# Detection Result Schemas
class BoundingBoxSchema(BaseModel):
    """Bounding box coordinates schema."""

    model_config = ConfigDict(from_attributes=True)

    x1: float = Field(description="Top-left X coordinate")
    y1: float = Field(description="Top-left Y coordinate")
    x2: float = Field(description="Bottom-right X coordinate")
    y2: float = Field(description="Bottom-right Y coordinate")
    width: float = Field(description="Bounding box width")
    height: float = Field(description="Bounding box height")
    area: float = Field(description="Bounding box area")


class DetectionResultCreateSchema(BaseModel):
    """Schema for creating detection results."""

    model_config = ConfigDict(from_attributes=True)

    frame_metadata_id: str = Field(description="Reference to frame metadata")
    detection_id: int = Field(description="Detection ID within frame")
    bbox_x1: float = Field(description="Bounding box top-left X coordinate")
    bbox_y1: float = Field(description="Bounding box top-left Y coordinate")
    bbox_x2: float = Field(description="Bounding box bottom-right X coordinate")
    bbox_y2: float = Field(description="Bounding box bottom-right Y coordinate")
    class_name: str = Field(description="Primary object class")
    class_confidence: float = Field(
        description="Classification confidence (0.0-1.0)", ge=0.0, le=1.0
    )
    vehicle_type: str | None = Field(None, description="Specific vehicle type")
    vehicle_confidence: float | None = Field(
        None, description="Vehicle type confidence (0.0-1.0)", ge=0.0, le=1.0
    )
    track_id: int | None = Field(None, description="Object tracking ID")
    color_primary: str | None = Field(None, description="Primary object color")
    color_secondary: str | None = Field(None, description="Secondary object color")
    license_plate: str | None = Field(None, description="License plate text")
    detection_zone: str | None = Field(None, description="Detection zone name")


class DetectionResultResponseSchema(BaseModel):
    """Schema for detection result API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Detection result unique identifier")
    frame_metadata_id: str = Field(description="Reference to frame metadata")
    detection_id: int = Field(description="Detection ID within frame")
    bounding_box: BoundingBoxSchema = Field(description="Bounding box coordinates")
    class_name: str = Field(description="Primary object class")
    class_confidence: float = Field(description="Classification confidence (0.0-1.0)")
    vehicle_type: str | None = Field(None, description="Specific vehicle type")
    vehicle_confidence: float | None = Field(
        None, description="Vehicle type confidence (0.0-1.0)"
    )
    track_id: int | None = Field(None, description="Object tracking ID")
    velocity_x: float | None = Field(
        None, description="Velocity in X direction (pixels/frame)"
    )
    velocity_y: float | None = Field(
        None, description="Velocity in Y direction (pixels/frame)"
    )
    velocity_magnitude: float | None = Field(
        None, description="Velocity magnitude"
    )
    direction: float | None = Field(
        None, description="Movement direction in degrees (0-360)"
    )
    color_primary: str | None = Field(None, description="Primary object color")
    color_secondary: str | None = Field(None, description="Secondary object color")
    license_plate: str | None = Field(None, description="License plate text")
    license_plate_confidence: float | None = Field(
        None, description="License plate recognition confidence"
    )
    detection_quality: float = Field(description="Overall detection quality score")
    is_verified: bool = Field(description="Human verified detection")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


# System Metrics Schemas
class SystemMetricsCreateSchema(BaseModel):
    """Schema for creating system metrics."""

    model_config = ConfigDict(from_attributes=True)

    metric_name: str = Field(description="Metric name identifier", max_length=100)
    metric_type: MetricType = Field(description="Metric type category")
    metric_unit: str = Field(description="Metric unit of measurement")
    value: float = Field(description="Metric value")
    source_type: str = Field(description="Source type (camera/system/process)")
    source_id: str | None = Field(None, description="Source identifier")
    hostname: str | None = Field(None, description="Server hostname")
    service_name: str | None = Field(None, description="Service name")
    labels: dict[str, str] | None = Field(
        None, description="Additional metric labels/tags"
    )
    warning_threshold: float | None = Field(
        None, description="Warning threshold value"
    )
    critical_threshold: float | None = Field(
        None, description="Critical threshold value"
    )


class SystemMetricsResponseSchema(BaseModel):
    """Schema for system metrics API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(description="Metric unique identifier")
    metric_name: str = Field(description="Metric name identifier")
    metric_type: str = Field(description="Metric type category")
    metric_unit: str = Field(description="Metric unit of measurement")
    value: float = Field(description="Metric value")
    timestamp: datetime = Field(description="Metric collection timestamp")
    source_type: str = Field(description="Source type (camera/system/process)")
    source_id: str | None = Field(None, description="Source identifier")
    hostname: str | None = Field(None, description="Server hostname")
    service_name: str | None = Field(None, description="Service name")
    labels: dict[str, Any] | None = Field(
        None, description="Additional metric labels/tags"
    )
    warning_threshold: float | None = Field(
        None, description="Warning threshold value"
    )
    critical_threshold: float | None = Field(
        None, description="Critical threshold value"
    )
    alert_level: str | None = Field(
        None, description="Current alert level (warning/critical)"
    )
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


# Batch Operation Schemas
class BatchFrameMetadataCreateSchema(BaseModel):
    """Schema for batch frame metadata creation."""

    model_config = ConfigDict(from_attributes=True)

    frames: list[FrameMetadataCreateSchema] = Field(
        description="List of frame metadata to create", max_length=1000
    )

    @field_validator("frames")
    @classmethod
    def validate_frames_limit(cls, v: list[FrameMetadataCreateSchema]) -> list[FrameMetadataCreateSchema]:
        """Validate batch size limit."""
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 frames")
        return v


class BatchDetectionResultCreateSchema(BaseModel):
    """Schema for batch detection result creation."""

    model_config = ConfigDict(from_attributes=True)

    detections: list[DetectionResultCreateSchema] = Field(
        description="List of detection results to create", max_length=5000
    )

    @field_validator("detections")
    @classmethod
    def validate_detections_limit(cls, v: list[DetectionResultCreateSchema]) -> list[DetectionResultCreateSchema]:
        """Validate batch size limit."""
        if len(v) > 5000:
            raise ValueError("Batch size cannot exceed 5000 detections")
        return v


class BatchOperationResultSchema(BaseModel):
    """Schema for batch operation results."""

    model_config = ConfigDict(from_attributes=True)

    total_items: int = Field(description="Total number of items in batch")
    successful_items: int = Field(description="Number of successfully processed items")
    failed_items: int = Field(description="Number of failed items")
    errors: list[dict[str, Any]] = Field(
        description="List of errors for failed items", default_factory=list
    )
    processing_time_ms: float = Field(
        description="Total batch processing time in milliseconds"
    )
    items_per_second: float = Field(
        description="Processing throughput (items/second)"
    )


# Analytics and Aggregation Schemas
class CameraAnalyticsSchema(BaseModel):
    """Schema for camera analytics data."""

    model_config = ConfigDict(from_attributes=True)

    camera_id: str = Field(description="Camera identifier")
    camera_name: str = Field(description="Camera name")
    period_start: datetime = Field(description="Analytics period start")
    period_end: datetime = Field(description="Analytics period end")
    total_frames: int = Field(description="Total frames processed")
    frames_with_detections: int = Field(description="Frames containing detections")
    total_detections: int = Field(description="Total number of detections")
    vehicle_detections: int = Field(description="Number of vehicle detections")
    avg_processing_time: float = Field(description="Average processing time (ms)")
    avg_quality_score: float = Field(description="Average frame quality score")
    uptime_percentage: float = Field(description="Camera uptime percentage")
    error_count: int = Field(description="Number of processing errors")
    storage_used_mb: float = Field(description="Storage used in megabytes")


class SystemOverviewSchema(BaseModel):
    """Schema for system overview analytics."""

    model_config = ConfigDict(from_attributes=True)

    total_cameras: int = Field(description="Total number of cameras")
    active_cameras: int = Field(description="Number of active cameras")
    streaming_cameras: int = Field(description="Number of streaming cameras")
    offline_cameras: int = Field(description="Number of offline cameras")
    total_frames_today: int = Field(description="Total frames processed today")
    total_detections_today: int = Field(description="Total detections today")
    avg_system_latency: float = Field(description="Average system latency (ms)")
    system_throughput: float = Field(description="System throughput (frames/sec)")
    storage_used_gb: float = Field(description="Total storage used (GB)")
    error_rate_percentage: float = Field(description="System error rate percentage")
