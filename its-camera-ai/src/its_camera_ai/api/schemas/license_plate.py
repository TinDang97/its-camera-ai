"""
API schemas for License Plate Recognition endpoints.

Pydantic models for request/response validation and serialization
for all LPR-related API operations.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, validator


class PlateRegion(str, Enum):
    """License plate regions."""

    US = "us"
    EU = "eu"
    ASIA = "asia"
    CA = "ca"
    MX = "mx"
    BR = "br"
    AU = "au"
    UK = "uk"
    AUTO = "auto"


class AlertType(str, Enum):
    """Alert types for watchlist."""

    STOLEN = "stolen"
    WANTED = "wanted"
    AMBER_ALERT = "amber_alert"
    BOLO = "bolo"
    TOLL_VIOLATION = "toll_violation"
    PARKING_VIOLATION = "parking_violation"
    EXPIRED_REGISTRATION = "expired_registration"
    UNINSURED = "uninsured"
    CUSTOM = "custom"


class AlertPriority(str, Enum):
    """Alert priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")

    @validator('x1', 'x2', 'y1', 'y2')
    def coordinates_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Coordinates must be non-negative')
        return v

    @validator('x2')
    def x2_must_be_greater_than_x1(cls, v, values):
        if 'x1' in values and v <= values['x1']:
            raise ValueError('x2 must be greater than x1')
        return v

    @validator('y2')
    def y2_must_be_greater_than_y1(cls, v, values):
        if 'y1' in values and v <= values['y1']:
            raise ValueError('y2 must be greater than y1')
        return v


class PlateDetectionRequest(BaseModel):
    """Request for real-time plate detection."""

    image_data: str = Field(..., description="Base64 encoded image data")
    camera_id: UUID = Field(..., description="Camera ID")
    vehicle_detections: list[dict[str, Any]] = Field(
        ..., description="Vehicle detection results"
    )
    region_hint: PlateRegion | None = Field(
        PlateRegion.AUTO, description="Expected plate region"
    )
    min_confidence: float | None = Field(
        0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    enable_caching: bool | None = Field(
        True, description="Enable result caching"
    )

    class Config:
        json_encoders = {UUID: str}


class PlateDetectionResponse(BaseModel):
    """Response from plate detection."""

    success: bool = Field(..., description="Whether detection was successful")
    detections: list["PlateDetectionResult"] = Field(
        default_factory=list, description="List of plate detections"
    )
    processing_time_ms: float = Field(..., description="Total processing time")
    cached: bool = Field(default=False, description="Result was cached")
    error_message: str | None = Field(None, description="Error message if failed")


class PlateDetectionResult(BaseModel):
    """Individual plate detection result."""

    # Detection status and text
    plate_text: str | None = Field(None, description="Recognized plate text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    ocr_confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence")

    # Bounding boxes
    vehicle_bbox: BoundingBox = Field(..., description="Vehicle bounding box")
    plate_bbox: BoundingBox | None = Field(None, description="Plate bounding box")

    # Quality metrics
    plate_quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Plate image quality score"
    )
    character_confidences: list[float] = Field(
        default_factory=list, description="Per-character confidence scores"
    )

    # Processing info
    processing_time_ms: float = Field(..., description="Processing time")
    ocr_time_ms: float = Field(..., description="OCR processing time")
    engine_used: str = Field(..., description="OCR engine used")
    region: PlateRegion = Field(..., description="Detected/used region")

    # Status flags
    is_reliable: bool = Field(..., description="Whether detection is reliable")
    is_high_confidence: bool = Field(..., description="Whether detection has high confidence")

    # Alert information
    triggered_alerts: list[str] = Field(
        default_factory=list, description="Alert IDs triggered"
    )


class PlateSearchRequest(BaseModel):
    """Request for searching plate history."""

    plate_text: str | None = Field(None, description="Plate text to search")
    camera_ids: list[UUID] | None = Field(None, description="Camera IDs to filter")
    start_date: datetime | None = Field(None, description="Start date for search")
    end_date: datetime | None = Field(None, description="End date for search")
    min_confidence: float | None = Field(
        0.5, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    limit: int | None = Field(100, ge=1, le=1000, description="Maximum results")
    offset: int | None = Field(0, ge=0, description="Results offset")
    include_false_positives: bool | None = Field(
        False, description="Include false positive detections"
    )

    class Config:
        json_encoders = {UUID: str, datetime: lambda v: v.isoformat()}


class PlateSearchResponse(BaseModel):
    """Response from plate search."""

    detections: list["PlateDetectionRecord"] = Field(
        default_factory=list, description="Search results"
    )
    total_count: int = Field(..., description="Total matching records")
    has_more: bool = Field(..., description="Whether more results available")
    search_time_ms: float = Field(..., description="Search execution time")


class PlateDetectionRecord(BaseModel):
    """Plate detection record from database."""

    id: UUID = Field(..., description="Detection ID")
    camera_id: UUID = Field(..., description="Camera ID")
    camera_name: str | None = Field(None, description="Camera name")

    # Plate information
    plate_text: str = Field(..., description="Recognized plate text")
    plate_region: PlateRegion = Field(..., description="Plate region")

    # Confidence and quality
    overall_confidence: float = Field(..., description="Overall confidence")
    ocr_confidence: float = Field(..., description="OCR confidence")
    plate_quality_score: float = Field(..., description="Quality score")

    # Timing
    detected_at: datetime = Field(..., description="Detection timestamp")
    processing_time_ms: float = Field(..., description="Processing time")

    # Validation status
    is_validated: bool = Field(..., description="Manually validated")
    is_false_positive: bool = Field(..., description="Marked as false positive")
    validated_by: str | None = Field(None, description="Validator user")

    # Alert information
    triggered_alerts: list[str] = Field(
        default_factory=list, description="Triggered alerts"
    )

    class Config:
        json_encoders = {UUID: str, datetime: lambda v: v.isoformat()}


class WatchlistCreateRequest(BaseModel):
    """Request to create watchlist entry."""

    plate_number: str = Field(..., min_length=1, max_length=20, description="Plate number")
    alert_type: AlertType = Field(..., description="Type of alert")
    alert_priority: AlertPriority = Field(
        AlertPriority.MEDIUM, description="Alert priority"
    )
    description: str | None = Field(None, max_length=500, description="Description")
    notes: str | None = Field(None, max_length=1000, description="Additional notes")
    expires_at: datetime | None = Field(None, description="Expiration time")

    # Contact and notification
    contact_info: dict[str, Any] | None = Field(None, description="Contact information")
    notification_channels: list[str] | None = Field(
        None, description="Notification channels"
    )

    # Owner information
    owner_name: str | None = Field(None, max_length=200, description="Owner name")
    owner_info: dict[str, Any] | None = Field(None, description="Owner details")

    # Agency information
    agency: str | None = Field(None, max_length=100, description="Agency")
    jurisdiction: str | None = Field(None, max_length=100, description="Jurisdiction")
    case_number: str | None = Field(None, max_length=100, description="Case number")

    @validator('plate_number')
    def normalize_plate_number(cls, v):
        return v.upper().strip()

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class WatchlistUpdateRequest(BaseModel):
    """Request to update watchlist entry."""

    alert_type: AlertType | None = Field(None, description="Type of alert")
    alert_priority: AlertPriority | None = Field(None, description="Alert priority")
    description: str | None = Field(None, max_length=500, description="Description")
    notes: str | None = Field(None, max_length=1000, description="Additional notes")
    expires_at: datetime | None = Field(None, description="Expiration time")
    is_active: bool | None = Field(None, description="Whether alert is active")

    # Contact and notification
    contact_info: dict[str, Any] | None = Field(None, description="Contact information")
    notification_channels: list[str] | None = Field(
        None, description="Notification channels"
    )

    # Owner information
    owner_name: str | None = Field(None, max_length=200, description="Owner name")
    owner_info: dict[str, Any] | None = Field(None, description="Owner details")

    # Agency information
    agency: str | None = Field(None, max_length=100, description="Agency")
    jurisdiction: str | None = Field(None, max_length=100, description="Jurisdiction")
    case_number: str | None = Field(None, max_length=100, description="Case number")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class WatchlistResponse(BaseModel):
    """Watchlist entry response."""

    id: UUID = Field(..., description="Watchlist entry ID")
    plate_number: str = Field(..., description="Plate number")
    alert_type: AlertType = Field(..., description="Alert type")
    alert_priority: AlertPriority = Field(..., description="Alert priority")

    # Status
    is_active: bool = Field(..., description="Whether alert is active")
    expires_at: datetime | None = Field(None, description="Expiration time")

    # Description and metadata
    description: str | None = Field(None, description="Description")
    notes: str | None = Field(None, description="Notes")

    # Contact information
    contact_info: dict[str, Any] | None = Field(None, description="Contact info")
    notification_channels: list[str] | None = Field(None, description="Notification channels")

    # Owner information
    owner_name: str | None = Field(None, description="Owner name")
    owner_info: dict[str, Any] | None = Field(None, description="Owner info")

    # Agency information
    agency: str | None = Field(None, description="Agency")
    jurisdiction: str | None = Field(None, description="Jurisdiction")
    case_number: str | None = Field(None, description="Case number")

    # Statistics
    total_detections: int = Field(..., description="Total detections")
    last_detected_at: datetime | None = Field(None, description="Last detected")
    last_detected_camera_id: UUID | None = Field(None, description="Last camera")

    # Audit
    created_at: datetime = Field(..., description="Created timestamp")
    created_by: str = Field(..., description="Created by user")
    updated_at: datetime | None = Field(None, description="Updated timestamp")
    updated_by: str | None = Field(None, description="Updated by user")

    class Config:
        json_encoders = {UUID: str, datetime: lambda v: v.isoformat()}


class WatchlistSearchRequest(BaseModel):
    """Request to search watchlist entries."""

    plate_number: str | None = Field(None, description="Plate number to search")
    alert_type: AlertType | None = Field(None, description="Alert type filter")
    alert_priority: AlertPriority | None = Field(None, description="Priority filter")
    is_active: bool | None = Field(None, description="Active status filter")
    agency: str | None = Field(None, description="Agency filter")
    include_expired: bool | None = Field(False, description="Include expired entries")

    limit: int | None = Field(100, ge=1, le=1000, description="Maximum results")
    offset: int | None = Field(0, ge=0, description="Results offset")


class WatchlistSearchResponse(BaseModel):
    """Response from watchlist search."""

    entries: list[WatchlistResponse] = Field(
        default_factory=list, description="Watchlist entries"
    )
    total_count: int = Field(..., description="Total matching entries")
    has_more: bool = Field(..., description="Whether more results available")


class AnalyticsRequest(BaseModel):
    """Request for plate analytics."""

    camera_ids: list[UUID] | None = Field(None, description="Camera IDs to include")
    start_date: datetime | None = Field(None, description="Start date")
    end_date: datetime | None = Field(None, description="End date")
    group_by: str | None = Field(
        "hour", description="Grouping: hour, day, week, month"
    )
    include_details: bool | None = Field(
        False, description="Include detailed metrics"
    )

    class Config:
        json_encoders = {UUID: str, datetime: lambda v: v.isoformat()}


class AnalyticsResponse(BaseModel):
    """Response with plate analytics."""

    summary: "AnalyticsSummary" = Field(..., description="Summary metrics")
    data_points: list["AnalyticsDataPoint"] = Field(
        default_factory=list, description="Time-series data"
    )
    cameras: list["CameraAnalytics"] = Field(
        default_factory=list, description="Per-camera analytics"
    )
    computation_time_ms: float = Field(..., description="Computation time")


class AnalyticsSummary(BaseModel):
    """Summary analytics metrics."""

    total_vehicles: int = Field(..., description="Total vehicles detected")
    total_plate_attempts: int = Field(..., description="Total plate detection attempts")
    successful_reads: int = Field(..., description="Successful plate reads")
    unique_plates: int = Field(..., description="Unique plates detected")

    # Rates
    success_rate: float = Field(..., description="Plate recognition success rate")
    avg_confidence: float = Field(..., description="Average confidence")
    avg_quality_score: float = Field(..., description="Average quality score")
    avg_processing_time_ms: float = Field(..., description="Average processing time")

    # Alerts
    total_alerts: int = Field(..., description="Total alerts triggered")
    alerts_by_type: dict[str, int] = Field(
        default_factory=dict, description="Alerts by type"
    )

    # Regional distribution
    plates_by_region: dict[str, int] = Field(
        default_factory=dict, description="Plates by region"
    )


class AnalyticsDataPoint(BaseModel):
    """Single analytics data point."""

    timestamp: datetime = Field(..., description="Data point timestamp")
    vehicles: int = Field(..., description="Vehicle count")
    plate_attempts: int = Field(..., description="Plate detection attempts")
    successful_reads: int = Field(..., description="Successful reads")
    unique_plates: int = Field(..., description="Unique plates")
    success_rate: float = Field(..., description="Success rate")
    avg_confidence: float = Field(..., description="Average confidence")
    avg_processing_time_ms: float = Field(..., description="Average processing time")
    alerts_triggered: int = Field(..., description="Alerts triggered")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class CameraAnalytics(BaseModel):
    """Per-camera analytics."""

    camera_id: UUID = Field(..., description="Camera ID")
    camera_name: str | None = Field(None, description="Camera name")

    # Metrics
    total_vehicles: int = Field(..., description="Total vehicles")
    successful_reads: int = Field(..., description="Successful reads")
    success_rate: float = Field(..., description="Success rate")
    avg_confidence: float = Field(..., description="Average confidence")
    avg_processing_time_ms: float = Field(..., description="Average processing time")

    # Performance indicators
    performance_score: float = Field(..., description="Overall performance score")
    reliability_score: float = Field(..., description="Reliability score")

    class Config:
        json_encoders = {UUID: str}


class ValidationRequest(BaseModel):
    """Request to validate/correct plate detection."""

    detection_id: UUID = Field(..., description="Detection ID to validate")
    action: str = Field(..., description="Action: validate, mark_false_positive, correct")
    corrected_text: str | None = Field(None, description="Corrected plate text")
    confidence: float | None = Field(
        1.0, ge=0.0, le=1.0, description="Validation confidence"
    )
    notes: str | None = Field(None, description="Validation notes")

    class Config:
        json_encoders = {UUID: str}


class ValidationResponse(BaseModel):
    """Response from validation action."""

    success: bool = Field(..., description="Whether validation was successful")
    detection_id: UUID = Field(..., description="Detection ID")
    action_taken: str = Field(..., description="Action that was taken")
    updated_at: datetime = Field(..., description="Update timestamp")
    updated_by: str = Field(..., description="User who performed validation")

    class Config:
        json_encoders = {UUID: str, datetime: lambda v: v.isoformat()}


class HealthCheckResponse(BaseModel):
    """LPR system health check response."""

    status: str = Field(..., description="Overall system status")
    ocr_engines: dict[str, bool] = Field(..., description="OCR engine availability")
    model_status: dict[str, str] = Field(..., description="Model loading status")
    performance_metrics: dict[str, float] = Field(..., description="Performance metrics")
    cache_status: dict[str, Any] = Field(..., description="Cache statistics")
    memory_usage: dict[str, float] = Field(..., description="Memory usage statistics")
    uptime_seconds: float = Field(..., description="System uptime")


# Update forward references
PlateDetectionResponse.model_rebuild()
PlateSearchResponse.model_rebuild()
AnalyticsResponse.model_rebuild()
