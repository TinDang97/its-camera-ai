"""Analytics and reporting schemas."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .common import Coordinates, TimeRange


class VehicleClass(str, Enum):
    """Vehicle classification enumeration."""

    CAR = "car"
    TRUCK = "truck"
    VAN = "van"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


class IncidentType(str, Enum):
    """Traffic incident type enumeration."""

    ACCIDENT = "accident"
    CONGESTION = "congestion"
    WRONG_WAY = "wrong_way"
    SPEEDING = "speeding"
    ILLEGAL_PARKING = "illegal_parking"
    PEDESTRIAN_VIOLATION = "pedestrian_violation"
    VEHICLE_BREAKDOWN = "vehicle_breakdown"
    ROAD_OBSTRUCTION = "road_obstruction"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class Severity(str, Enum):
    """Incident severity enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrafficDirection(str, Enum):
    """Traffic direction enumeration."""

    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTHEAST = "northeast"
    NORTHWEST = "northwest"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"
    UNKNOWN = "unknown"


class VehicleCount(BaseModel):
    """Vehicle count data schema."""

    camera_id: str = Field(description="Camera ID")
    timestamp: datetime = Field(description="Count timestamp")
    vehicle_class: VehicleClass = Field(description="Vehicle classification")
    direction: TrafficDirection | None = Field(None, description="Traffic direction")
    count: int = Field(description="Number of vehicles", ge=0)
    confidence: float = Field(description="Detection confidence", ge=0, le=1)
    lane: str | None = Field(None, description="Lane identifier")
    speed: float | None = Field(None, description="Average speed in km/h", ge=0)


class TrafficMetrics(BaseModel):
    """Traffic flow metrics schema."""

    camera_id: str = Field(description="Camera ID")
    period_start: datetime = Field(description="Metrics period start")
    period_end: datetime = Field(description="Metrics period end")
    total_vehicles: int = Field(description="Total vehicle count", ge=0)
    vehicle_breakdown: dict[VehicleClass, int] = Field(
        description="Count by vehicle class"
    )
    directional_flow: dict[TrafficDirection, int] = Field(
        description="Count by direction"
    )
    avg_speed: float | None = Field(None, description="Average speed in km/h", ge=0)
    peak_hour: datetime | None = Field(None, description="Peak traffic hour")
    occupancy_rate: float = Field(description="Lane occupancy percentage", ge=0, le=100)
    congestion_level: str = Field(description="Congestion level (low/medium/high)")
    queue_length: float | None = Field(
        None, description="Average queue length in meters", ge=0
    )

    @field_validator("congestion_level")
    @classmethod
    def validate_congestion_level(cls, v: str) -> str:
        """Validate congestion level."""
        if v not in ["low", "medium", "high"]:
            raise ValueError("Congestion level must be 'low', 'medium', or 'high'")
        return v


class IncidentAlert(BaseModel):
    """Traffic incident alert schema."""

    id: str = Field(description="Alert ID")
    camera_id: str = Field(description="Source camera ID")
    incident_type: IncidentType = Field(description="Type of incident")
    severity: Severity = Field(description="Incident severity")
    description: str = Field(description="Incident description")
    location: str = Field(description="Incident location")
    coordinates: Coordinates | None = Field(None, description="GPS coordinates")
    timestamp: datetime = Field(description="Incident timestamp")
    detected_at: datetime = Field(description="Detection timestamp")
    confidence: float = Field(description="Detection confidence", ge=0, le=1)
    status: str = Field(description="Alert status (active/resolved/false_positive)")
    vehicles_involved: list[str] | None = Field(
        None, description="Vehicle IDs involved"
    )
    estimated_duration: int | None = Field(
        None, description="Estimated duration in minutes", ge=0
    )
    traffic_impact: str | None = Field(None, description="Traffic impact assessment")
    images: list[str] | None = Field(None, description="Evidence image URLs")
    video_clip: str | None = Field(None, description="Evidence video URL")
    resolved_at: datetime | None = Field(None, description="Resolution timestamp")
    resolved_by: str | None = Field(None, description="Resolved by user ID")
    notes: str | None = Field(None, description="Additional notes")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate alert status."""
        valid_statuses = ["active", "resolved", "false_positive", "investigating"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v

    class Config:
        from_attributes = True


class AnalyticsResponse(BaseModel):
    """Real-time analytics response schema."""

    camera_id: str = Field(description="Camera ID")
    timestamp: datetime = Field(description="Analytics timestamp")
    vehicle_counts: list[VehicleCount] = Field(description="Current vehicle counts")
    traffic_metrics: TrafficMetrics | None = Field(
        None, description="Current traffic metrics"
    )
    active_incidents: list[IncidentAlert] = Field(
        description="Currently active incidents"
    )
    processing_time: float = Field(description="Processing time in ms")
    frame_rate: float = Field(description="Current processing frame rate")
    detection_zones: list[str] = Field(description="Active detection zones")


class ReportRequest(BaseModel):
    """Analytics report generation request schema."""

    report_type: str = Field(description="Type of report to generate")
    time_range: TimeRange = Field(description="Report time range")
    camera_ids: list[str] | None = Field(
        None, description="Specific cameras (all if not specified)"
    )
    zone_ids: list[str] | None = Field(
        None, description="Specific zones (all if not specified)"
    )
    vehicle_classes: list[VehicleClass] | None = Field(
        None, description="Specific vehicle classes"
    )
    incident_types: list[IncidentType] | None = Field(
        None, description="Specific incident types"
    )
    aggregation_level: str = Field(
        description="Data aggregation level", default="hourly"
    )
    format: str = Field(description="Report format", default="json")
    include_charts: bool = Field(False, description="Include chart visualizations")
    email_recipients: list[str] | None = Field(
        None, description="Email addresses for report delivery"
    )

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, v: str) -> str:
        """Validate report type."""
        valid_types = [
            "traffic_summary",
            "incident_report",
            "vehicle_counts",
            "speed_analysis",
            "congestion_analysis",
            "compliance_report",
            "custom",
        ]
        if v not in valid_types:
            raise ValueError(f"Report type must be one of: {', '.join(valid_types)}")
        return v

    @field_validator("aggregation_level")
    @classmethod
    def validate_aggregation_level(cls, v: str) -> str:
        """Validate aggregation level."""
        if v not in ["minute", "hourly", "daily", "weekly", "monthly"]:
            raise ValueError(
                "Aggregation level must be 'minute', 'hourly', 'daily', 'weekly', or 'monthly'"
            )
        return v

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate report format."""
        if v not in ["json", "csv", "pdf", "excel"]:
            raise ValueError("Format must be 'json', 'csv', 'pdf', or 'excel'")
        return v


class ReportResponse(BaseModel):
    """Analytics report response schema."""

    report_id: str = Field(description="Generated report ID")
    report_type: str = Field(description="Report type")
    status: str = Field(description="Report generation status")
    created_at: datetime = Field(description="Report creation timestamp")
    completed_at: datetime | None = Field(None, description="Completion timestamp")
    download_url: str | None = Field(None, description="Report download URL")
    file_size: int | None = Field(None, description="Report file size in bytes")
    parameters: dict[str, Any] = Field(description="Report generation parameters")
    error_message: str | None = Field(None, description="Error message if failed")
    expires_at: datetime | None = Field(None, description="Download expiration")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate report status."""
        valid_statuses = ["pending", "processing", "completed", "failed", "expired"]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v

    class Config:
        from_attributes = True


class HistoricalQuery(BaseModel):
    """Historical data query schema."""

    time_range: TimeRange = Field(description="Query time range")
    camera_ids: list[str] | None = Field(
        None, description="Specific cameras (all if not specified)"
    )
    metric_types: list[str] = Field(description="Types of metrics to retrieve")
    aggregation: str = Field(description="Data aggregation level", default="hourly")
    filters: dict[str, Any] | None = Field(None, description="Additional filters")
    limit: int = Field(description="Maximum number of records", default=1000, le=10000)
    offset: int = Field(description="Record offset for pagination", default=0, ge=0)

    @field_validator("metric_types")
    @classmethod
    def validate_metric_types(cls, v: list[str]) -> list[str]:
        """Validate metric types."""
        valid_types = [
            "vehicle_counts",
            "speed_data",
            "incidents",
            "occupancy",
            "flow_rate",
            "queue_length",
            "congestion",
        ]
        for metric_type in v:
            if metric_type not in valid_types:
                raise ValueError(
                    f"Invalid metric type '{metric_type}'. Valid types: {', '.join(valid_types)}"
                )
        return v

    @field_validator("aggregation")
    @classmethod
    def validate_aggregation(cls, v: str) -> str:
        """Validate aggregation level."""
        if v not in ["raw", "minute", "hourly", "daily"]:
            raise ValueError(
                "Aggregation must be 'raw', 'minute', 'hourly', or 'daily'"
            )
        return v


class HistoricalData(BaseModel):
    """Historical data response schema."""

    timestamp: datetime = Field(description="Data timestamp")
    camera_id: str = Field(description="Source camera ID")
    metric_type: str = Field(description="Type of metric")
    value: float | int | dict[str, Any] = Field(description="Metric value")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class DashboardConfig(BaseModel):
    """Analytics dashboard configuration schema."""

    name: str = Field(description="Dashboard name", max_length=100)
    description: str | None = Field(None, description="Dashboard description")
    layout: dict[str, Any] = Field(description="Dashboard layout configuration")
    widgets: list[dict[str, Any]] = Field(description="Dashboard widgets")
    filters: dict[str, Any] | None = Field(None, description="Default filters")
    refresh_interval: int = Field(
        description="Auto-refresh interval in seconds", default=30, ge=5
    )
    is_public: bool = Field(False, description="Whether dashboard is public")
    shared_with: list[str] | None = Field(None, description="User IDs with access")


class AlertRule(BaseModel):
    """Alert rule configuration schema."""

    name: str = Field(description="Alert rule name", max_length=100)
    description: str | None = Field(None, description="Rule description")
    condition: dict[str, Any] = Field(description="Alert condition definition")
    severity: Severity = Field(description="Alert severity level")
    cameras: list[str] | None = Field(None, description="Specific cameras")
    zones: list[str] | None = Field(None, description="Specific zones")
    schedule: dict[str, Any] | None = Field(None, description="Active schedule")
    notification_channels: list[str] = Field(
        description="Notification channels (email, sms, webhook)"
    )
    is_active: bool = Field(True, description="Whether rule is active")
    cooldown_minutes: int = Field(
        description="Cooldown period between alerts", default=5, ge=1
    )
