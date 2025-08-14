"""Analytics and reporting schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

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


class AlertRuleRequest(BaseModel):
    """Request schema for creating alert rules."""

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


class AlertRuleResponse(BaseModel):
    """Response schema for alert rules."""

    id: str = Field(description="Alert rule ID")
    name: str = Field(description="Alert rule name")
    description: str | None = Field(None, description="Rule description")
    condition: dict[str, Any] = Field(description="Alert condition definition")
    severity: Severity = Field(description="Alert severity level")
    cameras: list[str] | None = Field(None, description="Specific cameras")
    zones: list[str] | None = Field(None, description="Specific zones")
    schedule: dict[str, Any] | None = Field(None, description="Active schedule")
    notification_channels: list[str] = Field(
        description="Notification channels (email, sms, webhook)"
    )
    is_active: bool = Field(description="Whether rule is active")
    cooldown_minutes: int = Field(description="Cooldown period between alerts")
    created_at: datetime = Field(description="Rule creation timestamp")
    updated_at: datetime = Field(description="Rule last update timestamp")
    created_by: str = Field(description="User ID who created the rule")
    last_triggered: datetime | None = Field(None, description="Last trigger timestamp")
    trigger_count: int = Field(0, description="Total number of triggers")

    class Config:
        from_attributes = True


class DashboardResponse(BaseModel):
    """Response schema for dashboard data."""

    camera_id: str = Field(description="Camera ID")
    timestamp: datetime = Field(description="Dashboard data timestamp")
    real_time_metrics: TrafficMetrics = Field(description="Real-time traffic metrics")
    active_incidents: list[IncidentAlert] = Field(description="Active incidents")
    vehicle_counts: list[VehicleCount] = Field(description="Recent vehicle counts")
    recent_violations: list[dict[str, Any]] = Field(description="Recent violations")
    anomalies: list[dict[str, Any]] = Field(description="Detected anomalies")
    camera_status: dict[str, Any] = Field(description="Camera status information")
    performance_metrics: dict[str, Any] = Field(description="Performance metrics")
    alerts_summary: dict[str, Any] = Field(description="Alerts summary")
    hourly_trends: list[dict[str, Any]] = Field(description="Hourly trend data")
    congestion_heatmap: dict[str, Any] | None = Field(None, description="Congestion heatmap data")


class PredictionResponse(BaseModel):
    """Response schema for traffic predictions."""

    camera_id: str = Field(description="Camera ID")
    prediction_timestamp: datetime = Field(description="When prediction was made")
    forecast_start: datetime = Field(description="Forecast period start")
    forecast_end: datetime = Field(description="Forecast period end")
    predictions: list[dict[str, Any]] = Field(description="Traffic predictions")
    confidence_interval: dict[str, float] = Field(description="Prediction confidence")
    ml_model_version: str = Field(description="ML model version used")
    ml_model_accuracy: float = Field(description="Model accuracy score", ge=0, le=1)
    factors_considered: list[str] = Field(description="Factors considered in prediction")
    historical_baseline: dict[str, Any] = Field(description="Historical baseline data")


class HeatmapResponse(BaseModel):
    """Response schema for traffic heatmap data."""

    camera_id: str = Field(description="Camera ID")
    timestamp: datetime = Field(description="Heatmap data timestamp")
    time_range: TimeRange = Field(description="Data time range")
    heatmap_data: list[dict[str, Any]] = Field(description="Heatmap grid data")
    zones: list[dict[str, Any]] = Field(description="Zone definitions")
    intensity_scale: dict[str, float] = Field(description="Intensity scale mapping")
    aggregation_method: str = Field(description="Data aggregation method")
    spatial_resolution: dict[str, int] = Field(description="Spatial resolution (width x height)")
    metadata: dict[str, Any] = Field(description="Additional metadata")


class TrendAnalysisResponse(BaseModel):
    """Response schema for traffic trend analysis."""

    analysis_timestamp: datetime = Field(description="Analysis timestamp")
    time_range: TimeRange = Field(description="Analysis time range")
    cameras: list[str] = Field(description="Cameras included in analysis")
    trends: dict[str, Any] = Field(description="Trend analysis results")
    patterns: list[dict[str, Any]] = Field(description="Identified patterns")
    seasonal_analysis: dict[str, Any] = Field(description="Seasonal trend analysis")
    anomaly_periods: list[dict[str, Any]] = Field(description="Anomalous time periods")
    recommendations: list[str] = Field(description="Traffic optimization recommendations")
    statistical_summary: dict[str, Any] = Field(description="Statistical summary")
    confidence_metrics: dict[str, float] = Field(description="Analysis confidence metrics")


class AnomalyDetectionRequest(BaseModel):
    """Request schema for triggering anomaly detection."""

    camera_ids: list[str] = Field(description="Camera IDs to analyze")
    time_range: TimeRange = Field(description="Analysis time range")
    detection_method: str = Field("isolation_forest", description="Anomaly detection method")
    sensitivity: float = Field(0.1, description="Detection sensitivity", ge=0, le=1)
    min_anomaly_score: float = Field(0.5, description="Minimum anomaly score", ge=0, le=1)
    include_historical_context: bool = Field(True, description="Include historical context")
    zones: list[str] | None = Field(None, description="Specific zones to analyze")
    vehicle_types: list[VehicleClass] | None = Field(None, description="Vehicle types to include")

    @field_validator("detection_method")
    @classmethod
    def validate_detection_method(cls, v: str) -> str:
        """Validate detection method."""
        valid_methods = ["isolation_forest", "one_class_svm", "autoencoder", "statistical"]
        if v not in valid_methods:
            raise ValueError(f"Detection method must be one of: {', '.join(valid_methods)}")
        return v


class WebSocketAnalyticsUpdate(BaseModel):
    """WebSocket analytics update message schema."""

    event_type: Literal["metrics", "incident", "vehicle_count", "speed_update", "prediction"] = Field(
        description="Type of analytics event"
    )
    camera_id: str = Field(description="Source camera ID")
    timestamp: datetime = Field(description="Event timestamp")
    data: dict[str, Any] = Field(description="Event data payload")
    processing_latency_ms: float = Field(description="Processing latency in milliseconds", ge=0)
    confidence_score: float = Field(description="Data confidence score", ge=0, le=1)
    sequence_id: int | None = Field(None, description="Message sequence identifier")
    metadata: dict[str, Any] | None = Field(None, description="Additional metadata")


class PredictionData(BaseModel):
    """Prediction data for WebSocket updates."""

    timestamp: datetime = Field(description="Prediction timestamp")
    predicted_vehicle_count: int = Field(description="Predicted vehicle count", ge=0)
    predicted_avg_speed: float = Field(description="Predicted average speed", ge=0)
    predicted_congestion_level: str = Field(description="Predicted congestion level")
    confidence: float = Field(description="Prediction confidence", ge=0, le=1)
    horizon_minutes: int = Field(description="Prediction horizon in minutes", ge=1)


class SpeedData(BaseModel):
    """Speed data for WebSocket updates."""

    timestamp: datetime = Field(description="Speed measurement timestamp")
    average_speed: float = Field(description="Average speed in km/h", ge=0)
    speed_limit: float = Field(description="Speed limit in km/h", ge=0)
    violation_count: int = Field(description="Speed violations count", ge=0)
    vehicle_speeds: dict[str, float] = Field(description="Individual vehicle speeds")
    zone_id: str | None = Field(None, description="Speed measurement zone")


class ReportGenerationRequest(BaseModel):
    """Request schema for generating analytics reports."""

    report_type: str = Field(description="Type of report to generate")
    time_range: TimeRange = Field(description="Report time range")
    camera_ids: list[str] | None = Field(None, description="Specific cameras")
    format: str = Field("json", description="Report format")
    include_charts: bool = Field(False, description="Include chart visualizations")
    email_recipients: list[str] | None = Field(None, description="Email delivery recipients")
    filters: dict[str, Any] | None = Field(None, description="Additional filters")
    template: str | None = Field(None, description="Report template to use")
    priority: str = Field("normal", description="Report generation priority")

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
            "anomaly_report",
            "trend_analysis",
            "custom",
        ]
        if v not in valid_types:
            raise ValueError(f"Report type must be one of: {', '.join(valid_types)}")
        return v

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate report format."""
        if v not in ["json", "csv", "pdf", "excel", "html"]:
            raise ValueError("Format must be 'json', 'csv', 'pdf', 'excel', or 'html'")
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Validate priority level."""
        if v not in ["low", "normal", "high", "urgent"]:
            raise ValueError("Priority must be 'low', 'normal', 'high', or 'urgent'")
        return v
