"""Data Transfer Objects (DTOs) for Analytics Services.

This module provides comprehensive type-safe data structures to replace
dictionary usage throughout the analytics services.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from ..models.analytics import ViolationType

# ============================
# Enums
# ============================

class TimeWindow(str, Enum):
    """Time window enumeration for aggregations."""
    ONE_MIN = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    ONE_HOUR = "1hour"
    ONE_DAY = "1day"
    ONE_WEEK = "1week"
    ONE_MONTH = "1month"


class AggregationLevel(str, Enum):
    """Aggregation level enumeration."""
    RAW = "raw"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class IncidentStatus(str, Enum):
    """Incident status enumeration."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    INVESTIGATING = "investigating"


class TrafficImpact(str, Enum):
    """Traffic impact level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CongestionLevel(str, Enum):
    """Congestion level enumeration."""
    FREE_FLOW = "free_flow"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"


# ============================
# Aggregation DTOs
# ============================

@dataclass
class StatisticalMetrics:
    """Statistical metrics for aggregated data."""
    sample_count: int = 0
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0


@dataclass
class VehicleStatistics:
    """Vehicle-specific statistical data."""
    mean_vehicle_count: float = 0.0
    median_vehicle_count: float = 0.0
    std_vehicle_count: float = 0.0
    percentile_95_vehicle_count: float = 0.0
    min_vehicle_count: int = 0
    max_vehicle_count: int = 0
    total_vehicles: int = 0


@dataclass
class SpeedStatistics:
    """Speed-specific statistical data."""
    mean_speed: float = 0.0
    median_speed: float = 0.0
    std_speed: float = 0.0
    min_speed: float = 0.0
    max_speed: float = 0.0
    percentile_85_speed: float = 0.0  # Common traffic engineering metric


@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics."""
    data_quality_score: float = 1.0
    data_completeness: float = 1.0
    outlier_count: int = 0
    missing_data_points: int = 0
    sensor_reliability_score: float = 1.0


@dataclass
class AggregatedMetricsDTO:
    """Container for aggregated traffic metrics."""
    camera_id: str
    time_window: TimeWindow
    aggregation_level: AggregationLevel
    start_time: datetime
    end_time: datetime
    vehicle_stats: VehicleStatistics
    speed_stats: SpeedStatistics
    quality_metrics: DataQualityMetrics
    sample_count: int = 0
    mean_occupancy_rate: float | None = None
    raw_data: list[dict[str, Any]] = field(default_factory=list)


# ============================
# Incident Detection DTOs
# ============================

@dataclass
class IncidentMetadata:
    """Metadata for incident detection."""
    detection_source: str = "ml_model"
    model_version: str | None = None
    processing_pipeline: str | None = None
    detection_method: str | None = None


@dataclass
class IncidentLocation:
    """Location information for incidents."""
    camera_id: str
    location_description: str = "Unknown"
    coordinates: tuple[float, float] | None = None
    zone_id: str | None = None
    lane_id: str | None = None


@dataclass
class IncidentEvidence:
    """Evidence data for incidents."""
    images: list[str] = field(default_factory=list)
    video_clip: str | None = None
    detection_frames: list[int] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)


@dataclass
class IncidentAlertDTO:
    """Data structure for incident alerts."""
    id: str
    incident_type: str
    severity: str
    description: str
    location: IncidentLocation
    timestamp: datetime
    detected_at: datetime
    confidence: float
    status: IncidentStatus
    vehicles_involved: list[str]
    traffic_impact: TrafficImpact
    evidence: IncidentEvidence
    metadata: IncidentMetadata
    priority_score: float = 0.5
    rule_triggered: str | None = None
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    notes: str | None = None


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    incident_type: str
    severity: str
    thresholds: dict[str, float]
    conditions: dict[str, Any]
    cooldown_minutes: int = 5
    notification_channels: list[str] = field(default_factory=list)
    camera_ids: list[str] | None = None
    active: bool = True


# ============================
# Rule Engine DTOs
# ============================

@dataclass
class RuleDefinition:
    """Traffic rule definition."""
    rule_type: ViolationType
    speed_limit: float | None = None
    tolerance: float = 0.0
    zone_id: str | None = None
    vehicle_type: str | None = None
    time_restrictions: dict[str, Any] | None = None
    weather_adjustments: dict[str, float] | None = None


@dataclass
class ViolationRecord:
    """Traffic violation record."""
    violation_type: ViolationType
    severity: str
    measured_value: float
    threshold_value: float
    excess_amount: float | None = None
    confidence: float = 0.8
    detection_id: str | None = None
    camera_id: str | None = None
    track_id: int | None = None
    license_plate: str | None = None
    detection_time: datetime | None = None
    vehicle_type: str | None = None
    rule_definition: RuleDefinition | None = None
    evidence_urls: list[str] = field(default_factory=list)


@dataclass
class SpeedLimitInfo:
    """Speed limit information for a zone."""
    zone_id: str
    vehicle_type: str
    speed_limit_kmh: float
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    weather_conditions: dict[str, float] | None = None
    time_based_limits: dict[str, float] | None = None


# ============================
# Speed Calculation DTOs
# ============================

@dataclass
class Position:
    """Vehicle position data."""
    x: float
    y: float
    timestamp: float
    frame_id: int | None = None
    confidence: float = 1.0


@dataclass
class SpeedMeasurement:
    """Speed measurement result."""
    speed_kmh: float
    measurement_method: str
    confidence: float
    distance_traveled: float
    time_elapsed: float
    positions_used: list[Position]


@dataclass
class CameraCalibration:
    """Camera calibration parameters."""
    camera_id: str
    homography_matrix: np.ndarray | None = None
    pixel_to_meter_ratio: float = 0.05
    focal_length: float | None = None
    sensor_size: tuple[float, float] | None = None
    distortion_coefficients: np.ndarray | None = None


# ============================
# Anomaly Detection DTOs
# ============================

@dataclass
class TrafficDataPoint:
    """Traffic data point for anomaly detection."""
    timestamp: datetime
    camera_id: str
    vehicle_count: int
    average_speed: float
    traffic_density: float
    flow_rate: float
    occupancy_rate: float
    queue_length: float


@dataclass
class AnomalyResult:
    """Anomaly detection result."""
    data_point: TrafficDataPoint
    anomaly_score: float
    severity: str
    probable_cause: str
    features: list[float]
    detection_method: str = "isolation_forest"
    model_name: str = "IsolationForest"
    model_version: str = "1.0"
    confidence: float = 0.0


# ============================
# Trajectory Analysis DTOs
# ============================

@dataclass
class PathPoint:
    """Single point in a vehicle trajectory."""
    x: float
    y: float
    timestamp: float
    speed: float = 0.0
    heading: float | None = None
    acceleration: float | None = None


@dataclass
class TrajectoryMetrics:
    """Metrics for vehicle trajectory analysis."""
    total_distance: float
    straight_line_distance: float
    path_efficiency: float
    average_speed: float
    max_speed: float
    min_speed: float
    total_time: float
    stop_count: int = 0
    acceleration_events: int = 0
    deceleration_events: int = 0


@dataclass
class VehicleTrajectoryDTO:
    """Vehicle trajectory data."""
    vehicle_track_id: int
    camera_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    path_points: list[PathPoint]
    metrics: TrajectoryMetrics
    vehicle_type: str | None = None
    license_plate: str | None = None
    tracking_quality: float = 1.0
    path_completeness: float = 1.0
    is_anomalous: bool = False
    anomaly_score: float | None = None
    anomaly_reasons: list[str] | None = None


# ============================
# Real-time Metrics DTOs
# ============================

@dataclass
class RealtimeTrafficMetrics:
    """Real-time traffic metrics."""
    timestamp: datetime
    camera_id: str
    total_vehicles: int
    vehicle_breakdown: dict[str, int]
    average_speed: float | None
    traffic_density: float
    congestion_level: CongestionLevel
    flow_rate: float | None = None
    occupancy_rate: float | None = None
    queue_length: float | None = None


@dataclass
class ProcessingResult:
    """Analytics processing result."""
    camera_id: str
    timestamp: datetime
    vehicle_count: int
    metrics: RealtimeTrafficMetrics
    violations: list[ViolationRecord]
    anomalies: list[AnomalyResult]
    processing_time_ms: float
    frame_id: int | None = None
    detection_quality: float = 1.0


# ============================
# Analytics Report DTOs
# ============================

@dataclass
class AnalyticsSummary:
    """Analytics summary data."""
    total_vehicles: int
    total_violations: int
    total_incidents: int
    average_speed: float
    peak_hour: datetime | None
    peak_vehicle_count: int
    congestion_hours: float
    incident_response_time: float | None = None


@dataclass
class AnalyticsReportDTO:
    """Comprehensive analytics report."""
    report_type: str
    time_range: tuple[datetime, datetime]
    cameras: list[str]
    generated_at: datetime
    summary: AnalyticsSummary
    detailed_metrics: list[AggregatedMetricsDTO] | None = None
    violations: list[ViolationRecord] | None = None
    incidents: list[IncidentAlertDTO] | None = None
    anomalies: list[AnomalyResult] | None = None
    recommendations: list[str] | None = None


# ============================
# Detection Data DTOs (for input)
# ============================

@dataclass
class DetectionData:
    """Detection data from ML models."""
    camera_id: str
    timestamp: datetime
    vehicle_count: int
    detections: list[dict[str, Any]]  # Will be replaced with proper DetectionResult usage
    confidence: float = 0.8
    source: str = "ml_model"
    model_version: str | None = None
    pipeline_id: str | None = None


# ============================
# Cache Key DTOs
# ============================

@dataclass
class CacheKeyParams:
    """Parameters for generating cache keys."""
    prefix: str
    camera_ids: list[str]
    time_window: TimeWindow | None = None
    aggregation_level: AggregationLevel | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    additional_params: dict[str, Any] | None = None


# ============================
# Mock Data DTOs (for testing)
# ============================

@dataclass
class MockMetricData:
    """Mock metric data for testing."""
    timestamp: datetime
    camera_id: str
    total_vehicles: int
    average_speed: float
    occupancy_rate: float
    flow_rate: float
    vehicle_class_diversity: int = 3
