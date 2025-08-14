"""Data Transfer Objects (DTOs) for Analytics Services.

This module provides comprehensive type-safe data structures to replace
dictionary usage throughout the analytics services.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import numpy as np

from ..models.analytics import ViolationType
from ..models.detection_result import DetectionClass

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
# Traffic Prediction DTOs
# ============================

@dataclass
class TrafficPredictionDTO:
    """Traffic prediction result data transfer object."""
    camera_id: str
    prediction_timestamp: datetime
    forecast_start: datetime
    forecast_end: datetime
    horizon_minutes: int
    predicted_vehicle_count: float
    confidence_interval: dict[str, float]
    model_version: str
    model_accuracy: float
    processing_time_ms: float
    factors_considered: list[str]
    predictions: list[dict[str, Any]] = field(default_factory=list)
    is_fallback: bool = False

    @property
    def confidence_lower(self) -> float:
        """Get lower confidence bound."""
        return self.confidence_interval.get("lower", 0.0)

    @property
    def confidence_upper(self) -> float:
        """Get upper confidence bound."""
        return self.confidence_interval.get("upper", 0.0)

    @property
    def is_high_confidence(self) -> bool:
        """Check if prediction has high confidence."""
        return self.model_accuracy >= 0.8 and not self.is_fallback


@dataclass
class PredictionRequest:
    """Request parameters for traffic predictions."""
    camera_id: str
    horizon_minutes: int = 60
    model_version: str | None = None
    confidence_level: float = 0.95
    include_features: bool = False
    weather_condition: str | None = None
    historical_data: list[dict[str, Any]] | None = None


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

# ============================
# Detection Result DTOs
# ============================

@dataclass
class BoundingBoxDTO:
    """Bounding box coordinates for detection results."""
    x1: float
    y1: float
    x2: float
    y2: float
    normalized: bool = False  # Whether coordinates are normalized (0.0-1.0) or absolute pixels

    @property
    def width(self) -> float:
        """Calculate bounding box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Calculate bounding box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        return self.width * self.height

    @property
    def center_x(self) -> float:
        """Calculate center X coordinate."""
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        """Calculate center Y coordinate."""
        return (self.y1 + self.y2) / 2


@dataclass
class DetectionResultDTO:
    """Comprehensive detection result data transfer object.
    
    Maps directly to DetectionResult model but provides additional
    convenience methods and type safety for analytics processing.
    """
    detection_id: str
    frame_id: str
    camera_id: str
    timestamp: datetime
    track_id: str | None
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBoxDTO
    attributes: dict[str, Any]
    zone_id: str | None = None
    speed: float | None = None  # Speed in km/h
    direction: float | None = None  # Direction in degrees (0-360)
    vehicle_type: str | None = None
    vehicle_confidence: float | None = None
    license_plate: str | None = None
    license_plate_confidence: float | None = None
    color_primary: str | None = None
    color_secondary: str | None = None
    estimated_length: float | None = None  # In meters
    estimated_width: float | None = None  # In meters
    detection_quality: float = 1.0
    occlusion_ratio: float | None = None
    blur_score: float | None = None
    model_version: str | None = None
    processing_time_ms: float | None = None
    is_false_positive: bool = False
    is_verified: bool = False
    is_anomaly: bool = False

    @property
    def is_vehicle(self) -> bool:
        """Check if detection is a vehicle."""
        vehicle_classes = {
            DetectionClass.CAR.value,
            DetectionClass.TRUCK.value,
            DetectionClass.BUS.value,
            DetectionClass.MOTORCYCLE.value,
            DetectionClass.BICYCLE.value,
        }
        return self.class_name in vehicle_classes

    @property
    def is_high_confidence(self) -> bool:
        """Check if detection has high confidence."""
        return self.confidence >= 0.8

    @property
    def is_reliable(self) -> bool:
        """Check if detection is reliable for analytics."""
        return (
            self.is_high_confidence
            and self.detection_quality >= 0.7
            and not self.is_false_positive
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "detection_id": self.detection_id,
            "frame_id": self.frame_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": {
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
                "x2": self.bbox.x2,
                "y2": self.bbox.y2,
                "normalized": self.bbox.normalized,
            },
            "attributes": self.attributes,
            "zone_id": self.zone_id,
            "speed": self.speed,
            "direction": self.direction,
            "vehicle_type": self.vehicle_type,
            "detection_quality": self.detection_quality,
            "model_version": self.model_version,
            "is_reliable": self.is_reliable,
        }


@dataclass
class FrameMetadataDTO:
    """Frame metadata for detection processing."""
    frame_id: str
    camera_id: str
    timestamp: datetime
    frame_number: int
    width: int
    height: int
    quality_score: float | None = None
    model_version: str | None = None
    processing_time_ms: float | None = None
    storage_path: str | None = None
    weather_conditions: dict[str, Any] | None = None
    lighting_conditions: str | None = None


@dataclass
class DetectionData:
    """Detection data from ML models with proper DTO integration."""
    camera_id: str
    timestamp: datetime
    frame_id: str
    detections: list[DetectionResultDTO]  # Replaced with proper DetectionResultDTO
    vehicle_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    source: str = "ml_model"
    model_version: str | None = None
    pipeline_id: str | None = None
    frame_metadata: FrameMetadataDTO | None = None

    @property
    def reliable_detections(self) -> list[DetectionResultDTO]:
        """Get only reliable detections for analytics."""
        return [d for d in self.detections if d.is_reliable]

    @property
    def vehicle_detections(self) -> list[DetectionResultDTO]:
        """Get only vehicle detections."""
        return [d for d in self.detections if d.is_vehicle]

    @property
    def high_confidence_detections(self) -> list[DetectionResultDTO]:
        """Get only high confidence detections."""
        return [d for d in self.detections if d.is_high_confidence]

    def get_detections_by_class(self, class_name: str) -> list[DetectionResultDTO]:
        """Get detections filtered by class name."""
        return [d for d in self.detections if d.class_name == class_name]

    def get_detections_in_zone(self, zone_id: str) -> list[DetectionResultDTO]:
        """Get detections in specific zone."""
        return [d for d in self.detections if d.zone_id == zone_id]


# ============================
# ML Output Conversion Utilities
# ============================

class DetectionResultConverter:
    """Converter for transforming ML model outputs to DetectionResult DTOs.
    
    Handles coordinate normalization, type mapping, and data validation
    for various ML model output formats.
    """

    @staticmethod
    def from_ml_output(
        ml_detection: dict[str, Any],
        frame_metadata: FrameMetadataDTO,
        model_version: str | None = None
    ) -> DetectionResultDTO:
        """Convert ML model detection output to DetectionResultDTO.
        
        Args:
            ml_detection: Raw ML model detection output
            frame_metadata: Frame metadata information
            model_version: Model version used for detection
            
        Returns:
            DetectionResultDTO with properly mapped data
        """
        # Extract bounding box coordinates
        bbox_data = ml_detection.get("bbox", ml_detection.get("bounding_box", {}))
        if isinstance(bbox_data, list) and len(bbox_data) >= 4:
            # Handle [x1, y1, x2, y2] format
            x1, y1, x2, y2 = bbox_data[:4]
        else:
            # Handle dictionary format
            x1 = bbox_data.get("x1", bbox_data.get("left", 0))
            y1 = bbox_data.get("y1", bbox_data.get("top", 0))
            x2 = bbox_data.get("x2", bbox_data.get("right", x1 + bbox_data.get("width", 0)))
            y2 = bbox_data.get("y2", bbox_data.get("bottom", y1 + bbox_data.get("height", 0)))

        # Determine if coordinates are normalized
        normalized = ml_detection.get("normalized", False)
        if not normalized and frame_metadata:
            # Check if coordinates appear normalized (values between 0 and 1)
            normalized = all(0 <= coord <= 1 for coord in [x1, y1, x2, y2])

        bbox = BoundingBoxDTO(
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            normalized=normalized
        )

        # Extract detection ID
        detection_id = ml_detection.get("detection_id", ml_detection.get("id", str(hash(str(ml_detection)))))

        # Extract class information
        class_name = ml_detection.get("class", ml_detection.get("class_name", "unknown"))
        class_id = ml_detection.get("class_id", DetectionResultConverter._get_class_id(class_name))
        confidence = float(ml_detection.get("confidence", ml_detection.get("score", 0.0)))

        # Extract tracking information
        track_id = ml_detection.get("track_id", ml_detection.get("tracking_id"))
        if track_id is not None:
            track_id = str(track_id)

        # Extract vehicle-specific information
        vehicle_type = ml_detection.get("vehicle_type")
        vehicle_confidence = ml_detection.get("vehicle_confidence")

        # Extract motion information
        speed = ml_detection.get("speed")
        direction = ml_detection.get("direction", ml_detection.get("heading"))

        # Extract additional attributes
        attributes = ml_detection.get("attributes", {})
        if "additional_data" in ml_detection:
            attributes.update(ml_detection["additional_data"])

        return DetectionResultDTO(
            detection_id=detection_id,
            frame_id=frame_metadata.frame_id,
            camera_id=frame_metadata.camera_id,
            timestamp=frame_metadata.timestamp,
            track_id=track_id,
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            bbox=bbox,
            attributes=attributes,
            zone_id=ml_detection.get("zone_id"),
            speed=speed,
            direction=direction,
            vehicle_type=vehicle_type,
            vehicle_confidence=vehicle_confidence,
            license_plate=ml_detection.get("license_plate"),
            license_plate_confidence=ml_detection.get("license_plate_confidence"),
            color_primary=ml_detection.get("color_primary"),
            color_secondary=ml_detection.get("color_secondary"),
            estimated_length=ml_detection.get("estimated_length"),
            estimated_width=ml_detection.get("estimated_width"),
            detection_quality=ml_detection.get("quality_score", 1.0),
            occlusion_ratio=ml_detection.get("occlusion_ratio"),
            blur_score=ml_detection.get("blur_score"),
            model_version=model_version,
            processing_time_ms=ml_detection.get("processing_time_ms"),
        )

    @staticmethod
    def from_ml_batch(
        ml_detections: list[dict[str, Any]],
        frame_metadata: FrameMetadataDTO,
        model_version: str | None = None
    ) -> list[DetectionResultDTO]:
        """Convert batch of ML detections to DTOs.
        
        Args:
            ml_detections: List of ML detection outputs
            frame_metadata: Frame metadata information
            model_version: Model version used for detection
            
        Returns:
            List of DetectionResultDTO objects
        """
        return [
            DetectionResultConverter.from_ml_output(detection, frame_metadata, model_version)
            for detection in ml_detections
        ]

    @staticmethod
    def to_detection_data(
        ml_output: dict[str, Any],
        camera_id: str,
        timestamp: datetime | None = None
    ) -> DetectionData:
        """Convert complete ML output to DetectionData.
        
        Args:
            ml_output: Complete ML pipeline output
            camera_id: Camera identifier
            timestamp: Processing timestamp
            
        Returns:
            DetectionData with converted DTOs
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Extract frame metadata
        frame_info = ml_output.get("frame", {})
        frame_metadata = FrameMetadataDTO(
            frame_id=frame_info.get("frame_id", str(hash(str(ml_output)))),
            camera_id=camera_id,
            timestamp=timestamp,
            frame_number=frame_info.get("frame_number", 0),
            width=frame_info.get("width", 1920),
            height=frame_info.get("height", 1080),
            quality_score=frame_info.get("quality_score"),
            model_version=ml_output.get("model_version"),
            processing_time_ms=ml_output.get("processing_time_ms"),
            storage_path=frame_info.get("storage_path"),
            weather_conditions=frame_info.get("weather_conditions"),
            lighting_conditions=frame_info.get("lighting_conditions"),
        )

        # Convert detections
        raw_detections = ml_output.get("detections", [])
        detections = DetectionResultConverter.from_ml_batch(
            raw_detections, frame_metadata, ml_output.get("model_version")
        )

        # Count vehicles
        vehicle_count = len([d for d in detections if d.is_vehicle])

        return DetectionData(
            camera_id=camera_id,
            timestamp=timestamp,
            frame_id=frame_metadata.frame_id,
            detections=detections,
            vehicle_count=vehicle_count,
            metadata=ml_output.get("metadata", {}),
            confidence=ml_output.get("overall_confidence", 0.8),
            source=ml_output.get("source", "ml_model"),
            model_version=ml_output.get("model_version"),
            pipeline_id=ml_output.get("pipeline_id"),
            frame_metadata=frame_metadata,
        )

    @staticmethod
    def _get_class_id(class_name: str) -> int:
        """Map class name to class ID.
        
        Args:
            class_name: Detection class name
            
        Returns:
            Numeric class ID
        """
        class_mapping = {
            "car": 0,
            "truck": 1,
            "bus": 2,
            "motorcycle": 3,
            "bicycle": 4,
            "person": 5,
            "traffic_light": 6,
            "stop_sign": 7,
            "ambulance": 8,
            "fire_truck": 9,
            "police": 10,
            "unknown": 99,
        }
        return class_mapping.get(class_name.lower(), 99)


class CoordinateNormalizer:
    """Utility for coordinate system conversions.
    
    Handles conversion between normalized (0.0-1.0) and absolute pixel coordinates.
    """

    @staticmethod
    def normalize_bbox(
        bbox: BoundingBoxDTO,
        frame_width: int,
        frame_height: int
    ) -> BoundingBoxDTO:
        """Convert absolute coordinates to normalized.
        
        Args:
            bbox: Bounding box with absolute coordinates
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            BoundingBoxDTO with normalized coordinates
        """
        if bbox.normalized:
            return bbox

        return BoundingBoxDTO(
            x1=bbox.x1 / frame_width,
            y1=bbox.y1 / frame_height,
            x2=bbox.x2 / frame_width,
            y2=bbox.y2 / frame_height,
            normalized=True
        )

    @staticmethod
    def denormalize_bbox(
        bbox: BoundingBoxDTO,
        frame_width: int,
        frame_height: int
    ) -> BoundingBoxDTO:
        """Convert normalized coordinates to absolute.
        
        Args:
            bbox: Bounding box with normalized coordinates
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            BoundingBoxDTO with absolute coordinates
        """
        if not bbox.normalized:
            return bbox

        return BoundingBoxDTO(
            x1=bbox.x1 * frame_width,
            y1=bbox.y1 * frame_height,
            x2=bbox.x2 * frame_width,
            y2=bbox.y2 * frame_height,
            normalized=False
        )


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
