"""Analytics models for TimescaleDB-optimized traffic data storage.

This module provides SQLAlchemy models optimized for time-series analytics data,
including traffic metrics, rule violations, trajectories, and anomaly detection results.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class CongestionLevel(str, Enum):
    """Traffic congestion level enumeration."""

    FREE_FLOW = "free_flow"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"


class ViolationType(str, Enum):
    """Traffic rule violation type enumeration."""

    SPEEDING = "speeding"
    RED_LIGHT = "red_light"
    WRONG_WAY = "wrong_way"
    ILLEGAL_TURN = "illegal_turn"
    ILLEGAL_PARKING = "illegal_parking"
    STOP_SIGN = "stop_sign"
    LANE_VIOLATION = "lane_violation"
    RESTRICTED_AREA = "restricted_area"
    PEDESTRIAN_VIOLATION = "pedestrian_violation"


class AnomalyType(str, Enum):
    """Anomaly detection type enumeration."""

    TRAFFIC_PATTERN = "traffic_pattern"
    VEHICLE_BEHAVIOR = "vehicle_behavior"
    SPEED_ANOMALY = "speed_anomaly"
    DENSITY_ANOMALY = "density_anomaly"
    FLOW_ANOMALY = "flow_anomaly"
    INCIDENT = "incident"
    UNUSUAL_TRAJECTORY = "unusual_trajectory"


class TrafficMetrics(BaseModel):
    """Time-series traffic metrics model optimized for TimescaleDB.

    Stores aggregated traffic metrics with hypertable partitioning
    for high-performance time-series analytics.
    """

    __tablename__ = "traffic_metrics"

    # Time dimension (primary for TimescaleDB hypertable)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Measurement timestamp (hypertable partition key)",
    )

    # Location dimensions
    camera_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, comment="Source camera identifier"
    )
    zone_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Detection zone identifier"
    )
    lane_id: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Lane identifier"
    )

    # Aggregation metadata
    aggregation_period: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="1min",
        comment="Aggregation period (1min, 5min, 1hour, etc.)",
    )
    sample_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, comment="Number of samples in aggregation"
    )

    # Vehicle count metrics
    total_vehicles: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Total vehicle count"
    )
    vehicle_cars: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Car count"
    )
    vehicle_trucks: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Truck count"
    )
    vehicle_buses: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Bus count"
    )
    vehicle_motorcycles: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Motorcycle count"
    )
    vehicle_bicycles: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Bicycle count"
    )

    # Speed metrics (km/h)
    average_speed: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Average speed in km/h"
    )
    median_speed: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Median speed in km/h"
    )
    speed_85th_percentile: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="85th percentile speed in km/h"
    )
    min_speed: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Minimum observed speed in km/h"
    )
    max_speed: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Maximum observed speed in km/h"
    )
    speed_variance: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Speed variance"
    )

    # Traffic flow metrics
    flow_rate: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Traffic flow rate (vehicles/hour)"
    )
    headway_average: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Average headway between vehicles (seconds)"
    )
    gap_average: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Average gap between vehicles (meters)"
    )

    # Density and occupancy metrics
    traffic_density: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Traffic density (vehicles/km)"
    )
    occupancy_rate: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Lane occupancy rate (0.0-1.0)"
    )
    queue_length: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Queue length in meters"
    )

    # Congestion analysis
    congestion_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=CongestionLevel.FREE_FLOW,
        comment="Congestion level classification",
    )
    level_of_service: Mapped[str | None] = mapped_column(
        String(5), nullable=True, comment="Highway Level of Service (A-F)"
    )
    congestion_index: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Congestion index (0.0-1.0)"
    )

    # Direction-based metrics
    northbound_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Northbound vehicle count"
    )
    southbound_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Southbound vehicle count"
    )
    eastbound_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Eastbound vehicle count"
    )
    westbound_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Westbound vehicle count"
    )

    # Quality metrics
    detection_accuracy: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Detection accuracy score (0.0-1.0)"
    )
    tracking_accuracy: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Tracking accuracy score (0.0-1.0)"
    )
    data_completeness: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Data completeness ratio (0.0-1.0)"
    )

    # Weather and environmental factors
    weather_condition: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Weather condition during measurement"
    )
    visibility: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Visibility in meters"
    )
    temperature: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Temperature in Celsius"
    )

    # Processing metadata
    processing_latency_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Analytics processing latency in milliseconds"
    )
    model_version: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Analytics model version"
    )

    # Additional flexible attributes
    additional_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Additional metadata and custom metrics"
    )

    # TimescaleDB and analytics-optimized indexes
    __table_args__ = (
        # Primary time-series index (for hypertable)
        Index("idx_traffic_metrics_time_camera", "timestamp", "camera_id"),
        # Aggregation and filtering indexes
        Index("idx_traffic_metrics_period", "aggregation_period"),
        Index("idx_traffic_metrics_zone", "zone_id"),
        Index("idx_traffic_metrics_lane", "lane_id"),
        # Traffic metrics indexes
        Index("idx_traffic_metrics_congestion", "congestion_level"),
        Index("idx_traffic_metrics_flow", "flow_rate"),
        Index("idx_traffic_metrics_speed", "average_speed"),
        Index("idx_traffic_metrics_density", "traffic_density"),
        # Composite indexes for common queries
        Index("idx_traffic_metrics_camera_time", "camera_id", "timestamp"),
        Index("idx_traffic_metrics_congestion_time", "congestion_level", "timestamp"),
        {
            "comment": "Time-series traffic metrics optimized for TimescaleDB hypertables"
        },
    )


class RuleViolation(BaseModel):
    """Traffic rule violation detection results.

    Records detected violations of traffic rules with evidence
    and severity assessment.
    """

    __tablename__ = "rule_violations"

    # Violation identification
    violation_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True, comment="Type of traffic violation"
    )
    severity: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="medium",
        comment="Violation severity (low/medium/high/critical)",
    )

    # Time and location
    detection_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Violation detection timestamp",
    )
    camera_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, comment="Source camera identifier"
    )
    zone_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Detection zone identifier"
    )
    location_description: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="Human-readable location description"
    )

    # Vehicle and detection information
    vehicle_detection_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("detection_results.id"),
        nullable=True,
        comment="Related vehicle detection",
    )
    vehicle_track_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, index=True, comment="Vehicle tracking ID"
    )
    license_plate: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        index=True,
        comment="Vehicle license plate (if detected)",
    )
    vehicle_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Vehicle type classification"
    )

    # Violation specifics
    rule_definition: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, comment="Rule definition that was violated"
    )
    measured_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Measured value (e.g., speed for speeding violation)",
    )
    threshold_value: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Rule threshold value"
    )
    violation_duration: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Duration of violation in seconds"
    )

    # Confidence and validation
    detection_confidence: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Violation detection confidence (0.0-1.0)"
    )
    false_positive_probability: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Estimated false positive probability"
    )
    human_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, comment="Human verification status"
    )
    verification_notes: Mapped[str | None] = mapped_column(
        String(500), nullable=True, comment="Human verification notes"
    )

    # Evidence and documentation
    evidence_frame_ids: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True, comment="Frame IDs containing violation evidence"
    )
    evidence_video_url: Mapped[str | None] = mapped_column(
        String(500), nullable=True, comment="URL to violation video evidence"
    )
    evidence_images: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True, comment="URLs to violation image evidence"
    )

    # Processing and model information
    detection_model: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Model used for violation detection"
    )
    rule_engine_version: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Rule engine version"
    )
    processing_time_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Processing time in milliseconds"
    )

    # Status and resolution
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="active",
        comment="Violation status (active/resolved/dismissed)",
    )
    resolution_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Resolution timestamp"
    )
    resolution_action: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Action taken to resolve violation"
    )

    # Additional metadata
    additional_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Additional violation-specific data"
    )

    # Relationships
    vehicle_detection = relationship(
        "DetectionResult", back_populates="rule_violations"
    )

    __table_args__ = (
        # Primary query indexes
        Index("idx_violations_time", "detection_time"),
        Index("idx_violations_camera", "camera_id"),
        Index("idx_violations_type", "violation_type"),
        Index("idx_violations_severity", "severity"),
        # Vehicle tracking indexes
        Index("idx_violations_track_id", "vehicle_track_id"),
        Index("idx_violations_plate", "license_plate"),
        # Status and verification indexes
        Index("idx_violations_status", "status"),
        Index("idx_violations_verified", "human_verified"),
        # Composite indexes for common queries
        Index("idx_violations_camera_time", "camera_id", "detection_time"),
        Index("idx_violations_type_time", "violation_type", "detection_time"),
        Index(
            "idx_violations_active",
            "status",
            "detection_time",
            postgresql_where=text("status = 'active'"),
        ),
        {"comment": "Traffic rule violation records with evidence tracking"},
    )


class VehicleTrajectory(BaseModel):
    """Vehicle trajectory analysis and path tracking.

    Stores analyzed vehicle paths for anomaly detection,
    behavior analysis, and traffic pattern recognition.
    """

    __tablename__ = "vehicle_trajectories"

    # Vehicle identification
    vehicle_track_id: Mapped[int] = mapped_column(
        Integer, nullable=False, index=True, comment="Vehicle tracking ID"
    )
    camera_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, comment="Source camera identifier"
    )

    # Trajectory time span
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Trajectory start timestamp",
    )
    end_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Trajectory end timestamp",
    )
    duration_seconds: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Trajectory duration in seconds"
    )

    # Path geometry and statistics
    path_points: Mapped[list[dict[str, float]]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Trajectory path points with coordinates and timestamps",
    )
    total_distance: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Total distance traveled in meters"
    )
    straight_line_distance: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Straight-line distance in meters"
    )
    path_efficiency: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Path efficiency ratio (straight_line/total_distance)",
    )

    # Speed analysis
    average_speed: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Average speed in km/h"
    )
    max_speed: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Maximum speed in km/h"
    )
    speed_variance: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Speed variance"
    )
    acceleration_events: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of significant acceleration events",
    )
    deceleration_events: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of significant deceleration events",
    )

    # Direction and movement analysis
    primary_direction: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="Primary movement direction"
    )
    direction_changes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of significant direction changes",
    )
    turning_points: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB, nullable=True, comment="Significant turning points in trajectory"
    )

    # Zone and lane analysis
    zones_visited: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True, comment="Detection zones visited during trajectory"
    )
    lanes_used: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True, comment="Lanes used during trajectory"
    )
    lane_changes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Number of lane changes"
    )

    # Vehicle classification
    vehicle_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Vehicle type classification"
    )
    vehicle_size_category: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="Vehicle size category (small/medium/large)"
    )
    license_plate: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        index=True,
        comment="Vehicle license plate (if detected)",
    )

    # Anomaly and behavior flags
    is_anomalous: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, comment="Trajectory marked as anomalous"
    )
    anomaly_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Anomaly detection score (0.0-1.0)"
    )
    anomaly_reasons: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True, comment="Reasons for anomaly classification"
    )

    # Behavior analysis
    stopping_events: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Number of stopping events"
    )
    idle_time_seconds: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, comment="Total idle time in seconds"
    )
    aggressive_behavior_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Aggressive driving behavior score (0.0-1.0)"
    )

    # Quality and confidence metrics
    tracking_quality: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Overall tracking quality score (0.0-1.0)"
    )
    path_completeness: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Path completeness ratio (0.0-1.0)"
    )
    occlusion_percentage: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Percentage of trajectory under occlusion"
    )

    # Processing metadata
    analysis_model: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Model used for trajectory analysis"
    )
    processing_time_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Analysis processing time in milliseconds"
    )

    # Additional analysis data
    additional_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Additional trajectory metrics and analysis"
    )

    __table_args__ = (
        # Primary query indexes
        Index("idx_trajectory_track_id", "vehicle_track_id"),
        Index("idx_trajectory_camera", "camera_id"),
        Index("idx_trajectory_start_time", "start_time"),
        Index("idx_trajectory_duration", "duration_seconds"),
        # Speed and behavior indexes
        Index("idx_trajectory_avg_speed", "average_speed"),
        Index("idx_trajectory_max_speed", "max_speed"),
        Index("idx_trajectory_anomalous", "is_anomalous"),
        # Vehicle identification
        Index("idx_trajectory_plate", "license_plate"),
        Index("idx_trajectory_vehicle_type", "vehicle_type"),
        # Quality indexes
        Index("idx_trajectory_quality", "tracking_quality"),
        # Composite indexes for analytics
        Index("idx_trajectory_camera_time", "camera_id", "start_time"),
        Index(
            "idx_trajectory_anomalies",
            "is_anomalous",
            "anomaly_score",
            postgresql_where=text("is_anomalous = true"),
        ),
        {
            "comment": "Vehicle trajectory analysis with path geometry and behavior metrics"
        },
    )


class TrafficAnomaly(BaseModel):
    """Traffic anomaly detection results.

    Records detected anomalies in traffic patterns, vehicle behavior,
    or system performance with detailed analysis.
    """

    __tablename__ = "traffic_anomalies"

    # Anomaly identification
    anomaly_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True, comment="Type of detected anomaly"
    )
    severity: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="Anomaly severity (low/medium/high/critical)",
    )

    # Detection time and location
    detection_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Anomaly detection timestamp",
    )
    start_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Estimated anomaly start time"
    )
    end_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Estimated anomaly end time"
    )

    # Location information
    camera_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, comment="Source camera identifier"
    )
    zone_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Affected detection zone"
    )
    affected_area: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Geographic area affected by anomaly"
    )

    # Anomaly detection details
    anomaly_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Anomaly detection score (0.0-1.0, higher = more anomalous)",
    )
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Detection confidence (0.0-1.0)"
    )
    detection_method: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="Method used for anomaly detection"
    )

    # Pattern and baseline comparison
    baseline_value: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Baseline/expected value"
    )
    observed_value: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Observed anomalous value"
    )
    deviation_magnitude: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Magnitude of deviation from baseline"
    )
    statistical_significance: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Statistical significance of deviation"
    )

    # Affected metrics and patterns
    affected_metrics: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True, comment="Traffic metrics affected by the anomaly"
    )
    pattern_description: Mapped[str | None] = mapped_column(
        String(500), nullable=True, comment="Description of anomalous pattern"
    )

    # Related entities
    related_vehicles: Mapped[list[int] | None] = mapped_column(
        JSONB, nullable=True, comment="Vehicle track IDs related to anomaly"
    )
    related_incidents: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True, comment="Related incident or violation IDs"
    )

    # Impact assessment
    traffic_impact: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Assessed impact on traffic flow"
    )
    estimated_delay: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Estimated traffic delay in minutes"
    )
    affected_vehicle_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Number of vehicles potentially affected"
    )

    # Root cause analysis
    probable_cause: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="Probable cause of anomaly"
    )
    contributing_factors: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True, comment="Contributing factors to the anomaly"
    )
    environmental_conditions: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Environmental conditions during anomaly"
    )

    # Detection model information
    model_name: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="Anomaly detection model name"
    )
    model_version: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="Model version"
    )
    model_confidence: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Model confidence in detection"
    )

    # Validation and feedback
    human_validated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, comment="Human validation status"
    )
    validation_result: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        comment="Validation result (confirmed/false_positive/inconclusive)",
    )
    validator_notes: Mapped[str | None] = mapped_column(
        String(500), nullable=True, comment="Human validator notes"
    )

    # Resolution and follow-up
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="detected",
        comment="Anomaly status (detected/investigating/resolved)",
    )
    resolution_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Resolution timestamp"
    )
    resolution_action: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="Action taken to resolve anomaly"
    )

    # Processing metadata
    processing_time_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Anomaly detection processing time"
    )
    data_quality_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Quality score of input data (0.0-1.0)"
    )

    # Additional analysis data
    detailed_analysis: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Detailed anomaly analysis results"
    )

    __table_args__ = (
        # Primary query indexes
        Index("idx_anomaly_time", "detection_time"),
        Index("idx_anomaly_camera", "camera_id"),
        Index("idx_anomaly_type", "anomaly_type"),
        Index("idx_anomaly_severity", "severity"),
        # Scoring and confidence indexes
        Index("idx_anomaly_score", "anomaly_score"),
        Index("idx_anomaly_confidence", "confidence"),
        # Status and validation indexes
        Index("idx_anomaly_status", "status"),
        Index("idx_anomaly_validated", "human_validated"),
        # Model tracking indexes
        Index("idx_anomaly_model", "model_name", "model_version"),
        # Composite indexes for analytics
        Index("idx_anomaly_camera_time", "camera_id", "detection_time"),
        Index("idx_anomaly_type_time", "anomaly_type", "detection_time"),
        Index("idx_anomaly_severity_score", "severity", "anomaly_score"),
        Index(
            "idx_anomaly_active",
            "status",
            "detection_time",
            postgresql_where=text("status IN ('detected', 'investigating')"),
        ),
        {"comment": "Traffic anomaly detection results with detailed analysis"},
    )


class SpeedLimit(BaseModel):
    """Speed limit configuration by zone and vehicle type.

    Stores dynamic speed limits that can be looked up by traffic zone
    and vehicle classification for accurate violation detection.
    """

    __tablename__ = "speed_limits"

    # Zone and location information
    zone_id: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, comment="Traffic zone identifier"
    )
    zone_name: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="Human-readable zone name"
    )
    location_description: Mapped[str | None] = mapped_column(
        String(300), nullable=True, comment="Detailed location description"
    )

    # Vehicle type classification
    vehicle_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        default="general",
        comment="Vehicle type (general/car/truck/motorcycle/bus/emergency)"
    )

    # Speed limit values (km/h)
    speed_limit_kmh: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Speed limit in kilometers per hour"
    )
    tolerance_kmh: Mapped[float] = mapped_column(
        Float, nullable=False, default=5.0, comment="Tolerance threshold in km/h"
    )

    # Time-based restrictions
    effective_start_time: Mapped[str | None] = mapped_column(
        String(8), nullable=True, comment="Daily start time (HH:MM:SS format)"
    )
    effective_end_time: Mapped[str | None] = mapped_column(
        String(8), nullable=True, comment="Daily end time (HH:MM:SS format)"
    )
    days_of_week: Mapped[list[int] | None] = mapped_column(
        JSONB, nullable=True, comment="Days of week (0=Monday to 6=Sunday), null=all days"
    )

    # Environmental conditions
    weather_conditions: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=True, comment="Weather conditions when limit applies (null=all conditions)"
    )
    minimum_visibility: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Minimum visibility in meters for this limit"
    )

    # Enforcement configuration
    enforcement_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, comment="Whether to enforce this speed limit"
    )
    warning_threshold: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Speed threshold for warnings (km/h over limit)"
    )
    violation_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=10.0, comment="Speed threshold for violations (km/h over limit)"
    )

    # Priority and validity
    priority: Mapped[int] = mapped_column(
        Integer, nullable=False, default=100, comment="Priority when multiple limits apply (lower=higher priority)"
    )
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, comment="Speed limit validity start date"
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Speed limit validity end date (null=permanent)"
    )

    # Geographic boundaries (optional)
    geographic_bounds: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Geographic boundaries for the speed limit zone"
    )

    # Administrative information
    authority: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Authority that set this speed limit"
    )
    regulation_reference: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="Legal regulation reference"
    )
    last_updated_by: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="User who last updated this limit"
    )

    # Performance optimization indexes
    __table_args__ = (
        # Primary lookup indexes
        Index("idx_speed_limit_zone_vehicle", "zone_id", "vehicle_type"),
        Index("idx_speed_limit_zone_active", "zone_id", "enforcement_enabled"),
        Index("idx_speed_limit_vehicle_type", "vehicle_type"),
        # Time-based filtering
        Index("idx_speed_limit_validity", "valid_from", "valid_until"),
        Index("idx_speed_limit_priority", "priority"),
        # Composite index for efficient lookups
        Index("idx_speed_limit_lookup", "zone_id", "vehicle_type", "enforcement_enabled", "valid_from"),
        # Partial index for active limits
        Index(
            "idx_speed_limit_active",
            "zone_id",
            "vehicle_type",
            "priority",
            postgresql_where=text("enforcement_enabled = true AND (valid_until IS NULL OR valid_until > NOW())")
        ),
        {"comment": "Dynamic speed limits by zone and vehicle type with time-based restrictions"}
    )

    def is_valid_at(self, check_time: datetime) -> bool:
        """Check if speed limit is valid at given time.

        Args:
            check_time: Time to check validity

        Returns:
            True if speed limit is valid at the given time
        """
        # Check date validity
        if check_time < self.valid_from:
            return False
        if self.valid_until and check_time > self.valid_until:
            return False

        # Check day of week if specified
        if self.days_of_week:
            current_day = check_time.weekday()  # 0=Monday, 6=Sunday
            if current_day not in self.days_of_week:
                return False

        # Check time of day if specified
        if self.effective_start_time and self.effective_end_time:
            current_time = check_time.strftime("%H:%M:%S")
            if not (self.effective_start_time <= current_time <= self.effective_end_time):
                return False

        return True

    def applies_to_conditions(self, weather: str | None = None, visibility: float | None = None) -> bool:
        """Check if speed limit applies to current environmental conditions.

        Args:
            weather: Current weather condition
            visibility: Current visibility in meters

        Returns:
            True if speed limit applies to the conditions
        """
        # Check weather conditions if specified
        if self.weather_conditions and weather:
            if weather.lower() not in [w.lower() for w in self.weather_conditions]:
                return False

        # Check visibility if specified
        if self.minimum_visibility and visibility:
            if visibility < self.minimum_visibility:
                return False

        return True

    @property
    def effective_limit_with_tolerance(self) -> float:
        """Get effective speed limit including tolerance."""
        return self.speed_limit_kmh + self.tolerance_kmh

    def __repr__(self) -> str:
        return f"<SpeedLimit(zone={self.zone_id}, vehicle={self.vehicle_type}, limit={self.speed_limit_kmh})>"


class AlertNotification(BaseModel):
    """Alert notification tracking for rule violations and anomalies.

    Tracks delivery status and recipient responses for generated alerts.
    """

    __tablename__ = "alert_notifications"

    # Alert reference
    alert_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Type of alert (violation/anomaly/incident)",
    )
    reference_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="ID of referenced violation/anomaly",
    )

    # Notification details
    notification_channel: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Notification channel (email/sms/webhook/push)",
    )
    recipient: Mapped[str] = mapped_column(
        String(200), nullable=False, comment="Notification recipient identifier"
    )
    priority: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="medium",
        comment="Alert priority (low/medium/high/critical)",
    )

    # Timing
    created_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Alert creation timestamp",
    )
    sent_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Notification sent timestamp"
    )
    acknowledged_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Alert acknowledgment timestamp"
    )

    # Delivery status
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="pending",
        comment="Notification status (pending/sent/delivered/failed/acknowledged)",
    )
    delivery_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Number of delivery attempts"
    )
    last_attempt_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last delivery attempt timestamp",
    )

    # Content and formatting
    subject: Mapped[str | None] = mapped_column(
        String(200), nullable=True, comment="Alert subject/title"
    )
    message_content: Mapped[str] = mapped_column(
        String(2000), nullable=False, comment="Alert message content"
    )
    message_format: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="text",
        comment="Message format (text/html/json)",
    )

    # Response tracking
    acknowledged_by: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="User who acknowledged the alert"
    )
    response_action: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Response action taken"
    )
    response_notes: Mapped[str | None] = mapped_column(
        String(500), nullable=True, comment="Response notes"
    )

    # Delivery metadata
    external_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="External system notification ID"
    )
    delivery_details: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Delivery service response details"
    )
    error_message: Mapped[str | None] = mapped_column(
        String(500), nullable=True, comment="Error message if delivery failed"
    )

    __table_args__ = (
        # Primary query indexes
        Index("idx_alert_created_time", "created_time"),
        Index("idx_alert_reference", "alert_type", "reference_id"),
        Index("idx_alert_recipient", "recipient"),
        Index("idx_alert_status", "status"),
        # Priority and response tracking
        Index("idx_alert_priority", "priority"),
        Index("idx_alert_acknowledged", "acknowledged_time"),
        # Delivery tracking
        Index("idx_alert_channel", "notification_channel"),
        Index("idx_alert_attempts", "delivery_attempts"),
        # Composite indexes for common queries
        Index(
            "idx_alert_pending",
            "status",
            "created_time",
            postgresql_where=text("status IN ('pending', 'failed')"),
        ),
        Index("idx_alert_recipient_status", "recipient", "status"),
        {"comment": "Alert notification delivery and response tracking"},
    )
