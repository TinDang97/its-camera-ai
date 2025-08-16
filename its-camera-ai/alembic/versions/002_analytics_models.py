"""Add analytics models for traffic monitoring.

Creates analytics tables for traffic metrics, rule violations,
vehicle trajectories, anomaly detection, speed limits, and alert notifications
with TimescaleDB optimizations.

Revision ID: 002
Revises: 001
Create Date: 2025-01-16 00:00:00.000000
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create analytics tables with TimescaleDB optimizations."""

    # Enable TimescaleDB extension if not already enabled
    op.execute('CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;')

    # TrafficMetrics table (hypertable for TimescaleDB)
    op.create_table(
        'traffic_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        # Time dimension (primary for TimescaleDB hypertable)
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False,
                  comment='Measurement timestamp (hypertable partition key)'),
        # Location dimensions
        sa.Column('camera_id', sa.String(100), nullable=False,
                  comment='Source camera identifier'),
        sa.Column('zone_id', sa.String(100), nullable=True,
                  comment='Detection zone identifier'),
        sa.Column('lane_id', sa.String(50), nullable=True,
                  comment='Lane identifier'),
        # Aggregation metadata
        sa.Column('aggregation_period', sa.String(20), nullable=False, default='1min',
                  comment='Aggregation period (1min, 5min, 1hour, etc.)'),
        sa.Column('sample_count', sa.Integer(), nullable=False, default=1,
                  comment='Number of samples in aggregation'),
        # Vehicle count metrics
        sa.Column('total_vehicles', sa.Integer(), nullable=False, default=0,
                  comment='Total vehicle count'),
        sa.Column('vehicle_cars', sa.Integer(), nullable=False, default=0,
                  comment='Car count'),
        sa.Column('vehicle_trucks', sa.Integer(), nullable=False, default=0,
                  comment='Truck count'),
        sa.Column('vehicle_buses', sa.Integer(), nullable=False, default=0,
                  comment='Bus count'),
        sa.Column('vehicle_motorcycles', sa.Integer(), nullable=False, default=0,
                  comment='Motorcycle count'),
        sa.Column('vehicle_bicycles', sa.Integer(), nullable=False, default=0,
                  comment='Bicycle count'),
        # Speed metrics (km/h)
        sa.Column('average_speed', sa.Float(), nullable=True,
                  comment='Average speed in km/h'),
        sa.Column('median_speed', sa.Float(), nullable=True,
                  comment='Median speed in km/h'),
        sa.Column('speed_85th_percentile', sa.Float(), nullable=True,
                  comment='85th percentile speed in km/h'),
        sa.Column('min_speed', sa.Float(), nullable=True,
                  comment='Minimum observed speed in km/h'),
        sa.Column('max_speed', sa.Float(), nullable=True,
                  comment='Maximum observed speed in km/h'),
        sa.Column('speed_variance', sa.Float(), nullable=True,
                  comment='Speed variance'),
        # Traffic flow metrics
        sa.Column('flow_rate', sa.Float(), nullable=True,
                  comment='Traffic flow rate (vehicles/hour)'),
        sa.Column('headway_average', sa.Float(), nullable=True,
                  comment='Average headway between vehicles (seconds)'),
        sa.Column('gap_average', sa.Float(), nullable=True,
                  comment='Average gap between vehicles (meters)'),
        # Density and occupancy metrics
        sa.Column('traffic_density', sa.Float(), nullable=True,
                  comment='Traffic density (vehicles/km)'),
        sa.Column('occupancy_rate', sa.Float(), nullable=True,
                  comment='Lane occupancy rate (0.0-1.0)'),
        sa.Column('queue_length', sa.Float(), nullable=True,
                  comment='Queue length in meters'),
        # Congestion analysis
        sa.Column('congestion_level', sa.String(20), nullable=False, default='free_flow',
                  comment='Congestion level classification'),
        sa.Column('level_of_service', sa.String(5), nullable=True,
                  comment='Highway Level of Service (A-F)'),
        sa.Column('congestion_index', sa.Float(), nullable=True,
                  comment='Congestion index (0.0-1.0)'),
        # Direction-based metrics
        sa.Column('northbound_count', sa.Integer(), nullable=False, default=0,
                  comment='Northbound vehicle count'),
        sa.Column('southbound_count', sa.Integer(), nullable=False, default=0,
                  comment='Southbound vehicle count'),
        sa.Column('eastbound_count', sa.Integer(), nullable=False, default=0,
                  comment='Eastbound vehicle count'),
        sa.Column('westbound_count', sa.Integer(), nullable=False, default=0,
                  comment='Westbound vehicle count'),
        # Quality metrics
        sa.Column('detection_accuracy', sa.Float(), nullable=True,
                  comment='Detection accuracy score (0.0-1.0)'),
        sa.Column('tracking_accuracy', sa.Float(), nullable=True,
                  comment='Tracking accuracy score (0.0-1.0)'),
        sa.Column('data_completeness', sa.Float(), nullable=True,
                  comment='Data completeness ratio (0.0-1.0)'),
        # Weather and environmental factors
        sa.Column('weather_condition', sa.String(50), nullable=True,
                  comment='Weather condition during measurement'),
        sa.Column('visibility', sa.Float(), nullable=True,
                  comment='Visibility in meters'),
        sa.Column('temperature', sa.Float(), nullable=True,
                  comment='Temperature in Celsius'),
        # Processing metadata
        sa.Column('processing_latency_ms', sa.Float(), nullable=True,
                  comment='Analytics processing latency in milliseconds'),
        sa.Column('model_version', sa.String(50), nullable=True,
                  comment='Analytics model version'),
        # Additional flexible attributes
        sa.Column('additional_metadata', postgresql.JSONB(), nullable=True,
                  comment='Additional metadata and custom metrics'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        comment='Time-series traffic metrics optimized for TimescaleDB hypertables'
    )

    # Convert traffic_metrics to hypertable
    op.execute("""
        SELECT create_hypertable('traffic_metrics', 'timestamp', 
                                chunk_time_interval => INTERVAL '1 hour',
                                create_default_indexes => FALSE);
    """)

    # TimescaleDB and analytics-optimized indexes
    op.create_index('idx_traffic_metrics_time_camera', 'traffic_metrics', ['timestamp', 'camera_id'])
    op.create_index('idx_traffic_metrics_period', 'traffic_metrics', ['aggregation_period'])
    op.create_index('idx_traffic_metrics_zone', 'traffic_metrics', ['zone_id'])
    op.create_index('idx_traffic_metrics_lane', 'traffic_metrics', ['lane_id'])
    op.create_index('idx_traffic_metrics_congestion', 'traffic_metrics', ['congestion_level'])
    op.create_index('idx_traffic_metrics_flow', 'traffic_metrics', ['flow_rate'])
    op.create_index('idx_traffic_metrics_speed', 'traffic_metrics', ['average_speed'])
    op.create_index('idx_traffic_metrics_density', 'traffic_metrics', ['traffic_density'])
    op.create_index('idx_traffic_metrics_camera_time', 'traffic_metrics', ['camera_id', 'timestamp'])
    op.create_index('idx_traffic_metrics_congestion_time', 'traffic_metrics', ['congestion_level', 'timestamp'])

    # RuleViolation table
    op.create_table(
        'rule_violations',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        # Violation identification
        sa.Column('violation_type', sa.String(50), nullable=False,
                  comment='Type of traffic violation'),
        sa.Column('severity', sa.String(20), nullable=False, default='medium',
                  comment='Violation severity (low/medium/high/critical)'),
        # Time and location
        sa.Column('detection_time', sa.DateTime(timezone=True), nullable=False,
                  comment='Violation detection timestamp'),
        sa.Column('camera_id', sa.String(100), nullable=False,
                  comment='Source camera identifier'),
        sa.Column('zone_id', sa.String(100), nullable=True,
                  comment='Detection zone identifier'),
        sa.Column('location_description', sa.String(200), nullable=True,
                  comment='Human-readable location description'),
        # Vehicle and detection information
        sa.Column('vehicle_detection_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('vehicle_track_id', sa.Integer(), nullable=True,
                  comment='Vehicle tracking ID'),
        sa.Column('license_plate', sa.String(20), nullable=True,
                  comment='Vehicle license plate (if detected)'),
        sa.Column('vehicle_type', sa.String(50), nullable=True,
                  comment='Vehicle type classification'),
        # Violation specifics
        sa.Column('rule_definition', postgresql.JSONB(), nullable=False,
                  comment='Rule definition that was violated'),
        sa.Column('measured_value', sa.Float(), nullable=True,
                  comment='Measured value (e.g., speed for speeding violation)'),
        sa.Column('threshold_value', sa.Float(), nullable=True,
                  comment='Rule threshold value'),
        sa.Column('violation_duration', sa.Float(), nullable=True,
                  comment='Duration of violation in seconds'),
        # Confidence and validation
        sa.Column('detection_confidence', sa.Float(), nullable=False,
                  comment='Violation detection confidence (0.0-1.0)'),
        sa.Column('false_positive_probability', sa.Float(), nullable=True,
                  comment='Estimated false positive probability'),
        sa.Column('human_verified', sa.Boolean(), nullable=False, default=False,
                  comment='Human verification status'),
        sa.Column('verification_notes', sa.String(500), nullable=True,
                  comment='Human verification notes'),
        # Evidence and documentation
        sa.Column('evidence_frame_ids', postgresql.JSONB(), nullable=True,
                  comment='Frame IDs containing violation evidence'),
        sa.Column('evidence_video_url', sa.String(500), nullable=True,
                  comment='URL to violation video evidence'),
        sa.Column('evidence_images', postgresql.JSONB(), nullable=True,
                  comment='URLs to violation image evidence'),
        # Processing and model information
        sa.Column('detection_model', sa.String(100), nullable=True,
                  comment='Model used for violation detection'),
        sa.Column('rule_engine_version', sa.String(50), nullable=True,
                  comment='Rule engine version'),
        sa.Column('processing_time_ms', sa.Float(), nullable=True,
                  comment='Processing time in milliseconds'),
        # Status and resolution
        sa.Column('status', sa.String(20), nullable=False, default='active',
                  comment='Violation status (active/resolved/dismissed)'),
        sa.Column('resolution_time', sa.DateTime(timezone=True), nullable=True,
                  comment='Resolution timestamp'),
        sa.Column('resolution_action', sa.String(100), nullable=True,
                  comment='Action taken to resolve violation'),
        # Additional metadata
        sa.Column('additional_data', postgresql.JSONB(), nullable=True,
                  comment='Additional violation-specific data'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['vehicle_detection_id'], ['detection_results.id']),
        comment='Traffic rule violation records with evidence tracking'
    )

    # RuleViolation indexes
    op.create_index('idx_violations_time', 'rule_violations', ['detection_time'])
    op.create_index('idx_violations_camera', 'rule_violations', ['camera_id'])
    op.create_index('idx_violations_type', 'rule_violations', ['violation_type'])
    op.create_index('idx_violations_severity', 'rule_violations', ['severity'])
    op.create_index('idx_violations_track_id', 'rule_violations', ['vehicle_track_id'])
    op.create_index('idx_violations_plate', 'rule_violations', ['license_plate'])
    op.create_index('idx_violations_status', 'rule_violations', ['status'])
    op.create_index('idx_violations_verified', 'rule_violations', ['human_verified'])
    op.create_index('idx_violations_camera_time', 'rule_violations', ['camera_id', 'detection_time'])
    op.create_index('idx_violations_type_time', 'rule_violations', ['violation_type', 'detection_time'])
    op.create_index('idx_violations_active', 'rule_violations', ['status', 'detection_time'],
                   postgresql_where=sa.text("status = 'active'"))

    # VehicleTrajectory table
    op.create_table(
        'vehicle_trajectories',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        # Vehicle identification
        sa.Column('vehicle_track_id', sa.Integer(), nullable=False,
                  comment='Vehicle tracking ID'),
        sa.Column('camera_id', sa.String(100), nullable=False,
                  comment='Source camera identifier'),
        # Trajectory time span
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=False,
                  comment='Trajectory start timestamp'),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=False,
                  comment='Trajectory end timestamp'),
        sa.Column('duration_seconds', sa.Float(), nullable=False,
                  comment='Trajectory duration in seconds'),
        # Path geometry and statistics
        sa.Column('path_points', postgresql.JSONB(), nullable=False,
                  comment='Trajectory path points with coordinates and timestamps'),
        sa.Column('total_distance', sa.Float(), nullable=False,
                  comment='Total distance traveled in meters'),
        sa.Column('straight_line_distance', sa.Float(), nullable=False,
                  comment='Straight-line distance in meters'),
        sa.Column('path_efficiency', sa.Float(), nullable=False,
                  comment='Path efficiency ratio (straight_line/total_distance)'),
        # Speed analysis
        sa.Column('average_speed', sa.Float(), nullable=False,
                  comment='Average speed in km/h'),
        sa.Column('max_speed', sa.Float(), nullable=False,
                  comment='Maximum speed in km/h'),
        sa.Column('speed_variance', sa.Float(), nullable=True,
                  comment='Speed variance'),
        sa.Column('acceleration_events', sa.Integer(), nullable=False, default=0,
                  comment='Number of significant acceleration events'),
        sa.Column('deceleration_events', sa.Integer(), nullable=False, default=0,
                  comment='Number of significant deceleration events'),
        # Direction and movement analysis
        sa.Column('primary_direction', sa.String(20), nullable=True,
                  comment='Primary movement direction'),
        sa.Column('direction_changes', sa.Integer(), nullable=False, default=0,
                  comment='Number of significant direction changes'),
        sa.Column('turning_points', postgresql.JSONB(), nullable=True,
                  comment='Significant turning points in trajectory'),
        # Zone and lane analysis
        sa.Column('zones_visited', postgresql.JSONB(), nullable=True,
                  comment='Detection zones visited during trajectory'),
        sa.Column('lanes_used', postgresql.JSONB(), nullable=True,
                  comment='Lanes used during trajectory'),
        sa.Column('lane_changes', sa.Integer(), nullable=False, default=0,
                  comment='Number of lane changes'),
        # Vehicle classification
        sa.Column('vehicle_type', sa.String(50), nullable=True,
                  comment='Vehicle type classification'),
        sa.Column('vehicle_size_category', sa.String(20), nullable=True,
                  comment='Vehicle size category (small/medium/large)'),
        sa.Column('license_plate', sa.String(20), nullable=True,
                  comment='Vehicle license plate (if detected)'),
        # Anomaly and behavior flags
        sa.Column('is_anomalous', sa.Boolean(), nullable=False, default=False,
                  comment='Trajectory marked as anomalous'),
        sa.Column('anomaly_score', sa.Float(), nullable=True,
                  comment='Anomaly detection score (0.0-1.0)'),
        sa.Column('anomaly_reasons', postgresql.JSONB(), nullable=True,
                  comment='Reasons for anomaly classification'),
        # Behavior analysis
        sa.Column('stopping_events', sa.Integer(), nullable=False, default=0,
                  comment='Number of stopping events'),
        sa.Column('idle_time_seconds', sa.Float(), nullable=False, default=0.0,
                  comment='Total idle time in seconds'),
        sa.Column('aggressive_behavior_score', sa.Float(), nullable=True,
                  comment='Aggressive driving behavior score (0.0-1.0)'),
        # Quality and confidence metrics
        sa.Column('tracking_quality', sa.Float(), nullable=False,
                  comment='Overall tracking quality score (0.0-1.0)'),
        sa.Column('path_completeness', sa.Float(), nullable=False,
                  comment='Path completeness ratio (0.0-1.0)'),
        sa.Column('occlusion_percentage', sa.Float(), nullable=True,
                  comment='Percentage of trajectory under occlusion'),
        # Processing metadata
        sa.Column('analysis_model', sa.String(100), nullable=True,
                  comment='Model used for trajectory analysis'),
        sa.Column('processing_time_ms', sa.Float(), nullable=True,
                  comment='Analysis processing time in milliseconds'),
        # Additional analysis data
        sa.Column('additional_metrics', postgresql.JSONB(), nullable=True,
                  comment='Additional trajectory metrics and analysis'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        comment='Vehicle trajectory analysis with path geometry and behavior metrics'
    )

    # VehicleTrajectory indexes
    op.create_index('idx_trajectory_track_id', 'vehicle_trajectories', ['vehicle_track_id'])
    op.create_index('idx_trajectory_camera', 'vehicle_trajectories', ['camera_id'])
    op.create_index('idx_trajectory_start_time', 'vehicle_trajectories', ['start_time'])
    op.create_index('idx_trajectory_duration', 'vehicle_trajectories', ['duration_seconds'])
    op.create_index('idx_trajectory_avg_speed', 'vehicle_trajectories', ['average_speed'])
    op.create_index('idx_trajectory_max_speed', 'vehicle_trajectories', ['max_speed'])
    op.create_index('idx_trajectory_anomalous', 'vehicle_trajectories', ['is_anomalous'])
    op.create_index('idx_trajectory_plate', 'vehicle_trajectories', ['license_plate'])
    op.create_index('idx_trajectory_vehicle_type', 'vehicle_trajectories', ['vehicle_type'])
    op.create_index('idx_trajectory_quality', 'vehicle_trajectories', ['tracking_quality'])
    op.create_index('idx_trajectory_camera_time', 'vehicle_trajectories', ['camera_id', 'start_time'])
    op.create_index('idx_trajectory_anomalies', 'vehicle_trajectories', ['is_anomalous', 'anomaly_score'],
                   postgresql_where=sa.text('is_anomalous = true'))

    # TrafficAnomaly table
    op.create_table(
        'traffic_anomalies',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        # Anomaly identification
        sa.Column('anomaly_type', sa.String(50), nullable=False,
                  comment='Type of detected anomaly'),
        sa.Column('severity', sa.String(20), nullable=False,
                  comment='Anomaly severity (low/medium/high/critical)'),
        # Detection time and location
        sa.Column('detection_time', sa.DateTime(timezone=True), nullable=False,
                  comment='Anomaly detection timestamp'),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=True,
                  comment='Estimated anomaly start time'),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True,
                  comment='Estimated anomaly end time'),
        # Location information
        sa.Column('camera_id', sa.String(100), nullable=False,
                  comment='Source camera identifier'),
        sa.Column('zone_id', sa.String(100), nullable=True,
                  comment='Affected detection zone'),
        sa.Column('affected_area', postgresql.JSONB(), nullable=True,
                  comment='Geographic area affected by anomaly'),
        # Anomaly detection details
        sa.Column('anomaly_score', sa.Float(), nullable=False,
                  comment='Anomaly detection score (0.0-1.0, higher = more anomalous)'),
        sa.Column('confidence', sa.Float(), nullable=False,
                  comment='Detection confidence (0.0-1.0)'),
        sa.Column('detection_method', sa.String(100), nullable=False,
                  comment='Method used for anomaly detection'),
        # Pattern and baseline comparison
        sa.Column('baseline_value', sa.Float(), nullable=True,
                  comment='Baseline/expected value'),
        sa.Column('observed_value', sa.Float(), nullable=True,
                  comment='Observed anomalous value'),
        sa.Column('deviation_magnitude', sa.Float(), nullable=True,
                  comment='Magnitude of deviation from baseline'),
        sa.Column('statistical_significance', sa.Float(), nullable=True,
                  comment='Statistical significance of deviation'),
        # Affected metrics and patterns
        sa.Column('affected_metrics', postgresql.JSONB(), nullable=True,
                  comment='Traffic metrics affected by the anomaly'),
        sa.Column('pattern_description', sa.String(500), nullable=True,
                  comment='Description of anomalous pattern'),
        # Related entities
        sa.Column('related_vehicles', postgresql.JSONB(), nullable=True,
                  comment='Vehicle track IDs related to anomaly'),
        sa.Column('related_incidents', postgresql.JSONB(), nullable=True,
                  comment='Related incident or violation IDs'),
        # Impact assessment
        sa.Column('traffic_impact', sa.String(100), nullable=True,
                  comment='Assessed impact on traffic flow'),
        sa.Column('estimated_delay', sa.Float(), nullable=True,
                  comment='Estimated traffic delay in minutes'),
        sa.Column('affected_vehicle_count', sa.Integer(), nullable=True,
                  comment='Number of vehicles potentially affected'),
        # Root cause analysis
        sa.Column('probable_cause', sa.String(200), nullable=True,
                  comment='Probable cause of anomaly'),
        sa.Column('contributing_factors', postgresql.JSONB(), nullable=True,
                  comment='Contributing factors to the anomaly'),
        sa.Column('environmental_conditions', postgresql.JSONB(), nullable=True,
                  comment='Environmental conditions during anomaly'),
        # Detection model information
        sa.Column('model_name', sa.String(100), nullable=False,
                  comment='Anomaly detection model name'),
        sa.Column('model_version', sa.String(50), nullable=False,
                  comment='Model version'),
        sa.Column('model_confidence', sa.Float(), nullable=False,
                  comment='Model confidence in detection'),
        # Validation and feedback
        sa.Column('human_validated', sa.Boolean(), nullable=False, default=False,
                  comment='Human validation status'),
        sa.Column('validation_result', sa.String(20), nullable=True,
                  comment='Validation result (confirmed/false_positive/inconclusive)'),
        sa.Column('validator_notes', sa.String(500), nullable=True,
                  comment='Human validator notes'),
        # Resolution and follow-up
        sa.Column('status', sa.String(20), nullable=False, default='detected',
                  comment='Anomaly status (detected/investigating/resolved)'),
        sa.Column('resolution_time', sa.DateTime(timezone=True), nullable=True,
                  comment='Resolution timestamp'),
        sa.Column('resolution_action', sa.String(200), nullable=True,
                  comment='Action taken to resolve anomaly'),
        # Processing metadata
        sa.Column('processing_time_ms', sa.Float(), nullable=True,
                  comment='Anomaly detection processing time'),
        sa.Column('data_quality_score', sa.Float(), nullable=True,
                  comment='Quality score of input data (0.0-1.0)'),
        # Additional analysis data
        sa.Column('detailed_analysis', postgresql.JSONB(), nullable=True,
                  comment='Detailed anomaly analysis results'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        comment='Traffic anomaly detection results with detailed analysis'
    )

    # TrafficAnomaly indexes
    op.create_index('idx_anomaly_time', 'traffic_anomalies', ['detection_time'])
    op.create_index('idx_anomaly_camera', 'traffic_anomalies', ['camera_id'])
    op.create_index('idx_anomaly_type', 'traffic_anomalies', ['anomaly_type'])
    op.create_index('idx_anomaly_severity', 'traffic_anomalies', ['severity'])
    op.create_index('idx_anomaly_score', 'traffic_anomalies', ['anomaly_score'])
    op.create_index('idx_anomaly_confidence', 'traffic_anomalies', ['confidence'])
    op.create_index('idx_anomaly_status', 'traffic_anomalies', ['status'])
    op.create_index('idx_anomaly_validated', 'traffic_anomalies', ['human_validated'])
    op.create_index('idx_anomaly_model', 'traffic_anomalies', ['model_name', 'model_version'])
    op.create_index('idx_anomaly_camera_time', 'traffic_anomalies', ['camera_id', 'detection_time'])
    op.create_index('idx_anomaly_type_time', 'traffic_anomalies', ['anomaly_type', 'detection_time'])
    op.create_index('idx_anomaly_severity_score', 'traffic_anomalies', ['severity', 'anomaly_score'])
    op.create_index('idx_anomaly_active', 'traffic_anomalies', ['status', 'detection_time'],
                   postgresql_where=sa.text("status IN ('detected', 'investigating')"))

    # SpeedLimit table
    op.create_table(
        'speed_limits',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        # Zone and location information
        sa.Column('zone_id', sa.String(100), nullable=False,
                  comment='Traffic zone identifier'),
        sa.Column('zone_name', sa.String(200), nullable=True,
                  comment='Human-readable zone name'),
        sa.Column('location_description', sa.String(300), nullable=True,
                  comment='Detailed location description'),
        # Vehicle type classification
        sa.Column('vehicle_type', sa.String(50), nullable=False, default='general',
                  comment='Vehicle type (general/car/truck/motorcycle/bus/emergency)'),
        # Speed limit values (km/h)
        sa.Column('speed_limit_kmh', sa.Float(), nullable=False,
                  comment='Speed limit in kilometers per hour'),
        sa.Column('tolerance_kmh', sa.Float(), nullable=False, default=5.0,
                  comment='Tolerance threshold in km/h'),
        # Time-based restrictions
        sa.Column('effective_start_time', sa.String(8), nullable=True,
                  comment='Daily start time (HH:MM:SS format)'),
        sa.Column('effective_end_time', sa.String(8), nullable=True,
                  comment='Daily end time (HH:MM:SS format)'),
        sa.Column('days_of_week', postgresql.JSONB(), nullable=True,
                  comment='Days of week (0=Monday to 6=Sunday), null=all days'),
        # Environmental conditions
        sa.Column('weather_conditions', postgresql.JSONB(), nullable=True,
                  comment='Weather conditions when limit applies (null=all conditions)'),
        sa.Column('minimum_visibility', sa.Float(), nullable=True,
                  comment='Minimum visibility in meters for this limit'),
        # Enforcement configuration
        sa.Column('enforcement_enabled', sa.Boolean(), nullable=False, default=True,
                  comment='Whether to enforce this speed limit'),
        sa.Column('warning_threshold', sa.Float(), nullable=True,
                  comment='Speed threshold for warnings (km/h over limit)'),
        sa.Column('violation_threshold', sa.Float(), nullable=False, default=10.0,
                  comment='Speed threshold for violations (km/h over limit)'),
        # Priority and validity
        sa.Column('priority', sa.Integer(), nullable=False, default=100,
                  comment='Priority when multiple limits apply (lower=higher priority)'),
        sa.Column('valid_from', sa.DateTime(timezone=True), nullable=False,
                  comment='Speed limit validity start date'),
        sa.Column('valid_until', sa.DateTime(timezone=True), nullable=True,
                  comment='Speed limit validity end date (null=permanent)'),
        # Geographic boundaries (optional)
        sa.Column('geographic_bounds', postgresql.JSONB(), nullable=True,
                  comment='Geographic boundaries for the speed limit zone'),
        # Administrative information
        sa.Column('authority', sa.String(100), nullable=True,
                  comment='Authority that set this speed limit'),
        sa.Column('regulation_reference', sa.String(200), nullable=True,
                  comment='Legal regulation reference'),
        sa.Column('last_updated_by', sa.String(100), nullable=True,
                  comment='User who last updated this limit'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        comment='Dynamic speed limits by zone and vehicle type with time-based restrictions'
    )

    # SpeedLimit indexes
    op.create_index('idx_speed_limit_zone_vehicle', 'speed_limits', ['zone_id', 'vehicle_type'])
    op.create_index('idx_speed_limit_zone_active', 'speed_limits', ['zone_id', 'enforcement_enabled'])
    op.create_index('idx_speed_limit_vehicle_type', 'speed_limits', ['vehicle_type'])
    op.create_index('idx_speed_limit_validity', 'speed_limits', ['valid_from', 'valid_until'])
    op.create_index('idx_speed_limit_priority', 'speed_limits', ['priority'])
    op.create_index('idx_speed_limit_lookup', 'speed_limits', 
                   ['zone_id', 'vehicle_type', 'enforcement_enabled', 'valid_from'])
    op.create_index('idx_speed_limit_active', 'speed_limits', ['zone_id', 'vehicle_type', 'priority'],
                   postgresql_where=sa.text(
                       'enforcement_enabled = true AND (valid_until IS NULL OR valid_until > NOW())'
                   ))

    # AlertNotification table
    op.create_table(
        'alert_notifications',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        # Alert reference
        sa.Column('alert_type', sa.String(50), nullable=False,
                  comment='Type of alert (violation/anomaly/incident)'),
        sa.Column('reference_id', sa.String(100), nullable=False,
                  comment='ID of referenced violation/anomaly'),
        # Notification details
        sa.Column('notification_channel', sa.String(50), nullable=False,
                  comment='Notification channel (email/sms/webhook/push)'),
        sa.Column('recipient', sa.String(200), nullable=False,
                  comment='Notification recipient identifier'),
        sa.Column('priority', sa.String(20), nullable=False, default='medium',
                  comment='Alert priority (low/medium/high/critical)'),
        # Timing
        sa.Column('created_time', sa.DateTime(timezone=True), nullable=False,
                  comment='Alert creation timestamp'),
        sa.Column('sent_time', sa.DateTime(timezone=True), nullable=True,
                  comment='Notification sent timestamp'),
        sa.Column('acknowledged_time', sa.DateTime(timezone=True), nullable=True,
                  comment='Alert acknowledgment timestamp'),
        # Delivery status
        sa.Column('status', sa.String(20), nullable=False, default='pending',
                  comment='Notification status (pending/sent/delivered/failed/acknowledged)'),
        sa.Column('delivery_attempts', sa.Integer(), nullable=False, default=0,
                  comment='Number of delivery attempts'),
        sa.Column('last_attempt_time', sa.DateTime(timezone=True), nullable=True,
                  comment='Last delivery attempt timestamp'),
        # Content and formatting
        sa.Column('subject', sa.String(200), nullable=True,
                  comment='Alert subject/title'),
        sa.Column('message_content', sa.String(2000), nullable=False,
                  comment='Alert message content'),
        sa.Column('message_format', sa.String(20), nullable=False, default='text',
                  comment='Message format (text/html/json)'),
        # Response tracking
        sa.Column('acknowledged_by', sa.String(100), nullable=True,
                  comment='User who acknowledged the alert'),
        sa.Column('response_action', sa.String(100), nullable=True,
                  comment='Response action taken'),
        sa.Column('response_notes', sa.String(500), nullable=True,
                  comment='Response notes'),
        # Delivery metadata
        sa.Column('external_id', sa.String(100), nullable=True,
                  comment='External system notification ID'),
        sa.Column('delivery_details', postgresql.JSONB(), nullable=True,
                  comment='Delivery service response details'),
        sa.Column('error_message', sa.String(500), nullable=True,
                  comment='Error message if delivery failed'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        comment='Alert notification delivery and response tracking'
    )

    # AlertNotification indexes
    op.create_index('idx_alert_created_time', 'alert_notifications', ['created_time'])
    op.create_index('idx_alert_reference', 'alert_notifications', ['alert_type', 'reference_id'])
    op.create_index('idx_alert_recipient', 'alert_notifications', ['recipient'])
    op.create_index('idx_alert_status', 'alert_notifications', ['status'])
    op.create_index('idx_alert_priority', 'alert_notifications', ['priority'])
    op.create_index('idx_alert_acknowledged', 'alert_notifications', ['acknowledged_time'])
    op.create_index('idx_alert_channel', 'alert_notifications', ['notification_channel'])
    op.create_index('idx_alert_attempts', 'alert_notifications', ['delivery_attempts'])
    op.create_index('idx_alert_pending', 'alert_notifications', ['status', 'created_time'],
                   postgresql_where=sa.text("status IN ('pending', 'failed')"))
    op.create_index('idx_alert_recipient_status', 'alert_notifications', ['recipient', 'status'])

    # Create updated_at trigger for new tables
    analytics_tables = [
        'traffic_metrics', 'rule_violations', 'vehicle_trajectories',
        'traffic_anomalies', 'speed_limits', 'alert_notifications'
    ]

    for table in analytics_tables:
        op.execute(f"""
            CREATE TRIGGER trigger_update_{table}_timestamp
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION update_timestamp();
        """)

    # Add relationships for detection_results table to support rule violations
    op.execute("""
        ALTER TABLE detection_results 
        ADD COLUMN IF NOT EXISTS rule_violations_count INTEGER DEFAULT 0;
    """)

    # Create TimescaleDB retention policy for traffic_metrics (optional)
    op.execute("""
        SELECT add_retention_policy('traffic_metrics', INTERVAL '1 year');
    """)


def downgrade() -> None:
    """Drop analytics tables and hypertables."""

    # Drop retention policy first
    op.execute("SELECT remove_retention_policy('traffic_metrics', if_exists => true);")

    # Drop triggers
    analytics_tables = [
        'traffic_metrics', 'rule_violations', 'vehicle_trajectories', 
        'traffic_anomalies', 'speed_limits', 'alert_notifications'
    ]

    for table in analytics_tables:
        op.execute(f"DROP TRIGGER IF EXISTS trigger_update_{table}_timestamp ON {table};")

    # Drop foreign key column
    op.execute("ALTER TABLE detection_results DROP COLUMN IF EXISTS rule_violations_count;")

    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('alert_notifications')
    op.drop_table('speed_limits')
    op.drop_table('traffic_anomalies')
    op.drop_table('vehicle_trajectories')
    op.drop_table('rule_violations')
    
    # Drop hypertable (this automatically drops the table)
    op.execute("DROP TABLE IF EXISTS traffic_metrics CASCADE;")