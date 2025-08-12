"""Initial database schema for ITS Camera AI.

Creates all tables for camera registry, frame metadata, detection results,
system metrics, and user management with optimized indexes for high-throughput operations.

Revision ID: 001
Revises: 
Create Date: 2025-01-12 00:00:00.000000
"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema with all optimizations."""

    # Enable UUID extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm";')  # For trigram indexes

    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('username', sa.String(50), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('is_superuser', sa.Boolean(), nullable=False, default=False),
        sa.Column('roles', postgresql.JSONB(), nullable=False, default=list),
        sa.Column('permissions', postgresql.JSONB(), nullable=False, default=list),
        sa.Column('profile', postgresql.JSONB(), nullable=True),
        sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email'),
        comment='User management and authentication'
    )

    # Users indexes
    op.create_index('idx_users_username', 'users', ['username'])
    op.create_index('idx_users_email', 'users', ['email'])
    op.create_index('idx_users_active', 'users', ['is_active'])
    op.create_index('idx_users_roles_gin', 'users', ['roles'], postgresql_using='gin')

    # Cameras table
    op.create_table(
        'cameras',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('location', sa.String(200), nullable=False),
        sa.Column('coordinates', postgresql.JSONB(), nullable=True),
        sa.Column('camera_type', sa.String(20), nullable=False),
        sa.Column('stream_url', sa.String(500), nullable=False),
        sa.Column('stream_protocol', sa.String(20), nullable=False),
        sa.Column('backup_stream_url', sa.String(500), nullable=True),
        sa.Column('username', sa.String(100), nullable=True),
        sa.Column('password', sa.String(200), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='offline'),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.Column('config', postgresql.JSONB(), nullable=False, default=dict),
        sa.Column('zone_id', sa.String(50), nullable=True),
        sa.Column('tags', postgresql.JSONB(), nullable=False, default=list),
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_frame_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('total_frames_processed', sa.Integer(), nullable=False, default=0),
        sa.Column('uptime_percentage', sa.Float(), nullable=True),
        sa.Column('avg_processing_time', sa.Float(), nullable=True),
        sa.Column('calibration', postgresql.JSONB(), nullable=True),
        sa.Column('detection_zones', postgresql.JSONB(), nullable=False, default=list),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        comment='Camera registry with optimizations for high-throughput queries'
    )

    # Camera indexes for performance
    op.create_index('idx_camera_name', 'cameras', ['name'])
    op.create_index('idx_camera_location', 'cameras', ['location'])
    op.create_index('idx_camera_status_active', 'cameras', ['status', 'is_active'])
    op.create_index('idx_camera_zone_active', 'cameras', ['zone_id', 'is_active'])
    op.create_index('idx_camera_last_seen', 'cameras', ['last_seen_at'])
    op.create_index('idx_camera_location_type', 'cameras', ['location', 'camera_type'])
    op.create_index('idx_camera_tags_gin', 'cameras', ['tags'], postgresql_using='gin')
    op.create_index('idx_camera_config_gin', 'cameras', ['config'], postgresql_using='gin')
    op.create_index('idx_camera_coordinates_gin', 'cameras', ['coordinates'], postgresql_using='gin')

    # Camera Settings table
    op.create_table(
        'camera_settings',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('camera_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('detection_enabled', sa.Boolean(), nullable=False, default=True),
        sa.Column('tracking_enabled', sa.Boolean(), nullable=False, default=True),
        sa.Column('analytics_enabled', sa.Boolean(), nullable=False, default=True),
        sa.Column('model_name', sa.String(100), nullable=False, default='yolo11n'),
        sa.Column('confidence_threshold', sa.Float(), nullable=False, default=0.5),
        sa.Column('nms_threshold', sa.Float(), nullable=False, default=0.4),
        sa.Column('max_batch_size', sa.Integer(), nullable=False, default=8),
        sa.Column('frame_skip', sa.Integer(), nullable=False, default=0),
        sa.Column('resize_resolution', postgresql.JSONB(), nullable=True),
        sa.Column('quality_threshold', sa.Float(), nullable=False, default=0.7),
        sa.Column('max_processing_time', sa.Integer(), nullable=False, default=100),
        sa.Column('record_detections_only', sa.Boolean(), nullable=False, default=False),
        sa.Column('storage_retention_days', sa.Integer(), nullable=False, default=7),
        sa.Column('alert_thresholds', postgresql.JSONB(), nullable=False, default=dict),
        sa.Column('notification_settings', postgresql.JSONB(), nullable=False, default=dict),
        sa.Column('advanced_settings', postgresql.JSONB(), nullable=False, default=dict),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='CASCADE'),
        comment='Camera-specific processing and configuration settings'
    )

    # Camera settings indexes
    op.create_index('idx_camera_settings_camera_id', 'camera_settings', ['camera_id'])
    op.create_index('idx_camera_settings_model', 'camera_settings', ['model_name'])
    op.create_index('idx_camera_settings_detection', 'camera_settings', ['detection_enabled'])

    # Frame Metadata table (optimized for high throughput)
    op.create_table(
        'frame_metadata',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('camera_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('frame_number', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('processing_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('processing_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('quality_rating', sa.String(20), nullable=True),
        sa.Column('width', sa.Integer(), nullable=False),
        sa.Column('height', sa.Integer(), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('format', sa.String(20), nullable=False, default='jpeg'),
        sa.Column('storage_path', sa.String(500), nullable=True),
        sa.Column('storage_bucket', sa.String(100), nullable=True),
        sa.Column('is_stored', sa.Boolean(), nullable=False, default=False),
        sa.Column('has_detections', sa.Boolean(), nullable=False, default=False),
        sa.Column('detection_count', sa.Integer(), nullable=False, default=0),
        sa.Column('vehicle_count', sa.Integer(), nullable=False, default=0),
        sa.Column('traffic_density', sa.Float(), nullable=True),
        sa.Column('congestion_level', sa.String(20), nullable=True),
        sa.Column('weather_conditions', postgresql.JSONB(), nullable=True),
        sa.Column('lighting_conditions', sa.String(20), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, default=0),
        sa.Column('model_version', sa.String(50), nullable=True),
        sa.Column('processing_config', postgresql.JSONB(), nullable=True),
        sa.Column('inference_time_ms', sa.Float(), nullable=True),
        sa.Column('preprocessing_time_ms', sa.Float(), nullable=True),
        sa.Column('postprocessing_time_ms', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['camera_id'], ['cameras.id'], ondelete='CASCADE'),
        comment='Frame metadata optimized for high-throughput processing'
    )

    # Frame metadata indexes (critical for performance)
    op.create_index('idx_frame_camera_timestamp', 'frame_metadata', ['camera_id', 'timestamp'])
    op.create_index('idx_frame_camera_frame_num', 'frame_metadata', ['camera_id', 'frame_number'])
    op.create_index('idx_frame_status_timestamp', 'frame_metadata', ['status', 'timestamp'])
    op.create_index('idx_frame_detections_timestamp', 'frame_metadata', ['has_detections', 'timestamp'])
    op.create_index('idx_frame_processing_time', 'frame_metadata', ['processing_time_ms'])
    op.create_index('idx_frame_quality_score', 'frame_metadata', ['quality_score'])
    op.create_index('idx_frame_traffic_density', 'frame_metadata', ['traffic_density'])
    op.create_index('idx_frame_vehicle_count', 'frame_metadata', ['vehicle_count'])
    op.create_index('idx_frame_storage_status', 'frame_metadata', ['is_stored', 'timestamp'])
    op.create_index('idx_frame_storage_path', 'frame_metadata', ['storage_path'])
    op.create_index('idx_frame_error_retry', 'frame_metadata', ['status', 'retry_count'])
    op.create_index('idx_frame_camera_quality', 'frame_metadata', ['camera_id', 'quality_score', 'timestamp'])
    op.create_index('idx_frame_camera_detections', 'frame_metadata', ['camera_id', 'has_detections', 'timestamp'])

    # Partial indexes for specific queries
    op.create_index(
        'idx_frame_failed_status', 'frame_metadata', ['camera_id', 'timestamp'],
        postgresql_where=sa.text("status = 'failed'")
    )
    op.create_index(
        'idx_frame_with_detections', 'frame_metadata', ['camera_id', 'timestamp', 'vehicle_count'],
        postgresql_where=sa.text("has_detections = true")
    )

    # Detection Results table
    op.create_table(
        'detection_results',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('frame_metadata_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('detection_id', sa.Integer(), nullable=False),
        sa.Column('bbox_x1', sa.Float(), nullable=False),
        sa.Column('bbox_y1', sa.Float(), nullable=False),
        sa.Column('bbox_x2', sa.Float(), nullable=False),
        sa.Column('bbox_y2', sa.Float(), nullable=False),
        sa.Column('bbox_width', sa.Float(), nullable=False),
        sa.Column('bbox_height', sa.Float(), nullable=False),
        sa.Column('bbox_area', sa.Float(), nullable=False),
        sa.Column('class_name', sa.String(50), nullable=False),
        sa.Column('class_confidence', sa.Float(), nullable=False),
        sa.Column('vehicle_type', sa.String(50), nullable=True),
        sa.Column('vehicle_confidence', sa.Float(), nullable=True),
        sa.Column('track_id', sa.Integer(), nullable=True),
        sa.Column('track_confidence', sa.Float(), nullable=True),
        sa.Column('velocity_x', sa.Float(), nullable=True),
        sa.Column('velocity_y', sa.Float(), nullable=True),
        sa.Column('velocity_magnitude', sa.Float(), nullable=True),
        sa.Column('direction', sa.Float(), nullable=True),
        sa.Column('color_primary', sa.String(30), nullable=True),
        sa.Column('color_secondary', sa.String(30), nullable=True),
        sa.Column('color_confidence', sa.Float(), nullable=True),
        sa.Column('license_plate', sa.String(20), nullable=True),
        sa.Column('license_plate_confidence', sa.Float(), nullable=True),
        sa.Column('license_plate_region', postgresql.JSONB(), nullable=True),
        sa.Column('estimated_length', sa.Float(), nullable=True),
        sa.Column('estimated_width', sa.Float(), nullable=True),
        sa.Column('estimated_height', sa.Float(), nullable=True),
        sa.Column('detection_zone', sa.String(50), nullable=True),
        sa.Column('zone_entry_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('zone_exit_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('detection_quality', sa.Float(), nullable=False, default=1.0),
        sa.Column('occlusion_ratio', sa.Float(), nullable=True),
        sa.Column('blur_score', sa.Float(), nullable=True),
        sa.Column('model_version', sa.String(50), nullable=True),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('additional_attributes', postgresql.JSONB(), nullable=True),
        sa.Column('is_false_positive', sa.Boolean(), nullable=False, default=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False, default=False),
        sa.Column('is_anomaly', sa.Boolean(), nullable=False, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['frame_metadata_id'], ['frame_metadata.id'], ondelete='CASCADE'),
        comment='Detection results with spatial and classification indexes'
    )

    # Detection result indexes
    op.create_index('idx_detection_frame_id', 'detection_results', ['frame_metadata_id'])
    op.create_index('idx_detection_class', 'detection_results', ['class_name'])
    op.create_index('idx_detection_track_id', 'detection_results', ['track_id'])
    op.create_index('idx_detection_confidence', 'detection_results', ['class_confidence'])
    op.create_index('idx_detection_quality', 'detection_results', ['detection_quality'])
    op.create_index('idx_detection_bbox_center', 'detection_results', ['bbox_x1', 'bbox_y1'])
    op.create_index('idx_detection_bbox_area', 'detection_results', ['bbox_area'])
    op.create_index('idx_detection_velocity', 'detection_results', ['velocity_magnitude'])
    op.create_index('idx_detection_direction', 'detection_results', ['direction'])
    op.create_index('idx_detection_vehicle_type', 'detection_results', ['vehicle_type'])
    op.create_index('idx_detection_color', 'detection_results', ['color_primary'])
    op.create_index('idx_detection_license_plate', 'detection_results', ['license_plate'])
    op.create_index('idx_detection_zone', 'detection_results', ['detection_zone'])
    op.create_index('idx_detection_zone_entry', 'detection_results', ['detection_zone', 'zone_entry_time'])
    op.create_index('idx_detection_flags', 'detection_results', ['is_false_positive', 'is_verified'])
    op.create_index('idx_detection_class_confidence', 'detection_results', ['class_name', 'class_confidence'])
    op.create_index('idx_detection_track_time', 'detection_results', ['track_id', 'created_at'])

    # High-performance vehicle detection queries
    op.create_index(
        'idx_detection_vehicles', 'detection_results',
        ['frame_metadata_id', 'class_name', 'class_confidence'],
        postgresql_where=sa.text(
            "class_name IN ('car', 'truck', 'bus', 'motorcycle', 'bicycle') "
            "AND class_confidence >= 0.5"
        )
    )

    # License plate detections
    op.create_index(
        'idx_detection_with_plates', 'detection_results',
        ['frame_metadata_id', 'license_plate', 'license_plate_confidence'],
        postgresql_where=sa.text("license_plate IS NOT NULL")
    )

    # System Metrics table
    op.create_table(
        'system_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_type', sa.String(50), nullable=False),
        sa.Column('metric_unit', sa.String(20), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('source_type', sa.String(50), nullable=False),
        sa.Column('source_id', sa.String(100), nullable=True),
        sa.Column('hostname', sa.String(100), nullable=True),
        sa.Column('service_name', sa.String(100), nullable=True),
        sa.Column('aggregation_period', sa.String(20), nullable=True),
        sa.Column('sample_count', sa.Integer(), nullable=True),
        sa.Column('min_value', sa.Float(), nullable=True),
        sa.Column('max_value', sa.Float(), nullable=True),
        sa.Column('avg_value', sa.Float(), nullable=True),
        sa.Column('sum_value', sa.Float(), nullable=True),
        sa.Column('labels', postgresql.JSONB(), nullable=True),
        sa.Column('context', postgresql.JSONB(), nullable=True),
        sa.Column('warning_threshold', sa.Float(), nullable=True),
        sa.Column('critical_threshold', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        comment='System metrics optimized for time-series monitoring'
    )

    # System metrics indexes (time-series optimized)
    op.create_index('idx_metrics_name_timestamp', 'system_metrics', ['metric_name', 'timestamp'])
    op.create_index('idx_metrics_type_timestamp', 'system_metrics', ['metric_type', 'timestamp'])
    op.create_index('idx_metrics_source_timestamp', 'system_metrics', ['source_type', 'source_id', 'timestamp'])
    op.create_index('idx_metrics_hostname_timestamp', 'system_metrics', ['hostname', 'timestamp'])
    op.create_index('idx_metrics_service_timestamp', 'system_metrics', ['service_name', 'timestamp'])
    op.create_index('idx_metrics_aggregation', 'system_metrics', ['metric_name', 'aggregation_period', 'timestamp'])
    op.create_index('idx_metrics_value_thresholds', 'system_metrics', ['metric_name', 'value', 'timestamp'])
    op.create_index('idx_metrics_labels_gin', 'system_metrics', ['labels'], postgresql_using='gin')

    # Performance monitoring specific indexes
    op.create_index(
        'idx_metrics_performance', 'system_metrics',
        ['metric_type', 'source_id', 'timestamp'],
        postgresql_where=sa.text(
            "metric_type IN ('processing_time', 'throughput', 'latency', 'frame_rate')"
        )
    )

    # Resource utilization queries
    op.create_index(
        'idx_metrics_resources', 'system_metrics',
        ['metric_type', 'hostname', 'timestamp'],
        postgresql_where=sa.text(
            "metric_type IN ('cpu_usage', 'memory_usage', 'gpu_usage', 'disk_usage')"
        )
    )

    # Recent metrics (last 24 hours)
    op.create_index(
        'idx_metrics_recent', 'system_metrics',
        ['metric_name', 'timestamp', 'value'],
        postgresql_where=sa.text("timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'")
    )

    # Aggregated Metrics table
    op.create_table(
        'aggregated_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('metric_name', sa.String(100), nullable=False),
        sa.Column('metric_type', sa.String(50), nullable=False),
        sa.Column('aggregation_period', sa.String(20), nullable=False),
        sa.Column('period_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('period_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('source_type', sa.String(50), nullable=False),
        sa.Column('source_id', sa.String(100), nullable=True),
        sa.Column('sample_count', sa.Integer(), nullable=False),
        sa.Column('min_value', sa.Float(), nullable=False),
        sa.Column('max_value', sa.Float(), nullable=False),
        sa.Column('avg_value', sa.Float(), nullable=False),
        sa.Column('sum_value', sa.Float(), nullable=False),
        sa.Column('std_deviation', sa.Float(), nullable=True),
        sa.Column('p50_value', sa.Float(), nullable=True),
        sa.Column('p95_value', sa.Float(), nullable=True),
        sa.Column('p99_value', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        comment='Pre-computed metric aggregations for dashboard queries'
    )

    # Aggregated metrics indexes
    op.create_index('idx_agg_metrics_name_period', 'aggregated_metrics',
                   ['metric_name', 'aggregation_period', 'period_start'])
    op.create_index('idx_agg_metrics_source_period', 'aggregated_metrics',
                   ['source_type', 'source_id', 'period_start'])
    op.create_index('idx_agg_metrics_period_range', 'aggregated_metrics',
                   ['aggregation_period', 'period_start', 'period_end'])

    # Create database functions for better performance
    op.execute("""
        CREATE OR REPLACE FUNCTION update_timestamp()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    # Create triggers for automatic timestamp updates
    tables_with_updated_at = [
        'users', 'cameras', 'camera_settings', 'frame_metadata',
        'detection_results', 'system_metrics', 'aggregated_metrics'
    ]

    for table in tables_with_updated_at:
        op.execute(f"""
            CREATE TRIGGER trigger_update_{table}_timestamp
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION update_timestamp();
        """)


def downgrade() -> None:
    """Drop all tables and extensions."""

    # Drop triggers first
    tables_with_updated_at = [
        'users', 'cameras', 'camera_settings', 'frame_metadata',
        'detection_results', 'system_metrics', 'aggregated_metrics'
    ]

    for table in tables_with_updated_at:
        op.execute(f"DROP TRIGGER IF EXISTS trigger_update_{table}_timestamp ON {table};")

    # Drop function
    op.execute("DROP FUNCTION IF EXISTS update_timestamp();")

    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('aggregated_metrics')
    op.drop_table('system_metrics')
    op.drop_table('detection_results')
    op.drop_table('frame_metadata')
    op.drop_table('camera_settings')
    op.drop_table('cameras')
    op.drop_table('users')

    # Drop extensions
    op.execute('DROP EXTENSION IF EXISTS "pg_trgm";')
    op.execute('DROP EXTENSION IF EXISTS "uuid-ossp";')
