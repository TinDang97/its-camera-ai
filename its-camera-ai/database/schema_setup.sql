-- Optimized PostgreSQL schema setup for ITS Camera AI
-- High-throughput design for 3000+ frame inserts/sec and sub-10ms queries

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Performance optimization settings
SET shared_preload_libraries = 'pg_stat_statements';
SET max_connections = 200;
SET shared_buffers = '2GB';
SET effective_cache_size = '6GB';
SET work_mem = '32MB';
SET maintenance_work_mem = '512MB';
SET checkpoint_completion_target = 0.9;
SET wal_buffers = '64MB';
SET default_statistics_target = 500;
SET random_page_cost = 1.1;
SET effective_io_concurrency = 200;

-- Connection pooling optimizations
SET max_prepared_transactions = 100;
SET max_locks_per_transaction = 256;

-- Create base tables with optimized structure
CREATE TABLE IF NOT EXISTS cameras (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description VARCHAR(500),
    location VARCHAR(200) NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    camera_type VARCHAR(20) NOT NULL,
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    firmware_version VARCHAR(50),
    stream_url VARCHAR(500) NOT NULL,
    stream_protocol VARCHAR(20) NOT NULL,
    backup_stream_url VARCHAR(500),
    username VARCHAR(100),
    password VARCHAR(100),
    status VARCHAR(20) NOT NULL DEFAULT 'offline',
    is_active BOOLEAN DEFAULT TRUE,
    last_seen_at TIMESTAMPTZ,
    config JSONB NOT NULL DEFAULT '{}',
    zone_id VARCHAR(100),
    tags TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create frame_metadata table with partitioning support
CREATE TABLE IF NOT EXISTS frame_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence_id BIGSERIAL UNIQUE,
    camera_id UUID NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
    frame_number BIGINT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processing_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    processing_time_ms DOUBLE PRECISION,
    model_version VARCHAR(50),
    frame_size_bytes INTEGER NOT NULL,
    quality_score DOUBLE PRECISION,
    blur_score DOUBLE PRECISION,
    brightness_score DOUBLE PRECISION,
    total_detections INTEGER DEFAULT 0,
    vehicle_count INTEGER DEFAULT 0,
    person_count INTEGER DEFAULT 0,
    confidence_avg DOUBLE PRECISION,
    detection_results JSONB,
    processing_metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (timestamp);

-- Create detections table
CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    frame_metadata_id UUID NOT NULL REFERENCES frame_metadata(id) ON DELETE CASCADE,
    object_type VARCHAR(20) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    class_id INTEGER NOT NULL,
    bbox_x1 DOUBLE PRECISION NOT NULL,
    bbox_y1 DOUBLE PRECISION NOT NULL,
    bbox_x2 DOUBLE PRECISION NOT NULL,
    bbox_y2 DOUBLE PRECISION NOT NULL,
    track_id INTEGER,
    speed_kmh DOUBLE PRECISION,
    direction DOUBLE PRECISION,
    zone_name VARCHAR(100),
    is_in_roi BOOLEAN DEFAULT TRUE,
    attributes JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create system_metrics table with partitioning
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id UUID REFERENCES cameras(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(20),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    labels JSONB,
    source VARCHAR(100) NOT NULL DEFAULT 'system',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
) PARTITION BY RANGE (timestamp);

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    role VARCHAR(50) NOT NULL DEFAULT 'viewer',
    permissions TEXT[],
    last_login_at TIMESTAMPTZ,
    login_count INTEGER DEFAULT 0,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create user_sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    refresh_token VARCHAR(255) UNIQUE,
    ip_address VARCHAR(45),
    user_agent TEXT,
    expires_at TIMESTAMPTZ NOT NULL,
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    revoked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id UUID REFERENCES cameras(id) ON DELETE CASCADE,
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    priority INTEGER NOT NULL DEFAULT 5,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'active',
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create optimized indexes for cameras table
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_name ON cameras (name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_location ON cameras (location);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_status ON cameras (status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_is_active ON cameras (is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_last_seen_at ON cameras (last_seen_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_zone_id ON cameras (zone_id);

-- Composite indexes for cameras
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_active_location ON cameras (is_active, location);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_status_last_seen ON cameras (status, last_seen_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_coordinates ON cameras (latitude, longitude);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_zone_active ON cameras (zone_id, is_active);

-- GIN indexes for JSONB and arrays
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_config_gin ON cameras USING GIN (config);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_tags_gin ON cameras USING GIN (tags);

-- Spatial index for coordinates (requires PostGIS extension)
-- CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_location_gist 
--     ON cameras USING GIST (ST_Point(longitude, latitude));

-- Indexes for frame_metadata table
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_frame_metadata_camera_id ON frame_metadata (camera_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_frame_metadata_timestamp ON frame_metadata (timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_frame_metadata_sequence_id ON frame_metadata (sequence_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_frame_metadata_processing_status ON frame_metadata (processing_status);

-- Composite indexes for frame_metadata
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_frame_metadata_camera_timestamp 
    ON frame_metadata (camera_id, timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_frame_metadata_status_timestamp 
    ON frame_metadata (processing_status, timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_frame_metadata_detections 
    ON frame_metadata (camera_id, total_detections);

-- GIN index for detection results
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_frame_metadata_results_gin 
    ON frame_metadata USING GIN (detection_results);

-- Partial index for failed frames (optimization for monitoring)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_frame_metadata_failed 
    ON frame_metadata (camera_id, timestamp) 
    WHERE processing_status = 'failed';

-- Indexes for detections table
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_frame_metadata_id ON detections (frame_metadata_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_object_type ON detections (object_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_confidence ON detections (confidence);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_track_id ON detections (track_id);

-- Composite indexes for detections
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_type_confidence 
    ON detections (object_type, confidence);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_track_frame 
    ON detections (track_id, frame_metadata_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_bbox 
    ON detections (bbox_x1, bbox_y1, bbox_x2, bbox_y2);

-- GIN index for detection attributes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_attributes_gin 
    ON detections USING GIN (attributes);

-- Indexes for system_metrics table
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_camera_id ON system_metrics (camera_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_metric_name ON system_metrics (metric_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics (timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_metric_type ON system_metrics (metric_type);

-- Composite indexes for system_metrics
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_time 
    ON system_metrics (timestamp, metric_name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_camera 
    ON system_metrics (camera_id, metric_name, timestamp);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_type_time 
    ON system_metrics (metric_type, timestamp);

-- GIN index for metrics labels
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_labels_gin 
    ON system_metrics USING GIN (labels);

-- Partial index for high-value metrics (alerts)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_alerts 
    ON system_metrics (metric_name, value, timestamp) 
    WHERE value > 0.8;

-- Indexes for users table
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_username ON users (username);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users (email);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_is_active ON users (is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_role ON users (role);

-- Composite indexes for users
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_username_active ON users (username, is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email_active ON users (email, is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_role_active ON users (role, is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_failed_logins 
    ON users (failed_login_attempts, locked_until);

-- Indexes for user_sessions table
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_user_id ON user_sessions (user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_session_token ON user_sessions (session_token);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions (expires_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_is_active ON user_sessions (is_active);

-- Composite indexes for user_sessions
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_token_active 
    ON user_sessions (session_token, is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_user_active 
    ON user_sessions (user_id, is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_expires_active 
    ON user_sessions (expires_at, is_active);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_ip_created 
    ON user_sessions (ip_address, created_at);

-- Indexes for alerts table
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_camera_id ON alerts (camera_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_alert_type ON alerts (alert_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_severity ON alerts (severity);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_priority ON alerts (priority);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_status ON alerts (status);

-- Composite indexes for alerts
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_status_priority 
    ON alerts (status, priority, created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_camera_status 
    ON alerts (camera_id, status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_type_severity 
    ON alerts (alert_type, severity, created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_resolved 
    ON alerts (resolved_at, status);

-- GIN index for alert metadata
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_metadata_gin 
    ON alerts USING GIN (metadata);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at triggers to all tables
CREATE TRIGGER update_cameras_updated_at BEFORE UPDATE ON cameras
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_frame_metadata_updated_at BEFORE UPDATE ON frame_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_detections_updated_at BEFORE UPDATE ON detections
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_system_metrics_updated_at BEFORE UPDATE ON system_metrics
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_sessions_updated_at BEFORE UPDATE ON user_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_alerts_updated_at BEFORE UPDATE ON alerts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function for automatic session cleanup
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions 
    WHERE expires_at < CURRENT_TIMESTAMP 
       OR (revoked_at IS NOT NULL AND revoked_at < CURRENT_TIMESTAMP - INTERVAL '7 days');
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Performance monitoring views
CREATE OR REPLACE VIEW camera_performance_summary AS
SELECT 
    c.id,
    c.name,
    c.location,
    c.status,
    c.is_active,
    COUNT(fm.id) as total_frames,
    AVG(fm.processing_time_ms) as avg_processing_time,
    AVG(fm.total_detections) as avg_detections_per_frame,
    MAX(fm.timestamp) as last_frame_time,
    DATE_TRUNC('hour', MAX(fm.timestamp)) as last_activity_hour
FROM cameras c
LEFT JOIN frame_metadata fm ON c.id = fm.camera_id
    AND fm.timestamp >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY c.id, c.name, c.location, c.status, c.is_active;

-- Real-time performance view for SSE
CREATE OR REPLACE VIEW realtime_camera_status AS
SELECT 
    c.id,
    c.name,
    c.status,
    c.last_seen_at,
    CASE 
        WHEN c.last_seen_at > CURRENT_TIMESTAMP - INTERVAL '30 seconds' THEN 'streaming'
        WHEN c.last_seen_at > CURRENT_TIMESTAMP - INTERVAL '5 minutes' THEN 'recent'
        ELSE 'stale'
    END as connection_status,
    latest_metrics.fps_current,
    latest_metrics.detection_rate
FROM cameras c
LEFT JOIN LATERAL (
    SELECT 
        AVG(CASE WHEN metric_name = 'fps' THEN value END) as fps_current,
        AVG(CASE WHEN metric_name = 'detection_rate' THEN value END) as detection_rate
    FROM system_metrics sm
    WHERE sm.camera_id = c.id 
      AND sm.timestamp >= CURRENT_TIMESTAMP - INTERVAL '1 minute'
) latest_metrics ON true
WHERE c.is_active = true;

COMMENT ON TABLE cameras IS 'Camera registry optimized for CRUD operations with sub-10ms queries';
COMMENT ON TABLE frame_metadata IS 'High-throughput frame processing metadata with time-series partitioning';
COMMENT ON TABLE detections IS 'Individual detection results with spatial indexing for analytics';
COMMENT ON TABLE system_metrics IS 'Performance monitoring with time-series storage and alerting';
COMMENT ON TABLE users IS 'User management with role-based access control';
COMMENT ON TABLE user_sessions IS 'Session tracking for security and monitoring';
COMMENT ON TABLE alerts IS 'Real-time alerting system with priority-based indexing';