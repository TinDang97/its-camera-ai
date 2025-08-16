-- TimescaleDB initialization script for ITS Camera AI
-- This script sets up the time-series database schema for metrics and analytics

-- Create the TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schema for metrics
CREATE SCHEMA IF NOT EXISTS metrics;

-- Camera metrics table
CREATE TABLE IF NOT EXISTS metrics.camera_metrics (
    time TIMESTAMPTZ NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    fps NUMERIC(5,2),
    latency_ms NUMERIC(7,2),
    frames_processed BIGINT,
    frames_dropped BIGINT,
    cpu_usage NUMERIC(5,2),
    memory_usage NUMERIC(5,2),
    gpu_usage NUMERIC(5,2),
    gpu_memory_mb INTEGER,
    network_bandwidth_mbps NUMERIC(10,2),
    error_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active'
);

-- Convert to hypertable
SELECT create_hypertable('metrics.camera_metrics', 'time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Vehicle detection metrics
CREATE TABLE IF NOT EXISTS metrics.vehicle_detections (
    time TIMESTAMPTZ NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    detection_id UUID DEFAULT gen_random_uuid(),
    vehicle_type VARCHAR(50),
    confidence NUMERIC(3,2),
    speed_kmh NUMERIC(6,2),
    direction VARCHAR(20),
    lane_number INTEGER,
    bounding_box JSONB,
    metadata JSONB
);

SELECT create_hypertable('metrics.vehicle_detections', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Traffic analytics aggregates
CREATE TABLE IF NOT EXISTS metrics.traffic_analytics (
    time TIMESTAMPTZ NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    interval_minutes INTEGER DEFAULT 5,
    vehicle_count INTEGER,
    avg_speed_kmh NUMERIC(6,2),
    max_speed_kmh NUMERIC(6,2),
    min_speed_kmh NUMERIC(6,2),
    occupancy_rate NUMERIC(5,2),
    flow_rate INTEGER,
    congestion_level VARCHAR(20),
    incident_count INTEGER DEFAULT 0
);

SELECT create_hypertable('metrics.traffic_analytics', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Model inference metrics
CREATE TABLE IF NOT EXISTS metrics.model_inference (
    time TIMESTAMPTZ NOT NULL,
    model_id VARCHAR(100) NOT NULL,
    model_version VARCHAR(20),
    inference_time_ms NUMERIC(8,2),
    preprocessing_time_ms NUMERIC(8,2),
    postprocessing_time_ms NUMERIC(8,2),
    batch_size INTEGER,
    input_shape VARCHAR(50),
    accuracy NUMERIC(5,4),
    objects_detected INTEGER,
    confidence_threshold NUMERIC(3,2),
    device VARCHAR(20),
    memory_used_mb INTEGER
);

SELECT create_hypertable('metrics.model_inference', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- System performance metrics
CREATE TABLE IF NOT EXISTS metrics.system_performance (
    time TIMESTAMPTZ NOT NULL,
    node_id VARCHAR(50) NOT NULL,
    service_name VARCHAR(50),
    cpu_cores INTEGER,
    cpu_usage_percent NUMERIC(5,2),
    memory_total_gb NUMERIC(10,2),
    memory_used_gb NUMERIC(10,2),
    memory_percent NUMERIC(5,2),
    disk_used_gb NUMERIC(10,2),
    disk_percent NUMERIC(5,2),
    network_in_mbps NUMERIC(10,2),
    network_out_mbps NUMERIC(10,2),
    gpu_count INTEGER,
    gpu_memory_total_gb NUMERIC(10,2),
    gpu_memory_used_gb NUMERIC(10,2),
    temperature_celsius NUMERIC(5,2)
);

SELECT create_hypertable('metrics.system_performance', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Alert and incident history
CREATE TABLE IF NOT EXISTS metrics.incidents (
    time TIMESTAMPTZ NOT NULL,
    incident_id UUID DEFAULT gen_random_uuid(),
    camera_id VARCHAR(50),
    incident_type VARCHAR(100),
    severity VARCHAR(20),
    description TEXT,
    location JSONB,
    detected_objects JSONB,
    response_time_seconds INTEGER,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_time TIMESTAMPTZ,
    metadata JSONB
);

SELECT create_hypertable('metrics.incidents', 'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_camera_metrics_camera_time 
    ON metrics.camera_metrics (camera_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_vehicle_detections_camera_time 
    ON metrics.vehicle_detections (camera_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_vehicle_detections_type 
    ON metrics.vehicle_detections (vehicle_type, time DESC);

CREATE INDEX IF NOT EXISTS idx_traffic_analytics_camera_time 
    ON metrics.traffic_analytics (camera_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_model_inference_model_time 
    ON metrics.model_inference (model_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_system_performance_node_time 
    ON metrics.system_performance (node_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_incidents_camera_time 
    ON metrics.incidents (camera_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_incidents_type_severity 
    ON metrics.incidents (incident_type, severity, time DESC);

-- Create continuous aggregates for real-time analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.hourly_camera_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    camera_id,
    AVG(fps) AS avg_fps,
    AVG(latency_ms) AS avg_latency_ms,
    SUM(frames_processed) AS total_frames_processed,
    SUM(frames_dropped) AS total_frames_dropped,
    AVG(cpu_usage) AS avg_cpu_usage,
    AVG(memory_usage) AS avg_memory_usage,
    AVG(gpu_usage) AS avg_gpu_usage,
    COUNT(*) AS sample_count
FROM metrics.camera_metrics
GROUP BY bucket, camera_id
WITH NO DATA;

-- Refresh policy for continuous aggregates
SELECT add_continuous_aggregate_policy('metrics.hourly_camera_stats',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.daily_traffic_summary
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    camera_id,
    SUM(vehicle_count) AS total_vehicles,
    AVG(avg_speed_kmh) AS avg_speed,
    MAX(max_speed_kmh) AS max_speed,
    AVG(occupancy_rate) AS avg_occupancy,
    SUM(incident_count) AS total_incidents
FROM metrics.traffic_analytics
GROUP BY bucket, camera_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('metrics.daily_traffic_summary',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Data retention policies
SELECT add_retention_policy('metrics.camera_metrics', 
    INTERVAL '30 days',
    if_not_exists => TRUE
);

SELECT add_retention_policy('metrics.vehicle_detections', 
    INTERVAL '7 days',
    if_not_exists => TRUE
);

SELECT add_retention_policy('metrics.traffic_analytics', 
    INTERVAL '90 days',
    if_not_exists => TRUE
);

SELECT add_retention_policy('metrics.model_inference', 
    INTERVAL '14 days',
    if_not_exists => TRUE
);

SELECT add_retention_policy('metrics.system_performance', 
    INTERVAL '30 days',
    if_not_exists => TRUE
);

SELECT add_retention_policy('metrics.incidents', 
    INTERVAL '365 days',
    if_not_exists => TRUE
);

-- Compression policies for older data
SELECT add_compression_policy('metrics.camera_metrics', 
    INTERVAL '7 days',
    if_not_exists => TRUE
);

SELECT add_compression_policy('metrics.vehicle_detections', 
    INTERVAL '1 day',
    if_not_exists => TRUE
);

SELECT add_compression_policy('metrics.traffic_analytics', 
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Create user for application
CREATE USER IF NOT EXISTS its_metrics_user WITH PASSWORD 'its_metrics_password';
GRANT USAGE ON SCHEMA metrics TO its_metrics_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metrics TO its_metrics_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metrics TO its_metrics_user;

-- Grant permissions for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA metrics 
    GRANT ALL PRIVILEGES ON TABLES TO its_metrics_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA metrics 
    GRANT ALL PRIVILEGES ON SEQUENCES TO its_metrics_user;

-- Performance optimization settings
ALTER SYSTEM SET shared_preload_libraries = 'timescaledb';
ALTER SYSTEM SET timescaledb.max_background_workers = 8;
ALTER SYSTEM SET max_worker_processes = 32;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 16;

-- Create helper functions for common queries
CREATE OR REPLACE FUNCTION metrics.get_camera_stats(
    p_camera_id VARCHAR(50),
    p_start_time TIMESTAMPTZ,
    p_end_time TIMESTAMPTZ
)
RETURNS TABLE (
    avg_fps NUMERIC,
    avg_latency NUMERIC,
    total_frames BIGINT,
    uptime_percent NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        AVG(fps)::NUMERIC AS avg_fps,
        AVG(latency_ms)::NUMERIC AS avg_latency,
        SUM(frames_processed)::BIGINT AS total_frames,
        (COUNT(CASE WHEN status = 'active' THEN 1 END) * 100.0 / COUNT(*))::NUMERIC AS uptime_percent
    FROM metrics.camera_metrics
    WHERE camera_id = p_camera_id
        AND time >= p_start_time
        AND time <= p_end_time;
END;
$$ LANGUAGE plpgsql;

-- Notify on critical incidents
CREATE OR REPLACE FUNCTION metrics.notify_critical_incident()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.severity = 'critical' THEN
        PERFORM pg_notify('critical_incident', 
            json_build_object(
                'incident_id', NEW.incident_id,
                'camera_id', NEW.camera_id,
                'type', NEW.incident_type,
                'time', NEW.time
            )::text
        );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER critical_incident_notification
    AFTER INSERT ON metrics.incidents
    FOR EACH ROW
    EXECUTE FUNCTION metrics.notify_critical_incident();

-- Initial data for testing (optional)
-- INSERT INTO metrics.camera_metrics (time, camera_id, fps, latency_ms, frames_processed)
-- VALUES (NOW(), 'camera_001', 30.0, 45.5, 1800);

COMMENT ON SCHEMA metrics IS 'Time-series metrics and analytics data for ITS Camera AI system';
COMMENT ON TABLE metrics.camera_metrics IS 'Real-time camera performance metrics';
COMMENT ON TABLE metrics.vehicle_detections IS 'Individual vehicle detection events';
COMMENT ON TABLE metrics.traffic_analytics IS 'Aggregated traffic flow analytics';
COMMENT ON TABLE metrics.model_inference IS 'ML model inference performance metrics';
COMMENT ON TABLE metrics.system_performance IS 'System resource utilization metrics';
COMMENT ON TABLE metrics.incidents IS 'Traffic incidents and alerts history';