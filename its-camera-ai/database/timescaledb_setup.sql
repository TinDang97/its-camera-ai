-- TimescaleDB setup for ITS Camera AI Analytics
-- Configures hypertables, continuous aggregates, retention policies, and compression
-- Run after Alembic migrations to set up time-series optimizations

-- Enable TimescaleDB extension if not already enabled
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_buffercache;

-- Check if traffic_metrics is already a hypertable
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'traffic_metrics'
    ) THEN
        -- Convert traffic_metrics to hypertable with 1-hour chunks
        PERFORM create_hypertable(
            'traffic_metrics', 
            'timestamp',
            chunk_time_interval => INTERVAL '1 hour',
            create_default_indexes => FALSE,
            if_not_exists => TRUE
        );
        
        RAISE NOTICE 'Created hypertable for traffic_metrics with 1-hour chunks';
    ELSE
        RAISE NOTICE 'traffic_metrics is already a hypertable';
    END IF;
END
$$;

-- Configure data retention policy for traffic_metrics (keep data for 1 year)
SELECT add_retention_policy('traffic_metrics', INTERVAL '1 year', if_not_exists => TRUE);

-- Set up compression policy for traffic_metrics (compress data older than 7 days)
ALTER TABLE traffic_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'camera_id,zone_id,aggregation_period',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('traffic_metrics', INTERVAL '7 days', if_not_exists => TRUE);

-- Create continuous aggregates for real-time dashboard queries
-- 5-minute aggregates for recent data
CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_metrics_5min
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 minutes', timestamp) AS bucket,
    camera_id,
    zone_id,
    aggregation_period,
    AVG(total_vehicles) AS avg_total_vehicles,
    MAX(total_vehicles) AS max_total_vehicles,
    AVG(average_speed) AS avg_speed,
    MAX(max_speed) AS max_speed,
    AVG(traffic_density) AS avg_traffic_density,
    MAX(traffic_density) AS max_traffic_density,
    AVG(occupancy_rate) AS avg_occupancy_rate,
    MAX(occupancy_rate) AS max_occupancy_rate,
    MODE() WITHIN GROUP (ORDER BY congestion_level) AS dominant_congestion_level,
    COUNT(*) AS sample_count
FROM traffic_metrics
GROUP BY bucket, camera_id, zone_id, aggregation_period;

-- Add refresh policy for 5-minute aggregates
SELECT add_continuous_aggregate_policy('traffic_metrics_5min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE
);

-- 1-hour aggregates for historical analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', timestamp) AS bucket,
    camera_id,
    zone_id,
    AVG(total_vehicles) AS avg_total_vehicles,
    MAX(total_vehicles) AS max_total_vehicles,
    MIN(total_vehicles) AS min_total_vehicles,
    STDDEV(total_vehicles) AS stddev_total_vehicles,
    AVG(average_speed) AS avg_speed,
    MAX(max_speed) AS max_speed,
    MIN(min_speed) AS min_speed,
    STDDEV(average_speed) AS stddev_speed,
    AVG(traffic_density) AS avg_traffic_density,
    MAX(traffic_density) AS max_traffic_density,
    AVG(occupancy_rate) AS avg_occupancy_rate,
    AVG(flow_rate) AS avg_flow_rate,
    MODE() WITHIN GROUP (ORDER BY congestion_level) AS dominant_congestion_level,
    COUNT(*) AS sample_count,
    -- Direction-based aggregates
    AVG(northbound_count + southbound_count) AS avg_ns_traffic,
    AVG(eastbound_count + westbound_count) AS avg_ew_traffic,
    -- Quality metrics
    AVG(detection_accuracy) AS avg_detection_accuracy,
    AVG(data_completeness) AS avg_data_completeness
FROM traffic_metrics
GROUP BY bucket, camera_id, zone_id;

-- Add refresh policy for hourly aggregates
SELECT add_continuous_aggregate_policy('traffic_metrics_hourly',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Daily aggregates for long-term trends
CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_metrics_daily
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', timestamp) AS bucket,
    camera_id,
    zone_id,
    AVG(total_vehicles) AS avg_total_vehicles,
    MAX(total_vehicles) AS max_total_vehicles,
    MIN(total_vehicles) AS min_total_vehicles,
    SUM(total_vehicles) AS total_vehicles_per_day,
    AVG(average_speed) AS avg_speed,
    MAX(max_speed) AS max_speed,
    MIN(min_speed) AS min_speed,
    AVG(traffic_density) AS avg_traffic_density,
    AVG(occupancy_rate) AS avg_occupancy_rate,
    AVG(flow_rate) AS avg_flow_rate,
    -- Congestion analysis
    COUNT(*) FILTER (WHERE congestion_level = 'free_flow') AS free_flow_periods,
    COUNT(*) FILTER (WHERE congestion_level = 'light') AS light_congestion_periods,
    COUNT(*) FILTER (WHERE congestion_level = 'moderate') AS moderate_congestion_periods,
    COUNT(*) FILTER (WHERE congestion_level = 'heavy') AS heavy_congestion_periods,
    COUNT(*) FILTER (WHERE congestion_level = 'severe') AS severe_congestion_periods,
    -- Vehicle type breakdown
    AVG(vehicle_cars) AS avg_cars,
    AVG(vehicle_trucks) AS avg_trucks,
    AVG(vehicle_buses) AS avg_buses,
    AVG(vehicle_motorcycles) AS avg_motorcycles,
    AVG(vehicle_bicycles) AS avg_bicycles,
    -- Quality and completeness
    AVG(detection_accuracy) AS avg_detection_accuracy,
    AVG(data_completeness) AS avg_data_completeness,
    COUNT(*) AS total_measurements
FROM traffic_metrics
GROUP BY bucket, camera_id, zone_id;

-- Add refresh policy for daily aggregates
SELECT add_continuous_aggregate_policy('traffic_metrics_daily',
    start_offset => INTERVAL '1 week',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes on continuous aggregates for performance
CREATE INDEX IF NOT EXISTS idx_traffic_metrics_5min_bucket_camera 
    ON traffic_metrics_5min (bucket DESC, camera_id);
CREATE INDEX IF NOT EXISTS idx_traffic_metrics_5min_camera_zone 
    ON traffic_metrics_5min (camera_id, zone_id, bucket DESC);

CREATE INDEX IF NOT EXISTS idx_traffic_metrics_hourly_bucket_camera 
    ON traffic_metrics_hourly (bucket DESC, camera_id);
CREATE INDEX IF NOT EXISTS idx_traffic_metrics_hourly_camera_zone 
    ON traffic_metrics_hourly (camera_id, zone_id, bucket DESC);

CREATE INDEX IF NOT EXISTS idx_traffic_metrics_daily_bucket_camera 
    ON traffic_metrics_daily (bucket DESC, camera_id);
CREATE INDEX IF NOT EXISTS idx_traffic_metrics_daily_camera_zone 
    ON traffic_metrics_daily (camera_id, zone_id, bucket DESC);

-- Create functions for common analytics queries
CREATE OR REPLACE FUNCTION get_realtime_traffic_metrics(
    camera_ids TEXT[],
    minutes_back INTEGER DEFAULT 15
)
RETURNS TABLE (
    camera_id TEXT,
    latest_timestamp TIMESTAMPTZ,
    current_vehicles INTEGER,
    current_speed DECIMAL,
    current_density DECIMAL,
    current_congestion TEXT,
    vehicles_trend TEXT,
    speed_trend TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH current_metrics AS (
        SELECT DISTINCT ON (tm.camera_id)
            tm.camera_id::TEXT,
            tm.timestamp,
            tm.total_vehicles,
            tm.average_speed,
            tm.traffic_density,
            tm.congestion_level
        FROM traffic_metrics tm
        WHERE tm.camera_id = ANY(camera_ids)
            AND tm.timestamp >= NOW() - INTERVAL '1 minute' * minutes_back
        ORDER BY tm.camera_id, tm.timestamp DESC
    ),
    previous_metrics AS (
        SELECT DISTINCT ON (tm.camera_id)
            tm.camera_id::TEXT,
            tm.total_vehicles AS prev_vehicles,
            tm.average_speed AS prev_speed
        FROM traffic_metrics tm
        WHERE tm.camera_id = ANY(camera_ids)
            AND tm.timestamp >= NOW() - INTERVAL '1 minute' * (minutes_back + 5)
            AND tm.timestamp < NOW() - INTERVAL '1 minute' * minutes_back
        ORDER BY tm.camera_id, tm.timestamp DESC
    )
    SELECT 
        c.camera_id,
        c.timestamp,
        c.total_vehicles::INTEGER,
        c.average_speed::DECIMAL,
        c.traffic_density::DECIMAL,
        c.congestion_level,
        CASE 
            WHEN p.prev_vehicles IS NULL THEN 'stable'
            WHEN c.total_vehicles > p.prev_vehicles * 1.1 THEN 'increasing'
            WHEN c.total_vehicles < p.prev_vehicles * 0.9 THEN 'decreasing'
            ELSE 'stable'
        END AS vehicles_trend,
        CASE 
            WHEN p.prev_speed IS NULL THEN 'stable'
            WHEN c.average_speed > p.prev_speed * 1.1 THEN 'increasing'
            WHEN c.average_speed < p.prev_speed * 0.9 THEN 'decreasing'
            ELSE 'stable'
        END AS speed_trend
    FROM current_metrics c
    LEFT JOIN previous_metrics p ON c.camera_id = p.camera_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get traffic patterns for a camera
CREATE OR REPLACE FUNCTION get_traffic_patterns(
    target_camera_id TEXT,
    days_back INTEGER DEFAULT 7,
    granularity TEXT DEFAULT 'hourly'
)
RETURNS TABLE (
    time_period TIMESTAMPTZ,
    avg_vehicles DECIMAL,
    avg_speed DECIMAL,
    dominant_congestion TEXT,
    peak_vehicles INTEGER,
    data_points INTEGER
) AS $$
DECLARE
    interval_text TEXT;
BEGIN
    -- Determine time bucket interval
    CASE granularity
        WHEN 'hourly' THEN interval_text := '1 hour';
        WHEN 'daily' THEN interval_text := '1 day';
        WHEN '5min' THEN interval_text := '5 minutes';
        ELSE interval_text := '1 hour';
    END CASE;

    RETURN QUERY EXECUTE format('
        SELECT 
            time_bucket(%L, timestamp) AS time_period,
            AVG(total_vehicles)::DECIMAL AS avg_vehicles,
            AVG(average_speed)::DECIMAL AS avg_speed,
            MODE() WITHIN GROUP (ORDER BY congestion_level) AS dominant_congestion,
            MAX(total_vehicles)::INTEGER AS peak_vehicles,
            COUNT(*)::INTEGER AS data_points
        FROM traffic_metrics
        WHERE camera_id = %L
            AND timestamp >= NOW() - INTERVAL ''%s days''
        GROUP BY time_bucket(%L, timestamp)
        ORDER BY time_period DESC
    ', interval_text, target_camera_id, days_back, interval_text);
END;
$$ LANGUAGE plpgsql;

-- Function to detect traffic anomalies
CREATE OR REPLACE FUNCTION detect_traffic_anomalies(
    target_camera_id TEXT,
    hours_back INTEGER DEFAULT 24,
    threshold_multiplier DECIMAL DEFAULT 2.0
)
RETURNS TABLE (
    timestamp TIMESTAMPTZ,
    metric_name TEXT,
    actual_value DECIMAL,
    expected_value DECIMAL,
    anomaly_score DECIMAL,
    severity TEXT
) AS $$
BEGIN
    RETURN QUERY
    WITH historical_baseline AS (
        SELECT 
            camera_id,
            EXTRACT(hour FROM timestamp) AS hour_of_day,
            EXTRACT(dow FROM timestamp) AS day_of_week,
            AVG(total_vehicles) AS avg_vehicles,
            STDDEV(total_vehicles) AS stddev_vehicles,
            AVG(average_speed) AS avg_speed,
            STDDEV(average_speed) AS stddev_speed,
            AVG(traffic_density) AS avg_density,
            STDDEV(traffic_density) AS stddev_density
        FROM traffic_metrics
        WHERE camera_id = target_camera_id
            AND timestamp >= NOW() - INTERVAL '30 days'
            AND timestamp < NOW() - INTERVAL '1 day'
        GROUP BY camera_id, hour_of_day, day_of_week
    ),
    recent_data AS (
        SELECT 
            tm.*,
            EXTRACT(hour FROM tm.timestamp) AS hour_of_day,
            EXTRACT(dow FROM tm.timestamp) AS day_of_week
        FROM traffic_metrics tm
        WHERE tm.camera_id = target_camera_id
            AND tm.timestamp >= NOW() - INTERVAL '1 hour' * hours_back
    )
    SELECT 
        rd.timestamp,
        'vehicle_count'::TEXT AS metric_name,
        rd.total_vehicles::DECIMAL AS actual_value,
        hb.avg_vehicles::DECIMAL AS expected_value,
        CASE 
            WHEN hb.stddev_vehicles > 0 THEN 
                ABS(rd.total_vehicles - hb.avg_vehicles) / hb.stddev_vehicles
            ELSE 0
        END AS anomaly_score,
        CASE 
            WHEN hb.stddev_vehicles > 0 AND 
                 ABS(rd.total_vehicles - hb.avg_vehicles) / hb.stddev_vehicles > threshold_multiplier * 2 THEN 'critical'
            WHEN hb.stddev_vehicles > 0 AND 
                 ABS(rd.total_vehicles - hb.avg_vehicles) / hb.stddev_vehicles > threshold_multiplier THEN 'high'
            WHEN hb.stddev_vehicles > 0 AND 
                 ABS(rd.total_vehicles - hb.avg_vehicles) / hb.stddev_vehicles > threshold_multiplier * 0.5 THEN 'medium'
            ELSE 'low'
        END AS severity
    FROM recent_data rd
    LEFT JOIN historical_baseline hb ON 
        hb.camera_id = rd.camera_id AND
        hb.hour_of_day = rd.hour_of_day AND
        hb.day_of_week = rd.day_of_week
    WHERE hb.stddev_vehicles > 0 
        AND ABS(rd.total_vehicles - hb.avg_vehicles) / hb.stddev_vehicles > threshold_multiplier * 0.5
    
    UNION ALL
    
    SELECT 
        rd.timestamp,
        'average_speed'::TEXT AS metric_name,
        rd.average_speed::DECIMAL AS actual_value,
        hb.avg_speed::DECIMAL AS expected_value,
        CASE 
            WHEN hb.stddev_speed > 0 THEN 
                ABS(rd.average_speed - hb.avg_speed) / hb.stddev_speed
            ELSE 0
        END AS anomaly_score,
        CASE 
            WHEN hb.stddev_speed > 0 AND 
                 ABS(rd.average_speed - hb.avg_speed) / hb.stddev_speed > threshold_multiplier * 2 THEN 'critical'
            WHEN hb.stddev_speed > 0 AND 
                 ABS(rd.average_speed - hb.avg_speed) / hb.stddev_speed > threshold_multiplier THEN 'high'
            WHEN hb.stddev_speed > 0 AND 
                 ABS(rd.average_speed - hb.avg_speed) / hb.stddev_speed > threshold_multiplier * 0.5 THEN 'medium'
            ELSE 'low'
        END AS severity
    FROM recent_data rd
    LEFT JOIN historical_baseline hb ON 
        hb.camera_id = rd.camera_id AND
        hb.hour_of_day = rd.hour_of_day AND
        hb.day_of_week = rd.day_of_week
    WHERE hb.stddev_speed > 0 
        AND ABS(rd.average_speed - hb.avg_speed) / hb.stddev_speed > threshold_multiplier * 0.5
    
    ORDER BY timestamp DESC, anomaly_score DESC;
END;
$$ LANGUAGE plpgsql;

-- Create performance monitoring view
CREATE OR REPLACE VIEW v_traffic_performance_summary AS
SELECT 
    camera_id,
    DATE_TRUNC('hour', timestamp) AS hour_bucket,
    COUNT(*) AS measurements_count,
    AVG(total_vehicles) AS avg_vehicles,
    MAX(total_vehicles) AS peak_vehicles,
    AVG(average_speed) AS avg_speed,
    MIN(average_speed) AS min_speed,
    MAX(max_speed) AS max_speed,
    AVG(traffic_density) AS avg_density,
    AVG(occupancy_rate) AS avg_occupancy,
    MODE() WITHIN GROUP (ORDER BY congestion_level) AS dominant_congestion,
    AVG(processing_latency_ms) AS avg_processing_latency,
    AVG(data_completeness) AS avg_data_completeness,
    COUNT(*) FILTER (WHERE congestion_level IN ('heavy', 'severe')) AS congested_periods
FROM traffic_metrics
WHERE timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY camera_id, DATE_TRUNC('hour', timestamp)
ORDER BY hour_bucket DESC, camera_id;

-- Create alerts view for real-time monitoring
CREATE OR REPLACE VIEW v_traffic_alerts AS
WITH recent_metrics AS (
    SELECT DISTINCT ON (camera_id)
        camera_id,
        timestamp,
        total_vehicles,
        average_speed,
        congestion_level,
        traffic_density,
        occupancy_rate,
        processing_latency_ms,
        data_completeness
    FROM traffic_metrics
    WHERE timestamp >= NOW() - INTERVAL '10 minutes'
    ORDER BY camera_id, timestamp DESC
)
SELECT 
    camera_id,
    timestamp,
    ARRAY_AGG(DISTINCT alert_type) AS alert_types,
    COUNT(*) AS alert_count,
    MAX(severity_level) AS max_severity
FROM (
    SELECT 
        camera_id,
        timestamp,
        'high_congestion' AS alert_type,
        3 AS severity_level
    FROM recent_metrics
    WHERE congestion_level IN ('heavy', 'severe')
    
    UNION ALL
    
    SELECT 
        camera_id,
        timestamp,
        'low_speed' AS alert_type,
        2 AS severity_level
    FROM recent_metrics
    WHERE average_speed < 20 AND total_vehicles > 10
    
    UNION ALL
    
    SELECT 
        camera_id,
        timestamp,
        'high_density' AS alert_type,
        2 AS severity_level
    FROM recent_metrics
    WHERE traffic_density > 0.8
    
    UNION ALL
    
    SELECT 
        camera_id,
        timestamp,
        'processing_delay' AS alert_type,
        2 AS severity_level
    FROM recent_metrics
    WHERE processing_latency_ms > 200
    
    UNION ALL
    
    SELECT 
        camera_id,
        timestamp,
        'data_quality' AS alert_type,
        1 AS severity_level
    FROM recent_metrics
    WHERE data_completeness < 0.8
) alerts
GROUP BY camera_id, timestamp
ORDER BY max_severity DESC, timestamp DESC;

-- Create performance optimization statistics
ANALYZE traffic_metrics;
ANALYZE rule_violations;
ANALYZE vehicle_trajectories;
ANALYZE traffic_anomalies;

-- Create statistics for continuous aggregates
ANALYZE traffic_metrics_5min;
ANALYZE traffic_metrics_hourly;
ANALYZE traffic_metrics_daily;

-- Set up automatic statistics updates
-- This will help the query planner make better decisions
SELECT add_job('ANALYZE traffic_metrics;', '1 hour', if_not_exists => TRUE);

RAISE NOTICE 'TimescaleDB setup completed successfully';
RAISE NOTICE 'Created hypertables: traffic_metrics';
RAISE NOTICE 'Created continuous aggregates: traffic_metrics_5min, traffic_metrics_hourly, traffic_metrics_daily';
RAISE NOTICE 'Created functions: get_realtime_traffic_metrics, get_traffic_patterns, detect_traffic_anomalies';
RAISE NOTICE 'Created views: v_traffic_performance_summary, v_traffic_alerts';