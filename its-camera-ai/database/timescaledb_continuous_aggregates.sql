-- TimescaleDB Continuous Aggregates for ITS Camera AI
-- This script sets up continuous aggregates for real-time data rollups
-- Supporting 10TB/day data processing with sub-second performance

-- Enable TimescaleDB extension if not already enabled
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create hypertable for traffic_metrics if not exists
-- (This assumes the table already exists from the main schema)
SELECT create_hypertable('traffic_metrics', 'timestamp', 
                        chunk_time_interval => INTERVAL '1 hour',
                        if_not_exists => TRUE);

-- Configure chunk compression for older data
ALTER TABLE traffic_metrics SET (
    timescaledb.compress,
    timescaledb.compress_orderby = 'timestamp DESC',
    timescaledb.compress_segmentby = 'camera_id'
);

-- Add compression policy for data older than 1 day
SELECT add_compression_policy('traffic_metrics', INTERVAL '1 day');

-- Add retention policy for raw data older than 6 months
SELECT add_retention_policy('traffic_metrics', INTERVAL '6 months');

-- 1-Minute Continuous Aggregate
-- Aggregates raw detection data into 1-minute intervals
CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_metrics_1min
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 minute', timestamp) AS bucket,
    camera_id,
    '1min'::text AS aggregation_period,
    
    -- Vehicle counts
    SUM(total_vehicles) AS total_vehicles,
    SUM(vehicle_cars) AS vehicle_cars,
    SUM(vehicle_trucks) AS vehicle_trucks,
    SUM(vehicle_buses) AS vehicle_buses,
    SUM(vehicle_motorcycles) AS vehicle_motorcycles,
    SUM(vehicle_bicycles) AS vehicle_bicycles,
    
    -- Directional flow
    SUM(northbound_count) AS northbound_count,
    SUM(southbound_count) AS southbound_count,
    SUM(eastbound_count) AS eastbound_count,
    SUM(westbound_count) AS westbound_count,
    
    -- Speed metrics
    AVG(average_speed) AS average_speed,
    MIN(average_speed) AS min_speed,
    MAX(average_speed) AS max_speed,
    STDDEV(average_speed) AS speed_stddev,
    
    -- Traffic density and flow
    AVG(traffic_density) AS traffic_density,
    AVG(flow_rate) AS flow_rate,
    AVG(occupancy_rate) AS occupancy_rate,
    AVG(queue_length) AS queue_length,
    
    -- Congestion level (mode)
    MODE() WITHIN GROUP (ORDER BY congestion_level) AS congestion_level,
    
    -- Data quality metrics
    COUNT(*) AS sample_count,
    AVG(data_completeness) AS data_completeness,
    
    -- Statistical measures
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_vehicles) AS p25_vehicles,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY total_vehicles) AS p50_vehicles,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_vehicles) AS p75_vehicles,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_vehicles) AS p95_vehicles,
    
    -- Zone and lane information
    zone_id,
    lane_id,
    
    -- Timestamps
    MIN(timestamp) AS period_start,
    MAX(timestamp) AS period_end,
    NOW() AS created_at,
    NOW() AS updated_at
    
FROM traffic_metrics
WHERE aggregation_period = 'raw' OR aggregation_period IS NULL
GROUP BY bucket, camera_id, zone_id, lane_id;

-- Add refresh policy for 1-minute aggregates (refresh every 30 seconds)
SELECT add_continuous_aggregate_policy('traffic_metrics_1min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '30 seconds',
    schedule_interval => INTERVAL '30 seconds');

-- 5-Minute Continuous Aggregate
-- Aggregates 1-minute data into 5-minute intervals
CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_metrics_5min
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 minutes', bucket) AS bucket,
    camera_id,
    '5min'::text AS aggregation_period,
    
    -- Vehicle counts (summed)
    SUM(total_vehicles) AS total_vehicles,
    SUM(vehicle_cars) AS vehicle_cars,
    SUM(vehicle_trucks) AS vehicle_trucks,
    SUM(vehicle_buses) AS vehicle_buses,
    SUM(vehicle_motorcycles) AS vehicle_motorcycles,
    SUM(vehicle_bicycles) AS vehicle_bicycles,
    
    -- Directional flow (summed)
    SUM(northbound_count) AS northbound_count,
    SUM(southbound_count) AS southbound_count,
    SUM(eastbound_count) AS eastbound_count,
    SUM(westbound_count) AS westbound_count,
    
    -- Speed metrics (averaged)
    AVG(average_speed) AS average_speed,
    MIN(min_speed) AS min_speed,
    MAX(max_speed) AS max_speed,
    AVG(speed_stddev) AS speed_stddev,
    
    -- Traffic density and flow (averaged)
    AVG(traffic_density) AS traffic_density,
    AVG(flow_rate) AS flow_rate,
    AVG(occupancy_rate) AS occupancy_rate,
    AVG(queue_length) AS queue_length,
    
    -- Congestion level (mode of modes)
    MODE() WITHIN GROUP (ORDER BY congestion_level) AS congestion_level,
    
    -- Data quality metrics
    SUM(sample_count) AS sample_count,
    AVG(data_completeness) AS data_completeness,
    
    -- Statistical measures (averaged percentiles)
    AVG(p25_vehicles) AS p25_vehicles,
    AVG(p50_vehicles) AS p50_vehicles,
    AVG(p75_vehicles) AS p75_vehicles,
    AVG(p95_vehicles) AS p95_vehicles,
    
    -- Zone and lane information
    zone_id,
    lane_id,
    
    -- Timestamps
    MIN(period_start) AS period_start,
    MAX(period_end) AS period_end,
    NOW() AS created_at,
    NOW() AS updated_at
    
FROM traffic_metrics_1min
GROUP BY bucket, camera_id, zone_id, lane_id;

-- Add refresh policy for 5-minute aggregates (refresh every 2 minutes)
SELECT add_continuous_aggregate_policy('traffic_metrics_5min',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '2 minutes',
    schedule_interval => INTERVAL '2 minutes');

-- Hourly Continuous Aggregate
-- Aggregates 5-minute data into hourly intervals
CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', bucket) AS bucket,
    camera_id,
    '1hour'::text AS aggregation_period,
    
    -- Vehicle counts (summed)
    SUM(total_vehicles) AS total_vehicles,
    SUM(vehicle_cars) AS vehicle_cars,
    SUM(vehicle_trucks) AS vehicle_trucks,
    SUM(vehicle_buses) AS vehicle_buses,
    SUM(vehicle_motorcycles) AS vehicle_motorcycles,
    SUM(vehicle_bicycles) AS vehicle_bicycles,
    
    -- Directional flow (summed)
    SUM(northbound_count) AS northbound_count,
    SUM(southbound_count) AS southbound_count,
    SUM(eastbound_count) AS eastbound_count,
    SUM(westbound_count) AS westbound_count,
    
    -- Speed metrics (averaged with proper weighting)
    AVG(average_speed) AS average_speed,
    MIN(min_speed) AS min_speed,
    MAX(max_speed) AS max_speed,
    
    -- Weighted standard deviation
    SQRT(AVG(speed_stddev * speed_stddev)) AS speed_stddev,
    
    -- Traffic density and flow (averaged)
    AVG(traffic_density) AS traffic_density,
    AVG(flow_rate) AS flow_rate,
    AVG(occupancy_rate) AS occupancy_rate,
    AVG(queue_length) AS queue_length,
    
    -- Congestion level distribution
    MODE() WITHIN GROUP (ORDER BY congestion_level) AS congestion_level,
    
    -- Congestion distribution (JSON)
    json_object_agg(
        congestion_level, 
        COUNT(*)::float / SUM(COUNT(*)) OVER (PARTITION BY bucket, camera_id)
    ) AS congestion_distribution,
    
    -- Data quality metrics
    SUM(sample_count) AS sample_count,
    AVG(data_completeness) AS data_completeness,
    
    -- Statistical measures
    AVG(p25_vehicles) AS p25_vehicles,
    AVG(p50_vehicles) AS p50_vehicles,
    AVG(p75_vehicles) AS p75_vehicles,
    AVG(p95_vehicles) AS p95_vehicles,
    
    -- Additional hourly statistics
    STDDEV(total_vehicles) AS vehicles_stddev,
    MIN(total_vehicles) AS min_vehicles,
    MAX(total_vehicles) AS max_vehicles,
    
    -- Peak detection
    CASE 
        WHEN AVG(total_vehicles) > LAG(AVG(total_vehicles), 1) OVER (
            PARTITION BY camera_id ORDER BY bucket
        ) AND AVG(total_vehicles) > LEAD(AVG(total_vehicles), 1) OVER (
            PARTITION BY camera_id ORDER BY bucket
        ) THEN true 
        ELSE false 
    END AS is_peak_hour,
    
    -- Zone and lane information
    zone_id,
    lane_id,
    
    -- Time patterns
    EXTRACT(HOUR FROM bucket) AS hour_of_day,
    EXTRACT(DOW FROM bucket) AS day_of_week,
    EXTRACT(WEEK FROM bucket) AS week_of_year,
    
    -- Timestamps
    MIN(period_start) AS period_start,
    MAX(period_end) AS period_end,
    NOW() AS created_at,
    NOW() AS updated_at
    
FROM traffic_metrics_5min
GROUP BY bucket, camera_id, zone_id, lane_id;

-- Add refresh policy for hourly aggregates (refresh every 10 minutes)
SELECT add_continuous_aggregate_policy('traffic_metrics_hourly',
    start_offset => INTERVAL '12 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '10 minutes');

-- Daily Continuous Aggregate
-- Aggregates hourly data into daily summaries
CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_metrics_daily
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', bucket) AS bucket,
    camera_id,
    '1day'::text AS aggregation_period,
    
    -- Daily totals
    SUM(total_vehicles) AS total_vehicles,
    SUM(vehicle_cars) AS vehicle_cars,
    SUM(vehicle_trucks) AS vehicle_trucks,
    SUM(vehicle_buses) AS vehicle_buses,
    SUM(vehicle_motorcycles) AS vehicle_motorcycles,
    SUM(vehicle_bicycles) AS vehicle_bicycles,
    
    -- Daily directional flow
    SUM(northbound_count) AS northbound_count,
    SUM(southbound_count) AS southbound_count,
    SUM(eastbound_count) AS eastbound_count,
    SUM(westbound_count) AS westbound_count,
    
    -- Daily speed statistics
    AVG(average_speed) AS average_speed,
    MIN(min_speed) AS min_speed,
    MAX(max_speed) AS max_speed,
    STDDEV(average_speed) AS speed_stddev,
    
    -- Daily traffic patterns
    AVG(traffic_density) AS avg_traffic_density,
    MAX(traffic_density) AS max_traffic_density,
    AVG(flow_rate) AS avg_flow_rate,
    MAX(flow_rate) AS max_flow_rate,
    AVG(occupancy_rate) AS avg_occupancy_rate,
    MAX(occupancy_rate) AS max_occupancy_rate,
    
    -- Peak hours analysis
    COUNT(CASE WHEN is_peak_hour THEN 1 END) AS peak_hours_count,
    ARRAY_AGG(CASE WHEN is_peak_hour THEN hour_of_day END) 
        FILTER (WHERE is_peak_hour) AS peak_hours,
    
    -- Congestion analysis
    MODE() WITHIN GROUP (ORDER BY congestion_level) AS dominant_congestion_level,
    
    -- Hourly patterns (JSON object with hour -> avg_vehicles)
    json_object_agg(hour_of_day, total_vehicles) AS hourly_pattern,
    
    -- Data quality
    SUM(sample_count) AS sample_count,
    AVG(data_completeness) AS data_completeness,
    
    -- Daily statistics
    STDDEV(total_vehicles) AS daily_vehicles_stddev,
    MIN(total_vehicles) AS min_hourly_vehicles,
    MAX(total_vehicles) AS max_hourly_vehicles,
    
    -- Zone and lane information
    zone_id,
    lane_id,
    
    -- Date information
    EXTRACT(DOW FROM bucket) AS day_of_week,
    EXTRACT(WEEK FROM bucket) AS week_of_year,
    EXTRACT(MONTH FROM bucket) AS month,
    EXTRACT(QUARTER FROM bucket) AS quarter,
    
    -- Timestamps
    bucket AS date,
    MIN(period_start) AS period_start,
    MAX(period_end) AS period_end,
    NOW() AS created_at,
    NOW() AS updated_at
    
FROM traffic_metrics_hourly
GROUP BY bucket, camera_id, zone_id, lane_id;

-- Add refresh policy for daily aggregates (refresh every 4 hours)
SELECT add_continuous_aggregate_policy('traffic_metrics_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '4 hours',
    schedule_interval => INTERVAL '4 hours');

-- Create indexes for optimal query performance
CREATE INDEX IF NOT EXISTS idx_traffic_metrics_1min_camera_bucket 
    ON traffic_metrics_1min (camera_id, bucket DESC);
    
CREATE INDEX IF NOT EXISTS idx_traffic_metrics_5min_camera_bucket 
    ON traffic_metrics_5min (camera_id, bucket DESC);
    
CREATE INDEX IF NOT EXISTS idx_traffic_metrics_hourly_camera_bucket 
    ON traffic_metrics_hourly (camera_id, bucket DESC);
    
CREATE INDEX IF NOT EXISTS idx_traffic_metrics_daily_camera_bucket 
    ON traffic_metrics_daily (camera_id, bucket DESC);

-- Create indexes for zone and lane filtering
CREATE INDEX IF NOT EXISTS idx_traffic_metrics_hourly_zone_lane 
    ON traffic_metrics_hourly (zone_id, lane_id, bucket DESC) 
    WHERE zone_id IS NOT NULL AND lane_id IS NOT NULL;

-- Create partial indexes for peak hour analysis
CREATE INDEX IF NOT EXISTS idx_traffic_metrics_hourly_peaks 
    ON traffic_metrics_hourly (camera_id, bucket DESC) 
    WHERE is_peak_hour = true;

-- Create weekly and monthly aggregates for long-term analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_metrics_weekly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 week', bucket) AS bucket,
    camera_id,
    '1week'::text AS aggregation_period,
    
    SUM(total_vehicles) AS total_vehicles,
    AVG(average_speed) AS average_speed,
    AVG(avg_traffic_density) AS avg_traffic_density,
    
    -- Weekly patterns
    json_object_agg(day_of_week, total_vehicles) AS weekly_pattern,
    
    -- Summary statistics
    STDDEV(total_vehicles) AS weekly_vehicles_stddev,
    MIN(min_hourly_vehicles) AS min_vehicles,
    MAX(max_hourly_vehicles) AS max_vehicles,
    
    zone_id,
    lane_id,
    
    EXTRACT(WEEK FROM bucket) AS week_of_year,
    EXTRACT(MONTH FROM bucket) AS month,
    EXTRACT(YEAR FROM bucket) AS year,
    
    NOW() AS created_at,
    NOW() AS updated_at
    
FROM traffic_metrics_daily
GROUP BY bucket, camera_id, zone_id, lane_id;

-- Add refresh policy for weekly aggregates (refresh daily)
SELECT add_continuous_aggregate_policy('traffic_metrics_weekly',
    start_offset => INTERVAL '2 weeks',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- Monthly aggregates for long-term trend analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_metrics_monthly
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 month', bucket) AS bucket,
    camera_id,
    '1month'::text AS aggregation_period,
    
    SUM(total_vehicles) AS total_vehicles,
    AVG(average_speed) AS average_speed,
    AVG(avg_traffic_density) AS avg_traffic_density,
    
    -- Monthly growth calculation
    LAG(SUM(total_vehicles)) OVER (
        PARTITION BY camera_id ORDER BY time_bucket('1 month', bucket)
    ) AS prev_month_vehicles,
    
    CASE 
        WHEN LAG(SUM(total_vehicles)) OVER (
            PARTITION BY camera_id ORDER BY time_bucket('1 month', bucket)
        ) > 0 THEN
            ((SUM(total_vehicles) - LAG(SUM(total_vehicles)) OVER (
                PARTITION BY camera_id ORDER BY time_bucket('1 month', bucket)
            )) * 100.0 / LAG(SUM(total_vehicles)) OVER (
                PARTITION BY camera_id ORDER BY time_bucket('1 month', bucket)
            ))
        ELSE NULL
    END AS month_over_month_growth_percent,
    
    zone_id,
    lane_id,
    
    EXTRACT(MONTH FROM bucket) AS month,
    EXTRACT(QUARTER FROM bucket) AS quarter,
    EXTRACT(YEAR FROM bucket) AS year,
    
    NOW() AS created_at,
    NOW() AS updated_at
    
FROM traffic_metrics_daily
GROUP BY bucket, camera_id, zone_id, lane_id;

-- Add refresh policy for monthly aggregates (refresh weekly)
SELECT add_continuous_aggregate_policy('traffic_metrics_monthly',
    start_offset => INTERVAL '3 months',
    end_offset => INTERVAL '1 week',
    schedule_interval => INTERVAL '1 week');

-- Create a unified view that combines all aggregation levels
CREATE OR REPLACE VIEW traffic_metrics_unified AS
SELECT 
    bucket AS timestamp,
    camera_id,
    aggregation_period,
    total_vehicles,
    average_speed,
    traffic_density,
    congestion_level,
    zone_id,
    lane_id,
    created_at,
    updated_at
FROM traffic_metrics_1min
UNION ALL
SELECT 
    bucket AS timestamp,
    camera_id,
    aggregation_period,
    total_vehicles,
    average_speed,
    traffic_density,
    congestion_level,
    zone_id,
    lane_id,
    created_at,
    updated_at
FROM traffic_metrics_5min
UNION ALL
SELECT 
    bucket AS timestamp,
    camera_id,
    aggregation_period,
    total_vehicles,
    average_speed,
    traffic_density,
    congestion_level,
    zone_id,
    lane_id,
    created_at,
    updated_at
FROM traffic_metrics_hourly
UNION ALL
SELECT 
    bucket AS timestamp,
    camera_id,
    aggregation_period,
    total_vehicles,
    average_speed,
    avg_traffic_density AS traffic_density,
    dominant_congestion_level AS congestion_level,
    zone_id,
    lane_id,
    created_at,
    updated_at
FROM traffic_metrics_daily;

-- Grant permissions to application user
GRANT SELECT ON ALL TABLES IN SCHEMA public TO its_camera_ai_user;
GRANT SELECT ON traffic_metrics_unified TO its_camera_ai_user;

-- Create helper functions for querying aggregated data

-- Function to get optimal aggregation level based on time range
CREATE OR REPLACE FUNCTION get_optimal_aggregation_level(
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE
) RETURNS TEXT AS $$
DECLARE
    duration_hours INTEGER;
BEGIN
    duration_hours := EXTRACT(EPOCH FROM (end_time - start_time)) / 3600;
    
    IF duration_hours <= 2 THEN
        RETURN '1min';
    ELSIF duration_hours <= 12 THEN
        RETURN '5min';
    ELSIF duration_hours <= 168 THEN -- 1 week
        RETURN '1hour';
    ELSIF duration_hours <= 720 THEN -- 1 month
        RETURN '1day';
    ELSIF duration_hours <= 8760 THEN -- 1 year
        RETURN '1week';
    ELSE
        RETURN '1month';
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to query traffic metrics with automatic aggregation level selection
CREATE OR REPLACE FUNCTION get_traffic_metrics(
    p_camera_id TEXT,
    p_start_time TIMESTAMP WITH TIME ZONE,
    p_end_time TIMESTAMP WITH TIME ZONE,
    p_aggregation_level TEXT DEFAULT NULL
) RETURNS TABLE (
    timestamp TIMESTAMP WITH TIME ZONE,
    camera_id TEXT,
    aggregation_period TEXT,
    total_vehicles BIGINT,
    average_speed NUMERIC,
    traffic_density NUMERIC,
    congestion_level TEXT
) AS $$
DECLARE
    optimal_level TEXT;
BEGIN
    -- Determine optimal aggregation level if not specified
    IF p_aggregation_level IS NULL THEN
        optimal_level := get_optimal_aggregation_level(p_start_time, p_end_time);
    ELSE
        optimal_level := p_aggregation_level;
    END IF;
    
    -- Query appropriate materialized view
    RETURN QUERY
    SELECT 
        t.timestamp,
        t.camera_id,
        t.aggregation_period,
        t.total_vehicles,
        t.average_speed,
        t.traffic_density,
        t.congestion_level
    FROM traffic_metrics_unified t
    WHERE t.camera_id = p_camera_id
        AND t.timestamp >= p_start_time
        AND t.timestamp <= p_end_time
        AND t.aggregation_period = optimal_level
    ORDER BY t.timestamp;
END;
$$ LANGUAGE plpgsql;

-- Performance monitoring views
CREATE OR REPLACE VIEW continuous_aggregate_stats AS
SELECT 
    hypertable_name,
    view_name,
    completed_threshold,
    invalidation_threshold,
    job_id,
    last_run_duration,
    next_start,
    job_status
FROM timescaledb_information.jobs j
JOIN timescaledb_information.continuous_aggregates ca ON j.hypertable_name = ca.hypertable_name
WHERE j.proc_name = 'policy_refresh_continuous_aggregate'
ORDER BY hypertable_name, view_name;

-- Chunk compression stats
CREATE OR REPLACE VIEW chunk_compression_stats AS
SELECT 
    chunk_schema,
    chunk_name,
    table_name,
    compression_status,
    before_compression_table_bytes,
    before_compression_index_bytes,
    before_compression_toast_bytes,
    after_compression_table_bytes,
    after_compression_index_bytes,
    after_compression_toast_bytes,
    CASE 
        WHEN before_compression_table_bytes > 0 THEN
            ROUND(
                (1.0 - after_compression_table_bytes::numeric / before_compression_table_bytes) * 100, 
                2
            )
        ELSE 0
    END AS compression_ratio_percent
FROM timescaledb_information.chunks c
LEFT JOIN timescaledb_information.compressed_chunk_stats ccs 
    ON c.chunk_name = ccs.chunk_name
WHERE c.table_name = 'traffic_metrics'
ORDER BY c.range_start DESC;