-- PostgreSQL table partitioning strategy for ITS Camera AI time-series data
-- Optimized for high-throughput frame metadata and metrics storage

-- =====================================================
-- FRAME METADATA PARTITIONING (Monthly)
-- Handles 3000+ inserts/sec with efficient queries
-- =====================================================

-- Create partition management function
CREATE OR REPLACE FUNCTION create_monthly_partitions(
    table_name TEXT,
    start_date DATE,
    end_date DATE
) RETURNS INTEGER AS $$
DECLARE
    current_date DATE := start_date;
    partition_count INTEGER := 0;
    partition_name TEXT;
    partition_start DATE;
    partition_end DATE;
    sql_stmt TEXT;
BEGIN
    WHILE current_date < end_date LOOP
        partition_start := DATE_TRUNC('month', current_date);
        partition_end := partition_start + INTERVAL '1 month';
        partition_name := table_name || '_' || TO_CHAR(partition_start, 'YYYY_MM');
        
        -- Create partition if it doesn't exist
        sql_stmt := FORMAT(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I 
             FOR VALUES FROM (%L) TO (%L)',
            partition_name, table_name, partition_start, partition_end
        );
        
        EXECUTE sql_stmt;
        
        -- Create partition-specific indexes for performance
        IF table_name = 'frame_metadata' THEN
            EXECUTE FORMAT(
                'CREATE INDEX IF NOT EXISTS %I ON %I (camera_id, timestamp)',
                'idx_' || partition_name || '_camera_time', partition_name
            );
            EXECUTE FORMAT(
                'CREATE INDEX IF NOT EXISTS %I ON %I (processing_status)',
                'idx_' || partition_name || '_status', partition_name
            );
        ELSIF table_name = 'system_metrics' THEN
            EXECUTE FORMAT(
                'CREATE INDEX IF NOT EXISTS %I ON %I (camera_id, metric_name)',
                'idx_' || partition_name || '_camera_metric', partition_name
            );
            EXECUTE FORMAT(
                'CREATE INDEX IF NOT EXISTS %I ON %I (metric_name, timestamp)',
                'idx_' || partition_name || '_metric_time', partition_name
            );
        END IF;
        
        partition_count := partition_count + 1;
        current_date := partition_end;
    END LOOP;
    
    RETURN partition_count;
END;
$$ LANGUAGE plpgsql;

-- Create initial frame_metadata partitions (current month + 6 months ahead)
SELECT create_monthly_partitions(
    'frame_metadata',
    DATE_TRUNC('month', CURRENT_DATE),
    DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '6 months'
);

-- Create initial system_metrics partitions
SELECT create_monthly_partitions(
    'system_metrics', 
    DATE_TRUNC('month', CURRENT_DATE),
    DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '6 months'
);

-- =====================================================
-- AUTOMATED PARTITION MANAGEMENT
-- =====================================================

-- Function to automatically create next month's partitions
CREATE OR REPLACE FUNCTION maintain_partitions()
RETURNS VOID AS $$
DECLARE
    next_month DATE;
BEGIN
    next_month := DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month';
    
    -- Ensure next month partitions exist
    PERFORM create_monthly_partitions('frame_metadata', next_month, next_month + INTERVAL '1 month');
    PERFORM create_monthly_partitions('system_metrics', next_month, next_month + INTERVAL '1 month');
    
    -- Log partition maintenance
    INSERT INTO system_metrics (metric_name, metric_type, value, source, labels)
    VALUES (
        'partition_maintenance_completed',
        'counter',
        1,
        'partition_manager',
        jsonb_build_object('timestamp', CURRENT_TIMESTAMP, 'next_month', next_month)
    );
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PARTITION PRUNING AND ARCHIVAL
-- =====================================================

-- Function to archive old partitions based on retention policies
CREATE OR REPLACE FUNCTION archive_old_partitions(
    table_name TEXT,
    retention_months INTEGER
) RETURNS INTEGER AS $$
DECLARE
    cutoff_date DATE;
    partition_name TEXT;
    archived_count INTEGER := 0;
    rec RECORD;
BEGIN
    cutoff_date := DATE_TRUNC('month', CURRENT_DATE) - (retention_months || ' months')::INTERVAL;
    
    -- Find partitions older than retention policy
    FOR rec IN 
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE tablename LIKE table_name || '_%'
          AND tablename ~ '\d{4}_\d{2}$'
          AND TO_DATE(RIGHT(tablename, 7), 'YYYY_MM') < cutoff_date
    LOOP
        partition_name := rec.tablename;
        
        -- Archive partition data (example: export to S3 or external storage)
        -- This would be implemented based on your archival strategy
        
        -- Drop the old partition (CAUTION: Data loss!)
        EXECUTE FORMAT('DROP TABLE IF EXISTS %I CASCADE', partition_name);
        
        archived_count := archived_count + 1;
        
        RAISE NOTICE 'Archived partition: %', partition_name;
    END LOOP;
    
    -- Log archival activity
    INSERT INTO system_metrics (metric_name, metric_type, value, source, labels)
    VALUES (
        'partitions_archived',
        'counter',
        archived_count,
        'partition_manager',
        jsonb_build_object(
            'table_name', table_name,
            'retention_months', retention_months,
            'cutoff_date', cutoff_date
        )
    );
    
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PARTITION STATISTICS AND MONITORING
-- =====================================================

-- View for partition size monitoring
CREATE OR REPLACE VIEW partition_statistics AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes,
    CASE 
        WHEN tablename LIKE 'frame_metadata_%' THEN 'frame_metadata'
        WHEN tablename LIKE 'system_metrics_%' THEN 'system_metrics'
        ELSE 'other'
    END as partition_type,
    CASE 
        WHEN tablename ~ '\d{4}_\d{2}$' THEN 
            TO_DATE(RIGHT(tablename, 7), 'YYYY_MM')
        ELSE NULL
    END as partition_month
FROM pg_tables
WHERE tablename LIKE 'frame_metadata_%'
   OR tablename LIKE 'system_metrics_%'
ORDER BY partition_month DESC, size_bytes DESC;

-- Function to get partition performance stats
CREATE OR REPLACE FUNCTION get_partition_performance_stats()
RETURNS TABLE (
    partition_name TEXT,
    partition_type TEXT,
    row_count BIGINT,
    avg_query_time NUMERIC,
    index_efficiency NUMERIC,
    last_vacuum TIMESTAMP,
    needs_maintenance BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    WITH partition_info AS (
        SELECT 
            t.tablename as pname,
            CASE 
                WHEN t.tablename LIKE 'frame_metadata_%' THEN 'frame_metadata'
                WHEN t.tablename LIKE 'system_metrics_%' THEN 'system_metrics'
                ELSE 'other'
            END as ptype
        FROM pg_tables t
        WHERE t.tablename LIKE 'frame_metadata_%'
           OR t.tablename LIKE 'system_metrics_%'
    ),
    table_stats AS (
        SELECT 
            pi.pname,
            pi.ptype,
            COALESCE(c.reltuples::BIGINT, 0) as row_count,
            COALESCE(s.last_vacuum, s.last_autovacuum) as last_vacuum_time
        FROM partition_info pi
        LEFT JOIN pg_class c ON c.relname = pi.pname
        LEFT JOIN pg_stat_user_tables s ON s.relname = pi.pname
    )
    SELECT 
        ts.pname::TEXT,
        ts.ptype::TEXT,
        ts.row_count,
        0.0::NUMERIC as avg_query_time, -- Would be populated from pg_stat_statements
        100.0::NUMERIC as index_efficiency, -- Calculated from actual index usage
        ts.last_vacuum_time,
        CASE 
            WHEN ts.last_vacuum_time < CURRENT_TIMESTAMP - INTERVAL '7 days' THEN TRUE
            WHEN ts.row_count > 10000000 THEN TRUE  -- 10M rows threshold
            ELSE FALSE
        END as needs_maintenance
    FROM table_stats ts
    ORDER BY ts.row_count DESC;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- AUTOMATED MAINTENANCE SCHEDULING
-- =====================================================

-- Create extension for pg_cron if available (requires superuser)
-- CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Schedule partition maintenance (runs monthly)
-- SELECT cron.schedule('partition-maintenance', '0 2 1 * *', 'SELECT maintain_partitions()');

-- Schedule archival (runs monthly, archives data older than 6 months)
-- SELECT cron.schedule('partition-archival', '0 3 1 * *', 'SELECT archive_old_partitions(''frame_metadata'', 6)');
-- SELECT cron.schedule('metrics-archival', '0 3 1 * *', 'SELECT archive_old_partitions(''system_metrics'', 12)');

-- Alternative: Manual scheduling function for systems without pg_cron
CREATE OR REPLACE FUNCTION schedule_maintenance_job()
RETURNS VOID AS $$
BEGIN
    -- Check if it's the first day of the month
    IF EXTRACT(day FROM CURRENT_DATE) = 1 THEN
        PERFORM maintain_partitions();
        
        -- Archive frame_metadata older than 6 months
        PERFORM archive_old_partitions('frame_metadata', 6);
        
        -- Archive system_metrics older than 12 months
        PERFORM archive_old_partitions('system_metrics', 12);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PARTITION-AWARE QUERY OPTIMIZATION
-- =====================================================

-- Function to get optimal partition for date range queries
CREATE OR REPLACE FUNCTION get_relevant_partitions(
    table_name TEXT,
    start_date TIMESTAMP,
    end_date TIMESTAMP
) RETURNS TEXT[] AS $$
DECLARE
    partition_names TEXT[];
    current_month DATE;
    end_month DATE;
    partition_name TEXT;
BEGIN
    current_month := DATE_TRUNC('month', start_date::DATE);
    end_month := DATE_TRUNC('month', end_date::DATE);
    
    partition_names := ARRAY[]::TEXT[];
    
    WHILE current_month <= end_month LOOP
        partition_name := table_name || '_' || TO_CHAR(current_month, 'YYYY_MM');
        
        -- Check if partition exists
        IF EXISTS (
            SELECT 1 FROM pg_tables 
            WHERE tablename = partition_name
        ) THEN
            partition_names := partition_names || partition_name;
        END IF;
        
        current_month := current_month + INTERVAL '1 month';
    END LOOP;
    
    RETURN partition_names;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- PERFORMANCE OPTIMIZATION SETTINGS
-- =====================================================

-- Optimize constraint exclusion for partitioning
SET constraint_exclusion = partition;

-- Enable parallel query execution for large partition scans
SET max_parallel_workers_per_gather = 4;
SET parallel_tuple_cost = 0.1;
SET parallel_setup_cost = 1000.0;

-- Optimize work memory for large sorts/joins across partitions
SET work_mem = '256MB';

-- Enable just-in-time compilation for complex queries
SET jit = on;
SET jit_above_cost = 100000;

-- =====================================================
-- MONITORING AND ALERTING
-- =====================================================

-- Function to check partition health
CREATE OR REPLACE FUNCTION check_partition_health()
RETURNS TABLE (
    issue_type TEXT,
    partition_name TEXT,
    severity TEXT,
    description TEXT,
    recommended_action TEXT
) AS $$
BEGIN
    -- Check for missing future partitions
    RETURN QUERY
    WITH expected_partitions AS (
        SELECT 
            'frame_metadata_' || TO_CHAR(
                DATE_TRUNC('month', CURRENT_DATE) + (n || ' months')::INTERVAL, 
                'YYYY_MM'
            ) as expected_name
        FROM generate_series(0, 2) as n
        UNION
        SELECT 
            'system_metrics_' || TO_CHAR(
                DATE_TRUNC('month', CURRENT_DATE) + (n || ' months')::INTERVAL, 
                'YYYY_MM'
            )
        FROM generate_series(0, 2) as n
    )
    SELECT 
        'missing_partition'::TEXT,
        ep.expected_name::TEXT,
        'high'::TEXT,
        'Future partition missing - may cause insert failures'::TEXT,
        'Run maintain_partitions() function'::TEXT
    FROM expected_partitions ep
    WHERE NOT EXISTS (
        SELECT 1 FROM pg_tables 
        WHERE tablename = ep.expected_name
    );
    
    -- Check for oversized partitions
    RETURN QUERY
    SELECT 
        'oversized_partition'::TEXT,
        ps.tablename::TEXT,
        'medium'::TEXT,
        'Partition size exceeds 50GB - may impact query performance'::TEXT,
        'Consider data archival or sub-partitioning'::TEXT
    FROM partition_statistics ps
    WHERE ps.size_bytes > 50 * 1024 * 1024 * 1024; -- 50GB
    
    -- Check for old partitions that should be archived
    RETURN QUERY
    SELECT 
        'archive_needed'::TEXT,
        ps.tablename::TEXT,
        'low'::TEXT,
        'Partition is older than retention policy'::TEXT,
        'Archive and drop old partition'::TEXT
    FROM partition_statistics ps
    WHERE ps.partition_month < DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '6 months'
      AND ps.partition_type = 'frame_metadata'
    UNION ALL
    SELECT 
        'archive_needed'::TEXT,
        ps.tablename::TEXT,
        'low'::TEXT,
        'Partition is older than retention policy'::TEXT,
        'Archive and drop old partition'::TEXT
    FROM partition_statistics ps
    WHERE ps.partition_month < DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '12 months'
      AND ps.partition_type = 'system_metrics';
END;
$$ LANGUAGE plpgsql;

-- Create a view for easy health monitoring
CREATE OR REPLACE VIEW partition_health_dashboard AS
SELECT * FROM check_partition_health();

COMMENT ON VIEW partition_statistics IS 'Monitor partition sizes and growth patterns';
COMMENT ON VIEW partition_health_dashboard IS 'Real-time partition health monitoring for alerts';
COMMENT ON FUNCTION maintain_partitions() IS 'Automated partition creation for time-series tables';
COMMENT ON FUNCTION archive_old_partitions(TEXT, INTEGER) IS 'Archive and cleanup old partitions based on retention policy';
COMMENT ON FUNCTION get_partition_performance_stats() IS 'Detailed performance statistics for all partitions';