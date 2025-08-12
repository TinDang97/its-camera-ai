-- Database migration strategy for ITS Camera AI
-- Zero-downtime migration approach for high-availability systems

-- =====================================================
-- MIGRATION FRAMEWORK
-- =====================================================

-- Create migration tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    version VARCHAR(20) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    applied_by VARCHAR(100) NOT NULL DEFAULT CURRENT_USER,
    execution_time_ms INTEGER,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    error_message TEXT,
    rollback_sql TEXT
);

CREATE INDEX IF NOT EXISTS idx_schema_migrations_version ON schema_migrations (version);
CREATE INDEX IF NOT EXISTS idx_schema_migrations_applied_at ON schema_migrations (applied_at);

-- Migration execution function
CREATE OR REPLACE FUNCTION execute_migration(
    migration_version VARCHAR(20),
    migration_description TEXT,
    migration_sql TEXT,
    rollback_sql TEXT DEFAULT NULL
) RETURNS BOOLEAN AS $$
DECLARE
    start_time TIMESTAMP;
    execution_time INTEGER;
    migration_success BOOLEAN := TRUE;
    error_msg TEXT;
BEGIN
    -- Check if migration already applied
    IF EXISTS (SELECT 1 FROM schema_migrations WHERE version = migration_version AND success = TRUE) THEN
        RAISE NOTICE 'Migration % already applied', migration_version;
        RETURN TRUE;
    END IF;
    
    start_time := clock_timestamp();
    
    BEGIN
        -- Execute the migration
        EXECUTE migration_sql;
        
        -- Calculate execution time
        execution_time := EXTRACT(MILLISECONDS FROM clock_timestamp() - start_time);
        
        -- Record successful migration
        INSERT INTO schema_migrations (version, description, execution_time_ms, success, rollback_sql)
        VALUES (migration_version, migration_description, execution_time, TRUE, rollback_sql);
        
        RAISE NOTICE 'Migration % completed successfully in % ms', migration_version, execution_time;
        
    EXCEPTION WHEN OTHERS THEN
        migration_success := FALSE;
        error_msg := SQLERRM;
        execution_time := EXTRACT(MILLISECONDS FROM clock_timestamp() - start_time);
        
        -- Record failed migration
        INSERT INTO schema_migrations (version, description, execution_time_ms, success, error_message, rollback_sql)
        VALUES (migration_version, migration_description, execution_time, FALSE, error_msg, rollback_sql);
        
        RAISE EXCEPTION 'Migration % failed: %', migration_version, error_msg;
    END;
    
    RETURN migration_success;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- ZERO-DOWNTIME MIGRATION PATTERNS
-- =====================================================

-- 1. Add column migration (safe)
CREATE OR REPLACE FUNCTION add_column_safe(
    table_name TEXT,
    column_name TEXT,
    column_definition TEXT,
    default_value TEXT DEFAULT NULL
) RETURNS VOID AS $$
DECLARE
    sql_stmt TEXT;
BEGIN
    -- Add column with default if specified
    sql_stmt := FORMAT('ALTER TABLE %I ADD COLUMN IF NOT EXISTS %I %s', 
                       table_name, column_name, column_definition);
    
    IF default_value IS NOT NULL THEN
        sql_stmt := sql_stmt || ' DEFAULT ' || default_value;
    END IF;
    
    EXECUTE sql_stmt;
    
    -- Backfill existing data if needed (in batches for large tables)
    IF default_value IS NOT NULL AND table_name IN ('frame_metadata', 'detections') THEN
        PERFORM backfill_column_batches(table_name, column_name, default_value);
    END IF;
    
    RAISE NOTICE 'Added column %.% safely', table_name, column_name;
END;
$$ LANGUAGE plpgsql;

-- 2. Drop column migration (requires careful planning)
CREATE OR REPLACE FUNCTION drop_column_safe(
    table_name TEXT,
    column_name TEXT,
    verification_query TEXT DEFAULT NULL
) RETURNS VOID AS $$
DECLARE
    sql_stmt TEXT;
    verification_result BOOLEAN := TRUE;
BEGIN
    -- Optional verification that column is not in use
    IF verification_query IS NOT NULL THEN
        EXECUTE verification_query INTO verification_result;
        IF NOT verification_result THEN
            RAISE EXCEPTION 'Column %.% still in use, cannot drop safely', table_name, column_name;
        END IF;
    END IF;
    
    -- Drop column
    sql_stmt := FORMAT('ALTER TABLE %I DROP COLUMN IF EXISTS %I', table_name, column_name);
    EXECUTE sql_stmt;
    
    RAISE NOTICE 'Dropped column %.% safely', table_name, column_name;
END;
$$ LANGUAGE plpgsql;

-- 3. Rename column with backward compatibility
CREATE OR REPLACE FUNCTION rename_column_with_compatibility(
    table_name TEXT,
    old_column TEXT,
    new_column TEXT,
    column_type TEXT
) RETURNS VOID AS $$
DECLARE
    sql_stmt TEXT;
BEGIN
    -- Step 1: Add new column
    PERFORM add_column_safe(table_name, new_column, column_type);
    
    -- Step 2: Copy data from old to new column
    sql_stmt := FORMAT('UPDATE %I SET %I = %I WHERE %I IS NOT NULL', 
                       table_name, new_column, old_column, old_column);
    EXECUTE sql_stmt;
    
    -- Step 3: Create trigger to keep columns in sync during transition period
    sql_stmt := FORMAT('
        CREATE OR REPLACE FUNCTION sync_%s_%s_columns()
        RETURNS TRIGGER AS $trigger$
        BEGIN
            IF TG_OP = ''INSERT'' OR TG_OP = ''UPDATE'' THEN
                IF NEW.%I IS NOT NULL THEN
                    NEW.%I = NEW.%I;
                ELSIF NEW.%I IS NOT NULL THEN
                    NEW.%I = NEW.%I;
                END IF;
                RETURN NEW;
            END IF;
            RETURN NULL;
        END;
        $trigger$ LANGUAGE plpgsql;
        
        DROP TRIGGER IF EXISTS sync_%s_%s_trigger ON %I;
        CREATE TRIGGER sync_%s_%s_trigger
            BEFORE INSERT OR UPDATE ON %I
            FOR EACH ROW EXECUTE FUNCTION sync_%s_%s_columns();
    ', table_name, new_column, new_column, old_column, new_column, old_column, old_column, new_column,
       table_name, new_column, table_name, new_column, table_name, new_column, table_name, new_column);
    
    EXECUTE sql_stmt;
    
    RAISE NOTICE 'Created compatibility sync between %.% and %.%', table_name, old_column, table_name, new_column;
END;
$$ LANGUAGE plpgsql;

-- 4. Backfill data in batches for large tables
CREATE OR REPLACE FUNCTION backfill_column_batches(
    table_name TEXT,
    column_name TEXT,
    new_value TEXT,
    batch_size INTEGER DEFAULT 10000
) RETURNS INTEGER AS $$
DECLARE
    rows_updated INTEGER := 0;
    total_updated INTEGER := 0;
    sql_stmt TEXT;
BEGIN
    LOOP
        sql_stmt := FORMAT('
            UPDATE %I SET %I = %s 
            WHERE %I IS NULL 
            AND id IN (
                SELECT id FROM %I 
                WHERE %I IS NULL 
                LIMIT %s
            )', 
            table_name, column_name, new_value,
            column_name, table_name, column_name, batch_size
        );
        
        EXECUTE sql_stmt;
        GET DIAGNOSTICS rows_updated = ROW_COUNT;
        
        total_updated := total_updated + rows_updated;
        
        -- Exit when no more rows to update
        EXIT WHEN rows_updated = 0;
        
        -- Small delay to avoid overwhelming the database
        PERFORM pg_sleep(0.1);
        
        RAISE NOTICE 'Backfilled % rows, total: %', rows_updated, total_updated;
    END LOOP;
    
    RETURN total_updated;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- INDEX MIGRATION FUNCTIONS
-- =====================================================

-- Create index concurrently (non-blocking)
CREATE OR REPLACE FUNCTION create_index_safe(
    index_name TEXT,
    table_name TEXT,
    index_definition TEXT,
    is_unique BOOLEAN DEFAULT FALSE
) RETURNS VOID AS $$
DECLARE
    sql_stmt TEXT;
BEGIN
    sql_stmt := 'CREATE';
    
    IF is_unique THEN
        sql_stmt := sql_stmt || ' UNIQUE';
    END IF;
    
    sql_stmt := sql_stmt || FORMAT(' INDEX CONCURRENTLY IF NOT EXISTS %I ON %I %s', 
                                   index_name, table_name, index_definition);
    
    EXECUTE sql_stmt;
    
    RAISE NOTICE 'Created index % on % concurrently', index_name, table_name;
END;
$$ LANGUAGE plpgsql;

-- Drop index safely
CREATE OR REPLACE FUNCTION drop_index_safe(
    index_name TEXT,
    verify_unused BOOLEAN DEFAULT TRUE
) RETURNS VOID AS $$
DECLARE
    usage_count INTEGER;
BEGIN
    -- Optionally verify index is not heavily used
    IF verify_unused THEN
        SELECT COALESCE(idx_scan, 0) INTO usage_count
        FROM pg_stat_user_indexes
        WHERE indexrelname = index_name;
        
        IF usage_count > 1000 THEN
            RAISE WARNING 'Index % has % scans, consider if drop is safe', index_name, usage_count;
        END IF;
    END IF;
    
    EXECUTE FORMAT('DROP INDEX CONCURRENTLY IF EXISTS %I', index_name);
    
    RAISE NOTICE 'Dropped index % safely', index_name;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- DATA ARCHIVAL AND CLEANUP POLICIES
-- =====================================================

-- Comprehensive archival strategy
CREATE OR REPLACE FUNCTION archive_data_policy(
    policy_name TEXT,
    table_name TEXT,
    retention_period INTERVAL,
    archive_method TEXT DEFAULT 'delete', -- 'delete', 'export', 'compress'
    batch_size INTEGER DEFAULT 10000
) RETURNS INTEGER AS $$
DECLARE
    cutoff_date TIMESTAMP;
    total_archived INTEGER := 0;
    batch_count INTEGER;
    archive_table_name TEXT;
    sql_stmt TEXT;
BEGIN
    cutoff_date := CURRENT_TIMESTAMP - retention_period;
    archive_table_name := table_name || '_archive';
    
    RAISE NOTICE 'Starting archival for % with cutoff %', table_name, cutoff_date;
    
    -- Different archival methods
    CASE archive_method
        WHEN 'export' THEN
            -- Export to archive table
            sql_stmt := FORMAT('
                CREATE TABLE IF NOT EXISTS %I (LIKE %I INCLUDING ALL);
                
                WITH archived_data AS (
                    DELETE FROM %I 
                    WHERE created_at < %L
                    RETURNING *
                )
                INSERT INTO %I SELECT * FROM archived_data;
            ', archive_table_name, table_name, table_name, cutoff_date, archive_table_name);
            
            EXECUTE sql_stmt;
            GET DIAGNOSTICS total_archived = ROW_COUNT;
            
        WHEN 'compress' THEN
            -- Compress old partitions (requires pg_squeeze or similar)
            RAISE NOTICE 'Compression archival not implemented yet';
            
        WHEN 'delete' THEN
            -- Batch delete for large tables
            LOOP
                sql_stmt := FORMAT('
                    DELETE FROM %I 
                    WHERE created_at < %L
                    AND id IN (
                        SELECT id FROM %I 
                        WHERE created_at < %L 
                        LIMIT %s
                    )', 
                    table_name, cutoff_date, table_name, cutoff_date, batch_size
                );
                
                EXECUTE sql_stmt;
                GET DIAGNOSTICS batch_count = ROW_COUNT;
                
                total_archived := total_archived + batch_count;
                
                EXIT WHEN batch_count = 0;
                
                -- Brief pause between batches
                PERFORM pg_sleep(0.1);
                
                IF total_archived % 50000 = 0 THEN
                    RAISE NOTICE 'Archived % rows so far', total_archived;
                END IF;
            END LOOP;
    END CASE;
    
    -- Log archival activity
    INSERT INTO system_metrics (metric_name, metric_type, value, source, labels)
    VALUES (
        'data_archived',
        'counter',
        total_archived,
        'archival_system',
        jsonb_build_object(
            'policy_name', policy_name,
            'table_name', table_name,
            'method', archive_method,
            'cutoff_date', cutoff_date,
            'retention_period', retention_period::TEXT
        )
    );
    
    RAISE NOTICE 'Archived % rows from % using % method', total_archived, table_name, archive_method;
    
    RETURN total_archived;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- SPECIFIC MIGRATION SCRIPTS
-- =====================================================

-- Migration 001: Initial schema setup
SELECT execute_migration(
    '001_initial_schema',
    'Create initial camera AI database schema',
    'SELECT 1; -- Schema already created in schema_setup.sql',
    'DROP TABLE IF EXISTS alerts, user_sessions, users, system_metrics, detections, frame_metadata, cameras CASCADE;'
);

-- Migration 002: Add performance indexes
DO $$
BEGIN
    PERFORM execute_migration(
        '002_performance_indexes',
        'Add high-performance indexes for camera operations',
        '
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_cameras_location_trgm ON cameras USING GIN (location gin_trgm_ops);
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_frame_metadata_camera_hour ON frame_metadata (camera_id, DATE_TRUNC(''hour'', timestamp));
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_detections_type_time ON detections (object_type, (SELECT timestamp FROM frame_metadata WHERE id = frame_metadata_id));
        ',
        '
        DROP INDEX IF EXISTS idx_cameras_location_trgm;
        DROP INDEX IF EXISTS idx_frame_metadata_camera_hour;
        DROP INDEX IF EXISTS idx_detections_type_time;
        '
    );
END $$;

-- Migration 003: Add partitioning constraints
DO $$
BEGIN
    PERFORM execute_migration(
        '003_partition_constraints',
        'Add constraints to support partitioning',
        '
        -- Add check constraints for partitioning
        ALTER TABLE frame_metadata ADD CONSTRAINT frame_metadata_timestamp_check 
            CHECK (timestamp >= ''2024-01-01''::timestamptz);
        
        ALTER TABLE system_metrics ADD CONSTRAINT system_metrics_timestamp_check 
            CHECK (timestamp >= ''2024-01-01''::timestamptz);
        ',
        '
        ALTER TABLE frame_metadata DROP CONSTRAINT IF EXISTS frame_metadata_timestamp_check;
        ALTER TABLE system_metrics DROP CONSTRAINT IF EXISTS system_metrics_timestamp_check;
        '
    );
END $$;

-- =====================================================
-- ARCHIVAL POLICIES
-- =====================================================

-- Define retention policies for different data types
CREATE OR REPLACE FUNCTION setup_archival_policies()
RETURNS VOID AS $$
BEGIN
    -- Frame metadata: 6 months retention
    PERFORM archive_data_policy(
        'frame_metadata_retention',
        'frame_metadata',
        INTERVAL '6 months',
        'export'
    );
    
    -- System metrics: 12 months retention
    PERFORM archive_data_policy(
        'system_metrics_retention',
        'system_metrics', 
        INTERVAL '12 months',
        'export'
    );
    
    -- Detection data: 3 months retention
    PERFORM archive_data_policy(
        'detections_retention',
        'detections',
        INTERVAL '3 months',
        'delete'  -- Detections can be recreated from archived frame data
    );
    
    -- User sessions: 30 days retention
    PERFORM archive_data_policy(
        'user_sessions_cleanup',
        'user_sessions',
        INTERVAL '30 days',
        'delete'
    );
    
    -- Alerts: 90 days retention for resolved alerts
    EXECUTE '
        DELETE FROM alerts 
        WHERE status = ''resolved'' 
        AND resolved_at < CURRENT_TIMESTAMP - INTERVAL ''90 days''
    ';
    
    RAISE NOTICE 'All archival policies executed';
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- MIGRATION ROLLBACK FUNCTIONS
-- =====================================================

-- Rollback specific migration
CREATE OR REPLACE FUNCTION rollback_migration(migration_version VARCHAR(20))
RETURNS BOOLEAN AS $$
DECLARE
    rollback_sql_cmd TEXT;
    migration_record RECORD;
BEGIN
    -- Get migration details
    SELECT * INTO migration_record
    FROM schema_migrations 
    WHERE version = migration_version 
    AND success = TRUE
    ORDER BY applied_at DESC 
    LIMIT 1;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Migration % not found or not successful', migration_version;
    END IF;
    
    IF migration_record.rollback_sql IS NULL THEN
        RAISE EXCEPTION 'No rollback script available for migration %', migration_version;
    END IF;
    
    -- Execute rollback
    BEGIN
        EXECUTE migration_record.rollback_sql;
        
        -- Mark as rolled back
        UPDATE schema_migrations 
        SET success = FALSE, 
            error_message = 'Rolled back by user',
            applied_at = CURRENT_TIMESTAMP
        WHERE version = migration_version;
        
        RAISE NOTICE 'Successfully rolled back migration %', migration_version;
        RETURN TRUE;
        
    EXCEPTION WHEN OTHERS THEN
        RAISE EXCEPTION 'Rollback failed for migration %: %', migration_version, SQLERRM;
    END;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- MONITORING AND VALIDATION
-- =====================================================

-- Validate database schema integrity
CREATE OR REPLACE FUNCTION validate_schema_integrity()
RETURNS TABLE (
    check_type TEXT,
    status TEXT,
    message TEXT
) AS $$
BEGIN
    -- Check required tables exist
    RETURN QUERY
    SELECT 
        'table_exists'::TEXT,
        CASE WHEN COUNT(*) = 7 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        'Required tables: ' || COUNT(*)::TEXT || '/7'
    FROM information_schema.tables 
    WHERE table_name IN ('cameras', 'frame_metadata', 'detections', 'system_metrics', 'users', 'user_sessions', 'alerts');
    
    -- Check critical indexes exist
    RETURN QUERY
    SELECT 
        'critical_indexes'::TEXT,
        CASE WHEN COUNT(*) >= 20 THEN 'PASS' ELSE 'WARN' END::TEXT,
        'Critical indexes found: ' || COUNT(*)::TEXT
    FROM pg_indexes 
    WHERE indexname LIKE 'idx_%'
    AND tablename IN ('cameras', 'frame_metadata', 'detections');
    
    -- Check partition health
    RETURN QUERY
    SELECT 
        'partition_health'::TEXT,
        CASE WHEN COUNT(*) >= 2 THEN 'PASS' ELSE 'FAIL' END::TEXT,
        'Active partitions: ' || COUNT(*)::TEXT
    FROM pg_tables 
    WHERE tablename LIKE '%_2024_%' OR tablename LIKE '%_2025_%';
    
    -- Check foreign key constraints
    RETURN QUERY
    SELECT 
        'foreign_keys'::TEXT,
        CASE WHEN COUNT(*) >= 5 THEN 'PASS' ELSE 'WARN' END::TEXT,
        'Foreign key constraints: ' || COUNT(*)::TEXT
    FROM information_schema.table_constraints 
    WHERE constraint_type = 'FOREIGN KEY';
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION execute_migration(VARCHAR, TEXT, TEXT, TEXT) IS 'Execute database migration with rollback support';
COMMENT ON FUNCTION archive_data_policy(TEXT, TEXT, INTERVAL, TEXT, INTEGER) IS 'Automated data archival based on retention policies';
COMMENT ON FUNCTION validate_schema_integrity() IS 'Validate database schema health and integrity';