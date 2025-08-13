"""Migration script for Analytics Service database tables.

This migration creates TimescaleDB-optimized analytics tables with hypertables,
indexes, and retention policies for high-performance traffic analytics.

Usage:
    python -m migrations.analytics_migration
"""

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.its_camera_ai.core.config import get_settings
from src.its_camera_ai.core.logging import get_logger

logger = get_logger(__name__)


class AnalyticsMigration:
    """Analytics database migration manager."""

    def __init__(self):
        self.settings = get_settings()
        self.engine = None

    async def __aenter__(self):
        """Initialize database engine."""
        self.engine = create_async_engine(
            self.settings.database.url,
            echo=self.settings.database.echo
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup database engine."""
        if self.engine:
            await self.engine.dispose()

    async def run_migration(self):
        """Run the complete analytics migration."""
        logger.info("Starting analytics database migration")

        try:
            # Step 1: Enable TimescaleDB extension
            await self._enable_timescaledb()

            # Step 2: Create analytics tables (if they don't exist)
            await self._create_analytics_tables()

            # Step 3: Create hypertables for time-series data
            await self._create_hypertables()

            # Step 4: Create optimized indexes
            await self._create_analytics_indexes()

            # Step 5: Set up retention policies
            await self._create_retention_policies()

            # Step 6: Create continuous aggregates
            await self._create_continuous_aggregates()

            # Step 7: Set up compression policies
            await self._create_compression_policies()

            logger.info("Analytics migration completed successfully")

        except Exception as e:
            logger.error(f"Analytics migration failed: {e}")
            raise

    async def _enable_timescaledb(self):
        """Enable TimescaleDB extension."""
        logger.info("Enabling TimescaleDB extension")

        async with self.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))

            # Also create supporting extensions
            extensions = [
                "CREATE EXTENSION IF NOT EXISTS postgis",      # For spatial queries
                "CREATE EXTENSION IF NOT EXISTS pg_stat_statements",  # For query analysis
                "CREATE EXTENSION IF NOT EXISTS pg_trgm",     # For text search
            ]

            for ext_sql in extensions:
                try:
                    await conn.execute(text(ext_sql))
                    logger.debug(f"Created extension: {ext_sql}")
                except Exception as e:
                    logger.debug(f"Extension creation info: {e}")

    async def _create_analytics_tables(self):
        """Create analytics tables using SQLAlchemy models."""
        logger.info("Creating analytics tables")

        # Import models to trigger table creation
        from src.its_camera_ai.models.base import BaseModel

        async with self.engine.begin() as conn:
            # Create all analytics tables
            await conn.run_sync(BaseModel.metadata.create_all)

    async def _create_hypertables(self):
        """Create TimescaleDB hypertables for time-series data."""
        logger.info("Creating hypertables")

        hypertables = [
            {
                "table": "traffic_metrics",
                "time_column": "timestamp",
                "chunk_interval": "1 hour",
                "partitioning_column": "camera_id"
            },
            {
                "table": "rule_violations",
                "time_column": "detection_time",
                "chunk_interval": "1 day",
                "partitioning_column": "camera_id"
            },
            {
                "table": "traffic_anomalies",
                "time_column": "detection_time",
                "chunk_interval": "1 day",
                "partitioning_column": "camera_id"
            },
            {
                "table": "vehicle_trajectories",
                "time_column": "start_time",
                "chunk_interval": "1 day",
                "partitioning_column": "camera_id"
            },
            {
                "table": "alert_notifications",
                "time_column": "created_time",
                "chunk_interval": "1 day",
                "partitioning_column": None
            }
        ]

        async with self.engine.begin() as conn:
            for ht in hypertables:
                try:
                    # Create basic hypertable
                    sql = f"""
                        SELECT create_hypertable(
                            '{ht["table"]}',
                            '{ht["time_column"]}',
                            chunk_time_interval => INTERVAL '{ht["chunk_interval"]}',
                            if_not_exists => TRUE
                        );
                    """
                    await conn.execute(text(sql))
                    logger.info(f"Created hypertable: {ht['table']}")

                    # Add space partitioning if specified
                    if ht.get("partitioning_column"):
                        try:
                            partition_sql = f"""
                                SELECT add_dimension(
                                    '{ht["table"]}',
                                    '{ht["partitioning_column"]}',
                                    number_partitions => 4,
                                    if_not_exists => TRUE
                                );
                            """
                            await conn.execute(text(partition_sql))
                            logger.info(f"Added space partitioning to {ht['table']}")
                        except Exception as e:
                            logger.debug(f"Space partitioning info for {ht['table']}: {e}")

                except Exception as e:
                    logger.debug(f"Hypertable creation info for {ht['table']}: {e}")

    async def _create_analytics_indexes(self):
        """Create additional performance indexes for analytics queries."""
        logger.info("Creating analytics indexes")

        indexes = [
            # Traffic metrics indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_traffic_metrics_camera_hour ON traffic_metrics (camera_id, date_trunc('hour', timestamp))",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_traffic_metrics_congestion_time ON traffic_metrics (congestion_level, timestamp) WHERE congestion_level IN ('heavy', 'severe')",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_traffic_metrics_speed_outliers ON traffic_metrics (average_speed, timestamp) WHERE average_speed > 100 OR average_speed < 5",

            # Rule violations indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_violations_severity_camera ON rule_violations (severity, camera_id, detection_time)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_violations_plate_type ON rule_violations (license_plate, violation_type) WHERE license_plate IS NOT NULL",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_violations_active_recent ON rule_violations (status, detection_time) WHERE status = 'active'",

            # Traffic anomalies indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_anomalies_score_camera ON traffic_anomalies (anomaly_score DESC, camera_id, detection_time)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_anomalies_type_severity ON traffic_anomalies (anomaly_type, severity, detection_time)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_anomalies_unvalidated ON traffic_anomalies (human_validated, anomaly_score) WHERE human_validated = false",

            # Vehicle trajectories indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trajectories_track_camera ON vehicle_trajectories (vehicle_track_id, camera_id, start_time)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trajectories_anomalous ON vehicle_trajectories (is_anomalous, anomaly_score) WHERE is_anomalous = true",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trajectories_duration ON vehicle_trajectories (duration_seconds, start_time) WHERE duration_seconds > 300",

            # Alert notifications indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_pending_priority ON alert_notifications (status, priority, created_time) WHERE status IN ('pending', 'failed')",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_recipient_channel ON alert_notifications (recipient, notification_channel, created_time)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_delivery_stats ON alert_notifications (notification_channel, status, sent_time)",
        ]

        async with self.engine.begin() as conn:
            for idx_sql in indexes:
                try:
                    await conn.execute(text(idx_sql))
                    logger.debug(f"Created index: {idx_sql.split()[4]}")
                except Exception as e:
                    logger.debug(f"Index creation info: {e}")

    async def _create_retention_policies(self):
        """Create data retention policies for automatic cleanup."""
        logger.info("Creating retention policies")

        retention_policies = [
            {
                "table": "traffic_metrics",
                "retention": "90 days",
                "description": "Keep detailed metrics for 90 days"
            },
            {
                "table": "rule_violations",
                "retention": "365 days",
                "description": "Keep violations for 1 year for compliance"
            },
            {
                "table": "traffic_anomalies",
                "retention": "180 days",
                "description": "Keep anomalies for 6 months for analysis"
            },
            {
                "table": "vehicle_trajectories",
                "retention": "30 days",
                "description": "Keep trajectories for 30 days (privacy)"
            },
            {
                "table": "alert_notifications",
                "retention": "90 days",
                "description": "Keep alert history for 90 days"
            }
        ]

        async with self.engine.begin() as conn:
            for policy in retention_policies:
                try:
                    sql = f"""
                        SELECT add_retention_policy(
                            '{policy["table"]}',
                            INTERVAL '{policy["retention"]}',
                            if_not_exists => TRUE
                        );
                    """
                    await conn.execute(text(sql))
                    logger.info(f"Set retention policy for {policy['table']}: {policy['retention']}")
                except Exception as e:
                    logger.debug(f"Retention policy info for {policy['table']}: {e}")

    async def _create_continuous_aggregates(self):
        """Create continuous aggregates for common analytics queries."""
        logger.info("Creating continuous aggregates")

        # Hourly traffic metrics aggregate
        hourly_metrics_sql = """
            CREATE MATERIALIZED VIEW IF NOT EXISTS traffic_metrics_hourly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 hour', timestamp) AS hour,
                camera_id,
                AVG(total_vehicles) as avg_vehicles,
                MAX(total_vehicles) as peak_vehicles,
                AVG(average_speed) as avg_speed,
                MIN(average_speed) as min_speed,
                MAX(average_speed) as max_speed,
                AVG(traffic_density) as avg_density,
                MAX(traffic_density) as peak_density,
                COUNT(*) as sample_count
            FROM traffic_metrics
            GROUP BY hour, camera_id;
        """

        # Daily violations summary
        daily_violations_sql = """
            CREATE MATERIALIZED VIEW IF NOT EXISTS violations_daily
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 day', detection_time) AS day,
                camera_id,
                violation_type,
                severity,
                COUNT(*) as violation_count,
                AVG(detection_confidence) as avg_confidence
            FROM rule_violations
            WHERE status = 'active'
            GROUP BY day, camera_id, violation_type, severity;
        """

        # Weekly anomaly trends
        weekly_anomalies_sql = """
            CREATE MATERIALIZED VIEW IF NOT EXISTS anomalies_weekly
            WITH (timescaledb.continuous) AS
            SELECT
                time_bucket('1 week', detection_time) AS week,
                camera_id,
                anomaly_type,
                COUNT(*) as anomaly_count,
                AVG(anomaly_score) as avg_score,
                MAX(anomaly_score) as max_score
            FROM traffic_anomalies
            GROUP BY week, camera_id, anomaly_type;
        """

        aggregates = [
            ("traffic_metrics_hourly", hourly_metrics_sql),
            ("violations_daily", daily_violations_sql),
            ("anomalies_weekly", weekly_anomalies_sql)
        ]

        async with self.engine.begin() as conn:
            for name, sql in aggregates:
                try:
                    await conn.execute(text(sql))
                    logger.info(f"Created continuous aggregate: {name}")

                    # Add refresh policy
                    refresh_sql = f"""
                        SELECT add_continuous_aggregate_policy(
                            '{name}',
                            start_offset => INTERVAL '2 days',
                            end_offset => INTERVAL '1 hour',
                            schedule_interval => INTERVAL '1 hour',
                            if_not_exists => TRUE
                        );
                    """
                    await conn.execute(text(refresh_sql))
                    logger.info(f"Added refresh policy for {name}")

                except Exception as e:
                    logger.debug(f"Continuous aggregate info for {name}: {e}")

    async def _create_compression_policies(self):
        """Create compression policies for data storage optimization."""
        logger.info("Creating compression policies")

        compression_policies = [
            {
                "table": "traffic_metrics",
                "compress_after": "7 days",
                "description": "Compress metrics older than 7 days"
            },
            {
                "table": "rule_violations",
                "compress_after": "30 days",
                "description": "Compress violations older than 30 days"
            },
            {
                "table": "traffic_anomalies",
                "compress_after": "14 days",
                "description": "Compress anomalies older than 14 days"
            },
            {
                "table": "vehicle_trajectories",
                "compress_after": "3 days",
                "description": "Compress trajectories older than 3 days"
            }
        ]

        async with self.engine.begin() as conn:
            for policy in compression_policies:
                try:
                    # Enable compression
                    enable_sql = f"ALTER TABLE {policy['table']} SET (timescaledb.compress = true)"
                    await conn.execute(text(enable_sql))

                    # Add compression policy
                    policy_sql = f"""
                        SELECT add_compression_policy(
                            '{policy["table"]}',
                            INTERVAL '{policy["compress_after"]}',
                            if_not_exists => TRUE
                        );
                    """
                    await conn.execute(text(policy_sql))
                    logger.info(f"Set compression policy for {policy['table']}: {policy['compress_after']}")

                except Exception as e:
                    logger.debug(f"Compression policy info for {policy['table']}: {e}")


async def main():
    """Run analytics migration."""
    logging.basicConfig(level=logging.INFO)

    async with AnalyticsMigration() as migration:
        await migration.run_migration()

    print("✅ Analytics migration completed successfully!")
    print("\nNext steps:")
    print("1. Verify hypertables: SELECT * FROM timescaledb_information.hypertables;")
    print("2. Check retention policies: SELECT * FROM timescaledb_information.retention_policies;")
    print("3. Monitor compression: SELECT * FROM timescaledb_information.compression_settings;")
    print("4. View continuous aggregates: SELECT * FROM timescaledb_information.continuous_aggregates;")


if __name__ == "__main__":
    asyncio.run(main())
