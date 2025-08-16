"""Maintenance worker for system cleanup and optimization.

This worker handles routine maintenance tasks including:
- Old data cleanup and archival
- Database optimization and vacuuming
- Cache management and cleanup
- Log rotation and management
- System health monitoring
- Performance optimization

Features:
- Configurable retention policies
- Incremental cleanup to avoid system impact
- Database optimization with minimal downtime
- Comprehensive monitoring and alerting
"""

import asyncio
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from celery import current_task
from sqlalchemy import text

from ..containers import ApplicationContainer
from ..core.exceptions import DatabaseError
from . import celery_app, logger


@celery_app.task(
    bind=True,
    autoretry_for=(DatabaseError,),
    retry_kwargs={"max_retries": 2, "countdown": 1800},
    soft_time_limit=7200,  # 2 hours
    time_limit=10800,  # 3 hours
)
def cleanup_old_data(self, retention_days: int = 90) -> dict[str, Any]:
    """Clean up old data based on retention policies.
    
    Args:
        retention_days: Number of days to retain data
        
    Returns:
        dict: Cleanup results
    """
    return asyncio.run(_cleanup_old_data_async(retention_days))


async def _cleanup_old_data_async(retention_days: int) -> dict[str, Any]:
    """Async implementation of data cleanup."""
    container = ApplicationContainer()
    container.wire(modules=[__name__])

    try:
        await container.init_resources()
        db_manager = await container.infrastructure.database()

        cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)
        cleanup_results = {}

        async with db_manager.get_session() as session:
            # Update task status
            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={"stage": "initializing", "progress": 5}
                )

            # 1. Clean up old detection results
            detection_results = await _cleanup_detection_results(session, cutoff_date)
            cleanup_results["detection_results"] = detection_results

            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={"stage": "detection_cleanup", "progress": 25}
                )

            # 2. Clean up old camera frames
            frame_results = await _cleanup_camera_frames(session, cutoff_date)
            cleanup_results["camera_frames"] = frame_results

            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={"stage": "frame_cleanup", "progress": 45}
                )

            # 3. Archive old traffic metrics (keep aggregated, remove raw)
            metrics_results = await _archive_traffic_metrics(session, cutoff_date)
            cleanup_results["traffic_metrics"] = metrics_results

            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={"stage": "metrics_cleanup", "progress": 65}
                )

            # 4. Clean up resolved incidents and alerts
            incidents_results = await _cleanup_resolved_incidents(session, cutoff_date)
            cleanup_results["incidents"] = incidents_results

            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={"stage": "incidents_cleanup", "progress": 80}
                )

            # 5. Clean up old logs and temporary files
            files_results = await _cleanup_temporary_files(cutoff_date)
            cleanup_results["temporary_files"] = files_results

            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={"stage": "files_cleanup", "progress": 90}
                )

            # 6. Update database statistics
            await _update_database_statistics(session)

            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={"stage": "finalizing", "progress": 95}
                )

        # Calculate total cleanup metrics
        total_deleted = sum(
            result.get("deleted_count", 0) for result in cleanup_results.values()
        )
        total_space_freed = sum(
            result.get("space_freed_mb", 0) for result in cleanup_results.values()
        )

        logger.info(
            f"Data cleanup completed: {total_deleted} records deleted, {total_space_freed:.2f} MB freed",
            retention_days=retention_days,
            cutoff_date=cutoff_date.isoformat(),
        )

        return {
            "status": "completed",
            "retention_days": retention_days,
            "cutoff_date": cutoff_date.isoformat(),
            "total_deleted": total_deleted,
            "total_space_freed_mb": total_space_freed,
            "cleanup_results": cleanup_results,
            "cleanup_time": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        raise DatabaseError(f"Data cleanup failed: {e}") from e
    finally:
        await container.shutdown_resources()


async def _cleanup_detection_results(session, cutoff_date: datetime) -> dict[str, Any]:
    """Clean up old detection results."""
    try:
        # Count records to be deleted
        count_query = text("""
            SELECT COUNT(*) FROM detection_results 
            WHERE created_at < :cutoff_date
        """)
        count_result = await session.execute(count_query, {"cutoff_date": cutoff_date})
        total_count = count_result.scalar()

        if total_count == 0:
            return {"deleted_count": 0, "space_freed_mb": 0}

        # Delete in batches to avoid long locks
        batch_size = 10000
        deleted_count = 0

        while True:
            delete_query = text("""
                DELETE FROM detection_results 
                WHERE id IN (
                    SELECT id FROM detection_results 
                    WHERE created_at < :cutoff_date 
                    LIMIT :batch_size
                )
            """)

            result = await session.execute(
                delete_query,
                {"cutoff_date": cutoff_date, "batch_size": batch_size}
            )

            batch_deleted = result.rowcount
            deleted_count += batch_deleted

            await session.commit()

            if batch_deleted < batch_size:
                break

            # Small delay to reduce system impact
            await asyncio.sleep(1)

        # Estimate space freed (rough calculation)
        space_freed_mb = deleted_count * 0.002  # ~2KB per detection record

        logger.info(f"Cleaned up {deleted_count} detection results")

        return {
            "deleted_count": deleted_count,
            "space_freed_mb": space_freed_mb,
            "table": "detection_results",
        }

    except Exception as e:
        logger.error(f"Failed to cleanup detection results: {e}")
        raise


async def _cleanup_camera_frames(session, cutoff_date: datetime) -> dict[str, Any]:
    """Clean up old camera frames and associated files."""
    try:
        # Get frame records with file paths
        frames_query = text("""
            SELECT id, file_path FROM camera_frames 
            WHERE created_at < :cutoff_date
        """)
        frames_result = await session.execute(frames_query, {"cutoff_date": cutoff_date})
        frames_to_delete = frames_result.fetchall()

        deleted_count = 0
        space_freed_mb = 0

        # Delete files and records in batches
        batch_size = 100

        for i in range(0, len(frames_to_delete), batch_size):
            batch = frames_to_delete[i:i + batch_size]

            # Delete physical files
            for frame in batch:
                if frame.file_path and Path(frame.file_path).exists():
                    try:
                        file_size = Path(frame.file_path).stat().st_size
                        Path(frame.file_path).unlink()
                        space_freed_mb += file_size / (1024 * 1024)
                    except Exception as e:
                        logger.warning(f"Failed to delete file {frame.file_path}: {e}")

            # Delete database records
            frame_ids = [frame.id for frame in batch]
            delete_query = text("""
                DELETE FROM camera_frames WHERE id = ANY(:frame_ids)
            """)

            await session.execute(delete_query, {"frame_ids": frame_ids})
            deleted_count += len(batch)

            await session.commit()
            await asyncio.sleep(0.5)  # Brief pause

        logger.info(f"Cleaned up {deleted_count} camera frames, freed {space_freed_mb:.2f} MB")

        return {
            "deleted_count": deleted_count,
            "space_freed_mb": space_freed_mb,
            "table": "camera_frames",
        }

    except Exception as e:
        logger.error(f"Failed to cleanup camera frames: {e}")
        raise


async def _archive_traffic_metrics(session, cutoff_date: datetime) -> dict[str, Any]:
    """Archive old traffic metrics, keeping only aggregated data."""
    try:
        # Keep hourly/daily aggregates, remove 1-minute data older than cutoff
        archive_query = text("""
            DELETE FROM traffic_metrics 
            WHERE created_at < :cutoff_date 
            AND aggregation_period IN ('1min', '5min')
        """)

        result = await session.execute(archive_query, {"cutoff_date": cutoff_date})
        deleted_count = result.rowcount

        await session.commit()

        # Estimate space freed
        space_freed_mb = deleted_count * 0.001  # ~1KB per metric record

        logger.info(f"Archived {deleted_count} traffic metrics")

        return {
            "deleted_count": deleted_count,
            "space_freed_mb": space_freed_mb,
            "table": "traffic_metrics",
            "action": "archived_fine_grained_data",
        }

    except Exception as e:
        logger.error(f"Failed to archive traffic metrics: {e}")
        raise


async def _cleanup_resolved_incidents(session, cutoff_date: datetime) -> dict[str, Any]:
    """Clean up old resolved incidents and alerts."""
    try:
        # Delete resolved incidents older than cutoff
        incidents_query = text("""
            DELETE FROM incident_alerts 
            WHERE resolved_at < :cutoff_date 
            AND status = 'resolved'
        """)

        result = await session.execute(incidents_query, {"cutoff_date": cutoff_date})
        deleted_count = result.rowcount

        await session.commit()

        # Clean up related alert records
        alerts_query = text("""
            DELETE FROM alert_rules 
            WHERE created_at < :cutoff_date 
            AND is_active = false
        """)

        alerts_result = await session.execute(alerts_query, {"cutoff_date": cutoff_date})
        alerts_deleted = alerts_result.rowcount

        await session.commit()

        total_deleted = deleted_count + alerts_deleted
        space_freed_mb = total_deleted * 0.005  # ~5KB per incident/alert

        logger.info(f"Cleaned up {total_deleted} incidents and alerts")

        return {
            "deleted_count": total_deleted,
            "space_freed_mb": space_freed_mb,
            "incidents_deleted": deleted_count,
            "alerts_deleted": alerts_deleted,
        }

    except Exception as e:
        logger.error(f"Failed to cleanup incidents: {e}")
        raise


async def _cleanup_temporary_files(cutoff_date: datetime) -> dict[str, Any]:
    """Clean up temporary files and logs."""
    try:
        temp_dirs = [
            "/tmp/its-camera-ai",
            "/var/tmp/its-camera-ai",
            "/var/log/its-camera-ai/temp",
        ]

        total_deleted = 0
        space_freed_mb = 0

        for temp_dir in temp_dirs:
            temp_path = Path(temp_dir)
            if not temp_path.exists():
                continue

            # Clean up old temporary files
            for file_path in temp_path.rglob("*"):
                if file_path.is_file():
                    try:
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC)

                        if file_mtime < cutoff_date:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            total_deleted += 1
                            space_freed_mb += file_size / (1024 * 1024)

                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {file_path}: {e}")

        # Clean up old log files (keep last 30 days of logs)
        log_cleanup = await _cleanup_log_files()
        total_deleted += log_cleanup["files_deleted"]
        space_freed_mb += log_cleanup["space_freed_mb"]

        logger.info(f"Cleaned up {total_deleted} temporary files, freed {space_freed_mb:.2f} MB")

        return {
            "deleted_count": total_deleted,
            "space_freed_mb": space_freed_mb,
            "type": "temporary_files_and_logs",
        }

    except Exception as e:
        logger.error(f"Failed to cleanup temporary files: {e}")
        return {"deleted_count": 0, "space_freed_mb": 0}


async def _cleanup_log_files() -> dict[str, Any]:
    """Clean up old log files."""
    log_dirs = [
        "/var/log/its-camera-ai",
        "/var/log/celery",
        "/tmp/its-camera-ai/logs",
    ]

    files_deleted = 0
    space_freed_mb = 0
    log_retention_days = 30
    cutoff_date = datetime.now(UTC) - timedelta(days=log_retention_days)

    for log_dir in log_dirs:
        log_path = Path(log_dir)
        if not log_path.exists():
            continue

        for log_file in log_path.glob("*.log*"):
            try:
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=UTC)

                if file_mtime < cutoff_date:
                    file_size = log_file.stat().st_size
                    log_file.unlink()
                    files_deleted += 1
                    space_freed_mb += file_size / (1024 * 1024)

            except Exception as e:
                logger.warning(f"Failed to delete log file {log_file}: {e}")

    return {"files_deleted": files_deleted, "space_freed_mb": space_freed_mb}


async def _update_database_statistics(session) -> None:
    """Update database statistics after cleanup."""
    try:
        # Update table statistics for PostgreSQL query planner
        analyze_query = text("""
            ANALYZE detection_results, camera_frames, traffic_metrics, incident_alerts;
        """)

        await session.execute(analyze_query)
        await session.commit()

        logger.info("Updated database statistics")

    except Exception as e:
        logger.warning(f"Failed to update database statistics: {e}")


@celery_app.task(
    bind=True,
    autoretry_for=(DatabaseError,),
    retry_kwargs={"max_retries": 1, "countdown": 3600},
    soft_time_limit=3600,  # 1 hour
    time_limit=7200,  # 2 hours
)
def optimize_database(self) -> dict[str, Any]:
    """Optimize database performance through vacuuming and reindexing.
    
    Returns:
        dict: Optimization results
    """
    return asyncio.run(_optimize_database_async())


async def _optimize_database_async() -> dict[str, Any]:
    """Async implementation of database optimization."""
    container = ApplicationContainer()
    container.wire(modules=[__name__])

    try:
        await container.init_resources()
        db_manager = await container.infrastructure.database()

        optimization_results = {}

        async with db_manager.get_session() as session:
            # Update task status
            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={"stage": "vacuum_analysis", "progress": 10}
                )

            # 1. Vacuum analyze main tables
            vacuum_results = await _vacuum_analyze_tables(session)
            optimization_results["vacuum"] = vacuum_results

            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={"stage": "reindex", "progress": 40}
                )

            # 2. Reindex fragmented indexes
            reindex_results = await _reindex_fragmented_indexes(session)
            optimization_results["reindex"] = reindex_results

            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={"stage": "timescaledb_optimization", "progress": 70}
                )

            # 3. TimescaleDB specific optimizations
            timescale_results = await _optimize_timescaledb(session)
            optimization_results["timescaledb"] = timescale_results

            if current_task:
                current_task.update_state(
                    state="PROGRESS",
                    meta={"stage": "statistics_update", "progress": 90}
                )

            # 4. Update query planner statistics
            await _update_query_statistics(session)
            optimization_results["statistics"] = {"updated": True}

        logger.info("Database optimization completed successfully")

        return {
            "status": "completed",
            "optimization_results": optimization_results,
            "optimization_time": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Database optimization failed: {e}")
        raise DatabaseError(f"Database optimization failed: {e}") from e
    finally:
        await container.shutdown_resources()


async def _vacuum_analyze_tables(session) -> dict[str, Any]:
    """Vacuum and analyze main tables."""
    tables_to_vacuum = [
        "detection_results",
        "camera_frames",
        "traffic_metrics",
        "incident_alerts",
        "alert_rules",
        "users",
        "cameras",
    ]

    vacuum_results = {"tables_processed": 0, "errors": 0}

    for table in tables_to_vacuum:
        try:
            # VACUUM ANALYZE for each table
            vacuum_query = text(f"VACUUM ANALYZE {table};")
            await session.execute(vacuum_query)

            vacuum_results["tables_processed"] += 1
            logger.debug(f"Vacuumed and analyzed table: {table}")

        except Exception as e:
            logger.warning(f"Failed to vacuum table {table}: {e}")
            vacuum_results["errors"] += 1

    return vacuum_results


async def _reindex_fragmented_indexes(session) -> dict[str, Any]:
    """Reindex fragmented indexes to improve performance."""
    try:
        # Find indexes with high fragmentation
        fragmentation_query = text("""
            SELECT schemaname, tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND tablename IN ('detection_results', 'camera_frames', 'traffic_metrics')
        """)

        result = await session.execute(fragmentation_query)
        indexes = result.fetchall()

        reindex_results = {"indexes_processed": 0, "errors": 0}

        for index in indexes:
            try:
                # Reindex concurrently to minimize downtime
                reindex_query = text(f"REINDEX INDEX CONCURRENTLY {index.indexname};")
                await session.execute(reindex_query)

                reindex_results["indexes_processed"] += 1
                logger.debug(f"Reindexed: {index.indexname}")

            except Exception as e:
                logger.warning(f"Failed to reindex {index.indexname}: {e}")
                reindex_results["errors"] += 1

        return reindex_results

    except Exception as e:
        logger.error(f"Failed to reindex fragmented indexes: {e}")
        return {"indexes_processed": 0, "errors": 1}


async def _optimize_timescaledb(session) -> dict[str, Any]:
    """Perform TimescaleDB specific optimizations."""
    try:
        timescale_results = {"operations": []}

        # 1. Refresh continuous aggregates
        refresh_query = text("""
            SELECT format('CALL refresh_continuous_aggregate(%L, NULL, NULL);', 
                          view_name) as refresh_sql
            FROM timescaledb_information.continuous_aggregates
            WHERE view_schema = 'public';
        """)

        result = await session.execute(refresh_query)
        refresh_commands = result.fetchall()

        for command in refresh_commands:
            try:
                await session.execute(text(command.refresh_sql))
                timescale_results["operations"].append(f"Refreshed: {command.refresh_sql}")
            except Exception as e:
                logger.warning(f"Failed to refresh continuous aggregate: {e}")

        # 2. Compress old chunks
        compress_query = text("""
            SELECT compress_chunk(chunk_schema||'.'||chunk_name)
            FROM timescaledb_information.chunks
            WHERE NOT is_compressed
            AND range_end < NOW() - INTERVAL '1 day'
            AND hypertable_name = 'traffic_metrics';
        """)

        try:
            compress_result = await session.execute(compress_query)
            compressed_chunks = compress_result.rowcount
            timescale_results["operations"].append(f"Compressed {compressed_chunks} chunks")
        except Exception as e:
            logger.warning(f"Failed to compress chunks: {e}")

        # 3. Drop old chunks (older than retention policy)
        drop_query = text("""
            SELECT drop_chunks('traffic_metrics', INTERVAL '180 days');
        """)

        try:
            await session.execute(drop_query)
            timescale_results["operations"].append("Dropped old chunks")
        except Exception as e:
            logger.warning(f"Failed to drop old chunks: {e}")

        return timescale_results

    except Exception as e:
        logger.error(f"TimescaleDB optimization failed: {e}")
        return {"operations": [], "error": str(e)}


async def _update_query_statistics(session) -> None:
    """Update PostgreSQL query planner statistics."""
    try:
        # Update statistics for all tables
        analyze_query = text("ANALYZE;")
        await session.execute(analyze_query)

        logger.info("Updated query planner statistics")

    except Exception as e:
        logger.warning(f"Failed to update query statistics: {e}")


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 1, "countdown": 300},
)
def cleanup_cache(self, cache_patterns: list[str] | None = None) -> dict[str, Any]:
    """Clean up Redis cache based on patterns.
    
    Args:
        cache_patterns: List of cache key patterns to clean
        
    Returns:
        dict: Cache cleanup results
    """
    return asyncio.run(_cleanup_cache_async(cache_patterns))


async def _cleanup_cache_async(cache_patterns: list[str] | None) -> dict[str, Any]:
    """Async implementation of cache cleanup."""
    container = ApplicationContainer()
    container.wire(modules=[__name__])

    try:
        await container.init_resources()
        cache_service = await container.services.cache_service()

        if cache_patterns is None:
            # Default patterns for cleanup
            cache_patterns = [
                "analytics:*",
                "dashboard:*",
                "reports:*",
                "predictions:*",
                "temp:*",
            ]

        cleanup_results = {"patterns_processed": 0, "keys_deleted": 0}

        for pattern in cache_patterns:
            try:
                # Get keys matching pattern
                keys = await cache_service.redis_client.keys(pattern)

                if keys:
                    # Delete keys in batches
                    batch_size = 1000
                    for i in range(0, len(keys), batch_size):
                        batch = keys[i:i + batch_size]
                        await cache_service.redis_client.delete(*batch)
                        cleanup_results["keys_deleted"] += len(batch)

                cleanup_results["patterns_processed"] += 1
                logger.debug(f"Cleaned cache pattern: {pattern}, deleted {len(keys)} keys")

            except Exception as e:
                logger.warning(f"Failed to clean cache pattern {pattern}: {e}")

        logger.info(f"Cache cleanup completed: {cleanup_results['keys_deleted']} keys deleted")

        return {
            "status": "completed",
            "cleanup_results": cleanup_results,
            "cleanup_time": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
        return {"status": "failed", "error": str(e)}
    finally:
        await container.shutdown_resources()


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 1, "countdown": 600},
)
def system_health_check(self) -> dict[str, Any]:
    """Perform comprehensive system health check.
    
    Returns:
        dict: System health status
    """
    return asyncio.run(_system_health_check_async())


async def _system_health_check_async() -> dict[str, Any]:
    """Async implementation of system health check."""
    container = ApplicationContainer()
    container.wire(modules=[__name__])

    try:
        await container.init_resources()

        health_results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "overall_status": "healthy",
            "components": {}
        }

        # 1. Database health
        try:
            db_manager = await container.infrastructure.database()
            async with db_manager.get_session() as session:
                # Test database connectivity and performance
                test_query = text("SELECT 1;")
                start_time = datetime.now(UTC)
                await session.execute(test_query)
                db_response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

                health_results["components"]["database"] = {
                    "status": "healthy",
                    "response_time_ms": db_response_time,
                    "connection_pool": "active",
                }
        except Exception as e:
            health_results["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_results["overall_status"] = "degraded"

        # 2. Redis health
        try:
            cache_service = await container.services.cache_service()

            # Test Redis connectivity
            start_time = datetime.now(UTC)
            await cache_service.redis_client.ping()
            redis_response_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            # Get Redis info
            redis_info = await cache_service.redis_client.info()

            health_results["components"]["redis"] = {
                "status": "healthy",
                "response_time_ms": redis_response_time,
                "memory_usage": redis_info.get("used_memory_human"),
                "connected_clients": redis_info.get("connected_clients"),
            }
        except Exception as e:
            health_results["components"]["redis"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_results["overall_status"] = "degraded"

        # 3. Disk space check
        try:
            disk_usage = shutil.disk_usage("/")
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            usage_percent = (disk_usage.used / disk_usage.total) * 100

            disk_status = "healthy"
            if usage_percent > 90:
                disk_status = "critical"
                health_results["overall_status"] = "critical"
            elif usage_percent > 80:
                disk_status = "warning"
                if health_results["overall_status"] == "healthy":
                    health_results["overall_status"] = "warning"

            health_results["components"]["disk_space"] = {
                "status": disk_status,
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "usage_percent": round(usage_percent, 2),
            }
        except Exception as e:
            health_results["components"]["disk_space"] = {
                "status": "unknown",
                "error": str(e),
            }

        # 4. Celery worker health
        try:
            from celery import current_app
            inspect = current_app.control.inspect()

            active_workers = inspect.active()
            stats = inspect.stats()

            worker_count = len(active_workers) if active_workers else 0

            health_results["components"]["celery_workers"] = {
                "status": "healthy" if worker_count > 0 else "warning",
                "active_workers": worker_count,
                "worker_stats": stats,
            }
        except Exception as e:
            health_results["components"]["celery_workers"] = {
                "status": "unknown",
                "error": str(e),
            }

        logger.info(f"System health check completed: {health_results['overall_status']}")

        return health_results

    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "overall_status": "critical",
            "error": str(e),
        }
    finally:
        await container.shutdown_resources()
