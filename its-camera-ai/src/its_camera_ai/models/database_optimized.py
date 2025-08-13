"""High-performance database manager optimized for ITS Camera AI.

This module provides enhanced database management with:
- Optimized connection pooling for high-throughput operations
- Bulk insert operations for frame metadata
- Connection health monitoring and failover
- Performance metrics collection
- Automatic connection retry logic

Performance targets:
- Handle 3000+ frame inserts per second
- Sub-10ms query response times for camera lookups
- Efficient connection pooling and resource management
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import asyncpg
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import QueuePool, StaticPool

from ..core.config import Settings
from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from .base import BaseModel

logger = get_logger(__name__)


@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration for different workload types."""

    # Basic pool settings
    pool_size: int = 20
    max_overflow: int = 40
    pool_timeout: int = 30
    pool_recycle: int = 3600

    # Advanced settings
    pool_pre_ping: bool = True
    pool_reset_on_return: str = "commit"

    # High-throughput specific
    enable_bulk_operations: bool = True
    bulk_insert_batch_size: int = 1000

    # Connection health monitoring
    health_check_interval: int = 30
    max_retries: int = 3
    retry_delay: float = 0.1


class HighPerformanceDatabaseManager:
    """Enhanced database manager optimized for high-throughput operations."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine: AsyncEngine | None = None
        self.session_factory: async_sessionmaker[AsyncSession] | None = None
        self.connection_pool: asyncpg.Pool | None = None

        # Performance monitoring
        self._query_count = 0
        self._total_query_time = 0.0
        self._connection_errors = 0
        self._last_health_check = 0.0

        # Pool configuration based on workload
        self.pool_config = self._get_pool_config()

    def _get_pool_config(self) -> ConnectionPoolConfig:
        """Get optimized pool configuration based on environment."""
        if self.settings.is_production():
            return ConnectionPoolConfig(
                pool_size=50,
                max_overflow=100,
                pool_timeout=60,
                bulk_insert_batch_size=2000,
                health_check_interval=15
            )
        elif self.settings.environment == "staging":
            return ConnectionPoolConfig(
                pool_size=20,
                max_overflow=40,
                bulk_insert_batch_size=1000
            )
        else:  # development/testing
            return ConnectionPoolConfig(
                pool_size=10,
                max_overflow=20,
                bulk_insert_batch_size=500,
                pool_recycle=1800  # Shorter recycle for dev
            )

    async def initialize(self) -> None:
        """Initialize optimized database connections."""
        try:
            # Create SQLAlchemy async engine with optimized pooling
            self.engine = create_async_engine(
                self.settings.get_database_url(async_driver=True),
                echo=self.settings.database.echo and self.settings.is_development(),

                # Connection pool settings
                poolclass=QueuePool if not self.settings.is_development() else StaticPool,
                pool_size=self.pool_config.pool_size,
                max_overflow=self.pool_config.max_overflow,
                pool_timeout=self.pool_config.pool_timeout,
                pool_recycle=self.pool_config.pool_recycle,
                pool_pre_ping=self.pool_config.pool_pre_ping,
                pool_reset_on_return=self.pool_config.pool_reset_on_return,

                # Performance optimizations
                connect_args={
                    "command_timeout": 60,
                    "server_settings": {
                        "application_name": "its_camera_ai",
                        "jit": "off",  # Disable JIT for short queries
                        "random_page_cost": "1.1",
                        "effective_io_concurrency": "200"
                    }
                }
            )

            # Create session factory with optimized settings
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,  # Manual flushing for better control
                autocommit=False
            )

            # Create raw asyncpg pool for high-performance bulk operations
            await self._create_asyncpg_pool()

            # Setup connection monitoring
            self._setup_connection_monitoring()

            # Run initial health check
            await self.health_check()

            logger.info(
                "High-performance database initialized",
                pool_size=self.pool_config.pool_size,
                max_overflow=self.pool_config.max_overflow,
                bulk_batch_size=self.pool_config.bulk_insert_batch_size
            )

        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise DatabaseError("Database initialization failed", cause=e) from e

    async def _create_asyncpg_pool(self) -> None:
        """Create raw asyncpg connection pool for bulk operations."""
        try:
            # Parse connection URL for asyncpg
            db_url = self.settings.get_database_url(async_driver=True)
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

            self.connection_pool = await asyncpg.create_pool(
                db_url,
                min_size=10,
                max_size=self.pool_config.pool_size,
                command_timeout=60,
                max_queries=50000,
                max_inactive_connection_lifetime=300
            )

            logger.debug("AsyncPG connection pool created")

        except Exception as e:
            logger.error("Failed to create AsyncPG pool", error=str(e))
            # Continue without raw pool - bulk operations will use SQLAlchemy

    def _setup_connection_monitoring(self) -> None:
        """Setup connection and query monitoring."""
        if not self.engine:
            return

        @event.listens_for(self.engine.sync_engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()

        @event.listens_for(self.engine.sync_engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total_time = time.time() - context._query_start_time
            self._query_count += 1
            self._total_query_time += total_time

            # Log slow queries
            if total_time > 1.0:
                logger.warning(
                    "Slow query detected",
                    query_time=total_time,
                    statement=statement[:200]
                )

    async def create_tables(self) -> None:
        """Create all database tables."""
        if not self.engine:
            raise DatabaseError("Database not initialized")

        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(BaseModel.metadata.create_all)

            logger.info("Database tables created")

        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise DatabaseError("Table creation failed", cause=e) from e

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get optimized database session with error handling."""
        if not self.session_factory:
            raise DatabaseError("Database not initialized")

        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            await session.close()

    async def bulk_insert_frame_metadata(
        self,
        frame_data: list[dict[str, Any]]
    ) -> int:
        """High-performance bulk insert for frame metadata.

        Args:
            frame_data: List of frame metadata dictionaries

        Returns:
            Number of inserted records
        """
        if not frame_data:
            return 0

        if not self.connection_pool:
            # Fallback to SQLAlchemy bulk insert
            return await self._sqlalchemy_bulk_insert(frame_data)

        try:
            # Use raw asyncpg for maximum performance
            return await self._asyncpg_bulk_insert(frame_data)

        except Exception as e:
            logger.error("Bulk insert failed", error=str(e), count=len(frame_data))
            self._connection_errors += 1
            raise DatabaseError("Bulk insert operation failed", cause=e) from e

    async def _asyncpg_bulk_insert(self, frame_data: list[dict[str, Any]]) -> int:
        """Raw asyncpg bulk insert for maximum throughput."""
        async with self.connection_pool.acquire() as conn:
            # Prepare data for COPY command
            columns = [
                'id', 'camera_id', 'frame_number', 'timestamp', 'processing_status',
                'processing_time_ms', 'frame_size_bytes', 'total_detections',
                'vehicle_count', 'person_count', 'confidence_avg', 'detection_results'
            ]

            # Convert dict data to tuple format
            rows = []
            for data in frame_data:
                row = [
                    data.get('id'),
                    data.get('camera_id'),
                    data.get('frame_number'),
                    data.get('timestamp'),
                    data.get('processing_status', 'pending'),
                    data.get('processing_time_ms'),
                    data.get('frame_size_bytes', 0),
                    data.get('total_detections', 0),
                    data.get('vehicle_count', 0),
                    data.get('person_count', 0),
                    data.get('confidence_avg'),
                    data.get('detection_results')
                ]
                rows.append(row)

            # Use COPY for fastest possible insert
            await conn.copy_records_to_table(
                'frame_metadata',
                records=rows,
                columns=columns
            )

            return len(rows)

    async def _sqlalchemy_bulk_insert(self, frame_data: list[dict[str, Any]]) -> int:
        """SQLAlchemy bulk insert fallback."""
        async with self.get_session() as session:
            # Use bulk_insert_mappings for better performance than individual inserts
            await session.run_sync(
                lambda sync_session: sync_session.bulk_insert_mappings(
                    'FrameMetadata', frame_data
                )
            )
            return len(frame_data)

    async def bulk_insert_detections(
        self,
        detection_data: list[dict[str, Any]]
    ) -> int:
        """Bulk insert detection results."""
        if not detection_data:
            return 0

        try:
            if self.connection_pool:
                async with self.connection_pool.acquire() as conn:
                    columns = [
                        'id', 'frame_metadata_id', 'object_type', 'confidence',
                        'class_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
                        'track_id', 'speed_kmh', 'direction', 'attributes'
                    ]

                    rows = []
                    for data in detection_data:
                        row = [
                            data.get('id'),
                            data.get('frame_metadata_id'),
                            data.get('object_type'),
                            data.get('confidence'),
                            data.get('class_id'),
                            data.get('bbox_x1'),
                            data.get('bbox_y1'),
                            data.get('bbox_x2'),
                            data.get('bbox_y2'),
                            data.get('track_id'),
                            data.get('speed_kmh'),
                            data.get('direction'),
                            data.get('attributes')
                        ]
                        rows.append(row)

                    await conn.copy_records_to_table(
                        'detections',
                        records=rows,
                        columns=columns
                    )

                    return len(rows)
            else:
                # SQLAlchemy fallback
                async with self.get_session() as session:
                    await session.run_sync(
                        lambda sync_session: sync_session.bulk_insert_mappings(
                            'Detection', detection_data
                        )
                    )
                    return len(detection_data)

        except Exception as e:
            logger.error("Detection bulk insert failed", error=str(e))
            raise DatabaseError("Detection bulk insert failed", cause=e) from e

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute optimized query with connection retry."""
        for attempt in range(self.pool_config.max_retries):
            try:
                async with self.get_session() as session:
                    result = await session.execute(
                        text(query),
                        parameters or {}
                    )
                    return [dict(row) for row in result.mappings()]

            except Exception as e:
                if attempt < self.pool_config.max_retries - 1:
                    await asyncio.sleep(self.pool_config.retry_delay * (2 ** attempt))
                    continue
                raise DatabaseError(f"Query execution failed after {self.pool_config.max_retries} attempts", cause=e) from e

    async def health_check(self) -> dict[str, Any]:
        """Comprehensive database health check."""
        current_time = time.time()

        # Skip if recently checked
        if (current_time - self._last_health_check) < self.pool_config.health_check_interval:
            return {"status": "cached", "last_check": self._last_health_check}

        health_status = {
            "status": "healthy",
            "timestamp": current_time,
            "connection_pool": {},
            "performance": {},
            "errors": self._connection_errors
        }

        try:
            # Test SQLAlchemy connection
            start_time = time.time()
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            sqlalchemy_latency = (time.time() - start_time) * 1000

            # Test asyncpg pool if available
            asyncpg_latency = None
            if self.connection_pool:
                start_time = time.time()
                async with self.connection_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                asyncpg_latency = (time.time() - start_time) * 1000

            # Pool statistics
            if self.engine and hasattr(self.engine.pool, 'size'):
                pool_stats = self.engine.pool
                health_status["connection_pool"] = {
                    "size": pool_stats.size(),
                    "checked_in": pool_stats.checkedin(),
                    "checked_out": pool_stats.checkedout(),
                    "overflow": pool_stats.overflow(),
                    "invalid": getattr(pool_stats, 'invalid', 0)
                }

            # Performance metrics
            avg_query_time = self._total_query_time / max(self._query_count, 1)
            health_status["performance"] = {
                "total_queries": self._query_count,
                "avg_query_time_ms": avg_query_time * 1000,
                "sqlalchemy_latency_ms": sqlalchemy_latency,
                "asyncpg_latency_ms": asyncpg_latency
            }

            # Health assessment
            if sqlalchemy_latency > 100:  # > 100ms is concerning
                health_status["status"] = "degraded"
                health_status["warnings"] = ["High query latency detected"]

            if self._connection_errors > 10:
                health_status["status"] = "unhealthy"
                health_status["errors_count"] = self._connection_errors

            self._last_health_check = current_time

        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
            logger.error("Database health check failed", error=str(e))

        return health_status

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get detailed performance metrics."""
        return {
            "query_count": self._query_count,
            "total_query_time": self._total_query_time,
            "avg_query_time": self._total_query_time / max(self._query_count, 1),
            "connection_errors": self._connection_errors,
            "pool_config": {
                "pool_size": self.pool_config.pool_size,
                "max_overflow": self.pool_config.max_overflow,
                "bulk_batch_size": self.pool_config.bulk_insert_batch_size
            }
        }

    async def optimize_for_workload(self, workload_type: str) -> None:
        """Dynamically optimize database settings for different workloads."""
        optimizations = {
            "high_throughput_ingest": {
                "synchronous_commit": "off",
                "wal_writer_delay": "10ms",
                "commit_delay": "100",
                "commit_siblings": "10"
            },
            "real_time_queries": {
                "synchronous_commit": "local",
                "random_page_cost": "1.1",
                "seq_page_cost": "1.0",
                "cpu_tuple_cost": "0.01"
            },
            "analytics": {
                "work_mem": "256MB",
                "max_parallel_workers_per_gather": "4",
                "enable_partitionwise_join": "on",
                "enable_partitionwise_aggregate": "on"
            }
        }

        if workload_type in optimizations:
            settings = optimizations[workload_type]
            async with self.get_session() as session:
                for setting, value in settings.items():
                    await session.execute(text(f"SET {setting} = '{value}'"))

            logger.info("Database optimized for workload", workload=workload_type)

    async def close(self) -> None:
        """Close all database connections."""
        try:
            if self.connection_pool:
                await self.connection_pool.close()
                self.connection_pool = None

            if self.engine:
                await self.engine.dispose()
                self.engine = None
                self.session_factory = None

            logger.info("Database connections closed")

        except Exception as e:
            logger.error("Error closing database connections", error=str(e))


# Global instance
_db_manager: HighPerformanceDatabaseManager | None = None


async def create_database_engine(settings: Settings) -> HighPerformanceDatabaseManager:
    """Create and initialize high-performance database manager."""
    global _db_manager

    if _db_manager is None:
        _db_manager = HighPerformanceDatabaseManager(settings)
        await _db_manager.initialize()

    return _db_manager


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Get high-performance database session dependency."""
    if _db_manager is None:
        raise DatabaseError("Database not initialized")

    async with _db_manager.get_session() as session:
        yield session


async def close_database() -> None:
    """Close high-performance database connections."""
    global _db_manager

    if _db_manager:
        await _db_manager.close()
        _db_manager = None


async def get_db_manager() -> HighPerformanceDatabaseManager:
    """Get the global database manager instance."""
    if _db_manager is None:
        raise DatabaseError("Database not initialized")
    return _db_manager
