"""Connection pool optimization for streaming system performance.

This module optimizes Redis, database, and FFMPEG process pools for maximum
throughput and minimum latency with intelligent health monitoring and
automatic connection recovery.
"""

import asyncio
import subprocess
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from weakref import WeakSet

import structlog

from ..core.exceptions import ConnectionPoolError, DatabaseError, RedisConnectionError
from .optimization_config import ConnectionPoolConfig

logger = structlog.get_logger(__name__)

# Database imports with availability checks
try:
    import asyncpg
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.pool import QueuePool
    DATABASE_AVAILABLE = True
except ImportError:
    logger.warning("Database libraries not available - database optimization disabled")
    DATABASE_AVAILABLE = False
    asyncpg = None

# Redis imports
try:
    import redis.asyncio as redis
    from redis.asyncio.connection import ConnectionPool as RedisPool
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("Redis not available - Redis optimization disabled")
    REDIS_AVAILABLE = False
    redis = None


@dataclass
class ConnectionStats:
    """Connection statistics for monitoring."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    reconnections: int = 0
    average_response_time_ms: float = 0.0
    connection_errors: int = 0
    last_health_check: float = field(default_factory=time.time)


@dataclass
class PoolMetrics:
    """Pool performance metrics."""

    pool_name: str
    pool_size: int
    active_count: int
    idle_count: int
    wait_queue_size: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_wait_time_ms: float
    peak_active_count: int
    created_at: float = field(default_factory=time.time)


class HealthMonitor:
    """Health monitoring for connection pools."""

    def __init__(self, check_interval: float = 30.0):
        """Initialize health monitor.
        
        Args:
            check_interval: Health check interval in seconds
        """
        self.check_interval = check_interval
        self.pool_health: dict[str, ConnectionStats] = {}
        self.monitoring_task: asyncio.Task | None = None
        self.is_monitoring = False

    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Connection health monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Connection health monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.check_interval)

                if not self.is_monitoring:
                    break

                # Update health stats for all pools
                current_time = time.time()
                for pool_name in self.pool_health:
                    self.pool_health[pool_name].last_health_check = current_time

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    def record_connection_event(
        self,
        pool_name: str,
        event_type: str,
        response_time_ms: float | None = None
    ) -> None:
        """Record connection event for monitoring.
        
        Args:
            pool_name: Pool identifier
            event_type: Event type (success, failure, reconnect)
            response_time_ms: Response time in milliseconds
        """
        if pool_name not in self.pool_health:
            self.pool_health[pool_name] = ConnectionStats()

        stats = self.pool_health[pool_name]

        if event_type == "success":
            stats.total_connections += 1
        elif event_type == "failure":
            stats.failed_connections += 1
            stats.connection_errors += 1
        elif event_type == "reconnect":
            stats.reconnections += 1

        # Update response time (running average)
        if response_time_ms is not None:
            if stats.average_response_time_ms == 0:
                stats.average_response_time_ms = response_time_ms
            else:
                # Exponential moving average
                stats.average_response_time_ms = (
                    0.9 * stats.average_response_time_ms + 0.1 * response_time_ms
                )

    def get_pool_health(self, pool_name: str) -> ConnectionStats | None:
        """Get health stats for a pool.
        
        Args:
            pool_name: Pool identifier
            
        Returns:
            Optional[ConnectionStats]: Health statistics or None
        """
        return self.pool_health.get(pool_name)


class RedisPoolOptimizer:
    """Redis connection pool optimizer with health monitoring."""

    def __init__(self, config: ConnectionPoolConfig, health_monitor: HealthMonitor):
        """Initialize Redis pool optimizer.
        
        Args:
            config: Connection pool configuration
            health_monitor: Health monitoring instance
        """
        if not REDIS_AVAILABLE:
            raise ConnectionPoolError("Redis not available")

        self.config = config
        self.health_monitor = health_monitor
        self.redis_pool: RedisPool | None = None
        self.redis_client: redis.Redis | None = None

        # Pool metrics
        self.pool_metrics = PoolMetrics(
            pool_name="redis",
            pool_size=config.redis_pool_size,
            active_count=0,
            idle_count=0,
            wait_queue_size=0,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            average_wait_time_ms=0.0,
            peak_active_count=0,
        )

    async def initialize(
        self,
        redis_url: str = "redis://localhost:6379",
        **kwargs
    ) -> None:
        """Initialize Redis connection pool.
        
        Args:
            redis_url: Redis connection URL
            **kwargs: Additional Redis configuration
        """
        try:
            # Create connection pool with optimized settings
            self.redis_pool = redis.ConnectionPool.from_url(
                redis_url,
                max_connections=self.config.redis_max_connections,
                retry_on_timeout=self.config.redis_retry_on_timeout,
                socket_keepalive=self.config.redis_socket_keepalive,
                socket_keepalive_options=self.config.redis_socket_keepalive_options,
                health_check_interval=self.config.redis_health_check_interval,
                **kwargs
            )

            # Create Redis client
            self.redis_client = redis.Redis(
                connection_pool=self.redis_pool,
                decode_responses=False  # Keep as bytes for streaming
            )

            # Test connection
            await self._test_connection()

            logger.info(f"Redis pool initialized: {self.config.redis_max_connections} max connections")

        except Exception as e:
            logger.error(f"Redis pool initialization failed: {e}")
            raise ConnectionPoolError(f"Redis pool setup failed: {e}") from e

    async def _test_connection(self) -> None:
        """Test Redis connection."""
        if not self.redis_client:
            raise ConnectionPoolError("Redis client not initialized")

        start_time = time.perf_counter()

        try:
            await self.redis_client.ping()
            response_time_ms = (time.perf_counter() - start_time) * 1000

            self.health_monitor.record_connection_event(
                "redis", "success", response_time_ms
            )

        except Exception as e:
            self.health_monitor.record_connection_event("redis", "failure")
            raise RedisConnectionError(f"Redis connection test failed: {e}") from e

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[redis.Redis, None]:
        """Get Redis connection with automatic cleanup.
        
        Yields:
            redis.Redis: Redis connection
        """
        if not self.redis_client:
            raise ConnectionPoolError("Redis pool not initialized")

        start_time = time.perf_counter()

        try:
            # Update metrics
            self.pool_metrics.total_requests += 1
            self.pool_metrics.active_count += 1
            self.pool_metrics.peak_active_count = max(
                self.pool_metrics.peak_active_count,
                self.pool_metrics.active_count
            )

            yield self.redis_client

            # Record success
            response_time_ms = (time.perf_counter() - start_time) * 1000
            self.pool_metrics.successful_requests += 1
            self.health_monitor.record_connection_event(
                "redis", "success", response_time_ms
            )

        except Exception as e:
            self.pool_metrics.failed_requests += 1
            self.health_monitor.record_connection_event("redis", "failure")
            logger.error(f"Redis connection error: {e}")
            raise

        finally:
            self.pool_metrics.active_count = max(0, self.pool_metrics.active_count - 1)

    async def execute_pipeline(self, commands: list[tuple]) -> list[Any]:
        """Execute Redis commands in pipeline for better performance.
        
        Args:
            commands: List of (command, *args) tuples
            
        Returns:
            List[Any]: Command results
        """
        async with self.get_connection() as redis_conn:
            pipe = redis_conn.pipeline()

            # Add all commands to pipeline
            for cmd_tuple in commands:
                if len(cmd_tuple) == 2:
                    command, args = cmd_tuple
                    getattr(pipe, command)(*args if isinstance(args, (list, tuple)) else [args])
                else:
                    command = cmd_tuple[0]
                    args = cmd_tuple[1:] if len(cmd_tuple) > 1 else []
                    getattr(pipe, command)(*args)

            # Execute pipeline
            return await pipe.execute()

    def get_pool_stats(self) -> dict[str, Any]:
        """Get Redis pool statistics.
        
        Returns:
            Dict[str, Any]: Pool statistics
        """
        pool_info = {}

        if self.redis_pool:
            pool_info = {
                "max_connections": self.redis_pool.max_connections,
                "created_connections": getattr(self.redis_pool, "_created_connections", 0),
                "available_connections": len(getattr(self.redis_pool, "_available_connections", [])),
                "in_use_connections": len(getattr(self.redis_pool, "_in_use_connections", set())),
            }

        return {
            "pool_metrics": {
                "pool_size": self.pool_metrics.pool_size,
                "active_count": self.pool_metrics.active_count,
                "total_requests": self.pool_metrics.total_requests,
                "successful_requests": self.pool_metrics.successful_requests,
                "failed_requests": self.pool_metrics.failed_requests,
                "success_rate_percent": (
                    (self.pool_metrics.successful_requests /
                     max(1, self.pool_metrics.total_requests)) * 100
                ),
                "peak_active_count": self.pool_metrics.peak_active_count,
            },
            "pool_info": pool_info,
            "health": self.health_monitor.get_pool_health("redis"),
        }


class DatabasePoolOptimizer:
    """Database connection pool optimizer with async support."""

    def __init__(self, config: ConnectionPoolConfig, health_monitor: HealthMonitor):
        """Initialize database pool optimizer.
        
        Args:
            config: Connection pool configuration
            health_monitor: Health monitoring instance
        """
        if not DATABASE_AVAILABLE:
            raise ConnectionPoolError("Database libraries not available")

        self.config = config
        self.health_monitor = health_monitor
        self.engine = None

        # Pool metrics
        self.pool_metrics = PoolMetrics(
            pool_name="database",
            pool_size=config.db_pool_size,
            active_count=0,
            idle_count=0,
            wait_queue_size=0,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            average_wait_time_ms=0.0,
            peak_active_count=0,
        )

    async def initialize(self, database_url: str) -> None:
        """Initialize database connection pool.
        
        Args:
            database_url: Database connection URL
        """
        try:
            # Create async engine with optimized pool settings
            self.engine = create_async_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=self.config.db_pool_size,
                max_overflow=self.config.db_max_overflow,
                pool_recycle=self.config.db_pool_recycle,
                pool_pre_ping=self.config.db_pool_pre_ping,
                pool_timeout=self.config.db_pool_timeout,
                echo=False,  # Disable SQL logging for performance
            )

            # Test connection
            await self._test_connection()

            logger.info(f"Database pool initialized: {self.config.db_pool_size} pool size")

        except Exception as e:
            logger.error(f"Database pool initialization failed: {e}")
            raise ConnectionPoolError(f"Database pool setup failed: {e}") from e

    async def _test_connection(self) -> None:
        """Test database connection."""
        if not self.engine:
            raise ConnectionPoolError("Database engine not initialized")

        start_time = time.perf_counter()

        try:
            async with self.engine.begin() as conn:
                await conn.execute("SELECT 1")

            response_time_ms = (time.perf_counter() - start_time) * 1000
            self.health_monitor.record_connection_event(
                "database", "success", response_time_ms
            )

        except Exception as e:
            self.health_monitor.record_connection_event("database", "failure")
            raise DatabaseError(f"Database connection test failed: {e}") from e

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup.
        
        Yields:
            AsyncSession: Database session
        """
        if not self.engine:
            raise ConnectionPoolError("Database pool not initialized")

        start_time = time.perf_counter()

        try:
            # Update metrics
            self.pool_metrics.total_requests += 1
            self.pool_metrics.active_count += 1
            self.pool_metrics.peak_active_count = max(
                self.pool_metrics.peak_active_count,
                self.pool_metrics.active_count
            )

            async with AsyncSession(self.engine, expire_on_commit=False) as session:
                yield session
                await session.commit()

            # Record success
            response_time_ms = (time.perf_counter() - start_time) * 1000
            self.pool_metrics.successful_requests += 1
            self.health_monitor.record_connection_event(
                "database", "success", response_time_ms
            )

        except Exception as e:
            self.pool_metrics.failed_requests += 1
            self.health_monitor.record_connection_event("database", "failure")
            logger.error(f"Database session error: {e}")
            raise

        finally:
            self.pool_metrics.active_count = max(0, self.pool_metrics.active_count - 1)

    def get_pool_stats(self) -> dict[str, Any]:
        """Get database pool statistics.
        
        Returns:
            Dict[str, Any]: Pool statistics
        """
        pool_info = {}

        if self.engine and hasattr(self.engine.pool, 'size'):
            pool_info = {
                "pool_size": self.engine.pool.size(),
                "checked_in": self.engine.pool.checkedin(),
                "checked_out": self.engine.pool.checkedout(),
                "overflow": self.engine.pool.overflow(),
                "invalid": self.engine.pool.invalid(),
            }

        return {
            "pool_metrics": {
                "pool_size": self.pool_metrics.pool_size,
                "active_count": self.pool_metrics.active_count,
                "total_requests": self.pool_metrics.total_requests,
                "successful_requests": self.pool_metrics.successful_requests,
                "failed_requests": self.pool_metrics.failed_requests,
                "success_rate_percent": (
                    (self.pool_metrics.successful_requests /
                     max(1, self.pool_metrics.total_requests)) * 100
                ),
                "peak_active_count": self.pool_metrics.peak_active_count,
            },
            "pool_info": pool_info,
            "health": self.health_monitor.get_pool_health("database"),
        }


class FFMPEGProcessPool:
    """FFMPEG process pool for video processing optimization."""

    def __init__(self, config: ConnectionPoolConfig, health_monitor: HealthMonitor):
        """Initialize FFMPEG process pool.
        
        Args:
            config: Connection pool configuration
            health_monitor: Health monitoring instance
        """
        self.config = config
        self.health_monitor = health_monitor

        # Process management
        self.active_processes: WeakSet = WeakSet()
        self.process_semaphore = asyncio.Semaphore(config.ffmpeg_max_concurrent)
        self.job_counter = 0
        self.restart_counter = 0

        # Pool metrics
        self.pool_metrics = PoolMetrics(
            pool_name="ffmpeg",
            pool_size=config.ffmpeg_pool_size,
            active_count=0,
            idle_count=0,
            wait_queue_size=0,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            average_wait_time_ms=0.0,
            peak_active_count=0,
        )

    @asynccontextmanager
    async def execute_ffmpeg(
        self,
        command: list[str],
        timeout: float | None = None
    ) -> AsyncGenerator[subprocess.Popen, None]:
        """Execute FFMPEG command with process pool management.
        
        Args:
            command: FFMPEG command arguments
            timeout: Process timeout in seconds
            
        Yields:
            subprocess.Popen: FFMPEG process
        """
        timeout = timeout or self.config.ffmpeg_process_timeout
        start_time = time.perf_counter()

        async with self.process_semaphore:
            try:
                # Update metrics
                self.pool_metrics.total_requests += 1
                self.pool_metrics.active_count += 1
                self.pool_metrics.peak_active_count = max(
                    self.pool_metrics.peak_active_count,
                    self.pool_metrics.active_count
                )
                self.job_counter += 1

                # Create process
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=1024*1024  # 1MB buffer limit
                )

                self.active_processes.add(process)

                try:
                    yield process

                    # Wait for completion with timeout
                    await asyncio.wait_for(process.wait(), timeout=timeout)

                    if process.returncode == 0:
                        self.pool_metrics.successful_requests += 1
                        response_time_ms = (time.perf_counter() - start_time) * 1000
                        self.health_monitor.record_connection_event(
                            "ffmpeg", "success", response_time_ms
                        )
                    else:
                        self.pool_metrics.failed_requests += 1
                        self.health_monitor.record_connection_event("ffmpeg", "failure")

                except TimeoutError:
                    # Kill process on timeout
                    process.kill()
                    await process.wait()
                    self.pool_metrics.failed_requests += 1
                    self.health_monitor.record_connection_event("ffmpeg", "failure")
                    raise

                finally:
                    # Ensure process is terminated
                    if process.returncode is None:
                        process.terminate()
                        await process.wait()

                # Check if we need to restart pool (prevent memory leaks)
                if self.job_counter >= self.config.ffmpeg_restart_threshold:
                    await self._restart_pool()

            except Exception as e:
                self.pool_metrics.failed_requests += 1
                self.health_monitor.record_connection_event("ffmpeg", "failure")
                logger.error(f"FFMPEG process error: {e}")
                raise

            finally:
                self.pool_metrics.active_count = max(0, self.pool_metrics.active_count - 1)

    async def _restart_pool(self) -> None:
        """Restart process pool to prevent memory leaks."""
        logger.info("Restarting FFMPEG process pool")

        # Terminate all active processes
        for process in list(self.active_processes):
            if process.returncode is None:
                process.terminate()

        # Wait for all processes to complete
        await asyncio.sleep(1.0)

        # Reset counters
        self.job_counter = 0
        self.restart_counter += 1

        logger.info(f"FFMPEG process pool restarted (restart #{self.restart_counter})")

    def get_pool_stats(self) -> dict[str, Any]:
        """Get FFMPEG process pool statistics.
        
        Returns:
            Dict[str, Any]: Pool statistics
        """
        return {
            "pool_metrics": {
                "pool_size": self.pool_metrics.pool_size,
                "active_count": self.pool_metrics.active_count,
                "total_requests": self.pool_metrics.total_requests,
                "successful_requests": self.pool_metrics.successful_requests,
                "failed_requests": self.pool_metrics.failed_requests,
                "success_rate_percent": (
                    (self.pool_metrics.successful_requests /
                     max(1, self.pool_metrics.total_requests)) * 100
                ),
                "peak_active_count": self.pool_metrics.peak_active_count,
            },
            "process_info": {
                "active_processes": len(self.active_processes),
                "job_counter": self.job_counter,
                "restart_counter": self.restart_counter,
                "max_concurrent": self.config.ffmpeg_max_concurrent,
            },
            "health": self.health_monitor.get_pool_health("ffmpeg"),
        }


class ConnectionPoolOptimizer:
    """Comprehensive connection pool optimizer for all external connections.
    
    Coordinates Redis, database, and FFMPEG process pools with intelligent
    health monitoring and automatic connection recovery.
    """

    def __init__(self, config: ConnectionPoolConfig):
        """Initialize connection pool optimizer.
        
        Args:
            config: Connection pool configuration
        """
        self.config = config
        self.health_monitor = HealthMonitor(30.0)  # 30-second health checks

        # Pool optimizers
        self.redis_optimizer: RedisPoolOptimizer | None = None
        self.database_optimizer: DatabasePoolOptimizer | None = None
        self.ffmpeg_pool: FFMPEGProcessPool | None = None

        self.is_initialized = False

    async def initialize(
        self,
        redis_url: str | None = None,
        database_url: str | None = None
    ) -> None:
        """Initialize all connection pools.
        
        Args:
            redis_url: Redis connection URL
            database_url: Database connection URL
        """
        if self.is_initialized:
            return

        try:
            # Start health monitoring
            await self.health_monitor.start_monitoring()

            # Initialize Redis pool
            if redis_url and REDIS_AVAILABLE:
                self.redis_optimizer = RedisPoolOptimizer(self.config, self.health_monitor)
                await self.redis_optimizer.initialize(redis_url)

            # Initialize database pool
            if database_url and DATABASE_AVAILABLE:
                self.database_optimizer = DatabasePoolOptimizer(self.config, self.health_monitor)
                await self.database_optimizer.initialize(database_url)

            # Initialize FFMPEG process pool
            self.ffmpeg_pool = FFMPEGProcessPool(self.config, self.health_monitor)

            self.is_initialized = True
            logger.info("Connection pool optimizer initialized")

        except Exception as e:
            logger.error(f"Connection pool initialization failed: {e}")
            raise ConnectionPoolError(f"Pool initialization failed: {e}") from e

    async def cleanup(self) -> None:
        """Cleanup all connection pools."""
        try:
            # Stop health monitoring
            await self.health_monitor.stop_monitoring()

            # Cleanup pools
            if self.redis_optimizer and self.redis_optimizer.redis_pool:
                await self.redis_optimizer.redis_pool.disconnect()

            if self.database_optimizer and self.database_optimizer.engine:
                await self.database_optimizer.engine.dispose()

            if self.ffmpeg_pool:
                await self.ffmpeg_pool._restart_pool()

            self.is_initialized = False
            logger.info("Connection pool optimizer cleanup completed")

        except Exception as e:
            logger.error(f"Connection pool cleanup error: {e}")

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Get comprehensive statistics for all pools.
        
        Returns:
            Dict[str, Any]: All pool statistics
        """
        stats = {
            "initialized": self.is_initialized,
            "health_monitoring": {
                "enabled": self.health_monitor.is_monitoring,
                "check_interval": self.health_monitor.check_interval,
                "monitored_pools": list(self.health_monitor.pool_health.keys()),
            },
        }

        if self.redis_optimizer:
            stats["redis"] = self.redis_optimizer.get_pool_stats()

        if self.database_optimizer:
            stats["database"] = self.database_optimizer.get_pool_stats()

        if self.ffmpeg_pool:
            stats["ffmpeg"] = self.ffmpeg_pool.get_pool_stats()

        return stats


async def create_connection_pool_optimizer(
    config: ConnectionPoolConfig,
    redis_url: str | None = None,
    database_url: str | None = None
) -> ConnectionPoolOptimizer:
    """Create and initialize connection pool optimizer.
    
    Args:
        config: Connection pool configuration
        redis_url: Redis connection URL
        database_url: Database connection URL
        
    Returns:
        ConnectionPoolOptimizer: Initialized connection pool optimizer
    """
    optimizer = ConnectionPoolOptimizer(config)
    await optimizer.initialize(redis_url, database_url)
    return optimizer
