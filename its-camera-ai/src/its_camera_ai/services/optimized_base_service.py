"""Optimized base service with performance optimizations.

This module provides an optimized base service class with:
- Async patterns for high-throughput processing
- Connection pooling optimization
- Caching strategies
- Error handling and retry mechanisms
- Performance monitoring
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import QueuePool

from ..core.exceptions import DatabaseError, ServiceError
from ..core.logging import get_logger

T = TypeVar('T')
logger = get_logger(__name__)


class PerformanceMonitor:
    """Performance monitoring for service operations."""

    def __init__(self):
        self.metrics: dict[str, list[float]] = {}
        self.call_counts: dict[str, int] = {}

    def record_execution_time(self, operation: str, duration: float) -> None:
        """Record execution time for an operation."""
        if operation not in self.metrics:
            self.metrics[operation] = []
            self.call_counts[operation] = 0

        self.metrics[operation].append(duration)
        self.call_counts[operation] += 1

        # Keep only last 100 measurements
        if len(self.metrics[operation]) > 100:
            self.metrics[operation] = self.metrics[operation][-100:]

    def get_average_time(self, operation: str) -> float | None:
        """Get average execution time for an operation."""
        if operation in self.metrics and self.metrics[operation]:
            return sum(self.metrics[operation]) / len(self.metrics[operation])
        return None

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Get comprehensive statistics."""
        stats = {}
        for operation in self.metrics:
            if self.metrics[operation]:
                times = self.metrics[operation]
                stats[operation] = {
                    "call_count": self.call_counts[operation],
                    "avg_time_ms": sum(times) / len(times) * 1000,
                    "min_time_ms": min(times) * 1000,
                    "max_time_ms": max(times) * 1000,
                    "p95_time_ms": sorted(times)[int(len(times) * 0.95)] * 1000,
                }
        return stats


class ConnectionPoolManager:
    """Optimized connection pool management."""

    def __init__(self, max_connections: int = 20, overflow: int = 10):
        self.max_connections = max_connections
        self.overflow = overflow
        self.active_connections = 0
        self.pool_hits = 0
        self.pool_misses = 0

    def get_pool_config(self) -> dict[str, Any]:
        """Get optimized pool configuration."""
        return {
            "pool_class": QueuePool,
            "pool_size": self.max_connections,
            "max_overflow": self.overflow,
            "pool_pre_ping": True,
            "pool_recycle": 3600,  # Recycle connections every hour
            "pool_reset_on_return": "commit",
        }

    def record_connection_use(self, from_pool: bool = True) -> None:
        """Record connection usage statistics."""
        if from_pool:
            self.pool_hits += 1
        else:
            self.pool_misses += 1

    def get_pool_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        total_requests = self.pool_hits + self.pool_misses
        hit_rate = (self.pool_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "active_connections": self.active_connections,
            "pool_hits": self.pool_hits,
            "pool_misses": self.pool_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
        }


class CacheStrategy:
    """Caching strategy for service operations."""

    def __init__(self, cache_service, default_ttl: int = 300):
        self.cache_service = cache_service
        self.default_ttl = default_ttl
        self.cache_hits = 0
        self.cache_misses = 0

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: int | None = None
    ) -> Any:
        """Get from cache or set using factory function."""
        try:
            # Try to get from cache
            cached_value = await self.cache_service.get_json(key)
            if cached_value is not None:
                self.cache_hits += 1
                return cached_value

            # Cache miss - call factory
            self.cache_misses += 1
            value = await factory()

            # Set in cache
            await self.cache_service.set_json(key, value, ttl or self.default_ttl)
            return value

        except Exception as e:
            logger.warning(f"Cache operation failed for key {key}: {e}")
            # Fallback to factory on cache failure
            return await factory()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
        }


class RetryStrategy:
    """Retry strategy for resilient operations."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    async def execute_with_retry(
        self,
        operation: Callable[[], T],
        retry_on: tuple = (DatabaseError, ConnectionError),
        operation_name: str = "unknown"
    ) -> T:
        """Execute operation with exponential backoff retry."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await operation()
            except retry_on as e:
                last_exception = e

                if attempt == self.max_retries:
                    logger.error(
                        f"Operation {operation_name} failed after {self.max_retries} retries",
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    break

                # Calculate delay with exponential backoff
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)

                logger.warning(
                    f"Operation {operation_name} failed, retrying in {delay}s",
                    error=str(e),
                    attempt=attempt + 1,
                    delay=delay,
                )

                await asyncio.sleep(delay)

        # Re-raise the last exception
        raise last_exception


class OptimizedBaseService(ABC):
    """Optimized base service with performance features."""

    def __init__(
        self,
        cache_service=None,
        performance_monitoring: bool = True,
        connection_pool_size: int = 20,
        default_cache_ttl: int = 300,
    ):
        self.cache_service = cache_service
        self.performance_monitor = PerformanceMonitor() if performance_monitoring else None
        self.connection_pool = ConnectionPoolManager(max_connections=connection_pool_size)

        if cache_service:
            self.cache_strategy = CacheStrategy(cache_service, default_cache_ttl)
        else:
            self.cache_strategy = None

        self.retry_strategy = RetryStrategy()

        # Service-specific metrics
        self.operation_count = 0
        self.error_count = 0
        self.start_time = time.time()

    @asynccontextmanager
    async def performance_context(self, operation_name: str) -> AsyncGenerator[None, None]:
        """Context manager for performance monitoring."""
        start_time = time.time()
        self.operation_count += 1

        try:
            yield
        except Exception as e:
            self.error_count += 1
            logger.error(f"Operation {operation_name} failed", error=str(e))
            raise
        finally:
            if self.performance_monitor:
                duration = time.time() - start_time
                self.performance_monitor.record_execution_time(operation_name, duration)

                # Log slow operations
                if duration > 1.0:  # More than 1 second
                    logger.warning(
                        f"Slow operation detected: {operation_name}",
                        duration_ms=duration * 1000,
                    )

    async def execute_with_cache(
        self,
        cache_key: str,
        factory: Callable[[], Any],
        ttl: int | None = None,
        operation_name: str = "cached_operation",
    ) -> Any:
        """Execute operation with caching."""
        if not self.cache_strategy:
            return await factory()

        async with self.performance_context(f"{operation_name}_cached"):
            return await self.cache_strategy.get_or_set(cache_key, factory, ttl)

    async def execute_with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str = "retry_operation",
        retry_on: tuple = (DatabaseError, ConnectionError),
    ) -> T:
        """Execute operation with retry logic."""
        async with self.performance_context(f"{operation_name}_retry"):
            return await self.retry_strategy.execute_with_retry(
                operation, retry_on, operation_name
            )

    @asynccontextmanager
    async def get_optimized_session(self, session_factory) -> AsyncGenerator[AsyncSession, None]:
        """Get optimized database session with connection pooling."""
        async with session_factory() as session:
            try:
                self.connection_pool.record_connection_use(from_pool=True)
                yield session
            except Exception as e:
                await session.rollback()
                logger.error("Database session error", error=str(e))
                raise

    async def bulk_execute(
        self,
        operations: list[Callable[[], Any]],
        batch_size: int = 10,
        operation_name: str = "bulk_operation",
    ) -> list[Any]:
        """Execute operations in optimized batches."""
        results = []

        async with self.performance_context(f"{operation_name}_bulk"):
            # Process in batches to avoid overwhelming the system
            for i in range(0, len(operations), batch_size):
                batch = operations[i:i + batch_size]

                # Execute batch concurrently
                batch_results = await asyncio.gather(
                    *[op() for op in batch],
                    return_exceptions=True
                )

                # Handle exceptions in batch
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error("Batch operation failed", error=str(result))
                        self.error_count += 1
                    else:
                        results.append(result)

                # Small delay between batches to prevent overwhelming
                if i + batch_size < len(operations):
                    await asyncio.sleep(0.01)

        return results

    async def parallel_execute(
        self,
        operations: dict[str, Callable[[], Any]],
        timeout: float = 30.0,
        operation_name: str = "parallel_operation",
    ) -> dict[str, Any]:
        """Execute operations in parallel with timeout."""
        async with self.performance_context(f"{operation_name}_parallel"):
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(
                        *[op() for op in operations.values()],
                        return_exceptions=True
                    ),
                    timeout=timeout
                )

                # Map results back to operation names
                result_dict = {}
                for (name, _), result in zip(operations.items(), results, strict=False):
                    if isinstance(result, Exception):
                        logger.error(f"Parallel operation {name} failed", error=str(result))
                        self.error_count += 1
                        result_dict[name] = None
                    else:
                        result_dict[name] = result

                return result_dict

            except TimeoutError:
                logger.error(f"Parallel operations timed out after {timeout}s")
                self.error_count += 1
                raise ServiceError("Parallel operations timed out")

    def get_service_stats(self) -> dict[str, Any]:
        """Get comprehensive service statistics."""
        uptime = time.time() - self.start_time

        stats = {
            "uptime_seconds": round(uptime, 2),
            "operation_count": self.operation_count,
            "error_count": self.error_count,
            "error_rate_percent": (
                round(self.error_count / self.operation_count * 100, 2)
                if self.operation_count > 0 else 0
            ),
            "operations_per_second": round(self.operation_count / uptime, 2) if uptime > 0 else 0,
        }

        # Add performance monitoring stats
        if self.performance_monitor:
            stats["performance"] = self.performance_monitor.get_stats()

        # Add cache stats
        if self.cache_strategy:
            stats["cache"] = self.cache_strategy.get_cache_stats()

        # Add connection pool stats
        stats["connection_pool"] = self.connection_pool.get_pool_stats()

        return stats

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Perform service health check."""
        pass

    async def warm_up(self) -> None:
        """Warm up service (override in subclasses)."""
        logger.info(f"Warming up service: {self.__class__.__name__}")

    async def shutdown(self) -> None:
        """Graceful shutdown (override in subclasses)."""
        logger.info(f"Shutting down service: {self.__class__.__name__}")


class HighThroughputMixin:
    """Mixin for high-throughput processing optimizations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processing_queue = asyncio.Queue(maxsize=1000)
        self.batch_processor_task = None
        self.batch_size = 50
        self.batch_timeout = 5.0  # Process batch every 5 seconds

    async def start_batch_processor(self) -> None:
        """Start background batch processor."""
        self.batch_processor_task = asyncio.create_task(self._batch_processor())

    async def stop_batch_processor(self) -> None:
        """Stop background batch processor."""
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass

    async def _batch_processor(self) -> None:
        """Background task to process items in batches."""
        batch = []

        while True:
            try:
                # Wait for items with timeout
                try:
                    item = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=self.batch_timeout
                    )
                    batch.append(item)
                except TimeoutError:
                    # Process current batch on timeout
                    if batch:
                        await self._process_batch(batch)
                        batch = []
                    continue

                # Process batch when it reaches target size
                if len(batch) >= self.batch_size:
                    await self._process_batch(batch)
                    batch = []

            except asyncio.CancelledError:
                # Process remaining items before shutdown
                if batch:
                    await self._process_batch(batch)
                break
            except Exception as e:
                logger.error("Batch processor error", error=str(e))
                await asyncio.sleep(1)  # Brief delay before retry

    async def _process_batch(self, batch: list[Any]) -> None:
        """Process a batch of items (override in subclass)."""
        logger.debug(f"Processing batch of {len(batch)} items")

    async def queue_for_processing(self, item: Any) -> None:
        """Queue item for batch processing."""
        try:
            await self.processing_queue.put(item)
        except asyncio.QueueFull:
            logger.warning("Processing queue full, dropping item")


class CacheOptimizedMixin:
    """Mixin for advanced caching optimizations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_warming_tasks = []

    async def warm_cache(self, cache_keys: list[str]) -> None:
        """Warm up cache with commonly accessed data."""
        logger.info(f"Warming cache with {len(cache_keys)} keys")

        async def warm_key(key: str) -> None:
            try:
                # Check if key exists in cache
                if not await self.cache_service.exists(key):
                    # Key doesn't exist, trigger cache population
                    await self._populate_cache_key(key)
            except Exception as e:
                logger.warning(f"Failed to warm cache key {key}", error=str(e))

        # Warm keys concurrently
        await asyncio.gather(*[warm_key(key) for key in cache_keys], return_exceptions=True)

    async def _populate_cache_key(self, key: str) -> None:
        """Populate specific cache key (override in subclass)."""
        pass

    async def invalidate_cache_pattern(self, pattern: str) -> int:
        """Invalidate cache keys matching pattern."""
        try:
            if hasattr(self.cache_service, 'delete_pattern'):
                return await self.cache_service.delete_pattern(pattern)
            else:
                logger.warning("Cache service doesn't support pattern deletion")
                return 0
        except Exception as e:
            logger.error(f"Failed to invalidate cache pattern {pattern}", error=str(e))
            return 0
