"""Enhanced cache service with multi-level caching for high-throughput analytics.

This module provides a production-ready caching solution with:
- L1 in-memory caching with configurable LRU eviction
- L2 Redis caching with pipeline optimization
- Smart cache invalidation strategies
- Performance monitoring and metrics
- Cache warming and preloading capabilities
- Optimized for 10TB/day analytics workload
"""

import asyncio
import builtins
import hashlib
import json
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from redis.asyncio import Redis

from ..core.logging import get_logger

logger = get_logger(__name__)


class CacheMetrics:
    """Cache performance metrics collection."""

    def __init__(self):
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0
        self.invalidations = 0
        self.evictions = 0
        self.operation_times = defaultdict(list)
        self.start_time = time.time()

    def record_l1_hit(self):
        self.l1_hits += 1

    def record_l1_miss(self):
        self.l1_misses += 1

    def record_l2_hit(self):
        self.l2_hits += 1

    def record_l2_miss(self):
        self.l2_misses += 1

    def record_invalidation(self):
        self.invalidations += 1

    def record_eviction(self):
        self.evictions += 1

    def record_operation_time(self, operation: str, duration: float):
        self.operation_times[operation].append(duration)
        # Keep only last 1000 measurements per operation
        if len(self.operation_times[operation]) > 1000:
            self.operation_times[operation] = self.operation_times[operation][-1000:]

    @property
    def l1_hit_rate(self) -> float:
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0

    @property
    def l2_hit_rate(self) -> float:
        total = self.l2_hits + self.l2_misses
        return self.l2_hits / total if total > 0 else 0.0

    @property
    def overall_hit_rate(self) -> float:
        total_hits = self.l1_hits + self.l2_hits
        total_requests = self.l1_hits + self.l1_misses + self.l2_hits + self.l2_misses
        return total_hits / total_requests if total_requests > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        uptime = time.time() - self.start_time

        avg_times = {}
        for op, times in self.operation_times.items():
            if times:
                avg_times[f"{op}_avg_ms"] = sum(times) / len(times) * 1000
                avg_times[f"{op}_p95_ms"] = sorted(times)[int(len(times) * 0.95)] * 1000 if len(times) > 20 else avg_times[f"{op}_avg_ms"]

        return {
            "l1_hits": self.l1_hits,
            "l1_misses": self.l1_misses,
            "l1_hit_rate": self.l1_hit_rate,
            "l2_hits": self.l2_hits,
            "l2_misses": self.l2_misses,
            "l2_hit_rate": self.l2_hit_rate,
            "overall_hit_rate": self.overall_hit_rate,
            "invalidations": self.invalidations,
            "evictions": self.evictions,
            "uptime_seconds": uptime,
            "operations_per_second": (self.l1_hits + self.l1_misses + self.l2_hits + self.l2_misses) / max(uptime, 1),
            **avg_times
        }


class CacheEntry:
    """Cache entry with metadata and TTL support."""

    def __init__(self, value: Any, ttl: int | None = None, tags: set[str] | None = None):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.tags = tags or set()
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self):
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = time.time()

    def should_evict(self, max_age: float = 3600) -> bool:
        """Check if entry should be evicted based on age and access pattern."""
        age = time.time() - self.created_at
        idle_time = time.time() - self.last_accessed

        # Evict if too old or not accessed recently
        return age > max_age or (idle_time > 300 and self.access_count < 2)


class L1Cache:
    """High-performance in-memory cache with LRU eviction."""

    def __init__(self, max_size: int = 10000, max_memory_mb: int = 512):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
        self._current_memory = 0
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.is_expired():
                if entry and entry.is_expired():
                    await self._remove_key(key)
                return None

            entry.touch()
            self._update_access_order(key)
            return entry.value

    async def set(self, key: str, value: Any, ttl: int | None = None, tags: set[str] | None = None) -> bool:
        async with self._lock:
            # Estimate memory usage
            estimated_size = self._estimate_size(value)

            # Check if we need to evict
            await self._ensure_capacity(estimated_size)

            # Remove old entry if exists
            if key in self._cache:
                await self._remove_key(key)

            # Add new entry
            entry = CacheEntry(value, ttl, tags)
            self._cache[key] = entry
            self._access_order.append(key)
            self._current_memory += estimated_size

            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                await self._remove_key(key)
                return True
            return False

    async def invalidate_by_tags(self, tags: builtins.set[str]) -> int:
        """Invalidate all entries with matching tags."""
        async with self._lock:
            keys_to_remove = []
            for key, entry in self._cache.items():
                if entry.tags and entry.tags.intersection(tags):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                await self._remove_key(key)

            return len(keys_to_remove)

    async def cleanup_expired(self) -> int:
        """Remove expired entries."""
        async with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired() or entry.should_evict():
                    expired_keys.append(key)

            for key in expired_keys:
                await self._remove_key(key)

            return len(expired_keys)

    async def _remove_key(self, key: str):
        """Remove key from cache and update tracking."""
        if key in self._cache:
            entry = self._cache[key]
            estimated_size = self._estimate_size(entry.value)
            del self._cache[key]
            self._current_memory = max(0, self._current_memory - estimated_size)

        if key in self._access_order:
            self._access_order.remove(key)

    def _update_access_order(self, key: str):
        """Update LRU order."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    async def _ensure_capacity(self, new_item_size: int):
        """Ensure we have capacity for new item."""
        # Check size limit
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order[0]
            await self._remove_key(oldest_key)

        # Check memory limit
        while (self._current_memory + new_item_size > self.max_memory_bytes and
               self._access_order):
            oldest_key = self._access_order[0]
            await self._remove_key(oldest_key)

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v)
                          for k, v in value.items())
            else:
                # Fallback estimate
                return len(str(value)) * 2
        except Exception:
            return 1024  # Conservative fallback

    def get_stats(self) -> dict[str, Any]:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "memory_usage_mb": self._current_memory / (1024 * 1024),
            "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            "memory_utilization": self._current_memory / self.max_memory_bytes if self.max_memory_bytes > 0 else 0
        }


class EnhancedCacheService:
    """Enhanced multi-level cache service optimized for high-throughput analytics.
    
    Features:
    - L1 in-memory cache with LRU eviction (10,000 items, 512MB limit)
    - L2 Redis cache with pipeline optimization
    - Smart cache invalidation with tag-based strategies
    - Performance monitoring and metrics collection
    - Cache warming and preloading for analytics queries
    - Optimized for 10TB/day data processing workload
    """

    def __init__(self,
                 redis_client: Redis,
                 l1_max_size: int = 10000,
                 l1_max_memory_mb: int = 512,
                 enable_l1: bool = True,
                 key_prefix: str = "its_camera_ai") -> None:
        self.redis = redis_client
        self.enable_l1 = enable_l1
        self.key_prefix = key_prefix

        # L1 cache setup
        self.l1_cache = L1Cache(l1_max_size, l1_max_memory_mb) if enable_l1 else None

        # Metrics and monitoring
        self.metrics = CacheMetrics()

        # Cache invalidation tracking
        self._invalidation_patterns: dict[str, set[str]] = defaultdict(set)

        # Background tasks
        self._cleanup_task: asyncio.Task | None = None
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        if self.l1_cache:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        """Periodic cleanup of expired entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Clean every 5 minutes
                if self.l1_cache:
                    evicted = await self.l1_cache.cleanup_expired()
                    if evicted > 0:
                        self.metrics.record_eviction()
                        logger.debug(f"L1 cache cleanup: evicted {evicted} entries")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    def _build_key(self, key: str) -> str:
        """Build prefixed cache key."""
        return f"{self.key_prefix}:{key}"

    def _hash_key(self, key: str) -> str:
        """Create a hash of the key for very long keys."""
        if len(key) > 200:  # Redis key limit considerations
            return hashlib.md5(key.encode()).hexdigest()
        return key

    async def get(self, key: str, decode_json: bool = False) -> Any:
        """Get value with multi-level cache lookup.

        Args:
            key: Cache key
            decode_json: Whether to decode JSON automatically

        Returns:
            Value or None if not found
        """
        start_time = time.time()
        hashed_key = self._hash_key(key)

        try:
            # Try L1 cache first
            if self.l1_cache:
                l1_value = await self.l1_cache.get(hashed_key)
                if l1_value is not None:
                    self.metrics.record_l1_hit()
                    self.metrics.record_operation_time("get", time.time() - start_time)
                    return l1_value
                else:
                    self.metrics.record_l1_miss()

            # Try L2 (Redis) cache
            prefixed_key = self._build_key(hashed_key)
            redis_value = await self.redis.get(prefixed_key)

            if redis_value is not None:
                self.metrics.record_l2_hit()

                # Decode if needed
                if decode_json:
                    try:
                        decoded_value = json.loads(redis_value)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON in Redis cache", key=key)
                        decoded_value = redis_value
                else:
                    decoded_value = redis_value

                # Populate L1 cache
                if self.l1_cache:
                    await self.l1_cache.set(hashed_key, decoded_value)

                self.metrics.record_operation_time("get", time.time() - start_time)
                return decoded_value
            else:
                self.metrics.record_l2_miss()
                self.metrics.record_operation_time("get", time.time() - start_time)
                return None

        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            self.metrics.record_operation_time("get_error", time.time() - start_time)
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None,
                 tags: set[str] | None = None, encode_json: bool = False) -> bool:
        """Set value in multi-level cache.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds
            tags: Tags for cache invalidation
            encode_json: Whether to encode as JSON

        Returns:
            True if successful
        """
        start_time = time.time()
        hashed_key = self._hash_key(key)

        try:
            # Prepare value for storage
            if encode_json:
                try:
                    redis_value = json.dumps(value, default=str)
                    cache_value = value  # Store original for L1
                except (TypeError, ValueError) as e:
                    logger.error("JSON serialization error", key=key, error=str(e))
                    return False
            else:
                redis_value = str(value)
                cache_value = value

            # Store in L1 cache
            if self.l1_cache:
                await self.l1_cache.set(hashed_key, cache_value, ttl, tags)

            # Store in L2 (Redis) cache
            prefixed_key = self._build_key(hashed_key)
            success = await self.redis.set(prefixed_key, redis_value, ex=ttl)

            # Track invalidation patterns
            if tags:
                for tag in tags:
                    self._invalidation_patterns[tag].add(hashed_key)

            self.metrics.record_operation_time("set", time.time() - start_time)
            return success

        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))
            self.metrics.record_operation_time("set_error", time.time() - start_time)
            return False

    async def get_json(self, key: str) -> Any:
        """Get JSON value by key (convenience method).

        Args:
            key: Cache key

        Returns:
            Parsed JSON value or None
        """
        return await self.get(key, decode_json=True)

    async def set_json(self, key: str, value: Any, ttl: int | None = None,
                      tags: builtins.set[str] | None = None) -> bool:
        """Set JSON value by key (convenience method).

        Args:
            key: Cache key
            value: Value to serialize and store
            ttl: Time to live in seconds
            tags: Tags for cache invalidation

        Returns:
            True if successful
        """
        return await self.set(key, value, ttl, tags, encode_json=True)

    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels.

        Args:
            key: Cache key

        Returns:
            True if key was deleted from any level
        """
        start_time = time.time()
        hashed_key = self._hash_key(key)
        deleted = False

        try:
            # Delete from L1 cache
            if self.l1_cache:
                l1_deleted = await self.l1_cache.delete(hashed_key)
                deleted = deleted or l1_deleted

            # Delete from L2 (Redis) cache
            prefixed_key = self._build_key(hashed_key)
            result = await self.redis.delete(prefixed_key)
            deleted = deleted or (result > 0)

            self.metrics.record_operation_time("delete", time.time() - start_time)
            return deleted

        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))
            self.metrics.record_operation_time("delete_error", time.time() - start_time)
            return False

    async def invalidate_by_tags(self, tags: builtins.set[str]) -> int:
        """Invalidate cache entries by tags.

        Args:
            tags: Set of tags to invalidate

        Returns:
            Number of entries invalidated
        """
        start_time = time.time()
        total_invalidated = 0

        try:
            # Invalidate L1 cache by tags
            if self.l1_cache:
                l1_invalidated = await self.l1_cache.invalidate_by_tags(tags)
                total_invalidated += l1_invalidated

            # Invalidate L2 cache by patterns
            keys_to_delete = set()
            for tag in tags:
                if tag in self._invalidation_patterns:
                    keys_to_delete.update(self._invalidation_patterns[tag])
                    del self._invalidation_patterns[tag]

            if keys_to_delete:
                # Use pipeline for efficient batch deletion
                pipe = self.redis.pipeline()
                for key in keys_to_delete:
                    prefixed_key = self._build_key(key)
                    pipe.delete(prefixed_key)

                results = await pipe.execute()
                l2_invalidated = sum(1 for result in results if result > 0)
                total_invalidated += l2_invalidated

            self.metrics.record_invalidation()
            self.metrics.record_operation_time("invalidate", time.time() - start_time)

            logger.debug(f"Invalidated {total_invalidated} entries for tags: {tags}")
            return total_invalidated

        except Exception as e:
            logger.error("Cache invalidation error", tags=tags, error=str(e))
            return 0

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., "analytics:camera_*")

        Returns:
            Number of entries invalidated
        """
        start_time = time.time()

        try:
            prefixed_pattern = self._build_key(pattern)
            keys = []

            # Scan for matching keys
            async for key in self.redis.scan_iter(match=prefixed_pattern, count=1000):
                keys.append(key)

            if keys:
                # Use pipeline for efficient batch deletion
                pipe = self.redis.pipeline()
                for key in keys:
                    pipe.delete(key)

                results = await pipe.execute()
                invalidated = sum(1 for result in results if result > 0)

                # Also clear from L1 cache
                if self.l1_cache:
                    for key in keys:
                        # Remove prefix to get original key
                        original_key = key.decode() if isinstance(key, bytes) else key
                        if original_key.startswith(self.key_prefix + ":"):
                            cache_key = original_key[len(self.key_prefix) + 1:]
                            await self.l1_cache.delete(cache_key)

                self.metrics.record_invalidation()
                self.metrics.record_operation_time("invalidate_pattern", time.time() - start_time)

                logger.debug(f"Invalidated {invalidated} entries for pattern: {pattern}")
                return invalidated

            return 0

        except Exception as e:
            logger.error("Cache pattern invalidation error", pattern=pattern, error=str(e))
            return 0

    async def warm_cache(self, cache_warmer: Callable[[str], Any], keys: list[str],
                        ttl: int | None = None, batch_size: int = 100) -> int:
        """Warm cache with pre-computed values.

        Args:
            cache_warmer: Function to generate values for keys
            keys: List of keys to warm
            ttl: TTL for warmed entries
            batch_size: Batch size for operations

        Returns:
            Number of entries successfully warmed
        """
        start_time = time.time()
        warmed_count = 0

        try:
            # Process in batches to avoid overwhelming the system
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]

                # Use pipeline for batch operations
                pipe = self.redis.pipeline()
                batch_values = []

                for key in batch_keys:
                    try:
                        value = await cache_warmer(key)
                        if value is not None:
                            batch_values.append((key, value))
                    except Exception as e:
                        logger.warning(f"Cache warmer failed for key {key}: {e}")

                # Store in Redis
                for key, value in batch_values:
                    hashed_key = self._hash_key(key)
                    prefixed_key = self._build_key(hashed_key)

                    if isinstance(value, (dict, list)):
                        redis_value = json.dumps(value, default=str)
                    else:
                        redis_value = str(value)

                    pipe.set(prefixed_key, redis_value, ex=ttl)

                if batch_values:
                    results = await pipe.execute()
                    warmed_count += sum(1 for result in results if result)

                # Also warm L1 cache
                if self.l1_cache:
                    for key, value in batch_values:
                        hashed_key = self._hash_key(key)
                        await self.l1_cache.set(hashed_key, value, ttl)

                # Small delay between batches to avoid overwhelming
                if i + batch_size < len(keys):
                    await asyncio.sleep(0.01)

            self.metrics.record_operation_time("warm_cache", time.time() - start_time)
            logger.info(f"Cache warming completed: {warmed_count}/{len(keys)} entries")
            return warmed_count

        except Exception as e:
            logger.error("Cache warming error", error=str(e))
            return warmed_count

    async def get_counter(self, key: str, _window: int) -> int:
        """Get counter value for rate limiting.

        Args:
            key: Counter key
            window: Time window in seconds (unused but kept for compatibility)

        Returns:
            Current counter value
        """
        try:
            prefixed_key = self._build_key(key)
            value = await self.redis.get(prefixed_key)
            return int(value) if value else 0
        except Exception:
            return 0

    async def increment_counter(self, key: str, window: int) -> int:
        """Increment counter for rate limiting.

        Args:
            key: Counter key
            window: Time window in seconds

        Returns:
            New counter value
        """
        try:
            prefixed_key = self._build_key(key)
            pipe = self.redis.pipeline()
            pipe.incr(prefixed_key)
            pipe.expire(prefixed_key, window)
            results = await pipe.execute()
            return results[0] if results else 1
        except Exception:
            return 1

    async def multi_get(self, keys: list[str], decode_json: bool = False) -> dict[str, Any]:
        """Get multiple values efficiently using pipeline.

        Args:
            keys: List of cache keys
            decode_json: Whether to decode JSON automatically

        Returns:
            Dictionary of key-value pairs
        """
        start_time = time.time()
        result = {}

        if not keys:
            return result

        try:
            # Check L1 cache first
            l1_misses = []
            if self.l1_cache:
                for key in keys:
                    hashed_key = self._hash_key(key)
                    l1_value = await self.l1_cache.get(hashed_key)
                    if l1_value is not None:
                        result[key] = l1_value
                        self.metrics.record_l1_hit()
                    else:
                        l1_misses.append(key)
                        self.metrics.record_l1_miss()
            else:
                l1_misses = keys

            # Get remaining keys from Redis
            if l1_misses:
                pipe = self.redis.pipeline()
                for key in l1_misses:
                    hashed_key = self._hash_key(key)
                    prefixed_key = self._build_key(hashed_key)
                    pipe.get(prefixed_key)

                redis_results = await pipe.execute()

                for key, redis_value in zip(l1_misses, redis_results, strict=False):
                    if redis_value is not None:
                        self.metrics.record_l2_hit()

                        # Decode if needed
                        if decode_json:
                            try:
                                decoded_value = json.loads(redis_value)
                            except json.JSONDecodeError:
                                decoded_value = redis_value
                        else:
                            decoded_value = redis_value

                        result[key] = decoded_value

                        # Populate L1 cache
                        if self.l1_cache:
                            hashed_key = self._hash_key(key)
                            await self.l1_cache.set(hashed_key, decoded_value)
                    else:
                        self.metrics.record_l2_miss()

            self.metrics.record_operation_time("multi_get", time.time() - start_time)
            return result

        except Exception as e:
            logger.error("Multi-get error", keys_count=len(keys), error=str(e))
            return result

    async def multi_set(self, items: dict[str, Any], ttl: int | None = None,
                       tags: builtins.set[str] | None = None, encode_json: bool = False) -> int:
        """Set multiple values efficiently using pipeline.

        Args:
            items: Dictionary of key-value pairs
            ttl: Time to live in seconds
            tags: Tags for cache invalidation
            encode_json: Whether to encode as JSON

        Returns:
            Number of successfully set items
        """
        start_time = time.time()

        if not items:
            return 0

        try:
            # Prepare items for Redis
            pipe = self.redis.pipeline()
            prepared_items = []

            for key, value in items.items():
                hashed_key = self._hash_key(key)
                prefixed_key = self._build_key(hashed_key)

                if encode_json:
                    try:
                        redis_value = json.dumps(value, default=str)
                        cache_value = value
                    except (TypeError, ValueError):
                        continue  # Skip invalid items
                else:
                    redis_value = str(value)
                    cache_value = value

                prepared_items.append((key, hashed_key, cache_value))
                pipe.set(prefixed_key, redis_value, ex=ttl)

            # Execute Redis pipeline
            results = await pipe.execute()
            success_count = sum(1 for result in results if result)

            # Store in L1 cache
            if self.l1_cache:
                for key, hashed_key, cache_value in prepared_items:
                    await self.l1_cache.set(hashed_key, cache_value, ttl, tags)

            # Track invalidation patterns
            if tags:
                for key, hashed_key, _ in prepared_items:
                    for tag in tags:
                        self._invalidation_patterns[tag].add(hashed_key)

            self.metrics.record_operation_time("multi_set", time.time() - start_time)
            return success_count

        except Exception as e:
            logger.error("Multi-set error", items_count=len(items), error=str(e))
            return 0

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive cache performance metrics."""
        metrics = self.metrics.get_stats()

        if self.l1_cache:
            metrics["l1_cache"] = self.l1_cache.get_stats()

        # Redis connection info
        try:
            redis_info = {
                "connected": True,
                "db_size": "unknown",  # Would need DBSIZE command
                "memory_usage": "unknown"  # Would need INFO memory
            }
        except Exception:
            redis_info = {"connected": False}

        metrics["redis"] = redis_info
        return metrics

    async def clear_all(self) -> bool:
        """Clear all cache entries (use with caution)."""
        try:
            # Clear L1 cache
            if self.l1_cache:
                self.l1_cache._cache.clear()
                self.l1_cache._access_order.clear()
                self.l1_cache._current_memory = 0

            # Clear Redis cache (only keys with our prefix)
            keys = []
            pattern = self._build_key("*")
            async for key in self.redis.scan_iter(match=pattern, count=1000):
                keys.append(key)

            if keys:
                await self.redis.delete(*keys)

            # Clear invalidation patterns
            self._invalidation_patterns.clear()

            logger.info(f"Cleared cache: {len(keys)} Redis keys")
            return True

        except Exception as e:
            logger.error("Cache clear error", error=str(e))
            return False

    async def shutdown(self):
        """Shutdown cache service and cleanup resources."""
        try:
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Final cleanup
            if self.l1_cache:
                await self.l1_cache.cleanup_expired()

            logger.info("Cache service shutdown completed")

        except Exception as e:
            logger.error("Cache shutdown error", error=str(e))


# Backward compatibility alias
CacheService = EnhancedCacheService
