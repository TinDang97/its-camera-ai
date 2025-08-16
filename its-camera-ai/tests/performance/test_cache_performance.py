"""Comprehensive cache performance tests for multi-level caching system.

This module tests the enhanced caching system's performance under various scenarios:
- High-throughput read/write operations
- Cache hit rate optimization
- Multi-level cache coherency
- Memory usage and eviction policies
- Cache invalidation performance
- Concurrent access patterns
"""

import asyncio
import json
import random
import time
import pytest
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock

import redis.asyncio as redis

from src.its_camera_ai.services.cache import EnhancedCacheService, CacheMetrics, L1Cache
from src.its_camera_ai.core.config import Settings


class TestCachePerformance:
    """Performance tests for the enhanced cache service."""
    
    @pytest.fixture
    async def redis_client(self):
        """Mock Redis client for testing."""
        mock_redis = AsyncMock(spec=redis.Redis)
        
        # Mock Redis operations with realistic behavior
        storage = {}
        
        async def mock_get(key):
            return storage.get(key)
        
        async def mock_set(key, value, ex=None):
            storage[key] = value
            return True
        
        async def mock_delete(*keys):
            deleted = 0
            for key in keys:
                if key in storage:
                    del storage[key]
                    deleted += 1
            return deleted
        
        async def mock_pipeline():
            pipe_mock = AsyncMock()
            pipe_commands = []
            
            def mock_pipe_get(key):
                pipe_commands.append(('get', key))
                return pipe_mock
            
            def mock_pipe_set(key, value, ex=None):
                pipe_commands.append(('set', key, value))
                return pipe_mock
            
            def mock_pipe_delete(key):
                pipe_commands.append(('delete', key))
                return pipe_mock
            
            async def mock_execute():
                results = []
                for cmd in pipe_commands:
                    if cmd[0] == 'get':
                        results.append(storage.get(cmd[1]))
                    elif cmd[0] == 'set':
                        storage[cmd[1]] = cmd[2]
                        results.append(True)
                    elif cmd[0] == 'delete':
                        if cmd[1] in storage:
                            del storage[cmd[1]]
                            results.append(1)
                        else:
                            results.append(0)
                return results
            
            pipe_mock.get = mock_pipe_get
            pipe_mock.set = mock_pipe_set
            pipe_mock.delete = mock_pipe_delete
            pipe_mock.execute = mock_execute
            pipe_mock.incr = lambda key: pipe_mock
            pipe_mock.expire = lambda key, ttl: pipe_mock
            
            return pipe_mock
        
        async def mock_scan_iter(match=None, count=None):
            for key in storage.keys():
                if match is None or match.replace('*', '') in key:
                    yield key
        
        mock_redis.get = mock_get
        mock_redis.set = mock_set
        mock_redis.delete = mock_delete
        mock_redis.pipeline = mock_pipeline
        mock_redis.scan_iter = mock_scan_iter
        
        return mock_redis
    
    @pytest.fixture
    async def cache_service(self, redis_client):
        """Create cache service for testing."""
        return EnhancedCacheService(
            redis_client=redis_client,
            l1_max_size=1000,
            l1_max_memory_mb=10,
            enable_l1=True,
            key_prefix="test"
        )
    
    @pytest.mark.asyncio
    async def test_high_throughput_operations(self, cache_service):
        """Test cache performance under high throughput."""
        # Test configuration
        num_operations = 1000
        concurrent_workers = 50
        
        start_time = time.time()
        
        async def worker(worker_id: int):
            operations = 0
            for i in range(num_operations // concurrent_workers):
                key = f"worker_{worker_id}_key_{i}"
                value = f"value_{worker_id}_{i}" * 100  # ~1KB values
                
                # Write operation
                await cache_service.set(key, value, ttl=300, encode_json=False)
                operations += 1
                
                # Read operation
                result = await cache_service.get(key)
                assert result == value
                operations += 1
                
                # Update operation
                new_value = f"updated_{value}"
                await cache_service.set(key, new_value, ttl=300, encode_json=False)
                operations += 1
            
            return operations
        
        # Run concurrent workers
        tasks = [worker(i) for i in range(concurrent_workers)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        total_operations = sum(results)
        
        # Performance assertions
        ops_per_second = total_operations / total_time
        
        print(f"\\nHigh Throughput Performance:")
        print(f"Total operations: {total_operations}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Operations per second: {ops_per_second:.2f}")
        print(f"Average latency: {(total_time / total_operations) * 1000:.2f}ms")
        
        # Performance requirements
        assert ops_per_second > 5000, f"Throughput too low: {ops_per_second} ops/sec"
        assert total_time < 30, f"Total time too high: {total_time}s"
        
        # Check cache metrics
        metrics = cache_service.get_metrics()
        assert metrics["overall_hit_rate"] > 0.8, f"Hit rate too low: {metrics['overall_hit_rate']}"
    
    @pytest.mark.asyncio
    async def test_cache_hit_rate_optimization(self, cache_service):
        """Test cache hit rate under realistic access patterns."""
        # Generate test data with realistic distribution
        num_keys = 500
        total_accesses = 2000
        
        # Create keys with different access frequencies (Zipf distribution)
        keys = [f"key_{i}" for i in range(num_keys)]
        values = [f"value_{i}" * random.randint(10, 100) for i in range(num_keys)]
        
        # Pre-populate cache
        for key, value in zip(keys, values):
            await cache_service.set(key, value, ttl=600)
        
        # Simulate realistic access pattern (80/20 rule)
        hot_keys = keys[:num_keys // 5]  # 20% of keys
        cold_keys = keys[num_keys // 5:]  # 80% of keys
        
        start_time = time.time()
        
        for _ in range(total_accesses):
            # 80% of accesses go to 20% of keys (hot data)
            if random.random() < 0.8:
                key = random.choice(hot_keys)
            else:
                key = random.choice(cold_keys)
            
            result = await cache_service.get(key)
            assert result is not None
        
        access_time = time.time() - start_time
        
        # Get performance metrics
        metrics = cache_service.get_metrics()
        
        print(f"\\nCache Hit Rate Performance:")
        print(f"Total accesses: {total_accesses}")
        print(f"Access time: {access_time:.2f}s")
        print(f"L1 hit rate: {metrics['l1_hit_rate']:.1%}")
        print(f"L2 hit rate: {metrics['l2_hit_rate']:.1%}")
        print(f"Overall hit rate: {metrics['overall_hit_rate']:.1%}")
        print(f"Average access time: {access_time / total_accesses * 1000:.2f}ms")
        
        # Performance requirements
        assert metrics["overall_hit_rate"] > 0.95, f"Overall hit rate too low: {metrics['overall_hit_rate']}"
        assert metrics["l1_hit_rate"] > 0.7, f"L1 hit rate too low: {metrics['l1_hit_rate']}"
        assert access_time / total_accesses < 0.01, "Average access time too high"
    
    @pytest.mark.asyncio
    async def test_memory_usage_and_eviction(self, cache_service):
        """Test memory usage patterns and LRU eviction."""
        # Test L1 cache memory limits
        l1_cache = cache_service.l1_cache
        initial_memory = l1_cache._current_memory
        
        # Add data until we hit memory limits
        large_values = []
        for i in range(100):
            # Create increasingly large values
            value = "x" * (1024 * (i + 1))  # 1KB, 2KB, 3KB, etc.
            large_values.append(value)
            
            await cache_service.set(f"large_key_{i}", value, ttl=300)
        
        # Check memory management
        final_memory = l1_cache._current_memory
        max_memory = l1_cache.max_memory_bytes
        
        print(f"\\nMemory Usage Performance:")
        print(f"Initial memory: {initial_memory / 1024:.1f} KB")
        print(f"Final memory: {final_memory / 1024:.1f} KB")
        print(f"Max memory: {max_memory / 1024 / 1024:.1f} MB")
        print(f"Memory utilization: {final_memory / max_memory:.1%}")
        print(f"L1 cache size: {len(l1_cache._cache)}")
        
        # Verify memory limits are respected
        assert final_memory <= max_memory, f"Memory limit exceeded: {final_memory} > {max_memory}"
        
        # Test access pattern affects eviction
        access_start = time.time()
        
        # Access some keys to test LRU
        for i in range(0, 50, 5):  # Access every 5th key
            result = await cache_service.get(f"large_key_{i}")
            assert result is not None
        
        access_time = time.time() - access_start
        
        # Add more data to trigger eviction
        for i in range(100, 150):
            value = "y" * 2048  # 2KB values
            await cache_service.set(f"eviction_test_{i}", value, ttl=300)
        
        # Verify some old keys were evicted but accessed keys remain
        evicted_count = 0
        accessed_count = 0
        
        for i in range(50):
            result = await cache_service.l1_cache.get(f"large_key_{i}")
            if result is None:
                evicted_count += 1
            elif i % 5 == 0:  # Keys we accessed
                accessed_count += 1
        
        print(f"Evicted keys: {evicted_count}")
        print(f"Accessed keys remaining: {accessed_count}")
        
        # LRU should prefer recently accessed keys
        assert accessed_count > evicted_count / 2, "LRU eviction not working correctly"
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_performance(self, cache_service):
        """Test cache invalidation performance with tags."""
        # Create test data with tags
        num_cameras = 20
        num_time_windows = 5
        
        setup_start = time.time()
        
        for camera_id in range(num_cameras):
            for window in range(num_time_windows):
                key = f"analytics:camera_{camera_id}:window_{window}"
                value = {"camera_id": camera_id, "window": window, "data": "x" * 1000}
                tags = {f"camera:{camera_id}", f"analytics", f"window:{window}"}
                
                await cache_service.set_json(key, value, ttl=3600, tags=tags)
        
        setup_time = time.time() - setup_start
        total_entries = num_cameras * num_time_windows
        
        print(f"\\nCache Invalidation Performance:")
        print(f"Setup time: {setup_time:.2f}s for {total_entries} entries")
        
        # Test tag-based invalidation
        invalidation_tests = [
            ("single_camera", {f"camera:5"}, num_time_windows),
            ("single_window", {f"window:2"}, num_cameras),
            ("analytics_all", {"analytics"}, total_entries),
        ]
        
        for test_name, tags, expected_count in invalidation_tests:
            start_time = time.time()
            invalidated = await cache_service.invalidate_by_tags(tags)
            invalidation_time = time.time() - start_time
            
            print(f"{test_name}: {invalidated} entries in {invalidation_time:.3f}s")
            assert invalidated == expected_count, f"Expected {expected_count}, got {invalidated}"
            assert invalidation_time < 1.0, f"Invalidation too slow: {invalidation_time}s"
    
    @pytest.mark.asyncio
    async def test_multi_get_set_performance(self, cache_service):
        """Test batch operations performance."""
        # Test multi-set performance
        num_items = 1000
        items = {
            f"batch_key_{i}": {"id": i, "data": "x" * random.randint(100, 1000)}
            for i in range(num_items)
        }
        
        # Single operations baseline
        single_start = time.time()
        for key, value in list(items.items())[:100]:  # Test with subset
            await cache_service.set_json(key, value, ttl=300)
        single_time = time.time() - single_start
        
        # Batch operations
        batch_start = time.time()
        batch_count = await cache_service.multi_set(items, ttl=300, encode_json=True)
        batch_time = time.time() - batch_start
        
        print(f"\\nBatch Operations Performance:")
        print(f"Single operations (100 items): {single_time:.3f}s")
        print(f"Batch operations ({num_items} items): {batch_time:.3f}s")
        print(f"Batch success rate: {batch_count}/{num_items} ({batch_count/num_items:.1%})")
        print(f"Batch speedup: {(single_time/100*num_items)/batch_time:.1f}x")
        
        assert batch_count == num_items, f"Not all items were set: {batch_count}/{num_items}"
        assert batch_time < single_time * 2, "Batch operations not efficient enough"
        
        # Test multi-get performance
        keys = list(items.keys())
        
        multi_get_start = time.time()
        results = await cache_service.multi_get(keys, decode_json=True)
        multi_get_time = time.time() - multi_get_start
        
        print(f"Multi-get ({len(keys)} keys): {multi_get_time:.3f}s")
        print(f"Multi-get rate: {len(keys)/multi_get_time:.0f} keys/sec")
        
        assert len(results) == num_items, f"Not all items retrieved: {len(results)}/{num_items}"
        assert multi_get_time < 5.0, f"Multi-get too slow: {multi_get_time}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self, cache_service):
        """Test cache safety under concurrent access."""
        num_workers = 20
        operations_per_worker = 100
        shared_keys = [f"shared_key_{i}" for i in range(10)]
        
        start_time = time.time()
        
        async def concurrent_worker(worker_id: int):
            operations = 0
            for i in range(operations_per_worker):
                key = random.choice(shared_keys)
                
                if random.random() < 0.7:  # 70% reads
                    result = await cache_service.get(key)
                    operations += 1
                else:  # 30% writes
                    value = f"worker_{worker_id}_op_{i}_{random.randint(1000, 9999)}"
                    await cache_service.set(key, value, ttl=300)
                    operations += 1
                
                # Random invalidation
                if random.random() < 0.05:  # 5% invalidations
                    await cache_service.delete(key)
                    operations += 1
            
            return operations
        
        # Run concurrent workers
        tasks = [concurrent_worker(i) for i in range(num_workers)]
        worker_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Check for exceptions
        exceptions = [r for r in worker_results if isinstance(r, Exception)]
        successful_ops = [r for r in worker_results if not isinstance(r, Exception)]
        
        total_operations = sum(successful_ops)
        
        print(f"\\nConcurrent Access Performance:")
        print(f"Workers: {num_workers}")
        print(f"Total operations: {total_operations}")
        print(f"Exceptions: {len(exceptions)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Concurrent ops/sec: {total_operations/total_time:.0f}")
        
        # Safety assertions
        assert len(exceptions) == 0, f"Concurrent access caused {len(exceptions)} exceptions"
        assert total_operations > num_workers * operations_per_worker * 0.9, "Too many operations failed"
        
        # Performance assertions
        assert total_time < 30, f"Concurrent operations too slow: {total_time}s"
        
        # Verify cache integrity
        metrics = cache_service.get_metrics()
        print(f"Final cache metrics: {metrics}")
        
        # Cache should still be functional
        test_key = "integrity_test"
        test_value = "integrity_value"
        await cache_service.set(test_key, test_value)
        result = await cache_service.get(test_key)
        assert result == test_value, "Cache integrity compromised"
    
    @pytest.mark.asyncio
    async def test_cache_warming_performance(self, cache_service):
        """Test cache warming strategies."""
        # Simulate cache warmer function
        async def analytics_cache_warmer(key: str) -> Dict[str, Any]:
            await asyncio.sleep(0.001)  # Simulate database query
            return {
                "key": key,
                "metrics": {
                    "vehicle_count": random.randint(1, 50),
                    "avg_speed": random.uniform(20, 80),
                    "timestamp": time.time()
                }
            }
        
        # Test cache warming
        warm_keys = [f"analytics:camera_{i}:hourly" for i in range(100)]
        
        start_time = time.time()
        warmed_count = await cache_service.warm_cache(
            cache_warmer=analytics_cache_warmer,
            keys=warm_keys,
            ttl=3600,
            batch_size=20
        )
        warm_time = time.time() - start_time
        
        print(f"\\nCache Warming Performance:")
        print(f"Keys to warm: {len(warm_keys)}")
        print(f"Successfully warmed: {warmed_count}")
        print(f"Warming time: {warm_time:.2f}s")
        print(f"Warming rate: {warmed_count/warm_time:.1f} keys/sec")
        
        assert warmed_count == len(warm_keys), f"Not all keys warmed: {warmed_count}/{len(warm_keys)}"
        assert warm_time < 10, f"Cache warming too slow: {warm_time}s"
        
        # Verify warmed data is accessible
        verification_start = time.time()
        for key in warm_keys[:10]:  # Check first 10 keys
            result = await cache_service.get_json(key)
            assert result is not None, f"Warmed key not found: {key}"
            assert "metrics" in result, f"Invalid warmed data for key: {key}"
        
        verification_time = time.time() - verification_start
        print(f"Verification time: {verification_time:.3f}s for 10 keys")
        
        # Warmed data should be fast to access (L1 cache hits)
        assert verification_time < 0.1, "Warmed data access too slow"
    
    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, cache_service):
        """Test cache behavior under memory pressure."""
        l1_cache = cache_service.l1_cache
        max_memory = l1_cache.max_memory_bytes
        
        # Fill cache to near capacity
        large_values = []
        memory_used = 0
        
        while memory_used < max_memory * 0.8:  # Fill to 80%
            value = "x" * 10240  # 10KB values
            large_values.append(value)
            
            key = f"memory_test_{len(large_values)}"
            await cache_service.set(key, value, ttl=300)
            
            memory_used = l1_cache._current_memory
        
        print(f"\\nMemory Pressure Test:")
        print(f"Filled to: {memory_used / max_memory:.1%} of capacity")
        print(f"Cache entries: {len(l1_cache._cache)}")
        
        # Add more data to trigger aggressive eviction
        pressure_start = time.time()
        for i in range(100):
            key = f"pressure_test_{i}"
            value = "y" * 20480  # 20KB values
            await cache_service.set(key, value, ttl=300)
        
        pressure_time = time.time() - pressure_start
        final_memory = l1_cache._current_memory
        
        print(f"Pressure test time: {pressure_time:.3f}s")
        print(f"Final memory: {final_memory / max_memory:.1%} of capacity")
        print(f"Final cache entries: {len(l1_cache._cache)}")
        
        # Memory should stay within limits
        assert final_memory <= max_memory, f"Memory limit exceeded: {final_memory} > {max_memory}"
        
        # Cache should remain responsive
        assert pressure_time < 5.0, f"Cache too slow under pressure: {pressure_time}s"
        
        # Cache should still be functional
        test_key = "pressure_functionality_test"
        test_value = "test_value"
        await cache_service.set(test_key, test_value)
        result = await cache_service.get(test_key)
        assert result == test_value, "Cache non-functional under memory pressure"
    
    def test_cache_metrics_accuracy(self, cache_service):
        """Test cache metrics collection accuracy."""
        # Reset metrics
        cache_service.metrics = CacheMetrics()
        
        # Perform known operations
        asyncio.run(self._metrics_test_operations(cache_service))
        
        metrics = cache_service.get_metrics()
        
        print(f"\\nCache Metrics Accuracy:")
        print(f"L1 hits: {metrics['l1_hits']}")
        print(f"L1 misses: {metrics['l1_misses']}")
        print(f"L2 hits: {metrics['l2_hits']}")
        print(f"L2 misses: {metrics['l2_misses']}")
        print(f"Overall hit rate: {metrics['overall_hit_rate']:.1%}")
        
        # Verify metrics make sense
        assert metrics["l1_hits"] + metrics["l1_misses"] > 0, "No L1 operations recorded"
        assert metrics["l2_hits"] + metrics["l2_misses"] > 0, "No L2 operations recorded"
        assert 0 <= metrics["overall_hit_rate"] <= 1, "Invalid hit rate"
        
        # Performance metrics should be present
        assert "operations_per_second" in metrics, "Operations per second not tracked"
        assert metrics["operations_per_second"] > 0, "Invalid operations per second"
    
    async def _metrics_test_operations(self, cache_service):
        """Helper method for metrics testing."""
        # Generate cache misses
        for i in range(10):
            await cache_service.get(f"miss_key_{i}")
        
        # Generate cache hits
        for i in range(5):
            key = f"hit_key_{i}"
            await cache_service.set(key, f"value_{i}")
            await cache_service.get(key)  # Should hit L1
            await cache_service.get(key)  # Should hit L1 again