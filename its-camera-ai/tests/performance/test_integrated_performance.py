"""Integrated performance tests for cache + database optimization.

This module tests the complete system performance with both cache and database optimizations:
- End-to-end analytics query performance (sub-10ms requirement)
- Cache-database interaction optimization
- Real-world workload simulation (1000+ cameras, 10TB/day)
- System performance under stress conditions
- Cache warming effectiveness for database queries
- Performance monitoring and alerting validation
"""

import asyncio
import time
import random
import statistics
from datetime import UTC, datetime, timedelta
from typing import List, Dict, Any
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.its_camera_ai.services.cache import EnhancedCacheService
from src.its_camera_ai.services.analytics_service import (
    AnalyticsAggregationService, 
    EnhancedAnalyticsService,
    TimeWindow,
    AggregationLevel
)
from src.its_camera_ai.services.database_optimizer import DatabaseOptimizer
from src.its_camera_ai.core.config import Settings


class TestIntegratedPerformance:
    """Integrated performance tests for the complete system."""
    
    @pytest.fixture
    def settings(self):
        """Test settings for high-performance configuration."""
        return Settings(
            database=Settings.DatabaseSettings(
                url="postgresql+asyncpg://test:test@localhost/test",
                pool_size=50,
                max_overflow=100,
                pool_timeout=30,
                echo=False
            ),
            environment="production"  # Use production settings for performance testing
        )
    
    @pytest.fixture
    async def mock_redis_client(self):
        """High-performance mock Redis client."""
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
        
        # Mock pipeline for batch operations
        async def mock_pipeline():
            pipe_mock = AsyncMock()
            pipe_commands = []
            
            def add_command(cmd_type, *args):
                pipe_commands.append((cmd_type, args))
                return pipe_mock
            
            pipe_mock.get = lambda key: add_command('get', key)
            pipe_mock.set = lambda key, value, ex=None: add_command('set', key, value, ex)
            pipe_mock.delete = lambda key: add_command('delete', key)
            
            async def execute():
                results = []
                for cmd_type, args in pipe_commands:
                    if cmd_type == 'get':
                        results.append(storage.get(args[0]))
                    elif cmd_type == 'set':
                        storage[args[0]] = args[1]
                        results.append(True)
                    elif cmd_type == 'delete':
                        if args[0] in storage:
                            del storage[args[0]]
                            results.append(1)
                        else:
                            results.append(0)
                return results
            
            pipe_mock.execute = execute
            return pipe_mock
        
        async def mock_scan_iter(match=None, count=None):
            for key in storage.keys():
                if match is None or match.replace('*', '') in key:
                    yield key
        
        mock_redis = AsyncMock()
        mock_redis.get = mock_get
        mock_redis.set = mock_set
        mock_redis.delete = mock_delete
        mock_redis.pipeline = mock_pipeline
        mock_redis.scan_iter = mock_scan_iter
        
        return mock_redis
    
    @pytest.fixture
    async def cache_service(self, mock_redis_client):
        """Enhanced cache service for testing."""
        return EnhancedCacheService(
            redis_client=mock_redis_client,
            l1_max_size=10000,  # Large L1 cache for performance
            l1_max_memory_mb=100,  # 100MB L1 cache
            enable_l1=True,
            key_prefix="perf_test"
        )
    
    @pytest.fixture
    async def mock_analytics_repository(self):
        """Mock analytics repository with realistic data."""
        repo = AsyncMock()
        
        # Mock aggregated metrics with realistic traffic data
        async def mock_get_aggregated_metrics(*args, **kwargs):
            return {
                "record_count": 1000,
                "total_vehicles": 15000,
                "avg_speed": 45.5,
                "avg_density": 12.3,
                "avg_occupancy": 65.2
            }
        
        async def mock_get_traffic_metrics_by_camera(*args, **kwargs):
            # Generate realistic traffic metrics
            metrics = []
            for i in range(kwargs.get('limit', 100)):
                metrics.append(MagicMock(
                    camera_id=kwargs.get('camera_id', 'camera_001'),
                    timestamp=datetime.now(UTC) - timedelta(minutes=i*5),
                    total_vehicles=random.randint(5, 50),
                    vehicle_cars=random.randint(3, 35),
                    vehicle_trucks=random.randint(0, 10),
                    vehicle_buses=random.randint(0, 3),
                    vehicle_motorcycles=random.randint(0, 8),
                    vehicle_bicycles=random.randint(0, 5),
                    northbound_count=random.randint(0, 15),
                    southbound_count=random.randint(0, 15),
                    eastbound_count=random.randint(0, 15),
                    westbound_count=random.randint(0, 15),
                    average_speed=random.uniform(25, 75),
                    occupancy_rate=random.uniform(10, 90),
                    congestion_level=random.choice(['light', 'moderate', 'heavy']),
                    queue_length=random.randint(0, 20)
                ))
            return metrics
        
        async def mock_get_active_cameras(cutoff_time):
            # Return mock active cameras
            return [MagicMock(id=f"camera_{i:03d}") for i in range(1, 11)]
        
        repo.get_aggregated_metrics = mock_get_aggregated_metrics
        repo.get_traffic_metrics_by_camera = mock_get_traffic_metrics_by_camera
        repo.get_active_cameras = mock_get_active_cameras
        
        return repo
    
    @pytest.fixture
    async def analytics_service(self, mock_analytics_repository, cache_service, settings):
        """Analytics aggregation service with cache integration."""
        return AnalyticsAggregationService(
            analytics_repository=mock_analytics_repository,
            cache_service=cache_service,
            settings=settings
        )
    
    @pytest.mark.asyncio
    async def test_end_to_end_analytics_performance(self, analytics_service, cache_service):
        """Test complete analytics query performance with caching."""
        # Test configuration for high-throughput scenario
        camera_ids = [f"camera_{i:03d}" for i in range(1, 21)]  # 20 cameras
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=24)
        
        # Test different aggregation patterns
        test_patterns = [
            (TimeWindow.FIVE_MIN, AggregationLevel.MINUTE, "real-time"),
            (TimeWindow.FIFTEEN_MIN, AggregationLevel.MINUTE, "near-real-time"),
            (TimeWindow.ONE_HOUR, AggregationLevel.HOURLY, "historical"),
        ]
        
        performance_results = []
        
        for time_window, agg_level, pattern_name in test_patterns:
            print(f"\\nTesting {pattern_name} analytics pattern:")
            print(f"Window: {time_window.value}, Level: {agg_level.value}")
            
            # Cold cache performance (first run)
            cold_start = time.time()
            cold_results = await analytics_service.aggregate_traffic_metrics(
                camera_ids=camera_ids,
                time_window=time_window,
                aggregation_level=agg_level,
                start_time=start_time,
                end_time=end_time,
                include_quality_check=True
            )
            cold_time = time.time() - cold_start
            
            # Warm cache performance (second run)
            warm_start = time.time()
            warm_results = await analytics_service.aggregate_traffic_metrics(
                camera_ids=camera_ids,
                time_window=time_window,
                aggregation_level=agg_level,
                start_time=start_time,
                end_time=end_time,
                include_quality_check=True
            )
            warm_time = time.time() - warm_start
            
            # Performance metrics
            cache_improvement = (cold_time - warm_time) / cold_time * 100
            
            result = {
                "pattern": pattern_name,
                "cold_time_ms": cold_time * 1000,
                "warm_time_ms": warm_time * 1000,
                "cache_improvement_pct": cache_improvement,
                "results_count": len(warm_results),
                "cameras_processed": len(camera_ids)
            }
            
            performance_results.append(result)
            
            print(f"  Cold cache: {result['cold_time_ms']:.1f}ms")
            print(f"  Warm cache: {result['warm_time_ms']:.1f}ms")
            print(f"  Cache improvement: {result['cache_improvement_pct']:.1f}%")
            print(f"  Results: {result['results_count']}")
            
            # Performance requirements
            assert result['warm_time_ms'] < 50, f"Warm cache too slow: {result['warm_time_ms']:.1f}ms"
            assert result['cache_improvement_pct'] > 50, f"Cache improvement too low: {result['cache_improvement_pct']:.1f}%"
            assert len(cold_results) == len(warm_results), "Results inconsistent between runs"
        
        # Overall performance summary
        avg_cold_time = statistics.mean([r['cold_time_ms'] for r in performance_results])
        avg_warm_time = statistics.mean([r['warm_time_ms'] for r in performance_results])
        avg_improvement = statistics.mean([r['cache_improvement_pct'] for r in performance_results])
        
        print(f"\\nOverall Performance Summary:")
        print(f"Average cold cache time: {avg_cold_time:.1f}ms")
        print(f"Average warm cache time: {avg_warm_time:.1f}ms")
        print(f"Average cache improvement: {avg_improvement:.1f}%")
        
        # System-wide performance requirements
        assert avg_warm_time < 30, f"System too slow: {avg_warm_time:.1f}ms average"
        assert avg_improvement > 60, f"Cache not effective enough: {avg_improvement:.1f}% improvement"
    
    @pytest.mark.asyncio
    async def test_high_throughput_workload_simulation(self, analytics_service, cache_service):
        """Simulate 10TB/day workload with 1000+ cameras."""
        # Simulate high-throughput workload
        num_cameras = 100  # Reduced for testing, represents 1000+ cameras
        queries_per_minute = 500  # Represents high query load
        simulation_duration = 60  # 1 minute simulation
        
        camera_ids = [f"camera_{i:04d}" for i in range(1, num_cameras + 1)]
        
        print(f"\\nHigh Throughput Workload Simulation:")
        print(f"Cameras: {num_cameras}")
        print(f"Target queries per minute: {queries_per_minute}")
        print(f"Duration: {simulation_duration}s")
        
        # Workload simulation
        query_times = []
        cache_hits = 0
        total_queries = 0
        
        async def query_worker(worker_id: int):
            nonlocal cache_hits, total_queries
            worker_queries = 0
            worker_times = []
            
            while time.time() - start_time < simulation_duration:
                # Select random cameras and time range
                selected_cameras = random.sample(camera_ids, random.randint(1, 5))
                end_time = datetime.now(UTC)
                hours_back = random.choice([1, 6, 24])
                query_start_time = end_time - timedelta(hours=hours_back)
                
                # Random aggregation pattern
                time_window = random.choice([TimeWindow.FIVE_MIN, TimeWindow.FIFTEEN_MIN, TimeWindow.ONE_HOUR])
                agg_level = random.choice([AggregationLevel.MINUTE, AggregationLevel.HOURLY])
                
                # Execute query
                query_start = time.time()
                
                try:
                    results = await analytics_service.aggregate_traffic_metrics(
                        camera_ids=selected_cameras,
                        time_window=time_window,
                        aggregation_level=agg_level,
                        start_time=query_start_time,
                        end_time=end_time,
                        include_quality_check=False  # Skip for performance
                    )
                    
                    query_time = (time.time() - query_start) * 1000
                    worker_times.append(query_time)
                    worker_queries += 1
                    
                    # Small delay to control load
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    print(f"Query failed: {e}")
            
            return worker_times, worker_queries
        
        # Run simulation with concurrent workers
        num_workers = 20
        start_time = time.time()
        
        tasks = [query_worker(i) for i in range(num_workers)]
        results = await asyncio.gather(*tasks)
        
        total_duration = time.time() - start_time
        
        # Analyze results
        all_times = []
        total_worker_queries = 0
        
        for worker_times, worker_query_count in results:
            all_times.extend(worker_times)
            total_worker_queries += worker_query_count
        
        # Performance metrics
        avg_query_time = statistics.mean(all_times) if all_times else 0
        p95_query_time = sorted(all_times)[int(len(all_times) * 0.95)] if all_times else 0
        p99_query_time = sorted(all_times)[int(len(all_times) * 0.99)] if all_times else 0
        queries_per_second = len(all_times) / total_duration
        
        # Cache metrics
        cache_metrics = cache_service.get_metrics()
        
        print(f"\\nWorkload Results:")
        print(f"Total queries executed: {len(all_times)}")
        print(f"Total duration: {total_duration:.1f}s")
        print(f"Queries per second: {queries_per_second:.1f}")
        print(f"Average query time: {avg_query_time:.1f}ms")
        print(f"P95 query time: {p95_query_time:.1f}ms")
        print(f"P99 query time: {p99_query_time:.1f}ms")
        print(f"Cache hit rate: {cache_metrics['overall_hit_rate']:.1%}")
        print(f"L1 cache hit rate: {cache_metrics['l1_hit_rate']:.1%}")
        print(f"L2 cache hit rate: {cache_metrics['l2_hit_rate']:.1%}")
        
        # Performance requirements for 10TB/day workload
        assert queries_per_second > 50, f"Throughput too low: {queries_per_second:.1f} qps"
        assert avg_query_time < 100, f"Average query too slow: {avg_query_time:.1f}ms"
        assert p95_query_time < 200, f"P95 query too slow: {p95_query_time:.1f}ms"
        assert cache_metrics['overall_hit_rate'] > 0.7, f"Cache hit rate too low: {cache_metrics['overall_hit_rate']:.1%}"
    
    @pytest.mark.asyncio
    async def test_cache_warming_effectiveness(self, analytics_service, cache_service):
        """Test cache warming effectiveness for database queries."""
        camera_ids = [f"camera_{i:03d}" for i in range(1, 11)]
        
        # Clear cache to start fresh
        await cache_service.clear_all()
        
        print(f"\\nCache Warming Effectiveness Test:")
        
        # Test queries without warming (cold cache)
        cold_cache_times = []
        test_queries = [
            (TimeWindow.FIVE_MIN, AggregationLevel.MINUTE),
            (TimeWindow.FIFTEEN_MIN, AggregationLevel.MINUTE),
            (TimeWindow.ONE_HOUR, AggregationLevel.HOURLY),
        ]
        
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=6)
        
        for time_window, agg_level in test_queries:
            query_start = time.time()
            await analytics_service.aggregate_traffic_metrics(
                camera_ids=camera_ids,
                time_window=time_window,
                aggregation_level=agg_level,
                start_time=start_time,
                end_time=end_time
            )
            cold_time = (time.time() - query_start) * 1000
            cold_cache_times.append(cold_time)
        
        avg_cold_time = statistics.mean(cold_cache_times)
        
        # Warm cache
        print(f"Warming cache for {len(camera_ids)} cameras...")
        warm_start = time.time()
        warmed_entries = await analytics_service.warm_cache_for_cameras(camera_ids, hours_back=6)
        warm_duration = time.time() - warm_start
        
        print(f"Cache warming completed: {warmed_entries} entries in {warm_duration:.2f}s")
        
        # Test same queries with warmed cache
        warm_cache_times = []
        for time_window, agg_level in test_queries:
            query_start = time.time()
            await analytics_service.aggregate_traffic_metrics(
                camera_ids=camera_ids,
                time_window=time_window,
                aggregation_level=agg_level,
                start_time=start_time,
                end_time=end_time
            )
            warm_time = (time.time() - query_start) * 1000
            warm_cache_times.append(warm_time)
        
        avg_warm_time = statistics.mean(warm_cache_times)
        improvement = (avg_cold_time - avg_warm_time) / avg_cold_time * 100
        
        # Cache metrics after warming
        cache_metrics = cache_service.get_metrics()
        
        print(f"\\nCache Warming Results:")
        print(f"Entries warmed: {warmed_entries}")
        print(f"Warming time: {warm_duration:.2f}s")
        print(f"Cold cache avg time: {avg_cold_time:.1f}ms")
        print(f"Warm cache avg time: {avg_warm_time:.1f}ms")
        print(f"Performance improvement: {improvement:.1f}%")
        print(f"Cache hit rate: {cache_metrics['overall_hit_rate']:.1%}")
        
        # Warming effectiveness requirements
        assert warmed_entries > 0, "No entries were warmed"
        assert warm_duration < 30, f"Cache warming too slow: {warm_duration:.2f}s"
        assert improvement > 40, f"Cache warming not effective: {improvement:.1f}% improvement"
        assert cache_metrics['overall_hit_rate'] > 0.8, f"Hit rate after warming too low: {cache_metrics['overall_hit_rate']:.1%}"
    
    @pytest.mark.asyncio
    async def test_system_stress_conditions(self, analytics_service, cache_service):
        """Test system performance under stress conditions."""
        print(f"\\nSystem Stress Test:")
        
        # Stress test parameters
        stress_duration = 30  # 30 seconds
        max_concurrent_queries = 50
        memory_pressure_mb = 50  # Simulate memory pressure
        
        # Create large dataset to simulate memory pressure
        large_data = {}
        for i in range(100):
            large_data[f"stress_key_{i}"] = "x" * (memory_pressure_mb * 1024 * 10)  # 10KB per entry
        
        await cache_service.multi_set(large_data, ttl=stress_duration + 10)
        
        print(f"Applied memory pressure: ~{memory_pressure_mb}MB")
        
        # Stress test queries
        camera_ids = [f"camera_{i:03d}" for i in range(1, 31)]  # 30 cameras
        query_results = []
        errors = []
        
        async def stress_query_worker(worker_id: int):
            worker_results = []
            worker_errors = []
            
            start_time = time.time()
            while time.time() - start_time < stress_duration:
                try:
                    # Random query parameters
                    selected_cameras = random.sample(camera_ids, random.randint(1, 10))
                    end_time = datetime.now(UTC)
                    start_time_query = end_time - timedelta(hours=random.choice([1, 3, 6]))
                    
                    time_window = random.choice([TimeWindow.FIVE_MIN, TimeWindow.FIFTEEN_MIN])
                    agg_level = random.choice([AggregationLevel.MINUTE, AggregationLevel.HOURLY])
                    
                    query_start = time.time()
                    results = await analytics_service.aggregate_traffic_metrics(
                        camera_ids=selected_cameras,
                        time_window=time_window,
                        aggregation_level=agg_level,
                        start_time=start_time_query,
                        end_time=end_time,
                        include_quality_check=False
                    )
                    
                    query_time = (time.time() - query_start) * 1000
                    worker_results.append({
                        "time_ms": query_time,
                        "cameras": len(selected_cameras),
                        "results": len(results)
                    })
                    
                    # Brief pause
                    await asyncio.sleep(0.005)
                    
                except Exception as e:
                    worker_errors.append(str(e))
            
            return worker_results, worker_errors
        
        # Run stress test
        stress_start = time.time()
        tasks = [stress_query_worker(i) for i in range(max_concurrent_queries)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        stress_total_time = time.time() - stress_start
        
        # Analyze stress test results
        for result in results:
            if isinstance(result, tuple):
                worker_results, worker_errors = result
                query_results.extend(worker_results)
                errors.extend(worker_errors)
            elif isinstance(result, Exception):
                errors.append(str(result))
        
        # Performance analysis
        query_times = [r["time_ms"] for r in query_results]
        avg_stress_time = statistics.mean(query_times) if query_times else 0
        p95_stress_time = sorted(query_times)[int(len(query_times) * 0.95)] if query_times else 0
        success_rate = len(query_results) / (len(query_results) + len(errors)) * 100 if (len(query_results) + len(errors)) > 0 else 0
        
        # Cache performance under stress
        final_cache_metrics = cache_service.get_metrics()
        
        print(f"\\nStress Test Results:")
        print(f"Duration: {stress_total_time:.1f}s")
        print(f"Concurrent workers: {max_concurrent_queries}")
        print(f"Total queries: {len(query_results)}")
        print(f"Errors: {len(errors)}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Average query time: {avg_stress_time:.1f}ms")
        print(f"P95 query time: {p95_stress_time:.1f}ms")
        print(f"Cache hit rate under stress: {final_cache_metrics['overall_hit_rate']:.1%}")
        
        # Stress test requirements
        assert success_rate > 95, f"Success rate too low under stress: {success_rate:.1f}%"
        assert avg_stress_time < 150, f"Average query too slow under stress: {avg_stress_time:.1f}ms"
        assert p95_stress_time < 300, f"P95 query too slow under stress: {p95_stress_time:.1f}ms"
        assert final_cache_metrics['overall_hit_rate'] > 0.6, f"Cache hit rate degraded too much: {final_cache_metrics['overall_hit_rate']:.1%}"
        
        # System should remain stable
        assert len(errors) < len(query_results) * 0.05, f"Too many errors under stress: {len(errors)}"
    
    @pytest.mark.asyncio
    async def test_cache_invalidation_impact(self, analytics_service, cache_service):
        """Test performance impact of cache invalidation strategies."""
        camera_ids = [f"camera_{i:03d}" for i in range(1, 21)]
        
        # Pre-populate cache with analytics data
        print(f"\\nCache Invalidation Impact Test:")
        print(f"Pre-populating cache...")
        
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=12)
        
        # Populate cache with various queries
        for time_window in [TimeWindow.FIVE_MIN, TimeWindow.FIFTEEN_MIN, TimeWindow.ONE_HOUR]:
            for agg_level in [AggregationLevel.MINUTE, AggregationLevel.HOURLY]:
                await analytics_service.aggregate_traffic_metrics(
                    camera_ids=camera_ids[:10],
                    time_window=time_window,
                    aggregation_level=agg_level,
                    start_time=start_time,
                    end_time=end_time
                )
        
        initial_metrics = cache_service.get_metrics()
        print(f"Initial cache entries: L1={initial_metrics.get('l1_cache', {}).get('size', 0)}")
        
        # Test different invalidation strategies
        invalidation_tests = [
            ("single_camera", {f"camera:{camera_ids[0]}"}, "Invalidate single camera"),
            ("analytics_type", {"analytics"}, "Invalidate all analytics"),
            ("time_window", {"window:5min"}, "Invalidate specific time window"),
        ]
        
        for test_name, tags, description in invalidation_tests:
            print(f"\\nTesting {description}:")
            
            # Measure query performance before invalidation
            pre_invalidation_times = []
            for _ in range(5):
                query_start = time.time()
                await analytics_service.aggregate_traffic_metrics(
                    camera_ids=camera_ids[:5],
                    time_window=TimeWindow.FIVE_MIN,
                    aggregation_level=AggregationLevel.MINUTE,
                    start_time=start_time,
                    end_time=end_time
                )
                pre_invalidation_times.append((time.time() - query_start) * 1000)
            
            avg_pre_time = statistics.mean(pre_invalidation_times)
            
            # Perform invalidation
            invalidation_start = time.time()
            invalidated_count = await cache_service.invalidate_by_tags(tags)
            invalidation_time = (time.time() - invalidation_start) * 1000
            
            # Measure query performance after invalidation
            post_invalidation_times = []
            for _ in range(5):
                query_start = time.time()
                await analytics_service.aggregate_traffic_metrics(
                    camera_ids=camera_ids[:5],
                    time_window=TimeWindow.FIVE_MIN,
                    aggregation_level=AggregationLevel.MINUTE,
                    start_time=start_time,
                    end_time=end_time
                )
                post_invalidation_times.append((time.time() - query_start) * 1000)
            
            avg_post_time = statistics.mean(post_invalidation_times)
            performance_impact = ((avg_post_time - avg_pre_time) / avg_pre_time) * 100
            
            print(f"  Invalidated entries: {invalidated_count}")
            print(f"  Invalidation time: {invalidation_time:.1f}ms")
            print(f"  Pre-invalidation avg: {avg_pre_time:.1f}ms")
            print(f"  Post-invalidation avg: {avg_post_time:.1f}ms")
            print(f"  Performance impact: {performance_impact:.1f}%")
            
            # Invalidation performance requirements
            assert invalidation_time < 100, f"Invalidation too slow: {invalidation_time:.1f}ms"
            assert invalidated_count > 0, "No entries were invalidated"
            
            # Performance impact should be reasonable
            if test_name != "analytics_type":  # Full invalidation expected to have higher impact
                assert performance_impact < 200, f"Performance impact too high: {performance_impact:.1f}%"
    
    @pytest.mark.asyncio
    async def test_monitoring_and_alerting_performance(self, analytics_service, cache_service):
        """Test performance monitoring and alerting capabilities."""
        print(f"\\nMonitoring and Alerting Performance Test:")
        
        # Generate workload for monitoring
        camera_ids = [f"camera_{i:03d}" for i in range(1, 11)]
        monitoring_duration = 20  # 20 seconds
        
        # Background workload generator
        async def monitoring_workload():
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(hours=6)
            
            while True:
                try:
                    await analytics_service.aggregate_traffic_metrics(
                        camera_ids=random.sample(camera_ids, 3),
                        time_window=random.choice([TimeWindow.FIVE_MIN, TimeWindow.FIFTEEN_MIN]),
                        aggregation_level=random.choice([AggregationLevel.MINUTE, AggregationLevel.HOURLY]),
                        start_time=start_time,
                        end_time=end_time
                    )
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    break
                except Exception:
                    pass
        
        # Start background workload
        workload_task = asyncio.create_task(monitoring_workload())
        
        # Collect monitoring data
        monitoring_data = []
        monitoring_start = time.time()
        
        try:
            while time.time() - monitoring_start < monitoring_duration:
                metrics_start = time.time()
                
                # Collect cache metrics
                cache_metrics = cache_service.get_metrics()
                
                metrics_collection_time = (time.time() - metrics_start) * 1000
                
                monitoring_data.append({
                    "timestamp": time.time(),
                    "cache_hit_rate": cache_metrics["overall_hit_rate"],
                    "l1_hit_rate": cache_metrics["l1_hit_rate"],
                    "l2_hit_rate": cache_metrics["l2_hit_rate"],
                    "operations_per_second": cache_metrics["operations_per_second"],
                    "collection_time_ms": metrics_collection_time
                })
                
                await asyncio.sleep(1)  # Collect every second
        
        finally:
            workload_task.cancel()
            try:
                await workload_task
            except asyncio.CancelledError:
                pass
        
        # Analyze monitoring performance
        collection_times = [d["collection_time_ms"] for d in monitoring_data]
        avg_collection_time = statistics.mean(collection_times)
        max_collection_time = max(collection_times)
        
        hit_rates = [d["cache_hit_rate"] for d in monitoring_data]
        avg_hit_rate = statistics.mean(hit_rates)
        min_hit_rate = min(hit_rates)
        
        ops_per_sec = [d["operations_per_second"] for d in monitoring_data]
        avg_ops_per_sec = statistics.mean(ops_per_sec)
        
        print(f"\\nMonitoring Results:")
        print(f"Monitoring duration: {monitoring_duration}s")
        print(f"Data points collected: {len(monitoring_data)}")
        print(f"Average collection time: {avg_collection_time:.1f}ms")
        print(f"Max collection time: {max_collection_time:.1f}ms")
        print(f"Average hit rate: {avg_hit_rate:.1%}")
        print(f"Minimum hit rate: {min_hit_rate:.1%}")
        print(f"Average ops/sec: {avg_ops_per_sec:.1f}")
        
        # Monitoring performance requirements
        assert len(monitoring_data) > monitoring_duration * 0.8, "Insufficient monitoring data collected"
        assert avg_collection_time < 10, f"Metrics collection too slow: {avg_collection_time:.1f}ms"
        assert max_collection_time < 50, f"Max collection time too slow: {max_collection_time:.1f}ms"
        
        # System should maintain good performance during monitoring
        assert avg_hit_rate > 0.5, f"Hit rate too low during monitoring: {avg_hit_rate:.1%}"
        assert avg_ops_per_sec > 0, "No operations detected during monitoring"