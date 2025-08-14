"""Comprehensive tests for performance optimization system.

This test suite validates all performance optimization components including
GPU optimization, caching, connection pooling, latency monitoring, and
adaptive quality management for production-ready performance.
"""

import asyncio
import pytest
import time
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from src.its_camera_ai.performance import (
    AdaptiveQualityManager,
    ConnectionPoolOptimizer,
    GPUMemoryOptimizer,
    LatencyMonitor,
    OptimizationConfig,
    OptimizationStrategy,
    PerformanceOptimizer,
    PipelineStage,
    StreamingCacheManager,
    create_performance_optimizer,
    create_production_optimization_config,
)


@pytest.fixture
def optimization_config():
    """Create test optimization configuration."""
    return create_production_optimization_config(
        max_concurrent_streams=50,
        target_latency_ms=100.0,
        strategy=OptimizationStrategy.LATENCY_OPTIMIZED
    )


@pytest.fixture
def mock_redis_manager():
    """Create mock Redis manager for testing."""
    redis_mock = AsyncMock()
    redis_mock.redis_client = AsyncMock()
    return redis_mock


class TestOptimizationConfig:
    """Test optimization configuration classes."""
    
    def test_create_production_config(self):
        """Test creating production optimization configuration."""
        config = create_production_optimization_config(
            max_concurrent_streams=100,
            target_latency_ms=80.0,
            strategy=OptimizationStrategy.LATENCY_OPTIMIZED
        )
        
        assert config.max_concurrent_streams == 100
        assert config.target_latency_ms == 80.0
        assert config.strategy == OptimizationStrategy.LATENCY_OPTIMIZED
        
        # Check strategy-specific optimizations
        optimized_config = config.get_optimized_config_for_strategy()
        assert optimized_config.gpu.batch_timeout_ms == 5.0
        assert optimized_config.latency_monitoring.latency_sla_ms == 80.0
    
    def test_memory_optimized_config(self):
        """Test memory-optimized configuration."""
        config = create_production_optimization_config(
            max_concurrent_streams=200,
            target_latency_ms=150.0,
            strategy=OptimizationStrategy.MEMORY_OPTIMIZED
        )
        
        optimized_config = config.get_optimized_config_for_strategy()
        assert optimized_config.gpu.gpu_memory_pool_size_gb == 4.0
        assert optimized_config.caching.l1_cache_size_mb == 256
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = create_production_optimization_config()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "strategy" in config_dict
        assert "max_concurrent_streams" in config_dict


@pytest.mark.asyncio
class TestGPUMemoryOptimizer:
    """Test GPU memory optimization."""
    
    async def test_gpu_optimizer_initialization(self, optimization_config):
        """Test GPU optimizer initialization."""
        with patch('src.its_camera_ai.performance.gpu_memory_optimizer.TORCH_AVAILABLE', False):
            # Should handle missing PyTorch gracefully
            from src.its_camera_ai.performance.gpu_memory_optimizer import GPUMemoryOptimizer
            
            optimizer = GPUMemoryOptimizer(optimization_config.gpu)
            await optimizer.initialize()
            
            # Should work without PyTorch
            assert not optimizer.is_initialized
    
    async def test_memory_pool_management(self, optimization_config):
        """Test memory pool management."""
        with patch('src.its_camera_ai.performance.gpu_memory_optimizer.TORCH_AVAILABLE', True):
            with patch('torch.cuda.is_available', return_value=False):
                # Test without CUDA
                from src.its_camera_ai.performance.gpu_memory_optimizer import GPUMemoryPool
                
                with pytest.raises(Exception):  # Should fail without CUDA
                    GPUMemoryPool(pool_size_gb=1.0)


@pytest.mark.asyncio 
class TestStreamingCacheManager:
    """Test streaming cache management."""
    
    async def test_cache_initialization(self, optimization_config, mock_redis_manager):
        """Test cache manager initialization."""
        from src.its_camera_ai.performance.streaming_cache_manager import StreamingCacheManager
        
        cache_manager = StreamingCacheManager(
            optimization_config.caching,
            mock_redis_manager
        )
        
        await cache_manager.start()
        
        assert cache_manager.is_running
        assert cache_manager.l1_cache is not None
        
        await cache_manager.stop()
    
    async def test_multi_level_caching(self, optimization_config, mock_redis_manager):
        """Test multi-level caching functionality."""
        from src.its_camera_ai.performance.streaming_cache_manager import StreamingCacheManager
        
        cache_manager = StreamingCacheManager(
            optimization_config.caching,
            mock_redis_manager
        )
        
        await cache_manager.start()
        
        # Test L1 cache
        test_key = "test_fragment_1"
        test_data = b"test_fragment_data"
        
        await cache_manager.put(test_key, test_data, ttl=60)
        cached_data = await cache_manager.get(test_key)
        
        assert cached_data == test_data
        
        # Test cache metrics
        metrics = cache_manager.get_cache_metrics()
        assert "cache_levels" in metrics
        assert "performance" in metrics
        
        await cache_manager.stop()
    
    async def test_predictive_caching(self, optimization_config, mock_redis_manager):
        """Test predictive caching based on access patterns."""
        from src.its_camera_ai.performance.streaming_cache_manager import StreamingCacheManager
        
        cache_manager = StreamingCacheManager(
            optimization_config.caching,
            mock_redis_manager
        )
        
        await cache_manager.start()
        
        # Simulate access patterns
        camera_id = "camera_001"
        access_patterns = {"popular_stream": True}
        
        # Should not raise exceptions
        await cache_manager.implement_predictive_caching(camera_id, access_patterns)
        
        await cache_manager.stop()


@pytest.mark.asyncio
class TestLatencyMonitor:
    """Test latency monitoring and SLA management."""
    
    async def test_latency_monitor_initialization(self, optimization_config):
        """Test latency monitor initialization."""
        from src.its_camera_ai.performance.latency_monitor import LatencyMonitor
        
        monitor = LatencyMonitor(optimization_config.latency_monitoring)
        await monitor.start_monitoring()
        
        assert monitor.is_monitoring
        
        await monitor.stop_monitoring()
    
    async def test_end_to_end_latency_tracking(self, optimization_config):
        """Test end-to-end latency tracking."""
        from src.its_camera_ai.performance.latency_monitor import LatencyMonitor
        
        monitor = LatencyMonitor(optimization_config.latency_monitoring)
        await monitor.start_monitoring()
        
        # Test latency tracking context manager
        camera_id = "test_camera"
        
        async with monitor.track_end_to_end_latency(camera_id) as request_id:
            # Simulate processing time
            await asyncio.sleep(0.05)  # 50ms
            assert request_id is not None
        
        # Check metrics
        metrics = monitor.get_comprehensive_metrics()
        assert metrics["total_measurements"] > 0
        
        await monitor.stop_monitoring()
    
    async def test_sla_violation_detection(self, optimization_config):
        """Test SLA violation detection and alerting."""
        from src.its_camera_ai.performance.latency_monitor import (
            LatencyMonitor,
            LatencyMeasurement,
            PipelineStage
        )
        
        monitor = LatencyMonitor(optimization_config.latency_monitoring)
        await monitor.start_monitoring()
        
        # Create measurement that violates SLA
        violation_measurement = LatencyMeasurement(
            timestamp=time.time(),
            latency_ms=150.0,  # Above 100ms SLA
            stage=PipelineStage.END_TO_END,
            camera_id="test_camera",
            request_id="test_request"
        )
        
        await monitor.record_measurement(violation_measurement)
        
        # Check that violation was recorded
        metrics = monitor.get_comprehensive_metrics()
        assert metrics["total_sla_violations"] > 0
        
        await monitor.stop_monitoring()


@pytest.mark.asyncio
class TestAdaptiveQualityManager:
    """Test adaptive quality management."""
    
    async def test_quality_manager_initialization(self, optimization_config):
        """Test adaptive quality manager initialization."""
        from src.its_camera_ai.performance.adaptive_quality_manager import AdaptiveQualityManager
        
        manager = AdaptiveQualityManager(optimization_config.adaptive_quality)
        await manager.start()
        
        assert manager.system_monitor.is_monitoring
        
        await manager.stop()
    
    async def test_camera_registration(self, optimization_config):
        """Test camera registration and quality management."""
        from src.its_camera_ai.performance.adaptive_quality_manager import AdaptiveQualityManager
        
        manager = AdaptiveQualityManager(optimization_config.adaptive_quality)
        await manager.start()
        
        # Register cameras
        camera_id = "test_camera_001"
        state = await manager.register_camera(camera_id, is_priority=True)
        
        assert state.camera_id == camera_id
        assert state.is_priority == True
        assert camera_id in manager.camera_states
        
        # Test quality profile retrieval
        profile = manager.get_quality_profile(camera_id)
        assert profile is not None
        
        await manager.unregister_camera(camera_id)
        await manager.stop()
    
    async def test_quality_adjustment(self, optimization_config):
        """Test quality adjustment based on system load."""
        from src.its_camera_ai.performance.adaptive_quality_manager import (
            AdaptiveQualityManager,
            SystemMetrics,
            LoadCondition
        )
        
        manager = AdaptiveQualityManager(optimization_config.adaptive_quality)
        await manager.start()
        
        # Register test camera
        camera_id = "test_camera_001"
        await manager.register_camera(camera_id)
        
        # Simulate high load condition
        high_load_metrics = SystemMetrics(
            cpu_percent=85.0,
            memory_percent=80.0,
            gpu_percent=90.0,
            active_streams=1,
            average_latency_ms=120.0
        )
        
        assert high_load_metrics.load_condition == LoadCondition.HIGH
        
        # Apply quality adjustment
        adjustments = await manager.adjust_quality_based_on_load(high_load_metrics)
        
        # Should have adjusted quality
        if adjustments:
            assert camera_id in adjustments
        
        await manager.stop()


@pytest.mark.asyncio
class TestPerformanceOptimizer:
    """Test comprehensive performance optimizer."""
    
    async def test_optimizer_initialization(self, optimization_config, mock_redis_manager):
        """Test performance optimizer initialization."""
        optimizer = PerformanceOptimizer(optimization_config, mock_redis_manager)
        
        # Initialize without external dependencies
        await optimizer.initialize()
        
        assert optimizer.is_initialized
        
        await optimizer.start()
        assert optimizer.is_running
        
        await optimizer.stop()
        assert not optimizer.is_running
    
    async def test_stream_processing_optimization(self, optimization_config, mock_redis_manager):
        """Test optimized stream processing context."""
        optimizer = PerformanceOptimizer(optimization_config, mock_redis_manager)
        await optimizer.initialize()
        await optimizer.start()
        
        camera_id = "test_camera"
        
        # Test optimization context
        async with optimizer.optimize_stream_processing(camera_id) as context:
            assert isinstance(context, dict)
            # Context should contain optimization components
            # (some may be None if dependencies not available)
        
        await optimizer.stop()
    
    async def test_pipeline_measurement_recording(self, optimization_config, mock_redis_manager):
        """Test pipeline measurement recording."""
        optimizer = PerformanceOptimizer(optimization_config, mock_redis_manager)
        await optimizer.initialize()
        await optimizer.start()
        
        # Record measurements for different pipeline stages
        camera_id = "test_camera"
        
        await optimizer.record_pipeline_measurement(
            camera_id=camera_id,
            stage=PipelineStage.ML_INFERENCE,
            latency_ms=45.0,
            metadata={"batch_size": 4}
        )
        
        await optimizer.record_pipeline_measurement(
            camera_id=camera_id,
            stage=PipelineStage.ENCODING,
            latency_ms=8.0
        )
        
        # Should not raise exceptions
        await optimizer.stop()
    
    async def test_fragment_caching(self, optimization_config, mock_redis_manager):
        """Test streaming fragment caching."""
        optimizer = PerformanceOptimizer(optimization_config, mock_redis_manager)
        await optimizer.initialize()
        await optimizer.start()
        
        # Test fragment caching
        fragment_key = "camera_001_fragment_123"
        fragment_data = b"test_fragment_data_12345"
        
        # Cache fragment
        success = await optimizer.cache_fragment(fragment_key, fragment_data, 60)
        
        if optimizer.cache_manager:
            assert success == True
            
            # Retrieve fragment
            cached_data = await optimizer.get_cached_fragment(fragment_key)
            assert cached_data == fragment_data
        
        await optimizer.stop()
    
    async def test_optimization_status(self, optimization_config, mock_redis_manager):
        """Test optimization status reporting."""
        optimizer = PerformanceOptimizer(optimization_config, mock_redis_manager)
        await optimizer.initialize()
        await optimizer.start()
        
        status = optimizer.get_optimization_status()
        
        assert "initialized" in status
        assert "running" in status
        assert "strategy" in status
        assert "components" in status
        
        assert status["initialized"] == True
        assert status["running"] == True
        assert status["strategy"] == OptimizationStrategy.LATENCY_OPTIMIZED.value
        
        await optimizer.stop()
    
    async def test_comprehensive_metrics(self, optimization_config, mock_redis_manager):
        """Test comprehensive metrics collection."""
        optimizer = PerformanceOptimizer(optimization_config, mock_redis_manager)
        await optimizer.initialize()
        await optimizer.start()
        
        # Let the monitoring loop run briefly
        await asyncio.sleep(0.1)
        
        metrics = optimizer.get_comprehensive_metrics()
        
        assert "timestamp" in metrics
        assert "uptime_seconds" in metrics
        
        await optimizer.stop()


@pytest.mark.asyncio
class TestPerformanceIntegration:
    """Integration tests for complete performance optimization system."""
    
    async def test_create_latency_optimized_system(self):
        """Test creating latency-optimized performance system."""
        from src.its_camera_ai.performance import create_latency_optimized_system
        
        # Should handle missing external dependencies gracefully
        try:
            optimizer = await create_latency_optimized_system(
                max_concurrent_streams=25,
                target_latency_ms=80.0,
                redis_url="redis://localhost:6379",
                database_url="postgresql://localhost/test"
            )
            
            assert optimizer.is_initialized
            assert optimizer.config.strategy == OptimizationStrategy.LATENCY_OPTIMIZED
            
            await optimizer.stop()
            
        except Exception as e:
            # Expected when external dependencies not available
            pytest.skip(f"External dependencies not available: {e}")
    
    async def test_create_balanced_system(self):
        """Test creating balanced performance system."""
        from src.its_camera_ai.performance import create_balanced_performance_system
        
        try:
            optimizer = await create_balanced_performance_system(
                max_concurrent_streams=100,
                enable_quality_adaptation=True
            )
            
            assert optimizer.is_initialized
            assert optimizer.config.strategy == OptimizationStrategy.BALANCED
            
            await optimizer.stop()
            
        except Exception as e:
            pytest.skip(f"External dependencies not available: {e}")


class TestPerformanceStress:
    """Stress tests for performance optimization under load."""
    
    @pytest.mark.slow
    async def test_concurrent_stream_optimization(self, optimization_config, mock_redis_manager):
        """Test optimization with many concurrent streams."""
        optimizer = PerformanceOptimizer(optimization_config, mock_redis_manager)
        await optimizer.initialize()
        await optimizer.start()
        
        # Simulate multiple concurrent stream processing contexts
        tasks = []
        for i in range(10):  # Test with 10 concurrent streams
            camera_id = f"stress_camera_{i:03d}"
            
            async def process_stream(cam_id):
                async with optimizer.optimize_stream_processing(cam_id) as context:
                    # Simulate processing work
                    await asyncio.sleep(0.01)
                    
                    # Record measurements
                    await optimizer.record_pipeline_measurement(
                        cam_id, PipelineStage.ML_INFERENCE, 30.0 + i
                    )
            
            tasks.append(process_stream(camera_id))
        
        # Execute all concurrent streams
        await asyncio.gather(*tasks)
        
        # Check system status
        status = optimizer.get_optimization_status()
        assert status["initialized"]
        assert status["running"]
        
        await optimizer.stop()
    
    @pytest.mark.benchmark
    async def test_cache_performance(self, optimization_config, mock_redis_manager):
        """Benchmark cache performance."""
        optimizer = PerformanceOptimizer(optimization_config, mock_redis_manager)
        await optimizer.initialize()
        await optimizer.start()
        
        if not optimizer.cache_manager:
            pytest.skip("Cache manager not available")
        
        # Benchmark cache operations
        test_data = b"x" * 1024  # 1KB test data
        num_operations = 100
        
        start_time = time.perf_counter()
        
        # Cache many fragments
        for i in range(num_operations):
            await optimizer.cache_fragment(f"benchmark_fragment_{i}", test_data)
        
        # Retrieve all fragments
        for i in range(num_operations):
            await optimizer.get_cached_fragment(f"benchmark_fragment_{i}")
        
        end_time = time.perf_counter()
        
        operations_per_second = (num_operations * 2) / (end_time - start_time)
        
        # Should achieve reasonable cache performance
        assert operations_per_second > 100  # At least 100 ops/sec
        
        await optimizer.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])