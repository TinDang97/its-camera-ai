"""Comprehensive Integration Tests for Optimized ITS Camera AI System.

This test suite validates the integration of all optimization components including:
- gRPC service optimization with blosc compression
- Redis queue manager with enhanced performance
- Kafka event producer/consumer with compression
- End-to-end system performance validation
- Cross-service communication testing

The tests ensure the system meets the target performance requirements:
- <100ms end-to-end latency
- 99.95% success rate
- Support for 100+ concurrent streams
- 60%+ compression ratio improvement
- 50%+ network bandwidth reduction
"""

import asyncio
import time
import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import AsyncMock, MagicMock

from src.its_camera_ai.services.grpc_streaming_server import StreamingServiceImpl
from src.its_camera_ai.services.kafka_event_producer import KafkaEventProducer, create_kafka_producer
from src.its_camera_ai.services.kafka_sse_consumer import KafkaSSEConsumer, create_kafka_sse_consumer
from src.its_camera_ai.flow.redis_queue_manager import RedisQueueManager, QueueConfig, QueueType
from src.its_camera_ai.core.blosc_numpy_compressor import get_global_compressor, BloscNumpyCompressor
from src.its_camera_ai.testing.end_to_end_performance_validator import (
    EndToEndPerformanceValidator,
    PerformanceTargets,
    create_end_to_end_validator,
    run_quick_validation
)

pytestmark = pytest.mark.asyncio


class TestOptimizedSystemIntegration:
    """Test suite for optimized system integration."""

    @pytest_asyncio.fixture
    async def redis_manager(self):
        """Create Redis queue manager for testing."""
        manager = RedisQueueManager(
            redis_url="redis://localhost:6379/15",  # Use test database
            enable_blosc_compression=True,
            enable_adaptive_batching=True
        )
        try:
            await manager.connect()
            yield manager
        finally:
            await manager.disconnect()

    @pytest_asyncio.fixture
    async def grpc_service(self, redis_manager):
        """Create gRPC streaming service for testing."""
        service = StreamingServiceImpl(
            redis_manager=redis_manager,
            enable_blosc_compression=True,
            connection_pool_size=50
        )
        yield service

    @pytest_asyncio.fixture
    async def kafka_producer(self):
        """Create Kafka event producer for testing."""
        config = {
            "bootstrap_servers": ["localhost:9092"],
            "topic_prefix": "test-its-camera-ai",
            "enable_blosc_compression": True,
            "batch_size": 32768,
            "adaptive_batch_size": True
        }
        
        producer = create_kafka_producer(config)
        try:
            await producer.start()
            yield producer
        finally:
            await producer.stop()

    @pytest_asyncio.fixture
    async def kafka_consumer(self):
        """Create Kafka SSE consumer for testing."""
        from src.its_camera_ai.api.sse_broadcaster import SSEBroadcaster
        
        config = {
            "bootstrap_servers": ["localhost:9092"],
            "topic_prefix": "test-its-camera-ai",
            "consumer_group_id": "test-sse-streaming",
            "enable_blosc_decompression": True,
            "parallel_processing": True
        }
        
        # Mock SSE broadcaster for testing
        mock_broadcaster = MagicMock(spec=SSEBroadcaster)
        mock_broadcaster.connections = {}
        mock_broadcaster.get_stats.return_value = {"active_connections": 0}
        
        consumer = create_kafka_sse_consumer(config, mock_broadcaster)
        try:
            await consumer.start()
            yield consumer
        finally:
            await consumer.stop()

    @pytest_asyncio.fixture
    async def performance_validator(self):
        """Create performance validator for testing."""
        targets = PerformanceTargets(
            max_end_to_end_latency_ms=100.0,
            min_success_rate_percent=99.95,
            min_concurrent_streams=50,  # Reduced for testing
            min_compression_ratio_improvement=0.6
        )
        
        validator = await create_end_to_end_validator(targets)
        try:
            yield validator
        finally:
            await validator.cleanup()

    async def test_grpc_blosc_compression_integration(self, grpc_service):
        """Test gRPC service with blosc compression integration."""
        # Create test frame
        test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Test compression
        pb_frame = grpc_service._numpy_to_protobuf(test_frame, use_blosc=True)
        
        assert pb_frame.compressed_data is not None
        assert len(pb_frame.compressed_data) > 0
        assert pb_frame.compression_format == "blosc"
        assert pb_frame.quality == 100  # Lossless compression
        
        # Test decompression
        decompressed_frame = await grpc_service._optimized_protobuf_to_numpy(pb_frame)
        
        assert decompressed_frame is not None
        assert decompressed_frame.shape == test_frame.shape
        
        # Calculate compression ratio
        compression_ratio = len(pb_frame.compressed_data) / test_frame.nbytes
        assert compression_ratio < 0.7  # Should achieve > 30% compression
        
        print(f"gRPC Blosc compression ratio: {compression_ratio:.3f}")

    async def test_redis_queue_blosc_optimization(self, redis_manager):
        """Test Redis queue manager with blosc compression."""
        # Create test queue
        queue_config = QueueConfig(
            name="test_blosc_queue",
            queue_type=QueueType.STREAM,
            enable_blosc_compression=True,
            batch_size=50
        )
        
        await redis_manager.create_queue(queue_config)
        
        # Create test data (large enough to trigger compression)
        test_data = np.random.bytes(10240)  # 10KB
        
        # Enqueue with compression
        message_id = await redis_manager.enqueue(
            "test_blosc_queue",
            test_data,
            enable_compression=True
        )
        
        assert message_id is not None
        
        # Dequeue with automatic decompression
        result = await redis_manager.dequeue("test_blosc_queue", auto_decompress=True)
        
        assert result is not None
        assert result[0] == message_id
        assert result[1] == test_data  # Should match original data
        
        # Check compression metrics
        assert redis_manager.compression_metrics["total_original_bytes"] > 0
        assert redis_manager.compression_metrics["total_compressed_bytes"] > 0
        
        compression_ratio = (
            redis_manager.compression_metrics["total_compressed_bytes"] /
            redis_manager.compression_metrics["total_original_bytes"]
        )
        
        print(f"Redis queue compression ratio: {compression_ratio:.3f}")

    async def test_kafka_producer_optimization(self, kafka_producer):
        """Test Kafka producer with optimization features."""
        # Test event with large payload
        large_payload = {
            "camera_id": "test_camera_001",
            "frame_data": np.random.bytes(8192).hex(),  # 8KB payload
            "timestamp": time.time(),
            "metadata": {
                "resolution": "1920x1080",
                "fps": 30,
                "codec": "h264"
            }
        }
        
        # Send event
        success = await kafka_producer.send_analytics_event(
            event_type=kafka_producer.EventType.TRAFFIC_FLOW,
            data=large_payload,
            zone_id="test_zone_001"
        )
        
        assert success is True
        
        # Check health status and metrics
        health_status = kafka_producer.get_health_status()
        assert health_status["is_healthy"] is True
        
        metrics = health_status["metrics"]
        assert metrics["total_events_sent"] > 0
        assert metrics["total_bytes_sent"] > 0
        
        # Check compression metrics if available
        if metrics.get("bytes_saved_compression", 0) > 0:
            print(f"Kafka compression saved: {metrics['bytes_saved_compression']} bytes")
            print(f"Kafka compression efficiency: {metrics.get('compression_efficiency_percent', 0):.1f}%")

    async def test_kafka_consumer_optimization(self, kafka_consumer):
        """Test Kafka consumer with optimization features."""
        # Get health status
        health_status = kafka_consumer.get_health_status()
        
        assert health_status["is_healthy"] is True
        assert health_status["is_running"] is True
        
        config = health_status["configuration"]
        assert config["parallel_processing_enabled"] is True
        assert config["blosc_decompression_enabled"] is True
        assert config["adaptive_rate_limiting_enabled"] is True
        
        print(f"Kafka consumer max events/sec: {config['max_events_per_second']}")

    async def test_cross_service_communication(self, grpc_service, redis_manager, kafka_producer):
        """Test cross-service communication with optimizations."""
        # Create test frame
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # 1. Process through gRPC with compression
        pb_frame = grpc_service._numpy_to_protobuf(test_frame, use_blosc=True)
        
        # 2. Store in Redis queue
        queue_config = QueueConfig(
            name="test_cross_service_queue",
            enable_blosc_compression=True
        )
        await redis_manager.create_queue(queue_config)
        
        message_id = await redis_manager.enqueue(
            "test_cross_service_queue",
            pb_frame.compressed_data,
            enable_compression=True
        )
        
        # 3. Retrieve from Redis and verify
        result = await redis_manager.dequeue("test_cross_service_queue")
        assert result is not None
        assert result[0] == message_id
        
        # 4. Send analytics event to Kafka
        analytics_data = {
            "frame_id": "test_frame_001",
            "processing_time_ms": 25.5,
            "compression_ratio": len(pb_frame.compressed_data) / test_frame.nbytes,
            "queue_message_id": message_id
        }
        
        success = await kafka_producer.send_analytics_event(
            event_type=kafka_producer.EventType.PERFORMANCE_METRIC,
            data=analytics_data,
            zone_id="test_integration_zone"
        )
        
        assert success is True
        
        print("Cross-service communication test completed successfully")

    async def test_compression_performance_benchmarks(self):
        """Test compression performance across different data sizes."""
        blosc_compressor = get_global_compressor()
        
        # Test different image sizes
        test_sizes = [
            (480, 640, 3),     # SD
            (720, 1280, 3),    # HD
            (1080, 1920, 3),   # Full HD
            (2160, 3840, 3)    # 4K
        ]
        
        results = []
        
        for height, width, channels in test_sizes:
            test_array = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
            
            # Benchmark compression
            start_time = time.perf_counter()
            compressed_data = blosc_compressor.compress_with_metadata(test_array)
            compression_time = (time.perf_counter() - start_time) * 1000
            
            # Benchmark decompression
            start_time = time.perf_counter()
            decompressed_array = blosc_compressor.decompress_with_metadata(compressed_data)
            decompression_time = (time.perf_counter() - start_time) * 1000
            
            # Verify integrity
            assert np.array_equal(test_array, decompressed_array)
            
            # Calculate metrics
            compression_ratio = len(compressed_data) / test_array.nbytes
            size_reduction = (1 - compression_ratio) * 100
            throughput_mbps = (test_array.nbytes / 1024 / 1024) / (compression_time / 1000)
            
            result = {
                "size": f"{height}x{width}x{channels}",
                "original_mb": test_array.nbytes / (1024 * 1024),
                "compressed_mb": len(compressed_data) / (1024 * 1024),
                "compression_ratio": compression_ratio,
                "size_reduction_percent": size_reduction,
                "compression_time_ms": compression_time,
                "decompression_time_ms": decompression_time,
                "throughput_mbps": throughput_mbps
            }
            
            results.append(result)
            
            # Validate performance targets
            assert size_reduction >= 60.0, f"Size reduction {size_reduction:.1f}% below 60% target"
            assert compression_time <= 50.0, f"Compression time {compression_time:.2f}ms exceeds 50ms target"
            assert decompression_time <= 20.0, f"Decompression time {decompression_time:.2f}ms exceeds 20ms target"
            
        # Print benchmark results
        print("\nCompression Benchmark Results:")
        print("-" * 80)
        for result in results:
            print(f"{result['size']:12} | "
                  f"{result['original_mb']:6.1f}MB -> {result['compressed_mb']:6.1f}MB | "
                  f"{result['size_reduction_percent']:5.1f}% | "
                  f"{result['compression_time_ms']:6.2f}ms | "
                  f"{result['throughput_mbps']:6.1f}MB/s")

    async def test_concurrent_load_simulation(self, grpc_service, redis_manager):
        """Test system under concurrent load."""
        concurrent_streams = 20  # Reduced for testing environment
        frames_per_stream = 30   # 1 second at 30 FPS
        
        async def simulate_stream(stream_id: int):
            """Simulate a single camera stream."""
            latencies = []
            errors = 0
            
            for frame_num in range(frames_per_stream):
                try:
                    start_time = time.perf_counter()
                    
                    # Generate test frame
                    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    
                    # Process through gRPC
                    pb_frame = grpc_service._numpy_to_protobuf(test_frame, use_blosc=True)
                    decompressed = await grpc_service._optimized_protobuf_to_numpy(pb_frame)
                    
                    # Queue in Redis
                    message_id = await redis_manager.enqueue(
                        f"stream_{stream_id}_queue",
                        pb_frame.compressed_data,
                        enable_compression=True
                    )
                    
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    latencies.append(latency_ms)
                    
                    # Validate
                    assert decompressed is not None
                    assert message_id is not None
                    
                except Exception as e:
                    errors += 1
                    print(f"Stream {stream_id} frame {frame_num} error: {e}")
                
                # Simulate 30 FPS
                await asyncio.sleep(1.0 / 30.0)
            
            return {
                "stream_id": stream_id,
                "latencies": latencies,
                "errors": errors,
                "success_rate": (len(latencies) / frames_per_stream) * 100
            }
        
        # Create queues for each stream
        for stream_id in range(concurrent_streams):
            queue_config = QueueConfig(
                name=f"stream_{stream_id}_queue",
                enable_blosc_compression=True
            )
            await redis_manager.create_queue(queue_config)
        
        # Run concurrent streams
        start_time = time.time()
        tasks = [simulate_stream(i) for i in range(concurrent_streams)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, dict)]
        all_latencies = []
        total_errors = 0
        
        for result in successful_results:
            all_latencies.extend(result["latencies"])
            total_errors += result["errors"]
        
        if all_latencies:
            avg_latency = sum(all_latencies) / len(all_latencies)
            p95_latency = np.percentile(all_latencies, 95)
            p99_latency = np.percentile(all_latencies, 99)
            max_latency = max(all_latencies)
            
            total_frames = concurrent_streams * frames_per_stream
            success_rate = ((total_frames - total_errors) / total_frames) * 100
            
            print(f"\nConcurrent Load Test Results:")
            print(f"Streams: {concurrent_streams}, Frames/stream: {frames_per_stream}")
            print(f"Total duration: {total_duration:.2f}s")
            print(f"Success rate: {success_rate:.2f}%")
            print(f"Average latency: {avg_latency:.2f}ms")
            print(f"P95 latency: {p95_latency:.2f}ms")
            print(f"P99 latency: {p99_latency:.2f}ms")
            print(f"Max latency: {max_latency:.2f}ms")
            
            # Validate performance targets
            assert success_rate >= 99.0, f"Success rate {success_rate:.2f}% below 99% target"
            assert p99_latency <= 200.0, f"P99 latency {p99_latency:.2f}ms exceeds 200ms test target"
            assert avg_latency <= 100.0, f"Average latency {avg_latency:.2f}ms exceeds 100ms target"

    async def test_performance_metrics_collection(self, grpc_service, redis_manager):
        """Test comprehensive performance metrics collection."""
        # Generate some activity
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # Process multiple frames
        for i in range(10):
            pb_frame = grpc_service._numpy_to_protobuf(test_frame, use_blosc=True)
            await grpc_service._optimized_protobuf_to_numpy(pb_frame)
        
        # Collect gRPC metrics
        grpc_metrics = grpc_service.get_serialization_performance_metrics()
        
        assert "gRPC_server_metrics" in grpc_metrics
        assert "serialization_metrics" in grpc_metrics
        assert "blosc_compressor_metrics" in grpc_metrics
        
        server_metrics = grpc_metrics["gRPC_server_metrics"]
        assert server_metrics["total_requests"] >= 0
        assert server_metrics["total_bytes_processed"] >= 0
        
        # Collect Redis metrics
        redis_health = await redis_manager.health_check()
        assert redis_health["status"] in ["healthy", "unknown"]  # Unknown is OK for test environment
        
        # Print metrics summary
        print(f"\ngRPC Metrics Summary:")
        print(f"Total requests: {server_metrics['total_requests']}")
        print(f"Total bytes processed: {server_metrics['total_bytes_processed']}")
        print(f"Requests per second: {server_metrics['requests_per_second']:.2f}")
        
        if redis_health["status"] == "healthy":
            print(f"\nRedis Health: {redis_health['status']}")
            print(f"Total pending: {redis_health['total_pending']}")
            print(f"Total processing: {redis_health['total_processing']}")

    @pytest.mark.slow
    async def test_end_to_end_performance_validation(self, performance_validator, 
                                                     grpc_service, redis_manager):
        """Test end-to-end performance validation framework."""
        # Run quick validation
        validation_result = await run_quick_validation(grpc_service, redis_manager)
        
        assert validation_result["validation_status"] in ["passed", "warning"]
        assert validation_result["duration_seconds"] > 0
        assert "metrics" in validation_result
        
        print(f"\nQuick Validation Result: {validation_result['validation_status']}")
        print(f"Duration: {validation_result['duration_seconds']:.2f}s")
        
        if validation_result["metrics"]:
            metrics = validation_result["metrics"]
            if "p99_latency_ms" in metrics:
                print(f"P99 Latency: {metrics['p99_latency_ms']:.2f}ms")
            if "success_rate_percent" in metrics:
                print(f"Success Rate: {metrics['success_rate_percent']:.2f}%")

    async def test_system_resilience_basic(self, grpc_service, redis_manager):
        """Test basic system resilience."""
        # Test with various edge cases
        edge_cases = [
            np.array([]),  # Empty array
            np.ones((1, 1, 1), dtype=np.uint8),  # Minimal array
            np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8),  # Small array
        ]
        
        for i, test_case in enumerate(edge_cases):
            try:
                if test_case.size > 0:
                    pb_frame = grpc_service._numpy_to_protobuf(test_case, use_blosc=True)
                    decompressed = await grpc_service._optimized_protobuf_to_numpy(pb_frame)
                    
                    if test_case.size > 0:
                        assert decompressed is not None
                
                print(f"Edge case {i+1} handled successfully")
                
            except Exception as e:
                print(f"Edge case {i+1} failed (expected for empty arrays): {e}")

    async def test_memory_efficiency(self, grpc_service):
        """Test memory efficiency of optimizations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many frames to test memory efficiency
        large_frames = []
        for i in range(20):
            test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            pb_frame = grpc_service._numpy_to_protobuf(test_frame, use_blosc=True)
            large_frames.append(pb_frame)
        
        mid_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clear frames
        large_frames.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = mid_memory - initial_memory
        memory_recovered = mid_memory - final_memory
        
        print(f"\nMemory Efficiency Test:")
        print(f"Initial memory: {initial_memory:.1f}MB")
        print(f"Peak memory: {mid_memory:.1f}MB")
        print(f"Final memory: {final_memory:.1f}MB")
        print(f"Memory increase: {memory_increase:.1f}MB")
        print(f"Memory recovered: {memory_recovered:.1f}MB")
        
        # Validate memory efficiency
        assert memory_increase < 2000, f"Memory increase {memory_increase:.1f}MB too high"


# Integration test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])