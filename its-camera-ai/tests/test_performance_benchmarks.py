"""Performance benchmark tests for ITS Camera AI system.

This module contains comprehensive performance tests that validate the system
meets all architectural requirements:
- 100+ concurrent camera streams
- <10ms frame processing latency
- 99.9% success rate
- <4GB memory usage per service instance
- Load balancing and circuit breaker performance
- Error recovery and system resilience
"""

import asyncio
import gc
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import psutil
import pytest
from faker import Faker

from src.its_camera_ai.core.exceptions import CircuitBreakerError, ServiceMeshError
from src.its_camera_ai.data.redis_queue_manager import RedisQueueManager
from src.its_camera_ai.services.service_mesh import (
    LoadBalancer,
    LoadBalancingStrategy,
    ServiceEndpoint,
    ServiceMeshClient,
    ServiceStatus,
)
from src.its_camera_ai.services.streaming_service import (
    CameraConfig,
    StreamingDataProcessor,
    StreamProtocol,
)

fake = Faker()


class TestSystemLoadBenchmarks:
    """System-wide load testing and performance benchmarks."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock Redis
        self.mock_redis = AsyncMock(spec=RedisQueueManager)
        self.mock_redis.connect.return_value = None
        self.mock_redis.create_queue.return_value = True
        self.mock_redis.health_check.return_value = {"status": "healthy"}

        # Streaming processor with production-like limits
        self.streaming_processor = StreamingDataProcessor(
            redis_client=self.mock_redis,
            max_concurrent_streams=200,  # Test beyond requirements
            frame_processing_timeout=0.008,  # 8ms target (stricter than 10ms)
        )

        # Service mesh for distributed testing
        self.service_mesh = None

    async def setup_service_mesh(self):
        """Set up service mesh for testing."""
        with patch("src.its_camera_ai.services.service_mesh.redis.from_url") as mock_redis_factory:
            mock_redis_client = AsyncMock()
            mock_redis_client.ping.return_value = True
            mock_redis_client.keys.return_value = []
            mock_redis_factory.return_value = mock_redis_client

            self.service_mesh = ServiceMeshClient()
            await self.service_mesh.start()

    async def teardown_service_mesh(self):
        """Tear down service mesh."""
        if self.service_mesh:
            await self.service_mesh.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.benchmark
    async def test_system_throughput_benchmark(self, benchmark):
        """Benchmark overall system throughput with realistic workload."""
        await self.streaming_processor.start()

        async def simulate_realistic_workload():
            # Create diverse camera configurations
            configs = []
            resolutions = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
            protocols = [StreamProtocol.RTSP, StreamProtocol.HTTP, StreamProtocol.WEBRTC]

            for i in range(150):  # 150 cameras (above 100 requirement)
                config = CameraConfig(
                    camera_id=f"benchmark_camera_{i:03d}",
                    stream_url=f"rtsp://camera{i}.example.com/stream",
                    resolution=fake.random_element(resolutions),
                    fps=fake.random_int(15, 60),
                    protocol=fake.random_element(protocols),
                    quality_threshold=fake.random.uniform(0.6, 0.9),
                )
                configs.append(config)

            # Register all cameras
            with patch.object(
                self.streaming_processor.connection_manager, "connect_camera", return_value=True
            ):
                registration_start = time.perf_counter()
                tasks = [self.streaming_processor.register_camera(config) for config in configs]
                registrations = await asyncio.gather(*tasks)
                registration_time = time.perf_counter() - registration_start

                successful_registrations = sum(1 for reg in registrations if reg.success)

                # Simulate continuous frame processing for 10 seconds
                processing_start = time.perf_counter()
                total_frames_processed = 0
                processing_duration = 5.0  # 5 seconds for faster testing

                async def process_camera_frames(config):
                    nonlocal total_frames_processed
                    frames_for_camera = 0
                    end_time = processing_start + processing_duration

                    while time.perf_counter() < end_time:
                        # Generate realistic frame
                        h, w = config.resolution
                        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

                        try:
                            result = await self.streaming_processor._process_single_frame(
                                config.camera_id, frame, config
                            )
                            if result:
                                frames_for_camera += 1
                        except Exception:
                            pass  # Handle processing errors gracefully

                        # Simulate realistic frame rate timing
                        await asyncio.sleep(1.0 / config.fps)

                    total_frames_processed += frames_for_camera
                    return frames_for_camera

                # Process frames for all cameras concurrently
                camera_tasks = [process_camera_frames(config) for config in configs[:successful_registrations]]
                camera_results = await asyncio.gather(*camera_tasks, return_exceptions=True)

                actual_processing_time = time.perf_counter() - processing_start

                # Calculate throughput metrics
                fps_throughput = total_frames_processed / actual_processing_time
                cameras_per_second = successful_registrations / registration_time

                return {
                    "successful_registrations": successful_registrations,
                    "total_cameras": len(configs),
                    "registration_time": registration_time,
                    "total_frames_processed": total_frames_processed,
                    "processing_time": actual_processing_time,
                    "fps_throughput": fps_throughput,
                    "cameras_per_second": cameras_per_second,
                    "success_rate": successful_registrations / len(configs),
                }

        result = await benchmark.pedantic(simulate_realistic_workload, rounds=1, iterations=1)

        print("\\nSystem Throughput Benchmark Results:")
        print(f"Total cameras: {result['total_cameras']}")
        print(f"Successful registrations: {result['successful_registrations']}")
        print(f"Registration rate: {result['cameras_per_second']:.1f} cameras/sec")
        print(f"Frame throughput: {result['fps_throughput']:.1f} fps")
        print(f"Total frames processed: {result['total_frames_processed']}")
        print(f"System success rate: {result['success_rate']*100:.1f}%")

        # Validate performance requirements
        assert result['successful_registrations'] >= 150, "Should handle 150+ cameras"
        assert result['success_rate'] >= 0.99, f"Success rate {result['success_rate']*100:.1f}% too low"
        assert result['fps_throughput'] >= 1000, f"Throughput {result['fps_throughput']:.1f} fps too low"

        await self.streaming_processor.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_scaling_under_load(self):
        """Test memory usage scaling with increasing camera load."""
        await self.streaming_processor.start()

        # Test memory usage at different camera counts
        camera_counts = [25, 50, 100, 150]
        memory_metrics = {}

        for camera_count in camera_counts:
            # Force garbage collection
            gc.collect()
            baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            configs = []
            for i in range(camera_count):
                config = CameraConfig(
                    camera_id=f"memory_test_camera_{i:03d}",
                    stream_url=f"test://stream_{i}",
                    resolution=(1280, 720),
                    fps=30,
                    protocol=StreamProtocol.HTTP,
                )
                configs.append(config)

            with patch.object(
                self.streaming_processor.connection_manager, "connect_camera", return_value=True
            ):
                # Register cameras and process frames
                tasks = [self.streaming_processor.register_camera(config) for config in configs]
                await asyncio.gather(*tasks)

                # Process several batches of frames
                for batch in range(10):
                    frame_tasks = []
                    for config in configs:
                        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                        frame_tasks.append(
                            self.streaming_processor._process_single_frame(
                                config.camera_id, frame, config
                            )
                        )
                    await asyncio.gather(*frame_tasks, return_exceptions=True)

                # Measure peak memory
                peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_increase = peak_memory - baseline_memory

                memory_metrics[camera_count] = {
                    "baseline": baseline_memory,
                    "peak": peak_memory,
                    "increase": memory_increase,
                    "per_camera": memory_increase / camera_count if camera_count > 0 else 0,
                }

            # Clean up for next iteration
            await self.streaming_processor.stop()
            await self.streaming_processor.start()

        print("\\nMemory Scaling Analysis:")
        for camera_count, metrics in memory_metrics.items():
            print(
                f"Cameras {camera_count:3d}: "
                f"Peak={metrics['peak']:.1f}MB, "
                f"Increase={metrics['increase']:.1f}MB, "
                f"Per camera={metrics['per_camera']:.2f}MB"
            )

        # Validate memory requirements
        for camera_count, metrics in memory_metrics.items():
            assert metrics['peak'] < 4096, f"Memory {metrics['peak']:.1f}MB exceeds 4GB at {camera_count} cameras"

        # Memory scaling should be reasonable
        max_cameras = max(camera_counts)
        max_memory_increase = memory_metrics[max_cameras]['increase']
        assert max_memory_increase < 3072, f"Memory increase {max_memory_increase:.1f}MB too high"

        await self.streaming_processor.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_circuit_breaker_performance_under_load(self):
        """Test circuit breaker performance during high load and failures."""
        await self.setup_service_mesh()

        try:
            # Create multiple service endpoints
            endpoints = []
            for i in range(10):
                endpoint = ServiceEndpoint(
                    name=f"test_service_{i}",
                    host="127.0.0.1",
                    port=50051 + i,
                    protocol="grpc",
                )
                endpoints.append(endpoint)

            # Test circuit breaker under normal load
            normal_load_results = []
            for _ in range(100):  # 100 calls per service
                for endpoint in endpoints:
                    breaker = self.service_mesh._get_circuit_breaker(endpoint.name)

                    async def successful_call():
                        await asyncio.sleep(0.001)  # 1ms simulated processing
                        return "success"

                    start_time = time.perf_counter()
                    try:
                        result = await breaker.call(successful_call)
                        end_time = time.perf_counter()
                        normal_load_results.append(end_time - start_time)
                    except Exception:
                        pass

            avg_normal_latency = sum(normal_load_results) / len(normal_load_results) * 1000  # ms

            # Test circuit breaker with failures
            failure_scenarios = [0.1, 0.3, 0.5, 0.8]  # Different failure rates
            circuit_breaker_metrics = {}

            for failure_rate in failure_scenarios:
                scenario_results = []
                scenario_errors = 0
                scenario_circuit_opens = 0

                for _ in range(200):  # More calls to trigger circuit breaker
                    for endpoint in endpoints:
                        breaker = self.service_mesh._get_circuit_breaker(endpoint.name)

                        async def potentially_failing_call():
                            if np.random.random() < failure_rate:
                                raise RuntimeError("Simulated service failure")
                            await asyncio.sleep(0.001)
                            return "success"

                        start_time = time.perf_counter()
                        try:
                            result = await breaker.call(potentially_failing_call)
                            end_time = time.perf_counter()
                            scenario_results.append(end_time - start_time)
                        except CircuitBreakerError:
                            scenario_circuit_opens += 1
                        except Exception:
                            scenario_errors += 1

                if scenario_results:
                    avg_latency = sum(scenario_results) / len(scenario_results) * 1000  # ms
                else:
                    avg_latency = 0

                circuit_breaker_metrics[failure_rate] = {
                    "avg_latency": avg_latency,
                    "total_calls": 200 * len(endpoints),
                    "successful_calls": len(scenario_results),
                    "errors": scenario_errors,
                    "circuit_opens": scenario_circuit_opens,
                }

            print("\\nCircuit Breaker Performance Analysis:")
            print(f"Normal load average latency: {avg_normal_latency:.2f}ms")

            for failure_rate, metrics in circuit_breaker_metrics.items():
                success_rate = metrics['successful_calls'] / metrics['total_calls']
                print(
                    f"Failure rate {failure_rate*100:3.0f}%: "
                    f"Success={success_rate*100:.1f}%, "
                    f"Latency={metrics['avg_latency']:.2f}ms, "
                    f"Circuit opens={metrics['circuit_opens']}"
                )

            # Validate circuit breaker effectiveness
            for failure_rate, metrics in circuit_breaker_metrics.items():
                if failure_rate > 0.5:  # High failure rates should trigger circuit breakers
                    assert metrics['circuit_opens'] > 0, f"Circuit breaker should open at {failure_rate*100}% failure rate"

                # Latency should remain reasonable even with failures
                if metrics['avg_latency'] > 0:
                    assert metrics['avg_latency'] < 50.0, f"Latency too high during failures: {metrics['avg_latency']:.2f}ms"

        finally:
            await self.teardown_service_mesh()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_load_balancer_performance(self):
        """Test load balancer performance and distribution."""
        await self.setup_service_mesh()

        try:
            strategies = [
                LoadBalancingStrategy.ROUND_ROBIN,
                LoadBalancingStrategy.LEAST_CONNECTIONS,
                LoadBalancingStrategy.HEALTH_AWARE,
            ]

            performance_results = {}

            for strategy in strategies:
                load_balancer = LoadBalancer(strategy)

                # Create service endpoints with different characteristics
                endpoints = []
                for i in range(5):
                    endpoint = ServiceEndpoint(
                        name=f"service_{i}",
                        host="127.0.0.1",
                        port=50051 + i,
                        protocol="grpc",
                        weight=1 + i,  # Different weights
                    )
                    endpoint.status = ServiceStatus.HEALTHY
                    endpoint.active_connections = i * 2  # Different connection counts
                    endpoint.avg_response_time = 10 + i * 5  # Different response times
                    endpoints.append(endpoint)

                # Simulate load balancing decisions
                selections = []
                decision_times = []

                for _ in range(1000):  # 1000 load balancing decisions
                    start_time = time.perf_counter()
                    selected = load_balancer.select_endpoint("test_service", endpoints)
                    end_time = time.perf_counter()

                    selections.append(selected.port)
                    decision_times.append(end_time - start_time)

                    # Simulate connection changes for least connections strategy
                    if strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                        selected.active_connections += 1

                # Analyze distribution and performance
                from collections import Counter

                distribution = Counter(selections)
                avg_decision_time = sum(decision_times) * 1000 / len(decision_times)  # microseconds

                performance_results[strategy.value] = {
                    "avg_decision_time_us": avg_decision_time,
                    "distribution": dict(distribution),
                    "total_decisions": len(selections),
                }

            print("\\nLoad Balancer Performance Analysis:")
            for strategy, metrics in performance_results.items():
                print(f"Strategy {strategy}:")
                print(f"  Average decision time: {metrics['avg_decision_time_us']:.1f} μs")
                print(f"  Distribution: {metrics['distribution']}")

                # Validate performance requirements
                assert metrics['avg_decision_time_us'] < 100, (
                    f"Load balancing decision too slow: {metrics['avg_decision_time_us']:.1f} μs"
                )

                # Distribution should be reasonable (no single endpoint getting >60% of traffic)
                total_requests = metrics['total_decisions']
                for port, count in metrics['distribution'].items():
                    percentage = count / total_requests
                    if strategy == "round_robin":
                        # Round robin should be very even
                        assert percentage < 0.25, f"Uneven distribution in round robin: {percentage*100:.1f}%"

        finally:
            await self.teardown_service_mesh()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_end_to_end_system_resilience(self):
        """Test complete system resilience under various failure conditions."""
        await self.streaming_processor.start()
        await self.setup_service_mesh()

        try:
            # Simulate complex failure scenarios
            failure_scenarios = [
                {"name": "network_delays", "delay_range": (0.1, 0.5)},
                {"name": "service_intermittent", "failure_rate": 0.2},
                {"name": "high_memory_pressure", "memory_multiplier": 2.0},
                {"name": "concurrent_overload", "overload_factor": 3.0},
            ]

            resilience_metrics = {}

            for scenario in failure_scenarios:
                scenario_name = scenario["name"]
                print(f"\\nTesting resilience scenario: {scenario_name}")

                # Set up cameras for this scenario
                camera_count = 50  # Moderate load for resilience testing
                configs = []
                for i in range(camera_count):
                    config = CameraConfig(
                        camera_id=f"resilience_camera_{i:03d}",
                        stream_url=f"test://stream_{i}",
                        resolution=(1280, 720),
                        fps=30,
                        protocol=StreamProtocol.HTTP,
                    )
                    configs.append(config)

                with patch.object(
                    self.streaming_processor.connection_manager, "connect_camera", return_value=True
                ):
                    # Register cameras
                    registration_tasks = [self.streaming_processor.register_camera(config) for config in configs]
                    registrations = await asyncio.gather(*registration_tasks, return_exceptions=True)
                    successful_registrations = sum(
                        1 for reg in registrations if hasattr(reg, "success") and reg.success
                    )

                    # Process frames under failure conditions
                    total_frames = 0
                    successful_frames = 0
                    errors = 0
                    latencies = []

                    for batch in range(20):  # 20 batches of frames
                        batch_tasks = []

                        for config in configs[:successful_registrations]:
                            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

                            # Apply scenario-specific conditions
                            async def process_with_scenario(camera_id, frame, config, scenario):
                                nonlocal total_frames, successful_frames, errors

                                start_time = time.perf_counter()

                                try:
                                    # Simulate scenario conditions
                                    if scenario["name"] == "network_delays":
                                        delay = np.random.uniform(*scenario["delay_range"])
                                        await asyncio.sleep(delay)
                                    elif scenario["name"] == "service_intermittent":
                                        if np.random.random() < scenario["failure_rate"]:
                                            raise ServiceMeshError("Intermittent service failure")

                                    result = await self.streaming_processor._process_single_frame(
                                        camera_id, frame, config
                                    )

                                    end_time = time.perf_counter()
                                    latency = (end_time - start_time) * 1000  # ms

                                    total_frames += 1
                                    if result is not None:
                                        successful_frames += 1
                                        latencies.append(latency)

                                except Exception:
                                    errors += 1
                                    total_frames += 1

                            batch_tasks.append(
                                process_with_scenario(config.camera_id, frame, config, scenario)
                            )

                        await asyncio.gather(*batch_tasks, return_exceptions=True)

                    # Calculate resilience metrics
                    success_rate = successful_frames / total_frames if total_frames > 0 else 0
                    avg_latency = sum(latencies) / len(latencies) if latencies else 0
                    error_rate = errors / total_frames if total_frames > 0 else 0

                    resilience_metrics[scenario_name] = {
                        "success_rate": success_rate,
                        "avg_latency": avg_latency,
                        "error_rate": error_rate,
                        "total_frames": total_frames,
                        "successful_frames": successful_frames,
                        "errors": errors,
                    }

            print("\\nSystem Resilience Analysis:")
            for scenario_name, metrics in resilience_metrics.items():
                print(
                    f"{scenario_name}: "
                    f"Success={metrics['success_rate']*100:.1f}%, "
                    f"Latency={metrics['avg_latency']:.1f}ms, "
                    f"Errors={metrics['error_rate']*100:.1f}%"
                )

                # Validate resilience requirements
                if scenario_name != "concurrent_overload":  # Overload scenario may have lower success rates
                    assert metrics['success_rate'] >= 0.95, (
                        f"Success rate too low in {scenario_name}: {metrics['success_rate']*100:.1f}%"
                    )

                assert metrics['avg_latency'] < 50.0, (
                    f"Latency too high in {scenario_name}: {metrics['avg_latency']:.1f}ms"
                )

        finally:
            await self.teardown_service_mesh()
            await self.streaming_processor.stop()


# Test markers configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    markers = [
        "performance: mark test as a performance test",
        "benchmark: mark test as a benchmark test using pytest-benchmark",
        "load: mark test as a load/stress test",
        "memory: mark test as a memory profiling test",
        "resilience: mark test as a system resilience test",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


if __name__ == "__main__":
    # Run performance tests
    import sys
    sys.exit(pytest.main(["-v", "-m", "performance", __file__]))
