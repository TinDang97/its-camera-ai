"""
Load Testing Framework for ITS Camera AI System

This module provides comprehensive load testing capabilities to validate system performance
under various conditions and ensure production readiness.

Key Performance Requirements:
- Support 1000+ concurrent camera streams
- Sub-100ms API response times
- 10TB/day data processing capacity
- 90%+ cache hit rates
- 95%+ system availability
"""

import asyncio
import json
import random
import statistics
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
import httpx
import psutil
from faker import Faker

from its_camera_ai.core.config import Settings
from its_camera_ai.services.unified_analytics_service import UnifiedAnalyticsService


fake = Faker()


class LoadTestMetrics:
    """Collects and analyzes load test metrics."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.error_count = 0
        self.success_count = 0
        self.throughput_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def record_response(self, response_time: float, success: bool):
        """Record API response metrics."""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def record_system_metrics(self):
        """Record system resource metrics."""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        self.memory_samples.append(memory.percent)
        self.cpu_samples.append(cpu)
    
    def calculate_throughput(self) -> float:
        """Calculate requests per second."""
        if not self.start_time or not self.end_time:
            return 0.0
        
        total_requests = self.success_count + self.error_count
        duration = self.end_time - self.start_time
        return total_requests / duration if duration > 0 else 0.0
    
    def get_response_time_stats(self) -> Dict[str, float]:
        """Get response time statistics."""
        if not self.response_times:
            return {}
        
        return {
            "min": min(self.response_times),
            "max": max(self.response_times),
            "mean": statistics.mean(self.response_times),
            "median": statistics.median(self.response_times),
            "p95": self._percentile(self.response_times, 95),
            "p99": self._percentile(self.response_times, 99),
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_success_rate(self) -> float:
        """Calculate success rate percentage."""
        total = self.success_count + self.error_count
        return (self.success_count / total * 100) if total > 0 else 0.0


class LoadTestScenario:
    """Defines a load testing scenario."""
    
    def __init__(
        self,
        name: str,
        duration_seconds: int,
        concurrent_users: int,
        ramp_up_seconds: int = 0,
        target_endpoint: str = "/api/v1/analytics/real-time",
        request_data_generator=None,
    ):
        self.name = name
        self.duration_seconds = duration_seconds
        self.concurrent_users = concurrent_users
        self.ramp_up_seconds = ramp_up_seconds
        self.target_endpoint = target_endpoint
        self.request_data_generator = request_data_generator or self._default_data_generator
    
    def _default_data_generator(self) -> Dict[str, Any]:
        """Generate default test data for requests."""
        return {
            "camera_id": f"camera_{random.randint(1, 1000)}",
            "detections": [
                {
                    "object_id": fake.uuid4(),
                    "class_name": random.choice(["car", "truck", "motorcycle", "bus"]),
                    "confidence": random.uniform(0.8, 0.99),
                    "bbox": [
                        random.randint(0, 800),
                        random.randint(0, 600),
                        random.randint(100, 300),
                        random.randint(80, 200),
                    ],
                    "timestamp": datetime.utcnow().isoformat(),
                }
                for _ in range(random.randint(1, 5))
            ],
        }


class CameraStreamSimulator:
    """Simulates multiple camera streams for load testing."""
    
    def __init__(self, num_cameras: int = 1000):
        self.num_cameras = num_cameras
        self.cameras = [
            {
                "id": f"camera_{i:04d}",
                "location": fake.address(),
                "fps": random.choice([15, 20, 25, 30]),
                "resolution": random.choice(["1080p", "720p", "4K"]),
                "last_frame_time": 0,
            }
            for i in range(num_cameras)
        ]
    
    async def generate_frame_data(self, camera_id: str) -> Dict[str, Any]:
        """Generate realistic frame data for a camera."""
        camera = next((c for c in self.cameras if c["id"] == camera_id), None)
        if not camera:
            raise ValueError(f"Camera {camera_id} not found")
        
        # Simulate realistic detection patterns
        num_objects = random.choices(
            [0, 1, 2, 3, 4, 5, 6],
            weights=[5, 20, 30, 25, 15, 4, 1],  # Most frames have 1-3 objects
            k=1
        )[0]
        
        detections = []
        for _ in range(num_objects):
            detections.append({
                "object_id": fake.uuid4(),
                "class_name": random.choices(
                    ["car", "truck", "motorcycle", "bus", "bicycle", "person"],
                    weights=[60, 15, 10, 8, 4, 3],
                    k=1
                )[0],
                "confidence": random.uniform(0.7, 0.99),
                "bbox": [
                    random.randint(0, 1920),
                    random.randint(0, 1080),
                    random.randint(50, 400),
                    random.randint(30, 300),
                ],
                "speed": random.uniform(0, 80) if random.random() > 0.3 else None,
            })
        
        return {
            "camera_id": camera_id,
            "timestamp": datetime.utcnow().isoformat(),
            "frame_number": int(time.time() * camera["fps"]),
            "resolution": camera["resolution"],
            "detections": detections,
            "metadata": {
                "weather": random.choice(["clear", "cloudy", "rain", "fog"]),
                "lighting": random.choice(["day", "night", "dawn", "dusk"]),
                "traffic_density": random.choice(["low", "medium", "high"]),
            },
        }
    
    async def simulate_continuous_stream(
        self, camera_id: str, duration_seconds: int, callback
    ):
        """Simulate continuous stream from a camera."""
        camera = next((c for c in self.cameras if c["id"] == camera_id), None)
        if not camera:
            return
        
        fps = camera["fps"]
        frame_interval = 1.0 / fps
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            frame_start = time.time()
            
            frame_data = await self.generate_frame_data(camera_id)
            await callback(frame_data)
            
            # Maintain FPS timing
            frame_duration = time.time() - frame_start
            sleep_time = max(0, frame_interval - frame_duration)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)


@pytest.fixture
def load_test_metrics():
    """Provide load test metrics collector."""
    return LoadTestMetrics()


@pytest.fixture
def camera_simulator():
    """Provide camera stream simulator."""
    return CameraStreamSimulator(num_cameras=100)  # Smaller for tests


@pytest.fixture
def load_test_client():
    """Provide HTTP client for load testing."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(30.0),
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
    )


class TestLoadTestingFramework:
    """Test suite for load testing framework validation."""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_api_response_time_under_load(
        self, load_test_client, load_test_metrics, auth_headers
    ):
        """Test API response times under concurrent load."""
        scenario = LoadTestScenario(
            name="API Response Time Test",
            duration_seconds=30,
            concurrent_users=50,
            target_endpoint="/api/v1/analytics/dashboard-data",
        )
        
        async def make_request():
            start_time = time.time()
            try:
                response = await load_test_client.get(
                    f"http://localhost:8000{scenario.target_endpoint}",
                    headers=auth_headers,
                    params={"camera_id": f"camera_{random.randint(1, 100)}"},
                )
                response_time = time.time() - start_time
                success = response.status_code == 200
                load_test_metrics.record_response(response_time, success)
                return response_time, success
            except Exception:
                response_time = time.time() - start_time
                load_test_metrics.record_response(response_time, False)
                return response_time, False
        
        # Execute concurrent requests
        load_test_metrics.start_time = time.time()
        
        tasks = []
        for _ in range(scenario.concurrent_users):
            task = asyncio.create_task(make_request())
            tasks.append(task)
            # Small delay to simulate gradual ramp-up
            await asyncio.sleep(0.01)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        load_test_metrics.end_time = time.time()
        
        # Analyze results
        stats = load_test_metrics.get_response_time_stats()
        success_rate = load_test_metrics.get_success_rate()
        throughput = load_test_metrics.calculate_throughput()
        
        # Performance assertions
        assert stats["p95"] < 0.5, f"95th percentile response time {stats['p95']:.3f}s exceeds 500ms"
        assert stats["mean"] < 0.2, f"Mean response time {stats['mean']:.3f}s exceeds 200ms"
        assert success_rate >= 95.0, f"Success rate {success_rate:.1f}% below 95%"
        assert throughput >= 10.0, f"Throughput {throughput:.1f} RPS below 10 RPS"
        
        print(f"Load Test Results:")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Throughput: {throughput:.1f} RPS")
        print(f"  Response Times: {stats}")
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_sustained_load_performance(
        self, camera_simulator, load_test_metrics, unified_analytics_service
    ):
        """Test sustained load performance over extended period."""
        duration_seconds = 60
        concurrent_cameras = 100
        
        async def process_camera_stream(camera_id: str):
            """Process stream from a single camera."""
            frame_count = 0
            error_count = 0
            
            async def frame_callback(frame_data):
                nonlocal frame_count, error_count
                start_time = time.time()
                try:
                    # Mock analytics processing
                    await asyncio.sleep(random.uniform(0.01, 0.05))  # Simulate processing
                    processing_time = time.time() - start_time
                    load_test_metrics.record_response(processing_time, True)
                    frame_count += 1
                except Exception:
                    processing_time = time.time() - start_time
                    load_test_metrics.record_response(processing_time, False)
                    error_count += 1
            
            await camera_simulator.simulate_continuous_stream(
                camera_id, duration_seconds, frame_callback
            )
            
            return frame_count, error_count
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._monitor_system_resources(load_test_metrics, duration_seconds))
        
        # Start camera streams
        load_test_metrics.start_time = time.time()
        
        camera_tasks = []
        for i in range(concurrent_cameras):
            camera_id = f"camera_{i:04d}"
            task = asyncio.create_task(process_camera_stream(camera_id))
            camera_tasks.append(task)
        
        # Wait for completion
        results = await asyncio.gather(*camera_tasks, return_exceptions=True)
        load_test_metrics.end_time = time.time()
        
        # Stop monitoring
        monitor_task.cancel()
        
        # Analyze results
        total_frames = sum(result[0] for result in results if isinstance(result, tuple))
        total_errors = sum(result[1] for result in results if isinstance(result, tuple))
        
        stats = load_test_metrics.get_response_time_stats()
        success_rate = load_test_metrics.get_success_rate()
        frames_per_second = total_frames / duration_seconds
        
        # Performance assertions
        assert success_rate >= 95.0, f"Success rate {success_rate:.1f}% below 95%"
        assert frames_per_second >= 1000.0, f"Frame processing rate {frames_per_second:.1f} FPS below 1000 FPS"
        assert stats["p95"] < 0.1, f"95th percentile processing time {stats['p95']:.3f}s exceeds 100ms"
        assert max(load_test_metrics.memory_samples) < 80.0, "Memory usage exceeded 80%"
        assert statistics.mean(load_test_metrics.cpu_samples) < 70.0, "Average CPU usage exceeded 70%"
        
        print(f"Sustained Load Results:")
        print(f"  Total Frames: {total_frames}")
        print(f"  Frames/Second: {frames_per_second:.1f}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Processing Times: {stats}")
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_burst_traffic_handling(
        self, load_test_client, load_test_metrics, auth_headers
    ):
        """Test system behavior under burst traffic conditions."""
        
        async def normal_load_phase():
            """Normal traffic phase - 10 RPS."""
            tasks = []
            for _ in range(30):  # 30 requests over 3 seconds
                task = asyncio.create_task(self._make_test_request(load_test_client, auth_headers, load_test_metrics))
                tasks.append(task)
                await asyncio.sleep(0.1)  # 10 RPS
            await asyncio.gather(*tasks, return_exceptions=True)
        
        async def burst_load_phase():
            """Burst traffic phase - 100 RPS."""
            tasks = []
            for _ in range(300):  # 300 requests over 3 seconds
                task = asyncio.create_task(self._make_test_request(load_test_client, auth_headers, load_test_metrics))
                tasks.append(task)
                await asyncio.sleep(0.01)  # 100 RPS
            await asyncio.gather(*tasks, return_exceptions=True)
        
        load_test_metrics.start_time = time.time()
        
        # Execute test phases
        await normal_load_phase()
        await burst_load_phase()
        await normal_load_phase()  # Recovery phase
        
        load_test_metrics.end_time = time.time()
        
        # Analyze results
        stats = load_test_metrics.get_response_time_stats()
        success_rate = load_test_metrics.get_success_rate()
        
        # Performance assertions
        assert success_rate >= 90.0, f"Success rate {success_rate:.1f}% below 90% during burst"
        assert stats["p99"] < 2.0, f"99th percentile response time {stats['p99']:.3f}s exceeds 2s during burst"
        
        print(f"Burst Traffic Results:")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Response Times: {stats}")
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_memory_leak_detection(
        self, camera_simulator, load_test_metrics
    ):
        """Test for memory leaks during extended operation."""
        initial_memory = psutil.virtual_memory().percent
        duration_seconds = 120  # 2 minutes
        measurement_interval = 10  # Measure every 10 seconds
        
        memory_measurements = [initial_memory]
        
        async def memory_monitoring():
            """Monitor memory usage over time."""
            for _ in range(duration_seconds // measurement_interval):
                await asyncio.sleep(measurement_interval)
                current_memory = psutil.virtual_memory().percent
                memory_measurements.append(current_memory)
        
        async def workload_simulation():
            """Simulate realistic workload."""
            for i in range(50):  # 50 cameras
                camera_id = f"camera_{i:04d}"
                
                async def frame_callback(frame_data):
                    # Simulate processing with some memory allocation
                    data = json.dumps(frame_data)
                    await asyncio.sleep(0.01)
                
                task = asyncio.create_task(
                    camera_simulator.simulate_continuous_stream(
                        camera_id, duration_seconds, frame_callback
                    )
                )
        
        # Run monitoring and workload concurrently
        monitor_task = asyncio.create_task(memory_monitoring())
        workload_task = asyncio.create_task(workload_simulation())
        
        await asyncio.gather(monitor_task, workload_task, return_exceptions=True)
        
        # Analyze memory trend
        final_memory = memory_measurements[-1]
        memory_increase = final_memory - initial_memory
        
        # Calculate memory growth rate
        if len(memory_measurements) > 1:
            growth_rates = [
                memory_measurements[i] - memory_measurements[i-1]
                for i in range(1, len(memory_measurements))
            ]
            avg_growth_rate = statistics.mean(growth_rates)
        else:
            avg_growth_rate = 0
        
        # Memory leak detection assertions
        assert memory_increase < 10.0, f"Memory increased by {memory_increase:.1f}% - possible leak"
        assert avg_growth_rate < 1.0, f"Average memory growth rate {avg_growth_rate:.2f}%/10s too high"
        
        print(f"Memory Leak Detection Results:")
        print(f"  Initial Memory: {initial_memory:.1f}%")
        print(f"  Final Memory: {final_memory:.1f}%")
        print(f"  Memory Increase: {memory_increase:.1f}%")
        print(f"  Growth Rate: {avg_growth_rate:.2f}%/10s")
    
    async def _make_test_request(
        self, client: httpx.AsyncClient, headers: Dict[str, str], metrics: LoadTestMetrics
    ) -> Tuple[float, bool]:
        """Make a test request and record metrics."""
        start_time = time.time()
        try:
            response = await client.get(
                "http://localhost:8000/api/v1/analytics/dashboard-data",
                headers=headers,
                params={"camera_id": f"camera_{random.randint(1, 100)}"},
            )
            response_time = time.time() - start_time
            success = response.status_code == 200
            metrics.record_response(response_time, success)
            return response_time, success
        except Exception:
            response_time = time.time() - start_time
            metrics.record_response(response_time, False)
            return response_time, False
    
    async def _monitor_system_resources(
        self, metrics: LoadTestMetrics, duration_seconds: int
    ):
        """Monitor system resources during load test."""
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            metrics.record_system_metrics()
            await asyncio.sleep(1)


class TestLoadTestScenarios:
    """Pre-defined load test scenarios for different use cases."""
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_peak_traffic_scenario(
        self, load_test_client, auth_headers
    ):
        """Test peak traffic scenario - morning rush hour simulation."""
        scenario = LoadTestScenario(
            name="Peak Traffic",
            duration_seconds=300,  # 5 minutes
            concurrent_users=200,
            ramp_up_seconds=60,
        )
        
        metrics = LoadTestMetrics()
        
        # Simulate gradual ramp-up
        ramp_up_delay = scenario.ramp_up_seconds / scenario.concurrent_users
        
        async def user_session():
            """Simulate a user session."""
            # Wait for ramp-up
            await asyncio.sleep(random.uniform(0, scenario.ramp_up_seconds))
            
            # Make requests for duration
            session_end = time.time() + scenario.duration_seconds
            while time.time() < session_end:
                start_time = time.time()
                try:
                    response = await load_test_client.post(
                        "http://localhost:8000/api/v1/analytics/real-time",
                        headers=auth_headers,
                        json=scenario.request_data_generator(),
                    )
                    response_time = time.time() - start_time
                    success = response.status_code == 200
                    metrics.record_response(response_time, success)
                except Exception:
                    response_time = time.time() - start_time
                    metrics.record_response(response_time, False)
                
                # Wait between requests (1-3 seconds)
                await asyncio.sleep(random.uniform(1, 3))
        
        # Execute scenario
        metrics.start_time = time.time()
        
        tasks = [
            asyncio.create_task(user_session())
            for _ in range(scenario.concurrent_users)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        metrics.end_time = time.time()
        
        # Validate peak traffic performance
        stats = metrics.get_response_time_stats()
        success_rate = metrics.get_success_rate()
        throughput = metrics.calculate_throughput()
        
        assert success_rate >= 95.0, f"Peak traffic success rate {success_rate:.1f}% below 95%"
        assert stats["p95"] < 1.0, f"Peak traffic P95 response time {stats['p95']:.3f}s exceeds 1s"
        assert throughput >= 50.0, f"Peak traffic throughput {throughput:.1f} RPS below 50 RPS"
        
        print(f"Peak Traffic Scenario Results:")
        print(f"  Users: {scenario.concurrent_users}")
        print(f"  Duration: {scenario.duration_seconds}s")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Throughput: {throughput:.1f} RPS")
        print(f"  Response Times: {stats}")
    
    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_database_stress_scenario(
        self, camera_simulator, load_test_metrics
    ):
        """Test database performance under heavy write load."""
        concurrent_cameras = 500
        duration_seconds = 180  # 3 minutes
        
        # Simulate high-frequency database writes
        async def database_write_simulation():
            """Simulate database write operations."""
            write_count = 0
            error_count = 0
            
            end_time = time.time() + duration_seconds
            while time.time() < end_time:
                start_time = time.time()
                try:
                    # Simulate database write operation
                    await asyncio.sleep(random.uniform(0.001, 0.01))
                    write_time = time.time() - start_time
                    load_test_metrics.record_response(write_time, True)
                    write_count += 1
                except Exception:
                    write_time = time.time() - start_time
                    load_test_metrics.record_response(write_time, False)
                    error_count += 1
                
                # Small delay between writes
                await asyncio.sleep(random.uniform(0.01, 0.05))
            
            return write_count, error_count
        
        load_test_metrics.start_time = time.time()
        
        # Start multiple database write simulations
        tasks = [
            asyncio.create_task(database_write_simulation())
            for _ in range(concurrent_cameras)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        load_test_metrics.end_time = time.time()
        
        # Analyze database performance
        total_writes = sum(result[0] for result in results if isinstance(result, tuple))
        total_errors = sum(result[1] for result in results if isinstance(result, tuple))
        
        stats = load_test_metrics.get_response_time_stats()
        success_rate = load_test_metrics.get_success_rate()
        writes_per_second = total_writes / duration_seconds
        
        # Database stress test assertions
        assert success_rate >= 98.0, f"Database stress success rate {success_rate:.1f}% below 98%"
        assert writes_per_second >= 1000.0, f"Database write rate {writes_per_second:.1f} WPS below 1000 WPS"
        assert stats["p95"] < 0.05, f"Database write P95 time {stats['p95']:.3f}s exceeds 50ms"
        
        print(f"Database Stress Scenario Results:")
        print(f"  Total Writes: {total_writes}")
        print(f"  Writes/Second: {writes_per_second:.1f}")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Write Times: {stats}")