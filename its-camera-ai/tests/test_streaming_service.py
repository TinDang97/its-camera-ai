"""Comprehensive test suite for the Streaming Service.

This module provides unit tests, integration tests, and load tests for the
Streaming Service implementation, ensuring it meets the performance requirements:
- Support 100+ concurrent camera streams
- Frame processing latency < 10ms
- 99.9% frame processing success rate
- Memory usage < 4GB per service instance
"""

import asyncio
import contextlib
import gc
import time
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import psutil
import pytest

from src.its_camera_ai.data.redis_queue_manager import (
    RedisQueueManager,
)
from src.its_camera_ai.data.streaming_processor import ProcessedFrame, ProcessingStage
from src.its_camera_ai.services.grpc_streaming_server import (
    StreamingServiceImpl,
)

# Internal imports
from src.its_camera_ai.services.streaming_service import (
    CameraConfig,
    CameraConnectionManager,
    FrameQualityValidator,
    StreamingDataProcessor,
    StreamProtocol,
)


class TestFrameQualityValidator:
    """Test suite for frame quality validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = FrameQualityValidator(
            min_resolution=(640, 480), min_quality_score=0.5, max_blur_threshold=100.0
        )

        self.camera_config = CameraConfig(
            camera_id="test_camera_001",
            stream_url="rtsp://test.example.com/stream",
            resolution=(1280, 720),
            fps=30,
            protocol=StreamProtocol.RTSP,
            quality_threshold=0.7,
        )

    @pytest.mark.asyncio
    async def test_validate_high_quality_frame(self):
        """Test validation of a high-quality frame."""
        # Create a high-quality synthetic frame
        frame = self._create_test_frame(1280, 720, quality="high")

        metrics = await self.validator.validate_frame_quality(frame, self.camera_config)

        assert metrics.overall_score >= 0.7
        assert metrics.passed_validation is True
        assert len(metrics.issues) == 0

    @pytest.mark.asyncio
    async def test_validate_low_quality_frame(self):
        """Test validation of a low-quality frame."""
        # Create a low-quality synthetic frame (blurry)
        frame = self._create_test_frame(1280, 720, quality="low")

        metrics = await self.validator.validate_frame_quality(frame, self.camera_config)

        assert metrics.overall_score < 0.7
        assert metrics.passed_validation is False
        assert len(metrics.issues) > 0

    @pytest.mark.asyncio
    async def test_validate_small_resolution_frame(self):
        """Test validation fails for frames below minimum resolution."""
        frame = self._create_test_frame(320, 240, quality="high")

        metrics = await self.validator.validate_frame_quality(frame, self.camera_config)

        assert metrics.passed_validation is False
        assert any("Resolution" in issue for issue in metrics.issues)

    def _create_test_frame(
        self, width: int, height: int, quality: str = "high"
    ) -> np.ndarray:
        """Create a synthetic test frame."""
        if quality == "high":
            # Create a frame with clear patterns and good contrast
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[::20, :] = 255  # Horizontal lines for contrast
            frame[:, ::20] = 255  # Vertical lines
            frame += np.random.randint(0, 50, frame.shape, dtype=np.uint8)  # Some noise
        else:
            # Create a blurry, low-contrast frame
            frame = np.random.randint(100, 150, (height, width, 3), dtype=np.uint8)
            # Apply blur
            import cv2

            frame = cv2.GaussianBlur(frame, (15, 15), 0)

        return frame


class TestCameraConnectionManager:
    """Test suite for camera connection management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = CameraConnectionManager()

        self.rtsp_config = CameraConfig(
            camera_id="rtsp_camera",
            stream_url="rtsp://test.example.com/stream",
            resolution=(1280, 720),
            fps=30,
            protocol=StreamProtocol.RTSP,
        )

        self.webrtc_config = CameraConfig(
            camera_id="webrtc_camera",
            stream_url="webrtc://test.example.com/stream",
            resolution=(1920, 1080),
            fps=30,
            protocol=StreamProtocol.WEBRTC,
        )

    @pytest.mark.asyncio
    @patch("cv2.VideoCapture")
    async def test_connect_rtsp_camera_success(self, mock_video_capture):
        """Test successful RTSP camera connection."""
        # Mock successful video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap

        success = await self.manager.connect_camera(self.rtsp_config)

        assert success is True
        assert self.manager.is_connected("rtsp_camera") is True

        stats = self.manager.get_connection_stats("rtsp_camera")
        assert stats is not None
        assert stats["protocol"] == "rtsp"

    @pytest.mark.asyncio
    @patch("cv2.VideoCapture")
    async def test_connect_rtsp_camera_failure(self, mock_video_capture):
        """Test failed RTSP camera connection."""
        # Mock failed video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap

        success = await self.manager.connect_camera(self.rtsp_config)

        assert success is False
        assert self.manager.is_connected("rtsp_camera") is False

    @pytest.mark.asyncio
    async def test_connect_webrtc_camera(self):
        """Test WebRTC camera connection (mock implementation)."""
        success = await self.manager.connect_camera(self.webrtc_config)

        # Should succeed with mock implementation
        assert success is True
        assert self.manager.is_connected("webrtc_camera") is True

    @pytest.mark.asyncio
    async def test_disconnect_camera(self):
        """Test camera disconnection."""
        # First connect a camera
        await self.manager.connect_camera(self.webrtc_config)
        assert self.manager.is_connected("webrtc_camera") is True

        # Then disconnect
        success = await self.manager.disconnect_camera("webrtc_camera")

        assert success is True
        assert self.manager.is_connected("webrtc_camera") is False

    @pytest.mark.asyncio
    async def test_capture_frame_mock_camera(self):
        """Test frame capture from mock camera."""
        await self.manager.connect_camera(self.webrtc_config)

        frame = await self.manager.capture_frame("webrtc_camera")

        assert frame is not None
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)  # Default mock frame size

        stats = self.manager.get_connection_stats("webrtc_camera")
        assert stats["frames_captured"] == 1


class TestStreamingDataProcessor:
    """Test suite for the main streaming data processor."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create mocked dependencies
        self.mock_redis = AsyncMock(spec=RedisQueueManager)
        self.mock_redis.connect.return_value = None
        self.mock_redis.create_queue.return_value = True
        self.mock_redis.health_check.return_value = {"status": "healthy"}

        self.processor = StreamingDataProcessor(
            redis_client=self.mock_redis,
            max_concurrent_streams=5,  # Low limit for testing
            frame_processing_timeout=0.1,  # 100ms for testing
        )

        self.test_config = CameraConfig(
            camera_id="test_camera",
            stream_url="test://stream",
            resolution=(1280, 720),
            fps=30,
            protocol=StreamProtocol.HTTP,
        )

    @pytest.mark.asyncio
    async def test_start_stop_processor(self):
        """Test processor startup and shutdown."""
        assert self.processor.is_running is False

        await self.processor.start()
        assert self.processor.is_running is True

        # Verify Redis connection and queue setup
        self.mock_redis.connect.assert_called_once()
        assert (
            self.mock_redis.create_queue.call_count >= 3
        )  # Should create multiple queues

        await self.processor.stop()
        assert self.processor.is_running is False

    @pytest.mark.asyncio
    async def test_register_camera_success(self):
        """Test successful camera registration."""
        await self.processor.start()

        with patch.object(
            self.processor.connection_manager, "connect_camera", return_value=True
        ):
            registration = await self.processor.register_camera(self.test_config)

        assert registration.success is True
        assert registration.camera_id == "test_camera"
        assert "test_camera" in self.processor.registered_cameras
        assert self.processor.processing_metrics["active_connections"] == 1

        await self.processor.stop()

    @pytest.mark.asyncio
    async def test_register_camera_max_streams_exceeded(self):
        """Test camera registration failure when max streams exceeded."""
        await self.processor.start()

        # Register cameras up to the limit
        with patch.object(
            self.processor.connection_manager, "connect_camera", return_value=True
        ):
            for i in range(5):  # Max limit is 5
                config = CameraConfig(
                    camera_id=f"camera_{i}",
                    stream_url=f"test://stream_{i}",
                    resolution=(640, 480),
                    fps=30,
                    protocol=StreamProtocol.HTTP,
                )
                registration = await self.processor.register_camera(config)
                assert registration.success is True

            # Try to register one more (should fail)
            extra_config = CameraConfig(
                camera_id="camera_extra",
                stream_url="test://stream_extra",
                resolution=(640, 480),
                fps=30,
                protocol=StreamProtocol.HTTP,
            )
            registration = await self.processor.register_camera(extra_config)
            assert registration.success is False
            assert "Maximum concurrent streams" in registration.message

        await self.processor.stop()

    @pytest.mark.asyncio
    async def test_register_camera_connection_failure(self):
        """Test camera registration failure due to connection issues."""
        await self.processor.start()

        with patch.object(
            self.processor.connection_manager, "connect_camera", return_value=False
        ):
            registration = await self.processor.register_camera(self.test_config)

        assert registration.success is False
        assert "Failed to connect" in registration.message
        assert "test_camera" not in self.processor.registered_cameras

        await self.processor.stop()

    @pytest.mark.asyncio
    async def test_process_single_frame_success(self):
        """Test successful single frame processing."""
        await self.processor.start()

        # Register camera first
        with patch.object(
            self.processor.connection_manager, "connect_camera", return_value=True
        ):
            await self.processor.register_camera(self.test_config)

        # Create high-quality test frame
        frame = self._create_test_frame(1280, 720, quality="high")

        # Process frame
        processed_frame = await self.processor._process_single_frame(
            "test_camera", frame, self.test_config
        )

        assert processed_frame is not None
        assert processed_frame.camera_id == "test_camera"
        assert processed_frame.validation_passed is True
        assert processed_frame.quality_score >= 0.5

        await self.processor.stop()

    @pytest.mark.asyncio
    async def test_process_single_frame_quality_failure(self):
        """Test frame processing with quality validation failure."""
        await self.processor.start()

        with patch.object(
            self.processor.connection_manager, "connect_camera", return_value=True
        ):
            await self.processor.register_camera(self.test_config)

        # Create low-quality test frame
        frame = self._create_test_frame(1280, 720, quality="low")

        processed_frame = await self.processor._process_single_frame(
            "test_camera", frame, self.test_config
        )

        # Should return None due to quality validation failure
        assert processed_frame is None
        assert self.processor.processing_metrics["frames_rejected"] == 1

        await self.processor.stop()

    @pytest.mark.asyncio
    async def test_queue_frame_batch(self):
        """Test batching and queuing of processed frames."""
        await self.processor.start()

        # Create test frames
        frames = []
        for i in range(5):
            frame = ProcessedFrame(
                frame_id=f"frame_{i}",
                camera_id="test_camera",
                timestamp=time.time(),
                original_image=np.zeros((480, 640, 3), dtype=np.uint8),
                quality_score=0.8,
                validation_passed=True,
                processing_stage=ProcessingStage.VALIDATION,
            )
            frames.append(frame)

        batch_id = await self.processor.queue_frame_batch(frames)

        assert batch_id.camera_id == "test_camera"
        assert batch_id.frame_count == 5
        assert batch_id.batch_id is not None

        # Verify Redis enqueue was called for each frame
        assert self.mock_redis.enqueue.call_count == 5

        await self.processor.stop()

    @pytest.mark.asyncio
    async def test_get_health_status(self):
        """Test health status reporting."""
        await self.processor.start()

        health_status = await self.processor.get_health_status()

        assert health_status["service_status"] == "healthy"
        assert health_status["redis_status"] == "healthy"
        assert "processing_metrics" in health_status
        assert "registered_cameras" in health_status
        assert "active_streams" in health_status
        assert "timestamp" in health_status

        await self.processor.stop()

    def _create_test_frame(
        self, width: int, height: int, quality: str = "high"
    ) -> np.ndarray:
        """Create a synthetic test frame."""
        if quality == "high":
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[::20, :] = 255
            frame[:, ::20] = 255
            frame += np.random.randint(0, 50, frame.shape, dtype=np.uint8)
        else:
            frame = np.random.randint(100, 150, (height, width, 3), dtype=np.uint8)
            import cv2

            frame = cv2.GaussianBlur(frame, (15, 15), 0)

        return frame


class TestStreamingServiceIntegration:
    """Integration tests for the complete streaming service."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.mock_redis = AsyncMock(spec=RedisQueueManager)
        self.mock_redis.connect.return_value = None
        self.mock_redis.create_queue.return_value = True
        self.mock_redis.health_check.return_value = {"status": "healthy"}

        self.streaming_processor = StreamingDataProcessor(
            redis_client=self.mock_redis, max_concurrent_streams=10
        )

        self.grpc_service = StreamingServiceImpl(
            streaming_processor=self.streaming_processor, redis_manager=self.mock_redis
        )

    @pytest.mark.asyncio
    async def test_end_to_end_camera_registration(self):
        """Test end-to-end camera registration through gRPC."""
        await self.streaming_processor.start()

        # Create gRPC request
        from src.its_camera_ai.proto import processed_frame_pb2 as frame_pb

        request = frame_pb.CameraStreamConfig(
            camera_id="integration_test_camera",
            location="Test Location",
            latitude=40.7128,
            longitude=-74.0060,
            width=1280,
            height=720,
            fps=30,
            encoding="h264",
            quality_threshold=0.7,
            processing_enabled=True,
        )

        # Mock context
        mock_context = AsyncMock()

        with patch.object(
            self.streaming_processor.connection_manager,
            "connect_camera",
            return_value=True,
        ):
            response = await self.grpc_service.RegisterStream(request, mock_context)

        assert response.success is True
        assert response.camera_id == "integration_test_camera"
        assert "integration_test_camera" in self.streaming_processor.registered_cameras

        await self.streaming_processor.stop()

    @pytest.mark.asyncio
    async def test_health_check_integration(self):
        """Test health check through gRPC interface."""
        await self.streaming_processor.start()

        from src.its_camera_ai.proto import streaming_service_pb2 as pb

        request = pb.HealthCheckRequest(service_name="streaming_service")
        mock_context = AsyncMock()

        response = await self.grpc_service.HealthCheck(request, mock_context)

        assert response.status == pb.HealthCheckResponse.Status.SERVING
        assert "healthy" in response.message.lower()
        assert response.response_time_ms > 0

        await self.streaming_processor.stop()


class TestStreamingServicePerformance:
    """Performance tests for streaming service."""

    def setup_method(self):
        """Set up performance test fixtures."""
        self.mock_redis = AsyncMock(spec=RedisQueueManager)
        self.mock_redis.connect.return_value = None
        self.mock_redis.create_queue.return_value = True

        # Use higher limits for performance testing
        self.processor = StreamingDataProcessor(
            redis_client=self.mock_redis,
            max_concurrent_streams=150,  # Test beyond 100 requirement
            frame_processing_timeout=0.01,  # 10ms target
        )

        # Track initial memory usage
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_single_frame_processing_latency(self):
        """Test that single frame processing meets <10ms latency requirement."""
        await self.processor.start()

        config = CameraConfig(
            camera_id="perf_test_camera",
            stream_url="test://stream",
            resolution=(1280, 720),
            fps=30,
            protocol=StreamProtocol.HTTP,
        )

        with patch.object(
            self.processor.connection_manager, "connect_camera", return_value=True
        ):
            await self.processor.register_camera(config)

        # Create test frame
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        # Measure processing time for multiple iterations
        latencies = []

        for _ in range(100):
            start_time = time.perf_counter()

            await self.processor._process_single_frame(
                "perf_test_camera", frame, config
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]

        print("\nFrame Processing Latency Statistics:")
        print(f"Average: {avg_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"P99: {p99_latency:.2f}ms")

        # Verify performance requirements
        assert avg_latency < 10.0, (
            f"Average latency {avg_latency:.2f}ms exceeds 10ms requirement"
        )
        assert p95_latency < 15.0, f"P95 latency {p95_latency:.2f}ms too high"

        await self.processor.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_camera_registration(self):
        """Test registration of many cameras concurrently."""
        await self.processor.start()

        # Create many camera configurations
        configs = []
        for i in range(50):  # Test with 50 concurrent cameras
            config = CameraConfig(
                camera_id=f"camera_{i:03d}",
                stream_url=f"test://stream_{i}",
                resolution=(640, 480),
                fps=30,
                protocol=StreamProtocol.HTTP,
            )
            configs.append(config)

        with patch.object(
            self.processor.connection_manager, "connect_camera", return_value=True
        ):
            # Register all cameras concurrently
            start_time = time.perf_counter()

            tasks = [self.processor.register_camera(config) for config in configs]
            registrations = await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            total_time = end_time - start_time

        # Verify all registrations succeeded
        successful_registrations = sum(1 for reg in registrations if reg.success)

        print("\nConcurrent Camera Registration Results:")
        print(f"Total cameras: {len(configs)}")
        print(f"Successful: {successful_registrations}")
        print(f"Failed: {len(configs) - successful_registrations}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per camera: {total_time / len(configs) * 1000:.2f}ms")

        assert successful_registrations == len(configs), (
            "Not all camera registrations succeeded"
        )
        assert len(self.processor.registered_cameras) == len(configs)

        await self.processor.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_frame_throughput(self):
        """Test frame processing throughput."""
        await self.processor.start()

        config = CameraConfig(
            camera_id="throughput_test_camera",
            stream_url="test://stream",
            resolution=(640, 480),
            fps=30,
            protocol=StreamProtocol.HTTP,
        )

        with patch.object(
            self.processor.connection_manager, "connect_camera", return_value=True
        ):
            await self.processor.register_camera(config)

        # Process many frames
        num_frames = 1000
        frames = []

        for _i in range(num_frames):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)

        start_time = time.perf_counter()

        # Process frames concurrently in batches
        batch_size = 50
        processed_count = 0

        for i in range(0, num_frames, batch_size):
            batch = frames[i : i + batch_size]
            tasks = [
                self.processor._process_single_frame(
                    "throughput_test_camera", frame, config
                )
                for frame in batch
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            processed_count += sum(
                1 for r in results if r is not None and not isinstance(r, Exception)
            )

        end_time = time.perf_counter()
        total_time = end_time - start_time

        throughput = processed_count / total_time
        success_rate = processed_count / num_frames

        print("\nFrame Processing Throughput Results:")
        print(f"Total frames: {num_frames}")
        print(f"Processed successfully: {processed_count}")
        print(f"Success rate: {success_rate * 100:.1f}%")
        print(f"Processing time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.1f} fps")

        # Verify performance requirements
        assert success_rate >= 0.999, (
            f"Success rate {success_rate * 100:.1f}% below 99.9% requirement"
        )
        assert throughput >= 100, (
            f"Throughput {throughput:.1f} fps too low"
        )  # Should handle at least 100 fps

        await self.processor.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_under_load(self):
        """Test memory usage stays under 4GB requirement during high load."""
        await self.processor.start()

        # Force garbage collection to get baseline
        gc.collect()
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        configs = []
        for i in range(100):  # 100 concurrent streams
            config = CameraConfig(
                camera_id=f"memory_test_camera_{i:03d}",
                stream_url=f"test://stream_{i}",
                resolution=(1280, 720),
                fps=30,
                protocol=StreamProtocol.HTTP,
            )
            configs.append(config)

        with patch.object(
            self.processor.connection_manager, "connect_camera", return_value=True
        ):
            # Register all cameras
            tasks = [self.processor.register_camera(config) for config in configs]
            await asyncio.gather(*tasks)

            # Process frames for each camera
            frame_batches = []
            for _i in range(100):  # 100 frames per camera
                batch = []
                for config in configs:
                    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                    batch.append((config.camera_id, frame, config))
                frame_batches.append(batch)

            # Process all frames
            for batch in frame_batches:
                tasks = [
                    self.processor._process_single_frame(camera_id, frame, config)
                    for camera_id, frame, config in batch
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            # Check memory usage
            gc.collect()  # Force cleanup
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - baseline_memory

            print("\nMemory Usage Analysis:")
            print(f"Baseline memory: {baseline_memory:.1f} MB")
            print(f"Peak memory: {peak_memory:.1f} MB")
            print(f"Memory increase: {memory_increase:.1f} MB")

            # Memory should stay under 4GB (4096 MB)
            assert peak_memory < 4096, f"Memory usage {peak_memory:.1f} MB exceeds 4GB limit"

            # Memory increase should be reasonable for 100 streams
            assert memory_increase < 2048, f"Memory increase {memory_increase:.1f} MB too high"

        await self.processor.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.benchmark
    async def test_concurrent_stream_load(self, benchmark):
        """Benchmark concurrent stream processing for 100+ streams."""
        await self.processor.start()

        async def process_concurrent_streams():
            # Test with 120 concurrent streams (above requirement)
            configs = []
            for i in range(120):
                config = CameraConfig(
                    camera_id=f"load_test_camera_{i:03d}",
                    stream_url=f"test://stream_{i}",
                    resolution=(1280, 720),
                    fps=30,
                    protocol=StreamProtocol.HTTP,
                )
                configs.append(config)

            with patch.object(
                self.processor.connection_manager, "connect_camera", return_value=True
            ):
                # Register all cameras concurrently
                start_time = time.perf_counter()
                tasks = [self.processor.register_camera(config) for config in configs]
                registrations = await asyncio.gather(*tasks)
                registration_time = time.perf_counter() - start_time

                # Verify all succeeded
                successful = sum(1 for reg in registrations if reg.success)

                # Process frames for all cameras simultaneously
                frames = [
                    np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                    for _ in range(120)
                ]

                frame_start_time = time.perf_counter()
                frame_tasks = [
                    self.processor._process_single_frame(config.camera_id, frames[i], config)
                    for i, config in enumerate(configs)
                ]
                frame_results = await asyncio.gather(*frame_tasks, return_exceptions=True)
                frame_processing_time = time.perf_counter() - frame_start_time

                # Calculate metrics
                successful_frames = sum(1 for r in frame_results if r is not None and not isinstance(r, Exception))
                success_rate = successful_frames / len(frame_results)

                return {
                    "registration_time": registration_time,
                    "frame_processing_time": frame_processing_time,
                    "successful_registrations": successful,
                    "successful_frames": successful_frames,
                    "success_rate": success_rate,
                    "total_streams": 120
                }

        result = await benchmark.pedantic(process_concurrent_streams, rounds=1)

        print("\nConcurrent Load Test Results:")
        print(f"Total streams: {result['total_streams']}")
        print(f"Successful registrations: {result['successful_registrations']}")
        print(f"Registration time: {result['registration_time']:.3f}s")
        print(f"Frame processing time: {result['frame_processing_time']:.3f}s")
        print(f"Success rate: {result['success_rate']*100:.1f}%")

        # Performance requirements validation
        assert result['successful_registrations'] >= 120, "Failed to register all streams"
        assert result['success_rate'] >= 0.999, f"Success rate {result['success_rate']*100:.1f}% below 99.9%"
        assert result['frame_processing_time'] < 2.0, f"Frame processing too slow: {result['frame_processing_time']:.3f}s"

        await self.processor.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_latency_consistency_under_load(self):
        """Test that latency remains consistent under varying load conditions."""
        await self.processor.start()

        config = CameraConfig(
            camera_id="latency_test_camera",
            stream_url="test://stream",
            resolution=(1280, 720),
            fps=30,
            protocol=StreamProtocol.HTTP,
        )

        with patch.object(
            self.processor.connection_manager, "connect_camera", return_value=True
        ):
            await self.processor.register_camera(config)

            # Test latency at different load levels
            load_levels = [1, 10, 50, 100]  # Concurrent frames
            latency_results = {}

            for load_level in load_levels:
                latencies = []

                # Process multiple batches at this load level
                for _ in range(10):  # 10 batches per load level
                    frames = [
                        np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                        for _ in range(load_level)
                    ]

                    start_time = time.perf_counter()
                    tasks = [
                        self.processor._process_single_frame(
                            "latency_test_camera", frame, config
                        )
                        for frame in frames
                    ]
                    await asyncio.gather(*tasks)
                    end_time = time.perf_counter()

                    # Calculate per-frame latency
                    batch_latency = (end_time - start_time) / load_level * 1000  # ms
                    latencies.append(batch_latency)

                # Calculate statistics
                avg_latency = sum(latencies) / len(latencies)
                p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

                latency_results[load_level] = {
                    "avg": avg_latency,
                    "p95": p95_latency,
                    "max": max(latencies),
                    "min": min(latencies)
                }

            print("\nLatency Consistency Results:")
            for load_level, metrics in latency_results.items():
                print(f"Load {load_level:3d}: Avg={metrics['avg']:.2f}ms, "
                      f"P95={metrics['p95']:.2f}ms, Max={metrics['max']:.2f}ms")

            # Validate latency requirements
            for load_level, metrics in latency_results.items():
                assert metrics['avg'] < 10.0, (
                    f"Average latency {metrics['avg']:.2f}ms exceeds 10ms at load {load_level}"
                )
                assert metrics['p95'] < 15.0, (
                    f"P95 latency {metrics['p95']:.2f}ms exceeds 15ms at load {load_level}"
                )

            # Latency should not degrade significantly with load
            latency_degradation = (
                latency_results[100]['avg'] - latency_results[1]['avg']
            ) / latency_results[1]['avg']

            assert latency_degradation < 2.0, (
                f"Latency degradation {latency_degradation*100:.1f}% too high under load"
            )

        await self.processor.stop()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_error_recovery_performance(self):
        """Test system performance during error conditions and recovery."""
        await self.processor.start()

        config = CameraConfig(
            camera_id="error_recovery_camera",
            stream_url="test://stream",
            resolution=(1280, 720),
            fps=30,
            protocol=StreamProtocol.HTTP,
        )

        with patch.object(
            self.processor.connection_manager, "connect_camera", return_value=True
        ):
            await self.processor.register_camera(config)

            # Simulate error conditions with frame processing
            error_rates = [0.0, 0.1, 0.2, 0.5]  # 0%, 10%, 20%, 50% error rates
            performance_metrics = {}

            for error_rate in error_rates:
                successful_frames = 0
                total_frames = 100
                latencies = []

                for _i in range(total_frames):
                    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

                    # Inject errors based on error rate
                    if np.random.random() < error_rate:
                        # Simulate corrupted frame (wrong dimensions)
                        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

                    start_time = time.perf_counter()
                    with contextlib.suppress(Exception):
                        result = await self.processor._process_single_frame(
                            "error_recovery_camera", frame, config
                        )
                        if result is not None:
                            successful_frames += 1

                    end_time = time.perf_counter()
                    latencies.append((end_time - start_time) * 1000)  # ms

                success_rate = successful_frames / total_frames
                avg_latency = sum(latencies) / len(latencies)

                performance_metrics[error_rate] = {
                    "success_rate": success_rate,
                    "avg_latency": avg_latency,
                    "successful_frames": successful_frames
                }

            print("\nError Recovery Performance:")
            for error_rate, metrics in performance_metrics.items():
                print(f"Error rate {error_rate*100:3.0f}%: Success={metrics['success_rate']*100:.1f}%, "
                      f"Latency={metrics['avg_latency']:.2f}ms")

            # System should maintain performance even with errors
            for error_rate, metrics in performance_metrics.items():
                expected_success_rate = 1.0 - error_rate
                assert metrics['success_rate'] >= expected_success_rate * 0.95, (
                    f"Success rate too low at {error_rate*100}% error rate"
                )
                assert metrics['avg_latency'] < 20.0, (
                    f"Latency too high during error conditions: {metrics['avg_latency']:.2f}ms"
                )

        await self.processor.stop()


# Pytest configuration for performance tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark test")
    config.addinivalue_line("markers", "load: mark test as a load test")
    config.addinivalue_line("markers", "memory: mark test as a memory profiling test")


if __name__ == "__main__":
    # Run tests with performance markers
    import sys

    sys.exit(pytest.main(["-v", "-m", "not performance", __file__]))
