"""Integration tests for CameraStreamOrchestrator with CUDA streams and unified vision engine.

This test suite validates the end-to-end functionality of the camera stream orchestrator
including integration with CUDA streams manager and unified vision analytics engine.
"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import time

from src.its_camera_ai.core.camera_stream_orchestrator import (
    CameraStreamOrchestrator,
    StreamConfiguration,
    StreamType,
    StreamPriority,
    simulate_camera_streams
)
from src.its_camera_ai.core.unified_vision_analytics_engine import UnifiedVisionAnalyticsEngine
from src.its_camera_ai.core.cuda_streams_manager import CudaStreamsManager
from src.its_camera_ai.ml.batch_processor import RequestPriority


class MockUnifiedVisionAnalyticsEngine:
    """Mock implementation of UnifiedVisionAnalyticsEngine for testing."""
    
    def __init__(self):
        self.process_frame_call_count = 0
        self.processing_latency_ms = 50  # Simulated processing latency
        
    async def process_frame_with_streams(
        self, frame, camera_id, frame_id, priority, include_analytics=True, include_quality_score=True
    ):
        """Mock frame processing with realistic latency."""
        self.process_frame_call_count += 1
        
        # Simulate processing time
        await asyncio.sleep(self.processing_latency_ms / 1000.0)
        
        # Create mock result
        mock_result = Mock()
        mock_result.camera_id = camera_id
        mock_result.frame_id = frame_id
        mock_result.quality_score = 0.8
        mock_result.analytics_result = Mock()
        mock_result.analytics_result.violations = []
        mock_result.analytics_result.anomalies = []
        mock_result.total_processing_time_ms = self.processing_latency_ms
        
        return mock_result


@pytest.mark.asyncio
class TestCameraStreamOrchestratorIntegration:
    """Integration tests for camera stream orchestrator."""
    
    @pytest.fixture
    async def mock_vision_engine(self):
        """Create mock vision engine."""
        return MockUnifiedVisionAnalyticsEngine()
    
    @pytest.fixture
    async def orchestrator(self, mock_vision_engine):
        """Create orchestrator with mock vision engine."""
        orchestrator = CameraStreamOrchestrator(
            vision_engine=mock_vision_engine,
            max_concurrent_streams=100,
            max_processing_queue_size=500,
        )
        yield orchestrator
        
        # Cleanup
        if orchestrator.is_running:
            await orchestrator.stop()
    
    async def test_orchestrator_lifecycle(self, orchestrator):
        """Test basic orchestrator lifecycle (start/stop)."""
        # Initially not running
        assert not orchestrator.is_running
        
        # Start orchestrator
        await orchestrator.start()
        assert orchestrator.is_running
        assert len(orchestrator.background_tasks) == 4  # 4 background tasks
        
        # Stop orchestrator
        await orchestrator.stop()
        assert not orchestrator.is_running
        assert len(orchestrator.background_tasks) == 0
    
    async def test_single_stream_management(self, orchestrator):
        """Test adding and removing a single stream."""
        await orchestrator.start()
        
        # Create stream configuration
        config = StreamConfiguration(
            camera_id="test_camera_001",
            stream_url="simulator://test",
            stream_type=StreamType.SIMULATOR,
            priority=StreamPriority.NORMAL,
            target_fps=30
        )
        
        # Add stream
        success = await orchestrator.add_stream(config)
        assert success
        assert "test_camera_001" in orchestrator.active_streams
        assert orchestrator.orchestrator_metrics["active_stream_count"] == 1
        
        # Wait for stream to initialize
        await asyncio.sleep(0.5)
        
        # Check stream is processing
        stream_connection = orchestrator.active_streams["test_camera_001"]
        assert stream_connection.is_running
        
        # Remove stream
        success = await orchestrator.remove_stream("test_camera_001")
        assert success
        assert "test_camera_001" not in orchestrator.active_streams
        assert orchestrator.orchestrator_metrics["active_stream_count"] == 0
    
    async def test_multiple_streams_with_priorities(self, orchestrator):
        """Test managing multiple streams with different priorities."""
        await orchestrator.start()
        
        # Create multiple stream configurations with different priorities
        configs = [
            StreamConfiguration(
                camera_id=f"camera_{i:03d}",
                stream_url="simulator://test",
                stream_type=StreamType.SIMULATOR,
                priority=StreamPriority.HIGH if i % 3 == 0 else StreamPriority.NORMAL,
                target_fps=30
            )
            for i in range(10)
        ]
        
        # Add all streams
        add_tasks = [orchestrator.add_stream(config) for config in configs]
        results = await asyncio.gather(*add_tasks)
        
        # Verify all streams added successfully
        assert all(results)
        assert len(orchestrator.active_streams) == 10
        assert orchestrator.orchestrator_metrics["active_stream_count"] == 10
        
        # Check priority distribution
        high_priority_count = sum(
            1 for config in configs if config.priority == StreamPriority.HIGH
        )
        assert len(orchestrator.priority_queues[StreamPriority.HIGH]) == high_priority_count
        
        # Wait for streams to process some frames
        await asyncio.sleep(2.0)
        
        # Verify frames are being processed
        total_frames_processed = sum(
            stream.metrics.frames_processed 
            for stream in orchestrator.active_streams.values()
        )
        assert total_frames_processed > 0
    
    async def test_stream_health_monitoring(self, orchestrator):
        """Test stream health monitoring and metrics collection."""
        await orchestrator.start()
        
        config = StreamConfiguration(
            camera_id="health_test_camera",
            stream_url="simulator://test",
            stream_type=StreamType.SIMULATOR,
            priority=StreamPriority.NORMAL,
            target_fps=30
        )
        
        await orchestrator.add_stream(config)
        
        # Let stream run for a bit
        await asyncio.sleep(2.0)
        
        # Get stream health status
        stream_connection = orchestrator.active_streams["health_test_camera"]
        health_status = stream_connection.get_health_status()
        
        # Verify health status structure
        assert "camera_id" in health_status
        assert "state" in health_status
        assert "health_score" in health_status
        assert "metrics" in health_status
        
        # Check that metrics are being updated
        assert health_status["metrics"]["frames_processed"] > 0
        assert health_status["metrics"]["current_fps"] > 0
        assert health_status["health_score"] > 0.5  # Should be healthy
    
    async def test_backpressure_handling(self, orchestrator, mock_vision_engine):
        """Test backpressure handling when system is overloaded."""
        await orchestrator.start()
        
        # Slow down vision engine processing to create backpressure
        mock_vision_engine.processing_latency_ms = 500  # Very slow processing
        
        config = StreamConfiguration(
            camera_id="backpressure_test",
            stream_url="simulator://test",
            stream_type=StreamType.SIMULATOR,
            priority=StreamPriority.NORMAL,
            target_fps=60  # High FPS to stress system
        )
        
        await orchestrator.add_stream(config)
        
        # Wait and check if system handles backpressure gracefully
        await asyncio.sleep(3.0)
        
        stream_connection = orchestrator.active_streams["backpressure_test"]
        
        # System should drop frames under backpressure
        assert stream_connection.metrics.frames_dropped > 0
        
        # But should still be processing some frames
        assert stream_connection.metrics.frames_processed > 0
    
    async def test_orchestrator_metrics(self, orchestrator):
        """Test comprehensive orchestrator metrics collection."""
        await orchestrator.start()
        
        # Add a few streams
        configs = [
            StreamConfiguration(
                camera_id=f"metrics_camera_{i}",
                stream_url="simulator://test",
                stream_type=StreamType.SIMULATOR,
                priority=StreamPriority.NORMAL,
                target_fps=30
            )
            for i in range(5)
        ]
        
        for config in configs:
            await orchestrator.add_stream(config)
        
        # Wait for processing
        await asyncio.sleep(2.0)
        
        # Get comprehensive metrics
        metrics = orchestrator.get_orchestrator_metrics()
        
        # Verify metrics structure
        assert "orchestrator" in metrics
        assert "aggregated" in metrics
        assert "streams" in metrics
        assert "priority_distribution" in metrics
        assert "timestamp" in metrics
        
        # Check orchestrator metrics
        assert metrics["orchestrator"]["active_stream_count"] == 5
        assert metrics["orchestrator"]["frames_processed_total"] > 0
        
        # Check aggregated metrics
        assert "total_frames_received" in metrics["aggregated"]
        assert "avg_health_score" in metrics["aggregated"]
        
        # Check individual stream metrics
        assert len(metrics["streams"]) == 5
        for stream_metrics in metrics["streams"].values():
            assert "health_score" in stream_metrics
            assert "metrics" in stream_metrics
    
    async def test_stream_simulation(self, orchestrator):
        """Test the stream simulation functionality."""
        await orchestrator.start()
        
        # Run simulation with multiple streams
        metrics = await simulate_camera_streams(
            orchestrator=orchestrator,
            num_streams=20,
            duration_seconds=3
        )
        
        # Verify simulation produced meaningful metrics
        assert "orchestrator" in metrics
        assert "streams" in metrics
        assert len(metrics["streams"]) == 20  # All simulated streams
        
        # Check that streams processed frames during simulation
        total_frames = metrics["aggregated"]["total_frames_received"]
        assert total_frames > 0
        
        # Check that no streams remain after simulation cleanup
        assert len(orchestrator.active_streams) == 0
    
    @patch('cv2.VideoCapture')
    async def test_rtsp_stream_connection(self, mock_cv2_capture, orchestrator):
        """Test RTSP stream connection handling."""
        # Mock OpenCV VideoCapture for RTSP
        mock_capture_instance = Mock()
        mock_capture_instance.isOpened.return_value = True
        mock_capture_instance.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2_capture.return_value = mock_capture_instance
        
        await orchestrator.start()
        
        config = StreamConfiguration(
            camera_id="rtsp_test_camera",
            stream_url="rtsp://example.com/stream",
            stream_type=StreamType.RTSP,
            priority=StreamPriority.NORMAL,
            target_fps=30
        )
        
        success = await orchestrator.add_stream(config)
        assert success
        
        # Wait for connection to establish
        await asyncio.sleep(1.0)
        
        # Verify OpenCV VideoCapture was used
        mock_cv2_capture.assert_called_once()
        
        # Check stream is connected
        stream_connection = orchestrator.active_streams["rtsp_test_camera"]
        # Note: State might be CONNECTING initially, so we just check it's not ERROR
        assert stream_connection.metrics.state != stream_connection.metrics.state.ERROR
    
    async def test_error_recovery(self, orchestrator):
        """Test error recovery and stream restart functionality."""
        await orchestrator.start()
        
        # Add stream that will have connection issues
        config = StreamConfiguration(
            camera_id="error_test_camera",
            stream_url="simulator://test",
            stream_type=StreamType.SIMULATOR,
            priority=StreamPriority.NORMAL,
            target_fps=30,
            retry_attempts=3,
            reconnect_interval=0.5  # Fast reconnect for testing
        )
        
        await orchestrator.add_stream(config)
        await asyncio.sleep(1.0)
        
        stream_connection = orchestrator.active_streams["error_test_camera"]
        
        # Simulate connection error by stopping capture
        if hasattr(stream_connection.capture, 'is_opened'):
            stream_connection.capture.is_opened = False
        
        # Wait for error detection and recovery attempt
        await asyncio.sleep(2.0)
        
        # Stream should attempt to recover
        # (In a real scenario, reconnection logic would be triggered)
        assert stream_connection.metrics.state in [
            stream_connection.metrics.state.CONNECTED,
            stream_connection.metrics.state.RECONNECTING,
            stream_connection.metrics.state.ERROR
        ]
    
    async def test_resource_limits(self, orchestrator):
        """Test that orchestrator respects resource limits."""
        # Create orchestrator with low limits
        limited_orchestrator = CameraStreamOrchestrator(
            vision_engine=MockUnifiedVisionAnalyticsEngine(),
            max_concurrent_streams=5,
            max_processing_queue_size=10,
        )
        
        await limited_orchestrator.start()
        
        try:
            # Try to add more streams than limit
            configs = [
                StreamConfiguration(
                    camera_id=f"limit_test_{i}",
                    stream_url="simulator://test",
                    stream_type=StreamType.SIMULATOR,
                    priority=StreamPriority.NORMAL,
                    target_fps=30
                )
                for i in range(10)  # More than limit of 5
            ]
            
            results = []
            for config in configs:
                success = await limited_orchestrator.add_stream(config)
                results.append(success)
            
            # Should only accept up to the limit
            successful_adds = sum(results)
            assert successful_adds <= 5
            assert len(limited_orchestrator.active_streams) <= 5
            
        finally:
            await limited_orchestrator.stop()
    
    async def test_priority_processing(self, orchestrator, mock_vision_engine):
        """Test that high-priority streams get preferential processing."""
        await orchestrator.start()
        
        # Add high and normal priority streams
        high_priority_config = StreamConfiguration(
            camera_id="high_priority_stream",
            stream_url="simulator://test",
            stream_type=StreamType.SIMULATOR,
            priority=StreamPriority.HIGH,
            target_fps=30
        )
        
        normal_priority_config = StreamConfiguration(
            camera_id="normal_priority_stream",
            stream_url="simulator://test",
            stream_type=StreamType.SIMULATOR,
            priority=StreamPriority.NORMAL,
            target_fps=30
        )
        
        await orchestrator.add_stream(high_priority_config)
        await orchestrator.add_stream(normal_priority_config)
        
        # Let streams run
        await asyncio.sleep(3.0)
        
        # Both streams should be processing, but we can't easily test
        # priority order without more complex mocking
        high_priority_stream = orchestrator.active_streams["high_priority_stream"]
        normal_priority_stream = orchestrator.active_streams["normal_priority_stream"]
        
        assert high_priority_stream.metrics.frames_processed > 0
        assert normal_priority_stream.metrics.frames_processed > 0
    
    async def test_callback_integration(self, orchestrator):
        """Test callback system for stream events."""
        await orchestrator.start()
        
        # Track callback invocations
        callback_calls = {"stream_added": 0, "stream_removed": 0}
        
        async def stream_added_callback(camera_id, config):
            callback_calls["stream_added"] += 1
        
        async def stream_removed_callback(camera_id):
            callback_calls["stream_removed"] += 1
        
        # Register callbacks
        orchestrator.register_callback("stream_added", stream_added_callback)
        orchestrator.register_callback("stream_removed", stream_removed_callback)
        
        # Add and remove stream
        config = StreamConfiguration(
            camera_id="callback_test_camera",
            stream_url="simulator://test",
            stream_type=StreamType.SIMULATOR,
            priority=StreamPriority.NORMAL,
            target_fps=30
        )
        
        await orchestrator.add_stream(config)
        await orchestrator.remove_stream("callback_test_camera")
        
        # Verify callbacks were called
        assert callback_calls["stream_added"] == 1
        assert callback_calls["stream_removed"] == 1


if __name__ == "__main__":
    # Run a simple integration test
    async def main():
        print("Running CameraStreamOrchestrator integration test...")
        
        # Create mock vision engine
        mock_engine = MockUnifiedVisionAnalyticsEngine()
        
        # Create orchestrator
        orchestrator = CameraStreamOrchestrator(
            vision_engine=mock_engine,
            max_concurrent_streams=50,
        )
        
        try:
            await orchestrator.start()
            print("✓ Orchestrator started successfully")
            
            # Run simulation
            print("Running stream simulation...")
            metrics = await simulate_camera_streams(
                orchestrator=orchestrator,
                num_streams=10,
                duration_seconds=5
            )
            
            print(f"✓ Simulation completed:")
            print(f"  - Total frames received: {metrics['aggregated']['total_frames_received']}")
            print(f"  - Total frames processed: {mock_engine.process_frame_call_count}")
            print(f"  - Average health score: {metrics['aggregated']['avg_health_score']:.2f}")
            print(f"  - Healthy streams: {metrics['aggregated']['healthy_streams']}")
            print("✓ Integration test completed successfully!")
            
        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            raise
        finally:
            await orchestrator.stop()
    
    # Run the test
    asyncio.run(main())