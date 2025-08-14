#!/usr/bin/env python3
"""Standalone integration test for CameraStreamOrchestrator.

This test validates the basic functionality of the camera stream orchestrator
without external dependencies like pytest.
"""

import asyncio
import logging
import os
import sys
from unittest.mock import Mock

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from its_camera_ai.core.camera_stream_orchestrator import (
    CameraStreamOrchestrator,
    StreamConfiguration,
    StreamPriority,
    StreamType,
    simulate_camera_streams,
)

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MockUnifiedVisionAnalyticsEngine:
    """Mock implementation for testing."""

    def __init__(self):
        self.process_frame_call_count = 0

    async def process_frame_with_streams(self, **kwargs):
        """Mock frame processing."""
        self.process_frame_call_count += 1

        # Simulate processing latency
        await asyncio.sleep(0.01)

        # Create mock result
        result = Mock()
        result.camera_id = kwargs.get('camera_id', 'unknown')
        result.frame_id = kwargs.get('frame_id', 'unknown')
        result.quality_score = 0.85
        result.total_processing_time_ms = 10.0

        # Mock analytics result
        result.analytics_result = Mock()
        result.analytics_result.violations = []
        result.analytics_result.anomalies = []

        return result


async def test_basic_orchestrator_functionality():
    """Test basic orchestrator operations."""
    logger.info("=== Testing Basic Orchestrator Functionality ===")

    # Create mock vision engine
    mock_engine = MockUnifiedVisionAnalyticsEngine()

    # Create orchestrator
    orchestrator = CameraStreamOrchestrator(
        vision_engine=mock_engine,
        max_concurrent_streams=100,
        max_processing_queue_size=500,
    )

    success_count = 0
    total_tests = 0

    try:
        # Test 1: Start orchestrator
        total_tests += 1
        logger.info("Test 1: Starting orchestrator...")
        await orchestrator.start()
        assert orchestrator.is_running, "Orchestrator should be running"
        logger.info("✓ Orchestrator started successfully")
        success_count += 1

        # Test 2: Add single stream
        total_tests += 1
        logger.info("Test 2: Adding single stream...")
        config = StreamConfiguration(
            camera_id="test_camera_001",
            stream_url="simulator://test",
            stream_type=StreamType.SIMULATOR,
            priority=StreamPriority.NORMAL,
            target_fps=30
        )

        success = await orchestrator.add_stream(config)
        assert success, "Stream should be added successfully"
        assert "test_camera_001" in orchestrator.active_streams, "Stream should be in active streams"
        logger.info("✓ Single stream added successfully")
        success_count += 1

        # Test 3: Wait for stream processing
        total_tests += 1
        logger.info("Test 3: Waiting for stream processing...")
        await asyncio.sleep(2.0)  # Let stream process some frames

        stream_connection = orchestrator.active_streams["test_camera_001"]
        assert stream_connection.is_running, "Stream connection should be running"
        assert stream_connection.metrics.frames_received > 0, "Should have received frames"
        logger.info(f"✓ Stream processed {stream_connection.metrics.frames_received} frames")
        success_count += 1

        # Test 4: Check stream health
        total_tests += 1
        logger.info("Test 4: Checking stream health...")
        health_status = stream_connection.get_health_status()
        assert health_status["health_score"] > 0, "Health score should be positive"
        assert "metrics" in health_status, "Health status should include metrics"
        logger.info(f"✓ Stream health score: {health_status['health_score']:.2f}")
        success_count += 1

        # Test 5: Add multiple streams
        total_tests += 1
        logger.info("Test 5: Adding multiple streams...")
        configs = []
        for i in range(5):
            config = StreamConfiguration(
                camera_id=f"multi_camera_{i:03d}",
                stream_url="simulator://test",
                stream_type=StreamType.SIMULATOR,
                priority=StreamPriority.HIGH if i % 2 == 0 else StreamPriority.NORMAL,
                target_fps=30
            )
            configs.append(config)

        for config in configs:
            success = await orchestrator.add_stream(config)
            assert success, f"Stream {config.camera_id} should be added"

        assert len(orchestrator.active_streams) == 6, "Should have 6 active streams"
        logger.info("✓ Multiple streams added successfully")
        success_count += 1

        # Test 6: Wait for processing and check metrics
        total_tests += 1
        logger.info("Test 6: Checking orchestrator metrics...")
        await asyncio.sleep(3.0)  # Let all streams process

        metrics = orchestrator.get_orchestrator_metrics()
        assert "orchestrator" in metrics, "Metrics should have orchestrator section"
        assert "streams" in metrics, "Metrics should have streams section"
        assert metrics["orchestrator"]["active_stream_count"] == 6, "Should report 6 active streams"

        total_frames = metrics["aggregated"]["total_frames_received"]
        assert total_frames > 0, "Should have processed some frames"

        logger.info(f"✓ Orchestrator metrics: {total_frames} frames received")
        success_count += 1

        # Test 7: Remove streams
        total_tests += 1
        logger.info("Test 7: Removing streams...")
        for config in configs[:3]:  # Remove first 3 streams
            success = await orchestrator.remove_stream(config.camera_id)
            assert success, f"Stream {config.camera_id} should be removed"

        assert len(orchestrator.active_streams) == 3, "Should have 3 active streams remaining"
        logger.info("✓ Streams removed successfully")
        success_count += 1

        # Test 8: Stream simulation
        total_tests += 1
        logger.info("Test 8: Running stream simulation...")

        # First remove remaining streams
        remaining_cameras = list(orchestrator.active_streams.keys())
        for camera_id in remaining_cameras:
            await orchestrator.remove_stream(camera_id)

        # Run simulation
        simulation_metrics = await simulate_camera_streams(
            orchestrator=orchestrator,
            num_streams=10,
            duration_seconds=3
        )

        assert "streams" in simulation_metrics, "Simulation should return metrics"
        sim_total_frames = simulation_metrics["aggregated"]["total_frames_received"]
        assert sim_total_frames > 0, "Simulation should process frames"

        logger.info(f"✓ Simulation completed: {sim_total_frames} frames processed")
        success_count += 1

        # Test 9: Stop orchestrator
        total_tests += 1
        logger.info("Test 9: Stopping orchestrator...")
        await orchestrator.stop()
        assert not orchestrator.is_running, "Orchestrator should be stopped"
        assert len(orchestrator.active_streams) == 0, "No streams should remain active"
        logger.info("✓ Orchestrator stopped successfully")
        success_count += 1

    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup
        if orchestrator.is_running:
            await orchestrator.stop()

    logger.info("\n=== Test Summary ===")
    logger.info(f"Passed: {success_count}/{total_tests}")
    logger.info(f"Success Rate: {success_count/total_tests*100:.1f}%")

    return success_count == total_tests


async def test_performance_characteristics():
    """Test performance characteristics of the orchestrator."""
    logger.info("\n=== Testing Performance Characteristics ===")

    mock_engine = MockUnifiedVisionAnalyticsEngine()
    orchestrator = CameraStreamOrchestrator(
        vision_engine=mock_engine,
        max_concurrent_streams=50,
    )

    try:
        await orchestrator.start()
        logger.info("✓ Orchestrator started for performance test")

        # Test with 20 concurrent streams
        logger.info("Testing with 20 concurrent streams...")
        start_time = asyncio.get_event_loop().time()

        configs = [
            StreamConfiguration(
                camera_id=f"perf_camera_{i:03d}",
                stream_url="simulator://test",
                stream_type=StreamType.SIMULATOR,
                priority=StreamPriority.NORMAL,
                target_fps=30
            )
            for i in range(20)
        ]

        # Add all streams
        for config in configs:
            await orchestrator.add_stream(config)

        setup_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"✓ Setup time for 20 streams: {setup_time:.2f}s")

        # Run for 5 seconds
        await asyncio.sleep(5.0)

        # Get final metrics
        metrics = orchestrator.get_orchestrator_metrics()
        total_frames = metrics["aggregated"]["total_frames_received"]
        avg_health = metrics["aggregated"]["avg_health_score"]

        logger.info("✓ Performance results:")
        logger.info(f"  - Total frames processed: {total_frames}")
        logger.info(f"  - Average health score: {avg_health:.2f}")
        logger.info(f"  - Frames per second (system): {total_frames/5.0:.1f}")
        logger.info(f"  - Vision engine calls: {mock_engine.process_frame_call_count}")

        assert total_frames > 100, "Should process significant number of frames"
        assert avg_health > 0.5, "Average health should be good"

        logger.info("✓ Performance test completed successfully")

    except Exception as e:
        logger.error(f"✗ Performance test failed: {e}")
        return False
    finally:
        await orchestrator.stop()

    return True


async def main():
    """Run all tests."""
    logger.info("Starting CameraStreamOrchestrator Integration Tests")
    logger.info("=" * 60)

    try:
        # Test basic functionality
        basic_test_passed = await test_basic_orchestrator_functionality()

        if basic_test_passed:
            logger.info("\n✓ Basic functionality tests PASSED")

            # Test performance characteristics
            perf_test_passed = await test_performance_characteristics()

            if perf_test_passed:
                logger.info("\n✓ Performance tests PASSED")
                logger.info("\n🎉 ALL TESTS PASSED! CameraStreamOrchestrator is working correctly.")
                return True
            else:
                logger.error("\n✗ Performance tests FAILED")
        else:
            logger.error("\n✗ Basic functionality tests FAILED")

    except Exception as e:
        logger.error(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

    logger.error("\n❌ SOME TESTS FAILED")
    return False


if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
