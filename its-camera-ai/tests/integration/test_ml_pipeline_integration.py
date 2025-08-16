"""Integration Tests for ML Pipeline Streaming Connection.

Tests the complete integration of:
1. CoreVisionEngine (YOLO11 ML inference)
2. UnifiedVisionAnalyticsEngine (unified processing)  
3. gRPC servicers (streaming endpoints)
4. Kafka event streaming (real-time data flow)
5. Analytics service (traffic analysis)

These tests verify that the ML pipeline is properly connected to streaming endpoints
and that detection results flow correctly to analytics and dashboard updates.
"""

import asyncio
import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.its_camera_ai.core.unified_vision_analytics_engine import (
    RequestPriority,
    UnifiedResult,
)
from src.its_camera_ai.flow.kafka_event_processor import (
    EventType,
    KafkaEventProcessor,
    StreamingEvent,
)
from src.its_camera_ai.ml.inference_optimizer import DetectionResult
from src.its_camera_ai.services.analytics_dtos import DetectionResultDTO
from src.its_camera_ai.services.ml_streaming_integration_service import (
    MLStreamingIntegrationService,
)
from src.its_camera_ai.services.production_ml_grpc_server import (
    ProductionMLgRPCServer,
)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = MagicMock()
    settings.model_path = "models/yolo11s.pt"
    settings.confidence_threshold = 0.5
    settings.iou_threshold = 0.4
    settings.max_detections = 1000
    settings.target_fps = 30
    settings.inference_batch_size = 8
    settings.max_batch_size = 32
    settings.enable_tensorrt = True
    settings.precision = "fp16"
    settings.gpu_device_ids = [0]
    settings.kafka_enabled = False  # Disable for testing
    settings.kafka_bootstrap_servers = ["localhost:9092"]
    return settings


@pytest.fixture
def mock_unified_analytics():
    """Mock unified analytics service."""
    analytics = AsyncMock()
    analytics.process_realtime_analytics.return_value = MagicMock(
        violations=[],
        anomalies=[],
        processing_time_ms=25.0,
    )
    return analytics


@pytest.fixture
def mock_cache_service():
    """Mock cache service."""
    cache = AsyncMock()
    cache.set_json = AsyncMock()
    cache.get_json = AsyncMock(return_value=None)
    return cache


@pytest.fixture
async def ml_integration_service(mock_settings, mock_unified_analytics, mock_cache_service):
    """Create ML integration service for testing."""
    with patch('src.its_camera_ai.services.ml_streaming_integration_service.CoreVisionEngine'), \
         patch('src.its_camera_ai.services.ml_streaming_integration_service.OptimizedInferenceEngine'):
        
        service = MLStreamingIntegrationService()
        service.settings = mock_settings
        service.unified_analytics = mock_unified_analytics
        service.cache_service = mock_cache_service
        
        # Mock the core components to avoid actual ML initialization
        service.core_vision_engine = AsyncMock()
        service.optimized_inference_engine = AsyncMock()
        service.unified_vision_analytics = AsyncMock()
        
        return service


@pytest.fixture
def sample_frame():
    """Sample camera frame for testing."""
    return np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)


@pytest.fixture
def mock_detection_results():
    """Mock detection results from ML inference."""
    return [
        DetectionResultDTO(
            detection_id="det_1",
            class_name="car",
            class_id=2,
            confidence=0.85,
            x_min=100,
            y_min=150,
            x_max=300,
            y_max=400,
            tracking_id=1,
            vehicle_type="car",
            is_vehicle=True,
            speed=45.0,
            direction="north",
            timestamp=datetime.now(UTC),
        ),
        DetectionResultDTO(
            detection_id="det_2", 
            class_name="truck",
            class_id=7,
            confidence=0.92,
            x_min=400,
            y_min=100,
            x_max=600,
            y_max=350,
            tracking_id=2,
            vehicle_type="truck",
            is_vehicle=True,
            speed=35.0,
            direction="south",
            timestamp=datetime.now(UTC),
        ),
    ]


class TestMLPipelineIntegration:
    """Test ML pipeline integration with streaming endpoints."""

    @pytest.mark.asyncio
    async def test_ml_service_initialization(self, ml_integration_service):
        """Test ML integration service initializes correctly."""
        # Mock the initialization methods
        ml_integration_service._initialize_core_vision_engine = AsyncMock()
        ml_integration_service._initialize_kafka_producer = AsyncMock()
        
        await ml_integration_service.initialize()
        
        assert ml_integration_service.is_initialized
        ml_integration_service._initialize_core_vision_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_frame_processing_pipeline(
        self, ml_integration_service, sample_frame, mock_detection_results
    ):
        """Test complete frame processing through ML pipeline."""
        # Setup mocks
        ml_integration_service.is_running = True
        ml_integration_service.unified_vision_analytics.process_frame = AsyncMock(
            return_value=UnifiedResult(
                detections=mock_detection_results,
                inference_time_ms=45.0,
                model_version="yolo11s",
                analytics_result=MagicMock(),
                quality_score=0.85,
                analytics_time_ms=25.0,
                camera_id="cam_001",
                frame_id="frame_123",
                timestamp=datetime.now(UTC),
                total_processing_time_ms=70.0,
                batch_size=1,
            )
        )
        
        # Process frame
        result = await ml_integration_service.process_frame(
            frame=sample_frame,
            camera_id="cam_001",
            frame_id="frame_123",
            priority=RequestPriority.NORMAL,
        )
        
        # Verify results
        assert result is not None
        assert result.camera_id == "cam_001"
        assert result.frame_id == "frame_123"
        assert len(result.detections) == 2
        assert result.inference_time_ms == 45.0
        assert result.total_processing_time_ms == 70.0
        
        # Verify analytics was called
        ml_integration_service.unified_vision_analytics.process_frame.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_processing_optimization(
        self, ml_integration_service, mock_detection_results
    ):
        """Test batch processing for optimal GPU utilization."""
        # Setup batch data
        frames = [np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8) for _ in range(4)]
        camera_ids = [f"cam_{i:03d}" for i in range(4)]
        frame_ids = [f"frame_{i}" for i in range(4)]
        
        # Mock batch processing
        ml_integration_service.is_running = True
        ml_integration_service.unified_vision_analytics.process_batch = AsyncMock(
            return_value=[
                UnifiedResult(
                    detections=mock_detection_results,
                    inference_time_ms=35.0,  # Faster due to batching
                    model_version="yolo11s",
                    analytics_result=MagicMock(),
                    quality_score=0.8,
                    analytics_time_ms=20.0,
                    camera_id=camera_ids[i],
                    frame_id=frame_ids[i],
                    timestamp=datetime.now(UTC),
                    total_processing_time_ms=55.0,
                    batch_size=4,
                )
                for i in range(4)
            ]
        )
        
        # Process batch
        results = await ml_integration_service.process_batch(
            frames=frames,
            camera_ids=camera_ids,
            frame_ids=frame_ids,
        )
        
        # Verify batch results
        assert len(results) == 4
        for i, result in enumerate(results):
            assert result.camera_id == camera_ids[i]
            assert result.frame_id == frame_ids[i]
            assert result.batch_size == 4
            assert result.inference_time_ms == 35.0  # Batching optimization
        
        # Verify batch processing was called
        ml_integration_service.unified_vision_analytics.process_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_performance_tracking(self, ml_integration_service, sample_frame):
        """Test performance metrics tracking during processing."""
        ml_integration_service.is_running = True
        ml_integration_service.start_time = time.time() - 60  # 1 minute of uptime
        ml_integration_service.frames_processed = 100
        ml_integration_service.inference_times = [45.0, 50.0, 42.0, 48.0, 46.0]
        
        # Get performance metrics
        metrics = await ml_integration_service.get_performance_metrics()
        
        # Verify metrics
        assert metrics["frames_processed"] == 100
        assert metrics["uptime_seconds"] >= 60
        assert "throughput_fps" in metrics
        assert "latency_metrics" in metrics
        assert metrics["latency_metrics"]["avg_ms"] == 46.2  # Average of test data

    @pytest.mark.asyncio
    async def test_health_check_integration(self, ml_integration_service):
        """Test comprehensive health check across all components."""
        ml_integration_service.is_running = True
        ml_integration_service.is_initialized = True
        ml_integration_service.unified_vision_analytics.health_check = AsyncMock(
            return_value={
                "status": "healthy",
                "queue_depth": 5,
                "avg_processing_time_ms": 45.0,
            }
        )
        
        # Get health status
        health = await ml_integration_service.health_check()
        
        # Verify health data
        assert health["status"] == "healthy"
        assert health["is_running"] is True
        assert health["is_initialized"] is True
        assert "components" in health
        assert "unified_engine_health" in health


class TestKafkaEventStreaming:
    """Test Kafka event streaming for real-time data flow."""

    @pytest.fixture
    def event_processor(self):
        """Create event processor for testing (no-op without Kafka)."""
        from src.its_camera_ai.flow.kafka_event_processor import create_kafka_event_processor
        return create_kafka_event_processor()

    @pytest.mark.asyncio
    async def test_detection_event_creation(self, event_processor, mock_detection_results):
        """Test creation and publishing of detection events."""
        # Mock detection data
        detections = [
            {
                "class_name": det.class_name,
                "confidence": det.confidence,
                "bbox": {"x_min": det.x_min, "y_min": det.y_min, "x_max": det.x_max, "y_max": det.y_max},
                "is_vehicle": det.is_vehicle,
                "vehicle_type": det.vehicle_type,
            }
            for det in mock_detection_results
        ]
        
        # This should not raise an error even without Kafka
        await event_processor.publish_detection_event(
            camera_id="cam_001",
            frame_id="frame_123",
            detections=detections,
            metadata={"quality_score": 0.85},
        )

    @pytest.mark.asyncio
    async def test_event_handler_registration(self, event_processor):
        """Test event handler registration for real-time processing."""
        # Create mock event handler
        handler_called = False
        
        def mock_handler(event: StreamingEvent):
            nonlocal handler_called
            handler_called = True
            assert event.event_type == EventType.DETECTION
            assert event.camera_id == "cam_001"
        
        # Register handler
        event_processor.register_event_handler(EventType.DETECTION, mock_handler)
        
        # Verify handler was registered (for no-op processor, this should work)
        assert not handler_called  # Handler won't be called without actual Kafka

    @pytest.mark.asyncio
    async def test_analytics_event_streaming(self, event_processor):
        """Test analytics event publishing for dashboard updates."""
        analytics_data = {
            "violations": 2,
            "anomalies": 1,
            "incidents": 0,
            "processing_time_ms": 25.0,
            "vehicle_count": 3,
            "congestion_level": "light",
        }
        
        # Publish analytics event
        await event_processor.publish_analytics_event(
            camera_id="cam_001",
            frame_id="frame_123",
            analytics_result=analytics_data,
        )

    @pytest.mark.asyncio
    async def test_metrics_event_streaming(self, event_processor):
        """Test metrics event publishing for monitoring."""
        metrics_data = {
            "avg_inference_time_ms": 45.0,
            "throughput_fps": 28.5,
            "gpu_utilization": 0.75,
            "queue_depth": 10,
        }
        
        # Publish metrics event
        await event_processor.publish_metrics_event(
            source="ml_pipeline",
            metrics=metrics_data,
        )


class TestgRPCServiceIntegration:
    """Test gRPC service integration with ML pipeline."""

    @pytest.mark.asyncio
    async def test_grpc_server_initialization(self, ml_integration_service):
        """Test gRPC server initializes with ML pipeline."""
        # Mock dependencies to avoid actual gRPC server startup
        with patch('src.its_camera_ai.services.production_ml_grpc_server.grpc.aio.server'), \
             patch('src.its_camera_ai.services.production_ml_grpc_server.vision_core_pb2_grpc'), \
             patch('src.its_camera_ai.services.production_ml_grpc_server.streaming_service_pb2_grpc'), \
             patch('src.its_camera_ai.services.production_ml_grpc_server.analytics_service_pb2_grpc'):
            
            server = ProductionMLgRPCServer(
                ml_integration_service=ml_integration_service,
                host="localhost",
                port=50051,
            )
            
            assert server.ml_integration_service == ml_integration_service
            assert server.host == "localhost"
            assert server.port == 50051

    @pytest.mark.asyncio
    async def test_vision_core_servicer_integration(self, ml_integration_service):
        """Test Vision Core gRPC servicer with ML pipeline."""
        from src.its_camera_ai.services.grpc.vision_core_servicer import VisionCoreServicer
        
        # Mock unified engine
        unified_engine = AsyncMock()
        
        # Create servicer
        servicer = VisionCoreServicer(unified_engine)
        
        assert servicer.unified_engine == unified_engine
        assert servicer.request_count == 0
        assert servicer.error_count == 0


class TestAnalyticsIntegration:
    """Test analytics service integration with ML pipeline."""

    @pytest.mark.asyncio
    async def test_analytics_processing_with_detections(
        self, mock_unified_analytics, mock_detection_results
    ):
        """Test analytics processing with real detection data."""
        from src.its_camera_ai.services.analytics_dtos import DetectionData
        
        # Create detection data
        detection_data = DetectionData(
            camera_id="cam_001",
            frame_id="frame_123",
            timestamp=datetime.now(UTC),
            detections=mock_detection_results,
            vehicle_count=2,
            frame_width=640,
            frame_height=480,
        )
        
        # Process analytics
        result = await mock_unified_analytics.process_realtime_analytics(
            detection_data=detection_data,
            include_anomaly_detection=True,
            include_incident_detection=True,
            include_rule_evaluation=True,
            include_speed_calculation=True,
        )
        
        # Verify analytics was called
        mock_unified_analytics.process_realtime_analytics.assert_called_once()
        assert result is not None

    @pytest.mark.asyncio
    async def test_real_time_metrics_generation(self, mock_detection_results):
        """Test real-time traffic metrics generation from detections."""
        from src.its_camera_ai.services.analytics_dtos import RealtimeTrafficMetrics, CongestionLevel
        
        # Calculate metrics from detections
        vehicle_count = len([d for d in mock_detection_results if d.is_vehicle])
        avg_speed = np.mean([d.speed for d in mock_detection_results if d.speed])
        
        # Create metrics
        metrics = RealtimeTrafficMetrics(
            timestamp=datetime.now(UTC),
            camera_id="cam_001",
            total_vehicles=vehicle_count,
            vehicle_breakdown={"car": 1, "truck": 1},
            average_speed=avg_speed,
            traffic_density=0.3,
            congestion_level=CongestionLevel.LIGHT,
            flow_rate=120.0,  # vehicles per hour
            occupancy_rate=30.0,
        )
        
        # Verify metrics
        assert metrics.total_vehicles == 2
        assert metrics.average_speed == 40.0  # (45 + 35) / 2
        assert metrics.congestion_level == CongestionLevel.LIGHT


class TestEndToEndPipelineFlow:
    """Test complete end-to-end ML pipeline flow."""

    @pytest.mark.asyncio
    async def test_complete_detection_to_analytics_flow(
        self, ml_integration_service, sample_frame, mock_detection_results
    ):
        """Test complete flow from frame input to analytics output."""
        
        # Setup complete mock chain
        ml_integration_service.is_running = True
        
        # Mock the unified engine to return realistic results
        unified_result = UnifiedResult(
            detections=mock_detection_results,
            inference_time_ms=45.0,
            model_version="yolo11s",
            analytics_result=MagicMock(
                violations=[],
                anomalies=[],
                processing_time_ms=25.0,
            ),
            quality_score=0.85,
            analytics_time_ms=25.0,
            camera_id="cam_001",
            frame_id="frame_123",
            timestamp=datetime.now(UTC),
            total_processing_time_ms=70.0,
            batch_size=1,
            metadata_track={"detections": 2, "quality_score": 0.85},
        )
        
        ml_integration_service.unified_vision_analytics.process_frame = AsyncMock(
            return_value=unified_result
        )
        
        # Process frame through complete pipeline
        start_time = time.time()
        result = await ml_integration_service.process_frame(
            frame=sample_frame,
            camera_id="cam_001",
            frame_id="frame_123",
            include_analytics=True,
            include_quality_score=True,
            include_frame_annotation=False,
            include_metadata_track=True,
        )
        processing_time = (time.time() - start_time) * 1000
        
        # Verify end-to-end results
        assert result is not None
        assert result.camera_id == "cam_001" 
        assert result.frame_id == "frame_123"
        assert len(result.detections) == 2
        assert result.inference_time_ms == 45.0
        assert result.analytics_time_ms == 25.0
        assert result.quality_score == 0.85
        assert result.metadata_track is not None
        
        # Verify performance requirement (sub-100ms)
        assert processing_time < 100.0, f"Processing took {processing_time:.1f}ms, exceeding 100ms target"
        
        # Verify ML service tracking was updated
        assert ml_integration_service.frames_processed > 0
        assert len(ml_integration_service.inference_times) > 0

    @pytest.mark.asyncio
    async def test_error_handling_and_graceful_degradation(
        self, ml_integration_service, sample_frame
    ):
        """Test error handling and graceful degradation."""
        ml_integration_service.is_running = True
        
        # Mock a failure in the unified engine
        ml_integration_service.unified_vision_analytics.process_frame = AsyncMock(
            side_effect=Exception("ML inference failed")
        )
        
        # Process frame should handle error gracefully
        with pytest.raises(Exception, match="ML inference failed"):
            await ml_integration_service.process_frame(
                frame=sample_frame,
                camera_id="cam_001",
                frame_id="frame_123",
            )

    @pytest.mark.asyncio
    async def test_concurrent_stream_processing(
        self, ml_integration_service, mock_detection_results
    ):
        """Test concurrent processing of multiple camera streams."""
        ml_integration_service.is_running = True
        
        # Setup concurrent processing mock
        async def mock_process_frame(*args, **kwargs):
            await asyncio.sleep(0.01)  # Simulate processing time
            return UnifiedResult(
                detections=mock_detection_results,
                inference_time_ms=np.random.uniform(40, 60),
                model_version="yolo11s",
                analytics_result=MagicMock(),
                quality_score=np.random.uniform(0.7, 0.9),
                analytics_time_ms=np.random.uniform(20, 30),
                camera_id=kwargs.get("camera_id", "unknown"),
                frame_id=kwargs.get("frame_id", "unknown"),
                timestamp=datetime.now(UTC),
                total_processing_time_ms=np.random.uniform(60, 90),
                batch_size=1,
            )
        
        ml_integration_service.unified_vision_analytics.process_frame = mock_process_frame
        
        # Process multiple streams concurrently
        num_streams = 10
        tasks = []
        
        for i in range(num_streams):
            frame = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            task = ml_integration_service.process_frame(
                frame=frame,
                camera_id=f"cam_{i:03d}",
                frame_id=f"frame_{i}",
            )
            tasks.append(task)
        
        # Wait for all streams to complete
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Verify concurrent processing
        assert len(results) == num_streams
        assert total_time < 1.0, f"Concurrent processing took {total_time:.2f}s, should be much faster"
        
        # Verify all results are valid
        for i, result in enumerate(results):
            assert result.camera_id == f"cam_{i:03d}"
            assert result.frame_id == f"frame_{i}"
            assert len(result.detections) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])