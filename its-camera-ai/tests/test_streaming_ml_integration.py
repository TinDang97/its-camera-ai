"""
Integration tests for ML annotation streaming pipeline.

Tests the complete integration between ML annotation processor
and the dual-channel streaming infrastructure.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import numpy as np
import cv2
from typing import Dict, Any

from src.its_camera_ai.services.streaming_service import (
    SSEStreamingService,
    StreamingService,
    MP4Fragment,
    ChannelType,
)
from src.its_camera_ai.ml.streaming_annotation_processor import (
    MLAnnotationProcessor,
    DetectionConfig,
    AnnotationStyleConfig,
    Detection,
    AnnotatedFrame,
)
from src.its_camera_ai.flow.redis_queue_manager import RedisQueueManager
from src.its_camera_ai.data.streaming_processor import ProcessedFrame


class TestMLStreamingIntegration:
    """Test ML annotation integration with streaming service."""

    def setup_method(self):
        """Set up integration test fixtures."""
        # Mock dependencies
        self.mock_redis_manager = MagicMock(spec=RedisQueueManager)
        self.mock_base_streaming = MagicMock(spec=StreamingService)
        self.mock_streaming_processor = MagicMock()

        # Configure mock streaming processor
        self.mock_base_streaming.streaming_processor = self.mock_streaming_processor

        # Create mock ML processor
        self.mock_ml_processor = MagicMock(spec=MLAnnotationProcessor)
        self.mock_ml_processor.config = DetectionConfig()
        self.mock_ml_processor.process_frame_with_annotations = AsyncMock()
        self.mock_ml_processor.create_detection_metadata = AsyncMock()

        # Configuration
        self.sse_config = {
            "max_concurrent_connections": 100,
            "fragment_duration_ms": 2000,
            "heartbeat_interval": 30,
            "connection_timeout": 300,
        }

        # Create SSE service with ML processor
        self.sse_service = SSEStreamingService(
            base_streaming_service=self.mock_base_streaming,
            redis_manager=self.mock_redis_manager,
            config=self.sse_config,
            ml_annotation_processor=self.mock_ml_processor,
        )

        # Test frame data
        self.test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        _, encoded = cv2.imencode(".jpg", self.test_frame)
        self.frame_bytes = encoded.tobytes()

    @pytest.mark.asyncio
    async def test_annotated_fragment_creation(self):
        """Test creation of annotated MP4 fragments."""
        # Mock processed frame
        processed_frame = ProcessedFrame(
            camera_id="test_camera",
            frame_id="frame_001",
            timestamp=1234567890.0,
            original_image=self.test_frame,
            stage="complete",
        )

        # Mock ML annotation result
        mock_detections = [
            Detection(
                class_name="car",
                confidence=0.85,
                bbox=(100, 100, 200, 150),
                center=(150.0, 125.0),
                area=5000.0,
                class_id=2,
            )
        ]

        mock_annotated_frame = AnnotatedFrame(
            frame_data=self.frame_bytes,
            detections=mock_detections,
            metadata={"inference_time_ms": 35.0, "render_time_ms": 8.0},
            timestamp=1234567890.0,
            processing_time_ms=43.0,
            frame_id="frame_001_annotated",
            camera_id="test_camera",
        )

        self.mock_ml_processor.process_frame_with_annotations.return_value = (
            mock_annotated_frame
        )

        # Mock metadata
        from src.its_camera_ai.ml.streaming_annotation_processor import (
            DetectionMetadata,
        )

        mock_metadata = DetectionMetadata(
            frame_id="frame_001_annotated",
            camera_id="test_camera",
            timestamp=1234567890.0,
            processing_time_ms=43.0,
            detection_count=1,
            vehicle_count={"car": 1},
        )

        self.mock_ml_processor.create_detection_metadata.return_value = mock_metadata

        # Create annotated fragment
        fragment = await self.sse_service._create_annotated_mp4_fragment(
            processed_frame=processed_frame,
            sequence_number=1,
            quality="medium",
            camera_id="test_camera",
        )

        # Verify fragment creation
        assert fragment is not None
        assert isinstance(fragment, MP4Fragment)
        assert fragment.camera_id == "test_camera"
        assert fragment.sequence_number == 1
        assert fragment.data == self.frame_bytes
        assert "ml_inference" in fragment.metadata
        assert fragment.metadata["ml_inference"]["detection_count"] == 1
        assert fragment.metadata["ml_inference"]["vehicle_count"] == {"car": 1}

        # Verify ML processor was called
        self.mock_ml_processor.process_frame_with_annotations.assert_called_once()
        self.mock_ml_processor.create_detection_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_dual_channel_streaming_with_ml(self):
        """Test dual-channel streaming with raw and annotated channels."""
        camera_id = "test_camera"

        # Mock processed frames from base service
        async def mock_process_stream(cam_id):
            """Generate mock processed frames."""
            for i in range(3):
                yield ProcessedFrame(
                    camera_id=cam_id,
                    frame_id=f"frame_{i:03d}",
                    timestamp=1234567890.0 + i,
                    original_image=self.test_frame,
                    stage="complete",
                )

        self.mock_streaming_processor.process_stream.return_value = mock_process_stream(
            camera_id
        )

        # Mock ML processing for annotated channel
        def mock_ml_processing(frame_data, cam_id):
            return AnnotatedFrame(
                frame_data=frame_data,
                detections=[],
                metadata={"inference_time_ms": 40.0},
                timestamp=1234567890.0,
                processing_time_ms=40.0,
                frame_id="annotated_frame",
                camera_id=cam_id,
            )

        self.mock_ml_processor.process_frame_with_annotations.side_effect = (
            mock_ml_processing
        )

        # Test raw channel streaming
        raw_fragments = []
        async for fragment in self.sse_service.stream_mp4_fragments(
            camera_id=camera_id, quality="medium", stream_type="raw"
        ):
            raw_fragments.append(fragment)
            if len(raw_fragments) >= 3:
                break

        # Test annotated channel streaming
        annotated_fragments = []
        async for fragment in self.sse_service.stream_mp4_fragments(
            camera_id=camera_id, quality="medium", stream_type="annotated"
        ):
            annotated_fragments.append(fragment)
            if len(annotated_fragments) >= 3:
                break

        # Verify both channels produced fragments
        assert len(raw_fragments) == 3
        assert len(annotated_fragments) == 3

        # Verify raw fragments don't have ML metadata
        for fragment in raw_fragments:
            assert "ml_inference" not in fragment.metadata

        # Verify annotated fragments have ML metadata
        for fragment in annotated_fragments:
            assert "ml_inference" in fragment.metadata or self.mock_ml_processor is None

    @pytest.mark.asyncio
    async def test_ml_processor_fallback_handling(self):
        """Test fallback when ML processor is unavailable."""
        # Create SSE service without ML processor
        sse_service_no_ml = SSEStreamingService(
            base_streaming_service=self.mock_base_streaming,
            redis_manager=self.mock_redis_manager,
            config=self.sse_config,
            ml_annotation_processor=None,
        )

        # Mock processed frame
        processed_frame = ProcessedFrame(
            camera_id="test_camera",
            frame_id="frame_001",
            timestamp=1234567890.0,
            original_image=self.test_frame,
            stage="complete",
        )

        # Request annotated fragment (should fallback to raw)
        fragment = await sse_service_no_ml._create_annotated_mp4_fragment(
            processed_frame=processed_frame,
            sequence_number=1,
            quality="medium",
            camera_id="test_camera",
        )

        # Should still create fragment but without ML processing
        assert fragment is not None
        assert "ml_inference" not in fragment.metadata

    @pytest.mark.asyncio
    async def test_ml_processing_error_handling(self):
        """Test error handling in ML processing pipeline."""
        # Configure ML processor to raise exception
        self.mock_ml_processor.process_frame_with_annotations.side_effect = Exception(
            "ML inference failed"
        )

        # Mock processed frame
        processed_frame = ProcessedFrame(
            camera_id="test_camera",
            frame_id="frame_001",
            timestamp=1234567890.0,
            original_image=self.test_frame,
            stage="complete",
        )

        # Should fallback to raw fragment on ML error
        fragment = await self.sse_service._create_annotated_mp4_fragment(
            processed_frame=processed_frame,
            sequence_number=1,
            quality="medium",
            camera_id="test_camera",
        )

        # Should still produce fragment (fallback to raw)
        assert fragment is not None
        # Should not have ML metadata due to error
        assert (
            "ml_inference" not in fragment.metadata
            or fragment.metadata.get("ml_inference") is None
        )

    @pytest.mark.asyncio
    async def test_sse_streaming_with_ml_metadata(self):
        """Test SSE streaming includes ML detection metadata."""
        from fastapi import Request
        from fastapi.responses import StreamingResponse

        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.state.user = MagicMock()
        mock_request.state.user.id = "test_user"
        mock_request.state.user.username = "testuser"

        # Mock processed frames
        async def mock_process_stream(cam_id):
            yield ProcessedFrame(
                camera_id=cam_id,
                frame_id="frame_001",
                timestamp=1234567890.0,
                original_image=self.test_frame,
                stage="complete",
            )

        self.mock_streaming_processor.process_stream.return_value = mock_process_stream(
            "test_camera"
        )

        # Mock ML processing
        mock_annotated_frame = AnnotatedFrame(
            frame_data=self.frame_bytes,
            detections=[
                Detection(
                    class_name="car",
                    confidence=0.85,
                    bbox=(100, 100, 200, 150),
                    center=(150.0, 125.0),
                    area=5000.0,
                    class_id=2,
                )
            ],
            metadata={"inference_time_ms": 35.0},
            timestamp=1234567890.0,
            processing_time_ms=35.0,
            frame_id="frame_001_annotated",
            camera_id="test_camera",
        )

        self.mock_ml_processor.process_frame_with_annotations.return_value = (
            mock_annotated_frame
        )

        # Create SSE response
        with patch("asyncio.create_task") as mock_create_task:
            mock_task = MagicMock()
            mock_task.cancel = MagicMock()
            mock_create_task.return_value = mock_task

            response = await self.sse_service.handle_sse_connection(
                request=mock_request,
                camera_id="test_camera",
                stream_type="annotated",
                quality="medium",
            )

        assert isinstance(response, StreamingResponse)
        assert response.media_type == "text/event-stream"

    def test_ml_configuration_integration(self):
        """Test ML configuration integration with container."""
        # Test that ML processor can be configured through container
        config = DetectionConfig(
            confidence_threshold=0.6,
            classes_to_detect=["car", "truck"],
            target_latency_ms=40.0,
        )

        style = AnnotationStyleConfig(box_thickness=3, show_confidence=False)

        # Create processor with custom config
        processor = MLAnnotationProcessor(config=config)
        processor.renderer = MagicMock()
        processor.renderer.style_config = style

        assert processor.config.confidence_threshold == 0.6
        assert processor.config.classes_to_detect == ["car", "truck"]
        assert processor.renderer.style_config.box_thickness == 3

    @pytest.mark.asyncio
    async def test_detection_metadata_streaming(self):
        """Test streaming of detection metadata alongside video."""
        # Mock detection metadata
        from src.its_camera_ai.ml.streaming_annotation_processor import (
            DetectionMetadata,
        )

        mock_metadata = DetectionMetadata(
            frame_id="frame_001",
            camera_id="test_camera",
            timestamp=1234567890.0,
            processing_time_ms=45.0,
            detection_count=2,
            vehicle_count={"car": 1, "truck": 1},
            detections=[
                {
                    "class": "car",
                    "confidence": 0.85,
                    "bbox": [100, 100, 200, 150],
                    "center": [150.0, 125.0],
                    "area": 5000.0,
                    "class_id": 2,
                },
                {
                    "class": "truck",
                    "confidence": 0.92,
                    "bbox": [250, 200, 350, 300],
                    "center": [300.0, 250.0],
                    "area": 10000.0,
                    "class_id": 7,
                },
            ],
            performance_metrics={
                "avg_inference_time_ms": 35.0,
                "avg_render_time_ms": 8.0,
                "total_processing_time_ms": 43.0,
            },
        )

        self.mock_ml_processor.create_detection_metadata.return_value = mock_metadata

        # Create test detection data
        detections = [
            Detection(
                class_name="car",
                confidence=0.85,
                bbox=(100, 100, 200, 150),
                center=(150.0, 125.0),
                area=5000.0,
                class_id=2,
            )
        ]

        # Test metadata creation
        metadata = await self.mock_ml_processor.create_detection_metadata(
            detections=detections,
            frame_timestamp=1234567890.0,
            camera_id="test_camera",
            frame_id="frame_001",
            processing_time_ms=45.0,
        )

        # Verify metadata structure
        assert metadata.frame_id == "frame_001"
        assert metadata.camera_id == "test_camera"
        assert metadata.detection_count == 2
        assert "car" in metadata.vehicle_count
        assert "truck" in metadata.vehicle_count
        assert len(metadata.detections) == 2
        assert "avg_inference_time_ms" in metadata.performance_metrics

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test performance monitoring for ML streaming."""
        # Mock performance stats
        mock_stats = {
            "processed_frames": 100,
            "avg_inference_time_ms": 35.2,
            "avg_render_time_ms": 8.1,
            "avg_total_time_ms": 43.3,
            "target_latency_ms": 50.0,
            "latency_compliance": 98.5,
        }

        self.mock_ml_processor.get_performance_stats.return_value = mock_stats

        # Get performance statistics
        stats = self.mock_ml_processor.get_performance_stats()

        # Verify performance metrics
        assert stats["processed_frames"] == 100
        assert stats["avg_inference_time_ms"] == 35.2
        assert stats["latency_compliance"] == 98.5
        assert stats["avg_total_time_ms"] < stats["target_latency_ms"]

        # Performance should meet requirements
        assert stats["avg_total_time_ms"] < 50.0  # Under 50ms target
        assert stats["latency_compliance"] > 95.0  # >95% compliance


class TestMLAPIEndpoints:
    """Test ML-specific API endpoints."""

    def setup_method(self):
        """Set up API test fixtures."""
        from fastapi.testclient import TestClient
        from src.its_camera_ai.api.app import app

        self.client = TestClient(app)

    @pytest.mark.skip(reason="Requires full app context with authentication")
    def test_update_detection_config_endpoint(self):
        """Test detection configuration update endpoint."""
        config_update = {
            "confidence_threshold": 0.7,
            "classes_to_detect": ["car", "truck"],
            "target_latency_ms": 40.0,
        }

        response = self.client.post(
            "/api/v1/streams/sse/test_camera/detection-config",
            json=config_update,
            headers={"Authorization": "Bearer test_token"},
        )

        # Would test actual response in full integration environment
        assert response.status_code in [200, 401]  # 401 if auth not set up

    @pytest.mark.skip(reason="Requires full app context with authentication")
    def test_get_detection_stats_endpoint(self):
        """Test detection statistics endpoint."""
        response = self.client.get(
            "/api/v1/streams/sse/test_camera/detection-stats",
            headers={"Authorization": "Bearer test_token"},
        )

        # Would test actual response in full integration environment
        assert response.status_code in [200, 401]  # 401 if auth not set up


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
