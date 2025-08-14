"""
Tests for ML Annotation Processor for real-time streaming.

Comprehensive test suite covering ML inference integration,
annotation rendering, performance optimization, and streaming integration.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import numpy as np
import cv2
from typing import List, Dict, Any

from src.its_camera_ai.ml.streaming_annotation_processor import (
    MLAnnotationProcessor,
    DetectionConfig,
    AnnotationStyleConfig,
    AnnotationRenderer,
    Detection,
    AnnotatedFrame,
    DetectionMetadata,
)
from src.its_camera_ai.ml.core_vision_engine import CoreVisionEngine, VisionConfig
from src.its_camera_ai.core.exceptions import StreamProcessingError


class TestDetectionConfig:
    """Test detection configuration."""
    
    def test_default_config(self):
        """Test default detection configuration."""
        config = DetectionConfig()
        
        assert config.confidence_threshold == 0.5
        assert config.nms_threshold == 0.4
        assert config.max_detections == 100
        assert "car" in config.classes_to_detect
        assert "truck" in config.classes_to_detect
        assert config.enable_gpu_acceleration is True
        assert config.target_latency_ms == 50.0
    
    def test_custom_config(self):
        """Test custom detection configuration."""
        config = DetectionConfig(
            confidence_threshold=0.7,
            classes_to_detect=["car", "bus"],
            target_latency_ms=30.0,
            batch_size=16
        )
        
        assert config.confidence_threshold == 0.7
        assert config.classes_to_detect == ["car", "bus"]
        assert config.target_latency_ms == 30.0
        assert config.batch_size == 16


class TestAnnotationStyleConfig:
    """Test annotation styling configuration."""
    
    def test_default_style(self):
        """Test default annotation style."""
        style = AnnotationStyleConfig()
        
        assert style.box_thickness == 2
        assert style.font_scale == 0.6
        assert style.show_confidence is True
        assert style.show_class_labels is True
        assert "car" in style.box_color_map
        assert style.box_color_map["car"] == (0, 255, 0)  # Green
    
    def test_custom_style(self):
        """Test custom annotation style."""
        custom_colors = {"car": (255, 0, 0), "truck": (0, 0, 255)}
        style = AnnotationStyleConfig(
            box_color_map=custom_colors,
            box_thickness=3,
            font_scale=0.8,
            show_confidence=False
        )
        
        assert style.box_color_map == custom_colors
        assert style.box_thickness == 3
        assert style.font_scale == 0.8
        assert style.show_confidence is False


class TestAnnotationRenderer:
    """Test annotation rendering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.renderer = AnnotationRenderer()
        self.test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray frame
    
    def test_renderer_initialization(self):
        """Test renderer initialization."""
        renderer = AnnotationRenderer()
        assert renderer.style_config is not None
        assert hasattr(renderer, '_font')
    
    def test_render_single_detection(self):
        """Test rendering single detection."""
        detection = Detection(
            class_name="car",
            confidence=0.85,
            bbox=(100, 100, 200, 150),
            center=(150.0, 125.0),
            area=5000.0,
            class_id=2
        )
        
        result = asyncio.run(
            self.renderer.render_detections_on_frame(
                self.test_frame, [detection]
            )
        )
        
        assert result.shape == self.test_frame.shape
        assert not np.array_equal(result, self.test_frame)  # Should be modified
    
    def test_render_multiple_detections(self):
        """Test rendering multiple detections."""
        detections = [
            Detection(
                class_name="car",
                confidence=0.85,
                bbox=(100, 100, 200, 150),
                center=(150.0, 125.0),
                area=5000.0,
                class_id=2
            ),
            Detection(
                class_name="truck",
                confidence=0.92,
                bbox=(250, 200, 350, 300),
                center=(300.0, 250.0),
                area=10000.0,
                class_id=7
            )
        ]
        
        result = asyncio.run(
            self.renderer.render_detections_on_frame(
                self.test_frame, detections
            )
        )
        
        assert result.shape == self.test_frame.shape
        assert not np.array_equal(result, self.test_frame)
    
    def test_render_empty_detections(self):
        """Test rendering with no detections."""
        result = asyncio.run(
            self.renderer.render_detections_on_frame(
                self.test_frame, []
            )
        )
        
        # Should return original frame when no detections
        assert np.array_equal(result, self.test_frame)
    
    def test_create_overlay(self):
        """Test creating transparent overlay."""
        detection = Detection(
            class_name="car",
            confidence=0.85,
            bbox=(100, 100, 200, 150),
            center=(150.0, 125.0),
            area=5000.0,
            class_id=2
        )
        
        overlay = self.renderer.create_annotation_overlay(
            (640, 480), [detection]
        )
        
        assert overlay.shape == (480, 640, 4)  # RGBA
        assert overlay.dtype == np.uint8
    
    def test_render_with_custom_style(self):
        """Test rendering with custom style configuration."""
        custom_style = AnnotationStyleConfig(
            box_thickness=5,
            show_confidence=False,
            show_class_labels=False
        )
        
        detection = Detection(
            class_name="car",
            confidence=0.85,
            bbox=(100, 100, 200, 150),
            center=(150.0, 125.0),
            area=5000.0,
            class_id=2
        )
        
        result = asyncio.run(
            self.renderer.render_detections_on_frame(
                self.test_frame, [detection], custom_style
            )
        )
        
        assert result.shape == self.test_frame.shape
        assert not np.array_equal(result, self.test_frame)


class TestMLAnnotationProcessor:
    """Test ML annotation processor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_vision_engine = MagicMock(spec=CoreVisionEngine)
        self.mock_vision_engine.initialize = AsyncMock()
        self.mock_vision_engine.detect_objects = AsyncMock()
        
        self.config = DetectionConfig(
            confidence_threshold=0.6,
            target_latency_ms=40.0,
            batch_size=4
        )
        
        self.processor = MLAnnotationProcessor(
            vision_engine=self.mock_vision_engine,
            config=self.config
        )
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self):
        """Test processor initialization."""
        await self.processor.initialize()
        
        assert self.processor._active_engine is not None
        self.mock_vision_engine.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_frame_with_annotations(self):
        """Test frame processing with ML annotations."""
        # Mock vision engine response
        mock_detection_result = MagicMock()
        mock_detection_result.detections = [
            {
                'class': 'car',
                'confidence': 0.85,
                'bbox': (100, 100, 200, 150),
                'center': (150.0, 125.0),
                'area': 5000.0,
                'class_id': 2
            }
        ]
        self.mock_vision_engine.detect_objects.return_value = mock_detection_result
        
        await self.processor.initialize()
        
        # Create test frame bytes
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        _, encoded = cv2.imencode('.jpg', test_frame)
        frame_bytes = encoded.tobytes()
        
        # Process frame
        result = await self.processor.process_frame_with_annotations(
            frame_data=frame_bytes,
            camera_id="test_camera"
        )
        
        assert isinstance(result, AnnotatedFrame)
        assert result.camera_id == "test_camera"
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "car"
        assert result.processing_time_ms > 0
        assert result.frame_data is not None
    
    @pytest.mark.asyncio
    async def test_process_invalid_frame(self):
        """Test processing invalid frame data."""
        await self.processor.initialize()
        
        with pytest.raises(StreamProcessingError):
            await self.processor.process_frame_with_annotations(
                frame_data=b"invalid_frame_data",
                camera_id="test_camera"
            )
    
    @pytest.mark.asyncio
    async def test_filter_detections_by_config(self):
        """Test detection filtering based on configuration."""
        # Mock vision engine with multiple detections
        mock_detection_result = MagicMock()
        mock_detection_result.detections = [
            {
                'class': 'car',
                'confidence': 0.85,
                'bbox': (100, 100, 200, 150),
                'center': (150.0, 125.0),
                'area': 5000.0,
                'class_id': 2
            },
            {
                'class': 'person',  # Not in filter list
                'confidence': 0.75,
                'bbox': (300, 100, 320, 180),
                'center': (310.0, 140.0),
                'area': 1600.0,
                'class_id': 0
            },
            {
                'class': 'car',
                'confidence': 0.45,  # Below threshold
                'bbox': (400, 200, 450, 240),
                'center': (425.0, 220.0),
                'area': 2000.0,
                'class_id': 2
            }
        ]
        self.mock_vision_engine.detect_objects.return_value = mock_detection_result
        
        # Configure to only detect cars with high confidence
        self.processor.config.classes_to_detect = ["car"]
        self.processor.config.confidence_threshold = 0.6
        
        await self.processor.initialize()
        
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        _, encoded = cv2.imencode('.jpg', test_frame)
        frame_bytes = encoded.tobytes()
        
        result = await self.processor.process_frame_with_annotations(
            frame_data=frame_bytes,
            camera_id="test_camera"
        )
        
        # Should only have one detection (car with confidence 0.85)
        assert len(result.detections) == 1
        assert result.detections[0].class_name == "car"
        assert result.detections[0].confidence == 0.85
    
    @pytest.mark.asyncio
    async def test_vehicle_priority_boost(self):
        """Test vehicle priority confidence boost."""
        mock_detection_result = MagicMock()
        mock_detection_result.detections = [
            {
                'class': 'car',
                'confidence': 0.55,  # Will be boosted
                'bbox': (100, 100, 200, 150),
                'center': (150.0, 125.0),
                'area': 5000.0,
                'class_id': 2
            }
        ]
        self.mock_vision_engine.detect_objects.return_value = mock_detection_result
        
        # Enable vehicle priority
        self.processor.config.vehicle_priority = True
        self.processor.config.confidence_boost_factor = 1.2
        
        await self.processor.initialize()
        
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        _, encoded = cv2.imencode('.jpg', test_frame)
        frame_bytes = encoded.tobytes()
        
        result = await self.processor.process_frame_with_annotations(
            frame_data=frame_bytes,
            camera_id="test_camera"
        )
        
        # Confidence should be boosted
        assert len(result.detections) == 1
        assert result.detections[0].confidence == 0.66  # 0.55 * 1.2
    
    @pytest.mark.asyncio
    async def test_max_detections_limit(self):
        """Test maximum detections limit."""
        # Create many mock detections
        detections = []
        for i in range(150):  # More than max_detections (100)
            detections.append({
                'class': 'car',
                'confidence': 0.9 - i * 0.001,  # Decreasing confidence
                'bbox': (i, i, i+50, i+30),
                'center': (i+25.0, i+15.0),
                'area': 1500.0,
                'class_id': 2
            })
        
        mock_detection_result = MagicMock()
        mock_detection_result.detections = detections
        self.mock_vision_engine.detect_objects.return_value = mock_detection_result
        
        self.processor.config.max_detections = 50
        
        await self.processor.initialize()
        
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        _, encoded = cv2.imencode('.jpg', test_frame)
        frame_bytes = encoded.tobytes()
        
        result = await self.processor.process_frame_with_annotations(
            frame_data=frame_bytes,
            camera_id="test_camera"
        )
        
        # Should be limited to max_detections
        assert len(result.detections) == 50
        # Should be sorted by confidence (highest first)
        confidences = [det.confidence for det in result.detections]
        assert confidences == sorted(confidences, reverse=True)
    
    @pytest.mark.asyncio
    async def test_create_detection_metadata(self):
        """Test detection metadata creation."""
        detections = [
            Detection(
                class_name="car",
                confidence=0.85,
                bbox=(100, 100, 200, 150),
                center=(150.0, 125.0),
                area=5000.0,
                class_id=2
            ),
            Detection(
                class_name="truck",
                confidence=0.92,
                bbox=(250, 200, 350, 300),
                center=(300.0, 250.0),
                area=10000.0,
                class_id=7
            )
        ]
        
        metadata = await self.processor.create_detection_metadata(
            detections=detections,
            frame_timestamp=time.time(),
            camera_id="test_camera",
            frame_id="frame_123",
            processing_time_ms=45.5
        )
        
        assert isinstance(metadata, DetectionMetadata)
        assert metadata.camera_id == "test_camera"
        assert metadata.frame_id == "frame_123"
        assert metadata.detection_count == 2
        assert metadata.processing_time_ms == 45.5
        assert "car" in metadata.vehicle_count
        assert "truck" in metadata.vehicle_count
        assert metadata.vehicle_count["car"] == 1
        assert metadata.vehicle_count["truck"] == 1
        assert len(metadata.detections) == 2
    
    def test_get_performance_stats(self):
        """Test performance statistics retrieval."""
        # Simulate some processing
        self.processor.inference_times = [40.0, 42.0, 38.0, 41.0, 39.0]
        self.processor.annotation_times = [8.0, 9.0, 7.5, 8.5, 8.0]
        self.processor._processed_frames = 5
        
        stats = self.processor.get_performance_stats()
        
        assert stats["processed_frames"] == 5
        assert stats["avg_inference_time_ms"] == 40.0
        assert stats["avg_render_time_ms"] == 8.2
        assert stats["avg_total_time_ms"] == 48.2
        assert stats["target_latency_ms"] == 40.0
        assert stats["latency_compliance"] == 100.0  # All under target
    
    @pytest.mark.asyncio
    async def test_processor_cleanup(self):
        """Test processor cleanup."""
        mock_engine = MagicMock()
        mock_engine.cleanup = AsyncMock()
        self.processor._active_engine = mock_engine
        
        await self.processor.cleanup()
        
        mock_engine.cleanup.assert_called_once()


class TestPerformanceRequirements:
    """Test performance requirements and optimization."""
    
    @pytest.mark.asyncio
    async def test_latency_requirement(self):
        """Test that processing meets latency requirements."""
        # Mock fast vision engine
        mock_engine = AsyncMock()
        mock_engine.detect_objects.return_value = MagicMock(detections=[])
        
        config = DetectionConfig(target_latency_ms=50.0)
        processor = MLAnnotationProcessor(
            vision_engine=mock_engine,
            config=config
        )
        processor._active_engine = mock_engine
        
        # Create test frame
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        _, encoded = cv2.imencode('.jpg', test_frame)
        frame_bytes = encoded.tobytes()
        
        # Measure processing time
        start_time = time.time()
        result = await processor.process_frame_with_annotations(
            frame_data=frame_bytes,
            camera_id="test_camera"
        )
        end_time = time.time()
        
        actual_latency = (end_time - start_time) * 1000
        
        # Should meet latency requirement (with some tolerance for test environment)
        assert actual_latency < config.target_latency_ms * 2  # 2x tolerance
        assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent frame processing."""
        mock_engine = AsyncMock()
        mock_engine.detect_objects.return_value = MagicMock(detections=[])
        
        processor = MLAnnotationProcessor(vision_engine=mock_engine)
        processor._active_engine = mock_engine
        
        # Create test frame
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        _, encoded = cv2.imencode('.jpg', test_frame)
        frame_bytes = encoded.tobytes()
        
        # Process multiple frames concurrently
        tasks = []
        for i in range(5):
            task = processor.process_frame_with_annotations(
                frame_data=frame_bytes,
                camera_id=f"camera_{i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.camera_id == f"camera_{i}"
            assert isinstance(result, AnnotatedFrame)


class TestIntegrationScenarios:
    """Test integration with streaming infrastructure."""
    
    @pytest.mark.asyncio
    async def test_streaming_integration(self):
        """Test integration with streaming service."""
        # This would test integration with the actual streaming service
        # For now, we'll test the interface compatibility
        
        processor = MLAnnotationProcessor()
        
        # Test that processor can be initialized without engines
        # (will create default engine)
        await processor.initialize()
        
        assert processor._active_engine is not None
    
    def test_configuration_compatibility(self):
        """Test configuration compatibility with container."""
        # Test that configs can be created with container defaults
        config = DetectionConfig(
            confidence_threshold=0.5,
            classes_to_detect=["car", "truck", "bus", "motorcycle", "bicycle", "person"],
            target_latency_ms=50.0,
            batch_size=8
        )
        
        style = AnnotationStyleConfig(
            show_confidence=True,
            show_class_labels=True,
            box_thickness=2,
            font_scale=0.6
        )
        
        assert config.confidence_threshold == 0.5
        assert style.show_confidence is True
        
        # Test that processor can be created with these configs
        processor = MLAnnotationProcessor(config=config)
        assert processor.config == config
        assert processor.renderer.style_config.show_confidence is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])