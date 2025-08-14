"""Unit tests for Analytics DTOs.

Tests the DetectionResult DTO integration and ML output conversion utilities.
"""

import pytest
from datetime import UTC, datetime
from unittest.mock import Mock

from src.its_camera_ai.services.analytics_dtos import (
    BoundingBoxDTO,
    DetectionData,
    DetectionResultConverter,
    DetectionResultDTO,
    FrameMetadataDTO,
    TrafficPredictionDTO,
)


class TestBoundingBoxDTO:
    """Test BoundingBoxDTO functionality."""

    def test_bounding_box_properties(self):
        """Test bounding box property calculations."""
        bbox = BoundingBoxDTO(x1=10, y1=20, x2=50, y2=80, normalized=False)
        
        assert bbox.width == 40
        assert bbox.height == 60
        assert bbox.area == 2400
        assert bbox.center_x == 30
        assert bbox.center_y == 50

    def test_normalized_bounding_box(self):
        """Test normalized bounding box."""
        bbox = BoundingBoxDTO(x1=0.1, y1=0.2, x2=0.5, y2=0.8, normalized=True)
        
        assert bbox.width == 0.4
        assert bbox.height == 0.6
        assert bbox.normalized is True


class TestDetectionResultDTO:
    """Test DetectionResultDTO functionality."""

    def test_detection_result_creation(self):
        """Test DetectionResultDTO creation and properties."""
        bbox = BoundingBoxDTO(x1=10, y1=20, x2=50, y2=80, normalized=False)
        detection = DetectionResultDTO(
            detection_id="det_001",
            frame_id="frame_001",
            camera_id="cam_001",
            timestamp=datetime.now(UTC),
            track_id="track_001",
            class_id=0,
            class_name="car",
            confidence=0.95,
            bbox=bbox,
            attributes={"color": "red"},
            speed=50.0,
            vehicle_type="sedan",
        )
        
        assert detection.is_vehicle is True
        assert detection.is_high_confidence is True
        assert detection.is_reliable is True
        assert detection.class_name == "car"
        assert detection.speed == 50.0

    def test_detection_result_serialization(self):
        """Test DetectionResultDTO to_dict method."""
        bbox = BoundingBoxDTO(x1=10, y1=20, x2=50, y2=80, normalized=False)
        detection = DetectionResultDTO(
            detection_id="det_001",
            frame_id="frame_001",
            camera_id="cam_001",
            timestamp=datetime.now(UTC),
            track_id="track_001",
            class_id=0,
            class_name="car",
            confidence=0.95,
            bbox=bbox,
            attributes={"color": "red"},
        )
        
        result_dict = detection.to_dict()
        assert result_dict["detection_id"] == "det_001"
        assert result_dict["class_name"] == "car"
        assert result_dict["confidence"] == 0.95
        assert "bbox" in result_dict
        assert result_dict["bbox"]["x1"] == 10


class TestDetectionResultConverter:
    """Test DetectionResultConverter functionality."""

    def test_from_ml_output_basic(self):
        """Test basic ML output conversion."""
        frame_metadata = FrameMetadataDTO(
            frame_id="frame_001",
            camera_id="cam_001",
            timestamp=datetime.now(UTC),
            frame_number=1,
            width=1920,
            height=1080,
        )
        
        ml_detection = {
            "bbox": [100, 200, 300, 400],
            "class": "car",
            "confidence": 0.9,
            "track_id": 123,
            "speed": 60.0,
            "attributes": {"color": "blue"},
        }
        
        detection = DetectionResultConverter.from_ml_output(
            ml_detection, frame_metadata, "yolo_v1.0"
        )
        
        assert detection.class_name == "car"
        assert detection.confidence == 0.9
        assert detection.track_id == "123"
        assert detection.speed == 60.0
        assert detection.bbox.x1 == 100
        assert detection.bbox.x2 == 300
        assert detection.model_version == "yolo_v1.0"
        assert detection.attributes["color"] == "blue"

    def test_from_ml_output_normalized_coords(self):
        """Test ML output with normalized coordinates."""
        frame_metadata = FrameMetadataDTO(
            frame_id="frame_001",
            camera_id="cam_001",
            timestamp=datetime.now(UTC),
            frame_number=1,
            width=1920,
            height=1080,
        )
        
        ml_detection = {
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "class": "truck",
            "confidence": 0.85,
            "normalized": True,
        }
        
        detection = DetectionResultConverter.from_ml_output(
            ml_detection, frame_metadata
        )
        
        assert detection.bbox.normalized is True
        assert detection.bbox.x1 == 0.1
        assert detection.class_name == "truck"

    def test_to_detection_data(self):
        """Test conversion to DetectionData."""
        ml_output = {
            "frame": {
                "frame_id": "frame_001",
                "frame_number": 1,
                "width": 1920,
                "height": 1080,
            },
            "detections": [
                {
                    "bbox": [100, 200, 300, 400],
                    "class": "car",
                    "confidence": 0.9,
                },
                {
                    "bbox": [500, 600, 700, 800],
                    "class": "truck",
                    "confidence": 0.85,
                },
            ],
            "model_version": "yolo_v1.0",
        }
        
        detection_data = DetectionResultConverter.to_detection_data(
            ml_output, "cam_001"
        )
        
        assert detection_data.camera_id == "cam_001"
        assert len(detection_data.detections) == 2
        assert detection_data.vehicle_count == 2  # Both car and truck are vehicles
        assert detection_data.detections[0].class_name == "car"
        assert detection_data.detections[1].class_name == "truck"
        assert detection_data.frame_metadata.frame_id == "frame_001"


class TestDetectionData:
    """Test DetectionData functionality."""

    def test_detection_data_filtering(self):
        """Test DetectionData filtering methods."""
        bbox1 = BoundingBoxDTO(x1=10, y1=20, x2=50, y2=80, normalized=False)
        bbox2 = BoundingBoxDTO(x1=100, y1=200, x2=150, y2=280, normalized=False)
        
        detection1 = DetectionResultDTO(
            detection_id="det_001",
            frame_id="frame_001",
            camera_id="cam_001",
            timestamp=datetime.now(UTC),
            track_id="track_001",
            class_id=0,
            class_name="car",
            confidence=0.95,
            bbox=bbox1,
            attributes={},
            detection_quality=0.8,
        )
        
        detection2 = DetectionResultDTO(
            detection_id="det_002",
            frame_id="frame_001",
            camera_id="cam_001",
            timestamp=datetime.now(UTC),
            track_id="track_002",
            class_id=5,
            class_name="person",
            confidence=0.7,
            bbox=bbox2,
            attributes={},
            detection_quality=0.6,
        )
        
        detection_data = DetectionData(
            camera_id="cam_001",
            timestamp=datetime.now(UTC),
            frame_id="frame_001",
            detections=[detection1, detection2],
            vehicle_count=1,
        )
        
        # Test filtering methods
        vehicles = detection_data.vehicle_detections
        assert len(vehicles) == 1
        assert vehicles[0].class_name == "car"
        
        high_confidence = detection_data.high_confidence_detections
        assert len(high_confidence) == 1
        assert high_confidence[0].confidence == 0.95
        
        reliable = detection_data.reliable_detections
        assert len(reliable) == 1
        assert reliable[0].detection_id == "det_001"
        
        cars = detection_data.get_detections_by_class("car")
        assert len(cars) == 1
        assert cars[0].class_name == "car"


class TestTrafficPredictionDTO:
    """Test TrafficPredictionDTO functionality."""

    def test_traffic_prediction_creation(self):
        """Test TrafficPredictionDTO creation and properties."""
        prediction = TrafficPredictionDTO(
            camera_id="cam_001",
            prediction_timestamp=datetime.now(UTC),
            forecast_start=datetime.now(UTC),
            forecast_end=datetime.now(UTC),
            horizon_minutes=60,
            predicted_vehicle_count=25.5,
            confidence_interval={"lower": 20.0, "upper": 31.0, "mean": 25.5},
            model_version="traffic_rf_v2.1",
            model_accuracy=0.87,
            processing_time_ms=150.0,
            factors_considered=["historical_patterns", "time_of_day"],
        )
        
        assert prediction.confidence_lower == 20.0
        assert prediction.confidence_upper == 31.0
        assert prediction.is_high_confidence is True
        assert prediction.is_fallback is False
        assert prediction.horizon_minutes == 60

    def test_traffic_prediction_fallback(self):
        """Test fallback traffic prediction."""
        prediction = TrafficPredictionDTO(
            camera_id="cam_001",
            prediction_timestamp=datetime.now(UTC),
            forecast_start=datetime.now(UTC),
            forecast_end=datetime.now(UTC),
            horizon_minutes=60,
            predicted_vehicle_count=10.0,
            confidence_interval={"lower": 5.0, "upper": 15.0, "mean": 10.0},
            model_version="fallback_v1.0",
            model_accuracy=0.5,
            processing_time_ms=0.0,
            factors_considered=["fallback"],
            is_fallback=True,
        )
        
        assert prediction.is_high_confidence is False
        assert prediction.is_fallback is True
        assert prediction.model_accuracy == 0.5


if __name__ == "__main__":
    pytest.main([__file__])
