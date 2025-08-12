"""
Core Computer Vision Engine for ITS Camera AI Traffic Monitoring System.

This module implements the primary computer vision pipeline for real-time vehicle detection
with YOLO11 nano model optimization, focusing on <100ms latency and >95% accuracy requirements
for traffic monitoring at 30 FPS across multiple camera streams.

Key Features:
- YOLO11 nano model optimization for real-time inference
- TensorRT/ONNX acceleration with automatic CPU fallback
- Dynamic batch processing for optimal GPU utilization
- Multi-camera stream support with concurrent processing
- Memory-efficient frame preprocessing pipeline
- Performance monitoring with drift detection
- Model versioning and A/B testing integration

Architecture:
- CoreVisionEngine: Main orchestrator for vision processing
- ModelManager: Handles model loading, optimization, and versioning
- FrameProcessor: Optimized preprocessing pipeline
- InferenceEngine: High-performance prediction engine
- PostProcessor: Detection formatting and filtering
- PerformanceMonitor: Real-time metrics and monitoring
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .inference_optimizer import (
    DetectionResult,
    InferenceConfig,
    ModelType,
    OptimizationBackend,
    OptimizedInferenceEngine,
)

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Vision processing stages."""

    PREPROCESSING = "preprocessing"
    INFERENCE = "inference"
    POSTPROCESSING = "postprocessing"
    COMPLETE = "complete"


class VehicleClass(Enum):
    """Vehicle classification for traffic monitoring."""

    CAR = 2  # COCO class ID
    MOTORCYCLE = 3
    BUS = 5
    TRUCK = 7
    BICYCLE = 1
    EMERGENCY = 99  # Custom class for emergency vehicles


@dataclass
class VisionConfig:
    """Configuration for Core Computer Vision Engine."""

    # Model Configuration
    model_type: ModelType = ModelType.NANO
    model_path: Path | None = None
    optimization_backend: OptimizationBackend = OptimizationBackend.TENSORRT

    # Performance Requirements
    target_latency_ms: int = 100
    target_accuracy: float = 0.95
    target_fps: int = 30
    max_concurrent_cameras: int = 10

    # Processing Configuration
    input_resolution: tuple[int, int] = (640, 640)
    batch_size: int = 8
    max_batch_size: int = 32
    batch_timeout_ms: int = 10

    # Quality Thresholds
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    min_detection_size: int = 32
    max_detections_per_frame: int = 100

    # Hardware Configuration
    device_ids: list[int] | None = None
    memory_fraction: float = 0.8
    enable_cpu_fallback: bool = True

    # Monitoring
    enable_performance_monitoring: bool = True
    metrics_collection_interval: int = 60
    drift_detection_enabled: bool = True

    def __post_init__(self) -> None:
        if self.device_ids is None:
            self.device_ids = [0] if torch.cuda.is_available() else []

        if self.model_path is None:
            self.model_path = Path(f"models/{self.model_type.value}")


@dataclass
class VisionResult:
    """Structured result from computer vision processing."""

    # Detection Data
    detections: list[dict[str, Any]]
    detection_count: int
    vehicle_counts: dict[VehicleClass, int]

    # Frame Metadata
    frame_id: str
    camera_id: str
    timestamp: float
    frame_resolution: tuple[int, int]

    # Performance Metrics
    preprocessing_time_ms: float
    inference_time_ms: float
    postprocessing_time_ms: float
    total_processing_time_ms: float

    # Quality Metrics
    avg_confidence: float
    detection_density: float  # detections per square pixel
    processing_quality_score: float

    # System Metrics
    gpu_memory_used_mb: float
    cpu_utilization: float
    batch_size_used: int


class ModelManager:
    """Manages YOLO11 model loading, optimization, and versioning."""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.models: dict[str, Any] = {}
        self.optimization_cache: dict[str, Path] = {}
        self.model_metadata: dict[str, dict[str, Any]] = {}

        self.inference_engine = OptimizedInferenceEngine(
            InferenceConfig(
                model_type=config.model_type,
                backend=config.optimization_backend,
                precision="fp16" if torch.cuda.is_available() else "fp32",
                batch_size=config.batch_size,
                max_batch_size=config.max_batch_size,
                batch_timeout_ms=config.batch_timeout_ms,
                input_size=config.input_resolution,
                device_ids=config.device_ids,
                memory_fraction=config.memory_fraction,
                conf_threshold=config.confidence_threshold,
                iou_threshold=config.iou_threshold,
                max_detections=config.max_detections_per_frame,
            )
        )

        logger.info("Model manager initialized")

    async def initialize(self, model_path: Path | None = None) -> None:
        """Initialize the model manager with primary model."""
        model_path = model_path or self.config.model_path

        if not model_path or not model_path.exists():
            # Download YOLO11 nano if not available
            model_path = await self._ensure_yolo11_nano()

        # Initialize inference engine
        await self.inference_engine.initialize(model_path)

        # Store model metadata
        self.model_metadata["primary"] = {
            "path": str(model_path),
            "type": self.config.model_type.value,
            "loaded_at": time.time(),
            "optimization_backend": self.config.optimization_backend.value,
        }

        logger.info(f"Model manager initialized with {model_path}")

    async def _ensure_yolo11_nano(self) -> Path:
        """Ensure YOLO11 nano model is available."""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / "yolo11n.pt"

        if not model_path.exists():
            logger.info("Downloading YOLO11 nano model...")
            # YOLO11 will auto-download if not present
            model = YOLO("yolo11n.pt")
            # Move to our models directory
            import shutil

            yolo_cache_path = (
                model.model_path
                if hasattr(model, "model_path")
                else Path.home() / ".cache/ultralytics/yolo11n.pt"
            )
            if yolo_cache_path.exists():
                shutil.copy2(yolo_cache_path, model_path)

        return model_path

    async def predict_batch(
        self,
        frames: list[np.ndarray],
        frame_ids: list[str],
        camera_ids: list[str],
    ) -> list[DetectionResult]:
        """Perform batch inference using optimized engine."""
        return await self.inference_engine.predict_batch(frames, frame_ids, camera_ids)

    async def predict_single(
        self,
        frame: np.ndarray,
        frame_id: str,
        camera_id: str,
    ) -> DetectionResult:
        """Perform single frame inference."""
        return await self.inference_engine.predict_single(frame, frame_id, camera_id)

    def get_model_info(self) -> dict[str, Any]:
        """Get current model information."""
        return {
            "models": self.model_metadata,
            "performance": self.inference_engine.get_performance_stats(),
            "optimization_backend": self.config.optimization_backend.value,
        }

    async def cleanup(self) -> None:
        """Clean up model resources."""
        await self.inference_engine.cleanup()


class FrameProcessor:
    """Optimized frame preprocessing pipeline for YOLO11 models."""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.input_size = config.input_resolution

        # Preprocessing statistics
        self.processing_times: list[float] = []
        self.quality_scores: list[float] = []

    def preprocess_frame(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Optimized frame preprocessing for YOLO11.

        Returns:
            - preprocessed_frame: Ready for inference
            - metadata: Preprocessing information
        """
        start_time = time.time()

        # Input validation
        if frame is None or frame.size == 0:
            raise ValueError("Invalid input frame")

        original_shape = frame.shape[:2]  # (height, width)

        # Calculate optimal scaling to maintain aspect ratio
        scale = min(
            self.input_size[0] / original_shape[0],
            self.input_size[1] / original_shape[1],
        )

        # Resize with aspect ratio preservation
        new_size = (int(original_shape[1] * scale), int(original_shape[0] * scale))
        if scale != 1.0:
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)

        # Letterboxing with gray padding
        processed_frame = np.full(
            (self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8
        )

        # Calculate padding offsets
        y_offset = (self.input_size[0] - new_size[1]) // 2
        x_offset = (self.input_size[1] - new_size[0]) // 2

        # Place resized frame in padded canvas
        processed_frame[
            y_offset : y_offset + new_size[1], x_offset : x_offset + new_size[0]
        ] = frame

        # Calculate quality metrics
        quality_score = self._calculate_quality_score(processed_frame)

        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)
        self.quality_scores.append(quality_score)

        # Keep only recent statistics
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-500:]
            self.quality_scores = self.quality_scores[-500:]

        metadata = {
            "original_shape": original_shape,
            "scale_factor": scale,
            "padding": (x_offset, y_offset),
            "quality_score": quality_score,
            "processing_time_ms": processing_time,
        }

        return processed_frame, metadata

    def preprocess_batch(
        self, frames: list[np.ndarray]
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Batch preprocessing for optimal performance."""
        processed_frames = []
        metadata_list = []

        for frame in frames:
            processed_frame, metadata = self.preprocess_frame(frame)
            processed_frames.append(processed_frame)
            metadata_list.append(metadata)

        # Stack into batch tensor
        batch_array = np.stack(processed_frames, axis=0)

        return batch_array, metadata_list

    def _calculate_quality_score(self, frame: np.ndarray) -> float:
        """Calculate frame quality score for monitoring."""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 1000.0)

        # Brightness analysis
        mean_brightness = np.mean(gray)
        brightness_score = 1.0 if 50 <= mean_brightness <= 200 else 0.7

        # Contrast analysis
        contrast = np.std(gray)
        contrast_score = min(1.0, contrast / 50.0)

        # Combined quality score (weighted average)
        quality_score = blur_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3

        return quality_score

    def get_preprocessing_stats(self) -> dict[str, float]:
        """Get preprocessing performance statistics."""
        if not self.processing_times:
            return {}

        return {
            "avg_processing_time_ms": np.mean(self.processing_times),
            "p95_processing_time_ms": np.percentile(self.processing_times, 95),
            "avg_quality_score": np.mean(self.quality_scores),
            "min_quality_score": np.min(self.quality_scores),
        }


class PostProcessor:
    """Post-processing pipeline for detection results and traffic analysis."""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.vehicle_class_mapping = {
            1: VehicleClass.BICYCLE,
            2: VehicleClass.CAR,
            3: VehicleClass.MOTORCYCLE,
            5: VehicleClass.BUS,
            7: VehicleClass.TRUCK,
        }

        # Traffic analysis parameters
        self.emergency_keywords = ["ambulance", "police", "fire"]
        self.size_thresholds = {
            VehicleClass.BICYCLE: (20, 200),
            VehicleClass.CAR: (50, 500),
            VehicleClass.MOTORCYCLE: (30, 150),
            VehicleClass.BUS: (100, 800),
            VehicleClass.TRUCK: (80, 600),
        }

    def process_detections(
        self,
        detection_result: DetectionResult,
        preprocessing_metadata: dict[str, Any],
    ) -> VisionResult:
        """
        Process raw detections into structured traffic analysis result.
        """
        start_time = time.time()

        # Extract vehicle detections only
        vehicle_detections = self._filter_vehicle_detections(detection_result)

        # Apply size-based filtering
        filtered_detections = self._apply_size_filtering(vehicle_detections)

        # Convert to traffic analysis format
        structured_detections = self._structure_detections(
            filtered_detections, preprocessing_metadata
        )

        # Calculate vehicle counts by type
        vehicle_counts = self._count_vehicles_by_type(structured_detections)

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            structured_detections, detection_result
        )

        processing_time = (time.time() - start_time) * 1000

        # Create comprehensive result
        vision_result = VisionResult(
            detections=structured_detections,
            detection_count=len(structured_detections),
            vehicle_counts=vehicle_counts,
            frame_id=detection_result.frame_id,
            camera_id=detection_result.camera_id,
            timestamp=detection_result.timestamp,
            frame_resolution=self.config.input_resolution,
            preprocessing_time_ms=detection_result.preprocessing_time_ms,
            inference_time_ms=detection_result.inference_time_ms,
            postprocessing_time_ms=processing_time,
            total_processing_time_ms=detection_result.total_time_ms + processing_time,
            avg_confidence=quality_metrics["avg_confidence"],
            detection_density=quality_metrics["detection_density"],
            processing_quality_score=quality_metrics["processing_quality"],
            gpu_memory_used_mb=detection_result.gpu_memory_used_mb,
            cpu_utilization=0.0,  # TODO: Implement CPU monitoring
            batch_size_used=1,  # TODO: Track actual batch size
        )

        return vision_result

    def _filter_vehicle_detections(
        self, detection_result: DetectionResult
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter detections to keep only vehicles."""
        vehicle_classes = list(self.vehicle_class_mapping.keys())
        vehicle_mask = np.isin(detection_result.classes, vehicle_classes)

        return (
            detection_result.boxes[vehicle_mask],
            detection_result.scores[vehicle_mask],
            detection_result.classes[vehicle_mask],
        )

    def _apply_size_filtering(
        self, detections: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply size-based filtering to remove unrealistic detections."""
        boxes, scores, classes = detections

        if len(boxes) == 0:
            return detections

        # Calculate box areas
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights

        # Apply size thresholds per vehicle type
        valid_mask = np.zeros(len(boxes), dtype=bool)

        for i, vehicle_class in enumerate(classes):
            if vehicle_class in self.vehicle_class_mapping:
                vehicle_type = self.vehicle_class_mapping[vehicle_class]
                min_area, max_area = self.size_thresholds.get(
                    vehicle_type, (self.config.min_detection_size, float("inf"))
                )
                valid_mask[i] = min_area <= areas[i] <= max_area

        return (
            boxes[valid_mask],
            scores[valid_mask],
            classes[valid_mask],
        )

    def _structure_detections(
        self,
        detections: tuple[np.ndarray, np.ndarray, np.ndarray],
        preprocessing_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Convert detections to structured format with traffic analysis."""
        boxes, scores, classes = detections
        structured = []

        for i in range(len(boxes)):
            box = boxes[i]
            score = float(scores[i])
            vehicle_class = int(classes[i])

            # Convert to original image coordinates
            original_box = self._convert_to_original_coordinates(
                box, preprocessing_metadata
            )

            vehicle_type = self.vehicle_class_mapping.get(
                vehicle_class, VehicleClass.CAR
            )

            detection = {
                "id": f"det_{i}_{int(time.time() * 1000)}",
                "vehicle_type": vehicle_type.name.lower(),
                "confidence": score,
                "bbox": {
                    "x1": float(original_box[0]),
                    "y1": float(original_box[1]),
                    "x2": float(original_box[2]),
                    "y2": float(original_box[3]),
                    "width": float(original_box[2] - original_box[0]),
                    "height": float(original_box[3] - original_box[1]),
                },
                "area": float(
                    (original_box[2] - original_box[0])
                    * (original_box[3] - original_box[1])
                ),
                "processed_bbox": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                },
                "attributes": {
                    "size_category": self._categorize_vehicle_size(box, vehicle_type),
                    "position": self._analyze_position(box),
                    "quality_score": min(
                        1.0, score * preprocessing_metadata.get("quality_score", 1.0)
                    ),
                },
            }

            structured.append(detection)

        return structured

    def _convert_to_original_coordinates(
        self,
        box: np.ndarray,
        metadata: dict[str, Any],
    ) -> np.ndarray:
        """Convert processed coordinates back to original image coordinates."""
        scale_factor = metadata.get("scale_factor", 1.0)
        x_offset, y_offset = metadata.get("padding", (0, 0))

        # Remove padding offset
        adjusted_box = box.copy()
        adjusted_box[0] -= x_offset  # x1
        adjusted_box[1] -= y_offset  # y1
        adjusted_box[2] -= x_offset  # x2
        adjusted_box[3] -= y_offset  # y2

        # Scale back to original coordinates
        if scale_factor != 1.0:
            adjusted_box /= scale_factor

        return adjusted_box

    def _categorize_vehicle_size(
        self, box: np.ndarray, vehicle_type: VehicleClass
    ) -> str:
        """Categorize vehicle size for traffic analysis."""
        area = (box[2] - box[0]) * (box[3] - box[1])

        if vehicle_type in [VehicleClass.BICYCLE, VehicleClass.MOTORCYCLE]:
            return "small"
        elif vehicle_type == VehicleClass.CAR:
            return "medium" if area > 5000 else "small"
        elif vehicle_type in [VehicleClass.BUS, VehicleClass.TRUCK]:
            return "large"

        return "medium"

    def _analyze_position(self, box: np.ndarray) -> dict[str, Any]:
        """Analyze vehicle position in frame for traffic flow analysis."""
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2

        # Normalize to frame coordinates
        norm_x = center_x / self.config.input_resolution[1]
        norm_y = center_y / self.config.input_resolution[0]

        return {
            "center_x": float(center_x),
            "center_y": float(center_y),
            "normalized_x": float(norm_x),
            "normalized_y": float(norm_y),
            "lane_position": "left" if norm_x < 0.5 else "right",
            "depth_zone": (
                "near" if norm_y > 0.7 else "far" if norm_y < 0.3 else "middle"
            ),
        }

    def _count_vehicles_by_type(
        self, detections: list[dict[str, Any]]
    ) -> dict[VehicleClass, int]:
        """Count vehicles by type for traffic analysis."""
        counts = dict.fromkeys(VehicleClass, 0)

        for detection in detections:
            vehicle_type_str = detection["vehicle_type"]
            # Convert string back to enum
            for vehicle_type in VehicleClass:
                if vehicle_type.name.lower() == vehicle_type_str:
                    counts[vehicle_type] += 1
                    break

        return counts

    def _calculate_quality_metrics(
        self,
        detections: list[dict[str, Any]],
        detection_result: DetectionResult,
    ) -> dict[str, float]:
        """Calculate detection and processing quality metrics."""
        if not detections:
            return {
                "avg_confidence": 0.0,
                "detection_density": 0.0,
                "processing_quality": 0.5,
            }

        # Average confidence
        confidences = [det["confidence"] for det in detections]
        avg_confidence = np.mean(confidences)

        # Detection density (detections per 1000 pixels)
        frame_area = self.config.input_resolution[0] * self.config.input_resolution[1]
        detection_density = (len(detections) / frame_area) * 1000

        # Processing quality score (composite metric)
        processing_quality = min(
            1.0,
            (
                avg_confidence * 0.4
                + min(1.0, detection_density / 0.1) * 0.3  # Optimal density around 0.1
                + min(
                    1.0,
                    detection_result.inference_time_ms / self.config.target_latency_ms,
                )
                * 0.3
            ),
        )

        return {
            "avg_confidence": float(avg_confidence),
            "detection_density": float(detection_density),
            "processing_quality": float(processing_quality),
        }


class PerformanceMonitor:
    """Real-time performance monitoring and alerting system."""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.enabled = config.enable_performance_monitoring

        # Performance metrics storage
        self.latency_history: list[float] = []
        self.throughput_history: list[float] = []
        self.accuracy_history: list[float] = []
        self.memory_usage_history: list[float] = []

        # Alert thresholds
        self.latency_threshold = config.target_latency_ms * 1.2  # 20% tolerance
        self.accuracy_threshold = config.target_accuracy * 0.95  # 5% tolerance
        self.memory_threshold = config.memory_fraction * 0.9  # 90% of allocated

        # Monitoring state
        self.last_metrics_time = time.time()
        self.total_processed_frames = 0
        self.alerts_generated = 0

        logger.info("Performance monitor initialized")

    def record_processing(self, vision_result: VisionResult) -> None:
        """Record processing metrics from vision result."""
        if not self.enabled:
            return

        # Record latency
        self.latency_history.append(vision_result.total_processing_time_ms)

        # Record quality/accuracy proxy
        self.accuracy_history.append(vision_result.processing_quality_score)

        # Record memory usage
        self.memory_usage_history.append(vision_result.gpu_memory_used_mb)

        # Update counters
        self.total_processed_frames += 1

        # Limit history size
        max_history = 1000
        if len(self.latency_history) > max_history:
            self.latency_history = self.latency_history[-max_history // 2 :]
            self.accuracy_history = self.accuracy_history[-max_history // 2 :]
            self.memory_usage_history = self.memory_usage_history[-max_history // 2 :]

    def calculate_throughput(self) -> float:
        """Calculate current throughput in FPS."""
        current_time = time.time()
        time_window = current_time - self.last_metrics_time

        if time_window > 0:
            recent_frame_count = min(100, self.total_processed_frames)
            throughput = recent_frame_count / time_window
            self.throughput_history.append(throughput)

            # Reset for next calculation
            self.last_metrics_time = current_time

            return throughput

        return 0.0

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        if not self.enabled or not self.latency_history:
            return {}

        current_throughput = self.calculate_throughput()

        return {
            "latency": {
                "avg_ms": np.mean(self.latency_history),
                "p50_ms": np.percentile(self.latency_history, 50),
                "p95_ms": np.percentile(self.latency_history, 95),
                "p99_ms": np.percentile(self.latency_history, 99),
                "max_ms": np.max(self.latency_history),
                "target_ms": self.config.target_latency_ms,
                "meets_target": np.percentile(self.latency_history, 95)
                <= self.config.target_latency_ms,
            },
            "throughput": {
                "current_fps": current_throughput,
                "avg_fps": (
                    np.mean(self.throughput_history) if self.throughput_history else 0.0
                ),
                "target_fps": self.config.target_fps,
                "meets_target": current_throughput >= self.config.target_fps * 0.9,
            },
            "quality": {
                "avg_score": np.mean(self.accuracy_history),
                "min_score": np.min(self.accuracy_history),
                "target_score": self.config.target_accuracy,
                "meets_target": np.mean(self.accuracy_history)
                >= self.config.target_accuracy * 0.95,
            },
            "memory": {
                "avg_usage_mb": np.mean(self.memory_usage_history),
                "max_usage_mb": np.max(self.memory_usage_history),
                "threshold_mb": self.memory_threshold,
            },
            "system": {
                "total_frames_processed": self.total_processed_frames,
                "alerts_generated": self.alerts_generated,
                "monitoring_enabled": self.enabled,
            },
        }

    def check_performance_alerts(self) -> list[dict[str, Any]]:
        """Check for performance issues and generate alerts."""
        if not self.enabled or len(self.latency_history) < 10:
            return []

        alerts = []

        # Latency alerts
        p95_latency = np.percentile(self.latency_history, 95)
        if p95_latency > self.latency_threshold:
            alerts.append(
                {
                    "type": "latency_high",
                    "severity": (
                        "warning"
                        if p95_latency < self.latency_threshold * 1.5
                        else "critical"
                    ),
                    "message": f"P95 latency ({p95_latency:.1f}ms) exceeds threshold ({self.latency_threshold:.1f}ms)",
                    "current_value": p95_latency,
                    "threshold": self.latency_threshold,
                    "timestamp": time.time(),
                }
            )

        # Quality alerts
        if self.accuracy_history:
            recent_quality = np.mean(self.accuracy_history[-50:])
            if recent_quality < self.accuracy_threshold:
                alerts.append(
                    {
                        "type": "quality_low",
                        "severity": "warning",
                        "message": f"Processing quality ({recent_quality:.3f}) below threshold ({self.accuracy_threshold:.3f})",
                        "current_value": recent_quality,
                        "threshold": self.accuracy_threshold,
                        "timestamp": time.time(),
                    }
                )

        # Memory alerts
        if self.memory_usage_history:
            max_memory = np.max(self.memory_usage_history[-20:])
            if max_memory > self.memory_threshold:
                alerts.append(
                    {
                        "type": "memory_high",
                        "severity": (
                            "critical"
                            if max_memory > self.memory_threshold * 1.1
                            else "warning"
                        ),
                        "message": f"GPU memory usage ({max_memory:.1f}MB) high",
                        "current_value": max_memory,
                        "threshold": self.memory_threshold,
                        "timestamp": time.time(),
                    }
                )

        self.alerts_generated += len(alerts)
        return alerts

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.latency_history.clear()
        self.throughput_history.clear()
        self.accuracy_history.clear()
        self.memory_usage_history.clear()
        self.total_processed_frames = 0
        self.alerts_generated = 0
        self.last_metrics_time = time.time()


class CoreVisionEngine:
    """
    Main orchestrator for Core Computer Vision Engine.

    Provides high-level interface for real-time traffic monitoring with
    YOLO11 nano model optimization and comprehensive performance monitoring.
    """

    def __init__(self, config: VisionConfig | None = None):
        self.config = config or VisionConfig()

        # Initialize components
        self.model_manager = ModelManager(self.config)
        self.frame_processor = FrameProcessor(self.config)
        self.post_processor = PostProcessor(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)

        # Engine state
        self.initialized = False
        self.processing_stats = {
            "total_frames": 0,
            "successful_frames": 0,
            "failed_frames": 0,
            "start_time": time.time(),
        }

        logger.info("Core Vision Engine created")

    async def initialize(self, model_path: Path | None = None) -> None:
        """Initialize the complete vision engine."""
        logger.info("Initializing Core Vision Engine...")

        # Initialize model manager
        await self.model_manager.initialize(model_path)

        # Validate configuration
        self._validate_configuration()

        self.initialized = True
        logger.info("Core Vision Engine initialized successfully")

    def _validate_configuration(self) -> None:
        """Validate engine configuration."""
        if self.config.target_latency_ms <= 0:
            raise ValueError("Target latency must be positive")

        if not (0.0 < self.config.target_accuracy <= 1.0):
            raise ValueError("Target accuracy must be between 0 and 1")

        if self.config.max_concurrent_cameras <= 0:
            raise ValueError("Max concurrent cameras must be positive")

        logger.info("Configuration validated successfully")

    async def process_frame(
        self,
        frame: np.ndarray,
        frame_id: str,
        camera_id: str,
    ) -> VisionResult:
        """
        Process single frame for vehicle detection.

        Args:
            frame: Input frame as numpy array (H, W, C)
            frame_id: Unique frame identifier
            camera_id: Camera identifier

        Returns:
            VisionResult with detections and performance metrics
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # Stage 1: Preprocessing
            processed_frame, preprocessing_metadata = (
                self.frame_processor.preprocess_frame(frame)
            )

            # Stage 2: Inference
            detection_result = await self.model_manager.predict_single(
                processed_frame, frame_id, camera_id
            )

            # Stage 3: Post-processing
            vision_result = self.post_processor.process_detections(
                detection_result, preprocessing_metadata
            )

            # Record performance metrics
            self.performance_monitor.record_processing(vision_result)

            # Update statistics
            self.processing_stats["total_frames"] += 1
            self.processing_stats["successful_frames"] += 1

            return vision_result

        except Exception as e:
            self.processing_stats["total_frames"] += 1
            self.processing_stats["failed_frames"] += 1

            logger.error(f"Frame processing failed for {frame_id}: {e}")

            # Return empty result with error information
            return self._create_error_result(frame_id, camera_id, str(e), start_time)

    async def process_batch(
        self,
        frames: list[np.ndarray],
        frame_ids: list[str],
        camera_ids: list[str],
    ) -> list[VisionResult]:
        """
        Process batch of frames for optimal GPU utilization.

        Args:
            frames: List of input frames
            frame_ids: List of frame identifiers
            camera_ids: List of camera identifiers

        Returns:
            List of VisionResult objects
        """
        if not self.initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        if len(frames) != len(frame_ids) or len(frames) != len(camera_ids):
            raise ValueError("Frames, frame_ids, and camera_ids must have same length")

        if len(frames) > self.config.max_batch_size:
            logger.warning(
                f"Batch size {len(frames)} exceeds max {self.config.max_batch_size}"
            )
            # Process in chunks
            return await self._process_batch_chunked(frames, frame_ids, camera_ids)

        try:
            # Batch preprocessing
            processed_frames, preprocessing_metadata = (
                self.frame_processor.preprocess_batch(frames)
            )

            # Batch inference
            detection_results = await self.model_manager.predict_batch(
                [processed_frames[i] for i in range(len(processed_frames))],
                frame_ids,
                camera_ids,
            )

            # Post-process each result
            vision_results = []
            for i, detection_result in enumerate(detection_results):
                vision_result = self.post_processor.process_detections(
                    detection_result, preprocessing_metadata[i]
                )
                vision_results.append(vision_result)

                # Record metrics
                self.performance_monitor.record_processing(vision_result)

            # Update statistics
            self.processing_stats["total_frames"] += len(frames)
            self.processing_stats["successful_frames"] += len(vision_results)

            return vision_results

        except Exception as e:
            self.processing_stats["total_frames"] += len(frames)
            self.processing_stats["failed_frames"] += len(frames)

            logger.error(f"Batch processing failed: {e}")

            # Return error results for all frames
            start_time = time.time()
            return [
                self._create_error_result(frame_id, camera_id, str(e), start_time)
                for frame_id, camera_id in zip(frame_ids, camera_ids, strict=False)
            ]

    async def _process_batch_chunked(
        self,
        frames: list[np.ndarray],
        frame_ids: list[str],
        camera_ids: list[str],
    ) -> list[VisionResult]:
        """Process large batch in chunks."""
        chunk_size = self.config.max_batch_size
        all_results = []

        for i in range(0, len(frames), chunk_size):
            chunk_frames = frames[i : i + chunk_size]
            chunk_frame_ids = frame_ids[i : i + chunk_size]
            chunk_camera_ids = camera_ids[i : i + chunk_size]

            chunk_results = await self.process_batch(
                chunk_frames, chunk_frame_ids, chunk_camera_ids
            )
            all_results.extend(chunk_results)

        return all_results

    def _create_error_result(
        self,
        frame_id: str,
        camera_id: str,
        error_message: str,
        start_time: float,
    ) -> VisionResult:
        """Create error result for failed processing."""
        return VisionResult(
            detections=[],
            detection_count=0,
            vehicle_counts=dict.fromkeys(VehicleClass, 0),
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=time.time(),
            frame_resolution=self.config.input_resolution,
            preprocessing_time_ms=0.0,
            inference_time_ms=0.0,
            postprocessing_time_ms=0.0,
            total_processing_time_ms=(time.time() - start_time) * 1000,
            avg_confidence=0.0,
            detection_density=0.0,
            processing_quality_score=0.0,
            gpu_memory_used_mb=0.0,
            cpu_utilization=0.0,
            batch_size_used=0,
        )

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        # Get component metrics
        model_info = self.model_manager.get_model_info()
        preprocessing_stats = self.frame_processor.get_preprocessing_stats()
        performance_metrics = self.performance_monitor.get_performance_metrics()

        # Calculate uptime
        uptime_seconds = time.time() - self.processing_stats["start_time"]

        # Combine all metrics
        return {
            "engine": {
                "initialized": self.initialized,
                "uptime_seconds": uptime_seconds,
                "total_frames_processed": self.processing_stats["total_frames"],
                "success_rate": (
                    self.processing_stats["successful_frames"]
                    / max(1, self.processing_stats["total_frames"])
                ),
                "config": {
                    "model_type": self.config.model_type.value,
                    "target_latency_ms": self.config.target_latency_ms,
                    "target_accuracy": self.config.target_accuracy,
                    "max_concurrent_cameras": self.config.max_concurrent_cameras,
                },
            },
            "model": model_info,
            "preprocessing": preprocessing_stats,
            "performance": performance_metrics,
        }

    def get_health_status(self) -> dict[str, Any]:
        """Get engine health status and alerts."""
        alerts = self.performance_monitor.check_performance_alerts()
        metrics = self.get_performance_metrics()

        # Determine overall health
        health_score = 1.0
        if metrics.get("performance", {}).get("latency", {}).get("meets_target", True):
            health_score -= 0.2
        if (
            metrics.get("performance", {})
            .get("throughput", {})
            .get("meets_target", True)
        ):
            health_score -= 0.3
        if metrics.get("performance", {}).get("quality", {}).get("meets_target", True):
            health_score -= 0.3

        critical_alerts = [a for a in alerts if a["severity"] == "critical"]
        warning_alerts = [a for a in alerts if a["severity"] == "warning"]

        if critical_alerts:
            health_score -= 0.5
        if warning_alerts:
            health_score -= 0.2

        health_score = max(0.0, health_score)

        return {
            "health_score": health_score,
            "status": (
                "healthy"
                if health_score > 0.8
                else "warning" if health_score > 0.5 else "critical"
            ),
            "alerts": {
                "total": len(alerts),
                "critical": len(critical_alerts),
                "warning": len(warning_alerts),
                "recent": alerts[-5:] if alerts else [],
            },
            "requirements_met": {
                "latency": metrics.get("performance", {})
                .get("latency", {})
                .get("meets_target", False),
                "throughput": metrics.get("performance", {})
                .get("throughput", {})
                .get("meets_target", False),
                "accuracy": metrics.get("performance", {})
                .get("quality", {})
                .get("meets_target", False),
            },
            "timestamp": time.time(),
        }

    async def cleanup(self) -> None:
        """Clean up engine resources."""
        logger.info("Cleaning up Core Vision Engine...")

        await self.model_manager.cleanup()
        self.performance_monitor.reset_metrics()

        self.initialized = False
        logger.info("Core Vision Engine cleanup completed")


# Factory Functions and Utilities


def create_optimal_config(
    deployment_scenario: str = "edge",
    available_memory_gb: float = 4.0,
    target_cameras: int = 4,
) -> VisionConfig:
    """
    Create optimal configuration based on deployment scenario.

    Scenarios:
    - edge: Low-power edge deployment (NANO model, small batches)
    - cloud: High-performance cloud deployment (SMALL/MEDIUM model, large batches)
    - production: Balanced production deployment (SMALL model, optimized settings)
    """

    if deployment_scenario == "edge":
        return VisionConfig(
            model_type=ModelType.NANO,
            target_latency_ms=50,
            target_accuracy=0.90,
            batch_size=2,
            max_batch_size=4,
            max_concurrent_cameras=min(2, target_cameras),
            memory_fraction=0.6,
            enable_cpu_fallback=True,
        )

    elif deployment_scenario == "cloud":
        return VisionConfig(
            model_type=(
                ModelType.MEDIUM if available_memory_gb >= 8 else ModelType.SMALL
            ),
            target_latency_ms=80,
            target_accuracy=0.95,
            batch_size=16 if available_memory_gb >= 8 else 8,
            max_batch_size=32 if available_memory_gb >= 16 else 16,
            max_concurrent_cameras=target_cameras,
            memory_fraction=0.8,
            enable_cpu_fallback=False,
        )

    else:  # production
        return VisionConfig(
            model_type=ModelType.SMALL,
            target_latency_ms=100,
            target_accuracy=0.95,
            batch_size=8 if available_memory_gb >= 6 else 4,
            max_batch_size=16 if available_memory_gb >= 8 else 8,
            max_concurrent_cameras=min(10, target_cameras),
            memory_fraction=0.7,
            enable_cpu_fallback=True,
        )


async def benchmark_engine(
    config: VisionConfig,
    num_frames: int = 100,
    frame_size: tuple[int, int] = (640, 640),
) -> dict[str, Any]:
    """
    Benchmark Core Vision Engine performance.

    Returns comprehensive performance analysis including latency,
    throughput, accuracy, and resource utilization metrics.
    """

    logger.info(f"Starting benchmark with {num_frames} frames...")

    # Create engine
    engine = CoreVisionEngine(config)
    await engine.initialize()

    # Generate test frames
    test_frames = []
    frame_ids = []
    camera_ids = []

    for i in range(num_frames):
        # Create synthetic traffic scene
        frame = np.random.randint(0, 255, (*frame_size, 3), dtype=np.uint8)
        test_frames.append(frame)
        frame_ids.append(f"bench_frame_{i}")
        camera_ids.append(f"bench_camera_{i % 4}")  # 4 virtual cameras

    # Warmup
    warmup_frames = 10
    for i in range(min(warmup_frames, num_frames)):
        await engine.process_frame(test_frames[i], frame_ids[i], camera_ids[i])

    # Benchmark single frame processing
    single_frame_times = []
    single_frame_start = time.time()

    for i in range(warmup_frames, min(warmup_frames + 50, num_frames)):
        start = time.time()
        result = await engine.process_frame(test_frames[i], frame_ids[i], camera_ids[i])
        single_frame_times.append((time.time() - start) * 1000)

    single_frame_duration = time.time() - single_frame_start

    # Benchmark batch processing
    batch_size = min(config.batch_size, num_frames - warmup_frames - 50)
    batch_start_idx = warmup_frames + 50

    if batch_start_idx + batch_size < num_frames:
        batch_frames = test_frames[batch_start_idx : batch_start_idx + batch_size]
        batch_frame_ids = frame_ids[batch_start_idx : batch_start_idx + batch_size]
        batch_camera_ids = camera_ids[batch_start_idx : batch_start_idx + batch_size]

        batch_start = time.time()
        batch_results = await engine.process_batch(
            batch_frames, batch_frame_ids, batch_camera_ids
        )
        batch_duration = time.time() - batch_start
        batch_per_frame = (batch_duration / batch_size) * 1000
    else:
        batch_duration = 0
        batch_per_frame = 0
        batch_results = []

    # Get final metrics
    performance_metrics = engine.get_performance_metrics()
    health_status = engine.get_health_status()

    # Cleanup
    await engine.cleanup()

    # Calculate benchmark results
    benchmark_results = {
        "configuration": {
            "model_type": config.model_type.value,
            "batch_size": config.batch_size,
            "target_latency_ms": config.target_latency_ms,
            "target_accuracy": config.target_accuracy,
        },
        "single_frame_performance": {
            "avg_latency_ms": np.mean(single_frame_times),
            "p50_latency_ms": np.percentile(single_frame_times, 50),
            "p95_latency_ms": np.percentile(single_frame_times, 95),
            "p99_latency_ms": np.percentile(single_frame_times, 99),
            "max_latency_ms": np.max(single_frame_times),
            "throughput_fps": len(single_frame_times) / single_frame_duration,
            "meets_latency_target": np.percentile(single_frame_times, 95)
            <= config.target_latency_ms,
        },
        "batch_performance": {
            "batch_size": batch_size,
            "total_batch_time_ms": batch_duration * 1000,
            "avg_per_frame_ms": batch_per_frame,
            "batch_throughput_fps": (
                batch_size / batch_duration if batch_duration > 0 else 0
            ),
            "batch_efficiency": (
                np.mean(single_frame_times) / batch_per_frame
                if batch_per_frame > 0
                else 0
            ),
        },
        "system_metrics": performance_metrics,
        "health_status": health_status,
        "benchmark_summary": {
            "total_frames_tested": num_frames,
            "successful_frames": performance_metrics.get("engine", {}).get(
                "total_frames_processed", 0
            ),
            "success_rate": performance_metrics.get("engine", {}).get(
                "success_rate", 0
            ),
            "meets_performance_targets": (
                np.percentile(single_frame_times, 95) <= config.target_latency_ms
                and health_status["health_score"] > 0.8
            ),
        },
    }

    logger.info("Benchmark completed successfully")
    return benchmark_results
