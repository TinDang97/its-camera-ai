"""
ML Annotation Processor for Real-time Video Streaming.

This module integrates YOLO11 object detection with the dual-channel streaming pipeline
to provide real-time AI-annotated video streams with bounding box overlays.

Key Features:
- Real-time YOLO11 inference with <50ms processing time
- Annotation overlay generation with configurable styles
- Integration with existing streaming infrastructure
- GPU-optimized batch processing
- Detection metadata streaming alongside video fragments
- Configurable detection thresholds and object classes

Architecture:
- MLAnnotationProcessor: Main orchestrator for ML inference integration
- AnnotationRenderer: Visual overlay generation system
- DetectionConfig: Configurable detection and annotation settings
- AnnotationStyleConfig: Customizable visual styling
- DetectionMetadata: JSON detection data for client consumption
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

from ..core.exceptions import StreamProcessingError
from ..core.logging import get_logger
from .core_vision_engine import CoreVisionEngine, DetectionResult, VisionConfig

logger = get_logger(__name__)

# Optional import for ultra-fast engine (may not be available in all environments)
try:
    from .ultra_fast_yolo11_engine import UltraFastYOLOConfig, UltraFastYOLOEngine
    ULTRA_FAST_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Ultra-fast YOLO engine not available: {e}")
    UltraFastYOLOEngine = None
    UltraFastYOLOConfig = None
    ULTRA_FAST_AVAILABLE = False


@dataclass
class DetectionConfig:
    """Configuration for object detection and annotation."""

    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    max_detections: int = 100
    classes_to_detect: list[str] = field(default_factory=lambda: [
        "car", "truck", "bus", "motorcycle", "bicycle", "person"
    ])
    annotation_style: Optional['AnnotationStyleConfig'] = None

    # Performance settings
    enable_gpu_acceleration: bool = True
    batch_size: int = 8
    target_latency_ms: float = 50.0

    # Vehicle-specific settings
    vehicle_priority: bool = True
    emergency_vehicle_detection: bool = True
    confidence_boost_factor: float = 1.1


@dataclass
class AnnotationStyleConfig:
    """Configuration for annotation visual styling."""

    box_color_map: dict[str, tuple[int, int, int]] = field(default_factory=lambda: {
        "car": (0, 255, 0),        # Green
        "truck": (255, 0, 0),      # Blue
        "bus": (0, 0, 255),        # Red
        "motorcycle": (255, 255, 0),  # Cyan
        "bicycle": (255, 0, 255),     # Magenta
        "person": (0, 255, 255),      # Yellow
    })
    box_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2
    font_color: tuple[int, int, int] = (255, 255, 255)  # White
    show_confidence: bool = True
    show_class_labels: bool = True
    transparency: float = 0.8
    label_padding: int = 5
    label_background_alpha: float = 0.7


@dataclass
class Detection:
    """Single object detection result."""

    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: tuple[float, float]
    area: float
    class_id: int
    track_id: int | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class AnnotatedFrame:
    """Annotated frame with detections and metadata."""

    frame_data: bytes
    detections: list[Detection]
    metadata: dict[str, Any]
    timestamp: float
    processing_time_ms: float
    frame_id: str
    camera_id: str


class DetectionMetadata(BaseModel):
    """JSON metadata for detection results."""

    frame_id: str = Field(..., description="Unique frame identifier")
    camera_id: str = Field(..., description="Camera identifier")
    timestamp: float = Field(..., description="Frame timestamp")
    processing_time_ms: float = Field(..., description="ML processing time")
    detection_count: int = Field(..., description="Number of detections")
    vehicle_count: dict[str, int] = Field(
        default_factory=dict, description="Vehicle count by type"
    )
    detections: list[dict[str, Any]] = Field(
        default_factory=list, description="Detection details"
    )
    performance_metrics: dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )


class AnnotationRenderer:
    """Renders visual annotations on video frames."""

    def __init__(self, style_config: AnnotationStyleConfig | None = None):
        """Initialize annotation renderer.
        
        Args:
            style_config: Visual styling configuration
        """
        self.style_config = style_config or AnnotationStyleConfig()
        self._font = cv2.FONT_HERSHEY_SIMPLEX

        logger.info("Annotation renderer initialized")

    async def render_detections_on_frame(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        style_config: AnnotationStyleConfig | None = None
    ) -> np.ndarray:
        """Render detection overlays on frame.
        
        Args:
            frame: Input video frame
            detections: List of detected objects
            style_config: Optional style override
            
        Returns:
            Frame with rendered annotations
        """
        try:
            start_time = time.time()
            config = style_config or self.style_config
            annotated_frame = frame.copy()

            for detection in detections:
                # Get color for object class
                color = config.box_color_map.get(
                    detection.class_name, (255, 255, 255)
                )

                # Draw bounding box
                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(
                    annotated_frame, (x1, y1), (x2, y2),
                    color, config.box_thickness
                )

                # Prepare label text
                label_parts = []
                if config.show_class_labels:
                    label_parts.append(detection.class_name)
                if config.show_confidence:
                    label_parts.append(f"{detection.confidence:.2f}")

                if label_parts:
                    label = " ".join(label_parts)

                    # Get label size
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, self._font, config.font_scale, config.font_thickness
                    )

                    # Draw label background
                    if config.label_background_alpha > 0:
                        overlay = annotated_frame.copy()
                        cv2.rectangle(
                            overlay,
                            (x1, y1 - label_height - config.label_padding * 2),
                            (x1 + label_width + config.label_padding * 2, y1),
                            color,
                            -1
                        )
                        cv2.addWeighted(
                            annotated_frame, 1 - config.label_background_alpha,
                            overlay, config.label_background_alpha,
                            0, annotated_frame
                        )

                    # Draw label text
                    cv2.putText(
                        annotated_frame, label,
                        (x1 + config.label_padding, y1 - config.label_padding),
                        self._font, config.font_scale, config.font_color,
                        config.font_thickness
                    )

            processing_time = (time.time() - start_time) * 1000
            logger.debug(
                f"Rendered {len(detections)} annotations in {processing_time:.2f}ms"
            )

            return annotated_frame

        except Exception as e:
            logger.error(f"Annotation rendering failed: {e}")
            return frame  # Return original frame on error

    def create_annotation_overlay(
        self,
        frame_size: tuple[int, int],
        detections: list[Detection]
    ) -> np.ndarray:
        """Create transparent overlay with annotations.
        
        Args:
            frame_size: (width, height) of the frame
            detections: List of detected objects
            
        Returns:
            Transparent overlay with annotations
        """
        try:
            width, height = frame_size
            overlay = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA

            for detection in detections:
                color = self.style_config.box_color_map.get(
                    detection.class_name, (255, 255, 255)
                )

                # Add alpha channel
                rgba_color = (*color, int(255 * self.style_config.transparency))

                x1, y1, x2, y2 = detection.bbox
                cv2.rectangle(
                    overlay, (x1, y1), (x2, y2),
                    rgba_color, self.style_config.box_thickness
                )

            return overlay

        except Exception as e:
            logger.error(f"Overlay creation failed: {e}")
            return np.zeros((frame_size[1], frame_size[0], 4), dtype=np.uint8)


class MLAnnotationProcessor:
    """Main ML annotation processor for streaming integration."""

    def __init__(
        self,
        vision_engine: CoreVisionEngine | None = None,
        ultra_fast_engine: Any | None = None,  # UltraFastYOLOEngine when available
        config: DetectionConfig | None = None
    ):
        """Initialize ML annotation processor.
        
        Args:
            vision_engine: Core vision engine for inference
            ultra_fast_engine: Ultra-fast YOLO engine for <10ms inference
            config: Detection configuration
        """
        self.config = config or DetectionConfig()
        self.renderer = AnnotationRenderer(self.config.annotation_style)

        # Initialize engines
        self._vision_engine = vision_engine
        self._ultra_fast_engine = ultra_fast_engine
        self._active_engine = None

        # Performance tracking
        self.inference_times = []
        self.annotation_times = []
        self._processed_frames = 0

        # Initialize async components
        self._inference_semaphore = asyncio.Semaphore(self.config.batch_size)

        logger.info(
            f"ML annotation processor initialized with target latency: "
            f"{self.config.target_latency_ms}ms"
        )

    async def initialize(self) -> None:
        """Initialize ML engines and models."""
        try:
            # Initialize ultra-fast engine if available
            if self._ultra_fast_engine and ULTRA_FAST_AVAILABLE:
                await self._ultra_fast_engine.initialize()
                self._active_engine = self._ultra_fast_engine
                logger.info("Using ultra-fast YOLO engine for <10ms inference")

            # Fallback to vision engine
            elif self._vision_engine:
                await self._vision_engine.initialize()
                self._active_engine = self._vision_engine
                logger.info("Using core vision engine for inference")

            else:
                # Create default vision engine
                vision_config = VisionConfig(
                    confidence_threshold=self.config.confidence_threshold,
                    batch_size=self.config.batch_size,
                    target_latency_ms=self.config.target_latency_ms
                )
                self._vision_engine = CoreVisionEngine(config=vision_config)
                await self._vision_engine.initialize()
                self._active_engine = self._vision_engine
                logger.info("Created default vision engine for inference")

        except Exception as e:
            logger.error(f"Failed to initialize ML engines: {e}")
            raise StreamProcessingError(f"ML initialization failed: {e}")

    async def process_frame_with_annotations(
        self,
        frame_data: bytes,
        camera_id: str,
        detection_config: DetectionConfig | None = None
    ) -> AnnotatedFrame:
        """Process frame with ML inference and annotation rendering.
        
        Args:
            frame_data: Raw frame bytes
            camera_id: Camera identifier
            detection_config: Optional detection config override
            
        Returns:
            Annotated frame with detections and metadata
            
        Raises:
            StreamProcessingError: If processing fails
        """
        start_time = time.time()
        config = detection_config or self.config

        try:
            async with self._inference_semaphore:
                # Decode frame
                frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                if frame is None:
                    raise ValueError("Failed to decode frame")

                # Run ML inference
                inference_start = time.time()
                detections = await self._run_inference(frame, config)
                inference_time = (time.time() - inference_start) * 1000

                # Render annotations
                render_start = time.time()
                annotated_frame = await self.renderer.render_detections_on_frame(
                    frame, detections, config.annotation_style
                )
                render_time = (time.time() - render_start) * 1000

                # Encode annotated frame
                _, encoded_frame = cv2.imencode('.jpg', annotated_frame)
                annotated_data = encoded_frame.tobytes()

                # Create metadata
                total_time = (time.time() - start_time) * 1000

                # Update performance tracking
                self.inference_times.append(inference_time)
                self.annotation_times.append(render_time)
                self._processed_frames += 1

                # Keep only recent samples for metrics
                if len(self.inference_times) > 1000:
                    self.inference_times = self.inference_times[-1000:]
                    self.annotation_times = self.annotation_times[-1000:]

                frame_id = f"{camera_id}_{int(time.time() * 1000)}_{self._processed_frames}"

                metadata = {
                    "inference_time_ms": inference_time,
                    "render_time_ms": render_time,
                    "total_time_ms": total_time,
                    "detection_count": len(detections),
                    "frame_size": frame.shape[:2],
                    "config": {
                        "confidence_threshold": config.confidence_threshold,
                        "classes": config.classes_to_detect
                    }
                }

                logger.debug(
                    f"Processed frame {frame_id} in {total_time:.2f}ms "
                    f"(inference: {inference_time:.2f}ms, render: {render_time:.2f}ms, "
                    f"detections: {len(detections)})"
                )

                return AnnotatedFrame(
                    frame_data=annotated_data,
                    detections=detections,
                    metadata=metadata,
                    timestamp=time.time(),
                    processing_time_ms=total_time,
                    frame_id=frame_id,
                    camera_id=camera_id
                )

        except Exception as e:
            logger.error(f"Frame annotation processing failed: {e}")
            raise StreamProcessingError(f"Annotation processing failed: {e}")

    async def _run_inference(
        self,
        frame: np.ndarray,
        config: DetectionConfig
    ) -> list[Detection]:
        """Run ML inference on frame.
        
        Args:
            frame: Input frame
            config: Detection configuration
            
        Returns:
            List of detected objects
        """
        try:
            if not self._active_engine:
                raise RuntimeError("ML engine not initialized")

            # Run inference using active engine
            if (ULTRA_FAST_AVAILABLE and UltraFastYOLOEngine and
                isinstance(self._active_engine, UltraFastYOLOEngine)):
                # Use ultra-fast engine
                results = await self._active_engine.predict_batch([frame])
                detections = self._convert_ultra_fast_results(results[0])
            else:
                # Use core vision engine
                detection_result = await self._active_engine.detect_objects(
                    frame,
                    confidence_threshold=config.confidence_threshold,
                    classes_filter=config.classes_to_detect
                )
                detections = self._convert_vision_results(detection_result)

            # Filter detections by configured classes
            filtered_detections = [
                det for det in detections
                if det.class_name in config.classes_to_detect
                and det.confidence >= config.confidence_threshold
            ]

            # Apply vehicle priority boost if enabled
            if config.vehicle_priority:
                for detection in filtered_detections:
                    if detection.class_name in ["car", "truck", "bus", "motorcycle"]:
                        detection.confidence *= config.confidence_boost_factor
                        detection.confidence = min(detection.confidence, 1.0)

            # Limit number of detections
            if len(filtered_detections) > config.max_detections:
                filtered_detections = sorted(
                    filtered_detections,
                    key=lambda x: x.confidence,
                    reverse=True
                )[:config.max_detections]

            return filtered_detections

        except Exception as e:
            logger.error(f"ML inference failed: {e}")
            return []  # Return empty list on error

    def _convert_ultra_fast_results(self, results: Any) -> list[Detection]:
        """Convert ultra-fast engine results to Detection objects."""
        detections = []
        # Implementation depends on ultra-fast engine output format
        # This is a placeholder for the actual conversion logic
        return detections

    def _convert_vision_results(self, result: DetectionResult) -> list[Detection]:
        """Convert vision engine results to Detection objects."""
        detections = []

        if result.detections:
            for detection in result.detections:
                # Convert to our Detection format
                det = Detection(
                    class_name=detection.get('class', 'unknown'),
                    confidence=detection.get('confidence', 0.0),
                    bbox=detection.get('bbox', (0, 0, 0, 0)),
                    center=detection.get('center', (0.0, 0.0)),
                    area=detection.get('area', 0.0),
                    class_id=detection.get('class_id', 0)
                )
                detections.append(det)

        return detections

    async def create_detection_metadata(
        self,
        detections: list[Detection],
        frame_timestamp: float,
        camera_id: str,
        frame_id: str,
        processing_time_ms: float
    ) -> DetectionMetadata:
        """Create JSON metadata for detections.
        
        Args:
            detections: List of detected objects
            frame_timestamp: Frame timestamp
            camera_id: Camera identifier
            frame_id: Frame identifier
            processing_time_ms: Total processing time
            
        Returns:
            Detection metadata for client consumption
        """
        try:
            # Count vehicles by type
            vehicle_count = {}
            detection_data = []

            for detection in detections:
                # Update vehicle counts
                class_name = detection.class_name
                vehicle_count[class_name] = vehicle_count.get(class_name, 0) + 1

                # Create detection data
                det_data = {
                    "class": class_name,
                    "confidence": round(detection.confidence, 3),
                    "bbox": list(detection.bbox),
                    "center": list(detection.center),
                    "area": round(detection.area, 2),
                    "class_id": detection.class_id,
                    "timestamp": detection.timestamp
                }

                if detection.track_id is not None:
                    det_data["track_id"] = detection.track_id

                detection_data.append(det_data)

            # Performance metrics
            avg_inference_time = (
                sum(self.inference_times[-100:]) / len(self.inference_times[-100:])
                if self.inference_times else 0.0
            )

            avg_render_time = (
                sum(self.annotation_times[-100:]) / len(self.annotation_times[-100:])
                if self.annotation_times else 0.0
            )

            performance_metrics = {
                "avg_inference_time_ms": round(avg_inference_time, 2),
                "avg_render_time_ms": round(avg_render_time, 2),
                "total_processing_time_ms": round(processing_time_ms, 2),
                "processed_frames": self._processed_frames,
                "target_latency_ms": self.config.target_latency_ms
            }

            return DetectionMetadata(
                frame_id=frame_id,
                camera_id=camera_id,
                timestamp=frame_timestamp,
                processing_time_ms=processing_time_ms,
                detection_count=len(detections),
                vehicle_count=vehicle_count,
                detections=detection_data,
                performance_metrics=performance_metrics
            )

        except Exception as e:
            logger.error(f"Failed to create detection metadata: {e}")
            return DetectionMetadata(
                frame_id=frame_id,
                camera_id=camera_id,
                timestamp=frame_timestamp,
                processing_time_ms=processing_time_ms,
                detection_count=0
            )

    def get_performance_stats(self) -> dict[str, Any]:
        """Get ML processing performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.inference_times:
            return {
                "processed_frames": 0,
                "avg_inference_time_ms": 0.0,
                "avg_render_time_ms": 0.0,
                "avg_total_time_ms": 0.0
            }

        recent_inference = self.inference_times[-100:]
        recent_render = self.annotation_times[-100:]
        recent_total = [i + r for i, r in zip(recent_inference, recent_render, strict=False)]

        return {
            "processed_frames": self._processed_frames,
            "avg_inference_time_ms": round(sum(recent_inference) / len(recent_inference), 2),
            "avg_render_time_ms": round(sum(recent_render) / len(recent_render), 2),
            "avg_total_time_ms": round(sum(recent_total) / len(recent_total), 2),
            "min_inference_time_ms": round(min(recent_inference), 2),
            "max_inference_time_ms": round(max(recent_inference), 2),
            "target_latency_ms": self.config.target_latency_ms,
            "latency_compliance": sum(1 for t in recent_total if t <= self.config.target_latency_ms) / len(recent_total) * 100
        }

    async def cleanup(self) -> None:
        """Cleanup ML processor resources."""
        try:
            if self._active_engine and hasattr(self._active_engine, 'cleanup'):
                await self._active_engine.cleanup()

            logger.info(
                f"ML annotation processor cleaned up. "
                f"Processed {self._processed_frames} frames."
            )

        except Exception as e:
            logger.error(f"Error during ML processor cleanup: {e}")
