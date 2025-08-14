"""
Real-time Data Processing for ITS Camera AI Traffic Monitoring System.

This module handles real-time processing of camera streams with data validation,
feature extraction, and preparation for ML inference and training.

Key Features:
- Real-time stream processing with Apache Kafka integration
- Data quality validation and filtering
- Feature extraction for traffic analytics
- Data versioning and lineage tracking
- Scalable processing with asyncio and multiprocessing

Performance Targets:
- Process 1000+ concurrent camera streams
- Sub-10ms data validation latency
- 99.9% data quality compliance
- Fault-tolerant with automatic recovery
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2
import numpy as np
from PIL import Image


# Fallback classes for development (defined before use)
class RedisQueueManagerFallback:
    """Fallback Redis queue manager when Redis is not available."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def connect(self) -> None:
        """Connect to Redis (no-op in fallback)."""
        pass

    async def disconnect(self) -> None:
        """Disconnect from Redis (no-op in fallback)."""
        pass

    async def create_queue(self, *_args: Any, **_kwargs: Any) -> None:
        """Create a queue (no-op in fallback)."""
        pass

    async def enqueue(self, *_args: Any, **_kwargs: Any) -> bool:
        """Enqueue a message (always succeeds in fallback)."""
        return True

    async def dequeue(self, *_args: Any, **_kwargs: Any) -> None:
        """Dequeue a message (returns None in fallback)."""
        return None

    async def dequeue_batch(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        """Dequeue batch of messages (returns empty list in fallback)."""
        return []

    async def acknowledge(self, *_args: Any, **_kwargs: Any) -> bool:
        """Acknowledge a message (always succeeds in fallback)."""
        return True

    async def get_queue_metrics(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """Get queue metrics (returns empty dict in fallback)."""
        return {}

    async def health_check(self) -> dict[str, Any]:
        """Health check (always healthy in fallback)."""
        return {"status": "healthy", "fallback": True}

    async def __aenter__(self) -> "RedisQueueManagerFallback":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, *_args: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()


@dataclass
class QueueConfigFallback:
    """Fallback queue configuration."""

    name: str = "default"
    queue_type: Any = None
    batch_size: int = 10


class QueueTypeFallback(Enum):
    """Fallback queue type enum."""

    STREAM = "stream"
    LIST = "list"


class ProcessedFrameSerializerFallback:
    """Fallback frame serializer."""

    def serialize(self, *_args: Any, **_kwargs: Any) -> bytes:
        return b""

    def deserialize(self, *_args: Any, **_kwargs: Any) -> None:
        return None


# Async messaging and serialization
try:
    from .grpc_serialization import ProcessedFrameSerializer
    from .redis_queue_manager import QueueConfig, QueueType, RedisQueueManager

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

    # Alias fallback classes
    RedisQueueManager = RedisQueueManagerFallback  # type: ignore
    QueueConfig = QueueConfigFallback  # type: ignore
    QueueType = QueueTypeFallback  # type: ignore
    ProcessedFrameSerializer = ProcessedFrameSerializerFallback  # type: ignore


logger = logging.getLogger(__name__)


class StreamStatus(Enum):
    """Camera stream status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class ProcessingStage(Enum):
    """Data processing stages."""

    INGESTION = "ingestion"
    VALIDATION = "validation"
    FEATURE_EXTRACTION = "feature_extraction"
    QUALITY_CONTROL = "quality_control"
    OUTPUT = "output"


@dataclass
class CameraStream:
    """Camera stream metadata and configuration."""

    camera_id: str
    location: str
    coordinates: tuple[float, float]  # (latitude, longitude)

    # Stream configuration
    resolution: tuple[int, int] = (1920, 1080)
    fps: int = 30
    encoding: str = "h264"

    # Processing configuration
    roi_boxes: list[tuple[int, int, int, int]] = field(
        default_factory=list
    )  # Regions of interest
    quality_threshold: float = 0.7
    processing_enabled: bool = True

    # Status tracking
    status: StreamStatus = StreamStatus.ACTIVE
    last_frame_time: float = field(default_factory=time.time)
    total_frames_processed: int = 0

    # Performance metrics
    avg_processing_latency_ms: float = 0.0
    quality_score_avg: float = 0.0
    error_rate: float = 0.0


@dataclass
class ProcessedFrame:
    """Processed frame data with extracted features."""

    # Core identifiers
    frame_id: str
    camera_id: str
    timestamp: float

    # Image data
    original_image: np.ndarray[Any, Any]
    processed_image: np.ndarray[Any, Any] | None = None
    thumbnail: np.ndarray[Any, Any] | None = None

    # Quality metrics
    quality_score: float = 0.0
    blur_score: float = 0.0
    brightness_score: float = 0.0
    contrast_score: float = 0.0
    noise_level: float = 0.0

    # Traffic features
    vehicle_density: float = 0.0
    congestion_level: str = "unknown"
    weather_conditions: str = "unknown"
    lighting_conditions: str = "unknown"

    # ROI analysis
    roi_features: dict[str, Any] = field(default_factory=dict)

    # Processing metadata
    processing_time_ms: float = 0.0
    processing_stage: ProcessingStage = ProcessingStage.INGESTION
    validation_passed: bool = False

    # Data lineage
    source_hash: str = ""
    version: str = "1.0"


class ImageQualityAnalyzer:
    """Analyze image quality metrics for traffic monitoring."""

    def __init__(self) -> None:
        self.blur_threshold = 100.0  # Laplacian variance threshold
        self.brightness_range = (30, 220)  # Optimal brightness range
        self.contrast_threshold = 40.0  # Minimum contrast

    def analyze_quality(self, image: np.ndarray[Any, Any]) -> dict[str, float]:
        """Comprehensive image quality analysis."""

        metrics = {}

        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Blur detection using Laplacian variance
        metrics["blur_score"] = self._calculate_blur_score(gray)

        # Brightness analysis
        metrics["brightness_score"] = self._calculate_brightness_score(gray)

        # Contrast analysis
        metrics["contrast_score"] = self._calculate_contrast_score(gray)

        # Noise estimation
        metrics["noise_level"] = self._estimate_noise_level(gray)

        # Overall quality score (weighted combination)
        metrics["quality_score"] = self._calculate_overall_quality(metrics)

        return metrics

    def _calculate_blur_score(self, gray_image: np.ndarray[Any, Any]) -> float:
        """Calculate blur score using Laplacian variance."""
        laplacian_var = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

        # Normalize to 0-1 score (higher = sharper)
        score = min(1.0, laplacian_var / 1000.0)
        return score

    def _calculate_brightness_score(self, gray_image: np.ndarray[Any, Any]) -> float:
        """Calculate brightness adequacy score."""
        mean_brightness = np.mean(gray_image)

        # Score based on distance from optimal range
        if self.brightness_range[0] <= mean_brightness <= self.brightness_range[1]:
            score = 1.0
        else:
            distance = min(
                abs(mean_brightness - self.brightness_range[0]),
                abs(mean_brightness - self.brightness_range[1]),
            )
            score = max(0.0, 1.0 - (distance / 100.0))

        return score

    def _calculate_contrast_score(self, gray_image: np.ndarray[Any, Any]) -> float:
        """Calculate image contrast score."""
        # RMS contrast
        rms_contrast = np.std(gray_image)

        # Normalize to 0-1 score
        score = min(1.0, float(rms_contrast) / 80.0)
        return score

    def _estimate_noise_level(self, gray_image: np.ndarray[Any, Any]) -> float:
        """Estimate noise level using high-frequency content."""
        # Apply high-pass filter to estimate noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray_image, -1, kernel)

        # Calculate noise level as std deviation of filtered image
        noise = float(np.std(filtered))  # type: ignore

        # Normalize to 0-1 (lower = less noisy)
        score = max(0.0, 1.0 - (noise / 50.0))
        return float(score)

    def _calculate_overall_quality(self, metrics: dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        weights = {
            "blur_score": 0.3,
            "brightness_score": 0.25,
            "contrast_score": 0.25,
            "noise_level": 0.2,
        }

        score = sum(metrics[key] * weights[key] for key in weights if key in metrics)
        return min(1.0, max(0.0, score))


class FeatureExtractor:
    """Extract traffic-specific features from processed frames."""

    def __init__(self) -> None:
        self.density_calculator = VehicleDensityCalculator()
        self.weather_detector = WeatherConditionDetector()
        self.lighting_analyzer = LightingConditionAnalyzer()

    def extract_features(
        self,
        frame: np.ndarray[Any, Any],
        roi_boxes: list[tuple[int, int, int, int]] | None = None,
    ) -> dict[str, Any]:
        """Extract comprehensive traffic features from frame."""

        features = {}

        # Vehicle density analysis
        features["vehicle_density"] = self.density_calculator.calculate(frame)

        # Weather conditions
        features["weather_conditions"] = self.weather_detector.detect(frame)

        # Lighting conditions
        features["lighting_conditions"] = self.lighting_analyzer.analyze(frame)

        # ROI-specific features
        if roi_boxes:
            features["roi_features"] = self._extract_roi_features(frame, roi_boxes)

        # Traffic flow metrics
        features["congestion_level"] = self._estimate_congestion_level(
            features["vehicle_density"]
        )

        return features

    def _extract_roi_features(
        self, frame: np.ndarray[Any, Any], roi_boxes: list[tuple[int, int, int, int]]
    ) -> dict[str, Any]:
        """Extract features from regions of interest."""
        roi_features = {}

        for i, (x1, y1, x2, y2) in enumerate(roi_boxes):
            roi = frame[y1:y2, x1:x2]

            roi_features[f"roi_{i}"] = {
                "mean_intensity": float(np.mean(roi)),
                "std_intensity": float(np.std(roi)),
                "edge_density": self._calculate_edge_density(roi),
                "motion_score": 0.0,  # Placeholder for motion analysis
            }

        return roi_features

    def _calculate_edge_density(self, roi: np.ndarray[Any, Any]) -> float:
        """Calculate edge density in ROI."""
        edges = cv2.Canny(roi, 50, 150)
        density = float(np.mean(edges) / 255.0)
        return density

    def _estimate_congestion_level(self, vehicle_density: float) -> str:
        """Estimate traffic congestion level from vehicle density."""
        if vehicle_density < 0.2:
            return "free_flow"
        elif vehicle_density < 0.5:
            return "moderate"
        elif vehicle_density < 0.8:
            return "heavy"
        else:
            return "congested"


class VehicleDensityCalculator:
    """Calculate vehicle density from frame."""

    def calculate(self, frame: np.ndarray[Any, Any]) -> float:
        """Simple density estimation based on image features."""
        # This is a placeholder - in production, use YOLO detections
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Simple heuristic: edge density correlates with vehicle presence
        density = float(np.mean(edges) / 255.0)
        return min(1.0, density * 2.0)


class WeatherConditionDetector:
    """Detect weather conditions from frame."""

    def detect(self, frame: np.ndarray[Any, Any]) -> str:
        """Detect weather conditions using image analysis."""
        # Simplified weather detection based on image properties
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Check for fog/haze (low saturation, high value)
        saturation_mean = float(np.mean(hsv[:, :, 1]))
        value_mean = float(np.mean(hsv[:, :, 2]))

        if saturation_mean < 50 and value_mean > 150:
            return "foggy"
        elif saturation_mean < 30:
            return "overcast"
        elif value_mean < 50:
            return "dark"
        else:
            return "clear"


class LightingConditionAnalyzer:
    """Analyze lighting conditions in frame."""

    def analyze(self, frame: np.ndarray[Any, Any]) -> str:
        """Analyze lighting conditions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        mean_brightness = float(np.mean(gray))

        if mean_brightness < 30:
            return "night"
        elif mean_brightness < 80:
            return "dawn_dusk"
        elif mean_brightness > 200:
            return "bright_sunlight"
        else:
            return "daylight"


class DataValidator:
    """Validate data quality and compliance."""

    def __init__(
        self,
        min_quality_score: float = 0.5,
        required_resolution: tuple[int, int] | None = None,
        max_age_seconds: float = 5.0,
    ):
        self.min_quality_score = min_quality_score
        self.required_resolution = required_resolution or (640, 480)
        self.max_age_seconds = max_age_seconds

    def validate(self, frame: ProcessedFrame) -> tuple[bool, list[str]]:
        """Validate processed frame data."""
        errors = []

        # Quality validation
        if frame.quality_score < self.min_quality_score:
            errors.append(f"Quality score {frame.quality_score:.2f} below threshold")

        # Resolution validation
        if frame.original_image.shape[:2] < self.required_resolution:
            errors.append(
                f"Resolution {frame.original_image.shape[:2]} below required {self.required_resolution}"
            )

        # Age validation
        age = time.time() - frame.timestamp
        if age > self.max_age_seconds:
            errors.append(
                f"Frame age {age:.1f}s exceeds maximum {self.max_age_seconds}s"
            )

        # Data completeness
        if frame.source_hash == "":
            errors.append("Missing source hash for data lineage")

        return len(errors) == 0, errors


class StreamProcessor:
    """Main stream processing orchestrator."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        enable_quality_analysis: bool = True,
        enable_feature_extraction: bool = True,
        batch_size: int = 10,
        processing_timeout_ms: int = 100,
    ):
        self.redis_url = redis_url
        self.enable_quality_analysis = enable_quality_analysis
        self.enable_feature_extraction = enable_feature_extraction
        self.batch_size = batch_size
        self.processing_timeout_ms = processing_timeout_ms

        # Processing components
        self.quality_analyzer = ImageQualityAnalyzer()
        self.feature_extractor = FeatureExtractor()
        self.validator = DataValidator()

        # Queue manager
        self.queue_manager: RedisQueueManager | None = None
        self.serializer = (
            ProcessedFrameSerializer()
            if REDIS_AVAILABLE
            else ProcessedFrameSerializerFallback()
        )

        # Stream registry
        self.streams: dict[str, CameraStream] = {}
        self.processing_stats: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=1000)
        )

        # Control flags
        self.is_running = False
        self.processing_tasks: dict[str, asyncio.Task[Any]] = {}

        logger.info(
            f"Stream processor initialized with batch_size={batch_size}, "
            f"timeout={processing_timeout_ms}ms"
        )

    async def start(self) -> None:
        """Start the stream processor."""
        logger.info("Starting stream processor...")

        # Connect to Redis
        if REDIS_AVAILABLE:
            self.queue_manager = RedisQueueManager(self.redis_url)
            await self.queue_manager.connect()

            # Create processing queues
            await self._setup_queues()
        else:
            logger.warning("Redis not available, using fallback mode")
            self.queue_manager = RedisQueueManagerFallback()

        self.is_running = True

        # Start processing tasks
        for i in range(3):  # 3 parallel processors
            task = asyncio.create_task(self._process_stream_batch(f"processor_{i}"))
            self.processing_tasks[f"processor_{i}"] = task

        # Start metrics collection
        self.metrics_task = asyncio.create_task(self._collect_metrics())

        logger.info("Stream processor started")

    async def stop(self) -> None:
        """Stop the stream processor."""
        logger.info("Stopping stream processor...")

        self.is_running = False

        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            task.cancel()

        if hasattr(self, "metrics_task"):
            self.metrics_task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(
            *self.processing_tasks.values(),
            return_exceptions=True,
        )

        # Disconnect from Redis
        if self.queue_manager and REDIS_AVAILABLE:
            await self.queue_manager.disconnect()

        logger.info("Stream processor stopped")

    async def _setup_queues(self) -> None:
        """Setup Redis queues for processing."""
        if not self.queue_manager or not REDIS_AVAILABLE:
            return

        # Input queue for raw frames
        await self.queue_manager.create_queue(
            QueueConfig(
                name="camera_frames_input",
                queue_type=QueueType.STREAM,
                max_length=10000,
                batch_size=self.batch_size,
            )
        )

        # Output queue for processed frames
        await self.queue_manager.create_queue(
            QueueConfig(
                name="processed_frames_output",
                queue_type=QueueType.STREAM,
                max_length=5000,
                batch_size=self.batch_size,
            )
        )

        # Quality control queue for failed frames
        await self.queue_manager.create_queue(
            QueueConfig(
                name="quality_control_queue",
                queue_type=QueueType.LIST,
                max_length=1000,
            )
        )

        logger.info("Processing queues created")

    async def register_camera(self, stream: CameraStream) -> bool:
        """Register a new camera stream."""
        try:
            self.streams[stream.camera_id] = stream
            logger.info(f"Registered camera stream: {stream.camera_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register camera {stream.camera_id}: {e}")
            return False

    async def process_frame(
        self,
        camera_id: str,
        frame: np.ndarray[Any, Any],
        timestamp: float | None = None,
    ) -> ProcessedFrame | None:
        """Process a single frame from camera."""

        if camera_id not in self.streams:
            logger.warning(f"Unknown camera ID: {camera_id}")
            return None

        stream = self.streams[camera_id]

        if not stream.processing_enabled:
            return None

        timestamp = timestamp or time.time()
        frame_id = f"{camera_id}_{int(timestamp * 1000)}"

        # Create processed frame object
        processed = ProcessedFrame(
            frame_id=frame_id,
            camera_id=camera_id,
            timestamp=timestamp,
            original_image=frame,
        )

        try:
            # Stage 1: Quality analysis
            if self.enable_quality_analysis:
                quality_metrics = self.quality_analyzer.analyze_quality(frame)
                processed.quality_score = quality_metrics["quality_score"]
                processed.blur_score = quality_metrics["blur_score"]
                processed.brightness_score = quality_metrics["brightness_score"]
                processed.contrast_score = quality_metrics["contrast_score"]
                processed.noise_level = quality_metrics["noise_level"]
                processed.processing_stage = ProcessingStage.QUALITY_CONTROL

            # Stage 2: Feature extraction
            if self.enable_feature_extraction:
                features = self.feature_extractor.extract_features(
                    frame,
                    stream.roi_boxes,
                )
                processed.vehicle_density = features["vehicle_density"]
                processed.congestion_level = features["congestion_level"]
                processed.weather_conditions = features["weather_conditions"]
                processed.lighting_conditions = features["lighting_conditions"]
                processed.roi_features = features.get("roi_features", {})
                processed.processing_stage = ProcessingStage.FEATURE_EXTRACTION

            # Stage 3: Validation
            is_valid, errors = self.validator.validate(processed)
            processed.validation_passed = is_valid

            if not is_valid:
                logger.debug(f"Frame {frame_id} validation failed: {errors}")
                # Send to quality control queue
                if self.queue_manager and REDIS_AVAILABLE:
                    await self.queue_manager.enqueue(
                        "quality_control_queue",
                        json.dumps({"frame_id": frame_id, "errors": errors}).encode(),
                    )

            # Calculate processing time
            processed.processing_time_ms = (time.time() - timestamp) * 1000

            # Generate source hash for lineage
            processed.source_hash = hashlib.sha256(
                f"{camera_id}_{timestamp}".encode()
            ).hexdigest()[:16]

            # Update stream metrics
            stream.total_frames_processed += 1
            stream.last_frame_time = time.time()
            self.processing_stats[camera_id].append(processed.processing_time_ms)

            # Enqueue for downstream processing if valid
            if is_valid and self.queue_manager and REDIS_AVAILABLE:
                serialized = self.serializer.serialize(processed)
                await self.queue_manager.enqueue(
                    "processed_frames_output",
                    serialized,
                    metadata={
                        "camera_id": camera_id,
                        "quality_score": processed.quality_score,
                    },
                )

            return processed

        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
            return None

    async def _process_stream_batch(self, processor_id: str) -> None:
        """Process frames in batches."""
        logger.info(f"Batch processor {processor_id} started")

        while self.is_running:
            try:
                if not self.queue_manager or not REDIS_AVAILABLE:
                    await asyncio.sleep(0.1)
                    continue

                # Dequeue batch of frames
                messages = await self.queue_manager.dequeue_batch(
                    "camera_frames_input",
                    batch_size=self.batch_size,
                    timeout_ms=self.processing_timeout_ms,
                )

                if not messages:
                    await asyncio.sleep(0.01)  # Short sleep if no messages
                    continue

                # Process batch in parallel
                tasks = []
                for message_id, data in messages:
                    # Deserialize frame data
                    try:
                        frame_data = json.loads(data)
                        camera_id = frame_data["camera_id"]
                        timestamp = frame_data["timestamp"]

                        # Decode image
                        image_bytes = base64.b64decode(frame_data["image"])
                        image = Image.open(io.BytesIO(image_bytes))
                        frame = np.array(image)

                        # Process frame
                        task = self.process_frame(camera_id, frame, timestamp)
                        tasks.append((message_id, task))

                    except Exception as e:
                        logger.error(f"Failed to deserialize frame: {e}")
                        continue

                # Wait for all frames to be processed
                for message_id, task in tasks:
                    try:
                        result = await task
                        if result and self.queue_manager:
                            await self.queue_manager.acknowledge(
                                "camera_frames_input",
                                message_id,
                                processing_time_ms=result.processing_time_ms,
                            )
                    except Exception as e:
                        logger.error(f"Failed to process frame in batch: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing error in {processor_id}: {e}")
                await asyncio.sleep(1)  # Back off on error

        logger.info(f"Batch processor {processor_id} stopped")

    async def _collect_metrics(self) -> None:
        """Collect and log processing metrics."""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Collect metrics every 30 seconds

                total_processed = sum(
                    s.total_frames_processed for s in self.streams.values()
                )
                active_streams = sum(
                    1 for s in self.streams.values() if s.status == StreamStatus.ACTIVE
                )

                # Calculate average processing times
                avg_latencies = {}
                for camera_id, latencies in self.processing_stats.items():
                    if latencies:
                        avg_latencies[camera_id] = np.mean(list(latencies))

                # Get queue metrics if available
                queue_metrics = {}
                if self.queue_manager and REDIS_AVAILABLE:
                    for queue_name in [
                        "camera_frames_input",
                        "processed_frames_output",
                    ]:
                        metrics = await self.queue_manager.get_queue_metrics(queue_name)
                        if metrics:
                            queue_metrics[queue_name] = {
                                "pending": metrics.pending_count,
                                "processing": metrics.processing_count,
                                "completed": metrics.completed_count,
                            }

                logger.info(
                    f"Stream Processing Metrics - "
                    f"Total Processed: {total_processed}, "
                    f"Active Streams: {active_streams}, "
                    f"Avg Latency: {np.mean(list(avg_latencies.values())):.1f}ms, "
                    f"Queue Metrics: {queue_metrics}"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")

    def get_stream_status(self, camera_id: str) -> dict[str, Any] | None:
        """Get status for a specific camera stream."""
        if camera_id not in self.streams:
            return None

        stream = self.streams[camera_id]
        latencies = list(self.processing_stats[camera_id])

        return {
            "camera_id": camera_id,
            "status": stream.status.value,
            "total_frames": stream.total_frames_processed,
            "last_frame_time": stream.last_frame_time,
            "avg_latency_ms": np.mean(latencies) if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            "quality_score_avg": stream.quality_score_avg,
            "error_rate": stream.error_rate,
        }

    async def __aenter__(self) -> "StreamProcessor":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.stop()


# Example usage and testing
async def test_stream_processor() -> None:
    """Test the stream processor with simulated data."""

    processor = StreamProcessor(
        enable_quality_analysis=True,
        enable_feature_extraction=True,
        batch_size=5,
    )

    async with processor:
        # Register test camera
        camera = CameraStream(
            camera_id="test_cam_001",
            location="Intersection A",
            coordinates=(37.7749, -122.4194),
            roi_boxes=[(100, 100, 300, 300), (400, 400, 600, 600)],
        )

        await processor.register_camera(camera)

        # Process test frames
        test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        for i in range(10):
            result = await processor.process_frame(
                "test_cam_001",
                test_frame,
                time.time(),
            )

            if result:
                logger.info(
                    f"Processed frame {i}: "
                    f"Quality={result.quality_score:.2f}, "
                    f"Density={result.vehicle_density:.2f}, "
                    f"Latency={result.processing_time_ms:.1f}ms"
                )

            await asyncio.sleep(0.1)

        # Get final status
        status = processor.get_stream_status("test_cam_001")
        logger.info(f"Final stream status: {status}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_stream_processor())
