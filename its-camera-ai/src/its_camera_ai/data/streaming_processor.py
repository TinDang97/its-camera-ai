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
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Async messaging
try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


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
    roi_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)  # Regions of interest
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
    original_image: np.ndarray
    processed_image: np.ndarray | None = None
    thumbnail: np.ndarray | None = None

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

    def __init__(self):
        self.blur_threshold = 100.0  # Laplacian variance threshold
        self.brightness_range = (30, 220)  # Optimal brightness range
        self.contrast_threshold = 40.0  # Minimum contrast

    def analyze_quality(self, image: np.ndarray) -> dict[str, float]:
        """Comprehensive image quality analysis."""

        metrics = {}

        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Blur detection using Laplacian variance
        metrics['blur_score'] = self._calculate_blur_score(gray)

        # Brightness analysis
        metrics['brightness_score'] = self._calculate_brightness_score(gray)

        # Contrast analysis
        metrics['contrast_score'] = self._calculate_contrast_score(gray)

        # Noise estimation
        metrics['noise_level'] = self._estimate_noise_level(gray)

        # Overall quality score (weighted combination)
        metrics['quality_score'] = self._calculate_overall_quality(metrics)

        return metrics

    def _calculate_blur_score(self, gray_image: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance."""
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

        # Normalize to 0-1 score (higher = sharper)
        score = min(1.0, laplacian_var / 1000.0)
        return score

    def _calculate_brightness_score(self, gray_image: np.ndarray) -> float:
        """Calculate brightness adequacy score."""
        mean_brightness = np.mean(gray_image)

        # Score based on distance from optimal range
        if self.brightness_range[0] <= mean_brightness <= self.brightness_range[1]:
            score = 1.0
        else:
            distance = min(
                abs(mean_brightness - self.brightness_range[0]),
                abs(mean_brightness - self.brightness_range[1])
            )
            score = max(0.0, 1.0 - (distance / 100.0))

        return score

    def _calculate_contrast_score(self, gray_image: np.ndarray) -> float:
        """Calculate image contrast score."""
        # RMS contrast
        rms_contrast = np.std(gray_image)

        # Normalize to 0-1 score
        score = min(1.0, rms_contrast / 80.0)
        return score

    def _estimate_noise_level(self, gray_image: np.ndarray) -> float:
        """Estimate noise level using high-frequency content."""
        # Apply high-pass filter to estimate noise
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(gray_image, -1, kernel)

        # Calculate noise level as std deviation of filtered image
        noise = np.std(filtered)

        # Normalize to 0-1 (lower = less noisy)
        score = max(0.0, 1.0 - (noise / 50.0))
        return score

    def _calculate_overall_quality(self, metrics: dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        weights = {
            'blur_score': 0.3,
            'brightness_score': 0.25,
            'contrast_score': 0.25,
            'noise_level': 0.2
        }

        score = sum(metrics[key] * weights[key] for key in weights if key in metrics)
        return min(1.0, max(0.0, score))


class TrafficFeatureExtractor:
    """Extract traffic-specific features from camera frames."""

    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )

        # Traffic density thresholds
        self.density_thresholds = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.6
        }

    def extract_features(self, frame: ProcessedFrame) -> dict[str, Any]:
        """Extract traffic and environmental features."""

        image = frame.original_image
        features = {}

        # Vehicle density estimation
        features['vehicle_density'] = self._estimate_vehicle_density(image)
        features['congestion_level'] = self._classify_congestion(features['vehicle_density'])

        # Environmental conditions
        features['lighting_conditions'] = self._analyze_lighting(image)
        features['weather_conditions'] = self._estimate_weather(image)

        # Motion analysis
        features['motion_intensity'] = self._analyze_motion(image)

        # ROI-specific features
        if hasattr(frame, 'roi_boxes') and frame.roi_boxes:
            features['roi_analysis'] = self._analyze_rois(image, frame.roi_boxes)

        return features

    def _estimate_vehicle_density(self, image: np.ndarray) -> float:
        """Estimate vehicle density using background subtraction."""

        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(image)

        # Calculate density as percentage of foreground pixels
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        fg_pixels = np.sum(fg_mask > 0)

        density = fg_pixels / total_pixels
        return min(1.0, density)

    def _classify_congestion(self, density: float) -> str:
        """Classify traffic congestion level."""

        if density >= self.density_thresholds['high']:
            return 'high'
        elif density >= self.density_thresholds['medium']:
            return 'medium'
        elif density >= self.density_thresholds['low']:
            return 'low'
        else:
            return 'free_flow'

    def _analyze_lighting(self, image: np.ndarray) -> str:
        """Analyze lighting conditions."""

        # Convert to HSV for better light analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        brightness = np.mean(hsv[:, :, 2])  # V channel

        if brightness > 180:
            return 'bright'
        elif brightness > 100:
            return 'normal'
        elif brightness > 50:
            return 'dim'
        else:
            return 'dark'

    def _estimate_weather(self, image: np.ndarray) -> str:
        """Estimate weather conditions from image characteristics."""

        # Simple weather estimation based on image properties
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)

        # Basic weather classification
        if std_intensity < 30 and mean_intensity < 100:
            return 'foggy'
        elif mean_intensity < 80:
            return 'overcast'
        elif std_intensity > 60:
            return 'clear'
        else:
            return 'partly_cloudy'

    def _analyze_motion(self, image: np.ndarray) -> float:
        """Analyze motion intensity in the scene."""

        # Apply Gaussian blur and calculate optical flow
        if hasattr(self, 'previous_frame'):
            gray_current = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray_previous = self.previous_frame

            # Calculate optical flow
            flow = cv2.calcOpticalFlowPyrLK(
                gray_previous, gray_current,
                None, None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )

            # Calculate motion intensity
            if flow[0] is not None:
                motion_magnitude = np.mean(np.linalg.norm(flow[1], axis=2))
                self.previous_frame = gray_current
                return min(1.0, motion_magnitude / 10.0)

        # Store current frame for next analysis
        self.previous_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return 0.0

    def _analyze_rois(self, image: np.ndarray, roi_boxes: list[tuple[int, int, int, int]]) -> dict[str, Any]:
        """Analyze specific regions of interest."""

        roi_analysis = {}

        for i, (x, y, w, h) in enumerate(roi_boxes):
            roi_id = f"roi_{i}"

            # Extract ROI
            roi = image[y:y+h, x:x+w]

            if roi.size > 0:
                # Analyze ROI
                roi_density = self._estimate_vehicle_density(roi)
                roi_brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY))

                roi_analysis[roi_id] = {
                    'density': roi_density,
                    'brightness': roi_brightness,
                    'congestion': self._classify_congestion(roi_density)
                }

        return roi_analysis


class StreamProcessor:
    """Main stream processing engine with async pipeline."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

        # Stream management
        self.active_streams: dict[str, CameraStream] = {}
        self.processing_queues: dict[str, asyncio.Queue] = {}

        # Processing components
        self.quality_analyzer = ImageQualityAnalyzer()
        self.feature_extractor = TrafficFeatureExtractor()

        # Data storage
        self.processed_frames = deque(maxlen=10000)
        self.quality_stats = defaultdict(lambda: deque(maxlen=1000))

        # Kafka integration
        self.kafka_bootstrap_servers = config.get('kafka_servers', ['localhost:9092'])
        self.input_topic = config.get('input_topic', 'camera_frames')
        self.output_topic = config.get('output_topic', 'processed_frames')

        # Redis for caching
        self.redis_url = config.get('redis_url', 'redis://localhost:6379')

        # Processing settings
        self.max_concurrent_streams = config.get('max_concurrent_streams', 1000)
        self.quality_threshold = config.get('quality_threshold', 0.7)
        self.processing_timeout = config.get('processing_timeout', 5.0)

        # Performance tracking
        self.performance_metrics = {
            'frames_processed': 0,
            'frames_rejected': 0,
            'avg_processing_time': 0.0,
            'throughput_fps': 0.0,
            'error_count': 0
        }

        self.kafka_consumer = None
        self.kafka_producer = None
        self.redis_client = None

        logger.info("Stream processor initialized")

    async def start(self):
        """Start the stream processing pipeline."""

        # Initialize connections
        if KAFKA_AVAILABLE:
            await self._setup_kafka()

        if REDIS_AVAILABLE:
            await self._setup_redis()

        # Start processing tasks
        asyncio.create_task(self._consume_frames())
        asyncio.create_task(self._monitor_performance())

        logger.info("Stream processor started")

    async def _setup_kafka(self):
        """Setup Kafka consumer and producer."""

        self.kafka_consumer = AIOKafkaConsumer(
            self.input_topic,
            bootstrap_servers=self.kafka_bootstrap_servers,
            group_id='stream_processor',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            max_poll_records=100
        )

        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
        )

        await self.kafka_consumer.start()
        await self.kafka_producer.start()

        logger.info("Kafka connections established")

    async def _setup_redis(self):
        """Setup Redis connection."""

        self.redis_client = await aioredis.from_url(
            self.redis_url,
            decode_responses=True
        )

        logger.info("Redis connection established")

    async def _consume_frames(self):
        """Consume and process frames from Kafka."""

        if not self.kafka_consumer:
            logger.warning("Kafka consumer not available")
            return

        try:
            async for message in self.kafka_consumer:
                try:
                    frame_data = message.value
                    await self._process_frame_message(frame_data)

                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    self.performance_metrics['error_count'] += 1

        except Exception as e:
            logger.error(f"Frame consumption error: {e}")

    async def _process_frame_message(self, frame_data: dict[str, Any]):
        """Process individual frame message."""

        start_time = time.time()

        try:
            # Parse frame data
            frame = await self._parse_frame_data(frame_data)
            if not frame:
                return

            # Process the frame through pipeline
            processed_frame = await self._process_frame_pipeline(frame)

            # Store processed frame
            if processed_frame.validation_passed:
                self.processed_frames.append(processed_frame)
                await self._emit_processed_frame(processed_frame)

                self.performance_metrics['frames_processed'] += 1
            else:
                self.performance_metrics['frames_rejected'] += 1

            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_processing_metrics(processing_time)

        except Exception as e:
            logger.error(f"Frame pipeline error: {e}")
            self.performance_metrics['error_count'] += 1

    async def _parse_frame_data(self, frame_data: dict[str, Any]) -> ProcessedFrame | None:
        """Parse incoming frame data."""

        try:
            # Decode image data
            if 'image_base64' in frame_data:
                image_data = base64.b64decode(frame_data['image_base64'])
                image = np.array(Image.open(io.BytesIO(image_data)))
            elif 'image_array' in frame_data:
                image = np.array(frame_data['image_array'], dtype=np.uint8)
            else:
                logger.warning("No valid image data found in frame")
                return None

            # Create processed frame
            frame = ProcessedFrame(
                frame_id=frame_data['frame_id'],
                camera_id=frame_data['camera_id'],
                timestamp=frame_data.get('timestamp', time.time()),
                original_image=image,
                source_hash=hashlib.md5(image.tobytes()).hexdigest()[:16]
            )

            return frame

        except Exception as e:
            logger.error(f"Frame parsing error: {e}")
            return None

    async def _process_frame_pipeline(self, frame: ProcessedFrame) -> ProcessedFrame:
        """Process frame through the complete pipeline."""

        pipeline_start = time.time()

        # Stage 1: Image quality analysis
        frame.processing_stage = ProcessingStage.VALIDATION
        quality_metrics = self.quality_analyzer.analyze_quality(frame.original_image)

        frame.quality_score = quality_metrics['quality_score']
        frame.blur_score = quality_metrics['blur_score']
        frame.brightness_score = quality_metrics['brightness_score']
        frame.contrast_score = quality_metrics['contrast_score']
        frame.noise_level = quality_metrics['noise_level']

        # Stage 2: Quality validation
        frame.validation_passed = frame.quality_score >= self.quality_threshold

        if not frame.validation_passed:
            frame.processing_time_ms = (time.time() - pipeline_start) * 1000
            return frame

        # Stage 3: Feature extraction
        frame.processing_stage = ProcessingStage.FEATURE_EXTRACTION
        features = self.feature_extractor.extract_features(frame)

        frame.vehicle_density = features.get('vehicle_density', 0.0)
        frame.congestion_level = features.get('congestion_level', 'unknown')
        frame.weather_conditions = features.get('weather_conditions', 'unknown')
        frame.lighting_conditions = features.get('lighting_conditions', 'unknown')
        frame.roi_features = features.get('roi_analysis', {})

        # Stage 4: Image preprocessing for ML
        frame.processing_stage = ProcessingStage.QUALITY_CONTROL
        frame.processed_image = self._preprocess_for_ml(frame.original_image)
        frame.thumbnail = self._create_thumbnail(frame.original_image)

        # Final stage
        frame.processing_stage = ProcessingStage.OUTPUT
        frame.processing_time_ms = (time.time() - pipeline_start) * 1000

        return frame

    def _preprocess_for_ml(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ML inference."""

        # Resize to standard ML input size
        target_size = (640, 640)

        # Maintain aspect ratio with padding
        h, w = image.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)

        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        pad_w = (target_size[0] - new_w) // 2
        pad_h = (target_size[1] - new_h) // 2

        padded = np.full((*target_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        return padded

    def _create_thumbnail(self, image: np.ndarray, size: tuple[int, int] = (128, 128)) -> np.ndarray:
        """Create thumbnail for storage and display."""

        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    async def _emit_processed_frame(self, frame: ProcessedFrame):
        """Emit processed frame to output systems."""

        # Prepare output data
        output_data = {
            'frame_id': frame.frame_id,
            'camera_id': frame.camera_id,
            'timestamp': frame.timestamp,
            'quality_score': frame.quality_score,
            'vehicle_density': frame.vehicle_density,
            'congestion_level': frame.congestion_level,
            'weather_conditions': frame.weather_conditions,
            'lighting_conditions': frame.lighting_conditions,
            'processing_time_ms': frame.processing_time_ms,
            'validation_passed': frame.validation_passed,
            'source_hash': frame.source_hash
        }

        # Send to Kafka
        if self.kafka_producer:
            await self.kafka_producer.send_and_wait(self.output_topic, output_data)

        # Cache in Redis
        if self.redis_client:
            cache_key = f"frame:{frame.camera_id}:{frame.frame_id}"
            await self.redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(output_data, default=str)
            )

        # Update quality statistics
        self.quality_stats[frame.camera_id].append({
            'timestamp': frame.timestamp,
            'quality_score': frame.quality_score,
            'processing_time': frame.processing_time_ms
        })

    def _update_processing_metrics(self, processing_time_ms: float):
        """Update processing performance metrics."""

        # Update average processing time
        current_avg = self.performance_metrics['avg_processing_time']
        total_processed = self.performance_metrics['frames_processed'] + self.performance_metrics['frames_rejected']

        if total_processed > 0:
            self.performance_metrics['avg_processing_time'] = (
                (current_avg * (total_processed - 1) + processing_time_ms) / total_processed
            )

    async def _monitor_performance(self):
        """Monitor and log performance metrics."""

        last_frame_count = 0
        last_check_time = time.time()

        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                current_time = time.time()
                current_frame_count = self.performance_metrics['frames_processed']

                # Calculate throughput
                time_diff = current_time - last_check_time
                frame_diff = current_frame_count - last_frame_count

                if time_diff > 0:
                    throughput = frame_diff / time_diff
                    self.performance_metrics['throughput_fps'] = throughput

                # Log metrics
                logger.info(
                    f"Processing metrics: "
                    f"throughput={throughput:.1f}fps, "
                    f"avg_time={self.performance_metrics['avg_processing_time']:.1f}ms, "
                    f"processed={current_frame_count}, "
                    f"rejected={self.performance_metrics['frames_rejected']}, "
                    f"errors={self.performance_metrics['error_count']}"
                )

                last_frame_count = current_frame_count
                last_check_time = current_time

            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

    async def register_stream(self, stream_config: dict[str, Any]) -> bool:
        """Register new camera stream."""

        try:
            stream = CameraStream(
                camera_id=stream_config['camera_id'],
                location=stream_config['location'],
                coordinates=tuple(stream_config['coordinates']),
                resolution=tuple(stream_config.get('resolution', [1920, 1080])),
                fps=stream_config.get('fps', 30),
                quality_threshold=stream_config.get('quality_threshold', self.quality_threshold)
            )

            self.active_streams[stream.camera_id] = stream
            self.processing_queues[stream.camera_id] = asyncio.Queue(maxsize=100)

            logger.info(f"Registered camera stream: {stream.camera_id}")
            return True

        except Exception as e:
            logger.error(f"Stream registration error: {e}")
            return False

    def get_stream_status(self, camera_id: str) -> dict[str, Any] | None:
        """Get status of specific camera stream."""

        if camera_id not in self.active_streams:
            return None

        stream = self.active_streams[camera_id]
        quality_history = list(self.quality_stats[camera_id])

        return {
            'camera_id': stream.camera_id,
            'status': stream.status.value,
            'location': stream.location,
            'resolution': stream.resolution,
            'fps': stream.fps,
            'total_frames_processed': stream.total_frames_processed,
            'avg_processing_latency_ms': stream.avg_processing_latency_ms,
            'quality_score_avg': stream.quality_score_avg,
            'error_rate': stream.error_rate,
            'last_frame_time': stream.last_frame_time,
            'recent_quality_samples': len(quality_history)
        }

    def get_processing_stats(self) -> dict[str, Any]:
        """Get overall processing statistics."""

        return {
            'active_streams': len(self.active_streams),
            'performance_metrics': self.performance_metrics,
            'processed_frames_buffer': len(self.processed_frames),
            'quality_stats_cameras': len(self.quality_stats),
            'kafka_available': self.kafka_consumer is not None,
            'redis_available': self.redis_client is not None
        }

    async def stop(self):
        """Stop stream processing."""

        logger.info("Stopping stream processor")

        if self.kafka_consumer:
            await self.kafka_consumer.stop()

        if self.kafka_producer:
            await self.kafka_producer.stop()

        if self.redis_client:
            await self.redis_client.close()

        logger.info("Stream processor stopped")


# Factory function
async def create_stream_processor(config_path: str | Path = None) -> StreamProcessor:
    """Create and initialize stream processor."""

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'kafka_servers': ['localhost:9092'],
            'input_topic': 'camera_frames',
            'output_topic': 'processed_frames',
            'redis_url': 'redis://localhost:6379',
            'max_concurrent_streams': 1000,
            'quality_threshold': 0.7,
            'processing_timeout': 5.0
        }

    processor = StreamProcessor(config)
    await processor.start()

    return processor
