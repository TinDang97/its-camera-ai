"""
Production ML Pipeline for ITS Camera AI Traffic Monitoring System.

This module implements a comprehensive production ML pipeline for continuous improvement
of traffic monitoring models with sub-100ms latency requirements and 1000+ concurrent
camera streams support.

Features:
1. Real-time data ingestion from camera streams with quality validation
2. Continuous learning with federated training across edge nodes
3. Automated model validation, versioning, and deployment
4. A/B testing framework with statistical significance
5. Performance monitoring with drift detection and alerting
6. MLOps integration with experiment tracking and CI/CD

Architecture:
- Data Pipeline: Stream processing with Apache Kafka and data validation
- Training Pipeline: Distributed training with Ray and federated learning
- Model Management: Registry with versioning and automated rollback
- Serving Pipeline: High-performance inference with TensorRT optimization
- Monitoring Pipeline: Real-time dashboards and automated alerting

Performance Targets:
- Sub-100ms inference latency (95th percentile)
- 1000+ FPS aggregate throughput across cameras
- 95%+ vehicle detection accuracy
- 99.9% system uptime with automated failover
"""

import asyncio
import hashlib
import json
import logging
import shutil
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ML and monitoring components
from .inference_optimizer import (
    DetectionResult,
    InferenceConfig,
    ModelType,
    OptimizationBackend,
    OptimizedInferenceEngine,
)
from .model_pipeline import (
    ABTestingFramework,
    DeploymentPipeline,
    DeploymentStrategy,
    ModelValidationResult,
    ModelValidator,
    ModelVersion,
)
from .production_monitoring import (
    Alert,
    AlertSeverity,
    DriftType,
    ProductionDashboard,
)

# External integrations
try:
    import mlflow
    import mlflow.pytorch

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import ray  # noqa: F401
    from ray import tune  # noqa: F401
    from ray.air import session  # noqa: F401

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import kafka  # noqa: F401
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


class PipelineStatus(Enum):
    """ML Pipeline status."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    TRAINING = "training"
    DEPLOYING = "deploying"
    MONITORING = "monitoring"
    ERROR = "error"
    STOPPED = "stopped"


class DataQuality(Enum):
    """Data quality levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CORRUPTED = "corrupted"


@dataclass
class StreamingDataSample:
    """Streaming data sample from camera feeds."""

    # Identifiers
    sample_id: str
    camera_id: str
    timestamp: float

    # Data
    image_data: np.ndarray
    metadata: dict[str, Any]

    # Quality metrics
    quality_score: float
    blur_score: float
    brightness_score: float
    resolution: tuple[int, int]

    # Labels (if available)
    ground_truth: dict[str, Any] | None = None

    # Processing info
    ingestion_latency_ms: float = 0.0
    preprocessing_time_ms: float = 0.0


@dataclass
class TrainingBatch:
    """Training batch for continuous learning."""

    batch_id: str
    samples: list[StreamingDataSample]
    model_version: str
    training_type: str  # "incremental", "federated", "full_retrain"

    # Metadata
    created_at: float
    quality_distribution: dict[DataQuality, int]
    camera_distribution: dict[str, int]

    # Training configuration
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 1


@dataclass
class ExperimentResult:
    """ML experiment tracking result."""

    experiment_id: str
    run_id: str
    model_version: str

    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float

    # Efficiency metrics
    inference_latency_ms: float
    training_time_minutes: float
    model_size_mb: float

    # Resource usage
    gpu_memory_peak_gb: float
    cpu_utilization_avg: float

    # Configuration
    hyperparameters: dict[str, Any]
    artifacts: dict[str, str]  # artifact_name -> path

    # Status
    status: str = "completed"
    created_at: float = field(default_factory=time.time)


class DataIngestionPipeline:
    """Real-time data ingestion from camera streams with quality validation."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.kafka_bootstrap_servers = config.get("kafka_servers", ["localhost:9092"])
        self.redis_url = config.get("redis_url", "redis://localhost:6379")
        self.data_quality_threshold = config.get("quality_threshold", 0.7)

        # Data buffers
        self.streaming_buffer = deque(maxlen=10000)  # Hold recent samples
        self.training_buffer = deque(maxlen=50000)  # Training data accumulation

        # Quality tracking
        self.quality_stats = defaultdict(int)
        self.camera_stats = defaultdict(lambda: {"samples": 0, "quality_avg": 0.0})

        # Processing components
        self.quality_validator = DataQualityValidator()

        # Async connections
        self.kafka_consumer = None
        self.kafka_producer = None
        self.redis_client = None

        # Processing metrics
        self.ingestion_rate = 0.0
        self.processing_latency = deque(maxlen=1000)
        self.last_metrics_time = time.time()

        logger.info("Data ingestion pipeline initialized")

    async def start(self):
        """Start data ingestion pipeline."""
        if KAFKA_AVAILABLE:
            await self._setup_kafka()

        if REDIS_AVAILABLE:
            await self._setup_redis()

        # Start processing tasks
        asyncio.create_task(self._process_camera_streams())
        asyncio.create_task(self._generate_training_batches())
        asyncio.create_task(self._update_metrics())

        logger.info("Data ingestion pipeline started")

    async def _setup_kafka(self):
        """Setup Kafka consumer and producer."""
        self.kafka_consumer = AIOKafkaConsumer(
            "camera_streams",
            bootstrap_servers=self.kafka_bootstrap_servers,
            group_id="ml_pipeline",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )

        await self.kafka_consumer.start()
        await self.kafka_producer.start()

    async def _setup_redis(self):
        """Setup Redis connection for caching."""
        self.redis_client = await aioredis.from_url(self.redis_url)

    async def _process_camera_streams(self):
        """Process incoming camera stream data."""
        if not self.kafka_consumer:
            logger.warning("Kafka consumer not available, using synthetic data")
            await self._generate_synthetic_data()
            return

        try:
            async for message in self.kafka_consumer:
                processing_start = time.time()

                try:
                    # Parse message
                    data = message.value
                    sample = await self._parse_stream_message(data)

                    if sample:
                        # Validate quality
                        quality_result = await self.quality_validator.validate_sample(
                            sample
                        )
                        sample.quality_score = quality_result["overall_score"]

                        # Update statistics
                        self._update_ingestion_stats(sample)

                        # Add to buffers based on quality
                        if sample.quality_score >= self.data_quality_threshold:
                            self.streaming_buffer.append(sample)

                            # High quality samples go to training buffer
                            if sample.quality_score >= 0.85:
                                self.training_buffer.append(sample)

                    # Track processing latency
                    processing_time = (time.time() - processing_start) * 1000
                    self.processing_latency.append(processing_time)

                except Exception as e:
                    logger.error(f"Error processing camera stream message: {e}")

        except Exception as e:
            logger.error(f"Camera stream processing error: {e}")

    async def _generate_synthetic_data(self):
        """Generate synthetic camera data for development/testing."""
        camera_ids = [f"cam_{i:03d}" for i in range(10)]

        while True:
            try:
                for camera_id in camera_ids:
                    # Generate synthetic image data
                    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

                    sample = StreamingDataSample(
                        sample_id=f"{camera_id}_{int(time.time() * 1000)}",
                        camera_id=camera_id,
                        timestamp=time.time(),
                        image_data=image,
                        metadata={
                            "location": f"intersection_{camera_id}",
                            "weather": "clear",
                        },
                        quality_score=np.random.uniform(0.6, 0.95),
                        blur_score=np.random.uniform(0.7, 0.95),
                        brightness_score=np.random.uniform(0.8, 0.95),
                        resolution=(640, 640),
                    )

                    # Add to buffers
                    if sample.quality_score >= self.data_quality_threshold:
                        self.streaming_buffer.append(sample)
                        if sample.quality_score >= 0.85:
                            self.training_buffer.append(sample)

                    self._update_ingestion_stats(sample)

                # Control ingestion rate (simulate 30 FPS per camera)
                await asyncio.sleep(1.0 / 30.0)

            except Exception as e:
                logger.error(f"Synthetic data generation error: {e}")
                await asyncio.sleep(1.0)

    async def _parse_stream_message(
        self, data: dict[str, Any]
    ) -> StreamingDataSample | None:
        """Parse incoming stream message to data sample."""
        try:
            # Decode image data (base64 or binary)
            image_data = self._decode_image_data(data["image"])

            sample = StreamingDataSample(
                sample_id=data["sample_id"],
                camera_id=data["camera_id"],
                timestamp=data["timestamp"],
                image_data=image_data,
                metadata=data.get("metadata", {}),
                quality_score=0.0,  # Will be calculated
                blur_score=0.0,
                brightness_score=0.0,
                resolution=tuple(data.get("resolution", [640, 640])),
                ground_truth=data.get("ground_truth"),
            )

            return sample

        except Exception as e:
            logger.error(f"Error parsing stream message: {e}")
            return None

    def _decode_image_data(self, encoded_data: Any) -> np.ndarray:
        """Decode image data from various formats."""
        # Placeholder - in production this would handle base64, binary, etc.
        if isinstance(encoded_data, list | np.ndarray):
            return np.array(encoded_data, dtype=np.uint8)

        # Generate placeholder image for development
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    def _update_ingestion_stats(self, sample: StreamingDataSample):
        """Update ingestion statistics."""
        quality_level = (
            DataQuality.HIGH
            if sample.quality_score >= 0.9
            else DataQuality.MEDIUM
            if sample.quality_score >= 0.7
            else DataQuality.LOW
        )

        self.quality_stats[quality_level] += 1

        # Update camera statistics
        cam_stats = self.camera_stats[sample.camera_id]
        cam_stats["samples"] += 1
        cam_stats["quality_avg"] = (
            cam_stats["quality_avg"] * (cam_stats["samples"] - 1) + sample.quality_score
        ) / cam_stats["samples"]

    async def _generate_training_batches(self):
        """Generate training batches from accumulated data."""
        while True:
            try:
                if len(self.training_buffer) >= 100:  # Minimum batch size
                    # Create training batch
                    batch_samples = []
                    batch_size = min(100, len(self.training_buffer))

                    for _ in range(batch_size):
                        if self.training_buffer:
                            batch_samples.append(self.training_buffer.popleft())

                    if batch_samples:
                        batch = TrainingBatch(
                            batch_id=f"batch_{int(time.time())}",
                            samples=batch_samples,
                            model_version="current",
                            training_type="incremental",
                            created_at=time.time(),
                            quality_distribution=self._get_quality_distribution(
                                batch_samples
                            ),
                            camera_distribution=self._get_camera_distribution(
                                batch_samples
                            ),
                        )

                        # Send to training pipeline
                        await self._emit_training_batch(batch)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Training batch generation error: {e}")
                await asyncio.sleep(60)

    def _get_quality_distribution(
        self, samples: list[StreamingDataSample]
    ) -> dict[DataQuality, int]:
        """Get quality distribution for samples."""
        distribution = dict.fromkeys(DataQuality, 0)

        for sample in samples:
            if sample.quality_score >= 0.9:
                distribution[DataQuality.HIGH] += 1
            elif sample.quality_score >= 0.7:
                distribution[DataQuality.MEDIUM] += 1
            else:
                distribution[DataQuality.LOW] += 1

        return distribution

    def _get_camera_distribution(
        self, samples: list[StreamingDataSample]
    ) -> dict[str, int]:
        """Get camera distribution for samples."""
        distribution = defaultdict(int)
        for sample in samples:
            distribution[sample.camera_id] += 1
        return dict(distribution)

    async def _emit_training_batch(self, batch: TrainingBatch):
        """Emit training batch to training pipeline."""
        if self.kafka_producer:
            await self.kafka_producer.send_and_wait(
                "training_batches",
                {
                    "batch_id": batch.batch_id,
                    "sample_count": len(batch.samples),
                    "quality_distribution": batch.quality_distribution,
                    "camera_distribution": batch.camera_distribution,
                    "created_at": batch.created_at,
                },
            )

        logger.info(
            f"Generated training batch {batch.batch_id} with {len(batch.samples)} samples"
        )

    async def _update_metrics(self):
        """Update ingestion metrics periodically."""
        while True:
            try:
                current_time = time.time()

                # Calculate ingestion rate
                if hasattr(self, "last_metrics_time"):
                    time_diff = current_time - self.last_metrics_time
                    if time_diff > 0:
                        samples_in_period = sum(self.quality_stats.values())
                        self.ingestion_rate = samples_in_period / time_diff

                # Log metrics
                if self.processing_latency:
                    avg_latency = np.mean(self.processing_latency)
                    p95_latency = np.percentile(self.processing_latency, 95)

                    logger.info(
                        f"Ingestion metrics: rate={self.ingestion_rate:.1f} samples/sec, "
                        f"latency_avg={avg_latency:.1f}ms, latency_p95={p95_latency:.1f}ms"
                    )

                # Reset counters
                self.quality_stats.clear()
                self.last_metrics_time = current_time

                await asyncio.sleep(60)  # Update every minute

            except Exception as e:
                logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(60)

    def get_buffer_stats(self) -> dict[str, Any]:
        """Get current buffer statistics."""
        return {
            "streaming_buffer_size": len(self.streaming_buffer),
            "training_buffer_size": len(self.training_buffer),
            "ingestion_rate_per_sec": self.ingestion_rate,
            "avg_processing_latency_ms": (
                np.mean(self.processing_latency) if self.processing_latency else 0
            ),
            "camera_count": len(self.camera_stats),
            "quality_stats": dict(self.quality_stats),
        }


class DataQualityValidator:
    """Validate data quality for camera streams."""

    def __init__(self):
        self.blur_threshold = 0.1
        self.brightness_range = (30, 220)
        self.min_resolution = (320, 240)

    async def validate_sample(self, sample: StreamingDataSample) -> dict[str, Any]:
        """Comprehensive data quality validation."""

        # Image quality checks
        blur_score = self._calculate_blur_score(sample.image_data)
        brightness_score = self._calculate_brightness_score(sample.image_data)
        resolution_score = self._calculate_resolution_score(sample.resolution)

        # Metadata quality
        metadata_score = self._validate_metadata(sample.metadata)

        # Overall score (weighted average)
        overall_score = (
            blur_score * 0.3
            + brightness_score * 0.3
            + resolution_score * 0.2
            + metadata_score * 0.2
        )

        return {
            "overall_score": overall_score,
            "blur_score": blur_score,
            "brightness_score": brightness_score,
            "resolution_score": resolution_score,
            "metadata_score": metadata_score,
            "quality_level": self._determine_quality_level(overall_score),
        }

    def _calculate_blur_score(self, image: np.ndarray) -> float:
        """Calculate image blur score using Laplacian variance."""
        try:
            import cv2

            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Normalize to 0-1 score (higher = less blurry)
            score = min(1.0, laplacian_var / 1000)
            return score
        except Exception:
            # Fallback: random score for development
            return np.random.uniform(0.7, 0.95)

    def _calculate_brightness_score(self, image: np.ndarray) -> float:
        """Calculate image brightness score."""
        try:
            # Convert to grayscale and get mean brightness
            gray = np.mean(image)

            # Score based on optimal brightness range
            if self.brightness_range[0] <= gray <= self.brightness_range[1]:
                score = 1.0
            else:
                # Penalize images outside optimal range
                distance = min(
                    abs(gray - self.brightness_range[0]),
                    abs(gray - self.brightness_range[1]),
                )
                score = max(0.0, 1.0 - (distance / 100))

            return score
        except Exception:
            return np.random.uniform(0.8, 0.95)

    def _calculate_resolution_score(self, resolution: tuple[int, int]) -> float:
        """Calculate resolution adequacy score."""
        width, height = resolution
        min_width, min_height = self.min_resolution

        if width >= min_width and height >= min_height:
            # Bonus for higher resolutions
            area_ratio = (width * height) / (min_width * min_height)
            score = min(1.0, 0.8 + (area_ratio - 1.0) * 0.2)
        else:
            # Penalty for low resolution
            score = (width / min_width) * (height / min_height)

        return score

    def _validate_metadata(self, metadata: dict[str, Any]) -> float:
        """Validate metadata completeness and quality."""
        required_fields = ["location", "weather"]
        optional_fields = ["traffic_density", "lighting_conditions"]

        score = 0.0

        # Required fields
        for required_field in required_fields:
            if required_field in metadata and metadata[required_field]:
                score += 0.4

        # Optional fields (bonus)
        for optional_field in optional_fields:
            if optional_field in metadata and metadata[optional_field]:
                score += 0.1

        return min(1.0, score)

    def _determine_quality_level(self, score: float) -> DataQuality:
        """Determine quality level from score."""
        if score >= 0.9:
            return DataQuality.HIGH
        elif score >= 0.7:
            return DataQuality.MEDIUM
        elif score >= 0.4:
            return DataQuality.LOW
        else:
            return DataQuality.CORRUPTED


class ContinuousTrainingPipeline:
    """Continuous learning with distributed and federated training."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_registry = ModelRegistry(config.get("model_registry", {}))
        self.experiment_tracker = ExperimentTracker(config.get("mlflow", {}))

        # Training configuration
        self.base_model_path = Path(config.get("base_model_path", "models/yolo11s.pt"))
        self.output_dir = Path(config.get("output_dir", "training_outputs"))
        self.output_dir.mkdir(exist_ok=True)

        # Distributed training
        self.use_ray = config.get("use_ray", True) and RAY_AVAILABLE
        self.num_workers = config.get("num_workers", 4)
        self.resources_per_worker = config.get(
            "resources_per_worker", {"cpu": 1, "gpu": 0.25}
        )

        # Federated learning
        self.federated_enabled = config.get("federated_enabled", False)
        self.federation_rounds = config.get("federation_rounds", 5)

        # Training buffers
        self.training_queue = asyncio.Queue(maxsize=100)
        self.active_training_jobs = {}

        # Performance tracking
        self.training_history = deque(maxlen=100)

        if self.use_ray:
            self._initialize_ray()

        logger.info("Continuous training pipeline initialized")

    def _initialize_ray(self):
        """Initialize Ray for distributed training."""
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.config.get("ray_cpus", 8),
                num_gpus=self.config.get("ray_gpus", 2),
            )

    async def start(self):
        """Start continuous training pipeline."""
        asyncio.create_task(self._process_training_batches())
        asyncio.create_task(self._monitor_training_jobs())

        if self.federated_enabled:
            asyncio.create_task(self._federated_learning_coordinator())

        logger.info("Continuous training pipeline started")

    async def _process_training_batches(self):
        """Process incoming training batches."""
        while True:
            try:
                # Get training batch (this would come from data ingestion)
                batch = await self.training_queue.get()

                # Determine training strategy
                training_strategy = self._determine_training_strategy(batch)

                # Launch training job
                job_id = await self._launch_training_job(batch, training_strategy)

                logger.info(
                    f"Launched training job {job_id} with {len(batch.samples)} samples"
                )

            except Exception as e:
                logger.error(f"Training batch processing error: {e}")
                await asyncio.sleep(10)

    def _determine_training_strategy(self, batch: TrainingBatch) -> str:
        """Determine optimal training strategy based on batch characteristics."""

        # Check data diversity
        camera_count = len(batch.camera_distribution)
        sample_count = len(batch.samples)

        if sample_count >= 1000 and camera_count >= 5:
            return "full_retrain"
        elif sample_count >= 100:
            return "incremental"
        else:
            return "fine_tune"

    async def _launch_training_job(self, batch: TrainingBatch, strategy: str) -> str:
        """Launch training job with specified strategy."""

        job_id = f"training_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # Create training configuration
        training_config = {
            "job_id": job_id,
            "strategy": strategy,
            "batch_id": batch.batch_id,
            "sample_count": len(batch.samples),
            "base_model": str(self.base_model_path),
            "output_dir": str(self.output_dir / job_id),
            "hyperparameters": self._get_hyperparameters(strategy),
            "resources": self.resources_per_worker,
        }

        if self.use_ray:
            # Launch distributed training with Ray
            job_handle = self._launch_ray_training.remote(training_config, batch)
            self.active_training_jobs[job_id] = {
                "handle": job_handle,
                "start_time": time.time(),
                "config": training_config,
            }
        else:
            # Launch local training
            asyncio.create_task(self._run_local_training(training_config, batch))

        return job_id

    @ray.remote
    def _launch_ray_training(
        self, config: dict[str, Any], batch: TrainingBatch
    ) -> dict[str, Any]:
        """Ray remote training function."""
        return asyncio.run(self._run_training_job(config, batch))

    async def _run_local_training(self, config: dict[str, Any], batch: TrainingBatch):
        """Run training job locally."""
        try:
            result = await self._run_training_job(config, batch)
            await self._handle_training_completion(config["job_id"], result)
        except Exception as e:
            await self._handle_training_error(config["job_id"], str(e))

    async def _run_training_job(
        self, config: dict[str, Any], batch: TrainingBatch
    ) -> dict[str, Any]:
        """Core training job execution."""

        job_id = config["job_id"]
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        try:
            # Start experiment tracking
            experiment_id = await self.experiment_tracker.start_experiment(
                f"continuous_training_{job_id}", config["hyperparameters"]
            )

            # Load base model
            from ultralytics import YOLO

            model = YOLO(config["base_model"])

            # Prepare training dataset
            dataset_path = await self._prepare_training_dataset(batch, output_dir)

            # Training configuration
            train_args = {
                "data": str(dataset_path),
                "epochs": config["hyperparameters"]["epochs"],
                "imgsz": config["hyperparameters"]["image_size"],
                "batch": config["hyperparameters"]["batch_size"],
                "lr0": config["hyperparameters"]["learning_rate"],
                "project": str(output_dir),
                "name": "training_run",
                "save": True,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }

            # Run training
            results = model.train(**train_args)

            # Save trained model
            model_path = output_dir / "best.pt"

            # Calculate metrics
            training_time = time.time() - start_time
            model_size = (
                model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0
            )

            # Validation metrics (from training results)
            metrics = {
                "accuracy": float(results.results_dict.get("metrics/mAP50", 0.0)),
                "precision": float(results.results_dict.get("metrics/precision", 0.0)),
                "recall": float(results.results_dict.get("metrics/recall", 0.0)),
                "training_time_minutes": training_time / 60,
                "model_size_mb": model_size,
                "sample_count": len(batch.samples),
            }

            # Log experiment results
            await self.experiment_tracker.log_results(
                experiment_id, metrics, str(model_path)
            )

            return {
                "success": True,
                "job_id": job_id,
                "model_path": str(model_path),
                "metrics": metrics,
                "experiment_id": experiment_id,
            }

        except Exception as e:
            logger.error(f"Training job {job_id} failed: {e}")
            return {"success": False, "job_id": job_id, "error": str(e)}

    async def _prepare_training_dataset(
        self, batch: TrainingBatch, output_dir: Path
    ) -> Path:
        """Prepare training dataset in YOLO format."""

        dataset_dir = output_dir / "dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

        # Save images and labels
        for i, sample in enumerate(batch.samples):
            # Save image
            image_filename = f"image_{i:06d}.jpg"
            image_path = dataset_dir / "images" / "train" / image_filename

            # Convert numpy array to image and save
            from PIL import Image

            if sample.image_data.dtype != np.uint8:
                image_data = (sample.image_data * 255).astype(np.uint8)
            else:
                image_data = sample.image_data

            image = Image.fromarray(image_data)
            image.save(image_path)

            # Save label (if available)
            label_filename = f"image_{i:06d}.txt"
            label_path = dataset_dir / "labels" / "train" / label_filename

            if sample.ground_truth:
                await self._save_yolo_label(sample.ground_truth, label_path)
            else:
                # Create empty label file
                label_path.write_text("")

        # Create dataset YAML file
        dataset_yaml = dataset_dir / "data.yaml"
        yaml_content = f"""
train: {dataset_dir}/images/train
val: {dataset_dir}/images/train  # Use same for validation in continuous learning

nc: 5  # number of classes
names: ['vehicle', 'car', 'truck', 'bus', 'motorcycle']
"""
        dataset_yaml.write_text(yaml_content.strip())

        return dataset_yaml

    async def _save_yolo_label(self, ground_truth: dict[str, Any], label_path: Path):
        """Save ground truth in YOLO label format."""
        labels = []

        if "boxes" in ground_truth and "classes" in ground_truth:
            boxes = ground_truth["boxes"]
            classes = ground_truth["classes"]

            for box, cls in zip(boxes, classes, strict=False):
                # Convert to YOLO format: class x_center y_center width height (normalized)
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                width = box[2] - box[0]
                height = box[3] - box[1]

                labels.append(
                    f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

        label_path.write_text("\n".join(labels))

    def _get_hyperparameters(self, strategy: str) -> dict[str, Any]:
        """Get hyperparameters based on training strategy."""

        base_params = {
            "image_size": 640,
            "batch_size": 16,
            "learning_rate": 0.001,
            "epochs": 10,
            "optimizer": "AdamW",
            "weight_decay": 0.0005,
        }

        if strategy == "incremental":
            base_params.update(
                {
                    "epochs": 3,
                    "learning_rate": 0.0001,  # Lower LR for incremental
                    "freeze": 10,  # Freeze first 10 layers
                }
            )
        elif strategy == "fine_tune":
            base_params.update(
                {
                    "epochs": 1,
                    "learning_rate": 0.00005,  # Very low LR for fine-tuning
                    "freeze": 15,  # Freeze more layers
                }
            )
        elif strategy == "full_retrain":
            base_params.update(
                {
                    "epochs": 20,
                    "learning_rate": 0.001,
                    "freeze": 0,  # Don't freeze any layers
                }
            )

        return base_params

    async def _monitor_training_jobs(self):
        """Monitor active training jobs and handle completion."""
        while True:
            try:
                completed_jobs = []

                for job_id, job_info in self.active_training_jobs.items():
                    if self.use_ray:
                        # Check Ray job status
                        try:
                            result = ray.get(job_info["handle"], timeout=0.1)
                            await self._handle_training_completion(job_id, result)
                            completed_jobs.append(job_id)
                        except ray.exceptions.GetTimeoutError:
                            # Job still running
                            continue
                        except Exception as e:
                            await self._handle_training_error(job_id, str(e))
                            completed_jobs.append(job_id)

                # Remove completed jobs
                for job_id in completed_jobs:
                    del self.active_training_jobs[job_id]

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Training job monitoring error: {e}")
                await asyncio.sleep(10)

    async def _handle_training_completion(self, job_id: str, result: dict[str, Any]):
        """Handle successful training job completion."""

        if result["success"]:
            logger.info(f"Training job {job_id} completed successfully")

            # Register new model version
            model_version = await self.model_registry.register_model(
                model_path=Path(result["model_path"]),
                metrics=result["metrics"],
                training_config=result.get("config", {}),
            )

            # Add to training history
            self.training_history.append(
                {
                    "job_id": job_id,
                    "completion_time": time.time(),
                    "model_version": model_version.version if model_version else None,
                    "metrics": result["metrics"],
                    "status": "success",
                }
            )

        else:
            await self._handle_training_error(
                job_id, result.get("error", "Unknown error")
            )

    async def _handle_training_error(self, job_id: str, error: str):
        """Handle training job errors."""

        logger.error(f"Training job {job_id} failed: {error}")

        # Add to training history
        self.training_history.append(
            {
                "job_id": job_id,
                "completion_time": time.time(),
                "model_version": None,
                "error": error,
                "status": "failed",
            }
        )

    async def _federated_learning_coordinator(self):
        """Coordinate federated learning across edge nodes."""

        logger.info("Starting federated learning coordinator")

        while True:
            try:
                # Check if we have enough local models for federation
                available_models = await self.model_registry.get_federated_models()

                if len(available_models) >= 2:  # Minimum for federation
                    logger.info(
                        f"Starting federation round with {len(available_models)} models"
                    )

                    # Perform model aggregation (FedAvg)
                    aggregated_model = await self._federated_averaging(available_models)

                    if aggregated_model:
                        # Register aggregated model
                        await self.model_registry.register_federated_model(
                            aggregated_model
                        )
                        logger.info("Federated model aggregation completed")

                # Wait for next federation round
                await asyncio.sleep(3600)  # Every hour

            except Exception as e:
                logger.error(f"Federated learning error: {e}")
                await asyncio.sleep(1800)  # Wait 30 minutes on error

    async def _federated_averaging(
        self, models: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Perform federated averaging of model weights."""

        try:
            # Load all model weights
            model_weights = []
            total_samples = 0

            for model_info in models:
                model = torch.load(model_info["path"], map_location="cpu")
                weights = model["model"].state_dict()
                sample_count = model_info.get("training_samples", 1000)

                model_weights.append((weights, sample_count))
                total_samples += sample_count

            if not model_weights:
                return None

            # Perform weighted averaging
            averaged_weights = {}

            # Initialize with first model's structure
            first_weights = model_weights[0][0]
            for key in first_weights:
                averaged_weights[key] = torch.zeros_like(first_weights[key])

            # Weighted average
            for weights, sample_count in model_weights:
                weight = sample_count / total_samples

                for key in weights:
                    if key in averaged_weights:
                        averaged_weights[key] += weights[key] * weight

            # Create aggregated model
            base_model = torch.load(models[0]["path"], map_location="cpu")
            base_model["model"].load_state_dict(averaged_weights)

            # Save aggregated model
            output_path = self.output_dir / f"federated_model_{int(time.time())}.pt"
            torch.save(base_model, output_path)

            return {
                "path": str(output_path),
                "aggregation_method": "FedAvg",
                "participating_models": len(models),
                "total_samples": total_samples,
            }

        except Exception as e:
            logger.error(f"Federated averaging error: {e}")
            return None

    async def add_training_batch(self, batch: TrainingBatch):
        """Add training batch to processing queue."""
        try:
            await self.training_queue.put(batch)
            logger.info(f"Added training batch {batch.batch_id} to queue")
        except asyncio.QueueFull:
            logger.warning(f"Training queue full, dropping batch {batch.batch_id}")

    def get_training_stats(self) -> dict[str, Any]:
        """Get training pipeline statistics."""

        recent_jobs = [
            job
            for job in self.training_history
            if job["completion_time"] > time.time() - 3600
        ]

        success_rate = len([j for j in recent_jobs if j["status"] == "success"]) / max(
            1, len(recent_jobs)
        )

        return {
            "active_jobs": len(self.active_training_jobs),
            "queue_size": self.training_queue.qsize(),
            "recent_jobs_1h": len(recent_jobs),
            "success_rate_1h": success_rate,
            "total_completed": len(self.training_history),
            "ray_initialized": ray.is_initialized() if RAY_AVAILABLE else False,
        }


class ModelRegistry:
    """Model versioning and registry with metadata tracking."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.registry_path = Path(config.get("registry_path", "model_registry"))
        self.registry_path.mkdir(exist_ok=True)

        # Model storage
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Metadata storage
        self.metadata_file = self.registry_path / "registry.json"
        self.models_metadata = self._load_metadata()

        # Version tracking
        self.current_version = self._get_next_version()

        logger.info(f"Model registry initialized at {self.registry_path}")

    def _load_metadata(self) -> dict[str, Any]:
        """Load registry metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading registry metadata: {e}")

        return {"models": {}, "versions": [], "current_production": None}

    def _save_metadata(self):
        """Save registry metadata."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.models_metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving registry metadata: {e}")

    def _get_next_version(self) -> str:
        """Get next version number."""
        versions = self.models_metadata.get("versions", [])
        if not versions:
            return "v1.0.0"

        # Extract version numbers and increment
        latest = versions[-1] if versions else "v0.0.0"
        major, minor, patch = map(int, latest[1:].split("."))
        return f"v{major}.{minor}.{patch + 1}"

    async def register_model(
        self,
        model_path: Path,
        metrics: dict[str, float],
        training_config: dict[str, Any] = None,
    ) -> ModelVersion | None:
        """Register new model version."""

        try:
            version = self.current_version
            model_id = f"yolo11_traffic_{version}"

            # Copy model to registry
            registry_model_path = self.models_dir / f"{model_id}.pt"
            shutil.copy2(model_path, registry_model_path)

            # Create model version metadata
            model_metadata = {
                "model_id": model_id,
                "version": version,
                "created_at": time.time(),
                "model_path": str(registry_model_path),
                "metrics": metrics,
                "training_config": training_config or {},
                "status": "registered",
                "validation_status": "pending",
            }

            # Calculate model hash for integrity
            with open(registry_model_path, "rb") as f:
                model_hash = hashlib.sha256(f.read()).hexdigest()[:16]
            model_metadata["model_hash"] = model_hash

            # Update registry
            self.models_metadata["models"][model_id] = model_metadata
            self.models_metadata["versions"].append(version)
            self._save_metadata()

            # Create ModelVersion object
            model_version = ModelVersion(
                model_id=model_id,
                version=version,
                model_path=registry_model_path,
                config_path=registry_model_path.with_suffix(".json"),
                metadata_path=registry_model_path.with_suffix(".metadata.json"),
                accuracy_score=metrics.get("accuracy", 0.0),
                latency_p95_ms=metrics.get("inference_latency_ms", 0.0),
                throughput_fps=1000.0 / max(1, metrics.get("inference_latency_ms", 50)),
            )

            # Save additional metadata
            with open(model_version.metadata_path, "w") as f:
                json.dump(model_metadata, f, indent=2, default=str)

            # Update next version
            self.current_version = self._get_next_version()

            logger.info(f"Registered model {model_id} with metrics: {metrics}")
            return model_version

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None

    async def get_production_model(self) -> ModelVersion | None:
        """Get current production model."""

        production_id = self.models_metadata.get("current_production")
        if not production_id or production_id not in self.models_metadata["models"]:
            return None

        return self._create_model_version(production_id)

    async def set_production_model(self, model_id: str) -> bool:
        """Set model as production."""

        if model_id not in self.models_metadata["models"]:
            logger.error(f"Model {model_id} not found in registry")
            return False

        # Update production pointer
        self.models_metadata["current_production"] = model_id
        self.models_metadata["models"][model_id]["status"] = "production"
        self._save_metadata()

        logger.info(f"Set {model_id} as production model")
        return True

    async def get_model_by_version(self, version: str) -> ModelVersion | None:
        """Get model by version string."""

        for model_id, metadata in self.models_metadata["models"].items():
            if metadata["version"] == version:
                return self._create_model_version(model_id)

        return None

    async def list_models(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent models with metadata."""

        models = []
        for model_id, metadata in list(self.models_metadata["models"].items())[-limit:]:
            models.append(
                {
                    "model_id": model_id,
                    "version": metadata["version"],
                    "created_at": metadata["created_at"],
                    "metrics": metadata["metrics"],
                    "status": metadata["status"],
                }
            )

        return models

    async def get_federated_models(self) -> list[dict[str, Any]]:
        """Get models available for federated learning."""

        federated_models = []
        for model_id, metadata in self.models_metadata["models"].items():
            if (
                metadata.get("training_config", {}).get("strategy") == "federated"
                and metadata["status"] == "registered"
            ):
                federated_models.append(
                    {
                        "model_id": model_id,
                        "path": metadata["model_path"],
                        "training_samples": metadata.get("training_config", {}).get(
                            "sample_count", 1000
                        ),
                        "created_at": metadata["created_at"],
                    }
                )

        return federated_models

    async def register_federated_model(
        self, aggregated_model: dict[str, Any]
    ) -> str | None:
        """Register federated aggregated model."""

        version = self.current_version
        model_id = f"yolo11_federated_{version}"

        # Copy to registry
        source_path = Path(aggregated_model["path"])
        registry_path = self.models_dir / f"{model_id}.pt"
        shutil.copy2(source_path, registry_path)

        # Create metadata
        model_metadata = {
            "model_id": model_id,
            "version": version,
            "created_at": time.time(),
            "model_path": str(registry_path),
            "type": "federated",
            "aggregation_info": aggregated_model,
            "status": "registered",
        }

        # Update registry
        self.models_metadata["models"][model_id] = model_metadata
        self.models_metadata["versions"].append(version)
        self._save_metadata()

        self.current_version = self._get_next_version()

        logger.info(f"Registered federated model {model_id}")
        return model_id

    def _create_model_version(self, model_id: str) -> ModelVersion | None:
        """Create ModelVersion object from metadata."""

        if model_id not in self.models_metadata["models"]:
            return None

        metadata = self.models_metadata["models"][model_id]

        model_version = ModelVersion(
            model_id=model_id,
            version=metadata["version"],
            model_path=Path(metadata["model_path"]),
            config_path=Path(metadata["model_path"]).with_suffix(".json"),
            metadata_path=Path(metadata["model_path"]).with_suffix(".metadata.json"),
            accuracy_score=metadata.get("metrics", {}).get("accuracy", 0.0),
            latency_p95_ms=metadata.get("metrics", {}).get("inference_latency_ms", 0.0),
        )

        return model_version

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics."""

        total_models = len(self.models_metadata["models"])
        production_model = self.models_metadata.get("current_production")

        # Model status distribution
        status_counts = defaultdict(int)
        for metadata in self.models_metadata["models"].values():
            status_counts[metadata["status"]] += 1

        return {
            "total_models": total_models,
            "current_production": production_model,
            "status_distribution": dict(status_counts),
            "latest_version": (
                self.models_metadata["versions"][-1]
                if self.models_metadata["versions"]
                else None
            ),
            "registry_size_mb": self._calculate_registry_size(),
        }

    def _calculate_registry_size(self) -> float:
        """Calculate total registry size in MB."""

        total_size = 0
        for model_file in self.models_dir.glob("*.pt"):
            total_size += model_file.stat().st_size

        return total_size / (1024 * 1024)


class ExperimentTracker:
    """MLflow-based experiment tracking."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.mlflow_enabled = MLFLOW_AVAILABLE and config.get("enabled", True)

        if self.mlflow_enabled:
            # Configure MLflow
            tracking_uri = config.get("tracking_uri", "file:./mlflow_experiments")
            mlflow.set_tracking_uri(tracking_uri)

            # Set default experiment
            experiment_name = config.get("experiment_name", "its_camera_ai")
            try:
                mlflow.set_experiment(experiment_name)
            except Exception:
                mlflow.create_experiment(experiment_name)
                mlflow.set_experiment(experiment_name)

            logger.info(f"MLflow experiment tracking enabled: {tracking_uri}")
        else:
            logger.warning("MLflow not available or disabled - using local tracking")

        # Local experiment storage (fallback)
        self.experiments_dir = Path(config.get("local_experiments_dir", "experiments"))
        self.experiments_dir.mkdir(exist_ok=True)
        self.active_runs = {}

    async def start_experiment(
        self, experiment_name: str, parameters: dict[str, Any]
    ) -> str:
        """Start new experiment run."""

        run_id = f"{experiment_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        if self.mlflow_enabled:
            try:
                mlflow_run = mlflow.start_run(run_name=run_id)

                # Log parameters
                for key, value in parameters.items():
                    mlflow.log_param(key, value)

                self.active_runs[run_id] = {
                    "mlflow_run": mlflow_run,
                    "start_time": time.time(),
                    "parameters": parameters,
                }

                logger.info(f"Started MLflow experiment run: {run_id}")
                return run_id

            except Exception as e:
                logger.error(f"MLflow experiment start error: {e}")

        # Local tracking fallback
        experiment_dir = self.experiments_dir / run_id
        experiment_dir.mkdir(exist_ok=True)

        experiment_data = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "parameters": parameters,
            "start_time": time.time(),
            "status": "running",
        }

        with open(experiment_dir / "experiment.json", "w") as f:
            json.dump(experiment_data, f, indent=2, default=str)

        self.active_runs[run_id] = {
            "local_dir": experiment_dir,
            "start_time": time.time(),
            "parameters": parameters,
        }

        logger.info(f"Started local experiment run: {run_id}")
        return run_id

    async def log_metrics(
        self, run_id: str, metrics: dict[str, float], step: int = None
    ):
        """Log metrics for experiment run."""

        if run_id not in self.active_runs:
            logger.warning(f"Unknown experiment run: {run_id}")
            return

        run_info = self.active_runs[run_id]

        if self.mlflow_enabled and "mlflow_run" in run_info:
            try:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.error(f"MLflow metrics logging error: {e}")

        # Local tracking
        if "local_dir" in run_info:
            metrics_file = run_info["local_dir"] / "metrics.json"

            # Load existing metrics
            if metrics_file.exists():
                with open(metrics_file) as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}

            # Add new metrics with timestamp
            timestamp = time.time()
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(
                    {"value": value, "step": step, "timestamp": timestamp}
                )

            # Save updated metrics
            with open(metrics_file, "w") as f:
                json.dump(all_metrics, f, indent=2)

    async def log_artifacts(self, run_id: str, artifacts: dict[str, str]):
        """Log artifacts for experiment run."""

        if run_id not in self.active_runs:
            logger.warning(f"Unknown experiment run: {run_id}")
            return

        run_info = self.active_runs[run_id]

        if self.mlflow_enabled and "mlflow_run" in run_info:
            try:
                for artifact_name, artifact_path in artifacts.items():
                    if Path(artifact_path).exists():
                        mlflow.log_artifact(artifact_path, artifact_name)
            except Exception as e:
                logger.error(f"MLflow artifact logging error: {e}")

        # Local tracking - copy artifacts
        if "local_dir" in run_info:
            artifacts_dir = run_info["local_dir"] / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)

            for artifact_name, artifact_path in artifacts.items():
                source_path = Path(artifact_path)
                if source_path.exists():
                    dest_path = artifacts_dir / f"{artifact_name}_{source_path.name}"
                    shutil.copy2(source_path, dest_path)

    async def log_results(
        self, run_id: str, metrics: dict[str, float], model_path: str
    ):
        """Log final experiment results."""

        # Log metrics
        await self.log_metrics(run_id, metrics)

        # Log model artifact
        if Path(model_path).exists():
            await self.log_artifacts(run_id, {"model": model_path})

        # End experiment run
        await self.end_experiment(run_id)

    async def end_experiment(self, run_id: str):
        """End experiment run."""

        if run_id not in self.active_runs:
            return

        run_info = self.active_runs[run_id]

        if self.mlflow_enabled and "mlflow_run" in run_info:
            try:
                mlflow.end_run()
            except Exception as e:
                logger.error(f"MLflow end run error: {e}")

        # Local tracking - update status
        if "local_dir" in run_info:
            experiment_file = run_info["local_dir"] / "experiment.json"
            if experiment_file.exists():
                with open(experiment_file) as f:
                    experiment_data = json.load(f)

                experiment_data["status"] = "completed"
                experiment_data["end_time"] = time.time()
                experiment_data["duration_minutes"] = (
                    time.time() - run_info["start_time"]
                ) / 60

                with open(experiment_file, "w") as f:
                    json.dump(experiment_data, f, indent=2, default=str)

        # Remove from active runs
        del self.active_runs[run_id]

        logger.info(f"Ended experiment run: {run_id}")

    async def get_experiment_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get experiment history."""

        experiments = []

        if self.mlflow_enabled:
            try:
                runs = mlflow.search_runs(
                    max_results=limit, order_by=["start_time DESC"]
                )
                for _, run in runs.iterrows():
                    experiments.append(
                        {
                            "run_id": run["run_id"],
                            "experiment_name": run.get(
                                "tags.mlflow.runName", "unknown"
                            ),
                            "status": run["status"],
                            "start_time": run["start_time"],
                            "end_time": run["end_time"],
                            "metrics": {
                                col.replace("metrics.", ""): run[col]
                                for col in run.index
                                if col.startswith("metrics.")
                            },
                            "parameters": {
                                col.replace("params.", ""): run[col]
                                for col in run.index
                                if col.startswith("params.")
                            },
                        }
                    )
            except Exception as e:
                logger.error(f"MLflow history query error: {e}")

        # Add local experiments
        for exp_dir in sorted(
            self.experiments_dir.glob("*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )[:limit]:
            exp_file = exp_dir / "experiment.json"
            if exp_file.exists():
                try:
                    with open(exp_file) as f:
                        exp_data = json.load(f)
                    experiments.append(exp_data)
                except Exception as e:
                    logger.error(f"Error loading local experiment {exp_dir}: {e}")

        return experiments[:limit]


class ProductionMLPipeline:
    """Main production ML pipeline orchestrator."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.status = PipelineStatus.INITIALIZING

        # Initialize components
        self.data_pipeline = DataIngestionPipeline(config.get("data_ingestion", {}))
        self.training_pipeline = ContinuousTrainingPipeline(config.get("training", {}))
        self.model_registry = ModelRegistry(config.get("model_registry", {}))
        self.experiment_tracker = ExperimentTracker(
            config.get("experiment_tracking", {})
        )

        # Model management components
        validator_config = config.get("model_validation", {})
        self.model_validator = ModelValidator(validator_config)

        # Production dashboard for monitoring
        self.dashboard = ProductionDashboard(config.get("monitoring", {}))

        # A/B testing and deployment
        self.ab_framework = ABTestingFramework(self.dashboard)
        self.deployment_pipeline = DeploymentPipeline(
            self.model_validator, self.ab_framework, self.dashboard
        )

        # Inference engine for serving
        inference_config = InferenceConfig(
            model_type=ModelType.SMALL,  # Default for balanced performance
            backend=OptimizationBackend.TENSORRT,
            precision="fp16",
            batch_size=8,
            max_batch_size=32,
        )
        self.inference_engine = OptimizedInferenceEngine(inference_config)

        # Pipeline state
        self.current_production_model = None
        self.pipeline_metrics = {
            "uptime_start": time.time(),
            "total_predictions": 0,
            "total_training_jobs": 0,
            "total_deployments": 0,
        }

        logger.info("Production ML pipeline initialized")

    async def start(self):
        """Start the complete ML pipeline."""

        try:
            self.status = PipelineStatus.INITIALIZING

            # Start all components
            await self.data_pipeline.start()
            await self.training_pipeline.start()
            await self.dashboard.start()

            # Initialize inference engine with base model
            base_model_path = Path(
                self.config.get("base_model_path", "models/yolo11s.pt")
            )
            if base_model_path.exists():
                await self.inference_engine.initialize(base_model_path)
                logger.info(f"Inference engine initialized with {base_model_path}")

            # Start pipeline monitoring
            asyncio.create_task(self._monitor_pipeline_health())
            asyncio.create_task(self._coordinate_training_deployment())

            self.status = PipelineStatus.RUNNING
            logger.info("Production ML pipeline started successfully")

        except Exception as e:
            self.status = PipelineStatus.ERROR
            logger.error(f"Pipeline startup error: {e}")
            raise

    async def _monitor_pipeline_health(self):
        """Monitor overall pipeline health and performance."""

        while True:
            try:
                # Collect component health metrics
                data_stats = self.data_pipeline.get_buffer_stats()
                training_stats = self.training_pipeline.get_training_stats()
                registry_stats = self.model_registry.get_registry_stats()

                # Calculate uptime
                uptime_hours = (
                    time.time() - self.pipeline_metrics["uptime_start"]
                ) / 3600

                # Log comprehensive health metrics
                health_metrics = {
                    "status": self.status.value,
                    "uptime_hours": uptime_hours,
                    "data_ingestion_rate": data_stats.get("ingestion_rate_per_sec", 0),
                    "training_queue_size": training_stats.get("queue_size", 0),
                    "active_training_jobs": training_stats.get("active_jobs", 0),
                    "total_models": registry_stats.get("total_models", 0),
                    "inference_throughput": await self._calculate_inference_throughput(),
                }

                logger.info(f"Pipeline health: {health_metrics}")

                # Check for alerts
                await self._check_pipeline_alerts(health_metrics)

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Pipeline health monitoring error: {e}")
                await asyncio.sleep(300)

    async def _coordinate_training_deployment(self):
        """Coordinate training and deployment cycle."""

        while True:
            try:
                # Check for new models in registry
                recent_models = await self.model_registry.list_models(limit=5)

                for model_info in recent_models:
                    model_id = model_info["model_id"]

                    # Skip if already processed
                    if model_info["status"] != "registered":
                        continue

                    # Get model version for validation
                    model_version = await self.model_registry.get_model_by_version(
                        model_info["version"]
                    )

                    if not model_version:
                        continue

                    # Validate model
                    logger.info(f"Validating model {model_id}")
                    validation_result = await self.model_validator.validate_model(
                        model_version
                    )

                    if validation_result == ModelValidationResult.PASSED:
                        # Deploy using A/B testing
                        logger.info(f"Deploying model {model_id}")

                        deployment_result = await self.deployment_pipeline.deploy_model(
                            model_version,
                            DeploymentStrategy.CANARY,
                            self.current_production_model,
                        )

                        if deployment_result["success"]:
                            self.pipeline_metrics["total_deployments"] += 1
                            logger.info(f"Successfully deployed model {model_id}")
                        else:
                            logger.error(
                                f"Deployment failed for {model_id}: {deployment_result}"
                            )

                    else:
                        logger.warning(
                            f"Model {model_id} validation failed: {validation_result}"
                        )

                await asyncio.sleep(600)  # Check every 10 minutes

            except Exception as e:
                logger.error(f"Training-deployment coordination error: {e}")
                await asyncio.sleep(600)

    async def _calculate_inference_throughput(self) -> float:
        """Calculate current inference throughput."""

        try:
            perf_stats = self.inference_engine.get_performance_stats()
            return perf_stats.get("throughput_fps", 0.0)
        except Exception:
            return 0.0

    async def _check_pipeline_alerts(self, metrics: dict[str, float]):
        """Check pipeline metrics for alert conditions."""

        alerts = []

        # Data ingestion rate alert
        if metrics["data_ingestion_rate"] < 10:  # Below 10 samples/sec
            alerts.append(
                Alert(
                    alert_id=f"data_rate_{int(time.time())}",
                    severity=AlertSeverity.WARNING,
                    drift_type=DriftType.VOLUME_DRIFT,
                    message=f"Low data ingestion rate: {metrics['data_ingestion_rate']:.1f} samples/sec",
                    timestamp=time.time(),
                    current_value=metrics["data_ingestion_rate"],
                    expected_value=30.0,
                    threshold=10.0,
                )
            )

        # Training backlog alert
        if metrics["training_queue_size"] > 50:
            alerts.append(
                Alert(
                    alert_id=f"training_backlog_{int(time.time())}",
                    severity=AlertSeverity.CRITICAL,
                    drift_type=DriftType.VOLUME_DRIFT,
                    message=f"High training queue backlog: {metrics['training_queue_size']} batches",
                    timestamp=time.time(),
                    current_value=metrics["training_queue_size"],
                    expected_value=10.0,
                    threshold=50.0,
                )
            )

        # Inference throughput alert
        if metrics["inference_throughput"] < 100:  # Below 100 FPS
            alerts.append(
                Alert(
                    alert_id=f"inference_throughput_{int(time.time())}",
                    severity=AlertSeverity.WARNING,
                    drift_type=DriftType.LATENCY_DRIFT,
                    message=f"Low inference throughput: {metrics['inference_throughput']:.1f} FPS",
                    timestamp=time.time(),
                    current_value=metrics["inference_throughput"],
                    expected_value=500.0,
                    threshold=100.0,
                )
            )

        # Send alerts to dashboard
        for alert in alerts:
            await self.dashboard.add_alert(alert)

    async def predict(
        self, frame: np.ndarray, frame_id: str, camera_id: str
    ) -> DetectionResult:
        """Make prediction using current production model."""

        try:
            result = await self.inference_engine.predict_single(
                frame, frame_id, camera_id
            )
            self.pipeline_metrics["total_predictions"] += 1

            # Update monitoring
            if camera_id in self.dashboard.model_monitors:
                monitor = self.dashboard.model_monitors[camera_id]
                monitor.metrics.add_sample(result)

            return result

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    async def predict_batch(
        self, frames: list[np.ndarray], frame_ids: list[str], camera_ids: list[str]
    ) -> list[DetectionResult]:
        """Make batch predictions."""

        try:
            results = await self.inference_engine.predict_batch(
                frames, frame_ids, camera_ids
            )
            self.pipeline_metrics["total_predictions"] += len(results)

            # Update monitoring for each result
            for result in results:
                if result.camera_id in self.dashboard.model_monitors:
                    monitor = self.dashboard.model_monitors[result.camera_id]
                    monitor.metrics.add_sample(result)

            return results

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise

    async def trigger_retraining(self, trigger_reason: str = "manual"):
        """Trigger model retraining."""

        logger.info(f"Triggering retraining: {trigger_reason}")
        self.status = PipelineStatus.TRAINING

        try:
            # Create training batch from current data buffer
            if len(self.data_pipeline.training_buffer) >= 50:
                samples = []
                batch_size = min(200, len(self.data_pipeline.training_buffer))

                for _ in range(batch_size):
                    if self.data_pipeline.training_buffer:
                        samples.append(self.data_pipeline.training_buffer.popleft())

                batch = TrainingBatch(
                    batch_id=f"manual_retrain_{int(time.time())}",
                    samples=samples,
                    model_version="current",
                    training_type="full_retrain",
                    created_at=time.time(),
                    quality_distribution=self.data_pipeline._get_quality_distribution(
                        samples
                    ),
                    camera_distribution=self.data_pipeline._get_camera_distribution(
                        samples
                    ),
                    learning_rate=0.001,
                    epochs=10,
                )

                await self.training_pipeline.add_training_batch(batch)
                self.pipeline_metrics["total_training_jobs"] += 1

                logger.info(f"Triggered retraining with {len(samples)} samples")
            else:
                logger.warning("Insufficient training data for retraining")

        except Exception as e:
            logger.error(f"Retraining trigger error: {e}")
        finally:
            self.status = PipelineStatus.RUNNING

    async def get_pipeline_status(self) -> dict[str, Any]:
        """Get comprehensive pipeline status."""

        return {
            "status": self.status.value,
            "uptime_hours": (time.time() - self.pipeline_metrics["uptime_start"])
            / 3600,
            "metrics": self.pipeline_metrics,
            "components": {
                "data_pipeline": self.data_pipeline.get_buffer_stats(),
                "training_pipeline": self.training_pipeline.get_training_stats(),
                "model_registry": self.model_registry.get_registry_stats(),
                "inference_engine": self.inference_engine.get_performance_stats(),
            },
            "current_production_model": (
                self.current_production_model.model_id
                if self.current_production_model
                else None
            ),
            "active_experiments": len(self.ab_framework.active_experiments),
        }

    async def stop(self):
        """Stop the ML pipeline gracefully."""

        logger.info("Stopping production ML pipeline")
        self.status = PipelineStatus.STOPPED

        try:
            # Stop components
            await self.inference_engine.cleanup()

            # Stop training pipeline
            if RAY_AVAILABLE and ray.is_initialized():
                ray.shutdown()

            logger.info("Production ML pipeline stopped")

        except Exception as e:
            logger.error(f"Pipeline shutdown error: {e}")


# Pipeline Factory and Utilities


async def create_production_pipeline(
    config_path: str | Path = None,
) -> ProductionMLPipeline:
    """Factory function to create production ML pipeline."""

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "data_ingestion": {
                "kafka_servers": ["localhost:9092"],
                "redis_url": "redis://localhost:6379",
                "quality_threshold": 0.7,
            },
            "training": {
                "use_ray": True,
                "num_workers": 4,
                "federated_enabled": True,
                "output_dir": "training_outputs",
            },
            "model_registry": {"registry_path": "model_registry"},
            "experiment_tracking": {
                "enabled": True,
                "tracking_uri": "file:./mlflow_experiments",
            },
            "model_validation": {
                "min_accuracy": 0.85,
                "max_latency_ms": 100,
                "validation_dataset": "data/validation",
            },
            "monitoring": {"dashboard_port": 8080, "metrics_interval": 60},
        }

    pipeline = ProductionMLPipeline(config)
    await pipeline.start()

    return pipeline


# Example usage for testing
if __name__ == "__main__":

    async def main():
        # Create production pipeline
        pipeline = await create_production_pipeline()

        # Simulate some predictions
        for i in range(10):
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            result = await pipeline.predict(frame, f"frame_{i}", "test_camera")
            print(f"Prediction {i}: {result.detection_count} detections")
            await asyncio.sleep(1)

        # Get pipeline status
        status = await pipeline.get_pipeline_status()
        print(f"Pipeline status: {status}")

        # Stop pipeline
        await pipeline.stop()

    asyncio.run(main())
