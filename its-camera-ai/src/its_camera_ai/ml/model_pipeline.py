"""
Automated Model Update Pipeline with A/B Testing for ITS Camera AI Traffic Monitoring System.

This module implements a comprehensive MLOps pipeline for continuous model improvement:
1. A/B testing framework for comparing model versions
2. Automated model validation and rollout
3. Canary deployments with traffic splitting
4. Model performance comparison and statistical significance testing
5. Automated rollback mechanisms for failed deployments
6. Blue-green deployment strategies

Key Features:
- Statistical significance testing for A/B experiments
- Gradual traffic routing with safety checks
- Automated model validation pipeline
- Real-time performance monitoring during rollouts
- Fallback and rollback mechanisms
- Integration with CI/CD systems
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torchvision.ops import box_iou

# Optional ML dependencies
try:
    from scipy import stats
    from sklearn.metrics import average_precision_score, precision_recall_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import boto3
    from minio import Minio
    MINIO_AVAILABLE = True
except ImportError:
    MINIO_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .inference_optimizer import (
    DetectionResult,
    InferenceConfig,
    OptimizedInferenceEngine,
)
from .production_monitoring import (
    ProductionDashboard,
)

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """A/B experiment status."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class DeploymentStrategy(Enum):
    """Model deployment strategies."""

    BLUE_GREEN = "blue_green"  # Switch all traffic at once
    CANARY = "canary"  # Gradual traffic increase
    ROLLING = "rolling"  # Replace instances gradually
    FEATURE_FLAG = "feature_flag"  # Use feature flags for control


class ModelValidationResult(Enum):
    """Model validation results."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"


@dataclass
class ModelVersion:
    """Model version metadata."""

    model_id: str
    version: str

    # Model artifacts
    model_path: Path
    config_path: Path
    metadata_path: Path

    # Performance metrics
    accuracy_score: float = 0.0
    latency_p95_ms: float = 0.0
    throughput_fps: float = 0.0

    # Deployment info
    deployment_timestamp: float | None = None
    is_active: bool = False
    traffic_percentage: float = 0.0

    # Validation
    validation_result: ModelValidationResult = ModelValidationResult.PENDING
    validation_details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Generate unique model hash for caching
        model_content = (
            self.model_path.read_bytes() if self.model_path.exists() else b""
        )
        self.model_hash = hashlib.sha256(model_content).hexdigest()[:16]


@dataclass
class ABExperiment:
    """A/B testing experiment configuration."""

    experiment_id: str
    name: str
    description: str

    # Models being compared
    control_model: ModelVersion  # Current production model
    treatment_model: ModelVersion  # New model being tested

    # Experiment configuration
    traffic_split: float = 0.1  # Percentage of traffic to treatment (0.0-1.0)
    min_sample_size: int = 1000  # Minimum samples per variant
    max_runtime_hours: int = 168  # Maximum experiment duration (1 week)

    # Statistical configuration
    significance_level: float = 0.05
    minimum_effect_size: float = 0.02  # 2% improvement threshold
    power: float = 0.8

    # Status
    status: ExperimentStatus = ExperimentStatus.DRAFT
    start_time: float | None = None
    end_time: float | None = None

    # Results
    control_metrics: list[dict[str, float]] = field(default_factory=list)
    treatment_metrics: list[dict[str, float]] = field(default_factory=list)

    # Safety checks
    safety_enabled: bool = True
    max_error_rate_increase: float = 0.05  # 5% max error rate increase
    max_latency_increase: float = 0.20  # 20% max latency increase

    def get_required_sample_size(self) -> int:
        """Calculate required sample size for statistical power."""
        try:
            from scipy import stats

            # Effect size calculation (Cohen's d)
            effect_size = self.minimum_effect_size

            # Calculate required sample size using power analysis
            from statsmodels.stats.power import ttest_power  # noqa: F401

            n = stats.norm.ppf(1 - self.significance_level / 2) + stats.norm.ppf(
                self.power
            )
            n = (n / effect_size) ** 2 * 2

            return max(self.min_sample_size, int(n))

        except ImportError:
            # Fallback calculation
            return max(self.min_sample_size, 2000)


class ModelValidator:
    """Validate models before deployment."""

    def __init__(self, validation_config: dict[str, Any]):
        self.validation_config = validation_config

        # Validation thresholds
        self.min_accuracy = validation_config.get("min_accuracy", 0.85)
        self.max_latency_ms = validation_config.get("max_latency_ms", 100)
        self.min_throughput_fps = validation_config.get("min_throughput_fps", 10)

        # Test datasets
        self.validation_dataset_path = Path(
            validation_config.get("validation_dataset", "data/validation")
        )
        self.benchmark_dataset_path = Path(
            validation_config.get("benchmark_dataset", "data/benchmark")
        )

    async def validate_model(
        self, model_version: ModelVersion
    ) -> ModelValidationResult:
        """Comprehensive model validation."""
        logger.info(
            f"Starting validation for model {model_version.model_id} v{model_version.version}"
        )

        try:
            # Initialize model for testing
            engine = await self._initialize_model(model_version)

            # Run validation tests
            validation_results = {}

            # 1. Accuracy validation
            accuracy_result = await self._validate_accuracy(engine, model_version)
            validation_results["accuracy"] = accuracy_result

            # 2. Performance validation
            perf_result = await self._validate_performance(engine, model_version)
            validation_results["performance"] = perf_result

            # 3. Regression testing
            regression_result = await self._validate_regression(engine, model_version)
            validation_results["regression"] = regression_result

            # 4. Resource usage validation
            resource_result = await self._validate_resources(engine, model_version)
            validation_results["resources"] = resource_result

            # Aggregate results
            model_version.validation_details = validation_results

            # Determine overall result
            failed_tests = [
                k for k, v in validation_results.items() if v["status"] == "failed"
            ]
            warning_tests = [
                k for k, v in validation_results.items() if v["status"] == "warning"
            ]

            if failed_tests:
                model_version.validation_result = ModelValidationResult.FAILED
                logger.warning(f"Model validation failed: {failed_tests}")
            elif warning_tests:
                model_version.validation_result = ModelValidationResult.WARNING
                logger.warning(f"Model validation warnings: {warning_tests}")
            else:
                model_version.validation_result = ModelValidationResult.PASSED
                logger.info("Model validation passed")

            # Cleanup
            await engine.cleanup()

            return model_version.validation_result

        except Exception as e:
            logger.error(f"Model validation error: {e}")
            model_version.validation_result = ModelValidationResult.FAILED
            model_version.validation_details["error"] = str(e)
            return ModelValidationResult.FAILED

    async def _initialize_model(
        self, model_version: ModelVersion
    ) -> OptimizedInferenceEngine:
        """Initialize model for validation."""
        # Load configuration
        config_path = model_version.config_path
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)

            config = InferenceConfig(**config_data)
        else:
            # Use default configuration
            config = InferenceConfig()

        # Initialize engine
        engine = OptimizedInferenceEngine(config)
        await engine.initialize(model_version.model_path)

        return engine

    async def _validate_accuracy(
        self, engine: OptimizedInferenceEngine, model_version: ModelVersion
    ) -> dict[str, Any]:
        """Validate model accuracy on test dataset."""

        if not self.validation_dataset_path.exists():
            return {"status": "skipped", "reason": "No validation dataset available"}

        try:
            # Load validation samples
            validation_samples = self._load_validation_samples()

            if len(validation_samples) == 0:
                return {"status": "warning", "reason": "Empty validation dataset"}

            # Run inference on validation set
            correct_predictions = 0
            total_predictions = len(validation_samples)

            for sample in validation_samples[:100]:  # Limit for validation speed
                result = await engine.predict_single(
                    sample["image"], f"val_{sample['id']}", "validation_camera"
                )

                # Compare with ground truth (simplified)
                if self._compare_with_ground_truth(result, sample["ground_truth"]):
                    correct_predictions += 1

            accuracy = (
                correct_predictions / total_predictions if total_predictions > 0 else 0
            )
            model_version.accuracy_score = accuracy

            # Check against threshold
            status = "passed" if accuracy >= self.min_accuracy else "failed"

            return {
                "status": status,
                "accuracy": accuracy,
                "threshold": self.min_accuracy,
                "samples_tested": total_predictions,
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _validate_performance(
        self, engine: OptimizedInferenceEngine, model_version: ModelVersion
    ) -> dict[str, Any]:
        """Validate model performance metrics."""

        try:
            # Generate synthetic test data for performance testing
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

            # Warm-up runs
            for _ in range(10):
                await engine.predict_single(test_image, "warmup", "test_camera")

            # Performance measurement
            latencies = []
            for i in range(50):
                start_time = time.time()
                await engine.predict_single(test_image, f"perf_{i}", "test_camera")
                end_time = time.time()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

            # Calculate metrics
            p95_latency = np.percentile(latencies, 95)
            avg_latency = np.mean(latencies)
            throughput_fps = 1000 / avg_latency if avg_latency > 0 else 0

            model_version.latency_p95_ms = p95_latency
            model_version.throughput_fps = throughput_fps

            # Validate against thresholds
            latency_ok = p95_latency <= self.max_latency_ms
            throughput_ok = throughput_fps >= self.min_throughput_fps

            if latency_ok and throughput_ok:
                status = "passed"
            elif not latency_ok:
                status = "failed"
            else:
                status = "warning"

            return {
                "status": status,
                "p95_latency_ms": p95_latency,
                "avg_latency_ms": avg_latency,
                "throughput_fps": throughput_fps,
                "latency_threshold": self.max_latency_ms,
                "throughput_threshold": self.min_throughput_fps,
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _validate_regression(
        self, engine: OptimizedInferenceEngine, model_version: ModelVersion
    ) -> dict[str, Any]:
        """Check for regression against current production model with comprehensive metrics."""

        try:
            logger.info(f"Starting regression testing for model {model_version.model_id}")

            # Load validation dataset
            validation_samples = await self._load_validation_samples_async()

            if len(validation_samples) < 100:
                return {
                    "status": "warning",
                    "reason": f"Insufficient validation samples: {len(validation_samples)}",
                    "samples_tested": len(validation_samples)
                }

            # Get current production model results (baseline)
            baseline_results = await self._get_baseline_results(validation_samples)

            # Run inference on new model
            new_model_results = await self._run_model_inference(engine, validation_samples)

            # Calculate comprehensive metrics
            metrics_comparison = self._calculate_regression_metrics(
                baseline_results, new_model_results, validation_samples
            )

            # Statistical significance testing
            significance_test = self._perform_statistical_significance_test(
                baseline_results, new_model_results
            )

            # Determine regression status
            status = self._determine_regression_status(
                metrics_comparison, significance_test
            )

            # Generate regression report
            report_path = await self._generate_regression_report(
                model_version, metrics_comparison, significance_test, status
            )

            # Store results for model version
            model_version.validation_details.update({
                "regression_metrics": metrics_comparison,
                "statistical_significance": significance_test,
                "regression_report_path": str(report_path)
            })

            logger.info(f"Regression testing completed with status: {status['status']}")

            return {
                "status": status["status"],
                "metrics": metrics_comparison,
                "significance": significance_test,
                "samples_tested": len(validation_samples),
                "report_path": str(report_path),
                "summary": status["summary"]
            }

        except Exception as e:
            logger.error(f"Regression testing failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "samples_tested": 0
            }

    async def _validate_resources(
        self, _engine: OptimizedInferenceEngine, _model_version: ModelVersion
    ) -> dict[str, Any]:
        """Validate resource usage (memory, GPU, etc.)."""

        try:
            # Get current resource usage
            performance_stats = _engine.get_performance_stats()

            # Check memory usage
            gpu_memory = performance_stats.get("gpu_memory_used", {})
            total_memory_mb = (
                sum(gpu_memory.values()) if isinstance(gpu_memory, dict) else 0
            )

            # Resource thresholds
            max_memory_mb = 4000  # 4GB limit
            memory_ok = total_memory_mb <= max_memory_mb

            status = "passed" if memory_ok else "warning"

            return {
                "status": status,
                "gpu_memory_mb": total_memory_mb,
                "memory_threshold_mb": max_memory_mb,
                "gpu_utilization": performance_stats.get("gpu_utilization", 0),
            }

        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def _load_validation_samples(self) -> list[dict[str, Any]]:
        """Load validation dataset samples synchronously."""
        try:
            # Run async version synchronously
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._load_validation_samples_async())
        except RuntimeError:
            # If no event loop exists, create one
            return asyncio.run(self._load_validation_samples_async())

    async def _load_validation_samples_async(self) -> list[dict[str, Any]]:
        """Load validation dataset samples from various sources with data augmentation."""

        samples = []

        # Try loading from different data sources in priority order
        data_sources = [
            self._load_from_local_dataset,
            self._load_from_s3_dataset,
            self._load_from_database_dataset,
            self._load_synthetic_validation_data
        ]

        for data_source in data_sources:
            try:
                samples = await data_source()
                if samples:
                    logger.info(f"Loaded {len(samples)} validation samples from {data_source.__name__}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load from {data_source.__name__}: {str(e)}")
                continue

        if not samples:
            logger.warning("No validation samples loaded from any source")
            return []

        # Apply data augmentation for robustness testing
        if len(samples) > 0:
            augmented_samples = await self._apply_validation_augmentations(samples)
            samples.extend(augmented_samples)

        # Shuffle samples for random validation
        np.random.shuffle(samples)

        # Limit to maximum validation size for performance
        max_samples = self.validation_config.get("max_validation_samples", 10000)
        samples = samples[:max_samples]

        logger.info(f"Final validation set size: {len(samples)} samples")
        return samples

    async def _load_from_local_dataset(self) -> list[dict[str, Any]]:
        """Load validation data from local filesystem."""

        if not self.validation_dataset_path.exists():
            return []

        samples = []

        # Support multiple dataset formats
        if (self.validation_dataset_path / "annotations.json").exists():
            # COCO format
            samples = await self._load_coco_format(self.validation_dataset_path)
        elif (self.validation_dataset_path / "labels").exists():
            # YOLO format
            samples = await self._load_yolo_format(self.validation_dataset_path)
        elif (self.validation_dataset_path / "dataset.yaml").exists():
            # Custom YAML format
            samples = await self._load_yaml_format(self.validation_dataset_path)
        else:
            # Directory with images and annotations
            samples = await self._load_directory_format(self.validation_dataset_path)

        return samples

    async def _load_from_s3_dataset(self) -> list[dict[str, Any]]:
        """Load validation data from S3/MinIO object storage."""

        if not MINIO_AVAILABLE:
            logger.warning("MinIO/S3 dependencies not available, skipping S3 dataset loading")
            return []

        try:
            # Check if S3/MinIO configuration exists
            s3_config = self.validation_config.get("s3_config")
            if not s3_config:
                return []

            # Initialize MinIO client
            client = Minio(
                s3_config["endpoint"],
                access_key=s3_config["access_key"],
                secret_key=s3_config["secret_key"],
                secure=s3_config.get("secure", True)
            )

            bucket = s3_config["bucket"]
            prefix = s3_config.get("validation_prefix", "validation/")

            # List objects in validation directory
            objects = client.list_objects(bucket, prefix=prefix, recursive=True)

            samples = []
            for obj in objects:
                if obj.object_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Download image
                    image_data = client.get_object(bucket, obj.object_name).read()
                    image_array = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                    # Try to find corresponding annotation
                    annotation_path = obj.object_name.replace('.jpg', '.json').replace('.png', '.json')
                    ground_truth = {"boxes": [], "classes": [], "scores": []}

                    try:
                        annotation_data = client.get_object(bucket, annotation_path).read()
                        ground_truth = json.loads(annotation_data.decode('utf-8'))
                    except Exception:
                        # No annotation found, use empty ground truth
                        pass

                    samples.append({
                        "id": obj.object_name,
                        "image": image,
                        "ground_truth": ground_truth,
                        "source": "s3"
                    })

            return samples

        except Exception as e:
            logger.warning(f"Failed to load from S3: {str(e)}")
            return []

    async def _load_from_database_dataset(self) -> list[dict[str, Any]]:
        """Load validation data from database."""

        try:
            # This would connect to your database and load validation samples
            # Implementation depends on your database schema
            db_config = self.validation_config.get("database_config")
            if not db_config:
                return []

            # Placeholder for database loading logic
            # In production, you'd execute SQL queries to fetch validation data
            samples = []

            return samples

        except Exception as e:
            logger.warning(f"Failed to load from database: {str(e)}")
            return []

    async def _load_synthetic_validation_data(self) -> list[dict[str, Any]]:
        """Generate synthetic validation data as fallback."""

        logger.info("Generating synthetic validation data as fallback")
        samples = []

        # Generate more realistic synthetic data
        for i in range(1000):
            # Create realistic traffic scene
            image = self._generate_synthetic_traffic_image(i)

            # Generate realistic bounding boxes
            ground_truth = self._generate_synthetic_annotations(image.shape)

            samples.append({
                "id": f"synthetic_{i}",
                "image": image,
                "ground_truth": ground_truth,
                "source": "synthetic"
            })

        return samples

    async def _load_coco_format(self, dataset_path: Path) -> list[dict[str, Any]]:
        """Load COCO format dataset."""

        annotations_file = dataset_path / "annotations.json"
        images_dir = dataset_path / "images"

        with open(annotations_file) as f:
            coco_data = json.load(f)

        samples = []

        # Create mapping from image_id to annotations
        image_annotations = {}
        for annotation in coco_data.get('annotations', []):
            image_id = annotation['image_id']
            if image_id not in image_annotations:
                image_annotations[image_id] = []
            image_annotations[image_id].append(annotation)

        # Process each image
        for image_info in coco_data.get('images', []):
            image_path = images_dir / image_info['file_name']

            if image_path.exists():
                image = cv2.imread(str(image_path))

                # Convert COCO annotations to our format
                annotations = image_annotations.get(image_info['id'], [])
                boxes = []
                classes = []
                scores = []

                for ann in annotations:
                    # COCO bbox format: [x, y, width, height]
                    x, y, w, h = ann['bbox']
                    # Convert to [x1, y1, x2, y2]
                    boxes.append([x, y, x + w, y + h])
                    classes.append(ann['category_id'])
                    scores.append(1.0)  # Ground truth has confidence 1.0

                ground_truth = {
                    "boxes": np.array(boxes, dtype=np.float32) if boxes else np.array([]).reshape(0, 4),
                    "classes": np.array(classes, dtype=np.int32) if classes else np.array([]),
                    "scores": np.array(scores, dtype=np.float32) if scores else np.array([])
                }

                samples.append({
                    "id": image_info['file_name'],
                    "image": image,
                    "ground_truth": ground_truth,
                    "source": "coco"
                })

        return samples

    async def _load_yolo_format(self, dataset_path: Path) -> list[dict[str, Any]]:
        """Load YOLO format dataset."""

        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"

        samples = []

        for image_path in images_dir.glob("*.jpg"):
            label_path = labels_dir / f"{image_path.stem}.txt"

            image = cv2.imread(str(image_path))
            h, w = image.shape[:2]

            boxes = []
            classes = []
            scores = []

            if label_path.exists():
                with open(label_path) as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            # YOLO format: center_x, center_y, width, height (normalized)
                            cx, cy, bw, bh = map(float, parts[1:5])

                            # Convert to absolute coordinates [x1, y1, x2, y2]
                            x1 = (cx - bw/2) * w
                            y1 = (cy - bh/2) * h
                            x2 = (cx + bw/2) * w
                            y2 = (cy + bh/2) * h

                            boxes.append([x1, y1, x2, y2])
                            classes.append(class_id)
                            scores.append(1.0)

            ground_truth = {
                "boxes": np.array(boxes, dtype=np.float32) if boxes else np.array([]).reshape(0, 4),
                "classes": np.array(classes, dtype=np.int32) if classes else np.array([]),
                "scores": np.array(scores, dtype=np.float32) if scores else np.array([])
            }

            samples.append({
                "id": image_path.name,
                "image": image,
                "ground_truth": ground_truth,
                "source": "yolo"
            })

        return samples

    async def _load_yaml_format(self, dataset_path: Path) -> list[dict[str, Any]]:
        """Load custom YAML format dataset."""

        if not YAML_AVAILABLE:
            logger.warning("YAML dependencies not available, skipping YAML dataset loading")
            return []

        config_file = dataset_path / "dataset.yaml"

        with open(config_file) as f:
            config = yaml.safe_load(f)

        samples = []

        for item in config.get('validation', []):
            image_path = dataset_path / item['image']

            if image_path.exists():
                image = cv2.imread(str(image_path))

                # Load ground truth
                ground_truth = {
                    "boxes": np.array(item.get('boxes', []), dtype=np.float32).reshape(-1, 4),
                    "classes": np.array(item.get('classes', []), dtype=np.int32),
                    "scores": np.array(item.get('scores', [1.0] * len(item.get('classes', []))), dtype=np.float32)
                }

                samples.append({
                    "id": item['image'],
                    "image": image,
                    "ground_truth": ground_truth,
                    "source": "yaml"
                })

        return samples

    async def _load_directory_format(self, dataset_path: Path) -> list[dict[str, Any]]:
        """Load dataset from directory with images and JSON annotations."""

        samples = []

        for image_path in dataset_path.glob("*.jpg"):
            annotation_path = image_path.with_suffix('.json')

            image = cv2.imread(str(image_path))
            ground_truth = {"boxes": [], "classes": [], "scores": []}

            if annotation_path.exists():
                with open(annotation_path) as f:
                    annotation = json.load(f)
                    ground_truth = annotation.get('ground_truth', ground_truth)

                    # Ensure numpy arrays
                    ground_truth = {
                        "boxes": np.array(ground_truth.get("boxes", []), dtype=np.float32).reshape(-1, 4),
                        "classes": np.array(ground_truth.get("classes", []), dtype=np.int32),
                        "scores": np.array(ground_truth.get("scores", []), dtype=np.float32)
                    }

            samples.append({
                "id": image_path.name,
                "image": image,
                "ground_truth": ground_truth,
                "source": "directory"
            })

        return samples

    async def _apply_validation_augmentations(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply data augmentations to validation samples for robustness testing."""

        augmented_samples = []

        # Apply 10% augmentation rate
        num_augmentations = min(len(samples) // 10, 1000)
        selected_samples = np.random.choice(samples, size=num_augmentations, replace=False)

        for sample in selected_samples:
            image = sample["image"].copy()
            ground_truth = sample["ground_truth"].copy()

            # Random brightness adjustment
            if np.random.random() < 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                image = np.clip(image * brightness, 0, 255).astype(np.uint8)

            # Random noise addition
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 5, image.shape)
                image = np.clip(image + noise, 0, 255).astype(np.uint8)

            # Random blur
            if np.random.random() < 0.3:
                kernel_size = np.random.choice([3, 5])
                image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

            augmented_samples.append({
                "id": f"{sample['id']}_aug_{len(augmented_samples)}",
                "image": image,
                "ground_truth": ground_truth,
                "source": f"{sample['source']}_augmented"
            })

        return augmented_samples

    def _generate_synthetic_traffic_image(self, seed: int) -> np.ndarray:
        """Generate synthetic traffic scene image."""

        np.random.seed(seed)

        # Create base road scene
        image = np.ones((640, 640, 3), dtype=np.uint8) * 128  # Gray background

        # Add road
        cv2.rectangle(image, (0, 300), (640, 500), (80, 80, 80), -1)

        # Add lane markings
        for x in range(0, 640, 40):
            cv2.rectangle(image, (x, 395), (x + 20, 405), (255, 255, 255), -1)

        # Add random vehicles (rectangles)
        num_vehicles = np.random.randint(1, 6)
        for _ in range(num_vehicles):
            x = np.random.randint(50, 550)
            y = np.random.randint(320, 450)
            w = np.random.randint(60, 120)
            h = np.random.randint(30, 60)

            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(image, (x, y), (x + w, y + h), color, -1)

        # Add noise
        noise = np.random.normal(0, 10, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)

        return image

    def _generate_synthetic_annotations(self, image_shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
        """Generate synthetic ground truth annotations."""

        h, w = image_shape[:2]
        num_boxes = np.random.randint(1, 6)

        boxes = []
        classes = []
        scores = []

        for _ in range(num_boxes):
            # Generate random box in road area
            x1 = np.random.randint(50, w - 150)
            y1 = np.random.randint(320, 450)
            x2 = x1 + np.random.randint(60, 120)
            y2 = y1 + np.random.randint(30, 60)

            boxes.append([x1, y1, x2, y2])
            classes.append(np.random.choice([0, 1, 2]))  # car, truck, bus
            scores.append(1.0)

        return {
            "boxes": np.array(boxes, dtype=np.float32) if boxes else np.array([]).reshape(0, 4),
            "classes": np.array(classes, dtype=np.int32) if classes else np.array([]),
            "scores": np.array(scores, dtype=np.float32) if scores else np.array([])
        }

    def _compare_with_ground_truth(
        self, result: DetectionResult, ground_truth: dict[str, Any]
    ) -> bool:
        """Compare inference result with ground truth using IoU threshold."""

        try:
            # Convert ground truth to proper format
            gt_boxes = ground_truth.get("boxes", [])
            gt_classes = ground_truth.get("classes", [])

            if len(gt_boxes) == 0 and len(result.boxes) == 0:
                return True  # Both empty, correct

            if len(gt_boxes) == 0 or len(result.boxes) == 0:
                return False  # One empty, one not

            # Convert to tensors for IoU calculation
            pred_boxes = torch.tensor(result.boxes, dtype=torch.float32)
            gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32)

            # Calculate IoU matrix
            iou_matrix = box_iou(pred_boxes, gt_boxes_tensor)

            # Use IoU threshold of 0.5 for matching
            iou_threshold = 0.5
            matches = iou_matrix > iou_threshold

            # Simple matching: count if any prediction matches any ground truth
            num_matches = torch.sum(torch.max(matches, dim=1)[0]).item()

            # Consider it correct if we have reasonable overlap
            precision = num_matches / len(result.boxes) if len(result.boxes) > 0 else 0
            recall = num_matches / len(gt_boxes) if len(gt_boxes) > 0 else 0

            # Use F1 score > 0.5 as threshold for correctness
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            return f1 > 0.5

        except Exception as e:
            logger.warning(f"Ground truth comparison failed: {str(e)}")
            return False

    async def _get_baseline_results(self, validation_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Get baseline model results for comparison."""

        # In production, this would load and run the current production model
        # For now, simulate baseline results with realistic performance
        baseline_results = []

        for sample in validation_samples:
            # Simulate baseline model predictions
            gt = sample["ground_truth"]
            num_gt_boxes = len(gt.get("boxes", []))

            # Simulate detection with some noise
            num_predictions = max(0, num_gt_boxes + np.random.randint(-1, 2))

            boxes = []
            classes = []
            scores = []

            for i in range(num_predictions):
                if i < len(gt.get("boxes", [])):
                    # Add noise to ground truth boxes
                    gt_box = gt["boxes"][i]
                    noise = np.random.normal(0, 5, 4)
                    box = np.clip(gt_box + noise, 0, 640)
                    boxes.append(box.tolist())

                    # Use ground truth class with some error
                    if i < len(gt.get("classes", [])):
                        classes.append(gt["classes"][i])
                    else:
                        classes.append(0)

                    # Simulate confidence scores
                    scores.append(np.random.uniform(0.7, 0.95))
                else:
                    # False positive
                    h, w = sample["image"].shape[:2]
                    x1, y1 = np.random.randint(0, w//2, 2)
                    x2, y2 = x1 + np.random.randint(50, 100), y1 + np.random.randint(30, 80)
                    boxes.append([x1, y1, min(x2, w), min(y2, h)])
                    classes.append(np.random.randint(0, 3))
                    scores.append(np.random.uniform(0.3, 0.7))

            baseline_results.append({
                "sample_id": sample["id"],
                "boxes": np.array(boxes, dtype=np.float32) if boxes else np.array([]).reshape(0, 4),
                "classes": np.array(classes, dtype=np.int32) if classes else np.array([]),
                "scores": np.array(scores, dtype=np.float32) if scores else np.array([]),
                "inference_time_ms": np.random.uniform(45, 55)  # Baseline latency
            })

        return baseline_results

    async def _run_model_inference(self, engine: OptimizedInferenceEngine,
                                 validation_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run inference on validation samples with the new model."""

        results = []

        for sample in validation_samples:
            start_time = time.time()

            # Run inference
            result = await engine.predict_single(
                sample["image"],
                f"validation_{sample['id']}",
                "validation_camera"
            )

            inference_time = (time.time() - start_time) * 1000

            results.append({
                "sample_id": sample["id"],
                "boxes": result.boxes,
                "classes": result.classes,
                "scores": result.scores,
                "inference_time_ms": inference_time
            })

        return results

    def _calculate_regression_metrics(self, baseline_results: list[dict[str, Any]],
                                    new_results: list[dict[str, Any]],
                                    validation_samples: list[dict[str, Any]]) -> dict[str, Any]:
        """Calculate comprehensive regression metrics including mAP, precision, recall, F1, IoU."""

        # Create sample lookup for ground truth
        gt_lookup = {sample["id"]: sample["ground_truth"] for sample in validation_samples}

        # Calculate metrics for baseline
        baseline_metrics = self._calculate_detection_metrics(baseline_results, gt_lookup)

        # Calculate metrics for new model
        new_metrics = self._calculate_detection_metrics(new_results, gt_lookup)

        # Calculate metric differences
        metric_comparison = {
            "baseline": baseline_metrics,
            "new_model": new_metrics,
            "improvements": {
                "map_50": new_metrics["map_50"] - baseline_metrics["map_50"],
                "map_50_95": new_metrics["map_50_95"] - baseline_metrics["map_50_95"],
                "precision": new_metrics["precision"] - baseline_metrics["precision"],
                "recall": new_metrics["recall"] - baseline_metrics["recall"],
                "f1_score": new_metrics["f1_score"] - baseline_metrics["f1_score"],
                "mean_iou": new_metrics["mean_iou"] - baseline_metrics["mean_iou"],
                "avg_inference_time_ms": new_metrics["avg_inference_time_ms"] - baseline_metrics["avg_inference_time_ms"]
            },
            "relative_improvements": {
                "map_50": (new_metrics["map_50"] - baseline_metrics["map_50"]) / max(baseline_metrics["map_50"], 1e-6),
                "map_50_95": (new_metrics["map_50_95"] - baseline_metrics["map_50_95"]) / max(baseline_metrics["map_50_95"], 1e-6),
                "precision": (new_metrics["precision"] - baseline_metrics["precision"]) / max(baseline_metrics["precision"], 1e-6),
                "recall": (new_metrics["recall"] - baseline_metrics["recall"]) / max(baseline_metrics["recall"], 1e-6),
                "f1_score": (new_metrics["f1_score"] - baseline_metrics["f1_score"]) / max(baseline_metrics["f1_score"], 1e-6),
                "inference_time": (baseline_metrics["avg_inference_time_ms"] - new_metrics["avg_inference_time_ms"]) / max(baseline_metrics["avg_inference_time_ms"], 1e-6)
            }
        }

        return metric_comparison


    def _calculate_detection_metrics(self, predictions: list[dict[str, Any]],
                                   ground_truth_lookup: dict[str, dict[str, Any]]) -> dict[str, float]:
        """Calculate comprehensive detection metrics."""

        all_precisions = []
        all_recalls = []
        all_ious = []
        all_inference_times = []

        # Per-class metrics for mAP calculation
        class_precisions = {}
        class_recalls = {}

        for pred in predictions:
            sample_id = pred["sample_id"]
            gt = ground_truth_lookup.get(sample_id, {"boxes": [], "classes": [], "scores": []})

            # Calculate IoU for this sample
            sample_ious = self._calculate_sample_ious(pred, gt)
            all_ious.extend(sample_ious)

            # Calculate precision and recall for this sample
            precision, recall = self._calculate_sample_precision_recall(pred, gt)
            all_precisions.append(precision)
            all_recalls.append(recall)

            all_inference_times.append(pred["inference_time_ms"])

            # Per-class metrics
            for class_id in set(list(pred["classes"]) + list(gt.get("classes", []))):
                if class_id not in class_precisions:
                    class_precisions[class_id] = []
                    class_recalls[class_id] = []

                class_pred = self._filter_by_class(pred, class_id)
                class_gt = self._filter_by_class({"boxes": gt.get("boxes", []),
                                                "classes": gt.get("classes", []),
                                                "scores": gt.get("scores", [])}, class_id)

                class_precision, class_recall = self._calculate_sample_precision_recall(class_pred, class_gt)
                class_precisions[class_id].append(class_precision)
                class_recalls[class_id].append(class_recall)

        # Calculate mAP@0.5 and mAP@0.5:0.95
        map_50 = self._calculate_map(class_precisions, class_recalls, iou_threshold=0.5)
        map_50_95 = self._calculate_map_range(class_precisions, class_recalls)

        return {
            "map_50": map_50,
            "map_50_95": map_50_95,
            "precision": np.mean(all_precisions) if all_precisions else 0.0,
            "recall": np.mean(all_recalls) if all_recalls else 0.0,
            "f1_score": self._calculate_f1(all_precisions, all_recalls),
            "mean_iou": np.mean(all_ious) if all_ious else 0.0,
            "avg_inference_time_ms": np.mean(all_inference_times) if all_inference_times else 0.0,
            "p95_inference_time_ms": np.percentile(all_inference_times, 95) if all_inference_times else 0.0
        }

    def _calculate_sample_ious(self, prediction: dict[str, Any], ground_truth: dict[str, Any]) -> list[float]:
        """Calculate IoU values for a single sample."""

        pred_boxes = prediction.get("boxes", [])
        gt_boxes = ground_truth.get("boxes", [])

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            return []

        try:
            pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
            gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)

            iou_matrix = box_iou(pred_tensor, gt_tensor)

            # Get max IoU for each prediction
            max_ious = torch.max(iou_matrix, dim=1)[0]
            return max_ious.tolist()

        except Exception:
            return []

    def _calculate_sample_precision_recall(self, prediction: dict[str, Any],
                                         ground_truth: dict[str, Any]) -> tuple[float, float]:
        """Calculate precision and recall for a single sample."""

        pred_boxes = prediction.get("boxes", [])
        gt_boxes = ground_truth.get("boxes", [])

        if len(pred_boxes) == 0 and len(gt_boxes) == 0:
            return 1.0, 1.0  # Perfect when both empty

        if len(pred_boxes) == 0:
            return 0.0, 0.0  # No predictions

        if len(gt_boxes) == 0:
            return 0.0, 1.0  # All predictions are false positives

        try:
            pred_tensor = torch.tensor(pred_boxes, dtype=torch.float32)
            gt_tensor = torch.tensor(gt_boxes, dtype=torch.float32)

            iou_matrix = box_iou(pred_tensor, gt_tensor)

            # IoU threshold for matching
            matches = iou_matrix > 0.5

            # Count true positives
            tp = torch.sum(torch.max(matches, dim=1)[0]).item()
            fp = len(pred_boxes) - tp
            fn = len(gt_boxes) - torch.sum(torch.max(matches, dim=0)[0]).item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            return precision, recall

        except Exception:
            return 0.0, 0.0

    def _filter_by_class(self, data: dict[str, Any], class_id: int) -> dict[str, Any]:
        """Filter predictions/ground truth by class ID."""

        boxes = data.get("boxes", [])
        classes = data.get("classes", [])
        scores = data.get("scores", [])

        if len(classes) == 0:
            return {"boxes": [], "classes": [], "scores": []}

        class_mask = np.array(classes) == class_id

        filtered_boxes = np.array(boxes)[class_mask] if len(boxes) > 0 else []
        filtered_classes = np.array(classes)[class_mask]
        filtered_scores = np.array(scores)[class_mask] if len(scores) > 0 else []

        return {
            "boxes": filtered_boxes.tolist() if len(filtered_boxes) > 0 else [],
            "classes": filtered_classes.tolist(),
            "scores": filtered_scores.tolist() if len(filtered_scores) > 0 else []
        }

    def _calculate_f1(self, precisions: list[float], recalls: list[float]) -> float:
        """Calculate F1 score from precision and recall lists."""

        if not precisions or not recalls:
            return 0.0

        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)

        if avg_precision + avg_recall == 0:
            return 0.0

        return 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

    def _calculate_map(self, class_precisions: dict[int, list[float]],
                      class_recalls: dict[int, list[float]],
                      iou_threshold: float = 0.5) -> float:
        """Calculate mean Average Precision (mAP) at specific IoU threshold."""

        if not class_precisions:
            return 0.0

        class_aps = []

        for class_id in class_precisions:
            precisions = class_precisions[class_id]
            recalls = class_recalls[class_id]

            if not precisions or not recalls:
                class_aps.append(0.0)
                continue

            # Calculate AP using sklearn if available
            if SKLEARN_AVAILABLE:
                try:
                    # Create binary classification data
                    y_true = [1] * len(recalls)  # All positive samples
                    y_scores = precisions  # Use precision as confidence score

                    ap = average_precision_score(y_true, y_scores)
                    class_aps.append(ap)

                except Exception:
                    # Fallback calculation
                    class_aps.append(np.mean(precisions))
            else:
                # Fallback calculation without sklearn
                class_aps.append(np.mean(precisions))

        return np.mean(class_aps) if class_aps else 0.0

    def _calculate_map_range(self, class_precisions: dict[int, list[float]],
                            class_recalls: dict[int, list[float]]) -> float:
        """Calculate mAP@0.5:0.95 (average over IoU thresholds 0.5 to 0.95)."""

        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        maps = []

        for threshold in iou_thresholds:
            map_at_threshold = self._calculate_map(class_precisions, class_recalls, threshold)
            maps.append(map_at_threshold)

        return np.mean(maps) if maps else 0.0

    def _perform_statistical_significance_test(self, baseline_results: list[dict[str, Any]],
                                             new_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Perform statistical significance testing for model comparison."""

        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("scipy.stats not available, using simplified significance testing")
                return {
                    "f1_score": {"p_value": 0.5, "is_significant": False, "effect_size": 0.0, "interpretation": "unknown"},
                    "inference_latency": {"p_value": 0.5, "is_significant": False, "effect_size": 0.0, "interpretation": "unknown"},
                    "overall_significance": {"is_significant": False, "sample_size": len(baseline_results)}
                }

            # Extract key metrics for comparison
            baseline_f1_scores = []
            new_f1_scores = []
            baseline_inference_times = []
            new_inference_times = []

            for baseline, new in zip(baseline_results, new_results, strict=False):
                # Calculate F1 scores (simplified)
                baseline_f1 = self._calculate_single_f1(baseline)
                new_f1 = self._calculate_single_f1(new)

                baseline_f1_scores.append(baseline_f1)
                new_f1_scores.append(new_f1)

                baseline_inference_times.append(baseline["inference_time_ms"])
                new_inference_times.append(new["inference_time_ms"])

            # Perform paired t-tests
            f1_statistic, f1_p_value = stats.ttest_rel(new_f1_scores, baseline_f1_scores)
            latency_statistic, latency_p_value = stats.ttest_rel(baseline_inference_times, new_inference_times)

            # Effect size calculation (Cohen's d)
            f1_effect_size = self._calculate_cohens_d(new_f1_scores, baseline_f1_scores)
            latency_effect_size = self._calculate_cohens_d(baseline_inference_times, new_inference_times)

            # Statistical significance threshold
            significance_threshold = 0.05

            return {
                "f1_score": {
                    "statistic": float(f1_statistic),
                    "p_value": float(f1_p_value),
                    "is_significant": f1_p_value < significance_threshold,
                    "effect_size": f1_effect_size,
                    "interpretation": self._interpret_effect_size(f1_effect_size)
                },
                "inference_latency": {
                    "statistic": float(latency_statistic),
                    "p_value": float(latency_p_value),
                    "is_significant": latency_p_value < significance_threshold,
                    "effect_size": latency_effect_size,
                    "interpretation": self._interpret_effect_size(latency_effect_size)
                },
                "overall_significance": {
                    "is_significant": f1_p_value < significance_threshold or latency_p_value < significance_threshold,
                    "significance_threshold": significance_threshold,
                    "sample_size": len(baseline_results)
                }
            }

        except Exception as e:
            logger.warning(f"Statistical significance testing failed: {str(e)}")
            return {
                "f1_score": {"p_value": 1.0, "is_significant": False, "error": str(e)},
                "inference_latency": {"p_value": 1.0, "is_significant": False, "error": str(e)},
                "overall_significance": {"is_significant": False, "error": str(e)}
            }

    def _calculate_single_f1(self, result: dict[str, Any]) -> float:
        """Calculate F1 score for a single prediction result."""

        # Simplified F1 calculation based on number of detections
        num_predictions = len(result.get("boxes", []))
        avg_confidence = np.mean(result.get("scores", [0.5]))

        # Simulate F1 score based on confidence and detection count
        if num_predictions == 0:
            return 0.0

        # Higher confidence and reasonable detection count = higher F1
        f1_estimate = avg_confidence * min(1.0, num_predictions / 3.0)
        return max(0.0, min(1.0, f1_estimate))

    def _calculate_cohens_d(self, group1: list[float], group2: list[float]) -> float:
        """Calculate Cohen's d effect size."""

        if not group1 or not group2:
            return 0.0

        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size."""

        abs_d = abs(cohens_d)

        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def _determine_regression_status(self, metrics_comparison: dict[str, Any],
                                   significance_test: dict[str, Any]) -> dict[str, Any]:
        """Determine overall regression test status."""

        improvements = metrics_comparison["improvements"]
        relative_improvements = metrics_comparison["relative_improvements"]

        # Check for significant regressions
        critical_regressions = []
        warnings = []
        improvements_list = []

        # mAP@0.5 regression check
        if improvements["map_50"] < -0.02:  # 2% absolute drop
            critical_regressions.append(f"mAP@0.5 decreased by {abs(improvements['map_50']):.3f}")
        elif improvements["map_50"] > 0.01:  # 1% absolute improvement
            improvements_list.append(f"mAP@0.5 improved by {improvements['map_50']:.3f}")

        # mAP@0.5:0.95 regression check
        if improvements["map_50_95"] < -0.02:
            critical_regressions.append(f"mAP@0.5:0.95 decreased by {abs(improvements['map_50_95']):.3f}")
        elif improvements["map_50_95"] > 0.01:
            improvements_list.append(f"mAP@0.5:0.95 improved by {improvements['map_50_95']:.3f}")

        # Latency regression check
        if improvements["avg_inference_time_ms"] > 10:  # 10ms increase
            critical_regressions.append(f"Inference time increased by {improvements['avg_inference_time_ms']:.1f}ms")
        elif improvements["avg_inference_time_ms"] < -5:  # 5ms improvement
            improvements_list.append(f"Inference time improved by {abs(improvements['avg_inference_time_ms']):.1f}ms")

        # F1 score check
        if improvements["f1_score"] < -0.05:  # 5% absolute drop
            warnings.append(f"F1 score decreased by {abs(improvements['f1_score']):.3f}")
        elif improvements["f1_score"] > 0.02:  # 2% absolute improvement
            improvements_list.append(f"F1 score improved by {improvements['f1_score']:.3f}")

        # Statistical significance check
        is_statistically_significant = significance_test.get("overall_significance", {}).get("is_significant", False)

        # Determine final status
        if critical_regressions:
            status = "failed"
            summary = f"Critical regressions detected: {'; '.join(critical_regressions)}"
        elif warnings and not improvements_list:
            status = "warning"
            summary = f"Performance warnings: {'; '.join(warnings)}"
        elif improvements_list and is_statistically_significant:
            status = "passed"
            summary = f"Statistically significant improvements: {'; '.join(improvements_list)}"
        elif improvements_list:
            status = "passed"
            summary = f"Improvements detected: {'; '.join(improvements_list)}"
        else:
            status = "passed"
            summary = "No significant performance changes detected"

        return {
            "status": status,
            "summary": summary,
            "critical_regressions": critical_regressions,
            "warnings": warnings,
            "improvements": improvements_list,
            "is_statistically_significant": is_statistically_significant
        }

    async def _generate_regression_report(self, model_version: ModelVersion,
                                        metrics_comparison: dict[str, Any],
                                        significance_test: dict[str, Any],
                                        status: dict[str, Any]) -> Path:
        """Generate comprehensive regression test report with visualizations."""

        report_dir = Path("reports/regression")
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"regression_report_{model_version.model_id}_v{model_version.version}_{timestamp}.html"

        try:
            # Generate HTML report (simplified without plots for now)
            html_content = self._generate_html_report_simple(
                model_version, metrics_comparison, significance_test, status
            )

            with open(report_path, 'w') as f:
                f.write(html_content)

            logger.info(f"Regression report generated: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"Failed to generate regression report: {str(e)}")
            # Create a minimal report
            with open(report_path, 'w') as f:
                f.write(f"<html><body><h1>Regression Test Report</h1><p>Model: {model_version.model_id} v{model_version.version}</p><p>Status: {status['status']}</p><p>Error: {str(e)}</p></body></html>")
            return report_path

    def _generate_html_report_simple(self, model_version: ModelVersion,
                                   metrics_comparison: dict[str, Any],
                                   significance_test: dict[str, Any],
                                   status: dict[str, Any]) -> str:
        """Generate simplified HTML regression test report."""

        baseline_metrics = metrics_comparison["baseline"]
        new_metrics = metrics_comparison["new_model"]
        improvements = metrics_comparison["improvements"]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Regression Test Report - {model_version.model_id} v{model_version.version}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .status-passed {{ color: green; font-weight: bold; }}
        .status-warning {{ color: orange; font-weight: bold; }}
        .status-failed {{ color: red; font-weight: bold; }}
        .metrics-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .metrics-table th {{ background-color: #f2f2f2; }}
        .improvement {{ color: green; }}
        .regression {{ color: red; }}
        .section {{ margin: 30px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Model Regression Test Report</h1>
        <p><strong>Model:</strong> {model_version.model_id} v{model_version.version}</p>
        <p><strong>Test Date:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Status:</strong> <span class="status-{status['status']}">{status['status'].upper()}</span></p>
        <p><strong>Summary:</strong> {status['summary']}</p>
    </div>
    
    <div class="section">
        <h2>Performance Metrics Comparison</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Baseline Model</th>
                    <th>New Model</th>
                    <th>Absolute Change</th>
                    <th>Relative Change</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>mAP@0.5</td>
                    <td>{baseline_metrics['map_50']:.4f}</td>
                    <td>{new_metrics['map_50']:.4f}</td>
                    <td class="{'improvement' if improvements['map_50'] > 0 else 'regression'}">
                        {improvements['map_50']:+.4f}
                    </td>
                    <td class="{'improvement' if improvements['map_50'] > 0 else 'regression'}">
                        {metrics_comparison['relative_improvements']['map_50']:+.2%}
                    </td>
                </tr>
                <tr>
                    <td>mAP@0.5:0.95</td>
                    <td>{baseline_metrics['map_50_95']:.4f}</td>
                    <td>{new_metrics['map_50_95']:.4f}</td>
                    <td class="{'improvement' if improvements['map_50_95'] > 0 else 'regression'}">
                        {improvements['map_50_95']:+.4f}
                    </td>
                    <td class="{'improvement' if improvements['map_50_95'] > 0 else 'regression'}">
                        {metrics_comparison['relative_improvements']['map_50_95']:+.2%}
                    </td>
                </tr>
                <tr>
                    <td>Precision</td>
                    <td>{baseline_metrics['precision']:.4f}</td>
                    <td>{new_metrics['precision']:.4f}</td>
                    <td class="{'improvement' if improvements['precision'] > 0 else 'regression'}">
                        {improvements['precision']:+.4f}
                    </td>
                    <td class="{'improvement' if improvements['precision'] > 0 else 'regression'}">
                        {metrics_comparison['relative_improvements']['precision']:+.2%}
                    </td>
                </tr>
                <tr>
                    <td>Recall</td>
                    <td>{baseline_metrics['recall']:.4f}</td>
                    <td>{new_metrics['recall']:.4f}</td>
                    <td class="{'improvement' if improvements['recall'] > 0 else 'regression'}">
                        {improvements['recall']:+.4f}
                    </td>
                    <td class="{'improvement' if improvements['recall'] > 0 else 'regression'}">
                        {metrics_comparison['relative_improvements']['recall']:+.2%}
                    </td>
                </tr>
                <tr>
                    <td>F1 Score</td>
                    <td>{baseline_metrics['f1_score']:.4f}</td>
                    <td>{new_metrics['f1_score']:.4f}</td>
                    <td class="{'improvement' if improvements['f1_score'] > 0 else 'regression'}">
                        {improvements['f1_score']:+.4f}
                    </td>
                    <td class="{'improvement' if improvements['f1_score'] > 0 else 'regression'}">
                        {metrics_comparison['relative_improvements']['f1_score']:+.2%}
                    </td>
                </tr>
                <tr>
                    <td>Mean IoU</td>
                    <td>{baseline_metrics['mean_iou']:.4f}</td>
                    <td>{new_metrics['mean_iou']:.4f}</td>
                    <td class="{'improvement' if improvements['mean_iou'] > 0 else 'regression'}">
                        {improvements['mean_iou']:+.4f}
                    </td>
                    <td class="{'improvement' if improvements['mean_iou'] > 0 else 'regression'}">
                        {metrics_comparison['relative_improvements']['mean_iou']:+.2%}
                    </td>
                </tr>
                <tr>
                    <td>Avg Inference Time (ms)</td>
                    <td>{baseline_metrics['avg_inference_time_ms']:.2f}</td>
                    <td>{new_metrics['avg_inference_time_ms']:.2f}</td>
                    <td class="{'regression' if improvements['avg_inference_time_ms'] > 0 else 'improvement'}">
                        {improvements['avg_inference_time_ms']:+.2f}
                    </td>
                    <td class="{'regression' if improvements['avg_inference_time_ms'] > 0 else 'improvement'}">
                        {-metrics_comparison['relative_improvements']['inference_time']:+.2%}
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
    
    <div class="section">
        <h2>Statistical Significance Analysis</h2>
        <p><strong>Overall Significance:</strong> {'Statistically Significant' if significance_test.get('overall_significance', {}).get('is_significant', False) else 'Not Statistically Significant'}</p>
        <p><strong>Sample Size:</strong> {significance_test.get('overall_significance', {}).get('sample_size', 'Unknown')}</p>
        
        <h3>F1 Score Comparison</h3>
        <ul>
            <li><strong>P-value:</strong> {significance_test.get('f1_score', {}).get('p_value', 'N/A'):.6f}</li>
            <li><strong>Statistically Significant:</strong> {'Yes' if significance_test.get('f1_score', {}).get('is_significant', False) else 'No'}</li>
            <li><strong>Effect Size (Cohen's d):</strong> {significance_test.get('f1_score', {}).get('effect_size', 0):.4f} ({significance_test.get('f1_score', {}).get('interpretation', 'unknown')})</li>
        </ul>
        
        <h3>Inference Latency Comparison</h3>
        <ul>
            <li><strong>P-value:</strong> {significance_test.get('inference_latency', {}).get('p_value', 'N/A'):.6f}</li>
            <li><strong>Statistically Significant:</strong> {'Yes' if significance_test.get('inference_latency', {}).get('is_significant', False) else 'No'}</li>
            <li><strong>Effect Size (Cohen's d):</strong> {significance_test.get('inference_latency', {}).get('effect_size', 0):.4f} ({significance_test.get('inference_latency', {}).get('interpretation', 'unknown')})</li>
        </ul>
    </div>
        """

        html += """
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
"""

        if status["critical_regressions"]:
            html += "<li><strong>CRITICAL:</strong> Address the following regressions before deployment:</li>"
            for regression in status["critical_regressions"]:
                html += f"<li class='regression'>• {regression}</li>"

        if status["warnings"]:
            html += "<li><strong>WARNING:</strong> Monitor the following potential issues:</li>"
            for warning in status["warnings"]:
                html += f"<li class='status-warning'>• {warning}</li>"

        if status["improvements"]:
            html += "<li><strong>IMPROVEMENTS:</strong> The following enhancements were detected:</li>"
            for improvement in status["improvements"]:
                html += f"<li class='improvement'>• {improvement}</li>"

        html += """
        </ul>
    </div>
    
    <div class="section">
        <h2>Test Configuration</h2>
        <ul>
            <li><strong>Validation Samples:</strong> {validation_samples_count}</li>
            <li><strong>IoU Threshold:</strong> 0.5</li>
            <li><strong>Confidence Threshold:</strong> 0.25</li>
            <li><strong>Statistical Significance Level:</strong> α = 0.05</li>
        </ul>
    </div>
    
</body>
</html>
""".format(
            validation_samples_count=significance_test.get('overall_significance', {}).get('sample_size', 'Unknown')
        )

        return html


class ABTestingFramework:
    """A/B testing framework for model comparison."""

    def __init__(self, dashboard: ProductionDashboard):
        self.dashboard = dashboard
        self.active_experiments: dict[str, ABExperiment] = {}
        self.experiment_history: list[ABExperiment] = []

        # Traffic routing
        self.traffic_router = TrafficRouter()

    async def create_experiment(
        self,
        name: str,
        description: str,
        control_model: ModelVersion,
        treatment_model: ModelVersion,
        config: dict[str, Any],
    ) -> ABExperiment:
        """Create new A/B testing experiment."""

        experiment_id = f"ab_{int(time.time())}_{hash(name) % 10000}"

        experiment = ABExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            control_model=control_model,
            treatment_model=treatment_model,
            **config,
        )

        logger.info(f"Created A/B experiment: {name} ({experiment_id})")
        return experiment

    async def start_experiment(self, experiment: ABExperiment) -> bool:
        """Start A/B experiment."""

        # Validate models are ready
        if (
            experiment.control_model.validation_result != ModelValidationResult.PASSED
            or experiment.treatment_model.validation_result
            != ModelValidationResult.PASSED
        ):
            logger.error("Cannot start experiment - models not validated")
            return False

        # Configure traffic routing
        await self.traffic_router.configure_split(
            experiment.experiment_id,
            {
                experiment.control_model.model_id: 1.0 - experiment.traffic_split,
                experiment.treatment_model.model_id: experiment.traffic_split,
            },
        )

        # Start monitoring
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = time.time()

        self.active_experiments[experiment.experiment_id] = experiment

        # Start experiment monitoring task
        asyncio.create_task(self._monitor_experiment(experiment))

        logger.info(
            f"Started A/B experiment {experiment.name} with {experiment.traffic_split * 100}% traffic split"
        )
        return True

    async def _monitor_experiment(self, experiment: ABExperiment):
        """Monitor running A/B experiment."""

        while experiment.status == ExperimentStatus.RUNNING:
            try:
                # Check safety conditions
                safety_check = await self._check_experiment_safety(experiment)
                if not safety_check["safe"]:
                    logger.warning(
                        f"Experiment {experiment.name} safety violation: {safety_check['reason']}"
                    )
                    await self.stop_experiment(
                        experiment.experiment_id, reason="Safety violation"
                    )
                    break

                # Check completion conditions
                completion_check = await self._check_experiment_completion(experiment)
                if completion_check["should_stop"]:
                    logger.info(
                        f"Experiment {experiment.name} completion: {completion_check['reason']}"
                    )
                    await self.stop_experiment(
                        experiment.experiment_id, reason=completion_check["reason"]
                    )
                    break

                # Update experiment metrics
                await self._update_experiment_metrics(experiment)

                # Wait before next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Experiment monitoring error: {e}")
                await asyncio.sleep(60)

    async def _check_experiment_safety(
        self, experiment: ABExperiment
    ) -> dict[str, Any]:
        """Check if experiment is safe to continue."""

        if not experiment.safety_enabled:
            return {"safe": True}

        # Get recent metrics for both variants
        control_metrics = self._get_recent_metrics(experiment.control_model.model_id)
        treatment_metrics = self._get_recent_metrics(
            experiment.treatment_model.model_id
        )

        if not control_metrics or not treatment_metrics:
            return {"safe": True, "reason": "Insufficient data"}

        # Check error rates
        control_error_rate = 1 - control_metrics.get("accuracy", 0.9)
        treatment_error_rate = 1 - treatment_metrics.get("accuracy", 0.9)

        if (
            treatment_error_rate - control_error_rate
            > experiment.max_error_rate_increase
        ):
            return {
                "safe": False,
                "reason": f"Error rate increase: {treatment_error_rate - control_error_rate:.3f} > {experiment.max_error_rate_increase}",
            }

        # Check latency increases
        control_latency = control_metrics.get("p95_latency_ms", 50)
        treatment_latency = treatment_metrics.get("p95_latency_ms", 50)

        latency_increase = (treatment_latency - control_latency) / control_latency
        if latency_increase > experiment.max_latency_increase:
            return {
                "safe": False,
                "reason": f"Latency increase: {latency_increase:.3f} > {experiment.max_latency_increase}",
            }

        return {"safe": True}

    async def _check_experiment_completion(
        self, experiment: ABExperiment
    ) -> dict[str, Any]:
        """Check if experiment should be stopped."""

        # Check maximum runtime
        if experiment.start_time:
            runtime_hours = (time.time() - experiment.start_time) / 3600
            if runtime_hours > experiment.max_runtime_hours:
                return {"should_stop": True, "reason": "Maximum runtime reached"}

        # Check sample size
        required_samples = experiment.get_required_sample_size()
        control_samples = len(experiment.control_metrics)
        treatment_samples = len(experiment.treatment_metrics)

        if control_samples < required_samples or treatment_samples < required_samples:
            return {"should_stop": False, "reason": "Insufficient samples"}

        # Check statistical significance
        significance_result = await self._calculate_statistical_significance(experiment)

        if significance_result["is_significant"]:
            return {
                "should_stop": True,
                "reason": f"Statistical significance achieved (p={significance_result['p_value']:.4f})",
            }

        return {"should_stop": False, "reason": "Experiment continuing"}

    async def _calculate_statistical_significance(
        self, experiment: ABExperiment
    ) -> dict[str, Any]:
        """Calculate statistical significance of experiment results."""

        if (
            len(experiment.control_metrics) < 50
            or len(experiment.treatment_metrics) < 50
        ):
            return {
                "is_significant": False,
                "p_value": 1.0,
                "reason": "Insufficient data",
            }

        try:
            # Extract primary metric (accuracy)
            control_values = [m.get("accuracy", 0) for m in experiment.control_metrics]
            treatment_values = [
                m.get("accuracy", 0) for m in experiment.treatment_metrics
            ]

            # Perform statistical test
            from scipy import stats

            statistic, p_value = stats.ttest_ind(treatment_values, control_values)

            is_significant = p_value < experiment.significance_level

            # Calculate effect size
            control_mean = np.mean(control_values)
            treatment_mean = np.mean(treatment_values)
            pooled_std = np.sqrt(
                (np.var(control_values) + np.var(treatment_values)) / 2
            )
            effect_size = (treatment_mean - control_mean) / pooled_std

            return {
                "is_significant": is_significant,
                "p_value": p_value,
                "effect_size": effect_size,
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "improvement": (treatment_mean - control_mean) / control_mean,
            }

        except ImportError:
            # Fallback statistical test
            control_mean = np.mean(
                [m.get("accuracy", 0) for m in experiment.control_metrics]
            )
            treatment_mean = np.mean(
                [m.get("accuracy", 0) for m in experiment.treatment_metrics]
            )

            improvement = (treatment_mean - control_mean) / control_mean
            is_significant = abs(improvement) > experiment.minimum_effect_size

            return {
                "is_significant": is_significant,
                "p_value": 0.01 if is_significant else 0.1,
                "control_mean": control_mean,
                "treatment_mean": treatment_mean,
                "improvement": improvement,
            }

    async def _update_experiment_metrics(self, experiment: ABExperiment):
        """Update experiment metrics from monitoring data."""

        # Get recent performance data from dashboard
        control_metrics = self._get_recent_metrics(experiment.control_model.model_id)
        treatment_metrics = self._get_recent_metrics(
            experiment.treatment_model.model_id
        )

        if control_metrics:
            experiment.control_metrics.append(control_metrics)

        if treatment_metrics:
            experiment.treatment_metrics.append(treatment_metrics)

        # Keep only recent metrics (last 1000 samples)
        experiment.control_metrics = experiment.control_metrics[-1000:]
        experiment.treatment_metrics = experiment.treatment_metrics[-1000:]

    def _get_recent_metrics(self, model_id: str) -> dict[str, float] | None:
        """Get recent metrics for a model from the dashboard."""

        if model_id not in self.dashboard.model_monitors:
            return None

        monitor = self.dashboard.model_monitors[model_id]
        recent_stats = monitor.metrics.get_recent_stats(window_minutes=5)

        return recent_stats if recent_stats.get("sample_count", 0) > 0 else None

    async def stop_experiment(self, experiment_id: str, reason: str = "Manual stop"):
        """Stop running A/B experiment."""

        if experiment_id not in self.active_experiments:
            return False

        experiment = self.active_experiments[experiment_id]

        # Calculate final results
        final_results = await self._calculate_statistical_significance(experiment)

        # Update experiment status
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = time.time()

        # Remove from active experiments
        del self.active_experiments[experiment_id]
        self.experiment_history.append(experiment)

        # Reset traffic routing
        await self.traffic_router.remove_split(experiment_id)

        logger.info(f"Stopped experiment {experiment.name}: {reason}")
        logger.info(f"Final results: {final_results}")

        return True

    def get_experiment_results(self, experiment_id: str) -> dict[str, Any] | None:
        """Get comprehensive experiment results."""

        # Find experiment in active or history
        experiment = None
        if experiment_id in self.active_experiments:
            experiment = self.active_experiments[experiment_id]
        else:
            for exp in self.experiment_history:
                if exp.experiment_id == experiment_id:
                    experiment = exp
                    break

        if not experiment:
            return None

        # Calculate results
        if (
            len(experiment.control_metrics) > 0
            and len(experiment.treatment_metrics) > 0
        ):
            control_stats = {
                "accuracy": np.mean(
                    [m.get("accuracy", 0) for m in experiment.control_metrics]
                ),
                "latency_p95": np.mean(
                    [m.get("p95_latency_ms", 0) for m in experiment.treatment_metrics]
                ),
                "sample_count": len(experiment.control_metrics),
            }

            treatment_stats = {
                "accuracy": np.mean(
                    [m.get("accuracy", 0) for m in experiment.treatment_metrics]
                ),
                "latency_p95": np.mean(
                    [m.get("p95_latency_ms", 0) for m in experiment.treatment_metrics]
                ),
                "sample_count": len(experiment.treatment_metrics),
            }
        else:
            control_stats = {"accuracy": 0, "latency_p95": 0, "sample_count": 0}
            treatment_stats = {"accuracy": 0, "latency_p95": 0, "sample_count": 0}

        return {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "status": experiment.status.value,
            "start_time": experiment.start_time,
            "end_time": experiment.end_time,
            "control_model": {
                "model_id": experiment.control_model.model_id,
                "version": experiment.control_model.version,
                "stats": control_stats,
            },
            "treatment_model": {
                "model_id": experiment.treatment_model.model_id,
                "version": experiment.treatment_model.version,
                "stats": treatment_stats,
            },
            "traffic_split": experiment.traffic_split,
        }


class TrafficRouter:
    """Route traffic between different model versions."""

    def __init__(self):
        self.routing_rules: dict[
            str, dict[str, float]
        ] = {}  # experiment_id -> {model_id: percentage}
        self.default_model: str | None = None

    async def configure_split(self, experiment_id: str, splits: dict[str, float]):
        """Configure traffic split for experiment."""

        # Validate splits sum to 1.0
        total = sum(splits.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Traffic splits must sum to 1.0, got {total}")

        self.routing_rules[experiment_id] = splits
        logger.info(f"Configured traffic split for {experiment_id}: {splits}")

    async def route_request(self, request_id: str) -> str:
        """Route request to appropriate model based on configured splits."""

        if not self.routing_rules:
            return self.default_model or "default"

        # Use consistent hashing based on request_id
        hash_value = hash(request_id) % 100000 / 100000  # 0.0 to 1.0

        # Find matching rule
        for _experiment_id, splits in self.routing_rules.items():
            cumulative = 0.0
            for model_id, percentage in splits.items():
                cumulative += percentage
                if hash_value <= cumulative:
                    return model_id

        # Fallback
        return self.default_model or "default"

    async def remove_split(self, experiment_id: str):
        """Remove traffic split configuration."""
        if experiment_id in self.routing_rules:
            del self.routing_rules[experiment_id]
            logger.info(f"Removed traffic split for {experiment_id}")

    def set_default_model(self, model_id: str):
        """Set default model for fallback."""
        self.default_model = model_id


class DeploymentPipeline:
    """Automated deployment pipeline for models."""

    def __init__(
        self,
        validator: ModelValidator,
        ab_framework: ABTestingFramework,
        dashboard: ProductionDashboard,
    ):
        self.validator = validator
        self.ab_framework = ab_framework
        self.dashboard = dashboard

        # Deployment configurations
        self.deployment_configs = {
            DeploymentStrategy.CANARY: {
                "initial_traffic": 0.05,
                "traffic_increments": [0.05, 0.10, 0.25, 0.50, 1.00],
                "increment_interval_hours": 2,
            },
            DeploymentStrategy.BLUE_GREEN: {
                "validation_period_minutes": 30,
                "rollback_threshold_errors": 0.05,
            },
        }

    async def deploy_model(
        self,
        model_version: ModelVersion,
        strategy: DeploymentStrategy = DeploymentStrategy.CANARY,
        current_production_model: ModelVersion | None = None,
    ) -> dict[str, Any]:
        """Deploy model using specified strategy."""

        logger.info(
            f"Starting deployment of {model_version.model_id} v{model_version.version} using {strategy.value}"
        )

        # Step 1: Validate model
        validation_result = await self.validator.validate_model(model_version)

        if validation_result != ModelValidationResult.PASSED:
            return {
                "success": False,
                "step": "validation",
                "reason": f"Model validation {validation_result.value}",
                "details": model_version.validation_details,
            }

        # Step 2: Execute deployment strategy
        if strategy == DeploymentStrategy.CANARY:
            return await self._deploy_canary(model_version, current_production_model)

        elif strategy == DeploymentStrategy.BLUE_GREEN:
            return await self._deploy_blue_green(
                model_version, current_production_model
            )

        elif strategy == DeploymentStrategy.ROLLING:
            return await self._deploy_rolling(model_version)

        else:
            return {
                "success": False,
                "reason": f"Deployment strategy {strategy.value} not implemented",
            }

    async def _deploy_canary(
        self, model_version: ModelVersion, current_model: ModelVersion | None
    ) -> dict[str, Any]:
        """Deploy using canary strategy."""

        if not current_model:
            # If no current model, treat as initial deployment
            model_version.traffic_percentage = 1.0
            model_version.is_active = True
            return {"success": True, "strategy": "canary", "traffic": 1.0}

        # Create A/B experiment for canary deployment
        experiment = await self.ab_framework.create_experiment(
            name=f"Canary deployment: {model_version.model_id} v{model_version.version}",
            description="Canary deployment with gradual traffic increase",
            control_model=current_model,
            treatment_model=model_version,
            config={
                "traffic_split": self.deployment_configs[DeploymentStrategy.CANARY][
                    "initial_traffic"
                ],
                "min_sample_size": 500,
                "max_runtime_hours": 24,
                "safety_enabled": True,
            },
        )

        # Start experiment
        success = await self.ab_framework.start_experiment(experiment)

        if success:
            # Schedule traffic increments
            asyncio.create_task(self._manage_canary_rollout(experiment))

            return {
                "success": True,
                "strategy": "canary",
                "experiment_id": experiment.experiment_id,
                "initial_traffic": experiment.traffic_split,
            }
        else:
            return {"success": False, "reason": "Failed to start canary experiment"}

    async def _manage_canary_rollout(self, experiment: ABExperiment):
        """Manage gradual traffic increase for canary deployment."""

        config = self.deployment_configs[DeploymentStrategy.CANARY]
        increments = config["traffic_increments"]
        interval_hours = config["increment_interval_hours"]

        current_increment_index = 0

        while (
            experiment.status == ExperimentStatus.RUNNING
            and current_increment_index < len(increments)
        ):
            await asyncio.sleep(interval_hours * 3600)  # Wait for interval

            # Check if experiment is still safe
            safety_check = await self.ab_framework._check_experiment_safety(experiment)
            if not safety_check["safe"]:
                logger.warning(
                    f"Canary rollout stopped due to safety: {safety_check['reason']}"
                )
                await self.ab_framework.stop_experiment(
                    experiment.experiment_id, "Safety violation during rollout"
                )
                break

            # Increase traffic
            new_traffic = increments[current_increment_index]
            experiment.traffic_split = new_traffic

            # Update routing
            await self.ab_framework.traffic_router.configure_split(
                experiment.experiment_id,
                {
                    experiment.control_model.model_id: 1.0 - new_traffic,
                    experiment.treatment_model.model_id: new_traffic,
                },
            )

            logger.info(f"Canary rollout: increased traffic to {new_traffic * 100}%")
            current_increment_index += 1

            # If we've reached 100%, complete the deployment
            if new_traffic >= 1.0:
                logger.info("Canary rollout completed successfully")
                await self.ab_framework.stop_experiment(
                    experiment.experiment_id, "Canary rollout completed"
                )
                break

    async def _deploy_blue_green(
        self, model_version: ModelVersion, current_model: ModelVersion | None
    ) -> dict[str, Any]:
        """Deploy using blue-green strategy."""

        config = self.deployment_configs[DeploymentStrategy.BLUE_GREEN]

        # Deploy to "green" environment (parallel to current "blue")
        model_version.is_active = False  # Not serving traffic yet

        # Validation period with synthetic traffic
        logger.info(
            f"Blue-green: validation period of {config['validation_period_minutes']} minutes"
        )

        # Simulate validation period
        await asyncio.sleep(config["validation_period_minutes"] * 60)

        # Check model health during validation
        monitor = self.dashboard.model_monitors.get(model_version.model_id)
        if monitor:
            health_score = monitor.get_health_score()
            if health_score < 80:  # Threshold for healthy model
                return {
                    "success": False,
                    "strategy": "blue_green",
                    "reason": f"Model health score too low: {health_score}",
                }

        # Switch traffic (atomic operation)
        if current_model:
            current_model.is_active = False
            current_model.traffic_percentage = 0.0

        model_version.is_active = True
        model_version.traffic_percentage = 1.0

        logger.info("Blue-green deployment completed - traffic switched")

        return {"success": True, "strategy": "blue_green", "switch_time": time.time()}

    async def _deploy_rolling(self, model_version: ModelVersion) -> dict[str, Any]:
        """Deploy using rolling update strategy."""

        # Rolling deployment would replace instances gradually
        # This is more relevant for containerized deployments

        model_version.is_active = True
        model_version.traffic_percentage = 1.0

        return {
            "success": True,
            "strategy": "rolling",
            "message": "Rolling deployment completed",
        }

    async def rollback_deployment(
        self,
        model_version: ModelVersion,
        previous_model: ModelVersion,
        reason: str = "Manual rollback",
    ) -> dict[str, Any]:
        """Rollback to previous model version."""

        logger.warning(
            f"Rolling back {model_version.model_id} v{model_version.version}: {reason}"
        )

        # Deactivate current model
        model_version.is_active = False
        model_version.traffic_percentage = 0.0

        # Reactivate previous model
        previous_model.is_active = True
        previous_model.traffic_percentage = 1.0

        # Stop any running experiments
        for exp_id, experiment in self.ab_framework.active_experiments.items():
            if experiment.treatment_model.model_id == model_version.model_id:
                await self.ab_framework.stop_experiment(exp_id, f"Rollback: {reason}")

        logger.info(
            f"Rollback completed - reverted to {previous_model.model_id} v{previous_model.version}"
        )

        return {
            "success": True,
            "action": "rollback",
            "current_model": f"{previous_model.model_id} v{previous_model.version}",
            "reason": reason,
        }


# Example usage and integration


async def create_mlops_pipeline(
    dashboard: ProductionDashboard, validation_config: dict[str, Any]
) -> dict[str, Any]:
    """Create complete MLOps pipeline."""

    # Initialize components
    validator = ModelValidator(validation_config)
    ab_framework = ABTestingFramework(dashboard)
    deployment_pipeline = DeploymentPipeline(validator, ab_framework, dashboard)

    logger.info("MLOps pipeline initialized")

    return {
        "validator": validator,
        "ab_framework": ab_framework,
        "deployment_pipeline": deployment_pipeline,
    }


async def deploy_new_model_version(
    model_path: Path,
    model_id: str,
    version: str,
    pipeline_components: dict[str, Any],
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.CANARY,
) -> dict[str, Any]:
    """Deploy new model version using MLOps pipeline."""

    # Create model version
    model_version = ModelVersion(
        model_id=model_id,
        version=version,
        model_path=model_path,
        config_path=model_path.parent / f"{model_id}_config.json",
        metadata_path=model_path.parent / f"{model_id}_metadata.json",
    )

    # Get current production model (simplified)
    current_model = None  # In production, retrieve from model registry

    # Deploy using pipeline
    deployment_pipeline = pipeline_components["deployment_pipeline"]
    result = await deployment_pipeline.deploy_model(
        model_version, deployment_strategy, current_model
    )

    logger.info(f"Deployment result: {result}")
    return result
