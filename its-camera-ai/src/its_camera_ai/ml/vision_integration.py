"""
Vision Integration Layer for ITS Camera AI Traffic Monitoring System.

This module provides integration between the Core Computer Vision Engine and the 
broader ML pipeline architecture, including model registry, federated learning,
monitoring systems, and production deployment workflows.

Key Features:
- Seamless integration with existing MLOps pipeline
- Model registry integration for version management
- Performance monitoring and drift detection
- A/B testing framework integration
- Federated learning coordination
- Production deployment automation
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .core_vision_engine import (
    CoreVisionEngine,
    VisionConfig,
    VisionResult,
    create_optimal_config,
)
from .ml_pipeline import (
    ExperimentTracker,
    ModelRegistry,
    ProductionMLPipeline,
)
from .model_pipeline import (
    ABTestingFramework,
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

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for vision-ML pipeline integration."""

    # Vision Engine Configuration
    vision_config: VisionConfig = field(default_factory=VisionConfig)

    # Integration Settings
    enable_model_registry: bool = True
    enable_federated_learning: bool = False
    enable_ab_testing: bool = True
    enable_drift_detection: bool = True

    # Performance Monitoring
    metrics_update_interval: int = 30  # seconds
    drift_check_interval: int = 300  # seconds
    model_validation_interval: int = 3600  # seconds

    # Deployment Settings
    auto_deployment_enabled: bool = False
    validation_threshold_accuracy: float = 0.90
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.CANARY

    # Model Management
    model_retention_count: int = 10
    auto_model_optimization: bool = True
    performance_baseline_frames: int = 1000


class VisionPipelineIntegration:
    """
    Integration layer between Core Vision Engine and ML Pipeline.
    
    Coordinates computer vision processing with model management,
    monitoring, and deployment systems.
    """

    def __init__(
        self,
        config: IntegrationConfig,
        ml_pipeline: ProductionMLPipeline | None = None
    ):
        self.config = config
        self.ml_pipeline = ml_pipeline

        # Initialize core vision engine
        self.vision_engine = CoreVisionEngine(config.vision_config)

        # Integration components
        self.model_registry: ModelRegistry | None = None
        self.experiment_tracker: ExperimentTracker | None = None
        self.dashboard: ProductionDashboard | None = None
        self.ab_framework: ABTestingFramework | None = None
        self.model_validator: ModelValidator | None = None

        # Integration state
        self.initialized = False
        self.current_model_version: ModelVersion | None = None
        self.baseline_metrics: dict[str, float] = {}
        self.drift_detection_enabled = config.enable_drift_detection

        # Performance tracking
        self.integration_metrics = {
            "total_processed": 0,
            "registry_updates": 0,
            "drift_detections": 0,
            "model_deployments": 0,
            "start_time": time.time(),
        }

        # Background tasks
        self.monitoring_tasks: list[asyncio.Task] = []

        logger.info("Vision Pipeline Integration initialized")

    async def initialize(self, model_path: Path | None = None) -> None:
        """Initialize the complete integrated vision pipeline."""
        logger.info("Initializing Vision Pipeline Integration...")

        # Initialize core vision engine
        await self.vision_engine.initialize(model_path)

        # Initialize ML pipeline components if provided
        if self.ml_pipeline:
            await self._initialize_ml_components()

        # Start background monitoring tasks
        await self._start_monitoring_tasks()

        # Establish baseline metrics
        await self._establish_baseline_metrics()

        self.initialized = True
        logger.info("Vision Pipeline Integration initialized successfully")

    async def _initialize_ml_components(self) -> None:
        """Initialize ML pipeline component integrations."""
        if not self.ml_pipeline:
            return

        # Connect to model registry
        if self.config.enable_model_registry:
            self.model_registry = self.ml_pipeline.model_registry
            logger.info("Connected to model registry")

        # Connect to experiment tracker
        self.experiment_tracker = self.ml_pipeline.experiment_tracker
        logger.info("Connected to experiment tracker")

        # Connect to monitoring dashboard
        self.dashboard = self.ml_pipeline.dashboard
        logger.info("Connected to production dashboard")

        # Connect to A/B testing framework
        if self.config.enable_ab_testing:
            self.ab_framework = self.ml_pipeline.ab_framework
            logger.info("Connected to A/B testing framework")

        # Connect to model validator
        self.model_validator = self.ml_pipeline.deployment_pipeline.model_validator
        logger.info("Connected to model validator")

    async def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""

        # Metrics collection task
        self.monitoring_tasks.append(
            asyncio.create_task(self._metrics_collection_loop())
        )

        # Drift detection task
        if self.drift_detection_enabled:
            self.monitoring_tasks.append(
                asyncio.create_task(self._drift_detection_loop())
            )

        # Model validation task
        if self.model_validator:
            self.monitoring_tasks.append(
                asyncio.create_task(self._model_validation_loop())
            )

        # Registry synchronization task
        if self.model_registry:
            self.monitoring_tasks.append(
                asyncio.create_task(self._registry_sync_loop())
            )

        logger.info(f"Started {len(self.monitoring_tasks)} monitoring tasks")

    async def _establish_baseline_metrics(self) -> None:
        """Establish baseline performance metrics."""
        logger.info("Establishing baseline performance metrics...")

        # Generate synthetic frames for baseline
        baseline_frames = []
        for i in range(self.config.performance_baseline_frames):
            # Create realistic traffic scene
            frame = self._generate_synthetic_traffic_frame()
            baseline_frames.append(frame)

        # Process baseline frames
        baseline_results = []
        batch_size = self.config.vision_config.batch_size

        for i in range(0, len(baseline_frames), batch_size):
            batch = baseline_frames[i:i + batch_size]
            frame_ids = [f"baseline_{j}" for j in range(i, i + len(batch))]
            camera_ids = ["baseline_camera"] * len(batch)

            try:
                results = await self.vision_engine.process_batch(batch, frame_ids, camera_ids)
                baseline_results.extend(results)
            except Exception as e:
                logger.warning(f"Baseline processing error: {e}")

        # Calculate baseline metrics
        if baseline_results:
            latencies = [r.total_processing_time_ms for r in baseline_results]
            confidences = [r.avg_confidence for r in baseline_results]
            detection_counts = [r.detection_count for r in baseline_results]

            self.baseline_metrics = {
                "avg_latency_ms": np.mean(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "avg_confidence": np.mean(confidences),
                "avg_detections": np.mean(detection_counts),
                "established_at": time.time(),
                "frame_count": len(baseline_results),
            }

            logger.info(f"Baseline metrics established: {self.baseline_metrics}")
        else:
            logger.warning("Failed to establish baseline metrics")

    def _generate_synthetic_traffic_frame(self) -> np.ndarray:
        """Generate synthetic traffic scene for baseline testing."""
        h, w = self.config.vision_config.input_resolution

        # Create base road scene
        frame = np.random.randint(40, 80, (h, w, 3), dtype=np.uint8)  # Dark road

        # Add road markings
        frame[h//2-2:h//2+2, :] = 255  # Center line

        # Add some "vehicle-like" rectangles
        num_vehicles = np.random.randint(1, 6)
        for _ in range(num_vehicles):
            # Random vehicle position and size
            x = np.random.randint(50, w-100)
            y = np.random.randint(50, h-80)
            width = np.random.randint(60, 120)
            height = np.random.randint(30, 60)

            # Vehicle color
            color = np.random.randint(100, 255, 3)
            frame[y:y+height, x:x+width] = color

        return frame

    async def process_frame(
        self,
        frame: np.ndarray,
        frame_id: str,
        camera_id: str
    ) -> VisionResult:
        """
        Process frame through integrated vision pipeline.
        
        Includes vision processing, monitoring integration, and optional
        experiment tracking.
        """
        if not self.initialized:
            raise RuntimeError("Integration not initialized")

        start_time = time.time()

        try:
            # Core vision processing
            vision_result = await self.vision_engine.process_frame(
                frame, frame_id, camera_id
            )

            # Integrate with monitoring systems
            await self._integrate_monitoring(vision_result)

            # Integrate with experiment tracking
            await self._integrate_experiment_tracking(vision_result)

            # Update integration metrics
            self.integration_metrics["total_processed"] += 1

            return vision_result

        except Exception as e:
            logger.error(f"Integrated processing failed for {frame_id}: {e}")
            raise

    async def process_batch(
        self,
        frames: list[np.ndarray],
        frame_ids: list[str],
        camera_ids: list[str],
    ) -> list[VisionResult]:
        """Process batch through integrated vision pipeline."""
        if not self.initialized:
            raise RuntimeError("Integration not initialized")

        try:
            # Core vision processing
            vision_results = await self.vision_engine.process_batch(
                frames, frame_ids, camera_ids
            )

            # Integrate each result with monitoring systems
            for vision_result in vision_results:
                await self._integrate_monitoring(vision_result)
                await self._integrate_experiment_tracking(vision_result)

            # Update integration metrics
            self.integration_metrics["total_processed"] += len(vision_results)

            return vision_results

        except Exception as e:
            logger.error(f"Integrated batch processing failed: {e}")
            raise

    async def _integrate_monitoring(self, vision_result: VisionResult) -> None:
        """Integrate vision result with monitoring systems."""
        if not self.dashboard:
            return

        try:
            # Create monitoring sample
            camera_id = vision_result.camera_id

            # Ensure camera monitor exists
            if camera_id not in self.dashboard.model_monitors:
                from .production_monitoring import ModelMonitor
                self.dashboard.model_monitors[camera_id] = ModelMonitor(camera_id)

            monitor = self.dashboard.model_monitors[camera_id]

            # Add detection sample (convert VisionResult to DetectionResult-like format)
            detection_sample = self._convert_to_detection_result(vision_result)
            monitor.metrics.add_sample(detection_sample)

            # Check for drift
            await self._check_for_drift(vision_result, monitor)

        except Exception as e:
            logger.warning(f"Monitoring integration failed: {e}")

    def _convert_to_detection_result(self, vision_result: VisionResult) -> Any:
        """Convert VisionResult to DetectionResult format for monitoring."""
        # Create mock detection result for monitoring compatibility
        from .inference_optimizer import DetectionResult

        # Extract basic detection data
        if vision_result.detections:
            boxes = np.array([
                [d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]]
                for d in vision_result.detections
            ])
            scores = np.array([d["confidence"] for d in vision_result.detections])
            classes = np.array([
                # Map vehicle types back to COCO classes
                {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7, "bicycle": 1}.get(
                    d["vehicle_type"], 2
                )
                for d in vision_result.detections
            ])
            class_names = [d["vehicle_type"] for d in vision_result.detections]
        else:
            boxes = np.array([]).reshape(0, 4)
            scores = np.array([])
            classes = np.array([])
            class_names = []

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            classes=classes,
            class_names=class_names,
            frame_id=vision_result.frame_id,
            camera_id=vision_result.camera_id,
            timestamp=vision_result.timestamp,
            inference_time_ms=vision_result.inference_time_ms,
            preprocessing_time_ms=vision_result.preprocessing_time_ms,
            postprocessing_time_ms=vision_result.postprocessing_time_ms,
            total_time_ms=vision_result.total_processing_time_ms,
            detection_count=vision_result.detection_count,
            avg_confidence=vision_result.avg_confidence,
            gpu_memory_used_mb=vision_result.gpu_memory_used_mb,
        )

    async def _check_for_drift(self, vision_result: VisionResult, monitor: Any) -> None:
        """Check for performance or data drift."""
        if not self.baseline_metrics:
            return

        # Performance drift detection
        current_latency = vision_result.total_processing_time_ms
        baseline_latency = self.baseline_metrics["p95_latency_ms"]

        if current_latency > baseline_latency * 1.5:  # 50% degradation
            alert = Alert(
                alert_id=f"perf_drift_{int(time.time())}",
                severity=AlertSeverity.WARNING,
                drift_type=DriftType.LATENCY_DRIFT,
                message=f"Performance drift detected: {current_latency:.1f}ms vs baseline {baseline_latency:.1f}ms",
                timestamp=time.time(),
                current_value=current_latency,
                expected_value=baseline_latency,
                threshold=baseline_latency * 1.5,
            )
            await self.dashboard.add_alert(alert)
            self.integration_metrics["drift_detections"] += 1

        # Accuracy drift detection
        current_confidence = vision_result.avg_confidence
        baseline_confidence = self.baseline_metrics["avg_confidence"]

        if current_confidence < baseline_confidence * 0.8:  # 20% degradation
            alert = Alert(
                alert_id=f"acc_drift_{int(time.time())}",
                severity=AlertSeverity.CRITICAL,
                drift_type=DriftType.PREDICTION_DRIFT,
                message=f"Accuracy drift detected: {current_confidence:.3f} vs baseline {baseline_confidence:.3f}",
                timestamp=time.time(),
                current_value=current_confidence,
                expected_value=baseline_confidence,
                threshold=baseline_confidence * 0.8,
            )
            await self.dashboard.add_alert(alert)
            self.integration_metrics["drift_detections"] += 1

    async def _integrate_experiment_tracking(self, vision_result: VisionResult) -> None:
        """Integrate with experiment tracking system."""
        if not self.experiment_tracker:
            return

        try:
            # Track performance metrics
            metrics = {
                "latency_ms": vision_result.total_processing_time_ms,
                "detection_count": vision_result.detection_count,
                "avg_confidence": vision_result.avg_confidence,
                "processing_quality": vision_result.processing_quality_score,
            }

            # Log to active experiment if running
            # This would be enhanced to track A/B testing experiments
            active_experiments = getattr(self.experiment_tracker, 'active_runs', {})
            for run_id in active_experiments:
                await self.experiment_tracker.log_metrics(run_id, metrics)

        except Exception as e:
            logger.warning(f"Experiment tracking integration failed: {e}")

    async def _metrics_collection_loop(self) -> None:
        """Background task for metrics collection and reporting."""
        while True:
            try:
                await asyncio.sleep(self.config.metrics_update_interval)

                # Collect vision engine metrics
                vision_metrics = self.vision_engine.get_performance_metrics()
                health_status = self.vision_engine.get_health_status()

                # Update dashboard with metrics
                if self.dashboard:
                    # Convert to dashboard format and update
                    await self._update_dashboard_metrics(vision_metrics, health_status)

                # Log summary metrics
                logger.info(f"Integration metrics: {self.integration_metrics}")

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)  # Wait before retry

    async def _update_dashboard_metrics(
        self,
        vision_metrics: dict[str, Any],
        health_status: dict[str, Any]
    ) -> None:
        """Update production dashboard with vision metrics."""
        try:
            # Update pipeline health score
            if hasattr(self.dashboard, 'update_pipeline_health'):
                await self.dashboard.update_pipeline_health(health_status["health_score"])

            # Update performance metrics
            if "performance" in vision_metrics:
                perf_metrics = vision_metrics["performance"]

                if "latency" in perf_metrics:
                    # Update latency metrics
                    latency_metrics = perf_metrics["latency"]
                    if hasattr(self.dashboard, 'update_latency_metrics'):
                        await self.dashboard.update_latency_metrics(latency_metrics)

                if "throughput" in perf_metrics:
                    # Update throughput metrics
                    throughput_metrics = perf_metrics["throughput"]
                    if hasattr(self.dashboard, 'update_throughput_metrics'):
                        await self.dashboard.update_throughput_metrics(throughput_metrics)

        except Exception as e:
            logger.warning(f"Dashboard metrics update failed: {e}")

    async def _drift_detection_loop(self) -> None:
        """Background task for systematic drift detection."""
        while True:
            try:
                await asyncio.sleep(self.config.drift_check_interval)

                # Perform comprehensive drift analysis
                if self.dashboard and self.baseline_metrics:
                    await self._perform_drift_analysis()

            except Exception as e:
                logger.error(f"Drift detection error: {e}")
                await asyncio.sleep(60)

    async def _perform_drift_analysis(self) -> None:
        """Perform comprehensive drift analysis."""
        try:
            # Collect recent performance data
            recent_metrics = self.vision_engine.get_performance_metrics()

            if not recent_metrics.get("performance"):
                return

            # Analyze performance drift
            perf_data = recent_metrics["performance"]

            # Check latency drift
            if "latency" in perf_data and "p95_ms" in perf_data["latency"]:
                current_p95 = perf_data["latency"]["p95_ms"]
                baseline_p95 = self.baseline_metrics.get("p95_latency_ms", current_p95)

                drift_ratio = current_p95 / baseline_p95 if baseline_p95 > 0 else 1.0

                if drift_ratio > 1.3:  # 30% performance degradation
                    await self._trigger_drift_response("latency", drift_ratio, current_p95, baseline_p95)

            # Check accuracy drift
            if "quality" in perf_data and "avg_score" in perf_data["quality"]:
                current_quality = perf_data["quality"]["avg_score"]
                baseline_quality = self.baseline_metrics.get("avg_confidence", current_quality)

                if current_quality < baseline_quality * 0.85:  # 15% accuracy drop
                    drift_ratio = current_quality / baseline_quality
                    await self._trigger_drift_response("accuracy", drift_ratio, current_quality, baseline_quality)

        except Exception as e:
            logger.error(f"Drift analysis failed: {e}")

    async def _trigger_drift_response(
        self,
        drift_type: str,
        drift_ratio: float,
        current_value: float,
        baseline_value: float,
    ) -> None:
        """Trigger response to detected drift."""
        severity = AlertSeverity.CRITICAL if drift_ratio > 2.0 else AlertSeverity.WARNING

        alert = Alert(
            alert_id=f"drift_{drift_type}_{int(time.time())}",
            severity=severity,
            drift_type=DriftType.LATENCY_DRIFT if drift_type == "latency" else DriftType.PREDICTION_DRIFT,
            message=f"Significant {drift_type} drift detected: {drift_ratio:.2f}x degradation",
            timestamp=time.time(),
            current_value=current_value,
            expected_value=baseline_value,
            threshold=baseline_value * (1.3 if drift_type == "latency" else 0.85),
        )

        if self.dashboard:
            await self.dashboard.add_alert(alert)

        # Trigger automated response if configured
        if severity == AlertSeverity.CRITICAL and self.config.auto_deployment_enabled:
            await self._trigger_model_revalidation()

        self.integration_metrics["drift_detections"] += 1
        logger.warning(f"Drift response triggered: {alert.message}")

    async def _model_validation_loop(self) -> None:
        """Background task for periodic model validation."""
        while True:
            try:
                await asyncio.sleep(self.config.model_validation_interval)

                if self.model_validator and self.current_model_version:
                    await self._validate_current_model()

            except Exception as e:
                logger.error(f"Model validation error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def _validate_current_model(self) -> None:
        """Validate current model performance."""
        try:
            if not self.current_model_version:
                return

            logger.info(f"Validating model {self.current_model_version.model_id}...")

            # Perform model validation
            validation_result = await self.model_validator.validate_model(
                self.current_model_version
            )

            if validation_result != ModelValidationResult.PASSED:
                logger.warning(f"Model validation failed: {validation_result}")

                # Trigger model replacement if validation fails
                if self.config.auto_deployment_enabled:
                    await self._trigger_model_replacement()
            else:
                logger.info("Model validation passed")

        except Exception as e:
            logger.error(f"Model validation failed: {e}")

    async def _trigger_model_revalidation(self) -> None:
        """Trigger immediate model revalidation."""
        if self.current_model_version and self.model_validator:
            await self._validate_current_model()

    async def _trigger_model_replacement(self) -> None:
        """Trigger automated model replacement."""
        try:
            if not self.model_registry:
                return

            logger.info("Triggering automated model replacement...")

            # Find best alternative model
            recent_models = await self.model_registry.list_models(limit=5)

            for model_info in recent_models:
                if model_info["model_id"] == self.current_model_version.model_id:
                    continue  # Skip current model

                candidate_version = await self.model_registry.get_model_by_version(
                    model_info["version"]
                )

                if candidate_version:
                    # Validate candidate
                    validation_result = await self.model_validator.validate_model(
                        candidate_version
                    )

                    if validation_result == ModelValidationResult.PASSED:
                        # Deploy candidate model
                        await self._deploy_model(candidate_version)
                        break

        except Exception as e:
            logger.error(f"Model replacement failed: {e}")

    async def _deploy_model(self, model_version: ModelVersion) -> None:
        """Deploy new model version."""
        try:
            logger.info(f"Deploying model {model_version.model_id}...")

            # Reinitialize vision engine with new model
            await self.vision_engine.cleanup()
            await self.vision_engine.initialize(model_version.model_path)

            # Update current model tracking
            self.current_model_version = model_version

            # Re-establish baseline with new model
            await self._establish_baseline_metrics()

            # Update registry
            if self.model_registry:
                await self.model_registry.set_production_model(model_version.model_id)

            self.integration_metrics["model_deployments"] += 1
            logger.info(f"Successfully deployed model {model_version.model_id}")

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise

    async def _registry_sync_loop(self) -> None:
        """Background task for model registry synchronization."""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                if self.model_registry:
                    await self._sync_with_registry()

            except Exception as e:
                logger.error(f"Registry sync error: {e}")
                await asyncio.sleep(60)

    async def _sync_with_registry(self) -> None:
        """Synchronize with model registry."""
        try:
            # Check for new production model
            production_model = await self.model_registry.get_production_model()

            if (production_model and
                (not self.current_model_version or
                 production_model.model_id != self.current_model_version.model_id)):

                logger.info(f"New production model detected: {production_model.model_id}")

                # Validate and deploy new model
                if self.model_validator:
                    validation_result = await self.model_validator.validate_model(production_model)

                    if validation_result == ModelValidationResult.PASSED:
                        await self._deploy_model(production_model)
                    else:
                        logger.warning(f"New production model failed validation: {validation_result}")
                else:
                    # Deploy without validation if validator not available
                    await self._deploy_model(production_model)

                self.integration_metrics["registry_updates"] += 1

        except Exception as e:
            logger.warning(f"Registry synchronization failed: {e}")

    def get_integration_metrics(self) -> dict[str, Any]:
        """Get comprehensive integration metrics."""
        vision_metrics = self.vision_engine.get_performance_metrics()
        health_status = self.vision_engine.get_health_status()

        uptime = time.time() - self.integration_metrics["start_time"]

        return {
            "integration": {
                "initialized": self.initialized,
                "uptime_seconds": uptime,
                "total_processed": self.integration_metrics["total_processed"],
                "registry_updates": self.integration_metrics["registry_updates"],
                "drift_detections": self.integration_metrics["drift_detections"],
                "model_deployments": self.integration_metrics["model_deployments"],
                "current_model": (
                    self.current_model_version.model_id
                    if self.current_model_version else None
                ),
                "baseline_established": bool(self.baseline_metrics),
                "monitoring_tasks": len(self.monitoring_tasks),
            },
            "vision_engine": vision_metrics,
            "health_status": health_status,
            "baseline_metrics": self.baseline_metrics,
        }

    async def cleanup(self) -> None:
        """Clean up integration resources."""
        logger.info("Cleaning up Vision Pipeline Integration...")

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Clean up vision engine
        await self.vision_engine.cleanup()

        self.initialized = False
        logger.info("Vision Pipeline Integration cleanup completed")


# Factory Functions

async def create_integrated_vision_pipeline(
    deployment_scenario: str = "production",
    ml_pipeline: ProductionMLPipeline | None = None,
    enable_full_integration: bool = True,
) -> VisionPipelineIntegration:
    """
    Create fully integrated vision pipeline with optimal configuration.
    
    Args:
        deployment_scenario: Target deployment scenario
        ml_pipeline: Existing ML pipeline to integrate with
        enable_full_integration: Enable all integration features
    
    Returns:
        Configured and initialized VisionPipelineIntegration
    """

    # Create optimal vision configuration
    vision_config = create_optimal_config(deployment_scenario)

    # Create integration configuration
    integration_config = IntegrationConfig(
        vision_config=vision_config,
        enable_model_registry=enable_full_integration and ml_pipeline is not None,
        enable_federated_learning=False,  # Disable by default for stability
        enable_ab_testing=enable_full_integration and ml_pipeline is not None,
        enable_drift_detection=enable_full_integration,
        auto_deployment_enabled=deployment_scenario == "production" and enable_full_integration,
    )

    # Create integration
    integration = VisionPipelineIntegration(integration_config, ml_pipeline)

    # Initialize
    await integration.initialize()

    logger.info(f"Created integrated vision pipeline for {deployment_scenario} deployment")
    return integration


async def benchmark_integrated_pipeline(
    deployment_scenario: str = "production",
    num_frames: int = 100,
    ml_pipeline: ProductionMLPipeline | None = None,
) -> dict[str, Any]:
    """
    Comprehensive benchmark of integrated vision pipeline.
    
    Includes vision processing, ML integration, monitoring,
    and deployment coordination performance.
    """

    logger.info("Starting integrated pipeline benchmark...")

    # Create integrated pipeline
    integration = await create_integrated_vision_pipeline(
        deployment_scenario=deployment_scenario,
        ml_pipeline=ml_pipeline,
        enable_full_integration=True,
    )

    # Generate test data
    test_frames = []
    frame_ids = []
    camera_ids = []

    for i in range(num_frames):
        frame = integration._generate_synthetic_traffic_frame()
        test_frames.append(frame)
        frame_ids.append(f"benchmark_{i}")
        camera_ids.append(f"camera_{i % 4}")

    # Warmup
    warmup_count = min(10, num_frames)
    for i in range(warmup_count):
        await integration.process_frame(test_frames[i], frame_ids[i], camera_ids[i])

    # Benchmark processing
    benchmark_start = time.time()
    successful_results = []
    failed_count = 0

    for i in range(warmup_count, num_frames):
        try:
            result = await integration.process_frame(test_frames[i], frame_ids[i], camera_ids[i])
            successful_results.append(result)
        except Exception as e:
            failed_count += 1
            logger.warning(f"Benchmark frame {i} failed: {e}")

    benchmark_duration = time.time() - benchmark_start

    # Collect comprehensive metrics
    integration_metrics = integration.get_integration_metrics()

    # Calculate benchmark results
    if successful_results:
        latencies = [r.total_processing_time_ms for r in successful_results]
        confidences = [r.avg_confidence for r in successful_results]
        detection_counts = [r.detection_count for r in successful_results]

        benchmark_results = {
            "configuration": {
                "deployment_scenario": deployment_scenario,
                "frames_processed": len(successful_results),
                "failed_frames": failed_count,
                "success_rate": len(successful_results) / (len(successful_results) + failed_count),
            },
            "performance": {
                "total_duration_s": benchmark_duration,
                "avg_throughput_fps": len(successful_results) / benchmark_duration,
                "avg_latency_ms": np.mean(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "avg_confidence": np.mean(confidences),
                "avg_detections": np.mean(detection_counts),
            },
            "integration_metrics": integration_metrics,
            "monitoring": {
                "drift_detections": integration_metrics["integration"]["drift_detections"],
                "registry_updates": integration_metrics["integration"]["registry_updates"],
                "baseline_established": integration_metrics["integration"]["baseline_established"],
                "health_score": integration_metrics["health_status"]["health_score"],
            },
        }
    else:
        benchmark_results = {
            "configuration": {"deployment_scenario": deployment_scenario},
            "performance": {"error": "No successful results"},
            "integration_metrics": integration_metrics,
        }

    # Cleanup
    await integration.cleanup()

    logger.info("Integrated pipeline benchmark completed")
    return benchmark_results
