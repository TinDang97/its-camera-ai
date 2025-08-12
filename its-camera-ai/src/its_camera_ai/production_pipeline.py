"""
Production ML Pipeline Orchestrator for ITS Camera AI Traffic Monitoring System.

This module orchestrates the complete production ML pipeline integrating:
1. Real-time data ingestion and processing
2. Continuous model training and validation
3. Model deployment with A/B testing
4. Production monitoring and drift detection
5. Federated learning across edge nodes
6. MLOps CI/CD integration

Architecture:
- Microservices-based with async communication
- Event-driven pipeline with Kafka integration
- Auto-scaling with Kubernetes support
- Multi-GPU distributed training
- Edge-cloud hybrid deployment
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# Import all pipeline components
from .data.streaming_processor import StreamProcessor, create_stream_processor
from .ml.federated_learning import (
    ClientInfo,
    FederatedCoordinator,
    create_federated_coordinator,
)
from .ml.inference_optimizer import (
    ModelType,
    OptimizationBackend,
)
from .ml.ml_pipeline import ProductionMLPipeline, create_production_pipeline
from .ml.production_monitoring import (
    ProductionDashboard,
    create_production_monitoring_system,
)

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Overall pipeline status."""

    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    SCALING = "scaling"
    UPDATING = "updating"
    DEGRADED = "degraded"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class DeploymentMode(Enum):
    """Deployment modes."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    EDGE = "edge"


@dataclass
class PipelineConfig:
    """Production pipeline configuration."""

    # Deployment settings
    deployment_mode: DeploymentMode = DeploymentMode.PRODUCTION
    environment_name: str = "prod"

    # Scaling configuration
    max_concurrent_cameras: int = 1000
    auto_scaling_enabled: bool = True
    target_cpu_utilization: float = 70.0
    target_gpu_utilization: float = 80.0

    # Performance targets
    target_latency_ms: int = 100
    target_throughput_fps: int = 30000  # Aggregate across all cameras
    target_accuracy: float = 0.95

    # Model configuration
    base_model_type: ModelType = ModelType.SMALL
    optimization_backend: OptimizationBackend = OptimizationBackend.TENSORRT
    enable_quantization: bool = True

    # Training configuration
    continuous_training_enabled: bool = True
    federated_learning_enabled: bool = True
    training_data_retention_days: int = 30

    # Monitoring configuration
    monitoring_enabled: bool = True
    alerting_enabled: bool = True
    drift_detection_enabled: bool = True

    # Infrastructure
    kafka_cluster: str = "kafka-cluster:9092"
    redis_cluster: str = "redis-cluster:6379"
    mlflow_uri: str = "http://mlflow:5000"
    prometheus_gateway: str = "pushgateway:9091"

    # Security
    encryption_enabled: bool = True
    authentication_required: bool = True
    audit_logging_enabled: bool = True


class ProductionOrchestrator:
    """Main production pipeline orchestrator."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.status = PipelineStatus.INITIALIZING

        # Core pipeline components
        self.stream_processor: StreamProcessor | None = None
        self.ml_pipeline: ProductionMLPipeline | None = None
        self.monitoring_dashboard: ProductionDashboard | None = None
        self.federated_coordinator: FederatedCoordinator | None = None

        # Runtime state
        self.start_time: float | None = None
        self.active_cameras: set[str] = set()
        self.performance_metrics = {
            "total_frames_processed": 0,
            "total_predictions_made": 0,
            "avg_latency_ms": 0.0,
            "current_fps": 0.0,
            "system_health_score": 100.0,
        }

        # Health check and scaling
        self.health_check_interval = 30  # seconds
        self.last_health_check = 0.0

        logger.info(
            f"Production orchestrator initialized for {config.deployment_mode.value} environment"
        )

    async def initialize(self):
        """Initialize all pipeline components."""

        logger.info("Initializing production pipeline components...")
        self.status = PipelineStatus.INITIALIZING

        try:
            # Initialize data processing
            await self._initialize_data_pipeline()

            # Initialize ML pipeline
            await self._initialize_ml_pipeline()

            # Initialize monitoring
            await self._initialize_monitoring()

            # Initialize federated learning (if enabled)
            if self.config.federated_learning_enabled:
                await self._initialize_federated_learning()

            logger.info("All pipeline components initialized successfully")

        except Exception as e:
            self.status = PipelineStatus.ERROR
            logger.error(f"Pipeline initialization failed: {e}")
            raise

    async def _initialize_data_pipeline(self):
        """Initialize data processing pipeline."""

        self.stream_processor = await create_stream_processor()
        logger.info("Data processing pipeline initialized")

    async def _initialize_ml_pipeline(self):
        """Initialize ML training and inference pipeline."""

        self.ml_pipeline = await create_production_pipeline()
        logger.info("ML pipeline initialized")

    async def _initialize_monitoring(self):
        """Initialize production monitoring and alerting."""

        models_config = [{"model_id": "yolo11_traffic_main", "model_version": "v1.0.0"}]

        alert_config = {
            "email": {
                "host": "smtp.company.com",
                "port": 587,
                "username": "alerts@company.com",
                "password": "password",
            },
            "slack": {"webhook_url": "https://hooks.slack.com/services/..."},
            "prometheus_enabled": True,
            "prometheus_port": 8080,
        }

        self.monitoring_dashboard = await create_production_monitoring_system(
            models_config, alert_config
        )
        logger.info("Production monitoring initialized")

    async def _initialize_federated_learning(self):
        """Initialize federated learning coordinator."""

        self.federated_coordinator = await create_federated_coordinator()
        logger.info("Federated learning coordinator initialized")

    async def start(self):
        """Start the complete production pipeline."""

        logger.info("Starting production pipeline...")
        self.status = PipelineStatus.STARTING
        self.start_time = time.time()

        try:
            # Start all components in order
            if self.monitoring_dashboard:
                await self.monitoring_dashboard.start()

            if self.ml_pipeline:
                await self.ml_pipeline.start()

            if self.stream_processor:
                await self.stream_processor.start()

            if self.federated_coordinator and self.config.federated_learning_enabled:
                # Register some default edge nodes (in production, these would connect dynamically)
                await self._register_default_edge_nodes()

            # Start monitoring and management tasks
            asyncio.create_task(self._pipeline_health_monitor())
            asyncio.create_task(self._performance_optimizer())
            asyncio.create_task(self._auto_scaler())

            self.status = PipelineStatus.RUNNING
            logger.info("Production pipeline started successfully")

        except Exception as e:
            self.status = PipelineStatus.ERROR
            logger.error(f"Pipeline startup failed: {e}")
            raise

    async def _register_default_edge_nodes(self):
        """Register default edge nodes for federated learning."""

        if not self.federated_coordinator:
            return

        edge_nodes = [
            ClientInfo(
                client_id=f"edge_node_{i}",
                location=f"Traffic Hub {i}",
                camera_ids=[f"cam_{i}_{j}" for j in range(10)],
                data_samples=np.random.randint(1000, 5000),
                data_quality_score=np.random.uniform(0.8, 0.95),
            )
            for i in range(5)
        ]

        for node in edge_nodes:
            await self.federated_coordinator.register_client(node)

        logger.info(f"Registered {len(edge_nodes)} default edge nodes")

    async def _pipeline_health_monitor(self):
        """Monitor overall pipeline health."""

        while self.status in [
            PipelineStatus.RUNNING,
            PipelineStatus.SCALING,
            PipelineStatus.UPDATING,
        ]:
            try:
                current_time = time.time()

                # Collect health metrics from all components
                health_data = await self._collect_health_metrics()

                # Update system health score
                self.performance_metrics["system_health_score"] = health_data[
                    "overall_health"
                ]

                # Check for issues
                await self._check_system_health(health_data)

                # Log health status
                if current_time - self.last_health_check > 300:  # Every 5 minutes
                    self._log_health_summary(health_data)
                    self.last_health_check = current_time

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _collect_health_metrics(self) -> dict[str, Any]:
        """Collect health metrics from all pipeline components."""
        health_data = self._initialize_health_data()

        # Collect metrics from each component
        if self.stream_processor:
            health_data["stream_processor"] = self._get_stream_processor_health()

        if self.ml_pipeline:
            health_data["ml_pipeline"] = await self._get_ml_pipeline_health()

        if self.monitoring_dashboard:
            health_data["monitoring"] = self._get_monitoring_health()

        if self.federated_coordinator:
            health_data["federated_learning"] = self._get_federated_health()

        # Calculate overall health score
        health_data["overall_health"] = self._calculate_overall_health(health_data)
        return health_data

    def _initialize_health_data(self) -> dict[str, Any]:
        """Initialize base health data structure."""
        return {
            "timestamp": time.time(),
            "uptime_hours": (
                (time.time() - self.start_time) / 3600 if self.start_time else 0
            ),
            "overall_health": 100.0,
        }

    def _get_stream_processor_health(self) -> dict[str, Any]:
        """Get stream processor health metrics."""
        stream_stats = self.stream_processor.get_processing_stats()
        error_rate = stream_stats["performance_metrics"]["error_count"] / max(
            1, stream_stats["performance_metrics"]["frames_processed"]
        )
        health_score = (
            90.0
            if stream_stats["kafka_available"] and stream_stats["redis_available"]
            else 50.0
        )

        return {
            "active_streams": stream_stats["active_streams"],
            "throughput_fps": stream_stats["performance_metrics"]["throughput_fps"],
            "error_rate": error_rate,
            "health_score": health_score,
        }

    async def _get_ml_pipeline_health(self) -> dict[str, Any]:
        """Get ML pipeline health metrics."""
        ml_status = await self.ml_pipeline.get_pipeline_status()
        return {
            "status": ml_status["status"],
            "inference_engine_fps": ml_status["components"]["inference_engine"].get(
                "throughput_fps", 0
            ),
            "training_jobs": ml_status["components"]["training_pipeline"][
                "active_jobs"
            ],
            "model_count": ml_status["components"]["model_registry"]["total_models"],
            "health_score": 100.0 if ml_status["status"] == "running" else 50.0,
        }

    def _get_monitoring_health(self) -> dict[str, Any]:
        """Get monitoring dashboard health metrics."""
        dashboard_data = self.monitoring_dashboard.get_dashboard_data()
        metrics = dashboard_data["metrics"]

        return {
            "system_health_score": metrics["system_health_score"],
            "active_alerts": metrics["alerts_active"],
            "critical_alerts": metrics["alerts_critical"],
            "health_score": max(0, 100 - metrics["alerts_critical"] * 20),
        }

    def _get_federated_health(self) -> dict[str, Any]:
        """Get federated learning health metrics."""
        fed_status = self.federated_coordinator.get_training_status()
        return {
            "is_training": fed_status["is_training"],
            "active_clients": fed_status["active_clients"],
            "current_round": fed_status["current_round"],
            "latest_accuracy": fed_status["latest_accuracy"],
            "health_score": 100.0 if fed_status["active_clients"] >= 3 else 60.0,
        }

    def _calculate_overall_health(self, health_data: dict[str, Any]) -> float:
        """Calculate overall health score from component scores."""
        component_scores = [
            data["health_score"]
            for data in health_data.values()
            if isinstance(data, dict) and "health_score" in data
        ]
        return np.mean(component_scores) if component_scores else 100.0

    async def _check_system_health(self, health_data: dict[str, Any]):
        """Check system health and trigger alerts if needed."""

        # Check overall health threshold
        if health_data["overall_health"] < 70:
            if self.status == PipelineStatus.RUNNING:
                self.status = PipelineStatus.DEGRADED
                logger.warning(
                    f"System health degraded: {health_data['overall_health']:.1f}%"
                )
        elif (
            health_data["overall_health"] > 80
            and self.status == PipelineStatus.DEGRADED
        ):
            self.status = PipelineStatus.RUNNING
            logger.info("System health recovered")

        # Check component-specific issues
        if "stream_processor" in health_data:
            stream_health = health_data["stream_processor"]
            if stream_health["error_rate"] > 0.05:  # 5% error rate
                logger.warning(
                    f"High stream processing error rate: {stream_health['error_rate']:.2%}"
                )

        if "ml_pipeline" in health_data:
            ml_health = health_data["ml_pipeline"]
            if (
                ml_health["inference_engine_fps"]
                < self.config.target_throughput_fps * 0.5
            ):
                logger.warning(
                    f"Low inference throughput: {ml_health['inference_engine_fps']} FPS"
                )

    def _log_health_summary(self, health_data: dict[str, Any]):
        """Log comprehensive health summary."""

        logger.info(
            f"Pipeline Health Summary: "
            f"Overall={health_data['overall_health']:.1f}%, "
            f"Uptime={health_data['uptime_hours']:.1f}h, "
            f"Active Cameras={len(self.active_cameras)}, "
            f"Status={self.status.value}"
        )

        for component, data in health_data.items():
            if isinstance(data, dict) and "health_score" in data:
                logger.debug(f"{component}: {data['health_score']:.1f}% health")

    async def _performance_optimizer(self):
        """Continuously optimize pipeline performance."""

        while self.status in [PipelineStatus.RUNNING, PipelineStatus.DEGRADED]:
            try:
                # Collect performance metrics
                perf_data = await self._collect_performance_metrics()

                # Optimize based on current performance
                await self._apply_performance_optimizations(perf_data)

                # Sleep before next optimization cycle
                await asyncio.sleep(300)  # Every 5 minutes

            except Exception as e:
                logger.error(f"Performance optimization error: {e}")
                await asyncio.sleep(300)

    async def _collect_performance_metrics(self) -> dict[str, Any]:
        """Collect detailed performance metrics."""

        metrics = {}

        # Inference performance
        if self.ml_pipeline:
            inference_stats = self.ml_pipeline.inference_engine.get_performance_stats()
            metrics["inference"] = {
                "avg_latency_ms": inference_stats.get("avg_latency_ms", 0),
                "p95_latency_ms": inference_stats.get("p95_latency_ms", 0),
                "throughput_fps": inference_stats.get("throughput_fps", 0),
                "gpu_utilization": inference_stats.get("gpu_utilization", 0),
            }

        # Stream processing performance
        if self.stream_processor:
            stream_stats = self.stream_processor.get_processing_stats()
            metrics["streaming"] = {
                "processing_fps": stream_stats["performance_metrics"]["throughput_fps"],
                "avg_processing_time": stream_stats["performance_metrics"][
                    "avg_processing_time"
                ],
                "buffer_utilization": stream_stats["processed_frames_buffer"] / 10000,
            }

        return metrics

    async def _apply_performance_optimizations(self, perf_data: dict[str, Any]):
        """Apply performance optimizations based on current metrics."""

        # Optimize inference performance
        if "inference" in perf_data:
            inference_metrics = perf_data["inference"]

            # If latency is too high, suggest model optimization
            if (
                inference_metrics["p95_latency_ms"]
                > self.config.target_latency_ms * 1.2
            ):
                logger.info(
                    "High inference latency detected - consider model quantization"
                )

            # If GPU utilization is low, increase batch size
            if inference_metrics["gpu_utilization"] < 50 and self.ml_pipeline:
                logger.info("Low GPU utilization - consider increasing batch size")

    async def _auto_scaler(self):
        """Auto-scale pipeline components based on load."""

        if not self.config.auto_scaling_enabled:
            return

        while self.status in [PipelineStatus.RUNNING, PipelineStatus.DEGRADED]:
            try:
                await self._evaluate_scaling_needs()
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(60)

    async def _evaluate_scaling_needs(self):
        """Evaluate if scaling up or down is needed."""

        # Get current resource utilization
        health_data = await self._collect_health_metrics()

        # Check if scaling is needed
        scale_up_needed = False
        scale_down_possible = False

        # CPU/GPU utilization based scaling
        if "ml_pipeline" in health_data:
            ml_data = health_data["ml_pipeline"]
            inference_fps = ml_data.get("inference_engine_fps", 0)

            if inference_fps > self.config.target_throughput_fps * 0.9:
                scale_up_needed = True
                logger.info(
                    "High inference load detected - scaling consideration needed"
                )
            elif inference_fps < self.config.target_throughput_fps * 0.3:
                scale_down_possible = True
                logger.info("Low inference load detected - scale down possible")

        # Apply scaling decisions
        if scale_up_needed:
            await self._scale_up()
        elif scale_down_possible:
            await self._scale_down()

    async def _scale_up(self):
        """Scale up pipeline components."""

        logger.info("Scaling up pipeline components")
        self.status = PipelineStatus.SCALING

        # In production, this would trigger Kubernetes scaling
        # or add more worker processes

        await asyncio.sleep(5)  # Simulate scaling time
        self.status = PipelineStatus.RUNNING

    async def _scale_down(self):
        """Scale down pipeline components."""

        logger.info("Scaling down pipeline components")
        self.status = PipelineStatus.SCALING

        # In production, this would reduce worker count

        await asyncio.sleep(3)  # Simulate scaling time
        self.status = PipelineStatus.RUNNING

    async def add_camera_stream(self, camera_config: dict[str, Any]) -> bool:
        """Add new camera stream to pipeline."""

        if not self.stream_processor:
            return False

        success = await self.stream_processor.register_stream(camera_config)

        if success:
            self.active_cameras.add(camera_config["camera_id"])
            logger.info(f"Added camera stream: {camera_config['camera_id']}")

        return success

    async def remove_camera_stream(self, camera_id: str) -> bool:
        """Remove camera stream from pipeline."""

        if camera_id in self.active_cameras:
            self.active_cameras.remove(camera_id)
            logger.info(f"Removed camera stream: {camera_id}")
            return True

        return False

    async def trigger_model_retraining(self, reason: str = "Manual trigger") -> bool:
        """Trigger model retraining."""

        if not self.ml_pipeline:
            return False

        try:
            await self.ml_pipeline.trigger_retraining(reason)
            return True
        except Exception as e:
            logger.error(f"Model retraining trigger failed: {e}")
            return False

    async def start_federated_round(self) -> bool:
        """Start new federated learning round."""

        if not self.federated_coordinator or not self.config.federated_learning_enabled:
            return False

        try:
            return await self.federated_coordinator.start_federated_training()
        except Exception as e:
            logger.error(f"Federated training start failed: {e}")
            return False

    def get_pipeline_status(self) -> dict[str, Any]:
        """Get comprehensive pipeline status."""

        status_data = {
            "status": self.status.value,
            "deployment_mode": self.config.deployment_mode.value,
            "uptime_hours": (
                (time.time() - self.start_time) / 3600 if self.start_time else 0
            ),
            "active_cameras": len(self.active_cameras),
            "performance_metrics": self.performance_metrics.copy(),
            "components": {
                "stream_processor": self.stream_processor is not None,
                "ml_pipeline": self.ml_pipeline is not None,
                "monitoring": self.monitoring_dashboard is not None,
                "federated_learning": self.federated_coordinator is not None,
            },
        }

        return status_data

    async def stop(self):
        """Stop the complete production pipeline."""

        logger.info("Stopping production pipeline...")
        self.status = PipelineStatus.STOPPING

        try:
            # Stop components in reverse order
            if self.federated_coordinator:
                await self.federated_coordinator.stop_training()

            if self.stream_processor:
                await self.stream_processor.stop()

            if self.ml_pipeline:
                await self.ml_pipeline.stop()

            if self.monitoring_dashboard:
                await self.monitoring_dashboard.stop()

            self.status = PipelineStatus.STOPPED
            logger.info("Production pipeline stopped")

        except Exception as e:
            self.status = PipelineStatus.ERROR
            logger.error(f"Pipeline shutdown error: {e}")


# Factory functions


async def create_production_orchestrator(
    config_path: Path | str = None,
    deployment_mode: DeploymentMode = DeploymentMode.PRODUCTION,
) -> ProductionOrchestrator:
    """Create and initialize production pipeline orchestrator."""

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config_data = json.load(f)

        config = PipelineConfig(
            deployment_mode=DeploymentMode(
                config_data.get("deployment_mode", "production")
            ),
            **{k: v for k, v in config_data.items() if k != "deployment_mode"},
        )
    else:
        # Default production configuration
        config = PipelineConfig(
            deployment_mode=deployment_mode,
            max_concurrent_cameras=1000,
            target_latency_ms=100,
            target_throughput_fps=30000,
            target_accuracy=0.95,
            continuous_training_enabled=True,
            federated_learning_enabled=True,
            monitoring_enabled=True,
            auto_scaling_enabled=True,
        )

    orchestrator = ProductionOrchestrator(config)
    await orchestrator.initialize()

    logger.info(
        f"Production orchestrator created for {deployment_mode.value} deployment"
    )
    return orchestrator


async def deploy_its_camera_ai_pipeline(
    config_path: Path | str = None,
    deployment_mode: DeploymentMode = DeploymentMode.PRODUCTION,
) -> ProductionOrchestrator:
    """Deploy complete ITS Camera AI production pipeline."""

    logger.info(f"Deploying ITS Camera AI pipeline in {deployment_mode.value} mode")

    # Create orchestrator
    orchestrator = await create_production_orchestrator(config_path, deployment_mode)

    # Start pipeline
    await orchestrator.start()

    logger.info("ITS Camera AI pipeline deployment completed")
    return orchestrator


# Example usage and testing
if __name__ == "__main__":

    async def main():
        # Deploy production pipeline
        orchestrator = await deploy_its_camera_ai_pipeline(
            deployment_mode=DeploymentMode.DEVELOPMENT  # Use development for testing
        )

        # Add some test cameras
        for i in range(5):
            camera_config = {
                "camera_id": f"test_cam_{i}",
                "location": f"Test Location {i}",
                "coordinates": [37.7749 + i * 0.01, -122.4194 + i * 0.01],
                "resolution": [1920, 1080],
                "fps": 30,
            }
            await orchestrator.add_camera_stream(camera_config)

        # Run for a while
        try:
            for _ in range(10):
                status = orchestrator.get_pipeline_status()
                print(
                    f"Pipeline Status: {status['status']}, Cameras: {status['active_cameras']}"
                )
                await asyncio.sleep(30)

        except KeyboardInterrupt:
            print("Shutting down pipeline...")

        # Stop pipeline
        await orchestrator.stop()
        print("Pipeline stopped")

    asyncio.run(main())
