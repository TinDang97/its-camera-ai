#!/usr/bin/env python3
"""
Production Deployment Script for ITS Camera AI Traffic Monitoring System.

This script deploys the complete ML pipeline for traffic monitoring with:
- Real-time camera stream processing (1000+ cameras at 30 FPS)
- YOLO11 inference with sub-100ms latency
- Continuous learning and model updates
- Federated learning across edge nodes
- Production monitoring and alerting
- Auto-scaling and fault tolerance

Usage:
    python deploy_production.py --mode production --config production_config.json
    python deploy_production.py --mode staging --cameras 100
    python deploy_production.py --mode development --test-mode
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("production_pipeline.log"),
    ],
)

logger = logging.getLogger(__name__)

# Import pipeline components
try:
    from its_camera_ai.production_pipeline import (
        DeploymentMode,
        deploy_its_camera_ai_pipeline,
    )
except ImportError as e:
    logger.error(f"Failed to import pipeline components: {e}")
    logger.error("Please ensure you're running from the project root directory")
    sys.exit(1)


class ProductionDeployment:
    """Manages production deployment lifecycle."""

    def __init__(self, args):
        self.args = args
        self.orchestrator = None
        self.shutdown_event = asyncio.Event()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, _frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()

    async def deploy(self):
        """Deploy the production pipeline."""

        try:
            logger.info("Starting ITS Camera AI production deployment")

            # Load configuration
            self._load_configuration()

            # Deploy pipeline
            self.orchestrator = await deploy_its_camera_ai_pipeline(
                config_path=self.args.config if hasattr(self.args, "config") else None,
                deployment_mode=DeploymentMode(self.args.mode),
            )

            # Add test cameras if specified
            if self.args.test_mode or self.args.mode == "development":
                await self._setup_test_cameras()

            # Start monitoring
            await self._run_production_monitoring()

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
        finally:
            if self.orchestrator:
                await self.orchestrator.stop()

    def _load_configuration(self) -> dict:
        """Load deployment configuration."""

        if hasattr(self.args, "config") and self.args.config:
            config_path = Path(self.args.config)
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                return config
            else:
                logger.warning(f"Config file {config_path} not found, using defaults")

        # Default configurations based on deployment mode
        if self.args.mode == "production":
            return self._get_production_config()
        elif self.args.mode == "staging":
            return self._get_staging_config()
        else:  # development
            return self._get_development_config()

    def _get_production_config(self) -> dict:
        """Get production configuration."""
        return {
            "deployment_mode": "production",
            "max_concurrent_cameras": 1000,
            "target_latency_ms": 95,
            "target_throughput_fps": 30000,
            "target_accuracy": 0.95,
            "continuous_training_enabled": True,
            "federated_learning_enabled": True,
            "auto_scaling_enabled": True,
            "monitoring_enabled": True,
        }

    def _get_staging_config(self) -> dict:
        """Get staging configuration."""
        return {
            "deployment_mode": "staging",
            "max_concurrent_cameras": 100,
            "target_latency_ms": 100,
            "target_throughput_fps": 3000,
            "target_accuracy": 0.90,
            "continuous_training_enabled": True,
            "federated_learning_enabled": False,
            "auto_scaling_enabled": True,
            "monitoring_enabled": True,
        }

    def _get_development_config(self) -> dict:
        """Get development configuration."""
        return {
            "deployment_mode": "development",
            "max_concurrent_cameras": 10,
            "target_latency_ms": 150,
            "target_throughput_fps": 300,
            "target_accuracy": 0.85,
            "continuous_training_enabled": False,
            "federated_learning_enabled": False,
            "auto_scaling_enabled": False,
            "monitoring_enabled": True,
        }

    async def _setup_test_cameras(self):
        """Setup test cameras for development/testing."""

        num_cameras = getattr(self.args, "cameras", 5)
        logger.info(f"Setting up {num_cameras} test cameras")

        for i in range(num_cameras):
            camera_config = {
                "camera_id": f"test_cam_{i:03d}",
                "location": f"Test Intersection {i}",
                "coordinates": [37.7749 + i * 0.001, -122.4194 + i * 0.001],
                "resolution": [1920, 1080],
                "fps": 30,
                "quality_threshold": 0.7,
            }

            success = await self.orchestrator.add_camera_stream(camera_config)
            if success:
                logger.info(f"Added test camera: {camera_config['camera_id']}")
            else:
                logger.warning(
                    f"Failed to add test camera: {camera_config['camera_id']}"
                )

    async def _run_production_monitoring(self):
        """Run production monitoring and management."""

        logger.info("Starting production monitoring loop")

        # Monitoring loop
        status_interval = 60  # Log status every minute
        last_status_log = 0

        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()

                # Get pipeline status
                status = self.orchestrator.get_pipeline_status()

                # Log status periodically
                if current_time - last_status_log > status_interval:
                    self._log_pipeline_status(status)
                    last_status_log = current_time

                # Check for manual interventions
                if self.args.test_mode:
                    await self._handle_test_commands()

                # Wait before next check
                try:
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=10.0)
                    break  # Shutdown requested
                except TimeoutError:
                    continue  # Continue monitoring

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)

        logger.info("Production monitoring stopped")

    def _log_pipeline_status(self, status: dict):
        """Log comprehensive pipeline status."""

        logger.info(
            f"Pipeline Status: {status['status']}, "
            f"Uptime: {status['uptime_hours']:.1f}h, "
            f"Cameras: {status['active_cameras']}, "
            f"Health: {status['performance_metrics']['system_health_score']:.1f}%"
        )

        # Log component status
        components = status["components"]
        logger.debug(
            f"Components - Stream: {components['stream_processor']}, "
            f"ML: {components['ml_pipeline']}, "
            f"Monitor: {components['monitoring']}, "
            f"Federated: {components['federated_learning']}"
        )

    async def _handle_test_commands(self):
        """Handle test mode commands (interactive testing)."""

        # In a real implementation, this could handle interactive commands
        # For now, just simulate some test scenarios

        # Simulate triggering retraining every 10 minutes in test mode
        if hasattr(self, "_last_retrain_test"):
            if time.time() - self._last_retrain_test > 600:  # 10 minutes
                logger.info("Test mode: Triggering model retraining")
                await self.orchestrator.trigger_model_retraining("Test mode trigger")
                self._last_retrain_test = time.time()
        else:
            self._last_retrain_test = time.time()


def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Deploy ITS Camera AI Production Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Production deployment:
    python deploy_production.py --mode production --config production_config.json

  Staging deployment:
    python deploy_production.py --mode staging --cameras 50

  Development testing:
    python deploy_production.py --mode development --test-mode --cameras 5
""",
    )

    parser.add_argument(
        "--mode",
        choices=["production", "staging", "development"],
        default="development",
        help="Deployment mode (default: development)",
    )

    parser.add_argument(
        "--config", type=str, help="Configuration file path (JSON format)"
    )

    parser.add_argument(
        "--cameras",
        type=int,
        default=5,
        help="Number of test cameras to setup (default: 5)",
    )

    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Enable test mode with simulated scenarios",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--no-federated", action="store_true", help="Disable federated learning"
    )

    parser.add_argument(
        "--no-monitoring", action="store_true", help="Disable production monitoring"
    )

    return parser.parse_args()


async def main():
    """Main deployment function."""

    args = parse_arguments()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Print deployment info
    print(
        f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  ITS Camera AI Production Deployment                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Mode: {args.mode:<20} │ Cameras: {args.cameras:<10} │ Test: {str(args.test_mode):<10} ║
║ Config: {(args.config or "default"):<18} │ Log Level: {args.log_level:<8} │ Version: 1.0.0  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    )

    # Validate deployment environment
    if args.mode == "production":
        logger.warning(
            "PRODUCTION MODE: Ensure all infrastructure dependencies are available:\n"
            "- Kafka cluster (message queuing)\n"
            "- Redis cluster (caching)\n"
            "- MLflow server (experiment tracking)\n"
            "- Prometheus/Grafana (monitoring)\n"
            "- GPU resources (inference acceleration)"
        )

        # In production, you might want to add additional validation
        # response = input("Proceed with production deployment? [y/N]: ")
        # if response.lower() != 'y':
        #     print("Deployment cancelled")
        #     return

    # Start deployment
    deployment = ProductionDeployment(args)

    try:
        await deployment.deploy()
    except KeyboardInterrupt:
        logger.info("Deployment interrupted by user")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)

    logger.info("Deployment completed successfully")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
    except Exception as e:
        print(f"\nDeployment error: {e}")
        sys.exit(1)
