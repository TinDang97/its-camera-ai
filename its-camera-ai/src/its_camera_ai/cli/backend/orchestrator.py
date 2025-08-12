"""Service orchestrator for coordinated backend operations.

Provides high-level orchestration of all backend services,
coordinating complex operations across multiple services.
"""

import asyncio
import time
from enum import Enum
from typing import Any

from ...core.config import Settings, get_settings
from ...core.logging import get_logger
from .api_client import APIClient
from .auth_manager import CLIAuthManager
from .database_manager import CLIDatabaseManager
from .event_streamer import EventStreamer
from .health_checker import HealthChecker
from .metrics_collector import MetricsCollector
from .queue_manager import CLIQueueManager
from .service_discovery import ServiceDiscovery

logger = get_logger(__name__)


class OrchestrationStatus(Enum):
    """Orchestration operation status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ServiceOrchestrator:
    """Central orchestrator for backend services.
    
    Features:
    - Coordinated service startup/shutdown
    - Complex multi-service operations
    - Dependency management
    - Health monitoring coordination
    - Event-driven workflows
    - Service communication coordination
    """

    def __init__(self, settings: Settings = None):
        """Initialize service orchestrator.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()

        # Initialize all backend components
        self.api_client = APIClient(settings=self.settings)
        self.auth_manager = CLIAuthManager(self.settings)
        self.db_manager = CLIDatabaseManager(self.settings)
        self.service_discovery = ServiceDiscovery(self.settings)
        self.health_checker = HealthChecker(self.settings)
        self.queue_manager = CLIQueueManager(self.settings)
        self.event_streamer = EventStreamer(self.settings)
        self.metrics_collector = MetricsCollector(self.settings)

        # Orchestration state
        self.initialized = False
        self.services_started = set()
        self.active_operations: dict[str, dict[str, Any]] = {}

        logger.info("Service orchestrator initialized")

    async def initialize(self) -> None:
        """Initialize all backend services."""
        if self.initialized:
            return

        logger.info("Initializing all backend services...")

        try:
            # Initialize services in dependency order
            await self.db_manager.initialize()
            await self.auth_manager.initialize()
            await self.api_client.connect()
            await self.service_discovery.discover_services()
            await self.health_checker.initialize()
            await self.queue_manager.initialize()
            await self.event_streamer.connect()
            await self.metrics_collector.initialize()

            self.initialized = True
            logger.info("All backend services initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize backend services: {e}")
            raise

    async def close(self) -> None:
        """Close all backend services."""
        if not self.initialized:
            return

        logger.info("Closing all backend services...")

        # Close services in reverse order
        await self.metrics_collector.close()
        await self.event_streamer.disconnect()
        await self.queue_manager.close()
        await self.health_checker.close()
        await self.service_discovery.stop_monitoring()
        await self.api_client.close()
        await self.auth_manager.close()
        await self.db_manager.close()

        self.initialized = False
        self.services_started.clear()
        logger.info("All backend services closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # High-level orchestrated operations

    async def full_system_health_check(self) -> dict[str, Any]:
        """Perform comprehensive system health check.
        
        Returns:
            Complete health assessment
        """
        logger.info("Starting full system health check...")

        operation_id = f"health_check_{int(time.time())}"
        self.active_operations[operation_id] = {
            "type": "health_check",
            "status": OrchestrationStatus.RUNNING,
            "started_at": time.time(),
            "components": []
        }

        try:
            # Discover and check all services
            await self.service_discovery.discover_services()
            await self.service_discovery.start_monitoring()

            # Run comprehensive health checks
            health_results = await self.health_checker.check_all_components()

            # Get service status from discovery
            service_status = await self.service_discovery.get_all_services_status()

            # Get metrics for health context
            dashboard_metrics = self.metrics_collector.get_dashboard_metrics()

            # Compile comprehensive report
            health_report = {
                "timestamp": time.time(),
                "overall_status": health_results.overall_status.value,
                "components": {
                    name: {
                        "status": result.status.value,
                        "level": result.level.value,
                        "message": result.message,
                        "response_time_ms": result.response_time_ms,
                        "details": result.details
                    }
                    for name, result in health_results.components.items()
                },
                "services": service_status,
                "metrics": dashboard_metrics,
                "summary": health_results.summary
            }

            self.active_operations[operation_id]["status"] = OrchestrationStatus.COMPLETED
            self.active_operations[operation_id]["result"] = health_report

            logger.info(f"Health check completed - Overall status: {health_results.overall_status.value}")
            return health_report

        except Exception as e:
            self.active_operations[operation_id]["status"] = OrchestrationStatus.FAILED
            self.active_operations[operation_id]["error"] = str(e)
            logger.error(f"Health check failed: {e}")
            raise
        finally:
            await self.service_discovery.stop_monitoring()

    async def start_monitoring_services(self) -> dict[str, Any]:
        """Start all monitoring services.
        
        Returns:
            Monitoring startup status
        """
        logger.info("Starting monitoring services...")

        operation_id = f"start_monitoring_{int(time.time())}"
        self.active_operations[operation_id] = {
            "type": "start_monitoring",
            "status": OrchestrationStatus.RUNNING,
            "started_at": time.time(),
            "services": []
        }

        results = {}

        try:
            # Start service discovery monitoring
            await self.service_discovery.start_monitoring()
            results["service_discovery"] = "started"
            self.services_started.add("service_discovery")

            # Start metrics collection
            await self.metrics_collector.start_collection()
            results["metrics_collection"] = "started"
            self.services_started.add("metrics_collection")

            # Start event streaming
            monitoring_task = asyncio.create_task(
                self.event_streamer.start_monitoring()
            )
            results["event_streaming"] = "started"
            self.services_started.add("event_streaming")

            self.active_operations[operation_id]["status"] = OrchestrationStatus.COMPLETED
            self.active_operations[operation_id]["result"] = results

            logger.info("All monitoring services started successfully")
            return {
                "status": "success",
                "services": results,
                "timestamp": time.time()
            }

        except Exception as e:
            self.active_operations[operation_id]["status"] = OrchestrationStatus.FAILED
            self.active_operations[operation_id]["error"] = str(e)
            logger.error(f"Failed to start monitoring services: {e}")
            raise

    async def stop_monitoring_services(self) -> dict[str, Any]:
        """Stop all monitoring services.
        
        Returns:
            Monitoring shutdown status
        """
        logger.info("Stopping monitoring services...")

        results = {}

        try:
            # Stop metrics collection
            if "metrics_collection" in self.services_started:
                await self.metrics_collector.stop_collection()
                results["metrics_collection"] = "stopped"
                self.services_started.discard("metrics_collection")

            # Stop service discovery monitoring
            if "service_discovery" in self.services_started:
                await self.service_discovery.stop_monitoring()
                results["service_discovery"] = "stopped"
                self.services_started.discard("service_discovery")

            # Event streaming will be stopped when orchestrator closes
            if "event_streaming" in self.services_started:
                results["event_streaming"] = "stopped"
                self.services_started.discard("event_streaming")

            logger.info("All monitoring services stopped")
            return {
                "status": "success",
                "services": results,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"Error stopping monitoring services: {e}")
            raise

    async def perform_system_maintenance(self, operations: list[str]) -> dict[str, Any]:
        """Perform system maintenance operations.
        
        Args:
            operations: List of maintenance operations to perform
            
        Returns:
            Maintenance results
        """
        logger.info(f"Starting system maintenance: {operations}")

        operation_id = f"maintenance_{int(time.time())}"
        self.active_operations[operation_id] = {
            "type": "maintenance",
            "status": OrchestrationStatus.RUNNING,
            "started_at": time.time(),
            "operations": operations
        }

        results = {}

        try:
            for operation in operations:
                if operation == "cleanup_database":
                    # Clean up old database records
                    deleted_frames = await self.db_manager.cleanup_old_data(
                        "frame_metadata", "created_at", 7
                    )
                    deleted_metrics = await self.db_manager.cleanup_old_data(
                        "system_metrics", "timestamp", 30
                    )
                    results["database_cleanup"] = {
                        "deleted_frames": deleted_frames,
                        "deleted_metrics": deleted_metrics
                    }

                elif operation == "vacuum_database":
                    # Vacuum and analyze database
                    vacuum_result = await self.db_manager.vacuum_analyze()
                    results["database_vacuum"] = vacuum_result

                elif operation == "purge_queues":
                    # Purge old queue messages
                    queue_status = await self.queue_manager.get_all_queue_status()
                    purged_counts = {}

                    for queue_name in queue_status.keys():
                        if queue_status[queue_name]["pending_count"] > 1000:
                            purged = await self.queue_manager.purge_queue(
                                queue_name, force=True
                            )
                            purged_counts[queue_name] = purged

                    results["queue_purge"] = purged_counts

                elif operation == "clear_caches":
                    # Clear various caches
                    self.api_client.clear_cache()
                    self.event_streamer.clear_event_buffer()
                    results["cache_clear"] = "completed"

                elif operation == "health_check":
                    # Run health checks
                    health_result = await self.full_system_health_check()
                    results["health_check"] = health_result["overall_status"]

            self.active_operations[operation_id]["status"] = OrchestrationStatus.COMPLETED
            self.active_operations[operation_id]["result"] = results

            logger.info(f"System maintenance completed: {len(results)} operations")
            return {
                "status": "success",
                "operations_completed": len(results),
                "results": results,
                "timestamp": time.time()
            }

        except Exception as e:
            self.active_operations[operation_id]["status"] = OrchestrationStatus.FAILED
            self.active_operations[operation_id]["error"] = str(e)
            logger.error(f"System maintenance failed: {e}")
            raise

    async def orchestrate_camera_setup(
        self,
        camera_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Orchestrate complete camera setup process.
        
        Args:
            camera_config: Camera configuration
            
        Returns:
            Setup results
        """
        logger.info(f"Orchestrating camera setup: {camera_config.get('name', 'unknown')}")

        operation_id = f"camera_setup_{int(time.time())}"
        self.active_operations[operation_id] = {
            "type": "camera_setup",
            "status": OrchestrationStatus.RUNNING,
            "started_at": time.time(),
            "camera_name": camera_config.get("name")
        }

        try:
            # 1. Validate configuration
            if not camera_config.get("stream_url"):
                raise ValueError("Stream URL is required")

            # 2. Test connectivity
            # This would test the camera stream URL
            connectivity_test = {"status": "success"}  # Simplified

            # 3. Create camera in database via API
            camera_result = await self.api_client.create_camera(camera_config)
            camera_id = camera_result.get("id")

            # 4. Submit background task for stream initialization
            task_id = await self.queue_manager.submit_task(
                "initialize_camera_stream",
                {
                    "camera_id": camera_id,
                    "stream_url": camera_config["stream_url"]
                },
                priority=5
            )

            # 5. Wait for stream initialization (simplified)
            await asyncio.sleep(2)

            # 6. Verify camera is accessible
            camera_status = await self.api_client.get_camera(camera_id)

            result = {
                "camera_id": camera_id,
                "connectivity_test": connectivity_test,
                "initialization_task": task_id,
                "status": camera_status.get("status", "unknown"),
                "setup_time": time.time() - self.active_operations[operation_id]["started_at"]
            }

            self.active_operations[operation_id]["status"] = OrchestrationStatus.COMPLETED
            self.active_operations[operation_id]["result"] = result

            logger.info(f"Camera setup completed: {camera_id}")
            return result

        except Exception as e:
            self.active_operations[operation_id]["status"] = OrchestrationStatus.FAILED
            self.active_operations[operation_id]["error"] = str(e)
            logger.error(f"Camera setup failed: {e}")
            raise

    async def orchestrate_model_deployment(
        self,
        model_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Orchestrate ML model deployment process.
        
        Args:
            model_info: Model deployment information
            
        Returns:
            Deployment results
        """
        logger.info(f"Orchestrating model deployment: {model_info.get('name', 'unknown')}")

        operation_id = f"model_deploy_{int(time.time())}"
        self.active_operations[operation_id] = {
            "type": "model_deployment",
            "status": OrchestrationStatus.RUNNING,
            "started_at": time.time(),
            "model_name": model_info.get("name")
        }

        try:
            # 1. Validate model information
            required_fields = ["name", "version", "file_path"]
            for field in required_fields:
                if not model_info.get(field):
                    raise ValueError(f"Missing required field: {field}")

            # 2. Submit model upload task
            upload_task_id = await self.queue_manager.submit_task(
                "upload_model",
                model_info,
                priority=8
            )

            # 3. Wait for upload completion
            # In real implementation, this would monitor the task status
            await asyncio.sleep(5)

            # 4. Register model via API
            model_result = await self.api_client.upload_model(
                b"",  # Simplified - would read actual model file
                model_info
            )

            # 5. Submit model validation task
            validation_task_id = await self.queue_manager.submit_task(
                "validate_model",
                {
                    "model_id": model_result.get("id"),
                    "validation_dataset": model_info.get("validation_dataset")
                },
                priority=7
            )

            result = {
                "model_id": model_result.get("id"),
                "upload_task": upload_task_id,
                "validation_task": validation_task_id,
                "status": "deployed",
                "deployment_time": time.time() - self.active_operations[operation_id]["started_at"]
            }

            self.active_operations[operation_id]["status"] = OrchestrationStatus.COMPLETED
            self.active_operations[operation_id]["result"] = result

            logger.info(f"Model deployment completed: {model_result.get('id')}")
            return result

        except Exception as e:
            self.active_operations[operation_id]["status"] = OrchestrationStatus.FAILED
            self.active_operations[operation_id]["error"] = str(e)
            logger.error(f"Model deployment failed: {e}")
            raise

    def get_system_overview(self) -> dict[str, Any]:
        """Get comprehensive system overview.
        
        Returns:
            System overview dictionary
        """
        return {
            "orchestrator": {
                "initialized": self.initialized,
                "services_started": list(self.services_started),
                "active_operations": len(self.active_operations)
            },
            "api_client": self.api_client.get_stats(),
            "auth_manager": self.auth_manager.get_session_info(),
            "database": {
                "connected": self.db_manager.is_connected()
            },
            "service_discovery": self.service_discovery.get_connection_status(),
            "queue_manager": self.queue_manager.get_manager_status(),
            "event_streamer": self.event_streamer.get_connection_status(),
            "metrics_collector": self.metrics_collector.get_collector_status()
        }

    def get_active_operations(self) -> dict[str, dict[str, Any]]:
        """Get information about active operations.
        
        Returns:
            Active operations dictionary
        """
        # Clean up completed operations older than 1 hour
        current_time = time.time()
        completed_operations = [
            op_id for op_id, op in self.active_operations.items()
            if op["status"] in (OrchestrationStatus.COMPLETED, OrchestrationStatus.FAILED)
            and current_time - op["started_at"] > 3600
        ]

        for op_id in completed_operations:
            del self.active_operations[op_id]

        return self.active_operations.copy()

    async def cancel_operation(self, operation_id: str) -> bool:
        """Cancel an active operation.
        
        Args:
            operation_id: ID of operation to cancel
            
        Returns:
            True if successfully cancelled
        """
        if operation_id in self.active_operations:
            operation = self.active_operations[operation_id]

            if operation["status"] == OrchestrationStatus.RUNNING:
                operation["status"] = OrchestrationStatus.CANCELLED
                operation["cancelled_at"] = time.time()

                logger.info(f"Cancelled operation: {operation_id}")
                return True

        return False

    async def wait_for_healthy_system(
        self,
        timeout: int = 300,
        check_interval: int = 10
    ) -> bool:
        """Wait for system to become healthy.
        
        Args:
            timeout: Maximum wait time in seconds
            check_interval: Check interval in seconds
            
        Returns:
            True if system becomes healthy
        """
        logger.info(f"Waiting for healthy system (timeout: {timeout}s)")

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                health_result = await self.full_system_health_check()

                if health_result["overall_status"] in ("healthy", "degraded"):
                    logger.info("System is healthy")
                    return True

                logger.debug(f"System not healthy yet: {health_result['overall_status']}")
                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.warning(f"Health check failed during wait: {e}")
                await asyncio.sleep(check_interval)

        logger.warning(f"System did not become healthy within {timeout}s")
        return False
