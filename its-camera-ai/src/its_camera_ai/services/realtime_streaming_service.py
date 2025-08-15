"""Real-time Event Streaming Service for ITS Camera AI.

This service coordinates the entire real-time event streaming pipeline,
integrating Kafka event production and consumption with SSE broadcasting
for seamless real-time camera analytics streaming to web clients.

Features:
- Centralized coordination of Kafka producer/consumer and SSE broadcaster
- Automatic service lifecycle management and health monitoring
- Event transformation and routing between components
- Performance monitoring and metrics collection
- Graceful shutdown and error recovery
"""

import asyncio
import time
from typing import Any

from ..api.sse_broadcaster import SSEBroadcaster
from ..core.exceptions import ServiceError
from ..core.logging import get_logger
from ..integrations.kafka_analytics_connector import KafkaAnalyticsConnector
from ..services.kafka_event_producer import KafkaEventProducer
from ..services.kafka_sse_consumer import KafkaSSEConsumer

logger = get_logger(__name__)


class RealtimeStreamingService:
    """Coordinated real-time streaming service for camera analytics."""

    def __init__(
        self,
        kafka_producer: KafkaEventProducer,
        kafka_consumer: KafkaSSEConsumer,
        sse_broadcaster: SSEBroadcaster,
        analytics_connector: KafkaAnalyticsConnector,
        config: dict[str, Any],
    ):
        """Initialize real-time streaming service.
        
        Args:
            kafka_producer: Kafka event producer for publishing events
            kafka_consumer: Kafka consumer for SSE streaming
            sse_broadcaster: SSE broadcaster for web clients
            analytics_connector: Kafka analytics connector
            config: Service configuration dictionary
        """
        self.kafka_producer = kafka_producer
        self.kafka_consumer = kafka_consumer
        self.sse_broadcaster = sse_broadcaster
        self.analytics_connector = analytics_connector
        self.config = config

        # Service state
        self.is_running = False
        self.is_healthy = True
        self.start_time = 0.0

        # Health monitoring
        self.health_check_interval = config.get("health_check_interval", 30.0)
        self.auto_restart_on_failure = config.get("auto_restart_on_failure", True)
        self.max_restart_attempts = config.get("max_restart_attempts", 3)
        self.restart_attempts = 0

        # Performance tracking
        self.metrics = {
            "total_uptime": 0.0,
            "service_restarts": 0,
            "last_health_check": 0.0,
            "component_health": {
                "kafka_producer": False,
                "kafka_consumer": False,
                "sse_broadcaster": True,  # Always available
                "analytics_connector": False,
            },
        }

        # Background tasks
        self.monitoring_tasks: list[asyncio.Task] = []

        logger.info("Real-time streaming service initialized",
                   auto_restart=self.auto_restart_on_failure,
                   health_check_interval=self.health_check_interval)

    async def start(self):
        """Start the real-time streaming service and all components."""
        if self.is_running:
            logger.warning("Real-time streaming service already running")
            return

        self.start_time = time.time()
        self.restart_attempts = 0

        try:
            logger.info("Starting real-time streaming service...")

            # Start components in order
            await self._start_components()

            # Start health monitoring
            self.monitoring_tasks = [
                asyncio.create_task(self._health_monitor()),
                asyncio.create_task(self._metrics_collector()),
            ]

            self.is_running = True
            self.is_healthy = True

            logger.info("Real-time streaming service started successfully")

        except Exception as e:
            logger.error("Failed to start real-time streaming service", error=str(e))
            await self._cleanup()
            raise ServiceError(f"Real-time streaming service startup failed: {e}",
                             service="realtime_streaming") from e

    async def _start_components(self):
        """Start all service components."""
        component_errors = []

        # Start Kafka producer
        try:
            if hasattr(self.kafka_producer, 'start'):
                await self.kafka_producer.start()
            self.metrics["component_health"]["kafka_producer"] = True
            logger.info("Kafka producer started")
        except Exception as e:
            logger.error("Failed to start Kafka producer", error=str(e))
            component_errors.append(f"Kafka producer: {e}")

        # Start analytics connector (depends on producer)
        try:
            if hasattr(self.analytics_connector, 'start'):
                await self.analytics_connector.start()
            self.metrics["component_health"]["analytics_connector"] = True
            logger.info("Kafka analytics connector started")
        except Exception as e:
            logger.error("Failed to start analytics connector", error=str(e))
            component_errors.append(f"Analytics connector: {e}")

        # Start Kafka consumer (depends on SSE broadcaster)
        try:
            if hasattr(self.kafka_consumer, 'start'):
                await self.kafka_consumer.start()
            self.metrics["component_health"]["kafka_consumer"] = True
            logger.info("Kafka SSE consumer started")
        except Exception as e:
            logger.error("Failed to start Kafka consumer", error=str(e))
            component_errors.append(f"Kafka consumer: {e}")

        # SSE broadcaster is always available (started with container)
        self.metrics["component_health"]["sse_broadcaster"] = True

        if component_errors:
            error_msg = "; ".join(component_errors)
            logger.warning("Some components failed to start", errors=error_msg)

            # If all Kafka components failed, we can't provide real-time streaming
            kafka_components_failed = (
                not self.metrics["component_health"]["kafka_producer"] and
                not self.metrics["component_health"]["kafka_consumer"]
            )

            if kafka_components_failed:
                raise ServiceError("Critical components failed to start: " + error_msg,
                                 service="realtime_streaming")

    async def _health_monitor(self):
        """Background task for health monitoring and auto-restart."""
        while self.is_running:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Check component health
                await self._update_component_health()

                # Check overall service health
                unhealthy_components = [
                    name for name, healthy in self.metrics["component_health"].items()
                    if not healthy
                ]

                was_healthy = self.is_healthy
                self.is_healthy = len(unhealthy_components) <= 1  # Allow one component to be down

                if not self.is_healthy and was_healthy:
                    logger.warning("Service health degraded",
                                 unhealthy_components=unhealthy_components)

                # Auto-restart if enabled and service is unhealthy
                if (not self.is_healthy and
                    self.auto_restart_on_failure and
                    self.restart_attempts < self.max_restart_attempts):

                    logger.info("Attempting service restart",
                              attempt=self.restart_attempts + 1,
                              max_attempts=self.max_restart_attempts)

                    await self._attempt_restart()

                self.metrics["last_health_check"] = time.time()

            except Exception as e:
                logger.error("Health monitor error", error=str(e))
                await asyncio.sleep(self.health_check_interval)

    async def _update_component_health(self):
        """Update health status of all components."""
        try:
            # Check Kafka producer
            if hasattr(self.kafka_producer, 'is_healthy'):
                self.metrics["component_health"]["kafka_producer"] = self.kafka_producer.is_healthy
            elif hasattr(self.kafka_producer, 'get_health_status'):
                health = self.kafka_producer.get_health_status()
                self.metrics["component_health"]["kafka_producer"] = health.get("is_healthy", False)

            # Check Kafka consumer
            if hasattr(self.kafka_consumer, 'is_healthy'):
                self.metrics["component_health"]["kafka_consumer"] = self.kafka_consumer.is_healthy
            elif hasattr(self.kafka_consumer, 'get_health_status'):
                health = self.kafka_consumer.get_health_status()
                self.metrics["component_health"]["kafka_consumer"] = health.get("is_healthy", False)

            # Check analytics connector
            if hasattr(self.analytics_connector, 'is_running'):
                self.metrics["component_health"]["analytics_connector"] = self.analytics_connector.is_running
            elif hasattr(self.analytics_connector, 'get_health_status'):
                health = self.analytics_connector.get_health_status()
                self.metrics["component_health"]["analytics_connector"] = health.get("is_running", False)

            # SSE broadcaster health (check connection count)
            active_connections = len(self.sse_broadcaster.connections)
            max_connections = getattr(self.sse_broadcaster, 'max_connections', 200)
            self.metrics["component_health"]["sse_broadcaster"] = active_connections < max_connections

        except Exception as e:
            logger.error("Error updating component health", error=str(e))

    async def _attempt_restart(self):
        """Attempt to restart failed components."""
        self.restart_attempts += 1
        self.metrics["service_restarts"] += 1

        try:
            logger.info("Restarting failed components...")

            # Stop and restart components that are unhealthy
            for component_name, is_healthy in self.metrics["component_health"].items():
                if not is_healthy and component_name != "sse_broadcaster":
                    await self._restart_component(component_name)

            # Allow some time for components to stabilize
            await asyncio.sleep(5.0)

            # Update health after restart
            await self._update_component_health()

            logger.info("Component restart completed",
                       restart_attempt=self.restart_attempts)

        except Exception as e:
            logger.error("Component restart failed",
                        restart_attempt=self.restart_attempts,
                        error=str(e))

    async def _restart_component(self, component_name: str):
        """Restart a specific component."""
        try:
            component = getattr(self, component_name.replace("_", "_"), None)

            if component and hasattr(component, 'stop'):
                logger.info(f"Stopping {component_name}")
                await component.stop()

            if component and hasattr(component, 'start'):
                logger.info(f"Starting {component_name}")
                await component.start()
                self.metrics["component_health"][component_name] = True

        except Exception as e:
            logger.error(f"Failed to restart {component_name}", error=str(e))
            self.metrics["component_health"][component_name] = False

    async def _metrics_collector(self):
        """Background task for collecting performance metrics."""
        while self.is_running:
            try:
                await asyncio.sleep(60.0)  # Collect metrics every minute

                # Update uptime
                self.metrics["total_uptime"] = time.time() - self.start_time

                # Collect component metrics
                component_metrics = {}

                # Kafka producer metrics
                if hasattr(self.kafka_producer, 'get_health_status'):
                    component_metrics["kafka_producer"] = self.kafka_producer.get_health_status()

                # Kafka consumer metrics
                if hasattr(self.kafka_consumer, 'get_health_status'):
                    component_metrics["kafka_consumer"] = self.kafka_consumer.get_health_status()

                # SSE broadcaster metrics
                if hasattr(self.sse_broadcaster, 'get_stats'):
                    component_metrics["sse_broadcaster"] = self.sse_broadcaster.get_stats()

                # Analytics connector metrics
                if hasattr(self.analytics_connector, 'get_health_status'):
                    component_metrics["analytics_connector"] = self.analytics_connector.get_health_status()

                # Log summary metrics
                logger.info("Real-time streaming service metrics",
                           uptime=self.metrics["total_uptime"],
                           restarts=self.metrics["service_restarts"],
                           is_healthy=self.is_healthy,
                           component_health=self.metrics["component_health"],
                           active_sse_connections=len(self.sse_broadcaster.connections))

            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(60.0)

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive service health status."""
        uptime = time.time() - self.start_time if self.is_running else 0.0

        return {
            "service": "realtime_streaming",
            "is_running": self.is_running,
            "is_healthy": self.is_healthy,
            "uptime_seconds": uptime,
            "restart_attempts": self.restart_attempts,
            "max_restart_attempts": self.max_restart_attempts,
            "component_health": self.metrics["component_health"].copy(),
            "metrics": {
                **self.metrics,
                "current_uptime": uptime,
            },
            "configuration": {
                "health_check_interval": self.health_check_interval,
                "auto_restart_on_failure": self.auto_restart_on_failure,
            },
            "sse_connections": len(self.sse_broadcaster.connections),
        }

    async def publish_detection_result(self, camera_id: str, frame_id: str, detection_results: list[dict]):
        """Publish detection result through the analytics connector."""
        if not self.is_running or not self.metrics["component_health"]["analytics_connector"]:
            logger.warning("Analytics connector not available for detection publishing")
            return False

        try:
            return await self.analytics_connector.publish_detection_result(
                camera_id=camera_id,
                frame_id=frame_id,
                detection_results=detection_results
            )
        except Exception as e:
            logger.error("Failed to publish detection result", error=str(e))
            return False

    async def publish_camera_status(self, camera_id: str, status: str, metadata: dict):
        """Publish camera status through the analytics connector."""
        if not self.is_running or not self.metrics["component_health"]["analytics_connector"]:
            logger.warning("Analytics connector not available for status publishing")
            return False

        try:
            return await self.analytics_connector.publish_camera_status(
                camera_id=camera_id,
                status=status,
                metadata=metadata
            )
        except Exception as e:
            logger.error("Failed to publish camera status", error=str(e))
            return False

    async def publish_analytics_result(self, zone_id: str, analytics_type: str, analytics_data: dict):
        """Publish analytics result through the analytics connector."""
        if not self.is_running or not self.metrics["component_health"]["analytics_connector"]:
            logger.warning("Analytics connector not available for analytics publishing")
            return False

        try:
            return await self.analytics_connector.publish_analytics_result(
                zone_id=zone_id,
                analytics_type=analytics_type,
                analytics_data=analytics_data
            )
        except Exception as e:
            logger.error("Failed to publish analytics result", error=str(e))
            return False

    async def stop(self):
        """Stop the real-time streaming service and all components."""
        if not self.is_running:
            logger.warning("Real-time streaming service not running")
            return

        logger.info("Stopping real-time streaming service...")

        self.is_running = False

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()

        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)

        # Stop components
        await self._cleanup()

        # Update metrics
        self.metrics["total_uptime"] = time.time() - self.start_time

        logger.info("Real-time streaming service stopped",
                   total_uptime=self.metrics["total_uptime"],
                   total_restarts=self.metrics["service_restarts"])

    async def _cleanup(self):
        """Cleanup all components during shutdown."""
        cleanup_errors = []

        # Stop Kafka consumer
        try:
            if hasattr(self.kafka_consumer, 'stop'):
                await self.kafka_consumer.stop()
            logger.info("Kafka consumer stopped")
        except Exception as e:
            logger.error("Error stopping Kafka consumer", error=str(e))
            cleanup_errors.append(f"Kafka consumer: {e}")

        # Stop analytics connector
        try:
            if hasattr(self.analytics_connector, 'stop'):
                await self.analytics_connector.stop()
            logger.info("Analytics connector stopped")
        except Exception as e:
            logger.error("Error stopping analytics connector", error=str(e))
            cleanup_errors.append(f"Analytics connector: {e}")

        # Stop Kafka producer
        try:
            if hasattr(self.kafka_producer, 'stop'):
                await self.kafka_producer.stop()
            logger.info("Kafka producer stopped")
        except Exception as e:
            logger.error("Error stopping Kafka producer", error=str(e))
            cleanup_errors.append(f"Kafka producer: {e}")

        # Reset component health
        for component in self.metrics["component_health"]:
            self.metrics["component_health"][component] = False

        self.is_healthy = False

        if cleanup_errors:
            error_msg = "; ".join(cleanup_errors)
            logger.warning("Cleanup completed with errors", errors=error_msg)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Factory function for easy initialization
def create_realtime_streaming_service(
    kafka_producer: KafkaEventProducer,
    kafka_consumer: KafkaSSEConsumer,
    sse_broadcaster: SSEBroadcaster,
    analytics_connector: KafkaAnalyticsConnector,
    config: dict[str, Any] | None = None,
) -> RealtimeStreamingService:
    """Create and configure a real-time streaming service.
    
    Args:
        kafka_producer: Kafka event producer instance
        kafka_consumer: Kafka SSE consumer instance
        sse_broadcaster: SSE broadcaster instance
        analytics_connector: Analytics connector instance
        config: Optional service configuration
        
    Returns:
        Configured RealtimeStreamingService instance
    """
    default_config = {
        "health_check_interval": 30.0,
        "auto_restart_on_failure": True,
        "max_restart_attempts": 3,
    }

    if config:
        default_config.update(config)

    return RealtimeStreamingService(
        kafka_producer=kafka_producer,
        kafka_consumer=kafka_consumer,
        sse_broadcaster=sse_broadcaster,
        analytics_connector=analytics_connector,
        config=default_config,
    )
