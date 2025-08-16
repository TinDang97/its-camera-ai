"""Production ML gRPC Server Implementation.

This server provides the complete gRPC interface for the ITS Camera AI system,
integrating the ML pipeline with streaming endpoints for production deployment.

Features:
- Vision Core Service with real ML inference
- Streaming Service with frame processing
- Analytics Service with real-time metrics
- Health monitoring and performance tracking
- Proper error handling and graceful degradation
"""

import asyncio
import signal
import time
from concurrent import futures
from typing import Any

import grpc
from grpc_reflection.v1alpha import reflection

from ..core.logging import get_logger
from ..proto import (
    analytics_service_pb2_grpc,
    streaming_service_pb2_grpc,
    vision_core_pb2_grpc,
)
from ..services.analytics_grpc_server import AnalyticsServiceServicer
from ..services.grpc.vision_core_servicer import VisionCoreServicer
from ..services.grpc_streaming_server import StreamingServiceServicer
from ..services.ml_streaming_integration_service import MLStreamingIntegrationService

logger = get_logger(__name__)


class ProductionMLgRPCServer:
    """Production gRPC server with complete ML pipeline integration."""

    def __init__(
        self,
        ml_integration_service: MLStreamingIntegrationService,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 10,
        max_message_length: int = 100 * 1024 * 1024,  # 100MB
    ):
        """Initialize production gRPC server."""
        self.ml_integration_service = ml_integration_service
        self.host = host
        self.port = port
        self.max_workers = max_workers
        self.max_message_length = max_message_length

        # gRPC server
        self.server = None
        self.is_running = False

        # Servicers
        self.vision_core_servicer = None
        self.streaming_servicer = None
        self.analytics_servicer = None

        logger.info(f"Production ML gRPC Server initialized on {host}:{port}")

    async def start(self) -> None:
        """Start the production gRPC server."""
        if self.is_running:
            logger.warning("gRPC server already running")
            return

        logger.info("Starting production ML gRPC server...")

        try:
            # Initialize ML integration service
            await self.ml_integration_service.initialize()
            await self.ml_integration_service.start()

            # Create and configure gRPC server
            self.server = grpc.aio.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=[
                    ("grpc.max_send_message_length", self.max_message_length),
                    ("grpc.max_receive_message_length", self.max_message_length),
                    ("grpc.keepalive_time_ms", 10000),
                    ("grpc.keepalive_timeout_ms", 2000),
                    ("grpc.keepalive_permit_without_calls", True),
                    ("grpc.http2.max_pings_without_data", 0),
                    ("grpc.http2.min_time_between_pings_ms", 5000),
                ],
            )

            # Initialize and register servicers
            await self._initialize_servicers()

            # Add reflection for debugging
            service_names = (
                vision_core_pb2_grpc.DESCRIPTOR.services_by_name["VisionCoreService"].full_name,
                streaming_service_pb2_grpc.DESCRIPTOR.services_by_name["StreamingService"].full_name,
                analytics_service_pb2_grpc.DESCRIPTOR.services_by_name["AnalyticsService"].full_name,
                reflection.SERVICE_NAME,
            )
            reflection.enable_server_reflection(service_names, self.server)

            # Start server
            listen_addr = f"{self.host}:{self.port}"
            self.server.add_insecure_port(listen_addr)
            await self.server.start()

            self.is_running = True

            logger.info(f"Production ML gRPC server started on {listen_addr}")
            logger.info("Available services:")
            logger.info("- VisionCoreService: Complete ML inference + analytics")
            logger.info("- StreamingService: Real-time frame processing")
            logger.info("- AnalyticsService: Traffic analytics and metrics")

        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")
            raise

    async def stop(self, grace_period: float = 5.0) -> None:
        """Stop the gRPC server gracefully."""
        if not self.is_running:
            return

        logger.info("Stopping production ML gRPC server...")

        try:
            if self.server:
                await self.server.stop(grace_period)

            # Stop ML integration service
            await self.ml_integration_service.stop()

            self.is_running = False
            logger.info("Production ML gRPC server stopped")

        except Exception as e:
            logger.error(f"Error stopping gRPC server: {e}")

    async def wait_for_termination(self) -> None:
        """Wait for server termination."""
        if self.server:
            await self.server.wait_for_termination()

    async def health_check(self) -> dict[str, Any]:
        """Get comprehensive server health status."""
        try:
            # Get ML service health
            ml_health = await self.ml_integration_service.health_check()

            # Get servicer health
            servicer_health = {
                "vision_core_servicer": self.vision_core_servicer is not None,
                "streaming_servicer": self.streaming_servicer is not None,
                "analytics_servicer": self.analytics_servicer is not None,
            }

            return {
                "server_running": self.is_running,
                "address": f"{self.host}:{self.port}",
                "ml_service": ml_health,
                "servicers": servicer_health,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "server_running": self.is_running,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def _initialize_servicers(self) -> None:
        """Initialize and register all gRPC servicers."""
        logger.info("Initializing gRPC servicers...")

        # Get unified vision analytics engine from ML service
        unified_engine = self.ml_integration_service.unified_vision_analytics
        if not unified_engine:
            raise RuntimeError("Unified vision analytics engine not available")

        # Vision Core Servicer - Main ML inference service
        self.vision_core_servicer = VisionCoreServicer(unified_engine)
        vision_core_pb2_grpc.add_VisionCoreServiceServicer_to_server(
            self.vision_core_servicer, self.server
        )

        # Streaming Servicer - Real-time frame processing
        self.streaming_servicer = StreamingServiceServicer(
            ml_integration_service=self.ml_integration_service
        )
        streaming_service_pb2_grpc.add_StreamingServiceServicer_to_server(
            self.streaming_servicer, self.server
        )

        # Analytics Servicer - Traffic analytics and metrics
        self.analytics_servicer = AnalyticsServiceServicer(
            analytics_service=self.ml_integration_service.unified_analytics,
            ml_integration_service=self.ml_integration_service,
        )
        analytics_service_pb2_grpc.add_AnalyticsServiceServicer_to_server(
            self.analytics_servicer, self.server
        )

        logger.info("All gRPC servicers initialized successfully")


class ProductionMLgRPCServerManager:
    """Manager for production gRPC server with proper lifecycle management."""

    def __init__(self, ml_integration_service: MLStreamingIntegrationService):
        """Initialize server manager."""
        self.ml_integration_service = ml_integration_service
        self.server = None
        self.shutdown_event = asyncio.Event()

    async def run_server(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        max_workers: int = 10,
    ) -> None:
        """Run the production gRPC server with proper signal handling."""
        # Create server
        self.server = ProductionMLgRPCServer(
            ml_integration_service=self.ml_integration_service,
            host=host,
            port=port,
            max_workers=max_workers,
        )

        # Setup signal handlers for graceful shutdown
        def signal_handler():
            logger.info("Received shutdown signal")
            self.shutdown_event.set()

        # Register signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)

        try:
            # Start server
            await self.server.start()

            # Log server info
            health = await self.server.health_check()
            logger.info(f"Server health: {health}")

            # Wait for shutdown signal
            logger.info("Server running. Press Ctrl+C to stop.")
            await self.shutdown_event.wait()

        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            # Graceful shutdown
            logger.info("Initiating graceful shutdown...")
            if self.server:
                await self.server.stop(grace_period=10.0)

    async def stop(self) -> None:
        """Stop the server."""
        self.shutdown_event.set()


# Production server factory function
async def create_production_ml_grpc_server(
    ml_integration_service: MLStreamingIntegrationService = None,
    host: str = "0.0.0.0",
    port: int = 50051,
    max_workers: int = 10,
) -> ProductionMLgRPCServer:
    """Factory function to create production ML gRPC server."""

    if ml_integration_service is None:
        # Create default ML integration service
        from ..core.container import Container
        container = Container()
        container.wire(modules=[__name__])

        ml_integration_service = MLStreamingIntegrationService()
        await ml_integration_service.initialize()

    server = ProductionMLgRPCServer(
        ml_integration_service=ml_integration_service,
        host=host,
        port=port,
        max_workers=max_workers,
    )

    return server


# Main entry point for standalone server
async def main():
    """Main entry point for production ML gRPC server."""
    from ..core.container import Container

    # Initialize dependency injection container
    container = Container()
    container.wire(modules=[__name__])

    # Create ML integration service
    ml_service = MLStreamingIntegrationService()

    # Create and run server manager
    manager = ProductionMLgRPCServerManager(ml_service)
    await manager.run_server(
        host="0.0.0.0",
        port=50051,
        max_workers=20,
    )


if __name__ == "__main__":
    asyncio.run(main())
