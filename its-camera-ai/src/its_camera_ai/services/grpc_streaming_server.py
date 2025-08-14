"""gRPC Streaming Server Implementation.

This module implements the gRPC server for the ITS Camera AI streaming service,
handling camera registration, frame processing, and system monitoring.

Key Features:
- Full gRPC service implementation
- Bidirectional streaming for real-time frame processing
- Health checks and system metrics
- Queue management and monitoring
- High-performance async operations
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import grpc
import numpy as np

# Internal imports
from ..core.logging import get_logger
from ..flow.redis_queue_manager import RedisQueueManager
from ..proto import processed_frame_pb2 as frame_pb
from ..proto import streaming_service_pb2 as pb
from ..proto import streaming_service_pb2_grpc as pb_grpc
from .streaming_service import (
    CameraConfig,
    StreamingDataProcessor,
    StreamProtocol,
)

logger = get_logger(__name__)


class StreamingServiceImpl(pb_grpc.StreamingServiceServicer):
    """gRPC implementation of the streaming service."""

    def __init__(
        self,
        streaming_processor: StreamingDataProcessor | None = None,
        redis_manager: RedisQueueManager | None = None,
    ):
        """
        Initialize the gRPC streaming service.

        Args:
            streaming_processor: The streaming data processor instance
            redis_manager: Redis queue manager for metrics
        """
        self.streaming_processor = streaming_processor or StreamingDataProcessor()
        self.redis_manager = redis_manager or RedisQueueManager()
        self.server: grpc.aio.Server | None = None
        self.is_serving = False

        logger.info("StreamingServiceImpl initialized")

    async def start_server(self, host: str = "127.0.0.1", port: int = 50051) -> None:
        """Start the gRPC server."""
        try:
            # Start the streaming processor
            await self.streaming_processor.start()

            # Create and configure gRPC server
            self.server = grpc.aio.server()
            pb_grpc.add_StreamingServiceServicer_to_server(self, self.server)

            # Configure server options for high throughput
            options = [
                ("grpc.keepalive_time_ms", 10000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_time_between_pings_ms", 10000),
                ("grpc.http2.min_ping_interval_without_data_ms", 300000),
                ("grpc.max_receive_message_length", 16 * 1024 * 1024),  # 16MB
                ("grpc.max_send_message_length", 16 * 1024 * 1024),  # 16MB
            ]

            for _option in options:
                self.server.add_generic_rpc_handlers(
                    (grpc.method_handlers_generic_handler("", {}),)
                )

            listen_addr = f"{host}:{port}"
            self.server.add_insecure_port(listen_addr)

            await self.server.start()
            self.is_serving = True

            logger.info(f"gRPC Streaming Server started on {listen_addr}")

        except Exception as e:
            logger.error(f"Failed to start gRPC server: {e}")
            raise

    async def stop_server(self) -> None:
        """Stop the gRPC server gracefully."""
        if self.server:
            logger.info("Stopping gRPC server...")

            # Stop accepting new requests
            await self.server.stop(grace=5.0)
            self.is_serving = False

            # Stop the streaming processor
            await self.streaming_processor.stop()

            logger.info("gRPC server stopped")

    async def StreamFrames(
        self,
        request_iterator: AsyncIterator[frame_pb.ProcessedFrame],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[pb.StreamResponse]:
        """Handle bidirectional streaming of frames."""
        logger.info("StreamFrames called - starting bidirectional stream")

        try:
            async for request in request_iterator:
                start_time = time.perf_counter()

                try:
                    # Convert protobuf frame to numpy array (simplified)
                    frame_data = self._protobuf_to_numpy(request)

                    if frame_data is not None:
                        # Process frame through streaming processor
                        camera_config = CameraConfig(
                            camera_id=request.camera_id,
                            stream_url="grpc://stream",
                            resolution=(frame_data.shape[1], frame_data.shape[0]),
                            fps=30,
                            protocol=StreamProtocol.HTTP,  # gRPC transport
                        )

                        # Validate frame quality
                        quality_metrics = (
                            await self.streaming_processor.validate_frame_quality(
                                frame_data, camera_config
                            )
                        )

                        processing_time = (time.perf_counter() - start_time) * 1000

                        # Create response
                        response = pb.StreamResponse(
                            success=quality_metrics.passed_validation,
                            message=(
                                "Frame processed successfully"
                                if quality_metrics.passed_validation
                                else "Quality validation failed"
                            ),
                            frame_id=request.frame_id,
                            processing_time_ms=processing_time,
                        )

                        yield response
                    else:
                        # Invalid frame data
                        yield pb.StreamResponse(
                            success=False,
                            message="Invalid frame data",
                            frame_id=request.frame_id,
                            processing_time_ms=0.0,
                        )

                except Exception as e:
                    logger.error(f"Error processing frame {request.frame_id}: {e}")
                    yield pb.StreamResponse(
                        success=False,
                        message=f"Processing error: {str(e)}",
                        frame_id=request.frame_id,
                        processing_time_ms=0.0,
                    )

        except Exception as e:
            logger.error(f"StreamFrames error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Stream processing failed: {e}"
            )

    async def ProcessFrameBatch(
        self, request: frame_pb.ProcessedFrameBatch, context: grpc.aio.ServicerContext
    ) -> pb.BatchResponse:
        """Process a batch of frames for improved efficiency."""
        logger.info(f"ProcessFrameBatch called with {len(request.frames)} frames")

        start_time = time.perf_counter()
        processed_count = 0
        failed_count = 0
        errors = []

        try:
            for frame in request.frames:
                try:
                    # Convert and process frame
                    frame_data = self._protobuf_to_numpy(frame)

                    if frame_data is not None:
                        camera_config = CameraConfig(
                            camera_id=frame.camera_id,
                            stream_url="grpc://batch",
                            resolution=(frame_data.shape[1], frame_data.shape[0]),
                            fps=30,
                            protocol=StreamProtocol.HTTP,
                        )

                        quality_metrics = (
                            await self.streaming_processor.validate_frame_quality(
                                frame_data, camera_config
                            )
                        )

                        if quality_metrics.passed_validation:
                            processed_count += 1
                        else:
                            failed_count += 1
                            error = pb.ProcessingError(
                                error_id=str(uuid.uuid4()),
                                frame_id=frame.frame_id,
                                camera_id=frame.camera_id,
                                error_type="QUALITY_VALIDATION_FAILED",
                                error_message=f"Quality issues: {', '.join(quality_metrics.issues)}",
                                timestamp=time.time(),
                                failed_stage=frame_pb.PROCESSING_STAGE_VALIDATION,
                            )
                            errors.append(error)
                    else:
                        failed_count += 1
                        error = pb.ProcessingError(
                            error_id=str(uuid.uuid4()),
                            frame_id=frame.frame_id,
                            camera_id=frame.camera_id,
                            error_type="INVALID_FRAME_DATA",
                            error_message="Could not convert frame data",
                            timestamp=time.time(),
                            failed_stage=frame_pb.PROCESSING_STAGE_INGESTION,
                        )
                        errors.append(error)

                except Exception as e:
                    failed_count += 1
                    error = pb.ProcessingError(
                        error_id=str(uuid.uuid4()),
                        frame_id=(
                            frame.frame_id if hasattr(frame, "frame_id") else "unknown"
                        ),
                        camera_id=(
                            frame.camera_id
                            if hasattr(frame, "camera_id")
                            else "unknown"
                        ),
                        error_type="PROCESSING_EXCEPTION",
                        error_message=str(e),
                        timestamp=time.time(),
                        failed_stage=frame_pb.PROCESSING_STAGE_VALIDATION,
                    )
                    errors.append(error)

            total_processing_time = (time.perf_counter() - start_time) * 1000

            return pb.BatchResponse(
                success=failed_count == 0,
                batch_id=request.batch_id,
                processed_count=processed_count,
                failed_count=failed_count,
                errors=errors,
                total_processing_time_ms=total_processing_time,
            )

        except Exception as e:
            logger.error(f"BatchProcess error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Batch processing failed: {e}"
            )

    async def RegisterStream(
        self, request: frame_pb.CameraStreamConfig, context: grpc.aio.ServicerContext
    ) -> pb.StreamRegistrationResponse:
        """Register a new camera stream."""
        logger.info(f"RegisterStream called for camera: {request.camera_id}")

        try:
            # Convert protobuf config to internal format
            camera_config = CameraConfig(
                camera_id=request.camera_id,
                stream_url=f"stream://{request.camera_id}",  # Placeholder
                resolution=(request.width, request.height),
                fps=request.fps,
                protocol=StreamProtocol.HTTP,  # Default
                location=request.location,
                coordinates=(
                    (request.latitude, request.longitude)
                    if request.latitude and request.longitude
                    else None
                ),
                quality_threshold=request.quality_threshold,
                roi_boxes=[
                    (roi.x, roi.y, roi.width, roi.height) for roi in request.roi_boxes
                ],
                enabled=request.processing_enabled,
            )

            # Register with streaming processor
            registration = await self.streaming_processor.register_camera(camera_config)

            return pb.StreamRegistrationResponse(
                success=registration.success,
                camera_id=registration.camera_id,
                message=registration.message,
            )

        except Exception as e:
            logger.error(f"RegisterStream error: {e}")
            return pb.StreamRegistrationResponse(
                success=False,
                camera_id=request.camera_id,
                message=f"Registration failed: {str(e)}",
            )

    async def UpdateStreamConfig(
        self, request: frame_pb.CameraStreamConfig, context: grpc.aio.ServicerContext
    ) -> pb.StreamUpdateResponse:
        """Update camera stream configuration."""
        logger.info(f"UpdateStreamConfig called for camera: {request.camera_id}")

        try:
            # For now, we'll treat this as a re-registration
            # In a full implementation, this would update existing config

            camera_config = CameraConfig(
                camera_id=request.camera_id,
                stream_url=f"stream://{request.camera_id}",
                resolution=(request.width, request.height),
                fps=request.fps,
                protocol=StreamProtocol.HTTP,
                location=request.location,
                coordinates=(
                    (request.latitude, request.longitude)
                    if request.latitude and request.longitude
                    else None
                ),
                quality_threshold=request.quality_threshold,
                roi_boxes=[
                    (roi.x, roi.y, roi.width, roi.height) for roi in request.roi_boxes
                ],
                enabled=request.processing_enabled,
            )

            # Re-register (this would be an update in production)
            registration = await self.streaming_processor.register_camera(camera_config)

            return pb.StreamUpdateResponse(
                success=registration.success,
                camera_id=registration.camera_id,
                message=(
                    "Configuration updated successfully"
                    if registration.success
                    else registration.message
                ),
            )

        except Exception as e:
            logger.error(f"UpdateStreamConfig error: {e}")
            return pb.StreamUpdateResponse(
                success=False,
                camera_id=request.camera_id,
                message=f"Update failed: {str(e)}",
            )

    async def GetStreamStatus(
        self, request: pb.StreamStatusRequest, context: grpc.aio.ServicerContext
    ) -> frame_pb.CameraStreamConfig:
        """Get status and configuration for a camera stream."""
        logger.info(f"GetStreamStatus called for camera: {request.camera_id}")

        try:
            # Get camera config from streaming processor
            if request.camera_id in self.streaming_processor.registered_cameras:
                config = self.streaming_processor.registered_cameras[request.camera_id]

                # Create response with current configuration
                response = frame_pb.CameraStreamConfig(
                    camera_id=config.camera_id,
                    location=config.location or "",
                    latitude=config.coordinates[0] if config.coordinates else 0.0,
                    longitude=config.coordinates[1] if config.coordinates else 0.0,
                    width=config.resolution[0],
                    height=config.resolution[1],
                    fps=config.fps,
                    encoding="h264",  # Default
                    quality_threshold=config.quality_threshold,
                    processing_enabled=config.enabled,
                    status=(
                        frame_pb.STREAM_STATUS_ACTIVE
                        if config.enabled
                        else frame_pb.STREAM_STATUS_INACTIVE
                    ),
                    last_frame_time=time.time(),  # Placeholder
                    total_frames_processed=0,  # Placeholder
                )

                # Add ROI boxes
                for roi in config.roi_boxes:
                    roi_box = frame_pb.ROIBox(
                        x=roi[0], y=roi[1], width=roi[2], height=roi[3], label="roi"
                    )
                    response.roi_boxes.append(roi_box)

                return response
            else:
                await context.abort(
                    grpc.StatusCode.NOT_FOUND, f"Camera {request.camera_id} not found"
                )

        except Exception as e:
            logger.error(f"GetStreamStatus error: {e}")
            await context.abort(grpc.StatusCode.INTERNAL, f"Status check failed: {e}")

    async def GetQueueMetrics(
        self, request: pb.QueueMetricsRequest, context: grpc.aio.ServicerContext
    ) -> frame_pb.QueueMetrics:
        """Get metrics for a specific queue."""
        logger.info(f"GetQueueMetrics called for queue: {request.queue_name}")

        try:
            if self.redis_manager:
                metrics = await self.redis_manager.get_queue_metrics(request.queue_name)

                if metrics:
                    return frame_pb.QueueMetrics(
                        queue_name=metrics.queue_name,
                        pending_count=metrics.pending_count,
                        processing_count=metrics.processing_count,
                        completed_count=metrics.completed_count,
                        failed_count=metrics.failed_count,
                        avg_processing_time_ms=metrics.avg_processing_time_ms,
                        throughput_fps=metrics.throughput_fps,
                    )
                else:
                    await context.abort(
                        grpc.StatusCode.NOT_FOUND,
                        f"Queue {request.queue_name} not found",
                    )
            else:
                await context.abort(
                    grpc.StatusCode.UNAVAILABLE, "Redis manager not available"
                )

        except Exception as e:
            logger.error(f"GetQueueMetrics error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Metrics retrieval failed: {e}"
            )

    async def PurgeQueue(
        self, request: pb.PurgeQueueRequest, context: grpc.aio.ServicerContext
    ) -> pb.PurgeQueueResponse:
        """Purge all messages from a queue."""
        logger.info(f"PurgeQueue called for queue: {request.queue_name}")

        try:
            if self.redis_manager:
                purged_count = await self.redis_manager.purge_queue(
                    request.queue_name, request.force
                )

                return pb.PurgeQueueResponse(
                    success=purged_count > 0,
                    purged_count=purged_count,
                    message=(
                        f"Purged {purged_count} messages"
                        if purged_count > 0
                        else "No messages to purge"
                    ),
                )
            else:
                return pb.PurgeQueueResponse(
                    success=False, purged_count=0, message="Redis manager not available"
                )

        except Exception as e:
            logger.error(f"PurgeQueue error: {e}")
            return pb.PurgeQueueResponse(
                success=False, purged_count=0, message=f"Purge failed: {str(e)}"
            )

    async def HealthCheck(
        self, request: pb.HealthCheckRequest, context: grpc.aio.ServicerContext
    ) -> pb.HealthCheckResponse:
        """Perform health check on the service."""
        logger.debug(f"HealthCheck called for service: {request.service_name}")

        start_time = time.perf_counter()

        try:
            # Get health status from streaming processor
            health_status = await self.streaming_processor.get_health_status()

            response_time = (time.perf_counter() - start_time) * 1000

            # Determine overall health
            is_healthy = health_status.get(
                "service_status"
            ) == "healthy" and health_status.get("redis_status") in [
                "healthy",
                "connected",
            ]

            status = (
                pb.HealthCheckResponse.Status.SERVING
                if is_healthy
                else pb.HealthCheckResponse.Status.NOT_SERVING
            )
            message = "Service is healthy" if is_healthy else "Service has issues"

            return pb.HealthCheckResponse(
                status=status, message=message, response_time_ms=response_time
            )

        except Exception as e:
            logger.error(f"HealthCheck error: {e}")
            response_time = (time.perf_counter() - start_time) * 1000

            return pb.HealthCheckResponse(
                status=pb.HealthCheckResponse.Status.NOT_SERVING,
                message=f"Health check failed: {str(e)}",
                response_time_ms=response_time,
            )

    async def GetSystemMetrics(
        self, request: pb.SystemMetricsRequest, context: grpc.aio.ServicerContext
    ) -> pb.SystemMetricsResponse:
        """Get comprehensive system metrics."""
        logger.info("GetSystemMetrics called")

        try:
            # Get processing metrics
            processing_metrics = self.streaming_processor.get_processing_metrics()

            perf_metrics = pb.PerformanceMetrics(
                frames_processed=processing_metrics.get("frames_processed", 0),
                frames_rejected=processing_metrics.get("frames_rejected", 0),
                avg_processing_time_ms=processing_metrics.get(
                    "avg_processing_time_ms", 0.0
                ),
                throughput_fps=processing_metrics.get("throughput_fps", 0.0),
                error_count=processing_metrics.get("error_count", 0),
                memory_usage_mb=processing_metrics.get("memory_usage_mb", 0.0),
                cpu_usage_percent=0.0,  # TODO: Implement CPU monitoring
                active_connections=processing_metrics.get("active_connections", 0),
            )

            queue_metrics = []
            if request.include_queue_metrics and self.redis_manager:
                # Get metrics for all queues
                all_metrics = await self.redis_manager.get_all_metrics()

                for _queue_name, metrics in all_metrics.items():
                    queue_metric = frame_pb.QueueMetrics(
                        queue_name=metrics.queue_name,
                        pending_count=metrics.pending_count,
                        processing_count=metrics.processing_count,
                        completed_count=metrics.completed_count,
                        failed_count=metrics.failed_count,
                        avg_processing_time_ms=metrics.avg_processing_time_ms,
                        throughput_fps=metrics.throughput_fps,
                    )
                    queue_metrics.append(queue_metric)

            return pb.SystemMetricsResponse(
                queue_metrics=queue_metrics,
                performance_metrics=perf_metrics,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(f"GetSystemMetrics error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Metrics collection failed: {e}"
            )

    def _protobuf_to_numpy(self, frame: frame_pb.ProcessedFrame) -> np.ndarray | None:
        """Convert protobuf frame to numpy array."""
        try:
            if not frame.original_image or not frame.original_image.compressed_data:
                return None

            # For now, create a dummy frame based on dimensions
            # In production, this would decode the compressed_data
            height = frame.original_image.height or 480
            width = frame.original_image.width or 640
            channels = frame.original_image.channels or 3

            # Create placeholder frame
            frame_array = np.random.randint(
                0, 255, (height, width, channels), dtype=np.uint8
            )

            return frame_array

        except Exception as e:
            logger.error(f"Frame conversion error: {e}")
            return None

    def _numpy_to_protobuf(self, frame: np.ndarray) -> frame_pb.ImageData:
        """Convert numpy array to protobuf ImageData."""
        try:
            height, width = frame.shape[:2]
            channels = frame.shape[2] if len(frame.shape) == 3 else 1

            # In production, this would compress the frame data
            # For now, we'll create a placeholder

            return frame_pb.ImageData(
                compressed_data=b"placeholder_compressed_data",
                width=width,
                height=height,
                channels=channels,
                compression_format="jpeg",
                quality=85,
            )

        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
            return frame_pb.ImageData()


class StreamingServer:
    """High-level streaming server manager."""

    def __init__(self, host: str = "127.0.0.1", port: int = 50051):
        self.host = host
        self.port = port
        self.service_impl = StreamingServiceImpl()
        self.is_running = False

    async def start(self) -> None:
        """Start the streaming server."""
        if self.is_running:
            logger.warning("Server already running")
            return

        try:
            await self.service_impl.start_server(self.host, self.port)
            self.is_running = True

            logger.info(
                f"Streaming server started successfully on {self.host}:{self.port}"
            )

        except Exception as e:
            logger.error(f"Failed to start streaming server: {e}")
            raise

    async def stop(self) -> None:
        """Stop the streaming server."""
        if not self.is_running:
            return

        await self.service_impl.stop_server()
        self.is_running = False

        logger.info("Streaming server stopped")

    async def serve_forever(self) -> None:
        """Start server and run until interrupted."""
        await self.start()

        try:
            # Wait for server shutdown
            while self.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.stop()

    async def __aenter__(self) -> "StreamingServer":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()


# Main entry point for standalone server
async def main() -> None:
    """Main entry point for the gRPC streaming server."""
    import argparse

    parser = argparse.ArgumentParser(description="ITS Camera AI Streaming Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    server = StreamingServer(host=args.host, port=args.port)

    try:
        await server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
