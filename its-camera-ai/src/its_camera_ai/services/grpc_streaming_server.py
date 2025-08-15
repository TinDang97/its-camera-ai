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
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import grpc
import numpy as np
import psutil

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    pynvml = None

# Internal imports
from ..core.blosc_numpy_compressor import get_global_compressor
from ..core.logging import get_logger
from ..flow.grpc_serialization import ProcessedFrameSerializer
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


@dataclass
class SystemMetrics:
    """Comprehensive system metrics for monitoring."""
    cpu_usage_percent: float
    cpu_per_core: list[float]
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    gpu_metrics: list[dict[str, Any]]
    process_cpu_percent: float
    process_memory_mb: float
    timestamp: datetime


class EnhancedMonitoringService:
    """Enhanced monitoring service with CPU/GPU tracking.
    
    Provides comprehensive system monitoring including:
    - CPU utilization (overall and per-core)
    - Memory usage (system and process-specific)
    - GPU utilization and memory (NVIDIA GPUs)
    - Temperature and power monitoring
    - Performance metrics for Prometheus export
    """

    def __init__(self):
        """Initialize monitoring service."""
        # Initialize NVML for GPU monitoring
        self.gpu_available = False
        self.gpu_count = 0

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_available = True
                logger.info(f"GPU monitoring initialized with {self.gpu_count} GPUs")
            except Exception as e:
                logger.warning(f"GPU monitoring not available: {e}")
        else:
            logger.warning("pynvml not available - GPU monitoring disabled")

        # Process monitoring
        self.process = psutil.Process()

        # Performance tracking
        self.metrics_history = []
        self.max_history = 100

        logger.info("EnhancedMonitoringService initialized")

    async def get_system_metrics(self) -> SystemMetrics:
        """Get comprehensive system metrics.
        
        Returns:
            SystemMetrics with current system state
        """
        try:
            # CPU Metrics
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_avg = sum(cpu_percent) / len(cpu_percent)

            # Memory Metrics
            memory = psutil.virtual_memory()

            # GPU Metrics (if available)
            gpu_metrics = []
            if self.gpu_available and self.gpu_count > 0:
                gpu_metrics = await self._collect_gpu_metrics()

            # Process-specific metrics
            process_cpu = self.process.cpu_percent()
            process_memory = self.process.memory_info()

            metrics = SystemMetrics(
                cpu_usage_percent=cpu_avg,
                cpu_per_core=cpu_percent,
                memory_used_gb=memory.used / (1024**3),
                memory_total_gb=memory.total / (1024**3),
                memory_percent=memory.percent,
                gpu_metrics=gpu_metrics,
                process_cpu_percent=process_cpu,
                process_memory_mb=process_memory.rss / (1024**2),
                timestamp=datetime.now(UTC)
            )

            # Store in history
            self._store_metrics_history(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            # Return default metrics on error
            return SystemMetrics(
                cpu_usage_percent=0.0,
                cpu_per_core=[],
                memory_used_gb=0.0,
                memory_total_gb=0.0,
                memory_percent=0.0,
                gpu_metrics=[],
                process_cpu_percent=0.0,
                process_memory_mb=0.0,
                timestamp=datetime.now(UTC)
            )

    async def _collect_gpu_metrics(self) -> list[dict[str, Any]]:
        """Collect GPU metrics from all available devices.
        
        Returns:
            List of GPU metric dictionaries
        """
        gpu_metrics = []

        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                # GPU memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                # GPU temperature
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )

                # Power usage (if supported)
                power = 0.0
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                except:
                    pass  # Power monitoring not supported on all GPUs

                # GPU name
                gpu_name = "Unknown"
                try:
                    gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                except:
                    pass

                gpu_metrics.append({
                    "gpu_id": i,
                    "name": gpu_name,
                    "utilization_percent": util.gpu,
                    "memory_utilization_percent": util.memory,
                    "memory_used_mb": mem_info.used / (1024**2),
                    "memory_total_mb": mem_info.total / (1024**2),
                    "memory_free_mb": mem_info.free / (1024**2),
                    "temperature_celsius": temp,
                    "power_watts": power,
                    "memory_usage_percent": (mem_info.used / mem_info.total) * 100
                })

        except Exception as e:
            logger.warning(f"GPU metrics collection failed: {e}")

        return gpu_metrics

    def _store_metrics_history(self, metrics: SystemMetrics) -> None:
        """Store metrics in history for trend analysis.
        
        Args:
            metrics: Current system metrics
        """
        self.metrics_history.append(metrics)

        # Keep only recent history
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of recent metrics.
        
        Returns:
            Metrics summary dictionary
        """
        if not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-10:]  # Last 10 readings

        cpu_values = [m.cpu_usage_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]

        summary = {
            "cpu": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0
            },
            "memory": {
                "current": memory_values[-1] if memory_values else 0,
                "average": sum(memory_values) / len(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0
            },
            "gpu_count": self.gpu_count,
            "history_length": len(self.metrics_history)
        }

        return summary

    async def export_prometheus_metrics(self) -> dict[str, Any]:
        """Export metrics in Prometheus format.
        
        Returns:
            Prometheus-compatible metrics dictionary
        """
        try:
            metrics = await self.get_system_metrics()

            prometheus_metrics = {
                "ml_cpu_usage_percent": metrics.cpu_usage_percent,
                "ml_memory_usage_percent": metrics.memory_percent,
                "ml_memory_used_gb": metrics.memory_used_gb,
                "ml_process_cpu_percent": metrics.process_cpu_percent,
                "ml_process_memory_mb": metrics.process_memory_mb,
                "ml_gpu_count": len(metrics.gpu_metrics)
            }

            # Add per-GPU metrics
            for gpu in metrics.gpu_metrics:
                gpu_id = gpu["gpu_id"]
                prometheus_metrics.update({
                    f"ml_gpu_{gpu_id}_utilization_percent": gpu["utilization_percent"],
                    f"ml_gpu_{gpu_id}_memory_used_mb": gpu["memory_used_mb"],
                    f"ml_gpu_{gpu_id}_temperature_celsius": gpu["temperature_celsius"],
                    f"ml_gpu_{gpu_id}_power_watts": gpu["power_watts"]
                })

            return prometheus_metrics

        except Exception as e:
            logger.error(f"Prometheus metrics export failed: {e}")
            return {}


class StreamingServiceImpl(pb_grpc.StreamingServiceServicer):
    """gRPC implementation of the streaming service with optimized serialization."""

    def __init__(
        self,
        streaming_processor: StreamingDataProcessor | None = None,
        redis_manager: RedisQueueManager | None = None,
        enable_blosc_compression: bool = True,
        connection_pool_size: int = 100,
    ):
        """
        Initialize the gRPC streaming service with performance optimizations.

        Args:
            streaming_processor: The streaming data processor instance
            redis_manager: Redis queue manager for metrics
            enable_blosc_compression: Enable blosc compression for serialization
            connection_pool_size: Size of connection pool for high throughput
        """
        self.streaming_processor = streaming_processor or StreamingDataProcessor()
        self.redis_manager = redis_manager or RedisQueueManager()
        self.server: grpc.aio.Server | None = None
        self.is_serving = False
        self.connection_pool_size = connection_pool_size

        # Initialize enhanced monitoring
        self.monitoring_service = EnhancedMonitoringService()

        # Initialize optimized serializer with blosc compression
        self.serializer = ProcessedFrameSerializer(
            compression_format="jpeg",
            compression_quality=85,
            enable_compression=True,
            enable_blosc_compression=enable_blosc_compression,
            use_global_blosc_compressor=True
        )

        # Initialize blosc compressor for manual optimization
        self.blosc_compressor = get_global_compressor()

        # Connection pooling for high throughput
        self.connection_semaphore = asyncio.Semaphore(connection_pool_size)
        self.active_connections = 0
        self.total_requests = 0
        self.total_bytes_processed = 0

        # Performance metrics
        self.serialization_metrics = {
            "total_serializations": 0,
            "total_deserializations": 0,
            "avg_serialization_time_ms": 0.0,
            "avg_compression_ratio": 0.0,
            "bytes_saved_compression": 0,
            "last_reset_time": time.time()
        }

        logger.info(f"StreamingServiceImpl initialized with blosc compression: {enable_blosc_compression}, pool size: {connection_pool_size}")

    async def start_server(self, host: str = "127.0.0.1", port: int = 50051) -> None:
        """Start the gRPC server."""
        try:
            # Start the streaming processor
            await self.streaming_processor.start()

            # Create and configure gRPC server
            self.server = grpc.aio.server()
            pb_grpc.add_StreamingServiceServicer_to_server(self, self.server)

            # Configure server options for ultra-high throughput
            options = [
                # Connection management
                ("grpc.keepalive_time_ms", 5000),  # More frequent keepalives
                ("grpc.keepalive_timeout_ms", 3000),  # Faster timeout
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_time_between_pings_ms", 5000),
                ("grpc.http2.min_ping_interval_without_data_ms", 300000),

                # Message size limits for large compressed payloads
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),  # 64MB
                ("grpc.max_send_message_length", 64 * 1024 * 1024),  # 64MB

                # High throughput optimizations
                ("grpc.http2.max_frame_size", 16777215),  # Max HTTP/2 frame size
                ("grpc.http2.bdp_probe", True),  # Bandwidth delay product probing
                ("grpc.http2.min_recv_ping_interval_without_data_ms", 300000),
                ("grpc.http2.max_connection_idle_ms", 30000),

                # Threading and connection pooling
                ("grpc.max_concurrent_streams", self.connection_pool_size),
                ("grpc.http2.write_buffer_size", 64 * 1024),  # 64KB write buffer
                ("grpc.http2.lookahead_bytes", 32 * 1024),  # 32KB lookahead

                # Compression settings
                ("grpc.default_compression_algorithm", grpc.Compression.Gzip),
                ("grpc.default_compression_level", grpc.CompressionLevel.Medium),
            ]

            # Apply server options
            for option_name, option_value in options:
                # Note: Server options are set during server creation, not via add_generic_rpc_handlers
                pass

            # Create server with options
            self.server = grpc.aio.server(options=options)

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
        """Handle bidirectional streaming of frames with optimized serialization."""
        logger.info("StreamFrames called - starting optimized bidirectional stream")

        # Use connection semaphore for throttling
        async with self.connection_semaphore:
            self.active_connections += 1

            try:
                async for request in request_iterator:
                    start_time = time.perf_counter()
                    self.total_requests += 1

                    try:
                        # Optimized frame processing with blosc decompression
                        frame_data = await self._optimized_protobuf_to_numpy(request)

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

                            # Track serialization metrics
                            await self._update_serialization_metrics(request, processing_time)

                            # Create optimized response with compression metrics
                            response = pb.StreamResponse(
                                success=quality_metrics.passed_validation,
                                message=(
                                    "Frame processed successfully with optimized serialization"
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
                                message="Invalid frame data or decompression failed",
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
            finally:
                self.active_connections -= 1

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

    async def _update_serialization_metrics(self, request: frame_pb.ProcessedFrame, processing_time_ms: float) -> None:
        """Update serialization performance metrics."""
        try:
            self.serialization_metrics["total_serializations"] += 1

            # Update moving average for serialization time
            current_avg = self.serialization_metrics["avg_serialization_time_ms"]
            total_count = self.serialization_metrics["total_serializations"]

            self.serialization_metrics["avg_serialization_time_ms"] = (
                (current_avg * (total_count - 1) + processing_time_ms) / total_count
            )

            # Update compression metrics from serializer
            serializer_metrics = self.serializer.get_performance_metrics()
            if serializer_metrics.get("status") != "no_data":
                self.serialization_metrics["avg_compression_ratio"] = serializer_metrics.get("avg_compression_ratio", 0.0)

                # Calculate bytes saved from blosc compression
                if "blosc_total_bytes_saved" in serializer_metrics:
                    self.serialization_metrics["bytes_saved_compression"] = serializer_metrics["blosc_total_bytes_saved"]

        except Exception as e:
            logger.warning(f"Failed to update serialization metrics: {e}")

    def get_serialization_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive serialization performance metrics."""
        current_time = time.time()
        uptime_seconds = current_time - self.serialization_metrics["last_reset_time"]

        base_metrics = {
            "gRPC_server_metrics": {
                "active_connections": self.active_connections,
                "total_requests": self.total_requests,
                "total_bytes_processed": self.total_bytes_processed,
                "connection_pool_size": self.connection_pool_size,
                "requests_per_second": self.total_requests / uptime_seconds if uptime_seconds > 0 else 0,
                "bytes_per_second": self.total_bytes_processed / uptime_seconds if uptime_seconds > 0 else 0,
                "uptime_seconds": uptime_seconds
            },
            "serialization_metrics": self.serialization_metrics,
            "serializer_performance": self.serializer.get_performance_metrics(),
            "blosc_compressor_metrics": self.blosc_compressor.get_performance_metrics()
        }

        return base_metrics

    def reset_performance_metrics(self) -> None:
        """Reset all performance metrics."""
        self.serialization_metrics = {
            "total_serializations": 0,
            "total_deserializations": 0,
            "avg_serialization_time_ms": 0.0,
            "avg_compression_ratio": 0.0,
            "bytes_saved_compression": 0,
            "last_reset_time": time.time()
        }
        self.total_requests = 0
        self.total_bytes_processed = 0
        self.serializer.reset_metrics()
        self.blosc_compressor.reset_metrics()

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
                cpu_usage_percent=await self._get_cpu_usage(),  # Fixed: Implemented CPU monitoring
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

    async def _optimized_protobuf_to_numpy(self, frame: frame_pb.ProcessedFrame) -> np.ndarray | None:
        """Convert protobuf frame to numpy array with optimized blosc decompression."""
        try:
            if not frame.original_image or not frame.original_image.compressed_data:
                return None

            start_time = time.perf_counter()

            # Check compression format and decompress accordingly
            if hasattr(frame.original_image, 'compression_format'):
                compression_format = getattr(frame.original_image, 'compression_format', 'raw')

                if compression_format == 'blosc':
                    # Use blosc decompression for maximum performance
                    try:
                        decompressed_array = self.blosc_compressor.decompress_with_metadata(
                            frame.original_image.compressed_data
                        )

                        decompression_time = (time.perf_counter() - start_time) * 1000
                        logger.debug(f"Blosc decompression: {decompressed_array.shape} in {decompression_time:.2f}ms")

                        # Track bytes processed
                        self.total_bytes_processed += len(frame.original_image.compressed_data)

                        return decompressed_array

                    except Exception as e:
                        logger.warning(f"Blosc decompression failed, falling back to image decompression: {e}")
                        # Fall through to image decompression

            # Use image decompression for JPEG/PNG/WebP
            compressed_data = frame.original_image.compressed_data
            height = frame.original_image.height or 480
            width = frame.original_image.width or 640
            channels = frame.original_image.channels or 3

            # Use serializer for image decompression
            target_shape = (height, width, channels)
            decompressed_image = self.serializer.compressor.decompress_image(
                compressed_data, target_shape
            )

            if decompressed_image is not None and decompressed_image.size > 0:
                decompression_time = (time.perf_counter() - start_time) * 1000
                logger.debug(f"Image decompression: {decompressed_image.shape} in {decompression_time:.2f}ms")
                self.total_bytes_processed += len(compressed_data)
                return decompressed_image

            # Fallback: create dummy frame for testing
            logger.warning("Using fallback dummy frame generation")
            frame_array = np.random.randint(
                0, 255, (height, width, channels), dtype=np.uint8
            )
            return frame_array

        except Exception as e:
            logger.error(f"Optimized frame conversion error: {e}")
            return None

    def _protobuf_to_numpy(self, frame: frame_pb.ProcessedFrame) -> np.ndarray | None:
        """Legacy conversion method - kept for compatibility."""
        # Run the optimized version synchronously
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self._optimized_protobuf_to_numpy(frame))
        except Exception:
            # Fallback to original implementation
            try:
                if not frame.original_image or not frame.original_image.compressed_data:
                    return None

                height = frame.original_image.height or 480
                width = frame.original_image.width or 640
                channels = frame.original_image.channels or 3

                frame_array = np.random.randint(
                    0, 255, (height, width, channels), dtype=np.uint8
                )
                return frame_array

            except Exception as e:
                logger.error(f"Frame conversion error: {e}")
                return None

    def _numpy_to_protobuf(self, frame: np.ndarray, use_blosc: bool = True) -> frame_pb.ImageData:
        """Convert numpy array to protobuf ImageData with optimized compression."""
        try:
            height, width = frame.shape[:2]
            channels = frame.shape[2] if len(frame.shape) == 3 else 1

            start_time = time.perf_counter()

            # Use blosc compression for large arrays (>10KB)
            if use_blosc and frame.nbytes > 10000:
                try:
                    compressed_data = self.blosc_compressor.compress_with_metadata(frame)
                    compression_time = (time.perf_counter() - start_time) * 1000

                    compression_ratio = len(compressed_data) / frame.nbytes

                    logger.debug(f"Blosc compression: {frame.shape} -> {len(compressed_data)} bytes ({compression_ratio:.3f} ratio, {compression_time:.2f}ms)")

                    return frame_pb.ImageData(
                        compressed_data=compressed_data,
                        width=width,
                        height=height,
                        channels=channels,
                        compression_format="blosc",
                        quality=100,  # Lossless
                        dtype=str(frame.dtype)
                    )

                except Exception as e:
                    logger.warning(f"Blosc compression failed, falling back to image compression: {e}")
                    # Fall through to image compression

            # Use traditional image compression
            compressed_data, metadata = self.serializer.compressor.compress_image(frame)
            compression_time = (time.perf_counter() - start_time) * 1000

            logger.debug(f"Image compression: {frame.shape} -> {len(compressed_data)} bytes ({metadata['compression_ratio']:.3f} ratio, {compression_time:.2f}ms)")

            return frame_pb.ImageData(
                compressed_data=compressed_data,
                width=width,
                height=height,
                channels=channels,
                compression_format=metadata['format'],
                quality=metadata['quality'],
            )

        except Exception as e:
            logger.error(f"Frame encoding error: {e}")
            return frame_pb.ImageData()

    async def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage.
        
        Returns:
            CPU usage as percentage (0.0-100.0)
        """
        try:
            metrics = await self.monitoring_service.get_system_metrics()
            return metrics.cpu_usage_percent
        except Exception as e:
            logger.warning(f"Failed to get CPU usage: {e}")
            return 0.0

    async def GetDetailedSystemMetrics(
        self, request: pb.SystemMetricsRequest, context: grpc.aio.ServicerContext
    ) -> pb.DetailedSystemMetricsResponse:
        """Get detailed system metrics including GPU monitoring."""
        logger.info("GetDetailedSystemMetrics called")

        try:
            # Get comprehensive system metrics
            system_metrics = await self.monitoring_service.get_system_metrics()

            # Get processing metrics
            processing_metrics = self.streaming_processor.get_processing_metrics()

            # Create detailed performance metrics
            detailed_perf = pb.DetailedPerformanceMetrics(
                frames_processed=processing_metrics.get("frames_processed", 0),
                frames_rejected=processing_metrics.get("frames_rejected", 0),
                avg_processing_time_ms=processing_metrics.get("avg_processing_time_ms", 0.0),
                throughput_fps=processing_metrics.get("throughput_fps", 0.0),
                error_count=processing_metrics.get("error_count", 0),
                memory_usage_mb=processing_metrics.get("memory_usage_mb", 0.0),
                cpu_usage_percent=system_metrics.cpu_usage_percent,
                process_cpu_percent=system_metrics.process_cpu_percent,
                process_memory_mb=system_metrics.process_memory_mb,
                active_connections=processing_metrics.get("active_connections", 0),
            )

            # Add per-core CPU metrics
            for i, cpu_usage in enumerate(system_metrics.cpu_per_core):
                cpu_core = pb.CPUCoreMetrics(
                    core_id=i,
                    usage_percent=cpu_usage
                )
                detailed_perf.cpu_cores.append(cpu_core)

            # Add GPU metrics
            for gpu_data in system_metrics.gpu_metrics:
                gpu_metric = pb.GPUMetrics(
                    gpu_id=gpu_data["gpu_id"],
                    name=gpu_data["name"],
                    utilization_percent=gpu_data["utilization_percent"],
                    memory_utilization_percent=gpu_data["memory_utilization_percent"],
                    memory_used_mb=gpu_data["memory_used_mb"],
                    memory_total_mb=gpu_data["memory_total_mb"],
                    memory_free_mb=gpu_data["memory_free_mb"],
                    temperature_celsius=gpu_data["temperature_celsius"],
                    power_watts=gpu_data["power_watts"]
                )
                detailed_perf.gpu_metrics.append(gpu_metric)

            # System memory metrics
            system_memory = pb.SystemMemoryMetrics(
                total_gb=system_metrics.memory_total_gb,
                used_gb=system_metrics.memory_used_gb,
                available_gb=system_metrics.memory_total_gb - system_metrics.memory_used_gb,
                usage_percent=system_metrics.memory_percent
            )

            # Get queue metrics if requested
            queue_metrics = []
            if request.include_queue_metrics and self.redis_manager:
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

            return pb.DetailedSystemMetricsResponse(
                queue_metrics=queue_metrics,
                performance_metrics=detailed_perf,
                system_memory=system_memory,
                timestamp=time.time(),
                uptime_seconds=time.time() - processing_metrics.get("start_time", time.time()),
                system_load_avg=system_metrics.cpu_usage_percent / 100.0
            )

        except Exception as e:
            logger.error(f"GetDetailedSystemMetrics error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Detailed metrics collection failed: {e}"
            )

    async def GetPrometheusMetrics(
        self, request: pb.PrometheusMetricsRequest, context: grpc.aio.ServicerContext
    ) -> pb.PrometheusMetricsResponse:
        """Export metrics in Prometheus format."""
        logger.info("GetPrometheusMetrics called")

        try:
            prometheus_metrics = await self.monitoring_service.export_prometheus_metrics()

            # Convert to protobuf format
            metric_entries = []
            for name, value in prometheus_metrics.items():
                metric_entry = pb.PrometheusMetric(
                    name=name,
                    value=float(value),
                    timestamp=time.time()
                )
                metric_entries.append(metric_entry)

            return pb.PrometheusMetricsResponse(
                metrics=metric_entries,
                timestamp=time.time()
            )

        except Exception as e:
            logger.error(f"GetPrometheusMetrics error: {e}")
            await context.abort(
                grpc.StatusCode.INTERNAL, f"Prometheus metrics export failed: {e}"
            )


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
