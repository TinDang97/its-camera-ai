"""Vision Core gRPC Servicer Implementation.

This servicer implements the VisionCoreService using the UnifiedVisionAnalyticsEngine
for high-performance inference and analytics processing.

Features:
- Single RPC for unified inference + analytics
- Bidirectional streaming for real-time camera feeds
- Batch processing optimization
- Comprehensive error handling and monitoring
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator

import cv2
import grpc
import numpy as np
from google.protobuf.timestamp_pb2 import Timestamp

from ...core.logging import get_logger
from ...core.unified_vision_analytics_engine import (
    RequestPriority,
    UnifiedResult,
    UnifiedVisionAnalyticsEngine,
)
from ...proto import vision_core_pb2, vision_core_pb2_grpc

logger = get_logger(__name__)


class VisionCoreServicer(vision_core_pb2_grpc.VisionCoreServiceServicer):
    """gRPC servicer for unified vision and analytics processing."""

    def __init__(self, unified_engine: UnifiedVisionAnalyticsEngine):
        """Initialize servicer with unified engine."""
        self.unified_engine = unified_engine
        self.active_streams: dict[str, dict] = {}

        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.last_metrics_reset = time.time()

        logger.info("VisionCoreServicer initialized")

    async def ProcessFrame(
        self,
        request: vision_core_pb2.FrameRequest,
        context: grpc.aio.ServicerContext,
    ) -> vision_core_pb2.VisionResult:
        """Process single frame with unified inference and analytics."""

        start_time = time.time()
        self.request_count += 1

        try:
            # Validate request
            if not request.frame_data:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("frame_data is required")
                return vision_core_pb2.VisionResult()

            # Decode frame data
            frame = self._decode_frame_data(request.frame_data, request.format)
            if frame is None:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Failed to decode frame data")
                return vision_core_pb2.VisionResult()

            # Convert priority
            priority = self._convert_priority(request.priority)

            # Process frame through unified engine
            unified_result = await self.unified_engine.process_frame(
                frame=frame,
                camera_id=request.camera_id or "unknown",
                frame_id=request.frame_id,
                priority=priority,
                include_analytics=request.options.include_analytics,
                include_quality_score=request.options.include_quality_score,
                include_frame_annotation=request.options.include_frame_annotation,
                include_metadata_track=request.options.include_metadata_track,
            )

            # Convert result to protobuf
            pb_result = self._convert_unified_result_to_pb(unified_result)

            # Add timing information
            total_time = (time.time() - start_time) * 1000
            pb_result.timing.total_time_ms = total_time

            logger.debug(
                f"Processed frame {request.frame_id} from {request.camera_id} "
                f"in {total_time:.1f}ms"
            )

            return pb_result

        except Exception as e:
            self.error_count += 1
            logger.error(f"Frame processing failed: {e}")

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Processing error: {str(e)}")

            # Return error result
            error_result = vision_core_pb2.VisionResult()
            error_result.status.code = vision_core_pb2.ProcessingStatus.FAILED
            error_result.status.message = str(e)
            return error_result

    async def ProcessFrameBatch(
        self,
        request: vision_core_pb2.FrameBatchRequest,
        context: grpc.aio.ServicerContext,
    ) -> vision_core_pb2.VisionBatchResult:
        """Process batch of frames for optimal GPU utilization."""

        start_time = time.time()
        self.request_count += len(request.frames)

        try:
            if not request.frames:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("At least one frame is required")
                return vision_core_pb2.VisionBatchResult()

            # Decode all frames
            frames = []
            camera_ids = []
            frame_ids = []

            for frame_request in request.frames:
                frame = self._decode_frame_data(frame_request.frame_data, frame_request.format)
                if frame is None:
                    logger.warning(f"Failed to decode frame {frame_request.frame_id}")
                    continue

                frames.append(frame)
                camera_ids.append(frame_request.camera_id or "unknown")
                frame_ids.append(frame_request.frame_id)

            if not frames:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("No valid frames to process")
                return vision_core_pb2.VisionBatchResult()

            # Set processing timeout
            timeout = request.max_processing_time_ms / 1000.0 if request.max_processing_time_ms > 0 else 30.0

            # Process batch
            try:
                unified_results = await asyncio.wait_for(
                    self.unified_engine.process_batch(
                        frames=frames,
                        camera_ids=camera_ids,
                        frame_ids=frame_ids,
                        priority=RequestPriority.NORMAL,
                    ),
                    timeout=timeout
                )
            except TimeoutError:
                context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
                context.set_details(f"Batch processing timeout after {timeout}s")
                return vision_core_pb2.VisionBatchResult()

            # Convert results to protobuf
            batch_result = vision_core_pb2.VisionBatchResult()

            for unified_result in unified_results:
                pb_result = self._convert_unified_result_to_pb(unified_result)
                batch_result.results.append(pb_result)

            # Add batch statistics
            total_time = (time.time() - start_time) * 1000
            batch_result.batch_stats.total_frames = len(unified_results)
            batch_result.batch_stats.successful_frames = len(unified_results)
            batch_result.batch_stats.total_processing_time_ms = total_time
            batch_result.batch_stats.average_processing_time_ms = total_time / len(unified_results)

            logger.info(
                f"Processed batch of {len(unified_results)} frames in {total_time:.1f}ms"
            )

            return batch_result

        except Exception as e:
            self.error_count += len(request.frames)
            logger.error(f"Batch processing failed: {e}")

            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Batch processing error: {str(e)}")
            return vision_core_pb2.VisionBatchResult()

    async def ProcessCameraStream(
        self,
        request_iterator: AsyncIterator[vision_core_pb2.CameraStreamRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[vision_core_pb2.CameraStreamResult]:
        """Process bidirectional camera stream for real-time processing."""

        stream_id = None
        camera_id = None

        try:
            async for request in request_iterator:
                try:
                    if request.HasField("init"):
                        # Initialize stream
                        stream_id = request.init.stream_id
                        camera_id = request.init.camera_id

                        # Register stream
                        self.active_streams[stream_id] = {
                            "camera_id": camera_id,
                            "start_time": time.time(),
                            "frame_count": 0,
                            "error_count": 0,
                            "options": request.init.options,
                        }

                        # Send initialization confirmation
                        result = vision_core_pb2.CameraStreamResult()
                        result.status.stream_id = stream_id
                        result.status.camera_id = camera_id
                        result.status.state = vision_core_pb2.StreamStatus.INITIALIZED
                        result.status.message = "Stream initialized successfully"

                        yield result

                        logger.info(f"Initialized camera stream {stream_id} for {camera_id}")

                    elif request.HasField("frame"):
                        # Process frame
                        if stream_id not in self.active_streams:
                            error_result = vision_core_pb2.CameraStreamResult()
                            error_result.error.stream_id = stream_id or "unknown"
                            error_result.error.code = vision_core_pb2.StreamError.UNKNOWN_ERROR
                            error_result.error.message = "Stream not initialized"
                            yield error_result
                            continue

                        # Decode and process frame
                        frame = self._decode_frame_data(
                            request.frame.frame_data,
                            request.frame.format
                        )

                        if frame is not None:
                            stream_info = self.active_streams[stream_id]

                            # Process frame
                            unified_result = await self.unified_engine.process_frame(
                                frame=frame,
                                camera_id=request.frame.camera_id,
                                frame_id=request.frame.frame_id,
                                priority=self._convert_priority(request.frame.priority),
                                include_analytics=stream_info["options"].include_analytics,
                                include_quality_score=stream_info["options"].include_quality_score,
                                include_frame_annotation=stream_info["options"].include_frame_annotation,
                                include_metadata_track=stream_info["options"].include_metadata_track,
                            )

                            # Convert and return result
                            result = vision_core_pb2.CameraStreamResult()
                            result.vision_result.CopyFrom(
                                self._convert_unified_result_to_pb(unified_result)
                            )

                            yield result

                            # Update stream stats
                            stream_info["frame_count"] += 1

                        else:
                            # Frame decode error
                            error_result = vision_core_pb2.CameraStreamResult()
                            error_result.error.stream_id = stream_id
                            error_result.error.camera_id = camera_id or "unknown"
                            error_result.error.code = vision_core_pb2.StreamError.FORMAT_ERROR
                            error_result.error.message = "Failed to decode frame"
                            error_result.error.recoverable = True
                            yield error_result

                            self.active_streams[stream_id]["error_count"] += 1

                    elif request.HasField("control"):
                        # Handle stream control
                        if stream_id in self.active_streams:
                            control_type = request.control.type

                            if control_type == vision_core_pb2.StreamControl.STOP:
                                # Stop stream
                                del self.active_streams[stream_id]

                                result = vision_core_pb2.CameraStreamResult()
                                result.status.stream_id = stream_id
                                result.status.state = vision_core_pb2.StreamStatus.STOPPED
                                result.status.message = "Stream stopped"
                                yield result

                                logger.info(f"Stopped camera stream {stream_id}")
                                break

                            elif control_type == vision_core_pb2.StreamControl.UPDATE_OPTIONS:
                                # Update processing options
                                self.active_streams[stream_id]["options"] = request.control.updated_options

                                result = vision_core_pb2.CameraStreamResult()
                                result.status.stream_id = stream_id
                                result.status.state = vision_core_pb2.StreamStatus.RUNNING
                                result.status.message = "Options updated"
                                yield result

                except Exception as e:
                    logger.error(f"Stream processing error: {e}")

                    # Send error response
                    error_result = vision_core_pb2.CameraStreamResult()
                    error_result.error.stream_id = stream_id or "unknown"
                    error_result.error.camera_id = camera_id or "unknown"
                    error_result.error.code = vision_core_pb2.StreamError.PROCESSING_ERROR
                    error_result.error.message = str(e)
                    error_result.error.recoverable = True
                    yield error_result

        finally:
            # Cleanup stream on disconnect
            if stream_id and stream_id in self.active_streams:
                logger.info(f"Cleaning up disconnected stream {stream_id}")
                del self.active_streams[stream_id]

    async def GetEngineHealth(
        self,
        request: vision_core_pb2.HealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> vision_core_pb2.HealthResponse:
        """Get engine health status and metrics."""

        try:
            # Get health from unified engine
            health_data = await self.unified_engine.health_check()

            response = vision_core_pb2.HealthResponse()

            # Convert status
            if health_data["status"] == "healthy":
                response.status = vision_core_pb2.HealthResponse.HEALTHY
            elif health_data["status"] == "degraded":
                response.status = vision_core_pb2.HealthResponse.DEGRADED
            else:
                response.status = vision_core_pb2.HealthResponse.UNHEALTHY

            response.message = f"Engine state: {health_data.get('state', 'unknown')}"

            # Add system metrics
            if "metrics" in health_data:
                metrics = health_data["metrics"]
                response.system_metrics.avg_processing_time_ms = metrics.get("avg_queue_time_ms", 0)
                response.system_metrics.total_queue_depth = health_data.get("queue_depth", 0)
                response.system_metrics.requests_per_second = self._calculate_rps()
                response.system_metrics.error_rate_percent = self._calculate_error_rate()
                response.system_metrics.total_requests = self.request_count
                response.system_metrics.failed_requests = self.error_count

            # Add component health
            if "devices" in health_data:
                for device_name, device_info in health_data["devices"].items():
                    component_health = vision_core_pb2.ComponentHealth()
                    component_health.status = vision_core_pb2.HealthResponse.HEALTHY
                    component_health.message = f"Load: {device_info.get('load', 0):.2f}"
                    response.components[device_name] = component_health

            return response

        except Exception as e:
            logger.error(f"Health check failed: {e}")

            response = vision_core_pb2.HealthResponse()
            response.status = vision_core_pb2.HealthResponse.ERROR
            response.message = f"Health check failed: {str(e)}"
            return response

    async def GetEngineMetrics(
        self,
        request: vision_core_pb2.MetricsRequest,
        context: grpc.aio.ServicerContext,
    ) -> vision_core_pb2.MetricsResponse:
        """Get comprehensive engine performance metrics."""

        try:
            # Get metrics from unified engine (now async)
            engine_metrics = await self.unified_engine.get_comprehensive_metrics()

            response = vision_core_pb2.MetricsResponse()

            # Current metrics
            if "unified_engine" in engine_metrics:
                ue_metrics = engine_metrics["unified_engine"]

                response.current_metrics.requests_per_second = self._calculate_rps()
                response.current_metrics.total_queue_depth = ue_metrics.get("avg_queue_time_ms", 0)
                response.current_metrics.avg_queue_time_ms = ue_metrics.get("avg_queue_time_ms", 0)
                response.current_metrics.error_rate_percent = self._calculate_error_rate()
                response.current_metrics.total_requests = self.request_count
                response.current_requests = self.error_count

            # Engine-specific metrics
            for engine_name, metrics in engine_metrics.items():
                if isinstance(metrics, dict):
                    engine_metric = vision_core_pb2.EngineMetrics()
                    engine_metric.engine_name = engine_name
                    engine_metric.engine_version = "1.0.0"

                    # Convert metrics to custom_metrics map
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            engine_metric.custom_metrics[key] = float(value)

                    response.engine_metrics[engine_name] = engine_metric

            return response

        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Metrics error: {str(e)}")
            return vision_core_pb2.MetricsResponse()

    def _decode_frame_data(
        self,
        frame_data: bytes,
        format_info: vision_core_pb2.FrameFormat
    ) -> np.ndarray | None:
        """Decode frame data based on format."""

        try:
            if format_info.format == vision_core_pb2.FrameFormat.JPEG:
                # Decode JPEG
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            elif format_info.format == vision_core_pb2.FrameFormat.PNG:
                # Decode PNG
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            elif format_info.format == vision_core_pb2.FrameFormat.RAW_RGB:
                # Raw RGB data
                frame = np.frombuffer(frame_data, np.uint8)
                return frame.reshape(format_info.height, format_info.width, 3)

            elif format_info.format == vision_core_pb2.FrameFormat.RAW_BGR:
                # Raw BGR data
                frame = np.frombuffer(frame_data, np.uint8)
                frame = frame.reshape(format_info.height, format_info.width, 3)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            else:
                logger.error(f"Unsupported frame format: {format_info.format}")
                return None

        except Exception as e:
            logger.error(f"Frame decoding failed: {e}")
            return None

    def _convert_priority(self, priority: int) -> RequestPriority:
        """Convert gRPC priority to internal priority enum."""
        if priority == 2:
            return RequestPriority.EMERGENCY
        elif priority == 1:
            return RequestPriority.HIGH
        else:
            return RequestPriority.NORMAL

    def _convert_unified_result_to_pb(self, unified_result: UnifiedResult) -> vision_core_pb2.VisionResult:
        """Convert UnifiedResult to protobuf VisionResult."""

        pb_result = vision_core_pb2.VisionResult()

        # Basic information
        pb_result.frame_id = unified_result.frame_id
        pb_result.camera_id = unified_result.camera_id

        # Timestamp
        timestamp = Timestamp()
        timestamp.FromDatetime(unified_result.timestamp)
        pb_result.timestamp.CopyFrom(timestamp)

        # Timing information
        pb_result.timing.inference_time_ms = unified_result.inference_time_ms
        pb_result.timing.analytics_time_ms = unified_result.analytics_time_ms
        pb_result.timing.total_time_ms = unified_result.total_processing_time_ms
        pb_result.timing.batch_size = unified_result.batch_size

        # Inference results
        pb_result.inference.model_name = "yolo11"
        pb_result.inference.model_version = unified_result.model_version

        # Convert detections
        for detection_dto in unified_result.detections:
            detection = vision_core_pb2.Detection()

            # Bounding box
            detection.bbox.x_min = detection_dto.x_min
            detection.bbox.y_min = detection_dto.y_min
            detection.bbox.x_max = detection_dto.x_max
            detection.bbox.y_max = detection_dto.y_max
            detection.bbox.bbox_confidence = detection_dto.confidence

            # Classification
            detection.class_name = detection_dto.class_name
            detection.class_id = detection_dto.class_id
            detection.confidence = detection_dto.confidence

            # Tracking
            if detection_dto.tracking_id:
                detection.tracking_id = detection_dto.tracking_id
                detection.tracking_state = vision_core_pb2.TrackingState.TRACKED

            pb_result.inference.detections.append(detection)

        # Analytics results
        if unified_result.analytics_result:
            analytics = pb_result.analytics

            # Traffic metrics
            analytics.traffic.total_vehicle_count = len([
                d for d in unified_result.detections if d.is_vehicle
            ])

            # TODO: Add more analytics conversion based on analytics_result structure

        # Quality score
        if unified_result.quality_score is not None:
            pb_result.quality.overall_score = unified_result.quality_score

        # Metadata track
        if unified_result.metadata_track:
            pb_result.metadata_track.metadata_json = json.dumps(unified_result.metadata_track)
            pb_result.metadata_track.metadata_type = "detection_analytics"

            timestamp = Timestamp()
            timestamp.FromDatetime(unified_result.timestamp)
            pb_result.metadata_track.timestamp.CopyFrom(timestamp)

        # Annotated frame
        if unified_result.annotated_frame is not None:
            # Encode annotated frame as JPEG
            _, encoded_frame = cv2.imencode('.jpg', unified_result.annotated_frame)
            pb_result.annotated_frame = encoded_frame.tobytes()

        # Processing status
        pb_result.status.code = vision_core_pb2.ProcessingStatus.SUCCESS
        pb_result.status.message = "Processing completed successfully"

        return pb_result

    def _calculate_rps(self) -> float:
        """Calculate requests per second."""
        uptime = time.time() - self.last_metrics_reset
        return self.request_count / max(1, uptime)

    def _calculate_error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100
