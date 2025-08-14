"""ML Analytics Connector Service for ITS Camera AI System.

This service provides the critical data pipeline connecting BatchProcessor output
to UnifiedAnalyticsService, ensuring <100ms end-to-end latency while maintaining
high throughput and reliability.

Key Features:
- Asynchronous batch processing with backpressure handling
- Real-time Redis pub/sub for analytics subscribers
- ML output to DetectionResultDTO conversion
- Camera-based aggregation for efficient processing
- Comprehensive error handling and timeout management
- Performance monitoring and metrics tracking

Performance Targets:
- End-to-end latency: <100ms (ML processing + analytics)
- Throughput: 30+ FPS per camera stream
- Batch efficiency: 95%+ GPU memory utilization
- Monitoring overhead: <2% CPU impact
"""

import asyncio
import contextlib
import json
import time
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as redis
from dependency_injector.wiring import Provide

from ..core.config import Settings
from ..core.logging import get_logger
from ..ml.batch_processor import SmartBatchProcessor
from ..services.analytics_dtos import (
    DetectionData,
    DetectionResultConverter,
    DetectionResultDTO,
    FrameMetadataDTO,
)
from ..services.cache import CacheService
from ..services.unified_analytics_service import UnifiedAnalyticsService

logger = get_logger(__name__)


class MLAnalyticsConnector:
    """Connects ML BatchProcessor output to UnifiedAnalyticsService.

    Handles the critical data flow from ML inference to analytics processing
    with optimized performance and reliability guarantees.
    """

    def __init__(
        self,
        batch_processor: SmartBatchProcessor = Provide["services.batch_processor"],
        unified_analytics: UnifiedAnalyticsService = Provide[
            "services.unified_analytics_service"
        ],
        redis_client: redis.Redis = Provide["infrastructure.redis_client"],
        cache_service: CacheService = Provide["services.cache_service"],
        settings: Settings = Provide["config"],
    ):
        """Initialize ML Analytics Connector with dependency injection.

        Args:
            batch_processor: ML batch processing service
            unified_analytics: Unified analytics service
            redis_client: Redis client for pub/sub
            cache_service: Cache service for performance optimization
            settings: Application settings
        """
        self.batch_processor = batch_processor
        self.unified_analytics = unified_analytics
        self.redis_pub = redis_client
        self.cache_service = cache_service
        self.settings = settings

        # Performance optimization
        self.batch_queue = asyncio.Queue(maxsize=1000)  # Backpressure control
        self.processing_timeout_ms = 100  # 100ms total timeout
        self.ml_timeout_ms = 80  # Reserve 80ms for ML processing
        self.analytics_timeout_ms = 20  # Reserve 20ms for analytics

        # Metrics tracking
        self.metrics = {
            "batches_processed": 0,
            "total_detections": 0,
            "timeouts": 0,
            "errors": 0,
            "avg_latency_ms": 0.0,
            "last_reset": time.time(),
        }

        # Processing state
        self.is_running = False
        self.processing_task: asyncio.Task | None = None

        logger.info("MLAnalyticsConnector initialized with dependency injection")

    async def start(self) -> None:
        """Start the ML analytics connector service."""
        if self.is_running:
            logger.warning("MLAnalyticsConnector already running")
            return

        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_queue())
        logger.info("MLAnalyticsConnector started")

    async def stop(self) -> None:
        """Stop the ML analytics connector service."""
        self.is_running = False

        if self.processing_task:
            self.processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.processing_task

        logger.info("MLAnalyticsConnector stopped")

    async def process_ml_batch(
        self, batch_results: list[dict[str, Any]], frame_metadata: dict[str, Any]
    ) -> None:
        """Process ML batch results and send to analytics.

        This is the main entry point for ML batch processing results.
        Converts ML outputs to DTOs and processes through analytics pipeline.

        Args:
            batch_results: List of ML model detection outputs
            frame_metadata: Metadata about the processed frames

        Requirements:
        - Convert ML output to DetectionResultDTO
        - Aggregate results by camera_id
        - Maintain <100ms latency
        - Handle backpressure
        """
        start_time = time.time()

        try:
            # Enqueue for processing with timeout
            processing_request = {
                "batch_results": batch_results,
                "frame_metadata": frame_metadata,
                "timestamp": start_time,
            }

            try:
                await asyncio.wait_for(
                    self.batch_queue.put(processing_request),
                    timeout=0.01,  # 10ms max wait for queue
                )
            except TimeoutError:
                self.metrics["errors"] += 1
                logger.warning("ML batch queue full, dropping request")
                return

        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Failed to queue ML batch: {e}")

    async def _process_queue(self) -> None:
        """Process queued ML batch requests continuously."""
        logger.info("Started ML analytics queue processor")

        while self.is_running:
            try:
                # Get batch request with timeout
                try:
                    request = await asyncio.wait_for(
                        self.batch_queue.get(), timeout=1.0
                    )
                except TimeoutError:
                    continue

                # Process the batch request
                await self._process_batch_request(request)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

        logger.info("Stopped ML analytics queue processor")

    async def _process_batch_request(self, request: dict[str, Any]) -> None:
        """Process a single batch request with timeout management."""
        start_time = time.time()
        request_age_ms = (start_time - request["timestamp"]) * 1000

        # Check if request has already aged beyond acceptable limits
        if request_age_ms > self.processing_timeout_ms:
            self.metrics["timeouts"] += 1
            logger.warning(f"Dropping aged request: {request_age_ms:.1f}ms old")
            return

        try:
            # Extract data
            batch_results = request["batch_results"]
            frame_metadata = request["frame_metadata"]

            # Step 1: Convert ML outputs to DTOs (fast operation)
            detection_results = await self._convert_ml_outputs(
                batch_results, frame_metadata
            )

            # Step 2: Group by camera for efficient processing
            camera_groups = self._group_by_camera(detection_results)

            # Step 3: Process each camera's detections with timeout
            remaining_time_ms = self.processing_timeout_ms - request_age_ms
            if remaining_time_ms <= 0:
                self.metrics["timeouts"] += 1
                return

            await self._process_camera_groups(
                camera_groups, frame_metadata, remaining_time_ms
            )

            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_metrics(len(detection_results), processing_time_ms)

        except TimeoutError:
            self.metrics["timeouts"] += 1
            logger.warning("ML batch processing timeout exceeded")
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"ML batch processing failed: {e}")

    async def _convert_ml_outputs(
        self, batch_results: list[dict[str, Any]], frame_metadata: dict[str, Any]
    ) -> list[DetectionResultDTO]:
        """Convert ML model outputs to DetectionResultDTO objects.

        Args:
            batch_results: Raw ML detection outputs
            frame_metadata: Frame processing metadata

        Returns:
            List of converted DetectionResultDTO objects
        """
        detection_results = []

        try:
            # Create frame metadata DTO
            frame_meta_dto = FrameMetadataDTO(
                frame_id=frame_metadata.get("frame_id", str(time.time())),
                camera_id=frame_metadata.get("camera_id", "unknown"),
                timestamp=datetime.now(UTC),
                frame_number=frame_metadata.get("frame_number", 0),
                width=frame_metadata.get("width", 1920),
                height=frame_metadata.get("height", 1080),
                quality_score=frame_metadata.get("quality_score"),
                model_version=frame_metadata.get("model_version"),
                processing_time_ms=frame_metadata.get("processing_time_ms"),
            )

            # Convert each ML detection result
            for ml_result in batch_results:
                # Handle different ML output formats
                if "detections" in ml_result:
                    # Batch format with multiple detections
                    for detection in ml_result["detections"]:
                        dto = DetectionResultConverter.from_ml_output(
                            detection,
                            frame_meta_dto,
                            frame_metadata.get("model_version"),
                        )
                        detection_results.append(dto)
                else:
                    # Single detection format
                    dto = DetectionResultConverter.from_ml_output(
                        ml_result, frame_meta_dto, frame_metadata.get("model_version")
                    )
                    detection_results.append(dto)

        except Exception as e:
            logger.error(f"ML output conversion failed: {e}")
            # Return empty list to avoid blocking pipeline

        return detection_results

    def _group_by_camera(
        self, detection_results: list[DetectionResultDTO]
    ) -> dict[str, list[DetectionResultDTO]]:
        """Group detection results by camera ID for batch processing.

        Args:
            detection_results: List of detection results

        Returns:
            Dictionary mapping camera_id to detection lists
        """
        camera_groups: dict[str, list[DetectionResultDTO]] = {}

        for detection in detection_results:
            camera_id = detection.camera_id
            if camera_id not in camera_groups:
                camera_groups[camera_id] = []
            camera_groups[camera_id].append(detection)

        return camera_groups

    async def _process_camera_groups(
        self,
        camera_groups: dict[str, list[DetectionResultDTO]],
        frame_metadata: dict[str, Any],
        timeout_ms: float,
    ) -> None:
        """Process detection groups by camera with parallel execution.

        Args:
            camera_groups: Detection results grouped by camera
            frame_metadata: Original frame metadata
            timeout_ms: Remaining time budget in milliseconds
        """
        if not camera_groups:
            return

        # Create processing tasks for parallel execution
        tasks = []
        for camera_id, detections in camera_groups.items():
            task = self._process_camera_detections(
                camera_id, detections, frame_metadata
            )
            tasks.append(task)

        try:
            # Execute all camera processing in parallel with timeout
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout_ms / 1000.0,
            )
        except TimeoutError:
            logger.warning(f"Camera group processing timeout after {timeout_ms}ms")
            raise

    async def _process_camera_detections(
        self,
        camera_id: str,
        detections: list[DetectionResultDTO],
        frame_metadata: dict[str, Any],
    ) -> None:
        """Process detections for a single camera.

        Args:
            camera_id: Camera identifier
            detections: Detection results for this camera
            frame_metadata: Frame metadata
        """
        try:
            # Create DetectionData object for analytics
            detection_data = DetectionData(
                camera_id=camera_id,
                timestamp=datetime.now(UTC),
                frame_id=frame_metadata.get("frame_id", str(time.time())),
                detections=detections,
                vehicle_count=len([d for d in detections if d.is_vehicle]),
                metadata=frame_metadata,
                confidence=frame_metadata.get("confidence", 0.8),
                source="ml_batch_processor",
                model_version=frame_metadata.get("model_version"),
                pipeline_id=frame_metadata.get("pipeline_id"),
            )

            # Process through unified analytics (with timeout handled at higher level)
            analytics_result = await self.unified_analytics.process_realtime_analytics(
                detection_data,
                include_anomaly_detection=True,
                include_incident_detection=True,
                include_rule_evaluation=True,
                include_speed_calculation=True,
            )

            # Publish to Redis for real-time subscribers
            await self._publish_to_redis(camera_id, detection_data, analytics_result)

            # Cache results for dashboards
            await self._cache_results(camera_id, detection_data, analytics_result)

        except Exception as e:
            logger.error(f"Camera {camera_id} processing failed: {e}")

    async def _publish_to_redis(
        self, camera_id: str, detection_data: DetectionData, analytics_result: Any
    ) -> None:
        """Publish detection and analytics results to Redis pub/sub.

        Args:
            camera_id: Camera identifier
            detection_data: Detection data
            analytics_result: Analytics processing result
        """
        try:
            # Prepare real-time message
            message = {
                "camera_id": camera_id,
                "timestamp": detection_data.timestamp.isoformat(),
                "vehicle_count": detection_data.vehicle_count,
                "detection_count": len(detection_data.detections),
                "processing_time_ms": analytics_result.processing_time_ms,
                "violations": len(analytics_result.violations),
                "anomalies": len(analytics_result.anomalies),
                "frame_id": detection_data.frame_id,
            }

            # Publish to camera-specific channel
            channel = f"ml:detections:{camera_id}"
            await self.redis_pub.publish(channel, json.dumps(message))

            # Publish to global analytics channel
            global_channel = "analytics:realtime"
            await self.redis_pub.publish(global_channel, json.dumps(message))

        except Exception as e:
            logger.warning(f"Redis publish failed for camera {camera_id}: {e}")

    async def _cache_results(
        self, camera_id: str, detection_data: DetectionData, analytics_result: Any
    ) -> None:
        """Cache results for dashboard and API access.

        Args:
            camera_id: Camera identifier
            detection_data: Detection data
            analytics_result: Analytics result
        """
        try:
            # Cache key for latest results
            cache_key = f"latest_ml_analytics:{camera_id}"

            # Prepare cached data
            cached_data = {
                "timestamp": detection_data.timestamp.isoformat(),
                "vehicle_count": detection_data.vehicle_count,
                "detection_count": len(detection_data.detections),
                "high_confidence_detections": len(
                    detection_data.high_confidence_detections
                ),
                "processing_time_ms": analytics_result.processing_time_ms,
                "violations": len(analytics_result.violations),
                "anomalies": len(analytics_result.anomalies),
                "frame_id": detection_data.frame_id,
                "model_version": detection_data.model_version,
            }

            # Cache with 5 minute TTL
            await self.cache_service.set_json(cache_key, cached_data, ttl=300)

        except Exception as e:
            logger.warning(f"Caching failed for camera {camera_id}: {e}")

    def _update_metrics(self, detection_count: int, processing_time_ms: float) -> None:
        """Update processing metrics.

        Args:
            detection_count: Number of detections processed
            processing_time_ms: Total processing time
        """
        self.metrics["batches_processed"] += 1
        self.metrics["total_detections"] += detection_count

        # Update rolling average latency
        current_avg = self.metrics["avg_latency_ms"]
        batch_count = self.metrics["batches_processed"]

        self.metrics["avg_latency_ms"] = (
            current_avg * (batch_count - 1) + processing_time_ms
        ) / batch_count

        # Log metrics periodically
        if self.metrics["batches_processed"] % 100 == 0:
            self._log_metrics()

    def _log_metrics(self) -> None:
        """Log current performance metrics."""
        uptime_s = time.time() - self.metrics["last_reset"]
        throughput = self.metrics["total_detections"] / max(1, uptime_s)

        logger.info(
            f"ML Analytics Connector Metrics - "
            f"Batches: {self.metrics['batches_processed']}, "
            f"Detections: {self.metrics['total_detections']}, "
            f"Avg Latency: {self.metrics['avg_latency_ms']:.1f}ms, "
            f"Throughput: {throughput:.1f} det/s, "
            f"Timeouts: {self.metrics['timeouts']}, "
            f"Errors: {self.metrics['errors']}"
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive connector metrics.

        Returns:
            Dictionary containing performance metrics
        """
        uptime_s = time.time() - self.metrics["last_reset"]
        throughput = self.metrics["total_detections"] / max(1, uptime_s)

        return {
            "ml_analytics_connector": {
                "batches_processed": self.metrics["batches_processed"],
                "total_detections": self.metrics["total_detections"],
                "avg_latency_ms": self.metrics["avg_latency_ms"],
                "throughput_detections_per_sec": throughput,
                "timeouts": self.metrics["timeouts"],
                "errors": self.metrics["errors"],
                "queue_size": self.batch_queue.qsize(),
                "is_running": self.is_running,
                "uptime_seconds": uptime_s,
            }
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check of the connector.

        Returns:
            Health status information
        """
        try:
            # Check if processing is running
            is_healthy = (
                self.is_running
                and self.processing_task
                and not self.processing_task.done()
            )

            # Check queue status
            queue_depth = self.batch_queue.qsize()
            queue_healthy = queue_depth < self.batch_queue.maxsize * 0.8

            # Check error rate
            total_ops = self.metrics["batches_processed"]
            error_rate = self.metrics["errors"] / max(1, total_ops)
            error_healthy = error_rate < 0.05  # Less than 5% error rate

            # Check latency
            latency_healthy = (
                self.metrics["avg_latency_ms"] < self.processing_timeout_ms
            )

            overall_healthy = all(
                [is_healthy, queue_healthy, error_healthy, latency_healthy]
            )

            return {
                "status": "healthy" if overall_healthy else "degraded",
                "is_running": self.is_running,
                "queue_depth": queue_depth,
                "queue_capacity": self.batch_queue.maxsize,
                "error_rate": error_rate,
                "avg_latency_ms": self.metrics["avg_latency_ms"],
                "checks": {
                    "processing_running": is_healthy,
                    "queue_healthy": queue_healthy,
                    "error_rate_ok": error_healthy,
                    "latency_ok": latency_healthy,
                },
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}
