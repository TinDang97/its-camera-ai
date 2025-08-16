"""ML Streaming Integration Service.

This service provides the crucial integration layer that connects:
1. CoreVisionEngine (YOLO11 ML inference) 
2. UnifiedVisionAnalyticsEngine (unified processing)
3. StreamingService (real-time streams)
4. Kafka event streaming for detection results
5. gRPC servicers for client communication

This service addresses the critical missing piece in the ML pipeline by
providing actual ML inference capabilities to the gRPC streaming endpoints.

Performance Requirements:
- Sub-100ms inference latency
- 1000+ concurrent camera streams  
- Real-time analytics updates
- GPU optimization with CUDA memory management
"""

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from dependency_injector.wiring import Provide, inject

from ..core.config import Settings
from ..core.logging import get_logger
from ..core.unified_vision_analytics_engine import (
    RequestPriority,
    UnifiedResult,
    UnifiedVisionAnalyticsEngine,
)
from ..ml.core_vision_engine import CoreVisionEngine, VisionConfig
from ..ml.inference_optimizer import (
    InferenceConfig,
    ModelType,
    OptimizationBackend,
    OptimizedInferenceEngine,
)
from ..services.cache import CacheService
from ..services.streaming_service import StreamingDataProcessor
from ..services.unified_analytics_service import UnifiedAnalyticsService

logger = get_logger(__name__)


@dataclass
class MLStreamingConfig:
    """Configuration for ML streaming integration."""

    # Model configuration
    model_path: str = "models/yolo11s.pt"
    model_type: ModelType = ModelType.SMALL
    optimization_backend: OptimizationBackend = OptimizationBackend.TENSORRT
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.4
    max_detections: int = 1000

    # Performance configuration
    target_fps: int = 30
    inference_batch_size: int = 8
    max_batch_size: int = 32
    enable_tensorrt: bool = True
    precision: str = "fp16"

    # Device configuration
    gpu_device_ids: list[int] = None
    enable_multi_gpu: bool = True

    # Kafka streaming
    kafka_enabled: bool = True
    kafka_bootstrap_servers: list[str] = None

    # Quality thresholds
    min_quality_score: float = 0.7
    enable_quality_filtering: bool = True


class MLStreamingIntegrationService:
    """Integration service connecting ML pipeline to streaming endpoints.
    
    This service provides the missing link between the ML inference engine
    and the streaming/analytics services, enabling real-time processing
    of camera streams with sub-100ms latency.
    """

    @inject
    def __init__(
        self,
        settings: Settings = Provide["config"],
        unified_analytics: UnifiedAnalyticsService = Provide["services.unified_analytics_service"],
        cache_service: CacheService = Provide["services.cache_service"],
    ):
        """Initialize ML streaming integration service."""
        self.settings = settings
        self.unified_analytics = unified_analytics
        self.cache_service = cache_service

        # Create configuration from settings
        self.config = MLStreamingConfig(
            model_path=getattr(settings, "model_path", "models/yolo11s.pt"),
            model_type=getattr(settings, "model_type", ModelType.SMALL),
            optimization_backend=getattr(settings, "optimization_backend", OptimizationBackend.TENSORRT),
            confidence_threshold=getattr(settings, "confidence_threshold", 0.5),
            iou_threshold=getattr(settings, "iou_threshold", 0.4),
            max_detections=getattr(settings, "max_detections", 1000),
            target_fps=getattr(settings, "target_fps", 30),
            inference_batch_size=getattr(settings, "inference_batch_size", 8),
            max_batch_size=getattr(settings, "max_batch_size", 32),
            enable_tensorrt=getattr(settings, "enable_tensorrt", True),
            precision=getattr(settings, "precision", "fp16"),
            gpu_device_ids=getattr(settings, "gpu_device_ids", [0]),
            enable_multi_gpu=getattr(settings, "enable_multi_gpu", True),
            kafka_enabled=getattr(settings, "kafka_enabled", False),
            kafka_bootstrap_servers=getattr(settings, "kafka_bootstrap_servers", ["localhost:9092"]),
        )

        # Core ML components
        self.core_vision_engine = None
        self.optimized_inference_engine = None
        self.unified_vision_analytics = None

        # Streaming processor for frame handling
        self.streaming_processor = None

        # State tracking
        self.is_initialized = False
        self.is_running = False

        # Performance metrics
        self.frames_processed = 0
        self.inference_times = []
        self.start_time = None

        logger.info("ML Streaming Integration Service initialized")

    async def initialize(self) -> None:
        """Initialize all ML and streaming components."""
        if self.is_initialized:
            logger.warning("Service already initialized")
            return

        logger.info("Initializing ML Streaming Integration Service...")

        try:
            # Step 1: Initialize optimized inference engine
            await self._initialize_inference_engine()

            # Step 2: Initialize core vision engine
            await self._initialize_core_vision_engine()

            # Step 3: Initialize unified vision analytics engine
            await self._initialize_unified_vision_analytics()

            # Step 4: Initialize streaming processor
            await self._initialize_streaming_processor()

            self.is_initialized = True
            logger.info("ML Streaming Integration Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ML streaming service: {e}")
            raise

    async def start(self) -> None:
        """Start the ML streaming service."""
        if not self.is_initialized:
            await self.initialize()

        if self.is_running:
            logger.warning("Service already running")
            return

        logger.info("Starting ML Streaming Integration Service...")

        try:
            # Start all components
            if self.optimized_inference_engine:
                await self.optimized_inference_engine.start()

            if self.core_vision_engine:
                await self.core_vision_engine.start()

            if self.unified_vision_analytics:
                await self.unified_vision_analytics.start()

            self.is_running = True
            self.start_time = time.time()

            logger.info("ML Streaming Integration Service started successfully")

        except Exception as e:
            logger.error(f"Failed to start ML streaming service: {e}")
            raise

    async def stop(self) -> None:
        """Stop the ML streaming service."""
        if not self.is_running:
            return

        logger.info("Stopping ML Streaming Integration Service...")

        try:
            # Stop all components
            if self.unified_vision_analytics:
                await self.unified_vision_analytics.stop()

            if self.core_vision_engine:
                await self.core_vision_engine.stop()

            if self.optimized_inference_engine:
                await self.optimized_inference_engine.stop()

            self.is_running = False

            logger.info("ML Streaming Integration Service stopped")

        except Exception as e:
            logger.error(f"Error stopping ML streaming service: {e}")

    async def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str,
        frame_id: str = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        include_analytics: bool = True,
        include_quality_score: bool = True,
        include_frame_annotation: bool = False,
        include_metadata_track: bool = True,
    ) -> UnifiedResult:
        """Process single frame through unified ML pipeline.
        
        This is the main entry point for frame processing that connects
        the ML inference with real-time analytics.
        """
        if not self.is_running:
            raise RuntimeError("ML streaming service not running")

        start_time = time.time()

        try:
            # Process through unified vision analytics engine
            result = await self.unified_vision_analytics.process_frame(
                frame=frame,
                camera_id=camera_id,
                frame_id=frame_id,
                priority=priority,
                include_analytics=include_analytics,
                include_quality_score=include_quality_score,
                include_frame_annotation=include_frame_annotation,
                include_metadata_track=include_metadata_track,
            )

            # Update performance tracking
            processing_time = (time.time() - start_time) * 1000
            self.inference_times.append(processing_time)
            self.frames_processed += 1

            # Keep only recent measurements
            if len(self.inference_times) > 1000:
                self.inference_times = self.inference_times[-1000:]

            logger.debug(
                f"Processed frame {frame_id} from {camera_id} in {processing_time:.1f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            raise

    async def process_batch(
        self,
        frames: list[np.ndarray],
        camera_ids: list[str],
        frame_ids: list[str] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> list[UnifiedResult]:
        """Process batch of frames efficiently."""
        if not self.is_running:
            raise RuntimeError("ML streaming service not running")

        try:
            # Process through unified vision analytics engine
            results = await self.unified_vision_analytics.process_batch(
                frames=frames,
                camera_ids=camera_ids,
                frame_ids=frame_ids,
                priority=priority,
            )

            # Update performance tracking
            self.frames_processed += len(frames)

            logger.debug(f"Processed batch of {len(frames)} frames")

            return results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise

    async def health_check(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        try:
            # Get health from unified vision analytics
            unified_health = {}
            if self.unified_vision_analytics:
                unified_health = await self.unified_vision_analytics.health_check()

            # Calculate performance metrics
            avg_inference_time = (
                np.mean(self.inference_times) if self.inference_times else 0
            )
            throughput = (
                self.frames_processed / (time.time() - self.start_time)
                if self.start_time else 0
            )

            return {
                "status": "healthy" if self.is_running else "stopped",
                "is_running": self.is_running,
                "is_initialized": self.is_initialized,
                "frames_processed": self.frames_processed,
                "avg_inference_time_ms": avg_inference_time,
                "throughput_fps": throughput,
                "components": {
                    "core_vision_engine": self.core_vision_engine is not None,
                    "optimized_inference_engine": self.optimized_inference_engine is not None,
                    "unified_vision_analytics": self.unified_vision_analytics is not None,
                },
                "unified_engine_health": unified_health,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    async def get_performance_metrics(self) -> dict[str, Any]:
        """Get detailed performance metrics."""
        try:
            # Calculate statistics
            avg_latency = np.mean(self.inference_times) if self.inference_times else 0
            p95_latency = np.percentile(self.inference_times, 95) if self.inference_times else 0
            p99_latency = np.percentile(self.inference_times, 99) if self.inference_times else 0

            uptime = time.time() - self.start_time if self.start_time else 0
            throughput = self.frames_processed / uptime if uptime > 0 else 0

            # Get unified engine metrics
            unified_metrics = {}
            if self.unified_vision_analytics:
                unified_metrics = await self.unified_vision_analytics.get_comprehensive_metrics()

            return {
                "timestamp": datetime.now(UTC).isoformat(),
                "uptime_seconds": uptime,
                "frames_processed": self.frames_processed,
                "throughput_fps": throughput,
                "latency_metrics": {
                    "avg_ms": avg_latency,
                    "p95_ms": p95_latency,
                    "p99_ms": p99_latency,
                    "min_ms": min(self.inference_times) if self.inference_times else 0,
                    "max_ms": max(self.inference_times) if self.inference_times else 0,
                },
                "unified_engine_metrics": unified_metrics,
            }

        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}

    async def _initialize_inference_engine(self) -> None:
        """Initialize optimized inference engine."""
        logger.info("Initializing optimized inference engine...")

        inference_config = InferenceConfig(
            model_type=self.config.model_type,
            backend=self.config.optimization_backend,
            precision=self.config.precision,
            batch_size=self.config.inference_batch_size,
            max_batch_size=self.config.max_batch_size,
            confidence_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            max_detections=self.config.max_detections,
            device_ids=self.config.gpu_device_ids,
        )

        self.optimized_inference_engine = OptimizedInferenceEngine(inference_config)
        logger.info("Optimized inference engine initialized")

    async def _initialize_core_vision_engine(self) -> None:
        """Initialize core vision engine."""
        logger.info("Initializing core vision engine...")

        vision_config = VisionConfig(
            model_path=self.config.model_path,
            confidence_threshold=self.config.confidence_threshold,
            iou_threshold=self.config.iou_threshold,
            max_detections=self.config.max_detections,
            target_fps=self.config.target_fps,
            enable_tensorrt=self.config.enable_tensorrt,
            batch_size=self.config.inference_batch_size,
            num_workers=len(self.config.gpu_device_ids),
            device_ids=self.config.gpu_device_ids,
        )

        self.core_vision_engine = CoreVisionEngine(vision_config)
        logger.info("Core vision engine initialized")

    async def _initialize_unified_vision_analytics(self) -> None:
        """Initialize unified vision analytics engine."""
        logger.info("Initializing unified vision analytics engine...")

        self.unified_vision_analytics = UnifiedVisionAnalyticsEngine(
            inference_engine=self.optimized_inference_engine,
            unified_analytics=self.unified_analytics,
            cache_service=self.cache_service,
            settings=self.settings,
        )

        logger.info("Unified vision analytics engine initialized")

    async def _initialize_streaming_processor(self) -> None:
        """Initialize streaming data processor."""
        logger.info("Initializing streaming processor...")

        self.streaming_processor = StreamingDataProcessor(
            ml_integration_service=self,
            cache_service=self.cache_service,
        )

        logger.info("Streaming processor initialized")
