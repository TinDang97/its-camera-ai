"""Dependency Injection Container for Streaming Service.

This module provides dependency injection configuration for the streaming service,
following the project's architecture patterns and ensuring proper separation of concerns.

Key Features:
- Dependency injection using dependency-injector
- Configuration management
- Resource lifecycle management
- Environment-specific configurations
"""

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

# Internal imports
from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)
from ..flow.redis_queue_manager import RedisQueueManager
from .grpc_streaming_server import StreamingServer, StreamingServiceImpl
from .streaming_service import (
    CameraConnectionManager,
    FrameQualityValidator,
    SSEStreamingService,
    StreamingDataProcessor,
    StreamingService,
)

# Optional ML imports - may not be available in all environments
try:
    from ..ml.core_vision_engine import CoreVisionEngine, VisionConfig
    from ..ml.streaming_annotation_processor import (
        AnnotationStyleConfig,
        DetectionConfig,
        MLAnnotationProcessor,
    )

    ML_COMPONENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML components not available for streaming: {e}")
    MLAnnotationProcessor = None
    DetectionConfig = None
    AnnotationStyleConfig = None
    CoreVisionEngine = None
    VisionConfig = None
    ML_COMPONENTS_AVAILABLE = False


class StreamingContainer(containers.DeclarativeContainer):
    """Dependency injection container for streaming service components."""

    # Configuration
    config = providers.Configuration()

    # Core dependencies
    settings = providers.Singleton(get_settings)

    # Redis Queue Manager
    redis_queue_manager = providers.Singleton(
        RedisQueueManager,
        redis_url=config.redis.url.as_(str).provided,
        pool_size=config.redis.pool_size.as_(int).provided,
        timeout=config.redis.timeout.as_(int).provided,
        retry_on_failure=config.redis.retry_on_failure.as_(bool).provided,
    )

    # Frame Quality Validator
    frame_quality_validator = providers.Singleton(
        FrameQualityValidator,
        min_resolution=config.streaming.min_resolution.provided,
        min_quality_score=config.streaming.min_quality_score.as_(float).provided,
        max_blur_threshold=config.streaming.max_blur_threshold.as_(float).provided,
    )

    # Camera Connection Manager
    camera_connection_manager = providers.Singleton(CameraConnectionManager)

    # Main Streaming Data Processor
    streaming_data_processor = providers.Singleton(
        StreamingDataProcessor,
        redis_client=redis_queue_manager,
        quality_validator=frame_quality_validator,
        connection_manager=camera_connection_manager,
        max_concurrent_streams=config.streaming.max_concurrent_streams.as_(
            int
        ).provided,
        frame_processing_timeout=config.streaming.frame_processing_timeout.as_(
            float
        ).provided,
    )

    # gRPC Service Implementation
    grpc_service_impl = providers.Singleton(
        StreamingServiceImpl,
        streaming_processor=streaming_data_processor,
        redis_manager=redis_queue_manager,
    )

    # Main Streaming Service (independent of gRPC)
    streaming_service = providers.Singleton(
        StreamingService,
        streaming_processor=streaming_data_processor,
        redis_manager=redis_queue_manager,
        config=config.streaming.provided,
    )

    # ML Components for AI-Annotated Streaming (conditional on availability)
    if ML_COMPONENTS_AVAILABLE:
        ml_detection_config = providers.Singleton(
            DetectionConfig,
            confidence_threshold=config.ml_streaming.confidence_threshold.as_(
                float
            ).provided,
            classes_to_detect=config.ml_streaming.classes_to_detect.provided,
            target_latency_ms=config.ml_streaming.target_latency_ms.as_(float).provided,
            batch_size=config.ml_streaming.batch_size.as_(int).provided,
        )

        ml_annotation_style = providers.Singleton(
            AnnotationStyleConfig,
            show_confidence=config.ml_streaming.show_confidence.as_(bool).provided,
            show_class_labels=config.ml_streaming.show_class_labels.as_(bool).provided,
            box_thickness=config.ml_streaming.box_thickness.as_(int).provided,
            font_scale=config.ml_streaming.font_scale.as_(float).provided,
        )

        ml_vision_engine = providers.Singleton(
            CoreVisionEngine,
            config=providers.Factory(
                VisionConfig,
                confidence_threshold=ml_detection_config.provided.confidence_threshold,
                batch_size=ml_detection_config.provided.batch_size,
                target_latency_ms=ml_detection_config.provided.target_latency_ms,
            ),
        )

        ml_annotation_processor = providers.Singleton(
            MLAnnotationProcessor,
            vision_engine=ml_vision_engine,
            config=providers.Factory(
                DetectionConfig,
                confidence_threshold=ml_detection_config.provided.confidence_threshold,
                classes_to_detect=ml_detection_config.provided.classes_to_detect,
                target_latency_ms=ml_detection_config.provided.target_latency_ms,
                batch_size=ml_detection_config.provided.batch_size,
                annotation_style=ml_annotation_style,
            ),
        )
    else:
        # Provide None when ML components are not available
        ml_detection_config = providers.Object(None)
        ml_annotation_style = providers.Object(None)
        ml_vision_engine = providers.Object(None)
        ml_annotation_processor = providers.Object(None)

    # SSE Streaming Service for browser-native video viewing with ML support
    sse_streaming_service = providers.Singleton(
        SSEStreamingService,
        base_streaming_service=streaming_service,
        redis_manager=redis_queue_manager,
        config=config.sse_streaming.provided,
        ml_annotation_processor=ml_annotation_processor,
    )

    # Streaming Server
    streaming_server = providers.Singleton(
        StreamingServer,
        host=config.streaming.grpc_host.as_(str).provided,
        port=config.streaming.grpc_port.as_(int).provided,
    )


# Factory function to create and configure container
def create_streaming_container(config_dict: dict = None) -> StreamingContainer:
    """Create and configure a streaming container with the provided configuration.

    Args:
        config_dict: Optional configuration dictionary override

    Returns:
        StreamingContainer: Configured container instance
    """
    container = StreamingContainer()

    # Default configuration
    default_config = {
        "redis": {
            "url": "redis://localhost:6379",
            "pool_size": 20,
            "timeout": 30,
            "retry_on_failure": True,
        },
        "streaming": {
            "min_resolution": (640, 480),
            "min_quality_score": 0.5,
            "max_blur_threshold": 100.0,
            "max_concurrent_streams": 100,
            "frame_processing_timeout": 0.01,  # 10ms
            "grpc_host": "127.0.0.1",
            "grpc_port": 50051,
        },
        "sse_streaming": {
            "max_concurrent_connections": 100,
            "fragment_duration_ms": 2000,
            "heartbeat_interval": 30,
            "connection_timeout": 300,
            "sync_tolerance_ms": 50.0,
            "sync_check_interval": 1.0,
            "max_sync_violations": 10,
            "enable_dual_channel": True,
            "channel_switch_timeout": 0.1,
        },
        "ml_streaming": {
            "confidence_threshold": 0.5,
            "classes_to_detect": [
                "car",
                "truck",
                "bus",
                "motorcycle",
                "bicycle",
                "person",
            ],
            "target_latency_ms": 50.0,
            "batch_size": 8,
            "show_confidence": True,
            "show_class_labels": True,
            "box_thickness": 2,
            "font_scale": 0.6,
            "enable_gpu_acceleration": True,
            "vehicle_priority": True,
            "emergency_vehicle_detection": True,
        },
    }

    # Merge with provided config
    if config_dict:

        def deep_merge(base: dict, override: dict) -> dict:
            """Deep merge configuration dictionaries."""
            result = base.copy()
            for key, value in override.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        default_config = deep_merge(default_config, config_dict)

    # Configure the container
    container.config.from_dict(default_config)

    # Wire the container
    container.wire(
        modules=[
            "src.its_camera_ai.services.streaming_service",
            "src.its_camera_ai.services.grpc_streaming_server",
        ]
    )

    return container


# Global container instance (will be initialized by the application)
_streaming_container: StreamingContainer = None


def get_streaming_container() -> StreamingContainer:
    """Get the global streaming container instance."""
    global _streaming_container
    if _streaming_container is None:
        _streaming_container = create_streaming_container()
    return _streaming_container


def initialize_streaming_container(config_dict: dict = None) -> StreamingContainer:
    """Initialize the global streaming container with configuration."""
    global _streaming_container
    _streaming_container = create_streaming_container(config_dict)
    return _streaming_container


# Dependency injection decorators for easy access
@inject
def get_streaming_processor(
    processor: StreamingDataProcessor = Provide[
        StreamingContainer.streaming_data_processor
    ],
) -> StreamingDataProcessor:
    """Get the streaming data processor instance."""
    return processor


@inject
def get_grpc_service(
    service: StreamingServiceImpl = Provide[StreamingContainer.grpc_service_impl],
) -> StreamingServiceImpl:
    """Get the gRPC service implementation instance."""
    return service


@inject
def get_streaming_server(
    server: StreamingServer = Provide[StreamingContainer.streaming_server],
) -> StreamingServer:
    """Get the streaming server instance."""
    return server


@inject
def get_redis_manager(
    manager: RedisQueueManager = Provide[StreamingContainer.redis_queue_manager],
) -> RedisQueueManager:
    """Get the Redis queue manager instance."""
    return manager


@inject
def get_streaming_service(
    service: StreamingService = Provide[StreamingContainer.streaming_service],
) -> StreamingService:
    """Get the main streaming service instance."""
    return service


@inject
def get_sse_streaming_service(
    service: SSEStreamingService = Provide[StreamingContainer.sse_streaming_service],
) -> SSEStreamingService:
    """Get the SSE streaming service instance."""
    return service


# Conditional ML dependency injection helpers
if ML_COMPONENTS_AVAILABLE:

    @inject
    def get_ml_annotation_processor(
        processor: MLAnnotationProcessor = Provide[
            StreamingContainer.ml_annotation_processor
        ],
    ) -> MLAnnotationProcessor:
        """Get the ML annotation processor instance."""
        return processor

    @inject
    def get_ml_vision_engine(
        engine: CoreVisionEngine = Provide[StreamingContainer.ml_vision_engine],
    ) -> CoreVisionEngine:
        """Get the ML vision engine instance."""
        return engine

else:

    def get_ml_annotation_processor():
        """ML annotation processor not available."""
        return None

    def get_ml_vision_engine():
        """ML vision engine not available."""
        return None
