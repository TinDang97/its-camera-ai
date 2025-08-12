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
from ..data.redis_queue_manager import RedisQueueManager
from .grpc_streaming_server import StreamingServer, StreamingServiceImpl
from .streaming_service import (
    CameraConnectionManager,
    FrameQualityValidator,
    StreamingDataProcessor,
)


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
