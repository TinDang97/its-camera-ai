"""Dependency injection containers for ITS Camera AI System.

This module implements the dependency injection architecture using the
dependency-injector library, providing a clean separation between
infrastructure, repositories, services, and application layers.

Architecture:
- InfrastructureContainer: Database, Redis, external connections
- RepositoryContainer: Data access layer with repository pattern
- ServiceContainer: Business logic and service layer
- ApplicationContainer: Application-wide dependencies and wiring

The containers support async/await patterns for FastAPI integration
and provide proper resource management with cleanup.
"""

import redis.asyncio as redis
from dependency_injector import containers, providers
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from .core.logging import get_logger

logger = get_logger(__name__)


class InfrastructureContainer(containers.DeclarativeContainer):
    """Infrastructure container for external dependencies.

    Manages database connections, Redis, external APIs, and other
    infrastructure components using Resource providers for proper
    lifecycle management.
    """

    # Configuration
    config = providers.Configuration()

    # Database Engine - Singleton for connection pooling
    database_engine = providers.Resource(
        providers.Factory(
            create_async_engine,
            config.database.url,
            echo=config.database.echo,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    )

    # Session Factory - Factory for creating sessions
    session_factory = providers.Factory(
        sessionmaker,
        bind=database_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Database Session - Factory for individual sessions
    database_session = providers.Resource(providers.Factory(session_factory))

    # Redis Connection - Singleton with connection pooling
    redis_client = providers.Resource(
        providers.Factory(
            redis.from_url,
            config.redis.url,
            max_connections=config.redis.max_connections,
            socket_timeout=config.redis.timeout,
            retry_on_timeout=config.redis.retry_on_timeout,
            decode_responses=True,
        )
    )


class RepositoryContainer(containers.DeclarativeContainer):
    """Repository container for data access layer.

    Implements the repository pattern for clean data access,
    abstracting database operations from business logic.
    """

    # Infrastructure dependencies
    infrastructure = providers.DependenciesContainer()

    # User Repository
    user_repository = providers.Factory(
        "its_camera_ai.repositories.user_repository.UserRepository",
        session_factory=infrastructure.session_factory,
    )

    # Camera Repository
    camera_repository = providers.Factory(
        "its_camera_ai.repositories.camera_repository.CameraRepository",
        session_factory=infrastructure.session_factory,
    )

    # Frame Repository
    frame_repository = providers.Factory(
        "its_camera_ai.repositories.frame_repository.FrameRepository",
        session_factory=infrastructure.session_factory,
    )

    # Detection Repository
    detection_repository = providers.Factory(
        "its_camera_ai.repositories.detection_repository.DetectionRepository",
        session_factory=infrastructure.session_factory,
    )

    # Metrics Repository
    metrics_repository = providers.Factory(
        "its_camera_ai.repositories.metrics_repository.MetricsRepository",
        session_factory=infrastructure.session_factory,
    )

    # Alert Repository
    alert_repository = providers.Factory(
        "its_camera_ai.repositories.alert_repository.AlertRepository",
        session_factory=infrastructure.session_factory,
    )

    # Analytics Repository
    analytics_repository = providers.Factory(
        "its_camera_ai.repositories.analytics_repository.AnalyticsRepository",
        session_factory=infrastructure.session_factory,
    )


class ServiceContainer(containers.DeclarativeContainer):
    """Service container for business logic layer.

    Contains all business services with proper dependency injection.
    Services are created as Factory providers to ensure proper
    scoping and resource management.
    """

    # Dependencies
    infrastructure = providers.DependenciesContainer()
    repositories = providers.DependenciesContainer()
    config = providers.Configuration()

    # Cache Service - Singleton for shared state
    cache_service = providers.Singleton(
        "its_camera_ai.services.cache.CacheService",
        redis_client=infrastructure.redis_client,
    )

    # Authentication Service - Singleton for shared state
    auth_service = providers.Singleton(
        "its_camera_ai.services.auth_service.AuthenticationService",
        user_repository=repositories.user_repository,
        cache_service=cache_service,
        config=config.security,
    )

    # Token Service - Singleton for shared JWT keys
    token_service = providers.Singleton(
        "its_camera_ai.services.token_service.TokenService",
        cache_service=cache_service,
        config=config.security,
    )

    # MFA Service - Singleton for shared state
    mfa_service = providers.Singleton(
        "its_camera_ai.services.mfa_service.MFAService",
        cache_service=cache_service,
        config=config.security,
    )

    # Email Service - Factory for per-request instances
    email_service = providers.Factory(
        "its_camera_ai.services.email_service.EmailService",
        config=config.email,
    )

    # Camera Service - Factory for per-request instances
    camera_service = providers.Factory(
        "its_camera_ai.services.camera_service.CameraService",
        camera_repository=repositories.camera_repository,
    )

    # Frame Service - Factory for per-request instances
    frame_service = providers.Factory(
        "its_camera_ai.services.frame_service.FrameService",
        frame_repository=repositories.frame_repository,
    )

    # Detection Service - Factory for per-request instances
    detection_service = providers.Factory(
        "its_camera_ai.services.frame_service.DetectionService",
        detection_repository=repositories.detection_repository,
    )

    # Metrics Service - Factory for per-request instances
    metrics_service = providers.Factory(
        "its_camera_ai.services.metrics_service.MetricsService",
        metrics_repository=repositories.metrics_repository,
    )

    # Streaming Data Processor - Singleton for shared processing state
    streaming_data_processor = providers.Singleton(
        "its_camera_ai.services.streaming_service.StreamingDataProcessor",
        redis_client=infrastructure.redis_client,
        max_concurrent_streams=config.streaming.max_concurrent_streams,
        frame_processing_timeout=config.streaming.frame_processing_timeout,
    )

    # Main Streaming Service - Independent service with CLI support
    streaming_service = providers.Singleton(
        "its_camera_ai.services.streaming_service.StreamingService",
        streaming_processor=streaming_data_processor,
        redis_manager=infrastructure.redis_client,
        config=config.streaming,
    )

    # SSE Streaming Service for browser-native video viewing
    sse_streaming_service = providers.Singleton(
        "its_camera_ai.services.streaming_service.SSEStreamingService",
        base_streaming_service=streaming_service,
        redis_manager=infrastructure.redis_client,
        config=config.sse_streaming,
    )

    # Analytics Service - Factory for per-request instances
    analytics_service = providers.Factory(
        "its_camera_ai.services.analytics_service.AnalyticsService",
        analytics_repository=repositories.analytics_repository,
        metrics_repository=repositories.metrics_repository,
        detection_repository=repositories.detection_repository,
        settings=config,
        cache_service=cache_service,
    )

    # Alert Service - Factory for per-request instances
    alert_service = providers.Factory(
        "its_camera_ai.services.alert_service.AlertService",
        alert_repository=repositories.alert_repository,
        settings=config,
    )


class ApplicationContainer(containers.DeclarativeContainer):
    """Main application container that wires all dependencies.

    This is the root container that coordinates all other containers
    and provides the main dependency injection configuration for
    the FastAPI application.
    """

    # Configuration
    config = providers.Configuration()

    # Nested containers
    infrastructure = providers.Container(
        InfrastructureContainer,
        config=config,
    )

    repositories = providers.Container(
        RepositoryContainer,
        infrastructure=infrastructure,
    )

    services = providers.Container(
        ServiceContainer,
        infrastructure=infrastructure,
        repositories=repositories,
        config=config,
    )

    # Application-level services
    service_mesh = providers.Singleton(
        "its_camera_ai.services.service_mesh.ServiceMesh",
        camera_service=services.camera_service,
        streaming_service=services.streaming_service,
        analytics_service=services.analytics_service,
        alert_service=services.alert_service,
        config=config.service_mesh,
    )
