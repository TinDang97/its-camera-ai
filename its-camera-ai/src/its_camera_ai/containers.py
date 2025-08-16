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

from .core.config import get_settings
from .core.configs.workers import get_workers_config
from .core.logging import get_logger
from .models.database import DatabaseManager

logger = get_logger(__name__)


async def _init_database_manager(settings):
    """Initialize and return a database manager with proper lifecycle.
    
    This function creates a DatabaseManager, initializes it, and returns it.
    The Resource provider will handle cleanup automatically.
    
    Args:
        settings: Application settings
        
    Returns:
        Initialized DatabaseManager instance
    """
    db_manager = DatabaseManager(settings)
    await db_manager.initialize()

    # Return the database manager for use in the container
    # The Resource provider will call cleanup() when disposing
    try:
        yield db_manager
    finally:
        await db_manager.cleanup()


class InfrastructureContainer(containers.DeclarativeContainer):
    """Infrastructure container for external dependencies.

    Manages database connections, Redis, external APIs, and other
    infrastructure components using Resource providers for proper
    lifecycle management with the unified DatabaseManager pattern.
    """

    # Configuration
    config = providers.Configuration()

    # Settings provider
    settings = providers.Singleton(
        get_settings,
    )

    # Database Manager - Resource with proper lifecycle management
    database = providers.Resource(
        _init_database_manager,
        settings=settings,
    )

    # Session Factory - Direct access to the database manager's get_session method
    session_factory = providers.Factory(
        lambda db: db.get_session,
        db=database,
    )

    # Database Session - Convenience provider for database sessions
    database_session = providers.Factory(
        lambda db: db.get_session(),
        db=database,
    )

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

    # Legacy Analytics Service - Factory for per-request instances
    analytics_service = providers.Factory(
        "its_camera_ai.services.analytics_service.EnhancedAnalyticsService",
        analytics_repository=repositories.analytics_repository,
        metrics_repository=repositories.metrics_repository,
        detection_repository=repositories.detection_repository,
        settings=config,
        cache_service=cache_service,
    )

    # Refactored Analytics Components with DI

    # Real-time Analytics Service - Factory for real-time analytics generation
    realtime_analytics_service = providers.Factory(
        "its_camera_ai.services.realtime_analytics_service.RealtimeAnalyticsService",
        cache_service=cache_service,
        settings=config,
    )

    # Historical Analytics Service - Factory for historical data queries
    historical_analytics_service = providers.Factory(
        "its_camera_ai.services.historical_analytics_service.HistoricalAnalyticsService",
        analytics_repository=repositories.analytics_repository,
        database_manager=infrastructure.database,
        cache_service=cache_service,
        settings=config,
    )

    # Analytics Aggregation Service - Factory for time-series aggregation
    analytics_aggregation_service = providers.Factory(
        "its_camera_ai.services.analytics_aggregation_service.AnalyticsAggregationService",
        analytics_repository=repositories.analytics_repository,
        cache_service=cache_service,
        settings=config,
    )

    # Incident Detection Service - Factory for incident processing
    incident_detection_service = providers.Factory(
        "its_camera_ai.services.incident_detection_service.IncidentDetectionService",
        alert_repository=repositories.alert_repository,
        cache_service=cache_service,
        settings=config,
    )

    # Incident Management Service - Factory for incident CRUD operations
    incident_management_service = providers.Factory(
        "its_camera_ai.services.incident_management_service.IncidentManagementService",
        alert_repository=repositories.alert_repository,
        database_manager=infrastructure.database,
        cache_service=cache_service,
        settings=config,
    )

    # Traffic Rule Service - Factory for rule evaluation
    traffic_rule_service = providers.Factory(
        "its_camera_ai.services.traffic_rule_service.TrafficRuleService",
        analytics_repository=repositories.analytics_repository,
        cache_service=cache_service,
        settings=config,
    )

    # Speed Calculation Service - Factory for speed analysis
    speed_calculation_service = providers.Factory(
        "its_camera_ai.services.speed_calculation_service.SpeedCalculationService",
        analytics_repository=repositories.analytics_repository,
        cache_service=cache_service,
        settings=config,
    )

    # Anomaly Detection Service - Factory for ML-based anomaly detection
    anomaly_detection_service = providers.Factory(
        "its_camera_ai.services.anomaly_detection_service.AnomalyDetectionService",
        analytics_repository=repositories.analytics_repository,
        cache_service=cache_service,
        settings=config,
    )

    # Prediction Service - Factory for ML predictions
    prediction_service = providers.Factory(
        "its_camera_ai.services.prediction_service.PredictionService",
        cache_service=cache_service,
        settings=config,
    )

    # Unified Analytics Service - Factory for comprehensive analytics
    unified_analytics_service = providers.Factory(
        "its_camera_ai.services.unified_analytics_service.UnifiedAnalyticsService",
        aggregation_service=analytics_aggregation_service,
        incident_detection_service=incident_detection_service,
        traffic_rule_service=traffic_rule_service,
        speed_calculation_service=speed_calculation_service,
        anomaly_detection_service=anomaly_detection_service,
        prediction_service=prediction_service,
        analytics_repository=repositories.analytics_repository,
        cache_service=cache_service,
        settings=config,
    )

    # Legacy support - Incident Detection Engine alias
    incident_detection_engine = providers.Factory(
        "its_camera_ai.services.incident_detection_service.IncidentDetectionService",
        alert_repository=repositories.alert_repository,
        cache_service=cache_service,
        settings=config,
    )

    # Alert Service - Factory for per-request instances
    alert_service = providers.Factory(
        "its_camera_ai.services.alert_service.AlertService",
        alert_repository=repositories.alert_repository,
        settings=config,
    )

    # License Plate Recognition Service - Factory for LPR pipeline
    lpr_service = providers.Factory(
        "its_camera_ai.ml.license_plate_recognition.create_lpr_pipeline",
        region="AUTO",
        use_gpu=True,
        enable_caching=True,
        target_latency_ms=15.0,
    )

    # ML Pipeline Integration Services

    # ML Analytics Connector - Singleton for shared processing state
    ml_analytics_connector = providers.Singleton(
        "its_camera_ai.services.ml_analytics_connector.MLAnalyticsConnector",
        # Note: batch_processor would be injected when ML module is available
        unified_analytics=unified_analytics_service,
        redis_client=infrastructure.redis_client,
        cache_service=cache_service,
        settings=config,
    )

    # Quality Score Calculator - Singleton for shared caching
    quality_score_calculator = providers.Singleton(
        "its_camera_ai.ml.quality_score_calculator.QualityScoreCalculator",
        cache_service=cache_service,
    )

    # Model Metrics Service - Singleton for metrics tracking
    model_metrics_service = providers.Singleton(
        "its_camera_ai.services.model_metrics_service.ModelMetricsService",
        analytics_repository=repositories.analytics_repository,
        cache_service=cache_service,
        settings=config,
    )

    # Kafka Event Producer - Singleton for shared Kafka connection
    kafka_event_producer = providers.Singleton(
        "its_camera_ai.services.kafka_event_producer.KafkaEventProducer",
        config=config.kafka_producer,
    )

    # SSE Broadcaster - Singleton for managing SSE connections
    sse_broadcaster = providers.Singleton(
        "its_camera_ai.api.sse_broadcaster.get_broadcaster",
        kafka_config=config.kafka_consumer,
    )

    # Kafka SSE Consumer - Singleton for consuming events for SSE streaming
    kafka_sse_consumer = providers.Singleton(
        "its_camera_ai.services.kafka_sse_consumer.KafkaSSEConsumer",
        config=config.kafka_consumer,
        sse_broadcaster=sse_broadcaster,
    )

    # Kafka Analytics Connector - Enhanced with SSE integration
    kafka_analytics_connector = providers.Singleton(
        "its_camera_ai.integrations.kafka_analytics_connector.KafkaAnalyticsConnector",
        kafka_config=config.kafka_producer,
        event_filters=config.kafka_event_filters,
    )

    # MP4 Streaming Components

    # Fragmented MP4 Encoder Factory - Creates encoders for individual cameras
    fragmented_mp4_encoder_factory = providers.Factory(
        "its_camera_ai.services.fragmented_mp4_encoder.create_fragmented_mp4_encoder",
    )

    # Metadata Track Manager Factory - Creates metadata managers with default config
    metadata_track_manager_factory = providers.Factory(
        "its_camera_ai.services.metadata_track_manager.create_metadata_track_manager",
    )

    # DASH Fragment Generator Factory - Creates DASH generators for cameras
    dash_fragment_generator_factory = providers.Factory(
        "its_camera_ai.services.dash_fragment_generator.create_dash_fragment_generator",
    )

    # MP4 Streaming Service - Singleton for coordinating all MP4 streaming
    mp4_streaming_service = providers.Singleton(
        "its_camera_ai.services.mp4_streaming_service.create_mp4_streaming_service",
        base_config=config.mp4_streaming,
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
        analytics_service=services.unified_analytics_service,  # Use unified analytics
        alert_service=services.alert_service,
        config=config.service_mesh,
    )

    # Real-time Event Streaming Service - Coordinates Kafka and SSE
    realtime_streaming_service = providers.Singleton(
        "its_camera_ai.services.realtime_streaming_service.RealtimeStreamingService",
        kafka_producer=services.kafka_event_producer,
        kafka_consumer=services.kafka_sse_consumer,
        sse_broadcaster=services.sse_broadcaster,
        analytics_connector=services.kafka_analytics_connector,
        config=config.realtime_streaming,
    )

    # MP4 Streaming Service - Coordinates fragmented MP4 encoding and DASH streaming
    mp4_streaming_service = providers.Singleton(
        "its_camera_ai.services.mp4_streaming_service.MP4StreamingService",
        base_config=config.mp4_streaming,
    )

    # Background Workers Configuration
    workers_config = providers.Singleton(
        get_workers_config,
    )
