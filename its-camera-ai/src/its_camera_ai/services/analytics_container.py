"""Dependency Injection Container for Analytics Service.

This module provides dependency injection configuration for the analytics service,
following the project's architecture patterns and ensuring proper separation of concerns.

Key Features:
- Dependency injection using dependency-injector
- Analytics service configuration management
- TimescaleDB session management
- Alert service integration
- ML model lifecycle management
"""

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Internal imports
from ..core.config import get_settings
from .alert_service import AlertService
from .analytics_service import AnalyticsService


class AnalyticsContainer(containers.DeclarativeContainer):
    """Dependency injection container for analytics service components."""

    # Configuration
    config = providers.Configuration()

    # Core dependencies
    settings = providers.Singleton(get_settings)

    # Database engine for analytics (TimescaleDB optimized)
    analytics_engine = providers.Singleton(
        create_async_engine,
        settings.provided.database.url,
        # TimescaleDB optimized connection settings
        pool_size=config.database.pool_size.as_(int).provided.or_else(20),
        max_overflow=config.database.max_overflow.as_(int).provided.or_else(40),
        pool_timeout=config.database.pool_timeout.as_(int).provided.or_else(30),
        pool_pre_ping=True,  # Enable connection validation
        pool_recycle=3600,  # Recycle connections every hour
        echo=config.database.echo.as_(bool).provided.or_else(False),
        # TimescaleDB specific optimizations
        connect_args={
            "server_settings": {
                "application_name": "its_camera_ai_analytics",
                "jit": "off",  # Disable JIT for consistent performance
            }
        },
    )

    # Session factory for analytics database
    analytics_session_factory = providers.Singleton(
        sessionmaker,
        bind=analytics_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,  # Manual control over flushing for batch operations
    )

    # Analytics database session provider
    analytics_session = providers.Factory(
        lambda factory: factory(),
        factory=analytics_session_factory,
    )

    # Main Analytics Service
    analytics_service = providers.Factory(
        AnalyticsService,
        session=analytics_session,
        settings=settings,
    )

    # Alert Service
    alert_service = providers.Factory(
        AlertService,
        session=analytics_session,
        settings=settings,
    )

    # Analytics service repositories for direct database access
    traffic_metrics_repository = providers.Factory(
        lambda session: session,  # Direct session access for repositories
        session=analytics_session,
    )

    rule_violations_repository = providers.Factory(
        lambda session: session,
        session=analytics_session,
    )

    traffic_anomalies_repository = providers.Factory(
        lambda session: session,
        session=analytics_session,
    )

    vehicle_trajectories_repository = providers.Factory(
        lambda session: session,
        session=analytics_session,
    )

    alert_notifications_repository = providers.Factory(
        lambda session: session,
        session=analytics_session,
    )


# Factory function to create and configure analytics container
def create_analytics_container(config_dict: dict = None) -> AnalyticsContainer:
    """Create and configure an analytics container with the provided configuration.

    Args:
        config_dict: Optional configuration dictionary override

    Returns:
        AnalyticsContainer: Configured container instance
    """
    container = AnalyticsContainer()

    # Default analytics configuration
    default_config = {
        "database": {
            "pool_size": 20,  # Higher pool size for analytics workloads
            "max_overflow": 40,  # Large overflow for burst analytics
            "pool_timeout": 30,  # Connection timeout
            "echo": False,  # SQL logging (disable in production)
        },
        "analytics": {
            "rule_evaluation_batch_size": 1000,  # Batch size for rule evaluation
            "anomaly_detection_enabled": True,  # Enable ML anomaly detection
            "anomaly_training_samples": 1000,  # Training samples for anomaly detector
            "metrics_aggregation_interval": 300,  # 5 minutes aggregation
            "violation_cooldown_minutes": 5,  # Violation alert cooldown
            "anomaly_cooldown_minutes": 15,  # Anomaly alert cooldown
            "max_trajectory_points": 1000,  # Max points per trajectory
            "trajectory_cleanup_hours": 24,  # Clean old trajectories
        },
        "alerts": {
            "enabled": True,  # Enable alert generation
            "max_retries": 3,  # Max delivery retries
            "retry_delay_minutes": 5,  # Delay between retries
            "batch_alerts": True,  # Batch similar alerts
            "batch_window_minutes": 10,  # Alert batching window
            "notification_channels": ["email", "webhook"],  # Enabled channels
        },
        "performance": {
            "enable_connection_pooling": True,  # Use connection pooling
            "enable_query_optimization": True,  # Enable query optimizations
            "enable_batch_processing": True,  # Enable batch operations
            "max_concurrent_operations": 10,  # Max concurrent analytics ops
            "processing_timeout_seconds": 60,  # Processing timeout
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
            "src.its_camera_ai.services.analytics_service",
            "src.its_camera_ai.services.alert_service",
            "src.its_camera_ai.api.routers.analytics",
            "src.its_camera_ai.cli.commands.ml",
        ]
    )

    return container


# Global container instance (will be initialized by the application)
_analytics_container: AnalyticsContainer = None


def get_analytics_container() -> AnalyticsContainer:
    """Get the global analytics container instance."""
    global _analytics_container
    if _analytics_container is None:
        _analytics_container = create_analytics_container()
    return _analytics_container


def initialize_analytics_container(config_dict: dict = None) -> AnalyticsContainer:
    """Initialize the global analytics container with configuration."""
    global _analytics_container
    _analytics_container = create_analytics_container(config_dict)
    return _analytics_container


# Dependency injection decorators for easy access
@inject
def get_analytics_service(
    service: AnalyticsService = Provide[AnalyticsContainer.analytics_service],
) -> AnalyticsService:
    """Get the analytics service instance."""
    return service


@inject
def get_alert_service(
    service: AlertService = Provide[AnalyticsContainer.alert_service],
) -> AlertService:
    """Get the alert service instance."""
    return service


@inject
def get_analytics_session(
    session: AsyncSession = Provide[AnalyticsContainer.analytics_session],
) -> AsyncSession:
    """Get an analytics database session."""
    return session


@inject
def get_traffic_metrics_repository(
    repository=Provide[AnalyticsContainer.traffic_metrics_repository],
):
    """Get the traffic metrics repository."""
    return repository


@inject
def get_rule_violations_repository(
    repository=Provide[AnalyticsContainer.rule_violations_repository],
):
    """Get the rule violations repository."""
    return repository


@inject
def get_traffic_anomalies_repository(
    repository=Provide[AnalyticsContainer.traffic_anomalies_repository],
):
    """Get the traffic anomalies repository."""
    return repository


@inject
def get_vehicle_trajectories_repository(
    repository=Provide[AnalyticsContainer.vehicle_trajectories_repository],
):
    """Get the vehicle trajectories repository."""
    return repository


@inject
def get_alert_notifications_repository(
    repository=Provide[AnalyticsContainer.alert_notifications_repository],
):
    """Get the alert notifications repository."""
    return repository


# Lifecycle management functions
async def initialize_analytics_database() -> None:
    """Initialize analytics database with TimescaleDB extensions and hypertables."""
    from ..core.logging import get_logger

    logger = get_logger(__name__)
    container = get_analytics_container()

    try:
        # Get database engine
        engine = container.analytics_engine()

        # Create TimescaleDB extensions and hypertables
        async with engine.begin() as conn:
            # Enable TimescaleDB extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

            # Create hypertables for time-series data
            hypertables = [
                {
                    "table": "traffic_metrics",
                    "time_column": "timestamp",
                    "chunk_interval": "1 hour",
                },
                {
                    "table": "rule_violations",
                    "time_column": "detection_time",
                    "chunk_interval": "1 day",
                },
                {
                    "table": "traffic_anomalies",
                    "time_column": "detection_time",
                    "chunk_interval": "1 day",
                },
                {
                    "table": "vehicle_trajectories",
                    "time_column": "start_time",
                    "chunk_interval": "1 day",
                },
                {
                    "table": "alert_notifications",
                    "time_column": "created_time",
                    "chunk_interval": "1 day",
                },
            ]

            for hypertable in hypertables:
                try:
                    # Create hypertable if it doesn't exist
                    await conn.execute(f"""
                        SELECT create_hypertable(
                            '{hypertable["table"]}', 
                            '{hypertable["time_column"]}',
                            chunk_time_interval => INTERVAL '{hypertable["chunk_interval"]}',
                            if_not_exists => TRUE
                        );
                    """)

                    logger.info(f"Created/verified hypertable: {hypertable['table']}")

                except Exception as e:
                    # Table might already be a hypertable or not exist yet
                    logger.debug(
                        f"Hypertable creation info for {hypertable['table']}: {e}"
                    )

            # Set up retention policies for automatic data cleanup
            retention_policies = [
                {
                    "table": "traffic_metrics",
                    "retention": "30 days",  # Keep metrics for 30 days
                },
                {
                    "table": "rule_violations",
                    "retention": "90 days",  # Keep violations for 90 days
                },
                {
                    "table": "traffic_anomalies",
                    "retention": "90 days",  # Keep anomalies for 90 days
                },
                {
                    "table": "vehicle_trajectories",
                    "retention": "7 days",  # Keep trajectories for 7 days
                },
                {
                    "table": "alert_notifications",
                    "retention": "30 days",  # Keep alerts for 30 days
                },
            ]

            for policy in retention_policies:
                try:
                    await conn.execute(f"""
                        SELECT add_retention_policy(
                            '{policy["table"]}', 
                            INTERVAL '{policy["retention"]}',
                            if_not_exists => TRUE
                        );
                    """)

                    logger.info(
                        f"Set retention policy for {policy['table']}: {policy['retention']}"
                    )

                except Exception as e:
                    logger.debug(f"Retention policy info for {policy['table']}: {e}")

        logger.info("Analytics database initialization completed")

    except Exception as e:
        logger.error(f"Failed to initialize analytics database: {e}")
        raise


async def cleanup_analytics_resources() -> None:
    """Clean up analytics container resources."""
    from ..core.logging import get_logger

    logger = get_logger(__name__)

    try:
        container = get_analytics_container()

        # Close database engine
        if hasattr(container, "analytics_engine"):
            engine = container.analytics_engine()
            await engine.dispose()
            logger.info("Analytics database engine disposed")

        # Reset global container
        global _analytics_container
        _analytics_container = None

        logger.info("Analytics container resources cleaned up")

    except Exception as e:
        logger.error(f"Failed to cleanup analytics resources: {e}")


# Context manager for analytics operations
class AnalyticsContext:
    """Context manager for analytics operations with proper resource management."""

    def __init__(self, config_dict: dict = None):
        self.config_dict = config_dict
        self.container = None

    async def __aenter__(self):
        """Initialize analytics context."""
        self.container = initialize_analytics_container(self.config_dict)
        await initialize_analytics_database()
        return self.container

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup analytics context."""
        await cleanup_analytics_resources()


# Batch processing utilities
class BatchAnalyticsProcessor:
    """Utility class for batch analytics processing operations."""

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size

    @inject
    async def process_detections_batch(
        self,
        detections: list,
        analytics_service: AnalyticsService = Provide[
            AnalyticsContainer.analytics_service
        ],
    ) -> dict:
        """Process a batch of detections for analytics."""
        results = {
            "processed_count": 0,
            "violations_found": 0,
            "anomalies_detected": 0,
            "processing_time_ms": 0,
        }

        from datetime import datetime

        start_time = datetime.utcnow()

        try:
            # Process detections in smaller chunks for memory efficiency
            for i in range(0, len(detections), self.batch_size):
                batch = detections[i : i + self.batch_size]

                # Process batch
                for detection_group in batch:
                    camera_id = detection_group.get("camera_id")
                    frame_timestamp = detection_group.get("timestamp")
                    detections_list = detection_group.get("detections", [])

                    result = await analytics_service.process_detections(
                        detections_list, camera_id, frame_timestamp
                    )

                    results["processed_count"] += result.get("vehicle_count", 0)
                    results["violations_found"] += len(result.get("violations", []))
                    results["anomalies_detected"] += len(result.get("anomalies", []))

            # Calculate processing time
            end_time = datetime.utcnow()
            results["processing_time_ms"] = (
                end_time - start_time
            ).total_seconds() * 1000

            return results

        except Exception as e:
            from ..core.logging import get_logger

            logger = get_logger(__name__)
            logger.error(f"Batch analytics processing failed: {e}")
            raise


# Performance monitoring utilities
class AnalyticsPerformanceMonitor:
    """Monitor analytics service performance and resource usage."""

    @inject
    async def get_performance_metrics(
        self,
        analytics_service: AnalyticsService = Provide[
            AnalyticsContainer.analytics_service
        ],
        alert_service: AlertService = Provide[AnalyticsContainer.alert_service],
    ) -> dict:
        """Get current performance metrics for analytics services."""
        from datetime import datetime, timedelta

        metrics = {
            "timestamp": datetime.utcnow(),
            "analytics_service": {
                "active": True,
                "last_processing_time_ms": 0,
            },
            "alert_service": {
                "active": True,
                "pending_alerts": 0,
                "failed_alerts": 0,
            },
            "database": {
                "connection_pool_size": 0,
                "active_connections": 0,
            },
        }

        try:
            # Get alert statistics for the last hour
            time_range = (datetime.utcnow() - timedelta(hours=1), datetime.utcnow())

            alert_stats = await alert_service.get_alert_statistics(time_range)
            metrics["alert_service"]["pending_alerts"] = alert_stats["summary"][
                "pending_alerts"
            ]
            metrics["alert_service"]["failed_alerts"] = alert_stats["summary"][
                "failed_alerts"
            ]

        except Exception as e:
            from ..core.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(f"Failed to get performance metrics: {e}")

        return metrics
