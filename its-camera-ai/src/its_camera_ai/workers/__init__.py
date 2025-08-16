"""Background workers for ITS Camera AI System.

This module provides a comprehensive background processing architecture
for real-time data aggregation, analytics computation, and system maintenance.

Architecture:
- Celery for distributed task processing
- Redis as message broker and result backend
- TimescaleDB continuous aggregates for rollups
- Event-driven processing with Kafka integration
- Comprehensive monitoring and error handling

Components:
- AggregationWorker: Real-time to historical data rollups
- AnalyticsWorker: Complex analytics computations
- MaintenanceWorker: System cleanup and optimization
- EventWorker: Kafka event processing
"""

import logging
from typing import Any

from celery import Celery
from kombu import Queue

from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Configure Celery application
celery_app = Celery(
    "its-camera-ai-workers",
    broker=settings.redis.url,
    backend=settings.redis.url,
    include=[
        "its_camera_ai.workers.aggregation_worker",
        "its_camera_ai.workers.analytics_worker",
        "its_camera_ai.workers.maintenance_worker",
        "its_camera_ai.workers.event_worker",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Performance settings
    worker_prefetch_multiplier=1,  # For heavy tasks
    task_acks_late=True,
    worker_disable_rate_limits=False,

    # Reliability settings
    task_reject_on_worker_lost=True,
    task_ignore_result=False,
    result_expires=3600,  # 1 hour

    # Queue configuration
    task_routes={
        # High priority real-time tasks
        "its_camera_ai.workers.aggregation_worker.aggregate_detection_data": {
            "queue": "realtime",
            "priority": 9,
        },
        "its_camera_ai.workers.event_worker.process_ml_detection": {
            "queue": "realtime",
            "priority": 9,
        },

        # Medium priority analytics tasks
        "its_camera_ai.workers.analytics_worker.compute_traffic_metrics": {
            "queue": "analytics",
            "priority": 5,
        },
        "its_camera_ai.workers.analytics_worker.detect_anomalies": {
            "queue": "analytics",
            "priority": 5,
        },

        # Low priority maintenance tasks
        "its_camera_ai.workers.maintenance_worker.cleanup_old_data": {
            "queue": "maintenance",
            "priority": 1,
        },
        "its_camera_ai.workers.maintenance_worker.optimize_database": {
            "queue": "maintenance",
            "priority": 1,
        },
    },

    # Queue definitions
    task_queues=[
        Queue("realtime", routing_key="realtime"),
        Queue("analytics", routing_key="analytics"),
        Queue("maintenance", routing_key="maintenance"),
        Queue("default", routing_key="default"),
    ],

    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,

    # Beat schedule for periodic tasks
    beat_schedule={
        # Real-time aggregation every minute
        "aggregate-1min-rollup": {
            "task": "its_camera_ai.workers.aggregation_worker.create_1min_rollup",
            "schedule": 60.0,  # Every minute
        },

        # 5-minute aggregation
        "aggregate-5min-rollup": {
            "task": "its_camera_ai.workers.aggregation_worker.create_5min_rollup",
            "schedule": 300.0,  # Every 5 minutes
        },

        # Hourly aggregation
        "aggregate-hourly-rollup": {
            "task": "its_camera_ai.workers.aggregation_worker.create_hourly_rollup",
            "schedule": 3600.0,  # Every hour
        },

        # Daily aggregation
        "aggregate-daily-rollup": {
            "task": "its_camera_ai.workers.aggregation_worker.create_daily_rollup",
            "schedule": 86400.0,  # Every day at midnight
        },

        # Anomaly detection every 15 minutes
        "detect-anomalies": {
            "task": "its_camera_ai.workers.analytics_worker.detect_anomalies",
            "schedule": 900.0,  # Every 15 minutes
        },

        # Data cleanup every 6 hours
        "cleanup-old-data": {
            "task": "its_camera_ai.workers.maintenance_worker.cleanup_old_data",
            "schedule": 21600.0,  # Every 6 hours
        },

        # Database optimization daily
        "optimize-database": {
            "task": "its_camera_ai.workers.maintenance_worker.optimize_database",
            "schedule": 86400.0,  # Daily at 2 AM
            "options": {"countdown": 7200},  # 2 hours delay from midnight
        },
    },
)


class WorkerError(Exception):
    """Base exception for worker errors."""
    pass


class TaskRetryError(WorkerError):
    """Exception raised when a task should be retried."""
    pass


class TaskFailureError(WorkerError):
    """Exception raised when a task has permanently failed."""
    pass


def setup_worker_logging() -> None:
    """Configure logging for workers."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/var/log/its-camera-ai/workers.log"),
        ]
    )


def get_worker_status() -> dict[str, Any]:
    """Get worker status information."""
    from celery import current_app

    inspect = current_app.control.inspect()

    return {
        "active_tasks": inspect.active(),
        "scheduled_tasks": inspect.scheduled(),
        "reserved_tasks": inspect.reserved(),
        "stats": inspect.stats(),
        "registered_tasks": list(current_app.tasks.keys()),
    }
