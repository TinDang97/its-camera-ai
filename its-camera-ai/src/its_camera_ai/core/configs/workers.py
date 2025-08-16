"""Configuration for background workers and task processing.

This module provides configuration settings for Celery workers,
Redis queues, and task scheduling for the ITS Camera AI system.
"""

from typing import Any

from pydantic import BaseModel, Field


class RedisConfig(BaseModel):
    """Redis configuration for Celery broker and result backend."""

    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    max_connections: int = Field(
        default=20,
        description="Maximum number of Redis connections"
    )
    timeout: int = Field(
        default=30,
        description="Redis connection timeout in seconds"
    )
    retry_on_timeout: bool = Field(
        default=True,
        description="Retry on timeout"
    )
    socket_keepalive: bool = Field(
        default=True,
        description="Enable socket keepalive"
    )
    socket_keepalive_options: dict[str, int] = Field(
        default_factory=lambda: {
            "TCP_KEEPIDLE": 1,
            "TCP_KEEPINTVL": 3,
            "TCP_KEEPCNT": 5,
        },
        description="Socket keepalive options"
    )


class CeleryConfig(BaseModel):
    """Celery configuration for distributed task processing."""

    # Broker settings
    broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL"
    )
    result_backend: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL"
    )

    # Task settings
    task_serializer: str = Field(
        default="json",
        description="Task serializer format"
    )
    accept_content: list[str] = Field(
        default=["json"],
        description="Accepted content types"
    )
    result_serializer: str = Field(
        default="json",
        description="Result serializer format"
    )
    timezone: str = Field(
        default="UTC",
        description="Timezone for scheduling"
    )
    enable_utc: bool = Field(
        default=True,
        description="Enable UTC timestamps"
    )

    # Worker settings
    worker_prefetch_multiplier: int = Field(
        default=1,
        description="Number of tasks to prefetch per worker"
    )
    task_acks_late: bool = Field(
        default=True,
        description="Acknowledge tasks after completion"
    )
    worker_disable_rate_limits: bool = Field(
        default=False,
        description="Disable rate limiting"
    )

    # Reliability settings
    task_reject_on_worker_lost: bool = Field(
        default=True,
        description="Reject tasks when worker is lost"
    )
    task_ignore_result: bool = Field(
        default=False,
        description="Ignore task results"
    )
    result_expires: int = Field(
        default=3600,
        description="Result expiration time in seconds"
    )

    # Monitoring settings
    worker_send_task_events: bool = Field(
        default=True,
        description="Send task events for monitoring"
    )
    task_send_sent_event: bool = Field(
        default=True,
        description="Send task sent events"
    )

    # Queue configuration
    task_default_queue: str = Field(
        default="default",
        description="Default task queue"
    )
    task_default_exchange: str = Field(
        default="default",
        description="Default exchange"
    )
    task_default_routing_key: str = Field(
        default="default",
        description="Default routing key"
    )


class QueueConfig(BaseModel):
    """Configuration for task queues."""

    name: str = Field(description="Queue name")
    routing_key: str = Field(description="Routing key")
    priority: int = Field(default=5, description="Queue priority")
    exchange: str = Field(default="default", description="Exchange name")
    durable: bool = Field(default=True, description="Queue durability")


class TaskScheduleConfig(BaseModel):
    """Configuration for scheduled tasks."""

    task: str = Field(description="Task name")
    schedule: float = Field(description="Schedule interval in seconds")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Task options"
    )
    args: list[Any] = Field(
        default_factory=list,
        description="Task arguments"
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Task keyword arguments"
    )


class WorkersConfig(BaseModel):
    """Comprehensive workers configuration."""

    # Redis configuration
    redis: RedisConfig = Field(
        default_factory=RedisConfig,
        description="Redis configuration"
    )

    # Celery configuration
    celery: CeleryConfig = Field(
        default_factory=CeleryConfig,
        description="Celery configuration"
    )

    # Queue definitions
    queues: list[QueueConfig] = Field(
        default_factory=lambda: [
            QueueConfig(
                name="realtime",
                routing_key="realtime",
                priority=9,
            ),
            QueueConfig(
                name="analytics",
                routing_key="analytics",
                priority=5,
            ),
            QueueConfig(
                name="maintenance",
                routing_key="maintenance",
                priority=1,
            ),
            QueueConfig(
                name="default",
                routing_key="default",
                priority=5,
            ),
        ],
        description="Queue configurations"
    )

    # Scheduled tasks
    beat_schedule: list[TaskScheduleConfig] = Field(
        default_factory=lambda: [
            TaskScheduleConfig(
                task="its_camera_ai.workers.aggregation_worker.create_1min_rollup",
                schedule=60.0,
            ),
            TaskScheduleConfig(
                task="its_camera_ai.workers.aggregation_worker.create_5min_rollup",
                schedule=300.0,
            ),
            TaskScheduleConfig(
                task="its_camera_ai.workers.aggregation_worker.create_hourly_rollup",
                schedule=3600.0,
            ),
            TaskScheduleConfig(
                task="its_camera_ai.workers.aggregation_worker.create_daily_rollup",
                schedule=86400.0,
            ),
            TaskScheduleConfig(
                task="its_camera_ai.workers.analytics_worker.detect_anomalies",
                schedule=900.0,
            ),
            TaskScheduleConfig(
                task="its_camera_ai.workers.maintenance_worker.cleanup_old_data",
                schedule=21600.0,
            ),
            TaskScheduleConfig(
                task="its_camera_ai.workers.maintenance_worker.optimize_database",
                schedule=86400.0,
                options={"countdown": 7200},
            ),
        ],
        description="Scheduled task configurations"
    )

    # Performance settings
    max_workers: int = Field(
        default=4,
        description="Maximum number of worker processes"
    )
    worker_concurrency: int = Field(
        default=2,
        description="Worker concurrency level"
    )
    worker_pool: str = Field(
        default="prefork",
        description="Worker pool type (prefork, gevent, eventlet)"
    )

    # Monitoring settings
    monitoring_enabled: bool = Field(
        default=True,
        description="Enable worker monitoring"
    )
    flower_port: int = Field(
        default=5555,
        description="Flower monitoring port"
    )
    flower_url_prefix: str = Field(
        default="/flower",
        description="Flower URL prefix"
    )

    # Logging settings
    log_level: str = Field(
        default="INFO",
        description="Worker log level"
    )
    log_file: str = Field(
        default="/var/log/its-camera-ai/workers.log",
        description="Worker log file path"
    )

    # Health check settings
    health_check_interval: int = Field(
        default=300,
        description="Health check interval in seconds"
    )

    # Resource limits
    worker_memory_limit: int = Field(
        default=2048,
        description="Worker memory limit in MB"
    )
    task_time_limit: int = Field(
        default=7200,
        description="Task time limit in seconds"
    )
    task_soft_time_limit: int = Field(
        default=3600,
        description="Task soft time limit in seconds"
    )


def get_workers_config() -> WorkersConfig:
    """Get workers configuration instance."""
    return WorkersConfig()
