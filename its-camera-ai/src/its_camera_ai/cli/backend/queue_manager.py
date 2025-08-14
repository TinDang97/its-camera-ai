"""Queue management for CLI backend integration.

Provides interface to Redis queue systems for background task management,
event processing, and service communication.
"""

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

from ...core.config import Settings, get_settings
from ...core.logging import get_logger
from ...flow.redis_queue_manager import (
    QueueConfig,
    QueueType,
    RedisQueueManager,
)

logger = get_logger(__name__)


class CLIQueueManager:
    """Queue manager for CLI operations.

    Provides high-level queue operations for:
    - Background task management
    - Event processing
    - Service communication
    - Job scheduling
    """

    def __init__(self, settings: Settings = None):
        """Initialize CLI queue manager.

        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self._redis_manager = RedisQueueManager(
            redis_url=self.settings.redis_queue.url,
            pool_size=self.settings.redis_queue.pool_size,
            timeout=self.settings.redis_queue.timeout,
            retry_on_failure=self.settings.redis_queue.retry_on_failure,
        )

        # Task handlers
        self._task_handlers: dict[str, Callable] = {}

        # Queue configurations
        self._queue_configs = self._setup_queue_configs()

        self._initialized = False

        logger.info("CLI queue manager initialized")

    def _setup_queue_configs(self) -> dict[str, QueueConfig]:
        """Setup default queue configurations."""
        return {
            "cli_tasks": QueueConfig(
                name="cli_tasks",
                queue_type=QueueType.STREAM,
                consumer_group="cli_workers",
                consumer_name="cli",
                batch_size=10,
                max_length=1000,
            ),
            "background_jobs": QueueConfig(
                name="background_jobs",
                queue_type=QueueType.STREAM,
                consumer_group="job_workers",
                consumer_name="cli",
                batch_size=5,
                max_length=500,
            ),
            "events": QueueConfig(
                name="system_events", queue_type=QueueType.PUBSUB, batch_size=1
            ),
            "notifications": QueueConfig(
                name="cli_notifications",
                queue_type=QueueType.LIST,
                batch_size=20,
                max_length=100,
            ),
        }

    async def initialize(self) -> None:
        """Initialize queue manager and create queues."""
        if self._initialized:
            return

        try:
            await self._redis_manager.connect()

            # Create configured queues
            for config in self._queue_configs.values():
                success = await self._redis_manager.create_queue(config)
                if success:
                    logger.debug(f"Created queue: {config.name}")
                else:
                    logger.warning(f"Failed to create queue: {config.name}")

            self._initialized = True
            logger.info("Queue manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize queue manager: {e}")
            raise

    async def close(self) -> None:
        """Close queue manager connections."""
        if self._initialized:
            await self._redis_manager.disconnect()
            self._initialized = False
            logger.info("Queue manager closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    # Task Management

    async def submit_task(
        self,
        task_type: str,
        task_data: dict[str, Any],
        priority: int = 0,
        queue: str = "cli_tasks",
    ) -> str:
        """Submit a background task.

        Args:
            task_type: Type of task to execute
            task_data: Task parameters and data
            priority: Task priority (higher = more important)
            queue: Queue name to submit to

        Returns:
            Task ID
        """
        if not self._initialized:
            await self.initialize()

        task_payload = {
            "task_type": task_type,
            "task_data": task_data,
            "submitted_at": time.time(),
            "submitted_by": "cli",
        }

        task_json = json.dumps(task_payload)
        task_id = await self._redis_manager.enqueue(
            queue_name=queue,
            data=task_json.encode("utf-8"),
            priority=priority,
            metadata={"task_type": task_type},
        )

        logger.info(f"Submitted task {task_type} with ID: {task_id}")
        return task_id

    async def submit_batch_tasks(
        self, tasks: list[dict[str, Any]], queue: str = "cli_tasks"
    ) -> list[str]:
        """Submit multiple tasks in batch.

        Args:
            tasks: List of task dictionaries
            queue: Queue name to submit to

        Returns:
            List of task IDs
        """
        if not self._initialized:
            await self.initialize()

        task_payloads = []
        priorities = []
        metadata_batch = []

        for task in tasks:
            payload = {
                "task_type": task["task_type"],
                "task_data": task.get("task_data", {}),
                "submitted_at": time.time(),
                "submitted_by": "cli",
            }

            task_payloads.append(json.dumps(payload).encode("utf-8"))
            priorities.append(task.get("priority", 0))
            metadata_batch.append({"task_type": task["task_type"]})

        task_ids = await self._redis_manager.enqueue_batch(
            queue_name=queue,
            data_batch=task_payloads,
            priorities=priorities,
            metadata_batch=metadata_batch,
        )

        logger.info(f"Submitted {len(tasks)} tasks in batch")
        return task_ids

    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """Register handler for a task type.

        Args:
            task_type: Type of task to handle
            handler: Async function to handle the task
        """
        self._task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")

    async def process_tasks(
        self, queue: str = "cli_tasks", max_tasks: int | None = None, timeout: int = 30
    ) -> int:
        """Process tasks from queue.

        Args:
            queue: Queue name to process
            max_tasks: Maximum number of tasks to process
            timeout: Processing timeout per task

        Returns:
            Number of tasks processed
        """
        if not self._initialized:
            await self.initialize()

        processed_count = 0

        try:
            while max_tasks is None or processed_count < max_tasks:
                # Get task from queue
                result = await self._redis_manager.dequeue(
                    queue_name=queue, timeout_ms=timeout * 1000
                )

                if not result:
                    break  # No more tasks

                task_id, task_data = result

                try:
                    # Parse task data
                    task_payload = json.loads(task_data.decode("utf-8"))
                    task_type = task_payload.get("task_type")

                    if task_type in self._task_handlers:
                        # Execute task handler
                        start_time = time.time()

                        await self._task_handlers[task_type](task_payload["task_data"])

                        processing_time = (time.time() - start_time) * 1000

                        # Acknowledge successful processing
                        await self._redis_manager.acknowledge(
                            queue_name=queue,
                            message_id=task_id,
                            processing_time_ms=processing_time,
                        )

                        processed_count += 1
                        logger.debug(
                            f"Processed task {task_type} in {processing_time:.2f}ms"
                        )
                    else:
                        # No handler for task type
                        await self._redis_manager.reject(
                            queue_name=queue,
                            message_id=task_id,
                            reason=f"No handler for task type: {task_type}",
                        )

                        logger.warning(f"No handler for task type: {task_type}")

                except Exception as e:
                    # Task processing failed
                    await self._redis_manager.reject(
                        queue_name=queue,
                        message_id=task_id,
                        reason=f"Task processing failed: {str(e)}",
                    )

                    logger.error(f"Task processing failed: {e}")

        except Exception as e:
            logger.error(f"Task processing error: {e}")

        logger.info(f"Processed {processed_count} tasks from queue {queue}")
        return processed_count

    # Event Management

    async def publish_event(
        self, event_type: str, event_data: dict[str, Any], queue: str = "events"
    ) -> bool:
        """Publish an event.

        Args:
            event_type: Type of event
            event_data: Event data
            queue: Queue/channel to publish to

        Returns:
            Success status
        """
        if not self._initialized:
            await self.initialize()

        event_payload = {
            "event_type": event_type,
            "event_data": event_data,
            "timestamp": time.time(),
            "source": "cli",
        }

        try:
            event_json = json.dumps(event_payload)
            await self._redis_manager.enqueue(
                queue_name=queue,
                data=event_json.encode("utf-8"),
                metadata={"event_type": event_type},
            )

            logger.debug(f"Published event {event_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish event {event_type}: {e}")
            return False

    async def subscribe_to_events(
        self,
        event_handler: Callable,
        event_types: list[str] | None = None,
        queue: str = "events",
    ) -> None:
        """Subscribe to events and process them.

        Args:
            event_handler: Function to handle events
            event_types: List of event types to filter (None = all)
            queue: Queue/channel to subscribe to
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Subscribing to events on queue {queue}")

        try:
            while True:
                result = await self._redis_manager.dequeue(
                    queue_name=queue, timeout_ms=5000  # 5 second timeout
                )

                if result:
                    event_id, event_data = result

                    try:
                        event_payload = json.loads(event_data.decode("utf-8"))
                        event_type = event_payload.get("event_type")

                        # Filter by event types if specified
                        if event_types is None or event_type in event_types:
                            await event_handler(event_payload)

                        # Acknowledge event processing
                        await self._redis_manager.acknowledge(
                            queue_name=queue, message_id=event_id
                        )

                    except Exception as e:
                        logger.error(f"Event processing failed: {e}")
                        await self._redis_manager.reject(
                            queue_name=queue,
                            message_id=event_id,
                            reason=f"Event processing failed: {str(e)}",
                        )

        except asyncio.CancelledError:
            logger.info("Event subscription cancelled")
        except Exception as e:
            logger.error(f"Event subscription error: {e}")

    # Notification Management

    async def send_notification(
        self, message: str, level: str = "info", metadata: dict[str, Any] | None = None
    ) -> bool:
        """Send a notification message.

        Args:
            message: Notification message
            level: Notification level (info, warning, error)
            metadata: Additional metadata

        Returns:
            Success status
        """
        notification = {
            "message": message,
            "level": level,
            "timestamp": time.time(),
            "source": "cli",
            "metadata": metadata or {},
        }

        try:
            notification_json = json.dumps(notification)
            await self._redis_manager.enqueue(
                queue_name="notifications", data=notification_json.encode("utf-8")
            )

            logger.debug(f"Sent notification: {message}")
            return True

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    async def get_notifications(self, count: int = 10) -> list[dict[str, Any]]:
        """Get recent notifications.

        Args:
            count: Maximum number of notifications to retrieve

        Returns:
            List of notification dictionaries
        """
        if not self._initialized:
            await self.initialize()

        notifications = []

        try:
            for _ in range(count):
                result = await self._redis_manager.dequeue(
                    queue_name="notifications", timeout_ms=100  # Short timeout
                )

                if result:
                    _, notification_data = result
                    notification = json.loads(notification_data.decode("utf-8"))
                    notifications.append(notification)
                else:
                    break  # No more notifications

        except Exception as e:
            logger.error(f"Failed to get notifications: {e}")

        return notifications

    # Queue Management

    async def get_queue_status(self, queue: str) -> dict[str, Any] | None:
        """Get queue status information.

        Args:
            queue: Queue name

        Returns:
            Queue status dictionary
        """
        if not self._initialized:
            await self.initialize()

        metrics = await self._redis_manager.get_queue_metrics(queue)

        if metrics:
            return {
                "name": metrics.queue_name,
                "pending_count": metrics.pending_count,
                "processing_count": metrics.processing_count,
                "completed_count": metrics.completed_count,
                "failed_count": metrics.failed_count,
                "avg_processing_time_ms": metrics.avg_processing_time_ms,
                "throughput_fps": metrics.throughput_fps,
                "memory_usage_bytes": metrics.memory_usage_bytes,
                "last_updated": metrics.last_updated,
            }

        return None

    async def get_all_queue_status(self) -> dict[str, dict[str, Any]]:
        """Get status for all managed queues.

        Returns:
            Dictionary mapping queue names to status information
        """
        status = {}

        for queue_name in self._queue_configs:
            queue_status = await self.get_queue_status(queue_name)
            if queue_status:
                status[queue_name] = queue_status

        return status

    async def purge_queue(self, queue: str, force: bool = False) -> int:
        """Purge all messages from a queue.

        Args:
            queue: Queue name to purge
            force: Force purge even if active

        Returns:
            Number of messages purged
        """
        if not self._initialized:
            await self.initialize()

        purged_count = await self._redis_manager.purge_queue(queue, force=force)

        if purged_count > 0:
            logger.info(f"Purged {purged_count} messages from queue {queue}")

        return purged_count

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on queue system.

        Returns:
            Health status information
        """
        if not self._initialized:
            return {
                "status": "not_initialized",
                "error": "Queue manager not initialized",
            }

        return await self._redis_manager.health_check()

    def get_manager_status(self) -> dict[str, Any]:
        """Get queue manager status.

        Returns:
            Manager status information
        """
        return {
            "initialized": self._initialized,
            "redis_status": self._redis_manager.get_status().value,
            "configured_queues": list(self._queue_configs.keys()),
            "registered_handlers": list(self._task_handlers.keys()),
            "redis_url": self.settings.redis_queue.url,
        }
