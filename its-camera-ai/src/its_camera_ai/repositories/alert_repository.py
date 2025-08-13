"""Alert repository for alert notification data access.

Provides specialized methods for alert notification management,
delivery tracking, and statistics with optimized queries.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from ..models.analytics import AlertNotification
from .base_repository import BaseRepository

logger = get_logger(__name__)


class AlertRepository(BaseRepository[AlertNotification]):
    """Repository for alert notification data access operations.

    Specialized methods for alert delivery tracking, status management,
    and analytics with optimized queries for notification systems.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        super().__init__(session_factory, AlertNotification)

    async def create_notification(
        self,
        alert_type: str,
        reference_id: str,
        notification_channel: str,
        recipient: str,
        priority: str,
        subject: str | None = None,
        message_content: str = "",
        **kwargs: Any,
    ) -> AlertNotification:
        """Create a new alert notification.

        Args:
            alert_type: Type of alert (violation/anomaly/incident)
            reference_id: ID of referenced entity
            notification_channel: Notification channel name
            recipient: Recipient identifier
            priority: Alert priority
            subject: Alert subject/title
            message_content: Alert message content
            **kwargs: Additional notification fields

        Returns:
            Created AlertNotification instance

        Raises:
            DatabaseError: If creation fails
        """
        return await self.create(
            alert_type=alert_type,
            reference_id=reference_id,
            notification_channel=notification_channel,
            recipient=recipient,
            priority=priority,
            subject=subject,
            message_content=message_content,
            status="pending",
            delivery_attempts=0,
            created_time=datetime.now(),
            **kwargs,
        )

    async def get_by_reference(
        self, alert_type: str, reference_id: str, limit: int = 100
    ) -> list[AlertNotification]:
        """Get notifications by alert type and reference ID.

        Args:
            alert_type: Type of alert
            reference_id: Referenced entity ID
            limit: Maximum number of results

        Returns:
            List of notifications for the referenced entity

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = (
                    select(AlertNotification)
                    .where(
                        and_(
                            AlertNotification.alert_type == alert_type,
                            AlertNotification.reference_id == reference_id,
                        )
                    )
                    .order_by(AlertNotification.created_time.desc())
                    .limit(limit)
                )

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get notifications by reference",
                    alert_type=alert_type,
                    reference_id=reference_id,
                    error=str(e),
                )
                raise DatabaseError("Notification retrieval failed", cause=e) from e

    async def get_by_status(
        self,
        status: str,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[AlertNotification]:
        """Get notifications by status.

        Args:
            status: Notification status to filter by
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of notifications with specified status

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = (
                    select(AlertNotification)
                    .where(AlertNotification.status == status)
                    .order_by(AlertNotification.created_time.desc())
                    .limit(limit)
                    .offset(offset)
                )

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get notifications by status",
                    status=status,
                    error=str(e),
                )
                raise DatabaseError("Notification retrieval failed", cause=e) from e

    async def get_failed_retryable(
        self, max_attempts: int = 3, limit: int = 50
    ) -> list[AlertNotification]:
        """Get failed notifications that can be retried.

        Args:
            max_attempts: Maximum retry attempts
            limit: Maximum number of results

        Returns:
            List of failed notifications eligible for retry

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = (
                    select(AlertNotification)
                    .where(
                        and_(
                            AlertNotification.status == "failed",
                            AlertNotification.delivery_attempts < max_attempts,
                        )
                    )
                    .order_by(AlertNotification.created_time.desc())
                    .limit(limit)
                )

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get failed retryable notifications",
                    max_attempts=max_attempts,
                    error=str(e),
                )
                raise DatabaseError("Notification retrieval failed", cause=e) from e

    async def check_recent_similar_alerts(
        self,
        alert_type: str,
        camera_id: str,
        event_type: str,
        since: datetime,
    ) -> list[AlertNotification]:
        """Check for recent similar alerts for cooldown logic.

        Args:
            alert_type: Type of alert
            camera_id: Camera identifier
            event_type: Event type (violation_type/anomaly_type)
            since: Look for alerts since this time

        Returns:
            List of recent similar notifications

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(AlertNotification).where(
                    and_(
                        AlertNotification.alert_type == alert_type,
                        AlertNotification.created_time >= since,
                        AlertNotification.status.in_(["delivered", "pending"]),
                        AlertNotification.delivery_details.op("@>")(
                            {f"{alert_type}_type": event_type, "camera_id": camera_id}
                        ),
                    )
                )

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to check recent similar alerts",
                    alert_type=alert_type,
                    camera_id=camera_id,
                    event_type=event_type,
                    error=str(e),
                )
                raise DatabaseError(
                    "Recent alerts check failed", cause=e
                ) from e

    async def get_statistics(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        camera_id: str | None = None,
    ) -> dict[str, Any]:
        """Get alert delivery statistics.

        Args:
            start_time: Optional start time filter
            end_time: Optional end time filter
            camera_id: Optional camera filter

        Returns:
            Alert statistics dictionary

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(AlertNotification)

                conditions = []
                if start_time:
                    conditions.append(AlertNotification.created_time >= start_time)
                if end_time:
                    conditions.append(AlertNotification.created_time <= end_time)
                if camera_id:
                    conditions.append(
                        AlertNotification.delivery_details.op("@>")(
                            {"camera_id": camera_id}
                        )
                    )

                if conditions:
                    query = query.where(and_(*conditions))

                result = await session.execute(query)
                notifications = result.scalars().all()

                # Calculate statistics
                total_alerts = len(notifications)
                delivered_alerts = sum(
                    1 for n in notifications if n.status == "delivered"
                )
                failed_alerts = sum(1 for n in notifications if n.status == "failed")
                pending_alerts = sum(1 for n in notifications if n.status == "pending")

                # Group by channel
                channel_stats = {}
                for notification in notifications:
                    channel = notification.notification_channel
                    if channel not in channel_stats:
                        channel_stats[channel] = {
                            "total": 0,
                            "delivered": 0,
                            "failed": 0,
                            "pending": 0,
                        }

                    channel_stats[channel]["total"] += 1
                    channel_stats[channel][notification.status] += 1

                # Group by priority
                priority_stats = {}
                for notification in notifications:
                    priority = notification.priority
                    if priority not in priority_stats:
                        priority_stats[priority] = {
                            "total": 0,
                            "delivered": 0,
                            "failed": 0,
                            "pending": 0,
                        }

                    priority_stats[priority]["total"] += 1
                    priority_stats[priority][notification.status] += 1

                return {
                    "summary": {
                        "total_alerts": total_alerts,
                        "delivered_alerts": delivered_alerts,
                        "failed_alerts": failed_alerts,
                        "pending_alerts": pending_alerts,
                        "delivery_rate": delivered_alerts / total_alerts
                        if total_alerts > 0
                        else 0.0,
                    },
                    "by_channel": channel_stats,
                    "by_priority": priority_stats,
                    "time_range": {
                        "start": start_time,
                        "end": end_time,
                    },
                }

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get alert statistics",
                    start_time=start_time.isoformat() if start_time else None,
                    end_time=end_time.isoformat() if end_time else None,
                    camera_id=camera_id,
                    error=str(e),
                )
                raise DatabaseError("Statistics retrieval failed", cause=e) from e

    async def update_delivery_status(
        self,
        notification_id: str,
        status: str,
        delivery_attempts: int | None = None,
        sent_time: datetime | None = None,
        external_id: str | None = None,
        delivery_details: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> AlertNotification | None:
        """Update notification delivery status and metadata.

        Args:
            notification_id: Notification ID to update
            status: New status
            delivery_attempts: Updated attempt count
            sent_time: Delivery timestamp
            external_id: External system notification ID
            delivery_details: Delivery service response details
            error_message: Error message if delivery failed

        Returns:
            Updated notification or None if not found

        Raises:
            DatabaseError: If update fails
        """
        update_data = {"status": status}

        if delivery_attempts is not None:
            update_data["delivery_attempts"] = delivery_attempts
            update_data["last_attempt_time"] = datetime.now()

        if sent_time is not None:
            update_data["sent_time"] = sent_time

        if external_id is not None:
            update_data["external_id"] = external_id

        if delivery_details is not None:
            update_data["delivery_details"] = delivery_details

        if error_message is not None:
            update_data["error_message"] = error_message
        elif status == "delivered":
            # Clear error message on successful delivery
            update_data["error_message"] = None

        return await self.update(notification_id, **update_data)

    async def acknowledge_notification(
        self,
        notification_id: str,
        acknowledged_by: str,
        response_action: str | None = None,
        response_notes: str | None = None,
    ) -> AlertNotification | None:
        """Acknowledge a notification.

        Args:
            notification_id: Notification ID to acknowledge
            acknowledged_by: User acknowledging the notification
            response_action: Action taken in response
            response_notes: Additional response notes

        Returns:
            Updated notification or None if not found

        Raises:
            DatabaseError: If update fails
        """
        return await self.update(
            notification_id,
            acknowledged_time=datetime.now(),
            acknowledged_by=acknowledged_by,
            response_action=response_action,
            response_notes=response_notes,
        )
