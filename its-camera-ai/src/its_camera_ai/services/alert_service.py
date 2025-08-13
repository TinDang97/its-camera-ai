"""Alert Generation Service for traffic violations and anomalies.

This service handles the generation, delivery, and management of alerts
for traffic violations, anomalies, and incidents with support for multiple
notification channels and configurable alert rules.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import Settings
from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from ..models.analytics import (
    AlertNotification,
    RuleViolation,
    TrafficAnomaly,
)
from .base_service import BaseAsyncService

logger = get_logger(__name__)


class AlertServiceError(Exception):
    """Alert service specific exceptions."""

    pass


class NotificationChannel:
    """Base class for notification channels."""

    def __init__(self, channel_type: str, config: dict[str, Any]):
        self.channel_type = channel_type
        self.config = config

    async def send_notification(
        self,
        recipient: str,
        subject: str,
        message: str,
        priority: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send notification through this channel.

        Args:
            recipient: Recipient identifier
            subject: Notification subject
            message: Notification message
            priority: Alert priority level
            metadata: Additional metadata

        Returns:
            Delivery result with status and details
        """
        raise NotImplementedError("Subclasses must implement send_notification")


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("email", config)
        self.smtp_server = config.get("smtp_server", "localhost")
        self.smtp_port = config.get("smtp_port", 587)
        self.username = config.get("username")
        self.password = config.get("password")
        self.from_email = config.get("from_email", "alerts@its-camera-ai.local")

    async def send_notification(
        self,
        recipient: str,
        subject: str,
        message: str,
        priority: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send email notification."""
        try:
            # In a real implementation, this would use an actual email service
            # For now, we'll simulate the email sending

            logger.info(
                "Sending email alert",
                recipient=recipient,
                subject=subject,
                priority=priority,
            )

            # Simulate email delivery
            await asyncio.sleep(0.1)  # Simulate network delay

            return {
                "status": "delivered",
                "channel": "email",
                "recipient": recipient,
                "delivery_id": f"email_{datetime.utcnow().timestamp()}",
                "delivered_at": datetime.utcnow(),
                "details": {
                    "subject": subject,
                    "message_length": len(message),
                    "priority": priority,
                },
            }

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return {
                "status": "failed",
                "channel": "email",
                "recipient": recipient,
                "error": str(e),
                "failed_at": datetime.utcnow(),
            }


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel."""

    def __init__(self, config: dict[str, Any]):
        super().__init__("webhook", config)
        self.webhook_url = config.get("url")
        self.headers = config.get("headers", {})
        self.timeout = config.get("timeout", 30)

    async def send_notification(
        self,
        recipient: str,
        subject: str,
        message: str,
        priority: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send webhook notification."""
        try:
            payload = {
                "recipient": recipient,
                "subject": subject,
                "message": message,
                "priority": priority,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {},
            }

            logger.info(
                "Sending webhook alert",
                recipient=recipient,
                webhook_url=self.webhook_url,
                priority=priority,
            )

            # In a real implementation, this would make an HTTP request
            # For now, we'll simulate the webhook call
            await asyncio.sleep(0.1)  # Simulate network delay

            return {
                "status": "delivered",
                "channel": "webhook",
                "recipient": recipient,
                "delivery_id": f"webhook_{datetime.utcnow().timestamp()}",
                "delivered_at": datetime.utcnow(),
                "details": {
                    "webhook_url": self.webhook_url,
                    "payload_size": len(json.dumps(payload)),
                    "priority": priority,
                },
            }

        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return {
                "status": "failed",
                "channel": "webhook",
                "recipient": recipient,
                "error": str(e),
                "failed_at": datetime.utcnow(),
            }


class AlertRuleEngine:
    """Alert rule evaluation engine for determining when to send alerts."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._alert_rules = self._load_default_alert_rules()

    def _load_default_alert_rules(self) -> dict[str, dict[str, Any]]:
        """Load default alert rules configuration."""
        return {
            "violation_alerts": {
                "enabled": True,
                "severity_thresholds": {
                    "low": False,  # Don't alert on low severity
                    "medium": True,  # Alert on medium severity
                    "high": True,  # Alert on high severity
                    "critical": True,  # Alert on critical severity
                },
                "cooldown_minutes": 5,  # Cooldown between similar alerts
                "batch_alerts": True,  # Batch similar alerts
                "batch_window_minutes": 10,
            },
            "anomaly_alerts": {
                "enabled": True,
                "score_threshold": 0.7,  # Only alert on high anomaly scores
                "severity_thresholds": {
                    "low": False,
                    "medium": False,
                    "high": True,
                    "critical": True,
                },
                "cooldown_minutes": 15,
                "consecutive_anomalies": 2,  # Require 2 consecutive anomalies
            },
            "pattern_alerts": {
                "enabled": True,
                "traffic_spike_threshold": 3.0,  # 3x normal traffic
                "congestion_duration_minutes": 10,
                "speed_drop_threshold": 0.5,  # 50% speed drop
            },
        }

    def should_alert_for_violation(self, violation: RuleViolation) -> bool:
        """Determine if a violation should trigger an alert."""
        rules = self._alert_rules.get("violation_alerts", {})

        if not rules.get("enabled", False):
            return False

        # Check severity threshold
        severity_thresholds = rules.get("severity_thresholds", {})
        if not severity_thresholds.get(violation.severity, False):
            return False

        return True

    def should_alert_for_anomaly(self, anomaly: TrafficAnomaly) -> bool:
        """Determine if an anomaly should trigger an alert."""
        rules = self._alert_rules.get("anomaly_alerts", {})

        if not rules.get("enabled", False):
            return False

        # Check anomaly score threshold
        score_threshold = rules.get("score_threshold", 0.7)
        if anomaly.anomaly_score < score_threshold:
            return False

        # Check severity threshold
        severity_thresholds = rules.get("severity_thresholds", {})
        if not severity_thresholds.get(anomaly.severity, False):
            return False

        return True

    def get_alert_priority(self, entity: RuleViolation | TrafficAnomaly) -> str:
        """Get alert priority based on entity severity and type."""
        severity_priority_map = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "critical": "critical",
        }

        return severity_priority_map.get(entity.severity, "medium")

    def get_cooldown_period(self, alert_type: str) -> timedelta:
        """Get cooldown period for alert type."""
        if alert_type == "violation":
            minutes = self._alert_rules.get("violation_alerts", {}).get(
                "cooldown_minutes", 5
            )
        elif alert_type == "anomaly":
            minutes = self._alert_rules.get("anomaly_alerts", {}).get(
                "cooldown_minutes", 15
            )
        else:
            minutes = 5

        return timedelta(minutes=minutes)


class AlertService(BaseAsyncService):
    """Comprehensive alert service for traffic monitoring notifications.

    Handles alert generation, delivery, and management for traffic violations,
    anomalies, and incidents with support for multiple notification channels.
    """

    def __init__(self, session: AsyncSession, settings: Settings):
        super().__init__(session, AlertNotification)
        self.settings = settings
        self.rule_engine = AlertRuleEngine(settings)
        self.notification_channels = self._initialize_notification_channels()

    def _initialize_notification_channels(self) -> dict[str, NotificationChannel]:
        """Initialize available notification channels."""
        channels = {}

        # Email channel
        email_config = {
            "smtp_server": "localhost",
            "smtp_port": 587,
            "username": None,
            "password": None,
            "from_email": "alerts@its-camera-ai.local",
        }
        channels["email"] = EmailNotificationChannel(email_config)

        # Webhook channel
        webhook_config = {
            "url": "http://localhost:8080/api/v1/webhooks/alerts",
            "headers": {"Content-Type": "application/json"},
            "timeout": 30,
        }
        channels["webhook"] = WebhookNotificationChannel(webhook_config)

        return channels

    async def process_violation_alert(
        self,
        violation: RuleViolation,
        recipients: list[str],
        notification_channels: list[str] = None,
    ) -> list[dict[str, Any]]:
        """Process and send alerts for a traffic violation.

        Args:
            violation: Traffic violation record
            recipients: List of recipient identifiers
            notification_channels: List of channels to use (default: all)

        Returns:
            List of delivery results
        """
        if not self.rule_engine.should_alert_for_violation(violation):
            logger.debug(f"Violation {violation.id} does not meet alert criteria")
            return []

        # Check for cooldown period
        if await self._is_in_cooldown(
            "violation", violation.camera_id, violation.violation_type
        ):
            logger.debug(
                f"Violation alert in cooldown period for camera {violation.camera_id}"
            )
            return []

        # Generate alert content
        subject, message = self._generate_violation_alert_content(violation)
        priority = self.rule_engine.get_alert_priority(violation)

        # Send alerts through specified channels
        channels = notification_channels or list(self.notification_channels.keys())
        delivery_results = []

        for recipient in recipients:
            for channel_name in channels:
                result = await self._send_alert(
                    alert_type="violation",
                    reference_id=violation.id,
                    channel_name=channel_name,
                    recipient=recipient,
                    subject=subject,
                    message=message,
                    priority=priority,
                    metadata={
                        "violation_type": violation.violation_type,
                        "camera_id": violation.camera_id,
                        "severity": violation.severity,
                    },
                )
                delivery_results.append(result)

        # Update cooldown tracking
        await self._update_cooldown(
            "violation", violation.camera_id, violation.violation_type
        )

        return delivery_results

    async def process_anomaly_alert(
        self,
        anomaly: TrafficAnomaly,
        recipients: list[str],
        notification_channels: list[str] = None,
    ) -> list[dict[str, Any]]:
        """Process and send alerts for a traffic anomaly.

        Args:
            anomaly: Traffic anomaly record
            recipients: List of recipient identifiers
            notification_channels: List of channels to use (default: all)

        Returns:
            List of delivery results
        """
        if not self.rule_engine.should_alert_for_anomaly(anomaly):
            logger.debug(f"Anomaly {anomaly.id} does not meet alert criteria")
            return []

        # Check for cooldown period
        if await self._is_in_cooldown(
            "anomaly", anomaly.camera_id, anomaly.anomaly_type
        ):
            logger.debug(
                f"Anomaly alert in cooldown period for camera {anomaly.camera_id}"
            )
            return []

        # Generate alert content
        subject, message = self._generate_anomaly_alert_content(anomaly)
        priority = self.rule_engine.get_alert_priority(anomaly)

        # Send alerts through specified channels
        channels = notification_channels or list(self.notification_channels.keys())
        delivery_results = []

        for recipient in recipients:
            for channel_name in channels:
                result = await self._send_alert(
                    alert_type="anomaly",
                    reference_id=anomaly.id,
                    channel_name=channel_name,
                    recipient=recipient,
                    subject=subject,
                    message=message,
                    priority=priority,
                    metadata={
                        "anomaly_type": anomaly.anomaly_type,
                        "camera_id": anomaly.camera_id,
                        "severity": anomaly.severity,
                        "score": anomaly.anomaly_score,
                    },
                )
                delivery_results.append(result)

        # Update cooldown tracking
        await self._update_cooldown("anomaly", anomaly.camera_id, anomaly.anomaly_type)

        return delivery_results

    async def _send_alert(
        self,
        alert_type: str,
        reference_id: str,
        channel_name: str,
        recipient: str,
        subject: str,
        message: str,
        priority: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send individual alert through specified channel.

        Args:
            alert_type: Type of alert (violation/anomaly)
            reference_id: ID of referenced entity
            channel_name: Notification channel name
            recipient: Recipient identifier
            subject: Alert subject
            message: Alert message
            priority: Alert priority
            metadata: Additional metadata

        Returns:
            Delivery result with status and details
        """
        try:
            # Create notification record
            notification = AlertNotification(
                alert_type=alert_type,
                reference_id=reference_id,
                notification_channel=channel_name,
                recipient=recipient,
                priority=priority,
                subject=subject,
                message_content=message,
                status="pending",
                delivery_attempts=0,
            )

            self.session.add(notification)
            await self.session.commit()
            await self.session.refresh(notification)

            # Send through notification channel
            channel = self.notification_channels.get(channel_name)
            if not channel:
                raise AlertServiceError(f"Unknown notification channel: {channel_name}")

            # Update attempt count
            notification.delivery_attempts += 1
            notification.last_attempt_time = datetime.utcnow()

            # Send notification
            delivery_result = await channel.send_notification(
                recipient=recipient,
                subject=subject,
                message=message,
                priority=priority,
                metadata=metadata,
            )

            # Update notification status based on result
            if delivery_result.get("status") == "delivered":
                notification.status = "delivered"
                notification.sent_time = delivery_result.get(
                    "delivered_at", datetime.utcnow()
                )
                notification.external_id = delivery_result.get("delivery_id")
                notification.delivery_details = delivery_result.get("details", {})
            else:
                notification.status = "failed"
                notification.error_message = delivery_result.get(
                    "error", "Unknown error"
                )

            await self.session.commit()

            logger.info(
                "Alert sent",
                alert_type=alert_type,
                channel=channel_name,
                recipient=recipient,
                status=notification.status,
                priority=priority,
            )

            return {
                "notification_id": notification.id,
                "alert_type": alert_type,
                "channel": channel_name,
                "recipient": recipient,
                "status": notification.status,
                "delivery_result": delivery_result,
            }

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

            # Update notification status if record exists
            if "notification" in locals():
                notification.status = "failed"
                notification.error_message = str(e)
                await self.session.commit()

            return {
                "alert_type": alert_type,
                "channel": channel_name,
                "recipient": recipient,
                "status": "failed",
                "error": str(e),
            }

    def _generate_violation_alert_content(
        self, violation: RuleViolation
    ) -> tuple[str, str]:
        """Generate alert content for traffic violation."""
        violation_type_names = {
            "speeding": "Speed Limit Violation",
            "red_light": "Red Light Violation",
            "wrong_way": "Wrong Way Driving",
            "illegal_turn": "Illegal Turn",
            "illegal_parking": "Illegal Parking",
            "stop_sign": "Stop Sign Violation",
            "lane_violation": "Lane Violation",
        }

        violation_name = violation_type_names.get(
            violation.violation_type, violation.violation_type.title()
        )

        subject = f"🚨 {violation.severity.title()} {violation_name} Detected"

        message = f"""
Traffic Violation Alert

Type: {violation_name}
Severity: {violation.severity.upper()}
Location: Camera {violation.camera_id}
Time: {violation.detection_time.strftime("%Y-%m-%d %H:%M:%S")}
"""

        if violation.license_plate:
            message += f"Vehicle: {violation.license_plate}\n"

        if violation.measured_value and violation.threshold_value:
            if violation.violation_type == "speeding":
                message += f"Speed: {violation.measured_value:.1f} km/h (limit: {violation.threshold_value:.1f} km/h)\n"
            else:
                message += f"Measured: {violation.measured_value} (threshold: {violation.threshold_value})\n"

        message += f"Confidence: {violation.detection_confidence:.1%}\n"

        if violation.zone_id:
            message += f"Zone: {violation.zone_id}\n"

        message += f"\nViolation ID: {violation.id}"

        return subject, message.strip()

    def _generate_anomaly_alert_content(
        self, anomaly: TrafficAnomaly
    ) -> tuple[str, str]:
        """Generate alert content for traffic anomaly."""
        anomaly_type_names = {
            "traffic_pattern": "Traffic Pattern Anomaly",
            "vehicle_behavior": "Vehicle Behavior Anomaly",
            "speed_anomaly": "Speed Anomaly",
            "density_anomaly": "Traffic Density Anomaly",
            "flow_anomaly": "Traffic Flow Anomaly",
            "incident": "Traffic Incident",
        }

        anomaly_name = anomaly_type_names.get(
            anomaly.anomaly_type, anomaly.anomaly_type.title()
        )

        subject = f"⚠️ {anomaly.severity.title()} {anomaly_name} Detected"

        message = f"""
Traffic Anomaly Alert

Type: {anomaly_name}
Severity: {anomaly.severity.upper()}
Score: {anomaly.anomaly_score:.2f}
Location: Camera {anomaly.camera_id}
Time: {anomaly.detection_time.strftime("%Y-%m-%d %H:%M:%S")}
Confidence: {anomaly.confidence:.1%}
"""

        if anomaly.probable_cause:
            message += f"Probable Cause: {anomaly.probable_cause}\n"

        if anomaly.baseline_value and anomaly.observed_value:
            message += f"Baseline: {anomaly.baseline_value:.2f}\n"
            message += f"Observed: {anomaly.observed_value:.2f}\n"

            if anomaly.deviation_magnitude:
                message += f"Deviation: {anomaly.deviation_magnitude:.2f}\n"

        if anomaly.zone_id:
            message += f"Zone: {anomaly.zone_id}\n"

        message += f"Detection Method: {anomaly.detection_method}\n"
        message += f"\nAnomaly ID: {anomaly.id}"

        return subject, message.strip()

    async def _is_in_cooldown(
        self, alert_type: str, camera_id: str, event_type: str
    ) -> bool:
        """Check if alert type is in cooldown period for camera/event combination."""
        cooldown_period = self.rule_engine.get_cooldown_period(alert_type)
        cutoff_time = datetime.utcnow() - cooldown_period

        # Check for recent similar alerts
        query = select(AlertNotification).where(
            and_(
                AlertNotification.alert_type == alert_type,
                AlertNotification.created_time >= cutoff_time,
                AlertNotification.status.in_(["delivered", "pending"]),
            )
        )

        # Add metadata filter for camera and event type
        query = query.where(
            AlertNotification.delivery_details.op("@>")(
                {f"{alert_type}_type": event_type, "camera_id": camera_id}
            )
        )

        result = await self.session.execute(query)
        recent_alerts = result.scalars().first()

        return recent_alerts is not None

    async def _update_cooldown(
        self, alert_type: str, camera_id: str, event_type: str
    ) -> None:
        """Update cooldown tracking for alert type."""
        # Cooldown is implicitly tracked by checking recent alert timestamps
        # No explicit action needed as we use the alert creation time
        pass

    async def get_alert_statistics(
        self,
        time_range: tuple[datetime, datetime] | None = None,
        camera_id: str | None = None,
    ) -> dict[str, Any]:
        """Get alert delivery statistics.

        Args:
            time_range: Optional time range filter
            camera_id: Optional camera filter

        Returns:
            Alert statistics dictionary
        """
        try:
            query = select(AlertNotification)

            conditions = []
            if time_range:
                start_time, end_time = time_range
                conditions.append(AlertNotification.created_time >= start_time)
                conditions.append(AlertNotification.created_time <= end_time)

            if camera_id:
                conditions.append(
                    AlertNotification.delivery_details.op("@>")(
                        {"camera_id": camera_id}
                    )
                )

            if conditions:
                query = query.where(and_(*conditions))

            result = await self.session.execute(query)
            notifications = result.scalars().all()

            # Calculate statistics
            total_alerts = len(notifications)
            delivered_alerts = sum(1 for n in notifications if n.status == "delivered")
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
                    "start": time_range[0] if time_range else None,
                    "end": time_range[1] if time_range else None,
                },
            }

        except SQLAlchemyError as e:
            logger.error(f"Failed to get alert statistics: {e}")
            raise DatabaseError(f"Failed to retrieve alert statistics: {e}")

    async def retry_failed_alerts(self, max_retries: int = 3) -> dict[str, Any]:
        """Retry failed alert notifications.

        Args:
            max_retries: Maximum retry attempts

        Returns:
            Retry operation results
        """
        try:
            # Get failed notifications that haven't exceeded retry limit
            query = (
                select(AlertNotification)
                .where(
                    and_(
                        AlertNotification.status == "failed",
                        AlertNotification.delivery_attempts < max_retries,
                    )
                )
                .order_by(AlertNotification.created_time.desc())
                .limit(50)
            )

            result = await self.session.execute(query)
            failed_notifications = result.scalars().all()

            retry_results = []

            for notification in failed_notifications:
                try:
                    # Get notification channel
                    channel = self.notification_channels.get(
                        notification.notification_channel
                    )
                    if not channel:
                        continue

                    # Retry delivery
                    notification.delivery_attempts += 1
                    notification.last_attempt_time = datetime.utcnow()

                    delivery_result = await channel.send_notification(
                        recipient=notification.recipient,
                        subject=notification.subject or "Alert Retry",
                        message=notification.message_content,
                        priority=notification.priority,
                    )

                    # Update status based on result
                    if delivery_result.get("status") == "delivered":
                        notification.status = "delivered"
                        notification.sent_time = delivery_result.get(
                            "delivered_at", datetime.utcnow()
                        )
                        notification.external_id = delivery_result.get("delivery_id")
                        notification.delivery_details = delivery_result.get(
                            "details", {}
                        )
                        notification.error_message = None
                    else:
                        notification.error_message = delivery_result.get(
                            "error", "Retry failed"
                        )

                    retry_results.append(
                        {
                            "notification_id": notification.id,
                            "status": notification.status,
                            "attempts": notification.delivery_attempts,
                        }
                    )

                except Exception as e:
                    logger.error(f"Failed to retry notification {notification.id}: {e}")
                    notification.error_message = str(e)
                    retry_results.append(
                        {
                            "notification_id": notification.id,
                            "status": "failed",
                            "error": str(e),
                        }
                    )

            await self.session.commit()

            successful_retries = sum(
                1 for r in retry_results if r["status"] == "delivered"
            )

            return {
                "total_retried": len(retry_results),
                "successful_retries": successful_retries,
                "failed_retries": len(retry_results) - successful_retries,
                "results": retry_results,
            }

        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Failed to retry alerts: {e}")
            raise DatabaseError(f"Alert retry operation failed: {e}")

    async def acknowledge_alert(
        self,
        notification_id: str,
        acknowledged_by: str,
        response_action: str | None = None,
        notes: str | None = None,
    ) -> bool:
        """Acknowledge an alert notification.

        Args:
            notification_id: Notification ID to acknowledge
            acknowledged_by: User acknowledging the alert
            response_action: Action taken in response
            notes: Additional notes

        Returns:
            True if successfully acknowledged
        """
        try:
            notification = await self.get_by_id(notification_id)
            if not notification:
                return False

            notification.acknowledged_time = datetime.utcnow()
            notification.acknowledged_by = acknowledged_by
            notification.response_action = response_action
            notification.response_notes = notes

            await self.session.commit()

            logger.info(
                "Alert acknowledged",
                notification_id=notification_id,
                acknowledged_by=acknowledged_by,
            )

            return True

        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Failed to acknowledge alert: {e}")
            return False
