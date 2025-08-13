"""Comprehensive tests for the Alert Service implementation.

Tests email, webhook, and SMS notification channels with proper
retry logic, authentication, and error handling.
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from its_camera_ai.core.config import Settings
from its_camera_ai.models.analytics import RuleViolation, TrafficAnomaly
from its_camera_ai.services.alert_service import (
    AlertService,
    EmailNotificationChannel,
    SMSNotificationChannel,
    WebhookNotificationChannel,
)


class MockSettings(Settings):
    """Mock settings for testing."""

    smtp_server: str = "smtp.example.com"
    smtp_port: int = 587
    smtp_username: str = "test@example.com"
    smtp_password: str = "test_password"
    smtp_from_email: str = "alerts@its-camera-ai.com"
    smtp_timeout: int = 30
    smtp_max_retries: int = 3
    smtp_retry_delay: float = 1.0
    smtp_html_template: bool = True
    smtp_ssl_verify: bool = True
    smtp_no_tls: bool = False

    webhook_url: str = "https://api.example.com/webhooks/alerts"
    webhook_timeout: int = 30
    webhook_connect_timeout: int = 10
    webhook_max_retries: int = 3
    webhook_retry_delay: float = 1.0
    webhook_auth_type: str = "bearer"
    webhook_auth_token: str = "test_token_123"
    webhook_api_key: str = "test_api_key"
    webhook_api_key_header: str = "X-API-Key"
    webhook_username: str = "webhook_user"
    webhook_password: str = "webhook_pass"
    webhook_secret: str = "webhook_secret_key"
    webhook_ssl_verify: bool = True
    webhook_follow_redirects: bool = False

    sms_api_url: str = "https://api.sms-provider.com/send"
    sms_api_key: str = "sms_api_key_123"
    sms_from_number: str = "+1234567890"
    sms_provider: str = "generic"
    sms_timeout: int = 30
    sms_connect_timeout: int = 10
    sms_max_retries: int = 3
    sms_retry_delay: float = 1.0
    sms_max_length: int = 160


@pytest.fixture
def mock_settings():
    """Mock settings fixture."""
    return MockSettings()


@pytest.fixture
def mock_session():
    """Mock database session."""
    session = MagicMock(spec=AsyncSession)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def sample_violation():
    """Sample traffic violation for testing."""
    return RuleViolation(
        id=str(uuid4()),
        camera_id="CAM-001",
        violation_type="speeding",
        severity="high",
        detection_time=datetime.utcnow(),
        license_plate="ABC123",
        measured_value=85.0,
        threshold_value=60.0,
        detection_confidence=0.95,
        zone_id="ZONE-001",
    )


@pytest.fixture
def sample_anomaly():
    """Sample traffic anomaly for testing."""
    return TrafficAnomaly(
        id=str(uuid4()),
        camera_id="CAM-002",
        anomaly_type="traffic_pattern",
        severity="critical",
        detection_time=datetime.utcnow(),
        anomaly_score=0.85,
        confidence=0.90,
        probable_cause="Unusual traffic congestion",
        baseline_value=50.0,
        observed_value=200.0,
        deviation_magnitude=4.0,
        zone_id="ZONE-002",
        detection_method="ML_CLASSIFIER",
    )


class TestEmailNotificationChannel:
    """Tests for email notification channel."""

    @pytest.mark.asyncio
    async def test_email_send_success(self, mock_settings):
        """Test successful email sending."""
        config = {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "username": "test@example.com",
            "password": "password",
            "from_email": "alerts@example.com",
            "timeout": 30,
            "max_retries": 3,
            "retry_delay_base": 1.0,
            "html_template": True,
            "ssl_verify": True,
        }

        channel = EmailNotificationChannel(config)

        with patch("aiosmtplib.SMTP") as mock_smtp:
            mock_smtp_instance = AsyncMock()
            mock_smtp.return_value.__aenter__.return_value = mock_smtp_instance

            result = await channel.send_notification(
                recipient="user@example.com",
                subject="Test Alert",
                message="Test message",
                priority="high",
                metadata={"test": "data"},
            )

            assert result["status"] == "delivered"
            assert result["channel"] == "email"
            assert result["recipient"] == "user@example.com"
            assert "delivery_id" in result
            assert "delivered_at" in result

            # Verify SMTP calls
            mock_smtp_instance.starttls.assert_called_once()
            mock_smtp_instance.login.assert_called_once_with("test@example.com", "password")
            mock_smtp_instance.send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_email_send_with_retries(self, mock_settings):
        """Test email sending with retry logic."""
        config = {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "username": "test@example.com",
            "password": "password",
            "from_email": "alerts@example.com",
            "timeout": 30,
            "max_retries": 2,
            "retry_delay_base": 0.1,  # Fast retry for testing
            "html_template": True,
            "ssl_verify": True,
        }

        channel = EmailNotificationChannel(config)

        with patch("aiosmtplib.SMTP") as mock_smtp:
            # First attempt fails, second succeeds
            mock_smtp_instance = AsyncMock()
            mock_smtp.return_value.__aenter__.return_value = mock_smtp_instance
            mock_smtp_instance.send_message.side_effect = [Exception("SMTP Error"), None]

            result = await channel.send_notification(
                recipient="user@example.com",
                subject="Test Alert",
                message="Test message",
                priority="high",
            )

            assert result["status"] == "delivered"
            assert result["details"]["attempts"] == 2
            assert mock_smtp_instance.send_message.call_count == 2

    @pytest.mark.asyncio
    async def test_email_send_failure_after_retries(self, mock_settings):
        """Test email failure after all retries exhausted."""
        config = {
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "username": "test@example.com",
            "password": "password",
            "from_email": "alerts@example.com",
            "timeout": 30,
            "max_retries": 2,
            "retry_delay_base": 0.1,
            "html_template": True,
            "ssl_verify": True,
        }

        channel = EmailNotificationChannel(config)

        with patch("aiosmtplib.SMTP") as mock_smtp:
            mock_smtp_instance = AsyncMock()
            mock_smtp.return_value.__aenter__.return_value = mock_smtp_instance
            mock_smtp_instance.send_message.side_effect = Exception("Persistent SMTP Error")

            result = await channel.send_notification(
                recipient="user@example.com",
                subject="Test Alert",
                message="Test message",
                priority="high",
            )

            assert result["status"] == "failed"
            assert "Persistent SMTP Error" in result["error"]
            assert result["attempts"] == 3  # max_retries + 1

    def test_html_message_formatting(self):
        """Test HTML message formatting."""
        config = {"html_template": True}
        channel = EmailNotificationChannel(config)

        html_message = channel._format_html_message(
            "Test Alert",
            "This is a test message",
            "critical"
        )

        assert "<!DOCTYPE html>" in html_message
        assert "Test Alert" in html_message
        assert "This is a test message" in html_message
        assert "CRITICAL" in html_message
        assert "#dc3545" in html_message  # Critical color


class TestWebhookNotificationChannel:
    """Tests for webhook notification channel."""

    @pytest.mark.asyncio
    async def test_webhook_send_success(self):
        """Test successful webhook sending."""
        config = {
            "url": "https://api.example.com/webhooks",
            "headers": {"Content-Type": "application/json"},
            "timeout": 30,
            "max_retries": 3,
            "retry_delay_base": 1.0,
            "auth_type": "bearer",
            "auth_token": "test_token",
            "verify_ssl": True,
        }

        channel = WebhookNotificationChannel(config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client_instance.post.return_value = mock_response

            result = await channel.send_notification(
                recipient="system",
                subject="Test Alert",
                message="Test message",
                priority="high",
                metadata={"test": "data"},
            )

            assert result["status"] == "delivered"
            assert result["channel"] == "webhook"
            assert result["details"]["status_code"] == 200
            assert result["details"]["auth_type"] == "bearer"

            # Verify HTTP call was made with correct headers
            mock_client_instance.post.assert_called_once()
            call_args = mock_client_instance.post.call_args
            assert "Authorization" in call_args.kwargs["headers"]
            assert call_args.kwargs["headers"]["Authorization"] == "Bearer test_token"

    @pytest.mark.asyncio
    async def test_webhook_with_signature(self):
        """Test webhook with HMAC signature."""
        config = {
            "url": "https://api.example.com/webhooks",
            "headers": {"Content-Type": "application/json"},
            "timeout": 30,
            "max_retries": 3,
            "retry_delay_base": 1.0,
            "webhook_secret": "secret_key",
            "verify_ssl": True,
        }

        channel = WebhookNotificationChannel(config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client_instance.post.return_value = mock_response

            result = await channel.send_notification(
                recipient="system",
                subject="Test Alert",
                message="Test message",
                priority="high",
            )

            assert result["status"] == "delivered"

            # Verify signature header was added
            call_args = mock_client_instance.post.call_args
            headers = call_args.kwargs["headers"]
            assert "X-Webhook-Signature" in headers
            assert headers["X-Webhook-Signature"].startswith("sha256=")

    @pytest.mark.asyncio
    async def test_webhook_retry_on_failure(self):
        """Test webhook retry logic on HTTP errors."""
        config = {
            "url": "https://api.example.com/webhooks",
            "headers": {"Content-Type": "application/json"},
            "timeout": 30,
            "max_retries": 2,
            "retry_delay_base": 0.1,
        }

        channel = WebhookNotificationChannel(config)

        with patch("httpx.AsyncClient") as mock_client:
            # First two calls fail, third succeeds
            mock_responses = [
                Exception("Connection error"),
                Exception("Timeout error"),
                MagicMock(status_code=200, headers={}, raise_for_status=MagicMock()),
            ]

            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client_instance.post.side_effect = mock_responses

            result = await channel.send_notification(
                recipient="system",
                subject="Test Alert",
                message="Test message",
                priority="high",
            )

            assert result["status"] == "delivered"
            assert result["details"]["attempts"] == 3
            assert mock_client_instance.post.call_count == 3


class TestSMSNotificationChannel:
    """Tests for SMS notification channel."""

    @pytest.mark.asyncio
    async def test_sms_send_success(self):
        """Test successful SMS sending."""
        config = {
            "api_url": "https://sms.provider.com/send",
            "api_key": "sms_key_123",
            "from_number": "+1234567890",
            "provider": "generic",
            "timeout": 30,
            "max_retries": 3,
            "retry_delay_base": 1.0,
            "max_message_length": 160,
        }

        channel = SMSNotificationChannel(config)

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {}
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            mock_client_instance.post.return_value = mock_response

            result = await channel.send_notification(
                recipient="+1987654321",
                subject="🚨 Critical Speed Limit Violation Detected",
                message="Type: Speed Limit Violation\nSeverity: CRITICAL\nLocation: Camera CAM-001",
                priority="critical",
            )

            assert result["status"] == "delivered"
            assert result["channel"] == "sms"
            assert result["details"]["provider"] == "generic"

    def test_sms_message_formatting(self):
        """Test SMS message formatting and length constraints."""
        config = {
            "provider": "generic",
            "max_message_length": 160,
        }

        channel = SMSNotificationChannel(config)

        long_message = """
Traffic Violation Alert

Type: Speed Limit Violation  
Severity: CRITICAL
Location: Camera CAM-001
Time: 2023-12-01 10:30:00
Vehicle: ABC123
Speed: 85.0 km/h (limit: 60.0 km/h)
Confidence: 95.0%
Zone: ZONE-001
Violation ID: 12345-67890
        """

        formatted = channel._format_sms_message(
            "🚨 Critical Speed Limit Violation Detected",
            long_message,
            "critical"
        )

        assert len(formatted) <= 160
        assert "🚨" in formatted  # Critical emoji
        assert "Type:" in formatted
        assert "Severity:" in formatted
        assert "Location:" in formatted

    def test_sms_payload_preparation(self):
        """Test SMS payload preparation for different providers."""
        # Test generic provider
        config = {"provider": "generic", "from_number": "+1234567890"}
        channel = SMSNotificationChannel(config)
        payload = channel._prepare_sms_payload("+1987654321", "Test message")

        assert payload["from_number"] == "+1234567890"
        assert payload["to_number"] == "+1987654321"
        assert payload["message"] == "Test message"

        # Test Twilio provider
        config["provider"] = "twilio"
        channel = SMSNotificationChannel(config)
        payload = channel._prepare_sms_payload("+1987654321", "Test message")

        assert payload["From"] == "+1234567890"
        assert payload["To"] == "+1987654321"
        assert payload["Body"] == "Test message"

        # Test Nexmo provider
        config["provider"] = "nexmo"
        channel = SMSNotificationChannel(config)
        payload = channel._prepare_sms_payload("+1987654321", "Test message")

        assert payload["from"] == "+1234567890"
        assert payload["to"] == "+1987654321"
        assert payload["text"] == "Test message"


class TestAlertService:
    """Tests for the main AlertService class."""

    @pytest.mark.asyncio
    async def test_process_violation_alert(self, mock_session, mock_settings, sample_violation):
        """Test processing violation alerts."""
        service = AlertService(mock_session, mock_settings)

        # Mock the cooldown check
        with patch.object(service, '_is_in_cooldown', return_value=False), \
             patch.object(service, '_update_cooldown', return_value=None), \
             patch.object(service, '_send_alert') as mock_send_alert:

            mock_send_alert.return_value = {
                "status": "delivered",
                "notification_id": "test-123",
                "delivery_result": {"status": "delivered"},
            }

            results = await service.process_violation_alert(
                violation=sample_violation,
                recipients=["admin@example.com"],
                notification_channels=["email"]
            )

            assert len(results) == 1
            assert results[0]["status"] == "delivered"
            mock_send_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_anomaly_alert(self, mock_session, mock_settings, sample_anomaly):
        """Test processing anomaly alerts."""
        service = AlertService(mock_session, mock_settings)

        with patch.object(service, '_is_in_cooldown', return_value=False), \
             patch.object(service, '_update_cooldown', return_value=None), \
             patch.object(service, '_send_alert') as mock_send_alert:

            mock_send_alert.return_value = {
                "status": "delivered",
                "notification_id": "test-456",
                "delivery_result": {"status": "delivered"},
            }

            results = await service.process_anomaly_alert(
                anomaly=sample_anomaly,
                recipients=["ops@example.com"],
                notification_channels=["webhook", "email"]
            )

            assert len(results) == 2  # Two channels
            assert all(r["status"] == "delivered" for r in results)
            assert mock_send_alert.call_count == 2

    @pytest.mark.asyncio
    async def test_alert_service_initialization(self, mock_session, mock_settings):
        """Test alert service initialization with all channels."""
        service = AlertService(mock_session, mock_settings)

        # Should have email, webhook, and SMS channels
        assert "email" in service.notification_channels
        assert "webhook" in service.notification_channels
        assert "sms" in service.notification_channels

        # Verify channel configurations
        email_channel = service.notification_channels["email"]
        assert email_channel.smtp_server == "smtp.example.com"
        assert email_channel.smtp_port == 587

        webhook_channel = service.notification_channels["webhook"]
        assert webhook_channel.webhook_url == "https://api.example.com/webhooks/alerts"

        sms_channel = service.notification_channels["sms"]
        assert sms_channel.api_url == "https://api.sms-provider.com/send"

    @pytest.mark.asyncio
    async def test_cooldown_functionality(self, mock_session, mock_settings, sample_violation):
        """Test alert cooldown functionality."""
        service = AlertService(mock_session, mock_settings)

        # Mock recent alert exists (in cooldown)
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = MagicMock()  # Non-None = in cooldown
        mock_session.execute.return_value = mock_result

        with patch.object(service, '_update_cooldown', return_value=None):
            results = await service.process_violation_alert(
                violation=sample_violation,
                recipients=["admin@example.com"],
                notification_channels=["email"]
            )

            # Should return empty list due to cooldown
            assert results == []

    def test_violation_alert_content_generation(self, mock_session, mock_settings, sample_violation):
        """Test violation alert content generation."""
        service = AlertService(mock_session, mock_settings)

        subject, message = service._generate_violation_alert_content(sample_violation)

        assert "🚨" in subject
        assert "Speed Limit Violation" in subject
        assert "High" in subject

        assert "Traffic Violation Alert" in message
        assert "Speed Limit Violation" in message
        assert "HIGH" in message
        assert "CAM-001" in message
        assert "ABC123" in message
        assert "85.0 km/h" in message
        assert "60.0 km/h" in message

    def test_anomaly_alert_content_generation(self, mock_session, mock_settings, sample_anomaly):
        """Test anomaly alert content generation."""
        service = AlertService(mock_session, mock_settings)

        subject, message = service._generate_anomaly_alert_content(sample_anomaly)

        assert "⚠️" in subject
        assert "Traffic Pattern Anomaly" in subject
        assert "Critical" in subject

        assert "Traffic Anomaly Alert" in message
        assert "Traffic Pattern Anomaly" in message
        assert "CRITICAL" in message
        assert "CAM-002" in message
        assert "0.85" in message  # anomaly score
        assert "Unusual traffic congestion" in message


@pytest.mark.integration
class TestAlertServiceIntegration:
    """Integration tests for alert service with real dependencies."""

    @pytest.mark.asyncio
    async def test_end_to_end_alert_flow(self, mock_session, mock_settings, sample_violation):
        """Test complete end-to-end alert processing flow."""
        service = AlertService(mock_session, mock_settings)

        # Mock all external dependencies
        with patch("aiosmtplib.SMTP") as mock_smtp, \
             patch("httpx.AsyncClient") as mock_http_client:

            # Setup email mock
            mock_smtp_instance = AsyncMock()
            mock_smtp.return_value.__aenter__.return_value = mock_smtp_instance

            # Setup webhook mock
            mock_response = MagicMock(status_code=200, headers={}, raise_for_status=MagicMock())
            mock_http_instance = AsyncMock()
            mock_http_client.return_value.__aenter__.return_value = mock_http_instance
            mock_http_instance.post.return_value = mock_response

            # Mock database operations
            mock_session.execute.return_value.scalars.return_value.first.return_value = None  # No cooldown

            # Process alert
            results = await service.process_violation_alert(
                violation=sample_violation,
                recipients=["admin@example.com", "ops-webhook"],
                notification_channels=["email", "webhook"]
            )

            # Should have 4 results (2 recipients × 2 channels)
            assert len(results) == 4
            assert all(r["status"] == "delivered" for r in results)

            # Verify external calls were made
            assert mock_smtp_instance.send_message.call_count == 2  # 2 email recipients
            assert mock_http_instance.post.call_count == 2  # 2 webhook recipients
            assert mock_session.add.call_count == 4  # 4 notifications saved
            assert mock_session.commit.call_count == 8  # 4 saves + 4 updates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
