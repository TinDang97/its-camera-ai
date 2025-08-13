"""Email service for sending notifications and verification emails."""

import smtplib
from datetime import UTC, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ..core.config import Settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class EmailService:
    """Service for sending emails with templates."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.smtp_server = getattr(settings, 'smtp_server', 'localhost')
        self.smtp_port = getattr(settings, 'smtp_port', 587)
        self.smtp_username = getattr(settings, 'smtp_username', None)
        self.smtp_password = getattr(settings, 'smtp_password', None)
        self.smtp_use_tls = getattr(settings, 'smtp_use_tls', True)
        self.from_email = getattr(settings, 'from_email', 'noreply@its-camera-ai.com')
        self.from_name = getattr(settings, 'from_name', 'ITS Camera AI')

        # Setup Jinja2 template environment
        template_dir = Path(__file__).parent.parent / 'templates' / 'email'
        template_dir.mkdir(parents=True, exist_ok=True)
        self.template_env = Environment(loader=FileSystemLoader(str(template_dir)))

        # Create default templates if they don't exist
        self._ensure_default_templates(template_dir)

    def _ensure_default_templates(self, template_dir: Path) -> None:
        """Create default email templates if they don't exist."""
        templates = {
            'verification.html': self._get_verification_template(),
            'password_reset.html': self._get_password_reset_template(),
            'security_alert.html': self._get_security_alert_template(),
            'account_lockout.html': self._get_account_lockout_template(),
            'password_changed.html': self._get_password_changed_template(),
            'mfa_enabled.html': self._get_mfa_enabled_template(),
        }

        for filename, content in templates.items():
            template_path = template_dir / filename
            if not template_path.exists():
                template_path.write_text(content)

    def _get_verification_template(self) -> str:
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Email Verification - ITS Camera AI</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        .content { padding: 30px 20px; background: #f8f9fa; }
        .button { display: inline-block; background: #007bff; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Email Verification Required</h1>
        </div>
        <div class="content">
            <h2>Hello {{ full_name or username }}!</h2>
            <p>Welcome to ITS Camera AI! Please verify your email address to complete your registration.</p>
            <p>Click the button below to verify your email:</p>
            <a href="{{ verification_url }}" class="button">Verify Email Address</a>
            <p>Or copy and paste this link into your browser:</p>
            <p style="word-break: break-all;">{{ verification_url }}</p>
            <p><strong>This link will expire in 24 hours.</strong></p>
            <p>If you didn't create this account, please ignore this email.</p>
        </div>
        <div class="footer">
            <p>ITS Camera AI - AI-Powered Traffic Monitoring System</p>
        </div>
    </div>
</body>
</html>"""

    def _get_password_reset_template(self) -> str:
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Password Reset - ITS Camera AI</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #dc3545; color: white; padding: 20px; text-align: center; }
        .content { padding: 30px 20px; background: #f8f9fa; }
        .button { display: inline-block; background: #dc3545; color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Password Reset Request</h1>
        </div>
        <div class="content">
            <h2>Hello {{ full_name or username }}!</h2>
            <p>We received a request to reset your password for your ITS Camera AI account.</p>
            <p>Click the button below to reset your password:</p>
            <a href="{{ reset_url }}" class="button">Reset Password</a>
            <p>Or copy and paste this link into your browser:</p>
            <p style="word-break: break-all;">{{ reset_url }}</p>
            <p><strong>This link will expire in 1 hour.</strong></p>
            <p>If you didn't request a password reset, please ignore this email or contact support if you have concerns.</p>
        </div>
        <div class="footer">
            <p>ITS Camera AI - AI-Powered Traffic Monitoring System</p>
        </div>
    </div>
</body>
</html>"""

    def _get_security_alert_template(self) -> str:
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Security Alert - ITS Camera AI</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #ffc107; color: #212529; padding: 20px; text-align: center; }
        .content { padding: 30px 20px; background: #f8f9fa; }
        .alert { background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚨 Security Alert</h1>
        </div>
        <div class="content">
            <h2>Hello {{ full_name or username }}!</h2>
            <div class="alert">
                <strong>Security Event:</strong> {{ event_type }}<br>
                <strong>Time:</strong> {{ timestamp }}<br>
                <strong>IP Address:</strong> {{ ip_address or 'Unknown' }}<br>
                <strong>Location:</strong> {{ location or 'Unknown' }}
            </div>
            <p><strong>Event Details:</strong></p>
            <p>{{ description }}</p>
            <p>If this was you, you can safely ignore this email. If you don't recognize this activity, please:</p>
            <ul>
                <li>Change your password immediately</li>
                <li>Review your account settings</li>
                <li>Enable two-factor authentication if not already enabled</li>
                <li>Contact support if you need assistance</li>
            </ul>
        </div>
        <div class="footer">
            <p>ITS Camera AI - AI-Powered Traffic Monitoring System</p>
        </div>
    </div>
</body>
</html>"""

    def _get_account_lockout_template(self) -> str:
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Account Temporarily Locked - ITS Camera AI</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #dc3545; color: white; padding: 20px; text-align: center; }
        .content { padding: 30px 20px; background: #f8f9fa; }
        .warning { background: #f8d7da; border: 1px solid #f5c6cb; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔒 Account Locked</h1>
        </div>
        <div class="content">
            <h2>Hello {{ full_name or username }}!</h2>
            <div class="warning">
                Your account has been temporarily locked due to multiple failed login attempts.
            </div>
            <p><strong>Lockout Details:</strong></p>
            <ul>
                <li><strong>Time:</strong> {{ lockout_time }}</li>
                <li><strong>Duration:</strong> {{ lockout_duration }} minutes</li>
                <li><strong>Unlock Time:</strong> {{ unlock_time }}</li>
                <li><strong>Failed Attempts:</strong> {{ attempt_count }}</li>
            </ul>
            <p>Your account will be automatically unlocked at {{ unlock_time }}.</p>
            <p>If you didn't attempt to log in, this could indicate someone is trying to access your account. Please:</p>
            <ul>
                <li>Change your password once the account is unlocked</li>
                <li>Enable two-factor authentication</li>
                <li>Contact support if you have concerns</li>
            </ul>
        </div>
        <div class="footer">
            <p>ITS Camera AI - AI-Powered Traffic Monitoring System</p>
        </div>
    </div>
</body>
</html>"""

    def _get_password_changed_template(self) -> str:
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Password Changed - ITS Camera AI</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #28a745; color: white; padding: 20px; text-align: center; }
        .content { padding: 30px 20px; background: #f8f9fa; }
        .success { background: #d4edda; border: 1px solid #c3e6cb; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>✅ Password Changed Successfully</h1>
        </div>
        <div class="content">
            <h2>Hello {{ full_name or username }}!</h2>
            <div class="success">
                Your password has been successfully changed.
            </div>
            <p><strong>Change Details:</strong></p>
            <ul>
                <li><strong>Time:</strong> {{ change_time }}</li>
                <li><strong>IP Address:</strong> {{ ip_address or 'Unknown' }}</li>
                <li><strong>Device:</strong> {{ device or 'Unknown' }}</li>
            </ul>
            <p>If you didn't make this change, please contact support immediately.</p>
        </div>
        <div class="footer">
            <p>ITS Camera AI - AI-Powered Traffic Monitoring System</p>
        </div>
    </div>
</body>
</html>"""

    def _get_mfa_enabled_template(self) -> str:
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Two-Factor Authentication Enabled - ITS Camera AI</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background: #17a2b8; color: white; padding: 20px; text-align: center; }
        .content { padding: 30px 20px; background: #f8f9fa; }
        .info { background: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 Two-Factor Authentication Enabled</h1>
        </div>
        <div class="content">
            <h2>Hello {{ full_name or username }}!</h2>
            <div class="info">
                Two-factor authentication has been successfully enabled on your account.
            </div>
            <p>Your account is now more secure with an additional layer of protection.</p>
            <p><strong>Setup Details:</strong></p>
            <ul>
                <li><strong>Method:</strong> {{ mfa_method }}</li>
                <li><strong>Enabled:</strong> {{ setup_time }}</li>
                <li><strong>Backup Codes:</strong> {{ backup_codes_count }} generated</li>
            </ul>
            <p><strong>Important:</strong> Save your backup codes in a secure location. You can use them to access your account if you lose access to your authenticator device.</p>
            <p>If you didn't enable this feature, please contact support immediately.</p>
        </div>
        <div class="footer">
            <p>ITS Camera AI - AI-Powered Traffic Monitoring System</p>
        </div>
    </div>
</body>
</html>"""

    async def send_email(
        self,
        to_email: str,
        subject: str,
        template_name: str,
        template_vars: dict[str, Any],
        to_name: str | None = None
    ) -> bool:
        """Send email using template.

        Args:
            to_email: Recipient email address
            subject: Email subject
            template_name: Template filename
            template_vars: Variables for template rendering
            to_name: Recipient name

        Returns:
            True if email sent successfully
        """
        try:
            # Render template
            template = self.template_env.get_template(template_name)
            html_content = template.render(**template_vars)

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = f"{to_name} <{to_email}>" if to_name else to_email

            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # Send email
            if self.settings.environment == 'development':
                # In development, just log the email
                logger.info(
                    "Email would be sent",
                    to=to_email,
                    subject=subject,
                    template=template_name
                )
                return True

            # Send via SMTP
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.smtp_use_tls:
                    server.starttls()

                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)

                server.send_message(msg)

            logger.info(
                "Email sent successfully",
                to=to_email,
                subject=subject,
                template=template_name
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to send email",
                to=to_email,
                subject=subject,
                template=template_name,
                error=str(e)
            )
            return False

    async def send_verification_email(
        self, to_email: str, username: str, verification_token: str, full_name: str | None = None
    ) -> bool:
        """Send email verification email.

        Args:
            to_email: User email
            username: Username
            verification_token: Verification token
            full_name: User's full name

        Returns:
            True if sent successfully
        """
        base_url = getattr(self.settings, 'base_url', 'http://localhost:3000')
        verification_url = f"{base_url}/verify-email?token={verification_token}"

        return await self.send_email(
            to_email=to_email,
            subject="Verify Your Email Address - ITS Camera AI",
            template_name="verification.html",
            template_vars={
                'username': username,
                'full_name': full_name,
                'verification_url': verification_url,
                'verification_token': verification_token
            },
            to_name=full_name or username
        )

    async def send_password_reset_email(
        self, to_email: str, username: str, reset_token: str, full_name: str | None = None
    ) -> bool:
        """Send password reset email.

        Args:
            to_email: User email
            username: Username
            reset_token: Reset token
            full_name: User's full name

        Returns:
            True if sent successfully
        """
        base_url = getattr(self.settings, 'base_url', 'http://localhost:3000')
        reset_url = f"{base_url}/reset-password?token={reset_token}"

        return await self.send_email(
            to_email=to_email,
            subject="Password Reset Request - ITS Camera AI",
            template_name="password_reset.html",
            template_vars={
                'username': username,
                'full_name': full_name,
                'reset_url': reset_url,
                'reset_token': reset_token
            },
            to_name=full_name or username
        )

    async def send_security_alert(
        self,
        to_email: str,
        username: str,
        event_type: str,
        description: str,
        ip_address: str | None = None,
        timestamp: str | None = None,
        full_name: str | None = None
    ) -> bool:
        """Send security alert email.

        Args:
            to_email: User email
            username: Username
            event_type: Type of security event
            description: Event description
            ip_address: Client IP address
            timestamp: Event timestamp
            full_name: User's full name

        Returns:
            True if sent successfully
        """
        return await self.send_email(
            to_email=to_email,
            subject=f"Security Alert: {event_type} - ITS Camera AI",
            template_name="security_alert.html",
            template_vars={
                'username': username,
                'full_name': full_name,
                'event_type': event_type,
                'description': description,
                'ip_address': ip_address,
                'timestamp': timestamp or datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')
            },
            to_name=full_name or username
        )

    async def send_account_lockout_email(
        self,
        to_email: str,
        username: str,
        lockout_time: str,
        unlock_time: str,
        attempt_count: int,
        lockout_duration: int,
        full_name: str | None = None
    ) -> bool:
        """Send account lockout notification email.

        Args:
            to_email: User email
            username: Username
            lockout_time: When account was locked
            unlock_time: When account will be unlocked
            attempt_count: Number of failed attempts
            lockout_duration: Lockout duration in minutes
            full_name: User's full name

        Returns:
            True if sent successfully
        """
        return await self.send_email(
            to_email=to_email,
            subject="Account Temporarily Locked - ITS Camera AI",
            template_name="account_lockout.html",
            template_vars={
                'username': username,
                'full_name': full_name,
                'lockout_time': lockout_time,
                'unlock_time': unlock_time,
                'attempt_count': attempt_count,
                'lockout_duration': lockout_duration
            },
            to_name=full_name or username
        )

    async def send_password_changed_email(
        self,
        to_email: str,
        username: str,
        change_time: str,
        ip_address: str | None = None,
        device: str | None = None,
        full_name: str | None = None
    ) -> bool:
        """Send password changed notification email.

        Args:
            to_email: User email
            username: Username
            change_time: When password was changed
            ip_address: Client IP address
            device: User device info
            full_name: User's full name

        Returns:
            True if sent successfully
        """
        return await self.send_email(
            to_email=to_email,
            subject="Password Changed Successfully - ITS Camera AI",
            template_name="password_changed.html",
            template_vars={
                'username': username,
                'full_name': full_name,
                'change_time': change_time,
                'ip_address': ip_address,
                'device': device
            },
            to_name=full_name or username
        )

    async def send_mfa_enabled_email(
        self,
        to_email: str,
        username: str,
        mfa_method: str,
        setup_time: str,
        backup_codes_count: int,
        full_name: str | None = None
    ) -> bool:
        """Send MFA enabled notification email.

        Args:
            to_email: User email
            username: Username
            mfa_method: MFA method enabled
            setup_time: When MFA was enabled
            backup_codes_count: Number of backup codes generated
            full_name: User's full name

        Returns:
            True if sent successfully
        """
        return await self.send_email(
            to_email=to_email,
            subject="Two-Factor Authentication Enabled - ITS Camera AI",
            template_name="mfa_enabled.html",
            template_vars={
                'username': username,
                'full_name': full_name,
                'mfa_method': mfa_method.upper(),
                'setup_time': setup_time,
                'backup_codes_count': backup_codes_count
            },
            to_name=full_name or username
        )
