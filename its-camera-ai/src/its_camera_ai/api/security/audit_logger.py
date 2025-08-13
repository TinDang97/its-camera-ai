"""
Enhanced audit logging for security events and compliance.

Provides:
- Security event logging with structured data
- Failed authentication attempt tracking
- Permission denied event logging
- Rate limit violation tracking
- Suspicious pattern detection and alerting
- SIEM integration format
- Compliance reporting
"""

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import redis.asyncio as redis
from fastapi import Request

from ...core.config import get_settings
from ...core.logging import get_logger

logger = get_logger(__name__)


class SecurityEventType(Enum):
    """Security event types for audit logging."""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    MFA_SUCCESS = "mfa_success"
    MFA_FAILURE = "mfa_failure"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_ESCALATION = "permission_escalation"
    ROLE_CHANGE = "role_change"

    # API Security events
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    API_KEY_USED = "api_key_used"
    API_KEY_INVALID = "api_key_invalid"

    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    RATE_LIMIT_WARNING = "rate_limit_warning"
    RATE_LIMIT_RESET = "rate_limit_reset"

    # Security validation events
    SQL_INJECTION_ATTEMPT = "sql_injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"
    COMMAND_INJECTION_ATTEMPT = "command_injection_attempt"
    SSRF_ATTEMPT = "ssrf_attempt"
    XXE_ATTEMPT = "xxe_attempt"

    # Session events
    SESSION_CREATED = "session_created"
    SESSION_EXPIRED = "session_expired"
    SESSION_REVOKED = "session_revoked"
    SESSION_FINGERPRINT_MISMATCH = "session_fingerprint_mismatch"
    CONCURRENT_SESSION_LIMIT = "concurrent_session_limit"

    # CSRF events
    CSRF_TOKEN_MISMATCH = "csrf_token_mismatch"
    CSRF_TOKEN_MISSING = "csrf_token_missing"

    # File upload events
    MALICIOUS_FILE_UPLOAD = "malicious_file_upload"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    INVALID_FILE_TYPE = "invalid_file_type"

    # System events
    CONFIGURATION_CHANGE = "configuration_change"
    SECURITY_POLICY_VIOLATION = "security_policy_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    BRUTE_FORCE_ATTEMPT = "brute_force_attempt"

    # Compliance events
    DATA_EXPORT = "data_export"
    DATA_DELETION = "data_deletion"
    PRIVACY_REQUEST = "privacy_request"


class SecurityEventSeverity(Enum):
    """Security event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event data model."""

    # Event identification
    event_id: str
    event_type: SecurityEventType
    severity: SecurityEventSeverity
    timestamp: datetime

    # Request context
    client_ip: str
    user_agent: str
    request_path: str
    request_method: str

    # User context
    user_id: str | None = None
    session_id: str | None = None
    api_key_id: str | None = None

    # Event details
    message: str = ""
    details: dict[str, Any] = None

    # Response context
    response_status: int | None = None
    response_time_ms: float | None = None

    # Geolocation (if available)
    country: str | None = None
    region: str | None = None
    city: str | None = None

    # Risk assessment
    risk_score: int = 0  # 0-100
    is_suspicious: bool = False

    # Compliance tags
    compliance_tags: list[str] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.compliance_tags is None:
            self.compliance_tags = []


class SecurityAuditLogger:
    """Enhanced security audit logger."""

    def __init__(self, redis_client: redis.Redis | None = None, settings=None):
        self.redis = redis_client
        self.settings = settings or get_settings()

        # Configuration
        self.retention_days = self.settings.security.audit_log_retention_days
        self.high_risk_threshold = self.settings.security.high_risk_alert_threshold

        # Redis keys
        self.events_key = "security_events"
        self.alerts_key = "security_alerts"
        self.metrics_key = "security_metrics"

        # Event tracking for pattern detection
        self.failed_attempts_key = "failed_attempts"
        self.suspicious_ips_key = "suspicious_ips"

        # SIEM integration
        self.siem_enabled = bool(self.settings.security.security_alert_webhook)

    async def log_event(
        self,
        event_type: SecurityEventType,
        severity: SecurityEventSeverity,
        request: Request,
        message: str = "",
        details: dict[str, Any] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        api_key_id: str | None = None
    ) -> str:
        """
        Log a security event.

        Args:
            event_type: Type of security event
            severity: Event severity level
            request: HTTP request object
            message: Human-readable event message
            details: Additional event details
            user_id: Associated user ID
            session_id: Associated session ID
            api_key_id: Associated API key ID

        Returns:
            Event ID
        """
        # Generate event ID
        event_id = self._generate_event_id()

        # Create security event
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(UTC),
            client_ip=self._get_client_ip(request),
            user_agent=request.headers.get("user-agent", ""),
            request_path=request.url.path,
            request_method=request.method,
            user_id=user_id,
            session_id=session_id,
            api_key_id=api_key_id,
            message=message,
            details=details or {},
            risk_score=self._calculate_risk_score(event_type, severity, details),
            is_suspicious=self._is_suspicious_event(event_type, severity)
        )

        # Add geolocation if available
        await self._add_geolocation(event)

        # Add compliance tags
        self._add_compliance_tags(event)

        # Store event
        await self._store_event(event)

        # Update metrics
        await self._update_metrics(event)

        # Check for patterns and alerts
        await self._check_suspicious_patterns(event)

        # Send to SIEM if configured
        if self.siem_enabled:
            await self._send_to_siem(event)

        # Log to application logger
        self._log_to_app_logger(event)

        return event_id

    async def log_authentication_success(
        self,
        request: Request,
        user_id: str,
        session_id: str,
        login_method: str = "password"
    ) -> str:
        """Log successful authentication."""
        return await self.log_event(
            SecurityEventType.LOGIN_SUCCESS,
            SecurityEventSeverity.LOW,
            request,
            f"User {user_id} logged in successfully",
            {"login_method": login_method},
            user_id=user_id,
            session_id=session_id
        )

    async def log_authentication_failure(
        self,
        request: Request,
        username: str,
        reason: str = "invalid_credentials"
    ) -> str:
        """Log failed authentication attempt."""
        client_ip = self._get_client_ip(request)

        # Track failed attempts for brute force detection
        await self._track_failed_attempt(client_ip, username)

        return await self.log_event(
            SecurityEventType.LOGIN_FAILURE,
            SecurityEventSeverity.MEDIUM,
            request,
            f"Login failed for username: {username}",
            {"username": username, "failure_reason": reason}
        )

    async def log_permission_denied(
        self,
        request: Request,
        user_id: str,
        required_permission: str,
        resource: str
    ) -> str:
        """Log permission denied event."""
        return await self.log_event(
            SecurityEventType.ACCESS_DENIED,
            SecurityEventSeverity.MEDIUM,
            request,
            f"Access denied to {resource}",
            {
                "required_permission": required_permission,
                "resource": resource
            },
            user_id=user_id
        )

    async def log_rate_limit_exceeded(
        self,
        request: Request,
        limit: int,
        window: str,
        user_id: str | None = None,
        api_key_id: str | None = None
    ) -> str:
        """Log rate limit violation."""
        client_ip = self._get_client_ip(request)

        # Track for pattern detection
        await self._track_suspicious_ip(client_ip, "rate_limit_violation")

        return await self.log_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            SecurityEventSeverity.MEDIUM,
            request,
            f"Rate limit exceeded: {limit} requests per {window}",
            {
                "limit": limit,
                "window": window,
                "endpoint": request.url.path
            },
            user_id=user_id,
            api_key_id=api_key_id
        )

    async def log_security_validation_failure(
        self,
        request: Request,
        attack_type: str,
        payload: str,
        user_id: str | None = None
    ) -> str:
        """Log security validation failure (attack attempt)."""
        # Map attack types to event types
        event_type_mapping = {
            "sql_injection": SecurityEventType.SQL_INJECTION_ATTEMPT,
            "xss": SecurityEventType.XSS_ATTEMPT,
            "path_traversal": SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
            "command_injection": SecurityEventType.COMMAND_INJECTION_ATTEMPT,
            "ssrf": SecurityEventType.SSRF_ATTEMPT,
            "xxe": SecurityEventType.XXE_ATTEMPT,
        }

        event_type = event_type_mapping.get(attack_type, SecurityEventType.SECURITY_POLICY_VIOLATION)
        client_ip = self._get_client_ip(request)

        # Track suspicious IP
        await self._track_suspicious_ip(client_ip, attack_type)

        return await self.log_event(
            event_type,
            SecurityEventSeverity.HIGH,
            request,
            f"Security attack detected: {attack_type}",
            {
                "attack_type": attack_type,
                "payload_sample": payload[:100],  # Limit payload size in logs
                "blocked": True
            },
            user_id=user_id
        )

    async def log_api_key_usage(
        self,
        request: Request,
        api_key_id: str,
        success: bool = True
    ) -> str:
        """Log API key usage."""
        event_type = SecurityEventType.API_KEY_USED if success else SecurityEventType.API_KEY_INVALID
        severity = SecurityEventSeverity.LOW if success else SecurityEventSeverity.MEDIUM

        return await self.log_event(
            event_type,
            severity,
            request,
            f"API key {'used successfully' if success else 'invalid/expired'}",
            {"api_key_id": api_key_id, "success": success},
            api_key_id=api_key_id
        )

    async def log_csrf_violation(
        self,
        request: Request,
        violation_type: str,
        user_id: str | None = None
    ) -> str:
        """Log CSRF protection violation."""
        event_type = SecurityEventType.CSRF_TOKEN_MISMATCH if "mismatch" in violation_type else SecurityEventType.CSRF_TOKEN_MISSING

        return await self.log_event(
            event_type,
            SecurityEventSeverity.HIGH,
            request,
            f"CSRF violation: {violation_type}",
            {"violation_type": violation_type},
            user_id=user_id
        )

    async def get_security_metrics(self, hours: int = 24) -> dict[str, Any]:
        """Get security metrics for the specified time period."""
        if not self.redis:
            return {"error": "Redis not available"}

        # This is a simplified implementation
        # In practice, you'd implement comprehensive metrics aggregation

        current_time = int(time.time())
        start_time = current_time - (hours * 3600)

        try:
            # Get events from the time period
            events_data = await self.redis.zrangebyscore(
                self.events_key, start_time, current_time
            )

            # Parse events and calculate metrics
            total_events = len(events_data)
            high_severity_events = 0
            attack_attempts = 0
            failed_logins = 0

            for event_data in events_data:
                try:
                    event = json.loads(event_data.decode())

                    if event.get("severity") in ["high", "critical"]:
                        high_severity_events += 1

                    event_type = event.get("event_type", "")
                    if "attempt" in event_type or "injection" in event_type:
                        attack_attempts += 1

                    if event_type == "login_failure":
                        failed_logins += 1

                except json.JSONDecodeError:
                    continue

            return {
                "time_period_hours": hours,
                "total_events": total_events,
                "high_severity_events": high_severity_events,
                "attack_attempts": attack_attempts,
                "failed_logins": failed_logins,
                "events_per_hour": round(total_events / hours, 2) if hours > 0 else 0
            }

        except Exception as e:
            logger.error("Failed to get security metrics", error=str(e))
            return {"error": str(e)}

    async def get_suspicious_ips(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get list of suspicious IP addresses."""
        if not self.redis:
            return []

        try:
            # Get top suspicious IPs by score
            ips_with_scores = await self.redis.zrevrange(
                self.suspicious_ips_key, 0, limit - 1, withscores=True
            )

            suspicious_ips = []
            for ip_bytes, score in ips_with_scores:
                ip = ip_bytes.decode()
                suspicious_ips.append({
                    "ip": ip,
                    "risk_score": int(score),
                    "last_activity": await self._get_ip_last_activity(ip)
                })

            return suspicious_ips

        except Exception as e:
            logger.error("Failed to get suspicious IPs", error=str(e))
            return []

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import secrets
        timestamp = int(time.time() * 1000)  # milliseconds
        random_part = secrets.token_hex(4)
        return f"evt_{timestamp}_{random_part}"

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        return request.client.host if request.client else "unknown"

    def _calculate_risk_score(
        self,
        event_type: SecurityEventType,
        severity: SecurityEventSeverity,
        details: dict[str, Any] | None
    ) -> int:
        """Calculate risk score for event (0-100)."""
        base_scores = {
            SecurityEventSeverity.LOW: 10,
            SecurityEventSeverity.MEDIUM: 30,
            SecurityEventSeverity.HIGH: 70,
            SecurityEventSeverity.CRITICAL: 90
        }

        # Start with base score
        risk_score = base_scores.get(severity, 10)

        # Adjust based on event type
        high_risk_events = {
            SecurityEventType.SQL_INJECTION_ATTEMPT,
            SecurityEventType.XSS_ATTEMPT,
            SecurityEventType.COMMAND_INJECTION_ATTEMPT,
            SecurityEventType.BRUTE_FORCE_ATTEMPT,
            SecurityEventType.PERMISSION_ESCALATION
        }

        if event_type in high_risk_events:
            risk_score = min(100, risk_score + 20)

        return risk_score

    def _is_suspicious_event(
        self,
        event_type: SecurityEventType,
        severity: SecurityEventSeverity
    ) -> bool:
        """Determine if event is suspicious."""
        suspicious_events = {
            SecurityEventType.SQL_INJECTION_ATTEMPT,
            SecurityEventType.XSS_ATTEMPT,
            SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
            SecurityEventType.COMMAND_INJECTION_ATTEMPT,
            SecurityEventType.SSRF_ATTEMPT,
            SecurityEventType.XXE_ATTEMPT,
            SecurityEventType.BRUTE_FORCE_ATTEMPT,
            SecurityEventType.SESSION_FINGERPRINT_MISMATCH,
            SecurityEventType.MALICIOUS_FILE_UPLOAD
        }

        return (
            event_type in suspicious_events or
            severity in [SecurityEventSeverity.HIGH, SecurityEventSeverity.CRITICAL]
        )

    async def _add_geolocation(self, event: SecurityEvent) -> None:
        """Add geolocation data to event (placeholder)."""
        # TODO: Integrate with GeoIP service
        pass

    def _add_compliance_tags(self, event: SecurityEvent) -> None:
        """Add compliance tags based on event type."""
        # GDPR compliance tags
        gdpr_events = {
            SecurityEventType.DATA_EXPORT,
            SecurityEventType.DATA_DELETION,
            SecurityEventType.PRIVACY_REQUEST
        }

        if event.event_type in gdpr_events:
            event.compliance_tags.append("GDPR")

        # SOX compliance tags
        if "admin" in event.request_path or "financial" in event.request_path:
            event.compliance_tags.append("SOX")

        # PCI DSS compliance tags
        if "payment" in event.request_path or "card" in str(event.details):
            event.compliance_tags.append("PCI-DSS")

    async def _store_event(self, event: SecurityEvent) -> None:
        """Store security event in Redis."""
        if not self.redis:
            return

        try:
            # Convert event to JSON
            event_data = json.dumps(asdict(event), default=str)

            # Store with timestamp as score for time-based queries
            timestamp_score = int(event.timestamp.timestamp())

            await self.redis.zadd(self.events_key, {event_data: timestamp_score})

            # Set expiration based on retention policy
            await self.redis.expire(self.events_key, self.retention_days * 86400)

        except Exception as e:
            logger.error("Failed to store security event", error=str(e))

    async def _update_metrics(self, event: SecurityEvent) -> None:
        """Update security metrics counters."""
        if not self.redis:
            return

        try:
            current_hour = int(time.time() / 3600)

            # Update hourly counters
            await self.redis.hincrby(f"{self.metrics_key}:{current_hour}", "total_events", 1)
            await self.redis.hincrby(f"{self.metrics_key}:{current_hour}", f"severity_{event.severity.value}", 1)
            await self.redis.hincrby(f"{self.metrics_key}:{current_hour}", f"type_{event.event_type.value}", 1)

            # Set expiration
            await self.redis.expire(f"{self.metrics_key}:{current_hour}", 7 * 86400)  # 7 days

        except Exception as e:
            logger.error("Failed to update security metrics", error=str(e))

    async def _track_failed_attempt(self, client_ip: str, username: str) -> None:
        """Track failed authentication attempts for brute force detection."""
        if not self.redis:
            return

        try:
            int(time.time())
            window_key = f"{self.failed_attempts_key}:{client_ip}:{username}"

            # Increment counter
            count = await self.redis.incr(window_key)

            # Set expiration if first attempt in window
            if count == 1:
                await self.redis.expire(window_key, 300)  # 5 minute window

            # Check for brute force threshold
            if count >= 5:  # 5 failed attempts in 5 minutes
                await self._track_suspicious_ip(client_ip, "brute_force_attempt")

        except Exception as e:
            logger.error("Failed to track failed attempt", error=str(e))

    async def _track_suspicious_ip(self, client_ip: str, reason: str) -> None:
        """Track suspicious IP addresses."""
        if not self.redis:
            return

        try:
            # Increment suspicious score
            await self.redis.zincrby(self.suspicious_ips_key, 10, client_ip)

            # Store reason
            await self.redis.hset(f"suspicious_ip_reasons:{client_ip}", reason, int(time.time()))

            # Set expiration
            await self.redis.expire(f"suspicious_ip_reasons:{client_ip}", 86400)  # 24 hours

        except Exception as e:
            logger.error("Failed to track suspicious IP", error=str(e))

    async def _check_suspicious_patterns(self, event: SecurityEvent) -> None:
        """Check for suspicious patterns and generate alerts."""
        if event.risk_score >= self.high_risk_threshold:
            await self._generate_alert(event, "High risk security event detected")

        if event.is_suspicious:
            await self._generate_alert(event, f"Suspicious activity: {event.event_type.value}")

    async def _generate_alert(self, event: SecurityEvent, alert_message: str) -> None:
        """Generate security alert."""
        if not self.redis:
            return

        try:
            alert_data = {
                "alert_id": self._generate_event_id(),
                "timestamp": event.timestamp.isoformat(),
                "message": alert_message,
                "event_id": event.event_id,
                "client_ip": event.client_ip,
                "severity": event.severity.value,
                "risk_score": event.risk_score
            }

            # Store alert
            await self.redis.lpush(self.alerts_key, json.dumps(alert_data))
            await self.redis.ltrim(self.alerts_key, 0, 999)  # Keep last 1000 alerts

        except Exception as e:
            logger.error("Failed to generate security alert", error=str(e))

    async def _send_to_siem(self, event: SecurityEvent) -> None:
        """Send event to SIEM system (placeholder)."""
        # TODO: Implement SIEM integration (e.g., Splunk, ELK, etc.)
        pass

    def _log_to_app_logger(self, event: SecurityEvent) -> None:
        """Log event to application logger."""
        log_data = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "severity": event.severity.value,
            "client_ip": event.client_ip,
            "user_id": event.user_id,
            "message": event.message,
            "risk_score": event.risk_score
        }

        if event.severity in [SecurityEventSeverity.HIGH, SecurityEventSeverity.CRITICAL]:
            logger.error("Security event", **log_data)
        elif event.severity == SecurityEventSeverity.MEDIUM:
            logger.warning("Security event", **log_data)
        else:
            logger.info("Security event", **log_data)

    async def _get_ip_last_activity(self, ip: str) -> str | None:
        """Get last activity timestamp for IP."""
        if not self.redis:
            return None

        try:
            reasons = await self.redis.hgetall(f"suspicious_ip_reasons:{ip}")
            if reasons:
                latest_timestamp = max(int(timestamp) for timestamp in reasons.values())
                return datetime.fromtimestamp(latest_timestamp, tz=UTC).isoformat()
        except Exception:
            pass

        return None
