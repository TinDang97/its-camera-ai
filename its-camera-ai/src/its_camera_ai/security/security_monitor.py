"""
Real-time Security Monitoring and Alerting System.

Provides comprehensive security monitoring for the ITS Camera AI system:
- Real-time threat detection and analysis
- Security metrics collection and alerting  
- Automated incident response triggers
- Compliance monitoring and reporting
- Security dashboard and visualization
- Integration with SIEM systems
"""

import json
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import redis.asyncio as redis
from prometheus_client import Counter, Gauge, Histogram

from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEventType(Enum):
    """Types of security events to monitor."""
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_FAILURE = "authz_failure"
    RATE_LIMIT_VIOLATION = "rate_limit_violation"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    INTRUSION_ATTEMPT = "intrusion_attempt"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_PAYLOAD = "malicious_payload"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_id: str
    event_type: SecurityEventType
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: str
    user_id: str | None
    endpoint: str
    details: dict[str, Any]
    raw_data: dict[str, Any]

    # Risk scoring
    risk_score: int = 0
    confidence: float = 0.0

    # Response tracking
    responded: bool = False
    response_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "threat_level": self.threat_level.value,
            "timestamp": self.timestamp.isoformat(),
            "source_ip": self.source_ip,
            "user_id": self.user_id,
            "endpoint": self.endpoint,
            "details": self.details,
            "risk_score": self.risk_score,
            "confidence": self.confidence,
            "responded": self.responded,
            "response_actions": self.response_actions
        }


@dataclass
class SecurityMetrics:
    """Security metrics tracking."""
    total_events: int = 0
    events_by_type: dict[SecurityEventType, int] = field(default_factory=lambda: defaultdict(int))
    events_by_severity: dict[ThreatLevel, int] = field(default_factory=lambda: defaultdict(int))
    high_risk_ips: set[str] = field(default_factory=set)
    blocked_ips: set[str] = field(default_factory=set)
    average_risk_score: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


class SecurityMonitor:
    """Real-time security monitoring and alerting system."""

    def __init__(self, redis_client: redis.Redis | None = None, settings=None):
        self.redis = redis_client
        self.settings = settings or get_settings()

        # Event storage and processing
        self.recent_events: deque[SecurityEvent] = deque(maxlen=10000)
        self.event_handlers: dict[SecurityEventType, list[Callable]] = defaultdict(list)

        # Risk analysis
        self.ip_risk_scores: dict[str, int] = defaultdict(int)
        self.user_risk_scores: dict[str, int] = defaultdict(int)
        self.endpoint_risk_scores: dict[str, int] = defaultdict(int)

        # Anomaly detection
        self.baseline_metrics: dict[str, float] = {}
        self.anomaly_threshold = 3.0  # Standard deviations

        # Rate limiting for alerts to prevent spam
        self.alert_rate_limiter: dict[str, float] = {}
        self.alert_cooldown = 300  # 5 minutes

        # Prometheus metrics
        self.setup_prometheus_metrics()

        # Automated response actions
        self.response_actions: dict[ThreatLevel, list[Callable]] = {
            ThreatLevel.LOW: [],
            ThreatLevel.MEDIUM: [self._log_security_event],
            ThreatLevel.HIGH: [self._log_security_event, self._alert_security_team],
            ThreatLevel.CRITICAL: [
                self._log_security_event,
                self._alert_security_team,
                self._trigger_incident_response
            ]
        }

    def setup_prometheus_metrics(self) -> None:
        """Set up Prometheus metrics for security monitoring."""
        self.security_events_total = Counter(
            'security_events_total',
            'Total number of security events',
            ['event_type', 'severity', 'source']
        )

        self.security_risk_score = Gauge(
            'security_risk_score',
            'Current security risk score',
            ['ip_address', 'user_id']
        )

        self.security_response_time = Histogram(
            'security_response_time_seconds',
            'Time taken to respond to security events',
            ['event_type', 'severity']
        )

        self.blocked_requests_total = Counter(
            'blocked_requests_total',
            'Total number of blocked requests',
            ['reason', 'source_ip']
        )

    async def process_security_event(self, event: SecurityEvent) -> None:
        """Process a security event through the monitoring pipeline."""
        try:
            # Store event
            self.recent_events.append(event)

            # Calculate risk score
            event.risk_score = self._calculate_risk_score(event)
            event.confidence = self._calculate_confidence(event)

            # Update metrics
            await self._update_metrics(event)

            # Store in Redis for distributed processing
            if self.redis:
                await self._store_event_in_redis(event)

            # Trigger automated responses
            await self._trigger_responses(event)

            # Update risk tracking
            self._update_risk_tracking(event)

            # Check for anomalies
            await self._check_for_anomalies(event)

            # Log for audit trail
            logger.info("Security event processed",
                       event_id=event.event_id,
                       event_type=event.event_type.value,
                       threat_level=event.threat_level.value,
                       risk_score=event.risk_score,
                       source_ip=event.source_ip)

        except Exception as e:
            logger.error("Failed to process security event",
                        event_id=event.event_id,
                        error=str(e))

    def _calculate_risk_score(self, event: SecurityEvent) -> int:
        """Calculate risk score for security event."""
        base_scores = {
            SecurityEventType.AUTHENTICATION_FAILURE: 10,
            SecurityEventType.AUTHORIZATION_FAILURE: 15,
            SecurityEventType.RATE_LIMIT_VIOLATION: 5,
            SecurityEventType.INPUT_VALIDATION_FAILURE: 20,
            SecurityEventType.SUSPICIOUS_ACTIVITY: 25,
            SecurityEventType.INTRUSION_ATTEMPT: 50,
            SecurityEventType.DATA_BREACH_ATTEMPT: 80,
            SecurityEventType.PRIVILEGE_ESCALATION: 90,
            SecurityEventType.MALICIOUS_PAYLOAD: 70,
            SecurityEventType.ANOMALOUS_BEHAVIOR: 30,
        }

        risk_score = base_scores.get(event.event_type, 10)

        # Adjust based on threat level
        multipliers = {
            ThreatLevel.LOW: 1.0,
            ThreatLevel.MEDIUM: 1.5,
            ThreatLevel.HIGH: 2.0,
            ThreatLevel.CRITICAL: 3.0
        }
        risk_score = int(risk_score * multipliers[event.threat_level])

        # Adjust based on IP reputation
        ip_risk = self.ip_risk_scores.get(event.source_ip, 0)
        if ip_risk > 50:
            risk_score = int(risk_score * 1.5)

        # Adjust based on user reputation
        if event.user_id:
            user_risk = self.user_risk_scores.get(event.user_id, 0)
            if user_risk > 50:
                risk_score = int(risk_score * 1.3)

        # Adjust based on endpoint sensitivity
        endpoint_risk = self.endpoint_risk_scores.get(event.endpoint, 0)
        risk_score = int(risk_score * (1 + endpoint_risk / 100))

        # Check for repeat offenses (time-based)
        recent_events_from_ip = [
            e for e in self.recent_events
            if e.source_ip == event.source_ip
            and (event.timestamp - e.timestamp).total_seconds() < 3600  # Last hour
        ]

        if len(recent_events_from_ip) > 5:
            risk_score = int(risk_score * 1.5)  # Repeat offender

        return min(risk_score, 100)  # Cap at 100

    def _calculate_confidence(self, event: SecurityEvent) -> float:
        """Calculate confidence score for the security event."""
        # Base confidence based on event type reliability
        base_confidence = {
            SecurityEventType.AUTHENTICATION_FAILURE: 0.9,
            SecurityEventType.AUTHORIZATION_FAILURE: 0.85,
            SecurityEventType.RATE_LIMIT_VIOLATION: 0.95,
            SecurityEventType.INPUT_VALIDATION_FAILURE: 0.8,
            SecurityEventType.SUSPICIOUS_ACTIVITY: 0.6,
            SecurityEventType.INTRUSION_ATTEMPT: 0.7,
            SecurityEventType.DATA_BREACH_ATTEMPT: 0.8,
            SecurityEventType.PRIVILEGE_ESCALATION: 0.9,
            SecurityEventType.MALICIOUS_PAYLOAD: 0.85,
            SecurityEventType.ANOMALOUS_BEHAVIOR: 0.5,
        }

        confidence = base_confidence.get(event.event_type, 0.5)

        # Adjust based on data quality
        if event.details and len(event.details) > 3:
            confidence += 0.1

        # Adjust based on correlation with other events
        correlated_events = [
            e for e in self.recent_events[-100:]  # Last 100 events
            if e.source_ip == event.source_ip
            and e.event_type == event.event_type
            and (event.timestamp - e.timestamp).total_seconds() < 300  # Last 5 minutes
        ]

        if len(correlated_events) > 1:
            confidence = min(confidence + 0.2, 1.0)

        return confidence

    async def _update_metrics(self, event: SecurityEvent) -> None:
        """Update Prometheus metrics."""
        # Update counters
        self.security_events_total.labels(
            event_type=event.event_type.value,
            severity=event.threat_level.value,
            source=event.source_ip
        ).inc()

        # Update risk score gauge
        self.security_risk_score.labels(
            ip_address=event.source_ip,
            user_id=event.user_id or "anonymous"
        ).set(event.risk_score)

    async def _store_event_in_redis(self, event: SecurityEvent) -> None:
        """Store event in Redis for distributed processing."""
        try:
            event_data = json.dumps(event.to_dict())

            # Store individual event
            await self.redis.setex(
                f"security_event:{event.event_id}",
                3600,  # 1 hour TTL
                event_data
            )

            # Add to time-series for analysis
            await self.redis.zadd(
                "security_events_timeline",
                {event.event_id: time.time()}
            )

            # Add to IP-based tracking
            await self.redis.sadd(
                f"security_events_ip:{event.source_ip}",
                event.event_id
            )
            await self.redis.expire(f"security_events_ip:{event.source_ip}", 86400)  # 24 hours

            # Store metrics
            await self._update_redis_metrics(event)

        except Exception as e:
            logger.error("Failed to store event in Redis", error=str(e))

    async def _update_redis_metrics(self, event: SecurityEvent) -> None:
        """Update security metrics in Redis."""
        try:
            # Update event counters
            await self.redis.hincrby("security_metrics", "total_events", 1)
            await self.redis.hincrby("security_metrics", f"events_{event.event_type.value}", 1)
            await self.redis.hincrby("security_metrics", f"severity_{event.threat_level.value}", 1)

            # Update IP risk scores
            await self.redis.hincrby("ip_risk_scores", event.source_ip, event.risk_score)

            # Track high-risk IPs
            if event.risk_score > 50:
                await self.redis.sadd("high_risk_ips", event.source_ip)

            # Update hourly statistics
            hour_key = f"security_hourly:{datetime.now(UTC).strftime('%Y%m%d%H')}"
            await self.redis.hincrby(hour_key, "total_events", 1)
            await self.redis.hincrby(hour_key, f"severity_{event.threat_level.value}", 1)
            await self.redis.expire(hour_key, 86400 * 7)  # Keep for 7 days

        except Exception as e:
            logger.error("Failed to update Redis metrics", error=str(e))

    async def _trigger_responses(self, event: SecurityEvent) -> None:
        """Trigger automated responses based on threat level."""
        try:
            responses = self.response_actions.get(event.threat_level, [])

            for response_func in responses:
                try:
                    start_time = time.time()
                    await response_func(event)
                    response_time = time.time() - start_time

                    # Record response time metrics
                    self.security_response_time.labels(
                        event_type=event.event_type.value,
                        severity=event.threat_level.value
                    ).observe(response_time)

                    # Track successful response
                    event.response_actions.append(response_func.__name__)

                except Exception as e:
                    logger.error("Response action failed",
                               action=response_func.__name__,
                               event_id=event.event_id,
                               error=str(e))

            event.responded = len(event.response_actions) > 0

        except Exception as e:
            logger.error("Failed to trigger responses", event_id=event.event_id, error=str(e))

    def _update_risk_tracking(self, event: SecurityEvent) -> None:
        """Update risk tracking for IPs, users, and endpoints."""
        # Update IP risk score
        self.ip_risk_scores[event.source_ip] = min(
            self.ip_risk_scores[event.source_ip] + event.risk_score,
            1000  # Cap at 1000
        )

        # Update user risk score
        if event.user_id:
            self.user_risk_scores[event.user_id] = min(
                self.user_risk_scores[event.user_id] + event.risk_score,
                1000
            )

        # Update endpoint risk score
        endpoint_risk = min(event.risk_score / 10, 10)  # Scale down for endpoints
        self.endpoint_risk_scores[event.endpoint] = min(
            self.endpoint_risk_scores[event.endpoint] + endpoint_risk,
            100
        )

        # Decay risk scores over time (simple implementation)
        current_time = time.time()
        for ip in list(self.ip_risk_scores.keys()):
            # Reduce by 1 point per hour (simplified)
            hours_passed = max(1, int((current_time - time.time()) / 3600))
            self.ip_risk_scores[ip] = max(0, self.ip_risk_scores[ip] - hours_passed)

    async def _check_for_anomalies(self, event: SecurityEvent) -> None:
        """Check for anomalous patterns in security events."""
        try:
            # Check for unusual event frequency
            recent_count = len([
                e for e in self.recent_events
                if (event.timestamp - e.timestamp).total_seconds() < 300  # Last 5 minutes
            ])

            # Get baseline from Redis if available
            baseline_key = "baseline_event_rate"
            if self.redis:
                baseline = await self.redis.get(baseline_key)
                if baseline:
                    baseline_rate = float(baseline)
                    if recent_count > baseline_rate * self.anomaly_threshold:
                        await self._create_anomaly_event(
                            "unusual_event_frequency",
                            f"Event rate {recent_count} exceeds baseline {baseline_rate}",
                            event
                        )

            # Check for unusual IP behavior
            ip_events = [e for e in self.recent_events if e.source_ip == event.source_ip]
            if len(ip_events) > 20:  # More than 20 events from same IP
                await self._create_anomaly_event(
                    "suspicious_ip_activity",
                    f"IP {event.source_ip} generated {len(ip_events)} events",
                    event
                )

            # Check for privilege escalation patterns
            if event.user_id:
                user_events = [e for e in self.recent_events if e.user_id == event.user_id]
                auth_failures = [e for e in user_events if e.event_type == SecurityEventType.AUTHORIZATION_FAILURE]
                if len(auth_failures) > 5:
                    await self._create_anomaly_event(
                        "potential_privilege_escalation",
                        f"User {event.user_id} has {len(auth_failures)} authorization failures",
                        event
                    )

        except Exception as e:
            logger.error("Anomaly detection failed", error=str(e))

    async def _create_anomaly_event(self, anomaly_type: str, description: str, source_event: SecurityEvent) -> None:
        """Create an anomaly event based on detected patterns."""
        anomaly_event = SecurityEvent(
            event_id=f"anomaly_{int(time.time())}_{source_event.event_id}",
            event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
            threat_level=ThreatLevel.MEDIUM,
            timestamp=datetime.now(UTC),
            source_ip=source_event.source_ip,
            user_id=source_event.user_id,
            endpoint=source_event.endpoint,
            details={
                "anomaly_type": anomaly_type,
                "description": description,
                "source_event_id": source_event.event_id
            },
            raw_data={}
        )

        await self.process_security_event(anomaly_event)

    async def _log_security_event(self, event: SecurityEvent) -> None:
        """Log security event for audit trail."""
        logger.warning("Security event detected",
                      event_id=event.event_id,
                      event_type=event.event_type.value,
                      threat_level=event.threat_level.value,
                      risk_score=event.risk_score,
                      confidence=event.confidence,
                      source_ip=event.source_ip,
                      user_id=event.user_id,
                      endpoint=event.endpoint,
                      details=event.details)

    async def _alert_security_team(self, event: SecurityEvent) -> None:
        """Send alert to security team."""
        alert_key = f"alert_{event.event_type.value}_{event.source_ip}"
        current_time = time.time()

        # Check rate limiting
        if alert_key in self.alert_rate_limiter:
            if current_time - self.alert_rate_limiter[alert_key] < self.alert_cooldown:
                return  # Skip alert due to rate limiting

        self.alert_rate_limiter[alert_key] = current_time

        # Send alert (integrate with your alerting system)
        alert_message = {
            "title": f"Security Alert: {event.event_type.value}",
            "severity": event.threat_level.value,
            "description": f"Risk Score: {event.risk_score}, IP: {event.source_ip}",
            "details": event.details,
            "timestamp": event.timestamp.isoformat(),
            "event_id": event.event_id
        }

        # Log alert for now (replace with actual alerting integration)
        logger.error("SECURITY ALERT", **alert_message)

        # Store alert in Redis for dashboard
        if self.redis:
            await self.redis.lpush("security_alerts", json.dumps(alert_message))
            await self.redis.ltrim("security_alerts", 0, 1000)  # Keep last 1000 alerts

    async def _trigger_incident_response(self, event: SecurityEvent) -> None:
        """Trigger automated incident response."""
        # Create incident record
        incident_id = f"incident_{int(time.time())}_{event.event_id}"

        incident_data = {
            "incident_id": incident_id,
            "trigger_event": event.event_id,
            "severity": event.threat_level.value,
            "status": "open",
            "created_at": datetime.now(UTC).isoformat(),
            "source_ip": event.source_ip,
            "user_id": event.user_id,
            "auto_actions": []
        }

        # Determine automatic actions based on event type
        if event.event_type == SecurityEventType.INTRUSION_ATTEMPT:
            if event.risk_score > 80:
                await self._block_ip_address(event.source_ip, incident_id)
                incident_data["auto_actions"].append("ip_blocked")

        elif event.event_type == SecurityEventType.DATA_BREACH_ATTEMPT:
            if event.user_id:
                await self._suspend_user_account(event.user_id, incident_id)
                incident_data["auto_actions"].append("user_suspended")

        # Store incident
        if self.redis:
            await self.redis.setex(
                f"security_incident:{incident_id}",
                86400 * 30,  # 30 days
                json.dumps(incident_data)
            )

        logger.critical("Security incident triggered",
                       incident_id=incident_id,
                       event_id=event.event_id,
                       auto_actions=incident_data["auto_actions"])

    async def _block_ip_address(self, ip_address: str, incident_id: str) -> None:
        """Block IP address automatically."""
        try:
            # Add to blocked IPs set
            if self.redis:
                await self.redis.sadd("blocked_ips", ip_address)
                await self.redis.setex(f"ip_block:{ip_address}", 3600, incident_id)  # 1 hour block

            # Update internal tracking
            self.ip_risk_scores[ip_address] = 1000  # Maximum risk

            # Update metrics
            self.blocked_requests_total.labels(
                reason="automatic_incident_response",
                source_ip=ip_address
            ).inc()

            logger.warning("IP address blocked automatically",
                          ip_address=ip_address,
                          incident_id=incident_id)

        except Exception as e:
            logger.error("Failed to block IP address",
                        ip_address=ip_address,
                        incident_id=incident_id,
                        error=str(e))

    async def _suspend_user_account(self, user_id: str, incident_id: str) -> None:
        """Suspend user account automatically."""
        try:
            # Add to suspended users set
            if self.redis:
                await self.redis.sadd("suspended_users", user_id)
                await self.redis.setex(f"user_suspend:{user_id}", 3600, incident_id)  # 1 hour suspension

            logger.warning("User account suspended automatically",
                          user_id=user_id,
                          incident_id=incident_id)

        except Exception as e:
            logger.error("Failed to suspend user account",
                        user_id=user_id,
                        incident_id=incident_id,
                        error=str(e))

    async def get_security_dashboard_data(self) -> dict[str, Any]:
        """Get security dashboard data."""
        try:
            # Calculate metrics from recent events
            recent_events = list(self.recent_events)[-1000:]  # Last 1000 events

            # Event counts by type
            event_counts = defaultdict(int)
            severity_counts = defaultdict(int)

            for event in recent_events:
                event_counts[event.event_type.value] += 1
                severity_counts[event.threat_level.value] += 1

            # Top risk IPs
            top_risk_ips = sorted(
                self.ip_risk_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            # Recent high-risk events
            high_risk_events = [
                event.to_dict() for event in recent_events
                if event.risk_score > 50
            ][-20:]  # Last 20 high-risk events

            dashboard_data = {
                "summary": {
                    "total_events": len(recent_events),
                    "high_risk_events": len([e for e in recent_events if e.risk_score > 50]),
                    "blocked_ips": len(self.ip_risk_scores),
                    "average_risk_score": sum(e.risk_score for e in recent_events) / len(recent_events) if recent_events else 0
                },
                "event_counts": dict(event_counts),
                "severity_distribution": dict(severity_counts),
                "top_risk_ips": top_risk_ips,
                "recent_high_risk_events": high_risk_events,
                "timestamp": datetime.now(UTC).isoformat()
            }

            return dashboard_data

        except Exception as e:
            logger.error("Failed to generate dashboard data", error=str(e))
            return {"error": str(e)}

    async def cleanup_old_data(self) -> None:
        """Clean up old security data to prevent memory issues."""
        try:
            current_time = time.time()
            cutoff_time = current_time - 86400  # 24 hours

            # Clean up rate limiter
            self.alert_rate_limiter = {
                k: v for k, v in self.alert_rate_limiter.items()
                if current_time - v < self.alert_cooldown
            }

            # Clean up Redis data
            if self.redis:
                # Remove old events from timeline
                await self.redis.zremrangebyscore("security_events_timeline", 0, cutoff_time)

                # Clean up expired IP event sets
                # (This would be more sophisticated in production)
                pass

            logger.info("Security data cleanup completed")

        except Exception as e:
            logger.error("Failed to cleanup security data", error=str(e))


# Helper functions to create security events

def create_auth_failure_event(source_ip: str, endpoint: str, user_id: str | None = None, **details) -> SecurityEvent:
    """Create authentication failure event."""
    return SecurityEvent(
        event_id=f"auth_fail_{int(time.time())}_{source_ip.replace('.', '_')}",
        event_type=SecurityEventType.AUTHENTICATION_FAILURE,
        threat_level=ThreatLevel.MEDIUM,
        timestamp=datetime.now(UTC),
        source_ip=source_ip,
        user_id=user_id,
        endpoint=endpoint,
        details=details,
        raw_data={}
    )


def create_validation_failure_event(source_ip: str, endpoint: str, attack_type: str, payload: str, **details) -> SecurityEvent:
    """Create input validation failure event."""
    threat_level = ThreatLevel.HIGH if attack_type in ["sql_injection", "xss", "command_injection"] else ThreatLevel.MEDIUM

    return SecurityEvent(
        event_id=f"validation_fail_{int(time.time())}_{source_ip.replace('.', '_')}",
        event_type=SecurityEventType.INPUT_VALIDATION_FAILURE,
        threat_level=threat_level,
        timestamp=datetime.now(UTC),
        source_ip=source_ip,
        user_id=None,
        endpoint=endpoint,
        details={
            "attack_type": attack_type,
            "blocked_payload": payload[:200],  # Truncate for safety
            **details
        },
        raw_data={}
    )


def create_rate_limit_event(source_ip: str, endpoint: str, limit_type: str, **details) -> SecurityEvent:
    """Create rate limit violation event."""
    return SecurityEvent(
        event_id=f"rate_limit_{int(time.time())}_{source_ip.replace('.', '_')}",
        event_type=SecurityEventType.RATE_LIMIT_VIOLATION,
        threat_level=ThreatLevel.LOW,
        timestamp=datetime.now(UTC),
        source_ip=source_ip,
        user_id=details.get("user_id"),
        endpoint=endpoint,
        details={
            "limit_type": limit_type,
            **details
        },
        raw_data={}
    )


# Global security monitor instance
_security_monitor: SecurityMonitor | None = None


def get_security_monitor() -> SecurityMonitor:
    """Get the global security monitor instance."""
    global _security_monitor
    if _security_monitor is None:
        _security_monitor = SecurityMonitor()
    return _security_monitor
