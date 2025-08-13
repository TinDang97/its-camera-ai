"""
Session security manager for comprehensive session management.

Provides:
- Session timeout configuration
- Concurrent session limits
- Session fingerprinting
- Secure session storage
- Session invalidation on suspicious activity
- Session rotation and regeneration
"""

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import redis.asyncio as redis
from fastapi import Request

from ...core.config import get_settings
from ...core.logging import get_logger

logger = get_logger(__name__)


class SessionStatus(Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPICIOUS = "suspicious"
    LOCKED = "locked"


@dataclass
class SessionFingerprint:
    """Session fingerprint for security validation."""
    user_agent: str
    ip_address: str
    accept_language: str
    accept_encoding: str
    timezone: str | None = None
    screen_resolution: str | None = None
    fingerprint_hash: str = field(init=False)

    def __post_init__(self):
        """Generate fingerprint hash."""
        fingerprint_data = f"{self.user_agent}:{self.ip_address}:{self.accept_language}:{self.accept_encoding}"
        self.fingerprint_hash = hashlib.sha256(fingerprint_data.encode()).hexdigest()


@dataclass
class Session:
    """Session data model."""
    session_id: str
    user_id: str
    fingerprint: SessionFingerprint
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    status: SessionStatus
    device_name: str | None = None
    location: str | None = None
    is_remember_me: bool = False
    login_method: str = "password"  # password, mfa, oauth, etc.
    risk_score: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """Comprehensive session management."""

    def __init__(self, redis_client: redis.Redis | None = None, settings=None):
        self.redis = redis_client
        self.settings = settings or get_settings()

        # Session configuration
        self.session_timeout = timedelta(minutes=self.settings.security.session_timeout_minutes)
        self.max_sessions_per_user = self.settings.security.max_sessions_per_user
        self.sliding_expiration = self.settings.security.session_sliding_expiration
        self.remember_me_duration = timedelta(days=30)

        # Session security settings
        self.fingerprint_validation_enabled = True
        self.ip_change_threshold = 2  # Allow 2 IP changes before flagging
        self.suspicious_activity_threshold = 80

        # Redis key prefixes
        self.session_prefix = "session"
        self.user_sessions_prefix = "user_sessions"
        self.fingerprint_prefix = "fingerprint"
        self.activity_prefix = "activity"

    async def create_session(
        self,
        user_id: str,
        request: Request,
        remember_me: bool = False,
        device_name: str | None = None
    ) -> Session:
        """
        Create a new session.

        Args:
            user_id: User ID
            request: HTTP request for fingerprinting
            remember_me: Whether this is a "remember me" session
            device_name: Optional device name

        Returns:
            Created session
        """
        # Generate session ID
        session_id = self._generate_session_id()

        # Create fingerprint
        fingerprint = self._create_fingerprint(request)

        # Determine expiration
        now = datetime.now(UTC)
        if remember_me:
            expires_at = now + self.remember_me_duration
        else:
            expires_at = now + self.session_timeout

        # Check concurrent session limit
        await self._enforce_session_limit(user_id)

        # Create session
        session = Session(
            session_id=session_id,
            user_id=user_id,
            fingerprint=fingerprint,
            created_at=now,
            last_activity=now,
            expires_at=expires_at,
            status=SessionStatus.ACTIVE,
            device_name=device_name,
            is_remember_me=remember_me,
            location=await self._get_location_from_ip(fingerprint.ip_address)
        )

        # Store session
        await self._store_session(session)
        await self._add_user_session(user_id, session_id)

        # Log session creation
        logger.info(
            "Session created",
            session_id=session_id,
            user_id=user_id,
            ip=fingerprint.ip_address,
            device=device_name,
            remember_me=remember_me
        )

        return session

    async def validate_session(self, session_id: str, request: Request) -> Session | None:
        """
        Validate session and perform security checks.

        Args:
            session_id: Session ID to validate
            request: HTTP request for fingerprint validation

        Returns:
            Valid session or None if invalid
        """
        if not session_id:
            return None

        # Load session
        session = await self._load_session(session_id)
        if not session:
            return None

        # Check session status
        if session.status != SessionStatus.ACTIVE:
            logger.warning("Session validation failed - inactive status",
                         session_id=session_id, status=session.status.value)
            return None

        # Check expiration
        now = datetime.now(UTC)
        if now > session.expires_at:
            await self._expire_session(session_id)
            logger.warning("Session validation failed - expired", session_id=session_id)
            return None

        # Fingerprint validation
        if self.fingerprint_validation_enabled:
            current_fingerprint = self._create_fingerprint(request)
            if not self._validate_fingerprint(session.fingerprint, current_fingerprint):
                await self._flag_suspicious_session(session_id, "fingerprint_mismatch")
                logger.warning("Session validation failed - fingerprint mismatch",
                             session_id=session_id)
                return None

        # Update activity and extend session if sliding expiration enabled
        session.last_activity = now
        if self.sliding_expiration and not session.is_remember_me:
            session.expires_at = now + self.session_timeout

        # Store updated session
        await self._store_session(session)

        # Track activity
        await self._track_activity(session_id, request)

        return session

    async def regenerate_session(self, old_session_id: str) -> str | None:
        """
        Regenerate session ID for security (e.g., after login).

        Args:
            old_session_id: Current session ID

        Returns:
            New session ID or None if failed
        """
        # Load existing session
        session = await self._load_session(old_session_id)
        if not session:
            return None

        # Generate new session ID
        new_session_id = self._generate_session_id()

        # Update session
        session.session_id = new_session_id
        session.last_activity = datetime.now(UTC)

        # Store with new ID and remove old one
        await self._store_session(session)
        await self._delete_session(old_session_id)

        # Update user session list
        await self._remove_user_session(session.user_id, old_session_id)
        await self._add_user_session(session.user_id, new_session_id)

        logger.info("Session regenerated",
                   old_session_id=old_session_id,
                   new_session_id=new_session_id,
                   user_id=session.user_id)

        return new_session_id

    async def revoke_session(self, session_id: str, reason: str = "manual") -> bool:
        """
        Revoke a specific session.

        Args:
            session_id: Session ID to revoke
            reason: Reason for revocation

        Returns:
            True if successfully revoked
        """
        session = await self._load_session(session_id)
        if not session:
            return False

        # Update session status
        session.status = SessionStatus.REVOKED
        session.metadata["revocation_reason"] = reason
        session.metadata["revoked_at"] = datetime.now(UTC).isoformat()

        # Store updated session (keep for audit purposes)
        await self._store_session(session)

        # Remove from user sessions
        await self._remove_user_session(session.user_id, session_id)

        logger.info("Session revoked",
                   session_id=session_id,
                   user_id=session.user_id,
                   reason=reason)

        return True

    async def revoke_user_sessions(self, user_id: str, exclude_session: str | None = None) -> int:
        """
        Revoke all sessions for a user.

        Args:
            user_id: User ID
            exclude_session: Session ID to exclude from revocation

        Returns:
            Number of sessions revoked
        """
        session_ids = await self._get_user_sessions(user_id)
        revoked_count = 0

        for session_id in session_ids:
            if session_id != exclude_session:
                if await self.revoke_session(session_id, "user_logout_all"):
                    revoked_count += 1

        logger.info("User sessions revoked",
                   user_id=user_id,
                   count=revoked_count,
                   excluded=exclude_session)

        return revoked_count

    async def get_user_sessions(self, user_id: str) -> list[Session]:
        """
        Get all active sessions for a user.

        Args:
            user_id: User ID

        Returns:
            List of active sessions
        """
        session_ids = await self._get_user_sessions(user_id)
        sessions = []

        for session_id in session_ids:
            session = await self._load_session(session_id)
            if session and session.status == SessionStatus.ACTIVE:
                sessions.append(session)

        return sessions

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        if not self.redis:
            return 0

        # This would typically be run as a background task
        # For now, we'll just mark it as a placeholder
        logger.info("Session cleanup started")

        # TODO: Implement Redis SCAN to find expired sessions
        # This is a simplified implementation
        cleanup_count = 0

        logger.info("Session cleanup completed", count=cleanup_count)
        return cleanup_count

    def _generate_session_id(self) -> str:
        """Generate cryptographically secure session ID."""
        return secrets.token_urlsafe(32)

    def _create_fingerprint(self, request: Request) -> SessionFingerprint:
        """Create session fingerprint from request."""
        return SessionFingerprint(
            user_agent=request.headers.get("user-agent", ""),
            ip_address=self._get_client_ip(request),
            accept_language=request.headers.get("accept-language", ""),
            accept_encoding=request.headers.get("accept-encoding", ""),
            timezone=request.headers.get("x-timezone")
        )

    def _validate_fingerprint(self, stored: SessionFingerprint, current: SessionFingerprint) -> bool:
        """Validate session fingerprint."""
        # Allow some flexibility in fingerprint validation
        score = 0
        total_checks = 0

        # User agent check (strict)
        if stored.user_agent == current.user_agent:
            score += 3
        total_checks += 3

        # IP address check (allow some changes)
        if stored.ip_address == current.ip_address:
            score += 2
        elif self._is_similar_ip(stored.ip_address, current.ip_address):
            score += 1
        total_checks += 2

        # Language check
        if stored.accept_language == current.accept_language:
            score += 1
        total_checks += 1

        # Accept encoding check
        if stored.accept_encoding == current.accept_encoding:
            score += 1
        total_checks += 1

        # Calculate match percentage
        match_percentage = (score / total_checks) * 100

        # Require at least 70% match
        return match_percentage >= 70

    def _is_similar_ip(self, ip1: str, ip2: str) -> bool:
        """Check if IPs are from similar networks."""
        try:
            import ipaddress

            addr1 = ipaddress.ip_address(ip1)
            addr2 = ipaddress.ip_address(ip2)

            # Same /24 network
            if isinstance(addr1, ipaddress.IPv4Address) and isinstance(addr2, ipaddress.IPv4Address):
                network1 = ipaddress.ip_network(f"{addr1}/24", strict=False)
                return addr2 in network1

        except Exception:
            pass

        return False

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        return request.client.host if request.client else "unknown"

    async def _get_location_from_ip(self, ip_address: str) -> str | None:
        """Get location from IP address (placeholder)."""
        # TODO: Integrate with GeoIP service
        return None

    async def _store_session(self, session: Session) -> None:
        """Store session in Redis."""
        if not self.redis:
            return

        session_key = f"{self.session_prefix}:{session.session_id}"
        session_data = {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "status": session.status.value,
            "device_name": session.device_name or "",
            "location": session.location or "",
            "is_remember_me": str(session.is_remember_me),
            "login_method": session.login_method,
            "risk_score": str(session.risk_score),
            "fingerprint_hash": session.fingerprint.fingerprint_hash,
            "ip_address": session.fingerprint.ip_address,
            "user_agent": session.fingerprint.user_agent,
            "metadata": str(session.metadata)
        }

        # Calculate TTL
        ttl = int((session.expires_at - datetime.now(UTC)).total_seconds())
        if ttl > 0:
            await self.redis.hset(session_key, mapping=session_data)
            await self.redis.expire(session_key, ttl)

    async def _load_session(self, session_id: str) -> Session | None:
        """Load session from Redis."""
        if not self.redis:
            return None

        session_key = f"{self.session_prefix}:{session_id}"
        session_data = await self.redis.hgetall(session_key)

        if not session_data:
            return None

        try:
            # Reconstruct fingerprint
            fingerprint = SessionFingerprint(
                user_agent=session_data.get(b"user_agent", b"").decode(),
                ip_address=session_data.get(b"ip_address", b"").decode(),
                accept_language="",  # Not stored separately
                accept_encoding=""   # Not stored separately
            )

            # Reconstruct session
            session = Session(
                session_id=session_data.get(b"session_id", b"").decode(),
                user_id=session_data.get(b"user_id", b"").decode(),
                fingerprint=fingerprint,
                created_at=datetime.fromisoformat(session_data.get(b"created_at", b"").decode()),
                last_activity=datetime.fromisoformat(session_data.get(b"last_activity", b"").decode()),
                expires_at=datetime.fromisoformat(session_data.get(b"expires_at", b"").decode()),
                status=SessionStatus(session_data.get(b"status", b"active").decode()),
                device_name=session_data.get(b"device_name", b"").decode() or None,
                location=session_data.get(b"location", b"").decode() or None,
                is_remember_me=session_data.get(b"is_remember_me", b"False").decode() == "True",
                login_method=session_data.get(b"login_method", b"password").decode(),
                risk_score=int(session_data.get(b"risk_score", b"0").decode()),
                metadata=eval(session_data.get(b"metadata", b"{}").decode()) if session_data.get(b"metadata") else {}
            )

            return session

        except Exception as e:
            logger.error("Failed to load session", session_id=session_id, error=str(e))
            return None

    async def _delete_session(self, session_id: str) -> None:
        """Delete session from Redis."""
        if not self.redis:
            return

        session_key = f"{self.session_prefix}:{session_id}"
        await self.redis.delete(session_key)

    async def _add_user_session(self, user_id: str, session_id: str) -> None:
        """Add session to user's session list."""
        if not self.redis:
            return

        user_sessions_key = f"{self.user_sessions_prefix}:{user_id}"
        await self.redis.sadd(user_sessions_key, session_id)
        await self.redis.expire(user_sessions_key, int(self.remember_me_duration.total_seconds()))

    async def _remove_user_session(self, user_id: str, session_id: str) -> None:
        """Remove session from user's session list."""
        if not self.redis:
            return

        user_sessions_key = f"{self.user_sessions_prefix}:{user_id}"
        await self.redis.srem(user_sessions_key, session_id)

    async def _get_user_sessions(self, user_id: str) -> list[str]:
        """Get all session IDs for a user."""
        if not self.redis:
            return []

        user_sessions_key = f"{self.user_sessions_prefix}:{user_id}"
        session_ids = await self.redis.smembers(user_sessions_key)
        return [sid.decode() for sid in session_ids]

    async def _enforce_session_limit(self, user_id: str) -> None:
        """Enforce maximum sessions per user."""
        if not self.max_sessions_per_user:
            return

        session_ids = await self._get_user_sessions(user_id)
        active_sessions = []

        # Check which sessions are still active
        for session_id in session_ids:
            session = await self._load_session(session_id)
            if session and session.status == SessionStatus.ACTIVE:
                active_sessions.append(session)

        # If at limit, remove oldest sessions
        if len(active_sessions) >= self.max_sessions_per_user:
            # Sort by last activity
            active_sessions.sort(key=lambda s: s.last_activity)

            # Remove oldest sessions
            sessions_to_remove = len(active_sessions) - self.max_sessions_per_user + 1
            for session in active_sessions[:sessions_to_remove]:
                await self.revoke_session(session.session_id, "session_limit_exceeded")

    async def _expire_session(self, session_id: str) -> None:
        """Mark session as expired."""
        session = await self._load_session(session_id)
        if session:
            session.status = SessionStatus.EXPIRED
            await self._store_session(session)
            await self._remove_user_session(session.user_id, session_id)

    async def _flag_suspicious_session(self, session_id: str, reason: str) -> None:
        """Flag session as suspicious."""
        session = await self._load_session(session_id)
        if session:
            session.status = SessionStatus.SUSPICIOUS
            session.risk_score = min(100, session.risk_score + 20)
            session.metadata["suspicious_flags"] = session.metadata.get("suspicious_flags", [])
            session.metadata["suspicious_flags"].append(f"{reason}:{datetime.now(UTC).isoformat()}")

            await self._store_session(session)

            logger.warning("Session flagged as suspicious",
                         session_id=session_id,
                         reason=reason,
                         risk_score=session.risk_score)

    async def _track_activity(self, session_id: str, request: Request) -> None:
        """Track session activity for analytics."""
        if not self.redis:
            return

        activity_key = f"{self.activity_prefix}:{session_id}"
        activity_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "endpoint": request.url.path,
            "method": request.method,
            "ip": self._get_client_ip(request)
        }

        # Store last 10 activities
        await self.redis.lpush(activity_key, str(activity_data))
        await self.redis.ltrim(activity_key, 0, 9)
        await self.redis.expire(activity_key, 86400)  # 24 hours

    # Administrative and monitoring methods
    async def get_session_statistics(self) -> dict[str, Any]:
        """Get session statistics for monitoring."""
        if not self.redis:
            return {"error": "Redis not available"}

        # This is a simplified implementation
        # In practice, you'd implement more comprehensive statistics
        return {
            "total_active_sessions": 0,  # TODO: Implement
            "sessions_created_today": 0,  # TODO: Implement
            "suspicious_sessions": 0,     # TODO: Implement
            "average_session_duration": 0  # TODO: Implement
        }
