"""
Comprehensive test suite for Authentication Service.

Tests enterprise-grade authentication features including:
- JWT authentication with RS256 signatures
- Role-Based Access Control (RBAC)
- Multi-Factor Authentication (MFA)
- Session management with Redis
- Security audit logging
- Password policies and brute force protection
- OAuth2/OIDC integration
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import bcrypt
import jwt
import pyotp
import pytest
import redis.asyncio as redis
from cryptography.hazmat.primitives import serialization
from sqlalchemy.ext.asyncio import AsyncSession

from src.its_camera_ai.core.config import SecurityConfig
from src.its_camera_ai.core.exceptions import AuthenticationError
from src.its_camera_ai.models.user import Role, User
from src.its_camera_ai.services.auth_service import (
    AuthenticationService,
    AuthenticationStatus,
    BruteForceProtection,
    JWTManager,
    MFAMethod,
    PasswordPolicy,
    SecurityAuditLogger,
    SecurityEventType,
    SessionManager,
    UserCredentials,
    create_auth_service,
)


class TestPasswordPolicy:
    """Test password policy validation."""

    def test_valid_strong_password(self):
        """Test validation of a strong password."""
        result = PasswordPolicy.validate_password("MySecureP@ssw0rd2024!")
        assert result["valid"] is True
        assert result["strength"] == "strong"
        assert result["score"] >= 80
        assert len(result["errors"]) == 0

    def test_weak_password_too_short(self):
        """Test validation of password that's too short."""
        result = PasswordPolicy.validate_password("Short1!")
        assert result["valid"] is False
        assert "must be at least 12 characters" in " ".join(result["errors"])
        assert result["strength"] in ["weak", "very_weak"]

    def test_password_missing_requirements(self):
        """Test password missing character requirements."""
        result = PasswordPolicy.validate_password("alllowercase123")
        assert result["valid"] is False
        assert any("uppercase" in error for error in result["errors"])

        result = PasswordPolicy.validate_password("ALLUPPERCASE123")
        assert result["valid"] is False
        assert any("lowercase" in error for error in result["errors"])

        result = PasswordPolicy.validate_password("NoDigitsHere!")
        assert result["valid"] is False
        assert any("digits" in error for error in result["errors"])

        result = PasswordPolicy.validate_password("NoSpecialChars123")
        assert result["valid"] is False
        assert any("special" in error for error in result["errors"])

    def test_password_with_username(self):
        """Test password containing username."""
        result = PasswordPolicy.validate_password("MyUsername123!", "myusername")
        assert result["valid"] is False
        assert any("username" in error for error in result["errors"])

    def test_password_common_patterns(self):
        """Test password with common patterns."""
        result = PasswordPolicy.validate_password("Password123!")
        assert result["valid"] is False
        assert any("common patterns" in error for error in result["errors"])

    def test_password_repeated_characters(self):
        """Test password with too many repeated characters."""
        result = PasswordPolicy.validate_password("Mypassssssword123!")
        assert result["valid"] is False
        assert any("repeated characters" in error for error in result["errors"])


class TestJWTManager:
    """Test JWT token management."""

    def setup_method(self):
        """Setup test environment."""
        self.security_config = SecurityConfig()
        self.jwt_manager = JWTManager(self.security_config)

    def test_create_access_token(self):
        """Test JWT access token creation."""
        data = {"sub": "user123", "username": "testuser"}
        token = self.jwt_manager.create_access_token(data)

        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # JWT has 3 parts

        # Verify token can be decoded
        public_pem = self.jwt_manager.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        payload = jwt.decode(token, public_pem, algorithms=["RS256"])

        assert payload["sub"] == "user123"
        assert payload["username"] == "testuser"
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload

    def test_create_refresh_token(self):
        """Test JWT refresh token creation."""
        data = {"sub": "user123"}
        token = self.jwt_manager.create_refresh_token(data)

        # Verify token
        public_pem = self.jwt_manager.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        payload = jwt.decode(token, public_pem, algorithms=["RS256"])

        assert payload["sub"] == "user123"
        assert payload["type"] == "refresh"

    def test_verify_valid_token(self):
        """Test verification of valid token."""
        data = {"sub": "user123", "username": "testuser"}
        token = self.jwt_manager.create_access_token(data)

        payload = self.jwt_manager.verify_token(token)
        assert payload["sub"] == "user123"
        assert payload["username"] == "testuser"

    def test_verify_expired_token(self):
        """Test verification of expired token."""
        data = {"sub": "user123"}
        # Create token that expires immediately
        token = self.jwt_manager.create_access_token(data, timedelta(seconds=-1))

        with pytest.raises(AuthenticationError, match="Token has expired"):
            self.jwt_manager.verify_token(token)

    def test_verify_invalid_token(self):
        """Test verification of invalid token."""
        with pytest.raises(AuthenticationError, match="Invalid token"):
            self.jwt_manager.verify_token("invalid.token.here")


@pytest.mark.asyncio
class TestBruteForceProtection:
    """Test brute force protection."""

    async def setup_method(self):
        """Setup test environment."""
        # Mock Redis client
        self.redis_mock = AsyncMock(spec=redis.Redis)
        self.brute_force = BruteForceProtection(self.redis_mock)

    async def test_is_not_blocked_initially(self):
        """Test that user is not blocked initially."""
        self.redis_mock.exists.return_value = False

        is_blocked = await self.brute_force.is_blocked("test_user")
        assert is_blocked is False

    async def test_record_failure_under_limit(self):
        """Test recording failure under attempt limit."""
        self.redis_mock.incr.return_value = 2  # 2 attempts

        remaining = await self.brute_force.record_failure("test_user")
        assert remaining == 3  # 5 max - 2 attempts = 3 remaining

        self.redis_mock.incr.assert_called_once()
        self.redis_mock.expire.assert_called_once()

    async def test_record_failure_exceeds_limit(self):
        """Test recording failure that exceeds limit."""
        self.redis_mock.incr.return_value = 5  # Max attempts reached

        remaining = await self.brute_force.record_failure("test_user")
        assert remaining == 0

        # Should set lockout
        self.redis_mock.setex.assert_called_once()
        self.redis_mock.delete.assert_called_once()

    async def test_record_success_clears_attempts(self):
        """Test that successful auth clears failure attempts."""
        await self.brute_force.record_success("test_user")

        attempts_key = "auth:attempts:test_user"
        self.redis_mock.delete.assert_called_once_with(attempts_key)

    async def test_unlock_account(self):
        """Test manual account unlock."""
        await self.brute_force.unlock_account("test_user")

        lockout_key = "auth:lockout:test_user"
        attempts_key = "auth:attempts:test_user"
        self.redis_mock.delete.assert_called_once_with(lockout_key, attempts_key)


@pytest.mark.asyncio
class TestSessionManager:
    """Test Redis session management."""

    async def setup_method(self):
        """Setup test environment."""
        self.redis_mock = AsyncMock(spec=redis.Redis)
        self.session_manager = SessionManager(self.redis_mock)

    async def test_create_session(self):
        """Test session creation."""
        from src.its_camera_ai.services.auth_service import SessionInfo

        session_info = SessionInfo(
            session_id="test_session",
            user_id="user123",
            username="testuser",
            roles=["admin"],
            permissions=["users:read"],
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=8),
            mfa_verified=True,
        )

        self.redis_mock.scard.return_value = 2  # Current sessions

        result = await self.session_manager.create_session(session_info)

        assert result == "test_session"
        self.redis_mock.hset.assert_called_once()
        self.redis_mock.expire.assert_called()
        self.redis_mock.sadd.assert_called_once()

    async def test_get_existing_session(self):
        """Test getting existing session."""
        session_data = {
            "session_id": "test_session",
            "user_id": "user123",
            "username": "testuser",
            "roles": "admin,viewer",
            "permissions": "users:read,cameras:read",
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=8)).isoformat(),
            "mfa_verified": "True",
        }

        self.redis_mock.hgetall.return_value = session_data

        session = await self.session_manager.get_session("test_session")

        assert session is not None
        assert session.session_id == "test_session"
        assert session.user_id == "user123"
        assert session.roles == ["admin", "viewer"]
        assert session.permissions == ["users:read", "cameras:read"]
        assert session.mfa_verified is True

    async def test_get_nonexistent_session(self):
        """Test getting non-existent session."""
        self.redis_mock.hgetall.return_value = {}

        session = await self.session_manager.get_session("nonexistent")
        assert session is None

    async def test_delete_session(self):
        """Test session deletion."""
        session_data = {"user_id": "user123"}
        self.redis_mock.hgetall.return_value = session_data

        await self.session_manager.delete_session("test_session")

        self.redis_mock.srem.assert_called_once()
        self.redis_mock.delete.assert_called()

    async def test_delete_all_user_sessions(self):
        """Test deleting all sessions for a user."""
        self.redis_mock.smembers.return_value = ["session1", "session2", "session3"]

        await self.session_manager.delete_all_user_sessions("user123")

        # Should delete all session keys and user sessions set
        self.redis_mock.delete.assert_called()


@pytest.mark.asyncio
class TestSecurityAuditLogger:
    """Test security audit logging."""

    async def setup_method(self):
        """Setup test environment."""
        self.redis_mock = AsyncMock(spec=redis.Redis)
        self.audit_logger = SecurityAuditLogger(self.redis_mock)

    async def test_log_low_risk_event(self):
        """Test logging low-risk security event."""
        from src.its_camera_ai.services.auth_service import SecurityAuditEvent

        event = SecurityAuditEvent(
            event_id="test_event",
            event_type=SecurityEventType.LOGIN_SUCCESS,
            user_id="user123",
            username="testuser",
            success=True,
            timestamp=datetime.utcnow(),
            risk_score=10,
        )

        await self.audit_logger.log_event(event)

        self.redis_mock.hset.assert_called_once()
        self.redis_mock.expire.assert_called_once()
        self.redis_mock.xadd.assert_called_once()

    async def test_log_high_risk_event(self):
        """Test logging high-risk security event."""
        from src.its_camera_ai.services.auth_service import SecurityAuditEvent

        event = SecurityAuditEvent(
            event_id="test_event",
            event_type=SecurityEventType.BRUTE_FORCE_DETECTED,
            user_id="user123",
            username="testuser",
            success=False,
            timestamp=datetime.utcnow(),
            risk_score=90,
        )

        with patch.object(self.audit_logger, "_send_security_alert") as mock_alert:
            await self.audit_logger.log_event(event)
            mock_alert.assert_called_once_with(event)


@pytest.mark.asyncio
class TestAuthenticationService:
    """Test main authentication service."""

    async def setup_method(self):
        """Setup test environment."""
        # Mock database session
        self.session_mock = AsyncMock(spec=AsyncSession)

        # Mock Redis client
        self.redis_mock = AsyncMock(spec=redis.Redis)

        # Security configuration
        self.security_config = SecurityConfig()

        # Create service
        self.auth_service = AuthenticationService(
            self.session_mock, self.redis_mock, self.security_config
        )

        # Mock user data
        self.test_user = User(
            id="user123",
            username="testuser",
            email="test@example.com",
            hashed_password=bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode(),
            is_active=True,
            is_verified=True,
            mfa_enabled=False,
            roles=[],
        )

    async def test_authenticate_success_no_mfa(self):
        """Test successful authentication without MFA."""
        credentials = UserCredentials(
            username="testuser", password="password123", ip_address="192.168.1.100"
        )

        # Mock user lookup
        with (
            patch.object(
                self.auth_service, "_get_user_by_username", return_value=self.test_user
            ),
            patch.object(
                self.auth_service.brute_force_protection,
                "is_blocked",
                return_value=False,
            ),
            patch.object(
                self.auth_service.brute_force_protection, "record_success"
            ),
            patch.object(
                self.auth_service, "_create_user_session"
            ) as mock_session,
        ):
                        mock_session.return_value.session_id = "session123"

                        result = await self.auth_service.authenticate(credentials)

                        assert result.success is True
                        assert result.status == AuthenticationStatus.SUCCESS
                        assert result.user_id == "user123"
                        assert result.access_token is not None
                        assert result.refresh_token is not None
                        assert result.mfa_required is False

    async def test_authenticate_invalid_credentials(self):
        """Test authentication with invalid credentials."""
        credentials = UserCredentials(
            username="testuser", password="wrongpassword", ip_address="192.168.1.100"
        )

        with (
            patch.object(
                self.auth_service, "_get_user_by_username", return_value=self.test_user
            ),
            patch.object(
                self.auth_service.brute_force_protection,
                "is_blocked",
                return_value=False,
            ),
            patch.object(
                self.auth_service.brute_force_protection, "record_failure"
            ),
        ):
            result = await self.auth_service.authenticate(credentials)

            assert result.success is False
            assert result.status == AuthenticationStatus.FAILED
            assert result.error_message == "Invalid credentials"

    async def test_authenticate_user_not_found(self):
        """Test authentication with non-existent user."""
        credentials = UserCredentials(
            username="nonexistent", password="password123", ip_address="192.168.1.100"
        )

        with (
            patch.object(
                self.auth_service, "_get_user_by_username", return_value=None
            ),
            patch.object(
                self.auth_service.brute_force_protection,
                "is_blocked",
                return_value=False,
            ),
            patch.object(
                self.auth_service.brute_force_protection, "record_failure"
            ),
        ):
            result = await self.auth_service.authenticate(credentials)

            assert result.success is False
            assert result.status == AuthenticationStatus.FAILED

    async def test_authenticate_account_blocked(self):
        """Test authentication with blocked account."""
        credentials = UserCredentials(
            username="testuser", password="password123", ip_address="192.168.1.100"
        )

        with patch.object(
            self.auth_service.brute_force_protection, "is_blocked", return_value=True
        ):
            result = await self.auth_service.authenticate(credentials)

            assert result.success is False
            assert result.status == AuthenticationStatus.BLOCKED
            assert "temporarily locked" in result.error_message

    async def test_authenticate_mfa_required(self):
        """Test authentication requiring MFA."""
        # User with MFA enabled
        mfa_user = User(
            id="user123",
            username="testuser",
            email="test@example.com",
            hashed_password=bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode(),
            is_active=True,
            mfa_enabled=True,
            mfa_secret="JBSWY3DPEHPK3PXP",
            roles=[],
        )

        credentials = UserCredentials(
            username="testuser", password="password123", ip_address="192.168.1.100"
        )

        with (
            patch.object(
                self.auth_service, "_get_user_by_username", return_value=mfa_user
            ),
            patch.object(
                self.auth_service.brute_force_protection,
                "is_blocked",
                return_value=False,
            ),
        ):
            result = await self.auth_service.authenticate(credentials)

            assert result.success is False
            assert result.status == AuthenticationStatus.MFA_REQUIRED
            assert result.mfa_required is True
            assert MFAMethod.TOTP in result.mfa_methods

    async def test_verify_valid_token(self):
        """Test JWT token verification."""
        # Create valid token
        token_data = {
            "sub": "user123",
            "username": "testuser",
            "session_id": "session123",
            "roles": ["viewer"],
            "permissions": ["cameras:read"],
        }
        token = self.auth_service.jwt_manager.create_access_token(token_data)

        # Mock session
        from src.its_camera_ai.services.auth_service import SessionInfo

        mock_session = SessionInfo(
            session_id="session123",
            user_id="user123",
            username="testuser",
            roles=["viewer"],
            permissions=["cameras:read"],
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=8),
            mfa_verified=False,
        )

        with (
            patch.object(
                self.auth_service.session_manager, "get_session", return_value=mock_session
            ),
            patch.object(
                self.auth_service.session_manager, "update_session_activity"
            ),
        ):
            result = await self.auth_service.verify_token(token)

            assert result.valid is True
            assert result.user_id == "user123"
            assert result.session_id == "session123"
            assert result.roles == ["viewer"]
            assert result.permissions == ["cameras:read"]

    async def test_verify_expired_session(self):
        """Test token verification with expired session."""
        token_data = {
            "sub": "user123",
            "session_id": "session123",
            "roles": ["viewer"],
            "permissions": ["cameras:read"],
        }
        token = self.auth_service.jwt_manager.create_access_token(token_data)

        # Mock expired session
        from src.its_camera_ai.services.auth_service import SessionInfo

        expired_session = SessionInfo(
            session_id="session123",
            user_id="user123",
            username="testuser",
            roles=["viewer"],
            permissions=["cameras:read"],
            created_at=datetime.utcnow() - timedelta(hours=10),
            last_activity=datetime.utcnow() - timedelta(hours=2),
            expires_at=datetime.utcnow() - timedelta(hours=1),  # Expired
            mfa_verified=False,
        )

        with patch.object(
            self.auth_service.session_manager,
            "get_session",
            return_value=expired_session,
        ):
            result = await self.auth_service.verify_token(token)

            assert result.valid is False
            assert result.error_message == "Session expired"

    async def test_refresh_token_success(self):
        """Test successful token refresh."""
        # Create refresh token
        refresh_data = {"sub": "user123", "session_id": "session123"}
        refresh_token = self.auth_service.jwt_manager.create_refresh_token(refresh_data)

        # Mock valid session
        from src.its_camera_ai.services.auth_service import SessionInfo

        mock_session = SessionInfo(
            session_id="session123",
            user_id="user123",
            username="testuser",
            roles=["viewer"],
            permissions=["cameras:read"],
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=8),
            mfa_verified=False,
        )

        with (
            patch.object(
                self.auth_service.session_manager, "get_session", return_value=mock_session
            ),
            patch.object(
                self.auth_service.user_service, "get_by_id", return_value=self.test_user
            ),
            patch.object(
                self.auth_service,
                "_get_user_permissions",
                return_value=["cameras:read"],
            ),
        ):
            result = await self.auth_service.refresh_token(refresh_token)

            assert result.access_token is not None
            assert result.refresh_token is not None
            assert result.expires_in > 0

    async def test_enroll_totp_mfa(self):
        """Test TOTP MFA enrollment."""
        with (
            patch.object(
                self.auth_service.user_service, "get_by_id", return_value=self.test_user
            ),
            patch.object(self.auth_service.user_service, "update_by_id"),
        ):
            result = await self.auth_service.enroll_mfa("user123", MFAMethod.TOTP)

            assert result.success is True
            assert result.method == MFAMethod.TOTP
            assert result.secret is not None
            assert result.qr_code_url is not None
            assert len(result.backup_codes) == 8

    async def test_verify_totp_mfa(self):
        """Test TOTP MFA verification."""
        # Generate TOTP code
        secret = "JBSWY3DPEHPK3PXP"
        totp = pyotp.TOTP(secret)
        valid_code = totp.now()

        mfa_user = User(
            id="user123",
            username="testuser",
            email="test@example.com",
            hashed_password="hash",
            mfa_enabled=True,
            mfa_secret=secret,
            roles=[],
        )

        with patch.object(
            self.auth_service.user_service, "get_by_id", return_value=mfa_user
        ):
            result = await self.auth_service.verify_mfa(
                "user123", valid_code, MFAMethod.TOTP
            )

            assert result.success is True

    async def test_check_permission_success(self):
        """Test successful permission check."""
        role = Role(id="role1", name="viewer", description="Viewer role")
        user_with_roles = User(
            id="user123",
            username="testuser",
            email="test@example.com",
            hashed_password="hash",
            is_active=True,
            roles=[role],
        )

        with (
            patch.object(
                self.auth_service, "_get_user_with_roles", return_value=user_with_roles
            ),
            patch.object(
                self.auth_service,
                "_get_user_permissions",
                return_value=["cameras:read"],
            ),
        ):
            result = await self.auth_service.check_permission(
                "user123", "cameras", "read"
            )

            assert result is True

    async def test_check_permission_denied(self):
        """Test permission denied."""
        role = Role(id="role1", name="viewer", description="Viewer role")
        user_with_roles = User(
            id="user123",
            username="testuser",
            email="test@example.com",
            hashed_password="hash",
            is_active=True,
            roles=[role],
        )

        with (
            patch.object(
                self.auth_service, "_get_user_with_roles", return_value=user_with_roles
            ),
            patch.object(
                self.auth_service,
                "_get_user_permissions",
                return_value=["cameras:read"],
            ),
        ):
            result = await self.auth_service.check_permission(
                "user123", "users", "delete"
            )

            assert result is False

    async def test_superuser_has_all_permissions(self):
        """Test that superuser has all permissions."""
        superuser = User(
            id="admin123",
            username="admin",
            email="admin@example.com",
            hashed_password="hash",
            is_active=True,
            is_superuser=True,
            roles=[],
        )

        with patch.object(
            self.auth_service, "_get_user_with_roles", return_value=superuser
        ):
            result = await self.auth_service.check_permission(
                "admin123", "system", "delete"
            )

            assert result is True

    async def test_logout_success(self):
        """Test successful logout."""
        from src.its_camera_ai.services.auth_service import SessionInfo

        mock_session = SessionInfo(
            session_id="session123",
            user_id="user123",
            username="testuser",
            roles=["viewer"],
            permissions=["cameras:read"],
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=8),
            mfa_verified=False,
        )

        with (
            patch.object(
                self.auth_service.session_manager, "get_session", return_value=mock_session
            ),
            patch.object(self.auth_service.session_manager, "delete_session"),
        ):
            result = await self.auth_service.logout("session123")

            assert result is True

    async def test_change_password_success(self):
        """Test successful password change."""
        with (
            patch.object(
                self.auth_service.user_service, "get_by_id", return_value=self.test_user
            ),
            patch.object(self.auth_service.user_service, "update_by_id"),
            patch.object(
                self.auth_service.session_manager, "delete_all_user_sessions"
            ),
        ):
            result = await self.auth_service.change_password(
                "user123", "password123", "NewSecureP@ssw0rd2024!"
            )

            assert result is True

    async def test_change_password_wrong_old_password(self):
        """Test password change with wrong old password."""
        with patch.object(
            self.auth_service.user_service, "get_by_id", return_value=self.test_user
        ):
            result = await self.auth_service.change_password(
                "user123", "wrongpassword", "NewSecureP@ssw0rd2024!"
            )

            assert result is False

    async def test_change_password_weak_new_password(self):
        """Test password change with weak new password."""
        with (
            patch.object(
                self.auth_service.user_service, "get_by_id", return_value=self.test_user
            ),
            pytest.raises(ValueError, match="Password validation failed"),
        ):
            await self.auth_service.change_password(
                "user123", "password123", "weak"
            )


@pytest.mark.asyncio
class TestCreateAuthService:
    """Test authentication service factory function."""

    async def test_create_auth_service(self):
        """Test factory function creates service correctly."""
        session_mock = AsyncMock(spec=AsyncSession)

        with patch("redis.asyncio.from_url") as mock_redis:
            mock_redis_client = AsyncMock(spec=redis.Redis)
            mock_redis.return_value = mock_redis_client

            service = await create_auth_service(
                session_mock, "redis://localhost:6379/0"
            )

            assert isinstance(service, AuthenticationService)
            assert service.session == session_mock
            mock_redis.assert_called_once_with("redis://localhost:6379/0")


@pytest.mark.integration
class TestAuthenticationServiceIntegration:
    """Integration tests for authentication service with real components."""

    async def test_full_authentication_flow(self):
        """Test complete authentication flow."""
        # This would require actual database and Redis connections
        # Implementation would depend on test infrastructure setup
        pass

    async def test_concurrent_authentication_requests(self):
        """Test handling of concurrent authentication requests."""
        # Test race conditions and concurrent access
        pass

    async def test_session_cleanup_on_expiry(self):
        """Test automatic cleanup of expired sessions."""
        # Test session expiration handling
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
