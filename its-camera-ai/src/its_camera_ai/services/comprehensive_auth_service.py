"""Comprehensive authentication service that integrates all auth components."""

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..core.config import SecurityConfig
from ..core.logging import get_logger
from ..models.user import SecurityAuditLog, User
from ..services.auth import AuthService
from ..services.cache import CacheService
from ..services.email_service import EmailService
from ..services.mfa_service import MFAService
from ..services.token_service import TokenService

logger = get_logger(__name__)


class ComprehensiveAuthService:
    """Comprehensive authentication service with all features integrated."""

    def __init__(
        self,
        db: AsyncSession,
        cache_service: CacheService,
        security_config: SecurityConfig,
        email_service: EmailService
    ):
        self.db = db
        self.cache = cache_service
        self.security_config = security_config
        self.email_service = email_service

        # Initialize component services
        self.auth_service = AuthService(security_config)
        self.token_service = TokenService(cache_service, security_config)
        self.mfa_service = MFAService(cache_service, security_config)

    async def authenticate_user(
        self,
        username: str,
        password: str,
        mfa_code: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None
    ) -> dict[str, Any]:
        """Authenticate user with comprehensive security checks.

        Args:
            username: Username or email
            password: Password
            mfa_code: MFA code if required
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Authentication result with tokens or error information
        """
        auth_result = {
            "success": False,
            "user": None,
            "tokens": None,
            "mfa_required": False,
            "error": None,
            "locked_until": None
        }

        try:
            # Get user by username or email
            user = await self.auth_service.get_user_by_username(self.db, username)
            if not user:
                # Try email
                user_by_email = await self.db.execute(
                    select(User)
                    .options(selectinload(User.roles))
                    .where(User.email == username)
                )
                user = user_by_email.scalar_one_or_none()

            if not user:
                await self._log_failed_authentication(
                    username, "user_not_found", ip_address, user_agent
                )
                auth_result["error"] = "Invalid credentials"
                return auth_result

            # Check if account is locked
            is_locked, unlock_time = await self.auth_service.check_account_lockout(
                self.db, self.cache, user, ip_address
            )

            if is_locked:
                auth_result["error"] = "Account temporarily locked"
                auth_result["locked_until"] = unlock_time

                # Send lockout notification email
                if unlock_time:
                    await self.email_service.send_account_lockout_email(
                        to_email=user.email,
                        username=user.username,
                        lockout_time=datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC'),
                        unlock_time=unlock_time.strftime('%Y-%m-%d %H:%M:%S UTC'),
                        attempt_count=user.failed_login_attempts,
                        lockout_duration=self.security_config.lockout_duration_minutes,
                        full_name=user.full_name
                    )

                return auth_result

            # Check if user is active
            if not user.is_active:
                await self._log_failed_authentication(
                    username, "account_inactive", ip_address, user_agent, user.id
                )
                auth_result["error"] = "Account is inactive"
                return auth_result

            # Verify password
            from ..api.routers.auth import verify_password
            if not verify_password(password, user.hashed_password):
                await self.auth_service.record_failed_login(
                    self.db, self.cache, user, ip_address
                )
                await self._log_failed_authentication(
                    username, "invalid_password", ip_address, user_agent, user.id
                )
                auth_result["error"] = "Invalid credentials"
                return auth_result

            # Check MFA requirement
            if user.mfa_enabled:
                if not mfa_code:
                    auth_result["mfa_required"] = True
                    auth_result["error"] = "MFA code required"
                    return auth_result

                # Verify MFA code
                mfa_valid = await self.mfa_service.verify_totp(user, mfa_code)
                if not mfa_valid:
                    # Try backup code
                    mfa_valid = await self.mfa_service.verify_backup_code(
                        self.db, user, mfa_code
                    )

                if not mfa_valid:
                    await self._log_failed_authentication(
                        username, "invalid_mfa", ip_address, user_agent, user.id
                    )
                    auth_result["error"] = "Invalid MFA code"
                    return auth_result

            # Authentication successful
            await self.auth_service.clear_failed_login_attempts(
                self.db, self.cache, user
            )

            # Update user login information
            user.last_login = datetime.now(UTC)
            user.last_login_ip = ip_address
            user.last_login_device = user_agent
            await self.db.commit()

            # Create session and tokens
            import uuid
            session_id = str(uuid.uuid4())

            # Get user permissions
            permissions = await self.auth_service.get_user_permissions(self.db, user)

            # Create tokens
            access_token, access_exp = await self.token_service.create_access_token(
                user_id=user.id,
                session_id=session_id,
                permissions=permissions
            )

            refresh_token, refresh_exp = await self.token_service.create_refresh_token(
                user_id=user.id,
                session_id=session_id
            )

            # Store session information
            session_data = {
                "user_id": user.id,
                "username": user.username,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "created_at": datetime.now(UTC).isoformat(),
                "last_activity": datetime.now(UTC).isoformat(),
                "mfa_verified": user.mfa_enabled
            }

            session_key = f"session:{session_id}"
            session_ttl = int((refresh_exp - datetime.now(UTC)).total_seconds())
            await self.cache.set_json(session_key, session_data, session_ttl)

            # Log successful authentication
            await self.auth_service.create_security_audit_log(
                self.db,
                event_type="user_login",
                user_id=user.id,
                username=user.username,
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session_id,
                success=True,
                details={
                    "mfa_used": user.mfa_enabled,
                    "login_method": "password" + ("+mfa" if user.mfa_enabled else "")
                }
            )

            auth_result.update({
                "success": True,
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "roles": [role.name for role in user.roles],
                    "permissions": permissions,
                    "is_verified": user.is_verified,
                    "mfa_enabled": user.mfa_enabled
                },
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "bearer",
                    "expires_in": int((access_exp - datetime.now(UTC)).total_seconds())
                },
                "session_id": session_id
            })

            logger.info(
                "User authenticated successfully",
                user_id=user.id,
                username=user.username,
                ip_address=ip_address,
                mfa_used=user.mfa_enabled
            )

            return auth_result

        except Exception as e:
            logger.error(
                "Authentication error",
                username=username,
                error=str(e),
                ip_address=ip_address
            )
            auth_result["error"] = "Authentication failed"
            return auth_result

    async def logout_user(
        self,
        user_id: str,
        session_id: str | None = None,
        token: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None
    ) -> bool:
        """Logout user and invalidate session/tokens.

        Args:
            user_id: User ID
            session_id: Session ID to invalidate
            token: Token to blacklist
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            True if logout successful
        """
        try:
            # Blacklist token if provided
            if token:
                await self.token_service.blacklist_token(token, "user_logout")

            # Invalidate session if provided
            if session_id:
                await self.token_service.invalidate_session(session_id)

            # Log logout
            user = await self.auth_service.get_user_by_id(self.db, user_id)
            if user:
                await self.auth_service.create_security_audit_log(
                    self.db,
                    event_type="user_logout",
                    user_id=user.id,
                    username=user.username,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    session_id=session_id,
                    success=True,
                    details={"logout_type": "user_initiated"}
                )

            logger.info(
                "User logged out successfully",
                user_id=user_id,
                session_id=session_id
            )

            return True

        except Exception as e:
            logger.error(
                "Logout error",
                user_id=user_id,
                session_id=session_id,
                error=str(e)
            )
            return False

    async def change_password(
        self,
        user_id: str,
        current_password: str,
        new_password: str,
        ip_address: str | None = None,
        user_agent: str | None = None
    ) -> dict[str, Any]:
        """Change user password with security checks.

        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Change result
        """
        try:
            user = await self.auth_service.get_user_by_id(self.db, user_id)
            if not user:
                return {"success": False, "error": "User not found"}

            # Verify current password
            from ..api.routers.auth import get_password_hash, verify_password
            if not verify_password(current_password, user.hashed_password):
                await self.auth_service.create_security_audit_log(
                    self.db,
                    event_type="password_change_failed",
                    user_id=user.id,
                    username=user.username,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=False,
                    error_message="Invalid current password",
                    risk_score=60
                )
                return {"success": False, "error": "Current password is incorrect"}

            # Check password history to prevent reuse
            if user.password_history:
                try:
                    password_hashes = json.loads(user.password_history)
                    for old_hash in password_hashes:
                        if verify_password(new_password, old_hash):
                            return {
                                "success": False,
                                "error": "Cannot reuse recent passwords"
                            }
                except (json.JSONDecodeError, TypeError):
                    password_hashes = []
            else:
                password_hashes = []

            # Update password
            new_hash = get_password_hash(new_password)
            user.hashed_password = new_hash
            user.last_password_change = datetime.now(UTC)

            # Update password history
            password_hashes.append(user.hashed_password)
            # Keep only recent passwords
            max_history = self.security_config.password_history_size
            if len(password_hashes) > max_history:
                password_hashes = password_hashes[-max_history:]
            user.password_history = json.dumps(password_hashes)

            await self.db.commit()

            # Invalidate all other sessions (security measure)
            await self.token_service.invalidate_all_user_sessions(user.id)

            # Send notification email
            await self.email_service.send_password_changed_email(
                to_email=user.email,
                username=user.username,
                change_time=datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC'),
                ip_address=ip_address,
                device=user_agent,
                full_name=user.full_name
            )

            # Log password change
            await self.auth_service.create_security_audit_log(
                self.db,
                event_type="password_changed",
                user_id=user.id,
                username=user.username,
                ip_address=ip_address,
                user_agent=user_agent,
                success=True,
                details={"change_type": "user_initiated"}
            )

            logger.info(
                "Password changed successfully",
                user_id=user.id,
                username=user.username,
                ip_address=ip_address
            )

            return {"success": True, "message": "Password changed successfully"}

        except Exception as e:
            logger.error(
                "Password change error",
                user_id=user_id,
                error=str(e)
            )
            return {"success": False, "error": "Password change failed"}

    async def _log_failed_authentication(
        self,
        username: str,
        reason: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
        user_id: str | None = None
    ) -> None:
        """Log failed authentication attempt."""
        risk_scores = {
            "user_not_found": 30,
            "account_inactive": 50,
            "invalid_password": 60,
            "invalid_mfa": 70,
            "account_locked": 40
        }

        await self.auth_service.create_security_audit_log(
            self.db,
            event_type="failed_login",
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            success=False,
            error_message=reason,
            risk_score=risk_scores.get(reason, 50)
        )

    async def get_security_summary(self, user_id: str) -> dict[str, Any]:
        """Get security summary for user.

        Args:
            user_id: User ID

        Returns:
            Security summary
        """
        try:
            user = await self.auth_service.get_user_by_id(self.db, user_id)
            if not user:
                return {"error": "User not found"}

            # Get MFA status
            mfa_status = await self.mfa_service.get_mfa_status(user)

            # Get active sessions
            sessions = await self.token_service.get_user_sessions(user_id)

            # Get recent security events
            recent_events_query = await self.db.execute(
                select(SecurityAuditLog)
                .where(SecurityAuditLog.user_id == user_id)
                .order_by(SecurityAuditLog.timestamp.desc())
                .limit(10)
            )
            recent_events = recent_events_query.scalars().all()

            return {
                "user_id": user_id,
                "username": user.username,
                "account_status": {
                    "is_active": user.is_active,
                    "is_verified": user.is_verified,
                    "is_locked": bool(user.account_locked_until and user.account_locked_until > datetime.now(UTC)),
                    "failed_attempts": user.failed_login_attempts,
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "last_password_change": user.last_password_change.isoformat() if user.last_password_change else None
                },
                "mfa_status": mfa_status,
                "active_sessions": {
                    "count": len(sessions),
                    "sessions": sessions
                },
                "recent_events": [
                    {
                        "event_type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "success": event.success,
                        "ip_address": event.ip_address,
                        "risk_score": event.risk_score
                    }
                    for event in recent_events
                ]
            }

        except Exception as e:
            logger.error(
                "Failed to get security summary",
                user_id=user_id,
                error=str(e)
            )
            return {"error": "Failed to retrieve security summary"}
