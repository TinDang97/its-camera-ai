"""JWT token management and blacklisting service."""

from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt

from ..core.config import SecurityConfig
from ..core.logging import get_logger
from ..services.cache import CacheService

logger = get_logger(__name__)


class TokenService:
    """Service for JWT token management and blacklisting."""

    def __init__(self, cache_service: CacheService, security_config: SecurityConfig):
        self.cache = cache_service
        self.security_config = security_config

    def _get_blacklist_key(self, jti: str) -> str:
        """Get Redis key for blacklisted token."""
        return f"blacklisted_token:{jti}"

    def _get_user_sessions_key(self, user_id: str) -> str:
        """Get Redis key for user active sessions."""
        return f"user_sessions:{user_id}"

    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session data."""
        return f"session:{session_id}"

    async def create_access_token(
        self,
        user_id: str,
        session_id: str | None = None,
        permissions: list[str] | None = None,
        expires_delta: timedelta | None = None
    ) -> tuple[str, datetime]:
        """Create JWT access token.

        Args:
            user_id: User ID
            session_id: Session ID for tracking
            permissions: User permissions
            expires_delta: Custom expiration time

        Returns:
            Tuple of (token, expiration_time)
        """
        now = datetime.now(UTC)
        expire = now + (expires_delta or timedelta(
            minutes=self.security_config.access_token_expire_minutes
        ))

        import uuid
        jti = str(uuid.uuid4())  # JWT ID for blacklisting

        to_encode = {
            "sub": user_id,
            "jti": jti,
            "type": "access",
            "iat": now,
            "exp": expire,
            "permissions": permissions or []
        }

        if session_id:
            to_encode["session_id"] = session_id

        encoded_jwt = jwt.encode(
            to_encode,
            self.security_config.secret_key,
            algorithm=self.security_config.algorithm
        )

        # Store token metadata for session tracking
        if session_id:
            token_data = {
                "jti": jti,
                "user_id": user_id,
                "created_at": now.isoformat(),
                "expires_at": expire.isoformat(),
                "type": "access"
            }

            # Store token in session
            session_key = self._get_session_key(session_id)
            session_data = await self.cache.get_json(session_key) or {}
            session_data["access_token"] = token_data

            # Calculate TTL based on expiration
            ttl = int((expire - now).total_seconds())
            await self.cache.set_json(session_key, session_data, ttl)

        return encoded_jwt, expire

    async def create_refresh_token(
        self,
        user_id: str,
        session_id: str,
        expires_delta: timedelta | None = None
    ) -> tuple[str, datetime]:
        """Create JWT refresh token.

        Args:
            user_id: User ID
            session_id: Session ID
            expires_delta: Custom expiration time

        Returns:
            Tuple of (token, expiration_time)
        """
        now = datetime.now(UTC)
        expire = now + (expires_delta or timedelta(
            days=self.security_config.refresh_token_expire_days
        ))

        import uuid
        jti = str(uuid.uuid4())

        to_encode = {
            "sub": user_id,
            "jti": jti,
            "type": "refresh",
            "session_id": session_id,
            "iat": now,
            "exp": expire
        }

        encoded_jwt = jwt.encode(
            to_encode,
            self.security_config.secret_key,
            algorithm=self.security_config.algorithm
        )

        # Store token metadata
        token_data = {
            "jti": jti,
            "user_id": user_id,
            "session_id": session_id,
            "created_at": now.isoformat(),
            "expires_at": expire.isoformat(),
            "type": "refresh"
        }

        # Store in session and user sessions list
        session_key = self._get_session_key(session_id)
        session_data = await self.cache.get_json(session_key) or {}
        session_data["refresh_token"] = token_data

        ttl = int((expire - now).total_seconds())
        await self.cache.set_json(session_key, session_data, ttl)

        # Add to user's active sessions
        user_sessions_key = self._get_user_sessions_key(user_id)
        user_sessions = await self.cache.get_json(user_sessions_key) or []

        # Remove old sessions if exceeding max limit
        if len(user_sessions) >= self.security_config.max_sessions_per_user:
            # Sort by created time and remove oldest
            user_sessions.sort(key=lambda x: x.get('created_at', ''))
            old_sessions = user_sessions[:-self.security_config.max_sessions_per_user + 1]

            for old_session in old_sessions:
                await self.invalidate_session(old_session['session_id'])

            # Keep only recent sessions
            user_sessions = user_sessions[-self.security_config.max_sessions_per_user + 1:]

        # Add current session
        session_info = {
            "session_id": session_id,
            "created_at": now.isoformat(),
            "expires_at": expire.isoformat()
        }
        user_sessions.append(session_info)

        # Store with longest expiration time
        await self.cache.set_json(user_sessions_key, user_sessions, ttl)

        return encoded_jwt, expire

    async def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify JWT token and check if blacklisted.

        Args:
            token: JWT token to verify

        Returns:
            Token payload if valid, None otherwise
        """
        try:
            # Decode without verification first to get JTI
            unverified_payload = jwt.get_unverified_claims(token)
            jti = unverified_payload.get('jti')

            if jti:
                # Check if token is blacklisted
                blacklist_key = self._get_blacklist_key(jti)
                if await self.cache.get(blacklist_key):
                    logger.warning("Blacklisted token used", jti=jti)
                    return None

            # Verify token signature and expiration
            payload = jwt.decode(
                token,
                self.security_config.secret_key,
                algorithms=[self.security_config.algorithm]
            )

            # Additional validation
            token_type = payload.get('type')
            if token_type not in ['access', 'refresh']:
                return None

            return payload

        except JWTError as e:
            logger.debug("Token verification failed", error=str(e))
            return None

    async def blacklist_token(self, token: str, reason: str = "user_logout") -> bool:
        """Add token to blacklist.

        Args:
            token: JWT token to blacklist
            reason: Reason for blacklisting

        Returns:
            True if successfully blacklisted
        """
        try:
            # Get token payload
            payload = jwt.get_unverified_claims(token)
            jti = payload.get('jti')
            exp = payload.get('exp')

            if not jti or not exp:
                return False

            # Calculate TTL based on token expiration
            exp_datetime = datetime.fromtimestamp(exp)
            now = datetime.now(UTC)

            if exp_datetime <= now:
                # Token already expired, no need to blacklist
                return True

            ttl = int((exp_datetime - now).total_seconds())

            # Add to blacklist with metadata
            blacklist_key = self._get_blacklist_key(jti)
            blacklist_data = {
                "jti": jti,
                "reason": reason,
                "blacklisted_at": now.isoformat(),
                "expires_at": exp_datetime.isoformat(),
                "user_id": payload.get('sub'),
                "session_id": payload.get('session_id')
            }

            success = await self.cache.set_json(blacklist_key, blacklist_data, ttl)

            if success:
                logger.info(
                    "Token blacklisted",
                    jti=jti,
                    reason=reason,
                    user_id=payload.get('sub')
                )

            return success

        except Exception as e:
            logger.error("Failed to blacklist token", error=str(e))
            return False

    async def invalidate_session(self, session_id: str) -> bool:
        """Invalidate entire session (all tokens).

        Args:
            session_id: Session ID to invalidate

        Returns:
            True if successfully invalidated
        """
        try:
            session_key = self._get_session_key(session_id)
            session_data = await self.cache.get_json(session_key)

            if not session_data:
                return True  # Already invalidated

            # Blacklist all tokens in session
            success = True
            for token_type in ['access_token', 'refresh_token']:
                token_data = session_data.get(token_type)
                if token_data and token_data.get('jti'):
                    jti = token_data['jti']

                    # Create fake token payload for blacklisting
                    {
                        'jti': jti,
                        'exp': datetime.fromisoformat(token_data['expires_at']).timestamp(),
                        'sub': token_data.get('user_id'),
                        'session_id': session_id
                    }

                    # Blacklist token by JTI
                    blacklist_key = self._get_blacklist_key(jti)
                    now = datetime.now(UTC)
                    exp_datetime = datetime.fromisoformat(token_data['expires_at'])

                    if exp_datetime > now:
                        ttl = int((exp_datetime - now).total_seconds())
                        blacklist_data = {
                            "jti": jti,
                            "reason": "session_invalidated",
                            "blacklisted_at": now.isoformat(),
                            "expires_at": token_data['expires_at'],
                            "user_id": token_data.get('user_id'),
                            "session_id": session_id
                        }

                        if not await self.cache.set_json(blacklist_key, blacklist_data, ttl):
                            success = False

            # Remove session data
            await self.cache.delete(session_key)

            # Remove from user sessions list
            user_id = session_data.get('refresh_token', {}).get('user_id')
            if user_id:
                user_sessions_key = self._get_user_sessions_key(user_id)
                user_sessions = await self.cache.get_json(user_sessions_key) or []
                user_sessions = [s for s in user_sessions if s.get('session_id') != session_id]

                if user_sessions:
                    # Calculate new TTL based on latest session
                    max_exp = max(
                        datetime.fromisoformat(s['expires_at'])
                        for s in user_sessions
                    )
                    ttl = int((max_exp - datetime.now(UTC)).total_seconds())
                    if ttl > 0:
                        await self.cache.set_json(user_sessions_key, user_sessions, ttl)
                else:
                    await self.cache.delete(user_sessions_key)

            logger.info("Session invalidated", session_id=session_id, user_id=user_id)
            return success

        except Exception as e:
            logger.error("Failed to invalidate session", session_id=session_id, error=str(e))
            return False

    async def invalidate_all_user_sessions(self, user_id: str, except_session: str | None = None) -> bool:
        """Invalidate all sessions for a user.

        Args:
            user_id: User ID
            except_session: Session ID to exclude from invalidation

        Returns:
            True if successfully invalidated
        """
        try:
            user_sessions_key = self._get_user_sessions_key(user_id)
            user_sessions = await self.cache.get_json(user_sessions_key) or []

            success = True
            for session_info in user_sessions:
                session_id = session_info.get('session_id')
                if session_id and session_id != except_session:
                    if not await self.invalidate_session(session_id):
                        success = False

            # Update user sessions list to keep only the excepted session
            if except_session:
                user_sessions = [s for s in user_sessions if s.get('session_id') == except_session]
                if user_sessions:
                    session_info = user_sessions[0]
                    exp_datetime = datetime.fromisoformat(session_info['expires_at'])
                    ttl = int((exp_datetime - datetime.now(UTC)).total_seconds())
                    if ttl > 0:
                        await self.cache.set_json(user_sessions_key, user_sessions, ttl)
                    else:
                        await self.cache.delete(user_sessions_key)
                else:
                    await self.cache.delete(user_sessions_key)
            else:
                await self.cache.delete(user_sessions_key)

            logger.info(
                "All user sessions invalidated",
                user_id=user_id,
                except_session=except_session,
                session_count=len(user_sessions)
            )
            return success

        except Exception as e:
            logger.error(
                "Failed to invalidate user sessions",
                user_id=user_id,
                error=str(e)
            )
            return False

    async def get_user_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """Get active sessions for a user.

        Args:
            user_id: User ID

        Returns:
            List of active session information
        """
        try:
            user_sessions_key = self._get_user_sessions_key(user_id)
            user_sessions = await self.cache.get_json(user_sessions_key) or []

            # Filter out expired sessions
            now = datetime.now(UTC)
            active_sessions = []

            for session_info in user_sessions:
                try:
                    exp_datetime = datetime.fromisoformat(session_info['expires_at'])
                    if exp_datetime > now:
                        # Get additional session details
                        session_key = self._get_session_key(session_info['session_id'])
                        session_data = await self.cache.get_json(session_key)

                        if session_data:
                            session_info['last_activity'] = session_data.get('last_activity')
                            session_info['ip_address'] = session_data.get('ip_address')
                            session_info['user_agent'] = session_data.get('user_agent')

                        active_sessions.append(session_info)
                except ValueError:
                    # Invalid date format, skip
                    continue

            # Update stored sessions list if needed
            if len(active_sessions) != len(user_sessions):
                if active_sessions:
                    max_exp = max(
                        datetime.fromisoformat(s['expires_at'])
                        for s in active_sessions
                    )
                    ttl = int((max_exp - now).total_seconds())
                    if ttl > 0:
                        await self.cache.set_json(user_sessions_key, active_sessions, ttl)
                else:
                    await self.cache.delete(user_sessions_key)

            return active_sessions

        except Exception as e:
            logger.error("Failed to get user sessions", user_id=user_id, error=str(e))
            return []

    async def cleanup_expired_blacklist_entries(self) -> int:
        """Clean up expired blacklist entries.

        Returns:
            Number of entries cleaned up
        """
        # This would typically be done by Redis TTL automatically,
        # but we can implement manual cleanup if needed
        # For now, rely on Redis TTL
        return 0

    async def get_token_info(self, token: str) -> dict[str, Any] | None:
        """Get information about a token without verifying it.

        Args:
            token: JWT token

        Returns:
            Token information if valid format
        """
        try:
            payload = jwt.get_unverified_claims(token)

            # Check if blacklisted
            jti = payload.get('jti')
            is_blacklisted = False
            if jti:
                blacklist_key = self._get_blacklist_key(jti)
                is_blacklisted = bool(await self.cache.get(blacklist_key))

            return {
                'jti': jti,
                'user_id': payload.get('sub'),
                'session_id': payload.get('session_id'),
                'type': payload.get('type'),
                'issued_at': datetime.fromtimestamp(payload['iat']) if payload.get('iat') else None,
                'expires_at': datetime.fromtimestamp(payload['exp']) if payload.get('exp') else None,
                'permissions': payload.get('permissions', []),
                'is_blacklisted': is_blacklisted,
                'is_expired': datetime.fromtimestamp(payload['exp']) <= datetime.now(UTC) if payload.get('exp') else True
            }

        except Exception as e:
            logger.debug("Failed to get token info", error=str(e))
            return None
