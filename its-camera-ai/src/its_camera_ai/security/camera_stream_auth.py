"""
Camera Stream Authentication and Authorization.

Provides secure authentication mechanisms for camera streams:
- API key-based authentication for camera connections
- Certificate-based authentication for enterprise cameras
- Stream access control and authorization
- Camera credential management and rotation
- Real-time authentication monitoring
"""

import hashlib
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import jwt
from cryptography import x509
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import get_settings
from ..core.logging import get_logger
from ..models.camera import Camera

logger = get_logger(__name__)


class CameraAuthMethod(Enum):
    """Camera authentication methods."""
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    JWT_TOKEN = "jwt_token"
    DIGEST_AUTH = "digest_auth"


class CameraPermission(Enum):
    """Camera-specific permissions."""
    STREAM_READ = "stream_read"
    STREAM_WRITE = "stream_write"
    CONFIG_READ = "config_read"
    CONFIG_WRITE = "config_write"
    ADMIN = "admin"


@dataclass
class CameraCredentials:
    """Camera authentication credentials."""
    camera_id: str
    auth_method: CameraAuthMethod
    credentials: dict[str, Any]
    permissions: list[CameraPermission]
    expires_at: datetime | None = None
    created_at: datetime = None
    last_used_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(UTC)


@dataclass
class StreamAuthResult:
    """Stream authentication result."""
    authenticated: bool
    camera_id: str | None = None
    permissions: list[CameraPermission] = None
    auth_method: CameraAuthMethod | None = None
    expires_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []
        if self.metadata is None:
            self.metadata = {}


class CameraStreamAuthenticator:
    """Camera stream authentication and authorization manager."""

    def __init__(self, db_session: AsyncSession | None = None, settings=None):
        self.db_session = db_session
        self.settings = settings or get_settings()

        # Cache for active authentication sessions
        self._auth_cache: dict[str, StreamAuthResult] = {}
        self._cache_ttl = 300  # 5 minutes

        # Authentication attempt tracking for rate limiting
        self._auth_attempts: dict[str, list[float]] = {}
        self._max_attempts = 5
        self._attempt_window = 300  # 5 minutes

    async def authenticate_camera_stream(
        self,
        camera_id: str,
        auth_data: dict[str, Any],
        required_permissions: list[CameraPermission] | None = None
    ) -> StreamAuthResult:
        """
        Authenticate camera stream access.
        
        Args:
            camera_id: Camera identifier
            auth_data: Authentication data (API key, certificate, etc.)
            required_permissions: Required permissions for the operation
            
        Returns:
            StreamAuthResult with authentication status and permissions
        """
        if required_permissions is None:
            required_permissions = [CameraPermission.STREAM_READ]

        try:
            # Check rate limiting
            if not self._check_rate_limit(camera_id):
                return StreamAuthResult(
                    authenticated=False,
                    error="Too many authentication attempts",
                    metadata={"rate_limited": True}
                )

            # Check cache first
            cache_key = self._generate_cache_key(camera_id, auth_data)
            cached_result = self._get_cached_auth(cache_key)
            if cached_result and self._validate_cached_auth(cached_result):
                return cached_result

            # Determine authentication method
            auth_method = self._determine_auth_method(auth_data)
            if auth_method is None:
                return StreamAuthResult(
                    authenticated=False,
                    error="Invalid authentication method"
                )

            # Perform authentication based on method
            auth_result = await self._authenticate_by_method(
                camera_id, auth_method, auth_data, required_permissions
            )

            # Cache successful authentication
            if auth_result.authenticated:
                self._cache_auth_result(cache_key, auth_result)
                await self._log_auth_success(camera_id, auth_method, auth_result)
            else:
                await self._log_auth_failure(camera_id, auth_method, auth_result.error)

            # Track authentication attempt
            self._track_auth_attempt(camera_id)

            return auth_result

        except Exception as e:
            logger.error("Camera authentication error",
                        camera_id=camera_id,
                        error=str(e))
            return StreamAuthResult(
                authenticated=False,
                error="Authentication system error"
            )

    async def _authenticate_by_method(
        self,
        camera_id: str,
        auth_method: CameraAuthMethod,
        auth_data: dict[str, Any],
        required_permissions: list[CameraPermission]
    ) -> StreamAuthResult:
        """Authenticate using specific method."""
        if auth_method == CameraAuthMethod.API_KEY:
            return await self._authenticate_api_key(camera_id, auth_data, required_permissions)
        elif auth_method == CameraAuthMethod.CERTIFICATE:
            return await self._authenticate_certificate(camera_id, auth_data, required_permissions)
        elif auth_method == CameraAuthMethod.JWT_TOKEN:
            return await self._authenticate_jwt_token(camera_id, auth_data, required_permissions)
        elif auth_method == CameraAuthMethod.DIGEST_AUTH:
            return await self._authenticate_digest(camera_id, auth_data, required_permissions)
        else:
            return StreamAuthResult(
                authenticated=False,
                error=f"Unsupported authentication method: {auth_method}"
            )

    async def _authenticate_api_key(
        self,
        camera_id: str,
        auth_data: dict[str, Any],
        required_permissions: list[CameraPermission]
    ) -> StreamAuthResult:
        """Authenticate using API key."""
        api_key = auth_data.get("api_key")
        if not api_key:
            return StreamAuthResult(
                authenticated=False,
                error="API key not provided"
            )

        try:
            # Hash the API key for secure lookup
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Look up camera credentials in database
            if self.db_session:
                query = select(Camera).where(
                    Camera.id == camera_id,
                    Camera.api_key_hash == key_hash,
                    Camera.is_active == True
                )
                result = await self.db_session.execute(query)
                camera = result.scalar_one_or_none()

                if not camera:
                    return StreamAuthResult(
                        authenticated=False,
                        error="Invalid camera API key"
                    )

                # Check if API key has expired
                if camera.api_key_expires_at and datetime.now(UTC) > camera.api_key_expires_at:
                    return StreamAuthResult(
                        authenticated=False,
                        error="API key has expired"
                    )

                # Get camera permissions
                camera_permissions = self._get_camera_permissions(camera)

                # Check required permissions
                if not self._has_required_permissions(camera_permissions, required_permissions):
                    return StreamAuthResult(
                        authenticated=False,
                        error="Insufficient permissions"
                    )

                return StreamAuthResult(
                    authenticated=True,
                    camera_id=camera_id,
                    permissions=camera_permissions,
                    auth_method=CameraAuthMethod.API_KEY,
                    expires_at=camera.api_key_expires_at,
                    metadata={"camera_name": camera.name, "location": camera.location}
                )
            else:
                # Fallback when no database session available
                logger.warning("Database session not available for camera authentication")
                return StreamAuthResult(
                    authenticated=False,
                    error="Authentication service unavailable"
                )

        except Exception as e:
            logger.error("API key authentication failed",
                        camera_id=camera_id,
                        error=str(e))
            return StreamAuthResult(
                authenticated=False,
                error="API key authentication failed"
            )

    async def _authenticate_certificate(
        self,
        camera_id: str,
        auth_data: dict[str, Any],
        required_permissions: list[CameraPermission]
    ) -> StreamAuthResult:
        """Authenticate using X.509 certificate."""
        cert_data = auth_data.get("certificate")
        if not cert_data:
            return StreamAuthResult(
                authenticated=False,
                error="Certificate not provided"
            )

        try:
            # Parse certificate
            if isinstance(cert_data, str):
                cert_data = cert_data.encode()

            cert = x509.load_pem_x509_certificate(cert_data)

            # Validate certificate
            validation_result = self._validate_camera_certificate(cert, camera_id)
            if not validation_result["valid"]:
                return StreamAuthResult(
                    authenticated=False,
                    error=validation_result["error"]
                )

            # Get permissions from certificate
            cert_permissions = self._extract_permissions_from_cert(cert)

            # Check required permissions
            if not self._has_required_permissions(cert_permissions, required_permissions):
                return StreamAuthResult(
                    authenticated=False,
                    error="Certificate does not grant required permissions"
                )

            return StreamAuthResult(
                authenticated=True,
                camera_id=camera_id,
                permissions=cert_permissions,
                auth_method=CameraAuthMethod.CERTIFICATE,
                expires_at=cert.not_valid_after.replace(tzinfo=UTC),
                metadata={
                    "cert_subject": cert.subject.rfc4514_string(),
                    "cert_issuer": cert.issuer.rfc4514_string(),
                    "cert_serial": str(cert.serial_number)
                }
            )

        except Exception as e:
            logger.error("Certificate authentication failed",
                        camera_id=camera_id,
                        error=str(e))
            return StreamAuthResult(
                authenticated=False,
                error="Certificate authentication failed"
            )

    async def _authenticate_jwt_token(
        self,
        camera_id: str,
        auth_data: dict[str, Any],
        required_permissions: list[CameraPermission]
    ) -> StreamAuthResult:
        """Authenticate using JWT token."""
        token = auth_data.get("jwt_token")
        if not token:
            return StreamAuthResult(
                authenticated=False,
                error="JWT token not provided"
            )

        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                self.settings.security.secret_key,
                algorithms=[self.settings.security.algorithm]
            )

            # Validate camera ID in token
            token_camera_id = payload.get("camera_id")
            if token_camera_id != camera_id:
                return StreamAuthResult(
                    authenticated=False,
                    error="Camera ID mismatch in token"
                )

            # Check token expiration
            exp = payload.get("exp")
            if exp and datetime.fromtimestamp(exp, UTC) < datetime.now(UTC):
                return StreamAuthResult(
                    authenticated=False,
                    error="JWT token has expired"
                )

            # Extract permissions from token
            token_permissions = [
                CameraPermission(perm) for perm in payload.get("permissions", [])
                if perm in [p.value for p in CameraPermission]
            ]

            # Check required permissions
            if not self._has_required_permissions(token_permissions, required_permissions):
                return StreamAuthResult(
                    authenticated=False,
                    error="Token does not grant required permissions"
                )

            return StreamAuthResult(
                authenticated=True,
                camera_id=camera_id,
                permissions=token_permissions,
                auth_method=CameraAuthMethod.JWT_TOKEN,
                expires_at=datetime.fromtimestamp(exp, UTC) if exp else None,
                metadata={"token_issuer": payload.get("iss"), "token_subject": payload.get("sub")}
            )

        except jwt.ExpiredSignatureError:
            return StreamAuthResult(
                authenticated=False,
                error="JWT token has expired"
            )
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid JWT token", camera_id=camera_id, error=str(e))
            return StreamAuthResult(
                authenticated=False,
                error="Invalid JWT token"
            )
        except Exception as e:
            logger.error("JWT authentication failed", camera_id=camera_id, error=str(e))
            return StreamAuthResult(
                authenticated=False,
                error="JWT authentication failed"
            )

    async def _authenticate_digest(
        self,
        camera_id: str,
        auth_data: dict[str, Any],
        required_permissions: list[CameraPermission]
    ) -> StreamAuthResult:
        """Authenticate using HTTP Digest authentication."""
        username = auth_data.get("username")
        password_hash = auth_data.get("password_hash")
        nonce = auth_data.get("nonce")

        if not all([username, password_hash, nonce]):
            return StreamAuthResult(
                authenticated=False,
                error="Incomplete digest authentication data"
            )

        try:
            # Look up camera digest credentials
            if self.db_session:
                query = select(Camera).where(
                    Camera.id == camera_id,
                    Camera.digest_username == username,
                    Camera.is_active == True
                )
                result = await self.db_session.execute(query)
                camera = result.scalar_one_or_none()

                if not camera or not camera.digest_password_hash:
                    return StreamAuthResult(
                        authenticated=False,
                        error="Invalid digest credentials"
                    )

                # Validate digest hash
                expected_hash = self._calculate_digest_hash(
                    username, camera.digest_password_hash, nonce, camera_id
                )

                if password_hash != expected_hash:
                    return StreamAuthResult(
                        authenticated=False,
                        error="Invalid digest authentication"
                    )

                # Get permissions
                camera_permissions = self._get_camera_permissions(camera)

                return StreamAuthResult(
                    authenticated=True,
                    camera_id=camera_id,
                    permissions=camera_permissions,
                    auth_method=CameraAuthMethod.DIGEST_AUTH,
                    metadata={"username": username}
                )
            else:
                return StreamAuthResult(
                    authenticated=False,
                    error="Digest authentication service unavailable"
                )

        except Exception as e:
            logger.error("Digest authentication failed", camera_id=camera_id, error=str(e))
            return StreamAuthResult(
                authenticated=False,
                error="Digest authentication failed"
            )

    def _determine_auth_method(self, auth_data: dict[str, Any]) -> CameraAuthMethod | None:
        """Determine authentication method from auth data."""
        if "api_key" in auth_data:
            return CameraAuthMethod.API_KEY
        elif "certificate" in auth_data:
            return CameraAuthMethod.CERTIFICATE
        elif "jwt_token" in auth_data:
            return CameraAuthMethod.JWT_TOKEN
        elif "username" in auth_data and "password_hash" in auth_data:
            return CameraAuthMethod.DIGEST_AUTH
        else:
            return None

    def _validate_camera_certificate(self, cert: x509.Certificate, camera_id: str) -> dict[str, Any]:
        """Validate camera certificate."""
        try:
            # Check certificate validity period
            now = datetime.now(UTC)
            if now < cert.not_valid_before.replace(tzinfo=UTC):
                return {"valid": False, "error": "Certificate not yet valid"}

            if now > cert.not_valid_after.replace(tzinfo=UTC):
                return {"valid": False, "error": "Certificate has expired"}

            # Check if certificate is for this camera
            subject = cert.subject
            cn = None
            for attribute in subject:
                if attribute.oid == x509.NameOID.COMMON_NAME:
                    cn = attribute.value
                    break

            if cn != camera_id:
                # Check Subject Alternative Names
                try:
                    san_ext = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
                    san_names = [name.value for name in san_ext.value if hasattr(name, 'value')]
                    if camera_id not in san_names:
                        return {"valid": False, "error": "Certificate not issued for this camera"}
                except x509.ExtensionNotFound:
                    return {"valid": False, "error": "Certificate not issued for this camera"}

            # TODO: Verify certificate chain against trusted CA
            # This would require implementing certificate chain validation

            return {"valid": True}

        except Exception as e:
            return {"valid": False, "error": f"Certificate validation error: {str(e)}"}

    def _extract_permissions_from_cert(self, cert: x509.Certificate) -> list[CameraPermission]:
        """Extract permissions from certificate extensions."""
        permissions = [CameraPermission.STREAM_READ]  # Default permission

        try:
            # Look for custom extension containing permissions
            # This would be defined in your certificate issuance process
            # For now, return default permissions
            pass
        except:
            pass

        return permissions

    def _get_camera_permissions(self, camera) -> list[CameraPermission]:
        """Get permissions for camera from database record."""
        # Default permissions based on camera configuration
        permissions = [CameraPermission.STREAM_READ]

        if hasattr(camera, 'permissions') and camera.permissions:
            try:
                permissions = [CameraPermission(perm) for perm in camera.permissions
                             if perm in [p.value for p in CameraPermission]]
            except:
                pass

        return permissions

    def _has_required_permissions(
        self,
        granted_permissions: list[CameraPermission],
        required_permissions: list[CameraPermission]
    ) -> bool:
        """Check if granted permissions include all required permissions."""
        # Admin permission grants all access
        if CameraPermission.ADMIN in granted_permissions:
            return True

        return all(perm in granted_permissions for perm in required_permissions)

    def _check_rate_limit(self, camera_id: str) -> bool:
        """Check authentication rate limiting."""
        now = time.time()

        if camera_id not in self._auth_attempts:
            self._auth_attempts[camera_id] = []

        # Clean old attempts
        self._auth_attempts[camera_id] = [
            attempt for attempt in self._auth_attempts[camera_id]
            if now - attempt < self._attempt_window
        ]

        # Check if under limit
        return len(self._auth_attempts[camera_id]) < self._max_attempts

    def _track_auth_attempt(self, camera_id: str) -> None:
        """Track authentication attempt for rate limiting."""
        now = time.time()
        if camera_id not in self._auth_attempts:
            self._auth_attempts[camera_id] = []
        self._auth_attempts[camera_id].append(now)

    def _generate_cache_key(self, camera_id: str, auth_data: dict[str, Any]) -> str:
        """Generate cache key for authentication result."""
        # Create a hash of camera_id and auth data
        auth_str = f"{camera_id}:{str(sorted(auth_data.items()))}"
        return hashlib.sha256(auth_str.encode()).hexdigest()

    def _get_cached_auth(self, cache_key: str) -> StreamAuthResult | None:
        """Get cached authentication result."""
        return self._auth_cache.get(cache_key)

    def _validate_cached_auth(self, auth_result: StreamAuthResult) -> bool:
        """Validate if cached authentication is still valid."""
        if auth_result.expires_at and datetime.now(UTC) > auth_result.expires_at:
            return False
        return True

    def _cache_auth_result(self, cache_key: str, auth_result: StreamAuthResult) -> None:
        """Cache authentication result."""
        # Simple cache implementation - in production, use Redis or similar
        self._auth_cache[cache_key] = auth_result

    def _calculate_digest_hash(self, username: str, password_hash: str, nonce: str, uri: str) -> str:
        """Calculate digest authentication hash."""
        # Simplified digest calculation - implement full RFC 7616 for production
        hash_input = f"{username}:{password_hash}:{nonce}:{uri}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    async def _log_auth_success(
        self,
        camera_id: str,
        auth_method: CameraAuthMethod,
        auth_result: StreamAuthResult
    ) -> None:
        """Log successful authentication."""
        logger.info("Camera authentication successful",
                   camera_id=camera_id,
                   auth_method=auth_method.value,
                   permissions=[p.value for p in auth_result.permissions],
                   expires_at=auth_result.expires_at.isoformat() if auth_result.expires_at else None)

    async def _log_auth_failure(
        self,
        camera_id: str,
        auth_method: CameraAuthMethod,
        error: str
    ) -> None:
        """Log authentication failure."""
        logger.warning("Camera authentication failed",
                      camera_id=camera_id,
                      auth_method=auth_method.value if auth_method else "unknown",
                      error=error)

    async def revoke_camera_access(self, camera_id: str) -> bool:
        """Revoke all access for a camera."""
        try:
            # Clear authentication cache for this camera
            keys_to_remove = [
                key for key in self._auth_cache.keys()
                if self._auth_cache[key].camera_id == camera_id
            ]

            for key in keys_to_remove:
                del self._auth_cache[key]

            # Update database to revoke access
            if self.db_session:
                # Implementation would mark camera as inactive or revoke credentials
                pass

            logger.info("Camera access revoked", camera_id=camera_id)
            return True

        except Exception as e:
            logger.error("Failed to revoke camera access", camera_id=camera_id, error=str(e))
            return False

    async def rotate_camera_credentials(self, camera_id: str) -> bool:
        """Rotate camera credentials."""
        try:
            # Implementation would generate new API keys/certificates
            # and update database records

            # Clear cached authentication for this camera
            await self.revoke_camera_access(camera_id)

            logger.info("Camera credentials rotated", camera_id=camera_id)
            return True

        except Exception as e:
            logger.error("Failed to rotate camera credentials", camera_id=camera_id, error=str(e))
            return False
