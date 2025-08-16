"""
HashiCorp Vault client for ITS Camera AI secrets management.

Provides secure secret retrieval, caching, and automatic token renewal
for production Kubernetes environments with Vault integration.
"""

import asyncio
import logging
import os
import time
from typing import Any
from urllib.parse import urljoin

import aiohttp
from pydantic import BaseModel, Field

from its_camera_ai.core.config import get_settings

logger = logging.getLogger(__name__)


class VaultConfig(BaseModel):
    """Vault client configuration."""

    vault_addr: str = Field(default="https://vault.its-camera-ai.svc.cluster.local:8200")
    vault_namespace: str | None = Field(default=None)
    auth_method: str = Field(default="kubernetes")
    auth_mount_path: str = Field(default="auth/kubernetes")
    role: str = Field(default="its-camera-ai-api")

    # Kubernetes auth
    service_account_token_path: str = Field(default="/var/run/secrets/kubernetes.io/serviceaccount/token")

    # TLS configuration
    ca_cert_path: str | None = Field(default="/vault/tls/ca.crt")
    verify_ssl: bool = Field(default=True)

    # Token management
    token_renewal_threshold: int = Field(default=300)  # Renew token 5 minutes before expiry
    max_retries: int = Field(default=3)
    retry_delay: float = Field(default=1.0)

    # Secret caching
    cache_ttl: int = Field(default=3600)  # Cache secrets for 1 hour
    enable_cache: bool = Field(default=True)


class VaultSecret(BaseModel):
    """Represents a secret retrieved from Vault."""

    data: dict[str, Any]
    metadata: dict[str, Any]
    lease_duration: int
    renewable: bool
    retrieved_at: float = Field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        """Check if the secret is expired based on lease duration."""
        if self.lease_duration == 0:
            return False  # Non-renewable secrets don't expire
        return time.time() - self.retrieved_at >= self.lease_duration


class VaultAuthToken(BaseModel):
    """Vault authentication token information."""

    client_token: str
    accessor: str
    policies: list[str]
    token_policies: list[str]
    metadata: dict[str, Any]
    lease_duration: int
    renewable: bool
    entity_id: str
    token_type: str
    orphan: bool
    issued_at: float = Field(default_factory=time.time)

    @property
    def expires_at(self) -> float:
        """Calculate when the token expires."""
        return self.issued_at + self.lease_duration

    @property
    def needs_renewal(self) -> bool:
        """Check if token needs renewal based on threshold."""
        return time.time() >= (self.expires_at - 300)  # 5 minutes before expiry


class VaultClient:
    """
    Async HashiCorp Vault client for ITS Camera AI.
    
    Features:
    - Kubernetes authentication
    - Automatic token renewal
    - Secret caching with TTL
    - Connection pooling
    - Retry logic with exponential backoff
    - Structured logging
    """

    def __init__(self, config: VaultConfig | None = None):
        """Initialize Vault client with configuration."""
        self.config = config or VaultConfig()
        self._session: aiohttp.ClientSession | None = None
        self._token: VaultAuthToken | None = None
        self._secret_cache: dict[str, VaultSecret] = {}
        self._auth_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()

        # Configure TLS
        self._ssl_context = None
        if self.config.verify_ssl and self.config.ca_cert_path:
            import ssl
            self._ssl_context = ssl.create_default_context(cafile=self.config.ca_cert_path)

        logger.info(
            "Initialized Vault client",
            extra={
                "vault_addr": self.config.vault_addr,
                "auth_method": self.config.auth_method,
                "role": self.config.role,
                "verify_ssl": self.config.verify_ssl
            }
        )

    async def __aenter__(self) -> "VaultClient":
        """Async context manager entry."""
        await self._ensure_session()
        await self._ensure_authenticated()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=20,
                limit_per_host=10,
                ttl_dns_cache=300,
                use_dns_cache=True,
                ssl=self._ssl_context if self.config.verify_ssl else False
            )

            timeout = aiohttp.ClientTimeout(total=30, connect=10)

            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "ITS-Camera-AI-Vault-Client/1.0"}
            )

    async def _ensure_authenticated(self) -> None:
        """Ensure client is authenticated with valid token."""
        async with self._auth_lock:
            if self._token is None or self._token.needs_renewal:
                await self._authenticate()

    async def _authenticate(self) -> None:
        """Authenticate with Vault using Kubernetes auth method."""
        await self._ensure_session()

        try:
            # Read service account token
            if not os.path.exists(self.config.service_account_token_path):
                raise VaultAuthError(f"Service account token not found: {self.config.service_account_token_path}")

            with open(self.config.service_account_token_path) as f:
                jwt_token = f.read().strip()

            # Authenticate with Vault
            auth_url = urljoin(self.config.vault_addr, f"v1/{self.config.auth_mount_path}/login")

            auth_data = {
                "role": self.config.role,
                "jwt": jwt_token
            }

            headers = {}
            if self.config.vault_namespace:
                headers["X-Vault-Namespace"] = self.config.vault_namespace

            logger.debug("Authenticating with Vault", extra={"auth_url": auth_url, "role": self.config.role})

            async with self._session.post(auth_url, json=auth_data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise VaultAuthError(f"Authentication failed: {response.status} - {error_text}")

                auth_response = await response.json()
                auth_info = auth_response.get("auth", {})

                self._token = VaultAuthToken(
                    client_token=auth_info["client_token"],
                    accessor=auth_info["accessor"],
                    policies=auth_info["policies"],
                    token_policies=auth_info["token_policies"],
                    metadata=auth_info["metadata"],
                    lease_duration=auth_info["lease_duration"],
                    renewable=auth_info["renewable"],
                    entity_id=auth_info["entity_id"],
                    token_type=auth_info["token_type"],
                    orphan=auth_info["orphan"]
                )

                logger.info(
                    "Successfully authenticated with Vault",
                    extra={
                        "token_accessor": self._token.accessor[:8] + "...",
                        "policies": self._token.policies,
                        "lease_duration": self._token.lease_duration,
                        "renewable": self._token.renewable
                    }
                )

        except Exception as e:
            logger.error("Failed to authenticate with Vault", extra={"error": str(e)})
            raise VaultAuthError(f"Authentication failed: {e}") from e

    async def _make_request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make authenticated request to Vault API."""
        await self._ensure_session()
        await self._ensure_authenticated()

        url = urljoin(self.config.vault_addr, f"v1/{path}")

        headers = {
            "X-Vault-Token": self._token.client_token
        }
        if self.config.vault_namespace:
            headers["X-Vault-Namespace"] = self.config.vault_namespace

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.request(
                    method,
                    url,
                    json=data,
                    params=params,
                    headers=headers
                ) as response:

                    if response.status == 403:
                        # Token might be expired, try to re-authenticate
                        if attempt == 0:
                            logger.warning("Received 403, attempting re-authentication")
                            await self._authenticate()
                            headers["X-Vault-Token"] = self._token.client_token
                            continue
                        else:
                            raise VaultPermissionError("Access denied after re-authentication")

                    if response.status >= 400:
                        error_text = await response.text()
                        raise VaultAPIError(f"API request failed: {response.status} - {error_text}")

                    if response.status == 204:
                        return {}

                    return await response.json()

            except (TimeoutError, aiohttp.ClientError) as e:
                if attempt == self.config.max_retries - 1:
                    raise VaultConnectionError(f"Failed to connect to Vault after {self.config.max_retries} attempts") from e

                delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Request failed, retrying in {delay}s",
                    extra={"attempt": attempt + 1, "error": str(e)}
                )
                await asyncio.sleep(delay)

    async def get_secret(self, path: str, version: int | None = None, use_cache: bool = True) -> VaultSecret:
        """
        Retrieve a secret from Vault KV v2 engine.
        
        Args:
            path: Secret path (e.g., "its-camera-ai/data/api/jwt")
            version: Specific version to retrieve (latest if None)
            use_cache: Whether to use cached value if available
            
        Returns:
            VaultSecret containing the secret data and metadata
        """
        cache_key = f"{path}:v{version}" if version else path

        # Check cache first
        if use_cache and self.config.enable_cache:
            async with self._cache_lock:
                cached_secret = self._secret_cache.get(cache_key)
                if cached_secret and not cached_secret.is_expired:
                    logger.debug("Returning cached secret", extra={"path": path})
                    return cached_secret

        # Prepare request parameters
        params = {}
        if version:
            params["version"] = version

        # Retrieve secret from Vault
        logger.debug("Retrieving secret from Vault", extra={"path": path, "version": version})

        try:
            response = await self._make_request("GET", path, params=params)

            secret = VaultSecret(
                data=response.get("data", {}).get("data", {}),
                metadata=response.get("data", {}).get("metadata", {}),
                lease_duration=response.get("lease_duration", 0),
                renewable=response.get("renewable", False)
            )

            # Cache the secret
            if self.config.enable_cache:
                async with self._cache_lock:
                    self._secret_cache[cache_key] = secret

            logger.info(
                "Successfully retrieved secret",
                extra={
                    "path": path,
                    "version": version,
                    "data_keys": list(secret.data.keys()),
                    "renewable": secret.renewable
                }
            )

            return secret

        except Exception as e:
            logger.error("Failed to retrieve secret", extra={"path": path, "error": str(e)})
            raise

    async def get_database_config(self) -> dict[str, str]:
        """Get database configuration from Vault."""
        postgres_secret = await self.get_secret("its-camera-ai/data/database/postgres")
        redis_secret = await self.get_secret("its-camera-ai/data/database/redis")

        return {
            # PostgreSQL
            "DB_HOST": postgres_secret.data["host"],
            "DB_PORT": str(postgres_secret.data["port"]),
            "DB_NAME": postgres_secret.data["database"],
            "DB_USERNAME": postgres_secret.data["username"],
            "DB_PASSWORD": postgres_secret.data["password"],
            "DB_SSL_MODE": postgres_secret.data["ssl_mode"],

            # Redis
            "REDIS_HOST": redis_secret.data["host"],
            "REDIS_PORT": str(redis_secret.data["port"]),
            "REDIS_PASSWORD": redis_secret.data["password"],
        }

    async def get_api_config(self) -> dict[str, str]:
        """Get API configuration from Vault."""
        jwt_secret = await self.get_secret("its-camera-ai/data/api/jwt")
        encryption_secret = await self.get_secret("its-camera-ai/data/encryption/master")

        return {
            "JWT_SECRET_KEY": jwt_secret.data["secret_key"],
            "JWT_REFRESH_SECRET_KEY": jwt_secret.data["refresh_secret_key"],
            "ENCRYPTION_KEY": encryption_secret.data["key"],
            "ENCRYPTION_ALGORITHM": encryption_secret.data["algorithm"],
        }

    async def get_ml_config(self) -> dict[str, str]:
        """Get ML configuration from Vault."""
        ml_secret = await self.get_secret("its-camera-ai/data/ml/models")

        return {
            "MODEL_REGISTRY_KEY": ml_secret.data["model_registry_key"],
            "MODEL_ENCRYPTION_KEY": ml_secret.data["model_encryption_key"],
            "TENSORRT_LICENSE_KEY": ml_secret.data["tensorrt_license_key"],
        }

    async def put_secret(self, path: str, data: dict[str, Any]) -> None:
        """
        Store a secret in Vault KV v2 engine.
        
        Args:
            path: Secret path (e.g., "its-camera-ai/data/api/jwt")
            data: Secret data to store
        """
        logger.info("Storing secret in Vault", extra={"path": path, "data_keys": list(data.keys())})

        payload = {"data": data}
        await self._make_request("POST", path, data=payload)

        # Invalidate cache
        if self.config.enable_cache:
            async with self._cache_lock:
                # Remove all cached versions of this path
                keys_to_remove = [key for key in self._secret_cache.keys() if key.startswith(path)]
                for key in keys_to_remove:
                    del self._secret_cache[key]

        logger.info("Successfully stored secret", extra={"path": path})

    async def renew_token(self) -> None:
        """Renew the current authentication token."""
        if not self._token or not self._token.renewable:
            logger.warning("Token is not renewable, re-authenticating")
            await self._authenticate()
            return

        try:
            response = await self._make_request("POST", "auth/token/renew-self")
            auth_info = response.get("auth", {})

            # Update token information
            self._token.lease_duration = auth_info["lease_duration"]
            self._token.issued_at = time.time()

            logger.info("Successfully renewed token", extra={"new_lease_duration": self._token.lease_duration})

        except Exception as e:
            logger.error("Failed to renew token, re-authenticating", extra={"error": str(e)})
            await self._authenticate()

    async def clear_cache(self) -> None:
        """Clear the secret cache."""
        async with self._cache_lock:
            self._secret_cache.clear()
        logger.info("Cleared secret cache")

    async def health_check(self) -> dict[str, Any]:
        """Check Vault health status."""
        try:
            # Use a different session for health check to avoid auth requirements
            async with aiohttp.ClientSession() as session:
                url = urljoin(self.config.vault_addr, "v1/sys/health")
                async with session.get(url, ssl=self._ssl_context if self.config.verify_ssl else False) as response:
                    return await response.json()
        except Exception as e:
            logger.error("Health check failed", extra={"error": str(e)})
            raise VaultConnectionError(f"Health check failed: {e}") from e

    async def close(self) -> None:
        """Close the Vault client and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()

        self._token = None
        self._secret_cache.clear()

        logger.info("Vault client closed")


# Vault-specific exceptions
class VaultError(Exception):
    """Base exception for Vault-related errors."""
    pass


class VaultConnectionError(VaultError):
    """Raised when unable to connect to Vault."""
    pass


class VaultAuthError(VaultError):
    """Raised when authentication with Vault fails."""
    pass


class VaultPermissionError(VaultError):
    """Raised when access to a secret is denied."""
    pass


class VaultAPIError(VaultError):
    """Raised when Vault API returns an error."""
    pass


# Singleton pattern for application-wide Vault client
_vault_client: VaultClient | None = None


async def get_vault_client() -> VaultClient:
    """Get or create the global Vault client instance."""
    global _vault_client

    if _vault_client is None:
        settings = get_settings()

        # Create Vault config from environment/settings
        vault_config = VaultConfig(
            vault_addr=os.getenv("VAULT_ADDR", "https://vault.its-camera-ai.svc.cluster.local:8200"),
            vault_namespace=os.getenv("VAULT_NAMESPACE"),
            role=os.getenv("VAULT_ROLE", "its-camera-ai-api"),
            verify_ssl=os.getenv("VAULT_VERIFY_SSL", "true").lower() == "true"
        )

        _vault_client = VaultClient(vault_config)
        await _vault_client._ensure_session()
        await _vault_client._ensure_authenticated()

    return _vault_client


async def load_secrets_from_vault() -> dict[str, str]:
    """
    Load all application secrets from Vault.
    
    Returns:
        Dictionary of environment variables loaded from Vault
    """
    vault_client = await get_vault_client()

    try:
        # Load all secret configurations
        db_config = await vault_client.get_database_config()
        api_config = await vault_client.get_api_config()

        # Combine all configurations
        all_secrets = {**db_config, **api_config}

        logger.info(
            "Successfully loaded secrets from Vault",
            extra={"secret_count": len(all_secrets), "secret_keys": list(all_secrets.keys())}
        )

        return all_secrets

    except Exception as e:
        logger.error("Failed to load secrets from Vault", extra={"error": str(e)})
        raise
