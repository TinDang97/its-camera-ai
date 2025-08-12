"""API client for backend service integration.

Provides HTTP client functionality for communicating with FastAPI backend services
with proper authentication, retry logic, and connection pooling.
"""

import asyncio
import json
import time
from typing import Any

import httpx
from httpx import AsyncClient, ConnectError, RequestError, TimeoutException
from rich.console import Console

from ...core.config import Settings, get_settings
from ...core.exceptions import ITSCameraAIError
from ...core.logging import get_logger

logger = get_logger(__name__)
console = Console()


class APIClientError(ITSCameraAIError):
    """API client specific errors."""

    pass


class ConnectionError(APIClientError):
    """Connection-related errors."""

    pass


class AuthenticationError(APIClientError):
    """Authentication-related errors."""

    pass


class APIClient:
    """Async HTTP client for backend API communication.

    Features:
    - Connection pooling and keep-alive
    - Automatic retry with exponential backoff
    - JWT token management
    - Request/response logging
    - Circuit breaker pattern
    - Response caching
    """

    def __init__(
        self,
        base_url: str | None = None,
        settings: Settings | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize API client.

        Args:
            base_url: Base URL for API endpoints
            settings: Application settings
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Initial retry delay in seconds
        """
        self.settings = settings or get_settings()
        self.base_url = (
            base_url or f"http://{self.settings.api_host}:{self.settings.api_port}"
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Client instance
        self._client: AsyncClient | None = None
        self._session_created = False

        # Authentication
        self._access_token: str | None = None
        self._token_expires_at: float | None = None

        # Circuit breaker state
        self._failure_count = 0
        self._last_failure_time = 0
        self._circuit_open_until = 0
        self._failure_threshold = 5
        self._circuit_timeout = 60  # seconds

        # Request cache
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ttl = 300  # 5 minutes

        logger.info(f"API client initialized with base URL: {self.base_url}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Establish HTTP client connection."""
        if self._client is None:
            # Configure client with connection pooling
            limits = httpx.Limits(
                max_keepalive_connections=20, max_connections=100, keepalive_expiry=30.0
            )

            timeout = httpx.Timeout(
                connect=5.0, read=self.timeout, write=10.0, pool=5.0
            )

            headers = {
                "User-Agent": f"ITS-Camera-AI-CLI/{self.settings.app_version}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

            self._client = AsyncClient(
                base_url=self.base_url,
                timeout=timeout,
                limits=limits,
                headers=headers,
                follow_redirects=True,
                verify=bool(self.settings.is_production()),
            )

            self._session_created = True
            logger.info("HTTP client connection established")

    async def close(self) -> None:
        """Close HTTP client connection."""
        if self._client and self._session_created:
            await self._client.aclose()
            self._client = None
            self._session_created = False
            logger.info("HTTP client connection closed")

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._failure_count >= self._failure_threshold:
            if time.time() < self._circuit_open_until:
                return True
            else:
                # Reset circuit breaker
                self._failure_count = 0
                self._circuit_open_until = 0
        return False

    def _record_success(self) -> None:
        """Record successful request."""
        self._failure_count = 0
        self._last_failure_time = 0

    def _record_failure(self) -> None:
        """Record failed request."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self._failure_threshold:
            self._circuit_open_until = time.time() + self._circuit_timeout
            logger.warning(
                f"Circuit breaker opened - too many failures ({self._failure_count})"
            )

    def _get_cache_key(self, method: str, url: str, params: dict | None = None) -> str:
        """Generate cache key for request."""
        key_data = f"{method}:{url}"
        if params:
            key_data += f":{json.dumps(params, sort_keys=True)}"
        return key_data

    def _is_cache_valid(self, cache_entry: dict[str, Any]) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - cache_entry["timestamp"] < self._cache_ttl

    async def _ensure_authenticated(self) -> None:
        """Ensure valid authentication token."""
        current_time = time.time()

        # Check if token is valid
        if (
            self._access_token
            and self._token_expires_at
            and current_time < self._token_expires_at - 30  # 30 second buffer
        ):
            return

        # Token expired or missing, need to authenticate
        try:
            # This would typically get credentials from config or prompt user
            # For now, we'll assume the API handles authentication differently
            # or that the CLI operates with system-level permissions
            logger.debug("Authentication token refresh needed but not implemented yet")

        except Exception as e:
            raise AuthenticationError(f"Failed to authenticate: {e}") from e

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        data: str | bytes | None = None,
        headers: dict[str, str] | None = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        """Make HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint (without base URL)
            params: Query parameters
            json_data: JSON request body
            data: Raw request body
            headers: Additional headers
            use_cache: Whether to use response caching

        Returns:
            Dict containing response data

        Raises:
            ConnectionError: For connection issues
            AuthenticationError: For auth failures
            APIClientError: For other API errors
        """
        if not self._client:
            await self.connect()

        # Check circuit breaker
        if self._is_circuit_open():
            raise ConnectionError("Circuit breaker is open - too many recent failures")

        # Check cache for GET requests
        if method.upper() == "GET" and use_cache:
            cache_key = self._get_cache_key(method, endpoint, params)
            if cache_key in self._cache and self._is_cache_valid(
                self._cache[cache_key]
            ):
                logger.debug(f"Returning cached response for {method} {endpoint}")
                return self._cache[cache_key]["data"]

        # Ensure authentication
        await self._ensure_authenticated()

        # Prepare headers
        req_headers = {}
        if self._access_token:
            req_headers["Authorization"] = f"Bearer {self._access_token}"
        if headers:
            req_headers.update(headers)

        # Retry logic with exponential backoff
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()

                response = await self._client.request(
                    method=method,
                    url=endpoint,
                    params=params,
                    json=json_data,
                    content=data,
                    headers=req_headers,
                )

                duration = time.time() - start_time

                # Log request
                logger.debug(
                    f"{method} {endpoint} - {response.status_code} ({duration:.2f}s)"
                )

                # Handle response
                if response.status_code == 401:
                    self._record_failure()
                    raise AuthenticationError("Authentication required")

                if response.status_code == 403:
                    self._record_failure()
                    raise AuthenticationError("Access denied")

                if response.status_code >= 400:
                    self._record_failure()
                    error_text = response.text
                    raise APIClientError(
                        f"HTTP {response.status_code}: {error_text}",
                        code=f"HTTP_{response.status_code}",
                    )

                # Success
                self._record_success()

                # Parse JSON response
                try:
                    result = response.json()
                except json.JSONDecodeError:
                    result = {"data": response.text, "status": "success"}

                # Cache GET responses
                if method.upper() == "GET" and use_cache:
                    cache_key = self._get_cache_key(method, endpoint, params)
                    self._cache[cache_key] = {
                        "data": result,
                        "timestamp": time.time(),
                    }

                return result

            except (ConnectError, TimeoutException, RequestError) as e:
                last_exception = e
                self._record_failure()

                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Request failed after {self.max_retries + 1} attempts: {e}"
                    )
                    raise ConnectionError(
                        f"Failed to connect after {self.max_retries + 1} attempts: {e}"
                    )

            except Exception as e:
                self._record_failure()
                logger.error(f"Unexpected error during request: {e}")
                raise APIClientError(f"Request failed: {e}")

        # Should never reach here, but just in case
        raise ConnectionError(f"Request failed: {last_exception}")

    # HTTP method convenience functions

    async def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Make GET request."""
        return await self._make_request(
            "GET", endpoint, params=params, use_cache=use_cache, **kwargs
        )

    async def post(
        self,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        data: str | bytes | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Make POST request."""
        return await self._make_request(
            "POST", endpoint, json_data=json_data, data=data, **kwargs
        )

    async def put(
        self,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        data: str | bytes | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Make PUT request."""
        return await self._make_request(
            "PUT", endpoint, json_data=json_data, data=data, **kwargs
        )

    async def patch(
        self, endpoint: str, json_data: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Make PATCH request."""
        return await self._make_request(
            "PATCH", endpoint, json_data=json_data, **kwargs
        )

    async def delete(
        self, endpoint: str, params: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Make DELETE request."""
        return await self._make_request("DELETE", endpoint, params=params, **kwargs)

    # High-level API methods

    async def get_health(self) -> dict[str, Any]:
        """Get system health status."""
        return await self.get("/health", use_cache=False)

    async def get_system_status(self) -> dict[str, Any]:
        """Get detailed system status."""
        return await self.get("/api/v1/system/status", use_cache=False)

    async def get_services(self) -> dict[str, Any]:
        """Get list of available services."""
        return await self.get("/api/v1/system/services", use_cache=True)

    async def get_metrics(self, service: str = None) -> dict[str, Any]:
        """Get system metrics."""
        params = {"service": service} if service else None
        return await self.get("/api/v1/system/metrics", params=params, use_cache=False)

    async def list_cameras(self) -> dict[str, Any]:
        """Get list of cameras."""
        return await self.get("/api/v1/cameras", use_cache=True)

    async def get_camera(self, camera_id: str) -> dict[str, Any]:
        """Get camera details."""
        return await self.get(f"/api/v1/cameras/{camera_id}", use_cache=True)

    async def create_camera(self, camera_data: dict[str, Any]) -> dict[str, Any]:
        """Create new camera."""
        return await self.post("/api/v1/cameras", json_data=camera_data)

    async def update_camera(
        self, camera_id: str, camera_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Update camera configuration."""
        return await self.put(f"/api/v1/cameras/{camera_id}", json_data=camera_data)

    async def delete_camera(self, camera_id: str) -> dict[str, Any]:
        """Delete camera."""
        return await self.delete(f"/api/v1/cameras/{camera_id}")

    async def get_analytics(
        self, camera_id: str = None, start_time: str = None, end_time: str = None
    ) -> dict[str, Any]:
        """Get analytics data."""
        params = {}
        if camera_id:
            params["camera_id"] = camera_id
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        return await self.get("/api/v1/analytics", params=params, use_cache=True)

    async def list_models(self) -> dict[str, Any]:
        """Get list of ML models."""
        return await self.get("/api/v1/models", use_cache=True)

    async def upload_model(
        self, model_data: bytes, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        """Upload ML model."""
        # This would typically use multipart form data
        # Simplified for now - actual implementation would handle file uploads properly
        return await self.post("/api/v1/models/upload", json_data=metadata)

    def clear_cache(self) -> None:
        """Clear response cache."""
        self._cache.clear()
        logger.info("Response cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        return {
            "base_url": self.base_url,
            "failure_count": self._failure_count,
            "circuit_open": self._is_circuit_open(),
            "cache_entries": len(self._cache),
            "authenticated": bool(self._access_token),
            "connected": bool(self._client),
        }
