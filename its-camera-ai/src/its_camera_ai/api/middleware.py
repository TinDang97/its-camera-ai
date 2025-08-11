"""Custom middleware for FastAPI application.

Provides logging, metrics, security, and rate limiting middleware.
"""

import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from prometheus_client import Counter, Gauge, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..core.logging import get_logger

logger = get_logger(__name__)


# Prometheus metrics
REQUESTS_TOTAL = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

ACTIVE_CONNECTIONS = Gauge("http_active_connections", "Active HTTP connections")

RATE_LIMIT_HITS = Counter(
    "rate_limit_hits_total", "Total rate limit hits", ["ip_address"]
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
        )

        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration_seconds=duration,
            )

            return response

        except Exception as e:
            duration = time.time() - start_time

            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                duration_seconds=duration,
                error=str(e),
                exc_info=True,
            )
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for Prometheus metrics collection."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Track active connections
        ACTIVE_CONNECTIONS.inc()

        try:
            start_time = time.time()

            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Extract endpoint pattern
            endpoint = request.url.path
            if hasattr(request, "path_info"):
                endpoint = request.path_info

            # Record metrics
            REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=response.status_code,
            ).inc()

            REQUEST_DURATION.labels(method=request.method, endpoint=endpoint).observe(
                duration
            )

            return response

        finally:
            ACTIVE_CONNECTIONS.dec()


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and basic protection."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

        # Remove server header for security
        if "server" in response.headers:
            del response.headers["server"]

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware.

    For production use, consider Redis-based rate limiting.
    """

    def __init__(self, app: ASGIApp, calls: int = 100, period: int = 60) -> None:
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients: dict[str, dict[str, Any]] = defaultdict(dict)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        current_time = time.time()

        # Clean up old entries
        self._cleanup_old_entries(current_time)

        # Check rate limit
        client_data = self.clients[client_ip]

        # Initialize client data if needed
        if "requests" not in client_data:
            client_data["requests"] = []

        # Remove old requests outside the window
        client_data["requests"] = [
            req_time
            for req_time in client_data["requests"]
            if current_time - req_time < self.period
        ]

        # Check if rate limit exceeded
        if len(client_data["requests"]) >= self.calls:
            RATE_LIMIT_HITS.labels(ip_address=client_ip).inc()

            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                requests_count=len(client_data["requests"]),
                limit=self.calls,
                period=self.period,
            )

            from fastapi import HTTPException, status

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(self.period)},
            )

        # Add current request
        client_data["requests"].append(current_time)

        return await call_next(request)

    def _cleanup_old_entries(self, current_time: float) -> None:
        """Remove old client entries to prevent memory leaks."""
        clients_to_remove = []

        for client_ip, client_data in self.clients.items():
            if "requests" not in client_data:
                clients_to_remove.append(client_ip)
                continue

            # Remove old requests
            client_data["requests"] = [
                req_time
                for req_time in client_data["requests"]
                if current_time - req_time < self.period
            ]

            # Mark for removal if no recent requests
            if not client_data["requests"]:
                clients_to_remove.append(client_ip)

        # Remove old entries
        for client_ip in clients_to_remove:
            del self.clients[client_ip]
