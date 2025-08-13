"""
Enhanced rate limiting middleware with advanced features.

Provides:
- User-specific rate limiting
- IP-based rate limiting with subnet support
- Endpoint-specific rate limits
- Distributed rate limiting with Redis
- Adaptive rate limiting based on system load
- Rate limit headers (X-RateLimit-*)
- Exponential backoff for repeated violations
"""

import ipaddress
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import redis.asyncio as redis
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ...core.config import get_settings
from ...core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitRule:
    """Rate limit rule configuration."""
    requests_per_minute: int
    requests_per_hour: int = 0
    requests_per_day: int = 0
    burst_size: int = 10
    adaptive: bool = False
    backoff_multiplier: float = 2.0
    max_backoff_minutes: int = 60


@dataclass
class RateLimitStatus:
    """Current rate limit status."""
    allowed: bool
    limit: int
    remaining: int
    reset_time: int
    retry_after: int = 0
    backoff_until: int = 0


class EnhancedRateLimitMiddleware(BaseHTTPMiddleware):
    """Enhanced rate limiting middleware with advanced features."""

    def __init__(self, app, redis_client: redis.Redis | None = None, settings=None):
        super().__init__(app)
        self.redis = redis_client
        self.settings = settings or get_settings()

        # Default rate limit rules
        self.default_rules = {
            "general": RateLimitRule(
                requests_per_minute=100,
                requests_per_hour=1000,
                requests_per_day=10000,
                adaptive=True
            ),
            "auth": RateLimitRule(
                requests_per_minute=5,
                requests_per_hour=50,
                requests_per_day=500,
                burst_size=2,
                backoff_multiplier=3.0,
                max_backoff_minutes=30
            ),
            "upload": RateLimitRule(
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=1000,
                burst_size=5,
                adaptive=True
            ),
            "api_key": RateLimitRule(
                requests_per_minute=1000,
                requests_per_hour=10000,
                requests_per_day=100000,
                burst_size=50
            ),
            "premium": RateLimitRule(
                requests_per_minute=500,
                requests_per_hour=5000,
                requests_per_day=50000,
                burst_size=25,
                adaptive=True
            )
        }

        # Endpoint-specific rules
        self.endpoint_rules = {
            # Authentication endpoints
            "/api/v1/auth/login": "auth",
            "/api/v1/auth/register": "auth",
            "/api/v1/auth/refresh": "auth",
            "/api/v1/auth/mfa": "auth",
            "/api/v1/auth/password-reset": "auth",

            # Upload endpoints
            "/api/v1/cameras/upload": "upload",
            "/api/v1/models/upload": "upload",

            # High-frequency endpoints
            "/api/v1/health": RateLimitRule(requests_per_minute=1000, burst_size=100),
            "/metrics": RateLimitRule(requests_per_minute=500, burst_size=50),

            # Premium endpoints (would be set based on user tier)
            "/api/v1/analytics/real-time": "premium",
        }

        # Subnet configurations for IP-based limiting
        self.subnet_rules = {
            # Internal networks get higher limits
            "10.0.0.0/8": RateLimitRule(requests_per_minute=1000, burst_size=100),
            "172.16.0.0/12": RateLimitRule(requests_per_minute=1000, burst_size=100),
            "192.168.0.0/16": RateLimitRule(requests_per_minute=1000, burst_size=100),
        }

        # System load monitoring for adaptive limits
        self.adaptive_enabled = True
        self.load_threshold_low = 0.5
        self.load_threshold_high = 0.8
        self.last_load_check = 0
        self.current_load = 0.0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Apply enhanced rate limiting."""
        if not self.redis:
            logger.warning("Redis not available, skipping rate limiting")
            return await call_next(request)

        try:
            # Get client identifier and rule
            client_id, rule = await self._get_client_and_rule(request)

            # Check rate limits
            status_result = await self._check_rate_limits(client_id, request, rule)

            if not status_result.allowed:
                return self._create_rate_limit_response(status_result, rule)

            # Process request
            response = await call_next(request)

            # Add rate limit headers
            self._add_rate_limit_headers(response, status_result)

            return response

        except Exception as e:
            logger.error("Rate limiting error", error=str(e), path=request.url.path)
            # Continue without rate limiting on errors
            return await call_next(request)

    async def _get_client_and_rule(self, request: Request) -> tuple[str, RateLimitRule]:
        """Get client identifier and applicable rate limit rule."""
        # Check for API key first
        api_key = request.headers.get("x-api-key")
        if api_key:
            # TODO: Get user tier from API key service
            user_tier = await self._get_user_tier_from_api_key(api_key)
            rule = self.default_rules.get(user_tier, self.default_rules["api_key"])
            return f"api_key:{api_key}", rule

        # Check for authenticated user
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # TODO: Extract user ID from JWT token
            user_id = await self._get_user_id_from_token(auth_header[7:])
            if user_id:
                # TODO: Get user tier from database
                user_tier = await self._get_user_tier(user_id)
                rule = self.default_rules.get(user_tier, self.default_rules["general"])
                return f"user:{user_id}", rule

        # Fall back to IP-based limiting
        client_ip = self._get_client_ip(request)
        rule = self._get_ip_rule(client_ip, request)

        return f"ip:{client_ip}", rule

    def _get_ip_rule(self, client_ip: str, request: Request) -> RateLimitRule:
        """Get rate limit rule for IP address."""
        try:
            ip = ipaddress.ip_address(client_ip)

            # Check subnet rules
            for subnet_str, rule in self.subnet_rules.items():
                subnet = ipaddress.ip_network(subnet_str)
                if ip in subnet:
                    return rule

        except ValueError:
            # Invalid IP address
            pass

        # Check endpoint-specific rules
        endpoint = request.url.path
        if endpoint in self.endpoint_rules:
            rule_or_name = self.endpoint_rules[endpoint]
            if isinstance(rule_or_name, str):
                return self.default_rules[rule_or_name]
            return rule_or_name

        # Check endpoint patterns
        for pattern, rule_name in self._get_endpoint_patterns().items():
            if pattern in endpoint:
                return self.default_rules[rule_name]

        return self.default_rules["general"]

    def _get_endpoint_patterns(self) -> dict[str, str]:
        """Get endpoint patterns for rule matching."""
        return {
            "/auth/": "auth",
            "/upload": "upload",
            "/admin/": "premium",
        }

    async def _check_rate_limits(
        self, client_id: str, request: Request, rule: RateLimitRule
    ) -> RateLimitStatus:
        """Check if request is within rate limits."""
        current_time = int(time.time())

        # Apply adaptive limiting
        if rule.adaptive and self.adaptive_enabled:
            rule = await self._apply_adaptive_limits(rule)

        # Check for active backoff
        backoff_key = f"backoff:{client_id}"
        backoff_until = await self.redis.get(backoff_key)
        if backoff_until and int(backoff_until) > current_time:
            return RateLimitStatus(
                allowed=False,
                limit=rule.requests_per_minute,
                remaining=0,
                reset_time=current_time + 60,
                backoff_until=int(backoff_until),
                retry_after=int(backoff_until) - current_time
            )

        # Check minute limit (primary)
        minute_status = await self._check_time_window(
            client_id, "minute", 60, rule.requests_per_minute, rule.burst_size
        )

        if not minute_status.allowed:
            # Apply exponential backoff
            await self._apply_backoff(client_id, rule)
            return minute_status

        # Check hour limit if configured
        if rule.requests_per_hour > 0:
            hour_status = await self._check_time_window(
                client_id, "hour", 3600, rule.requests_per_hour, 0
            )
            if not hour_status.allowed:
                return hour_status

        # Check daily limit if configured
        if rule.requests_per_day > 0:
            day_status = await self._check_time_window(
                client_id, "day", 86400, rule.requests_per_day, 0
            )
            if not day_status.allowed:
                return day_status

        # All checks passed, increment counters
        await self._increment_counters(client_id)

        return minute_status

    async def _check_time_window(
        self, client_id: str, window: str, duration: int, limit: int, burst_size: int
    ) -> RateLimitStatus:
        """Check rate limit for a specific time window."""
        current_time = int(time.time())
        window_start = current_time - (current_time % duration)

        key = f"rate_limit:{client_id}:{window}:{window_start}"

        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()
        pipe.get(key)
        pipe.ttl(key)
        results = await pipe.execute()

        current_count = int(results[0]) if results[0] else 0
        results[1]

        # Calculate effective limit (base + burst)
        effective_limit = limit + burst_size

        # Check if limit exceeded
        if current_count >= effective_limit:
            reset_time = window_start + duration
            return RateLimitStatus(
                allowed=False,
                limit=limit,
                remaining=0,
                reset_time=reset_time,
                retry_after=max(1, reset_time - current_time)
            )

        # Calculate remaining requests
        remaining = max(0, effective_limit - current_count - 1)
        reset_time = window_start + duration

        return RateLimitStatus(
            allowed=True,
            limit=limit,
            remaining=remaining,
            reset_time=reset_time
        )

    async def _increment_counters(self, client_id: str) -> None:
        """Increment rate limit counters for all time windows."""
        current_time = int(time.time())

        # Calculate window starts
        minute_start = current_time - (current_time % 60)
        hour_start = current_time - (current_time % 3600)
        day_start = current_time - (current_time % 86400)

        # Use Redis pipeline for atomic operations
        pipe = self.redis.pipeline()

        # Increment minute counter
        minute_key = f"rate_limit:{client_id}:minute:{minute_start}"
        pipe.incr(minute_key)
        pipe.expire(minute_key, 60)

        # Increment hour counter
        hour_key = f"rate_limit:{client_id}:hour:{hour_start}"
        pipe.incr(hour_key)
        pipe.expire(hour_key, 3600)

        # Increment day counter
        day_key = f"rate_limit:{client_id}:day:{day_start}"
        pipe.incr(day_key)
        pipe.expire(day_key, 86400)

        await pipe.execute()

    async def _apply_adaptive_limits(self, rule: RateLimitRule) -> RateLimitRule:
        """Apply adaptive rate limiting based on system load."""
        current_time = time.time()

        # Update load info periodically
        if current_time - self.last_load_check > 10:  # Check every 10 seconds
            self.current_load = await self._get_system_load()
            self.last_load_check = current_time

        # Adjust limits based on load
        if self.current_load > self.load_threshold_high:
            # High load: reduce limits by 50%
            multiplier = 0.5
        elif self.current_load > self.load_threshold_low:
            # Medium load: reduce limits by 25%
            multiplier = 0.75
        else:
            # Low load: normal limits
            multiplier = 1.0

        # Create adjusted rule
        adjusted_rule = RateLimitRule(
            requests_per_minute=int(rule.requests_per_minute * multiplier),
            requests_per_hour=int(rule.requests_per_hour * multiplier) if rule.requests_per_hour else 0,
            requests_per_day=int(rule.requests_per_day * multiplier) if rule.requests_per_day else 0,
            burst_size=max(1, int(rule.burst_size * multiplier)),
            adaptive=rule.adaptive,
            backoff_multiplier=rule.backoff_multiplier,
            max_backoff_minutes=rule.max_backoff_minutes
        )

        return adjusted_rule

    async def _get_system_load(self) -> float:
        """Get current system load for adaptive limiting."""
        try:
            # Check Redis memory usage as a proxy for system load
            info = await self.redis.info("memory")
            used_memory = info.get("used_memory", 0)
            max_memory = info.get("maxmemory", 0)

            if max_memory > 0:
                memory_usage = used_memory / max_memory
                return min(1.0, memory_usage)

        except Exception as e:
            logger.warning("Failed to get system load", error=str(e))

        return 0.0  # Assume low load on error

    async def _apply_backoff(self, client_id: str, rule: RateLimitRule) -> None:
        """Apply exponential backoff for repeated violations."""
        backoff_key = f"backoff:{client_id}"
        violation_key = f"violations:{client_id}"

        # Get current violation count
        violations = await self.redis.get(violation_key)
        violations = int(violations) if violations else 0

        # Calculate backoff duration (exponential)
        backoff_seconds = min(
            60 * rule.max_backoff_minutes,
            int(60 * (rule.backoff_multiplier ** violations))
        )

        current_time = int(time.time())
        backoff_until = current_time + backoff_seconds

        # Set backoff and increment violations
        pipe = self.redis.pipeline()
        pipe.set(backoff_key, backoff_until, ex=backoff_seconds)
        pipe.incr(violation_key)
        pipe.expire(violation_key, 3600)  # Reset violations after 1 hour
        await pipe.execute()

        logger.warning(
            "Rate limit backoff applied",
            client_id=client_id,
            violations=violations + 1,
            backoff_seconds=backoff_seconds
        )

    def _create_rate_limit_response(
        self, status: RateLimitStatus, rule: RateLimitRule
    ) -> JSONResponse:
        """Create rate limit exceeded response."""
        content = {
            "error": "rate_limit_exceeded",
            "message": f"Rate limit exceeded. Limit: {status.limit} requests per minute",
            "limit": status.limit,
            "remaining": status.remaining,
            "reset_time": status.reset_time,
            "retry_after": status.retry_after
        }

        if status.backoff_until > 0:
            content["message"] = "Too many violations. Please wait before retrying."
            content["backoff_until"] = status.backoff_until

        headers = {
            "X-RateLimit-Limit": str(status.limit),
            "X-RateLimit-Remaining": str(status.remaining),
            "X-RateLimit-Reset": str(status.reset_time),
            "Retry-After": str(status.retry_after)
        }

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=content,
            headers=headers
        )

    def _add_rate_limit_headers(self, response: Response, status: RateLimitStatus) -> None:
        """Add rate limit headers to response."""
        response.headers["X-RateLimit-Limit"] = str(status.limit)
        response.headers["X-RateLimit-Remaining"] = str(status.remaining)
        response.headers["X-RateLimit-Reset"] = str(status.reset_time)

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers in order of preference
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP from the chain
            return forwarded_for.split(",")[0].strip()

        forwarded = request.headers.get("x-forwarded")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct connection
        return request.client.host if request.client else "unknown"

    async def _get_user_tier_from_api_key(self, api_key: str) -> str:
        """Get user tier from API key (placeholder)."""
        # TODO: Implement API key service integration
        return "api_key"

    async def _get_user_id_from_token(self, token: str) -> str | None:
        """Extract user ID from JWT token (placeholder)."""
        # TODO: Implement JWT token parsing
        return None

    async def _get_user_tier(self, user_id: str) -> str:
        """Get user tier from database (placeholder)."""
        # TODO: Implement database lookup
        return "general"

    # Administrative methods for rate limit management
    async def reset_rate_limit(self, client_id: str) -> bool:
        """Reset rate limits for a client."""
        try:
            current_time = int(time.time())

            # Calculate window starts
            minute_start = current_time - (current_time % 60)
            hour_start = current_time - (current_time % 3600)
            day_start = current_time - (current_time % 86400)

            # Delete rate limit keys
            keys_to_delete = [
                f"rate_limit:{client_id}:minute:{minute_start}",
                f"rate_limit:{client_id}:hour:{hour_start}",
                f"rate_limit:{client_id}:day:{day_start}",
                f"backoff:{client_id}",
                f"violations:{client_id}"
            ]

            deleted = await self.redis.delete(*keys_to_delete)

            logger.info("Rate limits reset", client_id=client_id, keys_deleted=deleted)
            return True

        except Exception as e:
            logger.error("Failed to reset rate limits", client_id=client_id, error=str(e))
            return False

    async def get_rate_limit_status(self, client_id: str) -> dict[str, Any]:
        """Get current rate limit status for a client."""
        try:
            current_time = int(time.time())

            # Get counters for all windows
            minute_start = current_time - (current_time % 60)
            hour_start = current_time - (current_time % 3600)
            day_start = current_time - (current_time % 86400)

            pipe = self.redis.pipeline()
            pipe.get(f"rate_limit:{client_id}:minute:{minute_start}")
            pipe.get(f"rate_limit:{client_id}:hour:{hour_start}")
            pipe.get(f"rate_limit:{client_id}:day:{day_start}")
            pipe.get(f"backoff:{client_id}")
            pipe.get(f"violations:{client_id}")

            results = await pipe.execute()

            return {
                "client_id": client_id,
                "minute_count": int(results[0]) if results[0] else 0,
                "hour_count": int(results[1]) if results[1] else 0,
                "day_count": int(results[2]) if results[2] else 0,
                "backoff_until": int(results[3]) if results[3] else 0,
                "violations": int(results[4]) if results[4] else 0,
                "current_time": current_time
            }

        except Exception as e:
            logger.error("Failed to get rate limit status", client_id=client_id, error=str(e))
            return {"error": str(e)}
