"""Service discovery and health checking for backend services.

Provides automatic discovery of backend services, health monitoring,
and service endpoint management for CLI operations.
"""

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import aiohttp
import redis.asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError

from ...core.config import Settings, get_settings
from ...core.logging import get_logger

logger = get_logger(__name__)


class ServiceStatus(Enum):
    """Service health status enumeration."""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNREACHABLE = "unreachable"


class ServiceType(Enum):
    """Service type enumeration."""
    API = "api"
    DATABASE = "database"
    REDIS = "redis"
    QUEUE = "queue"
    STORAGE = "storage"
    INFERENCE = "inference"
    MONITORING = "monitoring"
    WEB = "web"


@dataclass
class ServiceEndpoint:
    """Service endpoint information."""
    name: str
    service_type: ServiceType
    host: str
    port: int
    protocol: str = "http"
    path: str = "/"
    health_path: str = "/health"

    # Health check configuration
    timeout: float = 5.0
    check_interval: int = 30  # seconds
    failure_threshold: int = 3
    recovery_threshold: int = 2

    # Runtime state
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_check: float | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    response_time: float | None = None
    error_message: str | None = None

    @property
    def url(self) -> str:
        """Get full service URL."""
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"

    @property
    def health_url(self) -> str:
        """Get health check URL."""
        return f"{self.protocol}://{self.host}:{self.port}{self.health_path}"

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status in (ServiceStatus.HEALTHY, ServiceStatus.DEGRADED)


@dataclass
class ServiceRegistry:
    """Registry of discovered services."""
    services: dict[str, ServiceEndpoint] = field(default_factory=dict)
    last_discovery: float | None = None
    discovery_errors: list[str] = field(default_factory=list)

    def add_service(self, service: ServiceEndpoint) -> None:
        """Add service to registry."""
        self.services[service.name] = service
        logger.debug(f"Added service {service.name} to registry")

    def get_service(self, name: str) -> ServiceEndpoint | None:
        """Get service by name."""
        return self.services.get(name)

    def get_services_by_type(self, service_type: ServiceType) -> list[ServiceEndpoint]:
        """Get services by type."""
        return [svc for svc in self.services.values() if svc.service_type == service_type]

    def get_healthy_services(self) -> list[ServiceEndpoint]:
        """Get only healthy services."""
        return [svc for svc in self.services.values() if svc.is_healthy()]

    def remove_service(self, name: str) -> bool:
        """Remove service from registry."""
        if name in self.services:
            del self.services[name]
            logger.debug(f"Removed service {name} from registry")
            return True
        return False


class ServiceDiscovery:
    """Service discovery and health monitoring system.

    Features:
    - Automatic service discovery
    - Continuous health monitoring
    - Service endpoint management
    - Redis-based service registration
    - Load balancing support
    """

    def __init__(self, settings: Settings = None):
        """Initialize service discovery.

        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self.registry = ServiceRegistry()

        # Redis connection for service registration
        self._redis: aioredis.Redis | None = None
        self._redis_connected = False

        # Health monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_active = False

        # Default services from configuration
        self._setup_default_services()

        logger.info("Service discovery initialized")

    def _setup_default_services(self) -> None:
        """Setup default services from configuration."""
        # API service
        api_service = ServiceEndpoint(
            name="api",
            service_type=ServiceType.API,
            host=self.settings.api_host,
            port=self.settings.api_port,
            protocol="http",
            path="/api/v1",
            health_path="/health"
        )
        self.registry.add_service(api_service)

        # Database service
        db_config = self.settings.database
        if db_config.url:
            # Parse database URL for host/port
            import urllib.parse
            parsed = urllib.parse.urlparse(db_config.url)

            if parsed.hostname and parsed.port:
                db_service = ServiceEndpoint(
                    name="database",
                    service_type=ServiceType.DATABASE,
                    host=parsed.hostname,
                    port=parsed.port,
                    protocol="postgresql",
                    timeout=10.0
                )
                self.registry.add_service(db_service)

        # Redis service
        redis_config = self.settings.redis
        if redis_config.url:
            parsed = urllib.parse.urlparse(redis_config.url)
            if parsed.hostname and parsed.port:
                redis_service = ServiceEndpoint(
                    name="redis",
                    service_type=ServiceType.REDIS,
                    host=parsed.hostname,
                    port=parsed.port,
                    protocol="redis"
                )
                self.registry.add_service(redis_service)

        # MinIO storage service
        minio_config = self.settings.minio
        if minio_config.endpoint:
            host, port = minio_config.endpoint.split(":")
            port = int(port)

            storage_service = ServiceEndpoint(
                name="storage",
                service_type=ServiceType.STORAGE,
                host=host,
                port=port,
                protocol="https" if minio_config.secure else "http",
                health_path="/minio/health/ready"
            )
            self.registry.add_service(storage_service)

        # Monitoring service (Prometheus)
        if self.settings.monitoring.enable_metrics:
            monitoring_service = ServiceEndpoint(
                name="monitoring",
                service_type=ServiceType.MONITORING,
                host="localhost",
                port=self.settings.monitoring.prometheus_port,
                protocol="http",
                health_path="/-/healthy"
            )
            self.registry.add_service(monitoring_service)

    async def connect_redis(self) -> None:
        """Connect to Redis for service registration."""
        try:
            self._redis = aioredis.from_url(
                self.settings.redis.url,
                max_connections=5,
                retry_on_timeout=True,
                socket_timeout=5
            )

            # Test connection
            await self._redis.ping()
            self._redis_connected = True

            logger.info("Connected to Redis for service discovery")

        except Exception as e:
            logger.warning(f"Failed to connect to Redis for service discovery: {e}")
            self._redis_connected = False

    async def disconnect_redis(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._redis_connected = False
            logger.info("Disconnected from Redis")

    async def register_service(
        self,
        service: ServiceEndpoint,
        ttl: int = 60
    ) -> bool:
        """Register service in Redis.

        Args:
            service: Service to register
            ttl: Time to live in seconds

        Returns:
            True if registration successful
        """
        if not self._redis_connected:
            await self.connect_redis()

        if not self._redis:
            return False

        try:
            service_key = f"services:{service.name}"
            service_data = {
                "name": service.name,
                "type": service.service_type.value,
                "host": service.host,
                "port": service.port,
                "protocol": service.protocol,
                "url": service.url,
                "health_url": service.health_url,
                "registered_at": time.time()
            }

            await self._redis.setex(
                service_key,
                ttl,
                json.dumps(service_data)
            )

            logger.debug(f"Registered service {service.name} in Redis")
            return True

        except Exception as e:
            logger.error(f"Failed to register service {service.name}: {e}")
            return False

    async def discover_services(self) -> int:
        """Discover services from Redis registry.

        Returns:
            Number of services discovered
        """
        if not self._redis_connected:
            await self.connect_redis()

        if not self._redis:
            logger.warning("Redis not available for service discovery")
            return 0

        try:
            # Get all service keys
            service_keys = await self._redis.keys("services:*")
            discovered_count = 0

            for key in service_keys:
                try:
                    service_data = await self._redis.get(key)
                    if service_data:
                        data = json.loads(service_data)

                        service = ServiceEndpoint(
                            name=data["name"],
                            service_type=ServiceType(data["type"]),
                            host=data["host"],
                            port=data["port"],
                            protocol=data["protocol"]
                        )

                        self.registry.add_service(service)
                        discovered_count += 1

                except Exception as e:
                    logger.warning(f"Failed to parse service data from {key}: {e}")

            self.registry.last_discovery = time.time()
            logger.info(f"Discovered {discovered_count} services from Redis")
            return discovered_count

        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            self.registry.discovery_errors.append(str(e))
            return 0

    async def check_service_health(self, service: ServiceEndpoint) -> None:
        """Check health of a single service.

        Args:
            service: Service to check
        """
        start_time = time.time()

        try:
            if service.service_type == ServiceType.DATABASE:
                await self._check_database_health(service)
            elif service.service_type == ServiceType.REDIS:
                await self._check_redis_health(service)
            else:
                await self._check_http_health(service)

            # Record success
            service.consecutive_failures = 0
            service.consecutive_successes += 1
            service.response_time = (time.time() - start_time) * 1000  # milliseconds
            service.error_message = None

            # Determine status based on response time
            if service.response_time < 100:
                service.status = ServiceStatus.HEALTHY
            elif service.response_time < 1000:
                service.status = ServiceStatus.DEGRADED
            else:
                service.status = ServiceStatus.UNHEALTHY

        except Exception as e:
            # Record failure
            service.consecutive_successes = 0
            service.consecutive_failures += 1
            service.error_message = str(e)
            service.response_time = None

            # Determine status based on failure count
            if service.consecutive_failures >= service.failure_threshold:
                service.status = ServiceStatus.UNREACHABLE
            else:
                service.status = ServiceStatus.UNHEALTHY

            logger.debug(f"Health check failed for {service.name}: {e}")

        service.last_check = time.time()

    async def _check_http_health(self, service: ServiceEndpoint) -> None:
        """Check HTTP service health."""
        timeout = aiohttp.ClientTimeout(total=service.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(service.health_url) as response:
                if response.status >= 400:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")

    async def _check_database_health(self, service: ServiceEndpoint) -> None:
        """Check database service health."""
        # This would use a database connection to check health
        # For now, we'll use a simple TCP connection check
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(service.timeout)

        try:
            result = sock.connect_ex((service.host, service.port))
            if result != 0:
                raise Exception(f"Cannot connect to {service.host}:{service.port}")
        finally:
            sock.close()

    async def _check_redis_health(self, service: ServiceEndpoint) -> None:
        """Check Redis service health."""
        try:
            redis_url = f"redis://{service.host}:{service.port}"
            redis_client = aioredis.from_url(
                redis_url,
                socket_timeout=service.timeout
            )

            await redis_client.ping()
            await redis_client.close()

        except RedisConnectionError as e:
            raise Exception(f"Redis connection failed: {e}")

    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring_active:
            logger.warning("Monitoring is already active")
            return

        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started service health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._monitoring_active = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task
            self._monitoring_task = None

        logger.info("Stopped service health monitoring")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Check all services
                tasks = []
                for service in self.registry.services.values():
                    task = asyncio.create_task(self.check_service_health(service))
                    tasks.append(task)

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                # Wait before next check cycle
                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay on error

    async def get_service_status(self, service_name: str) -> dict[str, Any]:
        """Get detailed status for a service.

        Args:
            service_name: Name of the service

        Returns:
            Dictionary with service status information
        """
        service = self.registry.get_service(service_name)
        if not service:
            return {"error": f"Service {service_name} not found"}

        return {
            "name": service.name,
            "type": service.service_type.value,
            "status": service.status.value,
            "url": service.url,
            "health_url": service.health_url,
            "response_time_ms": service.response_time,
            "last_check": service.last_check,
            "consecutive_failures": service.consecutive_failures,
            "consecutive_successes": service.consecutive_successes,
            "error_message": service.error_message,
            "is_healthy": service.is_healthy()
        }

    async def get_all_services_status(self) -> dict[str, dict[str, Any]]:
        """Get status for all services.

        Returns:
            Dictionary mapping service names to status information
        """
        status = {}

        for service_name in self.registry.services:
            status[service_name] = await self.get_service_status(service_name)

        return status

    def get_healthy_service(self, service_type: ServiceType) -> ServiceEndpoint | None:
        """Get a healthy service of the specified type.

        Args:
            service_type: Type of service to find

        Returns:
            Healthy service endpoint or None
        """
        services = self.registry.get_services_by_type(service_type)
        healthy_services = [svc for svc in services if svc.is_healthy()]

        if healthy_services:
            # Return the one with best response time
            return min(healthy_services, key=lambda s: s.response_time or float('inf'))

        return None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.discover_services()
        await self.start_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()
        await self.disconnect_redis()
