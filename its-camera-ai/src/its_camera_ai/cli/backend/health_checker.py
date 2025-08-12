"""Comprehensive health checking system for backend services.

Provides detailed health monitoring, service dependency checking,
and system-wide health assessment capabilities.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil

from ...core.config import Settings, get_settings
from ...core.logging import get_logger
from .api_client import APIClient
from .database_manager import CLIDatabaseManager
from .service_discovery import ServiceDiscovery, ServiceStatus

logger = get_logger(__name__)


class HealthLevel(Enum):
    """Health check severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


class ComponentStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""
    component: str
    status: ComponentStatus
    level: HealthLevel
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    response_time_ms: float | None = None

    def is_healthy(self) -> bool:
        """Check if the component is healthy."""
        return self.status in (ComponentStatus.HEALTHY, ComponentStatus.DEGRADED)


@dataclass
class SystemHealth:
    """Overall system health assessment."""
    overall_status: ComponentStatus
    components: dict[str, HealthCheckResult] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def add_component(self, result: HealthCheckResult) -> None:
        """Add component health result."""
        self.components[result.component] = result
        self._update_overall_status()

    def _update_overall_status(self) -> None:
        """Update overall status based on components."""
        if not self.components:
            self.overall_status = ComponentStatus.UNKNOWN
            return

        statuses = [comp.status for comp in self.components.values()]

        if all(s == ComponentStatus.HEALTHY for s in statuses):
            self.overall_status = ComponentStatus.HEALTHY
        elif any(s == ComponentStatus.UNHEALTHY for s in statuses):
            self.overall_status = ComponentStatus.UNHEALTHY
        elif any(s == ComponentStatus.DEGRADED for s in statuses):
            self.overall_status = ComponentStatus.DEGRADED
        else:
            self.overall_status = ComponentStatus.UNKNOWN


class HealthChecker:
    """Comprehensive health checking system.
    
    Features:
    - Service health monitoring
    - Database connectivity checks
    - Resource utilization monitoring
    - Dependency validation
    - Performance benchmarking
    - Custom health check plugins
    """

    def __init__(self, settings: Settings = None):
        """Initialize health checker.
        
        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()

        # Initialize components
        self.service_discovery = ServiceDiscovery(self.settings)
        self.db_manager = CLIDatabaseManager(self.settings)
        self.api_client = APIClient(settings=self.settings)

        # Health check registry
        self._health_checks: dict[str, Callable] = {}
        self._register_default_checks()

        # Performance thresholds
        self.thresholds = {
            "response_time_ms": {
                "good": 100,
                "acceptable": 500,
                "poor": 1000
            },
            "cpu_usage_percent": {
                "good": 70,
                "acceptable": 85,
                "poor": 95
            },
            "memory_usage_percent": {
                "good": 70,
                "acceptable": 85,
                "poor": 95
            },
            "disk_usage_percent": {
                "good": 70,
                "acceptable": 85,
                "poor": 95
            }
        }

        logger.info("Health checker initialized")

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        self._health_checks.update({
            "api_service": self._check_api_service,
            "database": self._check_database,
            "redis": self._check_redis,
            "storage": self._check_storage,
            "system_resources": self._check_system_resources,
            "disk_space": self._check_disk_space,
            "network_connectivity": self._check_network_connectivity,
            "service_dependencies": self._check_service_dependencies
        })

    def register_health_check(self, name: str, check_func: Callable) -> None:
        """Register custom health check.
        
        Args:
            name: Name of the health check
            check_func: Async function that returns HealthCheckResult
        """
        self._health_checks[name] = check_func
        logger.info(f"Registered custom health check: {name}")

    async def check_component(self, component: str) -> HealthCheckResult:
        """Check health of a specific component.
        
        Args:
            component: Name of the component to check
            
        Returns:
            Health check result
        """
        if component not in self._health_checks:
            return HealthCheckResult(
                component=component,
                status=ComponentStatus.UNKNOWN,
                level=HealthLevel.WARNING,
                message=f"No health check registered for {component}"
            )

        try:
            start_time = time.time()
            result = await self._health_checks[component]()
            end_time = time.time()

            result.response_time_ms = (end_time - start_time) * 1000
            return result

        except Exception as e:
            logger.error(f"Health check failed for {component}: {e}")
            return HealthCheckResult(
                component=component,
                status=ComponentStatus.UNHEALTHY,
                level=HealthLevel.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)}
            )

    async def check_all_components(
        self,
        components: list[str] | None = None,
        parallel: bool = True
    ) -> SystemHealth:
        """Check health of all or specified components.
        
        Args:
            components: List of specific components to check
            parallel: Whether to run checks in parallel
            
        Returns:
            System health assessment
        """
        components = components or list(self._health_checks.keys())
        system_health = SystemHealth(overall_status=ComponentStatus.UNKNOWN)

        if parallel:
            # Run checks in parallel
            tasks = [self.check_component(comp) for comp in components]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                if isinstance(result, HealthCheckResult):
                    system_health.add_component(result)
                else:
                    # Handle exceptions
                    error_result = HealthCheckResult(
                        component=components[i],
                        status=ComponentStatus.UNHEALTHY,
                        level=HealthLevel.CRITICAL,
                        message=f"Health check crashed: {str(result)}",
                        details={"error": str(result)}
                    )
                    system_health.add_component(error_result)
        else:
            # Run checks sequentially
            for component in components:
                result = await self.check_component(component)
                system_health.add_component(result)

        # Generate summary
        system_health.summary = self._generate_summary(system_health)
        return system_health

    def _generate_summary(self, system_health: SystemHealth) -> dict[str, Any]:
        """Generate health summary statistics."""
        components = system_health.components

        status_counts = {
            ComponentStatus.HEALTHY.value: 0,
            ComponentStatus.DEGRADED.value: 0,
            ComponentStatus.UNHEALTHY.value: 0,
            ComponentStatus.UNKNOWN.value: 0
        }

        total_response_time = 0
        response_time_count = 0

        for result in components.values():
            status_counts[result.status.value] += 1

            if result.response_time_ms is not None:
                total_response_time += result.response_time_ms
                response_time_count += 1

        avg_response_time = (
            total_response_time / response_time_count
            if response_time_count > 0 else None
        )

        return {
            "total_components": len(components),
            "healthy_count": status_counts[ComponentStatus.HEALTHY.value],
            "degraded_count": status_counts[ComponentStatus.DEGRADED.value],
            "unhealthy_count": status_counts[ComponentStatus.UNHEALTHY.value],
            "unknown_count": status_counts[ComponentStatus.UNKNOWN.value],
            "average_response_time_ms": avg_response_time,
            "health_score": self._calculate_health_score(status_counts, len(components))
        }

    def _calculate_health_score(self, status_counts: dict[str, int], total: int) -> float:
        """Calculate overall health score (0-100)."""
        if total == 0:
            return 0.0

        healthy = status_counts[ComponentStatus.HEALTHY.value]
        degraded = status_counts[ComponentStatus.DEGRADED.value]

        # Healthy = 100%, Degraded = 50%, others = 0%
        score = (healthy * 100 + degraded * 50) / total
        return round(score, 2)

    # Default health check implementations

    async def _check_api_service(self) -> HealthCheckResult:
        """Check API service health."""
        try:
            health_data = await self.api_client.get_health()

            if health_data.get("status") == "healthy":
                return HealthCheckResult(
                    component="api_service",
                    status=ComponentStatus.HEALTHY,
                    level=HealthLevel.SUCCESS,
                    message="API service is healthy",
                    details=health_data
                )
            else:
                return HealthCheckResult(
                    component="api_service",
                    status=ComponentStatus.DEGRADED,
                    level=HealthLevel.WARNING,
                    message="API service reports degraded status",
                    details=health_data
                )

        except Exception as e:
            return HealthCheckResult(
                component="api_service",
                status=ComponentStatus.UNHEALTHY,
                level=HealthLevel.CRITICAL,
                message=f"Cannot connect to API service: {str(e)}",
                details={"error": str(e)}
            )

    async def _check_database(self) -> HealthCheckResult:
        """Check database connectivity and performance."""
        try:
            connectivity = await self.db_manager.check_connectivity()

            if connectivity.get("connected"):
                response_time = connectivity.get("response_time_ms", 0)

                if response_time < self.thresholds["response_time_ms"]["good"]:
                    status = ComponentStatus.HEALTHY
                    level = HealthLevel.SUCCESS
                elif response_time < self.thresholds["response_time_ms"]["acceptable"]:
                    status = ComponentStatus.DEGRADED
                    level = HealthLevel.WARNING
                else:
                    status = ComponentStatus.UNHEALTHY
                    level = HealthLevel.CRITICAL

                return HealthCheckResult(
                    component="database",
                    status=status,
                    level=level,
                    message=f"Database response time: {response_time:.2f}ms",
                    details=connectivity
                )
            else:
                return HealthCheckResult(
                    component="database",
                    status=ComponentStatus.UNHEALTHY,
                    level=HealthLevel.CRITICAL,
                    message="Database connection failed",
                    details=connectivity
                )

        except Exception as e:
            return HealthCheckResult(
                component="database",
                status=ComponentStatus.UNHEALTHY,
                level=HealthLevel.CRITICAL,
                message=f"Database health check failed: {str(e)}",
                details={"error": str(e)}
            )

    async def _check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        try:
            # Get Redis service from discovery
            redis_service = self.service_discovery.registry.get_service("redis")

            if not redis_service:
                return HealthCheckResult(
                    component="redis",
                    status=ComponentStatus.UNKNOWN,
                    level=HealthLevel.WARNING,
                    message="Redis service not registered"
                )

            await self.service_discovery.check_service_health(redis_service)

            if redis_service.status == ServiceStatus.HEALTHY:
                return HealthCheckResult(
                    component="redis",
                    status=ComponentStatus.HEALTHY,
                    level=HealthLevel.SUCCESS,
                    message="Redis is healthy",
                    details={
                        "response_time_ms": redis_service.response_time
                    }
                )
            else:
                return HealthCheckResult(
                    component="redis",
                    status=ComponentStatus.UNHEALTHY,
                    level=HealthLevel.CRITICAL,
                    message=f"Redis is {redis_service.status.value}",
                    details={
                        "error": redis_service.error_message
                    }
                )

        except Exception as e:
            return HealthCheckResult(
                component="redis",
                status=ComponentStatus.UNHEALTHY,
                level=HealthLevel.CRITICAL,
                message=f"Redis health check failed: {str(e)}",
                details={"error": str(e)}
            )

    async def _check_storage(self) -> HealthCheckResult:
        """Check MinIO storage service."""
        try:
            storage_service = self.service_discovery.registry.get_service("storage")

            if not storage_service:
                return HealthCheckResult(
                    component="storage",
                    status=ComponentStatus.UNKNOWN,
                    level=HealthLevel.WARNING,
                    message="Storage service not registered"
                )

            await self.service_discovery.check_service_health(storage_service)

            if storage_service.status == ServiceStatus.HEALTHY:
                return HealthCheckResult(
                    component="storage",
                    status=ComponentStatus.HEALTHY,
                    level=HealthLevel.SUCCESS,
                    message="Storage service is healthy",
                    details={
                        "response_time_ms": storage_service.response_time
                    }
                )
            else:
                return HealthCheckResult(
                    component="storage",
                    status=ComponentStatus.UNHEALTHY,
                    level=HealthLevel.CRITICAL,
                    message=f"Storage service is {storage_service.status.value}",
                    details={
                        "error": storage_service.error_message
                    }
                )

        except Exception as e:
            return HealthCheckResult(
                component="storage",
                status=ComponentStatus.UNHEALTHY,
                level=HealthLevel.CRITICAL,
                message=f"Storage health check failed: {str(e)}",
                details={"error": str(e)}
            )

    async def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource utilization."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Determine status based on resource usage
            cpu_status = self._get_resource_status(cpu_percent, "cpu_usage_percent")
            memory_status = self._get_resource_status(memory_percent, "memory_usage_percent")

            # Overall status is the worst of individual components
            if ComponentStatus.UNHEALTHY in (cpu_status, memory_status):
                status = ComponentStatus.UNHEALTHY
                level = HealthLevel.CRITICAL
            elif ComponentStatus.DEGRADED in (cpu_status, memory_status):
                status = ComponentStatus.DEGRADED
                level = HealthLevel.WARNING
            else:
                status = ComponentStatus.HEALTHY
                level = HealthLevel.SUCCESS

            return HealthCheckResult(
                component="system_resources",
                status=status,
                level=level,
                message=f"CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%",
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "cpu_count": psutil.cpu_count()
                }
            )

        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status=ComponentStatus.UNKNOWN,
                level=HealthLevel.WARNING,
                message=f"Could not check system resources: {str(e)}",
                details={"error": str(e)}
            )

    async def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability."""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            status = self._get_resource_status(disk_percent, "disk_usage_percent")

            if status == ComponentStatus.UNHEALTHY:
                level = HealthLevel.CRITICAL
            elif status == ComponentStatus.DEGRADED:
                level = HealthLevel.WARNING
            else:
                level = HealthLevel.SUCCESS

            return HealthCheckResult(
                component="disk_space",
                status=status,
                level=level,
                message=f"Disk usage: {disk_percent:.1f}%",
                details={
                    "used_percent": disk_percent,
                    "free_gb": disk.free / (1024**3),
                    "used_gb": disk.used / (1024**3),
                    "total_gb": disk.total / (1024**3)
                }
            )

        except Exception as e:
            return HealthCheckResult(
                component="disk_space",
                status=ComponentStatus.UNKNOWN,
                level=HealthLevel.WARNING,
                message=f"Could not check disk space: {str(e)}",
                details={"error": str(e)}
            )

    async def _check_network_connectivity(self) -> HealthCheckResult:
        """Check network connectivity."""
        try:
            import socket

            # Test DNS resolution
            socket.gethostbyname('google.com')

            # Test basic network interface stats
            net_io = psutil.net_io_counters()

            return HealthCheckResult(
                component="network_connectivity",
                status=ComponentStatus.HEALTHY,
                level=HealthLevel.SUCCESS,
                message="Network connectivity is healthy",
                details={
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                    "errors_in": net_io.errin,
                    "errors_out": net_io.errout
                }
            )

        except Exception as e:
            return HealthCheckResult(
                component="network_connectivity",
                status=ComponentStatus.UNHEALTHY,
                level=HealthLevel.CRITICAL,
                message=f"Network connectivity issues: {str(e)}",
                details={"error": str(e)}
            )

    async def _check_service_dependencies(self) -> HealthCheckResult:
        """Check service dependencies and their health."""
        try:
            # Get status of all registered services
            services_status = await self.service_discovery.get_all_services_status()

            healthy_count = 0
            total_count = len(services_status)
            critical_services_down = []

            critical_services = {'api', 'database', 'redis'}

            for service_name, status_data in services_status.items():
                if status_data.get('is_healthy', False):
                    healthy_count += 1
                elif service_name in critical_services:
                    critical_services_down.append(service_name)

            if critical_services_down:
                return HealthCheckResult(
                    component="service_dependencies",
                    status=ComponentStatus.UNHEALTHY,
                    level=HealthLevel.CRITICAL,
                    message=f"Critical services down: {', '.join(critical_services_down)}",
                    details={
                        "total_services": total_count,
                        "healthy_services": healthy_count,
                        "critical_services_down": critical_services_down,
                        "services_status": services_status
                    }
                )
            elif healthy_count == total_count:
                return HealthCheckResult(
                    component="service_dependencies",
                    status=ComponentStatus.HEALTHY,
                    level=HealthLevel.SUCCESS,
                    message=f"All {total_count} services are healthy",
                    details={
                        "total_services": total_count,
                        "healthy_services": healthy_count
                    }
                )
            else:
                return HealthCheckResult(
                    component="service_dependencies",
                    status=ComponentStatus.DEGRADED,
                    level=HealthLevel.WARNING,
                    message=f"{healthy_count}/{total_count} services healthy",
                    details={
                        "total_services": total_count,
                        "healthy_services": healthy_count,
                        "services_status": services_status
                    }
                )

        except Exception as e:
            return HealthCheckResult(
                component="service_dependencies",
                status=ComponentStatus.UNKNOWN,
                level=HealthLevel.WARNING,
                message=f"Could not check service dependencies: {str(e)}",
                details={"error": str(e)}
            )

    def _get_resource_status(self, usage_percent: float, threshold_key: str) -> ComponentStatus:
        """Get resource status based on usage percentage."""
        thresholds = self.thresholds[threshold_key]

        if usage_percent < thresholds["good"]:
            return ComponentStatus.HEALTHY
        elif usage_percent < thresholds["acceptable"]:
            return ComponentStatus.DEGRADED
        else:
            return ComponentStatus.UNHEALTHY

    async def benchmark_component(self, component: str, iterations: int = 5) -> dict[str, Any]:
        """Benchmark a component's performance.
        
        Args:
            component: Component to benchmark
            iterations: Number of test iterations
            
        Returns:
            Benchmark results
        """
        if component not in self._health_checks:
            return {"error": f"Component {component} not found"}

        results = []

        for i in range(iterations):
            result = await self.check_component(component)
            results.append({
                "iteration": i + 1,
                "status": result.status.value,
                "response_time_ms": result.response_time_ms,
                "timestamp": result.timestamp
            })

        # Calculate statistics
        response_times = [r["response_time_ms"] for r in results if r["response_time_ms"]]

        if response_times:
            avg_time = sum(response_times) / len(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
        else:
            avg_time = min_time = max_time = None

        success_rate = len([r for r in results if r["status"] in ("healthy", "degraded")]) / iterations * 100

        return {
            "component": component,
            "iterations": iterations,
            "success_rate_percent": success_rate,
            "avg_response_time_ms": avg_time,
            "min_response_time_ms": min_time,
            "max_response_time_ms": max_time,
            "results": results
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.api_client.connect()
        await self.db_manager.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.api_client.close()
        await self.db_manager.close()
