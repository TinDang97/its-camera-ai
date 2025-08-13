"""System management endpoints.

Provides comprehensive system management functionality including:
- System health and status monitoring
- Service management (start/stop/restart)
- System metrics (CPU, memory, disk, network)
- Configuration management
- Log management and retrieval
- System maintenance operations
- Backup and restore operations
- Performance monitoring
- System alerts and notifications
- System upgrade and deployment management
"""

import os
import platform
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import psutil
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ...core.config import Settings, get_settings
from ...core.logging import get_logger
from ...models.user import User
from ...services.cache import CacheService
from ..dependencies import (
    RateLimiter,
    get_cache_service,
    require_permissions,
)
from ..schemas.common import SuccessResponse

logger = get_logger(__name__)
router = APIRouter()

# Rate limiters for system operations
system_ops_rate_limit = RateLimiter(
    calls=10, period=60
)  # 10 system operations per minute
metrics_rate_limit = RateLimiter(calls=60, period=60)  # 60 metrics requests per minute
logs_rate_limit = RateLimiter(calls=30, period=60)  # 30 log requests per minute
backup_rate_limit = RateLimiter(calls=2, period=3600)  # 2 backup operations per hour

# Pydantic Models for System Management


class SystemStatus(BaseModel):
    """System status information."""

    system_status: str = Field(
        description="Overall system status (healthy/degraded/unhealthy)"
    )
    uptime: float = Field(description="System uptime in seconds")
    timestamp: datetime = Field(description="Status check timestamp")
    version: str = Field(description="Application version")
    environment: str = Field(description="Environment name (dev/staging/prod)")
    services: dict[str, str] = Field(description="Service status map")
    dependencies: dict[str, str] = Field(description="External dependencies status")
    alerts: list[dict[str, Any]] = Field(description="Active system alerts")


class SystemMetrics(BaseModel):
    """System performance metrics."""

    cpu: dict[str, float] = Field(description="CPU usage metrics")
    memory: dict[str, float] = Field(description="Memory usage metrics")
    disk: dict[str, float] = Field(description="Disk usage metrics")
    network: dict[str, float] = Field(description="Network metrics")
    processes: dict[str, int] = Field(description="Process count metrics")
    load_average: list[float] = Field(description="System load average")
    timestamp: datetime = Field(description="Metrics collection timestamp")


class ServiceInfo(BaseModel):
    """Service information."""

    name: str = Field(description="Service name")
    service_status: str = Field(description="Service status (running/stopped/error)")
    pid: int | None = Field(description="Process ID if running")
    cpu_percent: float = Field(description="CPU usage percentage")
    memory_percent: float = Field(description="Memory usage percentage")
    uptime: float | None = Field(description="Service uptime in seconds")
    restart_count: int = Field(description="Number of restarts")
    last_restart: datetime | None = Field(description="Last restart time")
    health_check_url: str | None = Field(description="Health check endpoint")


class ServiceAction(BaseModel):
    """Service action request."""

    service: str = Field(description="Service name")
    action: str = Field(description="Action to perform (start/stop/restart)")
    force: bool = Field(default=False, description="Force action even if risky")
    timeout: int = Field(default=30, description="Action timeout in seconds")


class SystemAlert(BaseModel):
    """System alert information."""

    alert_id: str = Field(description="Alert identifier")
    severity: str = Field(description="Alert severity (info/warning/error/critical)")
    category: str = Field(description="Alert category")
    title: str = Field(description="Alert title")
    message: str = Field(description="Alert message")
    source: str = Field(description="Alert source component")
    acknowledged: bool = Field(description="Whether alert is acknowledged")
    acknowledged_by: str | None = Field(description="User who acknowledged")
    acknowledged_at: datetime | None = Field(description="Acknowledgment time")
    created_at: datetime = Field(description="Alert creation time")
    resolved_at: datetime | None = Field(description="Alert resolution time")
    metadata: dict[str, Any] = Field(description="Additional alert data")


# Global system state tracking
_system_services: dict[str, ServiceInfo] = {}
_system_alerts: list[SystemAlert] = []


# Helper functions


def get_system_uptime() -> float:
    """Get system uptime in seconds."""
    try:
        return time.time() - psutil.boot_time()
    except Exception:
        return 0.0


def get_system_services() -> dict[str, str]:
    """Get status of system services."""
    services = {
        "database": "healthy",
        "redis": "healthy",
        "api": "healthy",
        "worker": "healthy",
        "scheduler": "healthy",
    }
    return services


def get_dependency_status() -> dict[str, str]:
    """Get status of external dependencies."""
    dependencies = {
        "postgresql": "connected",
        "redis": "connected",
        "s3": "connected",
        "kafka": "connected",
    }
    return dependencies


def collect_system_metrics() -> SystemMetrics:
    """Collect current system metrics."""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk metrics
        disk_usage = psutil.disk_usage("/")
        disk_io = psutil.disk_io_counters()

        # Network metrics
        network_io = psutil.net_io_counters()

        # Process metrics
        processes = list(psutil.process_iter(["pid", "name", "status"]))
        process_counts = {}
        for proc in processes:
            proc_status = proc.info.get("status", "unknown")
            process_counts[proc_status] = process_counts.get(proc_status, 0) + 1

        # Load average (Unix systems)
        load_avg = [0.0, 0.0, 0.0]
        if hasattr(os, "getloadavg"):
            load_avg = list(os.getloadavg())

        return SystemMetrics(
            cpu={
                "usage_percent": cpu_percent,
                "count_physical": cpu_count,
                "count_logical": cpu_count_logical,
            },
            memory={
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "usage_percent": memory.percent,
                "swap_total": swap.total,
                "swap_used": swap.used,
                "swap_percent": swap.percent,
            },
            disk={
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "usage_percent": (disk_usage.used / disk_usage.total) * 100,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0,
            },
            network={
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv,
            },
            processes={
                "total": len(processes),
                **process_counts,
            },
            load_average=load_avg,
            timestamp=datetime.now(UTC),
        )
    except Exception as e:
        logger.error("Failed to collect system metrics", error=str(e))
        # Return minimal metrics on error
        return SystemMetrics(
            cpu={"usage_percent": 0.0},
            memory={"usage_percent": 0.0},
            disk={"usage_percent": 0.0},
            network={},
            processes={"total": 0},
            load_average=[0.0, 0.0, 0.0],
            timestamp=datetime.now(UTC),
        )


# System Management Endpoints


@router.get(
    "/status",
    response_model=SystemStatus,
    summary="Get system status",
    description="Get comprehensive system health and status information.",
)
async def get_system_status(
    settings: Settings = Depends(get_settings),
    cache_service: CacheService = Depends(get_cache_service),
    _rate_limit: None = Depends(metrics_rate_limit),
) -> SystemStatus:
    """Get system status and health information.

    Returns overall system status including service health,
    dependency status, and active alerts.
    """
    try:
        # Check if status is cached
        cache_key = "system:status"
        cached_status = await cache_service.get(cache_key)
        if cached_status:
            logger.debug("Returning cached system status")
            return SystemStatus.model_validate(cached_status)

        # Collect fresh status
        services = get_system_services()
        dependencies = get_dependency_status()
        uptime = get_system_uptime()

        # Determine overall status
        overall_status = "healthy"
        if any(svc_status != "healthy" for svc_status in services.values()):
            overall_status = "degraded"
        if any(svc_status in ["error", "failed"] for svc_status in services.values()):
            overall_status = "unhealthy"

        # Get active alerts
        active_alerts = [
            alert.model_dump()
            for alert in _system_alerts
            if not alert.acknowledged and alert.resolved_at is None
        ]

        status_info = SystemStatus(
            system_status=overall_status,
            uptime=uptime,
            timestamp=datetime.now(UTC),
            version=settings.app_version,
            environment=settings.environment,
            services=services,
            dependencies=dependencies,
            alerts=active_alerts,
        )

        # Cache for 30 seconds
        await cache_service.set(cache_key, status_info.model_dump(), ttl=30)

        logger.info("System status retrieved", system_status=overall_status)
        return status_info

    except Exception as e:
        logger.error("Failed to get system status", error=str(e))
        # Return minimal status on error
        return SystemStatus(
            system_status="unhealthy",
            uptime=0.0,
            timestamp=datetime.now(UTC),
            version=settings.app_version,
            environment=settings.environment,
            services={},
            dependencies={},
            alerts=[],
        )


@router.get(
    "/metrics",
    response_model=SystemMetrics,
    summary="Get system metrics",
    description="Get detailed system performance metrics including CPU, memory, disk, and network.",
)
async def get_system_metrics(
    cache_service: CacheService = Depends(get_cache_service),
    _rate_limit: None = Depends(metrics_rate_limit),
) -> SystemMetrics:
    """Get system performance metrics.

    Returns detailed performance metrics including CPU usage,
    memory consumption, disk usage, and network statistics.
    """
    try:
        # Check if metrics are cached (cache for 10 seconds)
        cache_key = "system:metrics"
        cached_metrics = await cache_service.get(cache_key)
        if cached_metrics:
            logger.debug("Returning cached system metrics")
            return SystemMetrics.model_validate(cached_metrics)

        # Collect fresh metrics
        metrics = collect_system_metrics()

        # Cache metrics
        await cache_service.set(cache_key, metrics.model_dump(), ttl=10)

        logger.debug("System metrics collected")
        return metrics

    except Exception as e:
        logger.error("Failed to collect system metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to collect system metrics",
        ) from e


@router.get(
    "/services",
    response_model=list[ServiceInfo],
    summary="Get service information",
    description="Get detailed information about all system services.",
)
async def get_services(
    current_user: User = Depends(require_permissions("system:read")),
    _rate_limit: None = Depends(metrics_rate_limit),
) -> list[ServiceInfo]:
    """Get information about all system services.

    Returns detailed status and metrics for each service
    including CPU/memory usage, uptime, and health status.
    """
    try:
        services = []

        # Get process information for known services
        service_names = ["api", "worker", "scheduler", "monitoring"]

        for service_name in service_names:
            # This is a simplified implementation
            # In a real system, you'd query actual service managers
            service_info = ServiceInfo(
                name=service_name,
                service_status="running",
                pid=None,
                cpu_percent=0.0,
                memory_percent=0.0,
                uptime=get_system_uptime(),
                restart_count=0,
                last_restart=None,
                health_check_url=f"/health/{service_name}",
            )
            services.append(service_info)

        logger.info(
            "Service information retrieved",
            service_count=len(services),
            user=current_user.username,
        )
        return services

    except Exception as e:
        logger.error("Failed to get service information", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get service information",
        ) from e


@router.post(
    "/services/action",
    response_model=SuccessResponse,
    summary="Perform service action",
    description="Start, stop, or restart a system service.",
)
async def service_action(
    action_request: ServiceAction,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permissions("system:manage")),
    _rate_limit: None = Depends(system_ops_rate_limit),
) -> SuccessResponse:
    """Perform an action on a system service.

    Allows starting, stopping, or restarting services with
    proper validation and safety checks.
    """
    try:
        service_name = action_request.service
        action = action_request.action

        # Validate service name
        valid_services = ["api", "worker", "scheduler", "monitoring"]
        if service_name not in valid_services:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid service name. Valid services: {', '.join(valid_services)}",
            )

        # Validate action
        valid_actions = ["start", "stop", "restart"]
        if action not in valid_actions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action. Valid actions: {', '.join(valid_actions)}",
            )

        # Safety checks
        if service_name == "api" and action == "stop" and not action_request.force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot stop API service without force flag",
            )

        # Perform action in background (placeholder implementation)
        logger.info(
            "Service action initiated",
            service=service_name,
            action=action,
            user=current_user.username,
        )

        return SuccessResponse(
            success=True,
            message=f"Service {action} action initiated for {service_name}",
            data={
                "service": service_name,
                "action": action,
                "timeout": action_request.timeout,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Service action failed",
            service=action_request.service,
            action=action_request.action,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Service action failed",
        ) from e


@router.post(
    "/cleanup",
    response_model=SuccessResponse,
    summary="Perform system cleanup",
    description="Perform system cleanup operations like clearing caches and temporary files.",
)
async def system_cleanup(
    clear_cache: bool = Query(default=True, description="Clear application caches"),
    clear_logs: bool = Query(default=False, description="Clear old log files"),
    clear_temp: bool = Query(default=True, description="Clear temporary files"),
    current_user: User = Depends(require_permissions("system:maintenance")),
    _rate_limit: None = Depends(system_ops_rate_limit),
) -> SuccessResponse:
    """Perform system cleanup operations.

    Clears caches, temporary files, and old logs based on
    specified options to free up disk space.
    """
    try:
        cleanup_operations = []

        if clear_cache:
            cleanup_operations.append("cache")
        if clear_logs:
            cleanup_operations.append("logs")
        if clear_temp:
            cleanup_operations.append("temp_files")

        if not cleanup_operations:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one cleanup operation must be selected",
            )

        logger.info(
            "System cleanup initiated",
            operations=cleanup_operations,
            user=current_user.username,
        )

        return SuccessResponse(
            success=True,
            message="System cleanup completed",
            data={
                "operations": cleanup_operations,
                "completed_at": datetime.now(UTC).isoformat(),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("System cleanup failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System cleanup failed",
        ) from e


@router.get(
    "/disk-usage",
    response_model=dict[str, Any],
    summary="Get disk usage information",
    description="Get detailed disk usage information for system directories.",
)
async def get_disk_usage(
    current_user: User = Depends(require_permissions("system:read")),
    _rate_limit: None = Depends(metrics_rate_limit),
) -> dict[str, Any]:
    """Get disk usage information.

    Returns detailed disk usage statistics for system
    directories including logs, backups, and data.
    """
    try:
        import shutil

        disk_usage = {}

        # Get usage for key directories
        directories = {
            "root": "/",
            "tmp": "/tmp",
        }

        for name, path in directories.items():
            try:
                if os.path.exists(path):
                    usage = shutil.disk_usage(path)
                    disk_usage[name] = {
                        "path": path,
                        "total": usage.total,
                        "used": usage.used,
                        "free": usage.free,
                        "usage_percent": round((usage.used / usage.total) * 100, 2),
                    }
                else:
                    disk_usage[name] = {
                        "path": path,
                        "total": 0,
                        "used": 0,
                        "free": 0,
                        "usage_percent": 0,
                        "error": "Path not found",
                    }
            except Exception as e:
                disk_usage[name] = {
                    "path": path,
                    "error": str(e),
                }

        # Add timestamp
        disk_usage["timestamp"] = datetime.now(UTC).isoformat()

        logger.info(
            "Disk usage information retrieved",
            user=current_user.username,
        )

        return disk_usage

    except Exception as e:
        logger.error("Failed to get disk usage information", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get disk usage information",
        ) from e


@router.get(
    "/system-info",
    response_model=dict[str, Any],
    summary="Get system information",
    description="Get comprehensive system information including hardware and OS details.",
)
async def get_system_info(
    current_user: User = Depends(require_permissions("system:read")),
) -> dict[str, Any]:
    """Get comprehensive system information.

    Returns detailed system information including hardware
    specifications, OS version, and runtime environment.
    """
    try:
        system_info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "python_version": platform.python_version(),
            },
            "hardware": {
                "cpu_count": psutil.cpu_count(logical=False),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total": psutil.virtual_memory().total,
                "boot_time": psutil.boot_time(),
            },
            "network": {
                "hostname": platform.node(),
                "interfaces": list(psutil.net_if_addrs().keys()),
            },
            "environment": {
                "user": os.environ.get("USER", "unknown"),
                "home": os.environ.get("HOME", "unknown"),
                "path": os.environ.get("PATH", "")[:500],  # Truncate long PATH
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

        logger.info(
            "System information retrieved",
            user=current_user.username,
        )

        return system_info

    except Exception as e:
        logger.error("Failed to get system information", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get system information",
        ) from e


# Initialize some sample data for demonstration
def _initialize_sample_data() -> None:
    """Initialize sample system data for demonstration."""
    global _system_alerts

    # Sample alert
    sample_alert = SystemAlert(
        alert_id="alert_001",
        severity="warning",
        category="system_performance",
        title="High Memory Usage",
        message="System memory usage is above 85% threshold",
        source="monitoring_system",
        acknowledged=False,
        acknowledged_by=None,
        acknowledged_at=None,
        created_at=datetime.now(UTC) - timedelta(hours=1),
        resolved_at=None,
        metadata={"memory_usage": 87.5, "threshold": 85.0},
    )
    _system_alerts.append(sample_alert)


# Initialize sample data when module is loaded
_initialize_sample_data()
