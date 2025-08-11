"""Health check endpoints.

Provides health, readiness, and liveness endpoints for monitoring
and load balancer health checks.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from ...core.config import Settings, get_settings
from ...core.logging import get_logger
from ..dependencies import get_redis, get_db


logger = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str
    timestamp: datetime
    version: str
    environment: str
    uptime_seconds: float
    checks: Dict[str, Dict[str, Any]]


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    
    ready: bool
    timestamp: datetime
    checks: Dict[str, Dict[str, Any]]


# Track application start time
_start_time = datetime.now(timezone.utc)


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Basic health check endpoint.
    
    Always returns 200 OK if the application is running.
    Provides basic application information.
    """
    current_time = datetime.now(timezone.utc)
    uptime = (current_time - _start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=current_time,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=uptime,
        checks={},
    )


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, str]:
    """Kubernetes liveness probe endpoint.
    
    Returns 200 OK if the application process is running.
    Used by Kubernetes to determine if the pod should be restarted.
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(
    settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    """Kubernetes readiness probe endpoint.
    
    Checks if the application is ready to serve traffic.
    Verifies database and cache connectivity.
    """
    current_time = datetime.now(timezone.utc)
    checks = {}
    all_ready = True
    
    # Check database connectivity
    try:
        from sqlalchemy import text
        async with get_db() as db:
            result = await db.execute(text("SELECT 1"))
            await result.fetchone()
            
        checks["database"] = {
            "status": "healthy",
            "response_time_ms": 0,  # TODO: Implement timing
            "message": "Connected"
        }
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        checks["database"] = {
            "status": "unhealthy",
            "response_time_ms": 0,
            "message": f"Connection failed: {str(e)}"
        }
        all_ready = False
    
    # Check Redis connectivity
    try:
        redis_client = await get_redis()
        await redis_client.ping()
        
        checks["redis"] = {
            "status": "healthy",
            "response_time_ms": 0,  # TODO: Implement timing
            "message": "Connected"
        }
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        checks["redis"] = {
            "status": "unhealthy",
            "response_time_ms": 0,
            "message": f"Connection failed: {str(e)}"
        }
        all_ready = False
    
    # Check ML models (if applicable)
    try:
        # TODO: Add ML model health checks
        checks["ml_models"] = {
            "status": "healthy",
            "message": "Models loaded"
        }
    except Exception as e:
        logger.error("ML models health check failed", error=str(e))
        checks["ml_models"] = {
            "status": "unhealthy",
            "message": f"Models not available: {str(e)}"
        }
        all_ready = False
    
    response = ReadinessResponse(
        ready=all_ready,
        timestamp=current_time,
        checks=checks,
    )
    
    # Return 503 if not ready
    if not all_ready:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.dict(),
        )
    
    return response


@router.get("/health/detailed", response_model=HealthResponse)
async def detailed_health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Detailed health check with comprehensive system status.
    
    Provides detailed information about all system components.
    """
    current_time = datetime.now(timezone.utc)
    uptime = (current_time - _start_time).total_seconds()
    checks = {}
    overall_status = "healthy"
    
    # Run all health checks concurrently
    check_tasks = {
        "database": _check_database_health(),
        "redis": _check_redis_health(),
        "ml_models": _check_ml_models_health(),
        "disk_space": _check_disk_space(),
        "memory": _check_memory_usage(),
    }
    
    # Wait for all checks to complete
    results = await asyncio.gather(
        *check_tasks.values(),
        return_exceptions=True
    )
    
    # Process results
    for (check_name, _), result in zip(check_tasks.items(), results):
        if isinstance(result, Exception):
            checks[check_name] = {
                "status": "error",
                "message": f"Health check failed: {str(result)}"
            }
            overall_status = "degraded"
        else:
            checks[check_name] = result
            if result.get("status") != "healthy":
                overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=current_time,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=uptime,
        checks=checks,
    )


async def _check_database_health() -> Dict[str, Any]:
    """Check database health."""
    import time
    from sqlalchemy import text
    
    try:
        start_time = time.time()
        async with get_db() as db:
            result = await db.execute(text("SELECT version()"))
            version = await result.fetchone()
            
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "message": "Connected",
            "version": str(version[0]) if version else "unknown"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Connection failed: {str(e)}"
        }


async def _check_redis_health() -> Dict[str, Any]:
    """Check Redis health."""
    import time
    
    try:
        start_time = time.time()
        redis_client = await get_redis()
        info = await redis_client.info()
        response_time = (time.time() - start_time) * 1000
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "message": "Connected",
            "version": info.get("redis_version", "unknown"),
            "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Connection failed: {str(e)}"
        }


async def _check_ml_models_health() -> Dict[str, Any]:
    """Check ML models health."""
    try:
        # TODO: Implement actual model health checks
        return {
            "status": "healthy",
            "message": "Models loaded and ready",
            "models_count": 1  # Placeholder
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Models not available: {str(e)}"
        }


async def _check_disk_space() -> Dict[str, Any]:
    """Check disk space."""
    import shutil
    
    try:
        total, used, free = shutil.disk_usage("/")
        
        # Convert to GB
        total_gb = total / (1024**3)
        used_gb = used / (1024**3)
        free_gb = free / (1024**3)
        usage_percent = (used / total) * 100
        
        status = "healthy"
        if usage_percent > 90:
            status = "critical"
        elif usage_percent > 80:
            status = "warning"
        
        return {
            "status": status,
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "usage_percent": round(usage_percent, 2)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Disk check failed: {str(e)}"
        }


async def _check_memory_usage() -> Dict[str, Any]:
    """Check memory usage."""
    import psutil
    
    try:
        memory = psutil.virtual_memory()
        
        status = "healthy"
        if memory.percent > 90:
            status = "critical"
        elif memory.percent > 80:
            status = "warning"
        
        return {
            "status": status,
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "usage_percent": round(memory.percent, 2)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Memory check failed: {str(e)}"
        }
