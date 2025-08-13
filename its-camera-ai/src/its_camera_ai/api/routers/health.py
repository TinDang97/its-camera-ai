"""Health check endpoints.

Provides health, readiness, and liveness endpoints for monitoring
and load balancer health checks.
"""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import psutil
import torch
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from ...core.config import Settings, get_settings
from ...core.logging import get_logger
from ...ml.core_vision_engine import (
    CoreVisionEngine,
    create_optimal_config,
)
from ..dependencies import get_db, get_redis

logger = get_logger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    version: str
    environment: str
    uptime_seconds: float
    checks: dict[str, dict[str, Any]]


class ReadinessResponse(BaseModel):
    """Readiness check response model."""

    ready: bool
    timestamp: datetime
    checks: dict[str, dict[str, Any]]


# Track application start time
_start_time = datetime.now(UTC)


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Basic health check endpoint.

    Always returns 200 OK if the application is running.
    Provides basic application information.
    """
    current_time = datetime.now(UTC)
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
async def liveness_check() -> dict[str, str]:
    """Kubernetes liveness probe endpoint.

    Returns 200 OK if the application process is running.
    Used by Kubernetes to determine if the pod should be restarted.
    """
    return {"status": "alive", "timestamp": datetime.now(UTC).isoformat()}


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(
    _settings: Settings = Depends(get_settings),
) -> ReadinessResponse:
    """Kubernetes readiness probe endpoint.

    Checks if the application is ready to serve traffic.
    Verifies database and cache connectivity.
    """
    current_time = datetime.now(UTC)
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
            "message": "Connected",
        }
    except Exception as e:
        logger.error("Database health check failed", error=str(e))
        checks["database"] = {
            "status": "unhealthy",
            "response_time_ms": 0,
            "message": f"Connection failed: {str(e)}",
        }
        all_ready = False

    # Check Redis connectivity
    try:
        redis_client = await get_redis()
        await redis_client.ping()

        checks["redis"] = {
            "status": "healthy",
            "response_time_ms": 0,  # TODO: Implement timing
            "message": "Connected",
        }
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))
        checks["redis"] = {
            "status": "unhealthy",
            "response_time_ms": 0,
            "message": f"Connection failed: {str(e)}",
        }
        all_ready = False

    # Check ML models (if applicable)
    try:
        ml_health = await _check_ml_models_health_quick()
        checks["ml_models"] = ml_health
        if ml_health["status"] != "healthy":
            all_ready = False
    except Exception as e:
        logger.error("ML models health check failed", error=str(e))
        checks["ml_models"] = {
            "status": "unhealthy",
            "message": f"Models not available: {str(e)}",
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
    current_time = datetime.now(UTC)
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
    results = await asyncio.gather(*check_tasks.values(), return_exceptions=True)

    # Process results
    for (check_name, _), result in zip(check_tasks.items(), results, strict=False):
        if isinstance(result, Exception):
            checks[check_name] = {
                "status": "error",
                "message": f"Health check failed: {str(result)}",
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


async def _check_database_health() -> dict[str, Any]:
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
            "version": str(version[0]) if version else "unknown",
        }
    except Exception as e:
        return {"status": "unhealthy", "message": f"Connection failed: {str(e)}"}


async def _check_redis_health() -> dict[str, Any]:
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
            "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
        }
    except Exception as e:
        return {"status": "unhealthy", "message": f"Connection failed: {str(e)}"}


async def _check_ml_models_health_quick() -> dict[str, Any]:
    """Quick ML model health check for readiness probe."""
    try:
        # Quick check for CUDA availability and basic model loading capability
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0

        return {
            "status": "healthy",
            "message": f"CUDA available: {cuda_available}, GPUs: {gpu_count}",
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
        }
    except Exception as e:
        return {"status": "unhealthy", "message": f"ML environment check failed: {str(e)}"}


async def _check_ml_models_health() -> dict[str, Any]:
    """Comprehensive ML models health check with performance testing."""
    start_time = time.time()

    try:
        # Check hardware prerequisites
        hardware_status = await _check_ml_hardware()
        if hardware_status["status"] != "healthy":
            return {
                "status": "unhealthy",
                "message": "Hardware prerequisites not met",
                "hardware": hardware_status,
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
            }

        # Initialize a test vision engine with minimal configuration
        config = create_optimal_config("edge", available_memory_gb=2.0, target_cameras=1)
        test_engine = None

        try:
            test_engine = CoreVisionEngine(config)

            # Test model initialization (lightweight)
            init_start = time.time()
            await test_engine.initialize()
            init_time = (time.time() - init_start) * 1000

            # Test inference with dummy frame
            inference_start = time.time()
            dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            vision_result = await test_engine.process_frame(
                dummy_frame, "health_check_frame", "health_check_camera"
            )
            inference_time = (time.time() - inference_start) * 1000

            # Get model performance metrics
            test_engine.get_performance_metrics()
            health_status = test_engine.get_health_status()

            # Determine overall health
            is_healthy = (
                inference_time < config.target_latency_ms * 2  # Allow 2x target latency for health check
                and health_status["health_score"] > 0.5
                and vision_result.total_processing_time_ms > 0
            )

            total_time = (time.time() - start_time) * 1000

            result = {
                "status": "healthy" if is_healthy else "degraded",
                "message": "Models loaded and tested successfully" if is_healthy else "Models loaded but performance degraded",
                "response_time_ms": round(total_time, 2),
                "initialization_time_ms": round(init_time, 2),
                "inference_time_ms": round(inference_time, 2),
                "target_latency_ms": config.target_latency_ms,
                "meets_latency_target": inference_time <= config.target_latency_ms,
                "hardware": hardware_status,
                "model_info": {
                    "model_type": config.model_type.value,
                    "optimization_backend": config.optimization_backend.value,
                    "input_resolution": config.input_resolution,
                },
                "performance": {
                    "health_score": health_status["health_score"],
                    "detection_count": vision_result.detection_count,
                    "processing_quality": vision_result.processing_quality_score,
                    "gpu_memory_used_mb": vision_result.gpu_memory_used_mb,
                    "cpu_utilization": vision_result.cpu_utilization,
                    "batch_size_used": vision_result.batch_size_used,
                },
            }

            return result

        finally:
            # Always cleanup test engine
            if test_engine and test_engine.initialized:
                await test_engine.cleanup()

    except Exception as e:
        total_time = (time.time() - start_time) * 1000
        logger.error(f"ML models health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Models health check failed: {str(e)}",
            "response_time_ms": round(total_time, 2),
            "error": str(e),
        }


async def _check_ml_hardware() -> dict[str, Any]:
    """Check ML hardware prerequisites and GPU health."""
    try:
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0

        hardware_info = {
            "cuda_available": cuda_available,
            "gpu_count": gpu_count,
            "torch_version": torch.__version__,
        }

        if cuda_available and gpu_count > 0:
            # Get GPU information
            gpu_info = []
            total_memory_gb = 0
            available_memory_gb = 0

            for i in range(gpu_count):
                gpu_properties = torch.cuda.get_device_properties(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)

                # Get current GPU memory usage
                torch.cuda.set_device(i)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                memory_free = memory_total - memory_reserved

                gpu_info.append({
                    "device_id": i,
                    "name": gpu_properties.name,
                    "compute_capability": f"{gpu_properties.major}.{gpu_properties.minor}",
                    "total_memory_gb": round(memory_total, 2),
                    "allocated_memory_gb": round(memory_allocated, 2),
                    "reserved_memory_gb": round(memory_reserved, 2),
                    "free_memory_gb": round(memory_free, 2),
                    "utilization_percent": round((memory_reserved / memory_total) * 100, 1),
                })

                total_memory_gb += memory_total
                available_memory_gb += memory_free

            hardware_info.update({
                "gpus": gpu_info,
                "total_gpu_memory_gb": round(total_memory_gb, 2),
                "available_gpu_memory_gb": round(available_memory_gb, 2),
            })

            # Check if we have sufficient GPU memory (minimum 1GB free)
            has_sufficient_memory = available_memory_gb >= 1.0

            status = "healthy" if has_sufficient_memory else "warning"
            message = "GPU hardware ready" if has_sufficient_memory else "Low GPU memory available"

        else:
            # CPU-only mode
            status = "warning"
            message = "No GPU available, will use CPU-only inference"

        # Add CPU information
        cpu_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "unknown",
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        }

        hardware_info.update({"cpu": cpu_info})

        return {
            "status": status,
            "message": message,
            "hardware": hardware_info,
        }

    except Exception as e:
        logger.error(f"Hardware check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Hardware check failed: {str(e)}",
            "error": str(e),
        }


async def _check_disk_space() -> dict[str, Any]:
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
            "usage_percent": round(usage_percent, 2),
        }
    except Exception as e:
        return {"status": "error", "message": f"Disk check failed: {str(e)}"}


async def _check_memory_usage() -> dict[str, Any]:
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
            "usage_percent": round(memory.percent, 2),
        }
    except Exception as e:
        return {"status": "error", "message": f"Memory check failed: {str(e)}"}
