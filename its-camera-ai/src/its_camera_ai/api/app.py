"""FastAPI application factory and configuration.

Creates and configures the FastAPI application with proper middleware,
error handling, monitoring, and API route registration.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from ..core.config import Settings, get_settings
from ..core.exceptions import ITSCameraAIError
from ..core.logging import get_logger, setup_logging
from .dependencies import cleanup_dependencies, setup_dependencies
from .middleware import (
    LoggingMiddleware,
    MetricsMiddleware,
    RateLimitMiddleware,
    SecurityMiddleware,
)
from .routers import (
    analytics,
    auth,
    cameras,
    health,
    models,
    system,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Handles startup and shutdown events for the FastAPI application.
    """
    settings = get_settings()

    # Startup
    logger.info("Starting ITS Camera AI application", version=settings.app_version)

    try:
        # Setup dependencies (database, Redis, etc.)
        await setup_dependencies(settings)

        logger.info("Application startup complete")
        yield

    except Exception as e:
        logger.error("Failed to start application", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down ITS Camera AI application")
        try:
            await cleanup_dependencies()
            logger.info("Application shutdown complete")
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))


def create_app(settings: Settings = None) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        settings: Application settings (optional, will use default if not provided)

    Returns:
        FastAPI: Configured FastAPI application instance
    """
    if settings is None:
        settings = get_settings()

    # Setup logging first
    setup_logging(settings)

    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered traffic monitoring and analytics system",
        docs_url="/api/docs" if not settings.is_production() else None,
        redoc_url="/api/redoc" if not settings.is_production() else None,
        openapi_url="/api/openapi.json" if not settings.is_production() else None,
        lifespan=lifespan,
    )

    # Store settings in app state
    app.state.settings = settings

    # Add middleware (order matters - first added = outermost layer)
    _add_middleware(app, settings)

    # Add exception handlers
    _add_exception_handlers(app)

    # Include routers
    _include_routers(app, settings)

    # Mount Prometheus metrics endpoint
    if settings.monitoring.enable_metrics:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

    logger.info(
        "FastAPI application created",
        title=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        debug=settings.debug,
    )

    return app


def _add_middleware(app: FastAPI, settings: Settings) -> None:
    """Add middleware to the FastAPI application."""

    # Security middleware (outermost)
    app.add_middleware(SecurityMiddleware)

    # Trusted hosts
    if settings.is_production():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"],  # Configure properly in production
        )

    # CORS middleware
    if settings.security.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.security.allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            allow_headers=["*"],
        )

    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        calls=settings.security.rate_limit_per_minute,
        period=60,
    )

    # Compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Metrics collection
    if settings.monitoring.enable_metrics:
        app.add_middleware(MetricsMiddleware)

    # Logging middleware (innermost)
    app.add_middleware(LoggingMiddleware)


def _add_exception_handlers(app: FastAPI) -> None:
    """Add custom exception handlers to the FastAPI application."""

    @app.exception_handler(ITSCameraAIError)
    async def its_camera_ai_error_handler(
        request: Request, exc: ITSCameraAIError
    ) -> JSONResponse:
        """Handle custom application errors."""
        logger.error(
            "Application error",
            error=exc.__class__.__name__,
            message=exc.message,
            code=exc.code,
            details=exc.details,
            path=request.url.path,
            method=request.method,
        )

        # Map error types to HTTP status codes
        status_code = 500
        if "VALIDATION" in exc.code:
            status_code = 400
        elif "AUTHENTICATION" in exc.code:
            status_code = 401
        elif "AUTHORIZATION" in exc.code:
            status_code = 403
        elif "NOT_FOUND" in exc.code:
            status_code = 404
        elif "RATE_LIMIT" in exc.code:
            status_code = 429

        return JSONResponse(
            status_code=status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        logger.warning(
            "Validation error",
            errors=exc.errors(),
            path=request.url.path,
            method=request.method,
        )

        return JSONResponse(
            status_code=400,
            content={
                "error": "ValidationError",
                "message": "Request validation failed",
                "code": "VALIDATION_ERROR",
                "details": {
                    "errors": exc.errors(),
                },
            },
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle unexpected errors."""
        logger.error(
            "Unexpected error",
            error=exc.__class__.__name__,
            message=str(exc),
            path=request.url.path,
            method=request.method,
            exc_info=True,
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
                "code": "INTERNAL_SERVER_ERROR",
            },
        )


def _include_routers(app: FastAPI, settings: Settings) -> None:
    """Include API routers in the FastAPI application."""

    # Health check (no prefix)
    app.include_router(health.router, tags=["health"])

    # API routes with prefix
    api_prefix = settings.api_prefix

    app.include_router(
        auth.router, prefix=api_prefix + "/auth", tags=["authentication"]
    )

    app.include_router(cameras.router, prefix=api_prefix + "/cameras", tags=["cameras"])

    app.include_router(
        analytics.router, prefix=api_prefix + "/analytics", tags=["analytics"]
    )

    app.include_router(models.router, prefix=api_prefix + "/models", tags=["models"])

    app.include_router(system.router, prefix=api_prefix + "/system", tags=["system"])


# Global app instance for easy access
_app: FastAPI = None


def get_app() -> FastAPI:
    """Get the global FastAPI application instance.

    Returns:
        FastAPI: The application instance

    Raises:
        RuntimeError: If the app hasn't been created yet
    """
    global _app
    if _app is None:
        _app = create_app()
    return _app


# For debugging and development
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "its_camera_ai.api.app:get_app",
        factory=True,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )
