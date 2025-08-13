"""FastAPI application with dependency injection integration.

Enhanced FastAPI application factory that integrates with the
dependency injection container system for clean architecture
and proper resource management.
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

from ..containers import ApplicationContainer
from ..core.config import Settings, get_settings
from ..core.exceptions import ITSCameraAIError
from ..core.logging import get_logger, setup_logging
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
    """Application lifespan context manager with container instance binding.

    Handles startup and shutdown events for the FastAPI application
    with container instance bound to the app state.
    """
    settings = get_settings()

    # Startup
    logger.info(
        "Starting ITS Camera AI application with DI", version=settings.app_version
    )

    try:
        # Create and initialize container instance
        container = ApplicationContainer()
        container.config.from_dict(settings.model_dump())

        # Initialize container resources
        await container.init_resources()

        # Bind container instance to app state
        app.state.container = container

        # Wire dependencies for API modules
        container.wire(
            modules=[
                "src.its_camera_ai.api.routers.auth",
                "src.its_camera_ai.api.routers.cameras",
                "src.its_camera_ai.api.routers.analytics",
                "src.its_camera_ai.api.routers.system",
                "src.its_camera_ai.api.routers.health",
                "src.its_camera_ai.api.routers.models",
                "src.its_camera_ai.api.dependencies",
                "src.its_camera_ai.cli.commands.auth",
                "src.its_camera_ai.cli.commands.services",
                "src.its_camera_ai.cli.commands.config",
            ]
        )

        logger.info("Application startup complete with container instance binding")
        yield

    except Exception as e:
        logger.error("Failed to start application with DI", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down ITS Camera AI application")
        try:
            # Get container from app state
            container = getattr(app.state, "container", None)
            if container:
                # Unwire dependencies
                container.unwire()

                # Shutdown container resources
                await container.shutdown_resources()

                # Clear container from app state
                app.state.container = None

            logger.info("Application shutdown complete")
        except Exception as e:
            logger.error("Error during shutdown", error=str(e))


def create_app_with_di(settings: Settings | None = None) -> FastAPI:
    """Create and configure FastAPI application with dependency injection.

    Args:
        settings: Application settings (optional, will use default if not provided)

    Returns:
        FastAPI: Configured FastAPI application instance with DI
    """
    if settings is None:
        settings = get_settings()

    # Setup logging first
    setup_logging(settings)

    # Create FastAPI app with DI lifespan
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered traffic monitoring and analytics system with dependency injection",
        docs_url="/api/docs" if not settings.is_production() else None,
        redoc_url="/api/redoc" if not settings.is_production() else None,
        openapi_url="/api/openapi.json" if not settings.is_production() else None,
        lifespan=lifespan,
    )

    # Store settings in app state (container will be bound during lifespan)
    app.state.settings = settings

    # Add middleware (order matters - first added = outermost layer)
    _add_middleware(app, settings)

    # Add exception handlers
    _add_exception_handlers(app)

    # Include routers with DI support
    _include_routers(app)

    # Add Prometheus metrics endpoint
    if settings.monitoring.enable_metrics:
        metrics_app = make_asgi_app()
        app.mount("/metrics", metrics_app)

    logger.info("FastAPI application created with dependency injection")
    return app


def _add_middleware(app: FastAPI, settings: Settings) -> None:
    """Add middleware to FastAPI application."""

    # Security middleware (outermost)
    if settings.security.enabled:
        app.add_middleware(SecurityMiddleware)

    # Trusted host middleware
    if settings.security.allowed_hosts:
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=settings.security.allowed_hosts
        )

    # CORS middleware
    if settings.security.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.security.allowed_origins,
            allow_credentials=settings.security.allow_credentials,
            allow_methods=settings.security.allow_methods,
            allow_headers=settings.security.allow_headers,
        )

    # Compression middleware
    if settings.compression.enabled:
        app.add_middleware(
            GZipMiddleware,
            minimum_size=settings.compression.min_size,
            compresslevel=settings.compression.level,
        )

    # Rate limiting middleware
    if settings.rate_limit_enabled:
        app.add_middleware(RateLimitMiddleware)

    # Metrics middleware
    if settings.monitoring.enable_metrics:
        app.add_middleware(MetricsMiddleware)

    # Logging middleware (innermost - closest to request handlers)
    app.add_middleware(LoggingMiddleware)


def _add_exception_handlers(app: FastAPI) -> None:
    """Add custom exception handlers."""

    @app.exception_handler(ITSCameraAIError)
    async def its_camera_ai_exception_handler(
        request: Request, exc: ITSCameraAIError
    ) -> JSONResponse:
        """Handle custom ITS Camera AI exceptions."""
        logger.error(
            "ITS Camera AI error",
            error_type=type(exc).__name__,
            error=str(exc),
            path=request.url.path,
            method=request.method,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle request validation errors."""
        logger.warning(
            "Request validation error",
            errors=exc.errors(),
            path=request.url.path,
            method=request.method,
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(500)
    async def internal_server_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle internal server errors."""
        logger.error(
            "Internal server error",
            error_type=type(exc).__name__,
            error=str(exc),
            path=request.url.path,
            method=request.method,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An internal server error occurred",
                "details": None,
            },
        )


def _include_routers(app: FastAPI) -> None:
    """Include API routers with dependency injection support."""

    # Include all API routers with common prefix and tags
    app.include_router(health.router, prefix="/api/v1", tags=["health"])

    app.include_router(auth.router, prefix="/api/v1/auth", tags=["authentication"])

    app.include_router(cameras.router, prefix="/api/v1/cameras", tags=["cameras"])

    app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])

    app.include_router(system.router, prefix="/api/v1/system", tags=["system"])

    app.include_router(models.router, prefix="/api/v1/models", tags=["models"])


# Create the main app instance with dependency injection
def create_app(settings: Settings | None = None) -> FastAPI:
    """Main factory function for creating the FastAPI app with DI.

    This is the primary entry point for creating the application.

    Args:
        settings: Application settings

    Returns:
        FastAPI application with dependency injection
    """
    return create_app_with_di(settings)


# Convenience function for getting the application
def get_application() -> FastAPI:
    """Get configured FastAPI application with dependency injection.

    Returns:
        FastAPI application instance
    """
    settings = get_settings()
    return create_app_with_di(settings)


# Application instance for deployment
app = create_app_with_di()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.its_camera_ai.api.app_di:app",
        host="0.0.0.0",
        port=8000,
        reload=not settings.is_production(),
        log_config=None,  # Use our custom logging
    )
