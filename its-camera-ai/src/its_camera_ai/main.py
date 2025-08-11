"""Main application entry point for ITS Camera AI system.

This module provides the main entry point for running the application
in production or development environments.
"""

from .api.app import get_app
from .core.config import get_settings
from .core.logging import setup_logging


# Create the FastAPI application
app = get_app()


def main() -> None:
    """Main entry point for the application."""
    import uvicorn
    
    settings = get_settings()
    setup_logging(settings)
    
    uvicorn.run(
        "its_camera_ai.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload,
        workers=1 if settings.reload else settings.workers,
        log_level=settings.log_level.lower(),
        access_log=not settings.is_production(),
    )


if __name__ == "__main__":
    main()
