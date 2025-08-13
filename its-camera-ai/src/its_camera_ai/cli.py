"""Command-line interface for ITS Camera AI system.

Provides CLI commands for starting the server, running migrations,
and other administrative tasks.
"""

import asyncio
import sys
from pathlib import Path

import click
import uvicorn

from .core.config import get_settings
from .core.logging import get_logger, setup_logging
from .models.database import create_database_engine

logger = get_logger(__name__)


@click.group()
@click.version_option()
def main() -> None:
    """ITS Camera AI - Intelligent Traffic Monitoring System."""
    pass


@main.command()
@click.option(
    "--host",
    default=None,
    help="Host to bind the server to (defaults to environment-specific binding)",
    show_default=False,
)
@click.option(
    "--port",
    default=8080,
    type=int,
    help="Port to bind the server to",
    show_default=True,
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development",
)
@click.option(
    "--workers",
    default=1,
    type=int,
    help="Number of worker processes",
    show_default=True,
)
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    help="Log level",
    show_default=True,
)
def serve(
    host: str,
    port: int,
    reload: bool,
    workers: int,
    log_level: str,
) -> None:
    """Start the FastAPI server."""
    settings = get_settings()
    setup_logging(settings)

    # Security: Environment-specific host binding
    if host is None:
        if settings.is_production():
            # Production: bind to specific interface or localhost
            host = settings.get("api_host", "127.0.0.1")
        else:
            # Development: allow broader access for development
            host = "0.0.0.0"  # noqa: S104 - Development only

    logger.info(
        "Starting ITS Camera AI server",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level=log_level,
        environment=settings.environment,
    )

    # Override settings from CLI arguments
    settings.api_host = host
    settings.api_port = port
    settings.reload = reload
    settings.workers = workers
    settings.log_level = log_level.upper()

    uvicorn.run(
        "its_camera_ai.api.app:get_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        workers=1 if reload else workers,  # Reload only works with 1 worker
        log_level=log_level,
        access_log=not settings.is_production(),
    )


@main.command()
@click.option(
    "--create-tables",
    is_flag=True,
    help="Create database tables",
)
async def init_db(create_tables: bool) -> None:
    """Initialize the database."""
    settings = get_settings()
    setup_logging(settings)

    logger.info("Initializing database")

    try:
        db_manager = await create_database_engine(settings)

        if create_tables:
            await db_manager.create_tables()
            logger.info("Database tables created")

        logger.info("Database initialization complete")

    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        sys.exit(1)


@main.command()
@click.option(
    "--check",
    "check_only",
    is_flag=True,
    help="Only check configuration without starting services",
)
def config(check_only: bool) -> None:
    """Show configuration or check configuration validity."""
    try:
        settings = get_settings()
        setup_logging(settings)

        if check_only:
            logger.info("Configuration check passed")
            click.echo("✓ Configuration is valid")
        else:
            # Print configuration (excluding sensitive data)
            config_dict = settings.dict()

            # Mask sensitive fields
            sensitive_fields = [
                "secret_key",
                "database.url",
                "redis.url",
                "sentry_dsn",
            ]

            for field_path in sensitive_fields:
                keys = field_path.split(".")
                current = config_dict

                for key in keys[:-1]:
                    if key in current and isinstance(current[key], dict):
                        current = current[key]
                    else:
                        break
                else:
                    final_key = keys[-1]
                    if final_key in current:
                        current[final_key] = "***HIDDEN***"

            import json

            click.echo(json.dumps(config_dict, indent=2, default=str))

    except Exception as e:
        logger.error("Configuration error", error=str(e))
        click.echo(f"✗ Configuration error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option(
    "--component",
    type=click.Choice(["database", "redis", "all"]),
    default="all",
    help="Health check component",
    show_default=True,
)
async def health(component: str) -> None:
    """Check system health."""
    settings = get_settings()
    setup_logging(settings)

    logger.info(f"Running health check for: {component}")

    checks_passed = 0
    total_checks = 0

    if component in ["database", "all"]:
        total_checks += 1
        try:
            from sqlalchemy import text

            db_manager = await create_database_engine(settings)
            async with db_manager.get_session() as db:
                result = await db.execute(text("SELECT 1"))
                await result.fetchone()

            click.echo("✓ Database: Connected")
            checks_passed += 1
        except Exception as e:
            click.echo(f"✗ Database: Failed - {e}")

    if component in ["redis", "all"]:
        total_checks += 1
        try:
            import redis.asyncio as redis

            redis_client = redis.from_url(settings.redis.url)
            await redis_client.ping()
            await redis_client.close()

            click.echo("✓ Redis: Connected")
            checks_passed += 1
        except Exception as e:
            click.echo(f"✗ Redis: Failed - {e}")

    if checks_passed == total_checks:
        click.echo(f"\n✓ All {total_checks} health checks passed")
        sys.exit(0)
    else:
        click.echo(
            f"\n✗ {total_checks - checks_passed} of {total_checks} health checks failed"
        )
        sys.exit(1)


@main.command()
@click.option(
    "--output",
    type=click.Path(),
    help="Output file path (default: print to stdout)",
)
def version(output: str | None) -> None:
    """Show version information."""
    from . import __version__

    version_info = {
        "version": __version__,
        "python_version": sys.version,
        "platform": sys.platform,
    }

    if output:
        import json

        output_path = Path(output)
        output_path.write_text(json.dumps(version_info, indent=2))
        click.echo(f"Version information written to {output_path}")
    else:
        click.echo(f"ITS Camera AI v{__version__}")
        click.echo(f"Python {sys.version}")
        click.echo(f"Platform: {sys.platform}")


# Async command wrapper
def async_command(f):
    """Decorator to run async commands."""

    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


# Apply async wrapper to async commands
init_db = async_command(init_db)
health = async_command(health)


if __name__ == "__main__":
    main()
