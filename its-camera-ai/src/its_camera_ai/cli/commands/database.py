"""Database management CLI commands.

Provides database initialization, migration, seeding, and maintenance
commands for the ITS Camera AI system.
"""

import asyncio

import typer
from rich.console import Console
from rich.table import Table

from alembic import command
from alembic.config import Config

from ...core.config import get_settings
from ...core.logging import get_logger
from ...models import (
    CameraType,
    StreamProtocol,
    User,
    create_database_engine,
)

logger = get_logger(__name__)
console = Console()
app = typer.Typer(name="database", help="Database management commands")


@app.command()
def init(
    create_tables: bool = typer.Option(
        True, "--create-tables/--no-create-tables", help="Create database tables"
    ),
    run_migrations: bool = typer.Option(
        True, "--migrate/--no-migrate", help="Run database migrations"
    ),
) -> None:
    """Initialize the database with schema and basic data."""
    console.print("[bold blue]Initializing database...[/bold blue]")

    try:
        get_settings()

        if run_migrations:
            console.print("Running database migrations...")
            migrate()

        if create_tables:
            console.print("Creating database tables...")
            asyncio.run(_create_tables())

        console.print("[bold green]Database initialized successfully![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Database initialization failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def migrate(
    message: str | None = typer.Option(None, "--message", "-m", help="Migration message"),
    revision: str | None = typer.Option(None, "--revision", "-r", help="Target revision"),
    autogenerate: bool = typer.Option(False, "--auto", help="Auto-generate migration"),
) -> None:
    """Run database migrations."""
    try:
        # Get alembic configuration
        alembic_cfg = Config("alembic.ini")

        if message:
            # Create new migration
            console.print(f"Creating migration: {message}")
            if autogenerate:
                command.revision(alembic_cfg, message=message, autogenerate=True)
            else:
                command.revision(alembic_cfg, message=message)
        else:
            # Run migrations
            if revision:
                console.print(f"Migrating to revision: {revision}")
                command.upgrade(alembic_cfg, revision)
            else:
                console.print("Running all pending migrations...")
                command.upgrade(alembic_cfg, "head")

        console.print("[bold green]Migrations completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Migration failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def downgrade(
    revision: str = typer.Option("base", "--revision", "-r", help="Target revision"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Downgrade database to a specific revision."""
    if not confirm:
        console.print(f"[bold yellow]This will downgrade database to revision: {revision}[/bold yellow]")
        if not typer.confirm("Are you sure?"):
            console.print("Operation cancelled.")
            raise typer.Exit(0)

    try:
        alembic_cfg = Config("alembic.ini")
        console.print(f"Downgrading to revision: {revision}")
        command.downgrade(alembic_cfg, revision)

        console.print("[bold green]Downgrade completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Downgrade failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """Show database migration status."""
    try:
        alembic_cfg = Config("alembic.ini")
        command.current(alembic_cfg, verbose=True)

    except Exception as e:
        console.print(f"[bold red]Failed to get status: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def seed(
    cameras_count: int = typer.Option(5, "--cameras", help="Number of demo cameras to create"),
    admin_user: bool = typer.Option(True, "--admin/--no-admin", help="Create admin user"),
) -> None:
    """Seed database with sample data."""
    console.print("[bold blue]Seeding database with sample data...[/bold blue]")

    try:
        asyncio.run(_seed_database(cameras_count, admin_user))
        console.print("[bold green]Database seeded successfully![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Seeding failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def stats() -> None:
    """Show database statistics."""
    console.print("[bold blue]Database Statistics[/bold blue]")

    try:
        asyncio.run(_show_database_stats())

    except Exception as e:
        console.print(f"[bold red]Failed to get statistics: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def cleanup(
    frames_days: int = typer.Option(7, "--frames-days", help="Frame retention days"),
    metrics_days: int = typer.Option(30, "--metrics-days", help="Metrics retention days"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Clean up old data from database."""
    if not confirm:
        console.print("[bold yellow]This will delete:[/bold yellow]")
        console.print(f"- Frame metadata older than {frames_days} days")
        console.print(f"- System metrics older than {metrics_days} days")
        if not typer.confirm("Are you sure?"):
            console.print("Operation cancelled.")
            raise typer.Exit(0)

    try:
        asyncio.run(_cleanup_database(frames_days, metrics_days))
        console.print("[bold green]Database cleanup completed![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Cleanup failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def backup(
    output_file: str | None = typer.Option(None, "--output", "-o", help="Backup file path"),
) -> None:
    """Create database backup."""
    import subprocess
    from datetime import datetime

    try:
        settings = get_settings()

        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"its_camera_ai_backup_{timestamp}.sql"

        # Parse database URL
        db_url = settings.database.url

        console.print(f"Creating backup: {output_file}")

        # Use pg_dump for backup
        cmd = [
            "pg_dump",
            "--no-password",
            "--verbose",
            "--clean",
            "--no-acl",
            "--no-owner",
            "-f", output_file,
            db_url.replace("postgresql+asyncpg://", "postgresql://")
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            console.print(f"[bold green]Backup created successfully: {output_file}[/bold green]")
        else:
            console.print(f"[bold red]Backup failed: {result.stderr}[/bold red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[bold red]Backup failed: {e}[/bold red]")
        raise typer.Exit(1)


# Async helper functions

async def _create_tables() -> None:
    """Create database tables."""
    settings = get_settings()
    db_manager = await create_database_engine(settings)
    await db_manager.create_tables()
    await db_manager.close()


async def _seed_database(cameras_count: int, create_admin: bool) -> None:
    """Seed database with sample data."""

    from ...api.schemas.common import Coordinates
    from ...api.schemas.database import CameraConfigSchema, CameraCreateSchema
    from ...services import CameraService

    settings = get_settings()
    db_manager = await create_database_engine(settings)

    try:
        async with db_manager.get_session() as session:
            camera_service = CameraService(session)

            # Create admin user if requested
            if create_admin:
                from passlib.context import CryptContext
                pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

                admin_user = User(
                    username="admin",
                    email="admin@its-camera-ai.com",
                    full_name="System Administrator",
                    hashed_password=pwd_context.hash("admin123"),
                    is_active=True,
                    is_superuser=True,
                    roles=["admin"],
                    permissions=["*"],
                )
                session.add(admin_user)
                await session.commit()
                console.print("✓ Created admin user (username: admin, password: admin123)")

            # Create sample cameras
            sample_cameras = [
                {
                    "name": "Main Entrance Camera",
                    "description": "Primary camera monitoring main entrance traffic",
                    "location": "Main Entrance, Building A",
                    "coordinates": {"latitude": 40.7589, "longitude": -73.9851},
                    "camera_type": CameraType.FIXED,
                    "stream_protocol": StreamProtocol.RTSP,
                },
                {
                    "name": "Parking Lot Camera 1",
                    "description": "Monitoring north parking area",
                    "location": "North Parking Lot",
                    "coordinates": {"latitude": 40.7590, "longitude": -73.9850},
                    "camera_type": CameraType.PTZ,
                    "stream_protocol": StreamProtocol.RTSP,
                },
                {
                    "name": "Highway Overpass Camera",
                    "description": "Traffic monitoring on Highway 101",
                    "location": "Highway 101 Overpass",
                    "coordinates": {"latitude": 40.7591, "longitude": -73.9849},
                    "camera_type": CameraType.DOME,
                    "stream_protocol": StreamProtocol.RTMP,
                },
                {
                    "name": "Intersection Camera East",
                    "description": "Monitoring major intersection - east view",
                    "location": "5th Ave & Main St - East",
                    "coordinates": {"latitude": 40.7592, "longitude": -73.9848},
                    "camera_type": CameraType.BULLET,
                    "stream_protocol": StreamProtocol.RTSP,
                },
                {
                    "name": "Tunnel Exit Camera",
                    "description": "Monitoring tunnel exit traffic flow",
                    "location": "Lincoln Tunnel Exit",
                    "coordinates": {"latitude": 40.7593, "longitude": -73.9847},
                    "camera_type": CameraType.THERMAL,
                    "stream_protocol": StreamProtocol.HLS,
                },
            ]

            for i in range(min(cameras_count, len(sample_cameras))):
                camera_data = sample_cameras[i]

                # Generate random stream URL
                stream_url = f"rtsp://demo-server:554/camera_{i+1}"

                camera_create = CameraCreateSchema(
                    name=camera_data["name"],
                    description=camera_data["description"],
                    location=camera_data["location"],
                    coordinates=Coordinates(**camera_data["coordinates"]),
                    camera_type=camera_data["camera_type"],
                    stream_url=stream_url,
                    stream_protocol=camera_data["stream_protocol"],
                    username="demo_user",
                    password="demo_pass",
                    config=CameraConfigSchema(
                        resolution={"width": 1920, "height": 1080},
                        fps=30.0,
                        bitrate=2000,
                        quality=8,
                        night_vision=True,
                        motion_detection=True,
                        recording_enabled=True,
                        retention_days=30,
                        analytics_enabled=True,
                        alerts_enabled=True,
                    ),
                    zone_id=f"zone_{i+1}",
                    tags=["demo", "sample", camera_data["camera_type"].value],
                )

                await camera_service.create_camera(camera_create)
                console.print(f"✓ Created camera: {camera_data['name']}")

            console.print(f"\n[bold green]Created {cameras_count} sample cameras[/bold green]")

    finally:
        await db_manager.close()


async def _show_database_stats() -> None:
    """Show database statistics."""
    settings = get_settings()
    db_manager = await create_database_engine(settings)

    try:
        async with db_manager.get_session() as session:
            # Get table counts
            tables = [
                ("Users", "users"),
                ("Cameras", "cameras"),
                ("Camera Settings", "camera_settings"),
                ("Frame Metadata", "frame_metadata"),
                ("Detection Results", "detection_results"),
                ("System Metrics", "system_metrics"),
                ("Aggregated Metrics", "aggregated_metrics"),
            ]

            table = Table(title="Database Statistics")
            table.add_column("Table", style="bold cyan")
            table.add_column("Count", style="bold green", justify="right")

            for name, table_name in tables:
                result = await session.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                )
                count = result.scalar()
                table.add_row(name, f"{count:,}")

            console.print(table)

            # Show database size
            size_result = await session.execute("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as size
            """)
            db_size = size_result.scalar()
            console.print(f"\n[bold]Database Size:[/bold] {db_size}")

    finally:
        await db_manager.close()


async def _cleanup_database(frames_days: int, metrics_days: int) -> None:
    """Clean up old database records."""
    from ...services import FrameService, MetricsService

    settings = get_settings()
    db_manager = await create_database_engine(settings)

    try:
        async with db_manager.get_session() as session:
            # Clean up frame metadata
            frame_service = FrameService(session)
            deleted_frames = await frame_service.cleanup_old_frames(
                retention_days=frames_days
            )
            console.print(f"✓ Deleted {deleted_frames:,} old frame metadata records")

            # Clean up system metrics
            metrics_service = MetricsService(session)
            deleted_metrics = await metrics_service.cleanup_old_metrics(
                retention_days=metrics_days
            )
            console.print(f"✓ Deleted {deleted_metrics:,} old system metrics records")

    finally:
        await db_manager.close()


if __name__ == "__main__":
    app()
