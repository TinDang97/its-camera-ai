"""
Authentication CLI commands with dependency injection.

Enhanced CLI commands that use the dependency injection container
for proper service management and clean architecture.
"""

import asyncio
import getpass

import typer
from dependency_injector.wiring import Provide, inject
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from ...containers import ApplicationContainer
from ...core.config import get_settings
from ...services.auth_service import AuthenticationService as AuthService
from ...services.cache import CacheService

app = typer.Typer(
    name="auth-di",
    help="Authentication commands with dependency injection",
    rich_markup_mode="rich",
)
console = Console()


@app.command("create-user")
@inject
def create_user(
    username: str = typer.Option(..., help="Username for the new user"),
    email: str = typer.Option(..., help="Email address for the new user"),
    full_name: str = typer.Option(None, help="Full name of the user"),
    is_admin: bool = typer.Option(False, help="Create user as admin"),
    # Injected dependencies
    auth_service: AuthService = Provide[ApplicationContainer.services.auth_service],
) -> None:
    """Create a new user with dependency injection.

    This command demonstrates how CLI commands can use the dependency
    injection container to access services cleanly.
    """

    async def _create_user():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Creating user...", total=None)

                # Get password securely
                password = getpass.getpass("Password: ")
                confirm_password = getpass.getpass("Confirm password: ")

                if password != confirm_password:
                    rprint("[red]Passwords do not match[/red]")
                    raise typer.Exit(1)

                # Create user through injected service
                user_data = {
                    "username": username,
                    "email": email,
                    "password": password,
                    "full_name": full_name,
                    "is_superuser": is_admin,
                }

                progress.update(task, description="Creating user account...")
                user = await auth_service.create_user(**user_data)

                progress.update(task, description="User created successfully!")

            # Display user information
            table = Table(title="User Created Successfully")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("ID", str(user.id))
            table.add_row("Username", user.username)
            table.add_row("Email", user.email)
            table.add_row("Full Name", user.full_name or "Not set")
            table.add_row("Admin", "Yes" if user.is_superuser else "No")
            table.add_row("Created", user.created_at.strftime("%Y-%m-%d %H:%M:%S"))

            console.print(table)

        except Exception as e:
            rprint(f"[red]Error creating user: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_create_user())


@app.command("list-users")
@inject
def list_users(
    limit: int = typer.Option(10, help="Maximum number of users to display"),
    active_only: bool = typer.Option(False, help="Show only active users"),
    # Injected dependencies
    auth_service: AuthService = Provide[ApplicationContainer.services.auth_service],
) -> None:
    """List users with dependency injection."""

    async def _list_users():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching users...", total=None)

                # Get users through injected service
                users = await auth_service.list_users(
                    limit=limit, active_only=active_only
                )

                progress.update(task, description=f"Found {len(users)} users")

            if not users:
                rprint("[yellow]No users found[/yellow]")
                return

            # Display users table
            table = Table(title=f"Users ({len(users)} found)")
            table.add_column("ID", style="cyan")
            table.add_column("Username", style="green")
            table.add_column("Email", style="blue")
            table.add_column("Full Name", style="magenta")
            table.add_column("Active", style="yellow")
            table.add_column("Admin", style="red")
            table.add_column("Created", style="dim")

            for user in users:
                table.add_row(
                    str(user.id)[:8] + "...",
                    user.username,
                    user.email,
                    user.full_name or "Not set",
                    "Yes" if user.is_active else "No",
                    "Yes" if user.is_superuser else "No",
                    user.created_at.strftime("%Y-%m-%d"),
                )

            console.print(table)

        except Exception as e:
            rprint(f"[red]Error listing users: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_list_users())


@app.command("delete-user")
@inject
def delete_user(
    username: str = typer.Option(..., help="Username to delete"),
    force: bool = typer.Option(False, help="Skip confirmation prompt"),
    # Injected dependencies
    auth_service: AuthService = Provide[ApplicationContainer.services.auth_service],
) -> None:
    """Delete a user with dependency injection."""

    async def _delete_user():
        try:
            # Get user first to confirm existence
            user = await auth_service.get_user_by_username(username)
            if not user:
                rprint(f"[red]User '{username}' not found[/red]")
                raise typer.Exit(1)

            # Confirm deletion
            if not force:
                if not Confirm.ask(
                    f"Are you sure you want to delete user '{username}'? This action cannot be undone."
                ):
                    rprint("[yellow]Operation cancelled[/yellow]")
                    return

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Deleting user...", total=None)

                # Delete user through injected service
                await auth_service.delete_user(user.id)

                progress.update(task, description="User deleted successfully!")

            rprint(f"[green]User '{username}' has been deleted successfully[/green]")

        except Exception as e:
            rprint(f"[red]Error deleting user: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_delete_user())


@app.command("cache-stats")
@inject
def show_cache_stats(
    # Injected dependencies
    cache_service: CacheService = Provide[ApplicationContainer.services.cache_service],
) -> None:
    """Show cache statistics using dependency injection."""

    async def _show_cache_stats():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Getting cache stats...", total=None)

                # Get cache statistics through injected service
                stats = await cache_service.get_stats()

                progress.update(task, description="Cache stats retrieved!")

            # Display cache statistics
            table = Table(title="Cache Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            for key, value in stats.items():
                table.add_row(str(key).replace("_", " ").title(), str(value))

            console.print(table)

        except Exception as e:
            rprint(f"[red]Error getting cache stats: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_show_cache_stats())


@app.command("test-auth")
@inject
def test_authentication(
    username: str = typer.Option(..., help="Username to test"),
    # Injected dependencies
    auth_service: AuthService = Provide[ApplicationContainer.services.auth_service],
) -> None:
    """Test user authentication with dependency injection."""

    async def _test_auth():
        try:
            password = getpass.getpass("Password: ")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Testing authentication...", total=None)

                # Test authentication through injected service
                result = await auth_service.authenticate_user(username, password)

                if result:
                    progress.update(task, description="Authentication successful!")
                    rprint("[green]✓ Authentication successful[/green]")

                    # Show user info
                    table = Table(title="Authenticated User")
                    table.add_column("Field", style="cyan")
                    table.add_column("Value", style="green")

                    table.add_row("ID", str(result.id))
                    table.add_row("Username", result.username)
                    table.add_row("Email", result.email)
                    table.add_row("Full Name", result.full_name or "Not set")
                    table.add_row(
                        "Last Login",
                        str(result.last_login) if result.last_login else "Never",
                    )

                    console.print(table)

                else:
                    progress.update(task, description="Authentication failed!")
                    rprint("[red]✗ Authentication failed[/red]")

        except Exception as e:
            rprint(f"[red]Error testing authentication: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_test_auth())


@app.command("init-container")
def initialize_container() -> None:
    """Initialize the dependency injection container for CLI usage."""

    async def _init_container():
        try:
            from ...containers import init_container, wire_container

            settings = get_settings()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Initializing DI container...", total=None)

                # Initialize container
                await init_container(settings)

                progress.update(task, description="Wiring CLI modules...")

                # Wire CLI modules
                wire_container(
                    [
                        "src.its_camera_ai.cli.commands.auth_di",
                    ]
                )

                progress.update(task, description="DI container initialized!")

            rprint(
                "[green]✓ Dependency injection container initialized successfully[/green]"
            )
            rprint("[dim]You can now use CLI commands with dependency injection[/dim]")

        except Exception as e:
            rprint(f"[red]Error initializing container: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(_init_container())


# Additional utility commands for DI debugging
@app.command("container-info")
def show_container_info() -> None:
    """Show dependency injection container information."""

    try:
        from ...containers import ApplicationContainer

        table = Table(title="Dependency Injection Container Info")
        table.add_column("Container", style="cyan")
        table.add_column("Providers", style="green")
        table.add_column("Status", style="yellow")

        container = ApplicationContainer

        # Infrastructure container
        infra_providers = len(
            [attr for attr in dir(container.infrastructure) if not attr.startswith("_")]
        )
        table.add_row("Infrastructure", str(infra_providers), "Available")

        # Repositories container
        repo_providers = len(
            [attr for attr in dir(container.repositories) if not attr.startswith("_")]
        )
        table.add_row("Repositories", str(repo_providers), "Available")

        # Services container
        service_providers = len(
            [attr for attr in dir(container.services) if not attr.startswith("_")]
        )
        table.add_row("Services", str(service_providers), "Available")

        console.print(table)

        rprint("\n[dim]Available service providers:[/dim]")
        services = [
            attr for attr in dir(container.services) if not attr.startswith("_")
        ]
        for service in sorted(services):
            rprint(f"  • {service}")

    except Exception as e:
        rprint(f"[red]Error showing container info: {e}[/red]")


if __name__ == "__main__":
    app()
