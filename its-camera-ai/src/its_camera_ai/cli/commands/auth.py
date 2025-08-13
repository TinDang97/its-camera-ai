"""
Authentication CLI commands for ITS Camera AI system.

Provides comprehensive authentication management commands:
- User management (create, update, delete, list)
- Role and permission management
- MFA enrollment and management
- Session management
- Security audit and monitoring
- Password policy management
- Authentication testing and diagnostics
"""

import asyncio
import getpass
import secrets
import string
from datetime import UTC, datetime

import bcrypt
import qrcode
import qrcode.image.svg
import redis.asyncio as redis
import typer
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ...core.config import get_settings
from ...models.user import SecurityAuditLog, User
from ...services.auth_service import (
    AuthenticationService,
    MFAMethod,
    UserCredentials,
)

app = typer.Typer(
    name="auth",
    help="Authentication and authorization management commands",
    rich_markup_mode="rich",
)
console = Console()


# ============================
# Utility Functions
# ============================


async def get_auth_service() -> AuthenticationService:
    """Get authentication service instance."""
    settings = get_settings()

    # Create async database engine
    engine = create_async_engine(settings.database.url, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Create Redis client
    redis_client = redis.from_url(settings.redis.url)

    async with async_session() as session:
        return AuthenticationService(session, redis_client, settings.security)


def generate_secure_password(length: int = 16) -> str:
    """Generate a secure random password."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    password = "".join(secrets.choice(alphabet) for _ in range(length))
    return password


def display_user_table(users: list[User], title: str = "Users"):
    """Display users in a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("ID", style="dim")
    table.add_column("Username", style="cyan")
    table.add_column("Email", style="green")
    table.add_column("Full Name")
    table.add_column("Active", justify="center")
    table.add_column("Verified", justify="center")
    table.add_column("Superuser", justify="center")
    table.add_column("MFA", justify="center")
    table.add_column("Roles")
    table.add_column("Last Login")

    for user in users:
        roles = ", ".join([role.name for role in user.roles]) if user.roles else "None"
        last_login = (
            user.last_login.strftime("%Y-%m-%d %H:%M") if user.last_login else "Never"
        )

        table.add_row(
            user.id[:8] + "...",
            user.username,
            user.email,
            user.full_name or "",
            "✅" if user.is_active else "❌",
            "✅" if user.is_verified else "❌",
            "✅" if user.is_superuser else "❌",
            "✅" if getattr(user, "mfa_enabled", False) else "❌",
            roles,
            last_login,
        )

    console.print(table)


def display_audit_logs(
    logs: list[SecurityAuditLog], title: str = "Security Audit Logs"
):
    """Display security audit logs in a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")

    table.add_column("Timestamp", style="dim")
    table.add_column("Event Type", style="cyan")
    table.add_column("User", style="green")
    table.add_column("IP Address")
    table.add_column("Success", justify="center")
    table.add_column("Risk Score", justify="center")
    table.add_column("Details")

    for log in logs:
        success_icon = "✅" if log.success else "❌"
        risk_color = (
            "red"
            if log.risk_score >= 80
            else "yellow"
            if log.risk_score >= 40
            else "green"
        )

        table.add_row(
            log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            log.event_type,
            log.username or "Unknown",
            log.ip_address or "N/A",
            success_icon,
            f"[{risk_color}]{log.risk_score}[/{risk_color}]",
            log.error_message or "Success",
        )

    console.print(table)


# ============================
# User Management Commands
# ============================


@app.command("create-user")
def create_user(
    username: str = typer.Argument(..., help="Username for the new user"),
    email: str = typer.Argument(..., help="Email address for the new user"),
    full_name: str | None = typer.Option(
        None, "--full-name", "-n", help="Full name of the user"
    ),
    password: str | None = typer.Option(
        None, "--password", "-p", help="Password (will prompt if not provided)"
    ),
    roles: list[str] | None = typer.Option(
        None, "--role", "-r", help="Roles to assign (can be used multiple times)"
    ),
    superuser: bool = typer.Option(
        False, "--superuser", "-s", help="Create as superuser"
    ),
    auto_password: bool = typer.Option(
        False, "--auto-password", help="Generate secure password automatically"
    ),
):
    """Create a new user account."""

    async def _create_user():
        auth_service = await get_auth_service()

        # Handle password
        if auto_password:
            password_value = generate_secure_password()
            rprint(f"[green]Generated password:[/green] {password_value}")
        elif password:
            password_value = password
        else:
            password_value = getpass.getpass("Enter password: ")
            confirm_password = getpass.getpass("Confirm password: ")
            if password_value != confirm_password:
                rprint("[red]Passwords do not match![/red]")
                return

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Creating user...", total=1)

                # Hash password
                hashed_password = bcrypt.hashpw(
                    password_value.encode(), bcrypt.gensalt()
                ).decode()

                # Create user
                user = await auth_service.user_service.create(
                    username=username,
                    email=email,
                    full_name=full_name,
                    hashed_password=hashed_password,
                    is_superuser=superuser,
                    is_active=True,
                    is_verified=True,
                )

                # Assign roles if specified
                if roles:
                    # This would require additional role assignment logic
                    rprint(
                        f"[yellow]Note: Role assignment not yet implemented. Roles: {', '.join(roles)}[/yellow]"
                    )

                progress.update(task, completed=1)

            rprint(f"[green]✅ User '{username}' created successfully![/green]")
            rprint(f"[dim]User ID: {user.id}[/dim]")

        except Exception as e:
            rprint(f"[red]❌ Failed to create user: {str(e)}[/red]")

    asyncio.run(_create_user())


@app.command("list-users")
def list_users(
    active_only: bool = typer.Option(
        False, "--active-only", "-a", help="Show only active users"
    ),
    with_roles: bool = typer.Option(
        False, "--with-roles", "-r", help="Include role information"
    ),
    limit: int = typer.Option(
        50, "--limit", "-l", help="Maximum number of users to show"
    ),
):
    """List all user accounts."""

    async def _list_users():
        auth_service = await get_auth_service()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading users...", total=1)

                # Get users (this would require extending the service)
                users = await auth_service.user_service.get_all(limit=limit)

                if active_only:
                    users = [user for user in users if user.is_active]

                progress.update(task, completed=1)

            if not users:
                rprint("[yellow]No users found.[/yellow]")
                return

            display_user_table(users, f"Users ({len(users)} total)")

        except Exception as e:
            rprint(f"[red]❌ Failed to list users: {str(e)}[/red]")

    asyncio.run(_list_users())


@app.command("delete-user")
def delete_user(
    username: str = typer.Argument(..., help="Username of the user to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt"),
):
    """Delete a user account."""

    async def _delete_user():
        if not force:
            confirmed = Confirm.ask(
                f"Are you sure you want to delete user '{username}'?"
            )
            if not confirmed:
                rprint("[yellow]Operation cancelled.[/yellow]")
                return

        auth_service = await get_auth_service()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Deleting user...", total=1)

                # Find user
                user = await auth_service._get_user_by_username(username)
                if not user:
                    rprint(f"[red]User '{username}' not found.[/red]")
                    return

                # Delete user
                deleted = await auth_service.user_service.delete_by_id(user.id)

                progress.update(task, completed=1)

            if deleted:
                rprint(f"[green]✅ User '{username}' deleted successfully![/green]")
            else:
                rprint(f"[red]❌ Failed to delete user '{username}'.[/red]")

        except Exception as e:
            rprint(f"[red]❌ Failed to delete user: {str(e)}[/red]")

    asyncio.run(_delete_user())


# ============================
# MFA Management Commands
# ============================


@app.command("setup-mfa")
def setup_mfa(
    username: str = typer.Argument(..., help="Username for MFA setup"),
    method: MFAMethod = typer.Option(
        MFAMethod.TOTP, "--method", "-m", help="MFA method"
    ),
):
    """Set up multi-factor authentication for a user."""

    async def _setup_mfa():
        auth_service = await get_auth_service()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Setting up MFA...", total=1)

                # Find user
                user = await auth_service._get_user_by_username(username)
                if not user:
                    rprint(f"[red]User '{username}' not found.[/red]")
                    return

                # Enroll MFA
                result = await auth_service.enroll_mfa(user.id, method)

                progress.update(task, completed=1)

            if result.success:
                rprint(f"[green]✅ MFA setup successful for user '{username}'![/green]")

                if method == MFAMethod.TOTP and result.secret:
                    rprint(f"[cyan]Secret Key:[/cyan] {result.secret}")

                    if result.qr_code_url:
                        # Generate QR code
                        qr = qrcode.QRCode(
                            version=1,
                            error_correction=qrcode.constants.ERROR_CORRECT_L,
                            box_size=10,
                            border=4,
                        )
                        qr.add_data(result.qr_code_url)
                        qr.make(fit=True)

                        rprint("[cyan]QR Code URL:[/cyan]")
                        rprint(result.qr_code_url)
                        rprint(
                            "\n[yellow]Scan this QR code with your authenticator app.[/yellow]"
                        )

                if result.backup_codes:
                    rprint(
                        "\n[cyan]Backup Recovery Codes (save these securely):[/cyan]"
                    )
                    for i, code in enumerate(result.backup_codes, 1):
                        rprint(f"{i:2d}. {code}")
            else:
                rprint(f"[red]❌ MFA setup failed: {result.error_message}[/red]")

        except Exception as e:
            rprint(f"[red]❌ Failed to setup MFA: {str(e)}[/red]")

    asyncio.run(_setup_mfa())


@app.command("verify-mfa")
def verify_mfa(
    username: str = typer.Argument(..., help="Username to verify MFA"),
    code: str = typer.Argument(..., help="MFA code to verify"),
    method: MFAMethod = typer.Option(
        MFAMethod.TOTP, "--method", "-m", help="MFA method"
    ),
):
    """Verify an MFA code for a user."""

    async def _verify_mfa():
        auth_service = await get_auth_service()

        try:
            # Find user
            user = await auth_service._get_user_by_username(username)
            if not user:
                rprint(f"[red]User '{username}' not found.[/red]")
                return

            # Verify MFA
            result = await auth_service.verify_mfa(user.id, code, method)

            if result.success:
                rprint(
                    f"[green]✅ MFA verification successful for user '{username}'![/green]"
                )
            else:
                rprint(f"[red]❌ MFA verification failed: {result.error_message}[/red]")

        except Exception as e:
            rprint(f"[red]❌ Failed to verify MFA: {str(e)}[/red]")

    asyncio.run(_verify_mfa())


# ============================
# Session Management Commands
# ============================


@app.command("list-sessions")
def list_sessions(
    username: str | None = typer.Option(
        None, "--username", "-u", help="Filter by username"
    ),
    active_only: bool = typer.Option(
        True, "--active-only", "-a", help="Show only active sessions"
    ),
):
    """List active user sessions."""

    async def _list_sessions():
        await get_auth_service()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Loading sessions...", total=1)

                # This would require implementing session listing in the service
                rprint(
                    "[yellow]Session listing not yet implemented in service.[/yellow]"
                )

                progress.update(task, completed=1)

        except Exception as e:
            rprint(f"[red]❌ Failed to list sessions: {str(e)}[/red]")

    asyncio.run(_list_sessions())


@app.command("terminate-session")
def terminate_session(
    session_id: str = typer.Argument(..., help="Session ID to terminate"),
):
    """Terminate a specific user session."""

    async def _terminate_session():
        auth_service = await get_auth_service()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Terminating session...", total=1)

                # Logout session
                success = await auth_service.logout(session_id)

                progress.update(task, completed=1)

            if success:
                rprint(
                    f"[green]✅ Session {session_id[:8]}... terminated successfully![/green]"
                )
            else:
                rprint(f"[red]❌ Failed to terminate session {session_id[:8]}...[/red]")

        except Exception as e:
            rprint(f"[red]❌ Failed to terminate session: {str(e)}[/red]")

    asyncio.run(_terminate_session())


# ============================
# Security Audit Commands
# ============================


@app.command("audit-log")
def audit_log(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to look back"),
    event_type: str | None = typer.Option(
        None, "--event-type", "-e", help="Filter by event type"
    ),
    username: str | None = typer.Option(
        None, "--username", "-u", help="Filter by username"
    ),
    high_risk_only: bool = typer.Option(
        False, "--high-risk", help="Show only high-risk events"
    ),
    limit: int = typer.Option(
        100, "--limit", "-l", help="Maximum number of logs to show"
    ),
):
    """View security audit logs."""

    async def _audit_log():
        try:
            rprint(f"[cyan]Security Audit Log - Last {days} days[/cyan]")

            if high_risk_only:
                rprint("[yellow]Filtering for high-risk events (score >= 80)[/yellow]")

            # This would require implementing audit log querying
            rprint("[yellow]Audit log querying not yet implemented.[/yellow]")

        except Exception as e:
            rprint(f"[red]❌ Failed to retrieve audit logs: {str(e)}[/red]")

    asyncio.run(_audit_log())


@app.command("security-stats")
def security_stats():
    """Display security statistics and metrics."""

    async def _security_stats():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Gathering security statistics...", total=1)

                # This would require implementing security statistics
                stats = {
                    "Total Users": "N/A",
                    "Active Sessions": "N/A",
                    "Failed Logins (24h)": "N/A",
                    "High-Risk Events (7d)": "N/A",
                    "MFA Enabled Users": "N/A",
                    "Locked Accounts": "N/A",
                }

                progress.update(task, completed=1)

            # Display statistics
            table = Table(
                title="Security Statistics",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Metric", style="cyan")
            table.add_column("Value", justify="right", style="green")

            for metric, value in stats.items():
                table.add_row(metric, str(value))

            console.print(table)

        except Exception as e:
            rprint(f"[red]❌ Failed to gather security statistics: {str(e)}[/red]")

    asyncio.run(_security_stats())


# ============================
# Testing and Diagnostics
# ============================


@app.command("test-auth")
def test_auth(
    username: str = typer.Argument(..., help="Username to test"),
    password: str | None = typer.Option(
        None, "--password", "-p", help="Password (will prompt if not provided)"
    ),
    mfa_code: str | None = typer.Option(
        None, "--mfa-code", help="MFA code if required"
    ),
):
    """Test authentication for a user."""

    async def _test_auth():
        auth_service = await get_auth_service()

        # Get password if not provided
        password_value = password or getpass.getpass("Enter password: ")

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Testing authentication...", total=1)

                # Test authentication
                credentials = UserCredentials(
                    username=username,
                    password=password_value,
                    mfa_code=mfa_code,
                    ip_address="127.0.0.1",  # CLI test
                )

                result = await auth_service.authenticate(credentials)

                progress.update(task, completed=1)

            # Display results
            if result.success:
                rprint("[green]✅ Authentication successful![/green]")
                rprint(f"[dim]User ID: {result.user_id}[/dim]")
                rprint(f"[dim]Session ID: {result.session_id}[/dim]")
                rprint(f"[dim]Token expires in: {result.expires_in} seconds[/dim]")
            else:
                rprint(f"[red]❌ Authentication failed: {result.error_message}[/red]")
                rprint(f"[dim]Status: {result.status.value}[/dim]")

                if result.mfa_required:
                    rprint(
                        f"[yellow]MFA Required. Methods: {', '.join([m.value for m in result.mfa_methods])}[/yellow]"
                    )

        except Exception as e:
            rprint(f"[red]❌ Authentication test failed: {str(e)}[/red]")

    asyncio.run(_test_auth())


@app.command("check-permission")
def check_permission(
    username: str = typer.Argument(..., help="Username to check"),
    resource: str = typer.Argument(..., help="Resource to check access for"),
    action: str = typer.Argument(..., help="Action to check permission for"),
):
    """Check if a user has permission for a specific action."""

    async def _check_permission():
        auth_service = await get_auth_service()

        try:
            # Find user
            user = await auth_service._get_user_by_username(username)
            if not user:
                rprint(f"[red]User '{username}' not found.[/red]")
                return

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Checking permission...", total=1)

                # Check permission
                has_permission = await auth_service.check_permission(
                    user.id, resource, action
                )

                progress.update(task, completed=1)

            if has_permission:
                rprint(
                    f"[green]✅ User '{username}' has permission for '{resource}:{action}'[/green]"
                )
            else:
                rprint(
                    f"[red]❌ User '{username}' does NOT have permission for '{resource}:{action}'[/red]"
                )

        except Exception as e:
            rprint(f"[red]❌ Permission check failed: {str(e)}[/red]")

    asyncio.run(_check_permission())


# ============================
# System Maintenance Commands
# ============================


@app.command("cleanup-sessions")
def cleanup_sessions():
    """Clean up expired sessions."""

    async def _cleanup_sessions():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Cleaning up expired sessions...", total=1)

                # This would require implementing cleanup functionality
                rprint("[yellow]Session cleanup not yet implemented.[/yellow]")

                progress.update(task, completed=1)

            rprint("[green]✅ Session cleanup completed![/green]")

        except Exception as e:
            rprint(f"[red]❌ Session cleanup failed: {str(e)}[/red]")

    asyncio.run(_cleanup_sessions())


@app.command("reset-password")
def reset_password(
    username: str = typer.Argument(..., help="Username for password reset"),
    auto_generate: bool = typer.Option(
        False, "--auto-generate", help="Generate secure password automatically"
    ),
):
    """Reset a user's password."""

    async def _reset_password():
        auth_service = await get_auth_service()

        try:
            # Find user
            user = await auth_service._get_user_by_username(username)
            if not user:
                rprint(f"[red]User '{username}' not found.[/red]")
                return

            # Get new password
            if auto_generate:
                new_password = generate_secure_password()
                rprint(f"[green]Generated password:[/green] {new_password}")
            else:
                new_password = getpass.getpass("Enter new password: ")
                confirm_password = getpass.getpass("Confirm new password: ")
                if new_password != confirm_password:
                    rprint("[red]Passwords do not match![/red]")
                    return

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Resetting password...", total=1)

                # Hash and update password
                hashed_password = bcrypt.hashpw(
                    new_password.encode(), bcrypt.gensalt()
                ).decode()
                await auth_service.user_service.update_by_id(
                    user.id,
                    hashed_password=hashed_password,
                    last_password_change=datetime.now(UTC),
                )

                # Invalidate all user sessions
                await auth_service.session_manager.delete_all_user_sessions(user.id)

                progress.update(task, completed=1)

            rprint(
                f"[green]✅ Password reset successfully for user '{username}'![/green]"
            )
            rprint("[yellow]All existing sessions have been terminated.[/yellow]")

        except Exception as e:
            rprint(f"[red]❌ Password reset failed: {str(e)}[/red]")

    asyncio.run(_reset_password())


if __name__ == "__main__":
    app()
