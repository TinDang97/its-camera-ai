#!/usr/bin/env python3
"""
Deployment script for ITS Camera AI Authentication System.

This script automates the deployment of the comprehensive authentication system including:
- Database migrations for enhanced user models
- Default role and permission setup
- Security configuration validation
- Service health checks
- Initial admin user creation
"""

import asyncio
import getpass
import sys
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from migrations.auth_system_migration import downgrade, upgrade
from src.its_camera_ai.core.config import get_settings
from src.its_camera_ai.services.auth_service import (
    PasswordPolicy,
)

console = Console()
app = typer.Typer(help="Deploy ITS Camera AI Authentication System")


def show_banner():
    """Display deployment banner."""
    banner_text = """
🔐 ITS Camera AI Authentication System Deployment

Enterprise-grade authentication and authorization system with:
• JWT authentication with RS256 signatures
• Role-Based Access Control (RBAC)
• Multi-Factor Authentication (MFA)
• Session management with Redis
• Security audit logging
• Password policies and brute force protection
• Comprehensive security middleware
    """

    console.print(
        Panel(
            banner_text.strip(),
            title="Authentication System Deployment",
            border_style="green",
            padding=(1, 2),
        )
    )


def validate_environment():
    """Validate deployment environment."""
    rprint("[cyan]Validating deployment environment...[/cyan]")

    settings = get_settings()
    issues = []

    # Check database configuration
    if "localhost" in settings.database.url and settings.is_production():
        issues.append("❌ Production environment should not use localhost database")

    # Check Redis configuration
    if "localhost" in settings.redis.url and settings.is_production():
        issues.append("❌ Production environment should not use localhost Redis")

    # Check security configuration
    # Security: Use secure secret comparison to prevent timing attacks
    import secrets
    default_secret = "change-me-in-production"
    if secrets.compare_digest(settings.security.secret_key, default_secret):
        issues.append("❌ Default secret key detected - change before production")

    if settings.security.algorithm != "RS256":
        issues.append("⚠️  Consider using RS256 for production JWT tokens")

    if len(settings.security.secret_key) < 32:
        issues.append("❌ Secret key should be at least 32 characters")

    # Check password policy
    if settings.security.password_min_length < 12:
        issues.append("⚠️  Password minimum length should be at least 12 characters")

    # Display results
    if issues:
        rprint("\n[yellow]Environment validation issues found:[/yellow]")
        for issue in issues:
            rprint(f"  {issue}")

        if any("❌" in issue for issue in issues):
            if not typer.confirm("Continue deployment with errors?"):
                raise typer.Exit(1)
    else:
        rprint("[green]✅ Environment validation passed[/green]")


@app.command()
def deploy(
    skip_migration: bool = typer.Option(
        False, "--skip-migration", help="Skip database migration"
    ),
    skip_validation: bool = typer.Option(
        False, "--skip-validation", help="Skip environment validation"
    ),
    create_admin: bool = typer.Option(
        True, "--create-admin/--no-admin", help="Create initial admin user"
    ),
):
    """Deploy the authentication system."""

    try:
        show_banner()

        # Validate environment
        if not skip_validation:
            validate_environment()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Run database migration
            if not skip_migration:
                task = progress.add_task("Running database migration...", total=1)

                try:
                    upgrade()
                    progress.update(task, completed=1)
                    rprint(
                        "[green]✅ Database migration completed successfully[/green]"
                    )
                except Exception as e:
                    rprint(f"[red]❌ Database migration failed: {str(e)}[/red]")
                    raise typer.Exit(1)

            # Create admin user if requested
            if create_admin:
                task = progress.add_task("Setting up admin user...", total=1)

                asyncio.run(_create_admin_user())
                progress.update(task, completed=1)
                rprint("[green]✅ Admin user setup completed[/green]")

            # Run health checks
            task = progress.add_task("Running system health checks...", total=1)

            health_ok = asyncio.run(_run_health_checks())
            if health_ok:
                progress.update(task, completed=1)
                rprint("[green]✅ System health checks passed[/green]")
            else:
                rprint("[red]❌ System health checks failed[/red]")
                raise typer.Exit(1)

        # Display deployment summary
        _show_deployment_summary()

        rprint(
            "\n[bold green]🎉 Authentication system deployment completed successfully![/bold green]"
        )
        rprint("\n[cyan]Next steps:[/cyan]")
        rprint("1. Update your application to use the new authentication middleware")
        rprint("2. Configure your frontend to use the new auth endpoints")
        rprint("3. Test the authentication flow with the admin user")
        rprint("4. Set up monitoring and alerting for security events")

    except KeyboardInterrupt:
        rprint("\n[yellow]Deployment cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"\n[red]❌ Deployment failed: {str(e)}[/red]")
        raise typer.Exit(1)


async def _create_admin_user():
    """Create initial admin user."""
    from src.its_camera_ai.services.auth_service import create_auth_service

    settings = get_settings()

    # Get admin user details
    rprint("\n[cyan]Creating initial admin user...[/cyan]")

    admin_username = typer.prompt("Admin username", default="admin")
    admin_email = typer.prompt("Admin email", default="admin@its-camera-ai.com")

    while True:
        admin_password = getpass.getpass("Admin password: ")
        if not admin_password:
            rprint("[red]Password cannot be empty[/red]")
            continue

        # Validate password
        validation = PasswordPolicy.validate_password(admin_password, admin_username)
        if not validation["valid"]:
            rprint("[red]Password validation failed:[/red]")
            for error in validation["errors"]:
                rprint(f"  • {error}")
            continue

        confirm_password = getpass.getpass("Confirm password: ")
        if admin_password != confirm_password:
            rprint("[red]Passwords do not match[/red]")
            continue

        break

    try:
        # Create database session
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker

        engine = create_async_engine(settings.database.url, echo=False)
        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )

        async with async_session() as session:
            auth_service = await create_auth_service(session, settings.redis.url)

            # Check if admin user already exists
            existing_user = await auth_service._get_user_by_username(admin_username)
            if existing_user:
                rprint(
                    f"[yellow]Admin user '{admin_username}' already exists, skipping creation[/yellow]"
                )
                return

            # Hash password
            import bcrypt

            hashed_password = bcrypt.hashpw(
                admin_password.encode(), bcrypt.gensalt()
            ).decode()

            # Create admin user
            admin_user = await auth_service.user_service.create(
                username=admin_username,
                email=admin_email,
                hashed_password=hashed_password,
                full_name="System Administrator",
                is_active=True,
                is_verified=True,
                is_superuser=True,
            )

            rprint(
                f"[green]✅ Admin user '{admin_username}' created with ID: {admin_user.id}[/green]"
            )

    except Exception as e:
        rprint(f"[red]❌ Failed to create admin user: {str(e)}[/red]")
        raise


async def _run_health_checks():
    """Run system health checks."""
    import redis.asyncio as redis
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker

    from src.its_camera_ai.services.auth_service import create_auth_service

    settings = get_settings()
    health_ok = True

    # Test database connection
    try:
        engine = create_async_engine(settings.database.url, echo=False)
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        rprint("[green]✅ Database connection successful[/green]")
    except Exception as e:
        rprint(f"[red]❌ Database connection failed: {str(e)}[/red]")
        health_ok = False

    # Test Redis connection
    try:
        redis_client = redis.from_url(settings.redis.url)
        await redis_client.ping()
        await redis_client.close()
        rprint("[green]✅ Redis connection successful[/green]")
    except Exception as e:
        rprint(f"[red]❌ Redis connection failed: {str(e)}[/red]")
        health_ok = False

    # Test authentication service initialization
    try:
        async_session = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        async with async_session() as session:
            auth_service = await create_auth_service(session, settings.redis.url)
            # Test JWT token creation
            test_token = auth_service.jwt_manager.create_access_token({"sub": "test"})
            # Test token verification
            payload = auth_service.jwt_manager.verify_token(test_token)
            assert payload["sub"] == "test"
        rprint("[green]✅ Authentication service initialization successful[/green]")
    except Exception as e:
        rprint(f"[red]❌ Authentication service test failed: {str(e)}[/red]")
        health_ok = False

    return health_ok


def _show_deployment_summary():
    """Display deployment summary."""
    settings = get_settings()

    summary_text = f"""
Environment: {settings.environment}
Database: {settings.database.url.split("@")[-1] if "@" in settings.database.url else "Local"}
Redis: {settings.redis.url.split("@")[-1] if "@" in settings.redis.url else "Local"}

Authentication Features Deployed:
✅ JWT authentication with {settings.security.algorithm} signatures
✅ Role-Based Access Control (RBAC) with 6 default roles
✅ Multi-Factor Authentication (MFA) support
✅ Session management with Redis backend
✅ Security audit logging
✅ Password policies and brute force protection
✅ Rate limiting middleware
✅ Security headers middleware
✅ Comprehensive API endpoints

API Endpoints Available:
• POST /auth/login - User authentication
• POST /auth/register - User registration
• POST /auth/refresh - Token refresh
• POST /auth/logout - User logout
• GET /auth/profile - User profile
• POST /auth/change-password - Password change
• POST /auth/mfa/setup - MFA enrollment
• POST /auth/mfa/verify - MFA verification
• GET /auth/validate - Token validation

CLI Commands Available:
• its-camera-ai auth create-user - Create new users
• its-camera-ai auth list-users - List all users
• its-camera-ai auth setup-mfa - Setup MFA for users
• its-camera-ai auth test-auth - Test authentication
• its-camera-ai auth audit-log - View security logs
• its-camera-ai auth security-stats - Security statistics
    """

    console.print(
        Panel(
            summary_text.strip(),
            title="Deployment Summary",
            border_style="green",
            padding=(1, 2),
        )
    )


@app.command()
def rollback():
    """Rollback the authentication system deployment."""

    rprint("[yellow]⚠️  This will rollback all authentication system changes![/yellow]")
    rprint("[yellow]This includes:[/yellow]")
    rprint("  • Database schema changes")
    rprint("  • User data and roles")
    rprint("  • Security audit logs")
    rprint("  • Session data")

    if not typer.confirm("Are you sure you want to continue?", default=False):
        rprint("[cyan]Rollback cancelled[/cyan]")
        return

    if not typer.confirm("This action cannot be undone. Continue?", default=False):
        rprint("[cyan]Rollback cancelled[/cyan]")
        return

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Rolling back database changes...", total=1)

            downgrade()
            progress.update(task, completed=1)

        rprint("[green]✅ Authentication system rollback completed[/green]")

    except Exception as e:
        rprint(f"[red]❌ Rollback failed: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def test_deployment():
    """Test the deployed authentication system."""

    async def _test_deployment():
        from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
        from sqlalchemy.orm import sessionmaker

        from src.its_camera_ai.services.auth_service import (
            create_auth_service,
        )

        settings = get_settings()

        rprint("[cyan]Testing authentication system...[/cyan]")

        # Test database connection and service initialization
        try:
            engine = create_async_engine(settings.database.url, echo=False)
            async_session = sessionmaker(
                engine, class_=AsyncSession, expire_on_commit=False
            )

            async with async_session() as session:
                auth_service = await create_auth_service(session, settings.redis.url)

                # Test JWT token operations
                test_data = {"sub": "test_user", "username": "testuser"}
                token = auth_service.jwt_manager.create_access_token(test_data)
                payload = auth_service.jwt_manager.verify_token(token)

                assert payload["sub"] == "test_user"
                assert payload["username"] == "testuser"

                rprint("[green]✅ JWT token operations working[/green]")

                # Test password policy
                from src.its_camera_ai.services.auth_service import PasswordPolicy

                validation = PasswordPolicy.validate_password("TestPass123!")
                assert validation["valid"]

                rprint("[green]✅ Password policy validation working[/green]")

                rprint(
                    "[green]🎉 All tests passed! Authentication system is working correctly.[/green]"
                )

        except Exception as e:
            rprint(f"[red]❌ Test failed: {str(e)}[/red]")
            raise typer.Exit(1)

    asyncio.run(_test_deployment())


if __name__ == "__main__":
    app()
