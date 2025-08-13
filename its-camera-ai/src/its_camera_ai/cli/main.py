"""Main CLI entry point for ITS Camera AI.

Comprehensive command-line interface for managing the ITS Camera AI system.
Provides commands for service management, ML operations, data pipeline control,
security management, monitoring, and configuration.
"""

import sys
from pathlib import Path

import typer
from rich.panel import Panel
from rich.text import Text

# Import enhanced CLI features
from . import (
    completion,
    dashboard,
    history,
    interactive,
    plugins,
    profiles,
    shortcuts,
)
from . import (
    logging as cli_logging,
)

# Import CLI command groups
from .commands import (
    auth as auth_commands,
)
from .commands import (
    config as config_commands,
)
from .commands import (
    data as data_commands,
)
from .commands import (
    ml as ml_commands,
)
from .commands import (
    monitoring as monitoring_commands,
)
from .commands import (
    security as security_commands,
)
from .commands import (
    services as service_commands,
)
from .utils import (
    console,
    print_info,
    print_success,
    print_warning,
    setup_logging,
    show_banner,
)

# Create main CLI app
app = typer.Typer(
    name="its-camera-ai",
    help="🎥 ITS Camera AI - Intelligent Traffic Monitoring System",
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    add_completion=True,
)

# Add command groups
app.add_typer(
    service_commands.app, name="service", help="🚀 Service management commands"
)
app.add_typer(
    auth_commands.app, name="auth", help="🔐 Authentication and user management"
)
app.add_typer(ml_commands.app, name="ml", help="🤖 ML operations and model management")
app.add_typer(data_commands.app, name="data", help="📊 Data pipeline control")
app.add_typer(
    security_commands.app, name="security", help="🔒 Security and authentication"
)
app.add_typer(
    monitoring_commands.app, name="monitor", help="📈 System monitoring and health"
)
app.add_typer(config_commands.app, name="config", help="⚙️ Configuration management")

# Add enhanced CLI features
app.add_typer(completion.app, name="completion", help="🔧 Shell completion management")
app.add_typer(
    dashboard.app, name="dashboard", help="📊 Interactive dashboard and monitoring"
)
app.add_typer(history.app, name="history", help="📚 Command history and favorites")
app.add_typer(
    interactive.app, name="interactive", help="🧙‍♂️ Interactive mode and wizards"
)
app.add_typer(cli_logging.app, name="logging", help="📋 Advanced logging and debugging")
app.add_typer(plugins.app, name="plugin", help="🔌 Plugin management system")
app.add_typer(profiles.app, name="profile", help="🎭 Configuration profile management")
app.add_typer(shortcuts.app, name="shortcuts", help="⚡ Command shortcuts and aliases")


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress non-error output"
    ),
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
) -> None:
    """🎥 ITS Camera AI - Intelligent Traffic Monitoring System.

    A comprehensive AI-powered camera traffic monitoring system for real-time
    traffic analytics and vehicle tracking using computer vision.

    Key Features:
    • Real-time video processing with YOLO11 models
    • Edge-cloud hybrid architecture
    • Zero-trust security framework
    • MLOps integration with automated CI/CD
    • Comprehensive monitoring and analytics
    """
    # Store global options in context
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["config_file"] = config_file

    # Setup logging based on verbosity
    log_level = "DEBUG" if verbose else "WARNING" if quiet else "INFO"
    setup_logging(log_level)

    # Show banner only for main command (no subcommand)
    if ctx.invoked_subcommand is None:
        show_banner()

        # Load plugins automatically
        try:
            from .plugins import plugin_manager

            plugin_count = plugin_manager.load_all_plugins()
            if plugin_count > 0:
                print_info(f"Loaded {plugin_count} plugins")
        except Exception as e:
            if verbose:
                print_warning(f"Failed to load plugins: {e}")


@app.command()
def info() -> None:
    """📋 Show system information and status."""
    from ..__init__ import __version__
    from ..core.config import get_settings

    settings = get_settings()

    # Create info panel
    info_text = Text()
    info_text.append("Version: ", style="bold")
    info_text.append(f"{__version__}\n", style="green")
    info_text.append("Environment: ", style="bold")
    info_text.append(f"{settings.environment}\n", style="blue")
    info_text.append("Python: ", style="bold")
    info_text.append(f"{sys.version.split()[0]}\n", style="cyan")
    info_text.append("Platform: ", style="bold")
    info_text.append(f"{sys.platform}\n", style="magenta")

    console.print(
        Panel(
            info_text,
            title="🎥 ITS Camera AI System Information",
            border_style="blue",
        )
    )


@app.command()
def version() -> None:
    """📦 Show version information."""
    from ..__init__ import __version__

    console.print(
        f"[bold green]ITS Camera AI[/bold green] version [bold]{__version__}[/bold]"
    )


@app.command()
def status() -> None:
    """⚡ Show quick system status overview."""
    from .dashboard import QuickStatusChecker

    QuickStatusChecker.show_compact_status()


@app.command()
def setup() -> None:
    """🚀 Quick setup wizard for new installations."""
    print_info("Starting quick setup wizard...")

    # Install shell completion
    try:
        from .completion import detect_shell, install_completion

        shell = detect_shell()
        if shell:
            print_info(f"Installing shell completion for {shell}...")
            if install_completion(shell):
                print_success("Shell completion installed!")
        else:
            print_warning("Could not detect shell for completion installation")
    except Exception as e:
        print_warning(f"Shell completion setup failed: {e}")

    # Set up default profile
    try:
        from .profiles import profile_manager

        current = profile_manager.get_current_profile()
        print_success(f"Using profile: {current}")
    except Exception as e:
        print_warning(f"Profile setup failed: {e}")

    print_success(
        "Setup completed! Try 'its-camera-ai interactive start' for guided workflows."
    )


if __name__ == "__main__":
    app()
