"""CLI utilities for ITS Camera AI.

Common utilities and helpers for the CLI interface.
"""

import logging
import sys
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

# Global console instance
console = Console()


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging with Rich formatting.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def show_banner() -> None:
    """
    Display the ITS Camera AI banner.
    """
    banner_text = Text()
    banner_text.append("🎥 ITS Camera AI", style="bold blue")
    banner_text.append("\n")
    banner_text.append("Intelligent Traffic Monitoring System", style="italic cyan")
    banner_text.append("\n\n")
    banner_text.append("Real-time AI-powered traffic analytics and vehicle tracking", style="dim")

    console.print(Panel(
        banner_text,
        border_style="blue",
        padding=(1, 2),
    ))

    console.print("\n[dim]Use [bold]--help[/bold] to see available commands[/dim]\n")


def create_progress() -> Progress:
    """
    Create a Rich progress bar for long-running operations.

    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def create_status_table() -> Table:
    """
    Create a table for displaying service status.

    Returns:
        Configured Table instance
    """
    table = Table(title="Service Status")
    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Health", style="yellow")
    table.add_column("Uptime", style="blue")
    table.add_column("Details", style="dim")
    return table


def create_metrics_table() -> Table:
    """
    Create a table for displaying metrics.

    Returns:
        Configured Table instance
    """
    table = Table(title="System Metrics")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Unit", style="yellow")
    table.add_column("Status", style="blue")
    table.add_column("Threshold", style="dim")
    return table


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable format.

    Args:
        bytes_value: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1h 30m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


def format_percentage(value: float, total: float) -> str:
    """
    Format a percentage value.

    Args:
        value: Current value
        total: Total value

    Returns:
        Formatted percentage string
    """
    if total == 0:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"


def get_status_style(status: str) -> str:
    """
    Get Rich style for status value.

    Args:
        status: Status string

    Returns:
        Rich style string
    """
    status_lower = status.lower()
    if status_lower in ["running", "healthy", "active", "connected", "ok"]:
        return "green"
    elif status_lower in ["starting", "loading", "connecting", "warning"]:
        return "yellow"
    elif status_lower in ["stopped", "error", "failed", "disconnected", "critical"]:
        return "red"
    else:
        return "blue"


def print_error(message: str, exit_code: int = 1) -> None:
    """
    Print error message and exit.

    Args:
        message: Error message
        exit_code: Exit code (default: 1)
    """
    console.print(f"[bold red]Error:[/bold red] {message}", err=True)
    sys.exit(exit_code)


def print_warning(message: str) -> None:
    """
    Print warning message.

    Args:
        message: Warning message
    """
    console.print(f"[bold yellow]Warning:[/bold yellow] {message}")


def print_success(message: str) -> None:
    """
    Print success message.

    Args:
        message: Success message
    """
    console.print(f"[bold green]Success:[/bold green] {message}")


def print_info(message: str) -> None:
    """
    Print info message.

    Args:
        message: Info message
    """
    console.print(f"[bold blue]Info:[/bold blue] {message}")


def confirm_action(message: str, default: bool = False) -> bool:
    """
    Ask user for confirmation.

    Args:
        message: Confirmation message
        default: Default value if user just presses Enter

    Returns:
        True if confirmed, False otherwise
    """
    default_str = "Y/n" if default else "y/N"
    prompt = f"[bold yellow]?[/bold yellow] {message} [{default_str}]: "

    response = console.input(prompt).strip().lower()

    if not response:
        return default

    return response in ["y", "yes", "true", "1"]


def display_config(config: dict[str, Any], title: str = "Configuration") -> None:
    """
    Display configuration in a formatted table.

    Args:
        config: Configuration dictionary
        title: Table title
    """
    table = Table(title=title)
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    def add_config_items(items: dict[str, Any], prefix: str = "") -> None:
        for key, value in items.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                add_config_items(value, full_key)
            else:
                # Mask sensitive values
                if any(sensitive in key.lower() for sensitive in ["password", "secret", "key", "token"]):
                    display_value = "***HIDDEN***"
                else:
                    display_value = str(value)

                table.add_row(full_key, display_value)

    add_config_items(config)
    console.print(table)


class CLIError(Exception):
    """
    Custom CLI error for better error handling.
    """
    pass


def handle_async_command(func):
    """
    Decorator to handle async commands in Typer.

    Args:
        func: Async function to wrap

    Returns:
        Wrapped synchronous function
    """
    import asyncio
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return asyncio.run(func(*args, **kwargs))
        except KeyboardInterrupt:
            print_warning("Operation cancelled by user")
            sys.exit(130)  # SIGINT exit code
        except Exception as e:
            print_error(f"Command failed: {e}")

    return wrapper
