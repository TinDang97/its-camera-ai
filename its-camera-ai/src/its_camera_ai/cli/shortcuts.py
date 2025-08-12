"""Command shortcuts and aliases for ITS Camera AI CLI.

Provides convenient shortcuts, aliases, and quick access commands for
common operations with intuitive naming and developer-friendly patterns.
"""

import subprocess
import sys
from pathlib import Path

import typer
from rich.table import Table

from .utils import console, print_error, print_info, print_success, print_warning


class ShortcutManager:
    """Manages CLI shortcuts and aliases."""

    def __init__(self):
        self.shortcuts = {
            # Status and monitoring shortcuts
            "s": ("status", "Show quick system status"),
            "st": ("status", "Show quick system status"),
            "stat": ("status", "Show quick system status"),
            "h": ("dashboard health", "Run comprehensive health check"),
            "health": ("dashboard health", "Run comprehensive health check"),
            "m": ("dashboard monitor --duration 30", "Monitor system for 30 seconds"),
            "mon": ("dashboard monitor --duration 30", "Monitor system for 30 seconds"),

            # ML operations shortcuts
            "train": ("ml train", "Start model training"),
            "inf": ("ml inference", "Run model inference"),
            "deploy": ("ml deploy", "Deploy a model"),
            "models": ("ml models", "List available models"),
            "bench": ("ml benchmark", "Benchmark model performance"),

            # Service management shortcuts
            "up": ("service start", "Start all services"),
            "down": ("service stop", "Stop all services"),
            "restart": ("service restart", "Restart all services"),
            "ps": ("service status", "Show service status"),
            "logs": ("service logs", "Show service logs"),

            # Configuration shortcuts
            "conf": ("config get", "Show current configuration"),
            "cfg": ("config get", "Show current configuration"),
            "profile": ("profiles current", "Show current profile"),
            "prof": ("profiles current", "Show current profile"),
            "env": ("profiles list", "List available profiles"),

            # Interactive and help shortcuts
            "i": ("interactive start", "Start interactive mode"),
            "wizard": ("interactive start", "Start interactive mode"),
            "help": ("--help", "Show help information"),
            "?": ("--help", "Show help information"),

            # Monitoring and logging shortcuts
            "tail": ("logging tail", "Show recent log entries"),
            "debug": ("logging debug", "Toggle debug mode"),
            "clear-logs": ("logging clear --force", "Clear all log files"),

            # Plugin shortcuts
            "plugins": ("plugins list", "List available plugins"),
            "plug": ("plugins list", "List available plugins"),

            # Development shortcuts
            "dev": ("profiles switch development", "Switch to development profile"),
            "prod": ("profiles switch production", "Switch to production profile"),
            "stage": ("profiles switch staging", "Switch to staging profile"),
        }

        # Quick commands that execute immediately
        self.quick_commands = {
            "version": self._show_version,
            "v": self._show_version,
            "info": self._show_info,
            "uptime": self._show_uptime,
            "cpu": self._show_cpu_usage,
            "mem": self._show_memory_usage,
            "disk": self._show_disk_usage,
        }

    def get_shortcut(self, alias: str) -> tuple[str, str] | None:
        """Get the command and description for a shortcut."""
        if alias in self.shortcuts:
            return self.shortcuts[alias]
        return None

    def list_shortcuts(self) -> dict[str, tuple[str, str]]:
        """List all available shortcuts."""
        return self.shortcuts.copy()

    def execute_quick_command(self, command: str) -> bool:
        """Execute a quick command if it exists."""
        if command in self.quick_commands:
            self.quick_commands[command]()
            return True
        return False

    def _show_version(self) -> None:
        """Show version information."""
        try:
            from ..__init__ import __version__
            console.print(f"[bold green]ITS Camera AI[/bold green] v[bold]{__version__}[/bold]")
        except ImportError:
            console.print("[bold green]ITS Camera AI[/bold green] v[bold]0.1.0[/bold]")

    def _show_info(self) -> None:
        """Show basic system information."""
        import platform

        info_table = Table(title="System Information")
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="green")

        info_table.add_row("Platform", platform.platform())
        info_table.add_row("Python", sys.version.split()[0])
        info_table.add_row("Working Directory", str(Path.cwd()))

        console.print(info_table)

    def _show_uptime(self) -> None:
        """Show system uptime."""
        try:
            import psutil
            boot_time = psutil.boot_time()
            uptime_seconds = psutil.time.time() - boot_time

            days = int(uptime_seconds // 86400)
            hours = int((uptime_seconds % 86400) // 3600)
            minutes = int((uptime_seconds % 3600) // 60)

            uptime_str = f"{days}d {hours}h {minutes}m"
            console.print(f"[bold cyan]System Uptime:[/bold cyan] [green]{uptime_str}[/green]")
        except ImportError:
            print_error("psutil not available for uptime information")

    def _show_cpu_usage(self) -> None:
        """Show current CPU usage."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            status_icon = "🟢" if cpu_percent < 70 else "🟡" if cpu_percent < 90 else "🔴"

            console.print(f"[bold cyan]CPU Usage:[/bold cyan] {status_icon} [green]{cpu_percent:.1f}%[/green] ({cpu_count} cores)")
        except ImportError:
            print_error("psutil not available for CPU information")

    def _show_memory_usage(self) -> None:
        """Show current memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()

            status_icon = "🟢" if memory.percent < 70 else "🟡" if memory.percent < 90 else "🔴"

            from .utils import format_bytes
            console.print(f"[bold cyan]Memory Usage:[/bold cyan] {status_icon} [green]{memory.percent:.1f}%[/green] ({format_bytes(memory.used)}/{format_bytes(memory.total)})")
        except ImportError:
            print_error("psutil not available for memory information")

    def _show_disk_usage(self) -> None:
        """Show current disk usage."""
        try:
            import psutil
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100

            status_icon = "🟢" if disk_percent < 80 else "🟡" if disk_percent < 95 else "🔴"

            from .utils import format_bytes
            console.print(f"[bold cyan]Disk Usage:[/bold cyan] {status_icon} [green]{disk_percent:.1f}%[/green] ({format_bytes(disk.used)}/{format_bytes(disk.total)})")
        except ImportError:
            print_error("psutil not available for disk information")


# Global shortcut manager
shortcut_manager = ShortcutManager()


def execute_shortcut(alias: str, args: list[str] = None) -> bool:
    """Execute a shortcut command."""
    args = args or []

    # First check quick commands
    if shortcut_manager.execute_quick_command(alias):
        return True

    # Then check regular shortcuts
    shortcut_info = shortcut_manager.get_shortcut(alias)
    if shortcut_info:
        command, _ = shortcut_info

        # Build full command
        full_command = ["its-camera-ai"] + command.split() + args

        try:
            # Execute the command
            result = subprocess.run(full_command, check=False)
            return result.returncode == 0
        except FileNotFoundError:
            print_error("its-camera-ai command not found in PATH")
            return False
        except Exception as e:
            print_error(f"Failed to execute shortcut: {e}")
            return False

    return False


app = typer.Typer(help="⚡ Command shortcuts and aliases")


@app.command("list")
def list_shortcuts() -> None:
    """List all available shortcuts and aliases."""
    shortcuts = shortcut_manager.list_shortcuts()

    # Group shortcuts by category
    categories = {
        "Status & Monitoring": ["s", "st", "stat", "h", "health", "m", "mon"],
        "ML Operations": ["train", "inf", "deploy", "models", "bench"],
        "Service Management": ["up", "down", "restart", "ps", "logs"],
        "Configuration": ["conf", "cfg", "profile", "prof", "env"],
        "Interactive": ["i", "wizard", "help", "?"],
        "Logging": ["tail", "debug", "clear-logs"],
        "Plugins": ["plugins", "plug"],
        "Profiles": ["dev", "prod", "stage"],
    }

    for category, alias_list in categories.items():
        if not alias_list:
            continue

        table = Table(title=category)
        table.add_column("Shortcut", style="cyan", width=12)
        table.add_column("Command", style="green", width=30)
        table.add_column("Description", style="dim")

        for alias in alias_list:
            if alias in shortcuts:
                command, description = shortcuts[alias]
                table.add_row(alias, command, description)

        console.print(table)
        console.print()

    # Show quick commands
    quick_table = Table(title="Quick Commands")
    quick_table.add_column("Command", style="cyan")
    quick_table.add_column("Description", style="green")

    quick_commands = {
        "version, v": "Show version information",
        "info": "Show basic system information",
        "uptime": "Show system uptime",
        "cpu": "Show CPU usage",
        "mem": "Show memory usage",
        "disk": "Show disk usage",
    }

    for cmd, desc in quick_commands.items():
        quick_table.add_row(cmd, desc)

    console.print(quick_table)


@app.command("exec")
def execute_shortcut_command(
    alias: str = typer.Argument(..., help="Shortcut alias to execute"),
    args: list[str] = typer.Argument(None, help="Additional arguments"),
) -> None:
    """Execute a shortcut command."""
    if execute_shortcut(alias, args):
        print_success(f"Executed shortcut: {alias}")
    else:
        print_error(f"Unknown shortcut or execution failed: {alias}")


@app.command("add")
def add_shortcut(
    alias: str = typer.Argument(..., help="Shortcut alias"),
    command: str = typer.Argument(..., help="Command to execute"),
    description: str = typer.Option("", "--description", "-d", help="Shortcut description"),
) -> None:
    """Add a custom shortcut."""
    # For now, just show what would be added
    # In a full implementation, you'd store custom shortcuts in a config file
    print_info(f"Would add shortcut: {alias} -> {command}")
    if description:
        print_info(f"Description: {description}")
    print_warning("Custom shortcuts not yet implemented - coming soon!")


@app.command("remove")
def remove_shortcut(
    alias: str = typer.Argument(..., help="Shortcut alias to remove"),
) -> None:
    """Remove a custom shortcut."""
    print_info(f"Would remove shortcut: {alias}")
    print_warning("Custom shortcuts not yet implemented - coming soon!")


@app.command("help")
def show_shortcut_help(
    alias: str = typer.Argument(..., help="Shortcut alias to get help for"),
) -> None:
    """Show help for a specific shortcut."""
    shortcut_info = shortcut_manager.get_shortcut(alias)
    if shortcut_info:
        command, description = shortcut_info

        help_table = Table(title=f"Shortcut: {alias}")
        help_table.add_column("Property", style="cyan")
        help_table.add_column("Value", style="green")

        help_table.add_row("Alias", alias)
        help_table.add_row("Command", command)
        help_table.add_row("Description", description)

        console.print(help_table)

        # Show example usage
        console.print("\n[bold]Example usage:[/bold]")
        console.print(f"  [cyan]its-camera-ai shortcuts exec {alias}[/cyan]")
        console.print(f"  [dim]or directly:[/dim] [cyan]{alias}[/cyan] [dim](if using shell integration)[/dim]")

    elif alias in shortcut_manager.quick_commands:
        console.print(f"[bold cyan]{alias}[/bold cyan] is a quick command that executes immediately")
        shortcut_manager.execute_quick_command(alias)
    else:
        print_error(f"Unknown shortcut: {alias}")


# Development shortcuts as standalone commands
@app.command("dev")
def quick_dev_shortcuts() -> None:
    """Show common development shortcuts."""
    dev_shortcuts = {
        "Status Check": "its-camera-ai s",
        "Health Check": "its-camera-ai h",
        "Start Services": "its-camera-ai up",
        "View Logs": "its-camera-ai tail",
        "Interactive Mode": "its-camera-ai i",
        "Switch to Dev": "its-camera-ai dev",
        "Debug Mode": "its-camera-ai debug",
    }

    table = Table(title="🚀 Quick Development Commands")
    table.add_column("Task", style="cyan", width=20)
    table.add_column("Command", style="green")

    for task, command in dev_shortcuts.items():
        table.add_row(task, command)

    console.print(table)
    console.print("\n[dim]Tip: Use these shortcuts to quickly perform common development tasks[/dim]")


if __name__ == "__main__":
    app()
