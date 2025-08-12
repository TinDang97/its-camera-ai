"""Advanced logging and debugging for ITS Camera AI CLI.

Provides structured logging, debug modes, performance profiling, and
comprehensive error tracking with rich formatting and file rotation.
"""

import logging
import logging.handlers
import os
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psutil
import typer
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from .utils import (
    console,
    format_bytes,
    format_duration,
    print_error,
    print_info,
    print_success,
)


class PerformanceProfiler:
    """Performance profiler for CLI operations."""

    def __init__(self):
        self.measurements: dict[str, dict[str, Any]] = {}
        self.process = psutil.Process()

    @contextmanager
    def profile(self, operation_name: str) -> Generator[None, None, None]:
        """Profile a specific operation."""
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss
        start_cpu_times = self.process.cpu_times()

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss
            end_cpu_times = self.process.cpu_times()

            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            cpu_time = (end_cpu_times.user - start_cpu_times.user) + (end_cpu_times.system - start_cpu_times.system)

            self.measurements[operation_name] = {
                "duration": duration,
                "memory_delta": memory_delta,
                "memory_peak": end_memory,
                "cpu_time": cpu_time,
                "timestamp": time.time(),
            }

    def get_measurements(self) -> dict[str, dict[str, Any]]:
        """Get all performance measurements."""
        return self.measurements.copy()

    def clear_measurements(self) -> None:
        """Clear all measurements."""
        self.measurements.clear()

    def show_report(self) -> None:
        """Display performance report."""
        if not self.measurements:
            print_info("No performance measurements recorded")
            return

        table = Table(title="Performance Report")
        table.add_column("Operation", style="cyan")
        table.add_column("Duration", style="green")
        table.add_column("Memory Delta", style="yellow")
        table.add_column("Peak Memory", style="blue")
        table.add_column("CPU Time", style="magenta")

        for operation, metrics in self.measurements.items():
            table.add_row(
                operation,
                format_duration(metrics["duration"]),
                format_bytes(metrics["memory_delta"]),
                format_bytes(metrics["memory_peak"]),
                f"{metrics['cpu_time']:.3f}s",
            )

        console.print(table)


class CLILogger:
    """Advanced CLI logger with structured logging and rich formatting."""

    def __init__(self, log_dir: Path | None = None):
        self.log_dir = log_dir or Path.home() / ".its-camera-ai" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("its_camera_ai_cli")
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self.logger.handlers.clear()

        self.profiler = PerformanceProfiler()
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup logging handlers."""
        # Console handler with Rich formatting
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True,
            markup=True,
        )
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)

        # File handler with rotation
        log_file = self.log_dir / "its-camera-ai.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Debug file handler
        debug_file = self.log_dir / "debug.log"
        debug_handler = logging.handlers.RotatingFileHandler(
            debug_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=3,
            encoding="utf-8",
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d - %(process)d - %(thread)d - %(levelname)s - "
            "%(name)s.%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        debug_handler.setFormatter(debug_formatter)

        # Error file handler
        error_file = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=10,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)

        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(debug_handler)
        self.logger.addHandler(error_handler)

    def set_level(self, level: str) -> None:
        """Set the logging level."""
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")

        # Update console handler level
        for handler in self.logger.handlers:
            if isinstance(handler, RichHandler):
                handler.setLevel(numeric_level)

    def enable_debug_mode(self) -> None:
        """Enable debug mode with verbose logging."""
        self.set_level("DEBUG")
        self.logger.info("Debug mode enabled")

    def disable_debug_mode(self) -> None:
        """Disable debug mode."""
        self.set_level("INFO")
        self.logger.info("Debug mode disabled")

    @contextmanager
    def profile_operation(self, operation_name: str) -> Generator[None, None, None]:
        """Profile an operation with automatic logging."""
        self.logger.info(f"Starting operation: {operation_name}")

        with self.profiler.profile(operation_name):
            try:
                yield
                self.logger.info(f"Completed operation: {operation_name}")
            except Exception as e:
                self.logger.error(f"Operation failed: {operation_name} - {e}", exc_info=True)
                raise

    def log_system_info(self) -> None:
        """Log system information for debugging."""
        import platform

        info = {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
            "memory": format_bytes(psutil.virtual_memory().total),
            "disk_usage": format_bytes(psutil.disk_usage("/").free),
        }

        self.logger.debug(f"System info: {info}")

    def log_command(self, command: str, args: dict[str, Any]) -> None:
        """Log command execution."""
        self.logger.info(f"Executing command: {command} with args: {args}")

    def get_log_files(self) -> list[Path]:
        """Get all log files."""
        return list(self.log_dir.glob("*.log"))

    def clear_logs(self) -> None:
        """Clear all log files."""
        for log_file in self.get_log_files():
            try:
                log_file.unlink()
                self.logger.info(f"Cleared log file: {log_file}")
            except OSError as e:
                self.logger.error(f"Failed to clear log file {log_file}: {e}")

    def show_log_stats(self) -> None:
        """Show log file statistics."""
        table = Table(title="Log Files")
        table.add_column("File", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Lines", style="yellow")
        table.add_column("Modified", style="blue")

        for log_file in self.get_log_files():
            try:
                stats = log_file.stat()
                size = format_bytes(stats.st_size)

                # Count lines
                with open(log_file) as f:
                    line_count = sum(1 for _ in f)

                modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stats.st_mtime))

                table.add_row(log_file.name, size, str(line_count), modified)

            except (OSError, UnicodeDecodeError) as e:
                table.add_row(log_file.name, "Error", str(e), "")

        console.print(table)

    def tail_log(self, log_name: str = "its-camera-ai.log", lines: int = 50) -> None:
        """Show the last N lines of a log file."""
        log_file = self.log_dir / log_name

        if not log_file.exists():
            print_error(f"Log file not found: {log_file}")
            return

        try:
            with open(log_file) as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:]

            if recent_lines:
                console.print(Panel(
                    "".join(recent_lines),
                    title=f"Last {len(recent_lines)} lines of {log_name}",
                    border_style="blue",
                ))
            else:
                print_info(f"Log file {log_name} is empty")

        except Exception as e:
            print_error(f"Failed to read log file: {e}")


# Global logger instance
cli_logger = CLILogger()

app = typer.Typer(help="📋 Advanced logging and debugging")


@app.command("level")
def set_log_level(
    level: str = typer.Argument(
        ..., help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    ),
) -> None:
    """Set the logging level."""
    try:
        cli_logger.set_level(level)
        print_success(f"Log level set to {level.upper()}")
    except ValueError as e:
        print_error(str(e))


@app.command("debug")
def toggle_debug(
    enable: bool = typer.Option(True, "--enable/--disable", help="Enable or disable debug mode"),
) -> None:
    """Enable or disable debug mode."""
    if enable:
        cli_logger.enable_debug_mode()
        print_success("Debug mode enabled")
    else:
        cli_logger.disable_debug_mode()
        print_success("Debug mode disabled")


@app.command("stats")
def show_log_stats() -> None:
    """Show log file statistics."""
    cli_logger.show_log_stats()


@app.command("tail")
def tail_log(
    log_name: str = typer.Option("its-camera-ai.log", "--file", "-f", help="Log file name"),
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines to show"),
) -> None:
    """Show the last N lines of a log file."""
    cli_logger.tail_log(log_name, lines)


@app.command("clear")
def clear_logs(
    force: bool = typer.Option(False, "--force", help="Force clear without confirmation"),
) -> None:
    """Clear all log files."""
    if not force:
        if not typer.confirm("Are you sure you want to clear all log files?"):
            print_info("Operation cancelled")
            return

    cli_logger.clear_logs()
    print_success("All log files cleared")


@app.command("directory")
def show_log_directory() -> None:
    """Show the log directory path."""
    print_info(f"Log directory: {cli_logger.log_dir}")

    if cli_logger.log_dir.exists():
        files = list(cli_logger.log_dir.glob("*"))
        print_info(f"Contains {len(files)} files")
    else:
        print_info("Directory does not exist")


@app.command("profile")
def show_performance_profile() -> None:
    """Show performance profiling report."""
    cli_logger.profiler.show_report()


@app.command("profile-clear")
def clear_performance_profile() -> None:
    """Clear performance profiling data."""
    cli_logger.profiler.clear_measurements()
    print_success("Performance profiling data cleared")


@app.command("system-info")
def log_system_info() -> None:
    """Log and display system information."""
    cli_logger.log_system_info()

    # Also display to console
    import platform

    info_table = Table(title="System Information")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")

    system_info = [
        ("Platform", platform.platform()),
        ("Python Version", sys.version.split()[0]),
        ("CPU Count", str(os.cpu_count())),
        ("Total Memory", format_bytes(psutil.virtual_memory().total)),
        ("Available Memory", format_bytes(psutil.virtual_memory().available)),
        ("Disk Free Space", format_bytes(psutil.disk_usage("/").free)),
        ("Process ID", str(os.getpid())),
        ("Working Directory", str(Path.cwd())),
    ]

    for prop, value in system_info:
        info_table.add_row(prop, value)

    console.print(info_table)


@app.command("test")
def test_logging() -> None:
    """Test logging functionality with sample messages."""
    print_info("Testing logging functionality...")

    # Test different log levels
    cli_logger.logger.debug("This is a debug message")
    cli_logger.logger.info("This is an info message")
    cli_logger.logger.warning("This is a warning message")
    cli_logger.logger.error("This is an error message")

    # Test performance profiling
    with cli_logger.profile_operation("test_operation"):
        time.sleep(0.1)  # Simulate work
        cli_logger.logger.info("Doing some work...")
        time.sleep(0.05)

    print_success("Logging test completed")
    cli_logger.profiler.show_report()


if __name__ == "__main__":
    app()
