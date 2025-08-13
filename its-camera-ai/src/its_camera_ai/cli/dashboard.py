"""Interactive status dashboard for ITS Camera AI CLI.

Provides real-time system monitoring, quick status overviews, and
interactive dashboards with live updates and keyboard shortcuts.
"""

import asyncio
import time
from datetime import datetime
from typing import Any

import psutil
import typer
from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .utils import (
    console,
    format_bytes,
    format_duration,
    get_status_style,
    print_error,
    print_info,
    print_success,
)


class SystemMonitor:
    """System resource monitor for dashboard."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.cpu_history: list[float] = []
        self.memory_history: list[int] = []
        self.network_history: list[tuple[int, int]] = []
        self.max_history = 60  # Keep last 60 measurements

    def update_metrics(self) -> dict[str, Any]:
        """Update and return current system metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_history.append(cpu_percent)
        if len(self.cpu_history) > self.max_history:
            self.cpu_history.pop(0)

        # Memory metrics
        memory = psutil.virtual_memory()
        self.memory_history.append(memory.used)
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)

        # Network metrics
        network = psutil.net_io_counters()
        self.network_history.append((network.bytes_sent, network.bytes_recv))
        if len(self.network_history) > self.max_history:
            self.network_history.pop(0)

        # Process metrics
        process_memory = self.process.memory_info()
        process_cpu = self.process.cpu_percent()

        # Disk metrics
        disk = psutil.disk_usage("/")

        return {
            "cpu_percent": cpu_percent,
            "cpu_avg": sum(self.cpu_history) / len(self.cpu_history),
            "memory_total": memory.total,
            "memory_used": memory.used,
            "memory_percent": memory.percent,
            "process_memory": process_memory.rss,
            "process_cpu": process_cpu,
            "disk_total": disk.total,
            "disk_used": disk.used,
            "disk_free": disk.free,
            "disk_percent": (disk.used / disk.total) * 100,
            "uptime": time.time() - self.start_time,
            "network_sent": network.bytes_sent,
            "network_recv": network.bytes_recv,
        }

    def get_cpu_trend(self) -> str:
        """Get CPU usage trend indicator."""
        if len(self.cpu_history) < 5:
            return "📊"

        recent = sum(self.cpu_history[-5:]) / 5
        older = sum(self.cpu_history[-10:-5]) / 5 if len(self.cpu_history) >= 10 else recent

        if recent > older + 5:
            return "📈"
        elif recent < older - 5:
            return "📉"
        else:
            return "📊"

    def get_memory_trend(self) -> str:
        """Get memory usage trend indicator."""
        if len(self.memory_history) < 5:
            return "📊"

        recent = sum(self.memory_history[-5:]) / 5
        older = sum(self.memory_history[-10:-5]) / 5 if len(self.memory_history) >= 10 else recent

        if recent > older * 1.05:
            return "📈"
        elif recent < older * 0.95:
            return "📉"
        else:
            return "📊"


class ServiceStatusMonitor:
    """Monitor for application services."""

    def __init__(self):
        self.services = {
            "api_server": {"name": "API Server", "port": 8000, "healthy": True},
            "ml_pipeline": {"name": "ML Pipeline", "port": None, "healthy": True},
            "database": {"name": "PostgreSQL", "port": 5432, "healthy": True},
            "redis": {"name": "Redis Cache", "port": 6379, "healthy": True},
            "queue": {"name": "Message Queue", "port": 9092, "healthy": False},
            "monitoring": {"name": "Prometheus", "port": 9090, "healthy": True},
        }

    def check_service_health(self, service_id: str) -> bool:
        """Check if a service is healthy (simulated)."""
        import random
        # Simulate occasional service issues
        return random.random() > 0.1  # 90% uptime

    def update_service_status(self) -> dict[str, dict[str, Any]]:
        """Update and return service status."""
        for service_id, service in self.services.items():
            service["healthy"] = self.check_service_health(service_id)
            service["status"] = "running" if service["healthy"] else "error"
            service["uptime"] = "2d 5h" if service["healthy"] else "0m"

        return self.services


class Dashboard:
    """Interactive system dashboard with live updates."""

    def __init__(self):
        self.system_monitor = SystemMonitor()
        self.service_monitor = ServiceStatusMonitor()
        self.refresh_rate = 2.0  # seconds
        self.running = False

    def create_header(self) -> Panel:
        """Create dashboard header."""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header_text = Text()
        header_text.append("🎥 ITS Camera AI System Dashboard", style="bold blue")
        header_text.append(f"\n{current_time}", style="dim")
        header_text.append(f" • Refresh: {self.refresh_rate}s", style="dim")

        return Panel(
            Align.center(header_text),
            border_style="blue",
            height=4,
        )

    def create_system_metrics_panel(self, metrics: dict[str, Any]) -> Panel:
        """Create system metrics panel."""
        table = Table.grid(padding=(0, 2))
        table.add_column("Metric", style="cyan", width=15)
        table.add_column("Value", style="green", width=15)
        table.add_column("Bar", width=20)
        table.add_column("Trend", width=5)

        # CPU
        cpu_bar = "█" * int(metrics["cpu_percent"] / 5) + "░" * (20 - int(metrics["cpu_percent"] / 5))
        cpu_color = "red" if metrics["cpu_percent"] > 80 else "yellow" if metrics["cpu_percent"] > 60 else "green"
        table.add_row(
            "CPU",
            f"{metrics['cpu_percent']:.1f}%",
            f"[{cpu_color}]{cpu_bar}[/{cpu_color}]",
            self.system_monitor.get_cpu_trend()
        )

        # Memory
        mem_bar = "█" * int(metrics["memory_percent"] / 5) + "░" * (20 - int(metrics["memory_percent"] / 5))
        mem_color = "red" if metrics["memory_percent"] > 80 else "yellow" if metrics["memory_percent"] > 60 else "green"
        table.add_row(
            "Memory",
            f"{metrics['memory_percent']:.1f}%",
            f"[{mem_color}]{mem_bar}[/{mem_color}]",
            self.system_monitor.get_memory_trend()
        )

        # Disk
        disk_bar = "█" * int(metrics["disk_percent"] / 5) + "░" * (20 - int(metrics["disk_percent"] / 5))
        disk_color = "red" if metrics["disk_percent"] > 90 else "yellow" if metrics["disk_percent"] > 75 else "green"
        table.add_row(
            "Disk",
            f"{metrics['disk_percent']:.1f}%",
            f"[{disk_color}]{disk_bar}[/{disk_color}]",
            "💾"
        )

        # Uptime
        table.add_row(
            "Uptime",
            format_duration(metrics["uptime"]),
            "",
            "⏱️"
        )

        return Panel(
            table,
            title="System Metrics",
            border_style="green",
            height=10,
        )

    def create_services_panel(self, services: dict[str, dict[str, Any]]) -> Panel:
        """Create services status panel."""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Service", width=15)
        table.add_column("Status", width=10)
        table.add_column("Port", width=8)
        table.add_column("Uptime", width=10)

        for _service_id, service in services.items():
            status_style = get_status_style(service["status"])
            status_icon = "✅" if service["healthy"] else "❌"
            port_str = str(service["port"]) if service["port"] else "N/A"

            table.add_row(
                service["name"],
                f"{status_icon} [{status_style}]{service['status']}[/{status_style}]",
                port_str,
                service["uptime"]
            )

        healthy_count = sum(1 for s in services.values() if s["healthy"])
        total_count = len(services)

        return Panel(
            table,
            title=f"Services ({healthy_count}/{total_count} healthy)",
            border_style="yellow" if healthy_count < total_count else "green",
            height=10,
        )

    def create_quick_stats_panel(self, metrics: dict[str, Any]) -> Panel:
        """Create quick statistics panel."""
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column("Label", style="cyan")
        stats_table.add_column("Value", style="green")

        stats = [
            ("Total Memory", format_bytes(metrics["memory_total"])),
            ("Process Memory", format_bytes(metrics["process_memory"])),
            ("Process CPU", f"{metrics['process_cpu']:.1f}%"),
            ("Disk Free", format_bytes(metrics["disk_free"])),
            ("Network Sent", format_bytes(metrics["network_sent"])),
            ("Network Received", format_bytes(metrics["network_recv"])),
        ]

        for label, value in stats:
            stats_table.add_row(label, value)

        return Panel(
            stats_table,
            title="Quick Stats",
            border_style="blue",
            height=10,
        )

    def create_recent_activity_panel(self) -> Panel:
        """Create recent activity panel."""
        activities = [
            "🤖 Model inference completed (45ms avg)",
            "📊 Batch processing finished (1,250 frames)",
            "🔧 Configuration updated (ml.batch_size=32)",
            "📈 Metrics exported to Prometheus",
            "🔒 User authentication successful",
            "💾 Database backup completed",
        ]

        activity_text = "\n".join(f"• {activity}" for activity in activities[-6:])

        return Panel(
            activity_text,
            title="Recent Activity",
            border_style="magenta",
            height=10,
        )

    def create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )

        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        layout["left"].split_column(
            Layout(name="metrics"),
            Layout(name="services")
        )

        layout["right"].split_column(
            Layout(name="stats"),
            Layout(name="activity")
        )

        # Footer
        footer_text = Text()
        footer_text.append("Press ", style="dim")
        footer_text.append("Ctrl+C", style="bold")
        footer_text.append(" to exit • ", style="dim")
        footer_text.append("r", style="bold")
        footer_text.append(" to refresh • ", style="dim")
        footer_text.append("q", style="bold")
        footer_text.append(" to quit", style="dim")

        layout["footer"] = Panel(
            Align.center(footer_text),
            border_style="dim"
        )

        return layout

    def update_dashboard(self, layout: Layout) -> None:
        """Update dashboard with current data."""
        # Get current metrics
        metrics = self.system_monitor.update_metrics()
        services = self.service_monitor.update_service_status()

        # Update layout
        layout["header"] = self.create_header()
        layout["metrics"] = self.create_system_metrics_panel(metrics)
        layout["services"] = self.create_services_panel(services)
        layout["stats"] = self.create_quick_stats_panel(metrics)
        layout["activity"] = self.create_recent_activity_panel()

    async def run_dashboard(self) -> None:
        """Run the interactive dashboard."""
        self.running = True
        layout = self.create_layout()

        with Live(layout, console=console, refresh_per_second=0.5, screen=True):
            while self.running:
                try:
                    self.update_dashboard(layout)
                    await asyncio.sleep(self.refresh_rate)
                except KeyboardInterrupt:
                    break

    def stop_dashboard(self) -> None:
        """Stop the dashboard."""
        self.running = False


class QuickStatusChecker:
    """Quick system status checker for shortcuts."""

    @staticmethod
    def get_quick_status() -> dict[str, str]:
        """Get quick system status overview."""
        try:
            # CPU
            cpu = psutil.cpu_percent(interval=1)
            cpu_status = "🟢" if cpu < 70 else "🟡" if cpu < 90 else "🔴"

            # Memory
            memory = psutil.virtual_memory()
            mem_status = "🟢" if memory.percent < 70 else "🟡" if memory.percent < 90 else "🔴"

            # Disk
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            disk_status = "🟢" if disk_percent < 80 else "🟡" if disk_percent < 95 else "🔴"

            return {
                "cpu": f"{cpu_status} CPU: {cpu:.1f}%",
                "memory": f"{mem_status} RAM: {memory.percent:.1f}%",
                "disk": f"{disk_status} Disk: {disk_percent:.1f}%",
                "overall": "🟢 System Healthy" if all(s.startswith("🟢") for s in [cpu_status, mem_status, disk_status]) else "🟡 System Warning"
            }
        except Exception as e:
            return {"error": f"❌ Status check failed: {e}"}

    @staticmethod
    def show_compact_status() -> None:
        """Show compact status for quick viewing."""
        status = QuickStatusChecker.get_quick_status()

        if "error" in status:
            print_error(status["error"])
            return

        status_text = " | ".join([
            status["overall"],
            status["cpu"],
            status["memory"],
            status["disk"]
        ])

        console.print(f"[bold]{status_text}[/bold]")


app = typer.Typer(help="📊 Interactive dashboard and status monitoring")


@app.command()
def live() -> None:
    """Launch live interactive dashboard."""
    print_info("Starting interactive dashboard... Press Ctrl+C to exit")

    try:
        dashboard = Dashboard()
        asyncio.run(dashboard.run_dashboard())
    except KeyboardInterrupt:
        print_info("Dashboard stopped")
    except Exception as e:
        print_error(f"Dashboard failed: {e}")


@app.command("status")
def quick_status() -> None:
    """Show quick system status overview."""
    QuickStatusChecker.show_compact_status()


@app.command("health")
def health_check() -> None:
    """Comprehensive system health check."""
    print_info("Running system health check...")

    # System metrics
    metrics = SystemMonitor().update_metrics()

    # Health status
    issues = []

    if metrics["cpu_percent"] > 90:
        issues.append(f"❌ High CPU usage: {metrics['cpu_percent']:.1f}%")

    if metrics["memory_percent"] > 90:
        issues.append(f"❌ High memory usage: {metrics['memory_percent']:.1f}%")

    if metrics["disk_percent"] > 95:
        issues.append(f"❌ Low disk space: {metrics['disk_percent']:.1f}% used")

    # Display results
    health_table = Table(title="System Health Check")
    health_table.add_column("Component", style="cyan")
    health_table.add_column("Status", style="green")
    health_table.add_column("Value", style="yellow")

    components = [
        ("CPU Usage", "✅ Normal" if metrics["cpu_percent"] < 80 else "⚠️ High", f"{metrics['cpu_percent']:.1f}%"),
        ("Memory Usage", "✅ Normal" if metrics["memory_percent"] < 80 else "⚠️ High", f"{metrics['memory_percent']:.1f}%"),
        ("Disk Usage", "✅ Normal" if metrics["disk_percent"] < 90 else "⚠️ High", f"{metrics['disk_percent']:.1f}%"),
        ("Uptime", "✅ Running", format_duration(metrics["uptime"])),
    ]

    for component, status, value in components:
        health_table.add_row(component, status, value)

    console.print(health_table)

    if issues:
        console.print("\n[bold red]Issues Found:[/bold red]")
        for issue in issues:
            console.print(f"  {issue}")
    else:
        print_success("All systems healthy!")


@app.command("monitor")
def system_monitor(
    duration: int = typer.Option(60, "--duration", "-d", help="Monitoring duration in seconds"),
    interval: int = typer.Option(5, "--interval", "-i", help="Update interval in seconds"),
) -> None:
    """Monitor system metrics for a specified duration."""
    print_info(f"Monitoring system for {duration}s (updates every {interval}s)")

    monitor = SystemMonitor()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}s"),
        console=console,
    ) as progress:

        task = progress.add_task("Monitoring...", total=duration)

        for i in range(0, duration, interval):
            metrics = monitor.update_metrics()

            progress.update(
                task,
                description=f"CPU: {metrics['cpu_percent']:.1f}% | RAM: {metrics['memory_percent']:.1f}%",
                completed=i
            )

            time.sleep(min(interval, duration - i))

    print_success("Monitoring completed!")
    monitor.update_metrics()  # Final update

    # Show summary
    if monitor.cpu_history and monitor.memory_history:
        avg_cpu = sum(monitor.cpu_history) / len(monitor.cpu_history)
        max_cpu = max(monitor.cpu_history)
        avg_memory = sum(monitor.memory_history) / len(monitor.memory_history)
        max_memory = max(monitor.memory_history)

        summary_table = Table(title="Monitoring Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Average", style="green")
        summary_table.add_column("Peak", style="red")

        summary_table.add_row("CPU Usage", f"{avg_cpu:.1f}%", f"{max_cpu:.1f}%")
        summary_table.add_row("Memory Usage", format_bytes(int(avg_memory)), format_bytes(max_memory))

        console.print(summary_table)


if __name__ == "__main__":
    app()
