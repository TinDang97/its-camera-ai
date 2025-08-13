"""Monitoring and health check commands for ITS Camera AI.

Commands for system monitoring, health checks, and observability.
"""

import asyncio
import time

import typer
from rich.live import Live
from rich.table import Table

from ..utils import (
    console,
    create_metrics_table,
    create_progress,
    format_duration,
    get_status_style,
    handle_async_command,
    print_error,
    print_info,
    print_success,
)

app = typer.Typer(help="📈 System monitoring and health checks")


@app.command()
@handle_async_command
async def metrics(
    service: str | None = typer.Option(
        None, "--service", "-s", help="Filter metrics by service"
    ),
    watch: bool = typer.Option(
        False, "--watch", "-w", help="Watch metrics in real-time"
    ),
    interval: int = typer.Option(
        5, "--interval", "-i", help="Update interval in seconds for watch mode"
    ),
) -> None:
    """📊 Display system metrics.

    Show key performance metrics including throughput, latency, resource usage,
    and system health indicators.
    """
    if watch:
        await _watch_metrics(service, interval)
    else:
        await _show_metrics_once(service)


async def _show_metrics_once(service_filter: str | None) -> None:
    """
    Show metrics once.

    Args:
        service_filter: Optional service filter
    """
    print_info("Fetching system metrics...")

    # Get metrics data
    metrics_data = await _get_metrics_data(service_filter)

    # Display metrics in categories
    categories = {
        "Performance": [
            "inference_throughput_fps",
            "api_latency_ms",
            "queue_processing_rate",
            "memory_usage_percent",
        ],
        "System Health": [
            "cpu_usage_percent",
            "disk_usage_percent",
            "network_throughput_mbps",
            "error_rate_percent",
        ],
        "ML Pipeline": [
            "model_accuracy",
            "detection_count",
            "processing_latency_ms",
            "gpu_utilization_percent",
        ],
    }

    for category, metric_names in categories.items():
        table = Table(title=f"{category} Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Unit", style="yellow")
        table.add_column("Status", style="blue")
        table.add_column("Threshold", style="dim")

        for metric_name in metric_names:
            if metric_name in metrics_data:
                metric = metrics_data[metric_name]
                status_style = get_status_style(metric["status"])

                table.add_row(
                    metric["name"],
                    str(metric["value"]),
                    metric["unit"],
                    f"[{status_style}]{metric['status']}[/{status_style}]",
                    str(metric["threshold"]),
                )

        console.print(table)
        console.print()  # Add spacing between tables


async def _watch_metrics(service_filter: str | None, interval: int) -> None:
    """
    Watch metrics in real-time.

    Args:
        service_filter: Optional service filter
        interval: Update interval in seconds
    """
    print_info("Starting real-time metrics monitoring (Ctrl+C to stop)")

    try:
        with Live(auto_refresh=False) as live:
            while True:
                metrics_data = await _get_metrics_data(service_filter)

                # Create summary table
                table = create_metrics_table()

                # Add key metrics
                key_metrics = [
                    "inference_throughput_fps",
                    "api_latency_ms",
                    "cpu_usage_percent",
                    "memory_usage_percent",
                    "gpu_utilization_percent",
                    "error_rate_percent",
                ]

                for metric_name in key_metrics:
                    if metric_name in metrics_data:
                        metric = metrics_data[metric_name]
                        status_style = get_status_style(metric["status"])

                        table.add_row(
                            metric["name"],
                            str(metric["value"]),
                            metric["unit"],
                            f"[{status_style}]{metric['status']}[/{status_style}]",
                            str(metric["threshold"]),
                        )

                live.update(table)
                live.refresh()

                await asyncio.sleep(interval)

    except KeyboardInterrupt:
        print_info("Metrics monitoring stopped")


async def _get_metrics_data(service_filter: str | None) -> dict:
    """
    Get current metrics data.

    Args:
        service_filter: Optional service filter

    Returns:
        Dictionary of metrics data
    """
    # Simulate metrics data
    import random

    base_metrics = {
        "inference_throughput_fps": {
            "name": "Inference Throughput",
            "value": random.uniform(120, 180),
            "unit": "fps",
            "threshold": 100,
            "status": "healthy",
        },
        "api_latency_ms": {
            "name": "API Latency",
            "value": random.uniform(20, 80),
            "unit": "ms",
            "threshold": 100,
            "status": "healthy",
        },
        "queue_processing_rate": {
            "name": "Queue Processing Rate",
            "value": random.uniform(200, 300),
            "unit": "msg/s",
            "threshold": 150,
            "status": "healthy",
        },
        "memory_usage_percent": {
            "name": "Memory Usage",
            "value": random.uniform(45, 75),
            "unit": "%",
            "threshold": 80,
            "status": "healthy" if random.uniform(45, 75) < 70 else "warning",
        },
        "cpu_usage_percent": {
            "name": "CPU Usage",
            "value": random.uniform(30, 70),
            "unit": "%",
            "threshold": 80,
            "status": "healthy",
        },
        "disk_usage_percent": {
            "name": "Disk Usage",
            "value": random.uniform(40, 85),
            "unit": "%",
            "threshold": 90,
            "status": "healthy" if random.uniform(40, 85) < 80 else "warning",
        },
        "network_throughput_mbps": {
            "name": "Network Throughput",
            "value": random.uniform(50, 150),
            "unit": "Mbps",
            "threshold": 200,
            "status": "healthy",
        },
        "error_rate_percent": {
            "name": "Error Rate",
            "value": random.uniform(0, 2),
            "unit": "%",
            "threshold": 5,
            "status": "healthy" if random.uniform(0, 2) < 1 else "warning",
        },
        "model_accuracy": {
            "name": "Model Accuracy",
            "value": random.uniform(0.85, 0.95),
            "unit": "",
            "threshold": 0.8,
            "status": "healthy",
        },
        "detection_count": {
            "name": "Detection Count",
            "value": random.randint(500, 1500),
            "unit": "/hour",
            "threshold": 100,
            "status": "healthy",
        },
        "processing_latency_ms": {
            "name": "Processing Latency",
            "value": random.uniform(25, 65),
            "unit": "ms",
            "threshold": 100,
            "status": "healthy",
        },
        "gpu_utilization_percent": {
            "name": "GPU Utilization",
            "value": random.uniform(60, 90),
            "unit": "%",
            "threshold": 95,
            "status": "healthy" if random.uniform(60, 90) < 85 else "warning",
        },
    }

    # Filter by service if specified
    if service_filter:
        # Simulate service-specific filtering
        if service_filter == "api":
            return {k: v for k, v in base_metrics.items() if "api" in k or "latency" in k}
        elif service_filter == "ml":
            return {k: v for k, v in base_metrics.items() if any(term in k for term in ["model", "detection", "gpu", "inference"])}

    return base_metrics


@app.command()
@handle_async_command
async def health(
    component: str | None = typer.Option(
        None, "--component", "-c", help="Check specific component"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed health information"
    ),
    timeout: int = typer.Option(
        30, "--timeout", "-t", help="Health check timeout in seconds"
    ),
) -> None:
    """❤️ Comprehensive health checks.

    Run health checks on all system components or specific components.
    Includes connectivity, performance, and dependency checks.
    """
    components_to_check = [component] if component else [
        "database", "redis", "api", "inference", "monitoring", "storage"
    ]

    print_info(f"Running health checks with {timeout}s timeout...")

    health_results = []

    with create_progress() as progress:
        task = progress.add_task("Running health checks...", total=len(components_to_check))

        for comp in components_to_check:
            progress.update(task, description=f"Checking {comp}...")

            # Simulate health check
            result = await _check_component_health(comp, timeout)
            health_results.append(result)

            progress.advance(task)

    # Display results
    table = Table(title="Health Check Results")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Response Time", style="yellow")
    table.add_column("Details", style="blue")

    if verbose:
        table.add_column("Additional Info", style="dim")

    healthy_count = 0

    for result in health_results:
        status_style = get_status_style(result["status"])

        row_data = [
            result["component"],
            f"[{status_style}]{result['status']}[/{status_style}]",
            f"{result['response_time']:.1f}ms",
            result["details"],
        ]

        if verbose:
            row_data.append(result.get("additional_info", "N/A"))

        table.add_row(*row_data)

        if result["status"] in ["healthy", "ok"]:
            healthy_count += 1

    console.print(table)

    # Show summary
    total_checks = len(health_results)
    if healthy_count == total_checks:
        print_success(f"All {total_checks} components are healthy")
    else:
        unhealthy_count = total_checks - healthy_count
        print_error(f"{unhealthy_count} of {total_checks} components are unhealthy")

        # List unhealthy components
        unhealthy_components = [
            r["component"] for r in health_results
            if r["status"] not in ["healthy", "ok"]
        ]
        print_info(f"Unhealthy components: {', '.join(unhealthy_components)}")


async def _check_component_health(component: str, timeout: int) -> dict:
    """
    Check health of a specific component.

    Args:
        component: Component name
        timeout: Timeout in seconds

    Returns:
        Health check result dictionary
    """
    import random

    # Simulate health check time
    check_time = random.uniform(0.1, 1.0)
    await asyncio.sleep(check_time)

    # Simulate health check results
    health_scenarios = {
        "database": {
            "status": "healthy",
            "details": "PostgreSQL connection active",
            "additional_info": "Pool: 8/20 connections, Latency: 2.3ms",
        },
        "redis": {
            "status": "healthy",
            "details": "Redis cache responsive",
            "additional_info": "Memory: 45% used, Hit rate: 95.2%",
        },
        "api": {
            "status": "healthy",
            "details": "FastAPI server running",
            "additional_info": "Workers: 4/4 active, Requests: 150/min",
        },
        "inference": {
            "status": "healthy" if random.random() > 0.1 else "warning",
            "details": "ML inference engine active",
            "additional_info": "GPU: 75% utilization, Queue: 12 items",
        },
        "monitoring": {
            "status": "healthy",
            "details": "Prometheus metrics collecting",
            "additional_info": "Scrape interval: 15s, Targets: 8/8 up",
        },
        "storage": {
            "status": "healthy",
            "details": "MinIO object storage accessible",
            "additional_info": "Buckets: 4, Objects: 15.2k, Size: 2.1TB",
        },
    }

    if component in health_scenarios:
        result = health_scenarios[component].copy()
    else:
        result = {
            "status": "unknown",
            "details": f"Unknown component: {component}",
            "additional_info": "Component not recognized",
        }

    result.update({
        "component": component.title(),
        "response_time": check_time * 1000,  # Convert to milliseconds
    })

    return result


@app.command()
@handle_async_command
async def alerts(
    status: str | None = typer.Option(
        None, "--status", "-s", help="Filter by status (active, resolved, muted)"
    ),
    severity: str | None = typer.Option(
        None, "--severity", help="Filter by severity (low, medium, high, critical)"
    ),
    limit: int = typer.Option(
        20, "--limit", "-l", help="Maximum number of alerts to show"
    ),
) -> None:
    """🚨 View system alerts.

    Display active and recent system alerts with filtering options.
    """
    print_info("Fetching system alerts...")

    # Simulate alerts data
    alerts_data = await _get_alerts_data()

    # Apply filters
    filtered_alerts = alerts_data[:]

    if status:
        filtered_alerts = [a for a in filtered_alerts if a["status"] == status]

    if severity:
        filtered_alerts = [a for a in filtered_alerts if a["severity"] == severity]

    # Limit results
    filtered_alerts = filtered_alerts[:limit]

    # Display alerts table
    table = Table(title="System Alerts")
    table.add_column("ID", style="cyan")
    table.add_column("Severity", style="red")
    table.add_column("Title", style="green")
    table.add_column("Component", style="yellow")
    table.add_column("Status", style="blue")
    table.add_column("Triggered", style="white")
    table.add_column("Duration", style="dim")

    for alert in filtered_alerts:
        severity_style = {
            "low": "green",
            "medium": "yellow",
            "high": "red",
            "critical": "bold red",
        }.get(alert["severity"], "white")

        status_style = {
            "active": "red",
            "resolved": "green",
            "muted": "dim",
        }.get(alert["status"], "white")

        # Calculate duration
        duration = time.time() - alert["triggered_at"]
        duration_str = format_duration(duration)

        table.add_row(
            alert["id"],
            f"[{severity_style}]{alert['severity'].upper()}[/{severity_style}]",
            alert["title"],
            alert["component"],
            f"[{status_style}]{alert['status']}[/{status_style}]",
            alert["triggered"],
            duration_str,
        )

    console.print(table)

    # Show alert summary
    active_alerts = len([a for a in alerts_data if a["status"] == "active"])
    critical_alerts = len([
        a for a in alerts_data
        if a["status"] == "active" and a["severity"] == "critical"
    ])

    if critical_alerts > 0:
        print_error(f"{critical_alerts} critical alerts require immediate attention!")
    elif active_alerts > 0:
        print_info(f"{active_alerts} active alerts")
    else:
        print_success("No active alerts")


async def _get_alerts_data() -> list:
    """
    Get current alerts data.

    Returns:
        List of alert dictionaries
    """
    import time

    # Simulate alerts data
    alerts = [
        {
            "id": "alert_001",
            "severity": "high",
            "title": "High Memory Usage",
            "component": "Inference Engine",
            "status": "active",
            "triggered": "2024-01-16 14:25:30",
            "triggered_at": time.time() - 1800,  # 30 minutes ago
            "description": "Memory usage exceeded 85% threshold",
        },
        {
            "id": "alert_002",
            "severity": "medium",
            "title": "Increased API Latency",
            "component": "API Server",
            "status": "active",
            "triggered": "2024-01-16 14:30:15",
            "triggered_at": time.time() - 1200,  # 20 minutes ago
            "description": "API response time increased to 150ms average",
        },
        {
            "id": "alert_003",
            "severity": "critical",
            "title": "Database Connection Lost",
            "component": "Database",
            "status": "resolved",
            "triggered": "2024-01-16 13:45:00",
            "triggered_at": time.time() - 4200,  # 70 minutes ago
            "description": "Lost connection to PostgreSQL database",
        },
        {
            "id": "alert_004",
            "severity": "low",
            "title": "Disk Space Warning",
            "component": "Storage",
            "status": "muted",
            "triggered": "2024-01-16 12:00:00",
            "triggered_at": time.time() - 10800,  # 3 hours ago
            "description": "Disk usage reached 75% on /data partition",
        },
    ]

    return alerts


@app.command()
@handle_async_command
async def dashboard(
    component: str | None = typer.Option(
        None, "--component", "-c", help="Focus on specific component"
    ),
    refresh: int = typer.Option(
        10, "--refresh", "-r", help="Auto-refresh interval in seconds (0 to disable)"
    ),
) -> None:
    """📊 Live monitoring dashboard.

    Display a comprehensive live dashboard with key metrics, health status,
    and alerts in a single view.
    """
    print_info("Starting monitoring dashboard...")

    if refresh == 0:
        # Show static dashboard
        await _show_static_dashboard(component)
    else:
        # Show live dashboard
        await _show_live_dashboard(component, refresh)


async def _show_static_dashboard(component_filter: str | None) -> None:
    """
    Show static dashboard snapshot.

    Args:
        component_filter: Optional component filter
    """
    # Get data
    metrics_data = await _get_metrics_data(component_filter)
    health_results = []

    components = ["database", "redis", "api", "inference"]
    for comp in components:
        result = await _check_component_health(comp, 5)
        health_results.append(result)

    alerts_data = await _get_alerts_data()

    # Display dashboard sections
    _display_dashboard_summary(metrics_data, health_results, alerts_data)


async def _show_live_dashboard(component_filter: str | None, refresh_interval: int) -> None:
    """
    Show live updating dashboard.

    Args:
        component_filter: Optional component filter
        refresh_interval: Refresh interval in seconds
    """
    print_info(f"Live dashboard updating every {refresh_interval}s (Ctrl+C to stop)")

    try:
        while True:
            console.clear()

            # Get fresh data
            metrics_data = await _get_metrics_data(component_filter)
            health_results = []

            components = ["database", "redis", "api", "inference"]
            for comp in components:
                result = await _check_component_health(comp, 2)
                health_results.append(result)

            alerts_data = await _get_alerts_data()

            # Display dashboard
            _display_dashboard_summary(metrics_data, health_results, alerts_data)

            console.print(f"\n[dim]Last updated: {time.strftime('%H:%M:%S')} | Refreshing every {refresh_interval}s[/dim]")

            await asyncio.sleep(refresh_interval)

    except KeyboardInterrupt:
        print_info("Dashboard monitoring stopped")


def _display_dashboard_summary(
    metrics_data: dict,
    health_results: list,
    alerts_data: list,
) -> None:
    """
    Display dashboard summary tables.

    Args:
        metrics_data: Metrics data
        health_results: Health check results
        alerts_data: Alerts data
    """
    # Key metrics summary
    metrics_table = Table(title="📈 Key Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    metrics_table.add_column("Status", style="yellow")

    key_metrics = [
        "inference_throughput_fps",
        "api_latency_ms",
        "memory_usage_percent",
        "error_rate_percent",
    ]

    for metric_name in key_metrics:
        if metric_name in metrics_data:
            metric = metrics_data[metric_name]
            status_style = get_status_style(metric["status"])

            value_str = f"{metric['value']:.1f} {metric['unit']}"
            if metric_name == "model_accuracy":
                value_str = f"{metric['value']:.3f}"

            metrics_table.add_row(
                metric["name"],
                value_str,
                f"[{status_style}]{metric['status']}[/{status_style}]",
            )

    console.print(metrics_table)

    # Component health summary
    health_table = Table(title="❤️ Component Health")
    health_table.add_column("Component", style="cyan")
    health_table.add_column("Status", style="green")
    health_table.add_column("Response", style="yellow")

    for result in health_results:
        status_style = get_status_style(result["status"])

        health_table.add_row(
            result["component"],
            f"[{status_style}]{result['status']}[/{status_style}]",
            f"{result['response_time']:.1f}ms",
        )

    console.print(health_table)

    # Active alerts summary
    active_alerts = [a for a in alerts_data if a["status"] == "active"]

    if active_alerts:
        alerts_table = Table(title="🚨 Active Alerts")
        alerts_table.add_column("Severity", style="red")
        alerts_table.add_column("Component", style="cyan")
        alerts_table.add_column("Title", style="green")
        alerts_table.add_column("Duration", style="yellow")

        for alert in active_alerts[:5]:  # Show top 5 alerts
            severity_style = {
                "low": "green",
                "medium": "yellow",
                "high": "red",
                "critical": "bold red",
            }.get(alert["severity"], "white")

            duration = time.time() - alert["triggered_at"]
            duration_str = format_duration(duration)

            alerts_table.add_row(
                f"[{severity_style}]{alert['severity'].upper()}[/{severity_style}]",
                alert["component"],
                alert["title"],
                duration_str,
            )

        console.print(alerts_table)
    else:
        console.print("🚨 [green]No active alerts[/green]")


@app.command()
def reports(
    report_type: str = typer.Argument(
        "summary", help="Report type: summary, performance, health, alerts"
    ),
    output_file: str | None = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    format_type: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, csv"
    ),
    period: str = typer.Option(
        "24h", "--period", "-p", help="Time period: 1h, 6h, 24h, 7d, 30d"
    ),
) -> None:
    """📊 Generate monitoring reports.

    Generate various types of monitoring reports including performance
    summaries, health reports, and alert analyses.
    """
    valid_report_types = ["summary", "performance", "health", "alerts"]
    if report_type not in valid_report_types:
        print_error(f"Invalid report type. Must be one of: {', '.join(valid_report_types)}")
        return

    print_info(f"Generating {report_type} report for period: {period}")

    if report_type == "summary":
        _generate_summary_report(output_file, format_type, period)
    elif report_type == "performance":
        _generate_performance_report(output_file, format_type, period)
    elif report_type == "health":
        _generate_health_report(output_file, format_type, period)
    elif report_type == "alerts":
        _generate_alerts_report(output_file, format_type, period)


def _generate_summary_report(output_file: str | None, format_type: str, period: str) -> None:
    """
    Generate summary report.

    Args:
        output_file: Optional output file
        format_type: Output format
        period: Time period
    """
    # Simulate report data
    report_data = {
        "report_type": "System Summary",
        "period": period,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "total_requests": 156420,
            "average_latency_ms": 45.2,
            "error_rate_percent": 0.8,
            "uptime_percent": 99.7,
            "data_processed_gb": 2.4,
        },
        "components": {
            "healthy": 6,
            "warning": 1,
            "critical": 0,
        },
        "alerts": {
            "total": 12,
            "resolved": 10,
            "active": 2,
        },
    }

    if format_type == "table":
        _display_summary_table(report_data)

    if output_file:
        _save_report_to_file(report_data, output_file, format_type)
        print_success(f"Report saved to {output_file}")


def _display_summary_table(report_data: dict) -> None:
    """
    Display summary report as table.

    Args:
        report_data: Report data dictionary
    """
    # System metrics
    metrics_table = Table(title=f"System Summary - {report_data['period']}")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")

    for metric, value in report_data["metrics"].items():
        display_metric = metric.replace("_", " ").title()

        if "gb" in metric:
            display_value = f"{value} GB"
        elif "percent" in metric:
            display_value = f"{value}%"
        elif "ms" in metric:
            display_value = f"{value} ms"
        else:
            display_value = f"{value:,}"

        metrics_table.add_row(display_metric, display_value)

    console.print(metrics_table)

    # Component health
    health_table = Table(title="Component Health")
    health_table.add_column("Status", style="cyan")
    health_table.add_column("Count", style="green")

    for status, count in report_data["components"].items():
        status_style = get_status_style(status)
        health_table.add_row(
            f"[{status_style}]{status.title()}[/{status_style}]",
            str(count),
        )

    console.print(health_table)

    # Alerts summary
    alerts_table = Table(title="Alerts Summary")
    alerts_table.add_column("Type", style="cyan")
    alerts_table.add_column("Count", style="green")

    for alert_type, count in report_data["alerts"].items():
        alerts_table.add_row(alert_type.title(), str(count))

    console.print(alerts_table)


def _save_report_to_file(report_data: dict, output_file: str, format_type: str) -> None:
    """
    Save report data to file.

    Args:
        report_data: Report data
        output_file: Output file path
        format_type: Output format
    """
    import csv
    import json
    from pathlib import Path

    output_path = Path(output_file)

    if format_type == "json":
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)
    elif format_type == "csv":
        # Flatten data for CSV
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Category", "Metric", "Value"])

            for category, data in report_data.items():
                if isinstance(data, dict):
                    for metric, value in data.items():
                        writer.writerow([category, metric, value])
                else:
                    writer.writerow(["general", category, data])
    else:
        # Default to plain text
        with open(output_path, "w") as f:
            f.write(f"Report: {report_data['report_type']}\n")
            f.write(f"Period: {report_data['period']}\n")
            f.write(f"Generated: {report_data['generated_at']}\n\n")

            for section, data in report_data.items():
                if isinstance(data, dict):
                    f.write(f"{section.title()}:\n")
                    for key, value in data.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")


def _generate_performance_report(output_file: str | None, format_type: str, period: str) -> None:
    """
    Generate performance report.

    Args:
        output_file: Optional output file
        format_type: Output format
        period: Time period
    """
    print_info("Performance report generation not implemented yet")


def _generate_health_report(output_file: str | None, format_type: str, period: str) -> None:
    """
    Generate health report.

    Args:
        output_file: Optional output file
        format_type: Output format
        period: Time period
    """
    print_info("Health report generation not implemented yet")


def _generate_alerts_report(output_file: str | None, format_type: str, period: str) -> None:
    """
    Generate alerts report.

    Args:
        output_file: Optional output file
        format_type: Output format
        period: Time period
    """
    print_info("Alerts report generation not implemented yet")
