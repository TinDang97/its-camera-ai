"""CLI Command for Comprehensive Optimization Validation.

This command runs the complete validation suite for all system optimizations
including gRPC performance, Redis/Kafka optimization, blosc compression,
and end-to-end system validation.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import typer
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ...core.blosc_numpy_compressor import get_global_compressor
from ...flow.redis_queue_manager import RedisQueueManager
from ...services.grpc_streaming_server import StreamingServiceImpl
from ...services.kafka_event_producer import create_kafka_producer
from ...testing.end_to_end_performance_validator import (
    PerformanceTargets,
    create_end_to_end_validator,
)
from ...testing.performance_regression_monitor import (
    create_performance_monitor,
)

console = Console()
app = typer.Typer(name="validate-optimizations", help="Validate system optimization performance")


@app.command("quick")
def quick_validation(
    redis_url: str = typer.Option("redis://localhost:6379", help="Redis connection URL"),
    kafka_servers: str = typer.Option("localhost:9092", help="Kafka bootstrap servers"),
    output_file: str = typer.Option("", help="Output results to JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run quick optimization validation (< 2 minutes)."""

    async def run_quick_validation():
        with console.status("[bold green]Starting quick optimization validation..."):
            try:
                # Initialize components
                redis_manager = RedisQueueManager(redis_url=redis_url)
                await redis_manager.connect()

                grpc_service = StreamingServiceImpl(
                    redis_manager=redis_manager,
                    enable_blosc_compression=True
                )

                # Run quick tests
                results = {
                    "validation_type": "quick",
                    "timestamp": time.time(),
                    "tests": {}
                }

                # Test 1: Blosc compression efficiency
                console.print("🔍 Testing blosc compression efficiency...")
                blosc_compressor = get_global_compressor()

                import numpy as np
                test_array = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

                start_time = time.perf_counter()
                compressed_data = blosc_compressor.compress_with_metadata(test_array)
                compression_time = (time.perf_counter() - start_time) * 1000

                compression_ratio = len(compressed_data) / test_array.nbytes
                size_reduction = (1 - compression_ratio) * 100

                results["tests"]["blosc_compression"] = {
                    "compression_ratio": compression_ratio,
                    "size_reduction_percent": size_reduction,
                    "compression_time_ms": compression_time,
                    "passed": size_reduction >= 60.0 and compression_time <= 50.0
                }

                # Test 2: gRPC serialization performance
                console.print("🚀 Testing gRPC serialization performance...")

                serialization_times = []
                for i in range(10):
                    start_time = time.perf_counter()
                    pb_frame = grpc_service._numpy_to_protobuf(test_array, use_blosc=True)
                    decompressed = await grpc_service._optimized_protobuf_to_numpy(pb_frame)
                    serialization_time = (time.perf_counter() - start_time) * 1000
                    serialization_times.append(serialization_time)

                avg_serialization_time = sum(serialization_times) / len(serialization_times)
                max_serialization_time = max(serialization_times)

                results["tests"]["grpc_serialization"] = {
                    "avg_time_ms": avg_serialization_time,
                    "max_time_ms": max_serialization_time,
                    "passed": avg_serialization_time <= 30.0 and max_serialization_time <= 100.0
                }

                # Test 3: Redis queue performance
                console.print("📊 Testing Redis queue performance...")

                from ...flow.redis_queue_manager import QueueConfig

                queue_config = QueueConfig(
                    name="validation_test_queue",
                    enable_blosc_compression=True
                )
                await redis_manager.create_queue(queue_config)

                queue_times = []
                test_data = np.random.bytes(10240)  # 10KB

                for i in range(10):
                    start_time = time.perf_counter()
                    message_id = await redis_manager.enqueue(
                        "validation_test_queue", test_data, enable_compression=True
                    )
                    result = await redis_manager.dequeue("validation_test_queue")
                    queue_time = (time.perf_counter() - start_time) * 1000
                    queue_times.append(queue_time)

                avg_queue_time = sum(queue_times) / len(queue_times)

                results["tests"]["redis_queue"] = {
                    "avg_time_ms": avg_queue_time,
                    "passed": avg_queue_time <= 50.0
                }

                # Calculate overall result
                all_passed = all(test["passed"] for test in results["tests"].values())
                results["overall_status"] = "PASSED" if all_passed else "FAILED"
                results["duration_seconds"] = time.time() - results["timestamp"]

                await redis_manager.disconnect()

                return results

            except Exception as e:
                console.print(f"[red]Quick validation failed: {e}")
                return {"validation_type": "quick", "error": str(e), "overall_status": "ERROR"}

    # Run validation
    results = asyncio.run(run_quick_validation())

    # Display results
    display_quick_results(results, verbose)

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        console.print(f"✅ Results saved to {output_file}")


@app.command("comprehensive")
def comprehensive_validation(
    redis_url: str = typer.Option("redis://localhost:6379", help="Redis connection URL"),
    kafka_servers: str = typer.Option("localhost:9092", help="Kafka bootstrap servers"),
    test_duration: int = typer.Option(300, help="Test duration in seconds"),
    concurrent_streams: int = typer.Option(50, help="Number of concurrent streams to test"),
    output_dir: str = typer.Option("validation_results", help="Output directory for results"),
    enable_monitoring: bool = typer.Option(True, help="Enable performance monitoring"),
):
    """Run comprehensive optimization validation (5-15 minutes)."""

    async def run_comprehensive_validation():
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        try:
            # Initialize components with progress tracking
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:

                # Initialize components
                init_task = progress.add_task("Initializing system components...", total=None)

                redis_manager = RedisQueueManager(
                    redis_url=redis_url,
                    enable_blosc_compression=True,
                    enable_adaptive_batching=True
                )
                await redis_manager.connect()

                grpc_service = StreamingServiceImpl(
                    redis_manager=redis_manager,
                    enable_blosc_compression=True,
                    connection_pool_size=100
                )

                kafka_config = {
                    "bootstrap_servers": kafka_servers.split(","),
                    "topic_prefix": "validation-test",
                    "enable_blosc_compression": True
                }

                kafka_producer = create_kafka_producer(kafka_config)
                await kafka_producer.start()

                progress.update(init_task, description="Components initialized ✅")

                # Set up validation framework
                validation_task = progress.add_task("Setting up validation framework...", total=None)

                targets = PerformanceTargets(
                    max_end_to_end_latency_ms=100.0,
                    min_success_rate_percent=99.95,
                    min_concurrent_streams=concurrent_streams,
                    min_compression_ratio_improvement=0.6
                )

                validator = await create_end_to_end_validator(targets)

                progress.update(validation_task, description="Validation framework ready ✅")

                # Set up performance monitoring
                if enable_monitoring:
                    monitor_task = progress.add_task("Setting up performance monitoring...", total=None)

                    components = {
                        "grpc_service": grpc_service,
                        "redis_manager": redis_manager,
                        "kafka_producer": kafka_producer
                    }

                    monitor = await create_performance_monitor(
                        components,
                        baseline_file=str(output_path / "performance_baseline.json")
                    )

                    await monitor.start_monitoring()
                    progress.update(monitor_task, description="Performance monitoring active ✅")

            # Run comprehensive validation with live updates
            console.print("\n🚀 [bold]Starting Comprehensive Validation Suite[/bold]")
            console.print(f"Duration: {test_duration}s | Concurrent Streams: {concurrent_streams}")
            console.print("-" * 60)

            validation_results = await validator.run_comprehensive_validation(
                grpc_service=grpc_service,
                redis_manager=redis_manager,
                kafka_producer=kafka_producer,
                kafka_consumer=None  # Optional for this validation
            )

            # Run performance benchmark if monitoring enabled
            if enable_monitoring:
                console.print("\n📊 Running performance benchmark...")
                benchmark_results = await monitor.run_performance_benchmark(
                    duration_seconds=min(test_duration, 180)  # Max 3 minutes for benchmark
                )
                validation_results["benchmark_results"] = benchmark_results

                # Get performance report
                performance_report = monitor.get_performance_report()
                validation_results["performance_report"] = performance_report

                await monitor.stop_monitoring()

            # Cleanup
            await kafka_producer.stop()
            await redis_manager.disconnect()
            await validator.cleanup()

            return validation_results

        except Exception as e:
            console.print(f"[red]Comprehensive validation failed: {e}")
            import traceback
            console.print(f"[red]Error details: {traceback.format_exc()}")
            return {"validation_type": "comprehensive", "error": str(e), "overall_status": "ERROR"}

    # Run validation
    results = asyncio.run(run_comprehensive_validation())

    # Display results
    display_comprehensive_results(results)

    # Save detailed results
    output_path = Path(output_dir)
    results_file = output_path / f"comprehensive_validation_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"\n✅ Detailed results saved to {results_file}")


@app.command("monitor")
def start_monitoring(
    redis_url: str = typer.Option("redis://localhost:6379", help="Redis connection URL"),
    kafka_servers: str = typer.Option("localhost:9092", help="Kafka bootstrap servers"),
    baseline_file: str = typer.Option("performance_baseline.json", help="Performance baseline file"),
    interval: int = typer.Option(30, help="Monitoring interval in seconds"),
    duration: int = typer.Option(3600, help="Monitoring duration in seconds (0 for indefinite)"),
):
    """Start continuous performance monitoring and regression detection."""

    async def run_monitoring():
        try:
            # Initialize components
            console.print("🔍 [bold]Starting Performance Monitoring[/bold]")

            redis_manager = RedisQueueManager(redis_url=redis_url)
            await redis_manager.connect()

            grpc_service = StreamingServiceImpl(
                redis_manager=redis_manager,
                enable_blosc_compression=True
            )

            kafka_producer = create_kafka_producer({
                "bootstrap_servers": kafka_servers.split(","),
                "enable_blosc_compression": True
            })
            await kafka_producer.start()

            # Set up monitoring
            components = {
                "grpc_service": grpc_service,
                "redis_manager": redis_manager,
                "kafka_producer": kafka_producer
            }

            monitor = await create_performance_monitor(components, baseline_file)
            await monitor.start_monitoring(measurement_interval=interval)

            console.print(f"📊 Monitoring active with {interval}s intervals")

            # Monitor for specified duration
            start_time = time.time()

            try:
                while duration == 0 or (time.time() - start_time) < duration:
                    # Display current status
                    report = monitor.get_performance_report()
                    display_monitoring_status(report)

                    await asyncio.sleep(60)  # Update display every minute

            except KeyboardInterrupt:
                console.print("\n⏹️  Monitoring stopped by user")

            # Cleanup
            await monitor.stop_monitoring()
            await kafka_producer.stop()
            await redis_manager.disconnect()

            # Final report
            final_report = monitor.get_performance_report()
            console.print("\n📋 [bold]Final Performance Report[/bold]")
            display_performance_report(final_report)

        except Exception as e:
            console.print(f"[red]Monitoring failed: {e}")

    asyncio.run(run_monitoring())


def display_quick_results(results: dict[str, Any], verbose: bool = False):
    """Display quick validation results."""
    console.print("\n📊 [bold]Quick Validation Results[/bold]")

    if "error" in results:
        console.print(f"[red]❌ Validation failed: {results['error']}")
        return

    # Overall status
    status_color = "green" if results["overall_status"] == "PASSED" else "red"
    status_emoji = "✅" if results["overall_status"] == "PASSED" else "❌"

    console.print(f"\n{status_emoji} [bold {status_color}]Overall Status: {results['overall_status']}[/bold {status_color}]")
    console.print(f"⏱️  Duration: {results.get('duration_seconds', 0):.2f}s")

    # Test results table
    table = Table(title="Test Results", show_header=True, header_style="bold magenta")
    table.add_column("Test", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Key Metrics", style="yellow")

    tests = results.get("tests", {})

    for test_name, test_data in tests.items():
        status = "✅ PASS" if test_data.get("passed", False) else "❌ FAIL"

        if test_name == "blosc_compression":
            metrics = f"{test_data['size_reduction_percent']:.1f}% reduction, {test_data['compression_time_ms']:.2f}ms"
        elif test_name == "grpc_serialization":
            metrics = f"{test_data['avg_time_ms']:.2f}ms avg, {test_data['max_time_ms']:.2f}ms max"
        elif test_name == "redis_queue":
            metrics = f"{test_data['avg_time_ms']:.2f}ms avg"
        else:
            metrics = "N/A"

        table.add_row(test_name.replace("_", " ").title(), status, metrics)

    console.print(table)

    if verbose and "tests" in results:
        console.print("\n📋 [bold]Detailed Test Data[/bold]")
        for test_name, test_data in results["tests"].items():
            console.print(f"\n[cyan]{test_name.replace('_', ' ').title()}:[/cyan]")
            for key, value in test_data.items():
                if key != "passed":
                    console.print(f"  {key}: {value}")


def display_comprehensive_results(results: dict[str, Any]):
    """Display comprehensive validation results."""
    console.print("\n🎯 [bold]Comprehensive Validation Results[/bold]")

    if "error" in results:
        console.print(f"[red]❌ Validation failed: {results['error']}")
        return

    # Validation summary
    summary = results.get("validation_summary", {})

    summary_table = Table(title="Validation Summary", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow")

    summary_table.add_row("Total Tests", str(summary.get("total_tests", 0)))
    summary_table.add_row("Passed Tests", f"[green]{summary.get('passed_tests', 0)}[/green]")
    summary_table.add_row("Failed Tests", f"[red]{summary.get('failed_tests', 0)}[/red]")
    summary_table.add_row("Warning Tests", f"[yellow]{summary.get('warning_tests', 0)}[/yellow]")
    summary_table.add_row("Success Rate", f"{summary.get('overall_success_rate_percent', 0):.1f}%")
    summary_table.add_row("Duration", f"{summary.get('total_duration_seconds', 0):.1f}s")

    console.print(summary_table)

    # Key performance metrics
    key_metrics = results.get("key_performance_metrics", {})
    if key_metrics:
        console.print("\n🚀 [bold]Key Performance Metrics[/bold]")

        metrics_table = Table()
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="yellow")
        metrics_table.add_column("Target", style="green")

        metrics_table.add_row(
            "P99 Latency",
            f"{key_metrics.get('end_to_end_latency_p99_ms', 0):.2f}ms",
            "<100ms"
        )
        metrics_table.add_row(
            "Success Rate",
            f"{key_metrics.get('success_rate_percent', 0):.2f}%",
            "≥99.95%"
        )
        metrics_table.add_row(
            "Concurrent Streams",
            str(key_metrics.get('concurrent_streams_tested', 0)),
            "≥100"
        )
        metrics_table.add_row(
            "Throughput",
            f"{key_metrics.get('throughput_requests_per_second', 0):.1f} RPS",
            "High"
        )

        console.print(metrics_table)

    # Recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        console.print("\n💡 [bold]Recommendations[/bold]")
        for i, rec in enumerate(recommendations, 1):
            console.print(f"  {i}. {rec}")

    # Performance targets validation
    targets_validation = results.get("performance_targets_validation", {})
    if targets_validation:
        targets_met = targets_validation.get("targets_met", False)
        status_emoji = "✅" if targets_met else "❌"
        console.print(f"\n{status_emoji} [bold]Performance Targets: {'MET' if targets_met else 'NOT MET'}[/bold]")


def display_monitoring_status(report: dict[str, Any]):
    """Display current monitoring status."""
    # Clear screen and show current status
    console.clear()

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1)
    )

    # Header
    header_text = "[bold blue]🔍 Performance Monitoring Dashboard[/bold blue]"
    layout["header"].update(Panel(Align.center(header_text)))

    # Main content
    main_layout = Layout()
    main_layout.split_row(
        Layout(name="current", ratio=1),
        Layout(name="alerts", ratio=1)
    )

    # Current performance
    current_perf = report.get("current_performance", {})
    if current_perf:
        current_table = Table(title="Current Performance", show_header=True)
        current_table.add_column("Metric", style="cyan")
        current_table.add_column("Value", style="yellow")

        current_table.add_row("Avg Latency", f"{current_perf.get('avg_latency_ms', 0):.2f}ms")
        current_table.add_row("P95 Latency", f"{current_perf.get('p95_latency_ms', 0):.2f}ms")
        current_table.add_row("P99 Latency", f"{current_perf.get('p99_latency_ms', 0):.2f}ms")
        current_table.add_row("Throughput", f"{current_perf.get('avg_throughput_rps', 0):.1f} RPS")
        current_table.add_row("Success Rate", f"{current_perf.get('avg_success_rate', 0):.2f}%")
        current_table.add_row("Samples", str(current_perf.get('sample_count', 0)))

        main_layout["current"].update(current_table)

    # Active alerts
    alerts = report.get("active_alerts", [])
    if alerts:
        alerts_table = Table(title="Active Alerts", show_header=True)
        alerts_table.add_column("Metric", style="cyan")
        alerts_table.add_column("Severity", style="red")
        alerts_table.add_column("Regression", style="yellow")

        for alert in alerts[-5:]:  # Show last 5 alerts
            severity_color = "red" if alert["severity"] == "critical" else "yellow"
            alerts_table.add_row(
                alert["metric"],
                f"[{severity_color}]{alert['severity'].upper()}[/{severity_color}]",
                f"{alert['regression_percent']:.1f}%"
            )

        main_layout["alerts"].update(alerts_table)
    else:
        main_layout["alerts"].update(Panel("[green]No active alerts[/green]", title="Active Alerts"))

    layout["main"].update(main_layout)
    console.print(layout)


def display_performance_report(report: dict[str, Any]):
    """Display final performance report."""
    console.print(Panel.fit(f"📋 [bold]Performance Report - {datetime.fromtimestamp(report['report_timestamp']).strftime('%Y-%m-%d %H:%M:%S')}[/bold]"))

    # Current performance
    current_perf = report.get("current_performance", {})
    if current_perf:
        perf_table = Table(title="Performance Summary")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="yellow")

        for key, value in current_perf.items():
            if isinstance(value, float):
                perf_table.add_row(key.replace("_", " ").title(), f"{value:.2f}")
            else:
                perf_table.add_row(key.replace("_", " ").title(), str(value))

        console.print(perf_table)

    # Performance trends
    trends = report.get("performance_trends", {})
    if trends:
        console.print("\n📈 [bold]Performance Trends[/bold]")
        for metric, trend in trends.items():
            trend_emoji = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
            console.print(f"  {trend_emoji} {metric.replace('_', ' ').title()}: {trend:+.2f}%")


if __name__ == "__main__":
    app()
