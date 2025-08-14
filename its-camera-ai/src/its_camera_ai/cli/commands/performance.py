"""Performance optimization CLI commands for ITS Camera AI.

This module provides CLI commands to manage and monitor the performance
optimization system including GPU optimization, caching, connection pooling,
latency monitoring, and adaptive quality management.
"""

import asyncio
import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ...performance import (
    OptimizationStrategy,
    PipelineStage,
    create_performance_optimizer,
    create_production_optimization_config,
)

console = Console()
performance_cli = typer.Typer(name="performance", help="Performance optimization management")


@performance_cli.command("status")
def performance_status(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed metrics"),
    output_format: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
) -> None:
    """Show performance optimization status and metrics."""

    async def _get_status():
        try:
            # Create a basic optimizer to check status
            config = create_production_optimization_config(
                max_concurrent_streams=100,
                target_latency_ms=100.0,
                strategy=OptimizationStrategy.BALANCED
            )

            optimizer = await create_performance_optimizer(config)
            status = optimizer.get_optimization_status()

            if detailed:
                metrics = optimizer.get_comprehensive_metrics()
                status["detailed_metrics"] = metrics

            await optimizer.stop()
            return status

        except Exception as e:
            return {"error": str(e), "status": "unavailable"}

    status = asyncio.run(_get_status())

    if output_format == "json":
        console.print_json(json.dumps(status, indent=2))
        return

    # Table format
    if "error" in status:
        console.print(f"[red]Performance optimization unavailable: {status['error']}[/red]")
        return

    # Main status table
    table = Table(title="Performance Optimization Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")

    table.add_row("Initialized", "✓" if status["initialized"] else "✗", str(status["initialized"]))
    table.add_row("Running", "✓" if status["running"] else "✗", str(status["running"]))
    table.add_row("Strategy", status["strategy"], "Optimization strategy")

    if status.get("initialization_time_ms"):
        table.add_row("Init Time", f"{status['initialization_time_ms']:.2f}ms", "Initialization time")

    # Component status
    components = status.get("components", {})
    for component, enabled in components.items():
        table.add_row(
            component.replace("_", " ").title(),
            "✓" if enabled else "✗",
            "Enabled" if enabled else "Disabled"
        )

    console.print(table)

    # Performance summary if available
    if "performance_summary" in status:
        summary = status["performance_summary"]
        summary_table = Table(title="Performance Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        for key, value in summary.items():
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            summary_table.add_row(key.replace("_", " ").title(), formatted_value)

        console.print(summary_table)


@performance_cli.command("benchmark")
def performance_benchmark(
    streams: int = typer.Option(10, "--streams", "-s", help="Number of concurrent streams"),
    duration: int = typer.Option(30, "--duration", "-d", help="Benchmark duration in seconds"),
    strategy: str = typer.Option("latency_optimized", "--strategy", help="Optimization strategy"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Output file for results"),
) -> None:
    """Run performance benchmark with specified parameters."""

    async def _run_benchmark():
        try:
            # Parse strategy
            strategy_enum = OptimizationStrategy(strategy)

            config = create_production_optimization_config(
                max_concurrent_streams=streams,
                target_latency_ms=100.0,
                strategy=strategy_enum
            )

            console.print("[bold]Starting performance benchmark[/bold]")
            console.print(f"Streams: {streams}, Duration: {duration}s, Strategy: {strategy}")

            # Initialize optimizer
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                init_task = progress.add_task("Initializing optimizer...", total=None)

                optimizer = await create_performance_optimizer(config)
                progress.update(init_task, description="Optimizer initialized ✓")

                # Simulate concurrent stream processing
                benchmark_task = progress.add_task(f"Running benchmark for {duration}s...", total=duration)

                # Store benchmark results
                results = {
                    "config": {
                        "streams": streams,
                        "duration": duration,
                        "strategy": strategy
                    },
                    "metrics": [],
                    "start_time": time.time()
                }

                # Run benchmark
                start_time = time.time()
                tasks = []

                for i in range(streams):
                    camera_id = f"benchmark_camera_{i:03d}"

                    async def simulate_stream(cam_id: str):
                        async with optimizer.optimize_stream_processing(cam_id) as context:
                            # Simulate processing stages
                            stages = [
                                (PipelineStage.CAMERA_CAPTURE, 15.0),
                                (PipelineStage.PREPROCESSING, 5.0),
                                (PipelineStage.ML_INFERENCE, 35.0),
                                (PipelineStage.POSTPROCESSING, 3.0),
                                (PipelineStage.ENCODING, 8.0),
                                (PipelineStage.NETWORK_TRANSMISSION, 12.0)
                            ]

                            for stage, base_latency in stages:
                                # Add some random variation
                                import random
                                latency = base_latency + random.uniform(-5.0, 5.0)

                                await optimizer.record_pipeline_measurement(
                                    cam_id, stage, latency
                                )

                                # Small delay between stages
                                await asyncio.sleep(0.001)

                            # Cache some test fragments
                            fragment_key = f"{cam_id}_fragment_{int(time.time())}"
                            fragment_data = b"test_fragment_data_12345"
                            await optimizer.cache_fragment(fragment_key, fragment_data)

                    tasks.append(simulate_stream(camera_id))

                # Run all streams concurrently
                await asyncio.gather(*tasks)

                # Wait for benchmark duration
                elapsed = 0
                while elapsed < duration:
                    await asyncio.sleep(1.0)
                    elapsed = time.time() - start_time
                    progress.update(benchmark_task, completed=elapsed)

                # Collect final metrics
                final_metrics = optimizer.get_comprehensive_metrics()
                results["metrics"].append(final_metrics)
                results["end_time"] = time.time()
                results["actual_duration"] = elapsed

                await optimizer.stop()

                return results

        except Exception as e:
            console.print(f"[red]Benchmark failed: {e}[/red]")
            return {"error": str(e)}

    results = asyncio.run(_run_benchmark())

    if "error" in results:
        console.print(f"[red]Benchmark failed: {results['error']}[/red]")
        return

    # Display results
    console.print("\n[bold green]Benchmark Results[/bold green]")

    config = results["config"]
    console.print(f"Configuration: {config['streams']} streams, {config['duration']}s duration")
    console.print(f"Strategy: {config['strategy']}")
    console.print(f"Actual duration: {results['actual_duration']:.2f}s")

    # Extract key metrics
    if results["metrics"]:
        metrics = results["metrics"][-1]  # Get final metrics

        # Latency metrics
        if "latency_monitoring" in metrics:
            latency_data = metrics["latency_monitoring"]
            if "stages" in latency_data:
                latency_table = Table(title="Pipeline Latency Metrics")
                latency_table.add_column("Stage", style="cyan")
                latency_table.add_column("P50 (ms)", style="green")
                latency_table.add_column("P95 (ms)", style="yellow")
                latency_table.add_column("P99 (ms)", style="red")

                for stage, stage_data in latency_data["stages"].items():
                    if stage_data["short_window"]:
                        sw = stage_data["short_window"]
                        latency_table.add_row(
                            stage.replace("_", " ").title(),
                            f"{sw.get('median_latency_ms', 0):.1f}",
                            f"{sw.get('p95_latency_ms', 0):.1f}",
                            f"{sw.get('p99_latency_ms', 0):.1f}"
                        )

                console.print(latency_table)

        # Cache metrics
        if "caching" in metrics:
            cache_data = metrics["caching"]
            if "performance" in cache_data:
                perf = cache_data["performance"]
                console.print(f"Cache hit rate: {perf.get('overall_hit_rate_percent', 0):.1f}%")
                console.print(f"Total cache requests: {perf.get('total_requests', 0)}")

    # Save results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.write_text(json.dumps(results, indent=2))
        console.print(f"Results saved to: {output_path}")


@performance_cli.command("monitor")
def performance_monitor(
    interval: int = typer.Option(5, "--interval", "-i", help="Monitoring interval in seconds"),
    duration: int = typer.Option(60, "--duration", "-d", help="Monitoring duration in seconds"),
) -> None:
    """Monitor performance optimization metrics in real-time."""

    async def _monitor():
        try:
            config = create_production_optimization_config()
            optimizer = await create_performance_optimizer(config)

            console.print("[bold]Starting performance monitoring[/bold]")
            console.print(f"Interval: {interval}s, Duration: {duration}s")
            console.print("Press Ctrl+C to stop monitoring\n")

            start_time = time.time()

            while time.time() - start_time < duration:
                try:
                    # Collect current metrics
                    metrics = optimizer.get_comprehensive_metrics()

                    # Display timestamp
                    current_time = time.strftime("%H:%M:%S")
                    console.print(f"[bold blue]{current_time}[/bold blue]")

                    # Show key metrics
                    if "latency_monitoring" in metrics:
                        sla_rate = metrics["latency_monitoring"].get("overall_sla_violation_rate", 0)
                        console.print(f"SLA Compliance: {100-sla_rate:.1f}%")

                    if "caching" in metrics and "performance" in metrics["caching"]:
                        hit_rate = metrics["caching"]["performance"].get("overall_hit_rate_percent", 0)
                        console.print(f"Cache Hit Rate: {hit_rate:.1f}%")

                    if "gpu_optimization" in metrics:
                        gpu_util = metrics["gpu_optimization"].get("gpu_utilization_percent", 0)
                        console.print(f"GPU Utilization: {gpu_util:.1f}%")

                    console.print("-" * 40)

                    await asyncio.sleep(interval)

                except KeyboardInterrupt:
                    break

            await optimizer.stop()
            console.print("\n[green]Monitoring completed[/green]")

        except Exception as e:
            console.print(f"[red]Monitoring failed: {e}[/red]")

    try:
        asyncio.run(_monitor())
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring interrupted by user[/yellow]")


@performance_cli.command("optimize")
def optimize_streaming(
    strategy: str = typer.Option("latency_optimized", "--strategy", help="Optimization strategy"),
    max_streams: int = typer.Option(100, "--max-streams", help="Maximum concurrent streams"),
    target_latency: float = typer.Option(100.0, "--target-latency", help="Target latency in ms"),
    enable_gpu: bool = typer.Option(True, "--gpu/--no-gpu", help="Enable GPU optimization"),
    enable_cache: bool = typer.Option(True, "--cache/--no-cache", help="Enable caching"),
    config_file: str | None = typer.Option(None, "--config", "-c", help="Configuration file"),
) -> None:
    """Initialize and start performance optimization for streaming system."""

    async def _optimize():
        try:
            # Parse strategy
            strategy_enum = OptimizationStrategy(strategy)

            # Create configuration
            if config_file:
                # Load configuration from file
                config_path = Path(config_file)
                if config_path.exists():
                    config_data = json.loads(config_path.read_text())
                    # Create config from loaded data
                    from ...performance.optimization_config import OptimizationConfig
                    config = OptimizationConfig(**config_data)
                else:
                    console.print(f"[red]Configuration file not found: {config_file}[/red]")
                    return
            else:
                # Create default configuration with parameters
                config = create_production_optimization_config(
                    max_concurrent_streams=max_streams,
                    target_latency_ms=target_latency,
                    strategy=strategy_enum
                )

                # Apply CLI overrides
                if not enable_gpu:
                    config.gpu.gpu_memory_pool_size_gb = 0.0
                if not enable_cache:
                    config.caching.l1_cache_size_mb = 0

            console.print("[bold]Initializing performance optimization[/bold]")
            console.print(f"Strategy: {strategy}")
            console.print(f"Max streams: {max_streams}")
            console.print(f"Target latency: {target_latency}ms")
            console.print(f"GPU optimization: {'enabled' if enable_gpu else 'disabled'}")
            console.print(f"Caching: {'enabled' if enable_cache else 'disabled'}")

            # Initialize optimizer
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Initializing optimizer...", total=None)

                optimizer = await create_performance_optimizer(config)

                progress.update(task, description="Optimizer initialized ✓")

            # Show optimization status
            status = optimizer.get_optimization_status()

            panel = Panel.fit(
                f"""[green]Performance optimization initialized successfully![/green]

Status: {status['initialized'] and 'Initialized' or 'Not initialized'}
Strategy: {status['strategy']}
Components:
  • GPU Optimizer: {'✓' if status['components']['gpu_optimizer'] else '✗'}
  • Cache Manager: {'✓' if status['components']['cache_manager'] else '✗'}  
  • Connection Pool: {'✓' if status['components']['connection_optimizer'] else '✗'}
  • Latency Monitor: {'✓' if status['components']['latency_monitor'] else '✗'}
  • Quality Manager: {'✓' if status['components']['quality_manager'] else '✗'}

The optimizer is ready to handle streaming workloads with optimized performance.
""",
                title="Performance Optimization",
                border_style="green"
            )

            console.print(panel)

            # Keep running for demonstration
            console.print("\n[yellow]Optimizer running... Press Ctrl+C to stop[/yellow]")

            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                pass

            console.print("\n[blue]Stopping optimizer...[/blue]")
            await optimizer.stop()
            console.print("[green]Performance optimization stopped[/green]")

        except Exception as e:
            console.print(f"[red]Optimization failed: {e}[/red]")
            raise typer.Exit(1)

    try:
        asyncio.run(_optimize())
    except KeyboardInterrupt:
        console.print("\n[yellow]Optimization interrupted by user[/yellow]")


@performance_cli.command("config")
def generate_config(
    strategy: str = typer.Option("balanced", "--strategy", help="Optimization strategy"),
    output: str = typer.Option("performance_config.json", "--output", "-o", help="Output file"),
    pretty: bool = typer.Option(True, "--pretty/--compact", help="Pretty print JSON"),
) -> None:
    """Generate performance optimization configuration file."""

    try:
        # Parse strategy
        strategy_enum = OptimizationStrategy(strategy)

        # Create configuration
        config = create_production_optimization_config(
            max_concurrent_streams=100,
            target_latency_ms=100.0,
            strategy=strategy_enum
        )

        # Convert to dictionary
        config_dict = config.to_dict()

        # Write to file
        output_path = Path(output)
        if pretty:
            json_content = json.dumps(config_dict, indent=2)
        else:
            json_content = json.dumps(config_dict)

        output_path.write_text(json_content)

        console.print(f"[green]Configuration generated: {output_path}[/green]")
        console.print(f"Strategy: {strategy}")
        console.print(f"File size: {len(json_content)} bytes")

        # Show sample configuration
        sample_table = Table(title="Sample Configuration")
        sample_table.add_column("Section", style="cyan")
        sample_table.add_column("Key Settings", style="green")

        sample_table.add_row("Strategy", f"{strategy} optimization")
        sample_table.add_row("GPU", f"Memory pool: {config.gpu.gpu_memory_pool_size_gb}GB")
        sample_table.add_row("Caching", f"L1 cache: {config.caching.l1_cache_size_mb}MB")
        sample_table.add_row("Latency", f"SLA target: {config.latency_monitoring.latency_sla_ms}ms")
        sample_table.add_row("Quality", f"Adaptive: {config.adaptive_quality.enable_adaptive_quality}")

        console.print(sample_table)

    except Exception as e:
        console.print(f"[red]Configuration generation failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    performance_cli()
