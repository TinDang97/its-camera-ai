"""CLI commands for performance optimization and monitoring.

This module provides CLI commands for:
- Cache performance optimization and monitoring
- Database query optimization and tuning
- Performance testing and benchmarking
- Real-time performance monitoring and alerting
- System health checks and diagnostics
"""

import asyncio
import json
import time
from datetime import UTC, datetime

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...core.config import Settings
from ...core.logging import get_logger
from ...services.cache import EnhancedCacheService
from ...services.database_optimizer import DatabaseOptimizer
from ...services.performance_monitor import (
    AlertSeverity,
    PerformanceMonitor,
)
from ..backend.database_manager import DatabaseManager

console = Console()
logger = get_logger(__name__)

app = typer.Typer(name="performance", help="Performance optimization and monitoring commands")


@app.command("status")
async def performance_status(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed performance metrics"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format")
):
    """Show current system performance status."""
    try:
        settings = Settings()

        # Initialize components
        db_manager = DatabaseManager(settings)
        cache_service = await _get_cache_service()
        db_optimizer = DatabaseOptimizer(db_manager, settings)

        performance_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "cache": {},
            "database": {},
            "overall_status": "healthy"
        }

        # Get cache performance
        if cache_service:
            cache_metrics = cache_service.get_metrics()
            performance_data["cache"] = {
                "overall_hit_rate": cache_metrics.get("overall_hit_rate", 0),
                "l1_hit_rate": cache_metrics.get("l1_hit_rate", 0),
                "l2_hit_rate": cache_metrics.get("l2_hit_rate", 0),
                "operations_per_second": cache_metrics.get("operations_per_second", 0),
                "status": "healthy" if cache_metrics.get("overall_hit_rate", 0) > 0.8 else "degraded"
            }

        # Get database performance
        db_summary = db_optimizer.get_performance_summary()
        performance_data["database"] = {
            "avg_response_time_ms": db_summary.get("recent_avg_response_time_ms", 0),
            "slow_query_percentage": db_summary.get("slow_query_percentage", 0),
            "total_queries_analyzed": db_summary.get("total_queries_analyzed", 0),
            "index_recommendations": db_summary.get("index_recommendations", 0),
            "health_alerts": db_summary.get("health_alerts", 0),
            "status": db_summary.get("performance_status", "unknown")
        }

        # Determine overall status
        cache_ok = performance_data["cache"].get("status") == "healthy"
        db_ok = performance_data["database"].get("status") == "good"
        performance_data["overall_status"] = "healthy" if cache_ok and db_ok else "degraded"

        if json_output:
            rprint(json.dumps(performance_data, indent=2))
            return

        # Rich output
        status_color = "green" if performance_data["overall_status"] == "healthy" else "yellow"

        console.print(Panel(
            f"[{status_color}]System Performance Status: {performance_data['overall_status'].upper()}[/{status_color}]",
            title="ITS Camera AI Performance"
        ))

        # Cache metrics table
        cache_table = Table(title="Cache Performance")
        cache_table.add_column("Metric", style="cyan")
        cache_table.add_column("Value", style="magenta")
        cache_table.add_column("Status", style="green")

        cache_data = performance_data["cache"]
        cache_table.add_row("Overall Hit Rate", f"{cache_data.get('overall_hit_rate', 0):.1%}", cache_data.get('status', 'unknown'))
        cache_table.add_row("L1 Hit Rate", f"{cache_data.get('l1_hit_rate', 0):.1%}", "")
        cache_table.add_row("L2 Hit Rate", f"{cache_data.get('l2_hit_rate', 0):.1%}", "")
        cache_table.add_row("Operations/sec", f"{cache_data.get('operations_per_second', 0):.0f}", "")

        console.print(cache_table)

        # Database metrics table
        db_table = Table(title="Database Performance")
        db_table.add_column("Metric", style="cyan")
        db_table.add_column("Value", style="magenta")
        db_table.add_column("Status", style="green")

        db_data = performance_data["database"]
        db_table.add_row("Avg Response Time", f"{db_data.get('avg_response_time_ms', 0):.1f}ms", db_data.get('status', 'unknown'))
        db_table.add_row("Slow Queries", f"{db_data.get('slow_query_percentage', 0):.1f}%", "")
        db_table.add_row("Queries Analyzed", f"{db_data.get('total_queries_analyzed', 0):,}", "")
        db_table.add_row("Index Recommendations", f"{db_data.get('index_recommendations', 0)}", "")
        db_table.add_row("Health Alerts", f"{db_data.get('health_alerts', 0)}", "")

        console.print(db_table)

        if detailed:
            await _show_detailed_metrics(cache_service, db_optimizer)

    except Exception as e:
        console.print(f"[red]Error getting performance status: {e}[/red]")
        raise typer.Exit(1)


@app.command("optimize")
async def optimize_performance(
    cache_only: bool = typer.Option(False, "--cache-only", help="Optimize cache only"),
    database_only: bool = typer.Option(False, "--database-only", help="Optimize database only"),
    apply_recommendations: bool = typer.Option(False, "--apply", help="Apply optimization recommendations"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Show recommendations without applying")
):
    """Optimize system performance."""
    try:
        settings = Settings()

        console.print("[yellow]Starting performance optimization...[/yellow]")

        # Initialize services
        db_manager = DatabaseManager(settings)
        cache_service = await _get_cache_service()
        db_optimizer = DatabaseOptimizer(db_manager, settings)

        optimization_results = {
            "cache": {},
            "database": {},
            "applied": False
        }

        # Cache optimization
        if not database_only and cache_service:
            console.print("[cyan]Analyzing cache performance...[/cyan]")

            # Clear expired entries
            if cache_service.l1_cache:
                expired = await cache_service.l1_cache.cleanup_expired()
                optimization_results["cache"]["expired_cleaned"] = expired

            # Get cache metrics
            cache_metrics = cache_service.get_metrics()
            optimization_results["cache"]["current_metrics"] = cache_metrics

            console.print(f"Cache hit rate: {cache_metrics.get('overall_hit_rate', 0):.1%}")

        # Database optimization
        if not cache_only:
            console.print("[cyan]Analyzing database performance...[/cyan]")

            async with db_manager.get_session() as session:
                # Analyze query performance
                with console.status("[bold green]Analyzing query performance..."):
                    query_metrics = await db_optimizer.analyze_query_performance(session)

                optimization_results["database"]["query_analysis"] = len(query_metrics)

                # Generate index recommendations
                with console.status("[bold green]Generating index recommendations..."):
                    index_recommendations = await db_optimizer.generate_index_recommendations(session)

                optimization_results["database"]["index_recommendations"] = len(index_recommendations)

                # Display recommendations
                if index_recommendations:
                    rec_table = Table(title="Index Recommendations")
                    rec_table.add_column("Table", style="cyan")
                    rec_table.add_column("Type", style="yellow")
                    rec_table.add_column("Benefit", style="green")
                    rec_table.add_column("Reason", style="white")

                    for rec in index_recommendations[:10]:  # Show top 10
                        rec_table.add_row(
                            rec.table_name,
                            rec.index_type,
                            f"{rec.estimated_benefit:.1%}",
                            rec.reason
                        )

                    console.print(rec_table)

                # Apply recommendations if requested
                if apply_recommendations and not dry_run and index_recommendations:
                    if typer.confirm("Apply these optimization recommendations?"):
                        with console.status("[bold green]Applying optimizations..."):
                            apply_result = await db_optimizer.apply_optimization_recommendations(
                                session, index_recommendations
                            )

                        optimization_results["applied"] = True
                        optimization_results["database"]["applied"] = apply_result["applied"]
                        optimization_results["database"]["failed"] = apply_result["failed"]

                        console.print(f"[green]Applied {apply_result['applied']} optimizations[/green]")
                        if apply_result["failed"] > 0:
                            console.print(f"[yellow]Failed to apply {apply_result['failed']} optimizations[/yellow]")

        # Summary
        console.print(Panel(
            f"Optimization complete!\n"
            f"Cache optimizations: {len(optimization_results.get('cache', {}))}\n"
            f"Database recommendations: {optimization_results.get('database', {}).get('index_recommendations', 0)}\n"
            f"Applied: {'Yes' if optimization_results['applied'] else 'No (dry run)'}",
            title="Optimization Summary"
        ))

    except Exception as e:
        console.print(f"[red]Error during optimization: {e}[/red]")
        raise typer.Exit(1)


@app.command("monitor")
async def start_monitoring(
    duration: int = typer.Option(60, "--duration", "-d", help="Monitoring duration in seconds"),
    interval: int = typer.Option(10, "--interval", "-i", help="Monitoring interval in seconds"),
    alerts_only: bool = typer.Option(False, "--alerts-only", help="Show only alerts"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Output file for metrics")
):
    """Start real-time performance monitoring."""
    try:
        settings = Settings()

        # Initialize monitoring
        cache_service = await _get_cache_service()
        db_manager = DatabaseManager(settings)
        db_optimizer = DatabaseOptimizer(db_manager, settings)

        # Alert callback
        async def alert_callback(alert):
            severity_colors = {
                AlertSeverity.INFO: "blue",
                AlertSeverity.WARNING: "yellow",
                AlertSeverity.CRITICAL: "red",
                AlertSeverity.FATAL: "red bold"
            }
            color = severity_colors.get(alert.severity, "white")
            console.print(f"[{color}]ALERT [{alert.severity.value.upper()}]: {alert.message}[/{color}]")

        monitor = PerformanceMonitor(
            cache_service=cache_service,
            database_optimizer=db_optimizer,
            alert_callback=alert_callback
        )

        monitor.monitoring_interval = interval

        console.print(f"[green]Starting performance monitoring for {duration} seconds...[/green]")

        # Start monitoring
        await monitor.start_monitoring()

        # Collect metrics for specified duration
        metrics_data = []
        start_time = time.time()

        try:
            while time.time() - start_time < duration:
                if not alerts_only:
                    summary = await monitor.get_performance_summary()

                    # Display current metrics
                    current_time = datetime.now(UTC)
                    console.print(f"\\n[cyan][{current_time.strftime('%H:%M:%S')}] Performance Metrics[/cyan]")

                    for metric_name, metric_data in summary.get("metrics", {}).items():
                        value = metric_data.get("current_value", 0)
                        status = metric_data.get("status", "unknown")
                        status_color = {"healthy": "green", "warning": "yellow", "critical": "red"}.get(status, "white")

                        console.print(f"  {metric_name}: [white]{value:.2f}[/white] [{status_color}]{status}[/{status_color}]")

                    # Store for output file
                    if output_file:
                        metrics_data.append({
                            "timestamp": current_time.isoformat(),
                            "metrics": summary["metrics"]
                        })

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            console.print("\\n[yellow]Monitoring interrupted by user[/yellow]")

        finally:
            await monitor.stop_monitoring()

        # Final summary
        final_summary = await monitor.get_performance_summary()

        console.print(Panel(
            f"Monitoring completed!\n"
            f"Active alerts: {final_summary['active_alerts']}\n"
            f"Total alerts: {final_summary['total_alerts_today']}\n"
            f"SLA compliance: {len([c for c in final_summary.get('sla_compliance', {}).values() if c.get('compliance_pct', 0) >= 95])}/{len(final_summary.get('sla_compliance', {}))} metrics",
            title="Monitoring Summary"
        ))

        # Save metrics to file
        if output_file and metrics_data:
            with open(output_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            console.print(f"[green]Metrics saved to {output_file}[/green]")

    except Exception as e:
        console.print(f"[red]Error during monitoring: {e}[/red]")
        raise typer.Exit(1)


@app.command("benchmark")
async def performance_benchmark(
    cache_only: bool = typer.Option(False, "--cache-only", help="Benchmark cache only"),
    database_only: bool = typer.Option(False, "--database-only", help="Benchmark database only"),
    duration: int = typer.Option(30, "--duration", "-d", help="Benchmark duration in seconds"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Concurrent operations"),
    output_file: str | None = typer.Option(None, "--output", "-o", help="Output file for results")
):
    """Run performance benchmarks."""
    try:
        settings = Settings()

        console.print(f"[yellow]Starting performance benchmark (duration: {duration}s, concurrency: {concurrency})[/yellow]")

        benchmark_results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "configuration": {
                "duration": duration,
                "concurrency": concurrency,
                "cache_only": cache_only,
                "database_only": database_only
            },
            "results": {}
        }

        # Cache benchmark
        if not database_only:
            console.print("[cyan]Running cache benchmark...[/cyan]")
            cache_service = await _get_cache_service()

            if cache_service:
                cache_results = await _run_cache_benchmark(cache_service, duration, concurrency)
                benchmark_results["results"]["cache"] = cache_results

                # Display cache results
                cache_table = Table(title="Cache Benchmark Results")
                cache_table.add_column("Metric", style="cyan")
                cache_table.add_column("Value", style="magenta")

                cache_table.add_row("Operations", f"{cache_results['total_operations']:,}")
                cache_table.add_row("Operations/sec", f"{cache_results['ops_per_second']:.0f}")
                cache_table.add_row("Avg Latency", f"{cache_results['avg_latency_ms']:.2f}ms")
                cache_table.add_row("P95 Latency", f"{cache_results['p95_latency_ms']:.2f}ms")
                cache_table.add_row("Error Rate", f"{cache_results['error_rate']:.2%}")

                console.print(cache_table)

        # Database benchmark
        if not cache_only:
            console.print("[cyan]Running database benchmark...[/cyan]")

            db_manager = DatabaseManager(settings)
            db_results = await _run_database_benchmark(db_manager, duration, concurrency)
            benchmark_results["results"]["database"] = db_results

            # Display database results
            db_table = Table(title="Database Benchmark Results")
            db_table.add_column("Metric", style="cyan")
            db_table.add_column("Value", style="magenta")

            db_table.add_row("Queries", f"{db_results['total_queries']:,}")
            db_table.add_row("Queries/sec", f"{db_results['queries_per_second']:.0f}")
            db_table.add_row("Avg Response Time", f"{db_results['avg_response_time_ms']:.2f}ms")
            db_table.add_row("P95 Response Time", f"{db_results['p95_response_time_ms']:.2f}ms")
            db_table.add_row("Error Rate", f"{db_results['error_rate']:.2%}")

            console.print(db_table)

        # Overall assessment
        status = "PASS"
        issues = []

        if "cache" in benchmark_results["results"]:
            cache_res = benchmark_results["results"]["cache"]
            if cache_res["ops_per_second"] < 1000:
                issues.append("Cache throughput below 1000 ops/sec")
                status = "FAIL"
            if cache_res["avg_latency_ms"] > 10:
                issues.append("Cache latency above 10ms")
                status = "FAIL"

        if "database" in benchmark_results["results"]:
            db_res = benchmark_results["results"]["database"]
            if db_res["queries_per_second"] < 100:
                issues.append("Database throughput below 100 queries/sec")
                status = "FAIL"
            if db_res["avg_response_time_ms"] > 100:
                issues.append("Database response time above 100ms")
                status = "FAIL"

        status_color = "green" if status == "PASS" else "red"
        console.print(Panel(
            f"[{status_color}]Benchmark Status: {status}[/{status_color}]\\n" +
            ("\\n".join(f"⚠ {issue}" for issue in issues) if issues else "All performance targets met"),
            title="Benchmark Summary"
        ))

        # Save results
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            console.print(f"[green]Benchmark results saved to {output_file}[/green]")

    except Exception as e:
        console.print(f"[red]Error during benchmark: {e}[/red]")
        raise typer.Exit(1)


@app.command("health")
async def health_check(
    fix_issues: bool = typer.Option(False, "--fix", help="Attempt to fix detected issues"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format")
):
    """Perform comprehensive system health check."""
    try:
        settings = Settings()

        console.print("[yellow]Performing system health check...[/yellow]")

        health_results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "overall_status": "healthy",
            "components": {},
            "issues": [],
            "recommendations": []
        }

        # Cache health check
        cache_service = await _get_cache_service()
        if cache_service:
            cache_metrics = cache_service.get_metrics()
            cache_status = "healthy"
            cache_issues = []

            if cache_metrics.get("overall_hit_rate", 0) < 0.8:
                cache_status = "degraded"
                cache_issues.append("Low cache hit rate")

            if cache_metrics.get("operations_per_second", 0) < 100:
                cache_status = "degraded"
                cache_issues.append("Low cache throughput")

            health_results["components"]["cache"] = {
                "status": cache_status,
                "metrics": cache_metrics,
                "issues": cache_issues
            }

            if cache_issues:
                health_results["issues"].extend([f"Cache: {issue}" for issue in cache_issues])

        # Database health check
        db_manager = DatabaseManager(settings)
        db_optimizer = DatabaseOptimizer(db_manager, settings)

        async with db_manager.get_session() as session:
            db_health = await db_optimizer.monitor_database_health(session)
            db_status = "healthy"
            db_issues = []

            if db_health.cache_hit_ratio < 0.95:
                db_status = "degraded"
                db_issues.append("Low database cache hit ratio")

            if db_health.connection_count > 800:
                db_status = "degraded"
                db_issues.append("High connection count")

            if db_health.lock_waits > 5:
                db_status = "critical"
                db_issues.append("High lock waits")

            health_results["components"]["database"] = {
                "status": db_status,
                "metrics": {
                    "connections": db_health.connection_count,
                    "cache_hit_ratio": db_health.cache_hit_ratio,
                    "lock_waits": db_health.lock_waits,
                    "deadlocks": db_health.deadlocks
                },
                "issues": db_issues
            }

            if db_issues:
                health_results["issues"].extend([f"Database: {issue}" for issue in db_issues])

        # Determine overall status
        component_statuses = [comp["status"] for comp in health_results["components"].values()]
        if "critical" in component_statuses:
            health_results["overall_status"] = "critical"
        elif "degraded" in component_statuses:
            health_results["overall_status"] = "degraded"

        # Generate recommendations
        if health_results["issues"]:
            if any("cache hit rate" in issue for issue in health_results["issues"]):
                health_results["recommendations"].append("Consider cache warming or increased cache size")

            if any("connection count" in issue for issue in health_results["issues"]):
                health_results["recommendations"].append("Review connection pool configuration")

            if any("lock waits" in issue for issue in health_results["issues"]):
                health_results["recommendations"].append("Analyze query patterns and add indexes")

        # Output results
        if json_output:
            rprint(json.dumps(health_results, indent=2))
        else:
            status_color = {
                "healthy": "green",
                "degraded": "yellow",
                "critical": "red"
            }.get(health_results["overall_status"], "white")

            console.print(Panel(
                f"[{status_color}]Overall Health: {health_results['overall_status'].upper()}[/{status_color}]",
                title="System Health Check"
            ))

            # Component status table
            health_table = Table(title="Component Health")
            health_table.add_column("Component", style="cyan")
            health_table.add_column("Status", style="magenta")
            health_table.add_column("Issues", style="yellow")

            for comp_name, comp_data in health_results["components"].items():
                issues_text = ", ".join(comp_data["issues"]) if comp_data["issues"] else "None"
                health_table.add_row(comp_name.title(), comp_data["status"], issues_text)

            console.print(health_table)

            # Recommendations
            if health_results["recommendations"]:
                console.print("\\n[yellow]Recommendations:[/yellow]")
                for rec in health_results["recommendations"]:
                    console.print(f"  • {rec}")

        # Attempt fixes if requested
        if fix_issues and health_results["issues"]:
            if typer.confirm("Attempt to fix detected issues?"):
                console.print("[yellow]Attempting automatic fixes...[/yellow]")

                fixes_applied = []

                # Cache fixes
                if cache_service and any("cache" in issue for issue in health_results["issues"]):
                    if cache_service.l1_cache:
                        cleaned = await cache_service.l1_cache.cleanup_expired()
                        fixes_applied.append(f"Cleaned {cleaned} expired cache entries")

                # Database fixes could be added here

                if fixes_applied:
                    console.print("[green]Applied fixes:[/green]")
                    for fix in fixes_applied:
                        console.print(f"  • {fix}")
                else:
                    console.print("[yellow]No automatic fixes available[/yellow]")

    except Exception as e:
        console.print(f"[red]Error during health check: {e}[/red]")
        raise typer.Exit(1)


async def _get_cache_service():
    """Get cache service instance."""
    try:
        import redis.asyncio as redis
        redis_client = redis.from_url("redis://localhost:6379")
        return EnhancedCacheService(redis_client)
    except Exception:
        return None


async def _show_detailed_metrics(cache_service, db_optimizer):
    """Show detailed performance metrics."""
    if cache_service:
        console.print("\\n[cyan]Detailed Cache Metrics:[/cyan]")
        cache_metrics = cache_service.get_metrics()

        for key, value in cache_metrics.items():
            if isinstance(value, dict):
                console.print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    console.print(f"    {sub_key}: {sub_value}")
            else:
                console.print(f"  {key}: {value}")


async def _run_cache_benchmark(cache_service, duration: int, concurrency: int):
    """Run cache performance benchmark."""
    results = {
        "total_operations": 0,
        "total_time": 0,
        "operations": [],
        "errors": 0
    }

    async def worker():
        operations = []
        errors = 0

        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                # Mixed workload
                key = f"bench_key_{hash(time.time()) % 1000}"

                op_start = time.time()
                if hash(time.time()) % 3 == 0:  # 33% writes
                    await cache_service.set(key, f"value_{time.time()}", ttl=300)
                else:  # 67% reads
                    await cache_service.get(key)

                op_time = (time.time() - op_start) * 1000
                operations.append(op_time)

            except Exception:
                errors += 1

            await asyncio.sleep(0.001)  # Small delay

        return operations, errors

    # Run concurrent workers
    start_time = time.time()
    tasks = [worker() for _ in range(concurrency)]
    worker_results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    # Aggregate results
    all_operations = []
    total_errors = 0

    for operations, errors in worker_results:
        all_operations.extend(operations)
        total_errors += errors

    # Calculate metrics
    if all_operations:
        avg_latency = sum(all_operations) / len(all_operations)
        p95_latency = sorted(all_operations)[int(len(all_operations) * 0.95)]
    else:
        avg_latency = p95_latency = 0

    return {
        "total_operations": len(all_operations),
        "total_time": total_time,
        "ops_per_second": len(all_operations) / total_time if total_time > 0 else 0,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "errors": total_errors,
        "error_rate": total_errors / (len(all_operations) + total_errors) if (len(all_operations) + total_errors) > 0 else 0
    }


async def _run_database_benchmark(db_manager, duration: int, concurrency: int):
    """Run database performance benchmark."""
    results = {
        "total_queries": 0,
        "total_time": 0,
        "response_times": [],
        "errors": 0
    }

    async def worker():
        response_times = []
        errors = 0

        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                async with db_manager.get_session() as session:
                    query_start = time.time()

                    # Simple benchmark query
                    from sqlalchemy import text
                    await session.execute(text("SELECT 1"))

                    response_time = (time.time() - query_start) * 1000
                    response_times.append(response_time)

            except Exception:
                errors += 1

            await asyncio.sleep(0.01)  # Small delay

        return response_times, errors

    # Run concurrent workers
    start_time = time.time()
    tasks = [worker() for _ in range(concurrency)]
    worker_results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time

    # Aggregate results
    all_response_times = []
    total_errors = 0

    for response_times, errors in worker_results:
        all_response_times.extend(response_times)
        total_errors += errors

    # Calculate metrics
    if all_response_times:
        avg_response_time = sum(all_response_times) / len(all_response_times)
        p95_response_time = sorted(all_response_times)[int(len(all_response_times) * 0.95)]
    else:
        avg_response_time = p95_response_time = 0

    return {
        "total_queries": len(all_response_times),
        "total_time": total_time,
        "queries_per_second": len(all_response_times) / total_time if total_time > 0 else 0,
        "avg_response_time_ms": avg_response_time,
        "p95_response_time_ms": p95_response_time,
        "errors": total_errors,
        "error_rate": total_errors / (len(all_response_times) + total_errors) if (len(all_response_times) + total_errors) > 0 else 0
    }
