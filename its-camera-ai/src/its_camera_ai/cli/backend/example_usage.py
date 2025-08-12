"""Example usage of the CLI backend integration.

Demonstrates how to use the various backend integration components
for common operations.
"""

import asyncio
from typing import Any

from ...core.config import get_settings
from ...core.logging import get_logger
from .api_client import APIClient
from .auth_manager import CLIAuthManager
from .health_checker import HealthChecker
from .metrics_collector import MetricsCollector
from .orchestrator import ServiceOrchestrator
from .queue_manager import CLIQueueManager

logger = get_logger(__name__)


class BackendIntegrationExample:
    """Example demonstrating backend integration usage."""

    def __init__(self):
        self.settings = get_settings()

    async def example_api_operations(self) -> dict[str, Any]:
        """Example API client operations."""
        results = {}

        async with APIClient(settings=self.settings) as api_client:
            # Get system health
            health = await api_client.get_health()
            results["health"] = health

            # Get system status
            status = await api_client.get_system_status()
            results["status"] = status

            # List cameras
            cameras = await api_client.list_cameras()
            results["cameras"] = cameras

            # Get metrics
            metrics = await api_client.get_metrics()
            results["metrics"] = metrics

        return results

    async def example_authentication(self) -> dict[str, Any]:
        """Example authentication operations."""
        results = {}

        async with CLIAuthManager(self.settings) as auth_manager:
            # Attempt login (would normally prompt for credentials)
            try:
                login_result = await auth_manager.login("admin", "admin123")
                results["login"] = "success"
                results["user"] = login_result["user"]["username"]

                # Check permissions
                has_read_perm = await auth_manager.check_permission("cameras:read")
                has_write_perm = await auth_manager.check_permission("cameras:write")

                results["permissions"] = {
                    "cameras:read": has_read_perm,
                    "cameras:write": has_write_perm
                }

                # Get current user
                current_user = await auth_manager.get_current_user()
                results["current_user"] = current_user["username"] if current_user else None

            except Exception as e:
                results["login"] = f"failed: {e}"

        return results

    async def example_health_checking(self) -> dict[str, Any]:
        """Example health checking operations."""
        results = {}

        async with HealthChecker(self.settings) as health_checker:
            # Check all components
            system_health = await health_checker.check_all_components()

            results["overall_status"] = system_health.overall_status.value
            results["component_count"] = len(system_health.components)
            results["healthy_count"] = system_health.summary["healthy_count"]
            results["health_score"] = system_health.summary["health_score"]

            # Benchmark a specific component
            benchmark_results = await health_checker.benchmark_component(
                "api_service",
                iterations=3
            )

            if "error" not in benchmark_results:
                results["benchmark"] = {
                    "success_rate": benchmark_results["success_rate_percent"],
                    "avg_response_time": benchmark_results["avg_response_time_ms"]
                }

        return results

    async def example_queue_operations(self) -> dict[str, Any]:
        """Example queue management operations."""
        results = {}

        async with CLIQueueManager(self.settings) as queue_manager:
            # Submit a task
            task_id = await queue_manager.submit_task(
                "example_task",
                {"param1": "value1", "param2": 42},
                priority=5
            )
            results["task_submitted"] = task_id

            # Get queue status
            queue_status = await queue_manager.get_all_queue_status()
            results["queues"] = {
                name: {
                    "pending": status["pending_count"],
                    "processing": status["processing_count"]
                }
                for name, status in queue_status.items()
            }

            # Send a notification
            notification_sent = await queue_manager.send_notification(
                "Example notification from CLI backend integration",
                level="info",
                metadata={"source": "example"}
            )
            results["notification_sent"] = notification_sent

        return results

    async def example_metrics_collection(self) -> dict[str, Any]:
        """Example metrics collection operations."""
        results = {}

        async with MetricsCollector(self.settings) as metrics_collector:
            # Start collection for a short time
            await metrics_collector.start_collection(interval=10)
            await asyncio.sleep(15)  # Collect for 15 seconds
            await metrics_collector.stop_collection()

            # Get dashboard metrics
            dashboard = metrics_collector.get_dashboard_metrics()
            results["dashboard"] = dashboard

            # List available metrics
            available_metrics = metrics_collector.list_metrics()
            results["metric_count"] = len(available_metrics)

            # Get specific metric value
            cpu_usage = metrics_collector.get_metric_value("system_cpu_percent")
            if cpu_usage is not None:
                results["cpu_usage"] = cpu_usage

            # Export metrics
            exported_data = metrics_collector.export_metrics("json", duration=300)
            results["exported_metrics"] = len(exported_data.get("metrics", {}))

        return results

    async def example_full_orchestration(self) -> dict[str, Any]:
        """Example full system orchestration."""
        results = {}

        async with ServiceOrchestrator(self.settings) as orchestrator:
            # Get system overview
            overview = orchestrator.get_system_overview()
            results["system_initialized"] = overview["orchestrator"]["initialized"]

            # Run comprehensive health check
            health_result = await orchestrator.full_system_health_check()
            results["health_check"] = {
                "status": health_result["overall_status"],
                "components": len(health_result["components"]),
                "score": health_result["summary"]["health_score"]
            }

            # Start monitoring services
            monitoring_result = await orchestrator.start_monitoring_services()
            results["monitoring_started"] = monitoring_result["status"] == "success"

            # Run maintenance
            maintenance_result = await orchestrator.perform_system_maintenance([
                "clear_caches",
                "health_check"
            ])
            results["maintenance"] = {
                "status": maintenance_result["status"],
                "operations": maintenance_result["operations_completed"]
            }

            # Stop monitoring
            stop_result = await orchestrator.stop_monitoring_services()
            results["monitoring_stopped"] = stop_result["status"] == "success"

        return results

    async def run_all_examples(self) -> dict[str, Any]:
        """Run all example operations."""
        logger.info("Running backend integration examples...")

        all_results = {}

        try:
            all_results["api_operations"] = await self.example_api_operations()
        except Exception as e:
            all_results["api_operations"] = f"failed: {e}"

        try:
            all_results["authentication"] = await self.example_authentication()
        except Exception as e:
            all_results["authentication"] = f"failed: {e}"

        try:
            all_results["health_checking"] = await self.example_health_checking()
        except Exception as e:
            all_results["health_checking"] = f"failed: {e}"

        try:
            all_results["queue_operations"] = await self.example_queue_operations()
        except Exception as e:
            all_results["queue_operations"] = f"failed: {e}"

        try:
            all_results["metrics_collection"] = await self.example_metrics_collection()
        except Exception as e:
            all_results["metrics_collection"] = f"failed: {e}"

        try:
            all_results["orchestration"] = await self.example_full_orchestration()
        except Exception as e:
            all_results["orchestration"] = f"failed: {e}"

        logger.info("Backend integration examples completed")
        return all_results


async def main():
    """Main example function."""
    example = BackendIntegrationExample()
    results = await example.run_all_examples()

    print("\n=== Backend Integration Example Results ===")
    for category, result in results.items():
        print(f"\n{category.upper()}:")
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {result}")


if __name__ == "__main__":
    asyncio.run(main())
