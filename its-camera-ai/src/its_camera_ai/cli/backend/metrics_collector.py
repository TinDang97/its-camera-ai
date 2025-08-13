"""Metrics collection and monitoring for CLI operations.

Provides comprehensive metrics collection from all backend services
with aggregation, analysis, and reporting capabilities.
"""

import asyncio
import contextlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any

import aiohttp
from prometheus_client.parser import text_string_to_metric_families

from ...core.config import Settings, get_settings
from ...core.logging import get_logger
from .api_client import APIClient

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: int | float
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)
    help_text: str | None = None
    metric_type: str = "gauge"


@dataclass
class MetricSeries:
    """Time series of metric data points."""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: dict[str, str] = field(default_factory=dict)

    def add_point(self, value: int | float, timestamp: float = None) -> None:
        """Add a data point to the series."""
        self.points.append(MetricPoint(
            name=self.name,
            value=value,
            timestamp=timestamp or time.time(),
            labels=self.labels
        ))

    def get_latest(self) -> MetricPoint | None:
        """Get the latest data point."""
        return self.points[-1] if self.points else None

    def get_values(self, duration: int = 3600) -> list[float]:
        """Get values within duration (seconds)."""
        cutoff_time = time.time() - duration
        return [p.value for p in self.points if p.timestamp >= cutoff_time]

    def calculate_stats(self, duration: int = 3600) -> dict[str, float]:
        """Calculate statistics for the series."""
        values = self.get_values(duration)

        if not values:
            return {}

        return {
            "count": len(values),
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1] if values else 0
        }


class MetricsCollector:
    """Comprehensive metrics collection system.

    Features:
    - Prometheus metrics scraping
    - System metrics collection
    - Custom metric tracking
    - Metric aggregation and analysis
    - Historical data storage
    - Alert threshold monitoring
    """

    def __init__(self, settings: Settings = None):
        """Initialize metrics collector.

        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()
        self.api_client = APIClient(settings=self.settings)

        # Metric storage
        self.metrics: dict[str, MetricSeries] = {}
        self.metric_metadata: dict[str, dict[str, Any]] = {}

        # Collection settings
        self.collection_interval = 30  # seconds
        self.retention_duration = 86400  # 24 hours

        # Collection task
        self._collection_task: asyncio.Task | None = None
        self._collecting = False

        # Prometheus endpoints
        self.prometheus_endpoints = [
            f"http://localhost:{self.settings.monitoring.prometheus_port}/metrics",
            f"http://{self.settings.api_host}:{self.settings.api_port}/metrics"
        ]

        # HTTP session for metrics scraping
        self._session: aiohttp.ClientSession | None = None

        logger.info("Metrics collector initialized")

    async def initialize(self) -> None:
        """Initialize metrics collector."""
        # Create HTTP session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            connector=aiohttp.TCPConnector(limit=10)
        )

        await self.api_client.connect()
        logger.info("Metrics collector initialized")

    async def close(self) -> None:
        """Close metrics collector."""
        await self.stop_collection()

        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        await self.api_client.close()
        logger.info("Metrics collector closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def collect_prometheus_metrics(self, endpoint: str) -> dict[str, list[MetricPoint]]:
        """Collect metrics from Prometheus endpoint.

        Args:
            endpoint: Prometheus metrics endpoint URL

        Returns:
            Dictionary of metric families
        """
        if not self._session:
            await self.initialize()

        metrics = defaultdict(list)

        try:
            async with self._session.get(endpoint) as response:
                if response.status == 200:
                    content = await response.text()

                    # Parse Prometheus format
                    for family in text_string_to_metric_families(content):
                        for sample in family.samples:
                            metric_point = MetricPoint(
                                name=sample.name,
                                value=sample.value,
                                labels=dict(sample.labels),
                                help_text=family.documentation,
                                metric_type=family.type
                            )
                            metrics[family.name].append(metric_point)

                    logger.debug(f"Collected {len(metrics)} metric families from {endpoint}")
                else:
                    logger.warning(f"Failed to collect from {endpoint}: HTTP {response.status}")

        except Exception as e:
            logger.error(f"Error collecting from {endpoint}: {e}")

        return metrics

    async def collect_api_metrics(self) -> dict[str, Any]:
        """Collect metrics from API endpoints.

        Returns:
            API metrics dictionary
        """
        try:
            # Get system metrics
            system_metrics = await self.api_client.get_metrics()

            # Transform to metric points
            metrics = {}

            if "system" in system_metrics:
                system_data = system_metrics["system"]

                # CPU metrics
                if "cpu" in system_data:
                    metrics["system_cpu_usage"] = MetricPoint(
                        name="system_cpu_usage",
                        value=system_data["cpu"].get("usage_percent", 0),
                        labels={"type": "system"}
                    )

                # Memory metrics
                if "memory" in system_data:
                    metrics["system_memory_usage"] = MetricPoint(
                        name="system_memory_usage",
                        value=system_data["memory"].get("usage_percent", 0),
                        labels={"type": "system"}
                    )

                # Disk metrics
                if "disk" in system_data:
                    metrics["system_disk_usage"] = MetricPoint(
                        name="system_disk_usage",
                        value=system_data["disk"].get("usage_percent", 0),
                        labels={"type": "system"}
                    )

            # Service metrics
            if "services" in system_metrics:
                for service_name, service_data in system_metrics["services"].items():
                    if "response_time_ms" in service_data:
                        metrics["service_response_time"] = MetricPoint(
                            name="service_response_time",
                            value=service_data["response_time_ms"],
                            labels={"service": service_name}
                        )

                    if "request_count" in service_data:
                        metrics["service_request_count"] = MetricPoint(
                            name="service_request_count",
                            value=service_data["request_count"],
                            labels={"service": service_name}
                        )

            logger.debug(f"Collected {len(metrics)} API metrics")
            return metrics

        except Exception as e:
            logger.error(f"Error collecting API metrics: {e}")
            return {}

    async def collect_system_metrics(self) -> dict[str, MetricPoint]:
        """Collect local system metrics using psutil.

        Returns:
            System metrics dictionary
        """
        try:
            import psutil

            metrics = {}

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics["system_cpu_percent"] = MetricPoint(
                name="system_cpu_percent",
                value=cpu_percent,
                labels={"host": "localhost"}
            )

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics["system_memory_percent"] = MetricPoint(
                name="system_memory_percent",
                value=memory.percent,
                labels={"host": "localhost"}
            )
            metrics["system_memory_available_bytes"] = MetricPoint(
                name="system_memory_available_bytes",
                value=memory.available,
                labels={"host": "localhost"}
            )

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics["system_disk_percent"] = MetricPoint(
                name="system_disk_percent",
                value=disk_percent,
                labels={"host": "localhost", "mountpoint": "/"}
            )

            # Network metrics
            network = psutil.net_io_counters()
            metrics["system_network_bytes_sent"] = MetricPoint(
                name="system_network_bytes_sent",
                value=network.bytes_sent,
                labels={"host": "localhost"}
            )
            metrics["system_network_bytes_recv"] = MetricPoint(
                name="system_network_bytes_recv",
                value=network.bytes_recv,
                labels={"host": "localhost"}
            )

            # Process metrics
            process_count = len(psutil.pids())
            metrics["system_process_count"] = MetricPoint(
                name="system_process_count",
                value=process_count,
                labels={"host": "localhost"}
            )

            logger.debug(f"Collected {len(metrics)} system metrics")
            return metrics

        except ImportError:
            logger.warning("psutil not available, skipping system metrics")
            return {}
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}

    def add_metric_point(self, metric_point: MetricPoint) -> None:
        """Add a metric point to storage.

        Args:
            metric_point: Metric point to add
        """
        # Create series key with labels
        label_str = ",".join(f"{k}={v}" for k, v in sorted(metric_point.labels.items()))
        series_key = f"{metric_point.name}#{label_str}" if label_str else metric_point.name

        # Get or create metric series
        if series_key not in self.metrics:
            self.metrics[series_key] = MetricSeries(
                name=metric_point.name,
                labels=metric_point.labels
            )

            # Store metadata
            self.metric_metadata[metric_point.name] = {
                "help": metric_point.help_text,
                "type": metric_point.metric_type,
                "labels": list(metric_point.labels.keys())
            }

        # Add point to series
        self.metrics[series_key].add_point(
            metric_point.value,
            metric_point.timestamp
        )

    async def collect_all_metrics(self) -> None:
        """Collect metrics from all sources."""
        try:
            # Collect from Prometheus endpoints
            for endpoint in self.prometheus_endpoints:
                prometheus_metrics = await self.collect_prometheus_metrics(endpoint)

                for _family_name, metric_points in prometheus_metrics.items():
                    for metric_point in metric_points:
                        self.add_metric_point(metric_point)

            # Collect from API
            api_metrics = await self.collect_api_metrics()
            for metric_point in api_metrics.values():
                self.add_metric_point(metric_point)

            # Collect system metrics
            system_metrics = await self.collect_system_metrics()
            for metric_point in system_metrics.values():
                self.add_metric_point(metric_point)

            # Clean up old data
            self._cleanup_old_metrics()

            logger.debug(f"Collected metrics, total series: {len(self.metrics)}")

        except Exception as e:
            logger.error(f"Error during metric collection: {e}")

    def _cleanup_old_metrics(self) -> None:
        """Remove old metric data points."""
        cutoff_time = time.time() - self.retention_duration

        for series in self.metrics.values():
            # Remove old points
            while series.points and series.points[0].timestamp < cutoff_time:
                series.points.popleft()

    async def start_collection(self, interval: int = None) -> None:
        """Start continuous metric collection.

        Args:
            interval: Collection interval in seconds
        """
        if self._collecting:
            logger.warning("Metric collection is already running")
            return

        if interval:
            self.collection_interval = interval

        self._collecting = True
        self._collection_task = asyncio.create_task(self._collection_loop())

        logger.info(f"Started metric collection (interval: {self.collection_interval}s)")

    async def stop_collection(self) -> None:
        """Stop metric collection."""
        self._collecting = False

        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._collection_task
            self._collection_task = None

        logger.info("Stopped metric collection")

    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self._collecting:
            try:
                await self.collect_all_metrics()
                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(5)  # Short delay on error

    def get_metric_series(self, metric_name: str, labels: dict[str, str] = None) -> MetricSeries | None:
        """Get metric series by name and labels.

        Args:
            metric_name: Name of the metric
            labels: Labels to match

        Returns:
            Matching metric series or None
        """
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            series_key = f"{metric_name}#{label_str}"
        else:
            series_key = metric_name

        return self.metrics.get(series_key)

    def get_metric_value(
        self,
        metric_name: str,
        labels: dict[str, str] = None
    ) -> float | None:
        """Get latest value for a metric.

        Args:
            metric_name: Name of the metric
            labels: Labels to match

        Returns:
            Latest metric value or None
        """
        series = self.get_metric_series(metric_name, labels)
        if series:
            latest_point = series.get_latest()
            return latest_point.value if latest_point else None
        return None

    def get_metric_stats(
        self,
        metric_name: str,
        labels: dict[str, str] = None,
        duration: int = 3600
    ) -> dict[str, float]:
        """Get statistics for a metric.

        Args:
            metric_name: Name of the metric
            labels: Labels to match
            duration: Time window in seconds

        Returns:
            Statistics dictionary
        """
        series = self.get_metric_series(metric_name, labels)
        if series:
            return series.calculate_stats(duration)
        return {}

    def list_metrics(self) -> list[dict[str, Any]]:
        """List all available metrics.

        Returns:
            List of metric information
        """
        metrics_info = []

        for series_key, series in self.metrics.items():
            latest_point = series.get_latest()

            metrics_info.append({
                "name": series.name,
                "series_key": series_key,
                "labels": series.labels,
                "latest_value": latest_point.value if latest_point else None,
                "latest_timestamp": latest_point.timestamp if latest_point else None,
                "point_count": len(series.points),
                "metadata": self.metric_metadata.get(series.name, {})
            })

        return sorted(metrics_info, key=lambda x: x["name"])

    def get_dashboard_metrics(self) -> dict[str, Any]:
        """Get key metrics for dashboard display.

        Returns:
            Dashboard metrics dictionary
        """
        dashboard = {
            "system": {},
            "services": {},
            "summary": {}
        }

        # System metrics
        cpu_value = self.get_metric_value("system_cpu_percent")
        if cpu_value is not None:
            dashboard["system"]["cpu_percent"] = round(cpu_value, 1)

        memory_value = self.get_metric_value("system_memory_percent")
        if memory_value is not None:
            dashboard["system"]["memory_percent"] = round(memory_value, 1)

        disk_value = self.get_metric_value("system_disk_percent")
        if disk_value is not None:
            dashboard["system"]["disk_percent"] = round(disk_value, 1)

        # Service metrics
        for series_key, series in self.metrics.items():
            if "service_response_time" in series_key and "service" in series.labels:
                service_name = series.labels["service"]
                latest_point = series.get_latest()
                if latest_point:
                    if service_name not in dashboard["services"]:
                        dashboard["services"][service_name] = {}
                    dashboard["services"][service_name]["response_time_ms"] = round(
                        latest_point.value, 2
                    )

        # Summary
        dashboard["summary"] = {
            "total_metrics": len(self.metrics),
            "collection_active": self._collecting,
            "last_collection": time.time(),
            "retention_hours": self.retention_duration / 3600
        }

        return dashboard

    def export_metrics(
        self,
        format_type: str = "json",
        duration: int = 3600
    ) -> str | dict[str, Any]:
        """Export metrics in specified format.

        Args:
            format_type: Export format (json, csv, prometheus)
            duration: Time window in seconds

        Returns:
            Exported metrics data
        """
        if format_type == "json":
            export_data = {
                "timestamp": time.time(),
                "duration": duration,
                "metrics": {}
            }

            for series_key, series in self.metrics.items():
                export_data["metrics"][series_key] = {
                    "name": series.name,
                    "labels": series.labels,
                    "stats": series.calculate_stats(duration),
                    "values": [
                        {"timestamp": p.timestamp, "value": p.value}
                        for p in series.points
                        if p.timestamp >= time.time() - duration
                    ]
                }

            return export_data

        elif format_type == "prometheus":
            # Export in Prometheus format
            lines = []

            for series_key, series in self.metrics.items():
                latest_point = series.get_latest()
                if latest_point:
                    labels_str = ",".join(
                        f'{k}="{v}"' for k, v in series.labels.items()
                    )
                    if labels_str:
                        line = f"{series.name}{{{labels_str}}} {latest_point.value}"
                    else:
                        line = f"{series.name} {latest_point.value}"
                    lines.append(line)

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported export format: {format_type}")

    def get_collector_status(self) -> dict[str, Any]:
        """Get collector status information.

        Returns:
            Collector status dictionary
        """
        return {
            "collecting": self._collecting,
            "collection_interval": self.collection_interval,
            "retention_duration": self.retention_duration,
            "total_series": len(self.metrics),
            "total_points": sum(len(series.points) for series in self.metrics.values()),
            "prometheus_endpoints": self.prometheus_endpoints,
            "last_collection": time.time(),
            "memory_usage_mb": self._estimate_memory_usage() / 1024 / 1024
        }

    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of stored metrics.

        Returns:
            Estimated memory usage in bytes
        """
        # Rough estimate: each metric point ~100 bytes
        total_points = sum(len(series.points) for series in self.metrics.values())
        return total_points * 100
