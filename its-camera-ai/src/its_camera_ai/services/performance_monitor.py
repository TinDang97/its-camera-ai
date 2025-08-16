"""Performance monitoring and alerting service for cache and database optimization.

This module provides comprehensive performance monitoring capabilities:
- Real-time cache and database performance tracking
- Automated alerting for performance degradation
- Performance trend analysis and reporting
- Resource utilization monitoring
- SLA compliance tracking for 10TB/day workload
"""

import asyncio
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from ..core.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class MetricType(str, Enum):
    """Performance metric types."""
    CACHE_HIT_RATE = "cache_hit_rate"
    QUERY_RESPONSE_TIME = "query_response_time"
    DATABASE_CONNECTIONS = "database_connections"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"


@dataclass
class PerformanceAlert:
    """Performance alert container."""

    id: str
    severity: AlertSeverity
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetric:
    """Performance metric data point."""

    timestamp: datetime
    metric_type: MetricType
    value: float
    source: str
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""

    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    evaluation_window_seconds: int = 300  # 5 minutes
    min_samples: int = 10
    comparison_operator: str = "gt"  # gt, lt, eq
    enabled: bool = True


class MetricAggregator:
    """Aggregates and analyzes performance metrics."""

    def __init__(self, max_history_hours: int = 24):
        self.max_history_seconds = max_history_hours * 3600
        self.metrics: dict[MetricType, deque[PerformanceMetric]] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self._lock = asyncio.Lock()

    async def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric."""
        async with self._lock:
            self.metrics[metric.metric_type].append(metric)

            # Clean old metrics
            cutoff_time = datetime.now(UTC) - timedelta(seconds=self.max_history_seconds)
            metric_deque = self.metrics[metric.metric_type]

            # Remove old metrics from the front
            while metric_deque and metric_deque[0].timestamp < cutoff_time:
                metric_deque.popleft()

    async def get_recent_metrics(self,
                                metric_type: MetricType,
                                window_seconds: int = 300) -> list[PerformanceMetric]:
        """Get recent metrics within the specified window."""
        async with self._lock:
            cutoff_time = datetime.now(UTC) - timedelta(seconds=window_seconds)
            return [
                metric for metric in self.metrics[metric_type]
                if metric.timestamp >= cutoff_time
            ]

    async def calculate_statistics(self,
                                  metric_type: MetricType,
                                  window_seconds: int = 300) -> dict[str, float]:
        """Calculate statistics for a metric type."""
        metrics = await self.get_recent_metrics(metric_type, window_seconds)

        if not metrics:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }

        values = [m.value for m in metrics]
        sorted_values = sorted(values)

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "p95": sorted_values[int(len(sorted_values) * 0.95)] if len(sorted_values) > 0 else 0.0,
            "p99": sorted_values[int(len(sorted_values) * 0.99)] if len(sorted_values) > 0 else 0.0
        }


class PerformanceMonitor:
    """Comprehensive performance monitoring and alerting system.
    
    Features:
    - Real-time performance metric collection and analysis
    - Configurable alerting thresholds for different metric types
    - Trend analysis and anomaly detection
    - SLA compliance tracking for high-throughput workloads
    - Integration with cache and database optimization services
    """

    def __init__(self,
                 cache_service=None,
                 database_optimizer=None,
                 alert_callback: Callable[[PerformanceAlert], None] | None = None):
        self.cache_service = cache_service
        self.database_optimizer = database_optimizer
        self.alert_callback = alert_callback

        # Core components
        self.metric_aggregator = MetricAggregator()
        self.active_alerts: dict[str, PerformanceAlert] = {}
        self.alert_history: list[PerformanceAlert] = []

        # Configuration
        self.thresholds = self._setup_default_thresholds()
        self.monitoring_interval = 10  # seconds
        self.is_monitoring = False
        self._monitoring_task: asyncio.Task | None = None

        # Performance tracking
        self.sla_targets = {
            MetricType.CACHE_HIT_RATE: 0.90,      # 90% cache hit rate
            MetricType.QUERY_RESPONSE_TIME: 100,   # 100ms max query time
            MetricType.DATABASE_CONNECTIONS: 800,  # 80% of max connections
            MetricType.THROUGHPUT: 1000,           # 1000 ops/sec minimum
            MetricType.ERROR_RATE: 0.01,           # 1% max error rate
        }

        # Monitoring state
        self.last_collection_time = None
        self.collection_errors = 0

    def _setup_default_thresholds(self) -> dict[MetricType, PerformanceThreshold]:
        """Setup default performance thresholds."""
        return {
            MetricType.CACHE_HIT_RATE: PerformanceThreshold(
                metric_type=MetricType.CACHE_HIT_RATE,
                warning_threshold=0.85,   # 85% warning
                critical_threshold=0.80,  # 80% critical
                comparison_operator="lt"  # less than
            ),
            MetricType.QUERY_RESPONSE_TIME: PerformanceThreshold(
                metric_type=MetricType.QUERY_RESPONSE_TIME,
                warning_threshold=50.0,   # 50ms warning
                critical_threshold=100.0, # 100ms critical
                comparison_operator="gt"  # greater than
            ),
            MetricType.DATABASE_CONNECTIONS: PerformanceThreshold(
                metric_type=MetricType.DATABASE_CONNECTIONS,
                warning_threshold=700,    # 700 connections warning
                critical_threshold=900,   # 900 connections critical
                comparison_operator="gt"
            ),
            MetricType.MEMORY_USAGE: PerformanceThreshold(
                metric_type=MetricType.MEMORY_USAGE,
                warning_threshold=80.0,   # 80% memory usage warning
                critical_threshold=90.0,  # 90% memory usage critical
                comparison_operator="gt"
            ),
            MetricType.THROUGHPUT: PerformanceThreshold(
                metric_type=MetricType.THROUGHPUT,
                warning_threshold=500,    # 500 ops/sec warning
                critical_threshold=200,   # 200 ops/sec critical
                comparison_operator="lt"
            ),
            MetricType.ERROR_RATE: PerformanceThreshold(
                metric_type=MetricType.ERROR_RATE,
                warning_threshold=0.02,   # 2% error rate warning
                critical_threshold=0.05,  # 5% error rate critical
                comparison_operator="gt"
            )
        }

    async def start_monitoring(self):
        """Start performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring already running")
            return

        self.is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("Performance monitoring started")

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                collection_start = time.time()

                # Collect performance metrics
                await self._collect_metrics()

                # Evaluate thresholds and generate alerts
                await self._evaluate_thresholds()

                # Update monitoring state
                self.last_collection_time = datetime.now(UTC)
                collection_time = (time.time() - collection_start) * 1000

                # Monitor the monitoring system itself
                await self._add_metric(
                    MetricType.MEMORY_USAGE,  # Reuse for monitoring overhead
                    collection_time,
                    "performance_monitor",
                    {"type": "collection_time_ms"}
                )

                # Reset error count on successful collection
                self.collection_errors = 0

                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.collection_errors += 1
                logger.error(f"Performance monitoring error: {e}")

                # Exponential backoff on errors
                error_delay = min(60, self.monitoring_interval * (2 ** min(self.collection_errors, 5)))
                await asyncio.sleep(error_delay)

    async def _collect_metrics(self):
        """Collect performance metrics from various sources."""
        current_time = datetime.now(UTC)

        # Collect cache metrics
        if self.cache_service:
            await self._collect_cache_metrics(current_time)

        # Collect database metrics
        if self.database_optimizer:
            await self._collect_database_metrics(current_time)

        # Collect system metrics
        await self._collect_system_metrics(current_time)

    async def _collect_cache_metrics(self, timestamp: datetime):
        """Collect cache performance metrics."""
        try:
            cache_metrics = self.cache_service.get_metrics()

            # Cache hit rate
            await self._add_metric(
                MetricType.CACHE_HIT_RATE,
                cache_metrics.get("overall_hit_rate", 0.0),
                "cache_service",
                {"level": "overall"},
                timestamp
            )

            # L1 cache hit rate
            await self._add_metric(
                MetricType.CACHE_HIT_RATE,
                cache_metrics.get("l1_hit_rate", 0.0),
                "cache_service",
                {"level": "l1"},
                timestamp
            )

            # L2 cache hit rate
            await self._add_metric(
                MetricType.CACHE_HIT_RATE,
                cache_metrics.get("l2_hit_rate", 0.0),
                "cache_service",
                {"level": "l2"},
                timestamp
            )

            # Operations per second (throughput)
            await self._add_metric(
                MetricType.THROUGHPUT,
                cache_metrics.get("operations_per_second", 0.0),
                "cache_service",
                {"type": "cache_operations"},
                timestamp
            )

            # Memory usage (L1 cache)
            l1_cache_info = cache_metrics.get("l1_cache", {})
            if l1_cache_info:
                memory_utilization = l1_cache_info.get("memory_utilization", 0.0) * 100
                await self._add_metric(
                    MetricType.MEMORY_USAGE,
                    memory_utilization,
                    "cache_service",
                    {"component": "l1_cache"},
                    timestamp
                )

        except Exception as e:
            logger.error(f"Failed to collect cache metrics: {e}")

    async def _collect_database_metrics(self, timestamp: datetime):
        """Collect database performance metrics."""
        try:
            # Get performance summary from database optimizer
            db_summary = self.database_optimizer.get_performance_summary()

            # Query response time
            avg_response_time = db_summary.get("recent_avg_response_time_ms", 0.0)
            await self._add_metric(
                MetricType.QUERY_RESPONSE_TIME,
                avg_response_time,
                "database_optimizer",
                {"type": "average"},
                timestamp
            )

            # Error rate (slow queries percentage)
            slow_query_pct = db_summary.get("slow_query_percentage", 0.0) / 100
            await self._add_metric(
                MetricType.ERROR_RATE,
                slow_query_pct,
                "database_optimizer",
                {"type": "slow_queries"},
                timestamp
            )

            # Health alerts count (as a performance indicator)
            health_alerts = db_summary.get("health_alerts", 0)
            await self._add_metric(
                MetricType.ERROR_RATE,
                health_alerts / 100.0,  # Normalize to 0-1 range
                "database_optimizer",
                {"type": "health_alerts"},
                timestamp
            )

        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")

    async def _collect_system_metrics(self, timestamp: datetime):
        """Collect system-level performance metrics."""
        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            await self._add_metric(
                MetricType.MEMORY_USAGE,  # Reuse for CPU
                cpu_percent,
                "system",
                {"component": "cpu"},
                timestamp
            )

            # Memory usage
            memory = psutil.virtual_memory()
            await self._add_metric(
                MetricType.MEMORY_USAGE,
                memory.percent,
                "system",
                {"component": "memory"},
                timestamp
            )

            # Process count (as throughput indicator)
            process_count = len(psutil.pids())
            await self._add_metric(
                MetricType.THROUGHPUT,
                process_count,
                "system",
                {"type": "process_count"},
                timestamp
            )

        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

    async def _add_metric(self,
                         metric_type: MetricType,
                         value: float,
                         source: str,
                         tags: dict[str, str] = None,
                         timestamp: datetime = None):
        """Add a performance metric."""
        metric = PerformanceMetric(
            timestamp=timestamp or datetime.now(UTC),
            metric_type=metric_type,
            value=value,
            source=source,
            tags=tags or {},
        )

        await self.metric_aggregator.add_metric(metric)

    async def _evaluate_thresholds(self):
        """Evaluate performance thresholds and generate alerts."""
        for metric_type, threshold in self.thresholds.items():
            if not threshold.enabled:
                continue

            try:
                # Get recent statistics
                stats = await self.metric_aggregator.calculate_statistics(
                    metric_type, threshold.evaluation_window_seconds
                )

                if stats["count"] < threshold.min_samples:
                    continue  # Not enough data

                # Use mean value for threshold evaluation
                current_value = stats["mean"]

                # Check thresholds
                alert_severity = None
                threshold_value = None

                if self._exceeds_threshold(current_value, threshold.critical_threshold, threshold.comparison_operator):
                    alert_severity = AlertSeverity.CRITICAL
                    threshold_value = threshold.critical_threshold
                elif self._exceeds_threshold(current_value, threshold.warning_threshold, threshold.comparison_operator):
                    alert_severity = AlertSeverity.WARNING
                    threshold_value = threshold.warning_threshold

                # Generate or resolve alerts
                alert_id = f"{metric_type.value}_{alert_severity.value if alert_severity else 'resolved'}"

                if alert_severity:
                    await self._create_alert(
                        alert_id=alert_id,
                        severity=alert_severity,
                        metric_type=metric_type,
                        value=current_value,
                        threshold=threshold_value,
                        stats=stats
                    )
                else:
                    await self._resolve_alert(alert_id)

            except Exception as e:
                logger.error(f"Failed to evaluate threshold for {metric_type}: {e}")

    def _exceeds_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Check if value exceeds threshold based on operator."""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "eq":
            return abs(value - threshold) < 0.001  # Small epsilon for float comparison
        else:
            return False

    async def _create_alert(self,
                           alert_id: str,
                           severity: AlertSeverity,
                           metric_type: MetricType,
                           value: float,
                           threshold: float,
                           stats: dict[str, float]):
        """Create or update a performance alert."""
        # Check if alert already exists
        if alert_id in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[alert_id]
            alert.value = value
            alert.timestamp = datetime.now(UTC)
            return

        # Create new alert
        alert = PerformanceAlert(
            id=alert_id,
            severity=severity,
            metric_type=metric_type,
            message=self._generate_alert_message(metric_type, severity, value, threshold, stats),
            value=value,
            threshold=threshold,
            timestamp=datetime.now(UTC),
            metadata={
                "stats": stats,
                "evaluation_window": self.thresholds[metric_type].evaluation_window_seconds
            }
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Trigger alert callback
        if self.alert_callback:
            try:
                await self.alert_callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        logger.warning(f"Performance alert created: {alert.message}")

    async def _resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now(UTC)

            del self.active_alerts[alert_id]

            logger.info(f"Performance alert resolved: {alert.message}")

    def _generate_alert_message(self,
                               metric_type: MetricType,
                               severity: AlertSeverity,
                               value: float,
                               threshold: float,
                               stats: dict[str, float]) -> str:
        """Generate human-readable alert message."""
        metric_name = metric_type.value.replace("_", " ").title()

        if metric_type == MetricType.CACHE_HIT_RATE:
            return f"{metric_name} is {severity.value}: {value:.1%} (threshold: {threshold:.1%})"
        elif metric_type == MetricType.QUERY_RESPONSE_TIME:
            return f"{metric_name} is {severity.value}: {value:.1f}ms (threshold: {threshold:.1f}ms)"
        elif metric_type == MetricType.DATABASE_CONNECTIONS:
            return f"{metric_name} is {severity.value}: {value:.0f} (threshold: {threshold:.0f})"
        elif metric_type == MetricType.MEMORY_USAGE:
            return f"{metric_name} is {severity.value}: {value:.1f}% (threshold: {threshold:.1f}%)"
        elif metric_type == MetricType.THROUGHPUT:
            return f"{metric_name} is {severity.value}: {value:.0f} ops/sec (threshold: {threshold:.0f})"
        elif metric_type == MetricType.ERROR_RATE:
            return f"{metric_name} is {severity.value}: {value:.1%} (threshold: {threshold:.1%})"
        else:
            return f"{metric_name} {severity.value}: {value:.2f} (threshold: {threshold:.2f})"

    async def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            "monitoring_status": "active" if self.is_monitoring else "inactive",
            "last_collection": self.last_collection_time.isoformat() if self.last_collection_time else None,
            "collection_errors": self.collection_errors,
            "active_alerts": len(self.active_alerts),
            "total_alerts_today": len([
                alert for alert in self.alert_history
                if alert.timestamp >= datetime.now(UTC) - timedelta(days=1)
            ]),
            "metrics": {},
            "sla_compliance": {},
            "alerts": []
        }

        # Collect current metrics statistics
        for metric_type in MetricType:
            try:
                stats = await self.metric_aggregator.calculate_statistics(metric_type, 300)
                summary["metrics"][metric_type.value] = {
                    "current_value": stats["mean"],
                    "count": stats["count"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "p95": stats["p95"],
                    "status": self._get_metric_status(metric_type, stats["mean"])
                }

                # SLA compliance
                if metric_type in self.sla_targets:
                    target = self.sla_targets[metric_type]
                    current = stats["mean"]

                    if metric_type == MetricType.CACHE_HIT_RATE:
                        compliance = (current / target) * 100 if target > 0 else 0
                    elif metric_type == MetricType.QUERY_RESPONSE_TIME:
                        compliance = (target / current) * 100 if current > 0 else 100
                    elif metric_type == MetricType.THROUGHPUT:
                        compliance = (current / target) * 100 if target > 0 else 0
                    else:
                        compliance = 100 if current <= target else (target / current) * 100

                    summary["sla_compliance"][metric_type.value] = {
                        "target": target,
                        "current": current,
                        "compliance_pct": min(100, max(0, compliance))
                    }

            except Exception as e:
                logger.error(f"Failed to get summary for {metric_type}: {e}")

        # Active alerts
        summary["alerts"] = [
            {
                "id": alert.id,
                "severity": alert.severity.value,
                "metric": alert.metric_type.value,
                "message": alert.message,
                "value": alert.value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp.isoformat()
            }
            for alert in self.active_alerts.values()
        ]

        return summary

    def _get_metric_status(self, metric_type: MetricType, value: float) -> str:
        """Get status for a metric value."""
        if metric_type not in self.thresholds:
            return "unknown"

        threshold = self.thresholds[metric_type]

        if self._exceeds_threshold(value, threshold.critical_threshold, threshold.comparison_operator):
            return "critical"
        elif self._exceeds_threshold(value, threshold.warning_threshold, threshold.comparison_operator):
            return "warning"
        else:
            return "healthy"

    async def get_metric_history(self,
                                metric_type: MetricType,
                                hours: int = 24) -> list[dict[str, Any]]:
        """Get metric history for the specified time period."""
        window_seconds = hours * 3600
        metrics = await self.metric_aggregator.get_recent_metrics(metric_type, window_seconds)

        return [
            {
                "timestamp": metric.timestamp.isoformat(),
                "value": metric.value,
                "source": metric.source,
                "tags": metric.tags
            }
            for metric in metrics
        ]

    def update_threshold(self, metric_type: MetricType, threshold: PerformanceThreshold):
        """Update performance threshold configuration."""
        self.thresholds[metric_type] = threshold
        logger.info(f"Updated threshold for {metric_type.value}")

    async def test_alert_system(self) -> bool:
        """Test the alert system by generating a test alert."""
        try:
            test_alert = PerformanceAlert(
                id="test_alert",
                severity=AlertSeverity.INFO,
                metric_type=MetricType.CACHE_HIT_RATE,
                message="Test alert - system is functioning correctly",
                value=0.95,
                threshold=0.90,
                timestamp=datetime.now(UTC),
                metadata={"test": True}
            )

            if self.alert_callback:
                await self.alert_callback(test_alert)

            logger.info("Alert system test completed successfully")
            return True

        except Exception as e:
            logger.error(f"Alert system test failed: {e}")
            return False
