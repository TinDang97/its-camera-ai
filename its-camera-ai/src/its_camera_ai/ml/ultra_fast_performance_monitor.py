"""
Ultra-fast performance monitoring system with P99 latency tracking and automated alerts.

This module provides:
- Real-time P50/P95/P99 latency tracking
- Automated performance degradation detection
- SLA violation alerts and remediation
- Resource utilization monitoring
- Performance trend analysis
- Circuit breaker integration
"""

import asyncio
import logging
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class PerformanceMetric(Enum):
    """Performance metrics to track."""
    LATENCY_P50 = "latency_p50_ms"
    LATENCY_P95 = "latency_p95_ms"
    LATENCY_P99 = "latency_p99_ms"
    THROUGHPUT = "throughput_fps"
    ERROR_RATE = "error_rate_pct"
    GPU_UTILIZATION = "gpu_utilization_pct"
    MEMORY_USAGE = "memory_usage_mb"
    QUEUE_DEPTH = "queue_depth"


@dataclass
class PerformanceThresholds:
    """Performance thresholds for alerting."""

    # Latency thresholds (milliseconds)
    latency_p50_warning_ms: float = 30.0
    latency_p50_critical_ms: float = 40.0
    latency_p95_warning_ms: float = 45.0
    latency_p95_critical_ms: float = 50.0
    latency_p99_warning_ms: float = 55.0
    latency_p99_critical_ms: float = 60.0

    # Throughput thresholds (FPS)
    throughput_warning_fps: float = 25.0
    throughput_critical_fps: float = 20.0

    # Error rate thresholds (percentage)
    error_rate_warning_pct: float = 2.0
    error_rate_critical_pct: float = 5.0

    # Resource utilization thresholds
    gpu_utilization_warning_pct: float = 85.0
    gpu_utilization_critical_pct: float = 95.0
    memory_usage_warning_mb: float = 6000.0
    memory_usage_critical_mb: float = 7500.0

    # Queue depth thresholds
    queue_depth_warning: int = 50
    queue_depth_critical: int = 100


@dataclass
class PerformanceAlert:
    """Performance alert information."""

    timestamp: float
    severity: AlertSeverity
    metric: PerformanceMetric
    current_value: float
    threshold_value: float
    message: str
    component: str = "streaming_pipeline"
    camera_id: str | None = None
    suggested_actions: list[str] = field(default_factory=list)


@dataclass
class LatencyMeasurement:
    """Single latency measurement."""

    timestamp: float
    latency_ms: float
    camera_id: str
    frame_id: str
    component: str = "end_to_end"
    stage_breakdown: dict[str, float] | None = None


class RollingMetrics:
    """Rolling window metrics calculator."""

    def __init__(self, window_size: int = 1000, window_duration_seconds: int = 60):
        self.window_size = window_size
        self.window_duration_seconds = window_duration_seconds
        self.measurements: deque[LatencyMeasurement] = deque(maxlen=window_size)
        self.error_count = 0
        self.total_requests = 0

    def add_measurement(self, measurement: LatencyMeasurement) -> None:
        """Add a new measurement to the rolling window."""
        # Remove old measurements outside time window
        current_time = time.time()
        while (self.measurements and
               current_time - self.measurements[0].timestamp > self.window_duration_seconds):
            self.measurements.popleft()

        self.measurements.append(measurement)
        self.total_requests += 1

    def add_error(self) -> None:
        """Record an error occurrence."""
        self.error_count += 1
        self.total_requests += 1

    def get_percentiles(self) -> dict[str, float]:
        """Calculate latency percentiles."""
        if not self.measurements:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        latencies = [m.latency_ms for m in self.measurements]

        return {
            "p50": np.percentile(latencies, 50),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99),
            "avg": np.mean(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "std": np.std(latencies)
        }

    def get_throughput(self) -> float:
        """Calculate current throughput in FPS."""
        if not self.measurements:
            return 0.0

        time_span = time.time() - self.measurements[0].timestamp
        if time_span == 0:
            return 0.0

        return len(self.measurements) / time_span

    def get_error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.total_requests == 0:
            return 0.0

        return (self.error_count / self.total_requests) * 100

    def reset_error_count(self) -> None:
        """Reset error counters."""
        self.error_count = 0
        self.total_requests = len(self.measurements)


class ResourceMonitor:
    """System resource monitoring."""

    def __init__(self, gpu_device_id: int = 0):
        self.gpu_device_id = gpu_device_id
        self.gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()

    def get_gpu_metrics(self) -> dict[str, float]:
        """Get GPU utilization and memory metrics."""
        if not self.gpu_available:
            return {"utilization_pct": 0.0, "memory_used_mb": 0.0, "memory_total_mb": 0.0}

        try:
            device = torch.cuda.get_device_properties(self.gpu_device_id)
            memory_used = torch.cuda.memory_allocated(self.gpu_device_id) / (1024**2)
            memory_total = device.total_memory / (1024**2)

            # GPU utilization would require nvidia-ml-py in production
            utilization = 0.0  # Placeholder

            return {
                "utilization_pct": utilization,
                "memory_used_mb": memory_used,
                "memory_total_mb": memory_total,
                "memory_utilization_pct": (memory_used / memory_total) * 100
            }

        except Exception as e:
            logger.warning(f"Failed to get GPU metrics: {e}")
            return {"utilization_pct": 0.0, "memory_used_mb": 0.0, "memory_total_mb": 0.0}

    def get_cpu_metrics(self) -> dict[str, float]:
        """Get CPU utilization metrics."""
        try:
            return {
                "utilization_pct": psutil.cpu_percent(interval=None),
                "load_avg_1m": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0,
                "context_switches": psutil.cpu_stats().ctx_switches if hasattr(psutil, 'cpu_stats') else 0
            }
        except Exception as e:
            logger.warning(f"Failed to get CPU metrics: {e}")
            return {"utilization_pct": 0.0, "load_avg_1m": 0.0}

    def get_memory_metrics(self) -> dict[str, float]:
        """Get system memory metrics."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_mb": memory.total / (1024**2),
                "used_mb": memory.used / (1024**2),
                "available_mb": memory.available / (1024**2),
                "utilization_pct": memory.percent
            }
        except Exception as e:
            logger.warning(f"Failed to get memory metrics: {e}")
            return {"total_mb": 0.0, "used_mb": 0.0, "available_mb": 0.0, "utilization_pct": 0.0}


class PerformanceTrendAnalyzer:
    """Analyze performance trends and predict degradation."""

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.metric_history: dict[str, deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    def add_metric_point(self, metric_name: str, timestamp: float, value: float) -> None:
        """Add a metric data point."""
        self.metric_history[metric_name].append((timestamp, value))

    def detect_trend(self, metric_name: str, min_points: int = 10) -> str | None:
        """Detect performance trends (improving/degrading/stable)."""
        if metric_name not in self.metric_history:
            return None

        data = list(self.metric_history[metric_name])
        if len(data) < min_points:
            return None

        # Calculate trend using linear regression slope
        timestamps = np.array([point[0] for point in data])
        values = np.array([point[1] for point in data])

        # Normalize timestamps
        timestamps = timestamps - timestamps[0]

        # Calculate slope
        if len(timestamps) > 1:
            slope = np.polyfit(timestamps, values, 1)[0]

            # Determine trend based on slope and metric type
            if metric_name in ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "error_rate_pct"]:
                # For these metrics, positive slope is bad
                if slope > 0.1:
                    return "degrading"
                elif slope < -0.1:
                    return "improving"
            else:
                # For throughput, positive slope is good
                if slope > 0.1:
                    return "improving"
                elif slope < -0.1:
                    return "degrading"

        return "stable"

    def predict_sla_violation(
        self,
        metric_name: str,
        threshold: float,
        prediction_window_minutes: int = 5
    ) -> dict[str, Any]:
        """Predict if metric will violate SLA in the near future."""
        if metric_name not in self.metric_history:
            return {"violation_predicted": False, "confidence": 0.0}

        data = list(self.metric_history[metric_name])
        if len(data) < 5:
            return {"violation_predicted": False, "confidence": 0.0}

        try:
            # Simple linear extrapolation
            timestamps = np.array([point[0] for point in data])
            values = np.array([point[1] for point in data])

            # Fit linear model
            coeffs = np.polyfit(timestamps, values, 1)
            slope, intercept = coeffs

            # Predict value in prediction_window_minutes
            future_timestamp = timestamps[-1] + (prediction_window_minutes * 60)
            predicted_value = slope * future_timestamp + intercept

            # Check for violation
            violation_predicted = False
            confidence = 0.0

            if metric_name in ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "error_rate_pct"]:
                violation_predicted = predicted_value > threshold
                if violation_predicted:
                    confidence = min((predicted_value - threshold) / threshold, 1.0)
            else:
                violation_predicted = predicted_value < threshold
                if violation_predicted:
                    confidence = min((threshold - predicted_value) / threshold, 1.0)

            return {
                "violation_predicted": violation_predicted,
                "predicted_value": predicted_value,
                "threshold": threshold,
                "confidence": confidence,
                "time_to_violation_minutes": prediction_window_minutes if violation_predicted else None
            }

        except Exception as e:
            logger.warning(f"SLA violation prediction failed for {metric_name}: {e}")
            return {"violation_predicted": False, "confidence": 0.0}


class CircuitBreakerIntegration:
    """Circuit breaker integration for automatic load shedding."""

    def __init__(self, failure_threshold: int = 10, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open

    def record_success(self) -> None:
        """Record successful operation."""
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
            logger.info("Circuit breaker reset to closed state")

    def record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold and self.state == "closed":
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def should_allow_request(self) -> bool:
        """Check if request should be allowed through."""
        if self.state == "closed":
            return True

        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
                logger.info("Circuit breaker moved to half-open state")
                return True
            return False

        # half_open state - allow limited requests
        return True

    def get_state(self) -> dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "time_until_retry": max(0, self.recovery_timeout - (time.time() - self.last_failure_time))
        }


class UltraFastPerformanceMonitor:
    """
    Ultra-fast performance monitoring system with automated alerting and remediation.
    
    Features:
    - Real-time P99 latency tracking
    - Automated performance degradation detection
    - SLA violation prediction
    - Circuit breaker integration
    - Resource utilization monitoring
    """

    def __init__(
        self,
        thresholds: PerformanceThresholds | None = None,
        enable_alerting: bool = True,
        enable_trend_analysis: bool = True,
        enable_circuit_breaker: bool = True
    ):
        self.thresholds = thresholds or PerformanceThresholds()
        self.enable_alerting = enable_alerting
        self.enable_trend_analysis = enable_trend_analysis
        self.enable_circuit_breaker = enable_circuit_breaker

        # Metrics tracking
        self.rolling_metrics = RollingMetrics()
        self.resource_monitor = ResourceMonitor()
        self.trend_analyzer = PerformanceTrendAnalyzer() if enable_trend_analysis else None
        self.circuit_breaker = CircuitBreakerIntegration() if enable_circuit_breaker else None

        # Alerting
        self.alert_handlers: list[Callable[[PerformanceAlert], None]] = []
        self.active_alerts: dict[str, PerformanceAlert] = {}
        self.alert_cooldowns: dict[str, float] = {}
        self.cooldown_period = 300  # 5 minutes

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: asyncio.Task | None = None

        # Performance stats
        self.monitor_stats = {
            "alerts_generated": 0,
            "sla_violations": 0,
            "circuit_breaker_trips": 0,
            "monitoring_uptime_seconds": 0,
            "start_time": time.time()
        }

        logger.info("Ultra-fast performance monitor initialized")

    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]) -> None:
        """Add alert handler function."""
        self.alert_handlers.append(handler)

    def record_latency(
        self,
        latency_ms: float,
        camera_id: str,
        frame_id: str,
        component: str = "end_to_end",
        stage_breakdown: dict[str, float] | None = None
    ) -> None:
        """Record a latency measurement."""
        measurement = LatencyMeasurement(
            timestamp=time.time(),
            latency_ms=latency_ms,
            camera_id=camera_id,
            frame_id=frame_id,
            component=component,
            stage_breakdown=stage_breakdown
        )

        self.rolling_metrics.add_measurement(measurement)

        # Update trend analyzer
        if self.trend_analyzer:
            self.trend_analyzer.add_metric_point(
                "latency_end_to_end", measurement.timestamp, latency_ms
            )

        # Circuit breaker integration
        if self.circuit_breaker:
            if latency_ms > self.thresholds.latency_p99_critical_ms:
                self.circuit_breaker.record_failure()
            else:
                self.circuit_breaker.record_success()

    def record_error(self, camera_id: str = "unknown", error_type: str = "general") -> None:
        """Record an error occurrence."""
        self.rolling_metrics.add_error()

        if self.circuit_breaker:
            self.circuit_breaker.record_failure()

        # Update trend analyzer
        if self.trend_analyzer:
            self.trend_analyzer.add_metric_point(
                "error_rate_pct", time.time(), self.rolling_metrics.get_error_rate()
            )

    def should_allow_request(self) -> bool:
        """Check if new requests should be allowed (circuit breaker)."""
        if not self.circuit_breaker:
            return True

        return self.circuit_breaker.should_allow_request()

    async def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self.monitoring_active = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Starting performance monitoring loop")

        while self.monitoring_active:
            try:
                # Check latency thresholds
                await self._check_latency_thresholds()

                # Check resource utilization
                await self._check_resource_thresholds()

                # Check error rates
                await self._check_error_rate_thresholds()

                # Trend analysis and SLA prediction
                if self.trend_analyzer:
                    await self._check_performance_trends()

                # Clean up old alerts
                self._cleanup_old_alerts()

                # Update uptime
                self.monitor_stats["monitoring_uptime_seconds"] = (
                    time.time() - self.monitor_stats["start_time"]
                )

                # Sleep between checks
                await asyncio.sleep(1.0)  # Check every second

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5.0)  # Longer sleep on error

    async def _check_latency_thresholds(self) -> None:
        """Check latency thresholds and generate alerts."""
        percentiles = self.rolling_metrics.get_percentiles()

        # Check P50 latency
        await self._check_threshold_and_alert(
            PerformanceMetric.LATENCY_P50,
            percentiles["p50"],
            self.thresholds.latency_p50_warning_ms,
            self.thresholds.latency_p50_critical_ms,
            "P50 latency",
            ["Increase batch size", "Enable frame skipping", "Scale horizontally"]
        )

        # Check P95 latency
        await self._check_threshold_and_alert(
            PerformanceMetric.LATENCY_P95,
            percentiles["p95"],
            self.thresholds.latency_p95_warning_ms,
            self.thresholds.latency_p95_critical_ms,
            "P95 latency",
            ["Reduce batch timeout", "Enable CUDA graphs", "Optimize NMS"]
        )

        # Check P99 latency (most critical)
        await self._check_threshold_and_alert(
            PerformanceMetric.LATENCY_P99,
            percentiles["p99"],
            self.thresholds.latency_p99_warning_ms,
            self.thresholds.latency_p99_critical_ms,
            "P99 latency",
            ["Emergency frame skipping", "Circuit breaker activation", "Model switching"]
        )

    async def _check_resource_thresholds(self) -> None:
        """Check resource utilization thresholds."""
        # GPU metrics
        gpu_metrics = self.resource_monitor.get_gpu_metrics()
        if gpu_metrics["memory_used_mb"] > 0:
            await self._check_threshold_and_alert(
                PerformanceMetric.GPU_UTILIZATION,
                gpu_metrics.get("utilization_pct", 0),
                self.thresholds.gpu_utilization_warning_pct,
                self.thresholds.gpu_utilization_critical_pct,
                "GPU utilization",
                ["Reduce batch size", "Enable model quantization", "Scale to multiple GPUs"]
            )

        # Memory metrics
        memory_metrics = self.resource_monitor.get_memory_metrics()
        await self._check_threshold_and_alert(
            PerformanceMetric.MEMORY_USAGE,
            memory_metrics["used_mb"],
            self.thresholds.memory_usage_warning_mb,
            self.thresholds.memory_usage_critical_mb,
            "Memory usage",
            ["Clear tensor caches", "Reduce queue sizes", "Restart services"]
        )

    async def _check_error_rate_thresholds(self) -> None:
        """Check error rate thresholds."""
        error_rate = self.rolling_metrics.get_error_rate()

        await self._check_threshold_and_alert(
            PerformanceMetric.ERROR_RATE,
            error_rate,
            self.thresholds.error_rate_warning_pct,
            self.thresholds.error_rate_critical_pct,
            "Error rate",
            ["Check model health", "Validate input data", "Restart inference engine"]
        )

    async def _check_threshold_and_alert(
        self,
        metric: PerformanceMetric,
        current_value: float,
        warning_threshold: float,
        critical_threshold: float,
        metric_description: str,
        suggested_actions: list[str]
    ) -> None:
        """Check threshold and generate alert if needed."""
        alert_key = metric.value

        # Check if in cooldown
        if (alert_key in self.alert_cooldowns and
            time.time() - self.alert_cooldowns[alert_key] < self.cooldown_period):
            return

        severity = None
        threshold_value = 0.0

        if current_value >= critical_threshold:
            severity = AlertSeverity.CRITICAL
            threshold_value = critical_threshold
        elif current_value >= warning_threshold:
            severity = AlertSeverity.WARNING
            threshold_value = warning_threshold

        if severity:
            alert = PerformanceAlert(
                timestamp=time.time(),
                severity=severity,
                metric=metric,
                current_value=current_value,
                threshold_value=threshold_value,
                message=f"{metric_description} {current_value:.2f} exceeds {severity.value} threshold {threshold_value:.2f}",
                suggested_actions=suggested_actions
            )

            await self._handle_alert(alert)
            self.alert_cooldowns[alert_key] = time.time()

    async def _check_performance_trends(self) -> None:
        """Check performance trends and predict SLA violations."""
        if not self.trend_analyzer:
            return

        # Check latency trends
        for metric_name in ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]:
            trend = self.trend_analyzer.detect_trend(metric_name)

            if trend == "degrading":
                # Predict SLA violation
                threshold = getattr(self.thresholds, f"{metric_name.replace('latency_', '').replace('_ms', '')}_critical_ms")
                prediction = self.trend_analyzer.predict_sla_violation(
                    metric_name, threshold, prediction_window_minutes=5
                )

                if prediction["violation_predicted"] and prediction["confidence"] > 0.7:
                    alert = PerformanceAlert(
                        timestamp=time.time(),
                        severity=AlertSeverity.WARNING,
                        metric=PerformanceMetric(metric_name),
                        current_value=prediction["predicted_value"],
                        threshold_value=threshold,
                        message=f"SLA violation predicted for {metric_name} in {prediction['time_to_violation_minutes']} minutes",
                        suggested_actions=["Proactive scaling", "Enable load shedding", "Alert operations team"]
                    )

                    await self._handle_alert(alert)

    async def _handle_alert(self, alert: PerformanceAlert) -> None:
        """Handle performance alert."""
        self.active_alerts[alert.metric.value] = alert
        self.monitor_stats["alerts_generated"] += 1

        if alert.severity == AlertSeverity.CRITICAL:
            self.monitor_stats["sla_violations"] += 1

        logger.warning(f"Performance alert: {alert.message}")

        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def _cleanup_old_alerts(self) -> None:
        """Clean up old resolved alerts."""
        current_time = time.time()
        expired_alerts = []

        for alert_key, alert in self.active_alerts.items():
            if current_time - alert.timestamp > 300:  # 5 minutes
                expired_alerts.append(alert_key)

        for alert_key in expired_alerts:
            del self.active_alerts[alert_key]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        percentiles = self.rolling_metrics.get_percentiles()
        throughput = self.rolling_metrics.get_throughput()
        error_rate = self.rolling_metrics.get_error_rate()
        gpu_metrics = self.resource_monitor.get_gpu_metrics()
        memory_metrics = self.resource_monitor.get_memory_metrics()

        summary = {
            "latency_metrics": {
                "p50_ms": percentiles["p50"],
                "p95_ms": percentiles["p95"],
                "p99_ms": percentiles["p99"],
                "avg_ms": percentiles["avg"],
                "min_ms": percentiles["min"],
                "max_ms": percentiles["max"],
                "std_ms": percentiles["std"]
            },
            "throughput_metrics": {
                "current_fps": throughput,
                "target_fps": 30.0,  # Target FPS
                "throughput_efficiency": min(throughput / 30.0, 1.0) * 100
            },
            "error_metrics": {
                "error_rate_pct": error_rate,
                "total_errors": self.rolling_metrics.error_count,
                "total_requests": self.rolling_metrics.total_requests
            },
            "resource_metrics": {
                "gpu": gpu_metrics,
                "memory": memory_metrics
            },
            "sla_compliance": {
                "latency_p99_target_met": percentiles["p99"] <= self.thresholds.latency_p99_critical_ms,
                "error_rate_target_met": error_rate <= self.thresholds.error_rate_warning_pct,
                "throughput_target_met": throughput >= self.thresholds.throughput_warning_fps,
                "overall_sla_met": (
                    percentiles["p99"] <= self.thresholds.latency_p99_critical_ms and
                    error_rate <= self.thresholds.error_rate_warning_pct and
                    throughput >= self.thresholds.throughput_warning_fps
                )
            },
            "active_alerts": [
                {
                    "metric": alert.metric.value,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp
                }
                for alert in self.active_alerts.values()
            ],
            "circuit_breaker": self.circuit_breaker.get_state() if self.circuit_breaker else None,
            "monitor_stats": self.monitor_stats
        }

        return summary

    def get_health_status(self) -> str:
        """Get overall system health status."""
        summary = self.get_performance_summary()

        if summary["active_alerts"]:
            critical_alerts = [a for a in summary["active_alerts"] if a["severity"] == "critical"]
            if critical_alerts:
                return "critical"
            else:
                return "warning"

        if not summary["sla_compliance"]["overall_sla_met"]:
            return "degraded"

        return "healthy"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_monitoring()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_monitoring()


# Alert handler implementations
def console_alert_handler(alert: PerformanceAlert) -> None:
    """Simple console alert handler."""
    print(f"[{alert.severity.value.upper()}] {alert.message}")
    if alert.suggested_actions:
        print(f"Suggested actions: {', '.join(alert.suggested_actions)}")


def slack_alert_handler(webhook_url: str) -> Callable[[PerformanceAlert], None]:
    """Create Slack alert handler."""
    def handler(alert: PerformanceAlert) -> None:
        # In production, this would send to Slack webhook
        message = f"🚨 {alert.severity.value.upper()}: {alert.message}"
        logger.info(f"Would send Slack alert: {message}")

    return handler


# Factory function
async def create_performance_monitor(
    target_latency_p99_ms: float = 50.0,
    enable_alerting: bool = True,
    enable_slack_alerts: bool = False,
    slack_webhook_url: str | None = None
) -> UltraFastPerformanceMonitor:
    """Create and configure performance monitor."""

    thresholds = PerformanceThresholds(
        latency_p99_critical_ms=target_latency_p99_ms,
        latency_p95_critical_ms=target_latency_p99_ms * 0.9,
        latency_p50_critical_ms=target_latency_p99_ms * 0.8
    )

    monitor = UltraFastPerformanceMonitor(
        thresholds=thresholds,
        enable_alerting=enable_alerting,
        enable_trend_analysis=True,
        enable_circuit_breaker=True
    )

    # Add alert handlers
    monitor.add_alert_handler(console_alert_handler)

    if enable_slack_alerts and slack_webhook_url:
        monitor.add_alert_handler(slack_alert_handler(slack_webhook_url))

    return monitor
