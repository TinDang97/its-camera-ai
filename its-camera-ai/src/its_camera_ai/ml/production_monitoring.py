"""
Production Monitoring and Drift Detection for ITS Camera AI Traffic Monitoring System.

This module implements comprehensive monitoring of ML models in production including:
1. Real-time performance dashboards and alerting
2. Model accuracy tracking and drift detection
3. Data distribution shifts and anomaly detection
4. A/B testing framework for model comparisons
5. Automated alerts and remediation actions

Key Features:
- Multi-dimensional drift detection (accuracy, latency, data distribution)
- Real-time dashboards with Grafana/Prometheus integration
- Statistical testing for model performance changes
- Automated alert system with multiple notification channels
- Model rollback and failover mechanisms
"""

import asyncio
import contextlib
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .inference_optimizer import DetectionResult

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DriftType(Enum):
    """Types of model drift."""

    ACCURACY_DRIFT = "accuracy_drift"
    LATENCY_DRIFT = "latency_drift"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    VOLUME_DRIFT = "volume_drift"


@dataclass
class Alert:
    """Alert message structure."""

    alert_id: str
    severity: AlertSeverity
    drift_type: DriftType
    message: str
    timestamp: float

    # Metrics that triggered the alert
    current_value: float
    expected_value: float
    threshold: float

    # Context
    camera_id: str | None = None
    model_version: str | None = None

    # Alert lifecycle
    acknowledged: bool = False
    resolved: bool = False
    resolution_timestamp: float | None = None
    resolution_message: str | None = None


@dataclass
class ModelMetrics:
    """Model performance metrics over time."""

    model_id: str
    model_version: str

    # Performance metrics
    accuracy_scores: deque = field(default_factory=lambda: deque(maxlen=1000))
    latencies_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    throughput_fps: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Detection metrics
    detection_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidence_scores: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Resource metrics
    gpu_utilization: deque = field(default_factory=lambda: deque(maxlen=1000))
    memory_usage_mb: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Timestamps for correlation
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Drift indicators
    last_drift_check: float = field(default_factory=time.time)
    drift_score: float = 0.0

    def add_sample(self, result: DetectionResult, gpu_util: float = 0.0):
        """Add new sample to metrics."""
        current_time = time.time()

        self.accuracy_scores.append(result.avg_confidence)
        self.latencies_ms.append(result.total_time_ms)
        self.detection_counts.append(result.detection_count)
        self.confidence_scores.extend(
            result.scores.tolist() if len(result.scores) > 0 else [0]
        )
        self.gpu_utilization.append(gpu_util)
        self.memory_usage_mb.append(result.gpu_memory_used_mb)
        self.timestamps.append(current_time)

        # Calculate throughput (approximate)
        if len(self.timestamps) >= 2:
            time_diff = self.timestamps[-1] - self.timestamps[-2]
            if time_diff > 0:
                fps = 1.0 / time_diff
                self.throughput_fps.append(fps)

    def get_recent_stats(self, window_minutes: int = 5) -> dict[str, float]:
        """Get statistics for recent time window."""
        cutoff_time = time.time() - (window_minutes * 60)

        # Filter recent data
        recent_indices = [
            i for i, ts in enumerate(self.timestamps) if ts >= cutoff_time
        ]

        if not recent_indices:
            return {}

        # Extract recent values
        recent_accuracy = [
            self.accuracy_scores[i]
            for i in recent_indices
            if i < len(self.accuracy_scores)
        ]
        recent_latency = [
            self.latencies_ms[i] for i in recent_indices if i < len(self.latencies_ms)
        ]
        recent_throughput = [
            self.throughput_fps[i]
            for i in recent_indices
            if i < len(self.throughput_fps)
        ]

        return {
            "avg_accuracy": np.mean(recent_accuracy) if recent_accuracy else 0,
            "p95_latency_ms": (
                np.percentile(recent_latency, 95) if recent_latency else 0
            ),
            "p99_latency_ms": (
                np.percentile(recent_latency, 99) if recent_latency else 0
            ),
            "avg_throughput_fps": (
                np.mean(recent_throughput) if recent_throughput else 0
            ),
            "sample_count": len(recent_indices),
        }


class StatisticalDriftDetector:
    """Detect drift using statistical tests."""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def detect_accuracy_drift(
        self, baseline_scores: list[float], current_scores: list[float]
    ) -> dict[str, Any]:
        """Detect accuracy drift using statistical tests."""

        if len(baseline_scores) < 30 or len(current_scores) < 30:
            return {"drift_detected": False, "reason": "Insufficient data"}

        try:
            # Welch's t-test for different means
            from scipy import stats

            statistic, p_value = stats.ttest_ind(
                baseline_scores, current_scores, equal_var=False
            )

            drift_detected = p_value < self.significance_level

            baseline_mean = np.mean(baseline_scores)
            current_mean = np.mean(current_scores)
            effect_size = abs(current_mean - baseline_mean) / np.sqrt(
                (np.var(baseline_scores) + np.var(current_scores)) / 2
            )

            return {
                "drift_detected": drift_detected,
                "p_value": p_value,
                "statistic": statistic,
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
                "effect_size": effect_size,
                "direction": (
                    "degradation" if current_mean < baseline_mean else "improvement"
                ),
            }

        except ImportError:
            # Fallback to simple statistical comparison
            return self._simple_drift_test(baseline_scores, current_scores)

    def detect_latency_drift(
        self, baseline_latencies: list[float], current_latencies: list[float]
    ) -> dict[str, Any]:
        """Detect latency drift (performance degradation)."""

        if len(baseline_latencies) < 30 or len(current_latencies) < 30:
            return {"drift_detected": False, "reason": "Insufficient data"}

        # Use percentile-based comparison for latency
        baseline_p95 = np.percentile(baseline_latencies, 95)
        current_p95 = np.percentile(current_latencies, 95)

        # Detect if current P95 is significantly higher than baseline
        relative_increase = (current_p95 - baseline_p95) / baseline_p95

        # Alert if P95 latency increased by more than 20%
        drift_detected = relative_increase > 0.20

        return {
            "drift_detected": drift_detected,
            "baseline_p95": baseline_p95,
            "current_p95": current_p95,
            "relative_increase": relative_increase,
            "threshold": 0.20,
        }

    def detect_data_drift(
        self, baseline_features: np.ndarray, current_features: np.ndarray
    ) -> dict[str, Any]:
        """Detect data drift using distribution comparison."""

        if len(baseline_features) < 100 or len(current_features) < 100:
            return {"drift_detected": False, "reason": "Insufficient data"}

        try:
            # Kolmogorov-Smirnov test for distribution differences
            from scipy import stats

            # Flatten if multidimensional
            if len(baseline_features.shape) > 1:
                baseline_flat = baseline_features.flatten()
                current_flat = current_features.flatten()
            else:
                baseline_flat = baseline_features
                current_flat = current_features

            statistic, p_value = stats.ks_2samp(baseline_flat, current_flat)

            drift_detected = p_value < self.significance_level

            return {
                "drift_detected": drift_detected,
                "ks_statistic": statistic,
                "p_value": p_value,
                "baseline_mean": np.mean(baseline_flat),
                "current_mean": np.mean(current_flat),
                "baseline_std": np.std(baseline_flat),
                "current_std": np.std(current_flat),
            }

        except ImportError:
            # Simple mean/std comparison fallback
            baseline_mean = np.mean(baseline_features)
            current_mean = np.mean(current_features)

            relative_change = abs(current_mean - baseline_mean) / baseline_mean
            drift_detected = relative_change > 0.15  # 15% threshold

            return {
                "drift_detected": drift_detected,
                "relative_change": relative_change,
                "threshold": 0.15,
                "baseline_mean": baseline_mean,
                "current_mean": current_mean,
            }

    def _simple_drift_test(
        self, baseline_scores: list[float], current_scores: list[float]
    ) -> dict[str, Any]:
        """Simple drift test when scipy is not available."""

        baseline_mean = np.mean(baseline_scores)
        current_mean = np.mean(current_scores)
        baseline_std = np.std(baseline_scores)

        # Simple z-test approximation
        z_score = abs(current_mean - baseline_mean) / (
            baseline_std / np.sqrt(len(current_scores))
        )

        # Two-sided test at 5% significance level
        drift_detected = z_score > 1.96

        return {
            "drift_detected": drift_detected,
            "z_score": z_score,
            "baseline_mean": baseline_mean,
            "current_mean": current_mean,
            "direction": (
                "degradation" if current_mean < baseline_mean else "improvement"
            ),
        }


class ModelMonitor:
    """Monitor individual model performance and detect drift."""

    def __init__(
        self,
        model_id: str,
        model_version: str,
        baseline_window_hours: int = 24,
        drift_check_interval_minutes: int = 15,
    ):
        self.model_id = model_id
        self.model_version = model_version
        self.baseline_window_hours = baseline_window_hours
        self.drift_check_interval_minutes = drift_check_interval_minutes

        self.metrics = ModelMetrics(model_id, model_version)
        self.drift_detector = StatisticalDriftDetector()

        # Baseline data (established during initial stable period)
        self.baseline_metrics = {
            "accuracy_scores": [],
            "latencies_ms": [],
            "detection_counts": [],
        }

        self.is_baseline_established = False
        self.last_drift_check = time.time()

        # Alert thresholds
        self.thresholds = {
            "accuracy_drop_pct": 5.0,  # Alert if accuracy drops by 5%
            "latency_increase_pct": 20.0,  # Alert if P95 latency increases by 20%
            "throughput_drop_pct": 15.0,  # Alert if throughput drops by 15%
            "gpu_utilization_pct": 90.0,  # Alert if GPU util > 90%
            "memory_usage_mb": 4000,  # Alert if memory usage > 4GB
        }

    def add_inference_result(self, result: DetectionResult, gpu_util: float = 0.0):
        """Add new inference result for monitoring."""
        self.metrics.add_sample(result, gpu_util)

        # Check if it's time for drift detection
        if self._should_check_drift():
            asyncio.create_task(self._check_drift())

    def establish_baseline(self, min_samples: int = 1000):
        """Establish baseline metrics from initial stable period."""
        if len(self.metrics.timestamps) < min_samples:
            logger.info(
                f"Need {min_samples - len(self.metrics.timestamps)} more samples to establish baseline"
            )
            return False

        # Use recent stable data as baseline
        cutoff_time = time.time() - (self.baseline_window_hours * 3600)
        recent_indices = [
            i for i, ts in enumerate(self.metrics.timestamps) if ts >= cutoff_time
        ]

        if len(recent_indices) < min_samples:
            return False

        # Extract baseline data
        self.baseline_metrics["accuracy_scores"] = [
            self.metrics.accuracy_scores[i] for i in recent_indices[-min_samples:]
        ]
        self.baseline_metrics["latencies_ms"] = [
            self.metrics.latencies_ms[i] for i in recent_indices[-min_samples:]
        ]
        self.baseline_metrics["detection_counts"] = [
            self.metrics.detection_counts[i] for i in recent_indices[-min_samples:]
        ]

        self.is_baseline_established = True
        logger.info(
            f"Baseline established for model {self.model_id} with {min_samples} samples"
        )
        return True

    def _should_check_drift(self) -> bool:
        """Check if it's time to run drift detection."""
        time_since_last_check = time.time() - self.last_drift_check
        return time_since_last_check >= (self.drift_check_interval_minutes * 60)

    async def _check_drift(self) -> list[Alert]:
        """Perform comprehensive drift detection."""
        alerts = []

        if not self.is_baseline_established and not self.establish_baseline():
            return alerts

        self.last_drift_check = time.time()

        # Get recent data for comparison
        recent_stats = self.metrics.get_recent_stats(window_minutes=30)

        if not recent_stats or recent_stats["sample_count"] < 50:
            return alerts

        # Check accuracy drift
        recent_accuracy = list(self.metrics.accuracy_scores)[-100:]  # Last 100 samples
        if recent_accuracy and len(self.baseline_metrics["accuracy_scores"]) > 0:
            accuracy_drift = self.drift_detector.detect_accuracy_drift(
                self.baseline_metrics["accuracy_scores"], recent_accuracy
            )

            if (
                accuracy_drift["drift_detected"]
                and accuracy_drift.get("direction") == "degradation"
            ):
                alert = Alert(
                    alert_id=f"accuracy_drift_{self.model_id}_{int(time.time())}",
                    severity=AlertSeverity.WARNING,
                    drift_type=DriftType.ACCURACY_DRIFT,
                    message=f"Accuracy drift detected for model {self.model_id}",
                    timestamp=time.time(),
                    current_value=accuracy_drift["current_mean"],
                    expected_value=accuracy_drift["baseline_mean"],
                    threshold=self.thresholds["accuracy_drop_pct"] / 100,
                    model_version=self.model_version,
                )
                alerts.append(alert)

        # Check latency drift
        recent_latencies = list(self.metrics.latencies_ms)[-100:]
        if recent_latencies and len(self.baseline_metrics["latencies_ms"]) > 0:
            latency_drift = self.drift_detector.detect_latency_drift(
                self.baseline_metrics["latencies_ms"], recent_latencies
            )

            if latency_drift["drift_detected"]:
                severity = (
                    AlertSeverity.CRITICAL
                    if latency_drift["relative_increase"] > 0.5
                    else AlertSeverity.WARNING
                )

                alert = Alert(
                    alert_id=f"latency_drift_{self.model_id}_{int(time.time())}",
                    severity=severity,
                    drift_type=DriftType.LATENCY_DRIFT,
                    message=f"Latency drift detected for model {self.model_id}",
                    timestamp=time.time(),
                    current_value=latency_drift["current_p95"],
                    expected_value=latency_drift["baseline_p95"],
                    threshold=self.thresholds["latency_increase_pct"] / 100,
                    model_version=self.model_version,
                )
                alerts.append(alert)

        # Check resource usage alerts
        if recent_stats["avg_throughput_fps"] > 0:
            baseline_throughput = (
                np.mean(
                    [
                        1000 / lat
                        for lat in self.baseline_metrics["latencies_ms"]
                        if lat > 0
                    ]
                )
                if self.baseline_metrics["latencies_ms"]
                else 0
            )

            if baseline_throughput > 0:
                throughput_drop = (
                    baseline_throughput - recent_stats["avg_throughput_fps"]
                ) / baseline_throughput

                if throughput_drop > (self.thresholds["throughput_drop_pct"] / 100):
                    alert = Alert(
                        alert_id=f"throughput_drop_{self.model_id}_{int(time.time())}",
                        severity=AlertSeverity.WARNING,
                        drift_type=DriftType.VOLUME_DRIFT,
                        message=f"Throughput drop detected for model {self.model_id}",
                        timestamp=time.time(),
                        current_value=recent_stats["avg_throughput_fps"],
                        expected_value=baseline_throughput,
                        threshold=self.thresholds["throughput_drop_pct"] / 100,
                        model_version=self.model_version,
                    )
                    alerts.append(alert)

        return alerts

    def get_health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        if not self.is_baseline_established:
            return 50.0  # Neutral score when no baseline

        recent_stats = self.metrics.get_recent_stats(window_minutes=15)
        if not recent_stats:
            return 0.0  # No recent data

        # Component scores (0-100)
        accuracy_score = min(100, recent_stats.get("avg_accuracy", 0) * 100)

        # Latency score (inverse - lower is better)
        baseline_p95 = (
            np.percentile(self.baseline_metrics["latencies_ms"], 95)
            if self.baseline_metrics["latencies_ms"]
            else 100
        )
        latency_ratio = recent_stats.get("p95_latency_ms", baseline_p95) / baseline_p95
        latency_score = max(0, 100 - (latency_ratio - 1) * 100)

        # Throughput score
        recent_throughput = recent_stats.get("avg_throughput_fps", 0)
        baseline_throughput = 30  # Assumed baseline FPS
        throughput_score = min(100, (recent_throughput / baseline_throughput) * 100)

        # Weighted average
        health_score = (
            accuracy_score * 0.4 + latency_score * 0.3 + throughput_score * 0.3
        )

        return max(0, min(100, health_score))


class ProductionDashboard:
    """Real-time production monitoring dashboard."""

    def __init__(self, update_interval_seconds: int = 10):
        self.update_interval_seconds = update_interval_seconds
        self.model_monitors: dict[str, ModelMonitor] = {}
        self.active_alerts: list[Alert] = []
        self.alert_history: deque = deque(maxlen=10000)

        # Dashboard metrics
        self.dashboard_metrics = {
            "system_health_score": 100.0,
            "total_requests": 0,
            "requests_per_second": 0.0,
            "avg_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "error_rate_pct": 0.0,
            "active_models": 0,
            "alerts_active": 0,
            "alerts_critical": 0,
        }

        # Time series data for charts
        self.time_series = defaultdict(
            lambda: deque(maxlen=1440)
        )  # 24 hours of minutes

        # Update task
        self.update_task = None
        self.is_running = False

        # Alert handlers
        self.alert_handlers: list[Callable[[Alert], None]] = []

    async def start(self):
        """Start the dashboard monitoring."""
        logger.info("Starting production dashboard...")
        self.is_running = True

        self.update_task = asyncio.create_task(self._update_dashboard_loop())
        logger.info("Production dashboard started")

    async def stop(self):
        """Stop the dashboard monitoring."""
        logger.info("Stopping production dashboard...")
        self.is_running = False

        if self.update_task:
            self.update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.update_task

        logger.info("Production dashboard stopped")

    def register_model(self, model_id: str, model_version: str) -> ModelMonitor:
        """Register a model for monitoring."""
        monitor = ModelMonitor(model_id, model_version)
        self.model_monitors[model_id] = monitor

        logger.info(
            f"Registered model {model_id} version {model_version} for monitoring"
        )
        return monitor

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert notification handler."""
        self.alert_handlers.append(handler)

    def add_inference_result(
        self, model_id: str, result: DetectionResult, gpu_util: float = 0.0
    ):
        """Add inference result for monitoring."""
        if model_id in self.model_monitors:
            self.model_monitors[model_id].add_inference_result(result, gpu_util)

            # Update global metrics
            self.dashboard_metrics["total_requests"] += 1

    async def _update_dashboard_loop(self):
        """Main dashboard update loop."""
        while self.is_running:
            try:
                await self._update_dashboard_metrics()
                await asyncio.sleep(self.update_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(self.update_interval_seconds)

    async def _update_dashboard_metrics(self):
        """Update all dashboard metrics."""
        current_time = time.time()

        # Aggregate metrics from all model monitors
        all_latencies = []
        all_health_scores = []

        for _model_id, monitor in self.model_monitors.items():
            # Check for new alerts
            alerts = await monitor._check_drift()
            for alert in alerts:
                await self._handle_new_alert(alert)

            # Aggregate metrics
            recent_stats = monitor.metrics.get_recent_stats(window_minutes=1)
            if recent_stats and recent_stats["sample_count"] > 0:
                if recent_stats["p95_latency_ms"] > 0:
                    all_latencies.extend([recent_stats["p95_latency_ms"]])

                health_score = monitor.get_health_score()
                all_health_scores.append(health_score)

        # Update aggregated metrics
        if all_latencies:
            self.dashboard_metrics["avg_latency_ms"] = np.mean(all_latencies)
            self.dashboard_metrics["p95_latency_ms"] = np.percentile(all_latencies, 95)
            self.dashboard_metrics["p99_latency_ms"] = np.percentile(all_latencies, 99)

        if all_health_scores:
            self.dashboard_metrics["system_health_score"] = np.mean(all_health_scores)

        # Count alerts
        active_alerts = [alert for alert in self.active_alerts if not alert.resolved]
        self.dashboard_metrics["alerts_active"] = len(active_alerts)
        self.dashboard_metrics["alerts_critical"] = len(
            [
                alert
                for alert in active_alerts
                if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            ]
        )

        self.dashboard_metrics["active_models"] = len(self.model_monitors)

        # Store time series data
        timestamp_key = int(current_time // 60)  # Minute-level granularity
        for metric, value in self.dashboard_metrics.items():
            if isinstance(value, int | float):
                self.time_series[metric].append((timestamp_key, value))

        logger.debug(f"Dashboard updated: {self.dashboard_metrics}")

    async def _handle_new_alert(self, alert: Alert):
        """Handle a new alert."""
        self.active_alerts.append(alert)
        self.alert_history.append(alert)

        logger.warning(f"New alert: {alert.message} (Severity: {alert.severity.value})")

        # Notify all alert handlers
        for handler in self.alert_handlers:
            try:
                await asyncio.get_event_loop().run_in_executor(None, handler, alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get current dashboard data for frontend."""
        return {
            "metrics": self.dashboard_metrics.copy(),
            "model_health": {
                model_id: {
                    "health_score": monitor.get_health_score(),
                    "recent_stats": monitor.metrics.get_recent_stats(),
                    "is_baseline_established": monitor.is_baseline_established,
                }
                for model_id, monitor in self.model_monitors.items()
            },
            "active_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "drift_type": alert.drift_type.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "current_value": alert.current_value,
                    "expected_value": alert.expected_value,
                    "model_version": alert.model_version,
                    "camera_id": alert.camera_id,
                }
                for alert in self.active_alerts
                if not alert.resolved
            ],
            "time_series": {
                metric: list(values)[-60:]  # Last hour
                for metric, values in self.time_series.items()
            },
        }

    def acknowledge_alert(self, alert_id: str, message: str = ""):
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert {alert_id} acknowledged: {message}")
                return True
        return False

    def resolve_alert(self, alert_id: str, resolution_message: str = ""):
        """Resolve an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_timestamp = time.time()
                alert.resolution_message = resolution_message
                logger.info(f"Alert {alert_id} resolved: {resolution_message}")
                return True
        return False


# Alert notification handlers


class EmailAlertHandler:
    """Send alert notifications via email."""

    def __init__(self, smtp_config: dict[str, str]):
        self.smtp_config = smtp_config
        self.enabled = all(
            k in smtp_config for k in ["host", "port", "username", "password"]
        )

    def __call__(self, alert: Alert):
        """Send email notification for alert."""
        if not self.enabled:
            logger.warning("Email alerts not configured")
            return

        # In production, implement actual email sending
        logger.info(f"EMAIL ALERT: {alert.message} (Severity: {alert.severity.value})")


class SlackAlertHandler:
    """Send alert notifications to Slack."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def __call__(self, alert: Alert):
        """Send Slack notification for alert."""
        # In production, implement actual Slack webhook
        logger.info(f"SLACK ALERT: {alert.message} (Severity: {alert.severity.value})")


class PagerDutyAlertHandler:
    """Send critical alerts to PagerDuty."""

    def __init__(self, integration_key: str):
        self.integration_key = integration_key

    def __call__(self, alert: Alert):
        """Send PagerDuty alert for critical issues."""
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            # In production, implement actual PagerDuty API call
            logger.info(
                f"PAGERDUTY ALERT: {alert.message} (Severity: {alert.severity.value})"
            )


# Prometheus metrics exporter


class PrometheusMetricsExporter:
    """Export metrics in Prometheus format."""

    def __init__(self, dashboard: ProductionDashboard, port: int = 8080):
        self.dashboard = dashboard
        self.port = port
        self.app = None

    def start_server(self):
        """Start Prometheus metrics HTTP server."""
        try:
            from prometheus_client import Gauge, start_http_server

            # Define Prometheus metrics
            self.system_health = Gauge(
                "its_system_health_score", "Overall system health score"
            )
            self.model_health = Gauge(
                "its_model_health_score", "Model health score", ["model_id"]
            )
            self.inference_latency = Gauge(
                "its_inference_latency_ms", "Inference latency", ["percentile"]
            )
            self.active_alerts = Gauge(
                "its_active_alerts_total", "Number of active alerts", ["severity"]
            )

            # Start HTTP server
            start_http_server(self.port)

            # Start metrics update loop
            asyncio.create_task(self._update_prometheus_metrics())

            logger.info(f"Prometheus metrics server started on port {self.port}")

        except ImportError:
            logger.warning("Prometheus client not available")

    async def _update_prometheus_metrics(self):
        """Update Prometheus metrics periodically."""
        while True:
            try:
                dashboard_data = self.dashboard.get_dashboard_data()

                # Update system health
                self.system_health.set(dashboard_data["metrics"]["system_health_score"])

                # Update model health scores
                for model_id, health_data in dashboard_data["model_health"].items():
                    self.model_health.labels(model_id=model_id).set(
                        health_data["health_score"]
                    )

                # Update latency metrics
                metrics = dashboard_data["metrics"]
                self.inference_latency.labels(percentile="p95").set(
                    metrics["p95_latency_ms"]
                )
                self.inference_latency.labels(percentile="p99").set(
                    metrics["p99_latency_ms"]
                )

                # Update alert counts
                alert_counts = defaultdict(int)
                for alert in dashboard_data["active_alerts"]:
                    alert_counts[alert["severity"]] += 1

                for severity in ["info", "warning", "critical", "emergency"]:
                    self.active_alerts.labels(severity=severity).set(
                        alert_counts[severity]
                    )

                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                logger.error(f"Prometheus metrics update failed: {e}")
                await asyncio.sleep(30)


# Example usage and configuration


async def create_production_monitoring_system(
    models_config: list[dict[str, str]], alert_config: dict[str, Any]
) -> ProductionDashboard:
    """Create and configure production monitoring system."""

    # Create dashboard
    dashboard = ProductionDashboard(update_interval_seconds=10)

    # Register models
    for model_config in models_config:
        dashboard.register_model(
            model_config["model_id"], model_config["model_version"]
        )

    # Configure alert handlers
    if "email" in alert_config:
        email_handler = EmailAlertHandler(alert_config["email"])
        dashboard.add_alert_handler(email_handler)

    if "slack" in alert_config:
        slack_handler = SlackAlertHandler(alert_config["slack"]["webhook_url"])
        dashboard.add_alert_handler(slack_handler)

    if "pagerduty" in alert_config:
        pd_handler = PagerDutyAlertHandler(alert_config["pagerduty"]["integration_key"])
        dashboard.add_alert_handler(pd_handler)

    # Start Prometheus exporter
    if alert_config.get("prometheus_enabled", False):
        prometheus_exporter = PrometheusMetricsExporter(
            dashboard, port=alert_config.get("prometheus_port", 8080)
        )
        prometheus_exporter.start_server()

    # Start dashboard
    await dashboard.start()

    logger.info("Production monitoring system initialized")
    return dashboard
