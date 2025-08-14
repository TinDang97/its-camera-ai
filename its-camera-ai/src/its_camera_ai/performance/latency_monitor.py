"""Comprehensive latency monitoring and SLA management system.

This module implements real-time latency tracking, SLA violation detection,
performance regression analysis, and automated alerting for <100ms end-to-end
latency requirements with detailed pipeline segment monitoring.
"""

import asyncio
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from .optimization_config import LatencyMonitoringConfig

logger = structlog.get_logger(__name__)

# Optional monitoring dependencies
try:
    import prometheus_client
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.info("Prometheus client not available - using internal metrics only")
    PROMETHEUS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.info("Requests library not available - webhook alerts disabled")
    REQUESTS_AVAILABLE = False


class AlertLevel(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class PipelineStage(str, Enum):
    """Pipeline stages for detailed latency tracking."""

    CAMERA_CAPTURE = "camera_capture"
    PREPROCESSING = "preprocessing"
    ML_INFERENCE = "ml_inference"
    POSTPROCESSING = "postprocessing"
    ENCODING = "encoding"
    NETWORK_TRANSMISSION = "network_transmission"
    CLIENT_RENDERING = "client_rendering"
    END_TO_END = "end_to_end"


@dataclass
class LatencyMeasurement:
    """Individual latency measurement with context."""

    timestamp: float
    latency_ms: float
    stage: PipelineStage
    camera_id: str
    request_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_sla_violation(self) -> bool:
        """Check if this measurement violates SLA."""
        # Default SLA is 100ms for end-to-end
        sla_thresholds = {
            PipelineStage.CAMERA_CAPTURE: 20.0,
            PipelineStage.PREPROCESSING: 10.0,
            PipelineStage.ML_INFERENCE: 50.0,
            PipelineStage.POSTPROCESSING: 5.0,
            PipelineStage.ENCODING: 10.0,
            PipelineStage.NETWORK_TRANSMISSION: 20.0,
            PipelineStage.CLIENT_RENDERING: 10.0,
            PipelineStage.END_TO_END: 100.0,
        }
        return self.latency_ms > sla_thresholds.get(self.stage, 100.0)


@dataclass
class LatencyMetrics:
    """Aggregated latency metrics for monitoring."""

    stage: PipelineStage
    window_seconds: int
    sample_count: int
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    std_deviation_ms: float
    sla_violation_rate: float
    throughput_per_second: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class SLAViolation:
    """SLA violation record."""

    measurement: LatencyMeasurement
    violation_type: str
    severity: AlertLevel
    threshold_ms: float
    actual_ms: float
    context: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class RollingWindow:
    """Rolling window for latency measurements."""

    def __init__(self, window_seconds: int, max_samples: int = 10000):
        """Initialize rolling window.
        
        Args:
            window_seconds: Window size in seconds
            max_samples: Maximum samples to keep
        """
        self.window_seconds = window_seconds
        self.max_samples = max_samples
        self.measurements: deque[LatencyMeasurement] = deque(maxlen=max_samples)

    def add_measurement(self, measurement: LatencyMeasurement) -> None:
        """Add measurement to rolling window.
        
        Args:
            measurement: Latency measurement to add
        """
        self.measurements.append(measurement)
        self._cleanup_old_measurements()

    def _cleanup_old_measurements(self) -> None:
        """Remove measurements outside the window."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds

        while self.measurements and self.measurements[0].timestamp < cutoff_time:
            self.measurements.popleft()

    def get_metrics(self, stage: PipelineStage) -> LatencyMetrics | None:
        """Calculate metrics for the rolling window.
        
        Args:
            stage: Pipeline stage to calculate metrics for
            
        Returns:
            Optional[LatencyMetrics]: Calculated metrics or None
        """
        # Filter measurements for the stage
        stage_measurements = [
            m for m in self.measurements
            if m.stage == stage
        ]

        if not stage_measurements:
            return None

        # Extract latencies
        latencies = [m.latency_ms for m in stage_measurements]

        # Calculate statistics
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)

        # Percentiles
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        p95_index = int(0.95 * n)
        p99_index = int(0.99 * n)

        p95_latency = sorted_latencies[min(p95_index, n - 1)]
        p99_latency = sorted_latencies[min(p99_index, n - 1)]

        # SLA violations
        violations = sum(1 for m in stage_measurements if m.is_sla_violation)
        violation_rate = violations / len(stage_measurements)

        # Throughput (measurements per second)
        throughput = len(stage_measurements) / self.window_seconds

        return LatencyMetrics(
            stage=stage,
            window_seconds=self.window_seconds,
            sample_count=len(stage_measurements),
            mean_latency_ms=mean_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            max_latency_ms=max(latencies),
            min_latency_ms=min(latencies),
            std_deviation_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            sla_violation_rate=violation_rate,
            throughput_per_second=throughput,
        )


class PrometheusExporter:
    """Prometheus metrics exporter for latency monitoring."""

    def __init__(self, port: int = 8090):
        """Initialize Prometheus exporter.
        
        Args:
            port: HTTP server port for metrics
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus not available - metrics export disabled")
            return

        self.port = port
        self.server_started = False

        # Prometheus metrics
        self.latency_histogram = Histogram(
            'streaming_latency_seconds',
            'Streaming pipeline latency by stage',
            ['stage', 'camera_id'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )

        self.sla_violations = Counter(
            'streaming_sla_violations_total',
            'Total SLA violations by stage',
            ['stage', 'severity']
        )

        self.active_streams = Gauge(
            'streaming_active_streams',
            'Number of active streams'
        )

        self.throughput = Gauge(
            'streaming_throughput_fps',
            'Streaming throughput in frames per second',
            ['camera_id']
        )

    def start_server(self) -> None:
        """Start Prometheus HTTP server."""
        if not PROMETHEUS_AVAILABLE or self.server_started:
            return

        try:
            start_http_server(self.port)
            self.server_started = True
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")

    def record_measurement(self, measurement: LatencyMeasurement) -> None:
        """Record measurement in Prometheus metrics.
        
        Args:
            measurement: Latency measurement to record
        """
        if not PROMETHEUS_AVAILABLE:
            return

        try:
            # Record latency histogram
            self.latency_histogram.labels(
                stage=measurement.stage.value,
                camera_id=measurement.camera_id
            ).observe(measurement.latency_ms / 1000.0)  # Convert to seconds

            # Record SLA violation if applicable
            if measurement.is_sla_violation:
                severity = "critical" if measurement.latency_ms > 200 else "warning"
                self.sla_violations.labels(
                    stage=measurement.stage.value,
                    severity=severity
                ).inc()

        except Exception as e:
            logger.warning(f"Prometheus metrics recording failed: {e}")


class AlertManager:
    """Alert manager for SLA violations and performance issues."""

    def __init__(self, webhook_url: str | None = None):
        """Initialize alert manager.
        
        Args:
            webhook_url: Webhook URL for alert notifications
        """
        self.webhook_url = webhook_url
        self.alert_handlers: list[Callable] = []
        self.alert_history: deque[SLAViolation] = deque(maxlen=1000)
        self.alert_suppression: dict[str, float] = {}  # Alert type -> last sent time
        self.suppression_window = 300.0  # 5 minutes between same alert types

    def add_alert_handler(self, handler: Callable[[SLAViolation], None]) -> None:
        """Add custom alert handler.
        
        Args:
            handler: Alert handler function
        """
        self.alert_handlers.append(handler)

    async def send_alert(self, violation: SLAViolation) -> None:
        """Send alert for SLA violation.
        
        Args:
            violation: SLA violation to alert on
        """
        try:
            # Check alert suppression
            alert_key = f"{violation.violation_type}_{violation.measurement.stage.value}"
            current_time = time.time()

            if alert_key in self.alert_suppression:
                time_since_last = current_time - self.alert_suppression[alert_key]
                if time_since_last < self.suppression_window:
                    logger.debug(f"Alert suppressed: {alert_key}")
                    return

            # Record alert
            self.alert_history.append(violation)
            self.alert_suppression[alert_key] = current_time

            # Call custom handlers
            for handler in self.alert_handlers:
                try:
                    handler(violation)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")

            # Send webhook alert
            if self.webhook_url and REQUESTS_AVAILABLE:
                await self._send_webhook_alert(violation)

            logger.warning(
                f"SLA violation alert: {violation.violation_type} - "
                f"{violation.actual_ms:.2f}ms > {violation.threshold_ms:.2f}ms "
                f"({violation.measurement.stage.value})"
            )

        except Exception as e:
            logger.error(f"Alert sending failed: {e}")

    async def _send_webhook_alert(self, violation: SLAViolation) -> None:
        """Send webhook alert notification.
        
        Args:
            violation: SLA violation to send
        """
        if not self.webhook_url:
            return

        try:
            alert_data = {
                "timestamp": violation.timestamp,
                "severity": violation.severity.value,
                "violation_type": violation.violation_type,
                "stage": violation.measurement.stage.value,
                "camera_id": violation.measurement.camera_id,
                "threshold_ms": violation.threshold_ms,
                "actual_ms": violation.actual_ms,
                "context": violation.context,
            }

            # Send async webhook (non-blocking)
            asyncio.create_task(self._post_webhook(alert_data))

        except Exception as e:
            logger.warning(f"Webhook alert failed: {e}")

    async def _post_webhook(self, data: dict[str, Any]) -> None:
        """Post webhook data asynchronously.
        
        Args:
            data: Alert data to send
        """
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=10.0)
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Webhook returned status {response.status}")

        except ImportError:
            # Fall back to synchronous requests if aiohttp not available
            if REQUESTS_AVAILABLE:
                try:
                    requests.post(
                        self.webhook_url,
                        json=data,
                        timeout=10.0
                    )
                except Exception as e:
                    logger.warning(f"Synchronous webhook failed: {e}")
        except Exception as e:
            logger.warning(f"Async webhook failed: {e}")


class PerformanceRegressionDetector:
    """Detects performance regressions using statistical analysis."""

    def __init__(self, baseline_window_minutes: int = 60, regression_threshold: float = 0.2):
        """Initialize regression detector.
        
        Args:
            baseline_window_minutes: Baseline window in minutes
            regression_threshold: Regression threshold (20% = 0.2)
        """
        self.baseline_window = baseline_window_minutes * 60  # Convert to seconds
        self.regression_threshold = regression_threshold
        self.baseline_metrics: dict[PipelineStage, list[float]] = defaultdict(list)
        self.last_baseline_update = time.time()

    def update_baseline(self, measurements: list[LatencyMeasurement]) -> None:
        """Update baseline metrics with recent measurements.
        
        Args:
            measurements: Recent measurements for baseline
        """
        current_time = time.time()

        # Update baseline every hour
        if current_time - self.last_baseline_update < 3600:
            return

        # Group measurements by stage
        stage_latencies: dict[PipelineStage, list[float]] = defaultdict(list)

        for measurement in measurements:
            if current_time - measurement.timestamp <= self.baseline_window:
                stage_latencies[measurement.stage].append(measurement.latency_ms)

        # Update baseline with median values (robust to outliers)
        for stage, latencies in stage_latencies.items():
            if latencies:
                self.baseline_metrics[stage] = latencies[-100:]  # Keep last 100 samples

        self.last_baseline_update = current_time
        logger.debug(f"Updated performance baseline for {len(stage_latencies)} stages")

    def detect_regression(
        self,
        current_metrics: LatencyMetrics
    ) -> tuple[float, str] | None:
        """Detect performance regression.
        
        Args:
            current_metrics: Current performance metrics
            
        Returns:
            Optional[Tuple[float, str]]: (regression_factor, description) or None
        """
        stage = current_metrics.stage

        if stage not in self.baseline_metrics or not self.baseline_metrics[stage]:
            return None

        # Calculate baseline median
        baseline_median = statistics.median(self.baseline_metrics[stage])
        current_median = current_metrics.median_latency_ms

        # Calculate regression factor
        if baseline_median == 0:
            return None

        regression_factor = (current_median - baseline_median) / baseline_median

        # Check if regression exceeds threshold
        if regression_factor > self.regression_threshold:
            description = (
                f"Performance regression detected: {regression_factor:.1%} "
                f"increase in median latency ({baseline_median:.1f}ms -> "
                f"{current_median:.1f}ms)"
            )
            return regression_factor, description

        return None


class LatencyMonitor:
    """Comprehensive latency monitoring and SLA management system.
    
    Implements real-time latency tracking with <100ms SLA monitoring,
    performance regression detection, and automated alerting.
    """

    def __init__(self, config: LatencyMonitoringConfig):
        """Initialize latency monitor.
        
        Args:
            config: Latency monitoring configuration
        """
        self.config = config

        # Rolling windows for different time periods
        self.short_window = RollingWindow(config.monitoring_window_seconds)
        self.long_window = RollingWindow(config.trend_analysis_minutes * 60)

        # Monitoring components
        self.prometheus_exporter = PrometheusExporter()
        self.alert_manager = AlertManager(config.alert_webhook_url)
        self.regression_detector = PerformanceRegressionDetector(
            baseline_window_minutes=config.trend_analysis_minutes,
            regression_threshold=config.regression_threshold_percent / 100.0
        )

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: asyncio.Task | None = None
        self.active_requests: dict[str, dict[PipelineStage, float]] = {}

        # Statistics
        self.total_measurements = 0
        self.total_sla_violations = 0

        logger.info("LatencyMonitor initialized")

    async def start_monitoring(self) -> None:
        """Start latency monitoring with background tasks."""
        if self.is_monitoring:
            return

        self.is_monitoring = True

        # Start Prometheus server
        self.prometheus_exporter.start_server()

        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("Latency monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop latency monitoring."""
        self.is_monitoring = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Latency monitoring stopped")

    @asynccontextmanager
    async def track_end_to_end_latency(
        self,
        camera_id: str,
        request_id: str | None = None
    ):
        """Context manager for tracking end-to-end latency.
        
        Args:
            camera_id: Camera identifier
            request_id: Optional request identifier
        """
        request_id = request_id or f"{camera_id}_{int(time.time() * 1000000)}"
        start_time = time.perf_counter()

        # Initialize request tracking
        self.active_requests[request_id] = {PipelineStage.END_TO_END: start_time}

        try:
            yield request_id

        finally:
            # Record end-to-end latency
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            measurement = LatencyMeasurement(
                timestamp=time.time(),
                latency_ms=latency_ms,
                stage=PipelineStage.END_TO_END,
                camera_id=camera_id,
                request_id=request_id,
                metadata={"stages": list(self.active_requests.get(request_id, {}).keys())}
            )

            await self.record_measurement(measurement)

            # Cleanup request tracking
            self.active_requests.pop(request_id, None)

    async def record_measurement(self, measurement: LatencyMeasurement) -> None:
        """Record latency measurement.
        
        Args:
            measurement: Latency measurement to record
        """
        try:
            # Add to rolling windows
            self.short_window.add_measurement(measurement)
            self.long_window.add_measurement(measurement)

            # Export to Prometheus
            self.prometheus_exporter.record_measurement(measurement)

            # Update statistics
            self.total_measurements += 1

            # Check for SLA violation
            if measurement.is_sla_violation:
                self.total_sla_violations += 1
                await self._handle_sla_violation(measurement)

            # Log critical latencies
            if measurement.latency_ms > self.config.latency_sla_ms * 2:
                logger.warning(
                    f"Critical latency: {measurement.latency_ms:.2f}ms "
                    f"({measurement.stage.value}, {measurement.camera_id})"
                )

        except Exception as e:
            logger.error(f"Measurement recording failed: {e}")

    async def _handle_sla_violation(self, measurement: LatencyMeasurement) -> None:
        """Handle SLA violation.
        
        Args:
            measurement: Measurement that violated SLA
        """
        try:
            # Determine severity
            severity = AlertLevel.WARNING
            if measurement.latency_ms > self.config.latency_sla_ms * 2:
                severity = AlertLevel.CRITICAL
            elif measurement.latency_ms > self.config.latency_sla_ms * 3:
                severity = AlertLevel.EMERGENCY

            # Create violation record
            violation = SLAViolation(
                measurement=measurement,
                violation_type="latency_sla_violation",
                severity=severity,
                threshold_ms=self.config.latency_sla_ms,
                actual_ms=measurement.latency_ms,
                context={
                    "stage": measurement.stage.value,
                    "camera_id": measurement.camera_id,
                    "request_id": measurement.request_id,
                }
            )

            # Send alert if enabled
            if self.config.enable_sla_alerts:
                await self.alert_manager.send_alert(violation)

        except Exception as e:
            logger.error(f"SLA violation handling failed: {e}")

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for analysis and alerts."""
        while self.is_monitoring:
            try:
                await asyncio.sleep(self.config.monitoring_interval_seconds)

                if not self.is_monitoring:
                    break

                # Analyze current performance
                await self._analyze_performance()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

    async def _analyze_performance(self) -> None:
        """Analyze current performance and detect issues."""
        try:
            # Get current metrics for all stages
            for stage in PipelineStage:
                metrics = self.short_window.get_metrics(stage)
                if not metrics:
                    continue

                # Check SLA violation rate
                if (metrics.sla_violation_rate > self.config.sla_violation_threshold and
                    self.config.enable_sla_alerts):

                    violation = SLAViolation(
                        measurement=LatencyMeasurement(
                            timestamp=time.time(),
                            latency_ms=metrics.p95_latency_ms,
                            stage=stage,
                            camera_id="aggregate",
                            request_id="sla_analysis"
                        ),
                        violation_type="sla_violation_rate_exceeded",
                        severity=AlertLevel.WARNING,
                        threshold_ms=self.config.latency_sla_ms,
                        actual_ms=metrics.p95_latency_ms,
                        context={
                            "violation_rate": metrics.sla_violation_rate,
                            "threshold_rate": self.config.sla_violation_threshold,
                            "sample_count": metrics.sample_count,
                        }
                    )

                    await self.alert_manager.send_alert(violation)

                # Check for performance regression
                if self.config.regression_detection_enabled:
                    regression = self.regression_detector.detect_regression(metrics)
                    if regression:
                        regression_factor, description = regression

                        violation = SLAViolation(
                            measurement=LatencyMeasurement(
                                timestamp=time.time(),
                                latency_ms=metrics.median_latency_ms,
                                stage=stage,
                                camera_id="aggregate",
                                request_id="regression_analysis"
                            ),
                            violation_type="performance_regression",
                            severity=AlertLevel.WARNING,
                            threshold_ms=0.0,  # No threshold for regression
                            actual_ms=metrics.median_latency_ms,
                            context={
                                "regression_factor": regression_factor,
                                "description": description,
                            }
                        )

                        await self.alert_manager.send_alert(violation)

            # Update regression baseline
            all_measurements = list(self.long_window.measurements)
            self.regression_detector.update_baseline(all_measurements)

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")

    def get_comprehensive_metrics(self) -> dict[str, Any]:
        """Get comprehensive latency metrics.
        
        Returns:
            Dict[str, Any]: Complete latency monitoring metrics
        """
        metrics = {
            "monitoring_active": self.is_monitoring,
            "total_measurements": self.total_measurements,
            "total_sla_violations": self.total_sla_violations,
            "overall_sla_violation_rate": (
                (self.total_sla_violations / max(1, self.total_measurements)) * 100
            ),
            "active_requests": len(self.active_requests),
            "stages": {},
        }

        # Get metrics for each pipeline stage
        for stage in PipelineStage:
            short_metrics = self.short_window.get_metrics(stage)
            long_metrics = self.long_window.get_metrics(stage)

            stage_data = {
                "short_window": short_metrics.__dict__ if short_metrics else None,
                "long_window": long_metrics.__dict__ if long_metrics else None,
            }

            metrics["stages"][stage.value] = stage_data

        # Alert statistics
        metrics["alerts"] = {
            "total_alerts": len(self.alert_manager.alert_history),
            "recent_alerts": [
                {
                    "timestamp": alert.timestamp,
                    "severity": alert.severity.value,
                    "type": alert.violation_type,
                    "stage": alert.measurement.stage.value,
                }
                for alert in list(self.alert_manager.alert_history)[-10:]
            ],
        }

        return metrics


async def create_latency_monitor(
    config: LatencyMonitoringConfig
) -> LatencyMonitor:
    """Create and initialize latency monitor.
    
    Args:
        config: Latency monitoring configuration
        
    Returns:
        LatencyMonitor: Initialized latency monitor
    """
    monitor = LatencyMonitor(config)
    await monitor.start_monitoring()
    return monitor
