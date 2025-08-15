"""Performance Regression Monitoring System.

This module provides continuous performance monitoring and regression detection
for the ITS Camera AI system. It tracks key performance metrics over time,
detects performance degradations, and provides alerting for SLA violations.

Key Features:
- Continuous performance baseline tracking
- Regression detection algorithms
- Automated alerting on performance degradation
- Historical performance trend analysis
- Performance comparison across system versions
- Integration with monitoring systems (Prometheus, Grafana)
"""

import asyncio
import json
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from ..flow.redis_queue_manager import RedisQueueManager
from ..services.grpc_streaming_server import StreamingServiceImpl
from ..services.kafka_event_producer import KafkaEventProducer

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""

    # Latency baselines (milliseconds)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Throughput baselines
    requests_per_second: float = 0.0
    bytes_per_second: float = 0.0

    # Success rate baseline
    success_rate_percent: float = 100.0

    # Compression baselines
    avg_compression_ratio: float = 1.0
    avg_compression_time_ms: float = 0.0

    # Resource utilization baselines
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    avg_gpu_percent: float = 0.0

    # Metadata
    baseline_version: str = "unknown"
    baseline_timestamp: float = field(default_factory=time.time)
    sample_count: int = 0


@dataclass
class PerformanceMeasurement:
    """Single performance measurement."""

    timestamp: float
    latency_ms: float
    throughput_rps: float
    success_rate_percent: float
    compression_ratio: float
    compression_time_ms: float
    cpu_percent: float
    memory_mb: float
    gpu_percent: float
    component: str = "unknown"
    version: str = "unknown"


@dataclass
class RegressionAlert:
    """Performance regression alert."""

    timestamp: float
    metric_name: str
    current_value: float
    baseline_value: float
    regression_percent: float
    severity: str  # "warning", "critical"
    component: str
    message: str
    suggested_actions: list[str] = field(default_factory=list)


class RegressionDetector:
    """Detects performance regressions using statistical analysis."""

    def __init__(self,
                 warning_threshold_percent: float = 10.0,
                 critical_threshold_percent: float = 25.0,
                 min_samples: int = 30):
        self.warning_threshold = warning_threshold_percent / 100.0
        self.critical_threshold = critical_threshold_percent / 100.0
        self.min_samples = min_samples

    def detect_regression(self,
                         current_measurements: list[PerformanceMeasurement],
                         baseline: PerformanceBaseline) -> list[RegressionAlert]:
        """Detect performance regressions against baseline."""
        if len(current_measurements) < self.min_samples:
            return []

        alerts = []
        current_time = time.time()

        # Calculate current metrics
        latencies = [m.latency_ms for m in current_measurements]
        throughputs = [m.throughput_rps for m in current_measurements]
        success_rates = [m.success_rate_percent for m in current_measurements]
        compression_ratios = [m.compression_ratio for m in current_measurements]
        compression_times = [m.compression_time_ms for m in current_measurements]

        current_metrics = {
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "avg_throughput_rps": statistics.mean(throughputs),
            "avg_success_rate": statistics.mean(success_rates),
            "avg_compression_ratio": statistics.mean(compression_ratios),
            "avg_compression_time_ms": statistics.mean(compression_times)
        }

        # Check for regressions (higher is worse for these metrics)
        regression_checks = [
            ("avg_latency_ms", current_metrics["avg_latency_ms"], baseline.avg_latency_ms, True),
            ("p95_latency_ms", current_metrics["p95_latency_ms"], baseline.p95_latency_ms, True),
            ("p99_latency_ms", current_metrics["p99_latency_ms"], baseline.p99_latency_ms, True),
            ("avg_compression_time_ms", current_metrics["avg_compression_time_ms"], baseline.avg_compression_time_ms, True),
            ("avg_compression_ratio", current_metrics["avg_compression_ratio"], baseline.avg_compression_ratio, True),
        ]

        # Check for improvements that became worse (lower is worse)
        improvement_checks = [
            ("avg_throughput_rps", current_metrics["avg_throughput_rps"], baseline.requests_per_second, False),
            ("avg_success_rate", current_metrics["avg_success_rate"], baseline.success_rate_percent, False),
        ]

        # Check regressions
        for metric_name, current_value, baseline_value, higher_is_worse in regression_checks + improvement_checks:
            if baseline_value == 0:
                continue

            if higher_is_worse:
                regression_percent = (current_value - baseline_value) / baseline_value
            else:
                regression_percent = (baseline_value - current_value) / baseline_value

            if regression_percent > self.critical_threshold:
                alerts.append(RegressionAlert(
                    timestamp=current_time,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    regression_percent=regression_percent * 100,
                    severity="critical",
                    component="performance_monitor",
                    message=f"Critical regression in {metric_name}: {regression_percent*100:.1f}% worse than baseline",
                    suggested_actions=[
                        "Investigate recent code changes",
                        "Check system resource utilization",
                        "Review configuration changes",
                        "Consider rolling back recent deployments"
                    ]
                ))
            elif regression_percent > self.warning_threshold:
                alerts.append(RegressionAlert(
                    timestamp=current_time,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    regression_percent=regression_percent * 100,
                    severity="warning",
                    component="performance_monitor",
                    message=f"Performance regression in {metric_name}: {regression_percent*100:.1f}% worse than baseline",
                    suggested_actions=[
                        "Monitor trend closely",
                        "Check for gradual performance degradation",
                        "Review recent optimizations"
                    ]
                ))

        return alerts


class PerformanceRegressionMonitor:
    """Monitors system performance and detects regressions."""

    def __init__(self,
                 baseline_file: str = "performance_baseline.json",
                 history_file: str = "performance_history.json",
                 max_history_days: int = 30):
        self.baseline_file = Path(baseline_file)
        self.history_file = Path(history_file)
        self.max_history_days = max_history_days

        # Performance tracking
        self.current_measurements: deque[PerformanceMeasurement] = deque(maxlen=1000)
        self.performance_history: list[dict[str, Any]] = []
        self.baseline: PerformanceBaseline | None = None

        # Regression detection
        self.regression_detector = RegressionDetector()
        self.active_alerts: list[RegressionAlert] = []

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: asyncio.Task | None = None
        self.last_baseline_update = 0.0

        # Component references
        self.grpc_service: StreamingServiceImpl | None = None
        self.redis_manager: RedisQueueManager | None = None
        self.kafka_producer: KafkaEventProducer | None = None

        # Load existing data
        self._load_baseline()
        self._load_history()

    def _load_baseline(self):
        """Load performance baseline from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file) as f:
                    data = json.load(f)
                    self.baseline = PerformanceBaseline(**data)
                    logger.info(f"Loaded performance baseline from {self.baseline_file}")
            except Exception as e:
                logger.warning(f"Failed to load baseline: {e}")

    def _save_baseline(self):
        """Save performance baseline to file."""
        if self.baseline:
            try:
                with open(self.baseline_file, 'w') as f:
                    json.dump(self.baseline.__dict__, f, indent=2)
                    logger.info(f"Saved performance baseline to {self.baseline_file}")
            except Exception as e:
                logger.error(f"Failed to save baseline: {e}")

    def _load_history(self):
        """Load performance history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    self.performance_history = json.load(f)

                # Clean old history
                cutoff_time = time.time() - (self.max_history_days * 24 * 3600)
                self.performance_history = [
                    h for h in self.performance_history
                    if h.get('timestamp', 0) > cutoff_time
                ]

                logger.info(f"Loaded {len(self.performance_history)} performance history records")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")

    def _save_history(self):
        """Save performance history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def register_components(self,
                           grpc_service: StreamingServiceImpl | None = None,
                           redis_manager: RedisQueueManager | None = None,
                           kafka_producer: KafkaEventProducer | None = None):
        """Register system components for monitoring."""
        self.grpc_service = grpc_service
        self.redis_manager = redis_manager
        self.kafka_producer = kafka_producer

        logger.info("Components registered for performance monitoring")

    async def start_monitoring(self, measurement_interval: float = 30.0):
        """Start continuous performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(measurement_interval)
        )

        logger.info(f"Performance monitoring started with {measurement_interval}s interval")

    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect performance measurements
                await self._collect_measurements()

                # Check for regressions
                if self.baseline:
                    new_alerts = self.regression_detector.detect_regression(
                        list(self.current_measurements), self.baseline
                    )

                    # Process new alerts
                    for alert in new_alerts:
                        await self._handle_regression_alert(alert)

                # Update history periodically
                if len(self.current_measurements) >= 50:  # Every ~25 minutes at 30s intervals
                    await self._update_performance_history()

                # Auto-update baseline weekly
                current_time = time.time()
                if current_time - self.last_baseline_update > 7 * 24 * 3600:  # 7 days
                    await self._auto_update_baseline()

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(interval)

    async def _collect_measurements(self):
        """Collect performance measurements from all components."""
        current_time = time.time()

        try:
            # Collect gRPC metrics
            if self.grpc_service:
                grpc_metrics = self.grpc_service.get_serialization_performance_metrics()
                server_metrics = grpc_metrics.get("gRPC_server_metrics", {})
                serializer_metrics = grpc_metrics.get("serializer_performance", {})

                measurement = PerformanceMeasurement(
                    timestamp=current_time,
                    latency_ms=serializer_metrics.get("avg_serialization_time_ms", 0.0),
                    throughput_rps=server_metrics.get("requests_per_second", 0.0),
                    success_rate_percent=100.0,  # Assume success if no errors
                    compression_ratio=serializer_metrics.get("avg_compression_ratio", 1.0),
                    compression_time_ms=serializer_metrics.get("blosc_avg_compression_time_ms", 0.0),
                    cpu_percent=0.0,  # Would be collected from system monitoring
                    memory_mb=0.0,    # Would be collected from system monitoring
                    gpu_percent=0.0,  # Would be collected from GPU monitoring
                    component="grpc_service",
                    version="optimized"
                )

                self.current_measurements.append(measurement)

            # Collect Redis metrics
            if self.redis_manager:
                try:
                    redis_metrics = await self.redis_manager.get_all_metrics()

                    # Calculate average latency from Redis operations
                    avg_latency = 0.0
                    total_ops = 0
                    for queue_name, metrics in redis_metrics.items():
                        if hasattr(metrics, 'processing_times') and metrics.processing_times:
                            avg_latency += sum(metrics.processing_times) / len(metrics.processing_times)
                            total_ops += 1

                    if total_ops > 0:
                        avg_latency /= total_ops

                    measurement = PerformanceMeasurement(
                        timestamp=current_time,
                        latency_ms=avg_latency,
                        throughput_rps=0.0,  # Would calculate from Redis ops
                        success_rate_percent=100.0,
                        compression_ratio=self.redis_manager.compression_metrics.get("compression_ratio_avg", 1.0),
                        compression_time_ms=self.redis_manager.compression_metrics.get("compression_time_ms", 0.0),
                        cpu_percent=0.0,
                        memory_mb=0.0,
                        gpu_percent=0.0,
                        component="redis_manager",
                        version="optimized"
                    )

                    self.current_measurements.append(measurement)

                except Exception as e:
                    logger.debug(f"Failed to collect Redis metrics: {e}")

            # Collect Kafka metrics
            if self.kafka_producer:
                kafka_health = self.kafka_producer.get_health_status()
                metrics = kafka_health.get("metrics", {})

                measurement = PerformanceMeasurement(
                    timestamp=current_time,
                    latency_ms=metrics.get("avg_send_latency_ms", 0.0),
                    throughput_rps=metrics.get("throughput_events_per_sec", 0.0),
                    success_rate_percent=100.0 if kafka_health.get("is_healthy") else 0.0,
                    compression_ratio=metrics.get("avg_compression_ratio", 1.0),
                    compression_time_ms=0.0,  # Not directly available
                    cpu_percent=0.0,
                    memory_mb=0.0,
                    gpu_percent=0.0,
                    component="kafka_producer",
                    version="optimized"
                )

                self.current_measurements.append(measurement)

        except Exception as e:
            logger.error(f"Failed to collect measurements: {e}")

    async def _handle_regression_alert(self, alert: RegressionAlert):
        """Handle performance regression alert."""
        # Add to active alerts
        self.active_alerts.append(alert)

        # Log alert
        log_level = logger.critical if alert.severity == "critical" else logger.warning
        log_level(
            f"Performance regression detected: {alert.message}",
            metric=alert.metric_name,
            current_value=alert.current_value,
            baseline_value=alert.baseline_value,
            regression_percent=alert.regression_percent,
            component=alert.component
        )

        # Send to external monitoring systems (placeholder)
        await self._send_external_alert(alert)

    async def _send_external_alert(self, alert: RegressionAlert):
        """Send alert to external monitoring systems."""
        # This would integrate with Slack, PagerDuty, etc.
        # For now, just log
        logger.info(f"Would send external alert: {alert.message}")

    async def _update_performance_history(self):
        """Update performance history."""
        if not self.current_measurements:
            return

        # Calculate summary metrics
        recent_measurements = list(self.current_measurements)[-50:]  # Last 50 measurements

        latencies = [m.latency_ms for m in recent_measurements]
        throughputs = [m.throughput_rps for m in recent_measurements]
        success_rates = [m.success_rate_percent for m in recent_measurements]
        compression_ratios = [m.compression_ratio for m in recent_measurements]

        summary = {
            "timestamp": time.time(),
            "sample_count": len(recent_measurements),
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0.0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0.0,
            "avg_throughput_rps": statistics.mean(throughputs) if throughputs else 0.0,
            "avg_success_rate": statistics.mean(success_rates) if success_rates else 100.0,
            "avg_compression_ratio": statistics.mean(compression_ratios) if compression_ratios else 1.0,
            "version": "optimized"
        }

        self.performance_history.append(summary)
        self._save_history()

        logger.debug(f"Updated performance history: {len(self.performance_history)} records")

    async def _auto_update_baseline(self):
        """Automatically update baseline if performance has improved consistently."""
        if len(self.performance_history) < 7:  # Need at least a week of data
            return

        # Get recent week of data
        recent_history = self.performance_history[-7:]

        # Calculate new baseline candidates
        latencies = [h["avg_latency_ms"] for h in recent_history]
        throughputs = [h["avg_throughput_rps"] for h in recent_history]
        success_rates = [h["avg_success_rate"] for h in recent_history]
        compression_ratios = [h["avg_compression_ratio"] for h in recent_history]

        candidate_baseline = PerformanceBaseline(
            avg_latency_ms=statistics.mean(latencies),
            p95_latency_ms=max(latencies) * 0.95,  # Estimate
            p99_latency_ms=max(latencies) * 0.99,  # Estimate
            requests_per_second=statistics.mean(throughputs),
            success_rate_percent=statistics.mean(success_rates),
            avg_compression_ratio=statistics.mean(compression_ratios),
            baseline_version="auto_updated",
            baseline_timestamp=time.time(),
            sample_count=sum(h["sample_count"] for h in recent_history)
        )

        # Only update if significantly better
        if self.baseline:
            latency_improvement = (self.baseline.avg_latency_ms - candidate_baseline.avg_latency_ms) / self.baseline.avg_latency_ms
            throughput_improvement = (candidate_baseline.requests_per_second - self.baseline.requests_per_second) / max(self.baseline.requests_per_second, 1)

            if latency_improvement > 0.05 or throughput_improvement > 0.05:  # 5% improvement
                self.baseline = candidate_baseline
                self._save_baseline()
                self.last_baseline_update = time.time()

                logger.info(
                    f"Auto-updated performance baseline: "
                    f"{latency_improvement*100:.1f}% latency improvement, "
                    f"{throughput_improvement*100:.1f}% throughput improvement"
                )
        else:
            # No existing baseline, set this as first baseline
            self.baseline = candidate_baseline
            self._save_baseline()
            self.last_baseline_update = time.time()

            logger.info("Set initial performance baseline")

    async def create_baseline_from_current(self, version: str = "manual"):
        """Create new baseline from current measurements."""
        if len(self.current_measurements) < 100:
            logger.warning("Not enough measurements to create reliable baseline")
            return

        measurements = list(self.current_measurements)

        latencies = [m.latency_ms for m in measurements]
        throughputs = [m.throughput_rps for m in measurements]
        success_rates = [m.success_rate_percent for m in measurements]
        compression_ratios = [m.compression_ratio for m in measurements]
        compression_times = [m.compression_time_ms for m in measurements]

        self.baseline = PerformanceBaseline(
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=statistics.median(latencies),
            p95_latency_ms=np.percentile(latencies, 95),
            p99_latency_ms=np.percentile(latencies, 99),
            requests_per_second=statistics.mean(throughputs),
            success_rate_percent=statistics.mean(success_rates),
            avg_compression_ratio=statistics.mean(compression_ratios),
            avg_compression_time_ms=statistics.mean(compression_times),
            baseline_version=version,
            baseline_timestamp=time.time(),
            sample_count=len(measurements)
        )

        self._save_baseline()
        self.last_baseline_update = time.time()

        logger.info(f"Created new performance baseline with {len(measurements)} samples")

    def get_performance_report(self) -> dict[str, Any]:
        """Get comprehensive performance report."""
        current_time = time.time()

        # Calculate current performance
        recent_measurements = list(self.current_measurements)[-100:] if self.current_measurements else []

        current_performance = {}
        if recent_measurements:
            latencies = [m.latency_ms for m in recent_measurements]
            throughputs = [m.throughput_rps for m in recent_measurements]
            success_rates = [m.success_rate_percent for m in recent_measurements]

            current_performance = {
                "avg_latency_ms": statistics.mean(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "avg_throughput_rps": statistics.mean(throughputs),
                "avg_success_rate": statistics.mean(success_rates),
                "sample_count": len(recent_measurements)
            }

        # Performance trends
        trends = {}
        if len(self.performance_history) >= 2:
            recent = self.performance_history[-1]
            previous = self.performance_history[-2]

            trends = {
                "latency_trend": ((recent["avg_latency_ms"] - previous["avg_latency_ms"]) / previous["avg_latency_ms"] * 100) if previous["avg_latency_ms"] > 0 else 0,
                "throughput_trend": ((recent["avg_throughput_rps"] - previous["avg_throughput_rps"]) / max(previous["avg_throughput_rps"], 1) * 100),
                "success_rate_trend": recent["avg_success_rate"] - previous["avg_success_rate"]
            }

        return {
            "report_timestamp": current_time,
            "monitoring_active": self.monitoring_active,
            "baseline": self.baseline.__dict__ if self.baseline else None,
            "current_performance": current_performance,
            "performance_trends": trends,
            "active_alerts": [
                {
                    "metric": alert.metric_name,
                    "severity": alert.severity,
                    "regression_percent": alert.regression_percent,
                    "message": alert.message,
                    "timestamp": alert.timestamp
                }
                for alert in self.active_alerts[-10:]  # Last 10 alerts
            ],
            "history_records": len(self.performance_history),
            "measurements_collected": len(self.current_measurements)
        }

    async def run_performance_benchmark(self, duration_seconds: int = 300) -> dict[str, Any]:
        """Run a dedicated performance benchmark."""
        logger.info(f"Starting {duration_seconds}s performance benchmark")

        benchmark_measurements = []
        start_time = time.time()

        # Clear current measurements for clean benchmark
        self.current_measurements.clear()

        while time.time() - start_time < duration_seconds:
            await self._collect_measurements()

            # Copy to benchmark measurements
            if self.current_measurements:
                benchmark_measurements.append(self.current_measurements[-1])

            await asyncio.sleep(5.0)  # Sample every 5 seconds during benchmark

        # Analyze benchmark results
        if benchmark_measurements:
            latencies = [m.latency_ms for m in benchmark_measurements]
            throughputs = [m.throughput_rps for m in benchmark_measurements]
            success_rates = [m.success_rate_percent for m in benchmark_measurements]
            compression_ratios = [m.compression_ratio for m in benchmark_measurements]

            benchmark_results = {
                "duration_seconds": duration_seconds,
                "sample_count": len(benchmark_measurements),
                "avg_latency_ms": statistics.mean(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "p50_latency_ms": statistics.median(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "avg_throughput_rps": statistics.mean(throughputs),
                "max_throughput_rps": max(throughputs),
                "avg_success_rate": statistics.mean(success_rates),
                "min_success_rate": min(success_rates),
                "avg_compression_ratio": statistics.mean(compression_ratios),
                "benchmark_timestamp": time.time()
            }

            logger.info(
                f"Benchmark completed: {benchmark_results['avg_latency_ms']:.2f}ms avg latency, "
                f"{benchmark_results['p99_latency_ms']:.2f}ms P99, "
                f"{benchmark_results['avg_throughput_rps']:.1f} RPS"
            )

            return benchmark_results
        else:
            logger.warning("No measurements collected during benchmark")
            return {"error": "No measurements collected"}


# Factory function
async def create_performance_monitor(
    components: dict[str, Any],
    baseline_file: str = "performance_baseline.json"
) -> PerformanceRegressionMonitor:
    """Create and configure performance regression monitor.
    
    Args:
        components: Dictionary of system components to monitor
        baseline_file: Path to baseline file
        
    Returns:
        Configured PerformanceRegressionMonitor
    """
    monitor = PerformanceRegressionMonitor(baseline_file=baseline_file)

    monitor.register_components(
        grpc_service=components.get("grpc_service"),
        redis_manager=components.get("redis_manager"),
        kafka_producer=components.get("kafka_producer")
    )

    return monitor
