"""Model Performance Metrics Service for ITS Camera AI System.

This service provides comprehensive model performance tracking, drift detection,
and metrics aggregation for ML models in production. Integrates with Prometheus
for monitoring and alerting.

Key Features:
- Real-time inference metrics tracking
- Model drift detection using statistical methods
- Performance summary and trend analysis
- Prometheus metrics export
- Accuracy tracking with ground truth
- Multi-model performance comparison
- Alert generation for performance degradation

Performance Requirements:
- Metric collection: <1ms overhead per inference
- Drift detection: <10ms calculation time
- Update frequency: Real-time with 1s aggregation
- Memory usage: <100MB for 24h history
- Query response: <50ms for performance summaries
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

from ..core.config import Settings
from ..core.logging import get_logger
from ..repositories.analytics_repository import AnalyticsRepository
from ..services.analytics_dtos import DetectionResultDTO, TimeWindow
from ..services.cache import CacheService

logger = get_logger(__name__)


@dataclass
class ModelPerformanceSummary:
    """Comprehensive model performance summary."""
    model_name: str
    time_window: TimeWindow
    latency_p50: float
    latency_p95: float
    latency_p99: float
    avg_throughput_fps: float
    peak_throughput_fps: float
    total_inferences: int
    accuracy_metrics: dict[str, float]
    drift_score: float
    last_updated: datetime

    # Additional metrics
    error_rate: float = 0.0
    confidence_distribution: dict[str, int] = field(default_factory=dict)
    class_distribution: dict[str, int] = field(default_factory=dict)
    quality_score_avg: float = 0.0


@dataclass
class DriftDetectionResult:
    """Model drift detection result."""
    model_name: str
    drift_score: float
    drift_type: str  # "confidence", "class", "feature", "combined"
    severity: str    # "low", "medium", "high", "critical"
    baseline_date: datetime
    detection_date: datetime
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelAccuracyMetrics:
    """Model accuracy metrics when ground truth is available."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    confusion_matrix: dict[str, dict[str, int]]
    class_accuracies: dict[str, float]
    sample_count: int


class ModelMetricsService:
    """Track ML model performance metrics for monitoring and drift detection.
    
    Provides comprehensive model performance tracking with real-time metrics,
    drift detection, and alerting capabilities.
    """

    def __init__(
        self,
        analytics_repository: AnalyticsRepository,
        cache_service: CacheService,
        settings: Settings
    ):
        """Initialize model metrics service.
        
        Args:
            analytics_repository: Repository for analytics data access
            cache_service: Cache service for performance optimization
            settings: Application settings
        """
        self.analytics_repo = analytics_repository
        self.cache = cache_service
        self.settings = settings

        # Prometheus metrics
        self._init_prometheus_metrics()

        # Baseline distributions for drift detection
        self.baseline_distributions = {}
        self.drift_detection_window = 100  # Number of samples for drift detection

        # Performance tracking
        self.inference_history = {}  # model_name -> list of inference data
        self.max_history_size = 10000  # Maximum history per model

        # Drift detection configuration
        self.drift_config = {
            "confidence_threshold": settings.ml.drift_threshold if hasattr(settings, 'ml') else 0.15,
            "class_threshold": 0.2,
            "feature_threshold": 0.25,
            "min_samples": 50,
            "baseline_window_hours": 24
        }

        # Alert thresholds
        self.alert_thresholds = {
            "latency_p95_ms": 100.0,
            "accuracy_drop": 0.05,
            "drift_score": 0.3,
            "error_rate": 0.1
        }

        logger.info("ModelMetricsService initialized with Prometheus integration")

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Inference latency histogram
        self.inference_latency = Histogram(
            'ml_inference_latency_seconds',
            'Model inference latency in seconds',
            ['model_name', 'camera_id'],
            buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.0]
        )

        # Throughput gauge
        self.throughput_fps = Gauge(
            'ml_throughput_fps',
            'Model throughput in frames per second',
            ['model_name']
        )

        # Confidence distribution histogram
        self.confidence_histogram = Histogram(
            'ml_confidence_distribution',
            'Distribution of detection confidence scores',
            ['model_name', 'class_name'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

        # Drift score gauge
        self.drift_score = Gauge(
            'ml_drift_score',
            'Model drift detection score',
            ['model_name', 'drift_type']
        )

        # Accuracy gauge
        self.accuracy_gauge = Gauge(
            'ml_accuracy',
            'Model accuracy when ground truth available',
            ['model_name', 'metric_type']
        )

        # Error counters
        self.inference_errors = Counter(
            'ml_inference_errors_total',
            'Total number of inference errors',
            ['model_name', 'error_type']
        )

        # Detection counters
        self.detections_total = Counter(
            'ml_detections_total',
            'Total number of detections',
            ['model_name', 'class_name', 'camera_id']
        )

    async def track_inference(
        self,
        model_name: str,
        inference_time_ms: float,
        detections: list[DetectionResultDTO],
        frame_metadata: dict[str, Any],
        camera_id: str | None = None
    ) -> None:
        """Track inference metrics for a model.
        
        Args:
            model_name: Name of the ML model
            inference_time_ms: Inference time in milliseconds
            detections: List of detection results
            frame_metadata: Frame processing metadata
            camera_id: Camera identifier
        """
        try:
            start_time = time.time()

            # Extract camera ID
            camera_id = camera_id or frame_metadata.get("camera_id", "unknown")

            # Record Prometheus metrics
            self.inference_latency.labels(
                model_name=model_name,
                camera_id=camera_id
            ).observe(inference_time_ms / 1000.0)

            # Track confidence distribution
            for detection in detections:
                self.confidence_histogram.labels(
                    model_name=model_name,
                    class_name=detection.class_name
                ).observe(detection.confidence)

                # Count detections by class
                self.detections_total.labels(
                    model_name=model_name,
                    class_name=detection.class_name,
                    camera_id=camera_id
                ).inc()

            # Calculate and track throughput
            await self._update_throughput(model_name, inference_time_ms)

            # Store inference data for analysis
            inference_data = {
                "timestamp": datetime.now(UTC),
                "inference_time_ms": inference_time_ms,
                "detection_count": len(detections),
                "camera_id": camera_id,
                "detections": detections,
                "frame_metadata": frame_metadata
            }

            await self._store_inference_data(model_name, inference_data)

            # Check for drift (async to avoid blocking)
            asyncio.create_task(self._check_model_drift(model_name, detections))

            # Track processing overhead
            processing_overhead_ms = (time.time() - start_time) * 1000
            if processing_overhead_ms > 5.0:  # Log if overhead > 5ms
                logger.warning(
                    f"High metrics processing overhead: {processing_overhead_ms:.2f}ms "
                    f"for model {model_name}"
                )

        except Exception as e:
            logger.error(f"Failed to track inference for model {model_name}: {e}")
            self.inference_errors.labels(
                model_name=model_name,
                error_type="metrics_tracking"
            ).inc()

    async def _update_throughput(self, model_name: str, inference_time_ms: float) -> None:
        """Update throughput metrics for a model.
        
        Args:
            model_name: Model name
            inference_time_ms: Inference time in milliseconds
        """
        try:
            # Calculate instantaneous throughput
            if inference_time_ms > 0:
                instant_fps = 1000.0 / inference_time_ms

                # Get running average from cache
                cache_key = f"throughput_avg:{model_name}"
                cached_data = await self.cache.get_json(cache_key)

                if cached_data:
                    # Update running average
                    current_avg = cached_data.get("avg_fps", instant_fps)
                    sample_count = cached_data.get("sample_count", 0) + 1
                    new_avg = ((current_avg * (sample_count - 1)) + instant_fps) / sample_count

                    # Limit sample count to prevent overflow
                    sample_count = min(sample_count, 1000)
                else:
                    new_avg = instant_fps
                    sample_count = 1

                # Update Prometheus gauge
                self.throughput_fps.labels(model_name=model_name).set(new_avg)

                # Cache updated average
                await self.cache.set_json(
                    cache_key,
                    {"avg_fps": new_avg, "sample_count": sample_count},
                    ttl=3600  # 1 hour
                )

        except Exception as e:
            logger.warning(f"Throughput update failed for model {model_name}: {e}")

    async def _store_inference_data(
        self,
        model_name: str,
        inference_data: dict[str, Any]
    ) -> None:
        """Store inference data for analysis and drift detection.
        
        Args:
            model_name: Model name
            inference_data: Inference data to store
        """
        try:
            # Initialize history if needed
            if model_name not in self.inference_history:
                self.inference_history[model_name] = []

            # Add to history
            self.inference_history[model_name].append(inference_data)

            # Maintain history size limit
            if len(self.inference_history[model_name]) > self.max_history_size:
                self.inference_history[model_name] = self.inference_history[model_name][-self.max_history_size:]

            # Cache recent inference summary
            cache_key = f"model_inference_summary:{model_name}"
            summary = {
                "last_inference": inference_data["timestamp"].isoformat(),
                "total_inferences": len(self.inference_history[model_name]),
                "avg_inference_time_ms": np.mean([
                    d["inference_time_ms"]
                    for d in self.inference_history[model_name][-100:]  # Last 100 samples
                ]),
                "recent_detection_count": sum([
                    d["detection_count"]
                    for d in self.inference_history[model_name][-10:]  # Last 10 samples
                ])
            }

            await self.cache.set_json(cache_key, summary, ttl=300)  # 5 minutes

        except Exception as e:
            logger.warning(f"Failed to store inference data for model {model_name}: {e}")

    async def _check_model_drift(
        self,
        model_name: str,
        detections: list[DetectionResultDTO]
    ) -> DriftDetectionResult | None:
        """Check for model drift using statistical methods.
        
        Args:
            model_name: Model name
            detections: Current detections
            
        Returns:
            Drift detection result if drift detected, None otherwise
        """
        try:
            # Skip if insufficient samples
            if (model_name not in self.inference_history or
                len(self.inference_history[model_name]) < self.drift_config["min_samples"]):
                return None

            # Get baseline distribution
            baseline = await self._get_baseline_distribution(model_name)
            if not baseline:
                await self._initialize_baseline(model_name)
                return None

            # Calculate current distribution
            current_dist = self._calculate_current_distribution(detections)

            # Calculate different types of drift
            confidence_drift = self._calculate_confidence_drift(
                baseline.get("confidence_dist", {}),
                current_dist.get("confidence_dist", {})
            )

            class_drift = self._calculate_class_distribution_drift(
                baseline.get("class_dist", {}),
                current_dist.get("class_dist", {})
            )

            # Combined drift score
            combined_drift_score = (
                confidence_drift * 0.6 +
                class_drift * 0.4
            )

            # Update Prometheus metrics
            self.drift_score.labels(
                model_name=model_name,
                drift_type="confidence"
            ).set(confidence_drift)

            self.drift_score.labels(
                model_name=model_name,
                drift_type="class"
            ).set(class_drift)

            self.drift_score.labels(
                model_name=model_name,
                drift_type="combined"
            ).set(combined_drift_score)

            # Check if drift exceeds threshold
            if combined_drift_score > self.drift_config["confidence_threshold"]:
                drift_result = DriftDetectionResult(
                    model_name=model_name,
                    drift_score=combined_drift_score,
                    drift_type="combined",
                    severity=self._categorize_drift_severity(combined_drift_score),
                    baseline_date=datetime.fromisoformat(baseline.get("created_at", "1970-01-01")),
                    detection_date=datetime.now(UTC),
                    details={
                        "confidence_drift": confidence_drift,
                        "class_drift": class_drift,
                        "sample_count": len(detections),
                        "baseline_sample_count": baseline.get("sample_count", 0)
                    }
                )

                await self._handle_drift_detection(drift_result)
                return drift_result

            return None

        except Exception as e:
            logger.error(f"Drift detection failed for model {model_name}: {e}")
            return None

    def _calculate_confidence_drift(
        self,
        baseline_dist: dict[str, float],
        current_dist: dict[str, float]
    ) -> float:
        """Calculate confidence distribution drift using KL divergence.
        
        Args:
            baseline_dist: Baseline confidence distribution
            current_dist: Current confidence distribution
            
        Returns:
            Drift score (0.0-1.0)
        """
        try:
            if not baseline_dist or not current_dist:
                return 0.0

            # Ensure both distributions have same bins
            all_bins = set(baseline_dist.keys()) | set(current_dist.keys())

            baseline_probs = []
            current_probs = []

            for bin_key in sorted(all_bins):
                baseline_probs.append(baseline_dist.get(bin_key, 1e-10))
                current_probs.append(current_dist.get(bin_key, 1e-10))

            # Normalize to probabilities
            baseline_probs = np.array(baseline_probs)
            current_probs = np.array(current_probs)

            baseline_probs = baseline_probs / np.sum(baseline_probs)
            current_probs = current_probs / np.sum(current_probs)

            # Calculate KL divergence
            kl_div = np.sum(current_probs * np.log(
                (current_probs + 1e-10) / (baseline_probs + 1e-10)
            ))

            # Normalize to 0-1 range (empirical scaling)
            drift_score = min(1.0, kl_div / 2.0)

            return drift_score

        except Exception as e:
            logger.warning(f"Confidence drift calculation failed: {e}")
            return 0.0

    def _calculate_class_distribution_drift(
        self,
        baseline_dist: dict[str, int],
        current_dist: dict[str, int]
    ) -> float:
        """Calculate class distribution drift.
        
        Args:
            baseline_dist: Baseline class distribution
            current_dist: Current class distribution
            
        Returns:
            Drift score (0.0-1.0)
        """
        try:
            if not baseline_dist or not current_dist:
                return 0.0

            # Get all classes
            all_classes = set(baseline_dist.keys()) | set(current_dist.keys())

            # Calculate proportions
            baseline_total = sum(baseline_dist.values())
            current_total = sum(current_dist.values())

            if baseline_total == 0 or current_total == 0:
                return 0.0

            drift_sum = 0.0
            for class_name in all_classes:
                baseline_prop = baseline_dist.get(class_name, 0) / baseline_total
                current_prop = current_dist.get(class_name, 0) / current_total

                # Calculate absolute difference
                drift_sum += abs(baseline_prop - current_prop)

            # Normalize (max possible drift is 2.0)
            drift_score = min(1.0, drift_sum / 2.0)

            return drift_score

        except Exception as e:
            logger.warning(f"Class distribution drift calculation failed: {e}")
            return 0.0

    def _calculate_current_distribution(
        self, detections: list[DetectionResultDTO]
    ) -> dict[str, Any]:
        """Calculate current distribution from detections.
        
        Args:
            detections: Current detections
            
        Returns:
            Distribution dictionary
        """
        try:
            # Confidence distribution (binned)
            confidence_bins = {}
            for detection in detections:
                bin_key = f"{int(detection.confidence * 10) / 10:.1f}"  # 0.1 bin size
                confidence_bins[bin_key] = confidence_bins.get(bin_key, 0) + 1

            # Class distribution
            class_dist = {}
            for detection in detections:
                class_name = detection.class_name
                class_dist[class_name] = class_dist.get(class_name, 0) + 1

            return {
                "confidence_dist": confidence_bins,
                "class_dist": class_dist,
                "sample_count": len(detections),
                "timestamp": datetime.now(UTC).isoformat()
            }

        except Exception as e:
            logger.warning(f"Current distribution calculation failed: {e}")
            return {}

    async def _get_baseline_distribution(self, model_name: str) -> dict[str, Any] | None:
        """Get baseline distribution for drift detection.
        
        Args:
            model_name: Model name
            
        Returns:
            Baseline distribution or None if not available
        """
        try:
            cache_key = f"model_baseline:{model_name}"
            baseline = await self.cache.get_json(cache_key)

            if baseline:
                # Check if baseline is too old
                created_at = datetime.fromisoformat(baseline.get("created_at", "1970-01-01"))
                age_hours = (datetime.now(UTC) - created_at).total_seconds() / 3600

                if age_hours > self.drift_config["baseline_window_hours"]:
                    # Baseline too old, recreate
                    await self._initialize_baseline(model_name)
                    return None

                return baseline

            return None

        except Exception as e:
            logger.warning(f"Failed to get baseline for model {model_name}: {e}")
            return None

    async def _initialize_baseline(self, model_name: str) -> None:
        """Initialize baseline distribution from recent history.
        
        Args:
            model_name: Model name
        """
        try:
            if model_name not in self.inference_history:
                return

            # Use recent history for baseline
            recent_history = self.inference_history[model_name][-self.drift_detection_window:]

            if len(recent_history) < self.drift_config["min_samples"]:
                return

            # Aggregate distributions
            all_detections = []
            for inference_data in recent_history:
                all_detections.extend(inference_data.get("detections", []))

            baseline_dist = self._calculate_current_distribution(all_detections)
            baseline_dist["created_at"] = datetime.now(UTC).isoformat()

            # Cache baseline
            cache_key = f"model_baseline:{model_name}"
            await self.cache.set_json(
                cache_key,
                baseline_dist,
                ttl=self.drift_config["baseline_window_hours"] * 3600
            )

            logger.info(f"Initialized baseline for model {model_name} with {len(all_detections)} samples")

        except Exception as e:
            logger.error(f"Failed to initialize baseline for model {model_name}: {e}")

    def _categorize_drift_severity(self, drift_score: float) -> str:
        """Categorize drift severity based on score.
        
        Args:
            drift_score: Drift score (0.0-1.0)
            
        Returns:
            Severity category
        """
        if drift_score >= 0.5:
            return "critical"
        elif drift_score >= 0.3:
            return "high"
        elif drift_score >= 0.15:
            return "medium"
        else:
            return "low"

    async def _handle_drift_detection(self, drift_result: DriftDetectionResult) -> None:
        """Handle detected model drift.
        
        Args:
            drift_result: Drift detection result
        """
        try:
            # Log drift detection
            logger.warning(
                f"Model drift detected for {drift_result.model_name}: "
                f"score={drift_result.drift_score:.3f}, severity={drift_result.severity}"
            )

            # Cache drift alert
            cache_key = f"model_drift_alert:{drift_result.model_name}"
            alert_data = {
                "model_name": drift_result.model_name,
                "drift_score": drift_result.drift_score,
                "severity": drift_result.severity,
                "detection_date": drift_result.detection_date.isoformat(),
                "details": drift_result.details
            }

            await self.cache.set_json(cache_key, alert_data, ttl=3600)  # 1 hour

            # Trigger alert if severity is high or critical
            if drift_result.severity in ["high", "critical"]:
                await self._trigger_drift_alert(drift_result)

        except Exception as e:
            logger.error(f"Failed to handle drift detection: {e}")

    async def _trigger_drift_alert(self, drift_result: DriftDetectionResult) -> None:
        """Trigger alert for model drift.
        
        Args:
            drift_result: Drift detection result
        """
        # This would integrate with alerting system (Slack, email, etc.)
        logger.error(
            f"ALERT: Model drift detected for {drift_result.model_name} "
            f"(severity: {drift_result.severity}, score: {drift_result.drift_score:.3f})"
        )

        # Could implement additional alerting logic here
        pass

    async def get_performance_summary(
        self,
        model_name: str,
        time_window: TimeWindow = TimeWindow.ONE_HOUR
    ) -> ModelPerformanceSummary:
        """Get comprehensive model performance summary.
        
        Args:
            model_name: Model name
            time_window: Time window for analysis
            
        Returns:
            Model performance summary
        """
        try:
            # Calculate time range
            end_time = datetime.now(UTC)
            if time_window == TimeWindow.ONE_HOUR:
                start_time = end_time - timedelta(hours=1)
            elif time_window == TimeWindow.ONE_DAY:
                start_time = end_time - timedelta(days=1)
            elif time_window == TimeWindow.ONE_WEEK:
                start_time = end_time - timedelta(weeks=1)
            else:
                start_time = end_time - timedelta(hours=1)

            # Get inference history in time window
            if model_name not in self.inference_history:
                return self._create_empty_summary(model_name, time_window)

            window_history = [
                h for h in self.inference_history[model_name]
                if start_time <= h["timestamp"] <= end_time
            ]

            if not window_history:
                return self._create_empty_summary(model_name, time_window)

            # Calculate latency percentiles
            latencies = [h["inference_time_ms"] for h in window_history]
            latency_p50 = np.percentile(latencies, 50)
            latency_p95 = np.percentile(latencies, 95)
            latency_p99 = np.percentile(latencies, 99)

            # Calculate throughput
            total_time_hours = (end_time - start_time).total_seconds() / 3600
            total_inferences = len(window_history)
            avg_throughput_fps = total_inferences / max(total_time_hours * 3600, 1)

            # Peak throughput (max in any 1-minute window)
            peak_throughput_fps = self._calculate_peak_throughput(window_history)

            # Get current drift score
            current_drift = 0.0
            try:
                cache_key = f"model_drift_alert:{model_name}"
                drift_data = await self.cache.get_json(cache_key)
                if drift_data:
                    current_drift = drift_data.get("drift_score", 0.0)
            except:
                pass

            # Calculate confidence and class distributions
            all_detections = []
            for h in window_history:
                all_detections.extend(h.get("detections", []))

            confidence_dist = {}
            class_dist = {}
            quality_scores = []

            for detection in all_detections:
                # Confidence distribution
                conf_bin = f"{int(detection.confidence * 10) / 10:.1f}"
                confidence_dist[conf_bin] = confidence_dist.get(conf_bin, 0) + 1

                # Class distribution
                class_dist[detection.class_name] = class_dist.get(detection.class_name, 0) + 1

                # Quality scores
                quality_scores.append(detection.detection_quality)

            avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0

            # Create summary
            summary = ModelPerformanceSummary(
                model_name=model_name,
                time_window=time_window,
                latency_p50=latency_p50,
                latency_p95=latency_p95,
                latency_p99=latency_p99,
                avg_throughput_fps=avg_throughput_fps,
                peak_throughput_fps=peak_throughput_fps,
                total_inferences=total_inferences,
                accuracy_metrics={},  # Would be populated with ground truth data
                drift_score=current_drift,
                last_updated=datetime.now(UTC),
                error_rate=0.0,  # Would calculate from error tracking
                confidence_distribution=confidence_dist,
                class_distribution=class_dist,
                quality_score_avg=avg_quality_score
            )

            return summary

        except Exception as e:
            logger.error(f"Failed to get performance summary for model {model_name}: {e}")
            return self._create_empty_summary(model_name, time_window)

    def _create_empty_summary(
        self, model_name: str, time_window: TimeWindow
    ) -> ModelPerformanceSummary:
        """Create empty performance summary.
        
        Args:
            model_name: Model name
            time_window: Time window
            
        Returns:
            Empty performance summary
        """
        return ModelPerformanceSummary(
            model_name=model_name,
            time_window=time_window,
            latency_p50=0.0,
            latency_p95=0.0,
            latency_p99=0.0,
            avg_throughput_fps=0.0,
            peak_throughput_fps=0.0,
            total_inferences=0,
            accuracy_metrics={},
            drift_score=0.0,
            last_updated=datetime.now(UTC)
        )

    def _calculate_peak_throughput(self, history: list[dict[str, Any]]) -> float:
        """Calculate peak throughput in any 1-minute window.
        
        Args:
            history: Inference history
            
        Returns:
            Peak throughput in FPS
        """
        try:
            if len(history) < 2:
                return 0.0

            # Sort by timestamp
            sorted_history = sorted(history, key=lambda x: x["timestamp"])

            max_throughput = 0.0
            window_size = timedelta(minutes=1)

            for i in range(len(sorted_history)):
                window_start = sorted_history[i]["timestamp"]
                window_end = window_start + window_size

                # Count inferences in window
                window_count = sum(
                    1 for h in sorted_history[i:]
                    if h["timestamp"] <= window_end
                )

                # Calculate FPS for this window
                window_fps = window_count / 60.0  # 60 seconds
                max_throughput = max(max_throughput, window_fps)

            return max_throughput

        except Exception:
            return 0.0

    async def get_model_comparison(
        self, model_names: list[str], time_window: TimeWindow = TimeWindow.ONE_HOUR
    ) -> dict[str, ModelPerformanceSummary]:
        """Get performance comparison for multiple models.
        
        Args:
            model_names: List of model names
            time_window: Time window for analysis
            
        Returns:
            Dictionary mapping model names to performance summaries
        """
        try:
            tasks = [
                self.get_performance_summary(model_name, time_window)
                for model_name in model_names
            ]

            summaries = await asyncio.gather(*tasks, return_exceptions=True)

            results = {}
            for model_name, summary in zip(model_names, summaries, strict=False):
                if isinstance(summary, Exception):
                    logger.error(f"Failed to get summary for model {model_name}: {summary}")
                    results[model_name] = self._create_empty_summary(model_name, time_window)
                else:
                    results[model_name] = summary

            return results

        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {
                name: self._create_empty_summary(name, time_window)
                for name in model_names
            }

    def get_prometheus_metrics(self) -> dict[str, Any]:
        """Get current Prometheus metrics values.
        
        Returns:
            Dictionary of current metric values
        """
        try:
            # This would collect current values from Prometheus metrics
            # For now, return summary of tracked metrics
            metrics = {
                "models_tracked": len(self.inference_history),
                "total_inferences": sum(
                    len(history) for history in self.inference_history.values()
                ),
                "drift_alerts_active": 0,  # Would count active drift alerts
                "performance_overhead_ms": 0.0  # Average metrics collection overhead
            }

            return metrics

        except Exception as e:
            logger.error(f"Failed to get Prometheus metrics: {e}")
            return {}

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of metrics service.
        
        Returns:
            Health status dictionary
        """
        try:
            # Calculate health indicators
            total_models = len(self.inference_history)
            total_history = sum(len(h) for h in self.inference_history.values())

            # Check for recent activity
            recent_activity = False
            if self.inference_history:
                latest_timestamp = max(
                    max(h, key=lambda x: x["timestamp"])["timestamp"]
                    for h in self.inference_history.values()
                    if h
                )
                recent_activity = (datetime.now(UTC) - latest_timestamp).total_seconds() < 300  # 5 minutes

            health_score = 1.0
            if not recent_activity:
                health_score *= 0.5
            if total_models == 0:
                health_score *= 0.3

            return {
                "status": "healthy" if health_score > 0.7 else "degraded",
                "health_score": health_score,
                "models_tracked": total_models,
                "total_history_size": total_history,
                "recent_activity": recent_activity,
                "memory_usage_mb": sum(
                    len(str(h)) for h in self.inference_history.values()
                ) / (1024 * 1024),  # Rough estimate
                "drift_detection_enabled": True,
                "prometheus_integration": True
            }

        except Exception as e:
            logger.error(f"Health status check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
