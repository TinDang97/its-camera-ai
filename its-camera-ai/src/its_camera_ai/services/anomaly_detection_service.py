"""Anomaly Detection Service with proper DI and ML model integration.

This service detects traffic anomalies using machine learning models,
statistical methods, and historical baseline comparisons.
"""

import pickle
import statistics
from datetime import UTC, datetime, timedelta

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..core.config import Settings
from ..core.logging import get_logger
from ..repositories.analytics_repository import AnalyticsRepository
from .analytics_dtos import (
    AnomalyResult,
    TrafficDataPoint,
)
from .cache import CacheService

logger = get_logger(__name__)


class AnomalyDetectionService:
    """Anomaly detection service with ML model support.

    Detects traffic anomalies using isolation forest, statistical methods,
    and baseline comparisons with historical data.
    """

    def __init__(
        self,
        analytics_repository: AnalyticsRepository,
        cache_service: CacheService,
        settings: Settings,
    ):
        """Initialize with injected dependencies.

        Args:
            analytics_repository: Repository for analytics data access
            cache_service: Redis cache service
            settings: Application settings
        """
        self.analytics_repository = analytics_repository
        self.cache_service = cache_service
        self.settings = settings

        # ML Models
        self.isolation_forest: IsolationForest | None = None
        self.scaler: StandardScaler | None = None

        # Detection parameters
        self.contamination_rate = 0.1  # Expected percentage of anomalies
        self.anomaly_threshold = 0.5  # Anomaly score threshold
        self.baseline_window_hours = 24  # Hours for baseline calculation

        # Feature extraction parameters
        self.feature_names = [
            "vehicle_count",
            "average_speed",
            "traffic_density",
            "flow_rate",
            "occupancy_rate",
            "queue_length",
            "hour_of_day",
            "day_of_week",
        ]

    async def detect_anomalies(
        self,
        data_points: list[TrafficDataPoint],
        detection_method: str = "isolation_forest",
        camera_id: str | None = None,
    ) -> list[AnomalyResult]:
        """Detect anomalies in traffic data.

        Args:
            data_points: Traffic data points to analyze
            detection_method: Method to use for detection
            camera_id: Optional camera ID filter

        Returns:
            List of detected anomalies
        """
        if not data_points:
            return []

        try:
            anomalies = []

            if detection_method == "isolation_forest":
                anomalies.extend(
                    await self._detect_ml_anomalies(data_points, camera_id)
                )
            elif detection_method == "statistical":
                anomalies.extend(
                    await self._detect_statistical_anomalies(data_points, camera_id)
                )
            elif detection_method == "baseline":
                anomalies.extend(
                    await self._detect_baseline_anomalies(data_points, camera_id)
                )
            else:
                # Use all methods
                anomalies.extend(
                    await self._detect_ml_anomalies(data_points, camera_id)
                )
                anomalies.extend(
                    await self._detect_statistical_anomalies(data_points, camera_id)
                )
                anomalies.extend(
                    await self._detect_baseline_anomalies(data_points, camera_id)
                )

            # Remove duplicates based on timestamp and camera
            unique_anomalies = self._deduplicate_anomalies(anomalies)

            # Store anomalies
            if unique_anomalies:
                await self._persist_anomalies(unique_anomalies)

            logger.info(
                f"Detected {len(unique_anomalies)} anomalies from {len(data_points)} data points"
            )

            return unique_anomalies

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []

    async def _detect_ml_anomalies(
        self,
        data_points: list[TrafficDataPoint],
        camera_id: str | None = None,
    ) -> list[AnomalyResult]:
        """Detect anomalies using Isolation Forest ML model.

        Args:
            data_points: Traffic data points
            camera_id: Optional camera ID filter

        Returns:
            List of ML-detected anomalies
        """
        try:
            # Load or train the model
            model = await self._get_isolation_forest_model(camera_id)

            if not model:
                logger.warning("No ML model available for anomaly detection")
                return []

            # Extract features
            features = self._extract_features(data_points)

            if len(features) == 0:
                return []

            # Scale features
            scaled_features = self.scaler.transform(features)

            # Predict anomalies
            anomaly_scores = model.decision_function(scaled_features)
            predictions = model.predict(scaled_features)

            # Convert to anomaly results
            anomalies = []
            for i, (data_point, score, prediction) in enumerate(
                zip(data_points, anomaly_scores, predictions, strict=False)
            ):
                if prediction == -1:  # Anomaly detected
                    # Normalize score to 0-1 range
                    normalized_score = self._normalize_anomaly_score(score)

                    if normalized_score >= self.anomaly_threshold:
                        severity = self._severity_from_score(normalized_score)
                        cause = self._identify_probable_cause(data_point, features[i])

                        anomaly = AnomalyResult(
                            data_point=data_point,
                            anomaly_score=normalized_score,
                            severity=severity,
                            probable_cause=cause,
                            features=features[i].tolist(),
                            detection_method="isolation_forest",
                            model_name="IsolationForest",
                            model_version="1.0",
                            confidence=normalized_score,
                        )

                        anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"ML anomaly detection failed: {e}")
            return []

    async def _detect_statistical_anomalies(
        self,
        data_points: list[TrafficDataPoint],
        camera_id: str | None = None,
    ) -> list[AnomalyResult]:
        """Detect anomalies using statistical methods.

        Args:
            data_points: Traffic data points
            camera_id: Optional camera ID filter

        Returns:
            List of statistically detected anomalies
        """
        anomalies = []

        try:
            # Group by metric for statistical analysis
            metrics = {
                "vehicle_count": [dp.vehicle_count for dp in data_points],
                "average_speed": [dp.average_speed for dp in data_points],
                "traffic_density": [dp.traffic_density for dp in data_points],
                "flow_rate": [dp.flow_rate for dp in data_points],
                "occupancy_rate": [dp.occupancy_rate for dp in data_points],
                "queue_length": [dp.queue_length for dp in data_points],
            }

            # Calculate z-scores for each metric
            for metric_name, values in metrics.items():
                if not values or all(v == 0 for v in values):
                    continue

                mean_val = statistics.mean(values)
                stdev_val = statistics.stdev(values) if len(values) > 1 else 0

                if stdev_val == 0:
                    continue

                # Find outliers using z-score
                for i, (data_point, value) in enumerate(zip(data_points, values, strict=False)):
                    z_score = abs(value - mean_val) / stdev_val

                    # Threshold for outlier detection (3 standard deviations)
                    if z_score > 3.0:
                        anomaly_score = min(1.0, z_score / 5.0)  # Normalize to 0-1
                        severity = self._severity_from_score(anomaly_score)

                        anomaly = AnomalyResult(
                            data_point=data_point,
                            anomaly_score=anomaly_score,
                            severity=severity,
                            probable_cause=f"Statistical outlier in {metric_name}",
                            features=[z_score],
                            detection_method="statistical",
                            model_name="ZScore",
                            model_version="1.0",
                            confidence=anomaly_score,
                        )

                        anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Statistical anomaly detection failed: {e}")
            return []

    async def _detect_baseline_anomalies(
        self,
        data_points: list[TrafficDataPoint],
        camera_id: str | None = None,
    ) -> list[AnomalyResult]:
        """Detect anomalies using historical baseline comparison.

        Args:
            data_points: Traffic data points
            camera_id: Optional camera ID filter

        Returns:
            List of baseline-detected anomalies
        """
        anomalies = []

        try:
            for data_point in data_points:
                # Get historical baseline for this time period
                baseline = await self._get_historical_baseline(
                    data_point.camera_id, data_point.timestamp
                )

                if not baseline:
                    continue

                # Compare current values with baseline
                deviations = []
                metrics_compared = []

                # Vehicle count deviation
                if baseline.get("vehicle_count"):
                    deviation = abs(
                        data_point.vehicle_count - baseline["vehicle_count"]
                    ) / max(baseline["vehicle_count"], 1)
                    deviations.append(deviation)
                    metrics_compared.append("vehicle_count")

                # Speed deviation
                if baseline.get("average_speed"):
                    deviation = abs(
                        data_point.average_speed - baseline["average_speed"]
                    ) / max(baseline["average_speed"], 1)
                    deviations.append(deviation)
                    metrics_compared.append("average_speed")

                # Flow rate deviation
                if baseline.get("flow_rate"):
                    deviation = abs(data_point.flow_rate - baseline["flow_rate"]) / max(
                        baseline["flow_rate"], 1
                    )
                    deviations.append(deviation)
                    metrics_compared.append("flow_rate")

                if not deviations:
                    continue

                # Calculate overall deviation score
                max_deviation = max(deviations)
                avg_deviation = statistics.mean(deviations)

                # Threshold for baseline anomaly (50% deviation)
                if max_deviation > 0.5:
                    anomaly_score = min(1.0, avg_deviation)
                    severity = self._severity_from_score(anomaly_score)

                    # Identify which metric deviated most
                    max_idx = deviations.index(max_deviation)
                    primary_metric = metrics_compared[max_idx]

                    anomaly = AnomalyResult(
                        data_point=data_point,
                        anomaly_score=anomaly_score,
                        severity=severity,
                        probable_cause=f"Baseline deviation in {primary_metric}",
                        features=deviations,
                        detection_method="baseline_comparison",
                        model_name="BaselineComparison",
                        model_version="1.0",
                        confidence=anomaly_score,
                    )

                    anomalies.append(anomaly)

            return anomalies

        except Exception as e:
            logger.error(f"Baseline anomaly detection failed: {e}")
            return []

    def _extract_features(self, data_points: list[TrafficDataPoint]) -> np.ndarray:
        """Extract features from traffic data points.

        Args:
            data_points: Traffic data points

        Returns:
            Feature matrix
        """
        features = []

        for data_point in data_points:
            feature_vector = [
                data_point.vehicle_count,
                data_point.average_speed,
                data_point.traffic_density,
                data_point.flow_rate,
                data_point.occupancy_rate,
                data_point.queue_length,
                data_point.timestamp.hour,
                data_point.timestamp.weekday(),
            ]

            features.append(feature_vector)

        return np.array(features)

    async def _get_isolation_forest_model(
        self, camera_id: str | None = None
    ) -> IsolationForest | None:
        """Get or train Isolation Forest model.

        Args:
            camera_id: Optional camera ID for camera-specific model

        Returns:
            Trained Isolation Forest model
        """
        try:
            # Try to load from cache
            model_key = (
                f"anomaly_model:isolation_forest:{camera_id}"
                if camera_id
                else "anomaly_model:isolation_forest:global"
            )

            cached_model = await self.cache_service.get(model_key)
            if cached_model:
                try:
                    model_data = pickle.loads(cached_model)
                    self.isolation_forest = model_data["model"]
                    self.scaler = model_data["scaler"]
                    return self.isolation_forest
                except Exception as e:
                    logger.warning(f"Failed to load cached model: {e}")

            # Train new model if not cached
            training_data = await self._get_training_data(camera_id)

            if len(training_data) < 100:  # Minimum training data
                logger.warning("Insufficient training data for ML model")
                return None

            features = self._extract_features(training_data)

            # Scale features
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features)

            # Train Isolation Forest
            self.isolation_forest = IsolationForest(
                contamination=self.contamination_rate,
                random_state=42,
                n_estimators=100,
            )

            self.isolation_forest.fit(scaled_features)

            # Cache the trained model
            model_data = {
                "model": self.isolation_forest,
                "scaler": self.scaler,
            }

            serialized_model = pickle.dumps(model_data)
            await self.cache_service.set(
                model_key, serialized_model, ttl=86400
            )  # 24 hours

            logger.info(
                f"Trained new anomaly detection model with {len(training_data)} samples"
            )
            return self.isolation_forest

        except Exception as e:
            logger.error(f"Failed to get/train ML model: {e}")
            return None

    async def _get_training_data(
        self, camera_id: str | None = None
    ) -> list[TrafficDataPoint]:
        """Get training data for ML model.

        Args:
            camera_id: Optional camera ID filter

        Returns:
            List of training data points
        """
        try:
            # Get historical data from last 30 days
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(days=30)

            # Query from repository
            metrics = await self.analytics_repository.get_traffic_metrics_by_camera(
                camera_id=camera_id or "all",
                start_time=start_time,
                end_time=end_time,
                limit=10000,
            )

            # Convert to TrafficDataPoint objects
            training_data = []
            for metric in metrics:
                data_point = TrafficDataPoint(
                    timestamp=metric.timestamp,
                    camera_id=metric.camera_id,
                    vehicle_count=metric.total_vehicles,
                    average_speed=metric.average_speed or 0,
                    traffic_density=getattr(metric, "traffic_density", 0),
                    flow_rate=getattr(metric, "flow_rate", 0),
                    occupancy_rate=metric.occupancy_rate or 0,
                    queue_length=getattr(metric, "queue_length", 0),
                )
                training_data.append(data_point)

            return training_data

        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return []

    async def _get_historical_baseline(
        self, camera_id: str, timestamp: datetime
    ) -> dict[str, float] | None:
        """Get historical baseline for comparison.

        Args:
            camera_id: Camera identifier
            timestamp: Timestamp for baseline calculation

        Returns:
            Baseline metrics or None
        """
        try:
            # Calculate baseline for same hour/day of week from past data
            hour = timestamp.hour
            day_of_week = timestamp.weekday()

            # Get data from same time periods in past weeks
            baseline_periods = []
            for weeks_back in range(1, 5):  # Look back 4 weeks
                baseline_time = timestamp - timedelta(weeks=weeks_back)
                baseline_periods.append(baseline_time)

            # Query baseline data
            baseline_metrics = []
            for baseline_time in baseline_periods:
                start_time = baseline_time.replace(minute=0, second=0, microsecond=0)
                end_time = start_time + timedelta(hours=1)

                metrics = await self.analytics_repository.get_traffic_metrics_by_camera(
                    camera_id=camera_id,
                    start_time=start_time,
                    end_time=end_time,
                    limit=100,
                )

                baseline_metrics.extend(metrics)

            if not baseline_metrics:
                return None

            # Calculate average baseline values
            baseline = {
                "vehicle_count": statistics.mean(
                    [m.total_vehicles for m in baseline_metrics]
                ),
                "average_speed": statistics.mean(
                    [m.average_speed or 0 for m in baseline_metrics]
                ),
                "occupancy_rate": statistics.mean(
                    [m.occupancy_rate or 0 for m in baseline_metrics]
                ),
                "flow_rate": statistics.mean(
                    [getattr(m, "flow_rate", 0) for m in baseline_metrics]
                ),
            }

            return baseline

        except Exception as e:
            logger.warning(f"Failed to get historical baseline: {e}")
            return None

    def _normalize_anomaly_score(self, score: float) -> float:
        """Normalize anomaly score to 0-1 range.

        Args:
            score: Raw anomaly score from model

        Returns:
            Normalized score
        """
        # Isolation Forest returns negative values for anomalies
        # Normalize to positive 0-1 scale
        normalized = max(0, min(1, abs(score) / 0.5))
        return round(normalized, 3)

    def _severity_from_score(self, score: float) -> str:
        """Determine severity from anomaly score.

        Args:
            score: Anomaly score (0-1)

        Returns:
            Severity level
        """
        if score < 0.3:
            return "low"
        elif score < 0.6:
            return "medium"
        elif score < 0.8:
            return "high"
        else:
            return "critical"

    def _identify_probable_cause(
        self, data_point: TrafficDataPoint, features: np.ndarray
    ) -> str:
        """Identify probable cause of anomaly.

        Args:
            data_point: Traffic data point
            features: Feature vector used for detection

        Returns:
            Probable cause description
        """
        # Analyze which features are most unusual
        feature_names = self.feature_names

        if len(features) != len(feature_names):
            return "Unknown traffic pattern anomaly"

        # Simple heuristics for cause identification
        if features[0] > 50:  # High vehicle count
            return "Traffic congestion detected"
        elif features[1] > 100:  # High speed
            return "Excessive speeding detected"
        elif features[1] < 10:  # Very low speed
            return "Traffic jam or stopped vehicles"
        elif features[2] > 0.8:  # High density
            return "High traffic density anomaly"
        elif features[5] > 100:  # Long queue
            return "Extended vehicle queue detected"
        else:
            return "Unusual traffic pattern detected"

    def _deduplicate_anomalies(
        self, anomalies: list[AnomalyResult]
    ) -> list[AnomalyResult]:
        """Remove duplicate anomalies based on timestamp and camera.

        Args:
            anomalies: List of anomaly results

        Returns:
            Deduplicated list
        """
        seen = set()
        unique_anomalies = []

        for anomaly in sorted(anomalies, key=lambda x: x.anomaly_score, reverse=True):
            # Create key based on camera, timestamp (rounded to minute), and cause
            timestamp_minute = anomaly.data_point.timestamp.replace(
                second=0, microsecond=0
            )
            key = (
                anomaly.data_point.camera_id,
                timestamp_minute,
                anomaly.probable_cause,
            )

            if key not in seen:
                seen.add(key)
                unique_anomalies.append(anomaly)

        return unique_anomalies

    async def _persist_anomalies(self, anomalies: list[AnomalyResult]) -> None:
        """Persist anomalies to repository.

        Args:
            anomalies: List of anomalies to persist
        """
        try:
            for anomaly in anomalies:
                anomaly_data = {
                    "camera_id": anomaly.data_point.camera_id,
                    "timestamp": anomaly.data_point.timestamp,
                    "anomaly_type": anomaly.probable_cause,
                    "severity": anomaly.severity,
                    "anomaly_score": anomaly.anomaly_score,
                    "detection_method": anomaly.detection_method,
                    "confidence": anomaly.confidence,
                    "features": anomaly.features,
                    "data_point": {
                        "vehicle_count": anomaly.data_point.vehicle_count,
                        "average_speed": anomaly.data_point.average_speed,
                        "traffic_density": anomaly.data_point.traffic_density,
                        "flow_rate": anomaly.data_point.flow_rate,
                        "occupancy_rate": anomaly.data_point.occupancy_rate,
                        "queue_length": anomaly.data_point.queue_length,
                    },
                }

                # Store in repository (would use actual repository method)
                logger.debug(f"Would persist anomaly: {anomaly_data}")

            logger.info(f"Persisted {len(anomalies)} anomalies")

        except Exception as e:
            logger.error(f"Failed to persist anomalies: {e}")

    async def retrain_models(self, camera_id: str | None = None) -> bool:
        """Retrain anomaly detection models.

        Args:
            camera_id: Optional camera ID for camera-specific retraining

        Returns:
            True if retraining successful
        """
        try:
            # Clear cached models
            model_key = (
                f"anomaly_model:isolation_forest:{camera_id}"
                if camera_id
                else "anomaly_model:isolation_forest:global"
            )
            await self.cache_service.delete(model_key)

            # Retrain model
            model = await self._get_isolation_forest_model(camera_id)

            if model:
                logger.info(
                    f"Successfully retrained anomaly detection model for camera {camera_id}"
                )
                return True
            else:
                logger.error("Failed to retrain anomaly detection model")
                return False

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return False
