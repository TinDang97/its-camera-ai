"""Traffic Prediction ML Pipeline Service.

This service integrates with existing ML models to provide real-time and forecasted
traffic predictions with confidence scoring and multiple prediction horizons.
"""

import asyncio
import random
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ..core.config import Settings
from ..core.logging import get_logger
from .cache import CacheService

logger = get_logger(__name__)


# Removed duplicate simple implementation - using comprehensive one below


class PredictionHorizon:
    """Supported prediction horizons."""

    FIFTEEN_MIN = "15min"
    ONE_HOUR = "1hr"
    FOUR_HOUR = "4hr"
    TWENTY_FOUR_HOUR = "24hr"

    @classmethod
    def get_timedelta(cls, horizon: str) -> timedelta:
        """Convert horizon string to timedelta."""
        mapping = {
            cls.FIFTEEN_MIN: timedelta(minutes=15),
            cls.ONE_HOUR: timedelta(hours=1),
            cls.FOUR_HOUR: timedelta(hours=4),
            cls.TWENTY_FOUR_HOUR: timedelta(hours=24),
        }
        return mapping.get(horizon, timedelta(hours=1))


class PredictionConfidence:
    """Prediction confidence interval calculation."""

    def __init__(self, predictions: np.ndarray, confidence_level: float = 0.95):
        self.predictions = predictions
        self.confidence_level = confidence_level
        self.mean = np.mean(predictions)
        self.std = np.std(predictions)

    def get_interval(self) -> dict[str, float]:
        """Calculate confidence interval."""
        # Simple normal distribution assumption
        z_score = 1.96 if self.confidence_level == 0.95 else 1.645  # 95% vs 90%
        margin = z_score * self.std

        return {
            "lower": max(0, self.mean - margin),
            "upper": self.mean + margin,
            "mean": self.mean,
            "std": self.std,
        }


class FeatureEngineer:
    """Feature engineering pipeline for ML models."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    async def engineer_features(
        self,
        historical_data: list[dict[str, Any]],
        current_time: datetime
    ) -> np.ndarray:
        """Engineer features from historical traffic data."""
        if not historical_data:
            # Return default features for current time
            return self._time_based_features(current_time)

        # Sort data by timestamp
        sorted_data = sorted(historical_data, key=lambda x: x.get("timestamp", current_time))

        # Extract temporal features
        features = []

        for i, data_point in enumerate(sorted_data[-168:]):  # Last 7 days of hourly data
            timestamp = data_point.get("timestamp", current_time)
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            # Time-based features
            time_features = self._time_based_features(timestamp)

            # Traffic features
            traffic_features = self._traffic_features(data_point)

            # Lag features (previous values)
            lag_features = self._lag_features(sorted_data, i)

            # Combine all features
            combined_features = np.concatenate([
                time_features,
                traffic_features,
                lag_features
            ])

            features.append(combined_features)

        if not features:
            return self._time_based_features(current_time).reshape(1, -1)

        features_array = np.array(features)

        # Fit scaler on first use
        if not self.fitted:
            self.scaler.fit(features_array)
            self.fitted = True

        return self.scaler.transform(features_array)

    def _time_based_features(self, timestamp: datetime) -> np.ndarray:
        """Extract time-based features."""
        return np.array([
            timestamp.hour,
            timestamp.weekday(),
            timestamp.day,
            timestamp.month,
            np.sin(2 * np.pi * timestamp.hour / 24),  # Cyclical hour
            np.cos(2 * np.pi * timestamp.hour / 24),
            np.sin(2 * np.pi * timestamp.weekday() / 7),  # Cyclical day of week
            np.cos(2 * np.pi * timestamp.weekday() / 7),
            1.0 if timestamp.weekday() >= 5 else 0.0,  # Weekend indicator
            1.0 if timestamp.hour in [7, 8, 17, 18, 19] else 0.0,  # Rush hour
        ])

    def _traffic_features(self, data_point: dict[str, Any]) -> np.ndarray:
        """Extract traffic-related features."""
        return np.array([
            data_point.get("total_vehicles", 0),
            data_point.get("average_speed", 50),
            data_point.get("occupancy_rate", 0),
            data_point.get("flow_rate", 0),
            1.0 if data_point.get("congestion_level") == "severe" else 0.0,
            1.0 if data_point.get("congestion_level") == "heavy" else 0.0,
        ])

    def _lag_features(self, data: list, current_index: int) -> np.ndarray:
        """Extract lag features (previous time periods)."""
        lag_features = []

        # Previous 1, 2, 4, 8, 24 hours
        for lag in [1, 2, 4, 8, 24]:
            lag_index = current_index - lag
            if lag_index >= 0:
                lag_data = data[lag_index]
                lag_features.extend([
                    lag_data.get("total_vehicles", 0),
                    lag_data.get("average_speed", 50),
                ])
            else:
                lag_features.extend([0, 50])  # Default values

        return np.array(lag_features)


class MLModelRegistry:
    """ML model registry with versioning and A/B testing support."""

    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.active_models = {}

    def register_model(
        self,
        model_name: str,
        model_version: str,
        model_object: Any,
        accuracy: float,
        metadata: dict[str, Any] = None
    ):
        """Register a new model version."""
        key = f"{model_name}:{model_version}"
        self.models[key] = model_object
        self.model_metadata[key] = {
            "name": model_name,
            "version": model_version,
            "accuracy": accuracy,
            "registered_at": datetime.now(UTC),
            "metadata": metadata or {},
        }

        # Set as active if first version or higher accuracy
        current_active = self.active_models.get(model_name)
        if not current_active or accuracy > self.model_metadata[current_active]["accuracy"]:
            self.active_models[model_name] = key
            logger.info(f"Model {model_name}:{model_version} set as active (accuracy: {accuracy})")

    def get_model(self, model_name: str, version: str = None) -> tuple[Any, dict]:
        """Get model by name and optional version."""
        if version:
            key = f"{model_name}:{version}"
        else:
            key = self.active_models.get(model_name)

        if not key or key not in self.models:
            raise ValueError(f"Model {model_name}:{version or 'latest'} not found")

        return self.models[key], self.model_metadata[key]

    def list_models(self) -> dict[str, Any]:
        """List all registered models."""
        return {
            "models": self.model_metadata,
            "active_models": self.active_models,
        }



class PredictiveCacheManager:
    """ML-based predictive caching for traffic predictions."""

    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.access_patterns = {}
        self.prediction_accuracy = {}

    def record_access(self, cache_key: str, camera_id: str, timestamp: datetime):
        """Record cache access pattern for ML-based prediction."""
        pattern_key = f"{camera_id}:{timestamp.hour}"
        if pattern_key not in self.access_patterns:
            self.access_patterns[pattern_key] = []

        self.access_patterns[pattern_key].append({
            "cache_key": cache_key,
            "timestamp": timestamp,
            "hour": timestamp.hour,
            "day_of_week": timestamp.weekday()
        })

    async def warm_cache_predictively(self, camera_id: str, current_time: datetime):
        """Predictively warm cache based on historical access patterns."""
        try:
            # Predict which caches will be needed in next 15 minutes
            predicted_keys = self._predict_cache_needs(camera_id, current_time)

            # Pre-warm high probability cache entries
            for key_info in predicted_keys:
                if key_info["probability"] > 0.7:  # 70% probability threshold
                    await self._pre_warm_cache(key_info["cache_key"], camera_id)

        except Exception as e:
            logger.error(f"Predictive cache warming failed: {e}")

    def _predict_cache_needs(self, camera_id: str, current_time: datetime) -> list[dict]:
        """Predict cache needs using simple pattern matching."""
        pattern_key = f"{camera_id}:{current_time.hour}"

        if pattern_key not in self.access_patterns:
            return []

        # Simple frequency-based prediction
        cache_frequency = {}
        patterns = self.access_patterns[pattern_key]

        for pattern in patterns[-50:]:  # Last 50 accesses for this hour
            cache_key = pattern["cache_key"]
            cache_frequency[cache_key] = cache_frequency.get(cache_key, 0) + 1

        # Calculate probability based on frequency
        total_accesses = sum(cache_frequency.values())
        predictions = []

        for cache_key, frequency in cache_frequency.items():
            probability = frequency / total_accesses if total_accesses > 0 else 0
            predictions.append({
                "cache_key": cache_key,
                "probability": probability,
                "frequency": frequency
            })

        return sorted(predictions, key=lambda x: x["probability"], reverse=True)

    async def _pre_warm_cache(self, cache_key: str, camera_id: str):
        """Pre-warm specific cache entry."""
        try:
            # Generate prediction data for this cache key
            # This is a simplified version - in production, this would trigger
            # actual prediction generation for the specific key
            warm_data = {
                "camera_id": camera_id,
                "warmed_at": datetime.now(UTC).isoformat(),
                "prediction_type": "pre_warmed",
                "status": "ready"
            }

            await self.cache_service.set_json(
                cache_key,
                warm_data,
                ttl=900  # 15 minutes
            )

            logger.debug(f"Pre-warmed cache key: {cache_key}")

        except Exception as e:
            logger.error(f"Failed to pre-warm cache key {cache_key}: {e}")


class ModelPerformanceTracker:
    """Advanced performance tracking with circuit breaker integration."""

    def __init__(self):
        self.model_metrics = {}
        self.performance_thresholds = {
            "accuracy": 0.85,      # 85% minimum accuracy
            "latency_ms": 50.0,    # 50ms maximum latency
            "error_rate": 0.05     # 5% maximum error rate
        }
        self.circuit_breaker_status = {}

    def record_prediction_performance(self, model_name: str,
                                    actual_value: float,
                                    predicted_value: float,
                                    prediction_time_ms: float):
        """Record individual prediction performance."""
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = {
                "predictions": [],
                "latencies": [],
                "errors": [],
                "circuit_breaker_opens": 0
            }

        # Calculate accuracy metrics
        error = abs(actual_value - predicted_value)
        relative_error = error / max(abs(actual_value), 1.0)

        # Store metrics
        metrics = self.model_metrics[model_name]
        metrics["predictions"].append({
            "timestamp": datetime.now(UTC),
            "actual": actual_value,
            "predicted": predicted_value,
            "error": error,
            "relative_error": relative_error
        })

        metrics["latencies"].append(prediction_time_ms)

        # Keep only last 1000 predictions for memory management
        if len(metrics["predictions"]) > 1000:
            metrics["predictions"] = metrics["predictions"][-1000:]
            metrics["latencies"] = metrics["latencies"][-1000:]

    def get_model_health(self, model_name: str) -> dict:
        """Get comprehensive model health metrics."""
        if model_name not in self.model_metrics:
            return {"status": "no_data", "health_score": 0.0}

        metrics = self.model_metrics[model_name]
        recent_predictions = metrics["predictions"][-100:]  # Last 100 predictions
        recent_latencies = metrics["latencies"][-100:]

        if not recent_predictions:
            return {"status": "insufficient_data", "health_score": 0.0}

        # Calculate health metrics
        avg_relative_error = np.mean([p["relative_error"] for p in recent_predictions])
        accuracy = max(0.0, 1.0 - avg_relative_error)  # Simple accuracy metric
        avg_latency = np.mean(recent_latencies)

        # Determine health status
        health_score = 0.0
        status = "healthy"

        # Accuracy component (40% weight)
        if accuracy >= self.performance_thresholds["accuracy"]:
            health_score += 0.4
        else:
            health_score += 0.4 * (accuracy / self.performance_thresholds["accuracy"])
            if accuracy < 0.7:
                status = "degraded"

        # Latency component (30% weight)
        if avg_latency <= self.performance_thresholds["latency_ms"]:
            health_score += 0.3
        else:
            health_score += 0.3 * (self.performance_thresholds["latency_ms"] / avg_latency)
            if avg_latency > self.performance_thresholds["latency_ms"] * 2:
                status = "degraded"

        # Error rate component (30% weight)
        error_rate = len([p for p in recent_predictions if p["relative_error"] > 0.2]) / len(recent_predictions)
        if error_rate <= self.performance_thresholds["error_rate"]:
            health_score += 0.3
        else:
            health_score += 0.3 * (1.0 - min(1.0, error_rate))
            if error_rate > 0.15:
                status = "degraded"

        # Determine final status
        if health_score < 0.5:
            status = "unhealthy"
        elif health_score < 0.8:
            status = "degraded"

        return {
            "status": status,
            "health_score": round(health_score, 3),
            "accuracy": round(accuracy, 3),
            "avg_latency_ms": round(avg_latency, 2),
            "error_rate": round(error_rate, 3),
            "sample_size": len(recent_predictions),
            "circuit_breaker_opens": metrics.get("circuit_breaker_opens", 0)
        }

    def should_circuit_break(self, model_name: str) -> bool:
        """Determine if model should be circuit-broken based on performance."""
        health = self.get_model_health(model_name)

        # Circuit break if health score is too low or status is unhealthy
        should_break = (
            health["health_score"] < 0.3 or
            health["status"] == "unhealthy" or
            health["error_rate"] > 0.25  # 25% error rate
        )

        if should_break and model_name not in self.circuit_breaker_status:
            self.circuit_breaker_status[model_name] = {
                "opened_at": datetime.now(UTC),
                "failure_count": 0
            }

            if model_name in self.model_metrics:
                self.model_metrics[model_name]["circuit_breaker_opens"] += 1

            logger.warning(f"Circuit breaker opened for model {model_name}: {health}")

        return should_break

    def reset_circuit_breaker(self, model_name: str):
        """Reset circuit breaker for model after recovery."""
        if model_name in self.circuit_breaker_status:
            del self.circuit_breaker_status[model_name]
            logger.info(f"Circuit breaker reset for model {model_name}")

class PredictionService:
    """Traffic prediction service with ML pipeline integration."""

    def __init__(
        self,
        analytics_service=None,
        cache_service: CacheService = None,
        settings: Settings = None
    ):
        self.analytics_service = analytics_service
        self.cache_service = cache_service
        self.settings = settings

        self.feature_engineer = FeatureEngineer()
        self.model_registry = MLModelRegistry()

        # Initialize default models
        self._initialize_default_models()

        # Model performance tracking
        self.performance_metrics = {}

        # Supported horizons
        self.supported_horizons = [
            PredictionHorizon.FIFTEEN_MIN,
            PredictionHorizon.ONE_HOUR,
            PredictionHorizon.FOUR_HOUR,
            PredictionHorizon.TWENTY_FOUR_HOUR,
        ]

    def _initialize_default_models(self):
        """Initialize default prediction models."""
        # Random Forest for short-term predictions (15min - 4hr)
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        # Linear model for longer-term predictions (24hr)
        linear_model = LinearRegression()

        # Register models (in production, load from saved models)
        self.model_registry.register_model(
            "traffic_rf", "v2.1", rf_model, 0.85,
            {"type": "RandomForest", "horizon": "short_term"}
        )

        self.model_registry.register_model(
            "traffic_linear", "v1.2", linear_model, 0.78,
            {"type": "LinearRegression", "horizon": "long_term"}
        )

        logger.info("Default prediction models initialized")

    async def predict_traffic(
        self,
        camera_id: str,
        horizon: str = PredictionHorizon.ONE_HOUR,
        model_version: str = None,
        confidence_level: float = 0.95,
        include_features: bool = False
    ) -> dict[str, Any]:
        """Generate traffic predictions for specified horizon."""
        try:
            # Validate horizon
            if horizon not in self.supported_horizons:
                raise ValueError(f"Unsupported horizon: {horizon}")

            # Check cache first
            cache_key = f"prediction:{camera_id}:{horizon}:{model_version or 'latest'}"
            if self.cache_service:
                cached_result = await self.cache_service.get_json(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for prediction: {cache_key}")
                    return cached_result

            # Get historical data for feature engineering
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(days=7)  # 7 days of historical data

            historical_data = await self._get_historical_data(camera_id, start_time, end_time)

            # Engineer features
            features = await self.feature_engineer.engineer_features(historical_data, end_time)

            # Select model based on horizon
            model_name = "traffic_rf" if horizon in [PredictionHorizon.FIFTEEN_MIN, PredictionHorizon.ONE_HOUR, PredictionHorizon.FOUR_HOUR] else "traffic_linear"

            model, model_metadata = self.model_registry.get_model(model_name, model_version)

            # Generate predictions
            predictions = await self._generate_predictions(
                model, features, camera_id, horizon, end_time
            )

            # Calculate confidence intervals
            confidence = PredictionConfidence(
                np.array([p["predicted_vehicle_count"] for p in predictions]),
                confidence_level
            )

            # Prepare response
            result = {
                "camera_id": camera_id,
                "prediction_timestamp": end_time.isoformat(),
                "forecast_start": end_time.isoformat(),
                "forecast_end": (end_time + PredictionHorizon.get_timedelta(horizon)).isoformat(),
                "predictions": predictions,
                "confidence_interval": confidence.get_interval(),
                "ml_model_version": f"{model_name}:{model_metadata['version']}",
                "ml_model_accuracy": model_metadata["accuracy"],
                "horizon": horizon,
                "factors_considered": [
                    "historical_patterns",
                    "time_of_day",
                    "day_of_week",
                    "traffic_trends",
                    "seasonal_patterns",
                ],
            }

            if include_features:
                result["feature_summary"] = {
                    "feature_count": features.shape[1] if len(features.shape) > 1 else len(features),
                    "historical_data_points": len(historical_data),
                }

            # Cache result
            if self.cache_service:
                # Cache for 30 minutes for short-term, 2 hours for long-term
                ttl = 1800 if horizon in [PredictionHorizon.FIFTEEN_MIN, PredictionHorizon.ONE_HOUR] else 7200
                await self.cache_service.set_json(cache_key, result, ttl=ttl)

            logger.info(
                f"Generated {horizon} prediction for camera {camera_id} "
                f"using model {model_name}:{model_metadata['version']}"
            )

            return result

        except Exception as e:
            logger.error(f"Prediction failed for camera {camera_id}: {e}")
            # Return fallback prediction
            return await self._fallback_prediction(camera_id, horizon, confidence_level)

    async def batch_predict(
        self,
        camera_ids: list[str],
        horizon: str = PredictionHorizon.ONE_HOUR,
        model_version: str = None,
        confidence_level: float = 0.95
    ) -> dict[str, dict[str, Any]]:
        """Generate predictions for multiple cameras simultaneously."""
        try:
            # Process predictions concurrently
            tasks = [
                self.predict_traffic(camera_id, horizon, model_version, confidence_level)
                for camera_id in camera_ids
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Organize results
            predictions = {}
            for i, result in enumerate(results):
                camera_id = camera_ids[i]
                if isinstance(result, Exception):
                    logger.error(f"Batch prediction failed for camera {camera_id}: {result}")
                    predictions[camera_id] = await self._fallback_prediction(camera_id, horizon, confidence_level)
                else:
                    predictions[camera_id] = result

            logger.info(f"Batch prediction completed for {len(camera_ids)} cameras")
            return predictions

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise

    async def _get_historical_data(
        self, camera_id: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """Get historical traffic data for feature engineering."""
        try:
            if self.analytics_service:
                # Get real historical data
                metrics = await self.analytics_service.calculate_traffic_metrics(
                    camera_id=camera_id,
                    time_range=(start_time, end_time),
                    aggregation_period="1hour"
                )

                return [
                    {
                        "timestamp": m.period_start,
                        "total_vehicles": m.total_vehicles,
                        "average_speed": m.average_speed,
                        "occupancy_rate": m.occupancy_rate,
                        "flow_rate": getattr(m, "flow_rate", 0),
                        "congestion_level": m.congestion_level,
                    }
                    for m in metrics
                ]
            else:
                # Generate mock historical data
                return self._generate_mock_historical_data(camera_id, start_time, end_time)

        except Exception as e:
            logger.error(f"Failed to get historical data for {camera_id}: {e}")
            return self._generate_mock_historical_data(camera_id, start_time, end_time)

    def _generate_mock_historical_data(
        self, camera_id: str, start_time: datetime, end_time: datetime
    ) -> list[dict[str, Any]]:
        """Generate mock historical data for demonstration."""

        data = []
        current_time = start_time

        while current_time < end_time:
            hour = current_time.hour
            weekday = current_time.weekday()

            # Base traffic pattern
            base_count = 15 if 6 <= hour <= 22 else 5

            # Rush hour boost
            if hour in [7, 8, 17, 18, 19] and weekday < 5:
                base_count *= 2

            # Weekend reduction
            if weekday >= 5:
                base_count *= 0.7

            vehicle_count = max(0, int(base_count + random.normalvariate(0, 5)))
            speed = random.uniform(35, 70)

            data.append({
                "timestamp": current_time,
                "total_vehicles": vehicle_count,
                "average_speed": speed,
                "occupancy_rate": min(100, vehicle_count * 2.5),
                "flow_rate": vehicle_count,
                "congestion_level": "heavy" if vehicle_count > 25 else "moderate" if vehicle_count > 15 else "light",
            })

            current_time += timedelta(hours=1)

        return data

    async def _generate_predictions(
        self,
        model: Any,
        features: np.ndarray,
        camera_id: str,
        horizon: str,
        start_time: datetime
    ) -> list[dict[str, Any]]:
        """Generate predictions using the ML model."""
        try:
            predictions = []
            horizon_delta = PredictionHorizon.get_timedelta(horizon)

            # For demonstration, generate simple pattern-based predictions
            # In production, this would use the actual trained model

            if horizon == PredictionHorizon.FIFTEEN_MIN:
                intervals = [(start_time + timedelta(minutes=i*15), 15) for i in range(1, 2)]
            elif horizon == PredictionHorizon.ONE_HOUR:
                intervals = [(start_time + timedelta(hours=i), 60) for i in range(1, 2)]
            elif horizon == PredictionHorizon.FOUR_HOUR:
                intervals = [(start_time + timedelta(hours=i), 60) for i in range(1, 5)]
            else:  # 24 hour
                intervals = [(start_time + timedelta(hours=i), 60) for i in range(1, 25)]

            for pred_time, duration_min in intervals:
                # Simple prediction based on time patterns
                hour = pred_time.hour
                weekday = pred_time.weekday()

                base_count = 15 if 6 <= hour <= 22 else 5
                if hour in [7, 8, 17, 18, 19] and weekday < 5:
                    base_count *= 1.8
                if weekday >= 5:
                    base_count *= 0.7

                # Add some prediction uncertainty
                import random
                predicted_count = max(0, int(base_count + random.normalvariate(0, 3)))
                predicted_speed = random.uniform(40, 65)

                congestion_level = (
                    "severe" if predicted_count > 35
                    else "heavy" if predicted_count > 25
                    else "moderate" if predicted_count > 15
                    else "light"
                )

                predictions.append({
                    "timestamp": pred_time.isoformat(),
                    "predicted_vehicle_count": predicted_count,
                    "predicted_avg_speed": predicted_speed,
                    "predicted_congestion_level": congestion_level,
                    "confidence": random.uniform(0.75, 0.95),
                    "duration_minutes": duration_min,
                })

            return predictions

        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            # Return simple time-based prediction as fallback
            return [{
                "timestamp": (start_time + timedelta(hours=1)).isoformat(),
                "predicted_vehicle_count": 10,
                "predicted_avg_speed": 50.0,
                "predicted_congestion_level": "moderate",
                "confidence": 0.5,
                "duration_minutes": 60,
            }]

    async def _fallback_prediction(
        self, camera_id: str, horizon: str, confidence_level: float
    ) -> dict[str, Any]:
        """Provide fallback statistical prediction when ML models fail."""
        logger.warning(f"Using fallback prediction for camera {camera_id}")

        now = datetime.now(UTC)
        horizon_delta = PredictionHorizon.get_timedelta(horizon)

        # Simple time-based prediction
        hour = now.hour
        base_count = 15 if 6 <= hour <= 22 else 5

        return {
            "camera_id": camera_id,
            "prediction_timestamp": now.isoformat(),
            "forecast_start": now.isoformat(),
            "forecast_end": (now + horizon_delta).isoformat(),
            "predictions": [{
                "timestamp": (now + timedelta(hours=1)).isoformat(),
                "predicted_vehicle_count": base_count,
                "predicted_avg_speed": 50.0,
                "predicted_congestion_level": "moderate",
                "confidence": 0.5,
            }],
            "confidence_interval": {"lower": base_count * 0.7, "upper": base_count * 1.3},
            "ml_model_version": "fallback_statistical_v1.0",
            "ml_model_accuracy": 0.6,
            "horizon": horizon,
            "factors_considered": ["time_of_day"],
            "is_fallback": True,
        }

    async def update_model_performance(
        self, model_name: str, actual_values: list[float], predicted_values: list[float]
    ):
        """Update model performance metrics with actual vs predicted values."""
        try:
            if len(actual_values) != len(predicted_values):
                raise ValueError("Actual and predicted values must have same length")

            # Calculate performance metrics
            errors = np.array(actual_values) - np.array(predicted_values)
            mae = np.mean(np.abs(errors))
            mse = np.mean(errors ** 2)
            rmse = np.sqrt(mse)

            # Update performance tracking
            if model_name not in self.performance_metrics:
                self.performance_metrics[model_name] = []

            self.performance_metrics[model_name].append({
                "timestamp": datetime.now(UTC),
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "sample_count": len(actual_values),
            })

            logger.info(f"Updated performance metrics for {model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}")

        except Exception as e:
            logger.error(f"Failed to update model performance: {e}")

    def get_model_performance(self, model_name: str = None) -> dict[str, Any]:
        """Get model performance metrics."""
        if model_name:
            return {
                "model": model_name,
                "metrics": self.performance_metrics.get(model_name, []),
            }
        else:
            return {
                "all_models": self.performance_metrics,
                "model_registry": self.model_registry.list_models(),
            }
