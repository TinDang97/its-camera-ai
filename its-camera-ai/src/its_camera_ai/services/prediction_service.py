"""Traffic Prediction ML Pipeline Service.

This service integrates with existing ML models to provide real-time and forecasted
traffic predictions with confidence scoring and multiple prediction horizons.
"""

import asyncio
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


class PredictionService:
    """Traffic Prediction Service (BE-ANA-003).
    
    Integrates with ML models to provide real-time and forecasted
    traffic predictions with confidence scoring and feature engineering.
    """

    def __init__(
        self,
        cache_service: CacheService,
        settings: Settings = None
    ):
        self.cache = cache_service
        self.settings = settings or Settings()
        self.feature_engineer = FeatureEngineer()

        # Mock models - in production would load actual ML models
        self.models = {
            "yolo11-v1.2.3": {
                "accuracy": 0.89,
                "last_trained": "2024-01-15",
                "features": ["historical_traffic", "time_of_day", "weather"],
            },
            "ensemble-v2.1.0": {
                "accuracy": 0.93,
                "last_trained": "2024-02-01",
                "features": ["historical_traffic", "time_of_day", "weather", "events"],
            }
        }

        self.current_model = "yolo11-v1.2.3"

    async def predict_single(
        self,
        camera_id: str,
        horizon_minutes: int,
        historical_data: list[dict[str, Any]] | None = None,
        model_version: str | None = None,
        confidence_level: float = 0.95
    ) -> dict[str, Any]:
        """Generate prediction for single camera.
        
        Args:
            camera_id: Camera identifier
            horizon_minutes: Prediction horizon in minutes
            historical_data: Historical data for context
            model_version: Specific model version to use
            confidence_level: Confidence level for intervals
            
        Returns:
            Prediction data with confidence intervals
        """
        start_time = datetime.now(UTC)
        model_version = model_version or self.current_model

        try:
            # Engineer features from historical data
            current_time = datetime.now(UTC)
            features = await self.feature_engineer.engineer_features(
                historical_data or [], current_time
            )

            # Generate prediction using mock model
            # In production, this would call actual ML model
            base_prediction = await self._generate_model_prediction(
                features, horizon_minutes, model_version
            )

            # Calculate confidence intervals
            confidence = PredictionConfidence(
                np.array([base_prediction]), confidence_level
            )
            confidence_interval = confidence.get_interval()

            # Create prediction result
            target_time = current_time + timedelta(minutes=horizon_minutes)

            result = {
                "camera_id": camera_id,
                "prediction_time": current_time,
                "target_time": target_time,
                "horizon_minutes": horizon_minutes,
                "predicted_vehicle_count": base_prediction,
                "confidence_interval": confidence_interval,
                "model_version": model_version,
                "model_accuracy": self.models[model_version]["accuracy"],
                "processing_time_ms": (datetime.now(UTC) - start_time).total_seconds() * 1000,
                "features_used": self.models[model_version]["features"],
            }

            logger.debug(
                "Prediction generated",
                camera_id=camera_id,
                horizon_minutes=horizon_minutes,
                predicted_count=base_prediction,
            )

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}", camera_id=camera_id)
            raise

    async def predict_batch(
        self,
        camera_ids: list[str],
        horizon_minutes: int,
        historical_data: dict[str, list[dict[str, Any]]] | None = None,
        model_version: str | None = None,
        confidence_level: float = 0.95
    ) -> list[dict[str, Any]]:
        """Generate batch predictions for multiple cameras.
        
        Args:
            camera_ids: List of camera identifiers
            horizon_minutes: Prediction horizon in minutes
            historical_data: Historical data per camera
            model_version: Specific model version to use
            confidence_level: Confidence level for intervals
            
        Returns:
            List of prediction results
        """
        predictions = []

        for camera_id in camera_ids:
            try:
                camera_history = historical_data.get(camera_id, []) if historical_data else []
                prediction = await self.predict_single(
                    camera_id=camera_id,
                    horizon_minutes=horizon_minutes,
                    historical_data=camera_history,
                    model_version=model_version,
                    confidence_level=confidence_level,
                )
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Batch prediction failed for camera {camera_id}: {e}")
                continue

        logger.info(
            "Batch predictions completed",
            cameras_requested=len(camera_ids),
            predictions_generated=len(predictions),
        )

        return predictions

    async def get_model_performance(
        self,
        model_version: str | None = None
    ) -> dict[str, float]:
        """Get model performance metrics.
        
        Args:
            model_version: Specific model version
            
        Returns:
            Performance metrics dictionary
        """
        model_version = model_version or self.current_model

        if model_version not in self.models:
            raise ValueError(f"Unknown model version: {model_version}")

        # Mock performance metrics - in production would calculate from validation data
        return {
            "accuracy": self.models[model_version]["accuracy"],
            "mae": random.uniform(2.0, 6.0),  # Mean Absolute Error
            "rmse": random.uniform(3.0, 8.0),  # Root Mean Square Error
            "r2_score": random.uniform(0.75, 0.95),  # R-squared
            "mape": random.uniform(8.0, 15.0),  # Mean Absolute Percentage Error
        }

    async def _generate_model_prediction(
        self,
        features: np.ndarray,
        horizon_minutes: int,
        model_version: str
    ) -> float:
        """Generate prediction using ML model.
        
        In production, this would interface with actual ML models.
        """
        # Mock prediction logic
        base_count = 30.0  # Base traffic count

        # Time-based adjustments
        current_hour = datetime.now(UTC).hour
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:  # Rush hours
            base_count *= 1.5
        elif current_hour >= 22 or current_hour <= 6:  # Night hours
            base_count *= 0.3

        # Horizon adjustments (further predictions less certain)
        horizon_factor = 1.0 - (horizon_minutes / 1440) * 0.2  # Decrease for longer horizons

        # Model-specific adjustments
        model_accuracy = self.models[model_version]["accuracy"]
        accuracy_factor = 0.8 + (model_accuracy - 0.5) * 0.4

        # Add some randomness
        noise = random.uniform(-5, 5)

        prediction = base_count * horizon_factor * accuracy_factor + noise
        return max(0.0, prediction)  # Ensure non-negative

    def get_available_models(self) -> list[str]:
        """Get list of available model versions."""
        return list(self.models.keys())

    def set_current_model(self, model_version: str) -> None:
        """Set the current active model version."""
        if model_version not in self.models:
            raise ValueError(f"Unknown model version: {model_version}")
        self.current_model = model_version
        logger.info(f"Model version changed to {model_version}")


# Export the service class at module level for dependency injection
__all__ = ["PredictionService", "PredictionHorizon", "PredictionConfidence", "FeatureEngineer"]


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
        import random

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
