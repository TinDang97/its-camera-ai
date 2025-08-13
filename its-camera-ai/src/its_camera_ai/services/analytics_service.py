"""Analytics Service for traffic data processing and insights generation.

This service provides real-time traffic analytics, rule-based violation detection,
speed calculations, trajectory analysis, and anomaly detection using ML algorithms.
Optimized for high-performance processing with TimescaleDB integration.
"""

import statistics
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.schemas.analytics import (
    TrafficMetrics as TrafficMetricsResponse,
)
from ..core.config import Settings
from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from ..models.analytics import (
    RuleViolation,
    SpeedLimit,
    TrafficAnomaly,
    TrafficMetrics,
    VehicleTrajectory,
    ViolationType,
)
from ..models.detection_result import DetectionResult
from .base_service import BaseAsyncService
from .cache import CacheService

logger = get_logger(__name__)


class AnalyticsServiceError(Exception):
    """Analytics service specific exceptions."""

    pass


class RuleEngine:
    """Traffic rule evaluation engine with configurable rules."""

    def __init__(self, settings: Settings, session: AsyncSession | None = None, cache_service: CacheService | None = None):
        self.settings = settings
        self.session = session
        self.cache_service = cache_service
        self._rules = self._load_default_rules()

    def _load_default_rules(self) -> dict[str, dict[str, Any]]:
        """Load default traffic rules configuration."""
        return {
            "speed_limit": {
                "type": ViolationType.SPEEDING,
                "default_limit": 50.0,  # km/h
                "tolerance": 5.0,  # km/h
                "severity_thresholds": {
                    "medium": 15.0,  # 15+ km/h over limit
                    "high": 25.0,  # 25+ km/h over limit
                    "critical": 40.0,  # 40+ km/h over limit
                },
            },
            "wrong_way": {
                "type": ViolationType.WRONG_WAY,
                "min_confidence": 0.8,
                "min_distance": 10.0,  # meters
                "severity_thresholds": {"high": 0.8, "critical": 0.9},
            },
            "red_light": {
                "type": ViolationType.RED_LIGHT,
                "intersection_zones": [],
                "min_confidence": 0.7,
                "severity_thresholds": {"high": 0.7, "critical": 0.9},
            },
            "illegal_parking": {
                "type": ViolationType.ILLEGAL_PARKING,
                "min_duration": 300.0,  # 5 minutes
                "restricted_zones": [],
                "severity_thresholds": {
                    "low": 300.0,
                    "medium": 900.0,  # 15 minutes
                    "high": 1800.0,  # 30 minutes
                },
            },
        }

    async def evaluate_speed_rule(
        self, speed: float, zone_id: str | None = None, vehicle_type: str | None = None,
        check_time: datetime | None = None, weather: str | None = None, visibility: float | None = None
    ) -> dict[str, Any] | None:
        """Evaluate speed limit violation with dynamic database lookup."""
        rule = self._rules["speed_limit"]
        speed_limit = await self._get_speed_limit(zone_id, vehicle_type, check_time, weather, visibility)

        if speed <= speed_limit + rule["tolerance"]:
            return None

        excess_speed = speed - speed_limit
        severity = "low"

        for sev, threshold in rule["severity_thresholds"].items():
            if excess_speed >= threshold:
                severity = sev

        return {
            "violation_type": rule["type"],
            "severity": severity,
            "measured_value": speed,
            "threshold_value": speed_limit,
            "excess_amount": excess_speed,
            "rule_definition": {
                "speed_limit": speed_limit,
                "tolerance": rule["tolerance"],
                "zone_id": zone_id,
                "vehicle_type": vehicle_type,
                "check_time": check_time.isoformat() if check_time else None,
            },
        }

    def evaluate_wrong_way_rule(
        self, trajectory: list[dict[str, float]], expected_direction: str
    ) -> dict[str, Any] | None:
        """Evaluate wrong-way driving violation."""
        if len(trajectory) < 2:
            return None

        rule = self._rules["wrong_way"]

        # Calculate movement direction from trajectory
        start_point = trajectory[0]
        end_point = trajectory[-1]

        dx = end_point["x"] - start_point["x"]
        dy = end_point["y"] - start_point["y"]
        distance = (dx**2 + dy**2) ** 0.5

        if distance < rule["min_distance"]:
            return None

        # Calculate direction angle
        import math

        actual_direction = math.atan2(dy, dx) * 180 / math.pi

        # Compare with expected direction (simplified)
        direction_diff = abs(
            actual_direction - self._direction_to_angle(expected_direction)
        )
        direction_diff = min(direction_diff, 360 - direction_diff)  # Normalize to 0-180

        # Wrong way if driving in opposite direction (within tolerance)
        is_wrong_way = direction_diff > 135  # More than 135 degrees off

        if not is_wrong_way:
            return None

        confidence = min(1.0, (direction_diff - 135) / 45)  # Scale confidence

        if confidence < rule["min_confidence"]:
            return None

        severity = "medium"
        for sev, threshold in rule["severity_thresholds"].items():
            if confidence >= threshold:
                severity = sev

        return {
            "violation_type": rule["type"],
            "severity": severity,
            "measured_value": actual_direction,
            "threshold_value": self._direction_to_angle(expected_direction),
            "confidence": confidence,
            "rule_definition": {
                "expected_direction": expected_direction,
                "min_confidence": rule["min_confidence"],
                "min_distance": rule["min_distance"],
            },
        }

    async def _get_speed_limit(self, zone_id: str | None, vehicle_type: str | None,
                              check_time: datetime | None = None,
                              weather: str | None = None,
                              visibility: float | None = None) -> float:
        """Get dynamic speed limit for zone and vehicle type with Redis caching.
        
        Args:
            zone_id: Traffic zone identifier
            vehicle_type: Vehicle classification
            check_time: Time to check validity (defaults to current time)
            weather: Current weather condition
            visibility: Current visibility in meters
            
        Returns:
            Speed limit in km/h
        """
        if check_time is None:
            check_time = datetime.utcnow()

        # Normalize vehicle type
        normalized_vehicle_type = self._normalize_vehicle_type(vehicle_type)

        # Build cache key for Redis
        cache_key = f"speed_limit:{zone_id or 'default'}:{normalized_vehicle_type}:{check_time.strftime('%Y%m%d%H')}"

        # Try to get from cache first
        if self.cache_service:
            try:
                cached_limit = await self.cache_service.get_json(cache_key)
                if cached_limit is not None:
                    return float(cached_limit)
            except Exception as e:
                logger.warning(f"Cache retrieval failed for speed limit: {e}")

        # Query database for speed limits
        speed_limit = await self._query_speed_limit_from_db(
            zone_id, normalized_vehicle_type, check_time, weather, visibility
        )

        # Cache the result for 1 hour
        if self.cache_service and speed_limit is not None:
            try:
                await self.cache_service.set_json(cache_key, speed_limit, ttl=3600)
            except Exception as e:
                logger.warning(f"Cache storage failed for speed limit: {e}")

        return speed_limit or self._get_default_speed_limit(normalized_vehicle_type)

    async def _query_speed_limit_from_db(self, zone_id: str | None, vehicle_type: str,
                                       check_time: datetime, weather: str | None = None,
                                       visibility: float | None = None) -> float | None:
        """Query speed limit from database with environmental conditions.
        
        Args:
            zone_id: Traffic zone identifier
            vehicle_type: Normalized vehicle type
            check_time: Time to check validity
            weather: Current weather condition
            visibility: Current visibility in meters
            
        Returns:
            Speed limit in km/h or None if not found
        """
        if not self.session:
            return None

        try:
            # Build query for active speed limits
            query = select(SpeedLimit).where(
                and_(
                    SpeedLimit.enforcement_enabled == True,
                    SpeedLimit.valid_from <= check_time,
                    or_(
                        SpeedLimit.valid_until.is_(None),
                        SpeedLimit.valid_until > check_time
                    )
                )
            )

            # Add zone and vehicle type filters
            if zone_id:
                query = query.where(
                    or_(
                        SpeedLimit.zone_id == zone_id,
                        SpeedLimit.zone_id == "default"  # Fallback to default zone
                    )
                )
            else:
                query = query.where(SpeedLimit.zone_id == "default")

            query = query.where(
                or_(
                    SpeedLimit.vehicle_type == vehicle_type,
                    SpeedLimit.vehicle_type == "general"  # Fallback to general
                )
            )

            # Order by priority (lower number = higher priority) and specificity
            query = query.order_by(
                SpeedLimit.priority.asc(),
                SpeedLimit.zone_id.desc(),  # Specific zones before default
                SpeedLimit.vehicle_type.desc()  # Specific vehicle types before general
            )

            result = await self.session.execute(query)
            speed_limits = result.scalars().all()

            # Find the best matching speed limit
            for limit in speed_limits:
                # Check time-based restrictions
                if not limit.is_valid_at(check_time):
                    continue

                # Check environmental conditions
                if not limit.applies_to_conditions(weather, visibility):
                    continue

                logger.debug(
                    f"Found speed limit: {limit.speed_limit_kmh} km/h",
                    zone_id=zone_id,
                    vehicle_type=vehicle_type,
                    limit_id=limit.id
                )

                return limit.speed_limit_kmh

            return None

        except SQLAlchemyError as e:
            logger.error(f"Failed to query speed limit from database: {e}")
            return None

    def _normalize_vehicle_type(self, vehicle_type: str | None) -> str:
        """Normalize vehicle type for consistent lookup.
        
        Args:
            vehicle_type: Raw vehicle type from detection
            
        Returns:
            Normalized vehicle type
        """
        if not vehicle_type:
            return "general"

        vehicle_type_lower = vehicle_type.lower()

        # Map common variations to standard types
        type_mapping = {
            "car": "car",
            "automobile": "car",
            "sedan": "car",
            "suv": "car",
            "truck": "truck",
            "lorry": "truck",
            "van": "truck",
            "pickup": "truck",
            "motorcycle": "motorcycle",
            "motorbike": "motorcycle",
            "bike": "motorcycle",
            "bus": "bus",
            "coach": "bus",
            "emergency": "emergency",
            "ambulance": "emergency",
            "police": "emergency",
            "fire": "emergency",
        }

        return type_mapping.get(vehicle_type_lower, "general")

    def _get_default_speed_limit(self, vehicle_type: str) -> float:
        """Get default speed limit when database lookup fails.
        
        Args:
            vehicle_type: Normalized vehicle type
            
        Returns:
            Default speed limit in km/h
        """
        base_limit = self._rules["speed_limit"]["default_limit"]

        # Apply vehicle-specific adjustments
        adjustments = {
            "truck": -10.0,     # Trucks typically have lower limits
            "bus": -5.0,        # Buses slightly lower
            "emergency": 20.0,  # Emergency vehicles higher limits
            "motorcycle": 0.0,  # Same as cars
            "car": 0.0,         # Base limit
            "general": 0.0,     # Base limit
        }

        adjustment = adjustments.get(vehicle_type, 0.0)
        return max(20.0, base_limit + adjustment)  # Minimum 20 km/h

    def _direction_to_angle(self, direction: str) -> float:
        """Convert direction string to angle in degrees."""
        direction_map = {
            "north": 90.0,
            "south": 270.0,
            "east": 0.0,
            "west": 180.0,
            "northeast": 45.0,
            "northwest": 135.0,
            "southeast": 315.0,
            "southwest": 225.0,
        }
        return direction_map.get(direction.lower(), 0.0)


class SpeedCalculator:
    """Speed calculation using homography transformation and pixel-to-world mapping."""

    def __init__(self, camera_calibration: dict[str, Any] | None = None):
        self.calibration = camera_calibration or {}
        self.homography_matrix = self._load_homography_matrix()
        self.pixel_to_meter_ratio = self.calibration.get(
            "pixel_to_meter_ratio", 0.05
        )  # Default 5cm/pixel

    def _load_homography_matrix(self) -> np.ndarray | None:
        """Load homography transformation matrix for camera."""
        if "homography" in self.calibration:
            return np.array(self.calibration["homography"])
        return None

    def calculate_speed_from_positions(
        self, positions: list[tuple[float, float, float]], time_interval_seconds: float
    ) -> float | None:
        """Calculate speed from a sequence of positions and time interval.

        Args:
            positions: List of (x, y, timestamp) positions
            time_interval_seconds: Time interval between measurements

        Returns:
            Speed in km/h or None if calculation fails
        """
        if len(positions) < 2:
            return None

        total_distance = 0.0
        total_time = 0.0

        for i in range(1, len(positions)):
            prev_pos = positions[i - 1]
            curr_pos = positions[i]

            # Calculate distance
            if self.homography_matrix is not None:
                # Use homography transformation for accurate distance
                prev_world = self._pixel_to_world_coords(prev_pos[0], prev_pos[1])
                curr_world = self._pixel_to_world_coords(curr_pos[0], curr_pos[1])
                distance = self._world_distance(prev_world, curr_world)
            else:
                # Simple pixel distance with scaling factor
                pixel_distance = (
                    (curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2
                ) ** 0.5
                distance = pixel_distance * self.pixel_to_meter_ratio

            time_diff = curr_pos[2] - prev_pos[2]

            total_distance += distance
            total_time += time_diff

        if total_time <= 0:
            return None

        # Convert from m/s to km/h
        speed_ms = total_distance / total_time
        speed_kmh = speed_ms * 3.6

        return max(0.0, speed_kmh)  # Ensure non-negative speed

    def _pixel_to_world_coords(self, x: float, y: float) -> tuple[float, float]:
        """Transform pixel coordinates to world coordinates using homography."""
        if self.homography_matrix is None:
            return x * self.pixel_to_meter_ratio, y * self.pixel_to_meter_ratio

        # Apply homography transformation
        point = np.array([x, y, 1]).reshape(-1, 1)
        world_point = self.homography_matrix @ point

        # Normalize homogeneous coordinates
        if world_point[2, 0] != 0:
            world_x = world_point[0, 0] / world_point[2, 0]
            world_y = world_point[1, 0] / world_point[2, 0]
            return world_x, world_y

        return x * self.pixel_to_meter_ratio, y * self.pixel_to_meter_ratio

    def _world_distance(
        self, point1: tuple[float, float], point2: tuple[float, float]
    ) -> float:
        """Calculate Euclidean distance between two world coordinates."""
        return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2) ** 0.5


class AnomalyDetector:
    """ML-based anomaly detection for traffic patterns and vehicle behavior."""

    def __init__(self, contamination_rate: float = 0.1):
        self.contamination_rate = contamination_rate
        self.isolation_forest = IsolationForest(
            contamination=contamination_rate, random_state=42, n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, training_data: list[dict[str, float]]) -> None:
        """Train anomaly detection model on historical data."""
        if not training_data:
            logger.warning("No training data provided for anomaly detection")
            return

        # Extract features from training data
        features = self._extract_features(training_data)

        if len(features) == 0:
            logger.warning("No valid features extracted from training data")
            return

        # Fit scaler and model
        scaled_features = self.scaler.fit_transform(features)
        self.isolation_forest.fit(scaled_features)
        self.is_trained = True

        logger.info(f"Anomaly detector trained on {len(features)} samples")

    def detect_anomalies(
        self, data_points: list[dict[str, float]]
    ) -> list[dict[str, Any]]:
        """Detect anomalies in traffic data points."""
        if not self.is_trained:
            logger.warning("Anomaly detector not trained, skipping detection")
            return []

        features = self._extract_features(data_points)
        if len(features) == 0:
            return []

        # Scale features and predict anomalies
        scaled_features = self.scaler.transform(features)
        anomaly_scores = self.isolation_forest.decision_function(scaled_features)
        is_anomaly = self.isolation_forest.predict(scaled_features) == -1

        anomalies = []
        for i, (data_point, score, is_anom) in enumerate(
            zip(data_points, anomaly_scores, is_anomaly, strict=False)
        ):
            if is_anom:
                anomalies.append(
                    {
                        "data_point": data_point,
                        "anomaly_score": abs(score),  # Convert to positive score
                        "severity": self._calculate_severity(abs(score)),
                        "features": features[i].tolist() if len(features) > i else [],
                        "probable_cause": self._analyze_probable_cause(
                            data_point, features[i] if len(features) > i else None
                        ),
                    }
                )

        return anomalies

    def _extract_features(self, data_points: list[dict[str, float]]) -> np.ndarray:
        """Extract numerical features from data points."""
        features = []

        for data_point in data_points:
            feature_vector = []

            # Common traffic metrics features
            feature_keys = [
                "vehicle_count",
                "average_speed",
                "traffic_density",
                "flow_rate",
                "occupancy_rate",
                "queue_length",
            ]

            for key in feature_keys:
                value = data_point.get(key, 0.0)
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                else:
                    feature_vector.append(0.0)

            # Time-based features (hour of day, day of week)
            if "timestamp" in data_point:
                try:
                    if isinstance(data_point["timestamp"], datetime):
                        dt = data_point["timestamp"]
                    else:
                        dt = datetime.fromisoformat(str(data_point["timestamp"]))

                    feature_vector.extend(
                        [
                            dt.hour / 24.0,  # Normalized hour
                            dt.weekday() / 6.0,  # Normalized day of week
                            (dt.minute * 60 + dt.second)
                            / 86400.0,  # Normalized time within day
                        ]
                    )
                except (ValueError, TypeError):
                    feature_vector.extend([0.0, 0.0, 0.0])
            else:
                feature_vector.extend([0.0, 0.0, 0.0])

            if len(feature_vector) > 0:
                features.append(feature_vector)

        return (
            np.array(features) if features else np.array([]).reshape(0, 9)
        )  # 9 features expected

    def _calculate_severity(self, anomaly_score: float) -> str:
        """Calculate severity based on anomaly score."""
        if anomaly_score > 0.6:
            return "critical"
        elif anomaly_score > 0.4:
            return "high"
        elif anomaly_score > 0.2:
            return "medium"
        else:
            return "low"

    def _analyze_probable_cause(
        self, data_point: dict[str, float], features: np.ndarray | None
    ) -> str:
        """Analyze probable cause of anomaly based on data characteristics."""
        if features is None:
            return "unknown"

        # Simple rule-based cause analysis
        if data_point.get("average_speed", 0) > 80:
            return "excessive_speed"
        elif data_point.get("traffic_density", 0) > 50:
            return "unusual_congestion"
        elif data_point.get("vehicle_count", 0) == 0:
            return "traffic_absence"
        elif data_point.get("flow_rate", 0) < 100:
            return "low_traffic_flow"
        else:
            return "pattern_deviation"


class AnalyticsService(BaseAsyncService):
    """Comprehensive analytics service for traffic monitoring and insights.

    Provides real-time traffic analytics, rule-based violation detection,
    speed calculations, trajectory analysis, and ML-based anomaly detection.
    """

    def __init__(self, session: AsyncSession, settings: Settings, cache_service: CacheService | None = None):
        super().__init__(session, TrafficMetrics)
        self.settings = settings
        self.cache_service = cache_service
        self.rule_engine = RuleEngine(settings, session, cache_service)
        self.speed_calculator = SpeedCalculator()
        self.anomaly_detector = AnomalyDetector()
        self._initialize_anomaly_detector()

    async def _initialize_anomaly_detector(self) -> None:
        """Initialize anomaly detector with historical data."""
        try:
            # Get recent historical data for training
            training_data = await self._get_training_data()
            if training_data:
                self.anomaly_detector.train(training_data)
        except Exception as e:
            logger.warning(f"Failed to initialize anomaly detector: {e}")

    async def _get_training_data(self) -> list[dict[str, float]]:
        """Get historical traffic data for anomaly detector training."""
        try:
            # Get data from last 30 days for training
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)

            query = (
                select(TrafficMetrics)
                .where(
                    and_(
                        TrafficMetrics.timestamp >= start_time,
                        TrafficMetrics.timestamp <= end_time,
                        TrafficMetrics.aggregation_period
                        == "1hour",  # Use hourly aggregations
                    )
                )
                .order_by(TrafficMetrics.timestamp.desc())
                .limit(1000)
            )

            result = await self.session.execute(query)
            metrics = result.scalars().all()

            training_data = []
            for metric in metrics:
                training_data.append(
                    {
                        "vehicle_count": metric.total_vehicles or 0,
                        "average_speed": metric.average_speed or 0.0,
                        "traffic_density": metric.traffic_density or 0.0,
                        "flow_rate": metric.flow_rate or 0.0,
                        "occupancy_rate": metric.occupancy_rate or 0.0,
                        "queue_length": metric.queue_length or 0.0,
                        "timestamp": metric.timestamp,
                    }
                )

            return training_data

        except SQLAlchemyError as e:
            logger.error(f"Failed to fetch training data: {e}")
            return []

    async def process_detections(
        self,
        detections: list[DetectionResult],
        camera_id: str,
        frame_timestamp: datetime,
    ) -> dict[str, Any]:
        """Process vehicle detections and generate analytics results.

        Args:
            detections: List of detection results from current frame
            camera_id: Source camera identifier
            frame_timestamp: Frame timestamp

        Returns:
            Analytics processing results
        """
        start_time = datetime.utcnow()

        try:
            # Filter valid vehicle detections
            vehicle_detections = [
                det
                for det in detections
                if det.is_vehicle and det.class_confidence >= 0.5
            ]

            # Calculate real-time metrics
            metrics = await self._calculate_real_time_metrics(
                vehicle_detections, camera_id, frame_timestamp
            )

            # Evaluate traffic rules for violations
            violations = await self._evaluate_traffic_rules(
                vehicle_detections, camera_id
            )

            # Detect anomalies in current data
            anomalies = await self._detect_real_time_anomalies(metrics, camera_id)

            # Update trajectory analysis
            await self._update_trajectory_analysis(vehicle_detections, camera_id)

            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            result = {
                "camera_id": camera_id,
                "timestamp": frame_timestamp,
                "vehicle_count": len(vehicle_detections),
                "metrics": metrics,
                "violations": violations,
                "anomalies": anomalies,
                "processing_time_ms": processing_time,
            }

            logger.debug(
                f"Processed {len(vehicle_detections)} detections",
                camera_id=camera_id,
                processing_time_ms=processing_time,
                violations_count=len(violations),
                anomalies_count=len(anomalies),
            )

            return result

        except Exception as e:
            logger.error(f"Failed to process detections: {e}", camera_id=camera_id)
            raise AnalyticsServiceError(f"Detection processing failed: {e}")

    async def _calculate_real_time_metrics(
        self, detections: list[DetectionResult], camera_id: str, timestamp: datetime
    ) -> dict[str, Any]:
        """Calculate real-time traffic metrics from current detections."""
        if not detections:
            return {
                "total_vehicles": 0,
                "average_speed": None,
                "traffic_density": 0.0,
                "congestion_level": "free_flow",
            }

        # Count vehicles by type
        vehicle_counts = {}
        speeds = []

        for detection in detections:
            # Count by vehicle type
            vehicle_type = detection.vehicle_type or detection.class_name
            vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1

            # Collect speeds if available
            if detection.velocity_magnitude:
                # Convert pixel velocity to real speed (simplified)
                estimated_speed = (
                    detection.velocity_magnitude * 3.6
                )  # Rough conversion to km/h
                if 0 < estimated_speed < 200:  # Filter unrealistic speeds
                    speeds.append(estimated_speed)

        # Calculate metrics
        total_vehicles = len(detections)
        avg_speed = statistics.mean(speeds) if speeds else None

        # Estimate traffic density (vehicles per lane-km, simplified)
        estimated_lane_length = 1.0  # km, should be configured per camera
        traffic_density = total_vehicles / estimated_lane_length

        # Determine congestion level based on vehicle count and speed
        congestion_level = self._classify_congestion_level(total_vehicles, avg_speed)

        return {
            "total_vehicles": total_vehicles,
            "vehicle_breakdown": vehicle_counts,
            "average_speed": avg_speed,
            "traffic_density": traffic_density,
            "congestion_level": congestion_level,
            "timestamp": timestamp,
        }

    def _classify_congestion_level(
        self, vehicle_count: int, avg_speed: float | None
    ) -> str:
        """Classify traffic congestion level based on vehicle count and speed."""
        # Simple classification rules (should be calibrated per camera/location)
        if vehicle_count == 0:
            return "free_flow"
        elif vehicle_count <= 5:
            return "light" if not avg_speed or avg_speed > 40 else "moderate"
        elif vehicle_count <= 10:
            return "moderate" if not avg_speed or avg_speed > 20 else "heavy"
        elif vehicle_count <= 15:
            return "heavy"
        else:
            return "severe"

    async def _evaluate_traffic_rules(
        self, detections: list[DetectionResult], camera_id: str
    ) -> list[dict[str, Any]]:
        """Evaluate traffic rules and detect violations."""
        violations = []

        for detection in detections:
            # Speed limit evaluation
            if detection.velocity_magnitude:
                estimated_speed = detection.velocity_magnitude * 3.6  # Convert to km/h
                speed_violation = await self.rule_engine.evaluate_speed_rule(
                    estimated_speed, detection.detection_zone, detection.vehicle_type,
                    check_time=detection.created_at
                )

                if speed_violation:
                    violation_record = {
                        **speed_violation,
                        "detection_id": detection.id,
                        "camera_id": camera_id,
                        "track_id": detection.track_id,
                        "license_plate": detection.license_plate,
                        "detection_time": detection.created_at,
                    }
                    violations.append(violation_record)

        # Store violations in database
        if violations:
            await self._store_violations(violations)

        return violations

    async def _store_violations(self, violations: list[dict[str, Any]]) -> None:
        """Store detected violations in the database."""
        try:
            violation_records = []

            for violation in violations:
                record = RuleViolation(
                    violation_type=violation["violation_type"],
                    severity=violation["severity"],
                    detection_time=violation["detection_time"],
                    camera_id=violation["camera_id"],
                    vehicle_track_id=violation.get("track_id"),
                    license_plate=violation.get("license_plate"),
                    vehicle_type=violation.get("vehicle_type"),
                    rule_definition=violation["rule_definition"],
                    measured_value=violation.get("measured_value"),
                    threshold_value=violation.get("threshold_value"),
                    detection_confidence=violation.get("confidence", 0.8),
                )
                violation_records.append(record)

            # Bulk insert violations
            self.session.add_all(violation_records)
            await self.session.commit()

            logger.info(f"Stored {len(violation_records)} traffic violations")

        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Failed to store violations: {e}")

    async def _detect_real_time_anomalies(
        self, metrics: dict[str, Any], camera_id: str
    ) -> list[dict[str, Any]]:
        """Detect anomalies in current traffic metrics."""
        if not self.anomaly_detector.is_trained:
            return []

        # Prepare data point for anomaly detection
        data_point = {
            "vehicle_count": metrics.get("total_vehicles", 0),
            "average_speed": metrics.get("average_speed", 0.0) or 0.0,
            "traffic_density": metrics.get("traffic_density", 0.0),
            "flow_rate": 0.0,  # Would need historical data to calculate
            "occupancy_rate": 0.0,  # Would need zone configuration
            "queue_length": 0.0,  # Would need zone analysis
            "timestamp": metrics.get("timestamp", datetime.utcnow()),
        }

        anomalies = self.anomaly_detector.detect_anomalies([data_point])

        # Store significant anomalies
        if anomalies:
            await self._store_anomalies(anomalies, camera_id)

        return anomalies

    async def _store_anomalies(
        self, anomalies: list[dict[str, Any]], camera_id: str
    ) -> None:
        """Store detected anomalies in the database."""
        try:
            anomaly_records = []

            for anomaly in anomalies:
                if anomaly["anomaly_score"] < 0.3:  # Skip low-confidence anomalies
                    continue

                record = TrafficAnomaly(
                    anomaly_type="traffic_pattern",
                    severity=anomaly["severity"],
                    detection_time=datetime.utcnow(),
                    camera_id=camera_id,
                    anomaly_score=anomaly["anomaly_score"],
                    confidence=min(1.0, anomaly["anomaly_score"] * 1.2),
                    detection_method="isolation_forest",
                    probable_cause=anomaly.get("probable_cause", "unknown"),
                    model_name="IsolationForest",
                    model_version="1.0",
                    model_confidence=anomaly["anomaly_score"],
                    detailed_analysis={
                        "features": anomaly.get("features", []),
                        "data_point": anomaly["data_point"],
                    },
                )
                anomaly_records.append(record)

            if anomaly_records:
                self.session.add_all(anomaly_records)
                await self.session.commit()

                logger.info(f"Stored {len(anomaly_records)} traffic anomalies")

        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Failed to store anomalies: {e}")

    async def _update_trajectory_analysis(
        self, detections: list[DetectionResult], camera_id: str
    ) -> None:
        """Update vehicle trajectory analysis for tracking."""
        # Group detections by track_id
        tracked_vehicles = {}
        for detection in detections:
            if detection.track_id:
                if detection.track_id not in tracked_vehicles:
                    tracked_vehicles[detection.track_id] = []
                tracked_vehicles[detection.track_id].append(detection)

        # Update trajectory records for each tracked vehicle
        for track_id, track_detections in tracked_vehicles.items():
            try:
                await self._update_vehicle_trajectory(
                    track_id, track_detections, camera_id
                )
            except Exception as e:
                logger.warning(f"Failed to update trajectory for track {track_id}: {e}")

    async def _update_vehicle_trajectory(
        self, track_id: int, detections: list[DetectionResult], camera_id: str
    ) -> None:
        """Update or create vehicle trajectory record."""
        if not detections:
            return

        # Get existing trajectory or create new one
        query = select(VehicleTrajectory).where(
            and_(
                VehicleTrajectory.vehicle_track_id == track_id,
                VehicleTrajectory.camera_id == camera_id,
            )
        )

        result = await self.session.execute(query)
        trajectory = result.scalar_one_or_none()

        # Prepare path points from detections
        path_points = []
        for detection in sorted(detections, key=lambda d: d.created_at):
            path_points.append(
                {
                    "x": detection.bbox_center_x,
                    "y": detection.bbox_center_y,
                    "timestamp": detection.created_at.timestamp(),
                    "speed": detection.velocity_magnitude or 0.0,
                }
            )

        if trajectory:
            # Update existing trajectory
            trajectory.end_time = max(d.created_at for d in detections)
            trajectory.duration_seconds = (
                trajectory.end_time - trajectory.start_time
            ).total_seconds()

            # Merge new path points with existing ones
            existing_points = trajectory.path_points or []
            all_points = existing_points + path_points

            # Remove duplicates and sort by timestamp
            unique_points = {p["timestamp"]: p for p in all_points}
            trajectory.path_points = sorted(
                unique_points.values(), key=lambda p: p["timestamp"]
            )

            # Recalculate metrics
            await self._calculate_trajectory_metrics(trajectory)

        else:
            # Create new trajectory
            start_time = min(d.created_at for d in detections)
            end_time = max(d.created_at for d in detections)

            trajectory = VehicleTrajectory(
                vehicle_track_id=track_id,
                camera_id=camera_id,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=(end_time - start_time).total_seconds(),
                path_points=path_points,
                total_distance=0.0,
                straight_line_distance=0.0,
                path_efficiency=1.0,
                average_speed=0.0,
                max_speed=0.0,
                vehicle_type=detections[0].vehicle_type or detections[0].class_name,
                license_plate=detections[0].license_plate,
                tracking_quality=statistics.mean(
                    d.detection_quality for d in detections
                ),
                path_completeness=1.0,
            )

            await self._calculate_trajectory_metrics(trajectory)
            self.session.add(trajectory)

        await self.session.commit()

    async def _calculate_trajectory_metrics(
        self, trajectory: VehicleTrajectory
    ) -> None:
        """Calculate trajectory-specific metrics."""
        path_points = trajectory.path_points or []

        if len(path_points) < 2:
            return

        # Calculate distances and speeds
        total_distance = 0.0
        speeds = []

        for i in range(1, len(path_points)):
            prev_point = path_points[i - 1]
            curr_point = path_points[i]

            # Calculate distance (using pixel coordinates, should use real-world coordinates)
            dx = curr_point["x"] - prev_point["x"]
            dy = curr_point["y"] - prev_point["y"]
            distance = (dx**2 + dy**2) ** 0.5 * 0.05  # Rough pixel-to-meter conversion

            total_distance += distance

            # Calculate speed
            time_diff = curr_point["timestamp"] - prev_point["timestamp"]
            if time_diff > 0:
                speed = (distance / time_diff) * 3.6  # Convert m/s to km/h
                if 0 < speed < 200:  # Filter unrealistic speeds
                    speeds.append(speed)

        # Calculate straight-line distance
        first_point = path_points[0]
        last_point = path_points[-1]
        straight_distance = (
            (last_point["x"] - first_point["x"]) ** 2
            + (last_point["y"] - first_point["y"]) ** 2
        ) ** 0.5 * 0.05

        # Update trajectory metrics
        trajectory.total_distance = total_distance
        trajectory.straight_line_distance = straight_distance
        trajectory.path_efficiency = (
            straight_distance / total_distance if total_distance > 0 else 1.0
        )
        trajectory.average_speed = statistics.mean(speeds) if speeds else 0.0
        trajectory.max_speed = max(speeds) if speeds else 0.0

        # Detect anomalous behavior
        if trajectory.path_efficiency < 0.3:  # Very inefficient path
            trajectory.is_anomalous = True
            trajectory.anomaly_score = 1.0 - trajectory.path_efficiency
            trajectory.anomaly_reasons = ["inefficient_path"]

    async def calculate_traffic_metrics(
        self,
        camera_id: str,
        time_range: tuple[datetime, datetime],
        aggregation_period: str = "1hour",
    ) -> list[TrafficMetricsResponse]:
        """Calculate aggregated traffic metrics for a time range.

        Args:
            camera_id: Camera identifier
            time_range: (start_time, end_time) tuple
            aggregation_period: Aggregation period (1min, 5min, 1hour, etc.)

        Returns:
            List of aggregated traffic metrics
        """
        start_time, end_time = time_range

        try:
            # Query aggregated metrics from database
            query = (
                select(TrafficMetrics)
                .where(
                    and_(
                        TrafficMetrics.camera_id == camera_id,
                        TrafficMetrics.timestamp >= start_time,
                        TrafficMetrics.timestamp <= end_time,
                        TrafficMetrics.aggregation_period == aggregation_period,
                    )
                )
                .order_by(TrafficMetrics.timestamp)
            )

            result = await self.session.execute(query)
            metrics = result.scalars().all()

            # Convert to response format
            response_metrics = []
            for metric in metrics:
                response_metrics.append(
                    TrafficMetricsResponse(
                        camera_id=metric.camera_id,
                        period_start=metric.timestamp,
                        period_end=metric.timestamp
                        + self._get_period_duration(aggregation_period),
                        total_vehicles=metric.total_vehicles,
                        vehicle_breakdown={
                            "car": metric.vehicle_cars,
                            "truck": metric.vehicle_trucks,
                            "bus": metric.vehicle_buses,
                            "motorcycle": metric.vehicle_motorcycles,
                            "bicycle": metric.vehicle_bicycles,
                        },
                        directional_flow={
                            "north": metric.northbound_count,
                            "south": metric.southbound_count,
                            "east": metric.eastbound_count,
                            "west": metric.westbound_count,
                        },
                        avg_speed=metric.average_speed,
                        occupancy_rate=metric.occupancy_rate or 0.0,
                        congestion_level=metric.congestion_level,
                        queue_length=metric.queue_length,
                    )
                )

            return response_metrics

        except SQLAlchemyError as e:
            logger.error(f"Failed to calculate traffic metrics: {e}")
            raise DatabaseError(f"Traffic metrics calculation failed: {e}")

    def _get_period_duration(self, period: str) -> timedelta:
        """Get timedelta for aggregation period."""
        period_map = {
            "1min": timedelta(minutes=1),
            "5min": timedelta(minutes=5),
            "15min": timedelta(minutes=15),
            "1hour": timedelta(hours=1),
            "1day": timedelta(days=1),
        }
        return period_map.get(period, timedelta(hours=1))

    async def get_active_violations(
        self,
        camera_id: str | None = None,
        violation_type: str | None = None,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get active traffic violations with filtering options."""
        try:
            query = select(RuleViolation).where(RuleViolation.status == "active")

            if camera_id:
                query = query.where(RuleViolation.camera_id == camera_id)
            if violation_type:
                query = query.where(RuleViolation.violation_type == violation_type)
            if severity:
                query = query.where(RuleViolation.severity == severity)

            query = query.order_by(RuleViolation.detection_time.desc()).limit(limit)

            result = await self.session.execute(query)
            violations = result.scalars().all()

            return [
                {
                    "id": violation.id,
                    "type": violation.violation_type,
                    "severity": violation.severity,
                    "camera_id": violation.camera_id,
                    "detection_time": violation.detection_time,
                    "measured_value": violation.measured_value,
                    "threshold_value": violation.threshold_value,
                    "license_plate": violation.license_plate,
                    "confidence": violation.detection_confidence,
                }
                for violation in violations
            ]

        except SQLAlchemyError as e:
            logger.error(f"Failed to get active violations: {e}")
            raise DatabaseError(f"Failed to retrieve violations: {e}")

    async def get_traffic_anomalies(
        self,
        camera_id: str | None = None,
        anomaly_type: str | None = None,
        min_score: float = 0.5,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get detected traffic anomalies with filtering options."""
        try:
            query = select(TrafficAnomaly)

            conditions = [TrafficAnomaly.anomaly_score >= min_score]

            if camera_id:
                conditions.append(TrafficAnomaly.camera_id == camera_id)
            if anomaly_type:
                conditions.append(TrafficAnomaly.anomaly_type == anomaly_type)
            if time_range:
                start_time, end_time = time_range
                conditions.append(TrafficAnomaly.detection_time >= start_time)
                conditions.append(TrafficAnomaly.detection_time <= end_time)

            query = query.where(and_(*conditions))
            query = query.order_by(TrafficAnomaly.detection_time.desc()).limit(limit)

            result = await self.session.execute(query)
            anomalies = result.scalars().all()

            return [
                {
                    "id": anomaly.id,
                    "type": anomaly.anomaly_type,
                    "severity": anomaly.severity,
                    "camera_id": anomaly.camera_id,
                    "detection_time": anomaly.detection_time,
                    "score": anomaly.anomaly_score,
                    "confidence": anomaly.confidence,
                    "probable_cause": anomaly.probable_cause,
                    "status": anomaly.status,
                }
                for anomaly in anomalies
            ]

        except SQLAlchemyError as e:
            logger.error(f"Failed to get traffic anomalies: {e}")
            raise DatabaseError(f"Failed to retrieve anomalies: {e}")

    async def generate_analytics_report(
        self,
        camera_ids: list[str],
        time_range: tuple[datetime, datetime],
        report_type: str = "traffic_summary",
    ) -> dict[str, Any]:
        """Generate comprehensive analytics report."""
        start_time, end_time = time_range

        try:
            report_data = {
                "report_type": report_type,
                "time_range": {"start": start_time, "end": end_time},
                "cameras": camera_ids,
                "generated_at": datetime.utcnow(),
            }

            # Aggregate metrics across cameras
            total_vehicles = 0
            total_violations = 0
            total_anomalies = 0
            camera_summaries = []

            for camera_id in camera_ids:
                # Get traffic metrics
                metrics_query = select(func.sum(TrafficMetrics.total_vehicles)).where(
                    and_(
                        TrafficMetrics.camera_id == camera_id,
                        TrafficMetrics.timestamp >= start_time,
                        TrafficMetrics.timestamp <= end_time,
                    )
                )
                metrics_result = await self.session.execute(metrics_query)
                camera_vehicles = metrics_result.scalar() or 0

                # Get violations count
                violations_query = select(func.count(RuleViolation.id)).where(
                    and_(
                        RuleViolation.camera_id == camera_id,
                        RuleViolation.detection_time >= start_time,
                        RuleViolation.detection_time <= end_time,
                    )
                )
                violations_result = await self.session.execute(violations_query)
                camera_violations = violations_result.scalar() or 0

                # Get anomalies count
                anomalies_query = select(func.count(TrafficAnomaly.id)).where(
                    and_(
                        TrafficAnomaly.camera_id == camera_id,
                        TrafficAnomaly.detection_time >= start_time,
                        TrafficAnomaly.detection_time <= end_time,
                    )
                )
                anomalies_result = await self.session.execute(anomalies_query)
                camera_anomalies = anomalies_result.scalar() or 0

                camera_summaries.append(
                    {
                        "camera_id": camera_id,
                        "total_vehicles": camera_vehicles,
                        "violations": camera_violations,
                        "anomalies": camera_anomalies,
                    }
                )

                total_vehicles += camera_vehicles
                total_violations += camera_violations
                total_anomalies += camera_anomalies

            report_data.update(
                {
                    "summary": {
                        "total_vehicles": total_vehicles,
                        "total_violations": total_violations,
                        "total_anomalies": total_anomalies,
                        "cameras_analyzed": len(camera_ids),
                    },
                    "camera_summaries": camera_summaries,
                }
            )

            return report_data

        except SQLAlchemyError as e:
            logger.error(f"Failed to generate analytics report: {e}")
            raise DatabaseError(f"Report generation failed: {e}")
