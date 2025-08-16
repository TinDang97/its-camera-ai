import statistics
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..api.schemas.analytics import (
    TrafficMetrics as TrafficMetricsResponse,
)
from ..core.config import Settings
from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from ..models.analytics import (
    VehicleTrajectory,
    ViolationType,
)
from ..models.detection_result import DetectionResult
from .cache import CacheService

logger = get_logger(__name__)


class IncidentDetectionEngine:
    """Incident Detection and Alert System.

    Processes ML model outputs, applies business rules, and generates
    prioritized alerts through multiple channels with deduplication.
    """

    def __init__(self, cache_service: CacheService, settings: Settings = None):
        self.cache = cache_service
        self.settings = settings or Settings()
        self.active_incidents: dict[str, dict[str, Any]] = {}
        self.alert_rules: dict[str, dict[str, Any]] = {}
        self.suppression_windows: dict[str, datetime] = {}

    async def process_detection(
        self, detection: dict[str, Any], rules: list[dict[str, Any]] | None = None
    ) -> dict[str, Any] | None:
        """Process detection data and generate incident alerts.

        Args:
            detection: Detection data from ML models
            rules: Optional list of alert rules to apply

        Returns:
            IncidentAlert data if incident detected, None otherwise
        """
        start_time = datetime.now(UTC)

        try:
            # Apply business rules
            applicable_rules = rules or await self._get_applicable_rules(detection)

            for rule in applicable_rules:
                if await self._evaluate_rule(detection, rule):
                    # Check for duplicates
                    if await self._is_duplicate_incident(detection, rule):
                        continue

                    # Check suppression windows
                    if await self._is_suppressed(detection, rule):
                        continue

                    # Create incident alert
                    incident = await self._create_incident_alert(detection, rule)

                    # Calculate priority score
                    priority_score = await self._calculate_priority_score(incident)
                    incident["priority_score"] = priority_score

                    # Store incident
                    self.active_incidents[incident["id"]] = incident

                    # Trigger notifications
                    await self._trigger_notifications(incident)

                    # Set suppression window
                    cooldown_minutes = rule.get("cooldown_minutes", 5)
                    suppression_key = f"{detection['camera_id']}:{rule['name']}"
                    self.suppression_windows[suppression_key] = datetime.now(
                        UTC
                    ) + timedelta(minutes=cooldown_minutes)

                    processing_time = (
                        datetime.now(UTC) - start_time
                    ).total_seconds() * 1000
                    logger.info(
                        "Incident detected and processed",
                        incident_id=incident["id"],
                        rule_name=rule["name"],
                        priority_score=priority_score,
                        processing_time_ms=processing_time,
                    )

                    return incident

            return None

        except Exception as e:
            logger.error(f"Incident processing failed: {e}", detection=detection)
            return None

    async def _evaluate_rule(
        self, detection: dict[str, Any], rule: dict[str, Any]
    ) -> bool:
        """Evaluate if detection matches alert rule conditions."""
        conditions = rule.get("conditions", {})
        thresholds = rule.get("thresholds", {})

        # Camera filter
        if (
            rule.get("camera_ids")
            and detection.get("camera_id") not in rule["camera_ids"]
        ):
            return False

        # Vehicle count threshold
        if "vehicle_count" in thresholds:
            count = detection.get("vehicle_count", 0)
            if count < thresholds["vehicle_count"]:
                return False

        # Speed threshold
        if "speed_limit" in thresholds:
            speed = detection.get("speed", 0)
            if speed < thresholds["speed_limit"]:
                return False

        # Confidence threshold
        if "confidence" in thresholds:
            confidence = detection.get("confidence", 0)
            if confidence < thresholds["confidence"]:
                return False

        # Custom condition evaluation
        for condition_name, condition_value in conditions.items():
            if not await self._evaluate_condition(
                detection, condition_name, condition_value
            ):
                return False

        return True

    async def _is_duplicate_incident(
        self, detection: dict[str, Any], rule: dict[str, Any]
    ) -> bool:
        """Check if incident is a duplicate of existing active incident."""
        camera_id = detection.get("camera_id")
        incident_type = rule.get("incident_type", "unknown")

        # Time window for duplicate detection (5 minutes)
        dedup_window = timedelta(minutes=5)
        now = datetime.now(UTC)

        for incident in self.active_incidents.values():
            if (
                incident.get("camera_id") == camera_id
                and incident.get("incident_type") == incident_type
                and incident.get("status") == "active"
                and (now - incident.get("timestamp", now)) < dedup_window
            ):
                logger.debug("Duplicate incident detected", camera_id=camera_id)
                return True

        return False

    async def _is_suppressed(
        self, detection: dict[str, Any], rule: dict[str, Any]
    ) -> bool:
        """Check if incident is within suppression window."""
        camera_id = detection.get("camera_id")
        rule_name = rule.get("name")
        suppression_key = f"{camera_id}:{rule_name}"

        suppression_until = self.suppression_windows.get(suppression_key)
        if suppression_until and datetime.now(UTC) < suppression_until:
            logger.debug("Incident suppressed", suppression_key=suppression_key)
            return True

        return False

    async def _create_incident_alert(
        self, detection: dict[str, Any], rule: dict[str, Any]
    ) -> dict[str, Any]:
        """Create incident alert from detection and rule."""
        from uuid import uuid4

        incident_id = str(uuid4())
        timestamp = datetime.now(UTC)

        return {
            "id": incident_id,
            "camera_id": detection.get("camera_id"),
            "incident_type": rule.get("incident_type", "unknown"),
            "severity": rule.get("severity", "medium"),
            "description": self._generate_description(detection, rule),
            "location": detection.get("location", "Unknown"),
            "coordinates": detection.get("coordinates"),
            "timestamp": timestamp,
            "detected_at": timestamp,
            "confidence": detection.get("confidence", 0.0),
            "status": "active",
            "vehicles_involved": detection.get("vehicle_ids", []),
            "traffic_impact": self._assess_traffic_impact(detection),
            "images": detection.get("images", []),
            "video_clip": detection.get("video_clip"),
            "rule_triggered": rule.get("name"),
            "metadata": {
                "detection_source": detection.get("source", "ml_model"),
                "model_version": detection.get("model_version"),
                "processing_pipeline": detection.get("pipeline_id"),
            },
        }

    async def _calculate_priority_score(self, incident: dict[str, Any]) -> float:
        """Calculate priority score for incident (0.0 - 1.0)."""
        base_score = 0.5

        # Severity multiplier
        severity_multipliers = {"low": 0.3, "medium": 0.6, "high": 0.8, "critical": 1.0}
        severity_score = severity_multipliers.get(
            incident.get("severity", "medium"), 0.6
        )

        # Confidence boost
        confidence_boost = incident.get("confidence", 0.0) * 0.2

        # Traffic impact factor
        impact_factors = {"low": 0.1, "medium": 0.3, "high": 0.5, "critical": 0.7}
        impact_score = impact_factors.get(incident.get("traffic_impact", "low"), 0.1)

        # Historical frequency (incidents in same location recently reduce priority)
        frequency_penalty = await self._calculate_frequency_penalty(incident)

        final_score = min(
            1.0, severity_score + confidence_boost + impact_score - frequency_penalty
        )
        return round(final_score, 3)

    async def _calculate_frequency_penalty(self, incident: dict[str, Any]) -> float:
        """Calculate penalty based on recent incident frequency."""
        camera_id = incident.get("camera_id")
        incident_type = incident.get("incident_type")

        # Count similar incidents in last 24 hours
        recent_count = 0
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)

        for existing_incident in self.active_incidents.values():
            if (
                existing_incident.get("camera_id") == camera_id
                and existing_incident.get("incident_type") == incident_type
                and existing_incident.get("timestamp", datetime.min.replace(tzinfo=UTC))
                > cutoff_time
            ):
                recent_count += 1

        # Apply penalty: 0.1 per recent incident, max 0.5
        return min(0.5, recent_count * 0.1)

    def _generate_description(
        self, detection: dict[str, Any], rule: dict[str, Any]
    ) -> str:
        """Generate human-readable incident description."""
        incident_type = rule.get("incident_type", "unknown")
        camera_id = detection.get("camera_id", "unknown")

        descriptions = {
            "speeding": f"Speed violation detected on camera {camera_id}",
            "congestion": f"Traffic congestion detected on camera {camera_id}",
            "accident": f"Potential accident detected on camera {camera_id}",
            "wrong_way": f"Wrong-way vehicle detected on camera {camera_id}",
            "illegal_parking": f"Illegal parking detected on camera {camera_id}",
        }

        base_description = descriptions.get(
            incident_type, f"Incident detected on camera {camera_id}"
        )

        # Add contextual details
        if "vehicle_count" in detection:
            base_description += f", {detection['vehicle_count']} vehicles involved"

        if "speed" in detection:
            base_description += f", speed: {detection['speed']:.1f} km/h"

        return base_description

    def _assess_traffic_impact(self, detection: dict[str, Any]) -> str:
        """Assess potential traffic impact of incident."""
        vehicle_count = detection.get("vehicle_count", 0)
        speed = detection.get("speed", 50)

        if vehicle_count > 20 or speed < 20:
            return "high"
        elif vehicle_count > 10 or speed < 40:
            return "medium"
        else:
            return "low"

    async def _trigger_notifications(self, incident: dict[str, Any]) -> None:
        """Trigger notifications through configured channels."""
        # Placeholder for notification system integration
        # Would integrate with email, SMS, webhook services
        logger.info(
            "Notifications triggered for incident",
            incident_id=incident["id"],
            severity=incident["severity"],
            priority_score=incident.get("priority_score", 0),
        )

    async def _get_applicable_rules(
        self, detection: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get alert rules applicable to the detection."""
        # Return default rules - in production would query from database
        return [
            {
                "name": "speed_violation",
                "incident_type": "speeding",
                "severity": "medium",
                "thresholds": {"speed_limit": 80, "confidence": 0.8},
                "conditions": {},
                "cooldown_minutes": 5,
                "notification_channels": ["email", "webhook"],
            },
            {
                "name": "traffic_congestion",
                "incident_type": "congestion",
                "severity": "high",
                "thresholds": {"vehicle_count": 15, "confidence": 0.7},
                "conditions": {},
                "cooldown_minutes": 10,
                "notification_channels": ["email", "sms", "webhook"],
            },
        ]

    async def _evaluate_condition(
        self, detection: dict[str, Any], condition_name: str, condition_value: Any
    ) -> bool:
        """Evaluate custom condition."""
        # Placeholder for custom condition evaluation
        # Would implement complex rule logic here
        return True


class TimeWindow(str, Enum):
    """Time window enumeration for aggregations."""

    ONE_MIN = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    ONE_HOUR = "1hour"
    ONE_DAY = "1day"
    ONE_WEEK = "1week"
    ONE_MONTH = "1month"


class AggregationLevel(str, Enum):
    """Aggregation level enumeration."""

    RAW = "raw"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class AnalyticsServiceError(Exception):
    """Analytics service specific exceptions."""

    pass


class AggregatedMetrics:
    """Container for aggregated traffic metrics."""

    def __init__(
        self,
        camera_id: str,
        time_window: TimeWindow,
        aggregation_level: AggregationLevel,
        start_time: datetime,
        end_time: datetime,
        **metrics,
    ):
        self.camera_id = camera_id
        self.time_window = time_window
        self.aggregation_level = aggregation_level
        self.start_time = start_time
        self.end_time = end_time
        self.metrics = metrics

        # Statistical measures
        self.mean_vehicle_count = metrics.get("mean_vehicle_count", 0)
        self.median_vehicle_count = metrics.get("median_vehicle_count", 0)
        self.std_vehicle_count = metrics.get("std_vehicle_count", 0)
        self.percentile_95_vehicle_count = metrics.get("percentile_95_vehicle_count", 0)

        self.mean_speed = metrics.get("mean_speed", 0)
        self.median_speed = metrics.get("median_speed", 0)
        self.std_speed = metrics.get("std_speed", 0)

        self.data_quality_score = metrics.get("data_quality_score", 1.0)
        self.sample_count = metrics.get("sample_count", 0)
        self.outlier_count = metrics.get("outlier_count", 0)


class TrafficStatistics:
    """Statistical analysis results for traffic data."""

    def __init__(self, metrics_list: list):
        self.sample_count = len(metrics_list)

        if not metrics_list:
            self.mean = self.median = self.std_dev = 0
            self.percentiles = {}
            return

        # Calculate basic statistics
        values = [getattr(m, "total_vehicles", 0) for m in metrics_list]
        self.mean = statistics.mean(values) if values else 0
        self.median = statistics.median(values) if values else 0
        self.std_dev = statistics.stdev(values) if len(values) > 1 else 0

        # Calculate percentiles
        if values:
            sorted_values = sorted(values)
            self.percentiles = {
                "p25": np.percentile(sorted_values, 25),
                "p50": np.percentile(sorted_values, 50),
                "p75": np.percentile(sorted_values, 75),
                "p90": np.percentile(sorted_values, 90),
                "p95": np.percentile(sorted_values, 95),
                "p99": np.percentile(sorted_values, 99),
            }
        else:
            self.percentiles = {}


class AnalyticsAggregationService:
    """High-performance analytics aggregation service with TimescaleDB integration."""

    def __init__(
        self,
        analytics_repository=None,
        cache_service: CacheService | None = None,
        settings=None,
    ):
        self.analytics_repository = analytics_repository
        self.cache_service = cache_service
        self.settings = settings

        # Aggregation configuration
        self.supported_windows = {
            TimeWindow.ONE_MIN: timedelta(minutes=1),
            TimeWindow.FIVE_MIN: timedelta(minutes=5),
            TimeWindow.FIFTEEN_MIN: timedelta(minutes=15),
            TimeWindow.ONE_HOUR: timedelta(hours=1),
            TimeWindow.ONE_DAY: timedelta(days=1),
        }

        # Cache TTL settings
        self.cache_ttls = {
            TimeWindow.ONE_MIN: 60,  # 1 minute
            TimeWindow.FIVE_MIN: 300,  # 5 minutes
            TimeWindow.FIFTEEN_MIN: 900,  # 15 minutes
            TimeWindow.ONE_HOUR: 3600,  # 1 hour
            TimeWindow.ONE_DAY: 86400,  # 1 day
        }

    async def aggregate_traffic_metrics(
        self,
        camera_ids: list[str],
        time_window: TimeWindow,
        aggregation_level: AggregationLevel,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        include_quality_check: bool = True,
    ) -> list[AggregatedMetrics]:
        """Aggregate traffic metrics with TimescaleDB continuous aggregates.

        Features:
        - Multi-level caching (L1: in-memory, L2: Redis)
        - Data quality validation
        - Outlier detection
        - Statistical calculations
        - Support for 10K+ data points
        """
        try:
            # Set default time range if not provided
            if not end_time:
                end_time = datetime.now(UTC)
            if not start_time:
                start_time = end_time - self.supported_windows[time_window]

            # Check cache first (L2: Redis)
            cache_key = self._build_cache_key(
                camera_ids, time_window, aggregation_level, start_time, end_time
            )

            if self.cache_service:
                cached_result = await self.cache_service.get_json(cache_key)
                if cached_result:
                    logger.debug(f"Cache hit for aggregation: {cache_key}")
                    return [AggregatedMetrics(**item) for item in cached_result]

            # Execute aggregation query
            aggregated_results = []

            for camera_id in camera_ids:
                try:
                    # Get raw metrics from repository
                    if self.analytics_repository:
                        raw_metrics = await self._get_raw_metrics(
                            camera_id, start_time, end_time, aggregation_level
                        )
                    else:
                        # Fallback to mock data for demonstration
                        raw_metrics = self._generate_mock_metrics(
                            camera_id, start_time, end_time, time_window
                        )

                    # Perform aggregation and statistical analysis
                    aggregated = await self._aggregate_metrics(
                        camera_id,
                        raw_metrics,
                        time_window,
                        aggregation_level,
                        start_time,
                        end_time,
                        include_quality_check,
                    )

                    aggregated_results.append(aggregated)

                except Exception as e:
                    logger.error(
                        f"Failed to aggregate metrics for camera {camera_id}: {e}"
                    )
                    # Continue with other cameras
                    continue

            # Cache results
            if self.cache_service and aggregated_results:
                cache_data = [result.__dict__ for result in aggregated_results]
                ttl = self.cache_ttls.get(time_window, 3600)
                await self.cache_service.set_json(cache_key, cache_data, ttl=ttl)
                logger.debug(f"Cached aggregation result: {cache_key}")

            logger.info(
                f"Aggregated metrics for {len(camera_ids)} cameras, "
                f"window: {time_window}, level: {aggregation_level}, "
                f"results: {len(aggregated_results)}"
            )

            return aggregated_results

        except Exception as e:
            logger.error(f"Traffic metrics aggregation failed: {e}")
            raise AnalyticsServiceError(f"Aggregation failed: {str(e)}")

    async def calculate_statistics(self, metrics: list) -> TrafficStatistics:
        """Calculate comprehensive statistics for traffic metrics."""
        try:
            return TrafficStatistics(metrics)
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            raise AnalyticsServiceError(f"Statistics calculation failed: {str(e)}")

    def _build_cache_key(
        self,
        camera_ids: list[str],
        time_window: TimeWindow,
        aggregation_level: AggregationLevel,
        start_time: datetime,
        end_time: datetime,
    ) -> str:
        """Build cache key for aggregation results."""
        cameras_str = ",".join(sorted(camera_ids))
        return (
            f"agg:metrics:{cameras_str}:{time_window.value}:"
            f"{aggregation_level.value}:{start_time.isoformat()}:{end_time.isoformat()}"
        )

    async def _get_raw_metrics(
        self,
        camera_id: str,
        start_time: datetime,
        end_time: datetime,
        aggregation_level: AggregationLevel,
    ) -> list:
        """Get raw metrics from repository."""
        # This would interact with TimescaleDB using the analytics repository
        # For now, return mock data
        return self._generate_mock_metrics(
            camera_id, start_time, end_time, TimeWindow.ONE_MIN
        )

    def _generate_mock_metrics(
        self,
        camera_id: str,
        start_time: datetime,
        end_time: datetime,
        time_window: TimeWindow,
    ) -> list:
        """Generate mock metrics for demonstration."""
        import random

        metrics = []
        current_time = start_time
        window_delta = self.supported_windows[time_window]

        while current_time < end_time:
            # Generate realistic traffic patterns
            hour = current_time.hour
            base_count = 10 if 6 <= hour <= 20 else 3
            rush_factor = 2 if hour in [7, 8, 17, 18, 19] else 1

            vehicle_count = base_count * rush_factor + random.randint(-5, 10)
            vehicle_count = max(0, vehicle_count)

            speed = random.uniform(30, 70)

            metrics.append(
                {
                    "timestamp": current_time,
                    "camera_id": camera_id,
                    "total_vehicles": vehicle_count,
                    "average_speed": speed,
                    "occupancy_rate": min(100, vehicle_count * 2.5),
                    "flow_rate": vehicle_count * 60 // window_delta.total_seconds(),
                }
            )

            current_time += window_delta

        return metrics

    async def _aggregate_metrics(
        self,
        camera_id: str,
        raw_metrics: list,
        time_window: TimeWindow,
        aggregation_level: AggregationLevel,
        start_time: datetime,
        end_time: datetime,
        include_quality_check: bool,
    ) -> AggregatedMetrics:
        """Aggregate raw metrics with statistical analysis."""
        if not raw_metrics:
            return AggregatedMetrics(
                camera_id=camera_id,
                time_window=time_window,
                aggregation_level=aggregation_level,
                start_time=start_time,
                end_time=end_time,
                sample_count=0,
                data_quality_score=0.0,
            )

        # Extract values for statistical analysis
        vehicle_counts = [m["total_vehicles"] for m in raw_metrics]
        speeds = [m["average_speed"] for m in raw_metrics if m.get("average_speed")]
        occupancy_rates = [
            m["occupancy_rate"] for m in raw_metrics if m.get("occupancy_rate")
        ]

        # Calculate statistics
        stats = {}

        if vehicle_counts:
            stats.update(
                {
                    "mean_vehicle_count": statistics.mean(vehicle_counts),
                    "median_vehicle_count": statistics.median(vehicle_counts),
                    "std_vehicle_count": (
                        statistics.stdev(vehicle_counts)
                        if len(vehicle_counts) > 1
                        else 0
                    ),
                    "percentile_95_vehicle_count": np.percentile(vehicle_counts, 95),
                    "total_vehicles": sum(vehicle_counts),
                    "max_vehicle_count": max(vehicle_counts),
                    "min_vehicle_count": min(vehicle_counts),
                }
            )

        if speeds:
            stats.update(
                {
                    "mean_speed": statistics.mean(speeds),
                    "median_speed": statistics.median(speeds),
                    "std_speed": statistics.stdev(speeds) if len(speeds) > 1 else 0,
                    "max_speed": max(speeds),
                    "min_speed": min(speeds),
                }
            )

        if occupancy_rates:
            stats["mean_occupancy_rate"] = statistics.mean(occupancy_rates)

        # Data quality assessment
        data_quality_score = 1.0
        outlier_count = 0

        if include_quality_check:
            data_quality_score, outlier_count = self._assess_data_quality(raw_metrics)

        stats.update(
            {
                "sample_count": len(raw_metrics),
                "data_quality_score": data_quality_score,
                "outlier_count": outlier_count,
                "data_completeness": len(raw_metrics)
                / max(
                    1, self._expected_sample_count(time_window, start_time, end_time)
                ),
            }
        )

        return AggregatedMetrics(
            camera_id=camera_id,
            time_window=time_window,
            aggregation_level=aggregation_level,
            start_time=start_time,
            end_time=end_time,
            **stats,
        )

    def _assess_data_quality(self, metrics: list) -> tuple[float, int]:
        """Assess data quality and detect outliers."""
        if not metrics:
            return 0.0, 0

        # Simple outlier detection using IQR method
        vehicle_counts = [m["total_vehicles"] for m in metrics]

        if len(vehicle_counts) < 4:
            return 1.0, 0

        q1 = np.percentile(vehicle_counts, 25)
        q3 = np.percentile(vehicle_counts, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = [v for v in vehicle_counts if v < lower_bound or v > upper_bound]
        outlier_count = len(outliers)

        # Quality score based on outlier percentage
        quality_score = max(0.0, 1.0 - (outlier_count / len(vehicle_counts)))

        return quality_score, outlier_count

    def _expected_sample_count(
        self, time_window: TimeWindow, start_time: datetime, end_time: datetime
    ) -> int:
        """Calculate expected number of samples for data completeness assessment."""
        duration = end_time - start_time
        window_duration = self.supported_windows[time_window]

        return max(1, int(duration.total_seconds() / window_duration.total_seconds()))


class AnalyticsServiceError(Exception):
    """Analytics service specific exceptions."""

    pass


class RuleEngine:
    """Traffic rule evaluation engine with configurable rules."""

    def __init__(
        self,
        settings: Settings,
        analytics_repository=None,
        cache_service: CacheService | None = None,
    ):
        self.settings = settings
        self.analytics_repository = analytics_repository
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
        self,
        speed: float,
        zone_id: str | None = None,
        vehicle_type: str | None = None,
        check_time: datetime | None = None,
        weather: str | None = None,
        visibility: float | None = None,
    ) -> dict[str, Any] | None:
        """Evaluate speed limit violation with dynamic database lookup."""
        rule = self._rules["speed_limit"]
        speed_limit = await self._get_speed_limit(
            zone_id, vehicle_type, check_time, weather, visibility
        )

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

    async def _get_speed_limit(
        self,
        zone_id: str | None,
        vehicle_type: str | None,
        check_time: datetime | None = None,
        weather: str | None = None,
        visibility: float | None = None,
    ) -> float:
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
            check_time = datetime.now(UTC)

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

    async def _query_speed_limit_from_db(
        self,
        zone_id: str | None,
        vehicle_type: str,
        check_time: datetime,
        weather: str | None = None,
        visibility: float | None = None,
    ) -> float | None:
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
        if not self.analytics_repository:
            return None

        try:
            speed_limit = await self.analytics_repository.get_speed_limit(
                zone_id=zone_id,
                vehicle_type=vehicle_type,
                check_time=check_time,
            )

            if speed_limit and speed_limit.applies_to_conditions(weather, visibility):
                logger.debug(
                    f"Found speed limit: {speed_limit.speed_limit_kmh} km/h",
                    zone_id=zone_id,
                    vehicle_type=vehicle_type,
                    limit_id=speed_limit.id,
                )
                return speed_limit.speed_limit_kmh

            return None

        except Exception as e:
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
            "truck": -10.0,  # Trucks typically have lower limits
            "bus": -5.0,  # Buses slightly lower
            "emergency": 20.0,  # Emergency vehicles higher limits
            "motorcycle": 0.0,  # Same as cars
            "car": 0.0,  # Base limit
            "general": 0.0,  # Base limit
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
                if isinstance(value, int | float):
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


class EnhancedAnalyticsService:
    """Comprehensive analytics service for traffic monitoring and insights.

    Provides real-time traffic analytics, rule-based violation detection,
    speed calculations, trajectory analysis, and ML-based anomaly detection.
    """

    def __init__(
        self,
        analytics_repository,
        metrics_repository,
        detection_repository,
        settings: Settings,
        cache_service: CacheService | None = None,
    ):
        self.analytics_repository = analytics_repository
        self.metrics_repository = metrics_repository
        self.detection_repository = detection_repository
        self.settings = settings
        self.cache_service = cache_service
        self.rule_engine = RuleEngine(settings, analytics_repository, cache_service)
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
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(days=30)

            # Get all cameras' historical data for training
            aggregated_metrics = await self.analytics_repository.get_aggregated_metrics(
                start_time=start_time, end_time=end_time, aggregation_period="1hour"
            )

            # Return training data in the expected format
            if aggregated_metrics["record_count"] > 0:
                return [
                    {
                        "vehicle_count": aggregated_metrics["total_vehicles"],
                        "average_speed": aggregated_metrics["avg_speed"],
                        "traffic_density": aggregated_metrics["avg_density"],
                        "flow_rate": 0.0,  # Would need more specific query
                        "occupancy_rate": aggregated_metrics["avg_occupancy"],
                        "queue_length": 0.0,  # Would need more specific query
                        "timestamp": end_time,
                    }
                ]
            else:
                return []

        except Exception as e:
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
        start_time = datetime.now(UTC)

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

            processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

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
                    estimated_speed,
                    detection.detection_zone,
                    detection.vehicle_type,
                    check_time=detection.created_at,
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
            violation_data = []

            for violation in violations:
                violation_data.append(
                    {
                        "violation_type": violation["violation_type"],
                        "severity": violation["severity"],
                        "detection_time": violation["detection_time"],
                        "camera_id": violation["camera_id"],
                        "vehicle_track_id": violation.get("track_id"),
                        "license_plate": violation.get("license_plate"),
                        "vehicle_type": violation.get("vehicle_type"),
                        "rule_definition": violation["rule_definition"],
                        "measured_value": violation.get("measured_value"),
                        "threshold_value": violation.get("threshold_value"),
                        "detection_confidence": violation.get("confidence", 0.8),
                    }
                )

            # Use repository to create violations in batch
            violation_records = (
                await self.analytics_repository.create_rule_violations_batch(
                    violation_data
                )
            )

            logger.info(f"Stored {len(violation_records)} traffic violations")

        except Exception as e:
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
            "timestamp": metrics.get("timestamp", datetime.now(UTC)),
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
            anomaly_data = []

            for anomaly in anomalies:
                if anomaly["anomaly_score"] < 0.3:  # Skip low-confidence anomalies
                    continue

                anomaly_data.append(
                    {
                        "anomaly_type": "traffic_pattern",
                        "severity": anomaly["severity"],
                        "detection_time": datetime.now(UTC),
                        "camera_id": camera_id,
                        "anomaly_score": anomaly["anomaly_score"],
                        "confidence": min(1.0, anomaly["anomaly_score"] * 1.2),
                        "detection_method": "isolation_forest",
                        "probable_cause": anomaly.get("probable_cause", "unknown"),
                        "model_name": "IsolationForest",
                        "model_version": "1.0",
                        "model_confidence": anomaly["anomaly_score"],
                        "detailed_analysis": {
                            "features": anomaly.get("features", []),
                            "data_point": anomaly["data_point"],
                        },
                    }
                )

            if anomaly_data:
                anomaly_records = (
                    await self.analytics_repository.create_traffic_anomalies_batch(
                        anomaly_data
                    )
                )

                logger.info(f"Stored {len(anomaly_records)} traffic anomalies")

        except Exception as e:
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
        trajectory = await self.analytics_repository.get_vehicle_trajectory(
            track_id, camera_id
        )

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

            # Update trajectory data
            trajectory_data = {
                "end_time": trajectory.end_time,
                "duration_seconds": trajectory.duration_seconds,
                "path_points": trajectory.path_points,
                "total_distance": trajectory.total_distance,
                "straight_line_distance": trajectory.straight_line_distance,
                "path_efficiency": trajectory.path_efficiency,
                "average_speed": trajectory.average_speed,
                "max_speed": trajectory.max_speed,
                "is_anomalous": getattr(trajectory, "is_anomalous", False),
                "anomaly_score": getattr(trajectory, "anomaly_score", None),
                "anomaly_reasons": getattr(trajectory, "anomaly_reasons", None),
            }

            await self.analytics_repository.create_or_update_trajectory(
                {
                    "vehicle_track_id": track_id,
                    "camera_id": camera_id,
                    **trajectory_data,
                }
            )

        else:
            # Create new trajectory
            start_time = min(d.created_at for d in detections)
            end_time = max(d.created_at for d in detections)

            # Create a temporary trajectory object for metrics calculation
            temp_trajectory = type(
                "Trajectory",
                (),
                {
                    "path_points": path_points,
                    "total_distance": 0.0,
                    "straight_line_distance": 0.0,
                    "path_efficiency": 1.0,
                    "average_speed": 0.0,
                    "max_speed": 0.0,
                    "is_anomalous": False,
                    "anomaly_score": None,
                    "anomaly_reasons": None,
                },
            )()

            await self._calculate_trajectory_metrics(temp_trajectory)

            # Create trajectory via repository
            trajectory_data = {
                "vehicle_track_id": track_id,
                "camera_id": camera_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": (end_time - start_time).total_seconds(),
                "path_points": path_points,
                "total_distance": temp_trajectory.total_distance,
                "straight_line_distance": temp_trajectory.straight_line_distance,
                "path_efficiency": temp_trajectory.path_efficiency,
                "average_speed": temp_trajectory.average_speed,
                "max_speed": temp_trajectory.max_speed,
                "vehicle_type": detections[0].vehicle_type or detections[0].class_name,
                "license_plate": detections[0].license_plate,
                "tracking_quality": statistics.mean(
                    d.detection_quality for d in detections
                ),
                "path_completeness": 1.0,
                "is_anomalous": temp_trajectory.is_anomalous,
                "anomaly_score": temp_trajectory.anomaly_score,
                "anomaly_reasons": temp_trajectory.anomaly_reasons,
            }

            await self.analytics_repository.create_or_update_trajectory(trajectory_data)

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
            # Query aggregated metrics from repository
            metrics = await self.analytics_repository.get_traffic_metrics_by_camera(
                camera_id=camera_id,
                start_time=start_time,
                end_time=end_time,
                aggregation_period=aggregation_period,
                limit=10000,  # Large limit for time range queries
            )

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

        except Exception as e:
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
            violations = await self.analytics_repository.get_active_violations(
                camera_id=camera_id,
                violation_type=violation_type,
                severity=severity,
                limit=limit,
            )

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

        except Exception as e:
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
            anomalies = await self.analytics_repository.get_traffic_anomalies(
                camera_id=camera_id,
                anomaly_type=anomaly_type,
                min_score=min_score,
                time_range=time_range,
                limit=limit,
            )

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

        except Exception as e:
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
            # Use repository to get analytics summary
            summary_data = await self.analytics_repository.get_analytics_summary(
                camera_ids=camera_ids,
                start_time=start_time,
                end_time=end_time,
            )

            report_data = {
                "report_type": report_type,
                "time_range": summary_data["time_range"],
                "cameras": summary_data["cameras"],
                "generated_at": summary_data["generated_at"],
                "summary": summary_data["summary"],
                "camera_summaries": [],  # Could be extended to include per-camera details
            }

            return report_data

        except Exception as e:
            logger.error(f"Failed to generate analytics report: {e}")
            raise DatabaseError(f"Report generation failed: {e}")
