"""Traffic Rule Service with proper DI and repository integration.

This service evaluates traffic rules, detects violations, and manages
speed limits with zone-based configurations.
"""

from datetime import UTC, datetime
from typing import Any

from ..core.config import Settings
from ..core.logging import get_logger
from ..models.analytics import ViolationType
from ..repositories.analytics_repository import AnalyticsRepository
from .analytics_dtos import (
    DetectionData,
    RuleDefinition,
    SpeedLimitInfo,
    ViolationRecord,
)
from .cache import CacheService

logger = get_logger(__name__)


class TrafficRuleService:
    """Traffic rule evaluation service with proper DI.

    Evaluates traffic rules, detects violations, and manages
    zone-based speed limits with weather adjustments.
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

        # Default rule configurations (would come from database in production)
        self.default_rules = self._initialize_default_rules()

        # Weather adjustment factors
        self.weather_adjustments = {
            "clear": 1.0,
            "rain": 0.8,
            "heavy_rain": 0.7,
            "snow": 0.6,
            "fog": 0.7,
            "ice": 0.5,
        }

    def _initialize_default_rules(self) -> dict[ViolationType, RuleDefinition]:
        """Initialize default traffic rules.

        Returns:
            Dictionary of default rule definitions
        """
        return {
            ViolationType.SPEEDING: RuleDefinition(
                rule_type=ViolationType.SPEEDING,
                speed_limit=80.0,
                tolerance=10.0,
            ),
            ViolationType.RED_LIGHT: RuleDefinition(
                rule_type=ViolationType.RED_LIGHT,
            ),
            ViolationType.WRONG_WAY: RuleDefinition(
                rule_type=ViolationType.WRONG_WAY,
            ),
            ViolationType.ILLEGAL_TURN: RuleDefinition(
                rule_type=ViolationType.ILLEGAL_TURN,
            ),
            ViolationType.STOP_VIOLATION: RuleDefinition(
                rule_type=ViolationType.STOP_VIOLATION,
            ),
            ViolationType.ILLEGAL_PARKING: RuleDefinition(
                rule_type=ViolationType.ILLEGAL_PARKING,
            ),
            ViolationType.CONGESTION: RuleDefinition(
                rule_type=ViolationType.CONGESTION,
            ),
        }

    async def evaluate_violations(
        self,
        detection_data: DetectionData,
        zone_id: str | None = None,
        weather_condition: str = "clear",
    ) -> list[ViolationRecord]:
        """Evaluate traffic violations from detection data.

        Args:
            detection_data: Detection data from ML models
            zone_id: Traffic zone identifier
            weather_condition: Current weather condition

        Returns:
            List of detected violations
        """
        violations = []

        try:
            # Process each detection
            for detection in detection_data.detections:
                # Check speed violations
                speed_violation = await self._check_speed_violation(
                    detection, zone_id, weather_condition, detection_data.camera_id
                )
                if speed_violation:
                    violations.append(speed_violation)

                # Check congestion
                congestion_violation = await self._check_congestion_violation(
                    detection_data, zone_id
                )
                if congestion_violation:
                    violations.append(congestion_violation)

                # Check wrong way
                wrong_way_violation = await self._check_wrong_way_violation(
                    detection, detection_data.camera_id
                )
                if wrong_way_violation:
                    violations.append(wrong_way_violation)

                # Check illegal parking
                parking_violation = await self._check_illegal_parking_violation(
                    detection, zone_id, detection_data.camera_id
                )
                if parking_violation:
                    violations.append(parking_violation)

                # Check stop violations
                stop_violation = await self._check_stop_violation(
                    detection, zone_id, detection_data.camera_id
                )
                if stop_violation:
                    violations.append(stop_violation)

                # Check signal violations
                signal_violation = await self._check_signal_violation(
                    detection, zone_id, detection_data.camera_id
                )
                if signal_violation:
                    violations.append(signal_violation)

            # Store violations in repository if any
            if violations:
                await self._persist_violations(violations)

            logger.info(
                f"Evaluated {len(detection_data.detections)} detections, "
                f"found {len(violations)} violations"
            )

            return violations

        except Exception as e:
            logger.error(f"Violation evaluation failed: {e}")
            return []

    async def _check_speed_violation(
        self,
        detection: dict[str, Any],
        zone_id: str | None,
        weather_condition: str,
        camera_id: str,
    ) -> ViolationRecord | None:
        """Check for speed violations.

        Args:
            detection: Single detection data
            zone_id: Zone identifier
            weather_condition: Weather condition
            camera_id: Camera identifier

        Returns:
            Violation record if violation detected
        """
        speed = detection.get("speed")
        if not speed or speed <= 0:
            return None

        # Get speed limit for zone
        speed_limit_info = await self._get_speed_limit(
            zone_id, detection.get("vehicle_type", "car")
        )

        # Apply weather adjustments
        adjusted_limit = self._apply_weather_adjustment(
            speed_limit_info.speed_limit_kmh, weather_condition
        )

        # Check if violation with tolerance
        tolerance = speed_limit_info.speed_limit_kmh * 0.1  # 10% tolerance
        if speed > adjusted_limit + tolerance:
            excess = speed - adjusted_limit
            severity = self._severity_from_excess(excess, adjusted_limit)

            return ViolationRecord(
                violation_type=ViolationType.SPEEDING,
                severity=severity,
                measured_value=speed,
                threshold_value=adjusted_limit,
                excess_amount=excess,
                confidence=detection.get("confidence", 0.8),
                detection_id=detection.get("id"),
                camera_id=camera_id,
                track_id=detection.get("track_id"),
                license_plate=detection.get("license_plate"),
                detection_time=datetime.now(UTC),
                vehicle_type=detection.get("vehicle_type", "car"),
                rule_definition=RuleDefinition(
                    rule_type=ViolationType.SPEEDING,
                    speed_limit=speed_limit_info.speed_limit_kmh,
                    tolerance=tolerance,
                    zone_id=zone_id,
                    weather_adjustments={weather_condition: adjusted_limit},
                ),
            )

        return None

    async def _check_congestion_violation(
        self, detection_data: DetectionData, zone_id: str | None
    ) -> ViolationRecord | None:
        """Check for traffic congestion.

        Args:
            detection_data: Detection data
            zone_id: Zone identifier

        Returns:
            Violation record if congestion detected
        """
        vehicle_count = detection_data.vehicle_count

        # Get congestion threshold from cache or use default
        cache_key = (
            f"congestion_threshold:{zone_id}"
            if zone_id
            else "congestion_threshold:default"
        )
        threshold = await self.cache_service.get(cache_key)

        if not threshold:
            threshold = 20  # Default threshold
            await self.cache_service.set(cache_key, threshold, ttl=3600)
        else:
            threshold = int(threshold)

        if vehicle_count > threshold:
            severity = "high" if vehicle_count > threshold * 1.5 else "medium"

            return ViolationRecord(
                violation_type=ViolationType.CONGESTION,
                severity=severity,
                measured_value=float(vehicle_count),
                threshold_value=float(threshold),
                excess_amount=float(vehicle_count - threshold),
                confidence=0.9,
                camera_id=detection_data.camera_id,
                detection_time=datetime.now(UTC),
                rule_definition=RuleDefinition(
                    rule_type=ViolationType.CONGESTION,
                    zone_id=zone_id,
                ),
            )

        return None

    async def _check_wrong_way_violation(
        self, detection: dict[str, Any], camera_id: str
    ) -> ViolationRecord | None:
        """Check for wrong-way driving.

        Args:
            detection: Single detection data
            camera_id: Camera identifier

        Returns:
            Violation record if wrong-way detected
        """
        direction = detection.get("direction")
        expected_direction = detection.get("expected_direction")

        if direction and expected_direction and direction != expected_direction:
            # Check if it's opposite direction (180 degrees)
            if abs(direction - expected_direction) > 150:
                return ViolationRecord(
                    violation_type=ViolationType.WRONG_WAY,
                    severity="critical",
                    measured_value=direction,
                    threshold_value=expected_direction,
                    confidence=detection.get("confidence", 0.8),
                    detection_id=detection.get("id"),
                    camera_id=camera_id,
                    track_id=detection.get("track_id"),
                    license_plate=detection.get("license_plate"),
                    detection_time=datetime.now(UTC),
                    vehicle_type=detection.get("vehicle_type"),
                    rule_definition=RuleDefinition(
                        rule_type=ViolationType.WRONG_WAY,
                    ),
                )

        return None

    async def _check_illegal_parking_violation(
        self, detection: dict[str, Any], zone_id: str | None, camera_id: str
    ) -> ViolationRecord | None:
        """Check for illegal parking.

        Args:
            detection: Single detection data
            zone_id: Zone identifier
            camera_id: Camera identifier

        Returns:
            Violation record if illegal parking detected
        """
        if detection.get("status") != "parked":
            return None

        # Check if parking is allowed in zone
        no_parking_zone = await self._is_no_parking_zone(zone_id)

        if no_parking_zone:
            duration = detection.get("duration", 0)
            severity = "high" if duration > 300 else "medium"  # 5 minutes

            return ViolationRecord(
                violation_type=ViolationType.ILLEGAL_PARKING,
                severity=severity,
                measured_value=duration,
                threshold_value=0,
                confidence=detection.get("confidence", 0.8),
                detection_id=detection.get("id"),
                camera_id=camera_id,
                track_id=detection.get("track_id"),
                license_plate=detection.get("license_plate"),
                detection_time=datetime.now(UTC),
                vehicle_type=detection.get("vehicle_type"),
                rule_definition=RuleDefinition(
                    rule_type=ViolationType.ILLEGAL_PARKING,
                    zone_id=zone_id,
                ),
            )

        return None

    async def _check_stop_violation(
        self, detection: dict[str, Any], zone_id: str | None, camera_id: str
    ) -> ViolationRecord | None:
        """Check for stop sign violations.

        Args:
            detection: Single detection data
            zone_id: Zone identifier
            camera_id: Camera identifier

        Returns:
            Violation record if stop violation detected
        """
        at_stop_sign = detection.get("at_stop_sign", False)
        stopped = detection.get("stopped", False)

        if at_stop_sign and not stopped:
            return ViolationRecord(
                violation_type=ViolationType.STOP_VIOLATION,
                severity="high",
                measured_value=0,  # Did not stop
                threshold_value=1,  # Should stop
                confidence=detection.get("confidence", 0.8),
                detection_id=detection.get("id"),
                camera_id=camera_id,
                track_id=detection.get("track_id"),
                license_plate=detection.get("license_plate"),
                detection_time=datetime.now(UTC),
                vehicle_type=detection.get("vehicle_type"),
                rule_definition=RuleDefinition(
                    rule_type=ViolationType.STOP_VIOLATION,
                    zone_id=zone_id,
                ),
            )

        return None

    async def _check_signal_violation(
        self, detection: dict[str, Any], zone_id: str | None, camera_id: str
    ) -> ViolationRecord | None:
        """Check for traffic signal violations.

        Args:
            detection: Single detection data
            zone_id: Zone identifier
            camera_id: Camera identifier

        Returns:
            Violation record if signal violation detected
        """
        signal_state = detection.get("signal_state")
        vehicle_action = detection.get("action")

        if signal_state == "red" and vehicle_action in ["crossing", "entering"]:
            return ViolationRecord(
                violation_type=ViolationType.RED_LIGHT,
                severity="critical",
                measured_value=1,  # Crossed on red
                threshold_value=0,  # Should not cross
                confidence=detection.get("confidence", 0.8),
                detection_id=detection.get("id"),
                camera_id=camera_id,
                track_id=detection.get("track_id"),
                license_plate=detection.get("license_plate"),
                detection_time=datetime.now(UTC),
                vehicle_type=detection.get("vehicle_type"),
                rule_definition=RuleDefinition(
                    rule_type=ViolationType.RED_LIGHT,
                    zone_id=zone_id,
                ),
            )

        return None

    async def _get_speed_limit(
        self, zone_id: str | None, vehicle_type: str
    ) -> SpeedLimitInfo:
        """Get speed limit for zone and vehicle type.

        Args:
            zone_id: Zone identifier
            vehicle_type: Type of vehicle

        Returns:
            Speed limit information
        """
        # Try cache first
        cache_key = (
            f"speed_limit:{zone_id}:{vehicle_type}"
            if zone_id
            else f"speed_limit:default:{vehicle_type}"
        )
        cached_limit = await self.cache_service.get(cache_key)

        if cached_limit:
            return SpeedLimitInfo(
                zone_id=zone_id or "default",
                vehicle_type=vehicle_type,
                speed_limit_kmh=float(cached_limit),
            )

        # Query from repository
        try:
            # This would query from database in production
            # For now, use defaults based on vehicle type
            default_limits = {
                "car": 80.0,
                "truck": 60.0,
                "bus": 60.0,
                "motorcycle": 80.0,
                "bicycle": 30.0,
            }

            speed_limit = default_limits.get(vehicle_type, 80.0)

            # Cache the result
            await self.cache_service.set(cache_key, speed_limit, ttl=3600)

            return SpeedLimitInfo(
                zone_id=zone_id or "default",
                vehicle_type=vehicle_type,
                speed_limit_kmh=speed_limit,
            )

        except Exception as e:
            logger.warning(f"Failed to get speed limit: {e}")
            return SpeedLimitInfo(
                zone_id=zone_id or "default",
                vehicle_type=vehicle_type,
                speed_limit_kmh=80.0,  # Default fallback
            )

    def _apply_weather_adjustment(
        self, speed_limit: float, weather_condition: str
    ) -> float:
        """Apply weather-based speed limit adjustments.

        Args:
            speed_limit: Base speed limit
            weather_condition: Weather condition

        Returns:
            Adjusted speed limit
        """
        factor = self.weather_adjustments.get(weather_condition, 1.0)
        return speed_limit * factor

    def _severity_from_excess(self, excess: float, limit: float) -> str:
        """Determine violation severity from excess amount.

        Args:
            excess: Excess amount over limit
            limit: Speed limit

        Returns:
            Severity level
        """
        percentage = (excess / limit) * 100

        if percentage < 10:
            return "low"
        elif percentage < 25:
            return "medium"
        elif percentage < 50:
            return "high"
        else:
            return "critical"

    async def _is_no_parking_zone(self, zone_id: str | None) -> bool:
        """Check if zone is a no-parking zone.

        Args:
            zone_id: Zone identifier

        Returns:
            True if no-parking zone
        """
        if not zone_id:
            return False

        # Check cache
        cache_key = f"no_parking_zone:{zone_id}"
        cached_result = await self.cache_service.get(cache_key)

        if cached_result is not None:
            return cached_result == "1"

        # In production, query from database
        # For now, return False
        result = False

        # Cache result
        await self.cache_service.set(cache_key, "1" if result else "0", ttl=3600)

        return result

    async def _persist_violations(self, violations: list[ViolationRecord]) -> None:
        """Persist violations to repository.

        Args:
            violations: List of violations to persist
        """
        try:
            for violation in violations:
                violation_data = {
                    "violation_type": violation.violation_type.value,
                    "severity": violation.severity,
                    "camera_id": violation.camera_id,
                    "detection_time": violation.detection_time,
                    "confidence": violation.confidence,
                    "measured_value": violation.measured_value,
                    "threshold_value": violation.threshold_value,
                    "excess_amount": violation.excess_amount,
                    "track_id": violation.track_id,
                    "license_plate": violation.license_plate,
                    "vehicle_type": violation.vehicle_type,
                }

                await self.analytics_repository.create_violation(violation_data)

            logger.info(f"Persisted {len(violations)} violations")

        except Exception as e:
            logger.error(f"Failed to persist violations: {e}")

    async def batch_evaluate(
        self,
        detection_batch: list[DetectionData],
        zone_id: str | None = None,
        weather_condition: str = "clear",
    ) -> dict[str, list[ViolationRecord]]:
        """Evaluate violations for multiple detections.

        Args:
            detection_batch: Batch of detection data
            zone_id: Zone identifier
            weather_condition: Weather condition

        Returns:
            Dictionary mapping camera_id to violations
        """
        results = {}

        for detection_data in detection_batch:
            violations = await self.evaluate_violations(
                detection_data, zone_id, weather_condition
            )

            if violations:
                results[detection_data.camera_id] = violations

        return results

    async def get_active_rules(
        self, zone_id: str | None = None
    ) -> list[RuleDefinition]:
        """Get active traffic rules for a zone.

        Args:
            zone_id: Zone identifier

        Returns:
            List of active rule definitions
        """
        # In production, query from database
        # For now, return default rules
        return list(self.default_rules.values())

    async def update_speed_limit(
        self,
        zone_id: str,
        vehicle_type: str,
        new_limit: float,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
    ) -> bool:
        """Update speed limit for a zone.

        Args:
            zone_id: Zone identifier
            vehicle_type: Vehicle type
            new_limit: New speed limit
            valid_from: Start time for limit
            valid_until: End time for limit

        Returns:
            True if successfully updated
        """
        try:
            # Update in repository (would update database in production)
            speed_limit_info = SpeedLimitInfo(
                zone_id=zone_id,
                vehicle_type=vehicle_type,
                speed_limit_kmh=new_limit,
                valid_from=valid_from,
                valid_until=valid_until,
            )

            # Invalidate cache
            cache_key = f"speed_limit:{zone_id}:{vehicle_type}"
            await self.cache_service.delete(cache_key)

            logger.info(
                f"Updated speed limit for zone {zone_id}, "
                f"vehicle type {vehicle_type} to {new_limit} km/h"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to update speed limit: {e}")
            return False
