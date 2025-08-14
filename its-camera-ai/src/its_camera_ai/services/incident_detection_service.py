"""Incident Detection Service with proper DI and single responsibility.

This service handles incident detection, alert generation, and notification
management with deduplication and suppression logic.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from ..core.config import Settings
from ..core.logging import get_logger
from ..repositories.alert_repository import AlertRepository
from .analytics_dtos import (
    AlertRule,
    DetectionData,
    IncidentAlertDTO,
    IncidentEvidence,
    IncidentLocation,
    IncidentMetadata,
    IncidentStatus,
    TrafficImpact,
)
from .cache import CacheService

logger = get_logger(__name__)


class IncidentDetectionService:
    """Incident Detection and Alert System with proper DI.

    Processes ML model outputs, applies business rules, and generates
    prioritized alerts through multiple channels with deduplication.
    """

    def __init__(
        self,
        alert_repository: AlertRepository,
        cache_service: CacheService,
        settings: Settings,
    ):
        """Initialize with injected dependencies.

        Args:
            alert_repository: Repository for alert data access
            cache_service: Redis cache service
            settings: Application settings
        """
        self.alert_repository = alert_repository
        self.cache_service = cache_service
        self.settings = settings

        # In-memory state (could be moved to Redis for distributed systems)
        self.active_incidents: dict[str, IncidentAlertDTO] = {}
        self.suppression_windows: dict[str, datetime] = {}

    async def process_detection(
        self,
        detection_data: DetectionData,
        rules: list[AlertRule] | None = None,
    ) -> IncidentAlertDTO | None:
        """Process detection data and generate incident alerts.

        Args:
            detection_data: Detection data from ML models
            rules: Optional list of alert rules to apply

        Returns:
            Incident alert DTO if incident detected, None otherwise
        """
        start_time = datetime.now(UTC)

        try:
            # Get applicable rules
            applicable_rules = rules or await self._get_applicable_rules(detection_data)

            for rule in applicable_rules:
                if not await self._evaluate_rule(detection_data, rule):
                    continue

                # Check for duplicates
                if await self._is_duplicate_incident(detection_data, rule):
                    logger.debug(
                        f"Duplicate incident detected for camera {detection_data.camera_id}"
                    )
                    continue

                # Check suppression windows
                if await self._is_suppressed(detection_data, rule):
                    logger.debug(
                        f"Incident suppressed for camera {detection_data.camera_id}"
                    )
                    continue

                # Create incident alert
                incident = await self._create_incident_alert(detection_data, rule)

                # Calculate priority score
                incident.priority_score = await self._calculate_priority_score(incident)

                # Store incident
                self.active_incidents[incident.id] = incident
                await self._persist_incident(incident)

                # Trigger notifications
                await self._trigger_notifications(incident)

                # Set suppression window
                await self._set_suppression_window(detection_data, rule)

                processing_time = (
                    datetime.now(UTC) - start_time
                ).total_seconds() * 1000
                logger.info(
                    "Incident detected and processed",
                    incident_id=incident.id,
                    rule_name=rule.name,
                    priority_score=incident.priority_score,
                    processing_time_ms=processing_time,
                )

                return incident

            return None

        except Exception as e:
            logger.error(f"Incident processing failed: {e}")
            return None

    async def _evaluate_rule(
        self, detection_data: DetectionData, rule: AlertRule
    ) -> bool:
        """Evaluate if detection matches alert rule conditions.

        Args:
            detection_data: Detection data
            rule: Alert rule to evaluate

        Returns:
            True if rule conditions are met
        """
        # Camera filter
        if rule.camera_ids and detection_data.camera_id not in rule.camera_ids:
            return False

        # Vehicle count threshold
        if "vehicle_count" in rule.thresholds:
            if detection_data.vehicle_count < rule.thresholds["vehicle_count"]:
                return False

        # Confidence threshold
        if "confidence" in rule.thresholds:
            if detection_data.confidence < rule.thresholds["confidence"]:
                return False

        # Evaluate custom conditions
        for condition_name, condition_value in rule.conditions.items():
            if not await self._evaluate_condition(
                detection_data, condition_name, condition_value
            ):
                return False

        return True

    async def _is_duplicate_incident(
        self, detection_data: DetectionData, rule: AlertRule
    ) -> bool:
        """Check if incident is a duplicate of existing active incident.

        Args:
            detection_data: Detection data
            rule: Alert rule

        Returns:
            True if duplicate
        """
        camera_id = detection_data.camera_id
        incident_type = rule.incident_type

        # Time window for duplicate detection
        dedup_window = timedelta(minutes=5)
        now = datetime.now(UTC)

        for incident in self.active_incidents.values():
            if (
                incident.location.camera_id == camera_id
                and incident.incident_type == incident_type
                and incident.status == IncidentStatus.ACTIVE
                and (now - incident.timestamp) < dedup_window
            ):
                return True

        return False

    async def _is_suppressed(
        self, detection_data: DetectionData, rule: AlertRule
    ) -> bool:
        """Check if incident is within suppression window.

        Args:
            detection_data: Detection data
            rule: Alert rule

        Returns:
            True if suppressed
        """
        suppression_key = f"{detection_data.camera_id}:{rule.name}"

        # Check in-memory suppression
        suppression_until = self.suppression_windows.get(suppression_key)
        if suppression_until and datetime.now(UTC) < suppression_until:
            return True

        # Check Redis cache for distributed suppression
        cache_key = f"incident:suppression:{suppression_key}"
        try:
            cached_suppression = await self.cache_service.get(cache_key)
            if cached_suppression:
                return True
        except Exception as e:
            logger.warning(f"Failed to check suppression cache: {e}")

        return False

    async def _create_incident_alert(
        self, detection_data: DetectionData, rule: AlertRule
    ) -> IncidentAlertDTO:
        """Create incident alert from detection and rule.

        Args:
            detection_data: Detection data
            rule: Alert rule

        Returns:
            Incident alert DTO
        """
        incident_id = str(uuid4())
        timestamp = datetime.now(UTC)

        # Create location info
        location = IncidentLocation(
            camera_id=detection_data.camera_id,
            location_description=f"Camera {detection_data.camera_id} location",
            # Coordinates would come from camera configuration
            coordinates=None,
            zone_id=None,
            lane_id=None,
        )

        # Create evidence
        evidence = IncidentEvidence(
            images=[],  # Would be populated from detection data
            video_clip=None,
            detection_frames=[],
            confidence_scores=[detection_data.confidence],
        )

        # Create metadata
        metadata = IncidentMetadata(
            detection_source=detection_data.source,
            model_version=detection_data.model_version,
            processing_pipeline=detection_data.pipeline_id,
            detection_method="ml_detection",
        )

        # Generate description
        description = self._generate_description(detection_data, rule)

        # Assess traffic impact
        traffic_impact = self._assess_traffic_impact(detection_data)

        return IncidentAlertDTO(
            id=incident_id,
            incident_type=rule.incident_type,
            severity=rule.severity,
            description=description,
            location=location,
            timestamp=timestamp,
            detected_at=timestamp,
            confidence=detection_data.confidence,
            status=IncidentStatus.ACTIVE,
            vehicles_involved=[],  # Would be extracted from detection data
            traffic_impact=traffic_impact,
            evidence=evidence,
            metadata=metadata,
            priority_score=0.5,  # Will be calculated
            rule_triggered=rule.name,
        )

    async def _calculate_priority_score(self, incident: IncidentAlertDTO) -> float:
        """Calculate priority score for incident (0.0 - 1.0).

        Args:
            incident: Incident alert

        Returns:
            Priority score
        """
        base_score = 0.5

        # Severity multiplier
        severity_multipliers = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 1.0,
        }
        severity_score = severity_multipliers.get(incident.severity, 0.6)

        # Confidence boost
        confidence_boost = incident.confidence * 0.2

        # Traffic impact factor
        impact_factors = {
            TrafficImpact.LOW: 0.1,
            TrafficImpact.MEDIUM: 0.3,
            TrafficImpact.HIGH: 0.5,
            TrafficImpact.CRITICAL: 0.7,
        }
        impact_score = impact_factors.get(incident.traffic_impact, 0.1)

        # Historical frequency penalty
        frequency_penalty = await self._calculate_frequency_penalty(incident)

        final_score = min(
            1.0, severity_score + confidence_boost + impact_score - frequency_penalty
        )
        return round(final_score, 3)

    async def _calculate_frequency_penalty(self, incident: IncidentAlertDTO) -> float:
        """Calculate penalty based on recent incident frequency.

        Args:
            incident: Incident alert

        Returns:
            Frequency penalty
        """
        camera_id = incident.location.camera_id
        incident_type = incident.incident_type

        # Count similar incidents in last 24 hours
        recent_count = 0
        cutoff_time = datetime.now(UTC) - timedelta(hours=24)

        for existing_incident in self.active_incidents.values():
            if (
                existing_incident.location.camera_id == camera_id
                and existing_incident.incident_type == incident_type
                and existing_incident.timestamp > cutoff_time
            ):
                recent_count += 1

        # Apply penalty: 0.1 per recent incident, max 0.5
        return min(0.5, recent_count * 0.1)

    def _generate_description(
        self, detection_data: DetectionData, rule: AlertRule
    ) -> str:
        """Generate human-readable incident description.

        Args:
            detection_data: Detection data
            rule: Alert rule

        Returns:
            Description string
        """
        incident_type = rule.incident_type
        camera_id = detection_data.camera_id

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
        if detection_data.vehicle_count > 0:
            base_description += f", {detection_data.vehicle_count} vehicles involved"

        return base_description

    def _assess_traffic_impact(self, detection_data: DetectionData) -> TrafficImpact:
        """Assess potential traffic impact of incident.

        Args:
            detection_data: Detection data

        Returns:
            Traffic impact level
        """
        vehicle_count = detection_data.vehicle_count

        if vehicle_count > 20:
            return TrafficImpact.HIGH
        elif vehicle_count > 10:
            return TrafficImpact.MEDIUM
        else:
            return TrafficImpact.LOW

    async def _persist_incident(self, incident: IncidentAlertDTO) -> None:
        """Persist incident to database.

        Args:
            incident: Incident alert
        """
        try:
            # Convert DTO to repository format
            alert_data = {
                "alert_id": incident.id,
                "camera_id": incident.location.camera_id,
                "alert_type": incident.incident_type,
                "severity": incident.severity,
                "description": incident.description,
                "detected_at": incident.detected_at,
                "confidence": incident.confidence,
                "status": incident.status.value,
                "priority_score": incident.priority_score,
                "metadata": {
                    "location": incident.location.__dict__,
                    "evidence": incident.evidence.__dict__,
                    "metadata": incident.metadata.__dict__,
                },
            }

            await self.alert_repository.create_alert(alert_data)

        except Exception as e:
            logger.error(f"Failed to persist incident: {e}")

    async def _trigger_notifications(self, incident: IncidentAlertDTO) -> None:
        """Trigger notifications through configured channels.

        Args:
            incident: Incident alert
        """
        # This would integrate with notification services
        # For now, just log
        logger.info(
            "Notifications triggered for incident",
            incident_id=incident.id,
            severity=incident.severity,
            priority_score=incident.priority_score,
        )

    async def _set_suppression_window(
        self, detection_data: DetectionData, rule: AlertRule
    ) -> None:
        """Set suppression window for rule.

        Args:
            detection_data: Detection data
            rule: Alert rule
        """
        suppression_key = f"{detection_data.camera_id}:{rule.name}"
        suppression_until = datetime.now(UTC) + timedelta(minutes=rule.cooldown_minutes)

        # Set in-memory suppression
        self.suppression_windows[suppression_key] = suppression_until

        # Set in Redis for distributed suppression
        cache_key = f"incident:suppression:{suppression_key}"
        try:
            await self.cache_service.set(cache_key, "1", ttl=rule.cooldown_minutes * 60)
        except Exception as e:
            logger.warning(f"Failed to set suppression cache: {e}")

    async def _get_applicable_rules(
        self, detection_data: DetectionData
    ) -> list[AlertRule]:
        """Get alert rules applicable to the detection.

        Args:
            detection_data: Detection data

        Returns:
            List of applicable alert rules
        """
        # In production, this would query from database
        # For now, return default rules
        return [
            AlertRule(
                name="speed_violation",
                incident_type="speeding",
                severity="medium",
                thresholds={"speed_limit": 80, "confidence": 0.8},
                conditions={},
                cooldown_minutes=5,
                notification_channels=["email", "webhook"],
            ),
            AlertRule(
                name="traffic_congestion",
                incident_type="congestion",
                severity="high",
                thresholds={"vehicle_count": 15, "confidence": 0.7},
                conditions={},
                cooldown_minutes=10,
                notification_channels=["email", "sms", "webhook"],
            ),
        ]

    async def _evaluate_condition(
        self, detection_data: DetectionData, condition_name: str, condition_value: any
    ) -> bool:
        """Evaluate custom condition.

        Args:
            detection_data: Detection data
            condition_name: Condition name
            condition_value: Condition value

        Returns:
            True if condition is met
        """
        # Implement custom condition logic here
        # For now, return True
        return True

    async def get_active_incidents(
        self, camera_id: str | None = None, limit: int = 100
    ) -> list[IncidentAlertDTO]:
        """Get active incidents with optional filtering.

        Args:
            camera_id: Optional camera ID filter
            limit: Maximum number of results

        Returns:
            List of active incidents
        """
        incidents = list(self.active_incidents.values())

        # Filter by camera if specified
        if camera_id:
            incidents = [i for i in incidents if i.location.camera_id == camera_id]

        # Filter only active incidents
        incidents = [i for i in incidents if i.status == IncidentStatus.ACTIVE]

        # Sort by priority and limit
        incidents.sort(key=lambda x: x.priority_score, reverse=True)
        return incidents[:limit]

    async def resolve_incident(
        self, incident_id: str, resolved_by: str, notes: str | None = None
    ) -> bool:
        """Resolve an active incident.

        Args:
            incident_id: Incident ID
            resolved_by: User ID who resolved it
            notes: Optional resolution notes

        Returns:
            True if successfully resolved
        """
        if incident_id not in self.active_incidents:
            return False

        incident = self.active_incidents[incident_id]
        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.now(UTC)
        incident.resolved_by = resolved_by
        incident.notes = notes

        # Update in repository
        try:
            await self.alert_repository.update_alert_status(
                incident_id, IncidentStatus.RESOLVED.value, resolved_by
            )
            return True
        except Exception as e:
            logger.error(f"Failed to resolve incident: {e}")
            return False
