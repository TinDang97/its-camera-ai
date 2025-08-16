"""Incident Management Service with proper DI and comprehensive incident handling.

This service handles incident CRUD operations, filtering, status management,
database integration, and incident lifecycle management.
"""

import random
from datetime import datetime, timedelta
from uuid import uuid4

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.exceptions import (
    IncidentManagementError,
    ValidationError,
)
from ..api.schemas.analytics import (
    IncidentAlert,
    IncidentType,
    Severity,
)
from ..api.schemas.common import PaginatedResponse
from ..core.config import Settings
from ..core.logging import get_logger
from ..models.analytics import RuleViolation, TrafficAnomaly
from ..models.database import DatabaseManager
from ..repositories.alert_repository import AlertRepository
from .cache import CacheService

logger = get_logger(__name__)


class IncidentManagementService:
    """Incident Management Service with comprehensive incident handling.

    Handles incident CRUD operations, database integration, filtering,
    status management, and cache optimization.
    """

    def __init__(
        self,
        alert_repository: AlertRepository,
        database_manager: DatabaseManager,
        cache_service: CacheService,
        settings: Settings,
    ):
        """Initialize with injected dependencies.

        Args:
            alert_repository: Repository for alert/incident data access
            database_manager: Database manager for session lifecycle
            cache_service: Redis cache service for result caching
            settings: Application settings
        """
        self.alert_repository = alert_repository
        self.database_manager = database_manager
        self.cache_service = cache_service
        self.settings = settings

        # Cache TTL settings
        self.list_cache_ttl = 30  # 30 seconds for incident lists
        self.detail_cache_ttl = 60  # 1 minute for incident details

    async def list_incidents(
        self,
        page: int = 1,
        size: int = 20,
        camera_id: str | None = None,
        incident_type: IncidentType | None = None,
        severity: Severity | None = None,
        status: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> PaginatedResponse[IncidentAlert]:
        """List incidents with pagination and filtering.

        Args:
            page: Page number (1-based)
            size: Items per page
            camera_id: Filter by camera ID
            incident_type: Filter by incident type
            severity: Filter by severity
            status: Filter by status
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            PaginatedResponse[IncidentAlert]: Paginated incident list

        Raises:
            ValueError: If pagination parameters are invalid
            RuntimeError: If database query fails
        """
        try:
            # Validate pagination parameters
            if page < 1:
                raise ValidationError("Page must be greater than 0", field="page", value=str(page))
            if size < 1 or size > 100:
                raise ValidationError("Size must be between 1 and 100", field="size", value=str(size))

            # Build cache key
            cache_key = self._build_list_cache_key(
                page, size, camera_id, incident_type, severity, status, start_time, end_time
            )

            # Check cache
            cached_result = await self.cache_service.get_json(cache_key)
            if cached_result:
                logger.debug(
                    "Cache hit for incident list",
                    cache_key=cache_key,
                    page=page,
                    size=size,
                )
                return PaginatedResponse[IncidentAlert](**cached_result)

            # Query database with proper session management
            incidents = []
            async with self.database_manager.get_session() as db_session:
                incidents = await self._query_incidents_from_database(
                    db_session, camera_id, incident_type, severity, status, start_time, end_time
                )

            # Sort incidents by timestamp (most recent first)
            incidents.sort(key=lambda x: x.timestamp, reverse=True)

            # Apply pagination
            total = len(incidents)
            offset = (page - 1) * size
            paginated_incidents = incidents[offset : offset + size]

            # Create paginated response
            result = PaginatedResponse.create(
                items=paginated_incidents,
                total=total,
                page=page,
                size=size,
            )

            # Cache result
            await self.cache_service.set_json(
                cache_key, result.model_dump(), ttl=self.list_cache_ttl
            )

            logger.info(
                "Incidents listed successfully",
                total=total,
                page=page,
                size=size,
                returned=len(paginated_incidents),
            )

            return result

        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                "Failed to list incidents",
                error=str(e),
                page=page,
                size=size,
                exc_info=True,
            )
            raise IncidentManagementError(
                f"Failed to list incidents: {e}",
                operation="list",
                original_error=str(e),
            ) from e

    async def get_incident_by_id(
        self,
        incident_id: str,
    ) -> IncidentAlert | None:
        """Get incident by ID.

        Args:
            incident_id: Incident identifier

        Returns:
            IncidentAlert: Incident details, None if not found

        Raises:
            ValueError: If incident_id is invalid
        """
        if not incident_id or not incident_id.strip():
            raise ValidationError("Incident ID cannot be empty", field="incident_id", value=incident_id)

        try:
            # Check cache first
            cache_key = f"incident:detail:{incident_id}"
            cached_data = await self.cache_service.get_json(cache_key)
            if cached_data:
                logger.debug(
                    "Cache hit for incident detail",
                    incident_id=incident_id,
                )
                return IncidentAlert(**cached_data)

            # Query from database with proper session management
            async with self.database_manager.get_session() as db_session:
                # Try to find incident by converting from violations/anomalies
                incidents = await self._query_incidents_from_database(db_session)

                for incident in incidents:
                    if incident.id == incident_id:
                        # Cache the result
                        await self.cache_service.set_json(
                            cache_key, incident.model_dump(), ttl=self.detail_cache_ttl
                        )

                        logger.debug(
                            "Incident retrieved from database",
                            incident_id=incident_id,
                        )
                        return incident

            logger.debug(
                "Incident not found",
                incident_id=incident_id,
            )
            return None

        except Exception as e:
            logger.error(
                "Failed to get incident",
                incident_id=incident_id,
                error=str(e),
                exc_info=True,
            )
            return None

    async def update_incident_status(
        self,
        incident_id: str,
        status_update: str,
        notes: str | None = None,
        resolved_by: str | None = None,
        db_session: AsyncSession | None = None,
    ) -> IncidentAlert | None:
        """Update incident status and details.

        NOTE: This method currently requires a dedicated IncidentRepository for proper
        persistence. The current implementation using AlertRepository for violations/anomalies
        conversion does not support incident updates.

        Args:
            incident_id: Incident identifier
            status_update: New incident status
            notes: Additional notes
            resolved_by: User ID who resolved the incident
            db_session: Database session for updates

        Returns:
            IncidentAlert: Updated incident, None if not found

        Raises:
            ValueError: If parameters are invalid
            NotImplementedError: Until proper IncidentRepository is implemented
        """
        if not incident_id or not incident_id.strip():
            raise ValueError("Incident ID cannot be empty")
        if not status_update or not status_update.strip():
            raise ValueError("Status update cannot be empty")

        # TODO: Implement incident updates once IncidentRepository is available
        # Current architecture converts violations/anomalies to incidents read-only
        logger.warning(
            "Incident updates not yet implemented with current repository architecture",
            incident_id=incident_id,
            status_update=status_update,
        )

        raise NotImplementedError(
            "Incident updates require a dedicated IncidentRepository. "
            "Current implementation only supports read-only incident queries "
            "converted from violations and anomalies."
        )

    async def get_filtered_incidents(
        self,
        camera_id: str | None = None,
        status: str | None = None,
        severity: str | None = None,
        incident_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> PaginatedResponse[IncidentAlert]:
        """Get incidents with advanced filtering.

        Args:
            camera_id: Filter by camera ID
            status: Filter by status
            severity: Filter by severity
            incident_type: Filter by incident type
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            PaginatedResponse[IncidentAlert]: Filtered incidents

        Raises:
            ValueError: If parameters are invalid
        """
        if limit < 1 or limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        if offset < 0:
            raise ValueError("Offset must be non-negative")

        try:
            # Generate cache key
            filter_params = f"{camera_id}:{status}:{severity}:{incident_type}:{limit}:{offset}"
            cache_key = f"incidents:filtered:{hash(filter_params)}"

            # Check cache
            cached_result = await self.cache_service.get_json(cache_key)
            if cached_result:
                logger.debug(
                    "Cache hit for filtered incidents",
                    cache_key=cache_key,
                )
                return PaginatedResponse[IncidentAlert](**cached_result)

            # Query incidents from database with proper session management
            filtered_incidents = []
            async with self.database_manager.get_session() as db_session:
                # Convert IncidentType and Severity enums if provided as strings
                incident_type_enum = None
                if incident_type:
                    try:
                        incident_type_enum = IncidentType(incident_type)
                    except ValueError:
                        logger.warning(f"Invalid incident type: {incident_type}")

                severity_enum = None
                if severity:
                    try:
                        severity_enum = Severity(severity)
                    except ValueError:
                        logger.warning(f"Invalid severity: {severity}")

                # Query from database with filters
                all_incidents = await self._query_incidents_from_database(
                    db_session, camera_id, incident_type_enum, severity_enum, status
                )

                # Apply additional filtering and convert to filtered_incidents list
                for incident in all_incidents:
                    # Additional filtering if needed (database query should handle most)
                    filtered_incidents.append(incident)

            # Sort by timestamp (newest first) and priority
            filtered_incidents.sort(
                key=lambda x: (
                    x.timestamp,
                    x.severity == "critical",
                    x.severity == "high"
                ),
                reverse=True,
            )

            # Apply pagination
            total_count = len(filtered_incidents)
            paginated_incidents = filtered_incidents[offset : offset + limit]

            # Create response
            response = PaginatedResponse.create(
                items=paginated_incidents,
                total=total_count,
                page=offset // limit + 1,
                size=limit,
            )

            # Cache result
            await self.cache_service.set_json(
                cache_key, response.model_dump(), ttl=60
            )

            logger.info(
                "Filtered incidents retrieved successfully",
                total=total_count,
                returned=len(paginated_incidents),
                filters={
                    "camera_id": camera_id,
                    "status": status,
                    "severity": severity,
                    "incident_type": incident_type,
                },
            )

            return response

        except ValueError:
            raise
        except Exception as e:
            logger.error(
                "Failed to get filtered incidents",
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Failed to get filtered incidents: {e}") from e

    async def get_active_incidents(
        self,
        camera_id: str | None = None,
        limit: int = 100,
    ) -> list[IncidentAlert]:
        """Get currently active incidents.

        Args:
            camera_id: Optional camera ID filter
            limit: Maximum number of incidents

        Returns:
            list[IncidentAlert]: Active incidents
        """
        try:
            active_incidents = []

            async with self.database_manager.get_session() as db_session:
                # Query all incidents from database
                all_incidents = await self._query_incidents_from_database(
                    db_session, camera_id, status="active"
                )

                # Filter for active incidents and apply camera filter
                for incident in all_incidents:
                    # Check active status
                    if incident.status != "active":
                        continue

                    # Filter by camera if specified (redundant check since database query handles this)
                    if camera_id and incident.camera_id != camera_id:
                        continue

                    active_incidents.append(incident)

            # Sort by severity and timestamp
            active_incidents.sort(
                key=lambda x: (
                    x.severity == "critical",
                    x.severity == "high",
                    x.timestamp
                ),
                reverse=True,
            )

            # Apply limit
            return active_incidents[:limit]

        except Exception as e:
            logger.error(
                "Failed to get active incidents",
                camera_id=camera_id,
                error=str(e),
            )
            return []

    async def generate_mock_incidents(
        self,
        camera_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[IncidentAlert]:
        """Generate mock incident data for testing and development.

        Args:
            camera_id: Camera identifier
            start_time: Start time for incident generation
            end_time: End time for incident generation

        Returns:
            list[IncidentAlert]: Generated mock incidents
        """
        incidents = []
        current_time = start_time

        try:
            while current_time < end_time:
                # Random chance of incident (low probability)
                if random.random() < 0.05:  # 5% chance per time period
                    incident_type = random.choice(list(IncidentType))
                    severity = random.choice(list(Severity))

                    incident = IncidentAlert(
                        id=str(uuid4()),
                        camera_id=camera_id,
                        incident_type=incident_type,
                        severity=severity,
                        description=f"{incident_type.value.replace('_', ' ').title()} detected",
                        location=f"Camera {camera_id} vicinity",
                        coordinates=None,
                        timestamp=current_time,
                        detected_at=current_time,
                        confidence=random.uniform(0.7, 0.95),
                        status="resolved" if random.random() > 0.3 else "active",
                        vehicles_involved=None,
                        estimated_duration=(
                            random.randint(5, 30)
                            if incident_type == IncidentType.CONGESTION
                            else None
                        ),
                        traffic_impact=(
                            "medium"
                            if severity in [Severity.MEDIUM, Severity.HIGH]
                            else "low"
                        ),
                        images=None,
                        video_clip=None,
                        resolved_at=(
                            current_time + timedelta(minutes=random.randint(5, 30))
                            if random.random() > 0.3
                            else None
                        ),
                        resolved_by=None,
                        notes=None,
                    )
                    incidents.append(incident)

                current_time += timedelta(hours=1)

            logger.debug(
                "Mock incidents generated",
                camera_id=camera_id,
                count=len(incidents),
                time_range_hours=(end_time - start_time).total_seconds() / 3600,
            )

        except Exception as e:
            logger.error(
                "Failed to generate mock incidents",
                camera_id=camera_id,
                error=str(e),
            )

        return incidents

    async def _query_incidents_from_database(
        self,
        db_session: AsyncSession,
        camera_id: str | None = None,
        incident_type: IncidentType | None = None,
        severity: Severity | None = None,
        status: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[IncidentAlert]:
        """Query incidents from database by converting violations and anomalies.

        Args:
            db_session: Database session
            camera_id: Filter by camera ID
            incident_type: Filter by incident type
            severity: Filter by severity
            status: Filter by status
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            list[IncidentAlert]: Database incidents converted to alerts
        """
        incidents = []

        try:
            # Query rule violations
            violations = await self._query_violations(
                db_session, camera_id, incident_type, severity, status, start_time, end_time
            )

            # Convert violations to incident alerts
            for violation in violations:
                incident = self._convert_violation_to_incident(violation)
                incidents.append(incident)

            # Query traffic anomalies
            anomalies = await self._query_anomalies(
                db_session, camera_id, incident_type, severity, status, start_time, end_time
            )

            # Convert anomalies to incident alerts
            for anomaly in anomalies:
                incident = self._convert_anomaly_to_incident(anomaly)
                incidents.append(incident)

            logger.debug(
                "Incidents queried from database",
                violations_count=len(violations),
                anomalies_count=len(anomalies),
                total_incidents=len(incidents),
            )

        except Exception as e:
            logger.error(
                "Failed to query incidents from database",
                error=str(e),
                exc_info=True,
            )

        return incidents

    async def _query_violations(
        self,
        db_session: AsyncSession,
        camera_id: str | None,
        incident_type: IncidentType | None,
        severity: Severity | None,
        status: str | None,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> list[RuleViolation]:
        """Query rule violations from database."""
        violation_query = select(RuleViolation)

        # Build filters
        filters = []

        if start_time:
            filters.append(RuleViolation.detection_time >= start_time)
        if end_time:
            filters.append(RuleViolation.detection_time <= end_time)
        if camera_id:
            filters.append(RuleViolation.camera_id == camera_id)
        if severity:
            filters.append(RuleViolation.severity == severity)

        # Filter by incident type for violations
        if incident_type and incident_type in ["speeding", "red_light", "wrong_way", "illegal_parking"]:
            filters.append(RuleViolation.violation_type == incident_type)
        elif incident_type and incident_type not in ["traffic_anomaly"]:
            # If specific violation type requested but not matching, return empty
            return []

        if filters:
            violation_query = violation_query.where(and_(*filters))

        violation_query = violation_query.limit(50)  # Limit results
        result = await db_session.execute(violation_query)
        return result.scalars().all()

    async def _query_anomalies(
        self,
        db_session: AsyncSession,
        camera_id: str | None,
        incident_type: IncidentType | None,
        severity: Severity | None,
        status: str | None,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> list[TrafficAnomaly]:
        """Query traffic anomalies from database."""
        anomaly_query = select(TrafficAnomaly)

        # Build filters
        filters = []

        if start_time:
            filters.append(TrafficAnomaly.detection_time >= start_time)
        if end_time:
            filters.append(TrafficAnomaly.detection_time <= end_time)
        if camera_id:
            filters.append(TrafficAnomaly.camera_id == camera_id)
        if severity:
            filters.append(TrafficAnomaly.severity == severity)

        # Filter by incident type for anomalies
        if incident_type == "traffic_anomaly":
            pass  # Include all anomalies
        elif incident_type and incident_type not in ["speeding", "red_light", "wrong_way", "illegal_parking"]:
            filters.append(TrafficAnomaly.anomaly_type == incident_type)
        elif incident_type:
            # If specific violation type requested, skip anomalies
            return []

        if filters:
            anomaly_query = anomaly_query.where(and_(*filters))

        anomaly_query = anomaly_query.limit(50)  # Limit results
        result = await db_session.execute(anomaly_query)
        return result.scalars().all()

    def _convert_violation_to_incident(self, violation: RuleViolation) -> IncidentAlert:
        """Convert a RuleViolation to an IncidentAlert."""
        return IncidentAlert(
            id=f"violation_{violation.id}",
            incident_type=violation.violation_type,
            severity=violation.severity,
            timestamp=violation.detection_time,
            camera_id=violation.camera_id,
            description=f"Traffic violation: {violation.violation_type}",
            location=f"Camera {violation.camera_id}",
            coordinates=None,
            detected_at=violation.detection_time,
            confidence=getattr(violation, 'confidence', 0.9),
            status="resolved" if violation.resolution_time else "active",
            vehicles_involved=None,
            estimated_duration=None,
            traffic_impact="medium",
            images=None,
            video_clip=None,
            resolved_at=violation.resolution_time,
            resolved_by=getattr(violation, 'resolved_by', None),
            notes=getattr(violation, 'resolution_action', None),
        )

    def _convert_anomaly_to_incident(self, anomaly: TrafficAnomaly) -> IncidentAlert:
        """Convert a TrafficAnomaly to an IncidentAlert."""
        return IncidentAlert(
            id=f"anomaly_{anomaly.id}",
            incident_type="traffic_anomaly",
            severity=anomaly.severity,
            timestamp=anomaly.detection_time,
            camera_id=anomaly.camera_id,
            description=f"Traffic anomaly: {anomaly.anomaly_type}",
            location=f"Camera {anomaly.camera_id}",
            coordinates=None,
            detected_at=anomaly.detection_time,
            confidence=getattr(anomaly, 'confidence', 0.85),
            status="resolved" if anomaly.resolution_time else "active",
            vehicles_involved=None,
            estimated_duration=None,
            traffic_impact="low",
            images=None,
            video_clip=None,
            resolved_at=anomaly.resolution_time,
            resolved_by=getattr(anomaly, 'resolved_by', None),
            notes=getattr(anomaly, 'resolution_action', None),
        )

    def _build_list_cache_key(
        self,
        page: int,
        size: int,
        camera_id: str | None,
        incident_type: IncidentType | None,
        severity: Severity | None,
        status: str | None,
        start_time: datetime | None,
        end_time: datetime | None,
    ) -> str:
        """Build cache key for incident list queries."""
        key_parts = [
            "incidents:list",
            str(page),
            str(size),
            camera_id or "all",
            incident_type.value if incident_type else "all",
            severity.value if severity else "all",
            status or "all",
            start_time.isoformat() if start_time else "all",
            end_time.isoformat() if end_time else "all",
        ]
        return ":".join(key_parts)

    async def _invalidate_incident_cache(self, incident_id: str) -> None:
        """Invalidate cache entries related to an incident."""
        try:
            # Invalidate detail cache
            detail_cache_key = f"incident:detail:{incident_id}"
            await self.cache_service.delete(detail_cache_key)

            # Invalidate list caches (would need pattern-based deletion in production)
            await self.cache_service.delete_pattern("incidents:list:*")
            await self.cache_service.delete_pattern("incidents:filtered:*")

            logger.debug(
                "Incident cache invalidated",
                incident_id=incident_id,
            )

        except Exception as e:
            logger.error(
                "Failed to invalidate incident cache",
                incident_id=incident_id,
                error=str(e),
            )
