"""Analytics and reporting endpoints.

Provides real-time traffic analytics, historical data queries,
incident management, and custom report generation.
"""

import random
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status

from ...core.database import get_database_session
from ...core.logging import get_logger
from ...models.user import User
from ...services.cache import CacheService
from ..dependencies import (
    RateLimiter,
    get_cache_service,
    get_current_user,
    rate_limit_normal,
    require_permissions,
)
from ..schemas.analytics import (
    AnalyticsResponse,
    HistoricalData,
    HistoricalQuery,
    IncidentAlert,
    IncidentType,
    ReportRequest,
    ReportResponse,
    Severity,
    TrafficMetrics,
    VehicleClass,
    VehicleCount,
)
from ..schemas.common import PaginatedResponse

logger = get_logger(__name__)
router = APIRouter()

# Rate limiters
report_rate_limit = RateLimiter(calls=10, period=3600)  # 10 reports per hour
query_rate_limit = RateLimiter(calls=100, period=300)  # 100 queries per 5 min

# Simulated data stores
incidents_db: dict[str, dict[str, Any]] = {}
reports_db: dict[str, dict[str, Any]] = {}
vehicle_counts_db: list[dict[str, Any]] = []
traffic_metrics_db: dict[str, dict[str, Any]] = {}


async def generate_mock_vehicle_counts(
    camera_id: str, start_time: datetime, end_time: datetime
) -> list[VehicleCount]:
    """Generate mock vehicle count data.

    Args:
        camera_id: Camera identifier
        start_time: Start time for data
        end_time: End time for data

    Returns:
        list[VehicleCount]: Mock vehicle count data
    """

    counts = []
    current_time = start_time

    while current_time < end_time:
        # Generate random counts for different vehicle classes
        for vehicle_class in [VehicleClass.CAR, VehicleClass.TRUCK, VehicleClass.VAN]:
            count = random.randint(0, 20)
            if count > 0:
                counts.append(
                    VehicleCount(
                        camera_id=camera_id,
                        timestamp=current_time,
                        vehicle_class=vehicle_class,
                        direction=random.choice(["north", "south", "east", "west"]),
                        count=count,
                        confidence=random.uniform(0.8, 0.99),
                        lane=random.choice(["lane_1", "lane_2", "lane_3"]),
                        speed=random.uniform(30, 80),
                    )
                )

        current_time += timedelta(minutes=5)  # 5-minute intervals

    return counts


async def generate_mock_incidents(
    camera_id: str, start_time: datetime, end_time: datetime
) -> list[IncidentAlert]:
    """Generate mock incident data.

    Args:
        camera_id: Camera identifier
        start_time: Start time for data
        end_time: End time for data

    Returns:
        list[IncidentAlert]: Mock incident data
    """

    incidents = []
    current_time = start_time

    while current_time < end_time:
        # Random chance of incident (low probability)
        if random.random() < 0.05:  # 5% chance per hour
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
                    "medium" if severity in [Severity.MEDIUM, Severity.HIGH] else "low"
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

    return incidents


@router.get(
    "/real-time/{camera_id}",
    response_model=AnalyticsResponse,
    summary="Get real-time analytics",
    description="Get current real-time traffic analytics for a specific camera.",
)
async def get_real_time_analytics(
    camera_id: str,
    current_user: User = Depends(get_current_user),
    cache: CacheService = Depends(get_cache_service),
) -> AnalyticsResponse:
    """Get real-time analytics for a camera.

    Args:
        camera_id: Camera identifier
        current_user: Current user
        cache: Cache service

    Returns:
        AnalyticsResponse: Real-time analytics data
    """
    try:
        # Check cache first
        cache_key = f"analytics:realtime:{camera_id}"
        cached_data = await cache.get_json(cache_key)
        if cached_data:
            return AnalyticsResponse(**cached_data)

        # Generate current analytics (in production, this would query real data)
        now = datetime.now(UTC)
        last_5_min = now - timedelta(minutes=5)

        # Get recent vehicle counts
        vehicle_counts = await generate_mock_vehicle_counts(camera_id, last_5_min, now)

        # Generate traffic metrics
        import random

        traffic_metrics = TrafficMetrics(
            camera_id=camera_id,
            period_start=last_5_min,
            period_end=now,
            total_vehicles=sum(vc.count for vc in vehicle_counts),
            vehicle_breakdown={
                VehicleClass.CAR: sum(
                    vc.count
                    for vc in vehicle_counts
                    if vc.vehicle_class == VehicleClass.CAR
                ),
                VehicleClass.TRUCK: sum(
                    vc.count
                    for vc in vehicle_counts
                    if vc.vehicle_class == VehicleClass.TRUCK
                ),
                VehicleClass.VAN: sum(
                    vc.count
                    for vc in vehicle_counts
                    if vc.vehicle_class == VehicleClass.VAN
                ),
            },
            directional_flow={
                "north": random.randint(10, 50),
                "south": random.randint(10, 50),
                "east": random.randint(5, 30),
                "west": random.randint(5, 30),
            },
            avg_speed=random.uniform(45, 75),
            peak_hour=None,
            occupancy_rate=random.uniform(20, 80),
            congestion_level=random.choice(["low", "medium", "high"]),
            queue_length=random.uniform(0, 50),
        )

        # Get active incidents
        active_incidents = await generate_mock_incidents(camera_id, last_5_min, now)
        active_incidents = [inc for inc in active_incidents if inc.status == "active"]

        analytics_response = AnalyticsResponse(
            camera_id=camera_id,
            timestamp=now,
            vehicle_counts=vehicle_counts,
            traffic_metrics=traffic_metrics,
            active_incidents=active_incidents,
            processing_time=random.uniform(50, 150),
            frame_rate=random.uniform(25, 30),
            detection_zones=["main_road", "intersection", "parking_area"],
        )

        # Cache for 10 seconds
        await cache.set_json(cache_key, analytics_response.model_dump(), ttl=10)

        logger.debug(
            "Real-time analytics retrieved",
            camera_id=camera_id,
            user_id=current_user.id,
        )

        return analytics_response

    except Exception as e:
        logger.error(
            "Failed to get real-time analytics",
            camera_id=camera_id,
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analytics data",
        ) from e


@router.get(
    "/incidents",
    response_model=PaginatedResponse[IncidentAlert],
    summary="List incidents",
    description="Retrieve paginated list of traffic incidents with filtering.",
)
async def list_incidents(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    camera_id: str | None = Query(None, description="Filter by camera ID"),
    incident_type: IncidentType | None = Query(
        None, description="Filter by incident type"
    ),
    severity: Severity | None = Query(None, description="Filter by severity"),
    status: str | None = Query(None, description="Filter by status"),
    start_time: datetime | None = Query(None, description="Filter by start time"),
    end_time: datetime | None = Query(None, description="Filter by end time"),
    current_user: User = Depends(get_current_user),
    cache: CacheService = Depends(get_cache_service),
    _rate_limit: None = Depends(rate_limit_normal),
    db: Any = Depends(get_database_session),
) -> PaginatedResponse[IncidentAlert]:
    """List incidents with pagination and filtering.

    Args:
        page: Page number
        size: Items per page
        camera_id: Filter by camera ID
        incident_type: Filter by incident type
        severity: Filter by severity
        status: Filter by status
        start_time: Filter by start time
        end_time: Filter by end time
        current_user: Current user
        cache: Cache service
        _rate_limit: Rate limiting dependency

    Returns:
        PaginatedResponse[IncidentAlert]: Paginated incident list
    """
    try:
        # Build cache key
        cache_key = f"incidents:list:{page}:{size}:{camera_id}:{incident_type}:{severity}:{status}:{start_time}:{end_time}"

        # Check cache
        cached_result = await cache.get_json(cache_key)
        if cached_result:
            return PaginatedResponse[IncidentAlert](**cached_result)

        # Query database for violations and anomalies to create incident alerts
        from sqlalchemy import and_, select

        from ...models.analytics import RuleViolation, TrafficAnomaly

        incidents = []

        # Query rule violations
        violation_query = select(RuleViolation)

        # Apply filters to violation query
        filters = []
        if camera_id:
            filters.append(RuleViolation.camera_id == camera_id)
        if start_time:
            filters.append(RuleViolation.detection_time >= start_time)
        if end_time:
            filters.append(RuleViolation.detection_time <= end_time)
        if status:
            filters.append(RuleViolation.status == status)

        if filters:
            violation_query = violation_query.where(and_(*filters))

        # Add severity filter if specified
        if severity:
            violation_query = violation_query.where(RuleViolation.severity == severity)

        # Add incident type filter for violations
        if incident_type and incident_type in ["speeding", "red_light", "wrong_way", "illegal_parking"]:
            violation_query = violation_query.where(RuleViolation.violation_type == incident_type)
        elif incident_type and incident_type not in ["traffic_anomaly"]:
            # If specific violation type requested but not matching, skip violations
            violation_query = violation_query.where(False)

        violation_query = violation_query.order_by(RuleViolation.detection_time.desc())

        # Query traffic anomalies
        anomaly_query = select(TrafficAnomaly)

        # Apply filters to anomaly query
        anomaly_filters = []
        if camera_id:
            anomaly_filters.append(TrafficAnomaly.camera_id == camera_id)
        if start_time:
            anomaly_filters.append(TrafficAnomaly.detection_time >= start_time)
        if end_time:
            anomaly_filters.append(TrafficAnomaly.detection_time <= end_time)
        if status:
            # Map status to anomaly status
            status_map = {"active": "detected", "resolved": "resolved", "dismissed": "resolved"}
            anomaly_status = status_map.get(status, status)
            anomaly_filters.append(TrafficAnomaly.status == anomaly_status)

        if anomaly_filters:
            anomaly_query = anomaly_query.where(and_(*anomaly_filters))

        # Add severity filter if specified
        if severity:
            anomaly_query = anomaly_query.where(TrafficAnomaly.severity == severity)

        # Add incident type filter for anomalies
        if incident_type == "traffic_anomaly":
            pass  # Include all anomalies
        elif incident_type and incident_type not in ["speeding", "red_light", "wrong_way", "illegal_parking"]:
            anomaly_query = anomaly_query.where(TrafficAnomaly.anomaly_type == incident_type)
        elif incident_type:
            # If specific violation type requested, skip anomalies
            anomaly_query = anomaly_query.where(False)

        anomaly_query = anomaly_query.order_by(TrafficAnomaly.detection_time.desc())

        # Execute queries
        try:
            violation_result = await db.execute(violation_query.limit(size * 2))  # Get more for mixing
            violations = violation_result.scalars().all()

            anomaly_result = await db.execute(anomaly_query.limit(size * 2))  # Get more for mixing
            anomalies = anomaly_result.scalars().all()

            # Convert violations to incident alerts
            for violation in violations:
                incident = IncidentAlert(
                    id=f"violation_{violation.id}",
                    incident_type=violation.violation_type,
                    severity=violation.severity,
                    timestamp=violation.detection_time,
                    camera_id=violation.camera_id,
                    location=violation.location_description or f"Zone {violation.zone_id or 'Unknown'}",
                    description=f"{violation.violation_type.replace('_', ' ').title()} violation detected",
                    status=violation.status,
                    confidence_score=violation.detection_confidence,
                    vehicle_info={
                        "license_plate": violation.license_plate,
                        "vehicle_type": violation.vehicle_type,
                        "track_id": violation.vehicle_track_id,
                    } if any([violation.license_plate, violation.vehicle_type, violation.vehicle_track_id]) else None,
                    evidence_urls=violation.evidence_images or [],
                    detection_details={
                        "rule_definition": violation.rule_definition,
                        "measured_value": violation.measured_value,
                        "threshold_value": violation.threshold_value,
                        "violation_duration": violation.violation_duration,
                    },
                    alert_sent=True,  # Assume alerts are sent for all violations
                    acknowledged_by=None,
                    resolved_at=violation.resolution_time,
                    resolution_notes=violation.resolution_action,
                )
                incidents.append(incident)

            # Convert anomalies to incident alerts
            for anomaly in anomalies:
                incident = IncidentAlert(
                    id=f"anomaly_{anomaly.id}",
                    incident_type="traffic_anomaly",
                    severity=anomaly.severity,
                    timestamp=anomaly.detection_time,
                    camera_id=anomaly.camera_id,
                    location=f"Zone {anomaly.zone_id or 'Unknown'}",
                    description=f"{anomaly.anomaly_type.replace('_', ' ').title()} anomaly detected",
                    status="active" if anomaly.status == "detected" else anomaly.status,
                    confidence_score=anomaly.confidence,
                    vehicle_info=None,
                    evidence_urls=[],
                    detection_details={
                        "anomaly_type": anomaly.anomaly_type,
                        "anomaly_score": anomaly.anomaly_score,
                        "detection_method": anomaly.detection_method,
                        "probable_cause": anomaly.probable_cause,
                        "baseline_value": anomaly.baseline_value,
                        "observed_value": anomaly.observed_value,
                        "affected_metrics": anomaly.affected_metrics,
                    },
                    alert_sent=True,  # Assume alerts are sent for high-score anomalies
                    acknowledged_by=None,
                    resolved_at=anomaly.resolution_time,
                    resolution_notes=anomaly.resolution_action,
                )
                incidents.append(incident)

        except Exception as e:
            logger.error(f"Failed to query incidents from database: {e}")
            # Fallback to empty list
            incidents = []

        # Sort incidents by timestamp (most recent first)
        incidents.sort(key=lambda x: x.timestamp, reverse=True)

        # Apply pagination
        total = len(incidents)
        offset = (page - 1) * size
        paginated_incidents = incidents[offset : offset + size]

        incident_responses = paginated_incidents

        result = PaginatedResponse.create(
            items=incident_responses,
            total=total,
            page=page,
            size=size,
        )

        # Cache for 30 seconds
        await cache.set_json(cache_key, result.model_dump(), ttl=30)

        logger.info(
            "Incidents listed",
            total=total,
            page=page,
            size=size,
            user_id=current_user.id,
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to list incidents",
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve incidents",
        ) from e


@router.get(
    "/incidents/{incident_id}",
    response_model=IncidentAlert,
    summary="Get incident details",
    description="Retrieve detailed information about a specific incident.",
)
async def get_incident(
    incident_id: str,
    current_user: User = Depends(get_current_user),
) -> IncidentAlert:
    """Get incident by ID.

    Args:
        incident_id: Incident identifier
        current_user: Current user

    Returns:
        IncidentAlert: Incident details

    Raises:
        HTTPException: If incident not found
    """
    incident = incidents_db.get(incident_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )

    logger.debug(
        "Incident retrieved",
        incident_id=incident_id,
        user_id=current_user.id,
    )

    return IncidentAlert(**incident)


@router.put(
    "/incidents/{incident_id}",
    response_model=IncidentAlert,
    summary="Update incident",
    description="Update incident status and details.",
)
async def update_incident(
    incident_id: str,
    status_update: str = Query(description="New incident status"),
    notes: str | None = Query(None, description="Additional notes"),
    current_user: User = Depends(require_permissions("incidents:update")),
    _rate_limit: None = Depends(rate_limit_normal),
) -> IncidentAlert:
    """Update incident status.

    Args:
        incident_id: Incident identifier
        status_update: New status
        notes: Additional notes
        current_user: Current user with permissions
        _rate_limit: Rate limiting dependency

    Returns:
        IncidentAlert: Updated incident

    Raises:
        HTTPException: If incident not found or update fails
    """
    incident = incidents_db.get(incident_id)
    if not incident:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )

    # Update incident
    incident["status"] = status_update
    if notes:
        incident["notes"] = notes

    if status_update == "resolved":
        incident["resolved_at"] = datetime.now(UTC).isoformat()
        incident["resolved_by"] = current_user.id

    logger.info(
        "Incident updated",
        incident_id=incident_id,
        new_status=status_update,
        user_id=current_user.id,
    )

    return IncidentAlert(**incident)


@router.post(
    "/reports",
    response_model=ReportResponse,
    summary="Generate report",
    description="Generate a custom analytics report.",
)
async def generate_report(
    report_request: ReportRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permissions("reports:create")),
    _rate_limit: None = Depends(report_rate_limit),
) -> ReportResponse:
    """Generate analytics report.

    Args:
        report_request: Report generation parameters
        background_tasks: Background task manager
        current_user: Current user with permissions
        _rate_limit: Rate limiting dependency

    Returns:
        ReportResponse: Report generation status
    """
    try:
        # Generate report ID
        report_id = str(uuid4())

        # Create report record
        now = datetime.now(UTC)
        report_record = {
            "report_id": report_id,
            "report_type": report_request.report_type,
            "status": "pending",
            "created_at": now.isoformat(),
            "completed_at": None,
            "download_url": None,
            "file_size": None,
            "parameters": report_request.model_dump(),
            "error_message": None,
            "expires_at": (now + timedelta(days=7)).isoformat(),
            "created_by": current_user.id,
        }

        reports_db[report_id] = report_record

        # Start report generation in background
        background_tasks.add_task(
            generate_report_background,
            report_id,
            report_request,
        )

        logger.info(
            "Report generation started",
            report_id=report_id,
            report_type=report_request.report_type,
            user_id=current_user.id,
        )

        return ReportResponse(
            report_id=report_id,
            report_type=report_request.report_type,
            status="pending",
            created_at=now,
            completed_at=None,
            download_url=None,
            file_size=None,
            parameters=report_request.model_dump(),
            error_message=None,
            expires_at=now + timedelta(days=7),
        )

    except Exception as e:
        logger.error(
            "Report generation failed",
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report generation failed",
        ) from e


async def generate_report_background(
    report_id: str, _report_request: ReportRequest
) -> None:
    """Background task to generate report.

    Args:
        report_id: Report identifier
        report_request: Report parameters
    """
    try:
        import asyncio
        import random

        # Simulate report generation time
        await asyncio.sleep(random.uniform(5, 15))

        # Update report status
        report = reports_db.get(report_id)
        if report:
            report["status"] = "completed"
            report["completed_at"] = datetime.now(UTC).isoformat()
            report["download_url"] = f"/api/v1/analytics/reports/{report_id}/download"
            report["file_size"] = random.randint(1024, 1024 * 1024)  # 1KB to 1MB

        logger.info("Report generation completed", report_id=report_id)

    except Exception as e:
        # Update report with error
        report = reports_db.get(report_id)
        if report:
            report["status"] = "failed"
            report["error_message"] = str(e)

        logger.error("Report generation failed", report_id=report_id, error=str(e))


@router.get(
    "/reports",
    response_model=PaginatedResponse[ReportResponse],
    summary="List reports",
    description="List generated reports for the current user.",
)
async def list_reports(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    status: str | None = Query(None, description="Filter by status"),
    report_type: str | None = Query(None, description="Filter by report type"),
    current_user: User = Depends(get_current_user),
) -> PaginatedResponse[ReportResponse]:
    """List user's reports.

    Args:
        page: Page number
        size: Items per page
        status: Filter by status
        report_type: Filter by report type
        current_user: Current user

    Returns:
        PaginatedResponse[ReportResponse]: Paginated report list
    """
    # Filter reports by user
    user_reports = [
        report
        for report in reports_db.values()
        if report.get("created_by") == current_user.id
    ]

    # Apply filters
    if status:
        user_reports = [r for r in user_reports if r.get("status") == status]
    if report_type:
        user_reports = [r for r in user_reports if r.get("report_type") == report_type]

    # Sort by creation time (newest first)
    user_reports.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Pagination
    total = len(user_reports)
    offset = (page - 1) * size
    paginated_reports = user_reports[offset : offset + size]

    # Convert to response models
    report_responses = []
    for report in paginated_reports:
        report_responses.append(
            ReportResponse(
                report_id=report["report_id"],
                report_type=report["report_type"],
                status=report["status"],
                created_at=datetime.fromisoformat(report["created_at"]),
                completed_at=(
                    datetime.fromisoformat(report["completed_at"])
                    if report.get("completed_at")
                    else None
                ),
                download_url=report.get("download_url"),
                file_size=report.get("file_size"),
                parameters=report["parameters"],
                error_message=report.get("error_message"),
                expires_at=(
                    datetime.fromisoformat(report["expires_at"])
                    if report.get("expires_at")
                    else None
                ),
            )
        )

    return PaginatedResponse.create(
        items=report_responses,
        total=total,
        page=page,
        size=size,
    )


@router.get(
    "/reports/{report_id}",
    response_model=ReportResponse,
    summary="Get report status",
    description="Get the status and details of a specific report.",
)
async def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
) -> ReportResponse:
    """Get report status.

    Args:
        report_id: Report identifier
        current_user: Current user

    Returns:
        ReportResponse: Report status and details

    Raises:
        HTTPException: If report not found or access denied
    """
    report = reports_db.get(report_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report {report_id} not found",
        )

    # Check access (user can only access their own reports unless admin)
    if report.get("created_by") != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this report",
        )

    return ReportResponse(
        report_id=report["report_id"],
        report_type=report["report_type"],
        status=report["status"],
        created_at=datetime.fromisoformat(report["created_at"]),
        completed_at=(
            datetime.fromisoformat(report["completed_at"])
            if report.get("completed_at")
            else None
        ),
        download_url=report.get("download_url"),
        file_size=report.get("file_size"),
        parameters=report["parameters"],
        error_message=report.get("error_message"),
        expires_at=(
            datetime.fromisoformat(report["expires_at"])
            if report.get("expires_at")
            else None
        ),
    )


@router.post(
    "/historical-query",
    response_model=PaginatedResponse[HistoricalData],
    summary="Query historical data",
    description="Query historical traffic analytics data with flexible filtering.",
)
async def query_historical_data(
    query: HistoricalQuery,
    current_user: User = Depends(get_current_user),
    cache: CacheService = Depends(get_cache_service),
    _rate_limit: None = Depends(query_rate_limit),
    db: Any = Depends(get_database_session),
) -> PaginatedResponse[HistoricalData]:
    """Query historical analytics data.

    Args:
        query: Historical data query parameters
        current_user: Current user
        cache: Cache service
        _rate_limit: Rate limiting dependency

    Returns:
        PaginatedResponse[HistoricalData]: Historical data results
    """
    try:
        # Build cache key
        cache_key = f"historical:{hash(str(query.model_dump()))}"

        # Check cache
        cached_result = await cache.get_json(cache_key)
        if cached_result:
            return PaginatedResponse[HistoricalData](**cached_result)

        # Query real historical data from TrafficMetrics table
        from sqlalchemy import and_, select

        from ...models.analytics import TrafficMetrics

        historical_data = []

        try:
            # Build query for traffic metrics
            traffic_query = select(TrafficMetrics)

            # Apply filters
            filters = [
                TrafficMetrics.timestamp >= query.time_range.start_time,
                TrafficMetrics.timestamp <= query.time_range.end_time
            ]

            if query.camera_ids:
                filters.append(TrafficMetrics.camera_id.in_(query.camera_ids))

            # Map aggregation to database aggregation period
            aggregation_map = {
                "minute": "1min",
                "hourly": "1hour",
                "daily": "1day",
                "raw": "1min"  # Use 1min as finest granularity
            }

            db_aggregation = aggregation_map.get(query.aggregation, "1hour")
            filters.append(TrafficMetrics.aggregation_period == db_aggregation)

            traffic_query = traffic_query.where(and_(*filters))
            traffic_query = traffic_query.order_by(TrafficMetrics.timestamp)

            # Execute query
            result = await db.execute(traffic_query)
            metrics = result.scalars().all()

            # Convert to HistoricalData format
            for metric in metrics:
                for metric_type in query.metric_types:
                    value = None

                    if metric_type == "vehicle_counts":
                        value = metric.total_vehicles
                    elif metric_type == "speed_data":
                        value = metric.average_speed
                    elif metric_type == "occupancy":
                        value = (metric.occupancy_rate or 0.0) * 100  # Convert to percentage
                    elif metric_type == "traffic_density":
                        value = metric.traffic_density
                    elif metric_type == "flow_rate":
                        value = metric.flow_rate
                    elif metric_type == "congestion_level":
                        # Convert congestion level to numeric value
                        congestion_map = {
                            "free_flow": 0,
                            "light": 1,
                            "moderate": 2,
                            "heavy": 3,
                            "severe": 4
                        }
                        value = congestion_map.get(metric.congestion_level, 0)

                    if value is not None:
                        historical_data.append(
                            HistoricalData(
                                timestamp=metric.timestamp,
                                camera_id=metric.camera_id,
                                metric_type=metric_type,
                                value=value,
                                metadata={
                                    "aggregation": query.aggregation,
                                    "sample_count": metric.sample_count,
                                    "zone_id": metric.zone_id,
                                    "lane_id": metric.lane_id,
                                    "data_completeness": metric.data_completeness,
                                },
                            )
                        )

            # If no data found and specific cameras requested, add placeholder entries
            if not historical_data and query.camera_ids:
                logger.warning(
                    f"No historical data found for cameras {query.camera_ids} "
                    f"in time range {query.time_range.start_time} to {query.time_range.end_time}"
                )

        except Exception as e:
            logger.error(f"Failed to query historical data from database: {e}")
            # Return empty list on error
            historical_data = []

        # Apply pagination
        total = len(historical_data)
        offset = query.offset
        limit = query.limit

        paginated_data = historical_data[offset : offset + limit]

        result = PaginatedResponse.create(
            items=paginated_data,
            total=total,
            page=(offset // limit) + 1,
            size=limit,
        )

        # Cache for 5 minutes
        await cache.set_json(cache_key, result.model_dump(), ttl=300)

        logger.info(
            "Historical data queried",
            total_records=total,
            returned_records=len(paginated_data),
            user_id=current_user.id,
        )

        return result

    except Exception as e:
        logger.error(
            "Historical data query failed",
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Historical data query failed",
        ) from e
