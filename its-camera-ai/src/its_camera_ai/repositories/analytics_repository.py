"""Analytics repository for traffic analytics data access.

Provides specialized methods for traffic analytics, rule violations,
vehicle trajectories, and anomaly detection with optimized queries.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from ..models.analytics import (
    RuleViolation,
    SpeedLimit,
    TrafficAnomaly,
    TrafficMetrics,
    VehicleTrajectory,
)
from .base_repository import BaseRepository

logger = get_logger(__name__)


class AnalyticsRepository(BaseRepository[TrafficMetrics]):
    """Repository for traffic analytics data access operations.

    Specialized methods for traffic analytics, violations, trajectories,
    and anomaly detection with time-series optimizations for TimescaleDB.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        super().__init__(session_factory, TrafficMetrics)

    # ============================
    # Traffic Metrics Operations
    # ============================

    async def get_traffic_metrics_by_camera(
        self,
        camera_id: str,
        start_time: datetime,
        end_time: datetime,
        aggregation_period: str = "1hour",
        limit: int = 1000,
    ) -> list[TrafficMetrics]:
        """Get traffic metrics by camera and time range.

        Args:
            camera_id: Camera identifier
            start_time: Start of time range
            end_time: End of time range
            aggregation_period: Aggregation period (1min, 5min, 1hour, etc.)
            limit: Maximum number of results

        Returns:
            List of traffic metrics

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                conditions = [
                    TrafficMetrics.timestamp >= start_time,
                    TrafficMetrics.timestamp <= end_time,
                    TrafficMetrics.aggregation_period == aggregation_period,
                ]

                if camera_id:  # Only add camera_id filter if it's not empty
                    conditions.append(TrafficMetrics.camera_id == camera_id)

                query = (
                    select(TrafficMetrics)
                    .where(and_(*conditions))
                    .order_by(TrafficMetrics.timestamp.desc())
                    .limit(limit)
                )

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get traffic metrics by camera",
                    camera_id=camera_id,
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    error=str(e),
                )
                raise DatabaseError("Traffic metrics retrieval failed", cause=e) from e

    async def create_traffic_metrics(
        self, metrics_data: dict[str, Any]
    ) -> TrafficMetrics:
        """Create new traffic metrics record.

        Args:
            metrics_data: Traffic metrics data

        Returns:
            Created traffic metrics record

        Raises:
            DatabaseError: If creation fails
        """
        async with self._get_session() as session:
            try:
                metrics = TrafficMetrics(**metrics_data)
                session.add(metrics)
                await session.commit()
                await session.refresh(metrics)
                return metrics

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to create traffic metrics",
                    camera_id=metrics_data.get("camera_id"),
                    error=str(e),
                )
                raise DatabaseError("Traffic metrics creation failed", cause=e) from e

    async def get_aggregated_metrics(
        self,
        camera_ids: list[str] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        aggregation_period: str = "1hour",
    ) -> dict[str, Any]:
        """Get aggregated traffic metrics.

        Args:
            camera_ids: Optional list of camera IDs to filter by
            start_time: Optional start time filter
            end_time: Optional end time filter
            aggregation_period: Aggregation period

        Returns:
            Dictionary with aggregated metrics

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(
                    func.sum(TrafficMetrics.total_vehicles).label("total_vehicles"),
                    func.avg(TrafficMetrics.average_speed).label("avg_speed"),
                    func.avg(TrafficMetrics.traffic_density).label("avg_density"),
                    func.avg(TrafficMetrics.occupancy_rate).label("avg_occupancy"),
                    func.count(TrafficMetrics.id).label("record_count"),
                ).where(TrafficMetrics.aggregation_period == aggregation_period)

                conditions = []
                if camera_ids:
                    conditions.append(TrafficMetrics.camera_id.in_(camera_ids))
                if start_time:
                    conditions.append(TrafficMetrics.timestamp >= start_time)
                if end_time:
                    conditions.append(TrafficMetrics.timestamp <= end_time)

                if conditions:
                    query = query.where(and_(*conditions))

                result = await session.execute(query)
                row = result.first()

                if row and row.record_count > 0:
                    return {
                        "total_vehicles": int(row.total_vehicles or 0),
                        "avg_speed": float(row.avg_speed or 0),
                        "avg_density": float(row.avg_density or 0),
                        "avg_occupancy": float(row.avg_occupancy or 0),
                        "record_count": row.record_count,
                        "aggregation_period": aggregation_period,
                        "camera_ids": camera_ids,
                        "time_range": {
                            "start": start_time.isoformat() if start_time else None,
                            "end": end_time.isoformat() if end_time else None,
                        },
                    }
                else:
                    return {
                        "total_vehicles": 0,
                        "avg_speed": 0.0,
                        "avg_density": 0.0,
                        "avg_occupancy": 0.0,
                        "record_count": 0,
                        "aggregation_period": aggregation_period,
                        "camera_ids": camera_ids,
                        "time_range": {
                            "start": start_time.isoformat() if start_time else None,
                            "end": end_time.isoformat() if end_time else None,
                        },
                    }

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get aggregated metrics",
                    camera_ids=camera_ids,
                    error=str(e),
                )
                raise DatabaseError("Aggregated metrics retrieval failed", cause=e) from e

    # ============================
    # Rule Violations Operations
    # ============================

    async def create_rule_violation(
        self, violation_data: dict[str, Any]
    ) -> RuleViolation:
        """Create new rule violation record.

        Args:
            violation_data: Violation data

        Returns:
            Created violation record

        Raises:
            DatabaseError: If creation fails
        """
        async with self._get_session() as session:
            try:
                violation = RuleViolation(**violation_data)
                session.add(violation)
                await session.commit()
                await session.refresh(violation)
                return violation

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to create rule violation",
                    camera_id=violation_data.get("camera_id"),
                    violation_type=violation_data.get("violation_type"),
                    error=str(e),
                )
                raise DatabaseError("Rule violation creation failed", cause=e) from e

    async def create_rule_violations_batch(
        self, violations_data: list[dict[str, Any]]
    ) -> list[RuleViolation]:
        """Create multiple rule violation records in batch.

        Args:
            violations_data: List of violation data dictionaries

        Returns:
            List of created violation records

        Raises:
            DatabaseError: If batch creation fails
        """
        async with self._get_session() as session:
            try:
                violations = [RuleViolation(**data) for data in violations_data]
                session.add_all(violations)
                await session.commit()

                # Refresh all instances to get IDs
                for violation in violations:
                    await session.refresh(violation)

                return violations

            except SQLAlchemyError as e:
                await session.rollback()
                logger.error(
                    "Failed to create rule violations batch",
                    violation_count=len(violations_data),
                    error=str(e),
                )
                raise DatabaseError("Rule violations batch creation failed", cause=e) from e

    async def get_active_violations(
        self,
        camera_id: str | None = None,
        violation_type: str | None = None,
        severity: str | None = None,
        limit: int = 100,
    ) -> list[RuleViolation]:
        """Get active rule violations with filtering options.

        Args:
            camera_id: Optional camera ID filter
            violation_type: Optional violation type filter
            severity: Optional severity filter
            limit: Maximum number of results

        Returns:
            List of active violations

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(RuleViolation).where(RuleViolation.status == "active")

                if camera_id:
                    query = query.where(RuleViolation.camera_id == camera_id)
                if violation_type:
                    query = query.where(RuleViolation.violation_type == violation_type)
                if severity:
                    query = query.where(RuleViolation.severity == severity)

                query = query.order_by(RuleViolation.detection_time.desc()).limit(limit)

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get active violations",
                    camera_id=camera_id,
                    violation_type=violation_type,
                    severity=severity,
                    error=str(e),
                )
                raise DatabaseError("Active violations retrieval failed", cause=e) from e

    # ============================
    # Speed Limit Operations
    # ============================

    async def get_speed_limit(
        self,
        zone_id: str | None,
        vehicle_type: str,
        check_time: datetime,
    ) -> SpeedLimit | None:
        """Get applicable speed limit for zone and vehicle type.

        Args:
            zone_id: Traffic zone identifier
            vehicle_type: Vehicle type
            check_time: Time to check validity

        Returns:
            Speed limit record or None if not found

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(SpeedLimit).where(
                    and_(
                        SpeedLimit.enforcement_enabled,
                        SpeedLimit.valid_from <= check_time,
                        or_(
                            SpeedLimit.valid_until.is_(None),
                            SpeedLimit.valid_until > check_time,
                        ),
                    )
                )

                # Add zone and vehicle type filters
                if zone_id:
                    query = query.where(
                        or_(
                            SpeedLimit.zone_id == zone_id,
                            SpeedLimit.zone_id == "default",  # Fallback to default zone
                        )
                    )
                else:
                    query = query.where(SpeedLimit.zone_id == "default")

                query = query.where(
                    or_(
                        SpeedLimit.vehicle_type == vehicle_type,
                        SpeedLimit.vehicle_type == "general",  # Fallback to general
                    )
                )

                # Order by priority (lower number = higher priority) and specificity
                query = query.order_by(
                    SpeedLimit.priority.asc(),
                    SpeedLimit.zone_id.desc(),  # Specific zones before default
                    SpeedLimit.vehicle_type.desc(),  # Specific vehicle types before general
                )

                result = await session.execute(query)
                return result.scalar_one_or_none()

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get speed limit",
                    zone_id=zone_id,
                    vehicle_type=vehicle_type,
                    error=str(e),
                )
                raise DatabaseError("Speed limit retrieval failed", cause=e) from e

    # ============================
    # Vehicle Trajectory Operations
    # ============================

    async def get_vehicle_trajectory(
        self, track_id: int, camera_id: str
    ) -> VehicleTrajectory | None:
        """Get vehicle trajectory by track ID and camera.

        Args:
            track_id: Vehicle tracking ID
            camera_id: Camera identifier

        Returns:
            Vehicle trajectory or None if not found

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(VehicleTrajectory).where(
                    and_(
                        VehicleTrajectory.vehicle_track_id == track_id,
                        VehicleTrajectory.camera_id == camera_id,
                    )
                )

                result = await session.execute(query)
                return result.scalar_one_or_none()

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get vehicle trajectory",
                    track_id=track_id,
                    camera_id=camera_id,
                    error=str(e),
                )
                raise DatabaseError("Vehicle trajectory retrieval failed", cause=e) from e

    async def create_or_update_trajectory(
        self, trajectory_data: dict[str, Any]
    ) -> VehicleTrajectory:
        """Create new or update existing vehicle trajectory.

        Args:
            trajectory_data: Trajectory data

        Returns:
            Created or updated trajectory record

        Raises:
            DatabaseError: If operation fails
        """
        async with self._get_session() as session:
            try:
                track_id = trajectory_data.get("vehicle_track_id")
                camera_id = trajectory_data.get("camera_id")

                if track_id and camera_id:
                    # Try to find existing trajectory
                    existing = await session.get(
                        VehicleTrajectory,
                        {"vehicle_track_id": track_id, "camera_id": camera_id},
                    )

                    if existing:
                        # Update existing trajectory
                        for key, value in trajectory_data.items():
                            if hasattr(existing, key):
                                setattr(existing, key, value)
                        trajectory = existing
                    else:
                        # Create new trajectory
                        trajectory = VehicleTrajectory(**trajectory_data)
                        session.add(trajectory)
                else:
                    # Create new trajectory
                    trajectory = VehicleTrajectory(**trajectory_data)
                    session.add(trajectory)

                await session.commit()
                await session.refresh(trajectory)
                return trajectory

            except SQLAlchemyError as e:
                await session.rollback()
                logger.error(
                    "Failed to create or update trajectory",
                    track_id=trajectory_data.get("vehicle_track_id"),
                    camera_id=trajectory_data.get("camera_id"),
                    error=str(e),
                )
                raise DatabaseError("Trajectory operation failed", cause=e) from e

    # ============================
    # Traffic Anomaly Operations
    # ============================

    async def create_traffic_anomaly(
        self, anomaly_data: dict[str, Any]
    ) -> TrafficAnomaly:
        """Create new traffic anomaly record.

        Args:
            anomaly_data: Anomaly data

        Returns:
            Created anomaly record

        Raises:
            DatabaseError: If creation fails
        """
        async with self._get_session() as session:
            try:
                anomaly = TrafficAnomaly(**anomaly_data)
                session.add(anomaly)
                await session.commit()
                await session.refresh(anomaly)
                return anomaly

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to create traffic anomaly",
                    camera_id=anomaly_data.get("camera_id"),
                    anomaly_type=anomaly_data.get("anomaly_type"),
                    error=str(e),
                )
                raise DatabaseError("Traffic anomaly creation failed", cause=e) from e

    async def create_traffic_anomalies_batch(
        self, anomalies_data: list[dict[str, Any]]
    ) -> list[TrafficAnomaly]:
        """Create multiple traffic anomaly records in batch.

        Args:
            anomalies_data: List of anomaly data dictionaries

        Returns:
            List of created anomaly records

        Raises:
            DatabaseError: If batch creation fails
        """
        async with self._get_session() as session:
            try:
                anomalies = [TrafficAnomaly(**data) for data in anomalies_data]
                session.add_all(anomalies)
                await session.commit()

                # Refresh all instances to get IDs
                for anomaly in anomalies:
                    await session.refresh(anomaly)

                return anomalies

            except SQLAlchemyError as e:
                await session.rollback()
                logger.error(
                    "Failed to create traffic anomalies batch",
                    anomaly_count=len(anomalies_data),
                    error=str(e),
                )
                raise DatabaseError(
                    "Traffic anomalies batch creation failed", cause=e
                ) from e

    async def get_traffic_anomalies(
        self,
        camera_id: str | None = None,
        anomaly_type: str | None = None,
        min_score: float = 0.5,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 50,
    ) -> list[TrafficAnomaly]:
        """Get traffic anomalies with filtering options.

        Args:
            camera_id: Optional camera ID filter
            anomaly_type: Optional anomaly type filter
            min_score: Minimum anomaly score
            time_range: Optional time range filter
            limit: Maximum number of results

        Returns:
            List of traffic anomalies

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
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
                query = (
                    query.order_by(TrafficAnomaly.detection_time.desc()).limit(limit)
                )

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get traffic anomalies",
                    camera_id=camera_id,
                    anomaly_type=anomaly_type,
                    min_score=min_score,
                    error=str(e),
                )
                raise DatabaseError("Traffic anomalies retrieval failed", cause=e) from e

    # ============================
    # Analytics Report Operations
    # ============================

    async def get_analytics_summary(
        self,
        camera_ids: list[str],
        start_time: datetime,
        end_time: datetime,
    ) -> dict[str, Any]:
        """Get analytics summary for multiple cameras.

        Args:
            camera_ids: List of camera IDs
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Analytics summary dictionary

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                # Get traffic metrics summary
                metrics_query = select(
                    func.sum(TrafficMetrics.total_vehicles).label("total_vehicles")
                ).where(
                    and_(
                        TrafficMetrics.camera_id.in_(camera_ids),
                        TrafficMetrics.timestamp >= start_time,
                        TrafficMetrics.timestamp <= end_time,
                    )
                )
                metrics_result = await session.execute(metrics_query)
                total_vehicles = metrics_result.scalar() or 0

                # Get violations count
                violations_query = select(
                    func.count(RuleViolation.id).label("total_violations")
                ).where(
                    and_(
                        RuleViolation.camera_id.in_(camera_ids),
                        RuleViolation.detection_time >= start_time,
                        RuleViolation.detection_time <= end_time,
                    )
                )
                violations_result = await session.execute(violations_query)
                total_violations = violations_result.scalar() or 0

                # Get anomalies count
                anomalies_query = select(
                    func.count(TrafficAnomaly.id).label("total_anomalies")
                ).where(
                    and_(
                        TrafficAnomaly.camera_id.in_(camera_ids),
                        TrafficAnomaly.detection_time >= start_time,
                        TrafficAnomaly.detection_time <= end_time,
                    )
                )
                anomalies_result = await session.execute(anomalies_query)
                total_anomalies = anomalies_result.scalar() or 0

                return {
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                    },
                    "cameras": camera_ids,
                    "summary": {
                        "total_vehicles": int(total_vehicles),
                        "total_violations": int(total_violations),
                        "total_anomalies": int(total_anomalies),
                        "cameras_analyzed": len(camera_ids),
                    },
                    "generated_at": datetime.now().isoformat(),
                }

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get analytics summary",
                    camera_ids=camera_ids,
                    error=str(e),
                )
                raise DatabaseError("Analytics summary retrieval failed", cause=e) from e
