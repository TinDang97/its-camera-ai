"""Metrics repository for system metrics data access.

Provides specialized methods for metrics collection, analytics,
and performance monitoring with time-series optimizations.
"""

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from ..models.system_metrics import SystemMetrics
from .base_repository import BaseRepository

logger = get_logger(__name__)


class MetricsRepository(BaseRepository[SystemMetrics]):
    """Repository for system metrics data access operations.

    Specialized methods for metrics analytics, time-series data,
    and performance monitoring with optimized queries for TimescaleDB.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        super().__init__(session_factory, SystemMetrics)

    async def get_by_service(
        self, service_name: str, limit: int = 1000, offset: int = 0
    ) -> list[SystemMetrics]:
        """Get metrics by service name.

        Args:
            service_name: Service name to filter by
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of metrics for the specified service

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = (
                    select(SystemMetrics)
                    .where(SystemMetrics.service_name == service_name)
                    .order_by(SystemMetrics.timestamp.desc())
                    .limit(limit)
                    .offset(offset)
                )

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get metrics by service",
                    service_name=service_name,
                    error=str(e),
                )
                raise DatabaseError("Metrics retrieval failed", cause=e) from e

    async def get_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        service_name: str | None = None,
        metric_name: str | None = None,
        limit: int = 10000,
    ) -> list[SystemMetrics]:
        """Get metrics within a time range.

        Args:
            start_time: Start of time range
            end_time: End of time range
            service_name: Optional service name to filter by
            metric_name: Optional metric name to filter by
            limit: Maximum number of results

        Returns:
            List of metrics within the time range

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                query = select(SystemMetrics).where(
                    and_(
                        SystemMetrics.timestamp >= start_time,
                        SystemMetrics.timestamp <= end_time,
                    )
                )

                if service_name:
                    query = query.where(SystemMetrics.service_name == service_name)

                if metric_name:
                    query = query.where(SystemMetrics.metric_name == metric_name)

                query = query.order_by(SystemMetrics.timestamp.desc()).limit(limit)

                result = await session.execute(query)
                return list(result.scalars().all())

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get metrics by time range",
                    start_time=start_time.isoformat(),
                    end_time=end_time.isoformat(),
                    service_name=service_name,
                    metric_name=metric_name,
                    error=str(e),
                )
                raise DatabaseError("Metrics retrieval failed", cause=e) from e

    async def get_recent_metrics(
        self,
        service_name: str | None = None,
        metric_name: str | None = None,
        minutes: int = 5,
        limit: int = 1000,
    ) -> list[SystemMetrics]:
        """Get recent metrics.

        Args:
            service_name: Optional service name to filter by
            metric_name: Optional metric name to filter by
            minutes: Number of minutes to look back
            limit: Maximum number of results

        Returns:
            List of recent metrics

        Raises:
            DatabaseError: If query fails
        """
        start_time = datetime.now() - timedelta(minutes=minutes)
        return await self.get_by_time_range(
            start_time=start_time,
            end_time=datetime.now(),
            service_name=service_name,
            metric_name=metric_name,
            limit=limit,
        )

    async def get_average_metrics(
        self,
        metric_name: str,
        service_name: str | None = None,
        hours: int = 24,
        interval_minutes: int = 5,
    ) -> list[dict[str, Any]]:
        """Get averaged metrics over time intervals.

        Args:
            metric_name: Metric name to aggregate
            service_name: Optional service name to filter by
            hours: Number of hours to analyze
            interval_minutes: Aggregation interval in minutes

        Returns:
            List of averaged metrics

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                start_time = datetime.now() - timedelta(hours=hours)

                # PostgreSQL-specific time bucketing
                bucket_size = f"{interval_minutes} minutes"

                query = select(
                    func.date_trunc("hour", SystemMetrics.timestamp).label(
                        "time_bucket"
                    ),
                    func.avg(SystemMetrics.value).label("avg_value"),
                    func.min(SystemMetrics.value).label("min_value"),
                    func.max(SystemMetrics.value).label("max_value"),
                    func.count(SystemMetrics.id).label("count"),
                ).where(
                    and_(
                        SystemMetrics.metric_name == metric_name,
                        SystemMetrics.timestamp >= start_time,
                    )
                )

                if service_name:
                    query = query.where(SystemMetrics.service_name == service_name)

                query = query.group_by("time_bucket").order_by("time_bucket")

                result = await session.execute(query)

                return [
                    {
                        "timestamp": row.time_bucket,
                        "avg_value": float(row.avg_value or 0),
                        "min_value": float(row.min_value or 0),
                        "max_value": float(row.max_value or 0),
                        "count": row.count,
                    }
                    for row in result.all()
                ]

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get average metrics",
                    metric_name=metric_name,
                    service_name=service_name,
                    hours=hours,
                    error=str(e),
                )
                raise DatabaseError("Average metrics retrieval failed", cause=e) from e

    async def get_service_health_metrics(
        self, service_name: str, hours: int = 1
    ) -> dict[str, Any]:
        """Get comprehensive health metrics for a service.

        Args:
            service_name: Service name
            hours: Number of hours to analyze

        Returns:
            Dictionary with health metrics

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                start_time = datetime.now() - timedelta(hours=hours)

                # Get various metrics for the service
                query = (
                    select(
                        SystemMetrics.metric_name,
                        func.avg(SystemMetrics.value).label("avg_value"),
                        func.min(SystemMetrics.value).label("min_value"),
                        func.max(SystemMetrics.value).label("max_value"),
                        func.count(SystemMetrics.id).label("count"),
                    )
                    .where(
                        and_(
                            SystemMetrics.service_name == service_name,
                            SystemMetrics.timestamp >= start_time,
                        )
                    )
                    .group_by(SystemMetrics.metric_name)
                )

                result = await session.execute(query)

                metrics = {}
                for row in result.all():
                    metrics[row.metric_name] = {
                        "avg_value": float(row.avg_value or 0),
                        "min_value": float(row.min_value or 0),
                        "max_value": float(row.max_value or 0),
                        "count": row.count,
                    }

                # Calculate health score based on key metrics
                health_score = await self._calculate_health_score(metrics)

                return {
                    "service_name": service_name,
                    "time_range_hours": hours,
                    "metrics": metrics,
                    "health_score": health_score,
                    "analyzed_at": datetime.now(),
                }

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get service health metrics",
                    service_name=service_name,
                    hours=hours,
                    error=str(e),
                )
                raise DatabaseError("Health metrics retrieval failed", cause=e) from e

    async def _calculate_health_score(self, metrics: dict[str, Any]) -> float:
        """Calculate health score based on metrics.

        Args:
            metrics: Dictionary of metric statistics

        Returns:
            Health score between 0.0 and 1.0
        """
        score = 1.0

        # Penalize high error rates
        if "error_rate" in metrics:
            error_rate = metrics["error_rate"]["avg_value"]
            score *= max(0.0, 1.0 - error_rate)

        # Penalize high response times
        if "response_time_ms" in metrics:
            response_time = metrics["response_time_ms"]["avg_value"]
            if response_time > 1000:  # 1 second threshold
                score *= max(0.0, 1.0 - (response_time - 1000) / 10000)

        # Penalize high CPU usage
        if "cpu_usage_percent" in metrics:
            cpu_usage = metrics["cpu_usage_percent"]["avg_value"]
            if cpu_usage > 80:  # 80% threshold
                score *= max(0.0, 1.0 - (cpu_usage - 80) / 20)

        # Penalize high memory usage
        if "memory_usage_percent" in metrics:
            memory_usage = metrics["memory_usage_percent"]["avg_value"]
            if memory_usage > 80:  # 80% threshold
                score *= max(0.0, 1.0 - (memory_usage - 80) / 20)

        return max(0.0, min(1.0, score))

    async def get_performance_trends(
        self, metric_names: list[str], service_name: str | None = None, days: int = 7
    ) -> dict[str, list[dict[str, Any]]]:
        """Get performance trends for specified metrics.

        Args:
            metric_names: List of metric names to analyze
            service_name: Optional service name to filter by
            days: Number of days to analyze

        Returns:
            Dictionary with trend data for each metric

        Raises:
            DatabaseError: If query fails
        """
        async with self._get_session() as session:
            try:
                start_time = datetime.now() - timedelta(days=days)
                trends = {}

                for metric_name in metric_names:
                    query = select(
                        func.date_trunc("hour", SystemMetrics.timestamp).label("hour"),
                        func.avg(SystemMetrics.value).label("avg_value"),
                    ).where(
                        and_(
                            SystemMetrics.metric_name == metric_name,
                            SystemMetrics.timestamp >= start_time,
                        )
                    )

                    if service_name:
                        query = query.where(SystemMetrics.service_name == service_name)

                    query = query.group_by("hour").order_by("hour")

                    result = await session.execute(query)

                    trends[metric_name] = [
                        {
                            "timestamp": row.hour,
                            "value": float(row.avg_value or 0),
                        }
                        for row in result.all()
                    ]

                return trends

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to get performance trends",
                    metric_names=metric_names,
                    service_name=service_name,
                    days=days,
                    error=str(e),
                )
                raise DatabaseError(
                    "Performance trends retrieval failed", cause=e
                ) from e

    async def cleanup_old_metrics(
        self, older_than_days: int = 30, batch_size: int = 10000
    ) -> int:
        """Clean up old metrics to manage storage.

        Args:
            older_than_days: Delete metrics older than this many days
            batch_size: Number of metrics to delete per batch

        Returns:
            Number of metrics deleted

        Raises:
            DatabaseError: If cleanup fails
        """
        async with self._get_session() as session:
            try:
                cutoff_date = datetime.now() - timedelta(days=older_than_days)

                # Get metric IDs to delete
                query = (
                    select(SystemMetrics.id)
                    .where(SystemMetrics.timestamp < cutoff_date)
                    .limit(batch_size)
                )

                result = await session.execute(query)
                metric_ids = [row.id for row in result.all()]

                if not metric_ids:
                    return 0

                # Delete metrics in batch
                from sqlalchemy import delete

                delete_query = delete(SystemMetrics).where(
                    SystemMetrics.id.in_(metric_ids)
                )
                result = await session.execute(delete_query)
                await session.commit()

                deleted_count = result.rowcount

                logger.info(
                    "Cleaned up old metrics",
                    deleted_count=deleted_count,
                    cutoff_date=cutoff_date.isoformat(),
                )

                return deleted_count

            except SQLAlchemyError as e:
                logger.error(
                    "Failed to cleanup old metrics",
                    older_than_days=older_than_days,
                    error=str(e),
                )
                raise DatabaseError("Metrics cleanup failed", cause=e) from e
