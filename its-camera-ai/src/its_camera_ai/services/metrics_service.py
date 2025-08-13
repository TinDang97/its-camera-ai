"""Async service for system metrics collection and monitoring.

Provides time-series data management, aggregation, and monitoring
integration for performance tracking and alerting.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.schemas.database import SystemMetricsCreateSchema
from ..core.exceptions import DatabaseError
from ..core.logging import get_logger
from ..models import AggregatedMetrics, MetricType, SystemMetrics
from .base_service import BaseAsyncService

logger = get_logger(__name__)


class MetricsService(BaseAsyncService[SystemMetrics]):
    """High-performance async service for system metrics management.

    Optimized for time-series data collection, aggregation,
    and monitoring dashboard queries.
    """

    def __init__(self, session: AsyncSession):
        super().__init__(session, SystemMetrics)

    async def record_metric(
        self,
        metric_data: SystemMetricsCreateSchema,
        timestamp: datetime | None = None,
    ) -> SystemMetrics:
        """Record a single system metric.

        Args:
            metric_data: Metric data to record
            timestamp: Optional custom timestamp

        Returns:
            Created metrics record
        """
        try:
            metric_dict = metric_data.model_dump()

            # Set timestamp if not provided
            if timestamp:
                metric_dict["timestamp"] = timestamp
            else:
                metric_dict["timestamp"] = datetime.now(UTC)

            metric = SystemMetrics(**metric_dict)

            self.session.add(metric)
            await self.session.commit()
            await self.session.refresh(metric)

            logger.debug(
                "Metric recorded",
                metric_name=metric.metric_name,
                value=metric.value,
                source=metric.source_id,
            )

            return metric

        except Exception as e:
            await self.session.rollback()
            logger.error(
                "Failed to record metric",
                metric_name=metric_data.metric_name,
                error=str(e),
            )
            raise DatabaseError(f"Failed to record metric: {str(e)}") from e

    async def batch_record_metrics(
        self, metrics_data: list[SystemMetricsCreateSchema]
    ) -> int:
        """Batch record multiple metrics for high throughput.

        Args:
            metrics_data: List of metrics to record

        Returns:
            Number of successfully recorded metrics
        """
        try:
            successful_count = 0
            current_time = datetime.now(UTC)

            for metric_data in metrics_data:
                try:
                    metric_dict = metric_data.model_dump()
                    metric_dict["timestamp"] = current_time

                    metric = SystemMetrics(**metric_dict)
                    self.session.add(metric)
                    successful_count += 1

                except Exception as e:
                    logger.warning(
                        "Failed to add metric to batch",
                        metric_name=getattr(metric_data, "metric_name", "unknown"),
                        error=str(e),
                    )

            if successful_count > 0:
                await self.session.commit()

            logger.info(
                "Batch metrics recording completed",
                total_requested=len(metrics_data),
                successful_count=successful_count,
                failed_count=len(metrics_data) - successful_count,
            )

            return successful_count

        except Exception as e:
            await self.session.rollback()
            logger.error(
                "Batch metrics recording failed",
                total_metrics=len(metrics_data),
                error=str(e),
            )
            raise DatabaseError(f"Failed to batch record metrics: {str(e)}") from e

    async def get_metrics_by_name(
        self,
        metric_name: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        source_id: str | None = None,
        limit: int = 1000,
    ) -> list[SystemMetrics]:
        """Get metrics by name with time range and source filtering.

        Args:
            metric_name: Name of the metric
            start_time: Optional start time filter
            end_time: Optional end time filter
            source_id: Optional source ID filter
            limit: Maximum number of metrics to return

        Returns:
            List of matching metrics ordered by timestamp
        """
        try:
            query = (
                select(SystemMetrics)
                .where(SystemMetrics.metric_name == metric_name)
                .order_by(SystemMetrics.timestamp.desc())
                .limit(limit)
            )

            # Apply time filters
            if start_time:
                query = query.where(SystemMetrics.timestamp >= start_time)
            if end_time:
                query = query.where(SystemMetrics.timestamp <= end_time)

            # Apply source filter
            if source_id:
                query = query.where(SystemMetrics.source_id == source_id)

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(
                "Failed to get metrics by name",
                metric_name=metric_name,
                error=str(e),
            )
            raise DatabaseError(f"Failed to get metrics: {str(e)}") from e

    async def get_recent_metrics(
        self,
        metric_types: list[MetricType] | None = None,
        sources: list[str] | None = None,
        minutes: int = 15,
        limit: int = 500,
    ) -> list[SystemMetrics]:
        """Get recent metrics for real-time monitoring.

        Args:
            metric_types: Optional list of metric types to filter
            sources: Optional list of source IDs to filter
            minutes: Time window in minutes
            limit: Maximum number of metrics

        Returns:
            List of recent metrics
        """
        try:
            start_time = datetime.now(UTC) - timedelta(minutes=minutes)

            query = (
                select(SystemMetrics)
                .where(SystemMetrics.timestamp >= start_time)
                .order_by(SystemMetrics.timestamp.desc())
                .limit(limit)
            )

            # Apply type filters
            if metric_types:
                type_values = [mt.value for mt in metric_types]
                query = query.where(SystemMetrics.metric_type.in_(type_values))

            # Apply source filters
            if sources:
                query = query.where(SystemMetrics.source_id.in_(sources))

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(
                "Failed to get recent metrics",
                metric_types=metric_types,
                sources=sources,
                error=str(e),
            )
            raise DatabaseError(f"Failed to get recent metrics: {str(e)}") from e

    async def get_metric_aggregations(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Get statistical aggregations for a metric over time range.

        Args:
            metric_name: Name of the metric
            start_time: Start time for aggregation
            end_time: End time for aggregation
            source_id: Optional source ID filter

        Returns:
            Dictionary with aggregated statistics
        """
        try:
            query = (
                select(
                    func.count(SystemMetrics.value).label('count'),
                    func.min(SystemMetrics.value).label('min_value'),
                    func.max(SystemMetrics.value).label('max_value'),
                    func.avg(SystemMetrics.value).label('avg_value'),
                    func.stddev(SystemMetrics.value).label('std_dev'),
                    func.percentile_cont(0.5).within_group(SystemMetrics.value).label('p50'),
                    func.percentile_cont(0.95).within_group(SystemMetrics.value).label('p95'),
                    func.percentile_cont(0.99).within_group(SystemMetrics.value).label('p99'),
                )
                .where(
                    and_(
                        SystemMetrics.metric_name == metric_name,
                        SystemMetrics.timestamp >= start_time,
                        SystemMetrics.timestamp <= end_time,
                    )
                )
            )

            if source_id:
                query = query.where(SystemMetrics.source_id == source_id)

            result = await self.session.execute(query)
            row = result.first()

            if not row or row.count == 0:
                return {
                    "count": 0,
                    "min_value": None,
                    "max_value": None,
                    "avg_value": None,
                    "std_dev": None,
                    "p50": None,
                    "p95": None,
                    "p99": None,
                }

            return {
                "count": row.count,
                "min_value": float(row.min_value) if row.min_value else None,
                "max_value": float(row.max_value) if row.max_value else None,
                "avg_value": round(float(row.avg_value), 2) if row.avg_value else None,
                "std_dev": round(float(row.std_dev), 2) if row.std_dev else None,
                "p50": round(float(row.p50), 2) if row.p50 else None,
                "p95": round(float(row.p95), 2) if row.p95 else None,
                "p99": round(float(row.p99), 2) if row.p99 else None,
            }

        except Exception as e:
            logger.error(
                "Failed to get metric aggregations",
                metric_name=metric_name,
                error=str(e),
            )
            raise DatabaseError(f"Failed to get metric aggregations: {str(e)}") from e

    async def get_alert_metrics(
        self, threshold_type: str = "critical", limit: int = 100
    ) -> list[SystemMetrics]:
        """Get metrics that exceed alert thresholds.

        Args:
            threshold_type: Type of threshold ('warning' or 'critical')
            limit: Maximum number of alert metrics

        Returns:
            List of metrics exceeding thresholds
        """
        try:
            if threshold_type == "critical":
                query = (
                    select(SystemMetrics)
                    .where(
                        and_(
                            SystemMetrics.critical_threshold.is_not(None),
                            SystemMetrics.value >= SystemMetrics.critical_threshold,
                        )
                    )
                    .order_by(SystemMetrics.timestamp.desc())
                    .limit(limit)
                )
            else:  # warning
                query = (
                    select(SystemMetrics)
                    .where(
                        and_(
                            SystemMetrics.warning_threshold.is_not(None),
                            SystemMetrics.value >= SystemMetrics.warning_threshold,
                        )
                    )
                    .order_by(SystemMetrics.timestamp.desc())
                    .limit(limit)
                )

            result = await self.session.execute(query)
            return list(result.scalars().all())

        except Exception as e:
            logger.error(
                "Failed to get alert metrics",
                threshold_type=threshold_type,
                error=str(e),
            )
            raise DatabaseError(f"Failed to get alert metrics: {str(e)}") from e

    async def cleanup_old_metrics(
        self, retention_days: int = 30, batch_size: int = 1000
    ) -> int:
        """Clean up old metrics beyond retention period.

        Args:
            retention_days: Number of days to retain metrics
            batch_size: Number of metrics to delete per batch

        Returns:
            Total number of metrics deleted
        """
        try:
            cutoff_time = datetime.now(UTC) - timedelta(days=retention_days)
            total_deleted = 0

            while True:
                # Delete in batches to avoid large transactions
                delete_query = (
                    delete(SystemMetrics)
                    .where(SystemMetrics.timestamp < cutoff_time)
                    .limit(batch_size)
                )

                result = await self.session.execute(delete_query)
                deleted_count = result.rowcount

                if deleted_count == 0:
                    break

                await self.session.commit()
                total_deleted += deleted_count

                logger.debug(
                    "Deleted metrics batch",
                    batch_size=deleted_count,
                    total_deleted=total_deleted,
                )

            logger.info(
                "Metrics cleanup completed",
                retention_days=retention_days,
                total_deleted=total_deleted,
            )

            return total_deleted

        except Exception as e:
            await self.session.rollback()
            logger.error(
                "Metrics cleanup failed",
                retention_days=retention_days,
                error=str(e),
            )
            raise DatabaseError(f"Failed to cleanup metrics: {str(e)}") from e

    async def create_hourly_aggregations(
        self, target_hour: datetime | None = None
    ) -> int:
        """Create hourly aggregated metrics for dashboard performance.

        Args:
            target_hour: Specific hour to aggregate (defaults to previous hour)

        Returns:
            Number of aggregations created
        """
        try:
            if not target_hour:
                # Default to previous complete hour
                now = datetime.now(UTC)
                target_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)

            period_start = target_hour
            period_end = target_hour + timedelta(hours=1)

            # Get unique metric names and sources for the hour
            unique_metrics_query = (
                select(
                    SystemMetrics.metric_name,
                    SystemMetrics.metric_type,
                    SystemMetrics.source_type,
                    SystemMetrics.source_id,
                )
                .where(
                    and_(
                        SystemMetrics.timestamp >= period_start,
                        SystemMetrics.timestamp < period_end,
                    )
                )
                .distinct()
            )

            unique_result = await self.session.execute(unique_metrics_query)
            unique_metrics = list(unique_result.all())

            created_count = 0

            for metric_row in unique_metrics:
                # Calculate aggregations for this metric/source combination
                agg_query = (
                    select(
                        func.count(SystemMetrics.value).label('sample_count'),
                        func.min(SystemMetrics.value).label('min_value'),
                        func.max(SystemMetrics.value).label('max_value'),
                        func.avg(SystemMetrics.value).label('avg_value'),
                        func.sum(SystemMetrics.value).label('sum_value'),
                        func.stddev(SystemMetrics.value).label('std_deviation'),
                        func.percentile_cont(0.5).within_group(SystemMetrics.value).label('p50_value'),
                        func.percentile_cont(0.95).within_group(SystemMetrics.value).label('p95_value'),
                        func.percentile_cont(0.99).within_group(SystemMetrics.value).label('p99_value'),
                    )
                    .where(
                        and_(
                            SystemMetrics.metric_name == metric_row.metric_name,
                            SystemMetrics.metric_type == metric_row.metric_type,
                            SystemMetrics.source_type == metric_row.source_type,
                            SystemMetrics.source_id == metric_row.source_id,
                            SystemMetrics.timestamp >= period_start,
                            SystemMetrics.timestamp < period_end,
                        )
                    )
                )

                agg_result = await self.session.execute(agg_query)
                agg_data = agg_result.first()

                if agg_data and agg_data.sample_count > 0:
                    # Create aggregated metric
                    aggregated_metric = AggregatedMetrics(
                        metric_name=metric_row.metric_name,
                        metric_type=metric_row.metric_type,
                        aggregation_period="1h",
                        period_start=period_start,
                        period_end=period_end,
                        source_type=metric_row.source_type,
                        source_id=metric_row.source_id,
                        sample_count=agg_data.sample_count,
                        min_value=float(agg_data.min_value),
                        max_value=float(agg_data.max_value),
                        avg_value=float(agg_data.avg_value),
                        sum_value=float(agg_data.sum_value),
                        std_deviation=float(agg_data.std_deviation) if agg_data.std_deviation else None,
                        p50_value=float(agg_data.p50_value) if agg_data.p50_value else None,
                        p95_value=float(agg_data.p95_value) if agg_data.p95_value else None,
                        p99_value=float(agg_data.p99_value) if agg_data.p99_value else None,
                    )

                    self.session.add(aggregated_metric)
                    created_count += 1

            if created_count > 0:
                await self.session.commit()

            logger.info(
                "Hourly aggregations created",
                target_hour=target_hour,
                created_count=created_count,
            )

            return created_count

        except Exception as e:
            await self.session.rollback()
            logger.error(
                "Failed to create hourly aggregations",
                target_hour=target_hour,
                error=str(e),
            )
            raise DatabaseError(f"Failed to create aggregations: {str(e)}") from e

    async def get_system_health_summary(self) -> dict[str, Any]:
        """Get comprehensive system health summary.

        Returns:
            System health summary with key metrics
        """
        try:
            # Get metrics from last 5 minutes for health check
            recent_time = datetime.now(UTC) - timedelta(minutes=5)

            # Performance metrics
            performance_query = (
                select(
                    func.avg(SystemMetrics.value).label('avg_value'),
                    func.max(SystemMetrics.value).label('max_value'),
                )
                .where(
                    and_(
                        SystemMetrics.metric_type == MetricType.PROCESSING_TIME,
                        SystemMetrics.timestamp >= recent_time,
                    )
                )
            )

            # Error rate metrics
            error_query = (
                select(func.avg(SystemMetrics.value).label('avg_error_rate'))
                .where(
                    and_(
                        SystemMetrics.metric_type == MetricType.ERROR_RATE,
                        SystemMetrics.timestamp >= recent_time,
                    )
                )
            )

            # Throughput metrics
            throughput_query = (
                select(func.avg(SystemMetrics.value).label('avg_throughput'))
                .where(
                    and_(
                        SystemMetrics.metric_type == MetricType.THROUGHPUT,
                        SystemMetrics.timestamp >= recent_time,
                    )
                )
            )

            # Execute queries
            performance_result = await self.session.execute(performance_query)
            error_result = await self.session.execute(error_query)
            throughput_result = await self.session.execute(throughput_query)

            performance_data = performance_result.first()
            error_data = error_result.first()
            throughput_data = throughput_result.first()

            # Determine overall health status
            avg_processing_time = performance_data.avg_value if performance_data else 0
            max_processing_time = performance_data.max_value if performance_data else 0
            avg_error_rate = error_data.avg_error_rate if error_data else 0
            avg_throughput = throughput_data.avg_throughput if throughput_data else 0

            # Health scoring logic
            health_score = 100

            # Penalize high processing times
            if avg_processing_time > 100:  # 100ms threshold
                health_score -= min(30, (avg_processing_time - 100) / 10)

            # Penalize high error rates
            if avg_error_rate > 1:  # 1% threshold
                health_score -= min(40, avg_error_rate * 10)

            # Penalize low throughput
            if avg_throughput < 10:  # 10 FPS threshold
                health_score -= min(20, (10 - avg_throughput) * 2)

            health_score = max(0, health_score)

            # Determine health status
            if health_score >= 90:
                health_status = "excellent"
            elif health_score >= 75:
                health_status = "good"
            elif health_score >= 50:
                health_status = "fair"
            else:
                health_status = "poor"

            return {
                "health_score": round(health_score, 1),
                "health_status": health_status,
                "timestamp": datetime.now(UTC),
                "metrics_summary": {
                    "avg_processing_time_ms": round(avg_processing_time or 0, 2),
                    "max_processing_time_ms": round(max_processing_time or 0, 2),
                    "avg_error_rate_percent": round(avg_error_rate or 0, 2),
                    "avg_throughput_fps": round(avg_throughput or 0, 2),
                },
                "last_updated": recent_time,
            }

        except Exception as e:
            logger.error(
                "Failed to get system health summary",
                error=str(e),
            )
            raise DatabaseError(f"Failed to get system health: {str(e)}") from e
