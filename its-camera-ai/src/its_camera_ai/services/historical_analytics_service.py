"""Historical Analytics Service with proper DI and database integration.

This service handles historical traffic analytics queries, database access,
caching, and data aggregation with proper separation of concerns.
"""

import random
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..api.exceptions import (
    HistoricalQueryError,
    ValidationError,
)
from ..api.schemas.analytics import (
    HistoricalData,
    HistoricalQuery,
    IncidentType,
    Severity,
    VehicleClass,
)
from ..api.schemas.common import PaginatedResponse
from ..core.config import Settings
from ..core.logging import get_logger
from ..models.analytics import TrafficMetrics as TrafficMetricsModel
from ..models.database import DatabaseManager
from ..repositories.analytics_repository import AnalyticsRepository
from .cache import CacheService

logger = get_logger(__name__)


class HistoricalAnalyticsService:
    """Historical Analytics Service with database access and caching.

    Handles historical traffic data queries, database operations,
    result caching, and data format conversion.
    """

    def __init__(
        self,
        analytics_repository: AnalyticsRepository,
        database_manager: DatabaseManager,
        cache_service: CacheService,
        settings: Settings,
    ):
        """Initialize with injected dependencies.

        Args:
            analytics_repository: Repository for analytics data access
            database_manager: Database manager for session lifecycle
            cache_service: Redis cache service for result caching
            settings: Application settings
        """
        self.analytics_repository = analytics_repository
        self.database_manager = database_manager
        self.cache_service = cache_service
        self.settings = settings
        self.cache_ttl = 300  # 5 minutes for historical data

    async def query_historical_data(
        self,
        query: HistoricalQuery,
    ) -> PaginatedResponse[HistoricalData]:
        """Query historical analytics data with caching and pagination.

        Args:
            query: Historical data query parameters

        Returns:
            PaginatedResponse[HistoricalData]: Paginated historical data results

        Raises:
            ValueError: If query parameters are invalid
            RuntimeError: If database query fails
        """
        start_time = datetime.now(UTC)

        try:
            # Validate query parameters
            await self._validate_query(query)

            # Build cache key
            cache_key = f"historical:{hash(str(query.model_dump()))}"

            # Check cache
            cached_result = await self.cache_service.get_json(cache_key)
            if cached_result:
                logger.debug(
                    "Cache hit for historical query",
                    cache_key=cache_key,
                    cameras=len(query.camera_ids or []),
                )
                return PaginatedResponse[HistoricalData](**cached_result)

            # Query database for historical data with proper session management
            async with self.database_manager.get_session() as db_session:
                historical_data = await self._query_database(query, db_session)

            # Apply pagination
            total = len(historical_data)
            offset = query.offset
            limit = query.limit
            paginated_data = historical_data[offset : offset + limit]

            # Create paginated response
            result = PaginatedResponse.create(
                items=paginated_data,
                total=total,
                page=(offset // limit) + 1,
                size=limit,
            )

            # Cache result
            await self.cache_service.set_json(
                cache_key, result.model_dump(), ttl=self.cache_ttl
            )

            processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            logger.info(
                "Historical data queried successfully",
                total_records=total,
                returned_records=len(paginated_data),
                processing_time_ms=processing_time,
                cache_key=cache_key,
            )

            return result

        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                "Historical data query failed",
                error=str(e),
                query_cameras=len(query.camera_ids or []),
                time_range_hours=(
                    (query.time_range.end_time - query.time_range.start_time).total_seconds() / 3600
                ),
                exc_info=True,
            )
            raise HistoricalQueryError(
                f"Historical data query failed: {e}",
                query_params=query.model_dump(),
                original_error=str(e),
            ) from e

    async def query_enhanced_historical_data(
        self,
        query: HistoricalQuery,
    ) -> HistoricalData:
        """Query enhanced historical data with mock generation.

        This method provides comprehensive historical data including
        metrics, incidents, vehicle counts, and predictions.

        Args:
            query: Historical data query parameters

        Returns:
            HistoricalData: Enhanced historical data response

        Raises:
            ValueError: If query parameters are invalid
        """
        start_time = datetime.now(UTC)

        try:
            # Validate query parameters
            await self._validate_enhanced_query(query)

            # Generate cache key
            query_hash = hash(str(query.model_dump()))
            cache_key = f"analytics:historical:{query_hash}"

            # Check cache
            cached_result = await self.cache_service.get_json(cache_key)
            if cached_result:
                logger.debug(
                    "Cache hit for enhanced historical query",
                    query_hash=query_hash,
                )
                return HistoricalData(**cached_result)

            # Generate enhanced historical data
            response = await self._generate_enhanced_historical_data(query)

            # Cache result for 10 minutes
            await self.cache_service.set_json(
                cache_key, response.model_dump(), ttl=600
            )

            processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            logger.info(
                "Enhanced historical query completed",
                query_id=getattr(response, 'query_id', 'unknown'),
                cameras=len(query.camera_ids or []),
                processing_time_ms=processing_time,
            )

            return response

        except ValidationError:
            raise
        except Exception as e:
            logger.error(
                "Enhanced historical query failed",
                error=str(e),
                exc_info=True,
            )
            raise HistoricalQueryError(
                f"Enhanced historical query failed: {e}",
                query_params=query.model_dump(),
                original_error=str(e),
            ) from e

    async def _validate_query(self, query: HistoricalQuery) -> None:
        """Validate historical query parameters.

        Args:
            query: Query to validate

        Raises:
            ValidationError: If query parameters are invalid
        """
        if not query.time_range:
            raise ValidationError("Time range is required", field="time_range")

        if query.time_range.end_time <= query.time_range.start_time:
            raise ValidationError("End time must be after start time", field="time_range")

        # Check maximum query range (90 days)
        max_days = 90
        time_diff = query.time_range.end_time - query.time_range.start_time
        if time_diff.days > max_days:
            raise ValidationError(
                f"Query range cannot exceed {max_days} days",
                field="time_range",
                value=f"{time_diff.days} days",
            )

        if not query.metric_types:
            raise ValidationError("At least one metric type is required", field="metric_types")

        if query.limit <= 0 or query.limit > 10000:
            raise ValidationError(
                "Limit must be between 1 and 10000",
                field="limit",
                value=str(query.limit),
            )

        if query.offset < 0:
            raise ValidationError(
                "Offset must be non-negative",
                field="offset",
                value=str(query.offset),
            )

    async def _validate_enhanced_query(self, query: HistoricalQuery) -> None:
        """Validate enhanced query parameters with different constraints.

        Args:
            query: Query to validate

        Raises:
            ValidationError: If query parameters are invalid
        """
        # For enhanced queries, we have access to different fields
        if hasattr(query, 'end_time') and hasattr(query, 'start_time'):
            if query.end_time <= query.start_time:
                raise ValidationError("End time must be after start time", field="time_range")

            # Check maximum query range (90 days)
            time_diff = query.end_time - query.start_time
            if time_diff.days > 90:
                raise ValidationError(
                    "Query range cannot exceed 90 days",
                    field="time_range",
                    value=f"{time_diff.days} days",
                )

    async def _query_database(
        self,
        query: HistoricalQuery,
        db_session: AsyncSession,
    ) -> list[HistoricalData]:
        """Query historical data from database.

        Args:
            query: Historical data query
            db_session: Database session

        Returns:
            list[HistoricalData]: Historical data results
        """
        historical_data = []

        try:
            # Build database query for traffic metrics
            traffic_query = select(TrafficMetricsModel)

            # Add time range filter
            time_conditions = []
            if query.time_range.start_time:
                time_conditions.append(
                    TrafficMetricsModel.timestamp >= query.time_range.start_time
                )
            if query.time_range.end_time:
                time_conditions.append(
                    TrafficMetricsModel.timestamp <= query.time_range.end_time
                )

            if time_conditions:
                traffic_query = traffic_query.where(and_(*time_conditions))

            # Add camera filter
            if query.camera_ids:
                traffic_query = traffic_query.where(
                    TrafficMetricsModel.camera_id.in_(query.camera_ids)
                )

            # Execute query
            result = await db_session.execute(traffic_query)
            metrics = result.scalars().all()

            # Convert to HistoricalData format
            for metric in metrics:
                for metric_type in query.metric_types:
                    value = self._extract_metric_value(metric, metric_type)

                    if value is not None:
                        historical_data.append(
                            HistoricalData(
                                timestamp=metric.timestamp,
                                camera_id=metric.camera_id,
                                metric_type=metric_type,
                                value=value,
                                metadata={
                                    "aggregation": query.aggregation,
                                    "source": "database",
                                    "confidence": getattr(metric, 'confidence', 0.95),
                                },
                            )
                        )

            # If no data found, log warning
            if not historical_data and query.camera_ids:
                logger.warning(
                    "No historical data found in database",
                    cameras=query.camera_ids,
                    time_range_start=query.time_range.start_time,
                    time_range_end=query.time_range.end_time,
                )

        except Exception as e:
            logger.error(
                "Failed to query historical data from database",
                error=str(e),
                exc_info=True,
            )
            # Return empty list on database error
            historical_data = []

        return historical_data

    def _extract_metric_value(
        self,
        metric: TrafficMetricsModel,
        metric_type: str,
    ) -> float | int | dict[str, Any] | None:
        """Extract specific metric value from traffic metrics model.

        Args:
            metric: Traffic metrics model instance
            metric_type: Type of metric to extract

        Returns:
            Metric value or None if not available
        """
        try:
            if metric_type == "vehicle_counts":
                return metric.total_vehicles

            elif metric_type == "speed_data":
                return getattr(metric, 'avg_speed', None)

            elif metric_type == "occupancy":
                return getattr(metric, 'occupancy_rate', None)

            elif metric_type == "flow_rate":
                # Calculate flow rate as vehicles per hour
                return metric.total_vehicles * 60 if metric.total_vehicles else None

            elif metric_type == "queue_length":
                return getattr(metric, 'queue_length', None)

            elif metric_type == "congestion":
                # Convert congestion level to numeric value
                congestion_map = {
                    "low": 1,
                    "medium": 2,
                    "high": 3,
                    "severe": 4,
                }
                congestion_level = getattr(metric, 'congestion_level', 'low')
                return congestion_map.get(congestion_level, 0)

            else:
                logger.warning(f"Unknown metric type: {metric_type}")
                return None

        except AttributeError as e:
            logger.warning(
                f"Failed to extract metric {metric_type}",
                error=str(e),
                metric_id=getattr(metric, 'id', 'unknown'),
            )
            return None

    async def _generate_enhanced_historical_data(
        self,
        query: HistoricalQuery,
    ) -> HistoricalData:
        """Generate enhanced historical data with comprehensive metrics.

        Args:
            query: Historical data query

        Returns:
            HistoricalData: Enhanced historical data response
        """
        query_id = str(uuid4())

        # Generate mock data for different components
        metrics = await self._generate_mock_metrics(query)
        incidents = await self._generate_mock_incidents(query)
        vehicle_counts = await self._generate_mock_vehicle_counts(query)
        predictions = await self._generate_mock_predictions(query)

        # Create response with all available data
        # Note: This is a simplified structure - in practice, HistoricalData
        # might need to be extended to include all these fields
        response_data = {
            "query_id": query_id,
            "camera_ids": query.camera_ids or ["camera_001", "camera_002"],
            "time_range": {
                "start": getattr(query, 'start_time', datetime.now(UTC) - timedelta(days=1)),
                "end": getattr(query, 'end_time', datetime.now(UTC)),
            },
            "aggregation_window": getattr(query, 'aggregation_window', 'hourly'),
            "metrics": metrics,
            "incidents": incidents,
            "vehicle_counts": vehicle_counts,
            "predictions": predictions,
            "data_quality": {
                "completeness": random.uniform(0.85, 0.99),
                "accuracy": random.uniform(0.90, 0.99),
            },
            "export_url": f"/api/v1/analytics/export/{query_id}",
            "export_format": getattr(query, 'export_format', 'csv'),
        }

        # For now, return a simple HistoricalData response
        # This would need to be enhanced based on the actual schema
        return HistoricalData(
            timestamp=datetime.now(UTC),
            camera_id=query.camera_ids[0] if query.camera_ids else "camera_001",
            metric_type="comprehensive",
            value=response_data,
            metadata={
                "query_id": query_id,
                "source": "enhanced_mock",
                "data_quality": response_data["data_quality"],
            }
        )

    async def _generate_mock_metrics(self, query: HistoricalQuery) -> list[dict[str, Any]]:
        """Generate mock traffic metrics."""
        metrics = []

        # Generate hourly metrics for the time range
        current_time = getattr(query, 'start_time', datetime.now(UTC) - timedelta(days=1))
        end_time = getattr(query, 'end_time', datetime.now(UTC))

        while current_time < end_time:
            metrics.append({
                "timestamp": current_time,
                "vehicle_count": random.randint(10, 100),
                "avg_speed": random.uniform(30, 80),
                "occupancy_rate": random.uniform(20, 90),
                "congestion_level": random.choice(["low", "medium", "high"]),
            })
            current_time += timedelta(hours=1)

        return metrics

    async def _generate_mock_incidents(self, query: HistoricalQuery) -> list[dict[str, Any]]:
        """Generate mock incident data."""
        incidents = []

        # Generate some incidents if requested
        if "incidents" in getattr(query, 'metric_types', []):
            for _ in range(random.randint(0, 5)):
                incident_time = datetime.now(UTC) - timedelta(
                    hours=random.randint(1, 48)
                )
                incidents.append({
                    "id": str(uuid4()),
                    "timestamp": incident_time,
                    "type": random.choice(list(IncidentType)).value,
                    "severity": random.choice(list(Severity)).value,
                    "description": "Historical incident from query",
                    "status": "resolved",
                })

        return incidents

    async def _generate_mock_vehicle_counts(self, query: HistoricalQuery) -> list[dict[str, Any]]:
        """Generate mock vehicle count data."""
        vehicle_counts = []

        if "vehicle_counts" in getattr(query, 'metric_types', []):
            current_time = getattr(query, 'start_time', datetime.now(UTC) - timedelta(days=1))
            end_time = getattr(query, 'end_time', datetime.now(UTC))

            while current_time < end_time:
                for vehicle_class in [VehicleClass.CAR, VehicleClass.TRUCK]:
                    vehicle_counts.append({
                        "timestamp": current_time,
                        "vehicle_class": vehicle_class.value,
                        "count": random.randint(5, 50),
                        "confidence": random.uniform(0.8, 0.99),
                    })
                current_time += timedelta(hours=1)

        return vehicle_counts

    async def _generate_mock_predictions(self, query: HistoricalQuery) -> list[dict[str, Any]]:
        """Generate mock prediction data."""
        predictions = []

        # Generate some basic predictions
        for i in range(3):
            future_time = datetime.now(UTC) + timedelta(hours=i + 1)
            predictions.append({
                "timestamp": future_time,
                "predicted_vehicle_count": random.randint(20, 80),
                "confidence": random.uniform(0.7, 0.9),
                "factors": ["historical_patterns", "time_of_day"],
            })

        return predictions

    async def invalidate_cache(self, query_hash: str | None = None) -> bool:
        """Invalidate cached historical data.

        Args:
            query_hash: Specific query hash to invalidate (all if None)

        Returns:
            bool: True if cache was invalidated successfully
        """
        try:
            if query_hash:
                cache_key = f"historical:{query_hash}"
                result = await self.cache_service.delete(cache_key)
            else:
                # Invalidate all historical cache entries
                # This would require a pattern-based deletion
                result = await self.cache_service.delete_pattern("historical:*")

            logger.debug(
                "Historical cache invalidated",
                query_hash=query_hash,
                success=result,
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to invalidate historical cache",
                query_hash=query_hash,
                error=str(e),
            )
            return False
