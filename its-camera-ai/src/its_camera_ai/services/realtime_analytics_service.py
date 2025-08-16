"""Real-time Analytics Service with proper DI and single responsibility.

This service handles real-time traffic analytics generation, caching,
and data aggregation with proper separation of concerns from the router layer.
"""

import random
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from ..api.schemas.analytics import (
    AnalyticsResponse,
    IncidentAlert,
    IncidentType,
    Severity,
    TrafficDirection,
    TrafficMetrics,
    VehicleClass,
    VehicleCount,
)
from ..core.config import Settings
from ..core.logging import get_logger
from .cache import CacheService

logger = get_logger(__name__)


class RealtimeAnalyticsService:
    """Real-time Analytics Service with proper DI and caching.

    Handles generation of real-time traffic analytics, vehicle counts,
    traffic metrics, and incident detection with proper caching strategy.
    """

    def __init__(
        self,
        cache_service: CacheService,
        settings: Settings,
    ):
        """Initialize with injected dependencies.

        Args:
            cache_service: Redis cache service for result caching
            settings: Application settings
        """
        self.cache_service = cache_service
        self.settings = settings
        self.cache_ttl = 10  # seconds for real-time data

    async def get_realtime_analytics(
        self,
        camera_id: str,
        time_window_minutes: int = 5,
    ) -> AnalyticsResponse:
        """Get real-time analytics for a camera with caching.

        Args:
            camera_id: Camera identifier
            time_window_minutes: Time window for analytics in minutes

        Returns:
            AnalyticsResponse: Real-time analytics data

        Raises:
            ValueError: If camera_id is invalid
            RuntimeError: If analytics generation fails
        """
        if not camera_id or not camera_id.strip():
            raise ValueError("Camera ID cannot be empty")

        start_time = datetime.now(UTC)

        try:
            # Check cache first
            cache_key = f"analytics:realtime:{camera_id}"
            cached_data = await self.cache_service.get_json(cache_key)
            if cached_data:
                logger.debug(
                    "Returning cached real-time analytics",
                    camera_id=camera_id,
                    cache_key=cache_key,
                )
                return AnalyticsResponse(**cached_data)

            # Generate fresh analytics data
            analytics_response = await self._generate_realtime_analytics(
                camera_id, time_window_minutes
            )

            # Cache the result
            await self.cache_service.set_json(
                cache_key, analytics_response.model_dump(), ttl=self.cache_ttl
            )

            processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            logger.debug(
                "Real-time analytics generated",
                camera_id=camera_id,
                processing_time_ms=processing_time,
                vehicle_count=len(analytics_response.vehicle_counts),
                incident_count=len(analytics_response.active_incidents),
            )

            return analytics_response

        except Exception as e:
            logger.error(
                "Failed to get real-time analytics",
                camera_id=camera_id,
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Analytics generation failed: {e}") from e

    async def _generate_realtime_analytics(
        self,
        camera_id: str,
        time_window_minutes: int,
    ) -> AnalyticsResponse:
        """Generate real-time analytics data.

        Args:
            camera_id: Camera identifier
            time_window_minutes: Time window for analytics

        Returns:
            AnalyticsResponse: Generated analytics data
        """
        now = datetime.now(UTC)
        window_start = now - timedelta(minutes=time_window_minutes)

        # Generate vehicle counts
        vehicle_counts = await self._generate_vehicle_counts(
            camera_id, window_start, now
        )

        # Generate traffic metrics
        traffic_metrics = await self._generate_traffic_metrics(
            camera_id, window_start, now, vehicle_counts
        )

        # Generate active incidents
        active_incidents = await self._generate_active_incidents(
            camera_id, window_start, now
        )

        return AnalyticsResponse(
            camera_id=camera_id,
            timestamp=now,
            vehicle_counts=vehicle_counts,
            traffic_metrics=traffic_metrics,
            active_incidents=active_incidents,
            processing_time=random.uniform(50, 150),
            frame_rate=random.uniform(25, 30),
            detection_zones=["main_road", "intersection", "parking_area"],
        )

    async def _generate_vehicle_counts(
        self,
        camera_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[VehicleCount]:
        """Generate mock vehicle count data.

        Args:
            camera_id: Camera identifier
            start_time: Start time for data generation
            end_time: End time for data generation

        Returns:
            list[VehicleCount]: Generated vehicle count data
        """
        counts = []
        current_time = start_time

        while current_time < end_time:
            # Generate counts for different vehicle classes
            for vehicle_class in [VehicleClass.CAR, VehicleClass.TRUCK, VehicleClass.VAN]:
                count = random.randint(0, 20)
                if count > 0:
                    counts.append(
                        VehicleCount(
                            camera_id=camera_id,
                            timestamp=current_time,
                            vehicle_class=vehicle_class,
                            direction=random.choice(list(TrafficDirection)),
                            count=count,
                            confidence=random.uniform(0.8, 0.99),
                            lane=random.choice(["lane_1", "lane_2", "lane_3"]),
                            speed=random.uniform(30, 80),
                        )
                    )

            # Move to next interval (1-minute intervals)
            current_time += timedelta(minutes=1)

        return counts

    async def _generate_traffic_metrics(
        self,
        camera_id: str,
        start_time: datetime,
        end_time: datetime,
        vehicle_counts: list[VehicleCount],
    ) -> TrafficMetrics:
        """Generate traffic metrics from vehicle counts.

        Args:
            camera_id: Camera identifier
            start_time: Period start time
            end_time: Period end time
            vehicle_counts: Vehicle count data

        Returns:
            TrafficMetrics: Generated traffic metrics
        """
        # Calculate total vehicles
        total_vehicles = sum(vc.count for vc in vehicle_counts)

        # Calculate vehicle breakdown
        vehicle_breakdown = {}
        for vehicle_class in VehicleClass:
            count = sum(
                vc.count for vc in vehicle_counts if vc.vehicle_class == vehicle_class
            )
            if count > 0:
                vehicle_breakdown[vehicle_class] = count

        # Calculate directional flow
        directional_flow = {}
        for direction in TrafficDirection:
            count = sum(
                vc.count
                for vc in vehicle_counts
                if vc.direction == direction and vc.direction is not None
            )
            if count > 0:
                directional_flow[direction] = count

        # Generate additional metrics
        avg_speed = random.uniform(45, 75) if vehicle_counts else None
        occupancy_rate = random.uniform(20, 80)
        congestion_level = random.choice(["low", "medium", "high"])
        queue_length = random.uniform(0, 50)

        return TrafficMetrics(
            camera_id=camera_id,
            period_start=start_time,
            period_end=end_time,
            total_vehicles=total_vehicles,
            vehicle_breakdown=vehicle_breakdown,
            directional_flow=directional_flow,
            avg_speed=avg_speed,
            peak_hour=None,  # Would be calculated from historical data
            occupancy_rate=occupancy_rate,
            congestion_level=congestion_level,
            queue_length=queue_length,
        )

    async def _generate_active_incidents(
        self,
        camera_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[IncidentAlert]:
        """Generate mock active incident data.

        Args:
            camera_id: Camera identifier
            start_time: Time window start
            end_time: Time window end

        Returns:
            list[IncidentAlert]: Active incidents
        """
        incidents = []
        current_time = start_time

        while current_time < end_time:
            # Random chance of incident (low probability)
            if random.random() < 0.05:  # 5% chance per time period
                incident_type = random.choice(list(IncidentType))
                severity = random.choice(list(Severity))
                status = "active" if random.random() > 0.3 else "resolved"

                # Only include active incidents
                if status == "active":
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
                        status=status,
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
                        resolved_at=None,
                        resolved_by=None,
                        notes=None,
                    )
                    incidents.append(incident)

            current_time += timedelta(minutes=30)  # Check every 30 minutes

        return incidents

    async def invalidate_cache(self, camera_id: str) -> bool:
        """Invalidate cached analytics for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            bool: True if cache was invalidated successfully
        """
        try:
            cache_key = f"analytics:realtime:{camera_id}"
            result = await self.cache_service.delete(cache_key)

            logger.debug(
                "Cache invalidated",
                camera_id=camera_id,
                cache_key=cache_key,
                success=result,
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to invalidate cache",
                camera_id=camera_id,
                error=str(e),
            )
            return False

    async def get_analytics_status(self, camera_id: str) -> dict[str, Any]:
        """Get analytics generation status for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            dict: Analytics status information
        """
        try:
            cache_key = f"analytics:realtime:{camera_id}"
            cached_data = await self.cache_service.get_json(cache_key)

            return {
                "camera_id": camera_id,
                "cache_status": "hit" if cached_data else "miss",
                "cache_key": cache_key,
                "cache_ttl": self.cache_ttl,
                "service_status": "operational",
                "last_generated": (
                    cached_data.get("timestamp") if cached_data else None
                ),
            }

        except Exception as e:
            logger.error(
                "Failed to get analytics status",
                camera_id=camera_id,
                error=str(e),
            )
            return {
                "camera_id": camera_id,
                "service_status": "error",
                "error": str(e),
            }

    async def update_cache_ttl(self, new_ttl: int) -> None:
        """Update cache TTL for real-time analytics.

        Args:
            new_ttl: New TTL value in seconds
        """
        if new_ttl <= 0:
            raise ValueError("TTL must be positive")

        old_ttl = self.cache_ttl
        self.cache_ttl = new_ttl

        logger.info(
            "Cache TTL updated",
            old_ttl=old_ttl,
            new_ttl=new_ttl,
        )
