"""Unified Analytics Service with comprehensive traffic analysis capabilities.

This service consolidates all analytics functionality including metrics aggregation,
incident detection, rule evaluation, speed calculation, and anomaly detection.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from ..core.config import Settings
from ..core.logging import get_logger
from ..repositories.analytics_repository import AnalyticsRepository
from .analytics_aggregation_service import AnalyticsAggregationService
from .analytics_dtos import (
    AggregatedMetricsDTO,
    AggregationLevel,
    AnomalyResult,
    DetectionData,
    IncidentAlertDTO,
    ProcessingResult,
    RealtimeTrafficMetrics,
    SpeedMeasurement,
    TimeWindow,
    TrafficDataPoint,
    ViolationRecord,
)
from .anomaly_detection_service import AnomalyDetectionService
from .cache import CacheService
from .incident_detection_service import IncidentDetectionService
from .speed_calculation_service import SpeedCalculationService
from .traffic_rule_service import TrafficRuleService

logger = get_logger(__name__)


class UnifiedAnalyticsService:
    """Unified analytics service with comprehensive traffic analysis.

    Coordinates all analytics components: aggregation, incident detection,
    rule evaluation, speed calculation, and anomaly detection.
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

        # Initialize component services
        self.aggregation_service = AnalyticsAggregationService(
            analytics_repository=analytics_repository,
            cache_service=cache_service,
            settings=settings,
        )

        self.incident_detection_service = IncidentDetectionService(
            alert_repository=None,  # TODO: Create AlertRepository to handle alerts
            cache_service=cache_service,
            settings=settings,
        )

        self.traffic_rule_service = TrafficRuleService(
            analytics_repository=analytics_repository,
            cache_service=cache_service,
            settings=settings,
        )

        self.speed_calculation_service = SpeedCalculationService(
            analytics_repository=analytics_repository,
            cache_service=cache_service,
            settings=settings,
        )

        self.anomaly_detection_service = AnomalyDetectionService(
            analytics_repository=analytics_repository,
            cache_service=cache_service,
            settings=settings,
        )

    async def process_realtime_analytics(
        self,
        detection_data: DetectionData,
        include_anomaly_detection: bool = True,
        include_incident_detection: bool = True,
        include_rule_evaluation: bool = True,
        include_speed_calculation: bool = True,
    ) -> ProcessingResult:
        """Process comprehensive real-time analytics.

        Args:
            detection_data: Detection data from ML models
            include_anomaly_detection: Whether to run anomaly detection
            include_incident_detection: Whether to run incident detection
            include_rule_evaluation: Whether to evaluate traffic rules
            include_speed_calculation: Whether to calculate speeds

        Returns:
            Comprehensive processing result
        """
        start_time = datetime.now(UTC)

        try:
            # Create realtime metrics
            realtime_metrics = self._create_realtime_metrics(detection_data)

            # Initialize result containers
            violations: list[ViolationRecord] = []
            anomalies: list[AnomalyResult] = []
            incidents: list[IncidentAlertDTO] = []
            speed_measurements: list[SpeedMeasurement] = []

            # Traffic rule evaluation
            if include_rule_evaluation:
                violations = await self.traffic_rule_service.evaluate_violations(
                    detection_data
                )

            # Speed calculation
            if include_speed_calculation and detection_data.detections:
                trajectory_data = self._extract_trajectory_data(detection_data)
                if trajectory_data:
                    speed_measurements = (
                        await self.speed_calculation_service.calculate_speed_batch(
                            trajectory_data, detection_data.camera_id
                        )
                    )

            # Anomaly detection
            if include_anomaly_detection:
                traffic_data_point = self._convert_to_traffic_data_point(
                    detection_data, realtime_metrics
                )
                anomalies = await self.anomaly_detection_service.detect_anomalies(
                    [traffic_data_point], camera_id=detection_data.camera_id
                )

            # Incident detection
            if include_incident_detection:
                incident = await self.incident_detection_service.process_detection(
                    detection_data
                )
                if incident:
                    incidents.append(incident)

            # Calculate processing time
            processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            # Create comprehensive result
            result = ProcessingResult(
                camera_id=detection_data.camera_id,
                timestamp=detection_data.timestamp,
                vehicle_count=detection_data.vehicle_count,
                metrics=realtime_metrics,
                violations=violations,
                anomalies=anomalies,
                processing_time_ms=processing_time,
            )

            # Cache the result for real-time dashboards
            await self._cache_processing_result(result)

            logger.info(
                f"Processed analytics for camera {detection_data.camera_id}: "
                f"{len(violations)} violations, {len(anomalies)} anomalies, "
                f"{len(incidents)} incidents in {processing_time:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Real-time analytics processing failed: {e}")
            # Return minimal result on error
            processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            return ProcessingResult(
                camera_id=detection_data.camera_id,
                timestamp=detection_data.timestamp,
                vehicle_count=detection_data.vehicle_count,
                metrics=self._create_realtime_metrics(detection_data),
                violations=[],
                anomalies=[],
                processing_time_ms=processing_time,
                detection_quality=0.0,
            )

    async def generate_aggregated_report(
        self,
        camera_ids: list[str],
        start_time: datetime,
        end_time: datetime,
        aggregation_level: AggregationLevel = AggregationLevel.HOURLY,
        include_anomalies: bool = True,
        include_violations: bool = True,
    ) -> list[AggregatedMetricsDTO]:
        """Generate comprehensive aggregated analytics report.

        Args:
            camera_ids: List of camera IDs
            start_time: Report start time
            end_time: Report end time
            aggregation_level: Level of data aggregation
            include_anomalies: Whether to include anomaly analysis
            include_violations: Whether to include violation analysis

        Returns:
            List of aggregated metrics with enhanced analysis
        """
        try:
            # Get base aggregated metrics
            time_window = self._get_time_window_from_range(start_time, end_time)

            aggregated_metrics = (
                await self.aggregation_service.aggregate_traffic_metrics(
                    camera_ids=camera_ids,
                    time_window=time_window,
                    aggregation_level=aggregation_level,
                    start_time=start_time,
                    end_time=end_time,
                    include_quality_check=True,
                )
            )

            # Enhance with additional analytics if requested
            for metrics in aggregated_metrics:
                # Add anomaly analysis
                if include_anomalies:
                    anomaly_count = await self._count_anomalies(
                        metrics.camera_id, metrics.start_time, metrics.end_time
                    )
                    # Would add anomaly info to metrics object if extended

                # Add violation analysis
                if include_violations:
                    violation_count = await self._count_violations(
                        metrics.camera_id, metrics.start_time, metrics.end_time
                    )
                    # Would add violation info to metrics object if extended

            logger.info(
                f"Generated aggregated report for {len(camera_ids)} cameras, "
                f"period: {start_time} to {end_time}"
            )

            return aggregated_metrics

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return []

    async def get_camera_health_status(self, camera_id: str) -> dict[str, Any]:
        """Get comprehensive health status for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            Health status information
        """
        try:
            # Get recent metrics
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(hours=1)

            recent_metrics = await self.aggregation_service.aggregate_traffic_metrics(
                camera_ids=[camera_id],
                time_window=TimeWindow.ONE_HOUR,
                aggregation_level=AggregationLevel.RAW,
                start_time=start_time,
                end_time=end_time,
            )

            # Get calibration status
            calibration_status = (
                await self.speed_calculation_service.get_camera_calibration_status(
                    camera_id
                )
            )

            # Calculate health metrics
            if recent_metrics:
                metrics = recent_metrics[0]
                data_quality = metrics.quality_metrics.data_quality_score
                data_completeness = metrics.quality_metrics.data_completeness
                detection_count = metrics.sample_count
            else:
                data_quality = 0.0
                data_completeness = 0.0
                detection_count = 0

            # Check for recent anomalies
            anomaly_count = await self._count_anomalies(camera_id, start_time, end_time)

            # Check for recent violations
            violation_count = await self._count_violations(
                camera_id, start_time, end_time
            )

            # Calculate overall health score
            health_score = self._calculate_health_score(
                data_quality, data_completeness, anomaly_count, violation_count
            )

            return {
                "camera_id": camera_id,
                "timestamp": datetime.now(UTC),
                "health_score": health_score,
                "data_quality_score": data_quality,
                "data_completeness": data_completeness,
                "detection_count": detection_count,
                "anomaly_count": anomaly_count,
                "violation_count": violation_count,
                "calibration_status": calibration_status,
                "status": (
                    "healthy"
                    if health_score > 0.8
                    else "degraded" if health_score > 0.5 else "unhealthy"
                ),
            }

        except Exception as e:
            logger.error(f"Health status check failed: {e}")
            return {
                "camera_id": camera_id,
                "timestamp": datetime.now(UTC),
                "health_score": 0.0,
                "status": "error",
                "error": str(e),
            }

    def _create_realtime_metrics(
        self, detection_data: DetectionData
    ) -> RealtimeTrafficMetrics:
        """Create realtime traffic metrics from detection data.

        Args:
            detection_data: Detection data

        Returns:
            Realtime traffic metrics
        """
        # Extract vehicle breakdown
        vehicle_breakdown = {}
        for detection in detection_data.detections:
            vehicle_type = detection.get("vehicle_type", "unknown")
            vehicle_breakdown[vehicle_type] = vehicle_breakdown.get(vehicle_type, 0) + 1

        # Calculate average speed
        speeds = [
            d.get("speed", 0)
            for d in detection_data.detections
            if d.get("speed") and d.get("speed") > 0
        ]
        average_speed = sum(speeds) / len(speeds) if speeds else None

        # Estimate traffic density and congestion
        vehicle_count = detection_data.vehicle_count
        traffic_density = min(1.0, vehicle_count / 20.0)  # Normalize to 0-1

        # Determine congestion level
        from .analytics_dtos import CongestionLevel

        if traffic_density < 0.2:
            congestion_level = CongestionLevel.FREE_FLOW
        elif traffic_density < 0.4:
            congestion_level = CongestionLevel.LIGHT
        elif traffic_density < 0.6:
            congestion_level = CongestionLevel.MODERATE
        elif traffic_density < 0.8:
            congestion_level = CongestionLevel.HEAVY
        else:
            congestion_level = CongestionLevel.SEVERE

        return RealtimeTrafficMetrics(
            timestamp=detection_data.timestamp,
            camera_id=detection_data.camera_id,
            total_vehicles=vehicle_count,
            vehicle_breakdown=vehicle_breakdown,
            average_speed=average_speed,
            traffic_density=traffic_density,
            congestion_level=congestion_level,
            flow_rate=(
                vehicle_count * 60 if vehicle_count > 0 else None
            ),  # vehicles per hour
            occupancy_rate=min(100.0, traffic_density * 100),
        )

    def _extract_trajectory_data(
        self, detection_data: DetectionData
    ) -> list[dict[str, Any]]:
        """Extract trajectory data for speed calculation.

        Args:
            detection_data: Detection data

        Returns:
            List of trajectory data
        """
        trajectories = []

        for detection in detection_data.detections:
            if "trajectory" in detection and detection["trajectory"]:
                trajectory = {
                    "points": detection["trajectory"],
                    "track_id": detection.get("track_id"),
                    "vehicle_type": detection.get("vehicle_type"),
                }
                trajectories.append(trajectory)

        return trajectories

    def _convert_to_traffic_data_point(
        self, detection_data: DetectionData, metrics: RealtimeTrafficMetrics
    ) -> TrafficDataPoint:
        """Convert detection data to traffic data point for anomaly detection.

        Args:
            detection_data: Detection data
            metrics: Realtime traffic metrics

        Returns:
            Traffic data point
        """
        return TrafficDataPoint(
            timestamp=detection_data.timestamp,
            camera_id=detection_data.camera_id,
            vehicle_count=detection_data.vehicle_count,
            average_speed=metrics.average_speed or 0,
            traffic_density=metrics.traffic_density,
            flow_rate=metrics.flow_rate or 0,
            occupancy_rate=metrics.occupancy_rate or 0,
            queue_length=0,  # Would be calculated from detection data
        )

    def _get_time_window_from_range(
        self, start_time: datetime, end_time: datetime
    ) -> TimeWindow:
        """Determine appropriate time window from date range.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            Appropriate time window
        """
        duration = end_time - start_time

        if duration <= timedelta(hours=1):
            return TimeWindow.ONE_HOUR
        elif duration <= timedelta(days=1):
            return TimeWindow.ONE_DAY
        elif duration <= timedelta(weeks=1):
            return TimeWindow.ONE_WEEK
        else:
            return TimeWindow.ONE_MONTH

    async def _count_anomalies(
        self, camera_id: str, start_time: datetime, end_time: datetime
    ) -> int:
        """Count anomalies in time range.

        Args:
            camera_id: Camera identifier
            start_time: Start time
            end_time: End time

        Returns:
            Number of anomalies
        """
        # In production, query from repository
        # For now, return mock count
        return 0

    async def _count_violations(
        self, camera_id: str, start_time: datetime, end_time: datetime
    ) -> int:
        """Count violations in time range.

        Args:
            camera_id: Camera identifier
            start_time: Start time
            end_time: End time

        Returns:
            Number of violations
        """
        # In production, query from repository
        # For now, return mock count
        return 0

    def _calculate_health_score(
        self,
        data_quality: float,
        data_completeness: float,
        anomaly_count: int,
        violation_count: int,
    ) -> float:
        """Calculate overall camera health score.

        Args:
            data_quality: Data quality score (0-1)
            data_completeness: Data completeness score (0-1)
            anomaly_count: Number of recent anomalies
            violation_count: Number of recent violations

        Returns:
            Health score (0-1)
        """
        # Base score from data quality
        base_score = (data_quality + data_completeness) / 2

        # Penalty for anomalies (max 20% reduction)
        anomaly_penalty = min(0.2, anomaly_count * 0.05)

        # Penalty for violations (max 10% reduction)
        violation_penalty = min(0.1, violation_count * 0.02)

        health_score = max(0.0, base_score - anomaly_penalty - violation_penalty)
        return round(health_score, 3)

    async def _cache_processing_result(self, result: ProcessingResult) -> None:
        """Cache processing result for real-time dashboards.

        Args:
            result: Processing result to cache
        """
        try:
            cache_key = f"realtime_analytics:{result.camera_id}"

            # Serialize result (simplified for caching)
            cached_data = {
                "camera_id": result.camera_id,
                "timestamp": result.timestamp.isoformat(),
                "vehicle_count": result.vehicle_count,
                "violation_count": len(result.violations),
                "anomaly_count": len(result.anomalies),
                "processing_time_ms": result.processing_time_ms,
                "detection_quality": result.detection_quality,
            }

            await self.cache_service.set_json(
                cache_key, cached_data, ttl=300
            )  # 5 minutes

        except Exception as e:
            logger.warning(f"Failed to cache processing result: {e}")

    async def get_system_performance_metrics(self) -> dict[str, Any]:
        """Get overall system performance metrics.

        Returns:
            System performance metrics
        """
        try:
            # Get recent processing results from cache
            # This would aggregate across all cameras in production

            return {
                "timestamp": datetime.now(UTC),
                "total_cameras_active": 0,  # Would query from active cameras
                "average_processing_time_ms": 0.0,
                "total_vehicles_detected": 0,
                "total_violations_detected": 0,
                "total_anomalies_detected": 0,
                "system_health_score": 1.0,
                "error_rate": 0.0,
            }

        except Exception as e:
            logger.error(f"Failed to get system performance metrics: {e}")
            return {
                "timestamp": datetime.now(UTC),
                "error": str(e),
            }
