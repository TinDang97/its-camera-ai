"""Analytics processing worker for complex ML computations.

This worker handles computationally intensive analytics tasks including:
- Anomaly detection using ML models
- Traffic pattern analysis
- Predictive analytics
- Complex statistical computations
- Report generation

Features:
- GPU-accelerated ML inference when available
- Distributed processing across multiple workers
- Automatic model loading and caching
- Comprehensive error handling and retries
"""

import asyncio
import statistics
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from celery import current_task
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..containers import ApplicationContainer
from ..core.exceptions import DatabaseError
from ..models.analytics import TrafficAnomaly, TrafficMetrics
from . import celery_app, logger


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 120},
    soft_time_limit=1800,  # 30 minutes
    time_limit=3600,  # 1 hour
)
def compute_traffic_metrics(
    self,
    camera_ids: list[str],
    start_time: str,
    end_time: str,
    aggregation_level: str = "hourly",
) -> dict[str, Any]:
    """Compute complex traffic metrics for multiple cameras.

    Args:
        camera_ids: List of camera identifiers
        start_time: ISO format start time
        end_time: ISO format end time
        aggregation_level: Level of aggregation (hourly, daily)

    Returns:
        dict: Computed metrics results
    """
    return asyncio.run(
        _compute_traffic_metrics_async(
            camera_ids,
            datetime.fromisoformat(start_time),
            datetime.fromisoformat(end_time),
            aggregation_level,
        )
    )


async def _compute_traffic_metrics_async(
    camera_ids: list[str],
    start_time: datetime,
    end_time: datetime,
    aggregation_level: str,
) -> dict[str, Any]:
    """Async implementation of traffic metrics computation."""
    container = ApplicationContainer()
    container.wire(modules=[__name__])

    try:
        await container.init_resources()
        db_manager = await container.infrastructure.database()

        results = {}
        total_cameras = len(camera_ids)

        async with db_manager.get_session() as session:
            for i, camera_id in enumerate(camera_ids):
                try:
                    # Get base metrics for the camera
                    base_metrics = await _get_traffic_metrics(
                        session, camera_id, start_time, end_time, aggregation_level
                    )

                    if not base_metrics:
                        results[camera_id] = {"status": "no_data", "metrics": {}}
                        continue

                    # Compute advanced statistics
                    advanced_stats = await _compute_advanced_statistics(base_metrics)

                    # Compute traffic patterns
                    patterns = await _analyze_traffic_patterns(base_metrics)

                    # Compute correlation metrics
                    correlations = await _compute_correlations(base_metrics)

                    # Quality assessment
                    quality_metrics = await _assess_data_quality(base_metrics)

                    results[camera_id] = {
                        "status": "success",
                        "metrics": {
                            "basic_stats": _extract_basic_stats(base_metrics),
                            "advanced_stats": advanced_stats,
                            "patterns": patterns,
                            "correlations": correlations,
                            "quality": quality_metrics,
                        },
                        "sample_count": len(base_metrics),
                        "time_coverage": {
                            "start": start_time.isoformat(),
                            "end": end_time.isoformat(),
                            "duration_hours": (end_time - start_time).total_seconds()
                            / 3600,
                        },
                    }

                    # Update progress
                    if current_task:
                        current_task.update_state(
                            state="PROGRESS",
                            meta={
                                "processed": i + 1,
                                "total": total_cameras,
                                "current_camera": camera_id,
                            },
                        )

                except Exception as e:
                    logger.error(
                        f"Failed to compute metrics for camera {camera_id}: {e}"
                    )
                    results[camera_id] = {"status": "error", "error": str(e)}

        logger.info(f"Computed traffic metrics for {len(camera_ids)} cameras")

        return {
            "status": "completed",
            "total_cameras": total_cameras,
            "successful": len(
                [r for r in results.values() if r["status"] == "success"]
            ),
            "results": results,
            "computation_time": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Traffic metrics computation failed: {e}")
        raise DatabaseError(f"Metrics computation failed: {e}") from e
    finally:
        await container.shutdown_resources()


async def _get_traffic_metrics(
    session,
    camera_id: str,
    start_time: datetime,
    end_time: datetime,
    aggregation_level: str,
) -> list[TrafficMetrics]:
    """Fetch traffic metrics from database."""
    from sqlalchemy import select

    query = (
        select(TrafficMetrics)
        .where(TrafficMetrics.camera_id == camera_id)
        .where(TrafficMetrics.timestamp >= start_time)
        .where(TrafficMetrics.timestamp <= end_time)
        .where(TrafficMetrics.aggregation_period == aggregation_level)
        .order_by(TrafficMetrics.timestamp)
    )

    result = await session.execute(query)
    return list(result.scalars().all())


def _extract_basic_stats(metrics: list[TrafficMetrics]) -> dict[str, Any]:
    """Extract basic statistical measures."""
    if not metrics:
        return {}

    vehicle_counts = [m.total_vehicles for m in metrics]
    speeds = [m.average_speed for m in metrics if m.average_speed is not None]
    densities = [m.traffic_density for m in metrics if m.traffic_density is not None]

    return {
        "vehicle_count": {
            "mean": statistics.mean(vehicle_counts) if vehicle_counts else 0,
            "median": statistics.median(vehicle_counts) if vehicle_counts else 0,
            "std": statistics.stdev(vehicle_counts) if len(vehicle_counts) > 1 else 0,
            "min": min(vehicle_counts) if vehicle_counts else 0,
            "max": max(vehicle_counts) if vehicle_counts else 0,
        },
        "speed": (
            {
                "mean": statistics.mean(speeds) if speeds else 0,
                "median": statistics.median(speeds) if speeds else 0,
                "std": statistics.stdev(speeds) if len(speeds) > 1 else 0,
                "min": min(speeds) if speeds else 0,
                "max": max(speeds) if speeds else 0,
            }
            if speeds
            else {}
        ),
        "density": (
            {
                "mean": statistics.mean(densities) if densities else 0,
                "median": statistics.median(densities) if densities else 0,
                "std": statistics.stdev(densities) if len(densities) > 1 else 0,
            }
            if densities
            else {}
        ),
    }


async def _compute_advanced_statistics(metrics: list[TrafficMetrics]) -> dict[str, Any]:
    """Compute advanced statistical measures."""
    if not metrics:
        return {}

    vehicle_counts = np.array([m.total_vehicles for m in metrics])
    speeds = np.array([m.average_speed for m in metrics if m.average_speed is not None])

    advanced_stats = {}

    # Percentiles for vehicle counts
    if len(vehicle_counts) > 0:
        advanced_stats["vehicle_count_percentiles"] = {
            "p25": float(np.percentile(vehicle_counts, 25)),
            "p75": float(np.percentile(vehicle_counts, 75)),
            "p90": float(np.percentile(vehicle_counts, 90)),
            "p95": float(np.percentile(vehicle_counts, 95)),
            "p99": float(np.percentile(vehicle_counts, 99)),
        }

        # Variability measures
        advanced_stats["vehicle_count_variability"] = {
            "coefficient_of_variation": (
                float(np.std(vehicle_counts) / np.mean(vehicle_counts))
                if np.mean(vehicle_counts) > 0
                else 0
            ),
            "interquartile_range": float(
                np.percentile(vehicle_counts, 75) - np.percentile(vehicle_counts, 25)
            ),
        }

    # Speed statistics
    if len(speeds) > 0:
        advanced_stats["speed_percentiles"] = {
            "p25": float(np.percentile(speeds, 25)),
            "p75": float(np.percentile(speeds, 75)),
            "p90": float(np.percentile(speeds, 90)),
            "p95": float(np.percentile(speeds, 95)),
        }

    # Time-based patterns
    hourly_patterns = {}
    daily_patterns = {}

    for metric in metrics:
        hour = metric.timestamp.hour
        day = metric.timestamp.strftime("%A")

        if hour not in hourly_patterns:
            hourly_patterns[hour] = []
        hourly_patterns[hour].append(metric.total_vehicles)

        if day not in daily_patterns:
            daily_patterns[day] = []
        daily_patterns[day].append(metric.total_vehicles)

    # Average by hour and day
    advanced_stats["hourly_averages"] = {
        hour: statistics.mean(counts) for hour, counts in hourly_patterns.items()
    }
    advanced_stats["daily_averages"] = {
        day: statistics.mean(counts) for day, counts in daily_patterns.items()
    }

    return advanced_stats


async def _analyze_traffic_patterns(metrics: list[TrafficMetrics]) -> dict[str, Any]:
    """Analyze traffic patterns and trends."""
    if not metrics:
        return {}

    patterns = {}

    # Peak detection
    vehicle_counts = [m.total_vehicles for m in metrics]
    timestamps = [m.timestamp for m in metrics]

    if len(vehicle_counts) > 3:
        # Find local maxima (simple peak detection)
        peaks = []
        for i in range(1, len(vehicle_counts) - 1):
            if (
                vehicle_counts[i] > vehicle_counts[i - 1]
                and vehicle_counts[i] > vehicle_counts[i + 1]
                and vehicle_counts[i] > statistics.mean(vehicle_counts)
            ):
                peaks.append(
                    {
                        "timestamp": timestamps[i].isoformat(),
                        "value": vehicle_counts[i],
                        "hour": timestamps[i].hour,
                    }
                )

        patterns["peaks"] = peaks

        # Rush hour detection
        rush_hours = []
        hourly_avg = {}

        for metric in metrics:
            hour = metric.timestamp.hour
            if hour not in hourly_avg:
                hourly_avg[hour] = []
            hourly_avg[hour].append(metric.total_vehicles)

        overall_avg = statistics.mean(vehicle_counts)

        for hour, counts in hourly_avg.items():
            avg_count = statistics.mean(counts)
            if avg_count > overall_avg * 1.5:  # 50% above average
                rush_hours.append(
                    {
                        "hour": hour,
                        "average_vehicles": avg_count,
                        "intensity": avg_count / overall_avg,
                    }
                )

        patterns["rush_hours"] = sorted(
            rush_hours, key=lambda x: x["intensity"], reverse=True
        )

    # Congestion analysis
    congestion_levels = [m.congestion_level for m in metrics]
    congestion_distribution = {}

    for level in congestion_levels:
        congestion_distribution[level] = congestion_distribution.get(level, 0) + 1

    patterns["congestion_distribution"] = {
        level: count / len(congestion_levels)
        for level, count in congestion_distribution.items()
    }

    return patterns


async def _compute_correlations(metrics: list[TrafficMetrics]) -> dict[str, Any]:
    """Compute correlations between different traffic variables."""
    if len(metrics) < 3:
        return {}

    # Extract variables
    vehicle_counts = np.array([m.total_vehicles for m in metrics])
    speeds = np.array([m.average_speed for m in metrics if m.average_speed is not None])
    densities = np.array(
        [m.traffic_density for m in metrics if m.traffic_density is not None]
    )
    flow_rates = np.array([m.flow_rate for m in metrics if m.flow_rate is not None])

    correlations = {}

    # Speed vs Vehicle Count correlation
    if len(speeds) == len(vehicle_counts) and len(speeds) > 2:
        speed_count_corr = float(
            np.corrcoef(speeds[: len(vehicle_counts)], vehicle_counts)[0, 1]
        )
        correlations["speed_vs_count"] = speed_count_corr

    # Density vs Flow Rate correlation
    if len(densities) > 2 and len(flow_rates) > 2:
        min_len = min(len(densities), len(flow_rates))
        if min_len > 2:
            density_flow_corr = float(
                np.corrcoef(densities[:min_len], flow_rates[:min_len])[0, 1]
            )
            correlations["density_vs_flow"] = density_flow_corr

    # Time-based correlations
    hours = np.array([m.timestamp.hour for m in metrics])
    if len(hours) > 2:
        hour_count_corr = float(np.corrcoef(hours, vehicle_counts)[0, 1])
        correlations["time_vs_count"] = hour_count_corr

    return correlations


async def _assess_data_quality(metrics: list[TrafficMetrics]) -> dict[str, Any]:
    """Assess the quality of traffic data."""
    if not metrics:
        return {"overall_quality": "poor", "completeness": 0.0}

    quality_metrics = {}

    # Data completeness
    total_expected = len(metrics)
    non_null_counts = sum(1 for m in metrics if m.total_vehicles is not None)
    completeness = non_null_counts / total_expected if total_expected > 0 else 0

    quality_metrics["completeness"] = completeness

    # Consistency checks
    consistency_score = 1.0

    # Check for unrealistic values
    unrealistic_count = 0
    for metric in metrics:
        if (
            metric.total_vehicles and metric.total_vehicles > 200
        ):  # Unrealistically high
            unrealistic_count += 1
        if metric.average_speed and metric.average_speed > 150:  # Too fast
            unrealistic_count += 1
        if metric.traffic_density and metric.traffic_density < 0:  # Negative density
            unrealistic_count += 1

    if total_expected > 0:
        consistency_score = 1.0 - (unrealistic_count / total_expected)

    quality_metrics["consistency"] = consistency_score

    # Temporal continuity (check for gaps)
    if len(metrics) > 1:
        time_gaps = []
        for i in range(1, len(metrics)):
            gap = (metrics[i].timestamp - metrics[i - 1].timestamp).total_seconds()
            time_gaps.append(gap)

        expected_interval = 3600  # 1 hour for hourly data
        irregular_gaps = sum(
            1 for gap in time_gaps if abs(gap - expected_interval) > 300
        )  # 5 min tolerance
        continuity_score = 1.0 - (irregular_gaps / len(time_gaps)) if time_gaps else 1.0

        quality_metrics["temporal_continuity"] = continuity_score
    else:
        quality_metrics["temporal_continuity"] = 1.0

    # Overall quality score
    overall_quality = (
        completeness + consistency_score + quality_metrics["temporal_continuity"]
    ) / 3

    if overall_quality >= 0.9:
        quality_label = "excellent"
    elif overall_quality >= 0.7:
        quality_label = "good"
    elif overall_quality >= 0.5:
        quality_label = "fair"
    else:
        quality_label = "poor"

    quality_metrics["overall_quality"] = quality_label
    quality_metrics["overall_score"] = overall_quality

    return quality_metrics


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 300},
    soft_time_limit=900,  # 15 minutes
    time_limit=1800,  # 30 minutes
)
def detect_anomalies(
    self, camera_ids: list[str] | None = None, lookback_hours: int = 24
) -> dict[str, Any]:
    """Detect traffic anomalies using ML models.

    Args:
        camera_ids: List of camera IDs to analyze (None for all)
        lookback_hours: Hours of historical data to analyze

    Returns:
        dict: Anomaly detection results
    """
    return asyncio.run(_detect_anomalies_async(camera_ids, lookback_hours))


async def _detect_anomalies_async(
    camera_ids: list[str] | None, lookback_hours: int
) -> dict[str, Any]:
    """Async implementation of anomaly detection."""
    container = ApplicationContainer()
    container.wire(modules=[__name__])

    try:
        await container.init_resources()
        db_manager = await container.infrastructure.database()

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=lookback_hours)

        anomalies_detected = []
        cameras_processed = 0

        async with db_manager.get_session() as session:
            # Get cameras to process
            if camera_ids is None:
                from sqlalchemy import distinct, select

                camera_query = select(distinct(TrafficMetrics.camera_id))
                result = await session.execute(camera_query)
                camera_ids = [row[0] for row in result.fetchall()]

            for camera_id in camera_ids:
                try:
                    # Get recent metrics for this camera
                    metrics = await _get_traffic_metrics(
                        session, camera_id, start_time, end_time, "1hour"
                    )

                    if len(metrics) < 5:  # Need minimum data for anomaly detection
                        continue

                    # Detect anomalies using isolation forest
                    camera_anomalies = await _detect_camera_anomalies(
                        metrics, camera_id
                    )

                    # Store detected anomalies
                    if camera_anomalies:
                        stored_anomalies = await _store_anomalies(
                            session, camera_anomalies
                        )
                        anomalies_detected.extend(stored_anomalies)

                    cameras_processed += 1

                    # Update progress
                    if current_task:
                        current_task.update_state(
                            state="PROGRESS",
                            meta={
                                "processed": cameras_processed,
                                "total": len(camera_ids),
                                "current_camera": camera_id,
                                "anomalies_found": len(anomalies_detected),
                            },
                        )

                except Exception as e:
                    logger.error(
                        f"Anomaly detection failed for camera {camera_id}: {e}"
                    )
                    continue

        logger.info(
            f"Anomaly detection completed: {len(anomalies_detected)} anomalies found"
        )

        return {
            "status": "completed",
            "cameras_processed": cameras_processed,
            "total_cameras": len(camera_ids),
            "anomalies_detected": len(anomalies_detected),
            "anomalies": [
                {
                    "id": str(anomaly.id),
                    "camera_id": anomaly.camera_id,
                    "detection_time": anomaly.detection_time.isoformat(),
                    "anomaly_type": anomaly.anomaly_type,
                    "severity": anomaly.severity,
                    "score": anomaly.anomaly_score,
                }
                for anomaly in anomalies_detected[:10]  # Return top 10
            ],
            "detection_time": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise DatabaseError(f"Anomaly detection failed: {e}") from e
    finally:
        await container.shutdown_resources()


async def _detect_camera_anomalies(
    metrics: list[TrafficMetrics], camera_id: str
) -> list[TrafficAnomaly]:
    """Detect anomalies for a specific camera using ML."""
    if len(metrics) < 5:
        return []

    # Prepare feature matrix
    features = []
    timestamps = []

    for metric in metrics:
        feature_vector = [
            metric.total_vehicles or 0,
            metric.average_speed or 0,
            metric.traffic_density or 0,
            metric.flow_rate or 0,
            metric.occupancy_rate or 0,
            metric.timestamp.hour,  # Time feature
            metric.timestamp.weekday(),  # Day of week feature
        ]
        features.append(feature_vector)
        timestamps.append(metric.timestamp)

    features_array = np.array(features)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_array)

    # Detect anomalies using Isolation Forest
    isolation_forest = IsolationForest(
        contamination=0.1, random_state=42, n_estimators=100  # Expect 10% anomalies
    )

    anomaly_labels = isolation_forest.fit_predict(features_scaled)
    anomaly_scores = isolation_forest.decision_function(features_scaled)

    # Create anomaly records
    anomalies = []

    for i, (label, score, timestamp, metric) in enumerate(
        zip(anomaly_labels, anomaly_scores, timestamps, metrics, strict=False)
    ):
        if label == -1:  # Anomaly detected
            # Determine anomaly type and severity
            anomaly_type, severity = _classify_anomaly(metric, features_array[i])

            # Convert score to positive confidence
            confidence = min(1.0, abs(score) * 2)

            # Skip low-confidence anomalies
            if confidence < 0.3:
                continue

            anomaly = TrafficAnomaly(
                camera_id=camera_id,
                detection_time=timestamp,
                anomaly_type=anomaly_type,
                severity=severity,
                anomaly_score=confidence,
                confidence=confidence,
                detection_method="isolation_forest",
                probable_cause=_determine_probable_cause(metric, features_array[i]),
                model_name="IsolationForest",
                model_version="1.0",
                model_confidence=confidence,
                detailed_analysis={
                    "features": features_array[i].tolist(),
                    "anomaly_score": float(score),
                    "metric_values": {
                        "total_vehicles": metric.total_vehicles,
                        "average_speed": metric.average_speed,
                        "traffic_density": metric.traffic_density,
                        "flow_rate": metric.flow_rate,
                        "occupancy_rate": metric.occupancy_rate,
                    },
                },
                status="active",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )

            anomalies.append(anomaly)

    return anomalies


def _classify_anomaly(metric: TrafficMetrics, features: np.ndarray) -> tuple[str, str]:
    """Classify the type and severity of an anomaly."""
    vehicle_count = features[0]
    avg_speed = features[1]
    density = features[2]

    # Classify anomaly type
    if vehicle_count > 50:
        anomaly_type = "high_traffic_volume"
    elif vehicle_count == 0:
        anomaly_type = "no_traffic"
    elif avg_speed > 80:
        anomaly_type = "excessive_speed"
    elif avg_speed < 10 and vehicle_count > 5:
        anomaly_type = "traffic_jam"
    elif density > 10:
        anomaly_type = "high_density"
    else:
        anomaly_type = "pattern_deviation"

    # Determine severity
    if vehicle_count > 100 or avg_speed > 100 or density > 20:
        severity = "critical"
    elif vehicle_count > 50 or avg_speed > 80 or density > 15:
        severity = "high"
    elif vehicle_count > 30 or avg_speed > 60 or density > 10:
        severity = "medium"
    else:
        severity = "low"

    return anomaly_type, severity


def _determine_probable_cause(metric: TrafficMetrics, features: np.ndarray) -> str:
    """Determine the probable cause of an anomaly."""
    vehicle_count = features[0]
    avg_speed = features[1]
    hour = features[5]

    # Rule-based cause determination
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        if vehicle_count > 30:
            return "rush_hour_congestion"
        elif avg_speed > 80:
            return "rush_hour_speeding"

    if hour >= 22 or hour <= 6:
        if vehicle_count > 20:
            return "unusual_night_traffic"
        elif avg_speed > 80:
            return "night_speeding"

    if vehicle_count == 0:
        return "camera_malfunction_or_road_closure"

    if avg_speed > 100:
        return "emergency_vehicle_or_reckless_driving"

    if vehicle_count > 100:
        return "special_event_or_incident"

    return "unknown_pattern_deviation"


async def _store_anomalies(
    session, anomalies: list[TrafficAnomaly]
) -> list[TrafficAnomaly]:
    """Store detected anomalies in the database."""
    if not anomalies:
        return []

    session.add_all(anomalies)
    await session.commit()

    # Refresh to get IDs
    for anomaly in anomalies:
        await session.refresh(anomaly)

    return anomalies


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 1, "countdown": 600},
    soft_time_limit=3600,  # 1 hour
    time_limit=7200,  # 2 hours
)
def generate_analytics_report(self, report_params: dict[str, Any]) -> dict[str, Any]:
    """Generate comprehensive analytics report.

    Args:
        report_params: Report generation parameters

    Returns:
        dict: Report generation results
    """
    return asyncio.run(_generate_analytics_report_async(report_params))


async def _generate_analytics_report_async(
    report_params: dict[str, Any],
) -> dict[str, Any]:
    """Async implementation of report generation."""
    container = ApplicationContainer()
    container.wire(modules=[__name__])

    try:
        await container.init_resources()

        report_id = report_params.get("report_id")
        camera_ids = report_params.get("camera_ids", [])
        start_time = datetime.fromisoformat(report_params["start_time"])
        end_time = datetime.fromisoformat(report_params["end_time"])
        report_type = report_params.get("report_type", "comprehensive")

        # Update task status
        if current_task:
            current_task.update_state(
                state="PROGRESS", meta={"stage": "data_collection", "progress": 10}
            )

        # Collect data for all cameras
        analytics_data = {}
        for camera_id in camera_ids:
            metrics_result = await _compute_traffic_metrics_async(
                [camera_id], start_time, end_time, "hourly"
            )
            analytics_data[camera_id] = metrics_result["results"][camera_id]

        # Update progress
        if current_task:
            current_task.update_state(
                state="PROGRESS", meta={"stage": "analysis", "progress": 50}
            )

        # Generate comprehensive analysis
        report_data = {
            "report_id": report_id,
            "report_type": report_type,
            "generation_time": datetime.now(UTC).isoformat(),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_days": (end_time - start_time).days,
            },
            "cameras": camera_ids,
            "analytics": analytics_data,
            "summary": _generate_report_summary(analytics_data),
        }

        # Update final progress
        if current_task:
            current_task.update_state(
                state="PROGRESS", meta={"stage": "finalization", "progress": 90}
            )

        logger.info(f"Analytics report generated: {report_id}")

        return {
            "status": "completed",
            "report_id": report_id,
            "report_data": report_data,
            "file_size": len(str(report_data)),  # Approximate size
        }

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise DatabaseError(f"Report generation failed: {e}") from e
    finally:
        await container.shutdown_resources()


def _generate_report_summary(analytics_data: dict[str, Any]) -> dict[str, Any]:
    """Generate summary statistics for the report."""
    total_cameras = len(analytics_data)
    successful_cameras = len(
        [data for data in analytics_data.values() if data.get("status") == "success"]
    )

    # Aggregate metrics across all cameras
    total_vehicle_count = 0
    total_samples = 0

    for camera_data in analytics_data.values():
        if camera_data.get("status") == "success":
            metrics = camera_data.get("metrics", {})
            basic_stats = metrics.get("basic_stats", {})
            vehicle_stats = basic_stats.get("vehicle_count", {})

            if "mean" in vehicle_stats:
                sample_count = camera_data.get("sample_count", 0)
                total_vehicle_count += vehicle_stats["mean"] * sample_count
                total_samples += sample_count

    avg_vehicle_count = total_vehicle_count / total_samples if total_samples > 0 else 0

    return {
        "total_cameras": total_cameras,
        "successful_analysis": successful_cameras,
        "data_quality": successful_cameras / total_cameras if total_cameras > 0 else 0,
        "average_vehicle_count": avg_vehicle_count,
        "total_data_points": total_samples,
    }
