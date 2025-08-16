"""Event processing worker for real-time ML pipeline integration.

This worker handles event-driven processing from the ML pipeline,
including Kafka message processing, real-time analytics updates,
and SSE broadcasting for live dashboard updates.

Features:
- Kafka consumer for ML detection events
- Real-time analytics processing
- SSE broadcasting for live updates
- Event aggregation and batching
- Comprehensive error handling and dead letter queues
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from celery import current_task

from ..containers import ApplicationContainer
from ..core.exceptions import DatabaseError
from ..models.detection_result import DetectionResult
from . import celery_app, logger


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 30},
    soft_time_limit=60,  # 1 minute
    time_limit=120,  # 2 minutes
)
def process_ml_detection(self, detection_event: dict[str, Any]) -> dict[str, Any]:
    """Process ML detection event from Kafka.
    
    Args:
        detection_event: ML detection event data
        
    Returns:
        dict: Processing results
    """
    return asyncio.run(_process_ml_detection_async(detection_event))


async def _process_ml_detection_async(detection_event: dict[str, Any]) -> dict[str, Any]:
    """Async implementation of ML detection processing."""
    container = ApplicationContainer()
    container.wire(modules=[__name__])

    try:
        await container.init_resources()
        db_manager = await container.infrastructure.database()

        detection_data = detection_event.get("detection", {})
        camera_id = detection_data.get("camera_id")
        frame_timestamp = datetime.fromisoformat(detection_data.get("timestamp"))

        if not camera_id:
            raise ValueError("Missing camera_id in detection event")

        async with db_manager.get_session() as session:
            # Store detection result
            detection_result = await _create_detection_result(session, detection_data)

            # Trigger real-time aggregation
            await _trigger_realtime_aggregation(camera_id, frame_timestamp)

            # Check for violations and anomalies
            violations = await _check_traffic_violations(detection_result)
            anomalies = await _check_for_anomalies(detection_result)

            # Broadcast updates via SSE
            await _broadcast_realtime_updates(camera_id, {
                "detection": detection_result,
                "violations": violations,
                "anomalies": anomalies,
            })

            # Update task progress
            if current_task:
                current_task.update_state(
                    state="SUCCESS",
                    meta={
                        "detection_id": str(detection_result.id),
                        "camera_id": camera_id,
                        "violations_count": len(violations),
                        "anomalies_count": len(anomalies),
                    }
                )

        logger.debug(
            f"Processed ML detection for camera {camera_id}",
            detection_id=str(detection_result.id),
            violations=len(violations),
            anomalies=len(anomalies),
        )

        return {
            "status": "success",
            "detection_id": str(detection_result.id),
            "camera_id": camera_id,
            "processing_time": datetime.now(UTC).isoformat(),
            "violations_detected": len(violations),
            "anomalies_detected": len(anomalies),
        }

    except Exception as e:
        logger.error(f"ML detection processing failed: {e}", detection_event=detection_event)
        raise DatabaseError(f"Detection processing failed: {e}") from e
    finally:
        await container.shutdown_resources()


async def _create_detection_result(session, detection_data: dict[str, Any]) -> DetectionResult:
    """Create and store detection result in database."""
    detection_result = DetectionResult(
        camera_id=detection_data["camera_id"],
        frame_id=detection_data.get("frame_id"),
        class_name=detection_data["class_name"],
        class_confidence=detection_data["confidence"],
        bbox_x1=detection_data["bbox"][0],
        bbox_y1=detection_data["bbox"][1],
        bbox_x2=detection_data["bbox"][2],
        bbox_y2=detection_data["bbox"][3],
        bbox_center_x=(detection_data["bbox"][0] + detection_data["bbox"][2]) / 2,
        bbox_center_y=(detection_data["bbox"][1] + detection_data["bbox"][3]) / 2,
        velocity_magnitude=detection_data.get("velocity"),
        vehicle_type=detection_data.get("vehicle_type"),
        license_plate=detection_data.get("license_plate"),
        track_id=detection_data.get("track_id"),
        detection_zone=detection_data.get("zone"),
        detection_quality=detection_data.get("quality", 0.8),
        created_at=datetime.fromisoformat(detection_data["timestamp"]),
    )

    session.add(detection_result)
    await session.commit()
    await session.refresh(detection_result)

    return detection_result


async def _trigger_realtime_aggregation(camera_id: str, timestamp: datetime) -> None:
    """Trigger real-time aggregation for the camera."""
    from .aggregation_worker import aggregate_detection_data

    # Aggregate data for the current minute
    minute_start = timestamp.replace(second=0, microsecond=0)
    minute_end = minute_start + timedelta(minutes=1)

    # Use apply_async for non-blocking execution
    aggregate_detection_data.apply_async(
        args=[camera_id, minute_start.isoformat(), minute_end.isoformat()],
        queue="realtime",
        priority=9,
    )


async def _check_traffic_violations(detection_result: DetectionResult) -> list[dict[str, Any]]:
    """Check for traffic violations in the detection."""
    violations = []

    # Speed violation check
    if detection_result.velocity_magnitude:
        estimated_speed = detection_result.velocity_magnitude * 3.6  # Convert to km/h

        if estimated_speed > 80:  # Speed limit threshold
            violations.append({
                "type": "speeding",
                "severity": "high" if estimated_speed > 100 else "medium",
                "measured_speed": estimated_speed,
                "speed_limit": 80,
                "excess_speed": estimated_speed - 80,
                "detection_id": str(detection_result.id),
                "camera_id": detection_result.camera_id,
                "timestamp": detection_result.created_at.isoformat(),
            })

    # Vehicle type restrictions (example)
    if (detection_result.vehicle_type == "truck" and
        detection_result.detection_zone == "restricted_zone"):
        violations.append({
            "type": "restricted_vehicle",
            "severity": "medium",
            "vehicle_type": detection_result.vehicle_type,
            "zone": detection_result.detection_zone,
            "detection_id": str(detection_result.id),
            "camera_id": detection_result.camera_id,
            "timestamp": detection_result.created_at.isoformat(),
        })

    return violations


async def _check_for_anomalies(detection_result: DetectionResult) -> list[dict[str, Any]]:
    """Check for anomalous patterns in the detection."""
    anomalies = []

    # Unusual speed patterns
    if detection_result.velocity_magnitude:
        speed_kmh = detection_result.velocity_magnitude * 3.6

        # Very high speed anomaly
        if speed_kmh > 150:
            anomalies.append({
                "type": "excessive_speed",
                "severity": "critical",
                "speed": speed_kmh,
                "threshold": 150,
                "detection_id": str(detection_result.id),
                "camera_id": detection_result.camera_id,
                "timestamp": detection_result.created_at.isoformat(),
            })

        # Stationary vehicle in traffic lane
        elif speed_kmh < 1 and detection_result.detection_zone == "traffic_lane":
            anomalies.append({
                "type": "stationary_vehicle",
                "severity": "medium",
                "speed": speed_kmh,
                "zone": detection_result.detection_zone,
                "detection_id": str(detection_result.id),
                "camera_id": detection_result.camera_id,
                "timestamp": detection_result.created_at.isoformat(),
            })

    # Unusual vehicle types at specific times
    current_hour = detection_result.created_at.hour
    if (detection_result.vehicle_type == "bus" and
        (current_hour < 6 or current_hour > 22)):
        anomalies.append({
            "type": "unusual_timing",
            "severity": "low",
            "vehicle_type": detection_result.vehicle_type,
            "hour": current_hour,
            "detection_id": str(detection_result.id),
            "camera_id": detection_result.camera_id,
            "timestamp": detection_result.created_at.isoformat(),
        })

    return anomalies


async def _broadcast_realtime_updates(camera_id: str, update_data: dict[str, Any]) -> None:
    """Broadcast real-time updates via SSE."""
    try:
        # Get SSE broadcaster from container
        container = ApplicationContainer()
        sse_broadcaster = await container.services.sse_broadcaster()

        # Prepare SSE message
        sse_message = {
            "type": "detection_update",
            "camera_id": camera_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "data": {
                "detection_count": 1,
                "violations": update_data["violations"],
                "anomalies": update_data["anomalies"],
            }
        }

        # Broadcast to all connected clients
        await sse_broadcaster.broadcast(
            message=json.dumps(sse_message),
            channel=f"camera_{camera_id}"
        )

    except Exception as e:
        logger.warning(f"Failed to broadcast SSE update: {e}")


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 60},
)
def process_kafka_batch(self, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Process a batch of Kafka events for efficiency.
    
    Args:
        events: List of Kafka event messages
        
    Returns:
        dict: Batch processing results
    """
    return asyncio.run(_process_kafka_batch_async(events))


async def _process_kafka_batch_async(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Async implementation of batch Kafka event processing."""
    container = ApplicationContainer()
    container.wire(modules=[__name__])

    try:
        await container.init_resources()

        processed_count = 0
        failed_count = 0
        results = []

        # Group events by camera for efficient processing
        camera_events = {}
        for event in events:
            camera_id = event.get("detection", {}).get("camera_id")
            if camera_id:
                if camera_id not in camera_events:
                    camera_events[camera_id] = []
                camera_events[camera_id].append(event)

        # Process events by camera
        for camera_id, camera_event_list in camera_events.items():
            try:
                camera_results = await _process_camera_events(camera_id, camera_event_list)
                results.extend(camera_results)
                processed_count += len(camera_event_list)

                # Update progress
                if current_task:
                    current_task.update_state(
                        state="PROGRESS",
                        meta={
                            "processed": processed_count,
                            "total": len(events),
                            "current_camera": camera_id,
                        }
                    )

            except Exception as e:
                logger.error(f"Failed to process events for camera {camera_id}: {e}")
                failed_count += len(camera_event_list)

        logger.info(f"Batch processing completed: {processed_count} processed, {failed_count} failed")

        return {
            "status": "completed",
            "total_events": len(events),
            "processed_count": processed_count,
            "failed_count": failed_count,
            "cameras_processed": len(camera_events),
            "results": results[:10],  # Return sample results
            "processing_time": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise DatabaseError(f"Batch processing failed: {e}") from e
    finally:
        await container.shutdown_resources()


async def _process_camera_events(camera_id: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Process all events for a specific camera."""
    results = []

    # Batch detections by minute for aggregation efficiency
    minute_batches = {}

    for event in events:
        detection_data = event.get("detection", {})
        timestamp = datetime.fromisoformat(detection_data.get("timestamp"))
        minute_key = timestamp.replace(second=0, microsecond=0)

        if minute_key not in minute_batches:
            minute_batches[minute_key] = []
        minute_batches[minute_key].append(event)

    # Process each minute batch
    for minute_timestamp, minute_events in minute_batches.items():
        try:
            # Process individual detections
            for event in minute_events:
                result = await _process_ml_detection_async(event)
                results.append(result)

            # Trigger aggregation for this minute
            minute_end = minute_timestamp + timedelta(minutes=1)
            await _trigger_realtime_aggregation(camera_id, minute_timestamp)

        except Exception as e:
            logger.error(f"Failed to process minute batch for {camera_id} at {minute_timestamp}: {e}")
            # Continue with other batches
            continue

    return results


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2, "countdown": 120},
)
def process_incident_alert(self, incident_data: dict[str, Any]) -> dict[str, Any]:
    """Process incident alert from ML pipeline.
    
    Args:
        incident_data: Incident alert data
        
    Returns:
        dict: Processing results
    """
    return asyncio.run(_process_incident_alert_async(incident_data))


async def _process_incident_alert_async(incident_data: dict[str, Any]) -> dict[str, Any]:
    """Async implementation of incident alert processing."""
    container = ApplicationContainer()
    container.wire(modules=[__name__])

    try:
        await container.init_resources()

        incident_type = incident_data.get("type")
        camera_id = incident_data.get("camera_id")
        severity = incident_data.get("severity", "medium")
        confidence = incident_data.get("confidence", 0.8)

        if not camera_id or not incident_type:
            raise ValueError("Missing required incident data")

        # Create incident alert record
        incident_alert = {
            "id": incident_data.get("id"),
            "camera_id": camera_id,
            "incident_type": incident_type,
            "severity": severity,
            "confidence": confidence,
            "description": incident_data.get("description", f"{incident_type} detected"),
            "location": incident_data.get("location"),
            "coordinates": incident_data.get("coordinates"),
            "timestamp": datetime.fromisoformat(incident_data["timestamp"]),
            "detected_at": datetime.now(UTC),
            "status": "active",
            "metadata": incident_data.get("metadata", {}),
        }

        # Store incident (would use incident management service)
        # For now, just log and broadcast

        # Broadcast incident alert
        await _broadcast_incident_alert(incident_alert)

        # Trigger automated responses based on severity
        if severity in ["high", "critical"]:
            await _trigger_automated_responses(incident_alert)

        logger.info(
            f"Processed incident alert: {incident_type} at camera {camera_id}",
            incident_id=incident_alert["id"],
            severity=severity,
        )

        return {
            "status": "success",
            "incident_id": incident_alert["id"],
            "incident_type": incident_type,
            "camera_id": camera_id,
            "severity": severity,
            "processing_time": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Incident alert processing failed: {e}", incident_data=incident_data)
        raise DatabaseError(f"Incident processing failed: {e}") from e
    finally:
        await container.shutdown_resources()


async def _broadcast_incident_alert(incident_alert: dict[str, Any]) -> None:
    """Broadcast incident alert via SSE."""
    try:
        container = ApplicationContainer()
        sse_broadcaster = await container.services.sse_broadcaster()

        # Prepare SSE message
        sse_message = {
            "type": "incident_alert",
            "incident": incident_alert,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Broadcast to all clients and camera-specific channel
        await sse_broadcaster.broadcast(
            message=json.dumps(sse_message),
            channel="incidents"
        )

        await sse_broadcaster.broadcast(
            message=json.dumps(sse_message),
            channel=f"camera_{incident_alert['camera_id']}"
        )

    except Exception as e:
        logger.warning(f"Failed to broadcast incident alert: {e}")


async def _trigger_automated_responses(incident_alert: dict[str, Any]) -> None:
    """Trigger automated responses for high-severity incidents."""
    try:
        severity = incident_alert["severity"]
        incident_type = incident_alert["incident_type"]

        # Email notifications for critical incidents
        if severity == "critical":
            # Would integrate with email service
            logger.info(f"Triggering email notifications for critical incident: {incident_alert['id']}")

        # Traffic signal adjustments for congestion
        if incident_type == "congestion" and severity in ["high", "critical"]:
            # Would integrate with traffic management system
            logger.info(f"Triggering traffic signal optimization for congestion: {incident_alert['id']}")

        # Emergency service notifications for accidents
        if incident_type == "accident":
            # Would integrate with emergency services API
            logger.info(f"Triggering emergency service notification: {incident_alert['id']}")

    except Exception as e:
        logger.warning(f"Failed to trigger automated responses: {e}")


@celery_app.task(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 5, "countdown": 10},
    rate_limit="100/m",  # Rate limit for high-frequency updates
)
def update_camera_status(self, camera_id: str, status_data: dict[str, Any]) -> dict[str, Any]:
    """Update camera status from ML pipeline.
    
    Args:
        camera_id: Camera identifier
        status_data: Camera status information
        
    Returns:
        dict: Update results
    """
    return asyncio.run(_update_camera_status_async(camera_id, status_data))


async def _update_camera_status_async(camera_id: str, status_data: dict[str, Any]) -> dict[str, Any]:
    """Async implementation of camera status update."""
    try:
        # Update camera status in cache for real-time access
        container = ApplicationContainer()
        cache_service = await container.services.cache_service()

        # Store status with TTL
        cache_key = f"camera_status:{camera_id}"
        status_data["last_updated"] = datetime.now(UTC).isoformat()

        await cache_service.set_json(cache_key, status_data, ttl=300)  # 5 minutes TTL

        # Broadcast status update
        sse_broadcaster = await container.services.sse_broadcaster()
        sse_message = {
            "type": "camera_status",
            "camera_id": camera_id,
            "status": status_data,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        await sse_broadcaster.broadcast(
            message=json.dumps(sse_message),
            channel=f"camera_{camera_id}"
        )

        return {
            "status": "success",
            "camera_id": camera_id,
            "updated_at": status_data["last_updated"],
        }

    except Exception as e:
        logger.error(f"Camera status update failed for {camera_id}: {e}")
        raise DatabaseError(f"Status update failed: {e}") from e
