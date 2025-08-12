"""
Integration between Core Vision Engine and SSE broadcaster.

Automatically broadcasts camera events, detection results, and performance
metrics to connected dashboard clients via SSE.
"""

from datetime import datetime
from typing import Any

from ..api.sse_broadcaster import CameraEvent, SystemEvent, get_broadcaster
from ..core.logging import get_logger
from .core_vision_engine import VisionResult

logger = get_logger(__name__)


class SSEVisionIntegration:
    """Integration layer between vision processing and SSE broadcasting."""

    def __init__(self, enable_broadcasting: bool = True):
        self.enable_broadcasting = enable_broadcasting
        self.broadcaster = get_broadcaster() if enable_broadcasting else None
        self.performance_metrics_cache = {}
        self.last_health_broadcast = {}

    async def broadcast_detection_result(self, result: VisionResult):
        """Broadcast a new detection result to connected clients."""
        if not self.enable_broadcasting or not self.broadcaster:
            return

        try:
            event_data = {
                "frame_id": result.frame_id,
                "camera_id": result.camera_id,
                "timestamp": result.timestamp.isoformat(),
                "detection_count": result.detection_count,
                "vehicle_counts": {k.value: v for k, v in result.vehicle_counts.items()},
                "avg_confidence": result.avg_confidence,
                "processing_time_ms": result.total_processing_time_ms,
                "frame_resolution": result.frame_resolution,
                "gpu_memory_used_mb": result.gpu_memory_used_mb,
                "batch_size_used": result.batch_size_used,
                "detections": [
                    {
                        "bbox": detection.bbox,
                        "confidence": detection.confidence,
                        "class": detection.vehicle_class.value,
                        "track_id": getattr(detection, 'track_id', None),
                    }
                    for detection in result.detections
                ],
            }

            event = CameraEvent(
                camera_id=result.camera_id,
                event_type="detection_result",
                timestamp=result.timestamp,
                data=event_data,
            )

            await self.broadcaster.broadcast_camera_event(event)

        except Exception as e:
            logger.error(
                "Failed to broadcast detection result",
                camera_id=result.camera_id,
                error=str(e)
            )

    async def broadcast_camera_status_change(
        self,
        camera_id: str,
        new_status: str,
        previous_status: str | None = None,
        additional_data: dict[str, Any] | None = None
    ):
        """Broadcast camera status change event."""
        if not self.enable_broadcasting or not self.broadcaster:
            return

        try:
            event_data = {
                "new_status": new_status,
                "previous_status": previous_status,
                "change_time": datetime.now().isoformat(),
            }

            if additional_data:
                event_data.update(additional_data)

            event = CameraEvent(
                camera_id=camera_id,
                event_type="status_change",
                timestamp=datetime.now(),
                data=event_data,
            )

            await self.broadcaster.broadcast_camera_event(event)

            logger.info(
                "Camera status change broadcasted",
                camera_id=camera_id,
                new_status=new_status,
                previous_status=previous_status
            )

        except Exception as e:
            logger.error(
                "Failed to broadcast camera status change",
                camera_id=camera_id,
                error=str(e)
            )

    async def broadcast_camera_health_update(
        self,
        camera_id: str,
        health_metrics: dict[str, Any]
    ):
        """Broadcast camera health metrics update."""
        if not self.enable_broadcasting or not self.broadcaster:
            return

        # Throttle health updates to avoid spam (max once per minute per camera)
        current_time = datetime.now().timestamp()
        last_broadcast = self.last_health_broadcast.get(camera_id, 0)

        if current_time - last_broadcast < 60:  # 60 seconds throttle
            return

        try:
            self.last_health_broadcast[camera_id] = current_time

            event = CameraEvent(
                camera_id=camera_id,
                event_type="health_update",
                timestamp=datetime.now(),
                data=health_metrics,
            )

            await self.broadcaster.broadcast_camera_event(event)

        except Exception as e:
            logger.error(
                "Failed to broadcast camera health update",
                camera_id=camera_id,
                error=str(e)
            )

    async def broadcast_system_performance_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metrics: dict[str, Any] | None = None
    ):
        """Broadcast system performance alert."""
        if not self.enable_broadcasting or not self.broadcaster:
            return

        try:
            event_data = {
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }

            if metrics:
                event_data["metrics"] = metrics

            event = SystemEvent(
                event_type="performance_alert",
                timestamp=datetime.now(),
                data=event_data,
            )

            await self.broadcaster.broadcast_system_event(event)

            logger.info(
                "Performance alert broadcasted",
                alert_type=alert_type,
                severity=severity
            )

        except Exception as e:
            logger.error(
                "Failed to broadcast performance alert",
                error=str(e)
            )

    async def broadcast_system_statistics(self, stats: dict[str, Any]):
        """Broadcast system-wide processing statistics."""
        if not self.enable_broadcasting or not self.broadcaster:
            return

        try:
            event = SystemEvent(
                event_type="statistics",
                timestamp=datetime.now(),
                data=stats,
            )

            await self.broadcaster.broadcast_system_event(event)

        except Exception as e:
            logger.error(
                "Failed to broadcast system statistics",
                error=str(e)
            )

    async def broadcast_camera_error(
        self,
        camera_id: str,
        error_type: str,
        error_message: str,
        additional_data: dict[str, Any] | None = None
    ):
        """Broadcast camera error event."""
        if not self.enable_broadcasting or not self.broadcaster:
            return

        try:
            event_data = {
                "error_type": error_type,
                "error_message": error_message,
                "timestamp": datetime.now().isoformat(),
            }

            if additional_data:
                event_data.update(additional_data)

            event = CameraEvent(
                camera_id=camera_id,
                event_type="error",
                timestamp=datetime.now(),
                data=event_data,
            )

            await self.broadcaster.broadcast_camera_event(event)

            logger.warning(
                "Camera error broadcasted",
                camera_id=camera_id,
                error_type=error_type,
                error_message=error_message
            )

        except Exception as e:
            logger.error(
                "Failed to broadcast camera error",
                camera_id=camera_id,
                error=str(e)
            )

    async def broadcast_configuration_change(
        self,
        camera_id: str,
        changed_settings: dict[str, Any],
        user_id: str | None = None
    ):
        """Broadcast camera configuration change event."""
        if not self.enable_broadcasting or not self.broadcaster:
            return

        try:
            event_data = {
                "changed_settings": changed_settings,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
            }

            event = CameraEvent(
                camera_id=camera_id,
                event_type="configuration_change",
                timestamp=datetime.now(),
                data=event_data,
            )

            await self.broadcaster.broadcast_camera_event(event)

            logger.info(
                "Configuration change broadcasted",
                camera_id=camera_id,
                user_id=user_id,
                settings_count=len(changed_settings)
            )

        except Exception as e:
            logger.error(
                "Failed to broadcast configuration change",
                camera_id=camera_id,
                error=str(e)
            )


# Global SSE integration instance
_sse_integration: SSEVisionIntegration | None = None


def get_sse_integration() -> SSEVisionIntegration:
    """Get global SSE integration instance."""
    global _sse_integration
    if _sse_integration is None:
        _sse_integration = SSEVisionIntegration()
    return _sse_integration


async def setup_vision_sse_integration(vision_engine, enable_broadcasting: bool = True):
    """Set up integration between Vision Engine and SSE broadcasting.
    
    This function should be called during application startup to connect
    the vision processing pipeline to the SSE broadcasting system.
    """
    if not enable_broadcasting:
        logger.info("SSE broadcasting disabled")
        return

    sse_integration = get_sse_integration()

    # Hook into vision engine events (this would need to be implemented in the engine)
    # For now, we'll provide the integration instance that can be used manually

    logger.info("SSE vision integration set up successfully")
    return sse_integration
