"""gRPC Server Implementation for Alert Service.

This module implements the gRPC server for the alert service,
handling violation alerts, anomaly alerts, and notification management.
"""

import asyncio
import time
from datetime import datetime

import grpc

from ..core.config import get_settings
from ..core.logging import get_logger
from ..infrastructure.database.postgresql_service import get_async_session
from ..models.analytics import RuleViolation, TrafficAnomaly
from ..proto import alert_service_pb2 as alert_pb
from ..proto import alert_service_pb2_grpc as alert_grpc
from .alert_service import AlertService

logger = get_logger(__name__)


class AlertServiceImpl(alert_grpc.AlertServiceServicer):
    """gRPC implementation of the alert service."""

    def __init__(self, alert_service: AlertService = None):
        """Initialize alert gRPC service."""
        self.alert_service = alert_service
        self.settings = get_settings()

    async def get_alert_service(self) -> AlertService:
        """Get alert service instance."""
        if not self.alert_service:
            # Create session and alert service
            async with get_async_session() as session:
                self.alert_service = AlertService(session, self.settings)
        return self.alert_service

    async def ProcessViolationAlert(
        self, request: alert_pb.ViolationAlertRequest, context: grpc.aio.ServicerContext
    ) -> alert_pb.AlertResponse:
        """Process violation alert."""
        correlation_id = request.correlation_id or context.invocation_metadata().get(
            "correlation-id", ""
        )

        logger.info(
            "ProcessViolationAlert called",
            violation_id=request.violation.id,
            violation_type=request.violation.violation_type,
            severity=request.violation.severity,
            recipients_count=len(request.recipients),
            correlation_id=correlation_id,
        )

        try:
            alert_service = await self.get_alert_service()

            # Create RuleViolation object from protobuf
            violation = RuleViolation(
                id=request.violation.id,
                violation_type=request.violation.violation_type,
                severity=request.violation.severity,
                camera_id=request.violation.camera_id,
                detection_time=datetime.utcfromtimestamp(
                    request.violation.detection_time
                ),
                measured_value=request.violation.measured_value,
                threshold_value=request.violation.threshold_value,
                license_plate=request.violation.license_plate or None,
                detection_confidence=request.violation.confidence,
                vehicle_type=request.violation.vehicle_type or None,
                zone_id=request.violation.zone_id or None,
                status="active",
            )

            # Process violation alert
            delivery_results = await alert_service.process_violation_alert(
                violation,
                list(request.recipients),
                list(request.notification_channels)
                if request.notification_channels
                else None,
            )

            # Convert delivery results to protobuf
            response = alert_pb.AlertResponse(
                success=len(delivery_results) > 0,
                correlation_id=correlation_id,
                message="Violation alert processed successfully",
            )

            for result in delivery_results:
                delivery_result_pb = alert_pb.DeliveryResult(
                    notification_id=result.get("notification_id", ""),
                    alert_type=result.get("alert_type", ""),
                    channel=result.get("channel", ""),
                    recipient=result.get("recipient", ""),
                    status=result.get("status", ""),
                    error=result.get("error", ""),
                    delivered_at=time.time()
                    if result.get("status") == "delivered"
                    else 0.0,
                    details=result.get("delivery_result", {}).get("details", {}),
                )
                response.delivery_results.append(delivery_result_pb)

            logger.info(
                "ProcessViolationAlert completed",
                violation_id=request.violation.id,
                delivery_results_count=len(delivery_results),
                correlation_id=correlation_id,
            )

            return response

        except Exception as e:
            logger.error(
                f"ProcessViolationAlert failed: {e}",
                violation_id=request.violation.id,
                correlation_id=correlation_id,
            )
            return alert_pb.AlertResponse(
                success=False,
                correlation_id=correlation_id,
                message=f"Violation alert processing failed: {str(e)}",
            )

    async def ProcessAnomalyAlert(
        self, request: alert_pb.AnomalyAlertRequest, context: grpc.aio.ServicerContext
    ) -> alert_pb.AlertResponse:
        """Process anomaly alert."""
        correlation_id = request.correlation_id or context.invocation_metadata().get(
            "correlation-id", ""
        )

        logger.info(
            "ProcessAnomalyAlert called",
            anomaly_id=request.anomaly.id,
            anomaly_type=request.anomaly.anomaly_type,
            severity=request.anomaly.severity,
            recipients_count=len(request.recipients),
            correlation_id=correlation_id,
        )

        try:
            alert_service = await self.get_alert_service()

            # Create TrafficAnomaly object from protobuf
            anomaly = TrafficAnomaly(
                id=request.anomaly.id,
                anomaly_type=request.anomaly.anomaly_type,
                severity=request.anomaly.severity,
                camera_id=request.anomaly.camera_id,
                detection_time=datetime.utcfromtimestamp(
                    request.anomaly.detection_time
                ),
                anomaly_score=request.anomaly.score,
                confidence=request.anomaly.confidence,
                probable_cause=request.anomaly.probable_cause or "unknown",
                baseline_value=request.anomaly.baseline_value
                if request.anomaly.baseline_value > 0
                else None,
                observed_value=request.anomaly.observed_value
                if request.anomaly.observed_value > 0
                else None,
                deviation_magnitude=request.anomaly.deviation_magnitude
                if request.anomaly.deviation_magnitude > 0
                else None,
                zone_id=request.anomaly.zone_id or None,
                detection_method=request.anomaly.detection_method or "unknown",
                status="active",
            )

            # Process anomaly alert
            delivery_results = await alert_service.process_anomaly_alert(
                anomaly,
                list(request.recipients),
                list(request.notification_channels)
                if request.notification_channels
                else None,
            )

            # Convert delivery results to protobuf
            response = alert_pb.AlertResponse(
                success=len(delivery_results) > 0,
                correlation_id=correlation_id,
                message="Anomaly alert processed successfully",
            )

            for result in delivery_results:
                delivery_result_pb = alert_pb.DeliveryResult(
                    notification_id=result.get("notification_id", ""),
                    alert_type=result.get("alert_type", ""),
                    channel=result.get("channel", ""),
                    recipient=result.get("recipient", ""),
                    status=result.get("status", ""),
                    error=result.get("error", ""),
                    delivered_at=time.time()
                    if result.get("status") == "delivered"
                    else 0.0,
                    details=result.get("delivery_result", {}).get("details", {}),
                )
                response.delivery_results.append(delivery_result_pb)

            logger.info(
                "ProcessAnomalyAlert completed",
                anomaly_id=request.anomaly.id,
                delivery_results_count=len(delivery_results),
                correlation_id=correlation_id,
            )

            return response

        except Exception as e:
            logger.error(
                f"ProcessAnomalyAlert failed: {e}",
                anomaly_id=request.anomaly.id,
                correlation_id=correlation_id,
            )
            return alert_pb.AlertResponse(
                success=False,
                correlation_id=correlation_id,
                message=f"Anomaly alert processing failed: {str(e)}",
            )

    async def SendAlert(
        self, request: alert_pb.CustomAlertRequest, context: grpc.aio.ServicerContext
    ) -> alert_pb.AlertResponse:
        """Send custom alert."""
        correlation_id = request.correlation_id or context.invocation_metadata().get(
            "correlation-id", ""
        )

        logger.info(
            "SendAlert called",
            alert_type=request.alert_type,
            priority=request.priority,
            recipients_count=len(request.recipients),
            correlation_id=correlation_id,
        )

        try:
            alert_service = await self.get_alert_service()

            # Send custom alert through each channel
            delivery_results = []

            for recipient in request.recipients:
                for channel_name in request.notification_channels or ["email"]:
                    result = await alert_service._send_alert(
                        alert_type=request.alert_type,
                        reference_id="custom",
                        channel_name=channel_name,
                        recipient=recipient,
                        subject=request.subject,
                        message=request.message,
                        priority=request.priority,
                        metadata=dict(request.metadata),
                    )
                    delivery_results.append(result)

            # Convert delivery results to protobuf
            response = alert_pb.AlertResponse(
                success=len(delivery_results) > 0,
                correlation_id=correlation_id,
                message="Custom alert sent successfully",
            )

            for result in delivery_results:
                delivery_result_pb = alert_pb.DeliveryResult(
                    notification_id=result.get("notification_id", ""),
                    alert_type=result.get("alert_type", ""),
                    channel=result.get("channel", ""),
                    recipient=result.get("recipient", ""),
                    status=result.get("status", ""),
                    error=result.get("error", ""),
                    delivered_at=time.time()
                    if result.get("status") == "delivered"
                    else 0.0,
                )
                response.delivery_results.append(delivery_result_pb)

            return response

        except Exception as e:
            logger.error(f"SendAlert failed: {e}", correlation_id=correlation_id)
            return alert_pb.AlertResponse(
                success=False,
                correlation_id=correlation_id,
                message=f"Custom alert sending failed: {str(e)}",
            )

    async def GetAlertStatistics(
        self,
        request: alert_pb.AlertStatisticsRequest,
        context: grpc.aio.ServicerContext,
    ) -> alert_pb.AlertStatisticsResponse:
        """Get alert statistics."""
        try:
            alert_service = await self.get_alert_service()

            time_range = None
            if request.start_time > 0 and request.end_time > 0:
                time_range = (
                    datetime.utcfromtimestamp(request.start_time),
                    datetime.utcfromtimestamp(request.end_time),
                )

            stats = await alert_service.get_alert_statistics(
                time_range=time_range, camera_id=request.camera_id or None
            )

            response = alert_pb.AlertStatisticsResponse(
                success=True, message="Alert statistics retrieved successfully"
            )

            # Add summary
            summary = stats["summary"]
            response.summary.CopyFrom(
                alert_pb.StatisticsSummary(
                    total_alerts=summary["total_alerts"],
                    delivered_alerts=summary["delivered_alerts"],
                    failed_alerts=summary["failed_alerts"],
                    pending_alerts=summary["pending_alerts"],
                    delivery_rate=summary["delivery_rate"],
                )
            )

            # Add channel stats
            for channel, channel_stats in stats["by_channel"].items():
                response.by_channel[channel] = alert_pb.ChannelStats(
                    total=channel_stats["total"],
                    delivered=channel_stats["delivered"],
                    failed=channel_stats["failed"],
                    pending=channel_stats["pending"],
                )

            # Add priority stats
            for priority, priority_stats in stats["by_priority"].items():
                response.by_priority[priority] = alert_pb.PriorityStats(
                    total=priority_stats["total"],
                    delivered=priority_stats["delivered"],
                    failed=priority_stats["failed"],
                    pending=priority_stats["pending"],
                )

            # Add time range
            if stats["time_range"]["start"]:
                response.time_range.CopyFrom(
                    alert_pb.TimeRange(
                        start=stats["time_range"]["start"].timestamp(),
                        end=stats["time_range"]["end"].timestamp(),
                    )
                )

            return response

        except Exception as e:
            logger.error(f"GetAlertStatistics failed: {e}")
            return alert_pb.AlertStatisticsResponse(
                success=False, message=f"Failed to get alert statistics: {str(e)}"
            )

    async def RetryFailedAlerts(
        self, request: alert_pb.RetryRequest, context: grpc.aio.ServicerContext
    ) -> alert_pb.RetryResponse:
        """Retry failed alerts."""
        try:
            alert_service = await self.get_alert_service()

            max_retries = request.max_retries if request.max_retries > 0 else 3
            retry_result = await alert_service.retry_failed_alerts(max_retries)

            response = alert_pb.RetryResponse(
                success=True,
                total_retried=retry_result["total_retried"],
                successful_retries=retry_result["successful_retries"],
                failed_retries=retry_result["failed_retries"],
                message="Failed alerts retry completed",
            )

            for result in retry_result["results"]:
                retry_result_pb = alert_pb.RetryResult(
                    notification_id=result["notification_id"],
                    status=result["status"],
                    attempts=result.get("attempts", 0),
                    error=result.get("error", ""),
                )
                response.results.append(retry_result_pb)

            return response

        except Exception as e:
            logger.error(f"RetryFailedAlerts failed: {e}")
            return alert_pb.RetryResponse(
                success=False,
                total_retried=0,
                successful_retries=0,
                failed_retries=0,
                message=f"Failed alerts retry failed: {str(e)}",
            )

    async def AcknowledgeAlert(
        self, request: alert_pb.AcknowledgeRequest, context: grpc.aio.ServicerContext
    ) -> alert_pb.AcknowledgeResponse:
        """Acknowledge alert."""
        try:
            alert_service = await self.get_alert_service()

            success = await alert_service.acknowledge_alert(
                request.notification_id,
                request.acknowledged_by,
                request.response_action or None,
                request.notes or None,
            )

            return alert_pb.AcknowledgeResponse(
                success=success,
                notification_id=request.notification_id,
                message="Alert acknowledged successfully"
                if success
                else "Failed to acknowledge alert",
            )

        except Exception as e:
            logger.error(f"AcknowledgeAlert failed: {e}")
            return alert_pb.AcknowledgeResponse(
                success=False,
                notification_id=request.notification_id,
                message=f"Alert acknowledgment failed: {str(e)}",
            )

    async def HealthCheck(
        self, request: alert_pb.HealthCheckRequest, context: grpc.aio.ServicerContext
    ) -> alert_pb.HealthCheckResponse:
        """Perform health check."""
        start_time = time.perf_counter()

        try:
            # Check if alert service is available
            alert_service = await self.get_alert_service()

            response_time = (time.perf_counter() - start_time) * 1000

            return alert_pb.HealthCheckResponse(
                status=alert_pb.HealthCheckResponse.Status.SERVING,
                message="Alert service is healthy",
                response_time_ms=response_time,
            )

        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Alert health check failed: {e}")

            return alert_pb.HealthCheckResponse(
                status=alert_pb.HealthCheckResponse.Status.NOT_SERVING,
                message=f"Alert service unhealthy: {str(e)}",
                response_time_ms=response_time,
            )


class AlertGrpcServer:
    """Alert gRPC server manager."""

    def __init__(self, host: str = "127.0.0.1", port: int = 50053):
        self.host = host
        self.port = port
        self.server: grpc.aio.Server = None
        self.service_impl = None

    async def start(self):
        """Start the alert gRPC server."""
        try:
            # Create alert service implementation
            self.service_impl = AlertServiceImpl()

            # Create gRPC server
            self.server = grpc.aio.server()
            alert_grpc.add_AlertServiceServicer_to_server(
                self.service_impl, self.server
            )

            # Add server options for performance
            options = [
                ("grpc.keepalive_time_ms", 10000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_time_between_pings_ms", 10000),
                ("grpc.http2.min_ping_interval_without_data_ms", 300000),
                ("grpc.max_receive_message_length", 16 * 1024 * 1024),  # 16MB
                ("grpc.max_send_message_length", 16 * 1024 * 1024),  # 16MB
            ]

            listen_addr = f"{self.host}:{self.port}"
            self.server.add_insecure_port(listen_addr)

            await self.server.start()
            logger.info(f"Alert gRPC server started on {listen_addr}")

        except Exception as e:
            logger.error(f"Failed to start alert gRPC server: {e}")
            raise

    async def stop(self):
        """Stop the alert gRPC server."""
        if self.server:
            logger.info("Stopping alert gRPC server...")
            await self.server.stop(grace=5.0)
            logger.info("Alert gRPC server stopped")

    async def serve_forever(self):
        """Start server and run until interrupted."""
        await self.start()

        try:
            await self.server.wait_for_termination()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        finally:
            await self.stop()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Main entry point
async def main():
    """Main entry point for alert gRPC server."""
    import argparse

    parser = argparse.ArgumentParser(description="Alert gRPC Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=50053, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    server = AlertGrpcServer(host=args.host, port=args.port)

    try:
        await server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
