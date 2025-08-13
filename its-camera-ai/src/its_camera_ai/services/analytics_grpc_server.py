"""gRPC Server Implementation for Analytics Service.

This module implements the gRPC server for the analytics service,
handling detection processing, traffic metrics, violations, and anomalies.
"""

import asyncio
import time
from datetime import UTC, datetime

import grpc

from ..core.config import get_settings
from ..core.logging import get_logger
from ..infrastructure.database.postgresql_service import get_async_session
from ..models.detection_result import DetectionResult
from ..proto import analytics_service_pb2 as analytics_pb
from ..proto import analytics_service_pb2_grpc as analytics_grpc
from .analytics_service import AnalyticsService

logger = get_logger(__name__)


class AnalyticsServiceImpl(analytics_grpc.AnalyticsServiceServicer):
    """gRPC implementation of the analytics service."""

    def __init__(self, analytics_service: AnalyticsService = None):
        """Initialize analytics gRPC service."""
        self.analytics_service = analytics_service
        self.settings = get_settings()

    async def get_analytics_service(self) -> AnalyticsService:
        """Get analytics service instance."""
        if not self.analytics_service:
            # Create session and analytics service
            async with get_async_session() as session:
                self.analytics_service = AnalyticsService(session, self.settings)
        return self.analytics_service

    async def ProcessDetections(
        self, request: analytics_pb.DetectionRequest, context: grpc.aio.ServicerContext
    ) -> analytics_pb.AnalyticsResponse:
        """Process detections and generate analytics."""
        start_time = time.perf_counter()
        correlation_id = request.correlation_id or context.invocation_metadata().get(
            "correlation-id", ""
        )

        logger.info(
            "ProcessDetections called",
            camera_id=request.camera_id,
            detection_count=len(request.detections),
            correlation_id=correlation_id,
        )

        try:
            analytics_service = await self.get_analytics_service()

            # Convert protobuf detections to DetectionResult objects
            detections = []
            for det_pb in request.detections:
                detection = DetectionResult(
                    id=det_pb.id,
                    camera_id=request.camera_id,
                    class_name=det_pb.class_name,
                    class_confidence=det_pb.confidence,
                    bbox_x=det_pb.bbox.x,
                    bbox_y=det_pb.bbox.y,
                    bbox_width=det_pb.bbox.width,
                    bbox_height=det_pb.bbox.height,
                    bbox_center_x=det_pb.bbox.center_x,
                    bbox_center_y=det_pb.bbox.center_y,
                    track_id=int(det_pb.track_id) if det_pb.track_id else None,
                    vehicle_type=det_pb.vehicle_type or None,
                    license_plate=det_pb.license_plate or None,
                    velocity_magnitude=det_pb.velocity_magnitude
                    if det_pb.velocity_magnitude > 0
                    else None,
                    is_vehicle=det_pb.is_vehicle,
                    detection_quality=det_pb.detection_quality,
                    detection_zone=det_pb.detection_zone or None,
                    created_at=datetime.utcfromtimestamp(request.frame_timestamp),
                )
                detections.append(detection)

            # Process detections through analytics service
            frame_timestamp = datetime.utcfromtimestamp(request.frame_timestamp)
            result = await analytics_service.process_detections(
                detections, request.camera_id, frame_timestamp
            )

            processing_time = (time.perf_counter() - start_time) * 1000

            # Convert result to protobuf response
            response = analytics_pb.AnalyticsResponse(
                success=True,
                camera_id=result["camera_id"],
                timestamp=result["timestamp"].timestamp(),
                vehicle_count=result["vehicle_count"],
                processing_time_ms=processing_time,
                correlation_id=correlation_id,
                message="Detections processed successfully",
            )

            # Add traffic metrics
            if "metrics" in result:
                metrics = result["metrics"]
                response.metrics.CopyFrom(
                    analytics_pb.TrafficMetrics(
                        total_vehicles=metrics.get("total_vehicles", 0),
                        vehicle_breakdown=metrics.get("vehicle_breakdown", {}),
                        average_speed=metrics.get("average_speed", 0.0) or 0.0,
                        traffic_density=metrics.get("traffic_density", 0.0),
                        congestion_level=metrics.get("congestion_level", "unknown"),
                        timestamp=result["timestamp"].timestamp(),
                    )
                )

            # Add violations
            for violation in result.get("violations", []):
                violation_pb = analytics_pb.Violation(
                    id=str(violation.get("detection_id", "")),
                    violation_type=violation.get("violation_type", ""),
                    severity=violation.get("severity", ""),
                    camera_id=violation.get("camera_id", ""),
                    detection_time=violation.get(
                        "detection_time", datetime.now(UTC)
                    ).timestamp(),
                    measured_value=violation.get("measured_value", 0.0),
                    threshold_value=violation.get("threshold_value", 0.0),
                    license_plate=violation.get("license_plate", ""),
                    confidence=violation.get("confidence", 0.0),
                    track_id=str(violation.get("track_id", "")),
                )
                response.violations.append(violation_pb)

            # Add anomalies
            for anomaly in result.get("anomalies", []):
                anomaly_pb = analytics_pb.Anomaly(
                    id=str(anomaly.get("anomaly_id", "")),
                    anomaly_type="traffic_pattern",
                    severity=anomaly.get("severity", ""),
                    camera_id=request.camera_id,
                    detection_time=datetime.now(UTC).timestamp(),
                    score=anomaly.get("anomaly_score", 0.0),
                    confidence=anomaly.get("confidence", 0.0),
                    probable_cause=anomaly.get("probable_cause", ""),
                    status="active",
                )
                response.anomalies.append(anomaly_pb)

            logger.info(
                "ProcessDetections completed",
                camera_id=request.camera_id,
                processing_time_ms=processing_time,
                violations_count=len(response.violations),
                anomalies_count=len(response.anomalies),
                correlation_id=correlation_id,
            )

            return response

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"ProcessDetections failed: {e}",
                camera_id=request.camera_id,
                correlation_id=correlation_id,
            )
            return analytics_pb.AnalyticsResponse(
                success=False,
                camera_id=request.camera_id,
                timestamp=request.frame_timestamp,
                vehicle_count=0,
                processing_time_ms=processing_time,
                correlation_id=correlation_id,
                message=f"Processing failed: {str(e)}",
            )

    async def GetTrafficMetrics(
        self,
        request: analytics_pb.TrafficMetricsRequest,
        context: grpc.aio.ServicerContext,
    ) -> analytics_pb.TrafficMetricsResponse:
        """Get traffic metrics for time range."""
        try:
            analytics_service = await self.get_analytics_service()

            start_time = datetime.utcfromtimestamp(request.start_time)
            end_time = datetime.utcfromtimestamp(request.end_time)

            metrics = await analytics_service.calculate_traffic_metrics(
                request.camera_id,
                (start_time, end_time),
                request.aggregation_period or "1hour",
            )

            response = analytics_pb.TrafficMetricsResponse(
                success=True, message="Traffic metrics retrieved successfully"
            )

            for metric in metrics:
                period_pb = analytics_pb.TrafficMetricsPeriod(
                    camera_id=metric.camera_id,
                    period_start=metric.period_start.timestamp(),
                    period_end=metric.period_end.timestamp(),
                    total_vehicles=metric.total_vehicles,
                    vehicle_breakdown=metric.vehicle_breakdown,
                    directional_flow=metric.directional_flow,
                    avg_speed=metric.avg_speed or 0.0,
                    occupancy_rate=metric.occupancy_rate,
                    congestion_level=metric.congestion_level,
                    queue_length=metric.queue_length or 0.0,
                )
                response.metrics.append(period_pb)

            return response

        except Exception as e:
            logger.error(f"GetTrafficMetrics failed: {e}")
            return analytics_pb.TrafficMetricsResponse(
                success=False, message=f"Failed to get traffic metrics: {str(e)}"
            )

    async def GetActiveViolations(
        self, request: analytics_pb.ViolationsRequest, context: grpc.aio.ServicerContext
    ) -> analytics_pb.ViolationsResponse:
        """Get active traffic violations."""
        try:
            analytics_service = await self.get_analytics_service()

            violations = await analytics_service.get_active_violations(
                camera_id=request.camera_id or None,
                violation_type=request.violation_type or None,
                severity=request.severity or None,
                limit=request.limit if request.limit > 0 else 100,
            )

            response = analytics_pb.ViolationsResponse(
                success=True,
                total_count=len(violations),
                message="Active violations retrieved successfully",
            )

            for violation in violations:
                violation_pb = analytics_pb.Violation(
                    id=str(violation["id"]),
                    violation_type=violation["type"],
                    severity=violation["severity"],
                    camera_id=violation["camera_id"],
                    detection_time=violation["detection_time"].timestamp(),
                    measured_value=violation.get("measured_value", 0.0) or 0.0,
                    threshold_value=violation.get("threshold_value", 0.0) or 0.0,
                    license_plate=violation.get("license_plate", ""),
                    confidence=violation.get("confidence", 0.0) or 0.0,
                )
                response.violations.append(violation_pb)

            return response

        except Exception as e:
            logger.error(f"GetActiveViolations failed: {e}")
            return analytics_pb.ViolationsResponse(
                success=False,
                total_count=0,
                message=f"Failed to get violations: {str(e)}",
            )

    async def GetTrafficAnomalies(
        self, request: analytics_pb.AnomaliesRequest, context: grpc.aio.ServicerContext
    ) -> analytics_pb.AnomaliesResponse:
        """Get traffic anomalies."""
        try:
            analytics_service = await self.get_analytics_service()

            time_range = None
            if request.start_time > 0 and request.end_time > 0:
                time_range = (
                    datetime.utcfromtimestamp(request.start_time),
                    datetime.utcfromtimestamp(request.end_time),
                )

            anomalies = await analytics_service.get_traffic_anomalies(
                camera_id=request.camera_id or None,
                anomaly_type=request.anomaly_type or None,
                min_score=request.min_score if request.min_score > 0 else 0.5,
                time_range=time_range,
                limit=request.limit if request.limit > 0 else 50,
            )

            response = analytics_pb.AnomaliesResponse(
                success=True,
                total_count=len(anomalies),
                message="Traffic anomalies retrieved successfully",
            )

            for anomaly in anomalies:
                anomaly_pb = analytics_pb.Anomaly(
                    id=str(anomaly["id"]),
                    anomaly_type=anomaly["type"],
                    severity=anomaly["severity"],
                    camera_id=anomaly["camera_id"],
                    detection_time=anomaly["detection_time"].timestamp(),
                    score=anomaly["score"],
                    confidence=anomaly["confidence"],
                    probable_cause=anomaly["probable_cause"],
                    status=anomaly["status"],
                )
                response.anomalies.append(anomaly_pb)

            return response

        except Exception as e:
            logger.error(f"GetTrafficAnomalies failed: {e}")
            return analytics_pb.AnomaliesResponse(
                success=False,
                total_count=0,
                message=f"Failed to get anomalies: {str(e)}",
            )

    async def GenerateReport(
        self, request: analytics_pb.ReportRequest, context: grpc.aio.ServicerContext
    ) -> analytics_pb.ReportResponse:
        """Generate analytics report."""
        try:
            analytics_service = await self.get_analytics_service()

            start_time = datetime.utcfromtimestamp(request.start_time)
            end_time = datetime.utcfromtimestamp(request.end_time)

            report = await analytics_service.generate_analytics_report(
                list(request.camera_ids),
                (start_time, end_time),
                request.report_type or "traffic_summary",
            )

            response = analytics_pb.ReportResponse(
                success=True,
                report_id=str(report.get("report_id", "")),
                report_type=report["report_type"],
                generated_at=report["generated_at"].timestamp(),
                message="Report generated successfully",
            )

            # Add summary
            summary = report["summary"]
            response.summary.CopyFrom(
                analytics_pb.ReportSummary(
                    total_vehicles=summary["total_vehicles"],
                    total_violations=summary["total_violations"],
                    total_anomalies=summary["total_anomalies"],
                    cameras_analyzed=summary["cameras_analyzed"],
                )
            )

            # Add camera summaries
            for camera_summary in report["camera_summaries"]:
                camera_pb = analytics_pb.CameraSummary(
                    camera_id=camera_summary["camera_id"],
                    total_vehicles=camera_summary["total_vehicles"],
                    violations=camera_summary["violations"],
                    anomalies=camera_summary["anomalies"],
                )
                response.camera_summaries.append(camera_pb)

            return response

        except Exception as e:
            logger.error(f"GenerateReport failed: {e}")
            return analytics_pb.ReportResponse(
                success=False,
                report_id="",
                report_type=request.report_type,
                generated_at=time.time(),
                message=f"Report generation failed: {str(e)}",
            )

    async def HealthCheck(
        self,
        request: analytics_pb.HealthCheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> analytics_pb.HealthCheckResponse:
        """Perform health check."""
        start_time = time.perf_counter()

        try:
            # Check if analytics service is available
            analytics_service = await self.get_analytics_service()

            response_time = (time.perf_counter() - start_time) * 1000

            return analytics_pb.HealthCheckResponse(
                status=analytics_pb.HealthCheckResponse.Status.SERVING,
                message="Analytics service is healthy",
                response_time_ms=response_time,
            )

        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Analytics health check failed: {e}")

            return analytics_pb.HealthCheckResponse(
                status=analytics_pb.HealthCheckResponse.Status.NOT_SERVING,
                message=f"Analytics service unhealthy: {str(e)}",
                response_time_ms=response_time,
            )


class AnalyticsGrpcServer:
    """Analytics gRPC server manager."""

    def __init__(self, host: str = "127.0.0.1", port: int = 50052):
        self.host = host
        self.port = port
        self.server: grpc.aio.Server = None
        self.service_impl = None

    async def start(self):
        """Start the analytics gRPC server."""
        try:
            # Create analytics service implementation
            self.service_impl = AnalyticsServiceImpl()

            # Create gRPC server
            self.server = grpc.aio.server()
            analytics_grpc.add_AnalyticsServiceServicer_to_server(
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
            logger.info(f"Analytics gRPC server started on {listen_addr}")

        except Exception as e:
            logger.error(f"Failed to start analytics gRPC server: {e}")
            raise

    async def stop(self):
        """Stop the analytics gRPC server."""
        if self.server:
            logger.info("Stopping analytics gRPC server...")
            await self.server.stop(grace=5.0)
            logger.info("Analytics gRPC server stopped")

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
    """Main entry point for analytics gRPC server."""
    import argparse

    parser = argparse.ArgumentParser(description="Analytics gRPC Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=50052, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    server = AnalyticsGrpcServer(host=args.host, port=args.port)

    try:
        await server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
