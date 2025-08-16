"""Comprehensive Analytics API Integration Tests.

This test suite validates the complete analytics API functionality including:
- Real-time analytics endpoints
- Historical data queries with TimescaleDB
- Incident management operations
- Report generation with background workers
- Dashboard data aggregation
- Anomaly detection integration
- Performance under load
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import status
from fastapi.testclient import TestClient
from httpx import AsyncClient

from its_camera_ai.api.app import create_app
from its_camera_ai.core.config import get_settings_for_testing
from its_camera_ai.services.analytics_dtos import (
    AggregationLevel,
    DetectionData,
    DetectionResultDTO,
    BoundingBoxDTO,
    ProcessingResult,
    RealtimeTrafficMetrics,
    CongestionLevel,
)


@pytest.mark.integration
@pytest.mark.asyncio
class TestAnalyticsAPIIntegration:
    """Integration tests for Analytics API endpoints."""

    @pytest_asyncio.fixture
    async def app(self, test_settings):
        """Create test application with real services."""
        return create_app(test_settings)

    @pytest_asyncio.fixture
    async def async_client(self, app):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest_asyncio.fixture
    async def mock_analytics_service(self):
        """Mock analytics service with realistic responses."""
        service = AsyncMock()
        
        # Mock real-time analytics response
        now = datetime.now(UTC)
        mock_analytics = {
            "camera_id": "test_camera_1",
            "timestamp": now,
            "total_vehicles": 25,
            "vehicle_breakdown": {"car": 20, "truck": 3, "bus": 2},
            "average_speed": 45.2,
            "traffic_density": 0.6,
            "congestion_level": "moderate",
            "flow_rate": 1500,
            "occupancy_rate": 60.0,
            "violations": [],
            "anomalies": [],
            "processing_time_ms": 85.5,
            "detection_quality": 0.92,
        }
        
        service.get_realtime_analytics.return_value = mock_analytics
        service.get_active_violations.return_value = []
        service.get_traffic_anomalies.return_value = []
        service.calculate_traffic_metrics.return_value = []
        
        return service

    @pytest_asyncio.fixture
    async def mock_incident_service(self):
        """Mock incident management service."""
        service = AsyncMock()
        
        mock_incident = {
            "id": str(uuid4()),
            "camera_id": "test_camera_1",
            "incident_type": "congestion",
            "severity": "medium",
            "status": "active",
            "detected_at": datetime.now(UTC),
            "resolved_at": None,
            "description": "Traffic congestion detected",
            "location": {"lat": 10.762622, "lon": 106.660172},
        }
        
        service.list_incidents.return_value = {
            "items": [mock_incident],
            "total": 1,
            "page": 1,
            "size": 20,
            "total_pages": 1,
        }
        service.get_incident.return_value = mock_incident
        service.update_incident.return_value = mock_incident
        
        return service

    @pytest_asyncio.fixture
    async def sample_detection_data(self):
        """Create sample detection data for testing."""
        return DetectionData(
            camera_id="test_camera_1",
            timestamp=datetime.now(UTC),
            frame_id="frame_001",
            vehicle_count=15,
            detections=[
                DetectionResultDTO(
                    detection_id="det_001",
                    class_name="car",
                    confidence=0.95,
                    bbox=BoundingBoxDTO(x1=100, y1=150, x2=200, y2=250),
                    track_id="track_001",
                    timestamp=datetime.now(UTC),
                    vehicle_type="car",
                    speed=45.5,
                    direction="north",
                    attributes={"color": "blue"}
                ),
                DetectionResultDTO(
                    detection_id="det_002",
                    class_name="truck",
                    confidence=0.88,
                    bbox=BoundingBoxDTO(x1=300, y1=180, x2=450, y2=320),
                    track_id="track_002",
                    timestamp=datetime.now(UTC),
                    vehicle_type="truck",
                    speed=38.2,
                    direction="south",
                    attributes={"color": "red"}
                ),
            ],
            processing_metadata={
                "model_version": "yolo11n_v1.2",
                "inference_time_ms": 45.2,
                "gpu_utilization": 0.75,
            }
        )

    async def test_get_real_time_analytics_success(
        self, async_client, auth_headers, mock_analytics_service
    ):
        """Test successful real-time analytics retrieval."""
        camera_id = "test_camera_1"
        
        with patch(
            "its_camera_ai.api.dependencies.get_realtime_analytics_service",
            return_value=mock_analytics_service
        ):
            response = await async_client.get(
                f"/api/v1/analytics/real-time/{camera_id}",
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate response structure
        assert data["camera_id"] == camera_id
        assert "timestamp" in data
        assert "total_vehicles" in data
        assert "vehicle_breakdown" in data
        assert "average_speed" in data
        assert "traffic_density" in data
        assert "congestion_level" in data
        assert "processing_time_ms" in data
        
        # Validate data types and ranges
        assert isinstance(data["total_vehicles"], int)
        assert data["total_vehicles"] >= 0
        assert isinstance(data["average_speed"], (int, float))
        assert 0 <= data["traffic_density"] <= 1
        assert data["congestion_level"] in [
            "free_flow", "light", "moderate", "heavy", "severe"
        ]
        
        mock_analytics_service.get_realtime_analytics.assert_called_once_with(
            camera_id=camera_id
        )

    async def test_get_real_time_analytics_camera_not_found(
        self, async_client, auth_headers, mock_analytics_service
    ):
        """Test real-time analytics with non-existent camera."""
        camera_id = "non_existent_camera"
        
        mock_analytics_service.get_realtime_analytics.side_effect = ValueError(
            f"Camera {camera_id} not found"
        )
        
        with patch(
            "its_camera_ai.api.dependencies.get_realtime_analytics_service",
            return_value=mock_analytics_service
        ):
            response = await async_client.get(
                f"/api/v1/analytics/real-time/{camera_id}",
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not found" in response.json()["detail"].lower()

    async def test_get_real_time_analytics_service_error(
        self, async_client, auth_headers, mock_analytics_service
    ):
        """Test real-time analytics with service error."""
        camera_id = "test_camera_1"
        
        mock_analytics_service.get_realtime_analytics.side_effect = Exception(
            "Service temporarily unavailable"
        )
        
        with patch(
            "its_camera_ai.api.dependencies.get_realtime_analytics_service",
            return_value=mock_analytics_service
        ):
            response = await async_client.get(
                f"/api/v1/analytics/real-time/{camera_id}",
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to retrieve analytics data" in response.json()["detail"]

    async def test_list_incidents_success(
        self, async_client, auth_headers, mock_incident_service
    ):
        """Test successful incident listing with pagination."""
        with patch(
            "its_camera_ai.api.dependencies.get_incident_management_service",
            return_value=mock_incident_service
        ):
            response = await async_client.get(
                "/api/v1/analytics/incidents?page=1&size=20",
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate pagination structure
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "size" in data
        assert "total_pages" in data
        
        # Validate incident structure
        if data["items"]:
            incident = data["items"][0]
            assert "id" in incident
            assert "camera_id" in incident
            assert "incident_type" in incident
            assert "severity" in incident
            assert "status" in incident
            assert "detected_at" in incident
        
        mock_incident_service.list_incidents.assert_called_once()

    async def test_list_incidents_with_filters(
        self, async_client, auth_headers, mock_incident_service
    ):
        """Test incident listing with various filters."""
        filters = {
            "camera_id": "test_camera_1",
            "incident_type": "congestion",
            "severity": "high",
            "status": "active",
        }
        
        with patch(
            "its_camera_ai.api.dependencies.get_incident_management_service",
            return_value=mock_incident_service
        ):
            response = await async_client.get(
                "/api/v1/analytics/incidents",
                params=filters,
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify filters were passed to service
        call_args = mock_incident_service.list_incidents.call_args
        assert call_args[1]["camera_id"] == filters["camera_id"]
        assert call_args[1]["incident_type"] == filters["incident_type"]
        assert call_args[1]["severity"] == filters["severity"]
        assert call_args[1]["status"] == filters["status"]

    async def test_get_incident_success(
        self, async_client, auth_headers, mock_incident_service
    ):
        """Test successful incident retrieval by ID."""
        incident_id = str(uuid4())
        
        with patch(
            "its_camera_ai.api.dependencies.get_incident_management_service",
            return_value=mock_incident_service
        ):
            response = await async_client.get(
                f"/api/v1/analytics/incidents/{incident_id}",
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate incident structure
        assert "id" in data
        assert "camera_id" in data
        assert "incident_type" in data
        assert "severity" in data
        assert "status" in data
        
        mock_incident_service.get_incident.assert_called_once_with(
            incident_id=incident_id
        )

    async def test_get_incident_not_found(
        self, async_client, auth_headers, mock_incident_service
    ):
        """Test incident retrieval with non-existent ID."""
        incident_id = str(uuid4())
        
        mock_incident_service.get_incident.return_value = None
        
        with patch(
            "its_camera_ai.api.dependencies.get_incident_management_service",
            return_value=mock_incident_service
        ):
            response = await async_client.get(
                f"/api/v1/analytics/incidents/{incident_id}",
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Incident {incident_id} not found" in response.json()["detail"]

    async def test_update_incident_success(
        self, async_client, admin_auth_headers, mock_incident_service
    ):
        """Test successful incident update."""
        incident_id = str(uuid4())
        update_data = {
            "status": "resolved",
            "notes": "Issue resolved by maintenance team"
        }
        
        with patch(
            "its_camera_ai.api.dependencies.get_incident_management_service",
            return_value=mock_incident_service
        ):
            response = await async_client.put(
                f"/api/v1/analytics/incidents/{incident_id}",
                params=update_data,
                headers=admin_auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        
        # Verify update was called with correct data
        call_args = mock_incident_service.update_incident.call_args
        assert call_args[1]["incident_id"] == incident_id
        update_data_passed = call_args[1]["update_data"]
        assert update_data_passed["status"] == "resolved"
        assert update_data_passed["notes"] == update_data["notes"]
        assert "resolved_at" in update_data_passed
        assert "resolved_by" in update_data_passed

    async def test_generate_report_success(
        self, async_client, admin_auth_headers
    ):
        """Test successful report generation."""
        report_request = {
            "report_type": "traffic_summary",
            "time_range": {
                "start_time": (datetime.now(UTC) - timedelta(days=7)).isoformat(),
                "end_time": datetime.now(UTC).isoformat(),
            },
            "camera_ids": ["camera_1", "camera_2"],
            "format": "pdf",
            "include_charts": True,
        }
        
        with patch("its_camera_ai.workers.analytics_worker.generate_analytics_report") as mock_task:
            mock_task.apply_async.return_value = MagicMock(id="task_123")
            
            response = await async_client.post(
                "/api/v1/analytics/reports",
                json=report_request,
                headers=admin_auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate report response structure
        assert "report_id" in data
        assert "report_type" in data
        assert data["status"] == "pending"
        assert "created_at" in data
        assert "expires_at" in data
        assert data["report_type"] == report_request["report_type"]
        
        # Verify task was scheduled
        mock_task.apply_async.assert_called_once()

    async def test_query_historical_data_success(
        self, async_client, auth_headers, mock_analytics_service
    ):
        """Test successful historical data query."""
        query_data = {
            "camera_ids": ["camera_1", "camera_2"],
            "time_range": {
                "start_time": (datetime.now(UTC) - timedelta(days=7)).isoformat(),
                "end_time": datetime.now(UTC).isoformat(),
            },
            "aggregation_level": "hourly",
            "metrics": ["vehicle_count", "average_speed", "congestion_level"],
            "page": 1,
            "size": 100,
        }
        
        mock_result = {
            "items": [],
            "total": 0,
            "page": 1,
            "size": 100,
            "total_pages": 0,
        }
        
        with patch(
            "its_camera_ai.api.dependencies.get_historical_analytics_service"
        ) as mock_service:
            mock_service.return_value.query_historical_data.return_value = mock_result
            
            response = await async_client.post(
                "/api/v1/analytics/historical-query",
                json=query_data,
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate response structure
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "size" in data

    async def test_get_dashboard_data_success(
        self, async_client, auth_headers, mock_analytics_service
    ):
        """Test successful dashboard data retrieval."""
        camera_id = "test_camera_1"
        
        with patch(
            "its_camera_ai.api.dependencies.get_analytics_service",
            return_value=mock_analytics_service
        ):
            response = await async_client.get(
                f"/api/v1/analytics/dashboard/{camera_id}",
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate dashboard response structure
        assert data["camera_id"] == camera_id
        assert "timestamp" in data
        assert "real_time_metrics" in data
        assert "camera_status" in data
        assert "performance_metrics" in data
        assert "alerts_summary" in data
        assert "hourly_trends" in data
        
        # Validate real-time metrics structure
        real_time = data["real_time_metrics"]
        assert "total_vehicles" in real_time
        assert "average_speed" in real_time
        assert "congestion_level" in real_time
        assert "occupancy_rate" in real_time

    async def test_create_alert_rule_success(
        self, async_client, admin_auth_headers
    ):
        """Test successful alert rule creation."""
        rule_request = {
            "name": "High Traffic Alert",
            "description": "Alert when traffic exceeds threshold",
            "camera_ids": ["camera_1", "camera_2"],
            "conditions": {
                "vehicle_count": {"operator": ">", "value": 50},
                "congestion_level": {"operator": "==", "value": "severe"}
            },
            "severity": "high",
            "notification_channels": ["email", "webhook"],
            "is_active": True,
        }
        
        mock_rule = {
            "id": str(uuid4()),
            **rule_request,
            "created_at": datetime.now(UTC).isoformat(),
        }
        
        with patch(
            "its_camera_ai.api.dependencies.get_alert_service"
        ) as mock_service:
            mock_service.return_value.create_alert_rule.return_value = mock_rule
            
            response = await async_client.post(
                "/api/v1/analytics/alerts/rules",
                json=rule_request,
                headers=admin_auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate alert rule response
        assert "id" in data
        assert data["name"] == rule_request["name"]
        assert data["description"] == rule_request["description"]
        assert data["severity"] == rule_request["severity"]
        assert data["is_active"] == rule_request["is_active"]

    async def test_trigger_anomaly_detection_success(
        self, async_client, admin_auth_headers
    ):
        """Test successful anomaly detection trigger."""
        detection_request = {
            "camera_ids": ["camera_1", "camera_2"],
            "time_range": {
                "start_time": (datetime.now(UTC) - timedelta(hours=24)).isoformat(),
                "end_time": datetime.now(UTC).isoformat(),
            },
            "sensitivity": 0.8,
            "algorithm": "isolation_forest",
        }
        
        with patch("its_camera_ai.workers.analytics_worker.detect_anomalies") as mock_task:
            mock_task.apply_async.return_value = MagicMock(id="task_456")
            
            response = await async_client.post(
                "/api/v1/analytics/anomalies/detect",
                json=detection_request,
                headers=admin_auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate anomaly detection response
        assert "job_id" in data
        assert "task_id" in data
        assert data["status"] == "started"
        assert "message" in data
        assert "cameras" in data
        assert "estimated_completion" in data
        
        # Verify task was scheduled
        mock_task.apply_async.assert_called_once()

    async def test_get_traffic_predictions_success(
        self, async_client, auth_headers, mock_analytics_service
    ):
        """Test successful traffic predictions retrieval."""
        camera_id = "test_camera_1"
        forecast_hours = 24
        
        with patch(
            "its_camera_ai.api.dependencies.get_analytics_service",
            return_value=mock_analytics_service
        ):
            response = await async_client.get(
                f"/api/v1/analytics/predictions/{camera_id}?forecast_hours={forecast_hours}",
                headers=auth_headers
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Validate prediction response structure
        assert data["camera_id"] == camera_id
        assert "prediction_timestamp" in data
        assert "forecast_start" in data
        assert "forecast_end" in data
        assert "predictions" in data
        assert "confidence_interval" in data
        assert "ml_model_version" in data
        assert "ml_model_accuracy" in data
        assert "factors_considered" in data
        
        # Validate predictions structure
        if data["predictions"]:
            prediction = data["predictions"][0]
            assert "timestamp" in prediction
            assert "predicted_vehicle_count" in prediction
            assert "predicted_avg_speed" in prediction
            assert "predicted_congestion_level" in prediction
            assert "confidence" in prediction

    @pytest.mark.performance
    async def test_concurrent_real_time_analytics_requests(
        self, async_client, auth_headers, mock_analytics_service
    ):
        """Test concurrent real-time analytics requests performance."""
        camera_ids = [f"camera_{i}" for i in range(10)]
        
        with patch(
            "its_camera_ai.api.dependencies.get_realtime_analytics_service",
            return_value=mock_analytics_service
        ):
            # Create concurrent requests
            tasks = [
                async_client.get(
                    f"/api/v1/analytics/real-time/{camera_id}",
                    headers=auth_headers
                )
                for camera_id in camera_ids
            ]
            
            # Execute concurrently and measure time
            start_time = asyncio.get_event_loop().time()
            responses = await asyncio.gather(*tasks)
            end_time = asyncio.get_event_loop().time()
            
            # Validate all requests succeeded
            for response in responses:
                assert response.status_code == status.HTTP_200_OK
            
            # Validate performance
            total_time = end_time - start_time
            avg_time_per_request = total_time / len(camera_ids)
            
            # Should handle 10 concurrent requests in under 2 seconds
            assert total_time < 2.0
            # Average response time should be under 200ms
            assert avg_time_per_request < 0.2
            
            # Verify service was called for each camera
            assert mock_analytics_service.get_realtime_analytics.call_count == len(camera_ids)

    async def test_unauthorized_access(
        self, async_client
    ):
        """Test API access without authentication."""
        response = await async_client.get(
            "/api/v1/analytics/real-time/test_camera"
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_insufficient_permissions(
        self, async_client, auth_headers
    ):
        """Test API access with insufficient permissions."""
        # Test admin-only endpoint with regular user
        response = await async_client.post(
            "/api/v1/analytics/alerts/rules",
            json={"name": "test", "conditions": {}},
            headers=auth_headers
        )
        
        # Should require admin permissions
        assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_401_UNAUTHORIZED]

    @pytest.mark.performance
    async def test_api_response_times(
        self, async_client, auth_headers, mock_analytics_service
    ):
        """Test API response time requirements."""
        camera_id = "test_camera_1"
        
        with patch(
            "its_camera_ai.api.dependencies.get_realtime_analytics_service",
            return_value=mock_analytics_service
        ):
            # Measure response time for real-time analytics
            start_time = asyncio.get_event_loop().time()
            response = await async_client.get(
                f"/api/v1/analytics/real-time/{camera_id}",
                headers=auth_headers
            )
            end_time = asyncio.get_event_loop().time()
            
            assert response.status_code == status.HTTP_200_OK
            
            response_time = end_time - start_time
            # API should respond within 100ms for real-time endpoints
            assert response_time < 0.1

    async def test_data_validation_and_sanitization(
        self, async_client, auth_headers, mock_analytics_service
    ):
        """Test input validation and data sanitization."""
        # Test with invalid camera ID format
        invalid_camera_id = "<script>alert('xss')</script>"
        
        with patch(
            "its_camera_ai.api.dependencies.get_realtime_analytics_service",
            return_value=mock_analytics_service
        ):
            response = await async_client.get(
                f"/api/v1/analytics/real-time/{invalid_camera_id}",
                headers=auth_headers
            )
            
            # Should handle gracefully without script execution
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_404_NOT_FOUND,
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ]

    async def test_error_handling_and_logging(
        self, async_client, auth_headers, mock_analytics_service
    ):
        """Test comprehensive error handling and logging."""
        camera_id = "test_camera_1"
        
        # Simulate various error conditions
        error_scenarios = [
            (ValueError("Invalid input"), status.HTTP_400_BAD_REQUEST),
            (ConnectionError("Database unavailable"), status.HTTP_500_INTERNAL_SERVER_ERROR),
            (TimeoutError("Request timeout"), status.HTTP_500_INTERNAL_SERVER_ERROR),
            (Exception("Unknown error"), status.HTTP_500_INTERNAL_SERVER_ERROR),
        ]
        
        for error, expected_status in error_scenarios:
            mock_analytics_service.get_realtime_analytics.side_effect = error
            
            with patch(
                "its_camera_ai.api.dependencies.get_realtime_analytics_service",
                return_value=mock_analytics_service
            ):
                response = await async_client.get(
                    f"/api/v1/analytics/real-time/{camera_id}",
                    headers=auth_headers
                )
                
                assert response.status_code == expected_status
                
                # Validate error response structure
                error_data = response.json()
                assert "detail" in error_data
                assert isinstance(error_data["detail"], str)
                assert len(error_data["detail"]) > 0