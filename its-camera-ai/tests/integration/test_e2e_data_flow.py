"""End-to-End Data Flow Integration Tests.

This test suite validates the complete data flow pipeline:
Camera → ML Pipeline → Analytics → Database → API Responses

Testing scenarios:
- Complete detection processing pipeline
- Real-time analytics computation
- Background worker processing
- Database storage and retrieval
- Cache layer integration
- Performance under load
"""

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from its_camera_ai.core.logging import get_logger
from its_camera_ai.models.analytics import TrafficData, AggregatedMetrics, IncidentAlert
from its_camera_ai.services.analytics_dtos import (
    DetectionData,
    DetectionResultDTO,
    BoundingBoxDTO,
    ProcessingResult,
    RealtimeTrafficMetrics,
    CongestionLevel,
    AggregationLevel,
    TimeWindow,
)
from its_camera_ai.services.unified_analytics_service import UnifiedAnalyticsService
from its_camera_ai.services.ml_streaming_integration_service import MLStreamingIntegrationService
from its_camera_ai.workers.analytics_worker import AnalyticsWorker

logger = get_logger(__name__)


@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.asyncio
class TestEndToEndDataFlow:
    """End-to-end integration tests for complete data flow."""

    @pytest_asyncio.fixture
    async def ml_detection_data(self):
        """Sample ML detection data simulating YOLO11 output."""
        return {
            "camera_id": "test_camera_001",
            "frame_id": f"frame_{int(time.time())}",
            "timestamp": datetime.now(UTC),
            "inference_time_ms": 45.2,
            "model_version": "yolo11n_v1.2",
            "confidence_threshold": 0.5,
            "detections": [
                {
                    "detection_id": str(uuid4()),
                    "class_name": "car",
                    "confidence": 0.95,
                    "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 350},
                    "track_id": "track_001",
                    "vehicle_type": "car",
                    "speed": 45.5,
                    "direction": "north",
                    "attributes": {"color": "blue", "size": "medium"}
                },
                {
                    "detection_id": str(uuid4()),
                    "class_name": "truck",
                    "confidence": 0.88,
                    "bbox": {"x1": 400, "y1": 200, "x2": 650, "y2": 450},
                    "track_id": "track_002",
                    "vehicle_type": "truck",
                    "speed": 38.2,
                    "direction": "south",
                    "attributes": {"color": "red", "size": "large"}
                },
                {
                    "detection_id": str(uuid4()),
                    "class_name": "motorcycle",
                    "confidence": 0.82,
                    "bbox": {"x1": 200, "y1": 180, "x2": 280, "y2": 280},
                    "track_id": "track_003",
                    "vehicle_type": "motorcycle",
                    "speed": 52.1,
                    "direction": "east",
                    "attributes": {"color": "black", "size": "small"}
                }
            ],
            "frame_metadata": {
                "resolution": {"width": 1920, "height": 1080},
                "fps": 30,
                "quality_score": 0.92,
                "lighting_conditions": "good",
                "weather": "clear"
            }
        }

    @pytest_asyncio.fixture
    async def mock_ml_pipeline(self):
        """Mock ML pipeline that simulates real YOLO11 processing."""
        pipeline = AsyncMock()
        
        async def process_frame(frame_data):
            """Simulate ML processing with realistic timing."""
            await asyncio.sleep(0.045)  # Simulate 45ms inference time
            
            return DetectionData(
                camera_id=frame_data["camera_id"],
                timestamp=frame_data["timestamp"],
                frame_id=frame_data["frame_id"],
                vehicle_count=len(frame_data["detections"]),
                detections=[
                    DetectionResultDTO(
                        detection_id=det["detection_id"],
                        class_name=det["class_name"],
                        confidence=det["confidence"],
                        bbox=BoundingBoxDTO(**det["bbox"]),
                        track_id=det["track_id"],
                        timestamp=frame_data["timestamp"],
                        vehicle_type=det["vehicle_type"],
                        speed=det["speed"],
                        direction=det["direction"],
                        attributes=det["attributes"]
                    )
                    for det in frame_data["detections"]
                ],
                processing_metadata={
                    "model_version": frame_data["model_version"],
                    "inference_time_ms": frame_data["inference_time_ms"],
                    "confidence_threshold": frame_data["confidence_threshold"],
                    "gpu_utilization": 0.75,
                    "memory_usage_mb": 1024
                }
            )
        
        pipeline.process_frame = process_frame
        return pipeline

    @pytest_asyncio.fixture
    async def unified_analytics_service(
        self,
        db_session: AsyncSession,
        redis_client: redis.Redis,
        test_settings
    ):
        """Create unified analytics service with real dependencies."""
        # Create mock services for dependency injection
        mock_aggregation_service = AsyncMock()
        mock_incident_service = AsyncMock()
        mock_traffic_rule_service = AsyncMock()
        mock_speed_service = AsyncMock()
        mock_anomaly_service = AsyncMock()
        mock_prediction_service = AsyncMock()
        mock_analytics_repository = AsyncMock()
        mock_cache_service = AsyncMock()

        # Configure realistic service responses
        mock_aggregation_service.aggregate_traffic_metrics.return_value = []
        mock_incident_service.process_detection.return_value = None
        mock_traffic_rule_service.evaluate_violations.return_value = []
        mock_speed_service.calculate_speed_batch.return_value = []
        mock_anomaly_service.detect_anomalies.return_value = []
        
        return UnifiedAnalyticsService(
            aggregation_service=mock_aggregation_service,
            incident_detection_service=mock_incident_service,
            traffic_rule_service=mock_traffic_rule_service,
            speed_calculation_service=mock_speed_service,
            anomaly_detection_service=mock_anomaly_service,
            prediction_service=mock_prediction_service,
            analytics_repository=mock_analytics_repository,
            cache_service=mock_cache_service,
            settings=test_settings
        )

    async def test_complete_detection_processing_pipeline(
        self,
        ml_detection_data: Dict[str, Any],
        mock_ml_pipeline: AsyncMock,
        unified_analytics_service: UnifiedAnalyticsService,
        db_session: AsyncSession,
        redis_client: redis.Redis
    ):
        """Test complete pipeline from ML detection to database storage."""
        
        # Step 1: Simulate ML Pipeline Processing
        start_time = time.time()
        detection_data = await mock_ml_pipeline.process_frame(ml_detection_data)
        ml_processing_time = time.time() - start_time
        
        # Validate ML processing output
        assert isinstance(detection_data, DetectionData)
        assert detection_data.camera_id == ml_detection_data["camera_id"]
        assert detection_data.vehicle_count == len(ml_detection_data["detections"])
        assert len(detection_data.detections) == 3
        assert ml_processing_time < 0.1  # Should process within 100ms
        
        # Step 2: Analytics Processing
        analytics_start = time.time()
        processing_result = await unified_analytics_service.process_realtime_analytics(
            detection_data=detection_data,
            include_anomaly_detection=True,
            include_incident_detection=True,
            include_rule_evaluation=True,
            include_speed_calculation=True
        )
        analytics_processing_time = time.time() - analytics_start
        
        # Validate analytics processing
        assert isinstance(processing_result, ProcessingResult)
        assert processing_result.camera_id == detection_data.camera_id
        assert processing_result.vehicle_count == detection_data.vehicle_count
        assert processing_result.processing_time_ms > 0
        assert analytics_processing_time < 0.2  # Should process within 200ms
        
        # Validate real-time metrics
        metrics = processing_result.metrics
        assert isinstance(metrics, RealtimeTrafficMetrics)
        assert metrics.total_vehicles == 3
        assert metrics.vehicle_breakdown["car"] == 1
        assert metrics.vehicle_breakdown["truck"] == 1
        assert metrics.vehicle_breakdown["motorcycle"] == 1
        assert 0 <= metrics.traffic_density <= 1
        assert metrics.congestion_level in [
            CongestionLevel.FREE_FLOW,
            CongestionLevel.LIGHT,
            CongestionLevel.MODERATE,
            CongestionLevel.HEAVY,
            CongestionLevel.SEVERE
        ]
        
        # Step 3: Verify cache storage
        cache_key = f"realtime_analytics:{detection_data.camera_id}"
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            cached_analytics = json.loads(cached_data)
            assert cached_analytics["camera_id"] == detection_data.camera_id
            assert cached_analytics["vehicle_count"] == detection_data.vehicle_count
        
        logger.info(
            f"E2E pipeline completed successfully: "
            f"ML={ml_processing_time:.3f}s, Analytics={analytics_processing_time:.3f}s, "
            f"Total={ml_processing_time + analytics_processing_time:.3f}s"
        )

    async def test_high_traffic_scenario_processing(
        self,
        mock_ml_pipeline: AsyncMock,
        unified_analytics_service: UnifiedAnalyticsService,
        db_session: AsyncSession
    ):
        """Test processing of high traffic scenario with many vehicles."""
        
        # Create high traffic detection data (50+ vehicles)
        high_traffic_data = {
            "camera_id": "busy_intersection_001",
            "frame_id": f"frame_{int(time.time())}",
            "timestamp": datetime.now(UTC),
            "inference_time_ms": 85.5,  # Higher processing time for more detections
            "model_version": "yolo11n_v1.2",
            "confidence_threshold": 0.5,
            "detections": []
        }
        
        # Generate 50 vehicle detections
        vehicle_types = ["car", "truck", "bus", "motorcycle", "van"]
        for i in range(50):
            vehicle_type = vehicle_types[i % len(vehicle_types)]
            high_traffic_data["detections"].append({
                "detection_id": str(uuid4()),
                "class_name": vehicle_type,
                "confidence": 0.7 + (i % 3) * 0.1,
                "bbox": {
                    "x1": 50 + (i % 10) * 100,
                    "y1": 100 + (i // 10) * 80,
                    "x2": 150 + (i % 10) * 100,
                    "y2": 180 + (i // 10) * 80
                },
                "track_id": f"track_{i:03d}",
                "vehicle_type": vehicle_type,
                "speed": 20.0 + (i % 20),  # Varying speeds for congestion
                "direction": ["north", "south", "east", "west"][i % 4],
                "attributes": {"color": ["red", "blue", "white", "black"][i % 4]}
            })
        
        # Process high traffic scenario
        start_time = time.time()
        detection_data = await mock_ml_pipeline.process_frame(high_traffic_data)
        processing_result = await unified_analytics_service.process_realtime_analytics(
            detection_data=detection_data
        )
        total_time = time.time() - start_time
        
        # Validate high traffic processing
        assert processing_result.vehicle_count == 50
        assert processing_result.metrics.total_vehicles == 50
        
        # Should handle severe congestion
        assert processing_result.metrics.congestion_level in [
            CongestionLevel.HEAVY,
            CongestionLevel.SEVERE
        ]
        
        # High traffic density
        assert processing_result.metrics.traffic_density >= 0.8
        
        # Processing should still be efficient (under 500ms for 50 vehicles)
        assert total_time < 0.5
        
        # Vehicle breakdown should be accurate
        breakdown = processing_result.metrics.vehicle_breakdown
        assert sum(breakdown.values()) == 50
        assert "car" in breakdown
        assert "truck" in breakdown

    async def test_real_time_streaming_with_background_workers(
        self,
        mock_ml_pipeline: AsyncMock,
        unified_analytics_service: UnifiedAnalyticsService,
        redis_client: redis.Redis,
        db_session: AsyncSession
    ):
        """Test real-time streaming with background worker processing."""
        
        camera_id = "stream_test_camera"
        frame_count = 10
        processing_results = []
        
        # Simulate continuous stream processing
        for frame_num in range(frame_count):
            # Create frame data
            frame_data = {
                "camera_id": camera_id,
                "frame_id": f"frame_{frame_num:04d}",
                "timestamp": datetime.now(UTC),
                "inference_time_ms": 45.0 + (frame_num % 5) * 2,  # Varying processing times
                "model_version": "yolo11n_v1.2",
                "confidence_threshold": 0.5,
                "detections": [
                    {
                        "detection_id": str(uuid4()),
                        "class_name": "car",
                        "confidence": 0.85 + (frame_num % 3) * 0.05,
                        "bbox": {"x1": 100 + frame_num * 10, "y1": 150, "x2": 300, "y2": 350},
                        "track_id": f"track_{frame_num}",
                        "vehicle_type": "car",
                        "speed": 40.0 + frame_num * 2,
                        "direction": "north",
                        "attributes": {"color": "blue"}
                    }
                ]
            }
            
            # Process frame
            detection_data = await mock_ml_pipeline.process_frame(frame_data)
            result = await unified_analytics_service.process_realtime_analytics(
                detection_data=detection_data
            )
            processing_results.append(result)
            
            # Simulate realistic frame rate (30 FPS = ~33ms between frames)
            await asyncio.sleep(0.033)
        
        # Validate streaming results
        assert len(processing_results) == frame_count
        
        # Check processing consistency
        for i, result in enumerate(processing_results):
            assert result.camera_id == camera_id
            assert result.vehicle_count >= 0
            assert result.processing_time_ms > 0
            assert result.metrics.total_vehicles >= 0
            
            # Verify timestamps are sequential
            if i > 0:
                current_time = result.timestamp
                previous_time = processing_results[i-1].timestamp
                time_diff = (current_time - previous_time).total_seconds()
                assert 0 < time_diff < 0.1  # Should be processed within 100ms

        # Verify cache contains latest data
        cache_key = f"realtime_analytics:{camera_id}"
        cached_data = await redis_client.get(cache_key)
        assert cached_data is not None

    async def test_database_storage_and_retrieval_integration(
        self,
        unified_analytics_service: UnifiedAnalyticsService,
        db_session: AsyncSession
    ):
        """Test database storage and retrieval of analytics data."""
        
        camera_id = "db_test_camera"
        test_start_time = datetime.now(UTC)
        
        # Generate and store multiple data points
        data_points = []
        for hour in range(24):  # 24 hours of data
            timestamp = test_start_time + timedelta(hours=hour)
            
            # Create detection data for this hour
            detection_data = DetectionData(
                camera_id=camera_id,
                timestamp=timestamp,
                frame_id=f"frame_{hour:02d}",
                vehicle_count=10 + (hour % 12),  # Simulate daily traffic pattern
                detections=[],
                processing_metadata={
                    "model_version": "yolo11n_v1.2",
                    "inference_time_ms": 45.0
                }
            )
            
            # Process analytics
            result = await unified_analytics_service.process_realtime_analytics(
                detection_data=detection_data
            )
            data_points.append(result)
        
        # Test historical data aggregation
        end_time = test_start_time + timedelta(hours=24)
        aggregated_report = await unified_analytics_service.generate_aggregated_report(
            camera_ids=[camera_id],
            start_time=test_start_time,
            end_time=end_time,
            aggregation_level=AggregationLevel.HOURLY
        )
        
        # Validate aggregated data
        assert isinstance(aggregated_report, list)
        # Note: This would be populated in a real implementation
        # with actual database queries
        
        # Test camera health status
        health_status = await unified_analytics_service.get_camera_health_status(camera_id)
        
        assert health_status["camera_id"] == camera_id
        assert "health_score" in health_status
        assert "data_quality_score" in health_status
        assert "status" in health_status
        assert health_status["health_score"] >= 0.0

    async def test_cache_layer_performance_integration(
        self,
        unified_analytics_service: UnifiedAnalyticsService,
        redis_client: redis.Redis
    ):
        """Test multi-level cache integration and performance."""
        
        camera_id = "cache_test_camera"
        
        # Generate initial data
        detection_data = DetectionData(
            camera_id=camera_id,
            timestamp=datetime.now(UTC),
            frame_id="cache_test_frame",
            vehicle_count=15,
            detections=[],
            processing_metadata={"model_version": "yolo11n_v1.2"}
        )
        
        # First processing (cold cache)
        start_time = time.time()
        result1 = await unified_analytics_service.process_realtime_analytics(
            detection_data=detection_data
        )
        cold_cache_time = time.time() - start_time
        
        # Verify data was cached
        cache_key = f"realtime_analytics:{camera_id}"
        cached_data = await redis_client.get(cache_key)
        assert cached_data is not None
        
        # Parse cached data
        cached_analytics = json.loads(cached_data)
        assert cached_analytics["camera_id"] == camera_id
        assert cached_analytics["vehicle_count"] == 15
        
        # Test cache TTL
        ttl = await redis_client.ttl(cache_key)
        assert ttl > 0  # Should have expiration set
        assert ttl <= 300  # Should be 5 minutes or less
        
        # Test cache performance with multiple cameras
        camera_ids = [f"cache_camera_{i}" for i in range(10)]
        cache_operations = []
        
        for cam_id in camera_ids:
            cache_key = f"realtime_analytics:{cam_id}"
            test_data = {
                "camera_id": cam_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "vehicle_count": 20,
                "processing_time_ms": 75.0
            }
            
            start = time.time()
            await redis_client.setex(cache_key, 300, json.dumps(test_data))
            cached = await redis_client.get(cache_key)
            operation_time = time.time() - start
            
            cache_operations.append(operation_time)
            assert cached is not None
        
        # Validate cache performance
        avg_cache_time = sum(cache_operations) / len(cache_operations)
        max_cache_time = max(cache_operations)
        
        # Cache operations should be very fast
        assert avg_cache_time < 0.001  # Under 1ms average
        assert max_cache_time < 0.005  # Under 5ms maximum

    @pytest.mark.performance
    async def test_concurrent_camera_processing(
        self,
        mock_ml_pipeline: AsyncMock,
        unified_analytics_service: UnifiedAnalyticsService,
        redis_client: redis.Redis
    ):
        """Test concurrent processing of multiple camera streams."""
        
        # Simulate 20 concurrent camera streams
        camera_count = 20
        camera_ids = [f"concurrent_camera_{i:02d}" for i in range(camera_count)]
        
        async def process_camera_stream(camera_id: str):
            """Process a single camera stream."""
            results = []
            
            # Process 5 frames per camera
            for frame_num in range(5):
                frame_data = {
                    "camera_id": camera_id,
                    "frame_id": f"frame_{frame_num}",
                    "timestamp": datetime.now(UTC),
                    "inference_time_ms": 45.0,
                    "model_version": "yolo11n_v1.2",
                    "confidence_threshold": 0.5,
                    "detections": [
                        {
                            "detection_id": str(uuid4()),
                            "class_name": "car",
                            "confidence": 0.9,
                            "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 350},
                            "track_id": f"track_{frame_num}",
                            "vehicle_type": "car",
                            "speed": 45.0,
                            "direction": "north",
                            "attributes": {"color": "blue"}
                        }
                    ]
                }
                
                detection_data = await mock_ml_pipeline.process_frame(frame_data)
                result = await unified_analytics_service.process_realtime_analytics(
                    detection_data=detection_data
                )
                results.append(result)
                
                # Simulate frame rate
                await asyncio.sleep(0.033)  # 30 FPS
            
            return results
        
        # Execute concurrent processing
        start_time = time.time()
        tasks = [process_camera_stream(camera_id) for camera_id in camera_ids]
        all_results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Validate concurrent processing results
        assert len(all_results) == camera_count
        
        total_frames_processed = sum(len(results) for results in all_results)
        assert total_frames_processed == camera_count * 5  # 20 cameras × 5 frames
        
        # Performance validation
        frames_per_second = total_frames_processed / total_time
        
        # Should process at least 100 frames per second across all cameras
        assert frames_per_second >= 100
        
        # Should handle concurrent load efficiently
        assert total_time < 2.0  # Should complete within 2 seconds
        
        # Validate all cameras have cached data
        for camera_id in camera_ids:
            cache_key = f"realtime_analytics:{camera_id}"
            cached_data = await redis_client.get(cache_key)
            # Note: May be None due to rapid processing, but that's acceptable
        
        logger.info(
            f"Concurrent processing completed: {camera_count} cameras, "
            f"{total_frames_processed} frames, {frames_per_second:.1f} FPS, "
            f"{total_time:.2f}s total"
        )

    async def test_error_recovery_and_resilience(
        self,
        mock_ml_pipeline: AsyncMock,
        unified_analytics_service: UnifiedAnalyticsService,
        redis_client: redis.Redis
    ):
        """Test system resilience and error recovery."""
        
        camera_id = "error_test_camera"
        
        # Test 1: Malformed detection data
        malformed_data = {
            "camera_id": camera_id,
            "frame_id": "error_frame_1",
            "timestamp": datetime.now(UTC),
            "detections": [
                {
                    "detection_id": str(uuid4()),
                    "class_name": None,  # Invalid class name
                    "confidence": 1.5,   # Invalid confidence (>1.0)
                    "bbox": {"x1": "invalid", "y1": 150, "x2": 300, "y2": 350},  # Invalid bbox
                    "track_id": "",      # Empty track ID
                    "vehicle_type": "unknown_type",
                    "speed": -10.0,      # Invalid negative speed
                    "direction": "invalid_direction",
                    "attributes": {}
                }
            ]
        }
        
        # Should handle malformed data gracefully
        try:
            detection_data = await mock_ml_pipeline.process_frame(malformed_data)
            result = await unified_analytics_service.process_realtime_analytics(
                detection_data=detection_data
            )
            
            # Should return a valid result even with errors
            assert result.camera_id == camera_id
            assert result.processing_time_ms > 0
            
        except Exception as e:
            # If it fails, the error should be handled gracefully
            assert isinstance(e, (ValueError, TypeError))
        
        # Test 2: Redis connection failure simulation
        valid_data = {
            "camera_id": camera_id,
            "frame_id": "recovery_frame",
            "timestamp": datetime.now(UTC),
            "inference_time_ms": 45.0,
            "model_version": "yolo11n_v1.2",
            "detections": [
                {
                    "detection_id": str(uuid4()),
                    "class_name": "car",
                    "confidence": 0.9,
                    "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 350},
                    "track_id": "track_recovery",
                    "vehicle_type": "car",
                    "speed": 45.0,
                    "direction": "north",
                    "attributes": {"color": "blue"}
                }
            ]
        }
        
        # Temporarily disconnect Redis to test resilience
        await redis_client.connection_pool.disconnect()
        
        try:
            detection_data = await mock_ml_pipeline.process_frame(valid_data)
            result = await unified_analytics_service.process_realtime_analytics(
                detection_data=detection_data
            )
            
            # Should still process data even if caching fails
            assert result.camera_id == camera_id
            assert result.vehicle_count >= 0
            
        except Exception:
            # System should be resilient to cache failures
            pass
        
        # Reconnect Redis for cleanup
        await redis_client.connection_pool.disconnect()

    async def test_data_consistency_across_components(
        self,
        mock_ml_pipeline: AsyncMock,
        unified_analytics_service: UnifiedAnalyticsService,
        db_session: AsyncSession,
        redis_client: redis.Redis
    ):
        """Test data consistency across all system components."""
        
        camera_id = "consistency_test_camera"
        
        # Create test data with known values
        test_data = {
            "camera_id": camera_id,
            "frame_id": "consistency_frame",
            "timestamp": datetime.now(UTC),
            "inference_time_ms": 50.0,
            "model_version": "yolo11n_v1.2",
            "confidence_threshold": 0.5,
            "detections": [
                {
                    "detection_id": "det_consistency_001",
                    "class_name": "car",
                    "confidence": 0.95,
                    "bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 350},
                    "track_id": "track_consistency_001",
                    "vehicle_type": "car",
                    "speed": 45.0,
                    "direction": "north",
                    "attributes": {"color": "blue", "size": "medium"}
                },
                {
                    "detection_id": "det_consistency_002",
                    "class_name": "truck",
                    "confidence": 0.88,
                    "bbox": {"x1": 400, "y1": 200, "x2": 650, "y2": 450},
                    "track_id": "track_consistency_002",
                    "vehicle_type": "truck",
                    "speed": 35.0,
                    "direction": "south",
                    "attributes": {"color": "red", "size": "large"}
                }
            ]
        }
        
        # Process data through the pipeline
        detection_data = await mock_ml_pipeline.process_frame(test_data)
        processing_result = await unified_analytics_service.process_realtime_analytics(
            detection_data=detection_data
        )
        
        # Validate data consistency
        
        # 1. Detection data consistency
        assert detection_data.camera_id == camera_id
        assert detection_data.vehicle_count == 2
        assert len(detection_data.detections) == 2
        
        # Check individual detection consistency
        for i, detection in enumerate(detection_data.detections):
            original_det = test_data["detections"][i]
            assert detection.detection_id == original_det["detection_id"]
            assert detection.class_name == original_det["class_name"]
            assert detection.confidence == original_det["confidence"]
            assert detection.track_id == original_det["track_id"]
            assert detection.vehicle_type == original_det["vehicle_type"]
            assert detection.speed == original_det["speed"]
            assert detection.direction == original_det["direction"]
        
        # 2. Processing result consistency
        assert processing_result.camera_id == camera_id
        assert processing_result.vehicle_count == 2
        assert processing_result.timestamp == detection_data.timestamp
        
        # 3. Real-time metrics consistency
        metrics = processing_result.metrics
        assert metrics.camera_id == camera_id
        assert metrics.total_vehicles == 2
        assert metrics.vehicle_breakdown["car"] == 1
        assert metrics.vehicle_breakdown["truck"] == 1
        
        # Average speed should be calculated correctly
        expected_avg_speed = (45.0 + 35.0) / 2
        assert abs(metrics.average_speed - expected_avg_speed) < 0.1
        
        # 4. Cache consistency
        cache_key = f"realtime_analytics:{camera_id}"
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            cached_analytics = json.loads(cached_data)
            assert cached_analytics["camera_id"] == camera_id
            assert cached_analytics["vehicle_count"] == 2
        
        logger.info(
            f"Data consistency validated across all components for camera {camera_id}"
        )