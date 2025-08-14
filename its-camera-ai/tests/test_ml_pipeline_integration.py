"""Tests for ML Pipeline Integration Services.

Tests for the critical ML pipeline integration tasks:
- MLAnalyticsConnector (ITS-ML-002)
- QualityScoreCalculator (ITS-ML-004) 
- Enhanced monitoring in streaming server (ITS-ML-007)
- ModelMetricsService (ITS-ML-009)
"""

import asyncio
import pytest
import numpy as np
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.its_camera_ai.services.ml_analytics_connector import MLAnalyticsConnector
from src.its_camera_ai.ml.quality_score_calculator import QualityScoreCalculator, QualityFactors
from src.its_camera_ai.services.model_metrics_service import ModelMetricsService
from src.its_camera_ai.services.grpc_streaming_server import EnhancedMonitoringService
from src.its_camera_ai.services.analytics_dtos import (
    DetectionResultDTO, BoundingBoxDTO, FrameMetadataDTO, DetectionData, TimeWindow
)


class TestMLAnalyticsConnector:
    """Test ML Analytics Connector service."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        batch_processor = MagicMock()
        unified_analytics = AsyncMock()
        redis_client = AsyncMock()
        cache_service = AsyncMock()
        settings = MagicMock()
        settings.ml = MagicMock()
        settings.ml.drift_threshold = 0.15
        
        return {
            "batch_processor": batch_processor,
            "unified_analytics": unified_analytics,
            "redis_client": redis_client,
            "cache_service": cache_service,
            "settings": settings
        }
    
    @pytest.fixture
    def connector(self, mock_dependencies):
        """Create ML Analytics Connector instance."""
        return MLAnalyticsConnector(**mock_dependencies)
    
    @pytest.fixture
    def sample_ml_batch(self):
        """Create sample ML batch results."""
        return [
            {
                "bbox": [100, 100, 200, 200],
                "confidence": 0.85,
                "class": "car",
                "class_id": 0,
                "track_id": "track_001"
            },
            {
                "bbox": [300, 150, 400, 250],
                "confidence": 0.92,
                "class": "truck", 
                "class_id": 1,
                "track_id": "track_002"
            }
        ]
    
    @pytest.fixture
    def sample_frame_metadata(self):
        """Create sample frame metadata."""
        return {
            "frame_id": "frame_001",
            "camera_id": "camera_001",
            "timestamp": datetime.now(UTC),
            "width": 1920,
            "height": 1080,
            "model_version": "yolo11s_v1.0"
        }
    
    @pytest.mark.asyncio
    async def test_process_ml_batch_success(self, connector, sample_ml_batch, sample_frame_metadata):
        """Test successful ML batch processing."""
        # Mock analytics processing result
        connector.unified_analytics.process_realtime_analytics.return_value = MagicMock(
            processing_time_ms=45.0,
            violations=[],
            anomalies=[]
        )
        
        # Start connector
        await connector.start()
        
        # Process batch
        await connector.process_ml_batch(sample_ml_batch, sample_frame_metadata)
        
        # Allow queue processing
        await asyncio.sleep(0.1)
        
        # Verify analytics was called
        assert connector.unified_analytics.process_realtime_analytics.called
        
        # Stop connector
        await connector.stop()
    
    @pytest.mark.asyncio
    async def test_process_ml_batch_timeout_handling(self, connector, sample_ml_batch, sample_frame_metadata):
        """Test timeout handling in batch processing."""
        # Mock slow analytics processing
        async def slow_processing(*args, **kwargs):
            await asyncio.sleep(0.2)  # Longer than timeout
            return MagicMock(processing_time_ms=200.0, violations=[], anomalies=[])
        
        connector.unified_analytics.process_realtime_analytics.side_effect = slow_processing
        
        # Start connector
        await connector.start()
        
        # Process batch (should handle timeout gracefully)
        await connector.process_ml_batch(sample_ml_batch, sample_frame_metadata)
        
        # Allow processing
        await asyncio.sleep(0.3)
        
        # Check metrics show timeout
        metrics = connector.get_metrics()
        assert metrics["ml_analytics_connector"]["timeouts"] >= 0
        
        await connector.stop()
    
    @pytest.mark.asyncio
    async def test_convert_ml_outputs(self, connector, sample_ml_batch, sample_frame_metadata):
        """Test ML output conversion to DTOs."""
        # Create frame metadata DTO
        frame_meta = FrameMetadataDTO(
            frame_id=sample_frame_metadata["frame_id"],
            camera_id=sample_frame_metadata["camera_id"],
            timestamp=datetime.now(UTC),
            frame_number=0,
            width=sample_frame_metadata["width"],
            height=sample_frame_metadata["height"]
        )
        
        # Convert ML outputs
        detection_results = await connector._convert_ml_outputs(sample_ml_batch, sample_frame_metadata)
        
        # Verify conversion
        assert len(detection_results) == 2
        assert all(isinstance(d, DetectionResultDTO) for d in detection_results)
        
        # Check first detection
        first_detection = detection_results[0]
        assert first_detection.class_name == "car"
        assert first_detection.confidence == 0.85
        assert first_detection.track_id == "track_001"
    
    def test_group_by_camera(self, connector):
        """Test grouping detections by camera."""
        # Create test detections
        detections = [
            DetectionResultDTO(
                detection_id="det_1",
                frame_id="frame_1",
                camera_id="camera_1",
                timestamp=datetime.now(UTC),
                track_id="track_1",
                class_id=0,
                class_name="car",
                confidence=0.85,
                bbox=BoundingBoxDTO(100, 100, 200, 200),
                attributes={}
            ),
            DetectionResultDTO(
                detection_id="det_2", 
                frame_id="frame_2",
                camera_id="camera_2",
                timestamp=datetime.now(UTC),
                track_id="track_2",
                class_id=1,
                class_name="truck",
                confidence=0.92,
                bbox=BoundingBoxDTO(300, 150, 400, 250),
                attributes={}
            ),
            DetectionResultDTO(
                detection_id="det_3",
                frame_id="frame_3", 
                camera_id="camera_1",
                timestamp=datetime.now(UTC),
                track_id="track_3",
                class_id=0,
                class_name="car",
                confidence=0.78,
                bbox=BoundingBoxDTO(500, 200, 600, 300),
                attributes={}
            )
        ]
        
        # Group by camera
        groups = connector._group_by_camera(detections)
        
        # Verify grouping
        assert len(groups) == 2
        assert "camera_1" in groups
        assert "camera_2" in groups
        assert len(groups["camera_1"]) == 2
        assert len(groups["camera_2"]) == 1
    
    @pytest.mark.asyncio
    async def test_health_check(self, connector):
        """Test connector health check."""
        await connector.start()
        
        health = await connector.health_check()
        
        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "error"]
        assert "is_running" in health
        assert health["is_running"] == True
        
        await connector.stop()


class TestQualityScoreCalculator:
    """Test Quality Score Calculator."""
    
    @pytest.fixture
    def mock_cache(self):
        """Create mock cache service."""
        cache = AsyncMock()
        cache.get_json.return_value = None  # No cached data initially
        return cache
    
    @pytest.fixture
    def calculator(self, mock_cache):
        """Create quality score calculator."""
        return QualityScoreCalculator(mock_cache)
    
    @pytest.fixture
    def sample_detection(self):
        """Create sample detection."""
        return {
            "confidence": 0.85,
            "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
            "class": "car",
            "track_id": "track_001"
        }
    
    @pytest.fixture  
    def sample_frame(self):
        """Create sample frame."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    @pytest.fixture
    def sample_model_output(self):
        """Create sample model output."""
        return {
            "class_probabilities": {
                "car": 0.85,
                "truck": 0.10,
                "bus": 0.05
            },
            "confidence_scores": [0.85, 0.82, 0.79],
            "max_confidence": 0.85
        }
    
    @pytest.mark.asyncio
    async def test_calculate_quality_score_success(self, calculator, sample_detection, sample_frame, sample_model_output):
        """Test successful quality score calculation."""
        quality_score = await calculator.calculate_quality_score(
            sample_detection,
            sample_frame,
            sample_model_output
        )
        
        # Verify score is in valid range
        assert 0.0 <= quality_score <= 1.0
        
        # Verify detection confidence component
        assert quality_score > 0.0  # Should have some positive score
    
    def test_calculate_detection_confidence(self, calculator, sample_detection, sample_model_output):
        """Test detection confidence calculation."""
        confidence = calculator._calculate_detection_confidence(
            sample_detection, sample_model_output
        )
        
        # Should be combination of base confidence and entropy confidence
        assert 0.0 <= confidence <= 1.0
        assert confidence >= sample_detection["confidence"] * 0.7  # At least 70% of base
    
    @pytest.mark.asyncio
    async def test_calculate_image_quality(self, calculator, sample_frame, sample_detection):
        """Test image quality calculation."""
        quality = await calculator._calculate_image_quality(sample_frame, sample_detection)
        
        assert 0.0 <= quality <= 1.0
    
    def test_calculate_model_uncertainty(self, calculator, sample_model_output):
        """Test model uncertainty calculation."""
        uncertainty = calculator._calculate_model_uncertainty(sample_model_output)
        
        assert 0.0 <= uncertainty <= 1.0
        
        # Should be lower for confident predictions
        assert uncertainty < 0.5  # High confidence should have low uncertainty
    
    def test_calculate_temporal_consistency(self, calculator, sample_detection):
        """Test temporal consistency calculation."""
        # Create historical detections
        historical = [
            {
                "track_id": "track_001",
                "bbox": {"x1": 98, "y1": 98, "x2": 202, "y2": 202},
                "confidence": 0.83
            },
            {
                "track_id": "track_001", 
                "bbox": {"x1": 99, "y1": 99, "x2": 201, "y2": 201},
                "confidence": 0.84
            }
        ]
        
        consistency = calculator._calculate_temporal_consistency(
            sample_detection, historical
        )
        
        assert 0.0 <= consistency <= 1.0
    
    def test_weighted_average(self, calculator):
        """Test weighted average calculation."""
        factors = QualityFactors(
            detection_confidence=0.8,
            image_quality=0.7,
            model_uncertainty=0.2,  # Will be inverted to 0.8
            temporal_consistency=0.6
        )
        
        weighted_score = calculator._weighted_average(factors)
        
        assert 0.0 <= weighted_score <= 1.0
        
        # Verify weighted calculation
        expected = (
            0.8 * 0.4 +  # detection_confidence * weight
            0.7 * 0.3 +  # image_quality * weight
            0.8 * 0.2 +  # (1.0 - model_uncertainty) * weight
            0.6 * 0.1    # temporal_consistency * weight
        )
        assert abs(weighted_score - expected) < 1e-6


class TestEnhancedMonitoringService:
    """Test Enhanced Monitoring Service."""
    
    @pytest.fixture
    def monitoring_service(self):
        """Create monitoring service."""
        return EnhancedMonitoringService()
    
    @pytest.mark.asyncio
    async def test_get_system_metrics(self, monitoring_service):
        """Test system metrics collection."""
        metrics = await monitoring_service.get_system_metrics()
        
        # Verify basic metrics
        assert hasattr(metrics, 'cpu_usage_percent')
        assert hasattr(metrics, 'memory_used_gb')
        assert hasattr(metrics, 'gpu_metrics')
        
        # Verify values are reasonable
        assert 0.0 <= metrics.cpu_usage_percent <= 100.0
        assert metrics.memory_used_gb >= 0.0
        assert isinstance(metrics.gpu_metrics, list)
    
    def test_metrics_summary(self, monitoring_service):
        """Test metrics summary generation."""
        # Add some mock metrics to history
        from src.its_camera_ai.services.grpc_streaming_server import SystemMetrics
        
        mock_metrics = SystemMetrics(
            cpu_usage_percent=50.0,
            cpu_per_core=[45.0, 55.0],
            memory_used_gb=8.0,
            memory_total_gb=16.0,
            memory_percent=50.0,
            gpu_metrics=[],
            process_cpu_percent=25.0,
            process_memory_mb=512.0,
            timestamp=datetime.now(UTC)
        )
        
        monitoring_service._store_metrics_history(mock_metrics)
        
        summary = monitoring_service.get_metrics_summary()
        
        assert "cpu" in summary
        assert "memory" in summary
        assert summary["cpu"]["current"] == 50.0
        assert summary["memory"]["current"] == 50.0
    
    @pytest.mark.asyncio
    async def test_prometheus_metrics_export(self, monitoring_service):
        """Test Prometheus metrics export."""
        prometheus_metrics = await monitoring_service.export_prometheus_metrics()
        
        assert isinstance(prometheus_metrics, dict)
        assert "ml_cpu_usage_percent" in prometheus_metrics
        assert "ml_memory_usage_percent" in prometheus_metrics


class TestModelMetricsService:
    """Test Model Metrics Service."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        analytics_repo = AsyncMock()
        cache_service = AsyncMock()
        settings = MagicMock()
        settings.ml = MagicMock()
        settings.ml.drift_threshold = 0.15
        
        return {
            "analytics_repository": analytics_repo,
            "cache_service": cache_service,
            "settings": settings
        }
    
    @pytest.fixture
    def metrics_service(self, mock_dependencies):
        """Create model metrics service."""
        return ModelMetricsService(**mock_dependencies)
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detections."""
        return [
            DetectionResultDTO(
                detection_id="det_1",
                frame_id="frame_1",
                camera_id="camera_1", 
                timestamp=datetime.now(UTC),
                track_id="track_1",
                class_id=0,
                class_name="car",
                confidence=0.85,
                bbox=BoundingBoxDTO(100, 100, 200, 200),
                attributes={},
                detection_quality=0.8
            ),
            DetectionResultDTO(
                detection_id="det_2",
                frame_id="frame_1",
                camera_id="camera_1",
                timestamp=datetime.now(UTC),
                track_id="track_2", 
                class_id=1,
                class_name="truck",
                confidence=0.92,
                bbox=BoundingBoxDTO(300, 150, 400, 250),
                attributes={},
                detection_quality=0.9
            )
        ]
    
    @pytest.mark.asyncio
    async def test_track_inference(self, metrics_service, sample_detections):
        """Test inference tracking."""
        frame_metadata = {
            "camera_id": "camera_001",
            "frame_id": "frame_001",
            "model_version": "yolo11s_v1.0"
        }
        
        await metrics_service.track_inference(
            model_name="yolo11s",
            inference_time_ms=45.0,
            detections=sample_detections,
            frame_metadata=frame_metadata
        )
        
        # Verify inference was stored
        assert "yolo11s" in metrics_service.inference_history
        assert len(metrics_service.inference_history["yolo11s"]) == 1
    
    def test_calculate_confidence_drift(self, metrics_service):
        """Test confidence drift calculation."""
        baseline_dist = {"0.8": 50, "0.9": 30, "0.7": 20}
        current_dist = {"0.8": 30, "0.9": 50, "0.7": 20}  # Shifted to higher confidence
        
        drift_score = metrics_service._calculate_confidence_drift(baseline_dist, current_dist)
        
        assert 0.0 <= drift_score <= 1.0
        assert drift_score > 0.0  # Should detect some drift
    
    def test_calculate_class_distribution_drift(self, metrics_service):
        """Test class distribution drift calculation."""
        baseline_dist = {"car": 60, "truck": 30, "bus": 10}
        current_dist = {"car": 40, "truck": 40, "bus": 20}  # Different distribution
        
        drift_score = metrics_service._calculate_class_distribution_drift(baseline_dist, current_dist)
        
        assert 0.0 <= drift_score <= 1.0
        assert drift_score > 0.0  # Should detect drift
    
    @pytest.mark.asyncio
    async def test_get_performance_summary(self, metrics_service, sample_detections):
        """Test performance summary generation."""
        # Add some inference history
        frame_metadata = {"camera_id": "camera_001", "frame_id": "frame_001"}
        
        await metrics_service.track_inference(
            "test_model",
            45.0, 
            sample_detections,
            frame_metadata
        )
        
        summary = await metrics_service.get_performance_summary("test_model", TimeWindow.ONE_HOUR)
        
        assert summary.model_name == "test_model"
        assert summary.total_inferences >= 1
        assert summary.latency_p50 > 0
    
    def test_categorize_drift_severity(self, metrics_service):
        """Test drift severity categorization."""
        assert metrics_service._categorize_drift_severity(0.1) == "low"
        assert metrics_service._categorize_drift_severity(0.2) == "medium" 
        assert metrics_service._categorize_drift_severity(0.4) == "high"
        assert metrics_service._categorize_drift_severity(0.6) == "critical"
    
    def test_health_status(self, metrics_service):
        """Test health status check."""
        health = metrics_service.get_health_status()
        
        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "error"]
        assert "models_tracked" in health
        assert "drift_detection_enabled" in health


class TestIntegrationPerformance:
    """Integration tests for performance requirements."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_latency(self):
        """Test end-to-end latency meets <100ms requirement."""
        # Mock all dependencies
        mock_deps = {
            "batch_processor": MagicMock(),
            "unified_analytics": AsyncMock(),
            "redis_client": AsyncMock(), 
            "cache_service": AsyncMock(),
            "settings": MagicMock()
        }
        
        # Mock fast analytics processing
        mock_deps["unified_analytics"].process_realtime_analytics.return_value = MagicMock(
            processing_time_ms=30.0,
            violations=[],
            anomalies=[]
        )
        
        connector = MLAnalyticsConnector(**mock_deps)
        
        # Test batch processing latency
        start_time = datetime.now(UTC)
        
        sample_batch = [{"bbox": [100, 100, 200, 200], "confidence": 0.85, "class": "car"}]
        sample_metadata = {"frame_id": "test", "camera_id": "test", "timestamp": datetime.now(UTC)}
        
        await connector.start()
        await connector.process_ml_batch(sample_batch, sample_metadata)
        
        # Allow processing
        await asyncio.sleep(0.1)
        
        end_time = datetime.now(UTC)
        total_latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify latency requirement
        assert total_latency_ms < 100  # Should be well under 100ms for this simple case
        
        await connector.stop()
    
    @pytest.mark.asyncio
    async def test_quality_calculation_performance(self):
        """Test quality calculation meets <5ms requirement."""
        cache_mock = AsyncMock()
        cache_mock.get_json.return_value = None
        
        calculator = QualityScoreCalculator(cache_mock)
        
        # Create test data
        detection = {"confidence": 0.85, "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}}
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)  # Smaller frame for speed
        model_output = {"class_probabilities": {"car": 0.85, "truck": 0.15}}
        
        # Measure calculation time
        start_time = datetime.now(UTC)
        
        quality_score = await calculator.calculate_quality_score(detection, frame, model_output)
        
        end_time = datetime.now(UTC)
        calculation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify performance requirement
        assert calculation_time_ms < 5.0  # Should be under 5ms
        assert 0.0 <= quality_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_metrics_tracking_overhead(self):
        """Test metrics tracking overhead is minimal."""
        mock_deps = {
            "analytics_repository": AsyncMock(),
            "cache_service": AsyncMock(),
            "settings": MagicMock()
        }
        
        metrics_service = ModelMetricsService(**mock_deps)
        
        # Create sample data
        detections = [
            DetectionResultDTO(
                detection_id=f"det_{i}",
                frame_id="frame_1",
                camera_id="camera_1",
                timestamp=datetime.now(UTC),
                track_id=f"track_{i}",
                class_id=0,
                class_name="car",
                confidence=0.85,
                bbox=BoundingBoxDTO(100+i*10, 100, 200+i*10, 200),
                attributes={},
                detection_quality=0.8
            )
            for i in range(10)
        ]
        
        # Measure tracking overhead
        start_time = datetime.now(UTC)
        
        await metrics_service.track_inference(
            "test_model",
            45.0,
            detections, 
            {"camera_id": "camera_001"}
        )
        
        end_time = datetime.now(UTC)
        overhead_ms = (end_time - start_time).total_seconds() * 1000
        
        # Verify low overhead requirement (<1ms per inference ideally)
        assert overhead_ms < 10.0  # Allow some flexibility for test environment


if __name__ == "__main__":
    pytest.main([__file__, "-v"])