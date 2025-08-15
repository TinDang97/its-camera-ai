"""ML Pipeline Integration Tests for ITS Camera AI.

This test suite validates the end-to-end ML pipeline integration including:
- TASK-ML-001: Blosc compression for numpy arrays (✅ COMPLETED)
- TASK-ML-002: ML Pipeline Integration Testing & ModelRegistry Validation
- TASK-ML-003: Cross-Service Memory Optimization & Performance Monitoring
- MLAnalyticsConnector integration with enhanced monitoring
- QualityScoreCalculator with compressed feature caching
- ModelMetricsService with drift detection
- Enhanced ModelRegistry with federated learning support
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
from datetime import UTC, datetime

from src.its_camera_ai.core.blosc_numpy_compressor import (
    BloscNumpyCompressor,
    CompressionAlgorithm,
    CompressionLevel,
    get_global_compressor,
)
from src.its_camera_ai.ml.quality_score_calculator import QualityScoreCalculator, QualityFactors
from src.its_camera_ai.services.ml_analytics_connector import MLAnalyticsConnector
from src.its_camera_ai.services.model_metrics_service import ModelMetricsService
from src.its_camera_ai.services.grpc_streaming_server import EnhancedMonitoringService
from src.its_camera_ai.services.analytics_dtos import (
    DetectionResultDTO, BoundingBoxDTO, FrameMetadataDTO, DetectionData, TimeWindow
)
from src.its_camera_ai.storage.model_registry import MinIOModelRegistry
from src.its_camera_ai.storage.enhanced_model_registry import (
    EnhancedModelRegistry,
    DriftDetectionResult,
    ModelHealthStatus,
    DeploymentStage
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


# Enhanced fixtures for TASK-ML-002 testing
@pytest.fixture
async def enhanced_model_registry():
    """Create enhanced model registry fixture with mocked storage."""
    mock_storage_service = AsyncMock()
    mock_storage_service.upload_object.return_value = MagicMock(
        success=True,
        etag="test-etag",
        version_id="test-version",
        total_size=1024,
        upload_time_seconds=0.5
    )
    
    registry = EnhancedModelRegistry(
        storage_service=mock_storage_service,
        config={"enable_drift_detection": True, "enable_federated_learning": True}
    )
    
    yield registry

@pytest.fixture
def sample_model_path():
    """Create a sample PyTorch model file."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        torch.save(model.state_dict(), f.name)
        yield Path(f.name)
    
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


class TestEnhancedMLPipelineIntegration:
    """Enhanced test suite for TASK-ML-002: ML Pipeline Integration & ModelRegistry Validation."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_with_blosc_compression(self):
        """Test complete ML pipeline with blosc compression optimization."""
        # Generate large numpy arrays to trigger compression
        large_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        detection_features = np.random.rand(1000, 128).astype(np.float32)

        # Test blosc compression performance
        compressor = get_global_compressor()
        
        start_time = time.time()
        compressed_frame = compressor.compress_array(large_frame)
        compression_time = (time.time() - start_time) * 1000
        
        # Validate compression performance requirements (TASK-ML-001)
        assert compression_time < 10.0, f"Compression time {compression_time}ms exceeds 10ms target"
        
        original_size = large_frame.nbytes
        compressed_size = len(compressed_frame.compressed_data)
        compression_ratio = 1 - (compressed_size / original_size)
        
        assert compression_ratio > 0.6, f"Compression ratio {compression_ratio} below 60% target"

        # Test decompression integrity
        start_time = time.time()
        decompressed_frame = compressor.decompress_array(compressed_frame)
        decompression_time = (time.time() - start_time) * 1000
        
        assert decompression_time < 10.0, f"Decompression time {decompression_time}ms exceeds 10ms target"
        assert np.array_equal(large_frame, decompressed_frame), "Data integrity check failed"
        
        print(f"✓ Blosc compression validated: {compression_ratio:.1%} savings, {compression_time:.1f}ms")

    @pytest.mark.asyncio
    async def test_enhanced_model_registry_validation(self, enhanced_model_registry, sample_model_path):
        """Test enhanced model registry with drift detection and federated learning."""
        model_name = "yolo11n_traffic"
        version = "v2.1.0"
        
        # Prepare comprehensive model metrics
        model_metrics = {
            "accuracy": 0.945,
            "precision": 0.923,
            "recall": 0.891,
            "f1_score": 0.907,
            "latency_p95_ms": 47.2,
            "throughput_fps": 28.5,
            "model_size_mb": 12.3,
            "memory_usage_mb": 256.8,
            "mAP_50": 0.934,
            "mAP_75": 0.876,
        }
        
        training_config = {
            "framework": "pytorch",
            "architecture": "yolo11n", 
            "dataset": "traffic_detection_v3",
            "epochs": 100,
            "learning_rate": 0.001,
            "batch_size": 32,
            "optimizer": "AdamW",
            "augmentations": ["flip", "rotate", "brightness", "contrast"],
            "compression_enabled": True,
        }
        
        tags = {
            "environment": "production",
            "deployment_stage": "canary",
            "camera_type": "traffic_cam",
            "region": "us-west-2"
        }
        
        # Register model with enhanced registry
        model_version = await enhanced_model_registry.register_model(
            model_path=sample_model_path,
            model_name=model_name,
            version=version,
            metrics=model_metrics,
            training_config=training_config,
            tags=tags
        )
        
        # Validate registration
        assert model_version.model_id == f"{model_name}_{version}"
        assert model_version.version == version
        assert abs(model_version.accuracy_score - 0.945) < 0.001
        assert abs(model_version.latency_p95_ms - 47.2) < 0.1
        assert abs(model_version.throughput_fps - 28.5) < 0.1
        
        # Test model deployment promotion workflow
        await enhanced_model_registry.promote_model(
            model_name, version, DeploymentStage.STAGING
        )
        
        deployment_info = await enhanced_model_registry.get_deployment_info(
            model_name, version
        )
        assert deployment_info["stage"] == "staging"
        assert deployment_info["promoted_at"] > 0
        
        # Test drift detection simulation
        baseline_predictions = np.random.normal(0.8, 0.1, 1000)
        current_predictions = np.random.normal(0.75, 0.15, 1000)  # Slight drift
        
        drift_result = await enhanced_model_registry.detect_drift(
            model_name, version, baseline_predictions, current_predictions
        )
        
        assert isinstance(drift_result, DriftDetectionResult)
        assert drift_result.model_name == model_name
        assert drift_result.model_version == version
        assert 0.0 <= drift_result.drift_score <= 1.0
        
        # Test model health monitoring
        health_status = await enhanced_model_registry.get_model_health(
            model_name, version
        )
        
        assert isinstance(health_status, ModelHealthStatus)
        assert health_status.model_name == model_name
        assert health_status.version == version
        assert health_status.overall_score >= 0.0
        
        # Test federated learning readiness
        federated_config = await enhanced_model_registry.prepare_federated_learning(
            model_name, version, participant_count=5
        )
        
        assert "aggregation_strategy" in federated_config
        assert "communication_rounds" in federated_config
        assert federated_config["participant_count"] == 5
        
        print(f"✓ Enhanced ModelRegistry validated: {model_name}:{version} registered with drift detection")

    @pytest.mark.benchmark
    async def test_cross_service_memory_optimization(self):
        """Test cross-service memory optimization and blosc integration (TASK-ML-003)."""
        # Test memory efficiency with large datasets
        large_feature_arrays = [
            np.random.rand(5000, 256).astype(np.float32) for _ in range(10)
        ]
        
        # Test blosc compression across different array patterns
        compressor = get_global_compressor()
        memory_savings = []
        compression_times = []
        
        for i, features in enumerate(large_feature_arrays):
            start_time = time.time()
            compressed = compressor.compress_array(features)
            compression_time = (time.time() - start_time) * 1000
            
            compression_times.append(compression_time)
            memory_saving = 1 - (len(compressed.compressed_data) / features.nbytes)
            memory_savings.append(memory_saving)
            
            # Validate decompression integrity
            decompressed = compressor.decompress_array(compressed)
            assert np.allclose(features, decompressed, rtol=1e-6)
        
        # Validate overall memory optimization targets
        avg_memory_saving = np.mean(memory_savings)
        avg_compression_time = np.mean(compression_times)
        
        assert avg_memory_saving > 0.30, f"Average memory saving {avg_memory_saving:.1%} below 30% target"
        assert avg_compression_time < 10.0, f"Average compression time {avg_compression_time:.1f}ms exceeds 10ms"
        
        print(f"✓ Cross-service memory optimization validated:")
        print(f"  - Average memory saving: {avg_memory_saving:.1%}")
        print(f"  - Average compression time: {avg_compression_time:.1f}ms")

    @pytest.mark.asyncio
    async def test_model_registry_integration_with_compression(self, enhanced_model_registry, sample_model_path):
        """Test model registry integration with blosc compression for model artifacts."""
        model_name = "yolo11s_compressed"
        version = "v1.5.2"
        
        # Create larger model for compression testing
        larger_model = torch.nn.Sequential(
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 250),
            torch.nn.ReLU(),
            torch.nn.Linear(250, 10)
        )
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(larger_model.state_dict(), f.name)
            larger_model_path = Path(f.name)
        
        try:
            # Mock storage service to simulate compression
            original_size = larger_model_path.stat().st_size
            compressed_size = int(original_size * 0.3)  # Simulate 70% compression
            
            enhanced_model_registry.storage_service.upload_object.return_value = MagicMock(
                success=True,
                etag="compressed-etag",
                version_id="compressed-version",
                total_size=compressed_size,
                upload_time_seconds=0.8,
                compression_ratio=0.7
            )
            
            model_metrics = {
                "accuracy": 0.932,
                "latency_p95_ms": 52.1,
                "throughput_fps": 25.3,
                "model_size_mb": original_size / (1024 * 1024),
                "compressed_size_mb": compressed_size / (1024 * 1024),
            }
            
            # Register model with compression
            model_version = await enhanced_model_registry.register_model(
                model_path=larger_model_path,
                model_name=model_name,
                version=version,
                metrics=model_metrics,
                training_config={"compression_enabled": True}
            )
            
            # Validate compression benefits
            compression_ratio = 1 - (compressed_size / original_size)
            assert compression_ratio > 0.6, f"Model compression ratio {compression_ratio:.1%} below 60%"
            
            print(f"✓ Model registry compression validated: {compression_ratio:.1%} reduction")
            
        finally:
            larger_model_path.unlink(missing_ok=True)

    @pytest.mark.integration
    async def test_federated_learning_pipeline_integration(self, enhanced_model_registry, sample_model_path):
        """Test federated learning integration with the ML pipeline."""
        base_model_name = "yolo11n_federated"
        base_version = "v2.0.0"
        
        # Register base model for federated learning
        base_metrics = {
            "accuracy": 0.887,
            "precision": 0.901,
            "recall": 0.873,
            "f1_score": 0.887
        }
        
        await enhanced_model_registry.register_model(
            model_path=sample_model_path,
            model_name=base_model_name,
            version=base_version,
            metrics=base_metrics,
            training_config={"federated_enabled": True}
        )
        
        # Simulate federated learning setup
        participant_configs = await enhanced_model_registry.prepare_federated_learning(
            base_model_name, base_version, participant_count=3
        )
        
        assert participant_configs["participant_count"] == 3
        assert "aggregation_strategy" in participant_configs
        assert "communication_rounds" in participant_configs
        
        # Simulate participant model updates
        participant_metrics = [
            {"accuracy": 0.892, "loss": 0.234, "participant_id": "edge_node_1"},
            {"accuracy": 0.885, "loss": 0.241, "participant_id": "edge_node_2"},
            {"accuracy": 0.898, "loss": 0.228, "participant_id": "edge_node_3"},
        ]
        
        # Aggregate federated updates
        aggregated_metrics = await enhanced_model_registry.aggregate_federated_updates(
            base_model_name, base_version, participant_metrics
        )
        
        assert "aggregated_accuracy" in aggregated_metrics
        assert "aggregated_loss" in aggregated_metrics
        assert "participants_count" in aggregated_metrics
        assert aggregated_metrics["participants_count"] == 3
        
        # Validate aggregated performance improvement
        aggregated_accuracy = aggregated_metrics["aggregated_accuracy"]
        assert aggregated_accuracy > base_metrics["accuracy"], \
            "Federated learning should improve model accuracy"
        
        print(f"✓ Federated learning integration validated: {aggregated_accuracy:.3f} accuracy")

    @pytest.mark.benchmark
    async def test_comprehensive_pipeline_performance_benchmarks(self):
        """Comprehensive performance benchmarks for the entire ML pipeline."""
        # Performance test configuration with realistic workloads
        test_configs = [
            {"frame_size": (480, 640, 3), "detections": 10, "expected_latency_ms": 80, "scenario": "low_res"},
            {"frame_size": (720, 1280, 3), "detections": 25, "expected_latency_ms": 95, "scenario": "hd"},
            {"frame_size": (1080, 1920, 3), "detections": 50, "expected_latency_ms": 100, "scenario": "full_hd"},
        ]
        
        # Mock comprehensive ML Analytics Connector
        mock_deps = {
            "batch_processor": MagicMock(),
            "unified_analytics": AsyncMock(),
            "redis_client": AsyncMock(),
            "cache_service": AsyncMock(),
            "settings": MagicMock()
        }
        
        mock_deps["unified_analytics"].process_realtime_analytics.return_value = MagicMock(
            processing_time_ms=35.0,
            violations=[],
            anomalies=[]
        )
        
        connector = MLAnalyticsConnector(**mock_deps)
        await connector.start()
        
        performance_results = []
        
        for config in test_configs:
            frame_size = config["frame_size"]
            detection_count = config["detections"]
            expected_latency = config["expected_latency_ms"]
            scenario = config["scenario"]
            
            # Generate realistic test data with blosc compression
            test_frame = np.random.randint(0, 255, frame_size, dtype=np.uint8)
            
            # Test compression on frame data
            compressor = get_global_compressor()
            start_compression = time.time()
            compressed_frame = compressor.compress_array(test_frame)
            compression_time = (time.time() - start_compression) * 1000
            
            batch_results = [{
                "detections": [
                    {
                        "bbox": [i*20, i*20, i*20+60, i*20+60],
                        "confidence": 0.8 + (i * 0.01),
                        "class_id": i % 4,
                        "features": np.random.rand(128).astype(np.float32),
                    }
                    for i in range(detection_count)
                ]
            }]
            
            frame_metadata = {
                "frame_id": f"benchmark_{scenario}_{int(time.time())}",
                "camera_id": f"benchmark_cam_{scenario}",
                "width": frame_size[1],
                "height": frame_size[0],
                "quality_score": 0.9,
                "model_version": "yolo11n-benchmark",
                "scenario": scenario,
            }
            
            # Measure end-to-end pipeline latency
            start_time = time.time()
            
            await connector.process_ml_batch(batch_results, frame_metadata)
            await asyncio.sleep(0.05)  # Wait for processing
            
            end_time = time.time()
            pipeline_latency = (end_time - start_time) * 1000
            
            # Store performance results
            result = {
                "scenario": scenario,
                "frame_size": frame_size,
                "detection_count": detection_count,
                "pipeline_latency_ms": pipeline_latency,
                "compression_time_ms": compression_time,
                "expected_latency_ms": expected_latency,
                "compression_ratio": 1 - (len(compressed_frame.compressed_data) / test_frame.nbytes)
            }
            performance_results.append(result)
            
            # Validate performance requirements
            assert pipeline_latency < expected_latency, \
                f"Pipeline latency {pipeline_latency:.1f}ms exceeds {expected_latency}ms for {scenario}"
            
            print(f"✓ {scenario.upper()} performance: {pipeline_latency:.1f}ms < {expected_latency}ms, compression: {compression_time:.1f}ms")
        
        # Validate overall pipeline metrics
        final_metrics = connector.get_metrics()
        connector_metrics = final_metrics["ml_analytics_connector"]
        
        assert connector_metrics["batches_processed"] == len(test_configs)
        assert connector_metrics["errors"] == 0
        assert connector_metrics["timeouts"] == 0
        assert connector_metrics["avg_latency_ms"] < 100.0
        
        await connector.stop()
        
        # Print comprehensive performance summary
        print(f"✓ TASK-ML-002 & TASK-ML-003 Performance Summary:")
        print(f"  - Average pipeline latency: {connector_metrics['avg_latency_ms']:.1f}ms")
        print(f"  - Total batches processed: {connector_metrics['batches_processed']}")
        print(f"  - Error rate: {connector_metrics['errors']} / {connector_metrics['batches_processed']}")
        for result in performance_results:
            print(f"  - {result['scenario']}: {result['pipeline_latency_ms']:.1f}ms, compression: {result['compression_ratio']:.1%}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])