"""Comprehensive tests for MP4 encoder system.

This test suite validates the complete MP4 streaming pipeline including:
- Fragmented MP4 encoding
- Metadata track embedding  
- DASH fragment generation
- MP4 streaming service integration
- Performance and error handling
"""

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.its_camera_ai.services.fragmented_mp4_encoder import (
    AnalyticsMetadata, FragmentConfiguration, FragmentedMP4Encoder, 
    VideoConfiguration, VideoFrame, create_fragmented_mp4_encoder
)
from src.its_camera_ai.services.metadata_track_manager import (
    MetadataTrackConfig, MetadataTrackManager, MetadataTrackType, 
    create_metadata_track_manager
)
from src.its_camera_ai.services.dash_fragment_generator import (
    DashConfiguration, DashFragmentGenerator, DashQualityLevel,
    create_dash_fragment_generator
)
from src.its_camera_ai.services.mp4_streaming_service import (
    MP4StreamingService, StreamingConfiguration, StreamingSession,
    create_mp4_streaming_service
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_video_frame():
    """Create a sample video frame for testing."""
    # Create 480p RGB frame
    frame_data = np.random.randint(0, 255, (480, 854, 3), dtype=np.uint8)
    return frame_data


@pytest.fixture
def sample_analytics_metadata():
    """Create sample analytics metadata."""
    return AnalyticsMetadata(
        timestamp=time.time(),
        camera_id="test_camera_01",
        frame_id="frame_001",
        detections=[
            {
                "bbox": [100, 100, 50, 50],
                "confidence": 0.95,
                "class_id": 0,
                "class_name": "car"
            }
        ],
        traffic_analytics={"vehicle_count": 1, "avg_speed": 45.2},
        system_metrics={"fps": 30.0, "cpu_usage": 15.5}
    )


@pytest.fixture
def video_config():
    """Create video configuration for testing."""
    return VideoConfiguration(
        width=854,
        height=480,
        framerate=30.0,
        codec="h264",
        bitrate="1M",
        preset="ultrafast"  # Fast encoding for tests
    )


@pytest.fixture
def fragment_config(temp_output_dir):
    """Create fragment configuration for testing."""
    return FragmentConfiguration(
        fragment_duration=2.0,  # Short duration for tests
        output_directory=str(temp_output_dir / "fragments"),
        enable_metadata_tracks=True,
        enable_dash_manifest=True
    )


class TestFragmentedMP4Encoder:
    """Test cases for FragmentedMP4Encoder."""
    
    @pytest.mark.asyncio
    async def test_encoder_initialization(self, video_config, fragment_config):
        """Test encoder initialization."""
        encoder = FragmentedMP4Encoder(video_config, fragment_config, "test_camera")
        
        assert encoder.camera_id == "test_camera"
        assert encoder.video_config == video_config
        assert encoder.fragment_config == fragment_config
        assert not encoder.is_running
        assert encoder.fragment_index == 0
        assert encoder.total_frames_processed == 0
    
    @pytest.mark.asyncio
    async def test_encoder_start_stop(self, video_config, fragment_config):
        """Test encoder start/stop functionality."""
        encoder = FragmentedMP4Encoder(video_config, fragment_config, "test_camera")
        
        # Test start
        await encoder.start()
        assert encoder.is_running
        assert encoder.processing_task is not None
        
        # Test stop
        await encoder.stop()
        assert not encoder.is_running
        assert encoder.processing_task.cancelled() or encoder.processing_task.done()
    
    @pytest.mark.asyncio
    @patch('src.its_camera_ai.services.fragmented_mp4_encoder.ffmpeg')
    async def test_frame_processing(self, mock_ffmpeg, video_config, fragment_config, sample_video_frame):
        """Test frame processing and buffering."""
        encoder = FragmentedMP4Encoder(video_config, fragment_config, "test_camera")
        
        # Mock ffmpeg process
        mock_process = MagicMock()
        mock_process.wait.return_value = None
        mock_process.returncode = 0
        mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.run_async.return_value = mock_process
        
        await encoder.start()
        
        # Add frames
        for i in range(5):
            await encoder.add_frame(
                sample_video_frame, 
                time.time() + i * 0.1, 
                f"frame_{i:03d}"
            )
        
        assert encoder.total_frames_processed == 5
        assert encoder.fragment_buffer.frame_count <= 5
        
        await encoder.stop()
    
    @pytest.mark.asyncio
    async def test_metadata_addition(self, video_config, fragment_config, sample_analytics_metadata):
        """Test metadata addition to encoder."""
        encoder = FragmentedMP4Encoder(video_config, fragment_config, "test_camera")
        
        await encoder.start()
        
        # Add metadata
        await encoder.add_metadata(sample_analytics_metadata)
        
        # Verify metadata was buffered
        assert len(encoder.fragment_buffer.metadata_queue) == 1
        
        await encoder.stop()
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, video_config, fragment_config):
        """Test performance metrics collection."""
        encoder = FragmentedMP4Encoder(video_config, fragment_config, "test_camera")
        
        await encoder.start()
        
        metrics = encoder.get_performance_metrics()
        
        assert "camera_id" in metrics
        assert "is_running" in metrics
        assert "total_frames_processed" in metrics
        assert "buffer_frame_count" in metrics
        assert metrics["camera_id"] == "test_camera"
        assert metrics["is_running"] is True
        
        await encoder.stop()


class TestMetadataTrackManager:
    """Test cases for MetadataTrackManager."""
    
    def test_manager_initialization(self):
        """Test metadata track manager initialization."""
        track_configs = [
            MetadataTrackConfig(
                track_type=MetadataTrackType.DETECTION_RESULTS,
                enabled=True
            )
        ]
        
        manager = MetadataTrackManager(track_configs)
        
        assert MetadataTrackType.DETECTION_RESULTS in manager.track_configs
        assert MetadataTrackType.DETECTION_RESULTS in manager.metadata_buffers
    
    @pytest.mark.asyncio
    async def test_analytics_metadata_addition(self, sample_analytics_metadata):
        """Test adding analytics metadata to tracks."""
        manager = create_metadata_track_manager([MetadataTrackType.DETECTION_RESULTS])
        
        await manager.add_analytics_metadata(sample_analytics_metadata, 2.0)
        
        # Verify metadata was added
        detection_buffer = manager.metadata_buffers[MetadataTrackType.DETECTION_RESULTS]
        assert len(detection_buffer) == 1
        
        timed_metadata = detection_buffer[0]
        assert timed_metadata.timestamp == sample_analytics_metadata.timestamp
        assert len(timed_metadata.payload) > 0
    
    def test_metadata_encoding_formats(self, sample_analytics_metadata):
        """Test different metadata encoding formats."""
        from src.its_camera_ai.services.metadata_track_manager import MetadataEncoder
        
        encoder = MetadataEncoder()
        
        test_data = {
            "timestamp": sample_analytics_metadata.timestamp,
            "detections": sample_analytics_metadata.detections
        }
        
        # Test JSON encoding
        json_data = encoder.encode_json(test_data, compress=False)
        assert isinstance(json_data, bytes)
        assert len(json_data) > 0
        
        # Test compressed JSON encoding
        compressed_data = encoder.encode_json(test_data, compress=True)
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) <= len(json_data)
        
        # Test binary encoding
        binary_data = encoder.encode_binary(test_data)
        assert isinstance(binary_data, bytes)
        assert len(binary_data) > 0
        
        # Test KLV encoding
        klv_data = encoder.encode_klv(test_data)
        assert isinstance(klv_data, bytes)
        assert len(klv_data) > 0
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        manager = create_metadata_track_manager()
        
        metrics = manager.get_performance_metrics()
        
        assert "tracks_processed" in metrics
        assert "total_metadata_size_bytes" in metrics
        assert "encoding_time_ms" in metrics
    
    def test_metadata_summary(self, sample_analytics_metadata):
        """Test metadata summary generation."""
        manager = create_metadata_track_manager()
        
        # Add some metadata
        asyncio.run(manager.add_analytics_metadata(sample_analytics_metadata, 2.0))
        
        summary = manager.get_metadata_summary()
        
        assert "enabled_tracks" in summary
        assert "buffer_counts" in summary
        assert "total_size_bytes" in summary
        assert summary["total_size_bytes"] > 0


class TestDashFragmentGenerator:
    """Test cases for DashFragmentGenerator."""
    
    def test_generator_initialization(self, temp_output_dir):
        """Test DASH generator initialization."""
        config = DashConfiguration(output_directory=str(temp_output_dir))
        generator = DashFragmentGenerator(config, "test_camera")
        
        assert generator.camera_id == "test_camera"
        assert generator.config == config
        assert not generator.is_initialized
        assert generator.output_dir.exists()
    
    @pytest.mark.asyncio
    @patch('src.its_camera_ai.services.dash_fragment_generator.ffmpeg')
    async def test_initialization_segments(self, mock_ffmpeg, temp_output_dir):
        """Test initialization segment generation."""
        # Mock ffmpeg
        mock_ffmpeg.input.return_value.output.return_value.overwrite_output.return_value.run.return_value = None
        
        config = DashConfiguration(output_directory=str(temp_output_dir))
        generator = DashFragmentGenerator(config, "test_camera")
        
        await generator.initialize_streaming()
        
        assert generator.is_initialized
        
        # Verify initialization segments were "created" (mocked)
        mock_ffmpeg.input.assert_called()
    
    def test_manifest_generation(self, temp_output_dir):
        """Test DASH manifest generation."""
        config = DashConfiguration(output_directory=str(temp_output_dir))
        generator = DashFragmentGenerator(config, "test_camera")
        
        manifest_content = generator.manifest_generator.generate_mpd_manifest()
        
        assert manifest_content.startswith('<?xml version="1.0"')
        assert "MPD" in manifest_content
        assert "Period" in manifest_content
        assert "AdaptationSet" in manifest_content
        assert "Representation" in manifest_content
    
    def test_performance_metrics(self, temp_output_dir):
        """Test performance metrics collection."""
        config = DashConfiguration(output_directory=str(temp_output_dir))
        generator = DashFragmentGenerator(config, "test_camera")
        
        metrics = generator.get_performance_metrics()
        
        assert "camera_id" in metrics
        assert "is_initialized" in metrics
        assert "quality_levels" in metrics
        assert "segments_generated" in metrics
        assert metrics["camera_id"] == "test_camera"
        assert metrics["quality_levels"] == len(config.quality_levels)


class TestMP4StreamingService:
    """Test cases for MP4StreamingService."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test streaming service initialization."""
        service = create_mp4_streaming_service()
        
        assert not service.is_running
        assert len(service.sessions) == 0
        assert service.base_config is not None
    
    @pytest.mark.asyncio
    async def test_service_start_stop(self):
        """Test service start/stop functionality."""
        service = create_mp4_streaming_service()
        
        await service.start()
        assert service.is_running
        assert service.maintenance_task is not None
        assert service.metrics_task is not None
        
        await service.stop()
        assert not service.is_running
    
    @pytest.mark.asyncio
    @patch('src.its_camera_ai.services.mp4_streaming_service.StreamingSession')
    async def test_camera_streaming_management(self, mock_session_class):
        """Test camera streaming start/stop."""
        # Mock streaming session
        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_session.stop = AsyncMock()
        mock_session.is_active = True
        mock_session.session_id = "test_session"
        mock_session_class.return_value = mock_session
        
        service = create_mp4_streaming_service()
        await service.start()
        
        # Test start camera streaming
        result = await service.start_camera_streaming("test_camera")
        assert result is True
        assert "test_camera" in service.sessions
        mock_session.start.assert_called_once()
        
        # Test stop camera streaming
        result = await service.stop_camera_streaming("test_camera")
        assert result is True
        assert "test_camera" not in service.sessions
        mock_session.stop.assert_called_once()
        
        await service.stop()
    
    @pytest.mark.asyncio
    @patch('src.its_camera_ai.services.mp4_streaming_service.StreamingSession')
    async def test_frame_processing(self, mock_session_class, sample_video_frame, sample_analytics_metadata):
        """Test video frame processing."""
        # Mock streaming session
        mock_session = MagicMock()
        mock_session.start = AsyncMock()
        mock_session.stop = AsyncMock()
        mock_session.process_frame = AsyncMock()
        mock_session.is_active = True
        mock_session_class.return_value = mock_session
        
        service = create_mp4_streaming_service()
        await service.start()
        
        # Start camera streaming
        await service.start_camera_streaming("test_camera")
        
        # Process frame
        result = await service.process_camera_frame(
            "test_camera",
            sample_video_frame,
            time.time(),
            "frame_001",
            sample_analytics_metadata
        )
        
        assert result is True
        mock_session.process_frame.assert_called_once()
        
        await service.stop()
    
    def test_streaming_status(self):
        """Test streaming status reporting."""
        service = create_mp4_streaming_service()
        
        # Test service status
        status = service.get_streaming_status()
        
        assert "service_running" in status
        assert "active_cameras" in status
        assert "session_count" in status
        assert "service_metrics" in status
        assert status["service_running"] is False
        assert status["session_count"] == 0
    
    def test_service_metrics(self):
        """Test service metrics collection."""
        service = create_mp4_streaming_service()
        
        metrics = service.get_service_metrics()
        
        assert "active_sessions" in metrics
        assert "total_frames_processed" in metrics
        assert "total_fragments_generated" in metrics
        assert "service_uptime_seconds" in metrics


class TestStreamingSession:
    """Test cases for StreamingSession."""
    
    @pytest.mark.asyncio
    @patch('src.its_camera_ai.services.mp4_streaming_service.create_fragmented_mp4_encoder')
    @patch('src.its_camera_ai.services.mp4_streaming_service.create_metadata_track_manager')
    @patch('src.its_camera_ai.services.mp4_streaming_service.create_dash_fragment_generator')
    async def test_session_lifecycle(self, mock_dash, mock_metadata, mock_encoder):
        """Test streaming session lifecycle."""
        # Mock components
        mock_encoder_instance = MagicMock()
        mock_encoder_instance.start = AsyncMock()
        mock_encoder_instance.stop = AsyncMock()
        mock_encoder.return_value = mock_encoder_instance
        
        mock_metadata_instance = MagicMock()
        mock_metadata.return_value = mock_metadata_instance
        
        mock_dash_instance = MagicMock()
        mock_dash_instance.initialize_streaming = AsyncMock()
        mock_dash.return_value = mock_dash_instance
        
        config = StreamingConfiguration(camera_id="test_camera")
        session = StreamingSession(config)
        
        # Test start
        await session.start()
        assert session.is_active
        mock_encoder_instance.start.assert_called_once()
        mock_dash_instance.initialize_streaming.assert_called_once()
        
        # Test stop
        await session.stop()
        assert not session.is_active
        mock_encoder_instance.stop.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('src.its_camera_ai.services.mp4_streaming_service.create_fragmented_mp4_encoder')
    async def test_frame_processing(self, mock_encoder, sample_video_frame, sample_analytics_metadata):
        """Test frame processing in session."""
        # Mock encoder
        mock_encoder_instance = MagicMock()
        mock_encoder_instance.start = AsyncMock()
        mock_encoder_instance.stop = AsyncMock()
        mock_encoder_instance.add_frame = AsyncMock()
        mock_encoder.return_value = mock_encoder_instance
        
        config = StreamingConfiguration(camera_id="test_camera", enable_metadata_tracks=False)
        session = StreamingSession(config)
        
        await session.start()
        
        # Process frame
        await session.process_frame(
            sample_video_frame,
            time.time(),
            "frame_001",
            sample_analytics_metadata
        )
        
        assert session.frames_processed == 1
        mock_encoder_instance.add_frame.assert_called_once()
        
        await session.stop()


class TestIntegration:
    """Integration tests for the complete MP4 streaming pipeline."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_pipeline(self, temp_output_dir, sample_video_frame, sample_analytics_metadata):
        """Test the complete end-to-end streaming pipeline."""
        # This test would require actual ffmpeg installation
        pytest.skip("Integration test requires ffmpeg installation")
        
        # Configuration
        config = StreamingConfiguration(
            camera_id="integration_test_camera",
            video_config=VideoConfiguration(width=854, height=480, preset="ultrafast"),
            fragment_config=FragmentConfiguration(
                fragment_duration=1.0,
                output_directory=str(temp_output_dir / "integration")
            ),
            enable_dash_streaming=True,
            enable_metadata_tracks=True
        )
        
        # Create and start service
        service = create_mp4_streaming_service()
        await service.start()
        
        try:
            # Start camera streaming
            success = await service.start_camera_streaming("integration_test_camera", config)
            assert success
            
            # Process several frames
            for i in range(10):
                timestamp = time.time() + i * 0.033  # ~30fps
                frame_id = f"frame_{i:03d}"
                
                success = await service.process_camera_frame(
                    "integration_test_camera",
                    sample_video_frame,
                    timestamp,
                    frame_id,
                    sample_analytics_metadata
                )
                assert success
                
                # Small delay to allow processing
                await asyncio.sleep(0.01)
            
            # Allow some time for fragment generation
            await asyncio.sleep(2.0)
            
            # Verify outputs were created
            session = service.sessions["integration_test_camera"]
            if session.mp4_encoder:
                assert session.mp4_encoder.total_frames_processed == 10
            
            # Stop camera streaming
            success = await service.stop_camera_streaming("integration_test_camera")
            assert success
            
        finally:
            await service.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])