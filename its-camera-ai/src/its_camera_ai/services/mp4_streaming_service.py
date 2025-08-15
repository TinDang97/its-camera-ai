"""MP4 Streaming Service Integration for ITS Camera AI.

This module provides a comprehensive integration service that coordinates
fragmented MP4 encoding, metadata embedding, and DASH streaming for the
ITS Camera AI system, creating a unified streaming pipeline.

Features:
- Complete MP4 streaming pipeline coordination
- Integration with Kafka event streaming
- Real-time camera frame processing  
- DASH adaptive streaming with multiple quality levels
- Metadata track embedding with analytics synchronization
- Performance monitoring and health management
- Scalable multi-camera streaming support
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.logging import get_logger
from .dash_fragment_generator import (
    DashConfiguration,
    DashFragmentGenerator,
    create_dash_fragment_generator,
)
from .fragmented_mp4_encoder import (
    AnalyticsMetadata,
    FragmentConfiguration,
    FragmentedMP4Encoder,
    VideoConfiguration,
    create_fragmented_mp4_encoder,
)
from .metadata_track_manager import (
    MetadataTrackManager,
    MetadataTrackType,
    create_metadata_track_manager,
)

logger = get_logger(__name__)


@dataclass
class StreamingConfiguration:
    """Configuration for MP4 streaming service."""

    # Camera settings
    camera_id: str
    enable_streaming: bool = True

    # Video encoding settings
    video_config: VideoConfiguration = field(default_factory=VideoConfiguration)

    # Fragment settings
    fragment_config: FragmentConfiguration = field(default_factory=FragmentConfiguration)

    # DASH streaming settings
    dash_config: DashConfiguration = field(default_factory=DashConfiguration)

    # Metadata settings
    enabled_metadata_tracks: list[MetadataTrackType] = field(default_factory=lambda: [
        MetadataTrackType.DETECTION_RESULTS,
        MetadataTrackType.TRAFFIC_ANALYTICS,
        MetadataTrackType.SYSTEM_METRICS,
    ])

    # Performance settings
    max_concurrent_fragments: int = 3
    fragment_cleanup_interval: float = 300.0  # 5 minutes
    health_check_interval: float = 30.0
    metrics_collection_interval: float = 60.0

    # Output settings
    output_base_directory: str = "/tmp/its-camera-ai/streaming"
    enable_dash_streaming: bool = True
    enable_metadata_tracks: bool = True


class StreamingSession:
    """Represents an active streaming session for a camera."""

    def __init__(self, config: StreamingConfiguration):
        self.config = config
        self.camera_id = config.camera_id
        self.session_id = f"{config.camera_id}_{int(time.time())}"
        self.start_time = time.time()
        self.is_active = False

        # Components
        self.mp4_encoder: FragmentedMP4Encoder | None = None
        self.metadata_manager: MetadataTrackManager | None = None
        self.dash_generator: DashFragmentGenerator | None = None

        # State tracking
        self.frames_processed = 0
        self.fragments_generated = 0
        self.last_frame_time = 0.0
        self.last_fragment_time = 0.0

        # Performance metrics
        self.session_metrics = {
            "session_id": self.session_id,
            "camera_id": self.camera_id,
            "uptime_seconds": 0.0,
            "frames_processed": 0,
            "fragments_generated": 0,
            "average_fps": 0.0,
            "streaming_bitrate_mbps": 0.0,
            "last_activity": time.time(),
        }

    async def start(self) -> None:
        """Start the streaming session."""
        if self.is_active:
            logger.warning("Streaming session already active", session_id=self.session_id)
            return

        try:
            logger.info("Starting streaming session",
                       session_id=self.session_id,
                       camera_id=self.camera_id)

            # Initialize MP4 encoder
            self.mp4_encoder = create_fragmented_mp4_encoder(
                camera_id=self.camera_id,
                video_config=self.config.video_config,
                fragment_config=self.config.fragment_config
            )
            await self.mp4_encoder.start()

            # Initialize metadata manager
            if self.config.enable_metadata_tracks:
                self.metadata_manager = create_metadata_track_manager(
                    enabled_tracks=self.config.enabled_metadata_tracks
                )

            # Initialize DASH generator
            if self.config.enable_dash_streaming:
                self.dash_generator = create_dash_fragment_generator(
                    camera_id=self.camera_id,
                    config=self.config.dash_config
                )
                await self.dash_generator.initialize_streaming()

            self.is_active = True
            self.start_time = time.time()

            logger.info("Streaming session started successfully",
                       session_id=self.session_id)

        except Exception as e:
            logger.error("Failed to start streaming session",
                        session_id=self.session_id, error=str(e))
            await self._cleanup()
            raise

    async def stop(self) -> None:
        """Stop the streaming session."""
        if not self.is_active:
            logger.warning("Streaming session not active", session_id=self.session_id)
            return

        logger.info("Stopping streaming session", session_id=self.session_id)

        self.is_active = False
        await self._cleanup()

        # Update final metrics
        self.session_metrics["uptime_seconds"] = time.time() - self.start_time

        logger.info("Streaming session stopped",
                   session_id=self.session_id,
                   uptime_seconds=self.session_metrics["uptime_seconds"],
                   frames_processed=self.frames_processed,
                   fragments_generated=self.fragments_generated)

    async def process_frame(
        self,
        frame_data: np.ndarray,
        timestamp: float,
        frame_id: str,
        metadata: AnalyticsMetadata | None = None
    ) -> None:
        """Process a video frame for streaming."""
        if not self.is_active or not self.mp4_encoder:
            logger.warning("Cannot process frame - session not active",
                         session_id=self.session_id)
            return

        try:
            # Add frame to encoder
            await self.mp4_encoder.add_frame(frame_data, timestamp, frame_id)

            # Add metadata if provided
            if metadata and self.metadata_manager:
                fragment_duration = self.config.fragment_config.fragment_duration
                await self.metadata_manager.add_analytics_metadata(
                    metadata, fragment_duration
                )

            # Update tracking
            self.frames_processed += 1
            self.last_frame_time = time.time()
            self.session_metrics["last_activity"] = self.last_frame_time

            # Update performance metrics
            self._update_session_metrics()

        except Exception as e:
            logger.error("Failed to process frame",
                        session_id=self.session_id,
                        frame_id=frame_id, error=str(e))
            raise

    async def generate_dash_fragments(self) -> bool:
        """Generate DASH fragments from recent MP4 fragments."""
        if not self.dash_generator or not self.mp4_encoder:
            return False

        try:
            # Get latest fragment info
            fragment_info = self.mp4_encoder.get_fragment_info(self.fragments_generated)

            if not fragment_info:
                return False

            # Generate DASH segments
            segment_infos = await self.dash_generator.generate_fragments(
                fragment_info["file_path"]
            )

            if segment_infos:
                self.fragments_generated += 1
                self.last_fragment_time = time.time()

                logger.debug("DASH fragments generated",
                           session_id=self.session_id,
                           segment_count=len(segment_infos))
                return True

        except Exception as e:
            logger.error("Failed to generate DASH fragments",
                        session_id=self.session_id, error=str(e))

        return False

    def get_session_metrics(self) -> dict[str, Any]:
        """Get current session metrics."""
        if self.is_active:
            self.session_metrics["uptime_seconds"] = time.time() - self.start_time

        # Combine metrics from components
        metrics = self.session_metrics.copy()

        if self.mp4_encoder:
            encoder_metrics = self.mp4_encoder.get_performance_metrics()
            metrics.update({
                "encoder_" + k: v for k, v in encoder_metrics.items()
                if k not in ["camera_id"]
            })

        if self.dash_generator:
            dash_metrics = self.dash_generator.get_performance_metrics()
            metrics.update({
                "dash_" + k: v for k, v in dash_metrics.items()
                if k not in ["camera_id"]
            })

        return metrics

    def _update_session_metrics(self) -> None:
        """Update session performance metrics."""
        current_time = time.time()
        uptime = current_time - self.start_time

        if uptime > 0:
            self.session_metrics["average_fps"] = self.frames_processed / uptime

        self.session_metrics["frames_processed"] = self.frames_processed
        self.session_metrics["fragments_generated"] = self.fragments_generated
        self.session_metrics["uptime_seconds"] = uptime

    async def _cleanup(self) -> None:
        """Cleanup session resources."""
        try:
            if self.mp4_encoder:
                await self.mp4_encoder.stop()

            # Note: DASH generator and metadata manager are stateless
            # and don't require explicit cleanup

        except Exception as e:
            logger.error("Error during session cleanup",
                        session_id=self.session_id, error=str(e))


class MP4StreamingService:
    """Main MP4 streaming service for managing multiple camera streams.
    
    This service coordinates MP4 encoding, metadata embedding, and DASH streaming
    across multiple camera feeds, providing a unified streaming platform.
    """

    def __init__(self, base_config: StreamingConfiguration | None = None):
        """Initialize the MP4 streaming service.
        
        Args:
            base_config: Base configuration for streaming sessions
        """
        self.base_config = base_config or StreamingConfiguration(camera_id="default")

        # Active sessions
        self.sessions: dict[str, StreamingSession] = {}

        # Service state
        self.is_running = False
        self.start_time = 0.0

        # Background tasks
        self.maintenance_task: asyncio.Task | None = None
        self.metrics_task: asyncio.Task | None = None

        # Global metrics
        self.service_metrics = {
            "active_sessions": 0,
            "total_frames_processed": 0,
            "total_fragments_generated": 0,
            "service_uptime_seconds": 0.0,
            "average_sessions_fps": 0.0,
        }

        logger.info("MP4StreamingService initialized")

    async def start(self) -> None:
        """Start the MP4 streaming service."""
        if self.is_running:
            logger.warning("MP4StreamingService already running")
            return

        self.is_running = True
        self.start_time = time.time()

        # Start background tasks
        self.maintenance_task = asyncio.create_task(self._maintenance_loop())
        self.metrics_task = asyncio.create_task(self._metrics_collection_loop())

        logger.info("MP4StreamingService started")

    async def stop(self) -> None:
        """Stop the MP4 streaming service."""
        if not self.is_running:
            logger.warning("MP4StreamingService not running")
            return

        logger.info("Stopping MP4StreamingService")

        self.is_running = False

        # Cancel background tasks
        if self.maintenance_task:
            self.maintenance_task.cancel()
        if self.metrics_task:
            self.metrics_task.cancel()

        # Wait for tasks to complete
        tasks = [task for task in [self.maintenance_task, self.metrics_task] if task]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Stop all active sessions
        stop_tasks = []
        for session in self.sessions.values():
            if session.is_active:
                stop_tasks.append(session.stop())

        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)

        self.sessions.clear()

        logger.info("MP4StreamingService stopped",
                   total_uptime=time.time() - self.start_time)

    async def start_camera_streaming(
        self,
        camera_id: str,
        config: StreamingConfiguration | None = None
    ) -> bool:
        """Start streaming for a specific camera.
        
        Args:
            camera_id: Unique identifier for the camera
            config: Optional configuration override
            
        Returns:
            True if streaming started successfully
        """
        if camera_id in self.sessions:
            logger.warning("Streaming already active for camera", camera_id=camera_id)
            return False

        try:
            # Create configuration
            if config is None:
                config = StreamingConfiguration(
                    camera_id=camera_id,
                    video_config=self.base_config.video_config,
                    fragment_config=self.base_config.fragment_config,
                    dash_config=self.base_config.dash_config,
                    enabled_metadata_tracks=self.base_config.enabled_metadata_tracks,
                )
            else:
                config.camera_id = camera_id

            # Create and start session
            session = StreamingSession(config)
            await session.start()

            self.sessions[camera_id] = session

            logger.info("Camera streaming started",
                       camera_id=camera_id,
                       session_id=session.session_id)

            return True

        except Exception as e:
            logger.error("Failed to start camera streaming",
                        camera_id=camera_id, error=str(e))
            return False

    async def stop_camera_streaming(self, camera_id: str) -> bool:
        """Stop streaming for a specific camera.
        
        Args:
            camera_id: Unique identifier for the camera
            
        Returns:
            True if streaming stopped successfully
        """
        if camera_id not in self.sessions:
            logger.warning("No active streaming for camera", camera_id=camera_id)
            return False

        try:
            session = self.sessions[camera_id]
            await session.stop()
            del self.sessions[camera_id]

            logger.info("Camera streaming stopped", camera_id=camera_id)
            return True

        except Exception as e:
            logger.error("Failed to stop camera streaming",
                        camera_id=camera_id, error=str(e))
            return False

    async def process_camera_frame(
        self,
        camera_id: str,
        frame_data: np.ndarray,
        timestamp: float,
        frame_id: str,
        metadata: AnalyticsMetadata | None = None
    ) -> bool:
        """Process a video frame for a specific camera.
        
        Args:
            camera_id: Unique identifier for the camera
            frame_data: Video frame as numpy array
            timestamp: Frame timestamp in seconds
            frame_id: Unique frame identifier
            metadata: Optional analytics metadata
            
        Returns:
            True if frame processed successfully
        """
        if camera_id not in self.sessions:
            logger.warning("No active streaming session for camera", camera_id=camera_id)
            return False

        try:
            session = self.sessions[camera_id]
            await session.process_frame(frame_data, timestamp, frame_id, metadata)
            return True

        except Exception as e:
            logger.error("Failed to process camera frame",
                        camera_id=camera_id, frame_id=frame_id, error=str(e))
            return False

    def get_streaming_status(self, camera_id: str | None = None) -> dict[str, Any]:
        """Get streaming status for cameras.
        
        Args:
            camera_id: Optional specific camera ID
            
        Returns:
            Streaming status information
        """
        if camera_id:
            # Status for specific camera
            if camera_id in self.sessions:
                session = self.sessions[camera_id]
                return {
                    "camera_id": camera_id,
                    "is_active": session.is_active,
                    "session_id": session.session_id,
                    "metrics": session.get_session_metrics(),
                }
            else:
                return {"camera_id": camera_id, "is_active": False}

        # Status for all cameras
        return {
            "service_running": self.is_running,
            "active_cameras": list(self.sessions.keys()),
            "session_count": len(self.sessions),
            "service_metrics": self.get_service_metrics(),
            "camera_sessions": {
                cam_id: {
                    "is_active": session.is_active,
                    "session_id": session.session_id,
                    "uptime": time.time() - session.start_time,
                }
                for cam_id, session in self.sessions.items()
            }
        }

    def get_service_metrics(self) -> dict[str, Any]:
        """Get overall service metrics."""
        # Update global metrics
        self.service_metrics.update({
            "active_sessions": len(self.sessions),
            "service_uptime_seconds": time.time() - self.start_time if self.is_running else 0.0,
            "total_frames_processed": sum(
                session.frames_processed for session in self.sessions.values()
            ),
            "total_fragments_generated": sum(
                session.fragments_generated for session in self.sessions.values()
            ),
        })

        # Calculate average FPS across sessions
        if self.sessions:
            total_fps = sum(
                session.session_metrics.get("average_fps", 0.0)
                for session in self.sessions.values()
            )
            self.service_metrics["average_sessions_fps"] = total_fps / len(self.sessions)

        return self.service_metrics.copy()

    def get_dash_manifest_url(self, camera_id: str) -> str | None:
        """Get DASH manifest URL for a camera.
        
        Args:
            camera_id: Unique identifier for the camera
            
        Returns:
            DASH manifest URL or None if not available
        """
        if camera_id not in self.sessions:
            return None

        session = self.sessions[camera_id]
        if not session.dash_generator:
            return None

        manifest_path = session.dash_generator.get_manifest_path()
        if manifest_path:
            # Return relative URL path (actual HTTP serving depends on web server setup)
            return f"/streaming/{camera_id}/manifest.mpd"

        return None

    async def _maintenance_loop(self) -> None:
        """Background maintenance loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.base_config.fragment_cleanup_interval)

                # Cleanup old fragments for each session
                for session in self.sessions.values():
                    if session.dash_generator:
                        await asyncio.get_event_loop().run_in_executor(
                            None, session.dash_generator.cleanup_old_segments
                        )

                logger.debug("Maintenance cycle completed",
                           active_sessions=len(self.sessions))

            except Exception as e:
                logger.error("Maintenance loop error", error=str(e))
                await asyncio.sleep(30)  # Wait before retrying

    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        while self.is_running:
            try:
                await asyncio.sleep(self.base_config.metrics_collection_interval)

                # Update service metrics
                metrics = self.get_service_metrics()

                logger.info("Service metrics collected",
                           active_sessions=metrics["active_sessions"],
                           total_frames=metrics["total_frames_processed"],
                           total_fragments=metrics["total_fragments_generated"],
                           average_fps=metrics["average_sessions_fps"])

            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(60)  # Wait before retrying

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Factory function for creating the streaming service
def create_mp4_streaming_service(
    base_config: StreamingConfiguration | None = None
) -> MP4StreamingService:
    """Create an MP4 streaming service with default configuration.
    
    Args:
        base_config: Optional base configuration
        
    Returns:
        Configured MP4StreamingService instance
    """
    return MP4StreamingService(base_config)
