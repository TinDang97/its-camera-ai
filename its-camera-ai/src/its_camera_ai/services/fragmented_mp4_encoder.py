"""Fragmented MP4 Encoder for Real-time Streaming with Metadata Tracks.

This module implements a complete fragmented MP4 encoding system for the ITS Camera AI
project, providing DASH-compatible streaming with embedded analytics metadata.

Features:
- Real-time fragmented MP4 encoding with configurable fragment duration
- Embedded metadata tracks for synchronized analytics data
- DASH/HLS compatible output for adaptive streaming
- PyAV integration for low-level container manipulation
- ffmpeg-python integration for high-level encoding operations
- Memory-efficient streaming processing
- Integration with existing Kafka streaming pipeline
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import ffmpeg
import numpy as np

from ..core.exceptions import ServiceError
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VideoConfiguration:
    """Video encoding configuration for MP4 fragments."""

    width: int = 1920
    height: int = 1080
    framerate: float = 30.0
    codec: str = "h264"
    bitrate: str = "2M"
    pixel_format: str = "yuv420p"
    profile: str = "main"
    preset: str = "faster"
    crf: int = 23


@dataclass
class FragmentConfiguration:
    """Configuration for MP4 fragmentation and metadata."""

    fragment_duration: float = 4.0  # seconds
    segment_duration: float = 6.0   # DASH segment duration
    min_fragment_duration: float = 2.0  # minimum fragment duration
    max_fragment_duration: float = 8.0  # maximum fragment duration
    metadata_sync_tolerance: float = 0.1  # metadata sync tolerance in seconds
    enable_metadata_tracks: bool = True
    enable_dash_manifest: bool = True
    output_directory: str = "/tmp/its-camera-ai/fragments"
    manifest_name: str = "manifest.mpd"


@dataclass
class AnalyticsMetadata:
    """Analytics metadata to embed in MP4 fragments."""

    timestamp: float
    camera_id: str
    frame_id: str
    detections: list[dict[str, Any]] = field(default_factory=list)
    traffic_analytics: dict[str, Any] = field(default_factory=dict)
    system_metrics: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert metadata to JSON string for embedding."""
        return json.dumps({
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "detections": self.detections,
            "traffic_analytics": self.traffic_analytics,
            "system_metrics": self.system_metrics,
        })


class VideoFrame:
    """Video frame representation for fragmented encoding."""

    def __init__(
        self,
        data: np.ndarray,
        timestamp: float,
        frame_index: int,
        camera_id: str,
        frame_id: str
    ):
        self.data = data
        self.timestamp = timestamp
        self.frame_index = frame_index
        self.camera_id = camera_id
        self.frame_id = frame_id
        self.metadata: AnalyticsMetadata | None = None

    @property
    def height(self) -> int:
        """Get frame height."""
        return self.data.shape[0]

    @property
    def width(self) -> int:
        """Get frame width."""
        return self.data.shape[1]

    @property
    def channels(self) -> int:
        """Get number of color channels."""
        return self.data.shape[2] if len(self.data.shape) == 3 else 1


class FragmentBuffer:
    """Buffer for accumulating frames before fragment generation."""

    def __init__(self, target_duration: float, max_capacity: int = 1000):
        self.target_duration = target_duration
        self.max_capacity = max_capacity
        self.frames: deque[VideoFrame] = deque()
        self.metadata_queue: deque[AnalyticsMetadata] = deque()
        self.start_time: float | None = None
        self.current_duration = 0.0

    def add_frame(self, frame: VideoFrame) -> None:
        """Add a frame to the buffer."""
        if not self.frames:
            self.start_time = frame.timestamp

        self.frames.append(frame)
        self.current_duration = frame.timestamp - self.start_time

        # Prevent buffer overflow
        if len(self.frames) > self.max_capacity:
            dropped_frame = self.frames.popleft()
            if self.frames:
                self.start_time = self.frames[0].timestamp
                self.current_duration = self.frames[-1].timestamp - self.start_time
            else:
                self.start_time = None
                self.current_duration = 0.0

            logger.warning("Fragment buffer overflow, dropped frame",
                         frame_id=dropped_frame.frame_id)

    def add_metadata(self, metadata: AnalyticsMetadata) -> None:
        """Add metadata to the buffer."""
        self.metadata_queue.append(metadata)

        # Clean old metadata to prevent memory leaks
        while (len(self.metadata_queue) > self.max_capacity and
               self.metadata_queue):
            self.metadata_queue.popleft()

    def is_ready_for_fragment(self) -> bool:
        """Check if buffer is ready to generate a fragment."""
        return (self.current_duration >= self.target_duration and
                len(self.frames) > 0)

    def extract_fragment_data(self) -> tuple[list[VideoFrame], list[AnalyticsMetadata]]:
        """Extract data for fragment generation and clear buffer."""
        fragment_frames = list(self.frames)
        fragment_metadata = list(self.metadata_queue)

        self.frames.clear()
        self.metadata_queue.clear()
        self.start_time = None
        self.current_duration = 0.0

        return fragment_frames, fragment_metadata

    @property
    def frame_count(self) -> int:
        """Get current frame count in buffer."""
        return len(self.frames)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.frames) == 0


class FragmentedMP4Encoder:
    """Main fragmented MP4 encoder with metadata track support.
    
    This encoder creates DASH-compatible MP4 fragments with embedded analytics
    metadata tracks, suitable for real-time streaming applications.
    """

    def __init__(
        self,
        video_config: VideoConfiguration,
        fragment_config: FragmentConfiguration,
        camera_id: str,
    ):
        """Initialize the fragmented MP4 encoder.
        
        Args:
            video_config: Video encoding configuration
            fragment_config: Fragment generation configuration  
            camera_id: Unique identifier for the camera stream
        """
        self.video_config = video_config
        self.fragment_config = fragment_config
        self.camera_id = camera_id

        # Initialize components
        self.fragment_buffer = FragmentBuffer(
            target_duration=fragment_config.fragment_duration
        )

        # State tracking
        self.is_running = False
        self.fragment_index = 0
        self.total_frames_processed = 0
        self.total_fragments_generated = 0
        self.last_fragment_time = 0.0

        # Performance metrics
        self.metrics = {
            "encoding_latency_ms": 0.0,
            "fragment_generation_time_ms": 0.0,
            "buffer_utilization": 0.0,
            "frames_per_second": 0.0,
            "fragments_per_minute": 0.0,
        }

        # Output directory setup
        self.output_dir = Path(fragment_config.output_directory) / camera_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Background tasks
        self.processing_task: asyncio.Task | None = None

        logger.info("FragmentedMP4Encoder initialized",
                   camera_id=camera_id,
                   fragment_duration=fragment_config.fragment_duration,
                   video_config=video_config.__dict__)

    async def start(self) -> None:
        """Start the fragmented MP4 encoder."""
        if self.is_running:
            logger.warning("FragmentedMP4Encoder already running")
            return

        self.is_running = True
        self.processing_task = asyncio.create_task(self._processing_loop())

        logger.info("FragmentedMP4Encoder started", camera_id=self.camera_id)

    async def stop(self) -> None:
        """Stop the fragmented MP4 encoder and flush remaining data."""
        if not self.is_running:
            logger.warning("FragmentedMP4Encoder not running")
            return

        logger.info("Stopping FragmentedMP4Encoder", camera_id=self.camera_id)

        self.is_running = False

        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        # Generate final fragment from remaining buffer data
        await self._flush_remaining_data()

        logger.info("FragmentedMP4Encoder stopped",
                   camera_id=self.camera_id,
                   total_fragments=self.total_fragments_generated,
                   total_frames=self.total_frames_processed)

    async def add_frame(
        self,
        frame_data: np.ndarray,
        timestamp: float,
        frame_id: str
    ) -> None:
        """Add a video frame for encoding.
        
        Args:
            frame_data: Video frame as numpy array (H, W, C)
            timestamp: Frame timestamp in seconds
            frame_id: Unique frame identifier
        """
        if not self.is_running:
            logger.warning("Encoder not running, dropping frame", frame_id=frame_id)
            return

        frame = VideoFrame(
            data=frame_data,
            timestamp=timestamp,
            frame_index=self.total_frames_processed,
            camera_id=self.camera_id,
            frame_id=frame_id
        )

        self.fragment_buffer.add_frame(frame)
        self.total_frames_processed += 1

        # Update metrics
        self._update_performance_metrics()

        logger.debug("Frame added to encoder buffer",
                    frame_id=frame_id,
                    buffer_frames=self.fragment_buffer.frame_count,
                    buffer_duration=self.fragment_buffer.current_duration)

    async def add_metadata(self, metadata: AnalyticsMetadata) -> None:
        """Add analytics metadata for embedding in fragments.
        
        Args:
            metadata: Analytics metadata to embed
        """
        if not self.is_running:
            logger.warning("Encoder not running, dropping metadata",
                         camera_id=metadata.camera_id)
            return

        self.fragment_buffer.add_metadata(metadata)

        logger.debug("Metadata added to encoder buffer",
                    camera_id=metadata.camera_id,
                    timestamp=metadata.timestamp)

    def get_fragment_info(self, fragment_index: int) -> dict[str, Any] | None:
        """Get information about a generated fragment.
        
        Args:
            fragment_index: Index of the fragment
            
        Returns:
            Fragment information dictionary or None if not found
        """
        fragment_path = self.output_dir / f"fragment_{fragment_index:06d}.mp4"

        if not fragment_path.exists():
            return None

        return {
            "fragment_index": fragment_index,
            "camera_id": self.camera_id,
            "file_path": str(fragment_path),
            "file_size": fragment_path.stat().st_size,
            "created_time": fragment_path.stat().st_ctime,
        }

    def get_manifest_path(self) -> Path | None:
        """Get path to the DASH manifest file.
        
        Returns:
            Path to manifest file or None if not generated
        """
        manifest_path = self.output_dir / self.fragment_config.manifest_name
        return manifest_path if manifest_path.exists() else None

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        return {
            **self.metrics,
            "camera_id": self.camera_id,
            "is_running": self.is_running,
            "total_frames_processed": self.total_frames_processed,
            "total_fragments_generated": self.total_fragments_generated,
            "buffer_frame_count": self.fragment_buffer.frame_count,
            "buffer_duration_seconds": self.fragment_buffer.current_duration,
            "fragment_index": self.fragment_index,
        }

    async def _processing_loop(self) -> None:
        """Main processing loop for fragment generation."""
        try:
            while self.is_running:
                if self.fragment_buffer.is_ready_for_fragment():
                    await self._generate_fragment()
                else:
                    # Brief sleep to prevent busy waiting
                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.error("Processing loop error",
                        camera_id=self.camera_id, error=str(e))
            raise ServiceError(f"MP4 encoder processing failed: {e}",
                             service="fragmented_mp4_encoder") from e

    async def _generate_fragment(self) -> None:
        """Generate an MP4 fragment from buffered data."""
        start_time = time.time()

        try:
            # Extract data from buffer
            frames, metadata = self.fragment_buffer.extract_fragment_data()

            if not frames:
                logger.warning("No frames to generate fragment",
                             camera_id=self.camera_id)
                return

            fragment_path = self.output_dir / f"fragment_{self.fragment_index:06d}.mp4"

            logger.info("Generating MP4 fragment",
                       camera_id=self.camera_id,
                       fragment_index=self.fragment_index,
                       frame_count=len(frames),
                       metadata_count=len(metadata),
                       output_path=str(fragment_path))

            # Generate fragment using ffmpeg-python for encoding
            await self._encode_fragment_with_ffmpeg(frames, fragment_path)

            # Embed metadata tracks using PyAV for low-level manipulation
            if self.fragment_config.enable_metadata_tracks and metadata:
                await self._embed_metadata_tracks(fragment_path, metadata)

            # Update DASH manifest if enabled
            if self.fragment_config.enable_dash_manifest:
                await self._update_dash_manifest()

            self.fragment_index += 1
            self.total_fragments_generated += 1
            self.last_fragment_time = time.time()

            # Update performance metrics
            generation_time = (time.time() - start_time) * 1000
            self.metrics["fragment_generation_time_ms"] = generation_time

            logger.info("MP4 fragment generated successfully",
                       camera_id=self.camera_id,
                       fragment_index=self.fragment_index - 1,
                       generation_time_ms=generation_time,
                       fragment_size_bytes=fragment_path.stat().st_size)

        except Exception as e:
            logger.error("Fragment generation failed",
                        camera_id=self.camera_id,
                        fragment_index=self.fragment_index,
                        error=str(e))
            raise

    async def _encode_fragment_with_ffmpeg(
        self,
        frames: list[VideoFrame],
        output_path: Path
    ) -> None:
        """Encode video fragment using ffmpeg-python.
        
        Args:
            frames: List of video frames to encode
            output_path: Output path for the fragment
        """
        try:
            # Create temporary input for frame data
            frame_data = np.stack([frame.data for frame in frames])

            # Configure ffmpeg encoding pipeline
            input_config = {
                'format': 'rawvideo',
                'pix_fmt': 'rgb24',
                's': f'{frames[0].width}x{frames[0].height}',
                'r': self.video_config.framerate,
            }

            output_config = {
                'vcodec': self.video_config.codec,
                'pix_fmt': self.video_config.pixel_format,
                'preset': self.video_config.preset,
                'crf': self.video_config.crf,
                'profile:v': self.video_config.profile,
                'movflags': '+frag_keyframe+empty_moov+faststart',
                'f': 'mp4',
            }

            # Run ffmpeg encoding process
            process = (
                ffmpeg
                .input('pipe:', **input_config)
                .output(str(output_path), **output_config)
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stderr=True)
            )

            # Write frame data to ffmpeg
            frame_bytes = frame_data.astype(np.uint8).tobytes()
            await asyncio.get_event_loop().run_in_executor(
                None, self._write_frames_to_process, process, frame_bytes
            )

            # Wait for completion
            await asyncio.get_event_loop().run_in_executor(
                None, process.wait
            )

            if process.returncode != 0:
                stderr_output = process.stderr.read().decode() if process.stderr else ""
                raise ServiceError(
                    f"FFmpeg encoding failed: {stderr_output}",
                    service="fragmented_mp4_encoder"
                )

        except Exception as e:
            logger.error("FFmpeg encoding failed",
                        camera_id=self.camera_id, error=str(e))
            raise

    def _write_frames_to_process(self, process: Any, frame_bytes: bytes) -> None:
        """Write frame bytes to ffmpeg process stdin."""
        try:
            process.stdin.write(frame_bytes)
            process.stdin.close()
        except Exception as e:
            logger.error("Failed to write frames to process", error=str(e))
            raise

    async def _embed_metadata_tracks(
        self,
        fragment_path: Path,
        metadata: list[AnalyticsMetadata]
    ) -> None:
        """Embed metadata tracks in the MP4 fragment using PyAV.
        
        Args:
            fragment_path: Path to the MP4 fragment
            metadata: List of metadata to embed
        """
        try:
            # Create temporary file for metadata embedding
            temp_path = fragment_path.with_suffix('.temp.mp4')

            # Open input and output containers
            async with asyncio.Lock():  # Ensure thread safety for PyAV operations
                with av.open(str(fragment_path)) as input_container:
                    with av.open(str(temp_path), 'w') as output_container:

                        # Copy video and audio streams
                        stream_mapping = {}
                        for stream in input_container.streams:
                            if stream.type in ('video', 'audio'):
                                out_stream = output_container.add_stream_from_template(stream)
                                stream_mapping[stream] = out_stream

                        # Add metadata data stream
                        if metadata:
                            metadata_stream = output_container.add_data_stream('klv')  # KLV format for metadata

                            # Create metadata packet
                            metadata_json = json.dumps([meta.to_json() for meta in metadata])
                            metadata_packet = av.Packet(metadata_json.encode('utf-8'))
                            metadata_packet.stream = metadata_stream

                        # Copy packets and add metadata
                        for packet in input_container.demux():
                            if packet.stream in stream_mapping:
                                packet.stream = stream_mapping[packet.stream]
                                output_container.mux(packet)

                        # Mux metadata packet
                        if metadata:
                            output_container.mux(metadata_packet)

            # Replace original with metadata-enhanced version
            temp_path.replace(fragment_path)

            logger.debug("Metadata tracks embedded successfully",
                        camera_id=self.camera_id,
                        metadata_count=len(metadata),
                        fragment_path=str(fragment_path))

        except Exception as e:
            logger.error("Metadata embedding failed",
                        camera_id=self.camera_id,
                        fragment_path=str(fragment_path),
                        error=str(e))
            # Clean up temp file if it exists
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise

    async def _update_dash_manifest(self) -> None:
        """Update DASH manifest file for adaptive streaming."""
        try:
            manifest_path = self.output_dir / self.fragment_config.manifest_name

            # Basic DASH manifest template (MPD)
            manifest_content = self._generate_dash_manifest()

            # Write manifest file atomically
            temp_manifest = manifest_path.with_suffix('.temp.mpd')
            temp_manifest.write_text(manifest_content)
            temp_manifest.replace(manifest_path)

            logger.debug("DASH manifest updated",
                        camera_id=self.camera_id,
                        manifest_path=str(manifest_path))

        except Exception as e:
            logger.error("DASH manifest update failed",
                        camera_id=self.camera_id, error=str(e))
            raise

    def _generate_dash_manifest(self) -> str:
        """Generate DASH manifest XML content.
        
        Returns:
            DASH manifest XML as string
        """
        # Basic DASH MPD manifest structure
        manifest = f'''<?xml version="1.0" encoding="UTF-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" profiles="urn:mpeg:dash:profile:isoff-live:2011"
     type="dynamic" minimumUpdatePeriod="PT6S" suggestedPresentationDelay="PT18S"
     availabilityStartTime="2024-01-01T00:00:00Z">
  <Period id="0" start="PT0S">
    <AdaptationSet id="0" contentType="video" mimeType="video/mp4" codecs="{self.video_config.codec}">
      <Representation id="0" bandwidth="2000000" width="{self.video_config.width}" height="{self.video_config.height}" frameRate="{self.video_config.framerate}">
        <SegmentTemplate media="fragment_$Number%06d$.mp4" initialization="init.mp4" timescale="1000" duration="{int(self.fragment_config.fragment_duration * 1000)}" startNumber="0"/>
      </Representation>
    </AdaptationSet>
  </Period>
</MPD>'''
        return manifest

    async def _flush_remaining_data(self) -> None:
        """Flush any remaining data in buffers before shutdown."""
        if not self.fragment_buffer.is_empty:
            logger.info("Flushing remaining buffer data",
                       camera_id=self.camera_id,
                       remaining_frames=self.fragment_buffer.frame_count)

            # Force generate final fragment regardless of duration
            original_target = self.fragment_buffer.target_duration
            self.fragment_buffer.target_duration = 0.0

            try:
                await self._generate_fragment()
            finally:
                self.fragment_buffer.target_duration = original_target

    def _update_performance_metrics(self) -> None:
        """Update performance tracking metrics."""
        current_time = time.time()

        # Calculate frames per second
        if self.last_fragment_time > 0:
            time_diff = current_time - self.last_fragment_time
            if time_diff > 0:
                self.metrics["frames_per_second"] = self.fragment_buffer.frame_count / time_diff

        # Calculate buffer utilization
        max_buffer_duration = self.fragment_config.max_fragment_duration
        self.metrics["buffer_utilization"] = min(
            1.0, self.fragment_buffer.current_duration / max_buffer_duration
        )

        # Calculate fragments per minute
        if self.total_fragments_generated > 0 and self.last_fragment_time > 0:
            uptime_minutes = (current_time - self.last_fragment_time) / 60.0
            if uptime_minutes > 0:
                self.metrics["fragments_per_minute"] = self.total_fragments_generated / uptime_minutes

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


# Factory function for easy encoder creation
def create_fragmented_mp4_encoder(
    camera_id: str,
    video_config: VideoConfiguration | None = None,
    fragment_config: FragmentConfiguration | None = None,
) -> FragmentedMP4Encoder:
    """Create and configure a fragmented MP4 encoder.
    
    Args:
        camera_id: Unique identifier for the camera stream
        video_config: Optional video encoding configuration
        fragment_config: Optional fragment generation configuration
        
    Returns:
        Configured FragmentedMP4Encoder instance
    """
    if video_config is None:
        video_config = VideoConfiguration()

    if fragment_config is None:
        fragment_config = FragmentConfiguration()

    return FragmentedMP4Encoder(
        video_config=video_config,
        fragment_config=fragment_config,
        camera_id=camera_id,
    )
