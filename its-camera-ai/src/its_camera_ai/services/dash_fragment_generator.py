"""DASH Fragment Generator for Adaptive Streaming.

This module implements DASH (Dynamic Adaptive Streaming over HTTP) compatible
fragment generation for the ITS Camera AI system, providing professional-grade
adaptive streaming capabilities with multiple quality levels and bitrates.

Features:
- DASH-compliant fragment and manifest generation  
- Multi-bitrate adaptive streaming support
- Real-time fragment and initialization segment creation
- MPD (Media Presentation Description) manifest management
- Support for live and on-demand streaming profiles
- Integration with fragmented MP4 encoder
- Optimized for low-latency streaming
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC
from enum import Enum
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import ffmpeg

from ..core.exceptions import ServiceError
from ..core.logging import get_logger

logger = get_logger(__name__)


class DashProfile(Enum):
    """DASH streaming profiles."""
    LIVE = "urn:mpeg:dash:profile:isoff-live:2011"
    ON_DEMAND = "urn:mpeg:dash:profile:isoff-on-demand:2011"
    MAIN = "urn:mpeg:dash:profile:isoff-main:2011"


class PresentationType(Enum):
    """DASH presentation types."""
    STATIC = "static"
    DYNAMIC = "dynamic"


@dataclass
class DashQualityLevel:
    """Configuration for a DASH quality level/representation."""

    id: str
    width: int
    height: int
    bitrate: int  # bits per second
    framerate: float
    codec: str = "avc1.42E01E"  # H.264 Baseline Profile
    bandwidth: int | None = None  # Will be calculated if not provided

    def __post_init__(self):
        if self.bandwidth is None:
            self.bandwidth = self.bitrate


@dataclass
class DashConfiguration:
    """Configuration for DASH fragment generation."""

    profile: DashProfile = DashProfile.LIVE
    presentation_type: PresentationType = PresentationType.DYNAMIC
    segment_duration: float = 6.0  # seconds
    min_buffer_time: float = 18.0  # seconds
    suggested_presentation_delay: float = 18.0  # seconds
    minimum_update_period: float = 6.0  # seconds
    time_shift_buffer_depth: float = 300.0  # seconds (5 minutes)

    # Quality levels for adaptive streaming
    quality_levels: list[DashQualityLevel] = field(default_factory=lambda: [
        DashQualityLevel("low", 854, 480, 800000, 30.0),
        DashQualityLevel("medium", 1280, 720, 1500000, 30.0),
        DashQualityLevel("high", 1920, 1080, 3000000, 30.0),
    ])

    # Output settings
    output_directory: str = "/tmp/its-camera-ai/dash"
    manifest_filename: str = "manifest.mpd"
    init_segment_template: str = "init-$RepresentationID$.mp4"
    media_segment_template: str = "segment-$RepresentationID$-$Number$.mp4"

    # Advanced settings
    enable_faststart: bool = True
    enable_fragmented_mp4: bool = True
    utc_timing_scheme: str | None = "urn:mpeg:dash:utc:http-head:2014"
    utc_timing_value: str | None = "http://time.akamai.com"


@dataclass
class SegmentInfo:
    """Information about a generated segment."""

    representation_id: str
    segment_number: int
    start_time: float
    duration: float
    file_path: str
    file_size: int
    bitrate: int
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "representation_id": self.representation_id,
            "segment_number": self.segment_number,
            "start_time": self.start_time,
            "duration": self.duration,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "bitrate": self.bitrate,
            "created_at": self.created_at,
        }


class DashManifestGenerator:
    """Generates DASH MPD manifests for adaptive streaming."""

    def __init__(self, config: DashConfiguration):
        self.config = config
        self.segments_info: dict[str, list[SegmentInfo]] = {}
        self.availability_start_time = time.time()

        # Initialize segment tracking for each quality level
        for quality in config.quality_levels:
            self.segments_info[quality.id] = []

    def generate_mpd_manifest(self, current_time: float | None = None) -> str:
        """Generate DASH MPD manifest XML.
        
        Args:
            current_time: Current time for dynamic manifests
            
        Returns:
            MPD manifest XML as string
        """
        if current_time is None:
            current_time = time.time()

        # Create MPD root element
        mpd = ET.Element("MPD")
        mpd.set("xmlns", "urn:mpeg:dash:schema:mpd:2011")
        mpd.set("profiles", self.config.profile.value)
        mpd.set("type", self.config.presentation_type.value)

        # Set timing attributes for dynamic content
        if self.config.presentation_type == PresentationType.DYNAMIC:
            mpd.set("minimumUpdatePeriod", f"PT{self.config.minimum_update_period:.1f}S")
            mpd.set("suggestedPresentationDelay", f"PT{self.config.suggested_presentation_delay:.1f}S")
            mpd.set("timeShiftBufferDepth", f"PT{self.config.time_shift_buffer_depth:.1f}S")
            mpd.set("availabilityStartTime", self._format_datetime(self.availability_start_time))

        mpd.set("minBufferTime", f"PT{self.config.min_buffer_time:.1f}S")

        # Add UTCTiming if configured
        if self.config.utc_timing_scheme and self.config.utc_timing_value:
            utc_timing = ET.SubElement(mpd, "UTCTiming")
            utc_timing.set("schemeIdUri", self.config.utc_timing_scheme)
            utc_timing.set("value", self.config.utc_timing_value)

        # Create Period
        period = ET.SubElement(mpd, "Period")
        period.set("id", "0")
        period.set("start", "PT0S")

        # Create AdaptationSet for video
        adaptation_set = ET.SubElement(period, "AdaptationSet")
        adaptation_set.set("id", "0")
        adaptation_set.set("contentType", "video")
        adaptation_set.set("mimeType", "video/mp4")
        adaptation_set.set("startWithSAP", "1")

        # Add Representations for each quality level
        for quality in self.config.quality_levels:
            self._add_representation(adaptation_set, quality)

        # Format and return XML
        self._indent_xml(mpd)
        xml_str = ET.tostring(mpd, encoding='unicode')
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_str}'

    def _add_representation(self, adaptation_set: ET.Element, quality: DashQualityLevel) -> None:
        """Add a Representation element for a quality level."""
        representation = ET.SubElement(adaptation_set, "Representation")
        representation.set("id", quality.id)
        representation.set("codecs", quality.codec)
        representation.set("width", str(quality.width))
        representation.set("height", str(quality.height))
        representation.set("bandwidth", str(quality.bandwidth))
        representation.set("frameRate", str(quality.framerate))

        # Add SegmentTemplate
        segment_template = ET.SubElement(representation, "SegmentTemplate")
        segment_template.set("timescale", "1000")  # milliseconds
        segment_template.set("duration", str(int(self.config.segment_duration * 1000)))
        segment_template.set("initialization", self.config.init_segment_template.replace("$RepresentationID$", quality.id))
        segment_template.set("media", self.config.media_segment_template.replace("$RepresentationID$", quality.id))
        segment_template.set("startNumber", "1")

    def add_segment_info(self, segment_info: SegmentInfo) -> None:
        """Add information about a generated segment."""
        if segment_info.representation_id in self.segments_info:
            self.segments_info[segment_info.representation_id].append(segment_info)

            # Keep only recent segments for live streaming
            if self.config.presentation_type == PresentationType.DYNAMIC:
                max_segments = int(self.config.time_shift_buffer_depth / self.config.segment_duration)
                if len(self.segments_info[segment_info.representation_id]) > max_segments:
                    self.segments_info[segment_info.representation_id] = \
                        self.segments_info[segment_info.representation_id][-max_segments:]

    def get_segment_count(self, representation_id: str) -> int:
        """Get number of segments for a representation."""
        return len(self.segments_info.get(representation_id, []))

    def _format_datetime(self, timestamp: float) -> str:
        """Format timestamp as ISO 8601 datetime."""
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp, UTC)
        return dt.isoformat().replace('+00:00', 'Z')

    def _indent_xml(self, elem: ET.Element, level: int = 0) -> None:
        """Add indentation to XML for pretty printing."""
        indent = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent


class DashFragmentGenerator:
    """Generates DASH-compatible fragments and manifests.
    
    This generator creates DASH segments with multiple quality levels
    for adaptive streaming, managing both fragment generation and
    manifest updates for real-time streaming applications.
    """

    def __init__(self, config: DashConfiguration, camera_id: str):
        """Initialize the DASH fragment generator.
        
        Args:
            config: DASH configuration
            camera_id: Unique identifier for the camera stream
        """
        self.config = config
        self.camera_id = camera_id
        self.manifest_generator = DashManifestGenerator(config)

        # Output directory setup
        self.output_dir = Path(config.output_directory) / camera_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.segment_numbers: dict[str, int] = {}
        self.last_segment_time = time.time()
        self.is_initialized = False

        # Performance metrics
        self.metrics = {
            "segments_generated": 0,
            "total_output_size_bytes": 0,
            "generation_time_ms": 0.0,
            "manifest_updates": 0,
            "encoding_errors": 0,
        }

        # Initialize segment counters
        for quality in config.quality_levels:
            self.segment_numbers[quality.id] = 1

        logger.info("DashFragmentGenerator initialized",
                   camera_id=camera_id,
                   quality_levels=len(config.quality_levels),
                   output_dir=str(self.output_dir))

    async def initialize_streaming(self) -> None:
        """Initialize DASH streaming by generating initialization segments."""
        if self.is_initialized:
            logger.warning("DASH streaming already initialized")
            return

        try:
            logger.info("Initializing DASH streaming", camera_id=self.camera_id)

            # Generate initialization segments for each quality level
            init_tasks = []
            for quality in self.config.quality_levels:
                task = self._generate_initialization_segment(quality)
                init_tasks.append(task)

            await asyncio.gather(*init_tasks)

            # Generate initial manifest
            await self._update_manifest()

            self.is_initialized = True
            logger.info("DASH streaming initialized successfully",
                       camera_id=self.camera_id)

        except Exception as e:
            logger.error("DASH streaming initialization failed",
                        camera_id=self.camera_id, error=str(e))
            raise ServiceError(f"DASH initialization failed: {e}",
                             service="dash_fragment_generator") from e

    async def generate_fragments(
        self,
        input_video_path: str,
        segment_duration: float | None = None
    ) -> list[SegmentInfo]:
        """Generate DASH fragments from input video.
        
        Args:
            input_video_path: Path to input video file
            segment_duration: Optional segment duration override
            
        Returns:
            List of generated segment information
        """
        if not self.is_initialized:
            await self.initialize_streaming()

        start_time = time.time()
        duration = segment_duration or self.config.segment_duration

        try:
            logger.info("Generating DASH fragments",
                       camera_id=self.camera_id,
                       input_path=input_video_path,
                       quality_levels=len(self.config.quality_levels))

            # Generate fragments for each quality level
            fragment_tasks = []
            for quality in self.config.quality_levels:
                task = self._generate_segment_for_quality(
                    input_video_path, quality, duration
                )
                fragment_tasks.append(task)

            # Wait for all fragments to be generated
            segment_infos = await asyncio.gather(*fragment_tasks)

            # Update manifest with new segments
            for segment_info in segment_infos:
                if segment_info:
                    self.manifest_generator.add_segment_info(segment_info)

            await self._update_manifest()

            # Update metrics
            generation_time = (time.time() - start_time) * 1000
            self.metrics["generation_time_ms"] = generation_time
            self.metrics["segments_generated"] += len(segment_infos)

            # Filter out None results
            valid_segments = [seg for seg in segment_infos if seg is not None]

            logger.info("DASH fragments generated successfully",
                       camera_id=self.camera_id,
                       segments_count=len(valid_segments),
                       generation_time_ms=generation_time)

            return valid_segments

        except Exception as e:
            self.metrics["encoding_errors"] += 1
            logger.error("DASH fragment generation failed",
                        camera_id=self.camera_id,
                        input_path=input_video_path,
                        error=str(e))
            raise

    async def _generate_initialization_segment(self, quality: DashQualityLevel) -> None:
        """Generate initialization segment for a quality level."""
        init_filename = self.config.init_segment_template.replace("$RepresentationID$", quality.id)
        init_path = self.output_dir / init_filename

        try:
            logger.debug("Generating initialization segment",
                        quality_id=quality.id,
                        output_path=str(init_path))

            # Create a dummy input for initialization segment
            input_config = {
                'f': 'lavfi',
                'i': f'color=black:size={quality.width}x{quality.height}:duration=1:rate={quality.framerate}',
            }

            output_config = {
                'vcodec': 'libx264',
                'preset': 'ultrafast',
                'tune': 'zerolatency',
                'profile:v': 'baseline',
                'level': '3.1',
                'pix_fmt': 'yuv420p',
                'b:v': quality.bitrate,
                'r': quality.framerate,
                's': f'{quality.width}x{quality.height}',
                'movflags': '+frag_keyframe+empty_moov+default_base_moof',
                'f': 'mp4',
                'vframes': 0,  # Only initialization segment, no frames
            }

            # Run ffmpeg to generate initialization segment
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ffmpeg
                .input(**input_config)
                .output(str(init_path), **output_config)
                .overwrite_output()
                .run(capture_stderr=True)
            )

            logger.debug("Initialization segment generated",
                        quality_id=quality.id,
                        file_size=init_path.stat().st_size)

        except Exception as e:
            logger.error("Failed to generate initialization segment",
                        quality_id=quality.id, error=str(e))
            raise

    async def _generate_segment_for_quality(
        self,
        input_path: str,
        quality: DashQualityLevel,
        duration: float
    ) -> SegmentInfo | None:
        """Generate a segment for a specific quality level."""
        segment_number = self.segment_numbers[quality.id]
        segment_filename = (self.config.media_segment_template
                          .replace("$RepresentationID$", quality.id)
                          .replace("$Number$", str(segment_number)))
        segment_path = self.output_dir / segment_filename

        try:
            logger.debug("Generating segment for quality level",
                        quality_id=quality.id,
                        segment_number=segment_number,
                        output_path=str(segment_path))

            # Configure ffmpeg encoding
            output_config = {
                'vcodec': 'libx264',
                'preset': 'faster',
                'tune': 'zerolatency',
                'profile:v': 'baseline',
                'level': '3.1',
                'pix_fmt': 'yuv420p',
                'b:v': quality.bitrate,
                'maxrate': quality.bitrate,
                'bufsize': quality.bitrate * 2,
                'r': quality.framerate,
                's': f'{quality.width}x{quality.height}',
                'g': int(quality.framerate * 2),  # Keyframe interval
                'keyint_min': int(quality.framerate),
                'sc_threshold': 0,
                'movflags': '+frag_keyframe+empty_moov+default_base_moof',
                'f': 'mp4',
                't': duration,  # Segment duration
            }

            if self.config.enable_faststart:
                output_config['movflags'] += '+faststart'

            # Run ffmpeg encoding
            start_time = time.time()
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ffmpeg
                .input(input_path)
                .output(str(segment_path), **output_config)
                .overwrite_output()
                .run(capture_stderr=True)
            )

            # Create segment info
            segment_info = SegmentInfo(
                representation_id=quality.id,
                segment_number=segment_number,
                start_time=self.last_segment_time,
                duration=duration,
                file_path=str(segment_path),
                file_size=segment_path.stat().st_size,
                bitrate=quality.bitrate,
                created_at=time.time()
            )

            # Update counters
            self.segment_numbers[quality.id] += 1
            self.metrics["total_output_size_bytes"] += segment_info.file_size

            logger.debug("Segment generated successfully",
                        quality_id=quality.id,
                        segment_number=segment_number,
                        file_size=segment_info.file_size,
                        duration=duration)

            return segment_info

        except Exception as e:
            logger.error("Failed to generate segment",
                        quality_id=quality.id,
                        segment_number=segment_number,
                        error=str(e))
            return None

    async def _update_manifest(self) -> None:
        """Update the DASH manifest file."""
        try:
            manifest_path = self.output_dir / self.config.manifest_filename

            # Generate updated manifest
            manifest_content = self.manifest_generator.generate_mpd_manifest()

            # Write manifest atomically
            temp_manifest = manifest_path.with_suffix('.tmp')
            temp_manifest.write_text(manifest_content, encoding='utf-8')
            temp_manifest.replace(manifest_path)

            self.metrics["manifest_updates"] += 1

            logger.debug("DASH manifest updated",
                        camera_id=self.camera_id,
                        manifest_path=str(manifest_path))

        except Exception as e:
            logger.error("Failed to update DASH manifest",
                        camera_id=self.camera_id, error=str(e))
            raise

    def get_manifest_path(self) -> Path | None:
        """Get path to the DASH manifest file.
        
        Returns:
            Path to manifest file or None if not generated
        """
        manifest_path = self.output_dir / self.config.manifest_filename
        return manifest_path if manifest_path.exists() else None

    def get_segment_info(self, representation_id: str, segment_number: int) -> SegmentInfo | None:
        """Get information about a specific segment.
        
        Args:
            representation_id: ID of the representation/quality level
            segment_number: Segment number
            
        Returns:
            Segment information or None if not found
        """
        segments = self.manifest_generator.segments_info.get(representation_id, [])

        for segment in segments:
            if segment.segment_number == segment_number:
                return segment

        return None

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get current performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        return {
            **self.metrics,
            "camera_id": self.camera_id,
            "is_initialized": self.is_initialized,
            "quality_levels": len(self.config.quality_levels),
            "total_segments": sum(self.segment_numbers.values()) - len(self.segment_numbers),
            "segment_numbers": self.segment_numbers.copy(),
        }

    def cleanup_old_segments(self, max_age_seconds: float = 3600.0) -> int:
        """Cleanup old segment files.
        
        Args:
            max_age_seconds: Maximum age of segments to keep
            
        Returns:
            Number of segments cleaned up
        """
        try:
            current_time = time.time()
            cleanup_count = 0

            # Find and remove old segments
            for segment_file in self.output_dir.glob("segment-*.mp4"):
                if segment_file.stat().st_mtime < (current_time - max_age_seconds):
                    segment_file.unlink()
                    cleanup_count += 1

            logger.info("Old segments cleaned up",
                       camera_id=self.camera_id,
                       cleanup_count=cleanup_count)

            return cleanup_count

        except Exception as e:
            logger.error("Failed to cleanup old segments",
                        camera_id=self.camera_id, error=str(e))
            return 0


# Factory function for creating DASH generators
def create_dash_fragment_generator(
    camera_id: str,
    config: DashConfiguration | None = None
) -> DashFragmentGenerator:
    """Create a DASH fragment generator with default configuration.
    
    Args:
        camera_id: Unique identifier for the camera stream
        config: Optional DASH configuration
        
    Returns:
        Configured DashFragmentGenerator instance
    """
    if config is None:
        config = DashConfiguration()

    return DashFragmentGenerator(config, camera_id)
