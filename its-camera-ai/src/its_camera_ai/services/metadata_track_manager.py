"""Metadata Track Manager for MP4 Fragment Analytics Embedding.

This module manages the embedding of analytics metadata tracks in MP4 fragments,
providing synchronized embedding of detection results, traffic analytics, and system
metrics with video content for comprehensive streaming analytics.

Features:
- Multi-format metadata track support (JSON, Binary, KLV)
- Temporal synchronization with video timelines
- Efficient metadata compression and encoding
- Support for multiple analytics data types
- Real-time metadata streaming capabilities
- Integration with PyAV for low-level container manipulation
"""

import asyncio
import json
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import av

from ..core.exceptions import ServiceError
from ..core.logging import get_logger
from .fragmented_mp4_encoder import AnalyticsMetadata

logger = get_logger(__name__)


class MetadataFormat(Enum):
    """Supported metadata track formats."""
    JSON = "json"
    BINARY = "binary"
    KLV = "klv"  # Key-Length-Value format
    TIMED_TEXT = "timed_text"


class MetadataTrackType(Enum):
    """Types of metadata tracks for different analytics data."""
    DETECTION_RESULTS = "detection_results"
    TRAFFIC_ANALYTICS = "traffic_analytics"
    SYSTEM_METRICS = "system_metrics"
    CAMERA_STATUS = "camera_status"
    ZONE_ANALYTICS = "zone_analytics"
    SPEED_METRICS = "speed_metrics"
    VIOLATION_DATA = "violation_data"
    CUSTOM = "custom"


@dataclass
class MetadataTrackConfig:
    """Configuration for metadata track embedding."""

    track_type: MetadataTrackType
    format: MetadataFormat = MetadataFormat.JSON
    enabled: bool = True
    compression_enabled: bool = True
    sync_tolerance_ms: int = 100
    max_payload_size_kb: int = 64
    encoding: str = "utf-8"
    klv_universal_key: bytes | None = None


@dataclass
class TimedMetadata:
    """Timed metadata entry for synchronization with video."""

    timestamp: float
    duration: float
    payload: bytes
    track_type: MetadataTrackType
    format: MetadataFormat
    frame_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "duration": self.duration,
            "payload_size": len(self.payload),
            "track_type": self.track_type.value,
            "format": self.format.value,
            "frame_indices": self.frame_indices,
        }


class MetadataEncoder:
    """Encoder for different metadata formats."""

    @staticmethod
    def encode_json(data: dict[str, Any], compress: bool = True) -> bytes:
        """Encode data as JSON format.
        
        Args:
            data: Data to encode
            compress: Whether to compress the JSON
            
        Returns:
            Encoded JSON bytes
        """
        json_str = json.dumps(data, separators=(',', ':'))
        json_bytes = json_str.encode('utf-8')

        if compress:
            import gzip
            json_bytes = gzip.compress(json_bytes)

        return json_bytes

    @staticmethod
    def encode_binary(data: dict[str, Any]) -> bytes:
        """Encode data in compact binary format.
        
        Args:
            data: Data to encode
            
        Returns:
            Encoded binary bytes
        """
        # Simple binary encoding for numeric data
        result = bytearray()

        # Encode timestamp
        if 'timestamp' in data:
            result.extend(struct.pack('d', data['timestamp']))

        # Encode detection count
        detections = data.get('detections', [])
        result.extend(struct.pack('I', len(detections)))

        # Encode each detection
        for detection in detections:
            # Bounding box (4 floats)
            bbox = detection.get('bbox', [0.0, 0.0, 0.0, 0.0])
            result.extend(struct.pack('4f', *bbox[:4]))

            # Confidence score
            confidence = detection.get('confidence', 0.0)
            result.extend(struct.pack('f', confidence))

            # Class ID
            class_id = detection.get('class_id', 0)
            result.extend(struct.pack('I', class_id))

        return bytes(result)

    @staticmethod
    def encode_klv(data: dict[str, Any], universal_key: bytes | None = None) -> bytes:
        """Encode data in KLV (Key-Length-Value) format.
        
        Args:
            data: Data to encode
            universal_key: Optional universal key for KLV
            
        Returns:
            Encoded KLV bytes
        """
        if universal_key is None:
            # Default universal key for ITS camera analytics
            universal_key = b'\x06\x0e\x2b\x34\x01\x01\x01\x0e\x01\x03\x02\x01\x01\x00\x00\x00'

        # Encode payload as JSON
        payload = MetadataEncoder.encode_json(data, compress=False)

        # KLV structure: Key + Length + Value
        result = bytearray()
        result.extend(universal_key)  # Key (16 bytes)
        result.extend(struct.pack('>I', len(payload)))  # Length (4 bytes, big-endian)
        result.extend(payload)  # Value

        return bytes(result)


class MetadataTrackManager:
    """Manages embedding of analytics metadata tracks in MP4 fragments.
    
    This manager handles the creation, encoding, and embedding of multiple
    metadata tracks containing synchronized analytics data.
    """

    def __init__(self, track_configs: list[MetadataTrackConfig]):
        """Initialize the metadata track manager.
        
        Args:
            track_configs: List of metadata track configurations
        """
        self.track_configs = {config.track_type: config for config in track_configs}
        self.metadata_buffers: dict[MetadataTrackType, list[TimedMetadata]] = {}
        self.encoder = MetadataEncoder()

        # Performance metrics
        self.metrics = {
            "tracks_processed": 0,
            "total_metadata_size_bytes": 0,
            "encoding_time_ms": 0.0,
            "sync_errors": 0,
            "compression_ratio": 0.0,
        }

        # Initialize buffers for each enabled track type
        for track_type, config in self.track_configs.items():
            if config.enabled:
                self.metadata_buffers[track_type] = []

        logger.info("MetadataTrackManager initialized",
                   enabled_tracks=list(self.metadata_buffers.keys()))

    async def add_analytics_metadata(
        self,
        metadata: AnalyticsMetadata,
        fragment_duration: float,
        frame_indices: list[int] | None = None
    ) -> None:
        """Add analytics metadata for embedding in tracks.
        
        Args:
            metadata: Analytics metadata to add
            fragment_duration: Duration of the fragment
            frame_indices: Optional list of frame indices for sync
        """
        start_time = time.time()

        try:
            # Process detection results
            if metadata.detections and MetadataTrackType.DETECTION_RESULTS in self.metadata_buffers:
                await self._add_detection_metadata(
                    metadata, fragment_duration, frame_indices
                )

            # Process traffic analytics
            if metadata.traffic_analytics and MetadataTrackType.TRAFFIC_ANALYTICS in self.metadata_buffers:
                await self._add_traffic_metadata(
                    metadata, fragment_duration, frame_indices
                )

            # Process system metrics
            if metadata.system_metrics and MetadataTrackType.SYSTEM_METRICS in self.metadata_buffers:
                await self._add_system_metadata(
                    metadata, fragment_duration, frame_indices
                )

            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.metrics["encoding_time_ms"] = processing_time

        except Exception as e:
            logger.error("Failed to add analytics metadata",
                        camera_id=metadata.camera_id, error=str(e))
            raise

    async def embed_metadata_tracks(
        self,
        input_path: str,
        output_path: str,
        fragment_duration: float
    ) -> None:
        """Embed metadata tracks into an MP4 fragment.
        
        Args:
            input_path: Path to input MP4 file
            output_path: Path for output MP4 with metadata tracks
            fragment_duration: Duration of the fragment in seconds
        """
        try:
            logger.info("Embedding metadata tracks",
                       input_path=input_path,
                       output_path=output_path,
                       track_count=len(self.metadata_buffers))

            # Use PyAV for low-level container manipulation
            async with asyncio.Lock():  # Ensure thread safety
                with av.open(input_path) as input_container:
                    with av.open(output_path, 'w') as output_container:

                        # Copy existing streams
                        stream_mapping = {}
                        for stream in input_container.streams:
                            if stream.type in ('video', 'audio'):
                                out_stream = output_container.add_stream_from_template(stream)
                                stream_mapping[stream] = out_stream

                        # Add metadata streams for each track type
                        metadata_streams = {}
                        for track_type, metadata_list in self.metadata_buffers.items():
                            if metadata_list and track_type in self.track_configs:
                                config = self.track_configs[track_type]

                                # Create metadata stream based on format
                                if config.format == MetadataFormat.KLV:
                                    stream = output_container.add_data_stream('klv')
                                elif config.format == MetadataFormat.TIMED_TEXT:
                                    stream = output_container.add_stream('subrip')  # Subtitle track
                                else:
                                    stream = output_container.add_data_stream('bin')

                                metadata_streams[track_type] = stream

                        # Copy video/audio packets
                        packet_count = 0
                        for packet in input_container.demux():
                            if packet.stream in stream_mapping:
                                packet.stream = stream_mapping[packet.stream]
                                output_container.mux(packet)
                                packet_count += 1

                        # Embed metadata packets
                        await self._mux_metadata_packets(
                            output_container, metadata_streams, fragment_duration
                        )

                        logger.debug("Metadata tracks embedded successfully",
                                   packet_count=packet_count,
                                   metadata_streams=len(metadata_streams))

            # Clear metadata buffers after successful embedding
            self._clear_metadata_buffers()
            self.metrics["tracks_processed"] += len(self.metadata_buffers)

        except Exception as e:
            logger.error("Failed to embed metadata tracks",
                        input_path=input_path,
                        output_path=output_path,
                        error=str(e))
            raise ServiceError(f"Metadata embedding failed: {e}",
                             service="metadata_track_manager") from e

    async def _add_detection_metadata(
        self,
        metadata: AnalyticsMetadata,
        duration: float,
        frame_indices: list[int] | None = None
    ) -> None:
        """Add detection metadata to the track buffer."""
        config = self.track_configs[MetadataTrackType.DETECTION_RESULTS]

        detection_data = {
            "timestamp": metadata.timestamp,
            "camera_id": metadata.camera_id,
            "frame_id": metadata.frame_id,
            "detections": metadata.detections,
            "detection_count": len(metadata.detections),
        }

        # Encode based on configured format
        if config.format == MetadataFormat.JSON:
            payload = self.encoder.encode_json(detection_data, config.compression_enabled)
        elif config.format == MetadataFormat.BINARY:
            payload = self.encoder.encode_binary(detection_data)
        elif config.format == MetadataFormat.KLV:
            payload = self.encoder.encode_klv(detection_data, config.klv_universal_key)
        else:
            payload = self.encoder.encode_json(detection_data, False)

        timed_metadata = TimedMetadata(
            timestamp=metadata.timestamp,
            duration=duration,
            payload=payload,
            track_type=MetadataTrackType.DETECTION_RESULTS,
            format=config.format,
            frame_indices=frame_indices or []
        )

        self.metadata_buffers[MetadataTrackType.DETECTION_RESULTS].append(timed_metadata)
        self.metrics["total_metadata_size_bytes"] += len(payload)

    async def _add_traffic_metadata(
        self,
        metadata: AnalyticsMetadata,
        duration: float,
        frame_indices: list[int] | None = None
    ) -> None:
        """Add traffic analytics metadata to the track buffer."""
        config = self.track_configs[MetadataTrackType.TRAFFIC_ANALYTICS]

        traffic_data = {
            "timestamp": metadata.timestamp,
            "camera_id": metadata.camera_id,
            "analytics": metadata.traffic_analytics,
        }

        # Encode based on configured format
        payload = self.encoder.encode_json(traffic_data, config.compression_enabled)

        timed_metadata = TimedMetadata(
            timestamp=metadata.timestamp,
            duration=duration,
            payload=payload,
            track_type=MetadataTrackType.TRAFFIC_ANALYTICS,
            format=config.format,
            frame_indices=frame_indices or []
        )

        self.metadata_buffers[MetadataTrackType.TRAFFIC_ANALYTICS].append(timed_metadata)
        self.metrics["total_metadata_size_bytes"] += len(payload)

    async def _add_system_metadata(
        self,
        metadata: AnalyticsMetadata,
        duration: float,
        frame_indices: list[int] | None = None
    ) -> None:
        """Add system metrics metadata to the track buffer."""
        config = self.track_configs[MetadataTrackType.SYSTEM_METRICS]

        system_data = {
            "timestamp": metadata.timestamp,
            "camera_id": metadata.camera_id,
            "metrics": metadata.system_metrics,
        }

        # Encode based on configured format
        payload = self.encoder.encode_json(system_data, config.compression_enabled)

        timed_metadata = TimedMetadata(
            timestamp=metadata.timestamp,
            duration=duration,
            payload=payload,
            track_type=MetadataTrackType.SYSTEM_METRICS,
            format=config.format,
            frame_indices=frame_indices or []
        )

        self.metadata_buffers[MetadataTrackType.SYSTEM_METRICS].append(timed_metadata)
        self.metrics["total_metadata_size_bytes"] += len(payload)

    async def _mux_metadata_packets(
        self,
        output_container: av.container.OutputContainer,
        metadata_streams: dict[MetadataTrackType, av.stream.Stream],
        fragment_duration: float
    ) -> None:
        """Mux metadata packets into the output container."""
        try:
            for track_type, stream in metadata_streams.items():
                metadata_list = self.metadata_buffers[track_type]

                for timed_metadata in metadata_list:
                    # Create metadata packet
                    packet = av.Packet(timed_metadata.payload)
                    packet.stream = stream

                    # Set timing information
                    packet.pts = int(timed_metadata.timestamp * stream.time_base.denominator)
                    packet.dts = packet.pts
                    packet.duration = int(timed_metadata.duration * stream.time_base.denominator)

                    # Mux the packet
                    output_container.mux(packet)

                logger.debug("Metadata packets muxed",
                           track_type=track_type.value,
                           packet_count=len(metadata_list))

        except Exception as e:
            logger.error("Failed to mux metadata packets", error=str(e))
            raise

    def get_metadata_summary(self) -> dict[str, Any]:
        """Get summary of buffered metadata.
        
        Returns:
            Summary of metadata tracks and buffers
        """
        summary = {
            "enabled_tracks": list(self.metadata_buffers.keys()),
            "buffer_counts": {
                track_type.value: len(metadata_list)
                for track_type, metadata_list in self.metadata_buffers.items()
            },
            "total_size_bytes": sum(
                sum(len(meta.payload) for meta in metadata_list)
                for metadata_list in self.metadata_buffers.values()
            ),
            "metrics": self.metrics.copy(),
        }

        return summary

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        return self.metrics.copy()

    def _clear_metadata_buffers(self) -> None:
        """Clear all metadata buffers after successful embedding."""
        for track_type in self.metadata_buffers:
            self.metadata_buffers[track_type].clear()

        logger.debug("Metadata buffers cleared")

    def configure_track(self, track_type: MetadataTrackType, config: MetadataTrackConfig) -> None:
        """Update configuration for a metadata track.
        
        Args:
            track_type: Type of metadata track
            config: New configuration
        """
        self.track_configs[track_type] = config

        if config.enabled and track_type not in self.metadata_buffers:
            self.metadata_buffers[track_type] = []
        elif not config.enabled and track_type in self.metadata_buffers:
            del self.metadata_buffers[track_type]

        logger.info("Metadata track configuration updated",
                   track_type=track_type.value,
                   enabled=config.enabled)

    def get_track_config(self, track_type: MetadataTrackType) -> MetadataTrackConfig | None:
        """Get configuration for a metadata track.
        
        Args:
            track_type: Type of metadata track
            
        Returns:
            Track configuration or None if not found
        """
        return self.track_configs.get(track_type)


# Factory function for creating metadata track managers
def create_metadata_track_manager(
    enabled_tracks: list[MetadataTrackType] | None = None
) -> MetadataTrackManager:
    """Create a metadata track manager with default configurations.
    
    Args:
        enabled_tracks: Optional list of tracks to enable
        
    Returns:
        Configured MetadataTrackManager instance
    """
    if enabled_tracks is None:
        enabled_tracks = [
            MetadataTrackType.DETECTION_RESULTS,
            MetadataTrackType.TRAFFIC_ANALYTICS,
            MetadataTrackType.SYSTEM_METRICS,
        ]

    configs = []
    for track_type in enabled_tracks:
        if track_type == MetadataTrackType.DETECTION_RESULTS:
            config = MetadataTrackConfig(
                track_type=track_type,
                format=MetadataFormat.KLV,
                compression_enabled=True,
                max_payload_size_kb=128,
            )
        elif track_type == MetadataTrackType.TRAFFIC_ANALYTICS:
            config = MetadataTrackConfig(
                track_type=track_type,
                format=MetadataFormat.JSON,
                compression_enabled=True,
                max_payload_size_kb=64,
            )
        else:
            config = MetadataTrackConfig(
                track_type=track_type,
                format=MetadataFormat.JSON,
                compression_enabled=False,
                max_payload_size_kb=32,
            )

        configs.append(config)

    return MetadataTrackManager(configs)
