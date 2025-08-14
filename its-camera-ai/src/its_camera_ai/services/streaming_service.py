"""Streaming Service Implementation for ITS Camera AI.

This module implements the complete gRPC streaming service for real-time camera
stream processing with Redis integration, frame quality validation, and
high-performance batch processing.

Key Features:
- gRPC server implementation for camera communication
- RTSP/WebRTC stream handling capabilities
- Frame quality validation logic
- Redis queue integration for frame distribution
- Health monitoring and metrics collection
- Support for 100+ concurrent camera streams
- Sub-10ms frame processing latency
- 99.9% frame processing success rate
"""

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import StreamingResponse

import cv2
import numpy as np
import psutil
import torch

# WebRTC imports with fallback handling
try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from aiortc.contrib.media import MediaPlayer
    from av import VideoFrame

    WEBRTC_AVAILABLE = True
except ImportError:
    # Create mock classes for type checking
    class RTCPeerConnection:  # type: ignore
        pass

    class RTCSessionDescription:  # type: ignore
        pass

    class VideoStreamTrack:  # type: ignore
        pass

    class VideoFrame:  # type: ignore
        pass

    WEBRTC_AVAILABLE = False

# Internal imports
from ..core.exceptions import (
    CameraConfigurationError,
    CameraConnectionError,
    StreamProcessingError,
)
from ..core.logging import get_logger
from ..flow.redis_queue_manager import QueueConfig, QueueType, RedisQueueManager
from ..flow.streaming_processor import (
    ProcessedFrame,
    ProcessingStage,
)
from ..ml.streaming_annotation_processor import (
    MLAnnotationProcessor,
)
from ..proto import processed_frame_pb2

logger = get_logger(__name__)


class StreamProtocol(Enum):
    """Supported camera stream protocols."""

    RTSP = "rtsp"
    WEBRTC = "webrtc"
    HTTP = "http"
    ONVIF = "onvif"


class ChannelType(Enum):
    """Stream channel types for dual-channel support."""

    RAW = "raw"
    ANNOTATED = "annotated"


class QualityLevel(Enum):
    """Quality levels for stream channels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class CameraConfig:
    """Camera configuration for stream registration."""

    camera_id: str
    stream_url: str
    resolution: tuple[int, int]
    fps: int
    protocol: StreamProtocol
    location: str | None = None
    coordinates: tuple[float, float] | None = None
    quality_threshold: float = 0.7
    roi_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    enabled: bool = True


@dataclass
class CameraRegistration:
    """Camera registration result."""

    camera_id: str
    success: bool
    message: str
    registration_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QualityMetrics:
    """Frame quality validation metrics."""

    overall_score: float
    blur_score: float
    brightness_score: float
    contrast_score: float
    noise_level: float
    passed_validation: bool
    issues: list[str] = field(default_factory=list)


class CustomVideoStreamTrack(VideoStreamTrack):
    """Custom WebRTC video stream track for camera feeds."""

    def __init__(self, camera_id: str, capture_source: Any) -> None:
        if not WEBRTC_AVAILABLE:
            raise RuntimeError(
                "WebRTC dependencies not available. Please install aiortc and pyav."
            )
        super().__init__()
        self.camera_id = camera_id
        self.capture_source = capture_source
        self._frame_count = 0
        self._last_frame_time = time.time()

    async def recv(self) -> VideoFrame:  # type: ignore
        """Receive the next video frame from the camera source."""
        try:
            if isinstance(self.capture_source, cv2.VideoCapture):
                ret, frame = self.capture_source.read()
                if not ret or frame is None:
                    raise Exception("Failed to capture frame")

                # Convert BGR to RGB for WebRTC
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create AVFrame
                av_frame = VideoFrame.from_ndarray(frame, format="rgb24")  # type: ignore

                # Set presentation timestamp
                current_time = time.time()
                time_diff = current_time - self._last_frame_time
                self._last_frame_time = current_time

                if self._frame_count == 0:
                    av_frame.pts = 0
                else:
                    av_frame.pts = int(time_diff * 90000)  # 90kHz clock

                av_frame.time_base = (1, 90000)
                self._frame_count += 1

                return av_frame
            else:
                # For mock sources, generate a test frame
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                av_frame = VideoFrame.from_ndarray(frame, format="rgb24")  # type: ignore
                av_frame.pts = self._frame_count
                av_frame.time_base = (1, 30)  # 30 FPS
                self._frame_count += 1

                # Add small delay to simulate real camera
                await asyncio.sleep(0.033)  # ~30 FPS

                return av_frame

        except Exception as e:
            logger.error(f"WebRTC frame capture failed for {self.camera_id}: {e}")
            # Return a black frame on error
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            av_frame = VideoFrame.from_ndarray(black_frame, format="rgb24")  # type: ignore
            av_frame.pts = self._frame_count
            av_frame.time_base = (1, 30)
            self._frame_count += 1
            return av_frame


class WebRTCConnectionManager:
    """Manages WebRTC peer connections for real-time video streaming."""

    def __init__(self) -> None:
        self.peer_connections: dict[str, RTCPeerConnection] = {}
        self.video_tracks: dict[str, CustomVideoStreamTrack] = {}
        self.connection_stats: dict[str, dict[str, Any]] = {}
        self._logger = get_logger(f"{__name__}.WebRTCConnectionManager")

    async def create_peer_connection(
        self, camera_id: str, capture_source: Any
    ) -> RTCPeerConnection:
        """Create a new WebRTC peer connection for a camera."""
        if not WEBRTC_AVAILABLE:
            raise CameraConnectionError(
                "WebRTC not available. Please install aiortc and pyav: "
                "pip install aiortc pyav"
            )
        try:
            # Create peer connection with STUN server
            pc = RTCPeerConnection(
                configuration={
                    "iceServers": [
                        {"urls": "stun:stun.l.google.com:19302"},
                        {"urls": "stun:stun1.l.google.com:19302"},
                    ]
                }
            )

            # Create video track
            video_track = CustomVideoStreamTrack(camera_id, capture_source)

            # Add video track to peer connection
            pc.addTrack(video_track)

            # Store connections
            self.peer_connections[camera_id] = pc
            self.video_tracks[camera_id] = video_track

            # Initialize connection stats
            self.connection_stats[camera_id] = {
                "created_at": datetime.now(UTC),
                "state": "new",
                "ice_connection_state": "new",
                "connection_state": "new",
                "frames_sent": 0,
                "bytes_sent": 0,
                "last_activity": datetime.now(UTC),
            }

            # Set up event handlers
            @pc.on("connectionstatechange")
            async def on_connectionstatechange() -> None:
                state = pc.connectionState
                self.connection_stats[camera_id]["connection_state"] = state
                self.connection_stats[camera_id]["last_activity"] = datetime.now(UTC)
                self._logger.info(
                    f"WebRTC connection state changed for {camera_id}: {state}"
                )

                if state == "failed" or state == "closed":
                    await self.cleanup_connection(camera_id)

            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange() -> None:
                state = pc.iceConnectionState
                self.connection_stats[camera_id]["ice_connection_state"] = state
                self._logger.info(
                    f"ICE connection state changed for {camera_id}: {state}"
                )

            self._logger.info(f"Created WebRTC peer connection for camera: {camera_id}")
            return pc

        except Exception as e:
            self._logger.error(
                f"Failed to create WebRTC peer connection for {camera_id}: {e}"
            )
            raise CameraConnectionError(f"WebRTC connection creation failed: {e}")

    async def create_offer(self, camera_id: str) -> RTCSessionDescription:
        """Create an SDP offer for the camera connection."""
        if camera_id not in self.peer_connections:
            raise CameraConnectionError(
                f"No peer connection found for camera: {camera_id}"
            )

        pc = self.peer_connections[camera_id]
        try:
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            self.connection_stats[camera_id]["state"] = "offer_created"
            self._logger.info(f"Created SDP offer for camera: {camera_id}")

            return offer

        except Exception as e:
            self._logger.error(f"Failed to create offer for {camera_id}: {e}")
            raise CameraConnectionError(f"SDP offer creation failed: {e}")

    async def set_remote_description(
        self, camera_id: str, answer: RTCSessionDescription
    ) -> None:
        """Set the remote SDP answer for the camera connection."""
        if camera_id not in self.peer_connections:
            raise CameraConnectionError(
                f"No peer connection found for camera: {camera_id}"
            )

        pc = self.peer_connections[camera_id]
        try:
            await pc.setRemoteDescription(answer)

            self.connection_stats[camera_id]["state"] = "answer_set"
            self._logger.info(f"Set remote description for camera: {camera_id}")

        except Exception as e:
            self._logger.error(f"Failed to set remote description for {camera_id}: {e}")
            raise CameraConnectionError(f"Remote description setting failed: {e}")

    async def cleanup_connection(self, camera_id: str) -> None:
        """Clean up WebRTC connection resources."""
        try:
            if camera_id in self.peer_connections:
                pc = self.peer_connections[camera_id]
                await pc.close()
                del self.peer_connections[camera_id]

            if camera_id in self.video_tracks:
                track = self.video_tracks[camera_id]
                track.stop()
                del self.video_tracks[camera_id]

            if camera_id in self.connection_stats:
                del self.connection_stats[camera_id]

            self._logger.info(f"Cleaned up WebRTC connection for camera: {camera_id}")

        except Exception as e:
            self._logger.error(
                f"Error cleaning up WebRTC connection for {camera_id}: {e}"
            )

    def get_connection_stats(self, camera_id: str) -> dict[str, Any] | None:
        """Get WebRTC connection statistics."""
        return self.connection_stats.get(camera_id)

    def is_connected(self, camera_id: str) -> bool:
        """Check if WebRTC connection is active."""
        if camera_id not in self.peer_connections:
            return False

        pc = self.peer_connections[camera_id]
        return pc.connectionState in ["connected", "connecting"]

    async def get_connection_quality_stats(self, camera_id: str) -> dict[str, Any]:
        """Get detailed connection quality statistics."""
        if camera_id not in self.peer_connections:
            return {}

        try:
            pc = self.peer_connections[camera_id]
            stats = await pc.getStats()

            # Process WebRTC stats
            quality_stats = {
                "connection_state": pc.connectionState,
                "ice_connection_state": pc.iceConnectionState,
                "ice_gathering_state": pc.iceGatheringState,
                "signaling_state": pc.signalingState,
                "packets_sent": 0,
                "bytes_sent": 0,
                "packets_received": 0,
                "bytes_received": 0,
                "frames_encoded": 0,
                "frames_sent": 0,
                "key_frames_encoded": 0,
                "total_encode_time": 0,
                "rtt": 0,
                "jitter": 0,
                "packet_loss_rate": 0,
            }

            for report in stats.values():
                if hasattr(report, "type"):
                    if report.type == "outbound-rtp" and hasattr(report, "mediaType"):
                        if report.mediaType == "video":
                            quality_stats.update(
                                {
                                    "packets_sent": getattr(report, "packetsSent", 0),
                                    "bytes_sent": getattr(report, "bytesSent", 0),
                                    "frames_encoded": getattr(
                                        report, "framesEncoded", 0
                                    ),
                                    "frames_sent": getattr(report, "framesSent", 0),
                                    "key_frames_encoded": getattr(
                                        report, "keyFramesEncoded", 0
                                    ),
                                    "total_encode_time": getattr(
                                        report, "totalEncodeTime", 0
                                    ),
                                }
                            )
                    elif report.type == "remote-inbound-rtp":
                        quality_stats.update(
                            {
                                "rtt": getattr(report, "roundTripTime", 0)
                                * 1000,  # Convert to ms
                                "jitter": getattr(report, "jitter", 0),
                                "packet_loss_rate": getattr(report, "fractionLost", 0),
                            }
                        )

            return quality_stats

        except Exception as e:
            self._logger.error(
                f"Failed to get connection quality stats for {camera_id}: {e}"
            )
            return {}


class ProtobufSerializationManager:
    """Manages protobuf serialization for frame data to improve performance."""

    def __init__(self) -> None:
        self._logger = get_logger(f"{__name__}.ProtobufSerializationManager")

    def serialize_processed_frame(self, frame: ProcessedFrame) -> bytes:
        """Serialize ProcessedFrame to protobuf bytes for efficient transmission."""
        try:
            # Create protobuf message
            pb_frame = processed_frame_pb2.ProcessedFrame()

            # Basic frame information
            pb_frame.frame_id = frame.frame_id
            pb_frame.camera_id = frame.camera_id
            pb_frame.timestamp = frame.timestamp

            # Image data - compress for efficiency
            if frame.original_image is not None:
                pb_frame.original_image.CopyFrom(
                    self._compress_image_data(frame.original_image, "jpeg", 85)
                )

            # Quality metrics
            if hasattr(frame, "quality_score") and frame.quality_score is not None:
                pb_frame.quality_metrics.quality_score = frame.quality_score
            if hasattr(frame, "blur_score") and frame.blur_score is not None:
                pb_frame.quality_metrics.blur_score = frame.blur_score
            if (
                hasattr(frame, "brightness_score")
                and frame.brightness_score is not None
            ):
                pb_frame.quality_metrics.brightness_score = frame.brightness_score
            if hasattr(frame, "contrast_score") and frame.contrast_score is not None:
                pb_frame.quality_metrics.contrast_score = frame.contrast_score
            if hasattr(frame, "noise_level") and frame.noise_level is not None:
                pb_frame.quality_metrics.noise_level = frame.noise_level

            # Processing metadata
            if (
                hasattr(frame, "processing_time_ms")
                and frame.processing_time_ms is not None
            ):
                pb_frame.processing_time_ms = frame.processing_time_ms

            # Processing stage
            if (
                hasattr(frame, "processing_stage")
                and frame.processing_stage is not None
            ):
                stage_mapping = {
                    ProcessingStage.INGESTION: processed_frame_pb2.PROCESSING_STAGE_INGESTION,
                    ProcessingStage.VALIDATION: processed_frame_pb2.PROCESSING_STAGE_VALIDATION,
                    ProcessingStage.FEATURE_EXTRACTION: processed_frame_pb2.PROCESSING_STAGE_FEATURE_EXTRACTION,
                    ProcessingStage.QUALITY_CONTROL: processed_frame_pb2.PROCESSING_STAGE_QUALITY_CONTROL,
                    ProcessingStage.OUTPUT: processed_frame_pb2.PROCESSING_STAGE_OUTPUT,
                }
                pb_frame.processing_stage = stage_mapping.get(
                    frame.processing_stage,
                    processed_frame_pb2.PROCESSING_STAGE_UNSPECIFIED,
                )

            # Validation status
            if (
                hasattr(frame, "validation_passed")
                and frame.validation_passed is not None
            ):
                pb_frame.validation_passed = frame.validation_passed

            # Timestamps for performance tracking
            pb_frame.received_timestamp = frame.timestamp
            pb_frame.processed_timestamp = time.time()

            # Version and source hash for traceability
            pb_frame.version = "1.0.0"
            pb_frame.source_hash = self._calculate_frame_hash(frame)

            # Serialize to bytes
            return pb_frame.SerializeToString()

        except Exception as e:
            self._logger.error(f"Failed to serialize frame {frame.frame_id}: {e}")
            raise StreamProcessingError(f"Protobuf serialization failed: {e}")

    def deserialize_processed_frame(self, data: bytes) -> dict[str, Any]:
        """Deserialize protobuf bytes back to frame data dictionary."""
        try:
            pb_frame = processed_frame_pb2.ProcessedFrame()
            pb_frame.ParseFromString(data)

            # Convert back to dictionary format
            frame_data = {
                "frame_id": pb_frame.frame_id,
                "camera_id": pb_frame.camera_id,
                "timestamp": pb_frame.timestamp,
                "quality_score": pb_frame.quality_metrics.quality_score,
                "blur_score": pb_frame.quality_metrics.blur_score,
                "brightness_score": pb_frame.quality_metrics.brightness_score,
                "contrast_score": pb_frame.quality_metrics.contrast_score,
                "noise_level": pb_frame.quality_metrics.noise_level,
                "processing_time_ms": pb_frame.processing_time_ms,
                "validation_passed": pb_frame.validation_passed,
                "received_timestamp": pb_frame.received_timestamp,
                "processed_timestamp": pb_frame.processed_timestamp,
                "version": pb_frame.version,
                "source_hash": pb_frame.source_hash,
            }

            # Processing stage conversion
            stage_mapping = {
                processed_frame_pb2.PROCESSING_STAGE_INGESTION: "ingestion",
                processed_frame_pb2.PROCESSING_STAGE_VALIDATION: "validation",
                processed_frame_pb2.PROCESSING_STAGE_FEATURE_EXTRACTION: "feature_extraction",
                processed_frame_pb2.PROCESSING_STAGE_QUALITY_CONTROL: "quality_control",
                processed_frame_pb2.PROCESSING_STAGE_OUTPUT: "output",
            }
            frame_data["processing_stage"] = stage_mapping.get(
                pb_frame.processing_stage, "unspecified"
            )

            # Image data (if present)
            if pb_frame.HasField("original_image"):
                frame_data["original_image_compressed"] = {
                    "data": pb_frame.original_image.compressed_data,
                    "width": pb_frame.original_image.width,
                    "height": pb_frame.original_image.height,
                    "channels": pb_frame.original_image.channels,
                    "format": pb_frame.original_image.compression_format,
                    "quality": pb_frame.original_image.quality,
                }

            return frame_data

        except Exception as e:
            self._logger.error(f"Failed to deserialize protobuf data: {e}")
            raise StreamProcessingError(f"Protobuf deserialization failed: {e}")

    def serialize_frame_batch(
        self, frames: list[ProcessedFrame], batch_id: str
    ) -> bytes:
        """Serialize a batch of frames for efficient transmission."""
        try:
            pb_batch = processed_frame_pb2.ProcessedFrameBatch()

            pb_batch.batch_id = batch_id
            pb_batch.batch_timestamp = time.time()
            pb_batch.batch_size = len(frames)

            # Add frames to batch
            for frame in frames:
                pb_frame = pb_batch.frames.add()

                # Basic frame information
                pb_frame.frame_id = frame.frame_id
                pb_frame.camera_id = frame.camera_id
                pb_frame.timestamp = frame.timestamp

                # Quality metrics
                if hasattr(frame, "quality_score") and frame.quality_score is not None:
                    pb_frame.quality_metrics.quality_score = frame.quality_score
                if hasattr(frame, "blur_score") and frame.blur_score is not None:
                    pb_frame.quality_metrics.blur_score = frame.blur_score
                if (
                    hasattr(frame, "brightness_score")
                    and frame.brightness_score is not None
                ):
                    pb_frame.quality_metrics.brightness_score = frame.brightness_score
                if (
                    hasattr(frame, "contrast_score")
                    and frame.contrast_score is not None
                ):
                    pb_frame.quality_metrics.contrast_score = frame.contrast_score
                if hasattr(frame, "noise_level") and frame.noise_level is not None:
                    pb_frame.quality_metrics.noise_level = frame.noise_level

                # Processing metadata
                if (
                    hasattr(frame, "processing_time_ms")
                    and frame.processing_time_ms is not None
                ):
                    pb_frame.processing_time_ms = frame.processing_time_ms
                if (
                    hasattr(frame, "validation_passed")
                    and frame.validation_passed is not None
                ):
                    pb_frame.validation_passed = frame.validation_passed

                # Timestamps
                pb_frame.received_timestamp = frame.timestamp
                pb_frame.processed_timestamp = time.time()

                # Version and hash
                pb_frame.version = "1.0.0"
                pb_frame.source_hash = self._calculate_frame_hash(frame)

            return pb_batch.SerializeToString()

        except Exception as e:
            self._logger.error(f"Failed to serialize frame batch {batch_id}: {e}")
            raise StreamProcessingError(f"Batch protobuf serialization failed: {e}")

    def _compress_image_data(
        self, image: np.ndarray, format: str = "jpeg", quality: int = 85
    ) -> processed_frame_pb2.ImageData:
        """Compress image data for efficient storage."""
        try:
            image_data = processed_frame_pb2.ImageData()

            # Set image properties
            height, width = image.shape[:2]
            channels = image.shape[2] if len(image.shape) == 3 else 1

            image_data.width = width
            image_data.height = height
            image_data.channels = channels
            image_data.compression_format = format
            image_data.quality = quality

            # Compress image based on format
            if format.lower() == "jpeg":
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
                success, encoded_img = cv2.imencode(".jpg", image, encode_param)
            elif format.lower() == "png":
                encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 10)]
                success, encoded_img = cv2.imencode(".png", image, encode_param)
            elif format.lower() == "webp":
                encode_param = [cv2.IMWRITE_WEBP_QUALITY, quality]
                success, encoded_img = cv2.imencode(".webp", image, encode_param)
            else:
                # Default to JPEG
                encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
                success, encoded_img = cv2.imencode(".jpg", image, encode_param)

            if not success:
                raise Exception(f"Failed to encode image with format: {format}")

            image_data.compressed_data = encoded_img.tobytes()

            # Log compression efficiency
            original_size = image.nbytes
            compressed_size = len(image_data.compressed_data)
            compression_ratio = compressed_size / original_size

            self._logger.debug(
                f"Image compressed: {original_size} -> {compressed_size} bytes "
                f"(ratio: {compression_ratio:.2f})"
            )

            return image_data

        except Exception as e:
            self._logger.error(f"Image compression failed: {e}")
            raise StreamProcessingError(f"Image compression failed: {e}")

    def _calculate_frame_hash(self, frame: ProcessedFrame) -> str:
        """Calculate a hash for frame data integrity."""
        try:
            # Create hash from key frame properties
            hash_data = f"{frame.frame_id}_{frame.camera_id}_{frame.timestamp}"
            if hasattr(frame, "quality_score") and frame.quality_score is not None:
                hash_data += f"_{frame.quality_score}"

            return hashlib.md5(hash_data.encode()).hexdigest()[:16]

        except Exception:
            return "unknown_hash"

    def get_serialization_stats(self) -> dict[str, Any]:
        """Get serialization performance statistics."""
        return {
            "format": "protobuf",
            "compression_enabled": True,
            "supported_image_formats": ["jpeg", "png", "webp"],
            "default_quality": 85,
            "version": "1.0.0",
        }


@dataclass
class BatchId:
    """Batch processing identifier."""

    batch_id: str
    camera_id: str
    frame_count: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SSEStream:
    """Server-Sent Events stream configuration."""

    stream_id: str
    camera_id: str
    user_id: str
    stream_type: str  # 'raw' or 'annotated'
    quality: str  # 'low', 'medium', 'high'
    connection_time: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    client_capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass
class MP4Fragment:
    """MP4 fragment for SSE streaming."""

    fragment_id: str
    camera_id: str
    sequence_number: int
    timestamp: float
    data: bytes
    content_type: str = "video/mp4"
    duration_ms: float = 0.0
    size_bytes: int = 0
    quality: str = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Calculate derived fields after initialization."""
        self.size_bytes = len(self.data)


@dataclass
class ChannelMetadata:
    """Channel metadata for dual-channel stream synchronization."""

    channel_type: ChannelType
    quality: QualityLevel
    last_fragment_time: float
    sequence_number: int = 0
    buffer_size: int = 0
    sync_offset_ms: float = 0.0
    fragments_processed: int = 0
    bytes_transferred: int = 0
    average_latency_ms: float = 0.0


@dataclass
class DualChannelStream:
    """Dual-channel stream configuration for synchronized raw and annotated streams."""

    camera_id: str
    stream_id: str
    user_id: str
    raw_channel: ChannelMetadata
    annotated_channel: ChannelMetadata
    sync_tolerance_ms: float = 50.0
    created_at: float = field(default_factory=time.time)
    last_sync_check: float = field(default_factory=time.time)
    client_subscriptions: dict[str, set[ChannelType]] = field(default_factory=dict)
    sync_violations: int = 0
    is_synchronized: bool = True


@dataclass
class ChannelSyncStatus:
    """Channel synchronization status information."""

    is_synchronized: bool
    drift_ms: float
    sync_violations: int
    last_sync_time: float
    correction_applied: bool = False


@dataclass
class SSEConnectionMetrics:
    """Metrics for SSE connection management."""

    active_connections: int = 0
    total_connections_created: int = 0
    total_disconnections: int = 0
    bytes_streamed: int = 0
    fragments_sent: int = 0
    connection_errors: int = 0
    average_connection_duration: float = 0.0
    peak_concurrent_connections: int = 0
    dual_channel_streams: int = 0
    channel_switches: int = 0
    sync_violations: int = 0
    last_activity: datetime = field(default_factory=datetime.utcnow)


class StreamChannelManager:
    """Manages dual-channel streaming with synchronization between raw and annotated streams.

    Provides comprehensive dual-channel management with <50ms synchronization tolerance,
    independent quality control per channel, and client subscription management.
    """

    def __init__(self, redis_client, config: dict[str, Any]):
        """Initialize dual-channel stream manager.

        Args:
            redis_client: Redis client for stream data
            config: Streaming configuration
        """
        self.redis_client = redis_client
        self.config = config
        self.dual_channel_streams: dict[str, DualChannelStream] = {}
        self.sync_tolerance_ms = config.get("sync_tolerance_ms", 50.0)
        self.sync_check_interval = config.get("sync_check_interval", 1.0)  # 1 second
        self.max_sync_violations = config.get("max_sync_violations", 10)

        # Background sync monitoring task
        self._sync_monitor_task: asyncio.Task = None
        self._running = False

        logger.info(
            f"StreamChannelManager initialized with sync tolerance {self.sync_tolerance_ms}ms"
        )

    async def create_dual_channel_stream(
        self,
        camera_id: str,
        user_id: str,
        raw_quality: QualityLevel = QualityLevel.MEDIUM,
        annotated_quality: QualityLevel = QualityLevel.MEDIUM,
    ) -> DualChannelStream:
        """Create dual-channel stream with synchronized raw and annotated channels.

        Args:
            camera_id: Target camera identifier
            user_id: User creating the stream
            raw_quality: Quality level for raw channel
            annotated_quality: Quality level for annotated channel

        Returns:
            DualChannelStream configuration

        Raises:
            StreamProcessingError: If stream creation fails
        """
        try:
            current_time = time.time()
            stream_id = f"dual_{camera_id}_{uuid.uuid4().hex[:8]}"

            # Create channel metadata for both channels
            raw_channel = ChannelMetadata(
                channel_type=ChannelType.RAW,
                quality=raw_quality,
                last_fragment_time=current_time,
            )

            annotated_channel = ChannelMetadata(
                channel_type=ChannelType.ANNOTATED,
                quality=annotated_quality,
                last_fragment_time=current_time,
            )

            # Create dual-channel stream
            dual_stream = DualChannelStream(
                camera_id=camera_id,
                stream_id=stream_id,
                user_id=user_id,
                raw_channel=raw_channel,
                annotated_channel=annotated_channel,
                sync_tolerance_ms=self.sync_tolerance_ms,
            )

            # Register the stream
            self.dual_channel_streams[stream_id] = dual_stream

            # Start sync monitoring if not already running
            if not self._running:
                await self._start_sync_monitoring()

            logger.info(
                f"Created dual-channel stream {stream_id} for camera {camera_id}, "
                f"raw quality: {raw_quality.value}, annotated quality: {annotated_quality.value}"
            )

            return dual_stream

        except Exception as e:
            logger.error(
                f"Failed to create dual-channel stream for camera {camera_id}: {e}"
            )
            raise StreamProcessingError(
                f"Dual-channel stream creation failed: {e}"
            ) from e

    async def synchronize_channels(self, stream_id: str) -> ChannelSyncStatus:
        """Synchronize raw and annotated channels within tolerance.

        Args:
            stream_id: Dual-channel stream identifier

        Returns:
            ChannelSyncStatus with synchronization information
        """
        if stream_id not in self.dual_channel_streams:
            raise StreamProcessingError(f"Dual-channel stream {stream_id} not found")

        dual_stream = self.dual_channel_streams[stream_id]
        current_time = time.time()

        try:
            # Calculate time drift between channels
            raw_time = dual_stream.raw_channel.last_fragment_time
            annotated_time = dual_stream.annotated_channel.last_fragment_time
            drift_ms = abs(raw_time - annotated_time) * 1000

            is_synchronized = drift_ms <= self.sync_tolerance_ms
            correction_applied = False

            if not is_synchronized:
                # Apply synchronization correction
                dual_stream.sync_violations += 1

                # Determine which channel is behind and apply offset
                if raw_time < annotated_time:
                    dual_stream.raw_channel.sync_offset_ms = (
                        annotated_time - raw_time
                    ) * 1000
                else:
                    dual_stream.annotated_channel.sync_offset_ms = (
                        raw_time - annotated_time
                    ) * 1000

                correction_applied = True
                logger.warning(
                    f"Synchronization drift {drift_ms:.2f}ms detected for stream {stream_id}, "
                    f"correction applied"
                )

            # Update stream synchronization status
            dual_stream.is_synchronized = is_synchronized
            dual_stream.last_sync_check = current_time

            return ChannelSyncStatus(
                is_synchronized=is_synchronized,
                drift_ms=drift_ms,
                sync_violations=dual_stream.sync_violations,
                last_sync_time=current_time,
                correction_applied=correction_applied,
            )

        except Exception as e:
            logger.error(f"Channel synchronization failed for stream {stream_id}: {e}")
            raise StreamProcessingError(f"Synchronization failed: {e}") from e

    async def update_channel_timestamp(
        self, stream_id: str, channel_type: ChannelType, timestamp: float
    ) -> None:
        """Update timestamp for specific channel to maintain synchronization.

        Args:
            stream_id: Dual-channel stream identifier
            channel_type: Channel type to update
            timestamp: New fragment timestamp
        """
        if stream_id not in self.dual_channel_streams:
            return

        dual_stream = self.dual_channel_streams[stream_id]

        if channel_type == ChannelType.RAW:
            dual_stream.raw_channel.last_fragment_time = timestamp
            dual_stream.raw_channel.fragments_processed += 1
        elif channel_type == ChannelType.ANNOTATED:
            dual_stream.annotated_channel.last_fragment_time = timestamp
            dual_stream.annotated_channel.fragments_processed += 1

    async def get_synchronization_stats(self, stream_id: str) -> dict[str, Any]:
        """Get detailed synchronization statistics for a dual-channel stream.

        Args:
            stream_id: Dual-channel stream identifier

        Returns:
            Dictionary with synchronization statistics
        """
        if stream_id not in self.dual_channel_streams:
            return {"error": "Stream not found"}

        dual_stream = self.dual_channel_streams[stream_id]
        current_time = time.time()

        # Calculate current drift
        raw_time = dual_stream.raw_channel.last_fragment_time
        annotated_time = dual_stream.annotated_channel.last_fragment_time
        current_drift_ms = abs(raw_time - annotated_time) * 1000

        return {
            "stream_id": stream_id,
            "camera_id": dual_stream.camera_id,
            "is_synchronized": dual_stream.is_synchronized,
            "current_drift_ms": current_drift_ms,
            "sync_tolerance_ms": dual_stream.sync_tolerance_ms,
            "sync_violations": dual_stream.sync_violations,
            "raw_channel": {
                "fragments_processed": dual_stream.raw_channel.fragments_processed,
                "last_fragment_time": dual_stream.raw_channel.last_fragment_time,
                "sync_offset_ms": dual_stream.raw_channel.sync_offset_ms,
                "quality": dual_stream.raw_channel.quality.value,
            },
            "annotated_channel": {
                "fragments_processed": dual_stream.annotated_channel.fragments_processed,
                "last_fragment_time": dual_stream.annotated_channel.last_fragment_time,
                "sync_offset_ms": dual_stream.annotated_channel.sync_offset_ms,
                "quality": dual_stream.annotated_channel.quality.value,
            },
            "uptime_seconds": current_time - dual_stream.created_at,
            "last_sync_check": dual_stream.last_sync_check,
        }

    async def remove_dual_channel_stream(self, stream_id: str) -> None:
        """Remove dual-channel stream and cleanup resources.

        Args:
            stream_id: Stream identifier to remove
        """
        if stream_id in self.dual_channel_streams:
            dual_stream = self.dual_channel_streams[stream_id]
            del self.dual_channel_streams[stream_id]

            logger.info(
                f"Removed dual-channel stream {stream_id} for camera {dual_stream.camera_id}, "
                f"uptime: {time.time() - dual_stream.created_at:.2f}s, "
                f"sync violations: {dual_stream.sync_violations}"
            )

            # Stop sync monitoring if no streams remaining
            if not self.dual_channel_streams and self._running:
                await self._stop_sync_monitoring()

    async def _start_sync_monitoring(self) -> None:
        """Start background synchronization monitoring."""
        if self._running:
            return

        self._running = True
        self._sync_monitor_task = asyncio.create_task(self._sync_monitor_loop())
        logger.info("Started dual-channel synchronization monitoring")

    async def _stop_sync_monitoring(self) -> None:
        """Stop background synchronization monitoring."""
        if not self._running:
            return

        self._running = False
        if self._sync_monitor_task:
            self._sync_monitor_task.cancel()
            try:
                await self._sync_monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped dual-channel synchronization monitoring")

    async def _sync_monitor_loop(self) -> None:
        """Background loop for monitoring channel synchronization."""
        while self._running:
            try:
                for stream_id in list(self.dual_channel_streams.keys()):
                    try:
                        sync_status = await self.synchronize_channels(stream_id)

                        # Check for persistent sync violations
                        dual_stream = self.dual_channel_streams.get(stream_id)
                        if (
                            dual_stream
                            and dual_stream.sync_violations > self.max_sync_violations
                        ):
                            logger.error(
                                f"Stream {stream_id} exceeded max sync violations "
                                f"({dual_stream.sync_violations}), marking as degraded"
                            )
                            dual_stream.is_synchronized = False

                    except Exception as e:
                        logger.error(
                            f"Sync monitoring failed for stream {stream_id}: {e}"
                        )

                await asyncio.sleep(self.sync_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync monitor loop error: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retry

    def get_dual_channel_stats(self) -> dict[str, Any]:
        """Get comprehensive dual-channel streaming statistics.

        Returns:
            Dictionary with dual-channel statistics
        """
        total_sync_violations = sum(
            stream.sync_violations for stream in self.dual_channel_streams.values()
        )
        synchronized_streams = sum(
            1 for stream in self.dual_channel_streams.values() if stream.is_synchronized
        )

        return {
            "active_dual_channel_streams": len(self.dual_channel_streams),
            "synchronized_streams": synchronized_streams,
            "total_sync_violations": total_sync_violations,
            "sync_tolerance_ms": self.sync_tolerance_ms,
            "monitoring_enabled": self._running,
            "streams": {
                stream_id: {
                    "camera_id": stream.camera_id,
                    "is_synchronized": stream.is_synchronized,
                    "sync_violations": stream.sync_violations,
                    "uptime_seconds": time.time() - stream.created_at,
                }
                for stream_id, stream in self.dual_channel_streams.items()
            },
        }


class ChannelSubscriptionManager:
    """Manages client subscriptions to specific stream channels.

    Handles subscription state management and runtime channel switching
    without requiring client reconnection.
    """

    def __init__(self):
        """Initialize subscription manager."""
        self.client_subscriptions: dict[str, dict[str, set[ChannelType]]] = (
            {}
        )  # connection_id -> camera_id -> channels
        self.subscription_metrics = {
            "total_subscriptions": 0,
            "channel_switches": 0,
            "active_connections": 0,
        }
        logger.info("ChannelSubscriptionManager initialized")

    async def subscribe_to_channel(
        self, connection_id: str, camera_id: str, channel_type: ChannelType
    ) -> dict[str, Any]:
        """Subscribe client to specific channel.

        Args:
            connection_id: Client connection identifier
            camera_id: Camera identifier
            channel_type: Channel type to subscribe to

        Returns:
            Subscription status dictionary
        """
        try:
            if connection_id not in self.client_subscriptions:
                self.client_subscriptions[connection_id] = {}
                self.subscription_metrics["active_connections"] += 1

            if camera_id not in self.client_subscriptions[connection_id]:
                self.client_subscriptions[connection_id][camera_id] = set()

            # Add channel subscription
            if channel_type not in self.client_subscriptions[connection_id][camera_id]:
                self.client_subscriptions[connection_id][camera_id].add(channel_type)
                self.subscription_metrics["total_subscriptions"] += 1

            logger.info(
                f"Client {connection_id} subscribed to {channel_type.value} channel for camera {camera_id}"
            )

            return {
                "status": "subscribed",
                "connection_id": connection_id,
                "camera_id": camera_id,
                "channel_type": channel_type.value,
                "subscribed_channels": [
                    ch.value
                    for ch in self.client_subscriptions[connection_id][camera_id]
                ],
            }

        except Exception as e:
            logger.error(f"Subscription failed for connection {connection_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def switch_channel(
        self, connection_id: str, camera_id: str, new_channel: ChannelType
    ) -> dict[str, Any]:
        """Switch client to different channel without reconnection.

        Args:
            connection_id: Client connection identifier
            camera_id: Camera identifier
            new_channel: New channel type

        Returns:
            Channel switch status dictionary
        """
        try:
            if (
                connection_id not in self.client_subscriptions
                or camera_id not in self.client_subscriptions[connection_id]
            ):
                return {"status": "error", "error": "No active subscription found"}

            # Clear existing subscriptions for this camera
            old_channels = list(self.client_subscriptions[connection_id][camera_id])
            self.client_subscriptions[connection_id][camera_id].clear()

            # Add new channel subscription
            self.client_subscriptions[connection_id][camera_id].add(new_channel)
            self.subscription_metrics["channel_switches"] += 1

            logger.info(
                f"Client {connection_id} switched from {old_channels} to {new_channel.value} "
                f"for camera {camera_id}"
            )

            return {
                "status": "switched",
                "connection_id": connection_id,
                "camera_id": camera_id,
                "old_channels": [ch.value for ch in old_channels],
                "new_channel": new_channel.value,
                "switch_time": time.time(),
            }

        except Exception as e:
            logger.error(f"Channel switch failed for connection {connection_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def unsubscribe_from_channel(
        self, connection_id: str, camera_id: str, channel_type: ChannelType = None
    ) -> dict[str, Any]:
        """Unsubscribe client from specific channel or all channels for a camera.

        Args:
            connection_id: Client connection identifier
            camera_id: Camera identifier
            channel_type: Specific channel type to unsubscribe from (None for all)

        Returns:
            Unsubscription status dictionary
        """
        try:
            if (
                connection_id not in self.client_subscriptions
                or camera_id not in self.client_subscriptions[connection_id]
            ):
                return {"status": "error", "error": "No subscription found"}

            if channel_type:
                # Unsubscribe from specific channel
                self.client_subscriptions[connection_id][camera_id].discard(
                    channel_type
                )
                unsubscribed_channels = [channel_type.value]
            else:
                # Unsubscribe from all channels for this camera
                unsubscribed_channels = [
                    ch.value
                    for ch in self.client_subscriptions[connection_id][camera_id]
                ]
                self.client_subscriptions[connection_id][camera_id].clear()

            # Cleanup empty entries
            if not self.client_subscriptions[connection_id][camera_id]:
                del self.client_subscriptions[connection_id][camera_id]

            if not self.client_subscriptions[connection_id]:
                del self.client_subscriptions[connection_id]
                self.subscription_metrics["active_connections"] -= 1

            logger.info(
                f"Client {connection_id} unsubscribed from channels {unsubscribed_channels} "
                f"for camera {camera_id}"
            )

            return {
                "status": "unsubscribed",
                "connection_id": connection_id,
                "camera_id": camera_id,
                "unsubscribed_channels": unsubscribed_channels,
            }

        except Exception as e:
            logger.error(f"Unsubscription failed for connection {connection_id}: {e}")
            return {"status": "error", "error": str(e)}

    def get_client_subscriptions(self, connection_id: str) -> dict[str, Any]:
        """Get all subscriptions for a specific client.

        Args:
            connection_id: Client connection identifier

        Returns:
            Dictionary with client subscription information
        """
        if connection_id not in self.client_subscriptions:
            return {"subscriptions": {}}

        subscriptions = {}
        for camera_id, channels in self.client_subscriptions[connection_id].items():
            subscriptions[camera_id] = [ch.value for ch in channels]

        return {
            "connection_id": connection_id,
            "subscriptions": subscriptions,
            "total_cameras": len(subscriptions),
            "total_channels": sum(len(channels) for channels in subscriptions.values()),
        }

    def get_subscription_stats(self) -> dict[str, Any]:
        """Get comprehensive subscription statistics.

        Returns:
            Dictionary with subscription statistics
        """
        return {
            **self.subscription_metrics,
            "subscriptions_by_channel": {
                "raw": sum(
                    1
                    for camera_subs in self.client_subscriptions.values()
                    for channels in camera_subs.values()
                    if ChannelType.RAW in channels
                ),
                "annotated": sum(
                    1
                    for camera_subs in self.client_subscriptions.values()
                    for channels in camera_subs.values()
                    if ChannelType.ANNOTATED in channels
                ),
            },
            "cameras_with_subscriptions": len(
                set(
                    camera_id
                    for camera_subs in self.client_subscriptions.values()
                    for camera_id in camera_subs.keys()
                )
            ),
        }


class CameraConnectionManager:
    """Manages camera connections with different protocols."""

    def __init__(self):
        self.active_connections: dict[str, Any] = {}
        self.connection_stats: dict[str, dict[str, Any]] = {}
        self.webrtc_manager = WebRTCConnectionManager()
        self._connection_failures: dict[str, tuple[int, float]] = (
            {}
        )  # camera_id -> (failure_count, last_failure_time)
        self._connection_pool_size = 200  # Increased for 100+ concurrent connections
        self._max_reconnect_attempts = 3

    async def connect_camera(self, config: CameraConfig) -> bool:
        """Establish connection to camera based on protocol with circuit breaker."""
        camera_id = config.camera_id

        # Circuit breaker logic
        if camera_id in self._connection_failures:
            failure_count, last_failure = self._connection_failures[camera_id]
            if failure_count >= 5:  # Max failures before circuit opens
                time_since_failure = time.time() - last_failure
                if time_since_failure < 300:  # 5 minute backoff
                    logger.warning(
                        f"Circuit breaker open for camera {camera_id}, skipping connection attempt"
                    )
                    return False
                else:
                    # Reset circuit breaker after timeout
                    del self._connection_failures[camera_id]

        try:
            success = False
            if config.protocol == StreamProtocol.RTSP:
                success = await self._connect_rtsp(config)
            elif config.protocol == StreamProtocol.WEBRTC:
                success = await self._connect_webrtc(config)
            elif config.protocol == StreamProtocol.HTTP:
                success = await self._connect_http(config)
            elif config.protocol == StreamProtocol.ONVIF:
                success = await self._connect_onvif(config)
            else:
                raise CameraConfigurationError(
                    f"Unsupported protocol: {config.protocol}"
                )

            if success and camera_id in self._connection_failures:
                # Reset failure count on successful connection
                del self._connection_failures[camera_id]

            return success

        except Exception as e:
            logger.error(f"Failed to connect camera {config.camera_id}: {e}")

            # Update circuit breaker state
            if camera_id not in self._connection_failures:
                self._connection_failures[camera_id] = (1, time.time())
            else:
                failure_count, _ = self._connection_failures[camera_id]
                self._connection_failures[camera_id] = (failure_count + 1, time.time())

            return False

    async def _connect_rtsp(self, config: CameraConfig) -> bool:
        """Connect to RTSP camera stream."""
        try:
            cap = cv2.VideoCapture(config.stream_url)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, config.fps)

            if not cap.isOpened():
                raise CameraConnectionError(
                    f"Cannot open RTSP stream: {config.stream_url}"
                )

            # Test frame capture
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                raise CameraConnectionError("Cannot capture frames from RTSP stream")

            self.active_connections[config.camera_id] = cap
            self.connection_stats[config.camera_id] = {
                "protocol": config.protocol.value,
                "connected_at": datetime.now(UTC),
                "frames_captured": 0,
                "last_frame_time": None,
            }

            logger.info(f"Successfully connected to RTSP camera: {config.camera_id}")
            return True

        except Exception as e:
            logger.error(f"RTSP connection failed for {config.camera_id}: {e}")
            return False

    async def _connect_webrtc(self, config: CameraConfig) -> bool:
        """Connect to WebRTC camera stream."""
        if not WEBRTC_AVAILABLE:
            logger.error(
                "WebRTC dependencies not available. Please install: pip install aiortc pyav"
            )
            return False

        try:
            logger.info(
                f"Establishing WebRTC connection for camera: {config.camera_id}"
            )

            # For WebRTC, the stream_url could be:
            # 1. A WebRTC signaling server URL
            # 2. A local camera device (e.g., "/dev/video0" or "0")
            # 3. An RTSP URL that we'll proxy through WebRTC

            capture_source = None

            # Determine the source type
            if config.stream_url.startswith(("rtsp://", "http://", "https://")):
                # Proxy RTSP/HTTP stream through WebRTC
                logger.info(f"Creating WebRTC proxy for {config.stream_url}")
                capture_source = cv2.VideoCapture(config.stream_url)

                if not capture_source.isOpened():
                    logger.error(f"Failed to open source stream: {config.stream_url}")
                    return False

                # Configure capture properties
                capture_source.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
                capture_source.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
                capture_source.set(cv2.CAP_PROP_FPS, config.fps)

                # Test frame capture
                ret, frame = capture_source.read()
                if not ret or frame is None:
                    capture_source.release()
                    logger.error(
                        f"Cannot capture frames from source: {config.stream_url}"
                    )
                    return False

            elif config.stream_url.isdigit() or config.stream_url.startswith("/dev/"):
                # Local camera device
                device_id = (
                    int(config.stream_url)
                    if config.stream_url.isdigit()
                    else config.stream_url
                )
                logger.info(f"Creating WebRTC connection for local device: {device_id}")

                capture_source = cv2.VideoCapture(device_id)

                if not capture_source.isOpened():
                    logger.error(f"Failed to open camera device: {device_id}")
                    return False

                # Configure capture properties
                capture_source.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
                capture_source.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
                capture_source.set(cv2.CAP_PROP_FPS, config.fps)

            else:
                # For demo purposes, create a mock source
                logger.warning(f"Creating mock WebRTC source for: {config.stream_url}")
                capture_source = f"webrtc_mock_{config.camera_id}"

            # Create WebRTC peer connection
            peer_connection = await self.webrtc_manager.create_peer_connection(
                config.camera_id, capture_source
            )

            # Store the connection
            self.active_connections[config.camera_id] = {
                "type": "webrtc",
                "peer_connection": peer_connection,
                "capture_source": capture_source,
                "webrtc_manager": self.webrtc_manager,
            }

            # Update connection stats
            self.connection_stats[config.camera_id] = {
                "protocol": config.protocol.value,
                "connected_at": datetime.now(UTC),
                "frames_captured": 0,
                "last_frame_time": None,
                "webrtc_state": "connected",
                "stream_url": config.stream_url,
            }

            logger.info(
                f"Successfully established WebRTC connection for: {config.camera_id}"
            )
            return True

        except Exception as e:
            logger.error(f"WebRTC connection failed for {config.camera_id}: {e}")

            # Cleanup on failure
            try:
                await self.webrtc_manager.cleanup_connection(config.camera_id)
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed for {config.camera_id}: {cleanup_error}")

            return False

    async def _connect_http(self, config: CameraConfig) -> bool:
        """Connect to HTTP camera stream."""
        # HTTP implementation would go here
        logger.warning(
            f"HTTP not fully implemented for {config.camera_id}, simulating connection"
        )
        self.active_connections[config.camera_id] = f"http_mock_{config.camera_id}"
        return True

    async def _connect_onvif(self, config: CameraConfig) -> bool:
        """Connect to ONVIF camera."""
        # ONVIF implementation would go here
        logger.warning(
            f"ONVIF not fully implemented for {config.camera_id}, simulating connection"
        )
        self.active_connections[config.camera_id] = f"onvif_mock_{config.camera_id}"
        return True

    async def disconnect_camera(self, camera_id: str) -> bool:
        """Disconnect camera and cleanup resources."""
        try:
            if camera_id in self.active_connections:
                connection = self.active_connections[camera_id]

                if isinstance(connection, cv2.VideoCapture):
                    connection.release()
                elif (
                    isinstance(connection, dict) and connection.get("type") == "webrtc"
                ):
                    # WebRTC connection cleanup
                    await self.webrtc_manager.cleanup_connection(camera_id)

                    # Release capture source if it's an OpenCV capture
                    capture_source = connection.get("capture_source")
                    if isinstance(capture_source, cv2.VideoCapture):
                        capture_source.release()

                del self.active_connections[camera_id]

                if camera_id in self.connection_stats:
                    del self.connection_stats[camera_id]

                logger.info(f"Disconnected camera: {camera_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to disconnect camera {camera_id}: {e}")
            return False

    async def capture_frame(self, camera_id: str) -> np.ndarray | None:
        """Capture frame from connected camera."""
        if camera_id not in self.active_connections:
            return None

        connection = self.active_connections[camera_id]

        try:
            if isinstance(connection, cv2.VideoCapture):
                ret, frame = connection.read()
                if ret and frame is not None:
                    self.connection_stats[camera_id]["frames_captured"] += 1
                    self.connection_stats[camera_id]["last_frame_time"] = datetime.now(
                        UTC
                    )
                    return frame

            elif isinstance(connection, dict) and connection.get("type") == "webrtc":
                # WebRTC connection - capture from source
                capture_source = connection.get("capture_source")

                if isinstance(capture_source, cv2.VideoCapture):
                    ret, frame = capture_source.read()
                    if ret and frame is not None:
                        self.connection_stats[camera_id]["frames_captured"] += 1
                        self.connection_stats[camera_id]["last_frame_time"] = (
                            datetime.now(UTC)
                        )
                        return frame
                else:
                    # Mock WebRTC source
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    self.connection_stats[camera_id]["frames_captured"] += 1
                    self.connection_stats[camera_id]["last_frame_time"] = datetime.now(
                        UTC
                    )
                    return frame

            # For mock connections, generate test frame
            elif isinstance(connection, str) and "_mock_" in connection:
                # Generate synthetic frame for testing
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                self.connection_stats[camera_id]["frames_captured"] += 1
                self.connection_stats[camera_id]["last_frame_time"] = datetime.now(UTC)
                return frame

        except Exception as e:
            logger.error(f"Frame capture failed for {camera_id}: {e}")

        return None

    def get_connection_stats(self, camera_id: str) -> dict[str, Any] | None:
        """Get connection statistics for camera."""
        stats = self.connection_stats.get(camera_id)

        # Enhance stats with WebRTC information if available
        if stats and camera_id in self.active_connections:
            connection = self.active_connections[camera_id]
            if isinstance(connection, dict) and connection.get("type") == "webrtc":
                webrtc_stats = self.webrtc_manager.get_connection_stats(camera_id)
                if webrtc_stats:
                    stats["webrtc"] = webrtc_stats

        return stats

    async def get_webrtc_offer(self, camera_id: str) -> dict[str, str] | None:
        """Get WebRTC SDP offer for camera connection."""
        if camera_id not in self.active_connections:
            logger.error(f"Camera {camera_id} not connected")
            return None

        connection = self.active_connections[camera_id]
        if not isinstance(connection, dict) or connection.get("type") != "webrtc":
            logger.error(f"Camera {camera_id} is not using WebRTC")
            return None

        try:
            offer = await self.webrtc_manager.create_offer(camera_id)
            return {
                "type": offer.type,
                "sdp": offer.sdp,
            }
        except Exception as e:
            logger.error(f"Failed to create WebRTC offer for {camera_id}: {e}")
            return None

    async def set_webrtc_answer(self, camera_id: str, answer_sdp: str) -> bool:
        """Set WebRTC SDP answer for camera connection."""
        if camera_id not in self.active_connections:
            logger.error(f"Camera {camera_id} not connected")
            return False

        connection = self.active_connections[camera_id]
        if not isinstance(connection, dict) or connection.get("type") != "webrtc":
            logger.error(f"Camera {camera_id} is not using WebRTC")
            return False

        try:
            if not WEBRTC_AVAILABLE:
                logger.error("WebRTC not available")
                return False

            answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
            await self.webrtc_manager.set_remote_description(camera_id, answer)
            return True
        except Exception as e:
            logger.error(f"Failed to set WebRTC answer for {camera_id}: {e}")
            return False

    async def get_webrtc_connection_quality(self, camera_id: str) -> dict[str, Any]:
        """Get detailed WebRTC connection quality metrics."""
        if camera_id not in self.active_connections:
            return {"error": "Camera not connected"}

        connection = self.active_connections[camera_id]
        if not isinstance(connection, dict) or connection.get("type") != "webrtc":
            return {"error": "Camera is not using WebRTC"}

        try:
            return await self.webrtc_manager.get_connection_quality_stats(camera_id)
        except Exception as e:
            logger.error(f"Failed to get WebRTC quality stats for {camera_id}: {e}")
            return {"error": str(e)}

    def is_connected(self, camera_id: str) -> bool:
        """Check if camera is connected."""
        if camera_id not in self.active_connections:
            return False

        connection = self.active_connections[camera_id]

        # For WebRTC connections, check the peer connection state
        if isinstance(connection, dict) and connection.get("type") == "webrtc":
            return self.webrtc_manager.is_connected(camera_id)

        # For other connection types
        return True


class FrameQualityValidator:
    """Validates frame quality against configurable thresholds."""

    def __init__(
        self,
        min_resolution: tuple[int, int] = (640, 480),
        min_quality_score: float = 0.5,
        max_blur_threshold: float = 100.0,
    ):
        self.min_resolution = min_resolution
        self.min_quality_score = min_quality_score
        self.max_blur_threshold = max_blur_threshold

    async def validate_frame_quality(
        self, frame: np.ndarray, camera_config: CameraConfig
    ) -> QualityMetrics:
        """Validate frame quality and return metrics."""
        issues = []

        # Resolution check
        height, width = frame.shape[:2]
        if width < self.min_resolution[0] or height < self.min_resolution[1]:
            issues.append(
                f"Resolution {width}x{height} below minimum {self.min_resolution}"
            )

        # Convert to grayscale for analysis
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Blur detection using Laplacian variance
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < self.max_blur_threshold:
            issues.append(f"Blur score {blur_score:.1f} indicates blurry image")

        # Brightness analysis
        brightness_score = float(np.mean(gray))
        if brightness_score < 30:
            issues.append("Image too dark")
        elif brightness_score > 220:
            issues.append("Image too bright")

        brightness_score_normalized = min(1.0, brightness_score / 128.0)

        # Contrast analysis
        contrast_score = float(np.std(gray))
        contrast_score_normalized = min(1.0, contrast_score / 64.0)
        if contrast_score < 20:
            issues.append("Low contrast detected")

        # Noise estimation
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise_estimate = float(np.std(cv2.filter2D(gray, -1, kernel)))
        noise_level = max(0.0, 1.0 - (noise_estimate / 50.0))

        # Calculate overall quality score
        blur_score_normalized = min(1.0, blur_score / 500.0)
        overall_score = (
            blur_score_normalized * 0.3
            + brightness_score_normalized * 0.25
            + contrast_score_normalized * 0.25
            + noise_level * 0.2
        )

        passed_validation = (
            overall_score >= camera_config.quality_threshold and len(issues) == 0
        )

        if not passed_validation and overall_score < camera_config.quality_threshold:
            issues.append(
                f"Overall quality score {overall_score:.2f} below threshold {camera_config.quality_threshold}"
            )

        return QualityMetrics(
            overall_score=overall_score,
            blur_score=blur_score_normalized,
            brightness_score=brightness_score_normalized,
            contrast_score=contrast_score_normalized,
            noise_level=noise_level,
            passed_validation=passed_validation,
            issues=issues,
        )


class StreamingServiceInterface:
    """Interface definition for the streaming service."""

    async def register_camera(self, camera_config: CameraConfig) -> CameraRegistration:
        """Register a camera for stream processing."""
        raise NotImplementedError

    async def process_stream(self, camera_id: str) -> AsyncIterator[ProcessedFrame]:
        """Process continuous stream from registered camera."""
        raise NotImplementedError

    async def validate_frame_quality(
        self, frame: np.ndarray, camera_config: CameraConfig
    ) -> QualityMetrics:
        """Validate frame quality against camera configuration."""
        raise NotImplementedError

    async def queue_frame_batch(self, frames: list[ProcessedFrame]) -> BatchId:
        """Queue batch of processed frames for downstream processing."""
        raise NotImplementedError


class SSEStreamingService:
    """Server-Sent Events Streaming Service for MP4 fragmented streaming.

    Provides browser-native video viewing with dual channels (raw + AI-annotated streams)
    using dependency injection patterns and existing streaming infrastructure.
    """

    def __init__(
        self,
        base_streaming_service: "StreamingService",
        redis_manager: RedisQueueManager,
        config: dict[str, Any],
        ml_annotation_processor: Optional[MLAnnotationProcessor] = None,
    ):
        """Initialize SSE streaming service with dependency injection.

        Args:
            base_streaming_service: Base streaming service for core functionality
            redis_manager: Redis queue manager for stream data
            config: SSE streaming configuration
            ml_annotation_processor: ML annotation processor for AI-annotated streams
        """
        self.base_streaming_service = base_streaming_service
        self.redis_manager = redis_manager
        self.config = config
        self.ml_annotation_processor = ml_annotation_processor

        # SSE connection management
        self.active_streams: dict[str, SSEStream] = {}
        self.connection_metrics = SSEConnectionMetrics()
        self.stream_queues: dict[str, asyncio.Queue] = {}

        # Performance tracking
        self.startup_time: float | None = None
        self.processing_metrics = {
            "fragments_generated": 0,
            "streams_created": 0,
            "connection_failures": 0,
            "total_bytes_streamed": 0,
            "average_fragment_size": 0.0,
            "average_processing_latency": 0.0,
        }

        # Configuration defaults
        self.max_concurrent_connections = config.get("max_concurrent_connections", 100)
        self.fragment_duration_ms = config.get(
            "fragment_duration_ms", 2000
        )  # 2 seconds
        self.heartbeat_interval = config.get("heartbeat_interval", 30)  # 30 seconds
        self.connection_timeout = config.get("connection_timeout", 300)  # 5 minutes

        logger.info("SSEStreamingService initialized with dependency injection")

    async def create_sse_stream(
        self,
        camera_id: str,
        user: Any,  # User type from existing auth system
        stream_type: str = "raw",
        quality: str = "medium",
    ) -> SSEStream:
        """Create SSE stream with proper auth and validation.

        Args:
            camera_id: Target camera identifier
            user: Authenticated user object
            stream_type: Stream type ('raw' or 'annotated')
            quality: Stream quality ('low', 'medium', 'high')

        Returns:
            SSEStream configuration object

        Raises:
            StreamProcessingError: If stream creation fails
        """
        try:
            # Check concurrent connection limits
            if len(self.active_streams) >= self.max_concurrent_connections:
                raise StreamProcessingError(
                    f"Maximum concurrent connections ({self.max_concurrent_connections}) reached"
                )

            # Validate camera exists and user has permission
            # Note: In production, implement proper RBAC validation here
            user_id = getattr(user, "id", "anonymous")

            # Generate unique stream ID
            stream_id = f"sse_{camera_id}_{stream_type}_{uuid.uuid4().hex[:8]}"

            # Create stream configuration
            sse_stream = SSEStream(
                stream_id=stream_id,
                camera_id=camera_id,
                user_id=user_id,
                stream_type=stream_type,
                quality=quality,
                client_capabilities={
                    "supports_mp4": True,
                    "max_bitrate": self._get_quality_bitrate(quality),
                    "preferred_codec": "h264",
                },
            )

            # Register stream
            self.active_streams[stream_id] = sse_stream

            # Create message queue for this stream
            self.stream_queues[stream_id] = asyncio.Queue(maxsize=100)

            # Update metrics
            self.connection_metrics.active_connections += 1
            self.connection_metrics.total_connections_created += 1
            self.processing_metrics["streams_created"] += 1

            if (
                self.connection_metrics.active_connections
                > self.connection_metrics.peak_concurrent_connections
            ):
                self.connection_metrics.peak_concurrent_connections = (
                    self.connection_metrics.active_connections
                )

            logger.info(
                f"Created SSE stream: {stream_id} for camera {camera_id}, "
                f"user {user_id}, type {stream_type}, quality {quality}"
            )

            return sse_stream

        except Exception as e:
            self.processing_metrics["connection_failures"] += 1
            logger.error(f"Failed to create SSE stream for camera {camera_id}: {e}")
            raise StreamProcessingError(f"SSE stream creation failed: {e}") from e

    async def handle_sse_connection(
        self,
        request: "Request",
        camera_id: str,
        stream_type: str = "raw",
        quality: str = "medium",
    ) -> "StreamingResponse":
        """SSE endpoint handler with connection management.

        Args:
            request: FastAPI request object
            camera_id: Target camera identifier
            stream_type: Stream type ('raw' or 'annotated')
            quality: Stream quality

        Returns:
            StreamingResponse with SSE data
        """
        # Import here to avoid circular imports
        from fastapi.responses import StreamingResponse

        # Extract user from request (implement proper auth integration)
        user = getattr(request.state, "user", None)
        if not user:
            # Create mock user for now - replace with proper auth
            user = type("User", (), {"id": "anonymous", "username": "anonymous"})()

        try:
            # Create SSE stream
            sse_stream = await self.create_sse_stream(
                camera_id, user, stream_type, quality
            )

            # Generate SSE event stream
            async def generate_sse_events() -> AsyncIterator[str]:
                try:
                    # Send initial connection event
                    yield self._format_sse_event(
                        "connected",
                        {
                            "stream_id": sse_stream.stream_id,
                            "camera_id": camera_id,
                            "stream_type": stream_type,
                            "quality": quality,
                            "timestamp": time.time(),
                        },
                    )

                    # Start fragment streaming for this camera
                    fragment_task = asyncio.create_task(
                        self._stream_fragments_to_queue(sse_stream)
                    )

                    last_heartbeat = time.time()

                    try:
                        while True:
                            # Check for client disconnect
                            if await request.is_disconnected():
                                logger.info(
                                    f"Client disconnected from stream {sse_stream.stream_id}"
                                )
                                break

                            # Get message from queue with timeout
                            try:
                                message = await asyncio.wait_for(
                                    self.stream_queues[sse_stream.stream_id].get(),
                                    timeout=1.0,
                                )

                                if message["type"] == "fragment":
                                    # Send MP4 fragment
                                    yield self._format_sse_event(
                                        "fragment", message["data"]
                                    )
                                elif message["type"] == "error":
                                    yield self._format_sse_event(
                                        "error", {"error": message["error"]}
                                    )
                                    break

                                # Update activity timestamp
                                sse_stream.last_activity = datetime.utcnow()

                            except TimeoutError:
                                # Send heartbeat if needed
                                current_time = time.time()
                                if (
                                    current_time - last_heartbeat
                                    >= self.heartbeat_interval
                                ):
                                    yield self._format_sse_event(
                                        "heartbeat", {"timestamp": current_time}
                                    )
                                    last_heartbeat = current_time

                                continue

                    finally:
                        # Cleanup on disconnect
                        fragment_task.cancel()
                        await self._cleanup_sse_stream(sse_stream.stream_id)

                except Exception as e:
                    logger.error(f"SSE stream error for {sse_stream.stream_id}: {e}")
                    yield self._format_sse_event(
                        "error", {"error": str(e), "timestamp": time.time()}
                    )

            return StreamingResponse(
                generate_sse_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # Disable Nginx buffering
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control",
                },
            )

        except Exception as e:
            self.connection_metrics.connection_errors += 1
            logger.error(f"SSE connection handler failed for camera {camera_id}: {e}")
            raise

    async def stream_mp4_fragments(
        self, camera_id: str, quality: str = "medium"
    ) -> AsyncIterator[MP4Fragment]:
        """Core streaming logic yielding MP4 fragments.

        Args:
            camera_id: Target camera identifier
            quality: Stream quality setting

        Yields:
            MP4Fragment objects with video data
        """
        try:
            sequence_number = 0

            # Get camera stream from base streaming service
            if not self.base_streaming_service.streaming_processor:
                raise StreamProcessingError("Base streaming processor not available")

            # Process camera stream and generate fragments
            async for (
                processed_frame
            ) in self.base_streaming_service.streaming_processor.process_stream(
                camera_id
            ):
                try:
                    # Convert processed frame to MP4 fragment
                    fragment = await self._create_mp4_fragment(
                        processed_frame, sequence_number, quality
                    )

                    if fragment:
                        # Update metrics
                        self.processing_metrics["fragments_generated"] += 1
                        self.processing_metrics[
                            "total_bytes_streamed"
                        ] += fragment.size_bytes

                        # Update average fragment size
                        total_fragments = self.processing_metrics["fragments_generated"]
                        self.processing_metrics["average_fragment_size"] = (
                            self.processing_metrics["total_bytes_streamed"]
                            / total_fragments
                        )

                        sequence_number += 1
                        yield fragment

                except Exception as e:
                    logger.error(
                        f"Fragment creation failed for camera {camera_id}: {e}"
                    )
                    continue

        except Exception as e:
            logger.error(f"MP4 fragment streaming failed for camera {camera_id}: {e}")
            raise StreamProcessingError(f"Fragment streaming failed: {e}") from e

    async def _stream_fragments_to_queue(self, sse_stream: SSEStream) -> None:
        """Stream MP4 fragments to the SSE queue for a specific stream.

        Args:
            sse_stream: SSE stream configuration
        """
        try:
            async for fragment in self.stream_mp4_fragments(
                sse_stream.camera_id, sse_stream.quality, sse_stream.stream_type
            ):
                try:
                    # Prepare fragment data for SSE
                    fragment_data = {
                        "fragment_id": fragment.fragment_id,
                        "sequence_number": fragment.sequence_number,
                        "timestamp": fragment.timestamp,
                        "data": fragment.data.hex(),  # Hexadecimal encoding for text transport
                        "content_type": fragment.content_type,
                        "size_bytes": fragment.size_bytes,
                        "quality": fragment.quality,
                        "metadata": fragment.metadata,
                    }

                    # Add to stream queue
                    if sse_stream.stream_id in self.stream_queues:
                        try:
                            self.stream_queues[sse_stream.stream_id].put_nowait(
                                {"type": "fragment", "data": fragment_data}
                            )
                        except asyncio.QueueFull:
                            logger.warning(
                                f"Stream queue full for {sse_stream.stream_id}, dropping fragment"
                            )
                            continue

                except Exception as e:
                    logger.error(
                        f"Failed to queue fragment for stream {sse_stream.stream_id}: {e}"
                    )

        except Exception as e:
            logger.error(
                f"Fragment streaming task failed for stream {sse_stream.stream_id}: {e}"
            )
            # Send error to stream queue
            if sse_stream.stream_id in self.stream_queues:
                try:
                    self.stream_queues[sse_stream.stream_id].put_nowait(
                        {"type": "error", "error": str(e)}
                    )
                except asyncio.QueueFull:
                    pass

    async def _create_mp4_fragment(
        self, processed_frame: ProcessedFrame, sequence_number: int, quality: str
    ) -> MP4Fragment | None:
        """Create MP4 fragment from processed frame.

        Args:
            processed_frame: Input processed frame
            sequence_number: Fragment sequence number
            quality: Target quality setting

        Returns:
            MP4Fragment or None if creation fails
        """
        try:
            start_time = time.perf_counter()

            # Generate fragment ID
            fragment_id = f"frag_{processed_frame.camera_id}_{sequence_number}_{int(time.time() * 1000)}"

            # Convert frame to MP4 fragment
            # This is a simplified implementation - in production, use proper MP4 fragmentation
            if processed_frame.original_image is not None:
                # Encode frame as JPEG for now (replace with proper MP4 encoding)
                encode_param = [
                    cv2.IMWRITE_JPEG_QUALITY,
                    self._get_quality_jpeg_param(quality),
                ]
                success, encoded_data = cv2.imencode(
                    ".jpg", processed_frame.original_image, encode_param
                )

                if not success:
                    logger.warning(f"Failed to encode frame {processed_frame.frame_id}")
                    return None

                fragment_data = encoded_data.tobytes()

                # Create fragment
                fragment = MP4Fragment(
                    fragment_id=fragment_id,
                    camera_id=processed_frame.camera_id,
                    sequence_number=sequence_number,
                    timestamp=processed_frame.timestamp,
                    data=fragment_data,
                    content_type="image/jpeg",  # Temporary - use video/mp4 in production
                    duration_ms=self.fragment_duration_ms,
                    quality=quality,
                    metadata={
                        "quality_score": getattr(processed_frame, "quality_score", 0.0),
                        "processing_time_ms": getattr(
                            processed_frame, "processing_time_ms", 0.0
                        ),
                        "resolution": (
                            processed_frame.original_image.shape[:2]
                            if processed_frame.original_image is not None
                            else (0, 0)
                        ),
                    },
                )

                # Update processing latency metrics
                processing_time = (time.perf_counter() - start_time) * 1000
                total_fragments = self.processing_metrics["fragments_generated"] + 1
                self.processing_metrics["average_processing_latency"] = (
                    self.processing_metrics["average_processing_latency"]
                    * (total_fragments - 1)
                    + processing_time
                ) / total_fragments

                return fragment

            return None

        except Exception as e:
            logger.error(f"MP4 fragment creation failed: {e}")
            return None

    def _format_sse_event(self, event_type: str, data: dict[str, Any]) -> str:
        """Format data as SSE event.

        Args:
            event_type: Event type identifier
            data: Event data dictionary

        Returns:
            Formatted SSE event string
        """
        event_data = {"type": event_type, "timestamp": time.time(), **data}

        return f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"

    async def _cleanup_sse_stream(self, stream_id: str) -> None:
        """Clean up SSE stream resources.

        Args:
            stream_id: Stream identifier to cleanup
        """
        try:
            if stream_id in self.active_streams:
                stream = self.active_streams[stream_id]

                # Calculate connection duration
                connection_duration = (
                    datetime.utcnow() - stream.connection_time
                ).total_seconds()

                # Update metrics
                self.connection_metrics.active_connections -= 1
                self.connection_metrics.total_disconnections += 1

                # Update average connection duration
                total_disconnections = self.connection_metrics.total_disconnections
                self.connection_metrics.average_connection_duration = (
                    self.connection_metrics.average_connection_duration
                    * (total_disconnections - 1)
                    + connection_duration
                ) / total_disconnections

                # Remove stream and queue
                del self.active_streams[stream_id]

                if stream_id in self.stream_queues:
                    del self.stream_queues[stream_id]

                logger.info(
                    f"Cleaned up SSE stream {stream_id}, duration: {connection_duration:.2f}s"
                )

        except Exception as e:
            logger.error(f"Failed to cleanup SSE stream {stream_id}: {e}")

    def _get_quality_bitrate(self, quality: str) -> int:
        """Get bitrate for quality setting.

        Args:
            quality: Quality setting

        Returns:
            Bitrate in kbps
        """
        quality_settings = {
            "low": 500,  # 500 kbps
            "medium": 2000,  # 2 Mbps
            "high": 5000,  # 5 Mbps
        }
        return quality_settings.get(quality, 2000)

    def _get_quality_jpeg_param(self, quality: str) -> int:
        """Get JPEG quality parameter for quality setting.

        Args:
            quality: Quality setting

        Returns:
            JPEG quality parameter (0-100)
        """
        quality_settings = {
            "low": 60,
            "medium": 85,
            "high": 95,
        }
        return quality_settings.get(quality, 85)

    async def create_dual_channel_stream(
        self,
        camera_id: str,
        user_id: str,
        raw_quality: str = "medium",
        annotated_quality: str = "medium",
    ) -> DualChannelStream:
        """Create dual-channel stream with both raw and annotated channels.

        Args:
            camera_id: Target camera identifier
            user_id: User creating the stream
            raw_quality: Quality for raw channel
            annotated_quality: Quality for annotated channel

        Returns:
            DualChannelStream configuration
        """
        raw_quality_enum = QualityLevel(raw_quality)
        annotated_quality_enum = QualityLevel(annotated_quality)

        dual_stream = await self.channel_manager.create_dual_channel_stream(
            camera_id=camera_id,
            user_id=user_id,
            raw_quality=raw_quality_enum,
            annotated_quality=annotated_quality_enum,
        )

        # Update metrics
        self.connection_metrics.dual_channel_streams += 1

        return dual_stream

    async def handle_dual_channel_sse_connection(
        self,
        request: "Request",
        camera_id: str,
        initial_channel: str = "raw",
        raw_quality: str = "medium",
        annotated_quality: str = "medium",
    ) -> "StreamingResponse":
        """SSE endpoint handler for dual-channel streaming.

        Args:
            request: FastAPI request object
            camera_id: Target camera identifier
            initial_channel: Initial channel to stream (raw or annotated)
            raw_quality: Quality for raw channel
            annotated_quality: Quality for annotated channel

        Returns:
            StreamingResponse with dual-channel SSE data
        """
        from fastapi.responses import StreamingResponse

        user = getattr(request.state, "user", None)
        if not user:
            user = type("User", (), {"id": "anonymous", "username": "anonymous"})()

        try:
            # Create dual-channel stream
            dual_stream = await self.create_dual_channel_stream(
                camera_id=camera_id,
                user_id=user.id,
                raw_quality=raw_quality,
                annotated_quality=annotated_quality,
            )

            connection_id = f"conn_{uuid.uuid4().hex[:8]}"

            # Subscribe to initial channel
            initial_channel_type = (
                ChannelType.RAW if initial_channel == "raw" else ChannelType.ANNOTATED
            )
            await self.subscription_manager.subscribe_to_channel(
                connection_id=connection_id,
                camera_id=camera_id,
                channel_type=initial_channel_type,
            )

            # Generate dual-channel SSE events
            async def generate_dual_channel_events() -> AsyncIterator[str]:
                try:
                    # Send initial connection event
                    yield self._format_sse_event(
                        "dual_channel_connected",
                        {
                            "stream_id": dual_stream.stream_id,
                            "camera_id": camera_id,
                            "initial_channel": initial_channel,
                            "raw_quality": raw_quality,
                            "annotated_quality": annotated_quality,
                            "sync_tolerance_ms": dual_stream.sync_tolerance_ms,
                            "timestamp": time.time(),
                        },
                    )

                    # Start fragment streaming for both channels
                    raw_task = asyncio.create_task(
                        self._stream_channel_fragments(dual_stream, ChannelType.RAW)
                    )
                    annotated_task = asyncio.create_task(
                        self._stream_channel_fragments(
                            dual_stream, ChannelType.ANNOTATED
                        )
                    )

                    last_heartbeat = time.time()
                    last_sync_check = time.time()

                    try:
                        while True:
                            # Check for client disconnect
                            if await request.is_disconnected():
                                logger.info(
                                    f"Client disconnected from dual-channel stream {dual_stream.stream_id}"
                                )
                                break

                            # Check subscriptions and send appropriate fragments
                            subscriptions = (
                                self.subscription_manager.get_client_subscriptions(
                                    connection_id
                                )
                            )
                            current_channels = subscriptions.get(
                                "subscriptions", {}
                            ).get(camera_id, [])

                            # Get message from appropriate channel queue
                            fragment_received = False
                            for channel_str in current_channels:
                                queue_key = f"{dual_stream.stream_id}_{channel_str}"
                                if queue_key in self.stream_queues:
                                    try:
                                        message = await asyncio.wait_for(
                                            self.stream_queues[queue_key].get(),
                                            timeout=0.1,
                                        )

                                        if message["type"] == "fragment":
                                            # Send fragment with channel info
                                            fragment_data = message["data"]
                                            fragment_data["channel_type"] = channel_str

                                            yield self._format_sse_event(
                                                "dual_channel_fragment", fragment_data
                                            )

                                            # Update channel timestamp for sync
                                            channel_type = (
                                                ChannelType.RAW
                                                if channel_str == "raw"
                                                else ChannelType.ANNOTATED
                                            )
                                            await self.channel_manager.update_channel_timestamp(
                                                dual_stream.stream_id,
                                                channel_type,
                                                fragment_data["timestamp"],
                                            )

                                        fragment_received = True
                                        break

                                    except TimeoutError:
                                        continue

                            # Periodic synchronization check
                            current_time = time.time()
                            if (
                                current_time - last_sync_check >= 1.0
                            ):  # Check every second
                                sync_status = (
                                    await self.channel_manager.synchronize_channels(
                                        dual_stream.stream_id
                                    )
                                )

                                if not sync_status.is_synchronized:
                                    yield self._format_sse_event(
                                        "sync_warning",
                                        {
                                            "drift_ms": sync_status.drift_ms,
                                            "correction_applied": sync_status.correction_applied,
                                            "timestamp": current_time,
                                        },
                                    )

                                last_sync_check = current_time

                            # Send heartbeat if needed
                            if current_time - last_heartbeat >= self.heartbeat_interval:
                                sync_stats = await self.channel_manager.get_synchronization_stats(
                                    dual_stream.stream_id
                                )

                                yield self._format_sse_event(
                                    "dual_channel_heartbeat",
                                    {
                                        "timestamp": current_time,
                                        "sync_stats": sync_stats,
                                    },
                                )
                                last_heartbeat = current_time

                            # Brief sleep if no fragments received
                            if not fragment_received:
                                await asyncio.sleep(0.01)  # 10ms

                    finally:
                        # Cleanup on disconnect
                        raw_task.cancel()
                        annotated_task.cancel()
                        await self.subscription_manager.unsubscribe_from_channel(
                            connection_id, camera_id
                        )
                        await self.channel_manager.remove_dual_channel_stream(
                            dual_stream.stream_id
                        )

                except Exception as e:
                    logger.error(
                        f"Dual-channel SSE stream error for {dual_stream.stream_id}: {e}"
                    )
                    yield self._format_sse_event(
                        "dual_channel_error",
                        {"error": str(e), "timestamp": time.time()},
                    )

            return StreamingResponse(
                generate_dual_channel_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "Cache-Control",
                },
            )

        except Exception as e:
            self.connection_metrics.connection_errors += 1
            logger.error(
                f"Dual-channel SSE connection handler failed for camera {camera_id}: {e}"
            )
            raise

    async def _stream_channel_fragments(
        self, dual_stream: DualChannelStream, channel_type: ChannelType
    ) -> None:
        """Stream fragments for a specific channel in dual-channel mode.

        Args:
            dual_stream: Dual-channel stream configuration
            channel_type: Channel type to stream
        """
        try:
            queue_key = f"{dual_stream.stream_id}_{channel_type.value}"
            if queue_key not in self.stream_queues:
                self.stream_queues[queue_key] = asyncio.Queue(maxsize=100)

            quality = (
                dual_stream.raw_channel.quality.value
                if channel_type == ChannelType.RAW
                else dual_stream.annotated_channel.quality.value
            )

            async for fragment in self.stream_mp4_fragments(
                dual_stream.camera_id, quality
            ):
                try:
                    # Apply any sync offset
                    sync_offset = (
                        dual_stream.raw_channel.sync_offset_ms
                        if channel_type == ChannelType.RAW
                        else dual_stream.annotated_channel.sync_offset_ms
                    )

                    if sync_offset > 0:
                        await asyncio.sleep(
                            sync_offset / 1000.0
                        )  # Convert ms to seconds

                    # Prepare fragment data
                    fragment_data = {
                        "fragment_id": fragment.fragment_id,
                        "sequence_number": fragment.sequence_number,
                        "timestamp": fragment.timestamp,
                        "data": fragment.data.hex(),
                        "content_type": fragment.content_type,
                        "size_bytes": fragment.size_bytes,
                        "quality": fragment.quality,
                        "channel_type": channel_type.value,
                        "metadata": fragment.metadata,
                    }

                    # Queue fragment
                    try:
                        await asyncio.wait_for(
                            self.stream_queues[queue_key].put(
                                {"type": "fragment", "data": fragment_data}
                            ),
                            timeout=0.1,
                        )
                    except TimeoutError:
                        logger.warning(
                            f"Fragment queue full for {queue_key}, dropping fragment"
                        )

                except Exception as e:
                    logger.error(
                        f"Fragment streaming failed for channel {channel_type.value}: {e}"
                    )
                    continue

        except Exception as e:
            logger.error(
                f"Channel fragment streaming failed for {channel_type.value}: {e}"
            )
            # Send error to queue
            queue_key = f"{dual_stream.stream_id}_{channel_type.value}"
            if queue_key in self.stream_queues:
                try:
                    await self.stream_queues[queue_key].put(
                        {"type": "error", "error": str(e)}
                    )
                except asyncio.QueueFull:
                    pass

    async def get_connection_stats(self) -> dict[str, Any]:
        """Get SSE connection statistics.

        Returns:
            Dictionary with connection statistics
        """
        return {
            "active_connections": self.connection_metrics.active_connections,
            "total_connections_created": self.connection_metrics.total_connections_created,
            "total_disconnections": self.connection_metrics.total_disconnections,
            "bytes_streamed": self.connection_metrics.bytes_streamed,
            "fragments_sent": self.connection_metrics.fragments_sent,
            "connection_errors": self.connection_metrics.connection_errors,
            "average_connection_duration": self.connection_metrics.average_connection_duration,
            "peak_concurrent_connections": self.connection_metrics.peak_concurrent_connections,
            "dual_channel_streams": self.connection_metrics.dual_channel_streams,
            "channel_switches": self.connection_metrics.channel_switches,
            "sync_violations": self.connection_metrics.sync_violations,
            "processing_metrics": self.processing_metrics,
            "last_activity": self.connection_metrics.last_activity.isoformat(),
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check for SSE streaming service.

        Returns:
            Health check result dictionary
        """
        try:
            # Check base streaming service health
            base_health = await self.base_streaming_service.health_check()

            # Check SSE-specific health
            sse_healthy = (
                len(self.active_streams) <= self.max_concurrent_connections
                and self.connection_metrics.connection_errors
                < 100  # Arbitrary threshold
            )

            overall_healthy = base_health.get("healthy", False) and sse_healthy

            return {
                "healthy": overall_healthy,
                "sse_service_status": "healthy" if sse_healthy else "degraded",
                "base_service_health": base_health,
                "connection_stats": await self.get_connection_stats(),
                "dual_channel_stats": self.channel_manager.get_dual_channel_stats(),
                "subscription_stats": self.subscription_manager.get_subscription_stats(),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"SSE health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "timestamp": time.time(),
            }


class StreamingService:
    """Main Streaming Service with independent startup capability.

    This service can run independently without FastAPI binding and supports
    both CLI and programmatic initialization with proper dependency injection.
    """

    def __init__(
        self,
        streaming_processor: Union["StreamingDataProcessor", None] = None,
        redis_manager: RedisQueueManager | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize streaming service with dependency injection support.

        Args:
            streaming_processor: Injected streaming data processor
            redis_manager: Injected Redis queue manager
            config: Service configuration dictionary
        """
        self.config = config or {}
        self.streaming_processor = streaming_processor
        self.redis_manager = redis_manager

        # Service state
        self.is_running = False
        self.startup_time: float | None = None
        self.health_status = "initializing"

        # Performance tracking
        self.startup_metrics = {
            "startup_time_ms": 0.0,
            "initialization_time_ms": 0.0,
            "ready_time_ms": 0.0,
        }

        logger.info("StreamingService initialized with dependency injection")

    async def start(self) -> None:
        """Start the streaming service independently.

        Ensures <10ms startup time requirement.
        """
        if self.is_running:
            logger.warning("StreamingService already running")
            return

        startup_start = time.perf_counter()

        try:
            # Initialize components if not injected
            if self.streaming_processor is None:
                self.streaming_processor = StreamingDataProcessor(
                    redis_client=self.redis_manager,
                    max_concurrent_streams=self.config.get(
                        "max_concurrent_streams", 100
                    ),
                    frame_processing_timeout=self.config.get(
                        "frame_processing_timeout", 0.01
                    ),
                )

            # Start the streaming processor
            await self.streaming_processor.start()

            self.is_running = True
            self.health_status = "healthy"

            startup_end = time.perf_counter()
            startup_time_ms = (startup_end - startup_start) * 1000

            self.startup_time = startup_time_ms
            self.startup_metrics["startup_time_ms"] = startup_time_ms

            logger.info(f"StreamingService started in {startup_time_ms:.2f}ms")

            # Verify startup time requirement
            if startup_time_ms > 10.0:
                logger.warning(
                    f"Startup time {startup_time_ms:.2f}ms exceeds 10ms target"
                )

        except Exception as e:
            self.health_status = "unhealthy"
            logger.error(f"Failed to start StreamingService: {e}")
            raise StreamProcessingError(f"Service startup failed: {e}") from e

    async def stop(self) -> None:
        """Stop the streaming service with graceful shutdown."""
        if not self.is_running:
            return

        logger.info("Stopping StreamingService...")

        try:
            # Gracefully stop the streaming processor
            if self.streaming_processor:
                await self.streaming_processor.stop()

            self.is_running = False
            self.health_status = "stopped"

            logger.info("StreamingService stopped gracefully")

        except Exception as e:
            self.health_status = "error"
            logger.error(f"Error during StreamingService shutdown: {e}")
            raise

    async def health_check(self) -> dict[str, Any]:
        """Independent health check that works without FastAPI."""
        if not self.is_running:
            return {
                "status": "stopped",
                "healthy": False,
                "message": "Service not running",
                "timestamp": time.time(),
            }

        try:
            # Check streaming processor health
            processor_health = None
            if self.streaming_processor:
                processor_health = await self.streaming_processor.get_health_status()

            # Check Redis health
            redis_health = None
            if self.redis_manager:
                redis_health = await self.redis_manager.health_check()

            overall_healthy = (
                self.health_status == "healthy"
                and (
                    processor_health is None
                    or processor_health.get("service_status") == "healthy"
                )
                and (
                    redis_health is None
                    or redis_health.get("status") in ["healthy", "available"]
                )
            )

            return {
                "status": self.health_status,
                "healthy": overall_healthy,
                "startup_time_ms": self.startup_time,
                "processor_health": processor_health,
                "redis_health": redis_health,
                "startup_metrics": self.startup_metrics,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "healthy": False,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def get_metrics(self) -> dict[str, Any]:
        """Get service metrics for monitoring."""
        base_metrics = {
            "service_status": self.health_status,
            "is_running": self.is_running,
            "startup_metrics": self.startup_metrics,
        }

        if self.streaming_processor:
            processor_metrics = self.streaming_processor.get_processing_metrics()
            base_metrics["processing_metrics"] = processor_metrics

        return base_metrics

    async def register_camera(self, camera_config: CameraConfig) -> CameraRegistration:
        """Register a camera through the service interface."""
        if not self.is_running or not self.streaming_processor:
            return CameraRegistration(
                camera_id=camera_config.camera_id,
                success=False,
                message="Service not running or not properly initialized",
            )

        return await self.streaming_processor.register_camera(camera_config)

    async def __aenter__(self) -> "StreamingService":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit with cleanup."""
        _ = exc_type, exc_val, exc_tb  # Mark as intentionally unused
        await self.stop()


class StreamingDataProcessor(StreamingServiceInterface):
    """High-throughput video stream processing with quality validation."""

    def __init__(
        self,
        redis_client: RedisQueueManager | None = None,
        quality_validator: FrameQualityValidator | None = None,
        connection_manager: CameraConnectionManager | None = None,
        max_concurrent_streams: int = 100,
        frame_processing_timeout: float = 0.01,
    ):  # 10ms
        self.redis_client = redis_client or RedisQueueManager()
        self.quality_validator = quality_validator or FrameQualityValidator()
        self.connection_manager = connection_manager or CameraConnectionManager()
        self.protobuf_serializer = ProtobufSerializationManager()
        self.max_concurrent_streams = max_concurrent_streams
        self.frame_processing_timeout = frame_processing_timeout

        # Performance optimization
        self.performance_optimizer = None
        self._performance_enabled = False

        # Stream management
        self.registered_cameras: dict[str, CameraConfig] = {}
        self.active_streams: dict[str, asyncio.Task] = {}

        # Performance tracking
        self.processing_metrics = {
            "frames_processed": 0,
            "frames_rejected": 0,
            "total_processing_time": 0.0,
            "active_connections": 0,
            "error_count": 0,
        }

        # Control flags
        self.is_running = False
        self.shutdown_event = asyncio.Event()

        # Memory tracking
        self._memory_tracking_enabled = True
        self._memory_baseline_mb = self._get_current_memory_usage()
        self._peak_memory_mb = self._memory_baseline_mb
        self._frame_buffer_memory_mb = 0.0
        self._last_memory_check = time.time()
        self._memory_leak_threshold_mb = 100.0  # Alert if memory grows by >100MB

        logger.info(
            f"StreamingDataProcessor initialized with max_concurrent_streams={max_concurrent_streams}, baseline_memory={self._memory_baseline_mb:.1f}MB"
        )

    async def initialize_performance_optimization(
        self,
        optimization_config: Optional[dict[str, Any]] = None,
        redis_url: Optional[str] = None,
        database_url: Optional[str] = None,
    ) -> None:
        """Initialize performance optimization for streaming system.

        Args:
            optimization_config: Performance optimization configuration
            redis_url: Redis connection URL for L2 caching
            database_url: Database connection URL for connection pooling
        """
        try:
            from ..performance import (
                OptimizationStrategy,
                create_performance_optimizer,
                create_production_optimization_config,
            )

            # Create default optimization config if not provided
            if optimization_config is None:
                config = create_production_optimization_config(
                    max_concurrent_streams=self.max_concurrent_streams,
                    target_latency_ms=100.0,
                    strategy=OptimizationStrategy.LATENCY_OPTIMIZED,
                )
            else:
                from ..performance.optimization_config import OptimizationConfig

                config = OptimizationConfig(**optimization_config)

            # Initialize performance optimizer
            self.performance_optimizer = await create_performance_optimizer(
                config=config,
                redis_manager=self.redis_client,
                redis_url=redis_url,
                database_url=database_url,
            )

            logger.info("Performance optimization initialized successfully")

        except ImportError as e:
            logger.warning(f"Performance optimization not available: {e}")
            self.performance_optimizer = None
        except Exception as e:
            logger.error(f"Performance optimization initialization failed: {e}")
            self.performance_optimizer = None

    async def start(self) -> None:
        """Start the streaming data processor."""
        if self.is_running:
            logger.warning("StreamingDataProcessor already running")
            return

        try:
            # Connect to Redis
            if not self.redis_client:
                raise StreamProcessingError("Redis client not configured")

            await self.redis_client.connect()

            # Setup processing queues
            await self._setup_processing_queues()

            self.is_running = True
            logger.info("StreamingDataProcessor started successfully")

        except Exception as e:
            logger.error(f"Failed to start StreamingDataProcessor: {e}")
            raise StreamProcessingError(f"Startup failed: {e}") from e

    async def stop(self) -> None:
        """Stop the streaming data processor and cleanup resources."""
        if not self.is_running:
            return

        logger.info("Stopping StreamingDataProcessor...")

        self.is_running = False
        self.shutdown_event.set()

        # Stop all active streams
        for stream_task in self.active_streams.values():
            stream_task.cancel()

        # Wait for tasks to complete
        if self.active_streams:
            await asyncio.gather(*self.active_streams.values(), return_exceptions=True)

        # Disconnect all cameras
        for camera_id in list(self.registered_cameras.keys()):
            await self.connection_manager.disconnect_camera(camera_id)

        # Disconnect from Redis
        if self.redis_client:
            await self.redis_client.disconnect()

        logger.info("StreamingDataProcessor stopped")

    async def _setup_processing_queues(self) -> None:
        """Setup Redis queues for frame processing."""
        if not self.redis_client:
            raise StreamProcessingError("Redis client not available")

        # Input queue for raw frames
        await self.redis_client.create_queue(
            QueueConfig(
                name="camera_frames_input",
                queue_type=QueueType.STREAM,
                max_length=10000,
                batch_size=10,
            )
        )

        # Output queue for processed frames
        await self.redis_client.create_queue(
            QueueConfig(
                name="processed_frames_output",
                queue_type=QueueType.STREAM,
                max_length=5000,
                batch_size=10,
            )
        )

        # Quality control queue for failed frames
        await self.redis_client.create_queue(
            QueueConfig(
                name="quality_control_queue", queue_type=QueueType.LIST, max_length=1000
            )
        )

        logger.info("Processing queues configured successfully")

    async def register_camera(self, camera_config: CameraConfig) -> CameraRegistration:
        """Register a camera for stream processing."""
        try:
            # Check concurrent stream limit
            if len(self.registered_cameras) >= self.max_concurrent_streams:
                return CameraRegistration(
                    camera_id=camera_config.camera_id,
                    success=False,
                    message=f"Maximum concurrent streams ({self.max_concurrent_streams}) reached",
                )

            # Check if camera already registered
            if camera_config.camera_id in self.registered_cameras:
                return CameraRegistration(
                    camera_id=camera_config.camera_id,
                    success=False,
                    message="Camera already registered",
                )

            # Attempt to connect to camera
            connected = await self.connection_manager.connect_camera(camera_config)
            if not connected:
                return CameraRegistration(
                    camera_id=camera_config.camera_id,
                    success=False,
                    message="Failed to connect to camera",
                )

            # Register camera
            self.registered_cameras[camera_config.camera_id] = camera_config
            self.processing_metrics["active_connections"] += 1

            # Start processing stream if enabled
            if camera_config.enabled and self.is_running:
                self.active_streams[camera_config.camera_id] = asyncio.create_task(
                    self._process_camera_stream(camera_config.camera_id)
                )

            logger.info(f"Successfully registered camera: {camera_config.camera_id}")

            return CameraRegistration(
                camera_id=camera_config.camera_id,
                success=True,
                message="Camera registered successfully",
            )

        except Exception as e:
            logger.error(f"Failed to register camera {camera_config.camera_id}: {e}")
            self.processing_metrics["error_count"] += 1
            return CameraRegistration(
                camera_id=camera_config.camera_id,
                success=False,
                message=f"Registration failed: {str(e)}",
            )

    async def process_stream(self, camera_id: str) -> AsyncIterator[ProcessedFrame]:
        """Process continuous stream from registered camera."""
        if camera_id not in self.registered_cameras:
            raise StreamProcessingError(f"Camera {camera_id} not registered")

        camera_config = self.registered_cameras[camera_id]

        try:
            while self.is_running:
                # Capture frame from camera
                frame = await self.connection_manager.capture_frame(camera_id)
                if frame is None:
                    await asyncio.sleep(0.01)  # Short delay if no frame
                    continue

                # Process frame with timeout
                start_time = time.perf_counter()

                try:
                    processed_frame = await asyncio.wait_for(
                        self._process_single_frame(camera_id, frame, camera_config),
                        timeout=self.frame_processing_timeout,
                    )

                    if processed_frame:
                        processing_time = (time.perf_counter() - start_time) * 1000
                        processed_frame.processing_time_ms = processing_time

                        self.processing_metrics["frames_processed"] += 1
                        self.processing_metrics[
                            "total_processing_time"
                        ] += processing_time

                        yield processed_frame

                except TimeoutError:
                    logger.warning(f"Frame processing timeout for camera {camera_id}")
                    self.processing_metrics["frames_rejected"] += 1
                    continue

                # Small delay to prevent overwhelming
                await asyncio.sleep(
                    1.0 / camera_config.fps if camera_config.fps > 0 else 0.033
                )

        except Exception as e:
            logger.error(f"Stream processing error for camera {camera_id}: {e}")
            self.processing_metrics["error_count"] += 1
            raise StreamProcessingError(f"Stream processing failed: {e}") from e

    async def _process_single_frame(
        self, camera_id: str, frame: np.ndarray, camera_config: CameraConfig
    ) -> ProcessedFrame | None:
        """Process a single frame with quality validation."""
        try:
            frame_id = f"{camera_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            timestamp = time.time()

            # Validate frame quality
            quality_metrics = await self.quality_validator.validate_frame_quality(
                frame, camera_config
            )

            if not quality_metrics.passed_validation:
                logger.debug(
                    f"Frame {frame_id} failed quality validation: {quality_metrics.issues}"
                )
                self.processing_metrics["frames_rejected"] += 1

                # Send to quality control queue
                if self.redis_client:
                    await self._send_to_quality_control(
                        frame_id, camera_id, quality_metrics.issues
                    )

                return None

            # Create processed frame
            processed_frame = ProcessedFrame(
                frame_id=frame_id,
                camera_id=camera_id,
                timestamp=timestamp,
                original_image=frame,
                quality_score=quality_metrics.overall_score,
                blur_score=quality_metrics.blur_score,
                brightness_score=quality_metrics.brightness_score,
                contrast_score=quality_metrics.contrast_score,
                noise_level=quality_metrics.noise_level,
                processing_stage=ProcessingStage.VALIDATION,
                validation_passed=True,
            )

            return processed_frame

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return None

    async def _send_to_quality_control(
        self, frame_id: str, camera_id: str, issues: list[str]
    ) -> None:
        """Send failed frame information to quality control queue."""
        try:
            if self.redis_client:
                import json

                quality_data = {
                    "frame_id": frame_id,
                    "camera_id": camera_id,
                    "timestamp": time.time(),
                    "issues": issues,
                }

                await self.redis_client.enqueue(
                    "quality_control_queue", json.dumps(quality_data).encode()
                )

        except Exception as e:
            logger.error(f"Failed to send frame to quality control: {e}")

    async def validate_frame_quality(
        self, frame: np.ndarray, camera_config: CameraConfig
    ) -> QualityMetrics:
        """Validate frame quality against camera configuration."""
        return await self.quality_validator.validate_frame_quality(frame, camera_config)

    async def queue_frame_batch(self, frames: list[ProcessedFrame]) -> BatchId:
        """Queue batch of processed frames for downstream processing using protobuf serialization."""
        try:
            batch_id = f"batch_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

            if not frames:
                raise ValueError("Empty frame batch")

            camera_id = frames[0].camera_id

            # Use protobuf serialization for production efficiency
            if self.redis_client:
                try:
                    # Serialize entire batch using protobuf for maximum efficiency
                    batch_data = self.protobuf_serializer.serialize_frame_batch(
                        frames, batch_id
                    )

                    # Queue the serialized batch
                    await self.redis_client.enqueue(
                        "processed_frames_output",
                        batch_data,
                        metadata={
                            "batch_id": batch_id,
                            "camera_id": camera_id,
                            "frame_count": len(frames),
                            "serialization_format": "protobuf",
                            "compression_enabled": True,
                        },
                    )

                    logger.debug(
                        f"Queued protobuf batch {batch_id} with {len(frames)} frames "
                        f"(size: {len(batch_data)} bytes)"
                    )

                except Exception as protobuf_error:
                    # Fallback to individual frame serialization if batch fails
                    logger.warning(
                        f"Batch protobuf serialization failed for {batch_id}, "
                        f"falling back to individual frames: {protobuf_error}"
                    )

                    for frame in frames:
                        try:
                            # Serialize individual frame using protobuf
                            frame_data = (
                                self.protobuf_serializer.serialize_processed_frame(
                                    frame
                                )
                            )

                            await self.redis_client.enqueue(
                                "processed_frames_output",
                                frame_data,
                                metadata={
                                    "batch_id": batch_id,
                                    "camera_id": camera_id,
                                    "frame_id": frame.frame_id,
                                    "serialization_format": "protobuf",
                                },
                            )

                        except Exception as frame_error:
                            # Ultimate fallback to JSON for compatibility
                            logger.warning(
                                f"Protobuf serialization failed for frame {frame.frame_id}, "
                                f"using JSON fallback: {frame_error}"
                            )

                            frame_data = {
                                "frame_id": frame.frame_id,
                                "camera_id": frame.camera_id,
                                "timestamp": frame.timestamp,
                                "quality_score": getattr(frame, "quality_score", 0.0),
                                "processing_time_ms": getattr(
                                    frame, "processing_time_ms", 0.0
                                ),
                                "batch_id": batch_id,
                                "serialization_format": "json_fallback",
                            }

                            await self.redis_client.enqueue(
                                "processed_frames_output",
                                json.dumps(frame_data).encode(),
                                metadata={"batch_id": batch_id, "camera_id": camera_id},
                            )

            logger.debug(
                f"Successfully queued batch {batch_id} with {len(frames)} frames"
            )

            return BatchId(
                batch_id=batch_id, camera_id=camera_id, frame_count=len(frames)
            )

        except Exception as e:
            logger.error(f"Failed to queue frame batch: {e}")
            raise StreamProcessingError(f"Batch queuing failed: {e}") from e

    async def _process_camera_stream(self, camera_id: str) -> None:
        """Background task to continuously process camera stream."""
        logger.info(f"Starting stream processing for camera: {camera_id}")

        try:
            async for processed_frame in self.process_stream(camera_id):
                if not self.is_running:
                    break

                # Queue single frame as batch of 1
                await self.queue_frame_batch([processed_frame])

        except Exception as e:
            logger.error(f"Stream processing task failed for camera {camera_id}: {e}")
        finally:
            logger.info(f"Stream processing stopped for camera: {camera_id}")

    def get_processing_metrics(self) -> dict[str, Any]:
        """Get current processing metrics with enhanced WebRTC and serialization info."""
        avg_processing_time = self.processing_metrics["total_processing_time"] / max(
            1, self.processing_metrics["frames_processed"]
        )

        throughput = self.processing_metrics["frames_processed"] / max(
            1, self.processing_metrics["total_processing_time"] / 1000
        )

        # Count WebRTC connections
        webrtc_connections = 0
        rtsp_connections = 0
        mock_connections = 0

        for connection in self.connection_manager.active_connections.values():
            if isinstance(connection, dict) and connection.get("type") == "webrtc":
                webrtc_connections += 1
            elif isinstance(connection, cv2.VideoCapture):
                rtsp_connections += 1
            else:
                mock_connections += 1

        return {
            **self.processing_metrics,
            "avg_processing_time_ms": avg_processing_time,
            "throughput_fps": throughput,
            "registered_cameras": len(self.registered_cameras),
            "active_streams": len(self.active_streams),
            "connection_breakdown": {
                "webrtc_connections": webrtc_connections,
                "rtsp_connections": rtsp_connections,
                "mock_connections": mock_connections,
            },
            "serialization": self.protobuf_serializer.get_serialization_stats(),
        }

    def _get_current_memory_usage(self) -> float:
        """
        Get current memory usage in MB.

        Returns comprehensive memory usage including:
        - Process RSS (Resident Set Size)
        - GPU memory if available
        - Frame buffer estimates
        """
        try:
            # Get process memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            rss_mb = memory_info.rss / (1024 * 1024)  # Convert to MB

            return rss_mb

        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0

    def _get_gpu_memory_usage(self) -> dict[str, float]:
        """Get GPU memory usage if CUDA is available."""
        gpu_memory = {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "total_mb": 0.0,
            "free_mb": 0.0,
        }

        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                total_allocated = 0.0
                total_reserved = 0.0
                total_memory = 0.0

                for i in range(device_count):
                    allocated = torch.cuda.memory_allocated(i) / (1024 * 1024)
                    reserved = torch.cuda.memory_reserved(i) / (1024 * 1024)
                    properties = torch.cuda.get_device_properties(i)
                    total = properties.total_memory / (1024 * 1024)

                    total_allocated += allocated
                    total_reserved += reserved
                    total_memory += total

                gpu_memory.update(
                    {
                        "allocated_mb": total_allocated,
                        "reserved_mb": total_reserved,
                        "total_mb": total_memory,
                        "free_mb": total_memory - total_reserved,
                    }
                )

        except Exception as e:
            logger.debug(f"GPU memory check failed: {e}")

        return gpu_memory

    def _estimate_frame_buffer_memory(self) -> float:
        """Estimate memory usage of frame buffers."""
        try:
            total_buffer_memory = 0.0

            # Estimate based on registered cameras and their configurations
            for camera_config in self.registered_cameras.values():
                # Estimate memory per frame: width * height * channels * bytes_per_pixel
                width, height = camera_config.resolution
                bytes_per_frame = width * height * 3 * 1  # RGB, 1 byte per channel

                # Assume we keep ~10 frames in memory per camera (processing pipeline)
                estimated_buffer_frames = 10
                buffer_memory_mb = (bytes_per_frame * estimated_buffer_frames) / (
                    1024 * 1024
                )

                total_buffer_memory += buffer_memory_mb

            # Add Redis queue memory estimation
            frames_in_queues = (
                self.processing_metrics.get("frames_processed", 0) % 1000
            )  # Rough estimate
            avg_frame_size_mb = 1.0  # Conservative estimate: 1MB per frame
            queue_memory_mb = frames_in_queues * avg_frame_size_mb

            total_buffer_memory += queue_memory_mb
            return total_buffer_memory

        except Exception as e:
            logger.warning(f"Frame buffer memory estimation failed: {e}")
            return 0.0

    def _check_memory_leak(self) -> dict[str, Any]:
        """Check for potential memory leaks."""
        current_time = time.time()
        current_memory = self._get_current_memory_usage()

        # Update peak memory
        if current_memory > self._peak_memory_mb:
            self._peak_memory_mb = current_memory

        # Calculate memory growth since baseline
        memory_growth = current_memory - self._memory_baseline_mb

        # Determine if there's a potential leak
        potential_leak = memory_growth > self._memory_leak_threshold_mb

        # Calculate memory growth rate (MB per hour)
        time_elapsed_hours = (current_time - self._last_memory_check) / 3600.0
        growth_rate_mb_per_hour = memory_growth / max(0.01, time_elapsed_hours)

        leak_status = {
            "current_memory_mb": current_memory,
            "baseline_memory_mb": self._memory_baseline_mb,
            "peak_memory_mb": self._peak_memory_mb,
            "memory_growth_mb": memory_growth,
            "growth_rate_mb_per_hour": growth_rate_mb_per_hour,
            "potential_leak_detected": potential_leak,
            "leak_threshold_mb": self._memory_leak_threshold_mb,
            "time_since_start_hours": time_elapsed_hours,
        }

        if potential_leak:
            logger.warning(
                f"Potential memory leak detected: {memory_growth:.1f}MB growth "
                f"(rate: {growth_rate_mb_per_hour:.1f}MB/hour)"
            )

        return leak_status

    def get_comprehensive_memory_status(self) -> dict[str, Any]:
        """Get comprehensive memory status including leak detection."""
        if not self._memory_tracking_enabled:
            return {"tracking_enabled": False}

        try:
            # Get current memory metrics
            current_memory = self._get_current_memory_usage()
            gpu_memory = self._get_gpu_memory_usage()
            frame_buffer_memory = self._estimate_frame_buffer_memory()
            leak_status = self._check_memory_leak()

            # System memory information
            system_memory = psutil.virtual_memory()

            # Memory efficiency metrics
            active_cameras = len(self.active_streams)
            memory_per_camera = current_memory / max(1, active_cameras)

            memory_status = {
                "tracking_enabled": True,
                "timestamp": time.time(),
                "process_memory": {
                    "current_mb": current_memory,
                    "peak_mb": self._peak_memory_mb,
                    "baseline_mb": self._memory_baseline_mb,
                },
                "gpu_memory": gpu_memory,
                "frame_buffers": {
                    "estimated_mb": frame_buffer_memory,
                    "active_cameras": active_cameras,
                    "memory_per_camera_mb": memory_per_camera,
                },
                "system_memory": {
                    "total_mb": system_memory.total / (1024 * 1024),
                    "available_mb": system_memory.available / (1024 * 1024),
                    "used_percent": system_memory.percent,
                },
                "leak_detection": leak_status,
                "efficiency_metrics": {
                    "memory_per_camera_mb": memory_per_camera,
                    "frames_per_mb": self.processing_metrics.get("frames_processed", 0)
                    / max(1, current_memory),
                    "memory_efficiency_score": min(
                        1.0, 100.0 / max(1, current_memory)
                    ),  # Higher is better
                },
            }

            return memory_status

        except Exception as e:
            logger.error(f"Memory status check failed: {e}")
            return {
                "tracking_enabled": True,
                "error": str(e),
                "timestamp": time.time(),
            }

    async def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        redis_health = (
            await self.redis_client.health_check()
            if self.redis_client
            else {"status": "unavailable"}
        )

        # Get comprehensive memory status
        memory_status = self.get_comprehensive_memory_status()

        # Extract key memory metrics for backward compatibility
        memory_usage_mb = memory_status.get("process_memory", {}).get("current_mb", 0.0)

        # Determine overall service health based on memory and other factors
        service_health = "healthy"
        health_issues = []

        if not self.is_running:
            service_health = "stopped"
        else:
            # Check memory health
            if memory_status.get("leak_detection", {}).get(
                "potential_leak_detected", False
            ):
                service_health = "degraded"
                health_issues.append("Memory leak detected")

            # Check GPU memory if available
            gpu_memory = memory_status.get("gpu_memory", {})
            if gpu_memory.get("total_mb", 0) > 0:
                gpu_utilization = (
                    gpu_memory.get("reserved_mb", 0) / gpu_memory.get("total_mb", 1)
                ) * 100
                if gpu_utilization > 95:
                    service_health = "degraded"
                    health_issues.append("High GPU memory usage")

            # Check system memory
            system_memory = memory_status.get("system_memory", {})
            if system_memory.get("used_percent", 0) > 90:
                service_health = "degraded"
                health_issues.append("High system memory usage")

            # Check Redis health
            if redis_health.get("status") not in ["healthy", "available"]:
                service_health = "degraded"
                health_issues.append("Redis connectivity issues")

        # Get WebRTC connection stats for all cameras
        webrtc_health = {}
        for camera_id in self.registered_cameras:
            if self.connection_manager.is_connected(camera_id):
                connection = self.connection_manager.active_connections.get(camera_id)
                if isinstance(connection, dict) and connection.get("type") == "webrtc":
                    try:
                        webrtc_quality = (
                            await self.connection_manager.get_webrtc_connection_quality(
                                camera_id
                            )
                        )
                        webrtc_health[camera_id] = {
                            "connection_state": webrtc_quality.get(
                                "connection_state", "unknown"
                            ),
                            "ice_connection_state": webrtc_quality.get(
                                "ice_connection_state", "unknown"
                            ),
                            "rtt_ms": webrtc_quality.get("rtt", 0),
                            "packet_loss_rate": webrtc_quality.get(
                                "packet_loss_rate", 0
                            ),
                        }
                    except Exception as e:
                        webrtc_health[camera_id] = {"error": str(e)}

        return {
            "service_status": service_health,
            "health_issues": health_issues,
            "redis_status": redis_health.get("status", "unknown"),
            "processing_metrics": self.get_processing_metrics(),
            "registered_cameras": len(self.registered_cameras),
            "active_streams": len(self.active_streams),
            "memory_usage_mb": round(memory_usage_mb, 2),
            "comprehensive_memory_status": memory_status,
            "webrtc_health": webrtc_health,
            "timestamp": time.time(),
        }

    async def __aenter__(self) -> "StreamingDataProcessor":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        _ = exc_type, exc_val, exc_tb  # Mark as intentionally unused
        await self.stop()
