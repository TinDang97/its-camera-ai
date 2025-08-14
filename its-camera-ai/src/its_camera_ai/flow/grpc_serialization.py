"""gRPC Serialization utilities for ProcessedFrame optimization.

This module provides efficient serialization/deserialization for ProcessedFrame
data using Protocol Buffers, optimized for high-throughput video processing.

Key Features:
- Efficient binary serialization with protobuf
- Image compression (JPEG/WebP) for reduced payload size
- Automatic compression level optimization
- Batch serialization for improved performance
- Error handling and validation
"""

from __future__ import annotations

import io
import logging
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .streaming_processor import ProcessedFrame

# Import generated protobuf classes
# Note: These will be generated from the .proto files
try:
    from its_camera_ai.proto import processed_frame_pb2 as pb
except ImportError:
    # Fallback for development - these will be generated
    class MockProto:
        def __init__(self) -> None:
            pass

    pb_module = MockProto()
    pb_module.ProcessedFrame = dict  # type: ignore
    pb_module.ImageData = dict  # type: ignore
    pb_module.QualityMetrics = dict  # type: ignore
    pb_module.TrafficFeatures = dict  # type: ignore
    pb_module.ROIAnalysis = dict  # type: ignore
    pb_module.ProcessedFrameBatch = dict  # type: ignore

    pb = pb_module  # type: ignore


logger = logging.getLogger(__name__)


class ImageCompressor:
    """High-performance image compression for gRPC payloads."""

    def __init__(
        self,
        default_format: str = "jpeg",
        default_quality: int = 85,
        thumbnail_size: tuple[int, int] = (128, 128),
    ):
        """Initialize image compressor.

        Args:
            default_format: Default compression format (jpeg, png, webp)
            default_quality: Default compression quality (1-100)
            thumbnail_size: Size for thumbnail generation
        """
        self.default_format = default_format.lower()
        self.default_quality = default_quality
        self.thumbnail_size = thumbnail_size

        # Supported formats
        self.supported_formats = {"jpeg", "png", "webp"}

        # Format-specific settings
        self.format_settings = {
            "jpeg": {"format": "JPEG", "quality": default_quality, "optimize": True},
            "png": {"format": "PNG", "optimize": True},
            "webp": {"format": "WEBP", "quality": default_quality, "method": 6},
        }

    def compress_image(
        self,
        image: np.ndarray[Any, Any],
        format_type: str | None = None,
        quality: int | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        """Compress numpy image array to bytes.

        Args:
            image: Input image array (H, W, C) in RGB format
            format_type: Compression format override
            quality: Quality override

        Returns:
            tuple[bytes, dict]: Compressed image bytes and metadata
        """
        if image is None or image.size == 0:
            return b"", {}

        format_type = format_type or self.default_format
        quality = quality or self.default_quality

        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}")

        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB image
                pil_image = Image.fromarray(image, mode="RGB")
            elif len(image.shape) == 2:
                # Grayscale image
                pil_image = Image.fromarray(image, mode="L")
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")

            # Prepare compression settings
            settings = self.format_settings[format_type].copy()
            if "quality" in settings:
                settings["quality"] = quality

            # Compress to bytes
            buffer = io.BytesIO()
            # Save with explicit format parameter
            fmt = str(settings.pop("format", "JPEG"))
            pil_image.save(buffer, format=fmt, **settings)  # type: ignore
            compressed_data = buffer.getvalue()

            # Metadata
            metadata = {
                "original_size": image.nbytes,
                "compressed_size": len(compressed_data),
                "compression_ratio": len(compressed_data) / image.nbytes,
                "format": format_type,
                "quality": quality,
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
            }

            return compressed_data, metadata

        except Exception as e:
            logger.error(f"Image compression failed: {e}")
            raise

    def decompress_image(
        self, compressed_data: bytes, target_shape: tuple[int, ...] | None = None
    ) -> np.ndarray[Any, Any]:
        """Decompress bytes back to numpy array.

        Args:
            compressed_data: Compressed image bytes
            target_shape: Optional target shape for resizing

        Returns:
            np.ndarray: Decompressed image array
        """
        if not compressed_data:
            return np.array([])

        try:
            # Load from bytes
            buffer = io.BytesIO(compressed_data)
            pil_image = Image.open(buffer)

            # Convert to numpy array
            image_array = np.array(pil_image)

            # Resize if target shape specified
            if target_shape and len(target_shape) >= 2:
                height, width = target_shape[:2]
                if len(image_array.shape) == 3:
                    image_array = cv2.resize(image_array, (width, height))
                else:
                    image_array = cv2.resize(image_array, (width, height))
                    if len(target_shape) == 3 and target_shape[2] == 3:
                        # Convert grayscale to RGB
                        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

            return image_array

        except Exception as e:
            logger.error(f"Image decompression failed: {e}")
            return np.array([])

    def create_thumbnail(
        self, image: np.ndarray[Any, Any]
    ) -> tuple[bytes, dict[str, Any]]:
        """Create compressed thumbnail from image.

        Args:
            image: Input image array

        Returns:
            tuple[bytes, dict]: Thumbnail bytes and metadata
        """
        if image is None or image.size == 0:
            return b"", {}

        try:
            # Resize to thumbnail size
            thumbnail = cv2.resize(
                image, self.thumbnail_size, interpolation=cv2.INTER_AREA
            )

            # Compress with higher quality for thumbnails
            return self.compress_image(thumbnail, quality=90)

        except Exception as e:
            logger.error(f"Thumbnail creation failed: {e}")
            return b"", {}


class ProcessedFrameSerializer:
    """High-performance serializer for ProcessedFrame using gRPC."""

    def __init__(
        self,
        compression_format: str = "jpeg",
        compression_quality: int = 85,
        enable_compression: bool = True,
    ):
        """Initialize ProcessedFrame serializer.

        Args:
            compression_format: Image compression format
            compression_quality: Compression quality (1-100)
            enable_compression: Whether to enable image compression
        """
        self.compressor = ImageCompressor(
            default_format=compression_format, default_quality=compression_quality
        )
        self.enable_compression = enable_compression

        # Performance metrics
        self.serialization_times: list[float] = []
        self.compression_ratios: list[float] = []

    def serialize_processed_frame(self, frame: ProcessedFrame) -> bytes:
        """Serialize ProcessedFrame to protobuf bytes.

        Args:
            frame: ProcessedFrame instance from streaming_processor

        Returns:
            bytes: Serialized protobuf data
        """
        start_time = time.time()

        try:
            # Create protobuf message
            pb_frame = pb.ProcessedFrame()  # type: ignore

            # Core identifiers
            pb_frame.frame_id = frame.frame_id
            pb_frame.camera_id = frame.camera_id
            pb_frame.timestamp = frame.timestamp

            # Compress and set image data
            if frame.original_image is not None and frame.original_image.size > 0:
                pb_frame.original_image.CopyFrom(
                    self._serialize_image_data(frame.original_image, "original")
                )

            if frame.processed_image is not None and frame.processed_image.size > 0:
                pb_frame.processed_image.CopyFrom(
                    self._serialize_image_data(frame.processed_image, "processed")
                )

            if frame.thumbnail is not None and frame.thumbnail.size > 0:
                pb_frame.thumbnail.CopyFrom(
                    self._serialize_image_data(frame.thumbnail, "thumbnail")
                )

            # Quality metrics
            pb_frame.quality_metrics.quality_score = frame.quality_score
            pb_frame.quality_metrics.blur_score = frame.blur_score
            pb_frame.quality_metrics.brightness_score = frame.brightness_score
            pb_frame.quality_metrics.contrast_score = frame.contrast_score
            pb_frame.quality_metrics.noise_level = frame.noise_level

            # Traffic features
            pb_frame.traffic_features.vehicle_density = frame.vehicle_density
            pb_frame.traffic_features.congestion_level = frame.congestion_level
            pb_frame.traffic_features.weather_conditions = frame.weather_conditions
            pb_frame.traffic_features.lighting_conditions = frame.lighting_conditions

            # ROI features
            if hasattr(frame, "roi_features") and frame.roi_features:
                for roi_id, roi_data in frame.roi_features.items():
                    roi_analysis = pb_frame.roi_features.add()
                    roi_analysis.roi_id = roi_id
                    if isinstance(roi_data, dict):
                        roi_analysis.density = roi_data.get("density", 0.0)
                        roi_analysis.brightness = roi_data.get("brightness", 0.0)
                        roi_analysis.congestion = roi_data.get("congestion", "unknown")

            # Processing metadata
            pb_frame.processing_time_ms = frame.processing_time_ms
            pb_frame.processing_stage = self._convert_processing_stage(
                frame.processing_stage
            )
            pb_frame.validation_passed = frame.validation_passed

            # Data lineage
            pb_frame.source_hash = frame.source_hash
            pb_frame.version = frame.version

            # Timestamps
            pb_frame.received_timestamp = time.time()
            pb_frame.processed_timestamp = frame.timestamp

            # Serialize to bytes
            serialized_data = pb_frame.SerializeToString()

            # Track performance
            serialization_time = (time.time() - start_time) * 1000
            self.serialization_times.append(serialization_time)

            logger.debug(
                f"Serialized frame {frame.frame_id}: {len(serialized_data)} bytes in {serialization_time:.2f}ms"
            )

            return serialized_data  # type: ignore

        except Exception as e:
            logger.error(f"Failed to serialize ProcessedFrame: {e}")
            raise

    def _serialize_image_data(
        self, image: np.ndarray[Any, Any], image_type: str
    ) -> Any:
        """Serialize image data to protobuf ImageData.

        Args:
            image: Numpy image array
            image_type: Type of image (original, processed, thumbnail)

        Returns:
            pb.ImageData: Protobuf image data
        """
        image_data = pb.ImageData()  # type: ignore

        if image is None or image.size == 0:
            return image_data

        try:
            # Set dimensions
            if len(image.shape) == 3:
                height, width, channels = image.shape
            else:
                height, width = image.shape
                channels = 1

            image_data.width = width
            image_data.height = height
            image_data.channels = channels

            if self.enable_compression:
                # Compress image
                quality = 90 if image_type == "thumbnail" else 85
                compressed_data, metadata = self.compressor.compress_image(
                    image, quality=quality
                )

                image_data.compressed_data = compressed_data
                image_data.compression_format = metadata["format"]
                image_data.quality = metadata["quality"]

                # Track compression ratio
                self.compression_ratios.append(metadata["compression_ratio"])

            else:
                # Store raw data (not recommended for production)
                image_data.compressed_data = image.tobytes()
                image_data.compression_format = "raw"
                image_data.quality = 100

            return image_data

        except Exception as e:
            logger.error(f"Failed to serialize image data: {e}")
            return image_data

    def deserialize_processed_frame(self, serialized_data: bytes) -> ProcessedFrame:
        """Deserialize protobuf bytes to ProcessedFrame.

        Args:
            serialized_data: Serialized protobuf data

        Returns:
            ProcessedFrame: Deserialized frame object
        """
        try:
            # Parse protobuf
            pb_frame = pb.ProcessedFrame()
            pb_frame.ParseFromString(serialized_data)

            # Import ProcessedFrame here to avoid circular imports
            from .streaming_processor import ProcessedFrame

            # Deserialize image data
            original_image = self._deserialize_image_data(pb_frame.original_image)
            processed_image = self._deserialize_image_data(pb_frame.processed_image)
            thumbnail = self._deserialize_image_data(pb_frame.thumbnail)

            # Create ProcessedFrame instance
            frame = ProcessedFrame(
                frame_id=pb_frame.frame_id,
                camera_id=pb_frame.camera_id,
                timestamp=pb_frame.timestamp,
                original_image=original_image or np.array([]),
                processed_image=processed_image,
                thumbnail=thumbnail,
            )

            # Set quality metrics
            frame.quality_score = pb_frame.quality_metrics.quality_score
            frame.blur_score = pb_frame.quality_metrics.blur_score
            frame.brightness_score = pb_frame.quality_metrics.brightness_score
            frame.contrast_score = pb_frame.quality_metrics.contrast_score
            frame.noise_level = pb_frame.quality_metrics.noise_level

            # Set traffic features
            frame.vehicle_density = pb_frame.traffic_features.vehicle_density
            frame.congestion_level = pb_frame.traffic_features.congestion_level
            frame.weather_conditions = pb_frame.traffic_features.weather_conditions
            frame.lighting_conditions = pb_frame.traffic_features.lighting_conditions

            # Set ROI features
            roi_features = {}
            for roi in pb_frame.roi_features:
                roi_features[roi.roi_id] = {
                    "density": roi.density,
                    "brightness": roi.brightness,
                    "congestion": roi.congestion,
                }
            frame.roi_features = roi_features

            # Set processing metadata
            frame.processing_time_ms = pb_frame.processing_time_ms
            frame.processing_stage = self._convert_processing_stage_from_pb(
                pb_frame.processing_stage
            )
            frame.validation_passed = pb_frame.validation_passed

            # Set data lineage
            frame.source_hash = pb_frame.source_hash
            frame.version = pb_frame.version

            return frame

        except Exception as e:
            logger.error(f"Failed to deserialize ProcessedFrame: {e}")
            raise

    def _deserialize_image_data(self, image_data: Any) -> np.ndarray[Any, Any] | None:
        """Deserialize protobuf ImageData to numpy array.

        Args:
            image_data: Protobuf image data

        Returns:
            np.ndarray: Decompressed image array
        """
        if not image_data.compressed_data:
            return None

        try:
            if image_data.compression_format == "raw":
                # Raw data - reconstruct array
                shape: tuple[int, ...] = (image_data.height, image_data.width)
                if image_data.channels > 1:
                    shape = shape + (image_data.channels,)

                return np.frombuffer(
                    image_data.compressed_data, dtype=np.uint8
                ).reshape(shape)

            else:
                # Compressed data - decompress
                target_shape = (
                    image_data.height,
                    image_data.width,
                    image_data.channels,
                )
                return self.compressor.decompress_image(
                    image_data.compressed_data, target_shape
                )

        except Exception as e:
            logger.error(f"Failed to deserialize image data: {e}")
            return None

    def _convert_processing_stage(self, stage: Any) -> int:
        """Convert ProcessingStage enum to protobuf enum."""
        # This would map to the actual protobuf enum values
        stage_mapping = {
            "ingestion": 1,
            "validation": 2,
            "feature_extraction": 3,
            "quality_control": 4,
            "output": 5,
        }

        if hasattr(stage, "value"):
            return stage_mapping.get(stage.value, 0)
        elif isinstance(stage, str):
            return stage_mapping.get(stage, 0)
        else:
            return 0

    def _convert_processing_stage_from_pb(self, pb_stage: int) -> Any:
        """Convert protobuf enum to ProcessingStage."""
        from its_camera_ai.data.streaming_processor import ProcessingStage

        stage_mapping = {
            1: ProcessingStage.INGESTION,
            2: ProcessingStage.VALIDATION,
            3: ProcessingStage.FEATURE_EXTRACTION,
            4: ProcessingStage.QUALITY_CONTROL,
            5: ProcessingStage.OUTPUT,
        }

        return stage_mapping.get(pb_stage, ProcessingStage.INGESTION)

    def serialize_batch(
        self, frames: list[ProcessedFrame], batch_id: str | None = None
    ) -> bytes:
        """Serialize multiple ProcessedFrames into a batch.

        Args:
            frames: List of ProcessedFrame objects
            batch_id: Optional batch identifier

        Returns:
            bytes: Serialized batch data
        """
        try:
            batch = pb.ProcessedFrameBatch()  # type: ignore

            # Set batch metadata
            batch.batch_id = batch_id or f"batch_{int(time.time() * 1000)}"
            batch.batch_timestamp = time.time()
            batch.batch_size = len(frames)

            # Serialize each frame
            for frame in frames:
                frame_data = self.serialize_processed_frame(frame)
                # Note: This would require modification to protobuf to include serialized frames
                # For now, we'll serialize each frame separately
                pb_frame = pb.ProcessedFrame()  # type: ignore
                pb_frame.ParseFromString(frame_data)
                batch.frames.append(pb_frame)

            return batch.SerializeToString()  # type: ignore

        except Exception as e:
            logger.error(f"Failed to serialize batch: {e}")
            raise

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get serialization performance metrics.

        Returns:
            dict[str, Any]: Performance metrics
        """
        if not self.serialization_times:
            return {}

        return {
            "avg_serialization_time_ms": sum(self.serialization_times)
            / len(self.serialization_times),
            "max_serialization_time_ms": max(self.serialization_times),
            "min_serialization_time_ms": min(self.serialization_times),
            "total_serializations": len(self.serialization_times),
            "avg_compression_ratio": (
                sum(self.compression_ratios) / len(self.compression_ratios)
                if self.compression_ratios
                else 0
            ),
            "compression_enabled": self.enable_compression,
        }
