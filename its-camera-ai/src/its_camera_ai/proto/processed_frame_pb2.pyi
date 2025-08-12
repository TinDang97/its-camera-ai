from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class ProcessingStage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    PROCESSING_STAGE_UNSPECIFIED: _ClassVar[ProcessingStage]
    PROCESSING_STAGE_INGESTION: _ClassVar[ProcessingStage]
    PROCESSING_STAGE_VALIDATION: _ClassVar[ProcessingStage]
    PROCESSING_STAGE_FEATURE_EXTRACTION: _ClassVar[ProcessingStage]
    PROCESSING_STAGE_QUALITY_CONTROL: _ClassVar[ProcessingStage]
    PROCESSING_STAGE_OUTPUT: _ClassVar[ProcessingStage]

class StreamStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    STREAM_STATUS_UNSPECIFIED: _ClassVar[StreamStatus]
    STREAM_STATUS_ACTIVE: _ClassVar[StreamStatus]
    STREAM_STATUS_INACTIVE: _ClassVar[StreamStatus]
    STREAM_STATUS_ERROR: _ClassVar[StreamStatus]
    STREAM_STATUS_MAINTENANCE: _ClassVar[StreamStatus]

PROCESSING_STAGE_UNSPECIFIED: ProcessingStage
PROCESSING_STAGE_INGESTION: ProcessingStage
PROCESSING_STAGE_VALIDATION: ProcessingStage
PROCESSING_STAGE_FEATURE_EXTRACTION: ProcessingStage
PROCESSING_STAGE_QUALITY_CONTROL: ProcessingStage
PROCESSING_STAGE_OUTPUT: ProcessingStage
STREAM_STATUS_UNSPECIFIED: StreamStatus
STREAM_STATUS_ACTIVE: StreamStatus
STREAM_STATUS_INACTIVE: StreamStatus
STREAM_STATUS_ERROR: StreamStatus
STREAM_STATUS_MAINTENANCE: StreamStatus

class QualityMetrics(_message.Message):
    __slots__ = (
        "quality_score",
        "blur_score",
        "brightness_score",
        "contrast_score",
        "noise_level",
    )
    QUALITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    BLUR_SCORE_FIELD_NUMBER: _ClassVar[int]
    BRIGHTNESS_SCORE_FIELD_NUMBER: _ClassVar[int]
    CONTRAST_SCORE_FIELD_NUMBER: _ClassVar[int]
    NOISE_LEVEL_FIELD_NUMBER: _ClassVar[int]
    quality_score: float
    blur_score: float
    brightness_score: float
    contrast_score: float
    noise_level: float
    def __init__(
        self,
        quality_score: float | None = ...,
        blur_score: float | None = ...,
        brightness_score: float | None = ...,
        contrast_score: float | None = ...,
        noise_level: float | None = ...,
    ) -> None: ...

class TrafficFeatures(_message.Message):
    __slots__ = (
        "vehicle_density",
        "congestion_level",
        "weather_conditions",
        "lighting_conditions",
        "motion_intensity",
    )
    VEHICLE_DENSITY_FIELD_NUMBER: _ClassVar[int]
    CONGESTION_LEVEL_FIELD_NUMBER: _ClassVar[int]
    WEATHER_CONDITIONS_FIELD_NUMBER: _ClassVar[int]
    LIGHTING_CONDITIONS_FIELD_NUMBER: _ClassVar[int]
    MOTION_INTENSITY_FIELD_NUMBER: _ClassVar[int]
    vehicle_density: float
    congestion_level: str
    weather_conditions: str
    lighting_conditions: str
    motion_intensity: float
    def __init__(
        self,
        vehicle_density: float | None = ...,
        congestion_level: str | None = ...,
        weather_conditions: str | None = ...,
        lighting_conditions: str | None = ...,
        motion_intensity: float | None = ...,
    ) -> None: ...

class ROIAnalysis(_message.Message):
    __slots__ = (
        "roi_id",
        "density",
        "brightness",
        "congestion",
        "x",
        "y",
        "width",
        "height",
    )
    ROI_ID_FIELD_NUMBER: _ClassVar[int]
    DENSITY_FIELD_NUMBER: _ClassVar[int]
    BRIGHTNESS_FIELD_NUMBER: _ClassVar[int]
    CONGESTION_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    roi_id: str
    density: float
    brightness: float
    congestion: str
    x: int
    y: int
    width: int
    height: int
    def __init__(
        self,
        roi_id: str | None = ...,
        density: float | None = ...,
        brightness: float | None = ...,
        congestion: str | None = ...,
        x: int | None = ...,
        y: int | None = ...,
        width: int | None = ...,
        height: int | None = ...,
    ) -> None: ...

class ImageData(_message.Message):
    __slots__ = (
        "compressed_data",
        "width",
        "height",
        "channels",
        "compression_format",
        "quality",
    )
    COMPRESSED_DATA_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_FORMAT_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    compressed_data: bytes
    width: int
    height: int
    channels: int
    compression_format: str
    quality: int
    def __init__(
        self,
        compressed_data: bytes | None = ...,
        width: int | None = ...,
        height: int | None = ...,
        channels: int | None = ...,
        compression_format: str | None = ...,
        quality: int | None = ...,
    ) -> None: ...

class ProcessedFrame(_message.Message):
    __slots__ = (
        "frame_id",
        "camera_id",
        "timestamp",
        "original_image",
        "processed_image",
        "thumbnail",
        "quality_metrics",
        "traffic_features",
        "roi_features",
        "processing_time_ms",
        "processing_stage",
        "validation_passed",
        "source_hash",
        "version",
        "received_timestamp",
        "processed_timestamp",
    )
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ORIGINAL_IMAGE_FIELD_NUMBER: _ClassVar[int]
    PROCESSED_IMAGE_FIELD_NUMBER: _ClassVar[int]
    THUMBNAIL_FIELD_NUMBER: _ClassVar[int]
    QUALITY_METRICS_FIELD_NUMBER: _ClassVar[int]
    TRAFFIC_FEATURES_FIELD_NUMBER: _ClassVar[int]
    ROI_FEATURES_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_STAGE_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_PASSED_FIELD_NUMBER: _ClassVar[int]
    SOURCE_HASH_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    RECEIVED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    PROCESSED_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    frame_id: str
    camera_id: str
    timestamp: float
    original_image: ImageData
    processed_image: ImageData
    thumbnail: ImageData
    quality_metrics: QualityMetrics
    traffic_features: TrafficFeatures
    roi_features: _containers.RepeatedCompositeFieldContainer[ROIAnalysis]
    processing_time_ms: float
    processing_stage: ProcessingStage
    validation_passed: bool
    source_hash: str
    version: str
    received_timestamp: float
    processed_timestamp: float
    def __init__(
        self,
        frame_id: str | None = ...,
        camera_id: str | None = ...,
        timestamp: float | None = ...,
        original_image: ImageData | _Mapping | None = ...,
        processed_image: ImageData | _Mapping | None = ...,
        thumbnail: ImageData | _Mapping | None = ...,
        quality_metrics: QualityMetrics | _Mapping | None = ...,
        traffic_features: TrafficFeatures | _Mapping | None = ...,
        roi_features: _Iterable[ROIAnalysis | _Mapping] | None = ...,
        processing_time_ms: float | None = ...,
        processing_stage: ProcessingStage | str | None = ...,
        validation_passed: bool = ...,
        source_hash: str | None = ...,
        version: str | None = ...,
        received_timestamp: float | None = ...,
        processed_timestamp: float | None = ...,
    ) -> None: ...

class CameraStreamConfig(_message.Message):
    __slots__ = (
        "camera_id",
        "location",
        "latitude",
        "longitude",
        "width",
        "height",
        "fps",
        "encoding",
        "roi_boxes",
        "quality_threshold",
        "processing_enabled",
        "status",
        "last_frame_time",
        "total_frames_processed",
    )
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    LATITUDE_FIELD_NUMBER: _ClassVar[int]
    LONGITUDE_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FPS_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    ROI_BOXES_FIELD_NUMBER: _ClassVar[int]
    QUALITY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_ENABLED_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    LAST_FRAME_TIME_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FRAMES_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    camera_id: str
    location: str
    latitude: float
    longitude: float
    width: int
    height: int
    fps: int
    encoding: str
    roi_boxes: _containers.RepeatedCompositeFieldContainer[ROIBox]
    quality_threshold: float
    processing_enabled: bool
    status: StreamStatus
    last_frame_time: float
    total_frames_processed: int
    def __init__(
        self,
        camera_id: str | None = ...,
        location: str | None = ...,
        latitude: float | None = ...,
        longitude: float | None = ...,
        width: int | None = ...,
        height: int | None = ...,
        fps: int | None = ...,
        encoding: str | None = ...,
        roi_boxes: _Iterable[ROIBox | _Mapping] | None = ...,
        quality_threshold: float | None = ...,
        processing_enabled: bool = ...,
        status: StreamStatus | str | None = ...,
        last_frame_time: float | None = ...,
        total_frames_processed: int | None = ...,
    ) -> None: ...

class ROIBox(_message.Message):
    __slots__ = ("x", "y", "width", "height", "label")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    width: int
    height: int
    label: str
    def __init__(
        self,
        x: int | None = ...,
        y: int | None = ...,
        width: int | None = ...,
        height: int | None = ...,
        label: str | None = ...,
    ) -> None: ...

class QueueMetrics(_message.Message):
    __slots__ = (
        "queue_name",
        "pending_count",
        "processing_count",
        "completed_count",
        "failed_count",
        "avg_processing_time_ms",
        "throughput_fps",
    )
    QUEUE_NAME_FIELD_NUMBER: _ClassVar[int]
    PENDING_COUNT_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_COUNT_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_COUNT_FIELD_NUMBER: _ClassVar[int]
    FAILED_COUNT_FIELD_NUMBER: _ClassVar[int]
    AVG_PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    THROUGHPUT_FPS_FIELD_NUMBER: _ClassVar[int]
    queue_name: str
    pending_count: int
    processing_count: int
    completed_count: int
    failed_count: int
    avg_processing_time_ms: float
    throughput_fps: float
    def __init__(
        self,
        queue_name: str | None = ...,
        pending_count: int | None = ...,
        processing_count: int | None = ...,
        completed_count: int | None = ...,
        failed_count: int | None = ...,
        avg_processing_time_ms: float | None = ...,
        throughput_fps: float | None = ...,
    ) -> None: ...

class ProcessedFrameBatch(_message.Message):
    __slots__ = ("frames", "batch_id", "batch_timestamp", "batch_size")
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    BATCH_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    frames: _containers.RepeatedCompositeFieldContainer[ProcessedFrame]
    batch_id: str
    batch_timestamp: float
    batch_size: int
    def __init__(
        self,
        frames: _Iterable[ProcessedFrame | _Mapping] | None = ...,
        batch_id: str | None = ...,
        batch_timestamp: float | None = ...,
        batch_size: int | None = ...,
    ) -> None: ...

class ProcessingError(_message.Message):
    __slots__ = (
        "error_id",
        "frame_id",
        "camera_id",
        "error_type",
        "error_message",
        "timestamp",
        "failed_stage",
    )
    ERROR_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    ERROR_TYPE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    FAILED_STAGE_FIELD_NUMBER: _ClassVar[int]
    error_id: str
    frame_id: str
    camera_id: str
    error_type: str
    error_message: str
    timestamp: float
    failed_stage: ProcessingStage
    def __init__(
        self,
        error_id: str | None = ...,
        frame_id: str | None = ...,
        camera_id: str | None = ...,
        error_type: str | None = ...,
        error_message: str | None = ...,
        timestamp: float | None = ...,
        failed_stage: ProcessingStage | str | None = ...,
    ) -> None: ...
