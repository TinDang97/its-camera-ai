from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

import processed_frame_pb2 as _processed_frame_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import service as _service
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class StreamResponse(_message.Message):
    __slots__ = ("success", "message", "frame_id", "processing_time_ms")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    frame_id: str
    processing_time_ms: float
    def __init__(
        self,
        success: bool = ...,
        message: str | None = ...,
        frame_id: str | None = ...,
        processing_time_ms: float | None = ...,
    ) -> None: ...

class BatchResponse(_message.Message):
    __slots__ = (
        "success",
        "batch_id",
        "processed_count",
        "failed_count",
        "errors",
        "total_processing_time_ms",
    )
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    BATCH_ID_FIELD_NUMBER: _ClassVar[int]
    PROCESSED_COUNT_FIELD_NUMBER: _ClassVar[int]
    FAILED_COUNT_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    batch_id: str
    processed_count: int
    failed_count: int
    errors: _containers.RepeatedCompositeFieldContainer[
        _processed_frame_pb2.ProcessingError
    ]
    total_processing_time_ms: float
    def __init__(
        self,
        success: bool = ...,
        batch_id: str | None = ...,
        processed_count: int | None = ...,
        failed_count: int | None = ...,
        errors: _Iterable[_processed_frame_pb2.ProcessingError | _Mapping] | None = ...,
        total_processing_time_ms: float | None = ...,
    ) -> None: ...

class QueueMetricsRequest(_message.Message):
    __slots__ = ("queue_name",)
    QUEUE_NAME_FIELD_NUMBER: _ClassVar[int]
    queue_name: str
    def __init__(self, queue_name: str | None = ...) -> None: ...

class PurgeQueueRequest(_message.Message):
    __slots__ = ("queue_name", "force")
    QUEUE_NAME_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    queue_name: str
    force: bool
    def __init__(self, queue_name: str | None = ..., force: bool = ...) -> None: ...

class PurgeQueueResponse(_message.Message):
    __slots__ = ("success", "purged_count", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    PURGED_COUNT_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    purged_count: int
    message: str
    def __init__(
        self,
        success: bool = ...,
        purged_count: int | None = ...,
        message: str | None = ...,
    ) -> None: ...

class StreamRegistrationResponse(_message.Message):
    __slots__ = ("success", "camera_id", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    camera_id: str
    message: str
    def __init__(
        self,
        success: bool = ...,
        camera_id: str | None = ...,
        message: str | None = ...,
    ) -> None: ...

class StreamUpdateResponse(_message.Message):
    __slots__ = ("success", "camera_id", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    camera_id: str
    message: str
    def __init__(
        self,
        success: bool = ...,
        camera_id: str | None = ...,
        message: str | None = ...,
    ) -> None: ...

class StreamStatusRequest(_message.Message):
    __slots__ = ("camera_id",)
    CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
    camera_id: str
    def __init__(self, camera_id: str | None = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ("service_name",)
    SERVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    service_name: str
    def __init__(self, service_name: str | None = ...) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status", "message", "response_time_ms")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNKNOWN: _ClassVar[HealthCheckResponse.Status]
        SERVING: _ClassVar[HealthCheckResponse.Status]
        NOT_SERVING: _ClassVar[HealthCheckResponse.Status]
        SERVICE_UNKNOWN: _ClassVar[HealthCheckResponse.Status]

    UNKNOWN: HealthCheckResponse.Status
    SERVING: HealthCheckResponse.Status
    NOT_SERVING: HealthCheckResponse.Status
    SERVICE_UNKNOWN: HealthCheckResponse.Status
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    status: HealthCheckResponse.Status
    message: str
    response_time_ms: float
    def __init__(
        self,
        status: HealthCheckResponse.Status | str | None = ...,
        message: str | None = ...,
        response_time_ms: float | None = ...,
    ) -> None: ...

class SystemMetricsRequest(_message.Message):
    __slots__ = ("include_queue_metrics", "include_performance_metrics")
    INCLUDE_QUEUE_METRICS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_PERFORMANCE_METRICS_FIELD_NUMBER: _ClassVar[int]
    include_queue_metrics: bool
    include_performance_metrics: bool
    def __init__(
        self, include_queue_metrics: bool = ..., include_performance_metrics: bool = ...
    ) -> None: ...

class SystemMetricsResponse(_message.Message):
    __slots__ = ("queue_metrics", "performance_metrics", "timestamp")
    QUEUE_METRICS_FIELD_NUMBER: _ClassVar[int]
    PERFORMANCE_METRICS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    queue_metrics: _containers.RepeatedCompositeFieldContainer[
        _processed_frame_pb2.QueueMetrics
    ]
    performance_metrics: PerformanceMetrics
    timestamp: float
    def __init__(
        self,
        queue_metrics: _Iterable[_processed_frame_pb2.QueueMetrics | _Mapping]
        | None = ...,
        performance_metrics: PerformanceMetrics | _Mapping | None = ...,
        timestamp: float | None = ...,
    ) -> None: ...

class PerformanceMetrics(_message.Message):
    __slots__ = (
        "frames_processed",
        "frames_rejected",
        "avg_processing_time_ms",
        "throughput_fps",
        "error_count",
        "memory_usage_mb",
        "cpu_usage_percent",
        "active_connections",
    )
    FRAMES_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    FRAMES_REJECTED_FIELD_NUMBER: _ClassVar[int]
    AVG_PROCESSING_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    THROUGHPUT_FPS_FIELD_NUMBER: _ClassVar[int]
    ERROR_COUNT_FIELD_NUMBER: _ClassVar[int]
    MEMORY_USAGE_MB_FIELD_NUMBER: _ClassVar[int]
    CPU_USAGE_PERCENT_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_CONNECTIONS_FIELD_NUMBER: _ClassVar[int]
    frames_processed: int
    frames_rejected: int
    avg_processing_time_ms: float
    throughput_fps: float
    error_count: int
    memory_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    def __init__(
        self,
        frames_processed: int | None = ...,
        frames_rejected: int | None = ...,
        avg_processing_time_ms: float | None = ...,
        throughput_fps: float | None = ...,
        error_count: int | None = ...,
        memory_usage_mb: float | None = ...,
        cpu_usage_percent: float | None = ...,
        active_connections: int | None = ...,
    ) -> None: ...

class StreamingService(_service.service): ...
class StreamingService_Stub(StreamingService): ...
