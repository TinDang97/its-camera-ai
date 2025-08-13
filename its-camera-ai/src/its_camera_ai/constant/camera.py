from enum import Enum


class CameraStatus(str, Enum):
    """Camera operational status."""

    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    STREAMING = "streaming"
    STOPPED = "stopped"


class CameraType(str, Enum):
    """Camera hardware type."""

    FIXED = "fixed"
    PTZ = "ptz"
    DOME = "dome"
    BULLET = "bullet"
    THERMAL = "thermal"


class StreamProtocol(str, Enum):
    """Video stream protocols."""

    RTSP = "rtsp"
    RTMP = "rtmp"
    HLS = "hls"
    WEBRTC = "webrtc"
    HTTP = "http"


class ProcessingStatus(str, Enum):
    """Frame processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class VehicleType(str, Enum):
    """Detected vehicle types."""

    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    UNKNOWN = "unknown"
