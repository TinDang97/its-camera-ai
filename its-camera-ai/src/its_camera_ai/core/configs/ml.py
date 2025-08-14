"""Machine learning configuration settings."""

from pathlib import Path

from pydantic import BaseModel, Field


class MLConfig(BaseModel):
    """Machine learning configuration settings."""

    model_config = {"protected_namespaces": ()}

    model_path: Path = Field(default=Path("models"), description="Path to model files")
    batch_size: int = Field(default=32, description="Inference batch size")
    max_batch_size: int = Field(default=128, description="Maximum batch size")
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence threshold for predictions"
    )
    device: str = Field(default="cuda", description="Inference device")
    precision: str = Field(default="fp16", description="Model precision")
    enable_tensorrt: bool = Field(
        default=True, description="Enable TensorRT optimization"
    )


class MLStreamingConfig(BaseModel):
    """ML streaming configuration for AI-annotated video streams."""

    model_config = {"protected_namespaces": ()}

    # Detection Configuration
    confidence_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Object detection confidence threshold"
    )
    nms_threshold: float = Field(
        default=0.4, ge=0.0, le=1.0,
        description="Non-maximum suppression threshold"
    )
    max_detections: int = Field(
        default=100, ge=1, le=1000,
        description="Maximum number of detections per frame"
    )
    classes_to_detect: list[str] = Field(
        default=["car", "truck", "bus", "motorcycle", "bicycle", "person"],
        description="Object classes to detect"
    )

    # Performance Configuration
    target_latency_ms: float = Field(
        default=50.0, ge=10.0, le=1000.0,
        description="Target ML processing latency in milliseconds"
    )
    batch_size: int = Field(
        default=8, ge=1, le=64,
        description="ML inference batch size"
    )
    enable_gpu_acceleration: bool = Field(
        default=True, description="Enable GPU acceleration for ML inference"
    )

    # Annotation Style Configuration
    show_confidence: bool = Field(
        default=True, description="Show confidence scores in annotations"
    )
    show_class_labels: bool = Field(
        default=True, description="Show class labels in annotations"
    )
    box_thickness: int = Field(
        default=2, ge=1, le=10, description="Bounding box line thickness"
    )
    font_scale: float = Field(
        default=0.6, ge=0.1, le=2.0, description="Text font scale"
    )

    # Vehicle-Specific Settings
    vehicle_priority: bool = Field(
        default=True, description="Boost confidence for vehicle detections"
    )
    emergency_vehicle_detection: bool = Field(
        default=True, description="Enable emergency vehicle detection"
    )
    confidence_boost_factor: float = Field(
        default=1.1, ge=1.0, le=2.0, description="Vehicle confidence boost factor"
    )

    # Model Configuration
    model_path: str = Field(
        default="models/yolo11n.pt", description="Path to YOLO11 model file"
    )
    use_tensorrt: bool = Field(
        default=True, description="Use TensorRT optimization"
    )
    use_fp16: bool = Field(
        default=True, description="Use FP16 precision"
    )

    # Streaming Integration
    enable_metadata_streaming: bool = Field(
        default=True, description="Stream detection metadata alongside video"
    )
    metadata_include_performance: bool = Field(
        default=True, description="Include performance metrics in metadata"
    )
    enable_detection_history: bool = Field(
        default=True, description="Maintain detection history for analytics"
    )
    detection_history_limit: int = Field(
        default=1000, ge=100, le=10000, description="Maximum detection history entries"
    )
