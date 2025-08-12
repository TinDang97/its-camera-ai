"""ML model management schemas."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ModelType(str, Enum):
    """Model type enumeration."""

    DETECTION = "detection"
    CLASSIFICATION = "classification"
    TRACKING = "tracking"
    SEGMENTATION = "segmentation"
    POSE_ESTIMATION = "pose_estimation"
    ANOMALY_DETECTION = "anomaly_detection"


class ModelStatus(str, Enum):
    """Model status enumeration."""

    AVAILABLE = "available"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DEPRECATED = "deprecated"


class DeploymentStage(str, Enum):
    """Model deployment stage enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLBACK = "rollback"


class ModelFramework(str, Enum):
    """ML framework enumeration."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


class ModelVersion(BaseModel):
    """Model version information schema."""

    version: str = Field(description="Semantic version (e.g., 1.2.3)")
    name: str = Field(description="Model name")
    description: str | None = Field(None, description="Version description")
    changelog: str | None = Field(None, description="Changes in this version")
    created_at: datetime = Field(description="Version creation timestamp")
    created_by: str = Field(description="User who created this version")
    model_path: str = Field(description="Model file path")
    config_path: str | None = Field(None, description="Configuration file path")
    checksum: str = Field(description="Model file checksum")
    file_size: int = Field(description="Model file size in bytes")
    tags: list[str] = Field(default_factory=list, description="Version tags")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        import re

        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"
        if not re.match(pattern, v):
            raise ValueError("Version must follow semantic versioning (e.g., 1.2.3)")
        return v


class ModelMetrics(BaseModel):
    """Model performance metrics schema."""

    accuracy: float | None = Field(None, description="Model accuracy", ge=0, le=1)
    precision: float | None = Field(None, description="Model precision", ge=0, le=1)
    recall: float | None = Field(None, description="Model recall", ge=0, le=1)
    f1_score: float | None = Field(None, description="F1 score", ge=0, le=1)
    map_score: float | None = Field(
        None, description="Mean Average Precision", ge=0, le=1
    )
    inference_time: float = Field(description="Average inference time in ms", gt=0)
    throughput: float = Field(description="Throughput in FPS", gt=0)
    memory_usage: int = Field(description="Memory usage in MB", gt=0)
    gpu_utilization: float | None = Field(
        None, description="GPU utilization percentage", ge=0, le=100
    )
    batch_size: int = Field(description="Optimal batch size", gt=0)
    input_resolution: str = Field(description="Input resolution (e.g., 640x640)")
    custom_metrics: dict[str, float] | None = Field(
        None, description="Custom evaluation metrics"
    )
    benchmark_dataset: str | None = Field(
        None, description="Dataset used for benchmarking"
    )
    evaluation_date: datetime = Field(description="Metrics evaluation timestamp")


class ModelResponse(BaseModel):
    """Model information response schema."""

    id: str = Field(description="Model ID")
    name: str = Field(description="Model name")
    description: str | None = Field(None, description="Model description")
    model_type: ModelType = Field(description="Type of model")
    framework: ModelFramework = Field(description="ML framework used")
    status: ModelStatus = Field(description="Current model status")
    current_version: str = Field(description="Currently deployed version")
    latest_version: str = Field(description="Latest available version")
    versions: list[ModelVersion] = Field(description="Available versions")
    deployment_stage: DeploymentStage = Field(description="Current deployment stage")
    metrics: ModelMetrics | None = Field(None, description="Performance metrics")
    config: dict[str, Any] = Field(description="Model configuration")
    classes: list[str] = Field(description="Supported object classes")
    input_shape: list[int] = Field(description="Expected input shape")
    output_shape: list[int] = Field(description="Model output shape")
    requirements: dict[str, str] = Field(description="System requirements")
    created_at: datetime = Field(description="Model creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    created_by: str = Field(description="Model creator")
    tags: list[str] = Field(description="Model tags")
    is_active: bool = Field(description="Whether model is active")

    class Config:
        from_attributes = True


class ModelDeployment(BaseModel):
    """Model deployment request schema."""

    version: str = Field(description="Version to deploy")
    stage: DeploymentStage = Field(description="Target deployment stage")
    config_overrides: dict[str, Any] | None = Field(
        None, description="Configuration overrides"
    )
    batch_size: int | None = Field(None, description="Deployment batch size", gt=0)
    gpu_memory_limit: int | None = Field(
        None, description="GPU memory limit in MB", gt=0
    )
    scaling_config: dict[str, Any] | None = Field(
        None, description="Auto-scaling configuration"
    )
    rollback_on_failure: bool = Field(
        True, description="Auto-rollback on deployment failure"
    )
    health_check_timeout: int = Field(
        description="Health check timeout in seconds", default=300, gt=0
    )
    notes: str | None = Field(None, description="Deployment notes")


class ABTestConfig(BaseModel):
    """A/B testing configuration schema."""

    name: str = Field(description="A/B test name", max_length=100)
    description: str | None = Field(None, description="Test description")
    model_a_id: str = Field(description="Control model ID")
    model_b_id: str = Field(description="Variant model ID")
    traffic_split: float = Field(
        description="Traffic percentage for model B", ge=0, le=100
    )
    cameras: list[str] | None = Field(
        None, description="Specific cameras for test (all if not specified)"
    )
    start_time: datetime | None = Field(None, description="Test start time")
    end_time: datetime | None = Field(None, description="Test end time")
    success_criteria: dict[str, Any] = Field(description="Success criteria definition")
    is_active: bool = Field(True, description="Whether test is active")
    auto_promote: bool = Field(False, description="Auto-promote winner to production")
    confidence_threshold: float = Field(
        description="Statistical confidence threshold", default=0.95, ge=0, le=1
    )

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> "ABTestConfig":
        """Validate A/B test configuration."""
        instance = super().model_validate(obj, **kwargs)
        if instance.end_time and instance.start_time:
            if instance.end_time <= instance.start_time:
                raise ValueError("end_time must be after start_time")
        return instance


class ABTestResult(BaseModel):
    """A/B test results schema."""

    test_id: str = Field(description="A/B test ID")
    model_a_metrics: dict[str, float] = Field(description="Control model performance")
    model_b_metrics: dict[str, float] = Field(description="Variant model performance")
    statistical_significance: bool = Field(
        description="Whether results are statistically significant"
    )
    confidence_level: float = Field(description="Confidence level", ge=0, le=1)
    sample_size: int = Field(description="Total samples evaluated")
    winner: str | None = Field(None, description="Winning model ID")
    improvement: float | None = Field(
        None, description="Performance improvement percentage"
    )
    p_value: float | None = Field(None, description="Statistical p-value")
    evaluation_period: dict[str, datetime] = Field(description="Evaluation time period")
    recommendation: str = Field(description="Deployment recommendation")
    details: dict[str, Any] = Field(description="Detailed test results")


class ModelUpload(BaseModel):
    """Model upload request schema."""

    name: str = Field(description="Model name", max_length=100)
    version: str = Field(description="Model version")
    model_type: ModelType = Field(description="Type of model")
    framework: ModelFramework = Field(description="ML framework")
    description: str | None = Field(None, description="Model description")
    classes: list[str] = Field(description="Supported object classes")
    input_shape: list[int] = Field(description="Expected input shape")
    config: dict[str, Any] = Field(description="Model configuration")
    tags: list[str] = Field(default_factory=list, description="Model tags")
    benchmark_dataset: str | None = Field(
        None, description="Dataset used for training/evaluation"
    )
    training_config: dict[str, Any] | None = Field(
        None, description="Training configuration"
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        import re

        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"
        if not re.match(pattern, v):
            raise ValueError("Version must follow semantic versioning (e.g., 1.2.3)")
        return v


class ModelOptimization(BaseModel):
    """Model optimization request schema."""

    model_id: str = Field(description="Model to optimize")
    optimization_type: str = Field(description="Type of optimization")
    target_platform: str = Field(description="Target deployment platform")
    precision: str = Field(description="Target precision", default="fp16")
    max_batch_size: int = Field(description="Maximum batch size", default=16, gt=0)
    optimization_config: dict[str, Any] | None = Field(
        None, description="Optimization parameters"
    )

    @field_validator("optimization_type")
    @classmethod
    def validate_optimization_type(cls, v: str) -> str:
        """Validate optimization type."""
        valid_types = [
            "tensorrt",
            "onnx",
            "quantization",
            "pruning",
            "distillation",
            "tensorrt_fp16",
            "openvino",
        ]
        if v not in valid_types:
            raise ValueError(
                f"Optimization type must be one of: {', '.join(valid_types)}"
            )
        return v

    @field_validator("target_platform")
    @classmethod
    def validate_target_platform(cls, v: str) -> str:
        """Validate target platform."""
        valid_platforms = [
            "gpu",
            "cpu",
            "jetson",
            "edge_tpu",
            "intel_ncs",
            "arm",
            "x86",
        ]
        if v not in valid_platforms:
            raise ValueError(
                f"Target platform must be one of: {', '.join(valid_platforms)}"
            )
        return v

    @field_validator("precision")
    @classmethod
    def validate_precision(cls, v: str) -> str:
        """Validate precision format."""
        if v not in ["fp32", "fp16", "int8", "int4"]:
            raise ValueError("Precision must be 'fp32', 'fp16', 'int8', or 'int4'")
        return v


class OptimizationResult(BaseModel):
    """Model optimization result schema."""

    original_model_id: str = Field(description="Original model ID")
    optimized_model_id: str = Field(description="Optimized model ID")
    optimization_type: str = Field(description="Optimization type used")
    size_reduction: float = Field(description="Size reduction percentage", ge=0, le=100)
    speed_improvement: float = Field(description="Speed improvement percentage", ge=0)
    accuracy_loss: float = Field(description="Accuracy loss percentage", ge=0, le=100)
    before_metrics: ModelMetrics = Field(description="Original model metrics")
    after_metrics: ModelMetrics = Field(description="Optimized model metrics")
    optimization_time: float = Field(description="Optimization duration in seconds")
    created_at: datetime = Field(description="Optimization completion timestamp")
    success: bool = Field(description="Whether optimization succeeded")
    error_message: str | None = Field(None, description="Error message if failed")

    class Config:
        from_attributes = True


class ModelRegistry(BaseModel):
    """Model registry entry schema."""

    model_id: str = Field(description="Model ID")
    registry_url: str = Field(description="Model registry URL")
    artifact_path: str = Field(description="Model artifact path")
    metadata: dict[str, Any] = Field(description="Registry metadata")
    checksum: str = Field(description="Model checksum")
    download_count: int = Field(description="Download count", ge=0)
    last_accessed: datetime | None = Field(None, description="Last access time")
    is_public: bool = Field(False, description="Whether model is public")
    license: str | None = Field(None, description="Model license")
    documentation_url: str | None = Field(None, description="Documentation URL")

    class Config:
        from_attributes = True
