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
        if (
            instance.end_time
            and instance.start_time
            and instance.end_time <= instance.start_time
        ):
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


class ModelUploadRequest(BaseModel):
    """Enhanced model upload request schema with comprehensive metadata."""

    # Core model metadata
    name: str = Field(description="Model name", max_length=100, min_length=1)
    version: str = Field(description="Model version (semantic versioning)")
    model_type: ModelType = Field(description="Type of model")
    framework: ModelFramework = Field(description="ML framework used")
    description: str | None = Field(
        None, description="Model description", max_length=1000
    )

    # Model configuration
    classes: list[str] = Field(description="Supported object classes", min_length=1)
    input_shape: list[int] = Field(description="Expected input shape", min_length=1)
    output_shape: list[int] | None = Field(None, description="Expected output shape")
    config: dict[str, Any] = Field(description="Model configuration parameters")

    # Deployment preferences
    deployment_config: dict[str, Any] | None = Field(
        None, description="Deployment configuration preferences"
    )
    auto_deploy: bool = Field(
        False, description="Automatically deploy to development stage after upload"
    )
    target_stage: DeploymentStage = Field(
        DeploymentStage.DEVELOPMENT, description="Initial deployment stage"
    )

    # Metadata and classification
    tags: list[str] = Field(
        default_factory=list, description="Model tags for categorization"
    )
    license: str | None = Field(
        None, description="Model license (e.g., MIT, Apache-2.0)"
    )
    author: str | None = Field(None, description="Model author or organization")
    benchmark_dataset: str | None = Field(
        None, description="Dataset used for training/evaluation"
    )
    training_config: dict[str, Any] | None = Field(
        None, description="Training configuration and hyperparameters"
    )

    # System requirements
    min_python_version: str = Field(
        "3.8", description="Minimum Python version required"
    )
    required_packages: dict[str, str] | None = Field(
        None, description="Required packages with version constraints"
    )
    hardware_requirements: dict[str, Any] | None = Field(
        None, description="Hardware requirements (GPU, memory, etc.)"
    )

    # Validation and quality metadata
    expected_accuracy: float | None = Field(
        None, description="Expected model accuracy", ge=0, le=1
    )
    expected_latency_ms: float | None = Field(
        None, description="Expected inference latency in milliseconds", gt=0
    )
    quality_gates: dict[str, Any] | None = Field(
        None, description="Quality gate requirements for deployment"
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format."""
        import re

        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9\-\.]+)?$"
        if not re.match(pattern, v):
            raise ValueError(
                "Version must follow semantic versioning (e.g., 1.2.3 or 1.2.3-alpha.1)"
            )
        return v

    @field_validator("min_python_version")
    @classmethod
    def validate_python_version(cls, v: str) -> str:
        """Validate Python version format."""
        import re

        pattern = r"^\d+\.\d+(?:\.\d+)?$"
        if not re.match(pattern, v):
            raise ValueError("Python version must be in format X.Y or X.Y.Z")
        return v

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, v: list[str]) -> list[str]:
        """Validate that class names are non-empty and unique."""
        if not v:
            raise ValueError("At least one class must be specified")
        if len(v) != len(set(v)):
            raise ValueError("Class names must be unique")
        for class_name in v:
            if not class_name.strip():
                raise ValueError("Class names cannot be empty or whitespace")
        return v


class ModelUpload(BaseModel):
    """Legacy model upload schema for backward compatibility."""

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


class FileValidationResult(BaseModel):
    """File validation result schema."""

    is_valid: bool = Field(description="Whether the file passed validation")
    file_type: str = Field(description="Detected file type")
    file_size: int = Field(description="File size in bytes", ge=0)
    checksum: str = Field(description="SHA256 checksum of the file")
    mime_type: str | None = Field(None, description="MIME type of the file")
    encoding: str | None = Field(None, description="File encoding if applicable")

    # Validation details
    validation_errors: list[str] = Field(
        default_factory=list, description="List of validation errors"
    )
    validation_warnings: list[str] = Field(
        default_factory=list, description="List of validation warnings"
    )
    security_scan_passed: bool = Field(
        description="Whether the file passed security scanning"
    )
    malicious_patterns: list[str] = Field(
        default_factory=list, description="Detected malicious patterns if any"
    )

    # Framework-specific validation
    framework_metadata: dict[str, Any] | None = Field(
        None, description="Framework-specific metadata extracted from file"
    )
    model_architecture: str | None = Field(
        None, description="Detected model architecture"
    )
    model_size_mb: float | None = Field(
        None, description="Model size in megabytes", ge=0
    )
    estimated_parameters: int | None = Field(
        None, description="Estimated number of model parameters", ge=0
    )


class FileUploadInfo(BaseModel):
    """File upload information schema."""

    original_filename: str = Field(description="Original filename")
    stored_filename: str = Field(description="Filename used for storage")
    file_path: str = Field(description="Full file path in storage")
    file_size: int = Field(description="File size in bytes", ge=0)
    checksum: str = Field(description="SHA256 checksum")
    content_type: str | None = Field(None, description="Content type of uploaded file")
    upload_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(), description="Upload timestamp"
    )
    validation_result: FileValidationResult = Field(
        description="File validation results"
    )


class ModelUploadProgress(BaseModel):
    """Model upload progress tracking schema."""

    upload_id: str = Field(description="Unique upload identifier")
    status: str = Field(description="Current upload status")
    progress_percentage: float = Field(
        description="Upload progress percentage", ge=0, le=100
    )
    current_stage: str = Field(description="Current processing stage")
    stages_completed: list[str] = Field(
        default_factory=list, description="List of completed stages"
    )
    estimated_time_remaining: float | None = Field(
        None, description="Estimated time remaining in seconds"
    )
    error_message: str | None = Field(None, description="Error message if failed")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Upload start time"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Last update time"
    )


class ModelUploadResponse(BaseModel):
    """Comprehensive model upload response schema."""

    # Upload identification
    upload_id: str = Field(description="Unique upload identifier")
    model_id: str = Field(description="Generated model identifier")
    model_name: str = Field(description="Model name")
    version: str = Field(description="Model version")

    # Upload status
    status: str = Field(
        description="Upload status (pending, processing, completed, failed)"
    )
    progress: ModelUploadProgress = Field(description="Upload progress information")

    # File information
    files_uploaded: dict[str, FileUploadInfo] = Field(
        description="Information about uploaded files (model, config, requirements, etc.)"
    )
    total_file_size: int = Field(description="Total size of all uploaded files", ge=0)

    # Validation results
    validation_summary: dict[str, Any] = Field(
        description="Summary of all validation results"
    )
    validation_passed: bool = Field(description="Whether all validations passed")
    quality_checks: dict[str, Any] | None = Field(
        None, description="Results of quality gate checks"
    )

    # Model metadata
    model_metadata: dict[str, Any] = Field(
        description="Extracted model metadata and configuration"
    )
    storage_location: str = Field(description="Storage path of model files")
    artifact_urls: dict[str, str] | None = Field(
        None, description="URLs to access uploaded artifacts"
    )

    # Processing information
    processing_logs: list[str] = Field(
        default_factory=list, description="Processing log messages"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-critical warnings"
    )
    next_steps: list[str] = Field(
        default_factory=list, description="Suggested next steps"
    )

    # Deployment information
    deployment_ready: bool = Field(
        False, description="Whether model is ready for deployment"
    )
    auto_deployment_triggered: bool = Field(
        False, description="Whether auto-deployment was triggered"
    )
    deployment_stage: DeploymentStage | None = Field(
        None, description="Current or target deployment stage"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(), description="Upload initiation time"
    )
    completed_at: datetime | None = Field(None, description="Upload completion time")
    estimated_processing_time: float | None = Field(
        None, description="Estimated total processing time in seconds"
    )


class FileUploadValidation(BaseModel):
    """File upload validation configuration schema."""

    # File type restrictions
    allowed_extensions: list[str] = Field(description="List of allowed file extensions")
    allowed_mime_types: list[str] = Field(description="List of allowed MIME types")

    # Size restrictions
    max_file_size_mb: float = Field(description="Maximum file size in megabytes", gt=0)
    max_total_size_mb: float = Field(
        description="Maximum total size of all files in megabytes", gt=0
    )

    # Content validation
    require_signature_validation: bool = Field(
        True, description="Whether to validate file signatures"
    )
    scan_for_malicious_content: bool = Field(
        True, description="Whether to scan for malicious content"
    )
    validate_model_structure: bool = Field(
        True, description="Whether to validate model file structure"
    )

    # Framework-specific validation
    framework_validators: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Framework-specific validation rules"
    )

    # Quality gates
    quality_gates_enabled: bool = Field(
        False, description="Whether to apply quality gates"
    )
    quality_thresholds: dict[str, float] = Field(
        default_factory=dict, description="Quality thresholds for validation"
    )


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


# Model Management API Response Schemas
class ModelRegistrationResponse(BaseModel):
    """Response schema for model registration."""

    success: bool = Field(description="Whether registration was successful")
    model_id: str = Field(description="Generated unique model identifier")
    model_name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    storage_key: str = Field(description="Storage key in MinIO")
    storage_bucket: str = Field(description="Storage bucket name")
    size_mb: float = Field(description="Model size in megabytes", ge=0)
    metrics: dict[str, float] = Field(description="Performance metrics")
    tags: dict[str, str] = Field(description="Model tags")
    registration_time: datetime = Field(description="Registration timestamp")
    message: str = Field(description="Success/error message")


class ModelVersionResponse(BaseModel):
    """Response schema for model version details."""

    model_id: str = Field(description="Model identifier")
    model_name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    description: str | None = Field(None, description="Model description")
    storage_key: str = Field(description="Storage key in MinIO")
    storage_bucket: str = Field(description="Storage bucket name")
    size_mb: float = Field(description="Model size in megabytes", ge=0)
    created_at: datetime = Field(description="Model creation timestamp")
    metrics: dict[str, float] = Field(description="Performance metrics")
    tags: dict[str, str] = Field(description="Model tags")
    training_config: dict[str, Any] = Field(description="Training configuration")
    status: str = Field(description="Model status")

    class Config:
        from_attributes = True


class ModelPromotionRequest(BaseModel):
    """Request schema for model promotion."""

    target_stage: str = Field(
        description="Target deployment stage (development, staging, canary, production)"
    )
    force_promotion: bool = Field(
        False, description="Force promotion without validation"
    )
    min_accuracy: float | None = Field(
        None, ge=0, le=1, description="Minimum accuracy requirement"
    )
    max_latency_ms: float | None = Field(
        None, gt=0, description="Maximum latency requirement"
    )
    notes: str | None = Field(None, description="Promotion notes")


class ModelPromotionResponse(BaseModel):
    """Response schema for model promotion."""

    success: bool = Field(description="Whether promotion was successful")
    model_id: str = Field(description="Model identifier")
    model_name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    previous_stage: str = Field(description="Previous deployment stage")
    new_stage: str = Field(description="New deployment stage")
    promoted_by: str = Field(description="User who promoted the model")
    promoted_at: datetime = Field(description="Promotion timestamp")
    notes: str | None = Field(None, description="Promotion notes")
    message: str = Field(description="Success/error message")


class ModelListResponse(BaseModel):
    """Response schema for model listing."""

    models: list[dict[str, Any]] = Field(description="List of model information")
    total_count: int = Field(description="Total number of models found", ge=0)
    page_size: int = Field(description="Page size used for query", gt=0)
    has_more: bool = Field(description="Whether more results are available")
    filters: dict[str, Any] = Field(description="Applied filters")


class ModelDownloadResponse(BaseModel):
    """Response schema for model download."""

    success: bool = Field(description="Whether download preparation was successful")
    download_url: str = Field(description="Presigned download URL")
    expires_at: datetime = Field(description="URL expiration timestamp")
    model_id: str = Field(description="Model identifier")
    format: str = Field(description="Download format (binary, url)")
    size_mb: float | None = Field(None, ge=0, description="Model size in megabytes")
    message: str = Field(description="Success/error message")


class ModelDeleteResponse(BaseModel):
    """Response schema for model deletion."""

    success: bool = Field(description="Whether deletion was successful")
    model_id: str = Field(description="Model identifier")
    model_name: str = Field(description="Model name")
    version: str = Field(description="Model version")
    deleted_by: str = Field(description="User who deleted the model")
    deleted_at: datetime = Field(description="Deletion timestamp")
    message: str = Field(description="Success/error message")
