"""ML model management endpoints.

Provides model registry operations, deployment management,
A/B testing, and model optimization functionality.
"""

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.logging import get_logger
from ...models.user import User
from ...services.cache import CacheService
from ..dependencies import (
    RateLimiter,
    get_cache_service,
    get_current_user,
    get_db,
    rate_limit_strict,
    require_permissions,
)
from ..schemas.common import PaginatedResponse, SuccessResponse
from ..schemas.models import (
    ABTestConfig,
    DeploymentStage,
    ModelDeployment,
    ModelFramework,
    ModelMetrics,
    ModelOptimization,
    ModelResponse,
    ModelStatus,
    ModelType,
    ModelUpload,
    ModelVersion,
)

logger = get_logger(__name__)
router = APIRouter()

# Rate limiters
deployment_rate_limit = RateLimiter(calls=10, period=3600)  # 10 deployments per hour
optimization_rate_limit = RateLimiter(calls=5, period=3600)  # 5 optimizations per hour
upload_rate_limit = RateLimiter(calls=5, period=3600)  # 5 uploads per hour
enhanced_upload_rate_limit = RateLimiter(calls=3, period=3600)  # 3 enhanced uploads

# Model storage configuration
MODEL_STORAGE_BASE_PATH = Path(os.getenv("MODEL_STORAGE_PATH", "./models"))
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_MODEL_EXTENSIONS = {".pt", ".pth", ".onnx", ".engine", ".pb", ".tflite"}
ALLOWED_CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".txt"}
ALLOWED_REQUIREMENTS_EXTENSIONS = {".txt", ".yaml", ".yml"}

# Security scanning configuration
MALICIOUS_PATTERNS = [
    b"exec(",
    b"eval(",
    b"__import__",
    b"subprocess",
    b"os.system",
    b"pickle.loads",
]

# Simulated model database
models_db: dict[str, dict[str, Any]] = {
    "yolo11n": {
        "id": "yolo11n",
        "name": "YOLO11 Nano",
        "description": "Lightweight YOLO11 model optimized for edge deployment",
        "model_type": ModelType.DETECTION,
        "framework": ModelFramework.PYTORCH,
        "status": ModelStatus.ACTIVE,
        "current_version": "1.0.0",
        "latest_version": "1.2.0",
        "deployment_stage": DeploymentStage.PRODUCTION,
        "config": {
            "input_size": [640, 640],
            "classes": ["car", "truck", "bus", "motorcycle", "bicycle", "person"],
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
        },
        "classes": ["car", "truck", "bus", "motorcycle", "bicycle", "person"],
        "input_shape": [1, 3, 640, 640],
        "output_shape": [1, 25200, 11],
        "requirements": {
            "python": ">=3.8",
            "torch": ">=2.0.0",
            "ultralytics": ">=8.0.0",
            "opencv-python": ">=4.8.0",
        },
        "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        "updated_at": datetime.now(UTC),
        "created_by": "system",
        "tags": ["vehicle-detection", "real-time", "edge-optimized"],
        "is_active": True,
    },
    "yolo11s": {
        "id": "yolo11s",
        "name": "YOLO11 Small",
        "description": "Balanced YOLO11 model for accuracy and speed",
        "model_type": ModelType.DETECTION,
        "framework": ModelFramework.PYTORCH,
        "status": ModelStatus.AVAILABLE,
        "current_version": "1.1.0",
        "latest_version": "1.1.0",
        "deployment_stage": DeploymentStage.STAGING,
        "config": {
            "input_size": [640, 640],
            "classes": ["car", "truck", "bus", "motorcycle", "bicycle", "person"],
            "confidence_threshold": 0.5,
            "iou_threshold": 0.45,
        },
        "classes": ["car", "truck", "bus", "motorcycle", "bicycle", "person"],
        "input_shape": [1, 3, 640, 640],
        "output_shape": [1, 25200, 11],
        "requirements": {
            "python": ">=3.8",
            "torch": ">=2.0.0",
            "ultralytics": ">=8.0.0",
            "opencv-python": ">=4.8.0",
        },
        "created_at": datetime(2024, 1, 15, tzinfo=UTC),
        "updated_at": datetime.now(UTC),
        "created_by": "system",
        "tags": ["vehicle-detection", "balanced"],
        "is_active": True,
    },
}

deployments_db: dict[str, dict[str, Any]] = {}
ab_tests_db: dict[str, dict[str, Any]] = {}
optimizations_db: dict[str, dict[str, Any]] = {}


def validate_file_extension(filename: str, allowed_extensions: set[str]) -> bool:
    """Validate file extension against allowed extensions.

    Args:
        filename: Name of the file
        allowed_extensions: Set of allowed file extensions

    Returns:
        bool: True if extension is allowed
    """
    return Path(filename).suffix.lower() in allowed_extensions


def calculate_file_checksum(content: bytes) -> str:
    """Calculate SHA256 checksum of file content.

    Args:
        content: File content as bytes

    Returns:
        str: SHA256 checksum as hex string
    """
    return hashlib.sha256(content).hexdigest()


async def scan_for_malicious_content(content: bytes) -> list[str]:
    """Scan file content for potentially malicious patterns.

    Args:
        content: File content as bytes

    Returns:
        list[str]: List of detected malicious patterns
    """
    detected_patterns = []
    content_lower = content.lower()

    for pattern in MALICIOUS_PATTERNS:
        if pattern in content_lower:
            detected_patterns.append(pattern.decode("utf-8", errors="ignore"))

    return detected_patterns


async def validate_model_structure(
    file_path: Path, framework: ModelFramework
) -> dict[str, Any]:
    """Validate model file structure based on framework.

    Args:
        file_path: Path to the model file
        framework: ML framework of the model

    Returns:
        dict[str, Any]: Validation results with metadata

    Raises:
        ValueError: If model structure is invalid
    """
    try:
        validation_result = {
            "is_valid": False,
            "model_size": file_path.stat().st_size,
            "framework": framework,
            "metadata": {},
            "warnings": [],
        }

        if framework == ModelFramework.PYTORCH:
            # Basic PyTorch model validation
            try:
                import torch

                # Try to load model with weights_only=True for security
                model_data = torch.load(
                    file_path, map_location="cpu", weights_only=True
                )  # noqa: S301
                validation_result["is_valid"] = True
                validation_result["metadata"]["torch_version"] = torch.__version__

                # Extract model info if available
                if isinstance(model_data, dict):
                    if "model" in model_data:
                        validation_result["metadata"]["has_model_state"] = True
                    if "optimizer" in model_data:
                        validation_result["metadata"]["has_optimizer_state"] = True
                        validation_result["warnings"].append(
                            "Model contains optimizer state"
                        )

            except Exception as e:
                validation_result["warnings"].append(
                    f"PyTorch validation warning: {str(e)}"
                )
                validation_result["is_valid"] = True  # Allow with warnings

        elif framework == ModelFramework.ONNX:
            try:
                import onnx

                model = onnx.load(str(file_path))
                onnx.checker.check_model(model)
                validation_result["is_valid"] = True
                validation_result["metadata"]["onnx_version"] = getattr(
                    onnx, '__version__', 'unknown'
                )
                validation_result["metadata"]["ir_version"] = model.ir_version
                validation_result["metadata"]["opset_version"] = (
                    model.opset_import[0].version if model.opset_import else None
                )

            except Exception as e:
                raise ValueError(f"Invalid ONNX model: {str(e)}") from e

        elif framework == ModelFramework.TENSORRT:
            # Basic TensorRT engine validation
            if file_path.suffix.lower() == ".engine":
                validation_result["is_valid"] = True
                validation_result["metadata"]["engine_file"] = True
            else:
                validation_result["warnings"].append(
                    "TensorRT file should have .engine extension"
                )
                validation_result["is_valid"] = True

        else:
            # For other frameworks, basic file existence check
            validation_result["is_valid"] = True
            validation_result["warnings"].append(
                f"Limited validation for {framework} framework"
            )

        return validation_result

    except Exception as e:
        raise ValueError(f"Model validation failed: {str(e)}") from e


async def create_model_storage_directory(model_id: str, version: str) -> Path:
    """Create storage directory for model version.

    Args:
        model_id: Model identifier
        version: Model version

    Returns:
        Path: Created directory path
    """
    storage_dir = MODEL_STORAGE_BASE_PATH / model_id / version
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


async def save_uploaded_file(
    upload_file: UploadFile, destination: Path
) -> dict[str, Any]:
    """Save uploaded file to destination with validation.

    Args:
        upload_file: FastAPI UploadFile object
        destination: Destination file path

    Returns:
        dict[str, Any]: File save results with metadata

    Raises:
        HTTPException: If file save fails or validation fails
    """
    try:
        # Read file content
        content = await upload_file.read()

        # Validate file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)}MB",
            )

        # Scan for malicious content
        malicious_patterns = await scan_for_malicious_content(content)
        if malicious_patterns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Potentially malicious content detected: {', '.join(malicious_patterns)}",
            ) from None

        # Calculate checksum
        checksum = calculate_file_checksum(content)

        # Save file
        with open(destination, "wb") as f:
            f.write(content)

        return {
            "file_path": str(destination),
            "file_size": len(content),
            "checksum": checksum,
            "original_filename": upload_file.filename,
            "content_type": upload_file.content_type,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        ) from e


def generate_mock_metrics(model_type: ModelType) -> ModelMetrics:
    """Generate mock performance metrics for a model.

    Args:
        model_type: Type of model

    Returns:
        ModelMetrics: Mock performance metrics
    """
    import random

    if model_type == ModelType.DETECTION:
        return ModelMetrics(
            accuracy=None,
            precision=random.uniform(0.85, 0.95),
            recall=random.uniform(0.80, 0.90),
            f1_score=random.uniform(0.82, 0.92),
            map_score=random.uniform(0.75, 0.85),
            inference_time=random.uniform(20, 100),
            throughput=random.uniform(25, 60),
            memory_usage=random.randint(512, 2048),
            gpu_utilization=random.uniform(60, 90),
            batch_size=random.choice([1, 2, 4, 8]),
            input_resolution="640x640",
            custom_metrics={"detection_rate": random.uniform(0.9, 0.99)},
            benchmark_dataset="COCO-Traffic",
            evaluation_date=datetime.now(UTC),
        )
    else:
        return ModelMetrics(
            accuracy=random.uniform(0.90, 0.98),
            precision=random.uniform(0.88, 0.96),
            recall=random.uniform(0.85, 0.93),
            f1_score=random.uniform(0.86, 0.94),
            map_score=None,
            inference_time=random.uniform(10, 50),
            throughput=random.uniform(50, 120),
            memory_usage=random.randint(256, 1024),
            gpu_utilization=random.uniform(40, 70),
            batch_size=random.choice([4, 8, 16, 32]),
            input_resolution="224x224",
            custom_metrics={"classification_rate": random.uniform(0.9, 0.99)},
            benchmark_dataset="Custom-Dataset",
            evaluation_date=datetime.now(UTC),
        )


def generate_mock_versions(model_id: str, current_version: str) -> list[ModelVersion]:
    """Generate mock version history for a model.

    Args:
        model_id: Model identifier
        current_version: Current version string

    Returns:
        list[ModelVersion]: Mock version history
    """
    import random

    versions = []
    base_time = datetime.now(UTC) - timedelta(days=90)

    # Generate a few versions leading up to current
    version_nums = ["1.0.0", "1.1.0", "1.2.0"]
    if current_version not in version_nums:
        version_nums.append(current_version)

    for i, version in enumerate(version_nums):
        versions.append(
            ModelVersion(
                version=version,
                name=f"{model_id} v{version}",
                description=f"Model version {version} with improvements",
                changelog="Performance optimizations and bug fixes",
                created_at=base_time + timedelta(days=i * 30),
                created_by="system",
                model_path=f"/models/{model_id}/{version}/model.pt",
                config_path=f"/models/{model_id}/{version}/config.yaml",
                checksum=f"sha256:{random.getrandbits(256):064x}",
                file_size=random.randint(
                    10 * 1024 * 1024, 100 * 1024 * 1024
                ),  # 10MB to 100MB
                tags=["release"] if version == current_version else ["archived"],
            )
        )

    return versions


@router.post(
    "/upload",
    response_model=SuccessResponse,
    summary="Upload model",
    description="Upload a new ML model with files and metadata.",
)
async def upload_model(
    # Dependencies (must come first to avoid default parameter issues)
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permissions("models:upload")),
    cache: CacheService = Depends(get_cache_service),
    _rate_limit: None = Depends(upload_rate_limit),
    # Form data for model metadata
    name: str = Form(description="Model name"),
    version: str = Form(description="Model version (semantic versioning)"),
    model_type: ModelType = Form(description="Type of model"),
    framework: ModelFramework = Form(description="ML framework"),
    classes: str = Form(description="JSON array of supported object classes"),
    input_shape: str = Form(description="JSON array of input shape dimensions"),
    config: str = Form(description="JSON object of model configuration"),
    # File uploads
    model_file: UploadFile = File(description="Main model file (.pt, .onnx, etc.)"),
    # Optional form fields (with defaults)
    description: str = Form(None, description="Model description"),
    tags: str = Form(default="[]", description="JSON array of model tags"),
    benchmark_dataset: str = Form(
        None, description="Dataset used for training/evaluation"
    ),
    training_config: str = Form(
        default="{}", description="JSON object of training configuration"
    ),
    # Optional file uploads
    config_file: UploadFile = File(
        None, description="Configuration file (.yaml, .json)"
    ),
    requirements_file: UploadFile = File(
        None, description="Requirements file (.txt, .yaml)"
    ),
    # Database dependency (unused but kept for consistency)
    _db: AsyncSession = Depends(get_db),
) -> SuccessResponse:
    """Upload a new model with files and metadata.

    Args:
        name: Model name
        version: Model version
        model_type: Type of model
        framework: ML framework
        description: Model description
        classes: JSON string of supported classes
        input_shape: JSON string of input shape
        config: JSON string of model configuration
        tags: JSON string of model tags
        benchmark_dataset: Dataset used for evaluation
        training_config: JSON string of training configuration
        model_file: Main model file
        config_file: Optional configuration file
        requirements_file: Optional requirements file
        background_tasks: Background task manager
        current_user: Current user with permissions
        db: Database session
        cache: Cache service
        _rate_limit: Rate limiting dependency

    Returns:
        SuccessResponse: Upload confirmation with model details

    Raises:
        HTTPException: If upload fails or validation errors occur
    """
    try:
        # Validate and parse JSON fields
        try:
            classes_list = json.loads(classes)
            input_shape_list = json.loads(input_shape)
            config_dict = json.loads(config)
            tags_list = json.loads(tags)
            training_config_dict = json.loads(training_config)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON in form field: {str(e)}",
            ) from e

        # Create ModelUpload object for validation
        model_upload = ModelUpload(
            name=name,
            version=version,
            model_type=model_type,
            framework=framework,
            description=description,
            classes=classes_list,
            input_shape=input_shape_list,
            config=config_dict,
            tags=tags_list,
            benchmark_dataset=benchmark_dataset,
            training_config=training_config_dict,
        )

        # Generate model ID
        model_id = f"{name.lower().replace(' ', '-')}-{version}"

        # Check if model already exists
        if model_id in models_db:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Model {model_id} already exists",
            )

        # Validate model file
        if not model_file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Model file is required",
            )

        if not validate_file_extension(model_file.filename, ALLOWED_MODEL_EXTENSIONS):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model file extension. Allowed: {', '.join(ALLOWED_MODEL_EXTENSIONS)}",
            )

        # Validate optional config file
        if config_file and config_file.filename and not validate_file_extension(
            config_file.filename, ALLOWED_CONFIG_EXTENSIONS
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid config file extension. Allowed: {', '.join(ALLOWED_CONFIG_EXTENSIONS)}",
            )

        # Validate optional requirements file
        if requirements_file and requirements_file.filename and not validate_file_extension(
            requirements_file.filename, ALLOWED_REQUIREMENTS_EXTENSIONS
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid requirements file extension. Allowed: {', '.join(ALLOWED_REQUIREMENTS_EXTENSIONS)}",
            )

        # Create storage directory
        storage_dir = await create_model_storage_directory(model_id, version)

        # Save model file
        model_filename = f"model{Path(model_file.filename).suffix}"
        model_path = storage_dir / model_filename
        model_file_info = await save_uploaded_file(model_file, model_path)

        # Validate model structure
        try:
            validation_result = await validate_model_structure(model_path, framework)
            if not validation_result["is_valid"]:
                # Clean up uploaded file
                model_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Model file validation failed",
                ) from None
        except ValueError as e:
            # Clean up uploaded file
            model_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e

        # Save optional files
        config_path = None
        requirements_path = None

        if config_file and config_file.filename:
            config_filename = f"config{Path(config_file.filename).suffix}"
            config_path = storage_dir / config_filename
            await save_uploaded_file(config_file, config_path)

        if requirements_file and requirements_file.filename:
            requirements_filename = (
                f"requirements{Path(requirements_file.filename).suffix}"
            )
            requirements_path = storage_dir / requirements_filename
            await save_uploaded_file(requirements_file, requirements_path)

        # Model version information will be stored in the model record

        # Determine output shape based on model type and input shape
        output_shape = input_shape_list.copy()
        if model_type == ModelType.DETECTION:
            # YOLO-style output: [batch, detections, classes+5]
            num_classes = len(classes_list)
            output_shape = [input_shape_list[0], 25200, num_classes + 5]
        elif model_type == ModelType.CLASSIFICATION:
            # Classification output: [batch, num_classes]
            output_shape = [input_shape_list[0], len(classes_list)]

        # Create model record
        model_record = {
            "id": model_id,
            "name": name,
            "description": description,
            "model_type": model_type,
            "framework": framework,
            "status": ModelStatus.AVAILABLE,
            "current_version": version,
            "latest_version": version,
            "deployment_stage": DeploymentStage.DEVELOPMENT,
            "config": config_dict,
            "classes": classes_list,
            "input_shape": input_shape_list,
            "output_shape": output_shape,
            "requirements": {"python": ">=3.8"},  # Default requirements
            "created_at": datetime.now(UTC),
            "updated_at": datetime.now(UTC),
            "created_by": current_user.id,
            "tags": tags_list + ["uploaded"],
            "is_active": True,
            "validation_result": validation_result,
            "storage_path": str(storage_dir),
        }

        # Store in database
        models_db[model_id] = model_record

        # Schedule background benchmarking
        background_tasks.add_task(
            benchmark_uploaded_model,
            model_id,
            model_upload.model_type,
            current_user.id,
        )

        # Invalidate cache
        await cache.delete("models:list:*")

        logger.info(
            "Model uploaded successfully",
            model_id=model_id,
            name=name,
            version=version,
            framework=framework,
            file_size=model_file_info["file_size"],
            user_id=current_user.id,
        )

        return SuccessResponse(
            success=True,
            message="Model uploaded successfully",
            data={
                "model_id": model_id,
                "name": name,
                "version": version,
                "framework": framework,
                "file_size": model_file_info["file_size"],
                "checksum": model_file_info["checksum"],
                "validation_warnings": validation_result.get("warnings", []),
                "storage_path": str(storage_dir),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Model upload failed",
            name=name,
            version=version,
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model upload failed",
        ) from e


async def benchmark_uploaded_model(
    model_id: str, model_type: ModelType, user_id: str
) -> None:
    """Background task to benchmark newly uploaded model.

    Args:
        model_id: Model identifier
        model_type: Type of model
        user_id: User who uploaded the model
    """
    try:
        import asyncio

        # Simulate benchmarking process
        await asyncio.sleep(30)  # Simulate benchmark time

        # Generate benchmark metrics
        metrics = generate_mock_metrics(model_type)

        # Update model record with metrics
        model_record = models_db.get(model_id)
        if model_record:
            model_record["metrics"] = metrics.model_dump()
            model_record["status"] = ModelStatus.LOADED
            model_record["updated_at"] = datetime.now(UTC)

        logger.info(
            "Model benchmarking completed",
            model_id=model_id,
            inference_time=metrics.inference_time,
            throughput=metrics.throughput,
            user_id=user_id,
        )

    except Exception as e:
        logger.error(
            "Model benchmarking failed",
            model_id=model_id,
            error=str(e),
            user_id=user_id,
        )

        # Update model status to indicate benchmarking failed
        model_record = models_db.get(model_id)
        if model_record:
            model_record["status"] = ModelStatus.ERROR
            model_record["updated_at"] = datetime.now(UTC)


@router.get(
    "/",
    response_model=PaginatedResponse[ModelResponse],
    summary="List models",
    description="Retrieve a paginated list of available ML models.",
)
async def list_models(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    model_type: ModelType | None = Query(None, description="Filter by model type"),
    framework: ModelFramework | None = Query(None, description="Filter by framework"),
    status: ModelStatus | None = Query(None, description="Filter by status"),
    stage: DeploymentStage | None = Query(
        None, description="Filter by deployment stage"
    ),
    search: str | None = Query(None, description="Search in name and description"),
    current_user: User = Depends(get_current_user),
    _db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
) -> PaginatedResponse[ModelResponse]:
    """List available models with filtering.

    Args:
        page: Page number
        size: Items per page
        model_type: Filter by model type
        framework: Filter by framework
        status: Filter by status
        stage: Filter by deployment stage
        search: Search query
        current_user: Current user
        db: Database session
        cache: Cache service

    Returns:
        PaginatedResponse[ModelResponse]: Paginated model list
    """
    try:
        # Build cache key
        cache_key = f"models:list:{page}:{size}:{model_type}:{framework}:{status}:{stage}:{search}"

        # Check cache
        cached_result = await cache.get_json(cache_key)
        if cached_result:
            return PaginatedResponse[ModelResponse](**cached_result)

        # Filter models
        filtered_models = list(models_db.values())

        if model_type:
            filtered_models = [
                m for m in filtered_models if m.get("model_type") == model_type
            ]
        if framework:
            filtered_models = [
                m for m in filtered_models if m.get("framework") == framework
            ]
        if status:
            filtered_models = [m for m in filtered_models if m.get("status") == status]
        if stage:
            filtered_models = [
                m for m in filtered_models if m.get("deployment_stage") == stage
            ]
        if search:
            search_lower = search.lower()
            filtered_models = [
                m
                for m in filtered_models
                if search_lower in m.get("name", "").lower()
                or search_lower in m.get("description", "").lower()
            ]

        # Pagination
        total = len(filtered_models)
        offset = (page - 1) * size
        paginated_models = filtered_models[offset : offset + size]

        # Convert to response models
        model_responses = []
        for model in paginated_models:
            # Generate mock data
            metrics = generate_mock_metrics(model["model_type"])
            versions = generate_mock_versions(model["id"], model["current_version"])

            model_responses.append(
                ModelResponse(
                    id=model["id"],
                    name=model["name"],
                    description=model["description"],
                    model_type=model["model_type"],
                    framework=model["framework"],
                    status=model["status"],
                    current_version=model["current_version"],
                    latest_version=model["latest_version"],
                    versions=versions,
                    deployment_stage=model["deployment_stage"],
                    metrics=metrics,
                    config=model["config"],
                    classes=model["classes"],
                    input_shape=model["input_shape"],
                    output_shape=model["output_shape"],
                    requirements=model["requirements"],
                    created_at=model["created_at"],
                    updated_at=model["updated_at"],
                    created_by=model["created_by"],
                    tags=model["tags"],
                    is_active=model["is_active"],
                )
            )

        result = PaginatedResponse.create(
            items=model_responses,
            total=total,
            page=page,
            size=size,
        )

        # Cache for 2 minutes
        await cache.set_json(cache_key, result.model_dump(), ttl=120)

        logger.info(
            "Models listed",
            total=total,
            page=page,
            size=size,
            user_id=current_user.id,
        )

        return result

    except Exception as e:
        logger.error(
            "Failed to list models",
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve models",
        ) from e


@router.get(
    "/{model_id}",
    response_model=ModelResponse,
    summary="Get model details",
    description="Retrieve detailed information about a specific model.",
)
async def get_model(
    model_id: str,
    current_user: User = Depends(get_current_user),
    _db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
) -> ModelResponse:
    """Get model by ID.

    Args:
        model_id: Model identifier
        current_user: Current user
        db: Database session
        cache: Cache service

    Returns:
        ModelResponse: Model details

    Raises:
        HTTPException: If model not found
    """
    # Check cache first
    cache_key = f"model:{model_id}"
    cached_model = await cache.get_json(cache_key)
    if cached_model:
        return ModelResponse(**cached_model)

    # Get from database
    model = models_db.get(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Generate detailed response
    metrics = generate_mock_metrics(model["model_type"])
    versions = generate_mock_versions(model["id"], model["current_version"])

    model_response = ModelResponse(
        id=model["id"],
        name=model["name"],
        description=model["description"],
        model_type=model["model_type"],
        framework=model["framework"],
        status=model["status"],
        current_version=model["current_version"],
        latest_version=model["latest_version"],
        versions=versions,
        deployment_stage=model["deployment_stage"],
        metrics=metrics,
        config=model["config"],
        classes=model["classes"],
        input_shape=model["input_shape"],
        output_shape=model["output_shape"],
        requirements=model["requirements"],
        created_at=model["created_at"],
        updated_at=model["updated_at"],
        created_by=model["created_by"],
        tags=model["tags"],
        is_active=model["is_active"],
    )

    # Cache for 1 minute
    await cache.set_json(cache_key, model_response.model_dump(), ttl=60)

    logger.debug(
        "Model retrieved",
        model_id=model_id,
        user_id=current_user.id,
    )

    return model_response


@router.delete(
    "/{model_id}",
    response_model=SuccessResponse,
    summary="Delete model",
    description="Delete a model and all its associated files.",
)
async def delete_model(
    model_id: str,
    current_user: User = Depends(require_permissions("models:delete")),
    _db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
) -> SuccessResponse:
    """Delete a model and its files.

    Args:
        model_id: Model identifier
        current_user: Current user with permissions
        db: Database session
        cache: Cache service

    Returns:
        SuccessResponse: Deletion confirmation

    Raises:
        HTTPException: If model not found or deletion fails
    """
    model = models_db.get(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    # Check if model is in production
    if model.get("deployment_stage") == DeploymentStage.PRODUCTION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete model in production. Move to staging first.",
        )

    try:
        # Clean up model files
        storage_path = model.get("storage_path")
        if storage_path:
            storage_dir = Path(storage_path)
            if storage_dir.exists():
                import shutil

                shutil.rmtree(storage_dir)

        # Remove from database
        del models_db[model_id]

        # Clean up related records
        deployments_to_remove = []
        for dep_id, deployment in deployments_db.items():
            if deployment.get("model_id") == model_id:
                deployments_to_remove.append(dep_id)

        for dep_id in deployments_to_remove:
            del deployments_db[dep_id]

        # Clean up A/B tests
        ab_tests_to_remove = []
        for test_id, test in ab_tests_db.items():
            if test.get("model_a_id") == model_id or test.get("model_b_id") == model_id:
                ab_tests_to_remove.append(test_id)

        for test_id in ab_tests_to_remove:
            del ab_tests_db[test_id]

        # Clean up optimizations
        optimizations_to_remove = []
        for opt_id, optimization in optimizations_db.items():
            if optimization.get("original_model_id") == model_id:
                optimizations_to_remove.append(opt_id)

        for opt_id in optimizations_to_remove:
            del optimizations_db[opt_id]

        # Invalidate cache
        await cache.delete(f"model:{model_id}")
        await cache.delete("models:list:*")

        logger.info(
            "Model deleted successfully",
            model_id=model_id,
            name=model.get("name"),
            user_id=current_user.id,
        )

        return SuccessResponse(
            success=True,
            message="Model deleted successfully",
            data={"model_id": model_id, "name": model.get("name")},
        )

    except Exception as e:
        logger.error(
            "Model deletion failed",
            model_id=model_id,
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model deletion failed",
        ) from e


@router.post(
    "/{model_id}/deploy",
    response_model=SuccessResponse,
    summary="Deploy model",
    description="Deploy a model version to a specific stage.",
)
async def deploy_model(
    model_id: str,
    deployment: ModelDeployment,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permissions("models:deploy")),
    _db: AsyncSession = Depends(get_db),
    cache: CacheService = Depends(get_cache_service),
    _rate_limit: None = Depends(deployment_rate_limit),
) -> SuccessResponse:
    """Deploy model to a specific stage.

    Args:
        model_id: Model identifier
        deployment: Deployment configuration
        background_tasks: Background task manager
        current_user: Current user with permissions
        db: Database session
        cache: Cache service
        _rate_limit: Rate limiting dependency

    Returns:
        SuccessResponse: Deployment confirmation

    Raises:
        HTTPException: If model not found or deployment fails
    """
    model = models_db.get(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    try:
        # Create deployment record
        deployment_id = str(uuid4())
        deployment_record = {
            "id": deployment_id,
            "model_id": model_id,
            "version": deployment.version,
            "stage": deployment.stage,
            "status": "deploying",
            "config_overrides": deployment.config_overrides,
            "batch_size": deployment.batch_size,
            "gpu_memory_limit": deployment.gpu_memory_limit,
            "scaling_config": deployment.scaling_config,
            "rollback_on_failure": deployment.rollback_on_failure,
            "health_check_timeout": deployment.health_check_timeout,
            "notes": deployment.notes,
            "created_at": datetime.now(UTC),
            "created_by": current_user.id,
        }

        deployments_db[deployment_id] = deployment_record

        # Start deployment in background
        background_tasks.add_task(
            perform_deployment,
            deployment_id,
            model_id,
            deployment,
        )

        # Invalidate cache
        await cache.delete(f"model:{model_id}")
        await cache.delete("models:list:*")

        logger.info(
            "Model deployment started",
            model_id=model_id,
            version=deployment.version,
            stage=deployment.stage,
            deployment_id=deployment_id,
            user_id=current_user.id,
        )

        return SuccessResponse(
            success=True,
            message="Model deployment initiated",
            data={
                "deployment_id": deployment_id,
                "model_id": model_id,
                "version": deployment.version,
                "stage": deployment.stage,
            },
        )

    except Exception as e:
        logger.error(
            "Model deployment failed",
            model_id=model_id,
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model deployment failed",
        ) from e


async def perform_deployment(
    deployment_id: str, model_id: str, deployment: ModelDeployment
) -> None:
    """Background task to perform model deployment.

    Args:
        deployment_id: Deployment identifier
        model_id: Model identifier
        deployment: Deployment configuration
    """
    try:
        import asyncio
        import random

        # Simulate deployment process
        deployment_record = deployments_db.get(deployment_id)
        if not deployment_record:
            return

        # Simulate deployment time
        await asyncio.sleep(random.uniform(10, 30))

        # Update model deployment stage
        model = models_db.get(model_id)
        if model:
            model["deployment_stage"] = deployment.stage
            model["current_version"] = deployment.version
            model["status"] = ModelStatus.ACTIVE
            model["updated_at"] = datetime.now(UTC)

        # Update deployment status
        deployment_record["status"] = "completed"
        deployment_record["completed_at"] = datetime.now(UTC)

        logger.info(
            "Model deployment completed",
            deployment_id=deployment_id,
            model_id=model_id,
        )

    except Exception as e:
        # Update deployment with error
        deployment_record = deployments_db.get(deployment_id)
        if deployment_record:
            deployment_record["status"] = "failed"
            deployment_record["error_message"] = str(e)

        logger.error(
            "Model deployment failed",
            deployment_id=deployment_id,
            model_id=model_id,
            error=str(e),
        )


@router.post(
    "/ab-test",
    response_model=SuccessResponse,
    summary="Create A/B test",
    description="Setup A/B testing between two model versions.",
)
async def create_ab_test(
    ab_config: ABTestConfig,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permissions("models:ab_test")),
    _db: AsyncSession = Depends(get_db),
    _rate_limit: None = Depends(rate_limit_strict),
) -> SuccessResponse:
    """Create A/B test configuration.

    Args:
        ab_config: A/B test configuration
        background_tasks: Background task manager
        current_user: Current user with permissions
        db: Database session
        _rate_limit: Rate limiting dependency

    Returns:
        SuccessResponse: A/B test creation confirmation
    """
    try:
        # Validate models exist
        if ab_config.model_a_id not in models_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model A {ab_config.model_a_id} not found",
            )

        if ab_config.model_b_id not in models_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model B {ab_config.model_b_id} not found",
            )

        # Create A/B test record
        test_id = str(uuid4())
        test_record = {
            "id": test_id,
            "name": ab_config.name,
            "description": ab_config.description,
            "model_a_id": ab_config.model_a_id,
            "model_b_id": ab_config.model_b_id,
            "traffic_split": ab_config.traffic_split,
            "cameras": ab_config.cameras,
            "start_time": ab_config.start_time or datetime.now(UTC),
            "end_time": ab_config.end_time,
            "success_criteria": ab_config.success_criteria,
            "is_active": ab_config.is_active,
            "auto_promote": ab_config.auto_promote,
            "confidence_threshold": ab_config.confidence_threshold,
            "created_at": datetime.now(UTC),
            "created_by": current_user.id,
            "status": "active" if ab_config.is_active else "draft",
        }

        ab_tests_db[test_id] = test_record

        # Start A/B test monitoring if active
        if ab_config.is_active:
            background_tasks.add_task(monitor_ab_test, test_id)

        logger.info(
            "A/B test created",
            test_id=test_id,
            name=ab_config.name,
            model_a=ab_config.model_a_id,
            model_b=ab_config.model_b_id,
            user_id=current_user.id,
        )

        return SuccessResponse(
            success=True,
            message="A/B test created successfully",
            data={
                "test_id": test_id,
                "name": ab_config.name,
                "status": test_record["status"],
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "A/B test creation failed",
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="A/B test creation failed",
        ) from e


async def monitor_ab_test(test_id: str) -> None:
    """Background task to monitor A/B test performance.

    Args:
        test_id: A/B test identifier
    """
    try:
        import asyncio
        import random

        test_record = ab_tests_db.get(test_id)
        if not test_record:
            return

        # Simulate monitoring period
        await asyncio.sleep(60)  # Monitor for 1 minute in demo

        # Generate mock results
        model_a_accuracy = random.uniform(0.85, 0.92)
        model_b_accuracy = random.uniform(0.87, 0.94)

        # Determine statistical significance (simplified)
        improvement = ((model_b_accuracy - model_a_accuracy) / model_a_accuracy) * 100
        is_significant = abs(improvement) > 2.0  # 2% improvement threshold

        # Update test with results
        test_record["results"] = {
            "model_a_metrics": {"accuracy": model_a_accuracy, "samples": 1000},
            "model_b_metrics": {"accuracy": model_b_accuracy, "samples": 1000},
            "statistical_significance": is_significant,
            "confidence_level": 0.95,
            "improvement": improvement,
            "winner": (
                test_record["model_b_id"]
                if model_b_accuracy > model_a_accuracy
                else test_record["model_a_id"]
            ),
        }

        if (
            test_record["auto_promote"]
            and is_significant
            and model_b_accuracy > model_a_accuracy
        ):
            # Auto-promote model B
            model_b = models_db.get(test_record["model_b_id"])
            if model_b:
                model_b["deployment_stage"] = DeploymentStage.PRODUCTION
                test_record["status"] = "completed_promoted"
        else:
            test_record["status"] = "completed"

        logger.info(
            "A/B test monitoring completed",
            test_id=test_id,
            winner=test_record["results"]["winner"],
            improvement=improvement,
        )

    except Exception as e:
        logger.error(
            "A/B test monitoring failed",
            test_id=test_id,
            error=str(e),
        )


@router.get(
    "/ab-tests",
    response_model=PaginatedResponse[dict],
    summary="List A/B tests",
    description="List A/B tests with their current status and results.",
)
async def list_ab_tests(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    status: str | None = Query(None, description="Filter by status"),
    _current_user: User = Depends(get_current_user),
    _db: AsyncSession = Depends(get_db),
) -> PaginatedResponse[dict]:
    """List A/B tests.

    Args:
        page: Page number
        size: Items per page
        status: Filter by status
        current_user: Current user
        db: Database session

    Returns:
        PaginatedResponse[dict]: Paginated A/B test list
    """
    # Filter tests
    filtered_tests = list(ab_tests_db.values())

    if status:
        filtered_tests = [t for t in filtered_tests if t.get("status") == status]

    # Sort by creation time (newest first)
    filtered_tests.sort(
        key=lambda x: x.get("created_at", datetime.min.replace(tzinfo=UTC)),
        reverse=True,
    )

    # Pagination
    total = len(filtered_tests)
    offset = (page - 1) * size
    paginated_tests = filtered_tests[offset : offset + size]

    return PaginatedResponse.create(
        items=paginated_tests,
        total=total,
        page=page,
        size=size,
    )


@router.post(
    "/{model_id}/optimize",
    response_model=SuccessResponse,
    summary="Optimize model",
    description="Optimize a model for specific deployment targets.",
)
async def optimize_model(
    model_id: str,
    optimization: ModelOptimization,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_permissions("models:optimize")),
    _db: AsyncSession = Depends(get_db),
    _rate_limit: None = Depends(optimization_rate_limit),
) -> SuccessResponse:
    """Optimize model for deployment.

    Args:
        model_id: Model identifier
        optimization: Optimization configuration
        background_tasks: Background task manager
        current_user: Current user with permissions
        db: Database session
        _rate_limit: Rate limiting dependency

    Returns:
        SuccessResponse: Optimization confirmation

    Raises:
        HTTPException: If model not found or optimization fails
    """
    model = models_db.get(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    try:
        # Create optimization record
        optimization_id = str(uuid4())
        optimization_record = {
            "id": optimization_id,
            "original_model_id": model_id,
            "optimization_type": optimization.optimization_type,
            "target_platform": optimization.target_platform,
            "precision": optimization.precision,
            "max_batch_size": optimization.max_batch_size,
            "optimization_config": optimization.optimization_config,
            "status": "pending",
            "created_at": datetime.now(UTC),
            "created_by": current_user.id,
        }

        optimizations_db[optimization_id] = optimization_record

        # Start optimization in background
        background_tasks.add_task(
            perform_optimization,
            optimization_id,
            model_id,
            optimization,
        )

        logger.info(
            "Model optimization started",
            model_id=model_id,
            optimization_id=optimization_id,
            type=optimization.optimization_type,
            user_id=current_user.id,
        )

        return SuccessResponse(
            success=True,
            message="Model optimization initiated",
            data={
                "optimization_id": optimization_id,
                "model_id": model_id,
                "type": optimization.optimization_type,
            },
        )

    except Exception as e:
        logger.error(
            "Model optimization failed",
            model_id=model_id,
            error=str(e),
            user_id=current_user.id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model optimization failed",
        ) from e


async def perform_optimization(
    optimization_id: str, model_id: str, optimization: ModelOptimization
) -> None:
    """Background task to perform model optimization.

    Args:
        optimization_id: Optimization identifier
        model_id: Model identifier
        optimization: Optimization configuration
    """
    try:
        import asyncio
        import random

        optimization_record = optimizations_db.get(optimization_id)
        if not optimization_record:
            return

        # Update status to processing
        optimization_record["status"] = "processing"

        # Simulate optimization time
        await asyncio.sleep(random.uniform(30, 120))  # 30s to 2 min

        # Generate optimized model
        optimized_model_id = f"{model_id}_optimized_{optimization.optimization_type}"

        # Create optimized model record
        original_model = models_db[model_id]
        optimized_model = original_model.copy()
        optimized_model["id"] = optimized_model_id
        optimized_model["name"] = (
            f"{original_model['name']} ({optimization.optimization_type.upper()})"
        )
        optimized_model["description"] = (
            f"Optimized version of {original_model['name']} for {optimization.target_platform}"
        )
        optimized_model["tags"] = original_model["tags"] + [
            "optimized",
            optimization.optimization_type,
        ]
        optimized_model["created_at"] = datetime.now(UTC)

        models_db[optimized_model_id] = optimized_model

        # Generate optimization results
        size_reduction = random.uniform(20, 60)  # 20-60% size reduction
        speed_improvement = random.uniform(150, 300)  # 1.5x to 3x speed improvement
        accuracy_loss = random.uniform(0, 5)  # 0-5% accuracy loss

        # Update optimization record
        optimization_record.update(
            {
                "status": "completed",
                "optimized_model_id": optimized_model_id,
                "size_reduction": size_reduction,
                "speed_improvement": speed_improvement,
                "accuracy_loss": accuracy_loss,
                "completed_at": datetime.now(UTC),
            }
        )

        logger.info(
            "Model optimization completed",
            optimization_id=optimization_id,
            optimized_model_id=optimized_model_id,
            size_reduction=size_reduction,
            speed_improvement=speed_improvement,
        )

    except Exception as e:
        # Update optimization with error
        optimization_record = optimizations_db.get(optimization_id)
        if optimization_record:
            optimization_record["status"] = "failed"
            optimization_record["error_message"] = str(e)

        logger.error(
            "Model optimization failed",
            optimization_id=optimization_id,
            model_id=model_id,
            error=str(e),
        )
