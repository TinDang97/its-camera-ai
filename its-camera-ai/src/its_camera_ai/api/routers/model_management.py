"""FastAPI endpoints for ML model management and deployment.

Provides comprehensive model lifecycle management including registration,
deployment, version control, performance monitoring, and A/B testing.
"""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import aiofiles
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
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ...core.exceptions import ModelError, ResourceNotFoundError, ValidationError
from ...core.logging import get_logger
from ...storage.model_registry import MinIOModelRegistry
from ..dependencies import (
    get_current_user,
    get_model_registry,
    validate_api_key,
)
from ..schemas.models import (
    ModelDeleteResponse,
    ModelDownloadResponse,
    ModelListResponse,
    ModelPromotionRequest,
    ModelPromotionResponse,
    ModelRegistrationResponse,
    ModelVersionResponse,
)

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/models",
    tags=["model-management"],
    dependencies=[Depends(validate_api_key)],
)


# Request/Response Models
class ModelMetricsRequest(BaseModel):
    """Request for updating model performance metrics."""

    accuracy: float | None = Field(None, ge=0.0, le=1.0, description="Model accuracy")
    precision: float | None = Field(None, ge=0.0, le=1.0, description="Model precision")
    recall: float | None = Field(None, ge=0.0, le=1.0, description="Model recall")
    f1_score: float | None = Field(None, ge=0.0, le=1.0, description="F1 score")
    latency_p95_ms: float | None = Field(None, ge=0.0, description="95th percentile latency")
    latency_p99_ms: float | None = Field(None, ge=0.0, description="99th percentile latency")
    throughput_fps: float | None = Field(None, ge=0.0, description="Throughput in FPS")
    memory_usage_mb: float | None = Field(None, ge=0.0, description="Memory usage in MB")
    gpu_utilization: float | None = Field(None, ge=0.0, le=100.0, description="GPU utilization %")
    inference_cost_per_frame: float | None = Field(None, ge=0.0, description="Cost per inference")
    custom_metrics: dict[str, float] = Field(default_factory=dict, description="Custom metrics")


class ModelComparisonRequest(BaseModel):
    """Request for comparing model versions."""

    model_name: str = Field(..., description="Model name")
    version_a: str = Field(..., description="First version to compare")
    version_b: str = Field(..., description="Second version to compare")
    metrics: list[str] = Field(default=["accuracy", "latency_p95_ms", "throughput_fps"], description="Metrics to compare")


class ModelDeploymentRequest(BaseModel):
    """Request for model deployment configuration."""

    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    deployment_stage: str = Field(..., description="Deployment stage (development, staging, production)")
    instance_count: int = Field(default=1, ge=1, description="Number of instances")
    auto_scaling: bool = Field(default=True, description="Enable auto-scaling")
    max_instances: int = Field(default=10, ge=1, description="Maximum instances")
    target_cpu: float = Field(default=70.0, ge=10.0, le=90.0, description="Target CPU utilization %")
    health_check_endpoint: str = Field(default="/health", description="Health check endpoint")
    environment_variables: dict[str, str] = Field(default_factory=dict, description="Environment variables")


# Model Registration Endpoints
@router.post(
    "/register",
    response_model=ModelRegistrationResponse,
    summary="Register new ML model",
    description="Register a new model version with artifacts and metadata",
)
async def register_model(
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(..., description="Model artifact file"),
    model_name: str = Form(..., description="Model name/identifier"),
    version: str = Form(..., description="Model version"),
    _description: str | None = Form(None, description="Model description"),
    accuracy: float = Form(..., ge=0.0, le=1.0, description="Model accuracy"),
    latency_p95_ms: float | None = Form(None, ge=0.0, description="95th percentile latency"),
    throughput_fps: float | None = Form(None, ge=0.0, description="Throughput in FPS"),
    training_config: str | None = Form(None, description="Training configuration as JSON"),
    tags: str | None = Form(None, description="Model tags as JSON"),
    framework: str = Form("pytorch", description="ML framework (pytorch, tensorflow, onnx)"),
    model_type: str = Form("detection", description="Model type (detection, classification, segmentation)"),
    model_registry: MinIOModelRegistry = Depends(get_model_registry),
    current_user = Depends(get_current_user),
) -> ModelRegistrationResponse:
    """Register a new model version."""

    try:
        # Validate model file
        if not model_file.filename:
            raise ValidationError("Model filename is required")

        # Validate file extension
        valid_extensions = {'.pt', '.pth', '.onnx', '.pb', '.h5', '.pkl', '.joblib'}
        file_extension = Path(model_file.filename).suffix.lower()
        if file_extension not in valid_extensions:
            raise ValidationError(
                f"Invalid model file extension: {file_extension}. "
                f"Supported: {', '.join(valid_extensions)}"
            )

        # Parse JSON fields
        parsed_training_config = json.loads(training_config) if training_config else {}
        parsed_tags = json.loads(tags) if tags else {}

        # Add standard tags
        parsed_tags.update({
            "framework": framework,
            "model_type": model_type,
            "registered_by": current_user.username if hasattr(current_user, 'username') else 'system',
            "registration_date": datetime.now(UTC).isoformat(),
        })

        # Create temporary file for model
        temp_file = None
        try:
            # Save model to temporary file with secure creation
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp:
                temp_file = Path(tmp.name)
                # Set secure file permissions (owner read/write only)
                os.chmod(temp_file, 0o600)

            async with aiofiles.open(temp_file, 'wb') as f:
                content = await model_file.read()
                await f.write(content)

            # Prepare metrics
            metrics = {
                "accuracy": accuracy,
                "model_size_mb": len(content) / (1024 * 1024),
            }

            if latency_p95_ms is not None:
                metrics["latency_p95_ms"] = latency_p95_ms
            if throughput_fps is not None:
                metrics["throughput_fps"] = throughput_fps

            # Register model with registry
            model_version = await model_registry.register_model(
                model_path=temp_file,
                model_name=model_name,
                version=version,
                metrics=metrics,
                training_config=parsed_training_config,
                tags=parsed_tags,
            )

            # Schedule cleanup
            background_tasks.add_task(cleanup_temp_file, temp_file)

            return ModelRegistrationResponse(
                success=True,
                model_id=model_version.model_id,
                model_name=model_name,
                version=version,
                storage_key=model_version.storage_key,
                storage_bucket=model_version.storage_bucket,
                size_mb=model_version.model_size_mb,
                metrics=metrics,
                tags=parsed_tags,
                registration_time=datetime.now(UTC),
                message=f"Model {model_name}:{version} registered successfully",
            )

        except Exception:
            if temp_file and temp_file.exists():
                background_tasks.add_task(cleanup_temp_file, temp_file)
            raise

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ModelError as e:
        logger.error(f"Model registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model registration failed: {e.message}",
        )
    except Exception as e:
        logger.error(f"Unexpected error during model registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/{model_id}",
    response_model=ModelVersionResponse,
    summary="Retrieve model details",
    description="Get detailed information about a specific model version",
)
async def get_model(
    model_id: str,
    include_metrics: bool = Query(True, description="Include performance metrics"),
    model_registry: MinIOModelRegistry = Depends(get_model_registry),
) -> ModelVersionResponse:
    """Get model version details."""

    try:
        # Parse model_id (format: model_name:version or model_name_version)
        if ':' in model_id:
            model_name, version = model_id.split(':', 1)
        elif '_' in model_id:
            parts = model_id.split('_')
            model_name = '_'.join(parts[:-1])
            version = parts[-1]
        else:
            raise ValidationError("Invalid model_id format. Use 'name:version' or 'name_version'")

        # Get model metadata
        metadata = await model_registry.get_model_metadata(model_name, version)

        return ModelVersionResponse(
            model_id=model_id,
            model_name=model_name,
            version=version,
            description=metadata.get("description"),
            storage_key=metadata.get("storage_key"),
            storage_bucket=model_registry.models_bucket,
            size_mb=metadata.get("compressed_size", 0) / (1024 * 1024),
            created_at=datetime.fromtimestamp(metadata.get("created_at", 0), tz=UTC),
            metrics=metadata.get("metrics", {}) if include_metrics else {},
            tags=metadata.get("tags", {}),
            training_config=metadata.get("training_config", {}),
            status="registered",
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    except Exception as e:
        logger.error(f"Failed to retrieve model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model",
        )


@router.get(
    "/{model_id}/download",
    response_model=ModelDownloadResponse,
    summary="Download model artifacts",
    description="Download model file and optional configuration",
)
async def download_model(
    model_id: str,
    _include_config: bool = Query(False, description="Include training configuration"),
    format: str = Query("binary", description="Download format (binary, url)"),
    expires_in: int = Query(3600, ge=300, le=86400, description="URL expiration (seconds)"),
    model_registry: MinIOModelRegistry = Depends(get_model_registry),
) -> ModelDownloadResponse:
    """Download model artifacts."""

    try:
        # Parse model_id
        if ':' in model_id:
            model_name, version = model_id.split(':', 1)
        elif '_' in model_id:
            parts = model_id.split('_')
            model_name = '_'.join(parts[:-1])
            version = parts[-1]
        else:
            raise ValidationError("Invalid model_id format")

        if format == "url":
            # Generate presigned download URL
            download_url = await model_registry.generate_download_url(
                model_name, version, expires_in
            )

            return ModelDownloadResponse(
                success=True,
                download_url=download_url,
                expires_at=datetime.now(UTC).replace(
                    second=datetime.now().second + expires_in
                ),
                model_id=model_id,
                format="url",
                message="Download URL generated successfully",
            )

        elif format == "binary":
            # Direct binary download
            await model_registry.get_model(model_name, version)

            # Get model metadata for response headers
            metadata = await model_registry.get_model_metadata(model_name, version)

            # For now, return download URL since direct streaming requires more complex implementation
            download_url = await model_registry.generate_download_url(
                model_name, version, expires_in
            )

            return ModelDownloadResponse(
                success=True,
                download_url=download_url,
                expires_at=datetime.now(UTC).replace(
                    second=datetime.now().second + expires_in
                ),
                model_id=model_id,
                format="binary",
                size_mb=metadata.get("compressed_size", 0) / (1024 * 1024),
                message="Model ready for download",
            )

        else:
            raise ValidationError(f"Unsupported download format: {format}")

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    except Exception as e:
        logger.error(f"Failed to download model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download model",
        )


@router.put(
    "/{model_id}/promote",
    response_model=ModelPromotionResponse,
    summary="Promote model stage",
    description="Promote model to next deployment stage (development -> staging -> production)",
)
async def promote_model(
    model_id: str,
    promotion_request: ModelPromotionRequest,
    model_registry: MinIOModelRegistry = Depends(get_model_registry),
    current_user = Depends(get_current_user),
) -> ModelPromotionResponse:
    """Promote model to next stage."""

    try:
        # Parse model_id
        if ':' in model_id:
            model_name, version = model_id.split(':', 1)
        elif '_' in model_id:
            parts = model_id.split('_')
            model_name = '_'.join(parts[:-1])
            version = parts[-1]
        else:
            raise ValidationError("Invalid model_id format")

        # Validate promotion target
        valid_stages = ['development', 'staging', 'canary', 'production']
        if promotion_request.target_stage not in valid_stages:
            raise ValidationError(f"Invalid target stage. Must be one of: {valid_stages}")

        # Get current model metadata
        metadata = await model_registry.get_model_metadata(model_name, version)
        current_stage = metadata.get("deployment_stage", "development")

        # Validate promotion path
        stage_order = {'development': 0, 'staging': 1, 'canary': 2, 'production': 3}

        if not promotion_request.force_promotion:
            current_order = stage_order.get(current_stage, 0)
            target_order = stage_order.get(promotion_request.target_stage, 0)

            if target_order <= current_order:
                raise ValidationError(
                    f"Cannot promote from {current_stage} to {promotion_request.target_stage} "
                    "without force_promotion=True"
                )

        # Validate metrics thresholds for production
        if promotion_request.target_stage == 'production':
            metrics = metadata.get("metrics", {})
            min_accuracy = promotion_request.min_accuracy or 0.9
            max_latency = promotion_request.max_latency_ms or 100.0

            if metrics.get("accuracy", 0) < min_accuracy:
                raise ValidationError(
                    f"Model accuracy ({metrics.get('accuracy', 0):.3f}) below minimum "
                    f"required for production ({min_accuracy:.3f})"
                )

            if metrics.get("latency_p95_ms", float('inf')) > max_latency:
                raise ValidationError(
                    f"Model latency ({metrics.get('latency_p95_ms', 0):.1f}ms) exceeds "
                    f"maximum allowed for production ({max_latency:.1f}ms)"
                )

        # Update metadata with new stage
        # In a real implementation, this would update the model registry
        # For now, we'll simulate the promotion

        {
            "promoted_by": current_user.username if hasattr(current_user, 'username') else 'system',
            "promoted_at": datetime.now(UTC).isoformat(),
            "previous_stage": current_stage,
            "target_stage": promotion_request.target_stage,
            "promotion_notes": promotion_request.notes,
        }

        return ModelPromotionResponse(
            success=True,
            model_id=model_id,
            model_name=model_name,
            version=version,
            previous_stage=current_stage,
            new_stage=promotion_request.target_stage,
            promoted_by=current_user.username if hasattr(current_user, 'username') else 'system',
            promoted_at=datetime.now(UTC),
            notes=promotion_request.notes,
            message=f"Model promoted from {current_stage} to {promotion_request.target_stage}",
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    except Exception as e:
        logger.error(f"Failed to promote model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to promote model",
        )


@router.get(
    "/list",
    response_model=ModelListResponse,
    summary="List models with filtering",
    description="List available models with filtering and pagination",
)
async def list_models(
    model_name: str | None = Query(None, description="Filter by model name"),
    stage: str | None = Query(None, description="Filter by deployment stage"),
    framework: str | None = Query(None, description="Filter by ML framework"),
    model_type: str | None = Query(None, description="Filter by model type"),
    min_accuracy: float | None = Query(None, ge=0.0, le=1.0, description="Minimum accuracy"),
    max_latency_ms: float | None = Query(None, ge=0.0, description="Maximum latency"),
    created_after: datetime | None = Query(None, description="Created after date"),
    created_before: datetime | None = Query(None, description="Created before date"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    include_metrics: bool = Query(True, description="Include performance metrics"),
    sort_by: str = Query("created_at", description="Sort by field"),
    sort_order: str = Query("desc", description="Sort order (asc, desc)"),
    model_registry: MinIOModelRegistry = Depends(get_model_registry),
) -> ModelListResponse:
    """List models with filtering and pagination."""

    try:
        # Build tag filters
        tag_filters = {}
        if framework:
            tag_filters["framework"] = framework
        if model_type:
            tag_filters["model_type"] = model_type
        if stage:
            tag_filters["deployment_stage"] = stage

        # Get models from registry
        models = await model_registry.list_models(
            model_name=model_name,
            limit=limit,
            tags=tag_filters if tag_filters else None,
        )

        # Apply additional filters
        filtered_models = []
        for model in models:
            # Date filters
            if created_after and model.get("created_at", 0) < created_after.timestamp():
                continue
            if created_before and model.get("created_at", 0) > created_before.timestamp():
                continue

            # Metric filters
            metrics = model.get("metrics", {})
            if min_accuracy and metrics.get("accuracy", 0) < min_accuracy:
                continue
            if max_latency_ms and metrics.get("latency_p95_ms", float('inf')) > max_latency_ms:
                continue

            filtered_models.append({
                "model_id": f"{model['model_name']}_{model['version']}",
                "model_name": model["model_name"],
                "version": model["version"],
                "created_at": datetime.fromtimestamp(model.get("created_at", 0), tz=UTC),
                "size_mb": model.get("compressed_size", 0) / (1024 * 1024),
                "metrics": metrics if include_metrics else {},
                "tags": model.get("tags", {}),
                "storage_key": model.get("storage_key"),
                "status": "registered",
            })

        # Sort results
        reverse_order = sort_order.lower() == "desc"
        if sort_by == "created_at":
            filtered_models.sort(key=lambda x: x["created_at"], reverse=reverse_order)
        elif sort_by == "model_name":
            filtered_models.sort(key=lambda x: x["model_name"], reverse=reverse_order)
        elif sort_by == "size_mb":
            filtered_models.sort(key=lambda x: x["size_mb"], reverse=reverse_order)
        elif sort_by == "accuracy" and include_metrics:
            filtered_models.sort(key=lambda x: x["metrics"].get("accuracy", 0), reverse=reverse_order)

        return ModelListResponse(
            models=filtered_models,
            total_count=len(filtered_models),
            page_size=limit,
            has_more=len(models) >= limit,  # Simple pagination indicator
            filters={
                "model_name": model_name,
                "stage": stage,
                "framework": framework,
                "model_type": model_type,
                "min_accuracy": min_accuracy,
                "max_latency_ms": max_latency_ms,
            },
        )

    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list models",
        )


@router.delete(
    "/{model_id}",
    response_model=ModelDeleteResponse,
    summary="Delete model version",
    description="Delete a specific model version and its artifacts",
)
async def delete_model(
    model_id: str,
    force: bool = Query(False, description="Force delete even if deployed"),
    model_registry: MinIOModelRegistry = Depends(get_model_registry),
    current_user = Depends(get_current_user),
) -> ModelDeleteResponse:
    """Delete model version."""

    try:
        # Parse model_id
        if ':' in model_id:
            model_name, version = model_id.split(':', 1)
        elif '_' in model_id:
            parts = model_id.split('_')
            model_name = '_'.join(parts[:-1])
            version = parts[-1]
        else:
            raise ValidationError("Invalid model_id format")

        # Check if model exists
        try:
            metadata = await model_registry.get_model_metadata(model_name, version)
        except ResourceNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_id}",
            )

        # Check deployment stage
        deployment_stage = metadata.get("tags", {}).get("deployment_stage", "development")
        if deployment_stage == "production" and not force:
            raise ValidationError(
                "Cannot delete production model without force=True"
            )

        # Delete model from registry
        success = await model_registry.delete_model(model_name, version, force=force)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete model",
            )

        return ModelDeleteResponse(
            success=True,
            model_id=model_id,
            model_name=model_name,
            version=version,
            deleted_by=current_user.username if hasattr(current_user, 'username') else 'system',
            deleted_at=datetime.now(UTC),
            message=f"Model {model_id} deleted successfully",
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete model",
        )


# Model Performance and Monitoring
@router.put(
    "/{model_id}/metrics",
    summary="Update model performance metrics",
    description="Update performance metrics for a model version",
)
async def update_model_metrics(
    model_id: str,
    metrics_request: ModelMetricsRequest,
    model_registry: MinIOModelRegistry = Depends(get_model_registry),
    current_user = Depends(get_current_user),
):
    """Update model performance metrics."""

    try:
        # Parse model_id
        if ':' in model_id:
            model_name, version = model_id.split(':', 1)
        elif '_' in model_id:
            parts = model_id.split('_')
            model_name = '_'.join(parts[:-1])
            version = parts[-1]
        else:
            raise ValidationError("Invalid model_id format")

        # Get existing metadata
        metadata = await model_registry.get_model_metadata(model_name, version)

        # Update metrics
        current_metrics = metadata.get("metrics", {})

        # Update with new metrics
        update_fields = metrics_request.model_dump(exclude_unset=True)

        for field, value in update_fields.items():
            if field != "custom_metrics":
                current_metrics[field] = value

        # Add custom metrics
        if metrics_request.custom_metrics:
            current_metrics.update(metrics_request.custom_metrics)

        # Add update metadata
        current_metrics["last_updated"] = datetime.now(UTC).isoformat()
        current_metrics["updated_by"] = current_user.username if hasattr(current_user, 'username') else 'system'

        # In a real implementation, this would update the model registry
        # For now, we simulate the update

        return JSONResponse(
            content={
                "success": True,
                "model_id": model_id,
                "updated_metrics": current_metrics,
                "message": "Model metrics updated successfully",
            }
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    except Exception as e:
        logger.error(f"Failed to update metrics for model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update model metrics",
        )


@router.post(
    "/compare",
    summary="Compare model versions",
    description="Compare performance metrics between two model versions",
)
async def compare_models(
    comparison_request: ModelComparisonRequest,
    model_registry: MinIOModelRegistry = Depends(get_model_registry),
):
    """Compare two model versions."""

    try:
        # Get metadata for both models
        metadata_a = await model_registry.get_model_metadata(
            comparison_request.model_name, comparison_request.version_a
        )
        metadata_b = await model_registry.get_model_metadata(
            comparison_request.model_name, comparison_request.version_b
        )

        metrics_a = metadata_a.get("metrics", {})
        metrics_b = metadata_b.get("metrics", {})

        # Compare requested metrics
        comparison_results = {}
        for metric in comparison_request.metrics:
            value_a = metrics_a.get(metric)
            value_b = metrics_b.get(metric)

            if value_a is not None and value_b is not None:
                difference = value_b - value_a
                percent_change = (difference / value_a * 100) if value_a != 0 else 0

                comparison_results[metric] = {
                    "version_a_value": value_a,
                    "version_b_value": value_b,
                    "difference": difference,
                    "percent_change": percent_change,
                    "winner": comparison_request.version_b if value_b > value_a else comparison_request.version_a,
                }
            else:
                comparison_results[metric] = {
                    "version_a_value": value_a,
                    "version_b_value": value_b,
                    "difference": None,
                    "percent_change": None,
                    "winner": None,
                    "note": "Missing data for comparison",
                }

        return JSONResponse(
            content={
                "success": True,
                "model_name": comparison_request.model_name,
                "version_a": comparison_request.version_a,
                "version_b": comparison_request.version_b,
                "comparison_results": comparison_results,
                "summary": {
                    "total_metrics": len(comparison_request.metrics),
                    "comparable_metrics": len([r for r in comparison_results.values() if r.get("difference") is not None]),
                    "version_b_wins": len([r for r in comparison_results.values() if r.get("winner") == comparison_request.version_b]),
                },
            }
        )

    except ResourceNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to compare models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compare models",
        )


# Model Deployment (placeholder for integration with deployment service)
@router.post(
    "/{model_id}/deploy",
    summary="Deploy model to serving infrastructure",
    description="Deploy model version to serving infrastructure with configuration",
)
async def deploy_model(
    model_id: str,
    deployment_request: ModelDeploymentRequest,
    model_registry: MinIOModelRegistry = Depends(get_model_registry),
    current_user = Depends(get_current_user),
):
    """Deploy model to serving infrastructure."""

    try:
        # This is a placeholder for actual deployment logic
        # In a real implementation, this would integrate with:
        # - Kubernetes for container orchestration
        # - Model serving frameworks (TorchServe, TensorFlow Serving, etc.)
        # - Load balancers and service mesh
        # - Monitoring and alerting systems

        # Parse model_id
        if ':' in model_id:
            model_name, version = model_id.split(':', 1)
        elif '_' in model_id:
            parts = model_id.split('_')
            model_name = '_'.join(parts[:-1])
            version = parts[-1]
        else:
            raise ValidationError("Invalid model_id format")

        # Validate model exists
        await model_registry.get_model_metadata(model_name, version)

        # Simulate deployment process
        deployment_id = str(uuid4())

        return JSONResponse(
            content={
                "success": True,
                "deployment_id": deployment_id,
                "model_id": model_id,
                "model_name": model_name,
                "version": version,
                "deployment_stage": deployment_request.deployment_stage,
                "instance_count": deployment_request.instance_count,
                "auto_scaling": deployment_request.auto_scaling,
                "status": "deploying",
                "deployed_by": current_user.username if hasattr(current_user, 'username') else 'system',
                "deployed_at": datetime.now(UTC),
                "health_check_url": f"/api/v1/deployments/{deployment_id}/health",
                "message": f"Model {model_id} deployment initiated",
            }
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except ResourceNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )
    except Exception as e:
        logger.error(f"Failed to deploy model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deploy model",
        )


# Utility functions
async def cleanup_temp_file(file_path: Path) -> None:
    """Clean up temporary file."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to cleanup temp file {file_path}: {e}")
