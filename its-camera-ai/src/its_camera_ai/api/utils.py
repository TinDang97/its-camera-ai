"""API utility functions and helpers.

Common utilities for API endpoints including response formatting,
error handling, pagination, and data validation.
"""

from datetime import UTC, datetime
from typing import Any, TypeVar

from fastapi import HTTPException, Query, status

from ..core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class PaginationParams:
    """Common pagination parameters for list endpoints."""

    def __init__(
        self,
        page: int = Query(1, ge=1, description="Page number"),
        size: int = Query(20, ge=1, le=100, description="Items per page"),
    ):
        self.page = page
        self.size = size
        self.offset = (page - 1) * size


def format_timestamp(dt: datetime | None = None) -> str:
    """Format datetime to ISO 8601 string.

    Args:
        dt: Datetime object or None for current time

    Returns:
        ISO 8601 formatted string
    """
    if dt is None:
        dt = datetime.now(UTC)
    return dt.isoformat()


def validate_time_range(
    start_time: datetime, end_time: datetime, max_days: int = 90
) -> None:
    """Validate time range parameters.

    Args:
        start_time: Start of time range
        end_time: End of time range
        max_days: Maximum allowed days in range

    Raises:
        HTTPException: If time range is invalid
    """
    if end_time <= start_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="end_time must be after start_time",
        )

    time_diff = end_time - start_time
    if time_diff.days > max_days:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Time range cannot exceed {max_days} days",
        )


def sanitize_camera_id(camera_id: str) -> str:
    """Sanitize and validate camera ID.

    Args:
        camera_id: Raw camera ID string

    Returns:
        Sanitized camera ID

    Raises:
        HTTPException: If camera ID is invalid
    """
    # Remove whitespace and convert to lowercase
    camera_id = camera_id.strip().lower()

    # Validate format (alphanumeric with underscores/hyphens)
    if not camera_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Camera ID cannot be empty"
        )

    # Check for valid characters
    import re

    if not re.match(r"^[a-z0-9_-]+$", camera_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Camera ID must contain only letters, numbers, hyphens, and underscores",
        )

    return camera_id


def calculate_health_score(
    metrics: dict[str, Any], weights: dict[str, float] | None = None
) -> float:
    """Calculate weighted health score from metrics.

    Args:
        metrics: Dictionary of metric values
        weights: Optional weights for each metric

    Returns:
        Health score between 0 and 100
    """
    if not metrics:
        return 100.0

    if weights is None:
        # Default equal weights
        weights = dict.fromkeys(metrics.keys(), 1.0)

    total_weight = sum(weights.values())
    if total_weight == 0:
        return 100.0

    weighted_sum = sum(metrics.get(k, 0) * weights.get(k, 0) for k in metrics)

    return min(100.0, max(0.0, weighted_sum / total_weight))


def generate_request_id() -> str:
    """Generate unique request ID for tracking.

    Returns:
        UUID string for request tracking
    """
    from uuid import uuid4

    return str(uuid4())


def parse_coordinates(coord_str: str) -> tuple[float, float]:
    """Parse coordinate string to latitude/longitude tuple.

    Args:
        coord_str: Coordinate string in format "lat,lon"

    Returns:
        Tuple of (latitude, longitude)

    Raises:
        HTTPException: If coordinates are invalid
    """
    try:
        parts = coord_str.split(",")
        if len(parts) != 2:
            raise ValueError("Invalid format")

        lat = float(parts[0].strip())
        lon = float(parts[1].strip())

        # Validate ranges
        if not -90 <= lat <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        if not -180 <= lon <= 180:
            raise ValueError("Longitude must be between -180 and 180")

        return lat, lon

    except (ValueError, IndexError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid coordinates format: {str(e)}",
        )


class RateLimitManager:
    """Centralized rate limit management."""

    def __init__(self, default_calls: int = 100, default_period: int = 60):
        self.default_calls = default_calls
        self.default_period = default_period
        self._limiters: dict[str, Any] = {}

    def get_limiter(
        self, key: str, calls: int | None = None, period: int | None = None
    ) -> Any:
        """Get or create rate limiter for a key.

        Args:
            key: Unique key for the rate limiter
            calls: Number of allowed calls
            period: Time period in seconds

        Returns:
            Rate limiter instance
        """
        if key not in self._limiters:
            from ..dependencies import RateLimiter

            self._limiters[key] = RateLimiter(
                calls=calls or self.default_calls, period=period or self.default_period
            )
        return self._limiters[key]


# Global rate limit manager instance
rate_limit_manager = RateLimitManager()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted size string (e.g., '1.5 MB')
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def validate_semantic_version(version: str) -> bool:
    """Validate semantic version format.

    Args:
        version: Version string to validate

    Returns:
        True if version follows semantic versioning
    """
    import re

    pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"
    return bool(re.match(pattern, version))


def generate_model_id(name: str, version: str) -> str:
    """Generate standardized model ID from name and version.

    Args:
        name: Model name
        version: Model version

    Returns:
        Standardized model ID
    """
    import re

    # Normalize name: lowercase, replace spaces/special chars with hyphens
    normalized_name = re.sub(r"[^a-z0-9]+", "-", name.lower().strip())
    normalized_name = re.sub(r"-+", "-", normalized_name).strip("-")

    return f"{normalized_name}-{version}"


def calculate_model_performance_score(
    metrics: dict[str, Any],
    model_type: str,
    target_latency: float = 100.0,
    target_accuracy: float = 0.9,
) -> float:
    """Calculate overall performance score for a model.

    Args:
        metrics: Model performance metrics
        model_type: Type of model (detection, classification, etc.)
        target_latency: Target inference latency in ms
        target_accuracy: Target accuracy threshold

    Returns:
        Performance score between 0 and 100
    """
    score = 0.0
    weight_sum = 0.0

    # Latency score (lower is better)
    if "inference_time" in metrics:
        latency = metrics["inference_time"]
        latency_score = max(0, min(100, 100 * (target_latency / max(latency, 1))))
        score += latency_score * 0.3
        weight_sum += 0.3

    # Accuracy score
    accuracy_field = "map_score" if model_type == "detection" else "accuracy"
    if accuracy_field in metrics:
        accuracy = metrics[accuracy_field]
        accuracy_score = min(100, accuracy * 100 / target_accuracy)
        score += accuracy_score * 0.4
        weight_sum += 0.4

    # Throughput score (higher is better)
    if "throughput" in metrics:
        throughput = metrics["throughput"]
        throughput_score = min(100, throughput)
        score += throughput_score * 0.2
        weight_sum += 0.2

    # Memory efficiency score (lower usage is better)
    if "memory_usage" in metrics:
        memory_mb = metrics["memory_usage"]
        # Assume 2GB as reasonable target for edge deployment
        memory_score = max(0, min(100, 100 * (2048 / max(memory_mb, 1))))
        score += memory_score * 0.1
        weight_sum += 0.1

    return score / weight_sum if weight_sum > 0 else 0.0


def extract_model_metadata_from_filename(filename: str) -> dict[str, Any]:
    """Extract metadata from model filename using common naming patterns.

    Args:
        filename: Model filename

    Returns:
        Dictionary of extracted metadata
    """
    import re
    from pathlib import Path

    metadata = {
        "framework": None,
        "precision": None,
        "optimization": None,
        "size_variant": None,
    }

    # Extract framework from extension
    suffix = Path(filename).suffix.lower()
    framework_map = {
        ".pt": "pytorch",
        ".pth": "pytorch",
        ".onnx": "onnx",
        ".pb": "tensorflow",
        ".tflite": "tensorflow",
        ".engine": "tensorrt",
    }
    metadata["framework"] = framework_map.get(suffix)

    # Extract precision (fp32, fp16, int8, etc.)
    precision_pattern = r"(fp32|fp16|int8|int4)"
    precision_match = re.search(precision_pattern, filename.lower())
    if precision_match:
        metadata["precision"] = precision_match.group(1)

    # Extract optimization type
    opt_pattern = r"(quantized|pruned|distilled|optimized)"
    opt_match = re.search(opt_pattern, filename.lower())
    if opt_match:
        metadata["optimization"] = opt_match.group(1)

    # Extract model size variant (nano, small, medium, large, etc.)
    # Use word boundaries to avoid partial matches
    size_pattern = r"\b(nano|small|medium|large|xlarge)\b|\b([nsml]|xl)\b"
    size_match = re.search(size_pattern, filename.lower())
    if size_match:
        metadata["size_variant"] = size_match.group(1) or size_match.group(2)

    return metadata
