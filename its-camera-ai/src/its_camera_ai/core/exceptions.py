"""Custom exceptions for ITS Camera AI system.

Defines a hierarchy of custom exceptions with proper error codes,
messages, and context information for better error handling.
"""

from typing import Any


class ITSCameraAIError(Exception):
    """Base exception for all ITS Camera AI errors.

    Attributes:
        message: Human-readable error message
        code: Error code for programmatic handling
        details: Additional context information
        cause: Original exception that caused this error
    """

    status_code: int = 400

    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN_ERROR",
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary format."""
        result = {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "details": self.details,
        }
        if self.cause:
            result["cause"] = str(self.cause)
        return result

    def __str__(self) -> str:
        """String representation of the exception."""
        return f"[{self.code}] {self.message}"


class ValidationError(ITSCameraAIError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        code = "VALIDATION_ERROR"
        validation_details = details or {}
        if field:
            validation_details["field"] = field
        if value is not None:
            validation_details["value"] = str(value)

        super().__init__(message, code, validation_details)
        self.field = field
        self.value = value


class ConfigurationError(ITSCameraAIError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        code = "CONFIGURATION_ERROR"
        config_details = details or {}
        if config_key:
            config_details["config_key"] = config_key

        super().__init__(message, code, config_details)
        self.config_key = config_key


class DatabaseError(ITSCameraAIError):
    """Raised when database operations fail."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        table: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        code = "DATABASE_ERROR"
        db_details = details or {}
        if operation:
            db_details["operation"] = operation
        if table:
            db_details["table"] = table

        super().__init__(message, code, db_details, cause)
        self.operation = operation
        self.table = table


class ProcessingError(ITSCameraAIError):
    """Raised when data processing fails."""

    def __init__(
        self,
        message: str,
        processor: str | None = None,
        input_type: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        code = "PROCESSING_ERROR"
        proc_details = details or {}
        if processor:
            proc_details["processor"] = processor
        if input_type:
            proc_details["input_type"] = input_type

        super().__init__(message, code, proc_details, cause)
        self.processor = processor
        self.input_type = input_type


class ModelError(ITSCameraAIError):
    """Raised when ML model operations fail."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        code = "MODEL_ERROR"
        model_details = details or {}
        if model_name:
            model_details["model_name"] = model_name
        if operation:
            model_details["operation"] = operation

        super().__init__(message, code, model_details, cause)
        self.model_name = model_name
        self.operation = operation


class InferenceError(ModelError):
    """Raised when model inference fails."""

    def __init__(
        self,
        message: str,
        model_name: str | None = None,
        batch_size: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        inference_details = details or {}
        if batch_size:
            inference_details["batch_size"] = batch_size

        super().__init__(message, model_name, "inference", inference_details, cause)
        self.batch_size = batch_size


class StreamingError(ITSCameraAIError):
    """Raised when streaming operations fail."""

    def __init__(
        self,
        message: str,
        stream_id: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        code = "STREAMING_ERROR"
        stream_details = details or {}
        if stream_id:
            stream_details["stream_id"] = stream_id
        if operation:
            stream_details["operation"] = operation

        super().__init__(message, code, stream_details, cause)
        self.stream_id = stream_id
        self.operation = operation


class AuthenticationError(ITSCameraAIError):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, "AUTHENTICATION_ERROR", details)


class AuthorizationError(ITSCameraAIError):
    """Raised when authorization fails."""

    def __init__(
        self,
        message: str = "Access denied",
        resource: str | None = None,
        action: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        code = "AUTHORIZATION_ERROR"
        auth_details = details or {}
        if resource:
            auth_details["resource"] = resource
        if action:
            auth_details["action"] = action

        super().__init__(message, code, auth_details)
        self.resource = resource
        self.action = action


class RateLimitError(ITSCameraAIError):
    """Raised when rate limiting is triggered."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: int | None = None,
        window: int | None = None,
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        code = "RATE_LIMIT_ERROR"
        rate_details = details or {}
        if limit:
            rate_details["limit"] = limit
        if window:
            rate_details["window"] = window
        if retry_after:
            rate_details["retry_after"] = retry_after

        super().__init__(message, code, rate_details)
        self.limit = limit
        self.window = window
        self.retry_after = retry_after


class ExternalServiceError(ITSCameraAIError):
    """Raised when external service calls fail."""

    def __init__(
        self,
        message: str,
        service: str,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        code = "EXTERNAL_SERVICE_ERROR"
        service_details = details or {}
        service_details["service"] = service
        if status_code:
            service_details["status_code"] = status_code

        super().__init__(message, code, service_details, cause)
        self.service = service
        self.status_code = status_code


class ResourceNotFoundError(ITSCameraAIError):
    """Raised when a requested resource is not found."""

    def __init__(
        self,
        resource_type: str,
        resource_id: str | int,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"{resource_type} with ID '{resource_id}' not found"
        code = "RESOURCE_NOT_FOUND"
        resource_details = details or {}
        resource_details.update(
            {
                "resource_type": resource_type,
                "resource_id": str(resource_id),
            }
        )

        super().__init__(message, code, resource_details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConcurrencyError(ITSCameraAIError):
    """Raised when concurrency-related operations fail."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        code = "CONCURRENCY_ERROR"
        concurrency_details = details or {}
        if operation:
            concurrency_details["operation"] = operation

        super().__init__(message, code, concurrency_details, cause)
        self.operation = operation


class ServiceMeshError(ITSCameraAIError):
    """Raised when service mesh operations fail."""

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        code = "SERVICE_MESH_ERROR"
        mesh_details = details or {}
        if service_name:
            mesh_details["service_name"] = service_name
        if operation:
            mesh_details["operation"] = operation

        super().__init__(message, code, mesh_details, cause)
        self.service_name = service_name
        self.operation = operation


class CircuitBreakerError(ServiceMeshError):
    """Raised when circuit breaker operations fail."""

    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        state: str | None = None,
        failure_count: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        cb_details = details or {}
        if state:
            cb_details["circuit_breaker_state"] = state
        if failure_count is not None:
            cb_details["failure_count"] = failure_count

        super().__init__(message, service_name, "circuit_breaker", cb_details, cause)
        self.state = state
        self.failure_count = failure_count


class NotFoundError(ITSCameraAIError):
    """Raised when a requested resource is not found."""

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: str | None = None,
        resource_id: str | int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        code = "RESOURCE_NOT_FOUND"
        resource_details = details or {}
        if resource_type:
            resource_details["resource_type"] = resource_type
        if resource_id:
            resource_details["resource_id"] = str(resource_id)

        super().__init__(message, code, resource_details)
        self.resource_type = resource_type
        self.resource_id = resource_id


# Streaming-specific exceptions
class CameraConnectionError(StreamingError):
    """Raised when camera connection fails."""

    def __init__(
        self,
        message: str,
        camera_id: str | None = None,
        stream_url: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        connection_details = details or {}
        if camera_id:
            connection_details["camera_id"] = camera_id
        if stream_url:
            connection_details["stream_url"] = stream_url

        super().__init__(message, camera_id, "connection", connection_details, cause)
        self.camera_id = camera_id
        self.stream_url = stream_url


class CameraConfigurationError(StreamingError):
    """Raised when camera configuration is invalid."""

    def __init__(
        self,
        message: str,
        camera_id: str | None = None,
        config_field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        config_details = details or {}
        if camera_id:
            config_details["camera_id"] = camera_id
        if config_field:
            config_details["config_field"] = config_field

        super().__init__(message, camera_id, "configuration", config_details)
        self.camera_id = camera_id
        self.config_field = config_field


class StreamProcessingError(StreamingError):
    """Raised when stream processing fails."""

    def __init__(
        self,
        message: str,
        stream_id: str | None = None,
        processing_stage: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        processing_details = details or {}
        if processing_stage:
            processing_details["processing_stage"] = processing_stage

        super().__init__(message, stream_id, "processing", processing_details, cause)
        self.processing_stage = processing_stage


class QualityValidationError(StreamingError):
    """Raised when frame quality validation fails."""

    def __init__(
        self,
        message: str,
        frame_id: str | None = None,
        quality_score: float | None = None,
        issues: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        quality_details = details or {}
        if frame_id:
            quality_details["frame_id"] = frame_id
        if quality_score is not None:
            quality_details["quality_score"] = quality_score
        if issues:
            quality_details["issues"] = issues

        super().__init__(message, frame_id, "quality_validation", quality_details)
        self.frame_id = frame_id
        self.quality_score = quality_score
        self.issues = issues or []


class ServiceError(BaseException):
    """Base class for all service-related errors."""

    def __init__(
        self,
        message: str,
        service: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.service = service
        self.details = details or {}
