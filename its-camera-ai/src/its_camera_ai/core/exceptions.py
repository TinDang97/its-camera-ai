"""Custom exceptions for ITS Camera AI system.

Defines a hierarchy of custom exceptions with proper error codes,
messages, and context information for better error handling.
"""

from typing import Any, Dict, Optional, Union


class ITSCameraAIError(Exception):
    """Base exception for all ITS Camera AI errors.
    
    Attributes:
        message: Human-readable error message
        code: Error code for programmatic handling
        details: Additional context information
        cause: Original exception that caused this error
    """
    
    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN_ERROR",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause
        
    def to_dict(self) -> Dict[str, Any]:
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
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None,
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
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
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
        operation: Optional[str] = None,
        table: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
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
        processor: Optional[str] = None,
        input_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
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
        model_name: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
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
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        code = "INFERENCE_ERROR"
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
        stream_id: Optional[str] = None,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
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
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, "AUTHENTICATION_ERROR", details)


class AuthorizationError(ITSCameraAIError):
    """Raised when authorization fails."""
    
    def __init__(
        self,
        message: str = "Access denied",
        resource: Optional[str] = None,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
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
        limit: Optional[int] = None,
        window: Optional[int] = None,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
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
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
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
        resource_id: Union[str, int],
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"{resource_type} with ID '{resource_id}' not found"
        code = "RESOURCE_NOT_FOUND"
        resource_details = details or {}
        resource_details.update({
            "resource_type": resource_type,
            "resource_id": str(resource_id),
        })
        
        super().__init__(message, code, resource_details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class ConcurrencyError(ITSCameraAIError):
    """Raised when concurrency-related operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        code = "CONCURRENCY_ERROR"
        concurrency_details = details or {}
        if operation:
            concurrency_details["operation"] = operation
            
        super().__init__(message, code, concurrency_details, cause)
        self.operation = operation
