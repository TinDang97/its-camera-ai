"""Domain-specific exceptions for analytics services.

This module provides a hierarchy of exceptions for better error handling
and debugging in the analytics services layer.
"""


class AnalyticsServiceError(Exception):
    """Base exception for analytics service errors."""

    def __init__(self, message: str, service_name: str = "Analytics", **context):
        """Initialize analytics service error.
        
        Args:
            message: Error message
            service_name: Name of the service that raised the error
            **context: Additional context for logging/debugging
        """
        super().__init__(message)
        self.service_name = service_name
        self.context = context


class HistoricalQueryError(AnalyticsServiceError):
    """Exception raised when historical data queries fail."""

    def __init__(self, message: str, query_params: dict = None, **context):
        """Initialize historical query error.
        
        Args:
            message: Error message
            query_params: Query parameters that caused the failure
            **context: Additional context
        """
        super().__init__(message, service_name="HistoricalAnalytics", **context)
        self.query_params = query_params or {}


class RealtimeAnalyticsError(AnalyticsServiceError):
    """Exception raised when real-time analytics operations fail."""

    def __init__(self, message: str, camera_id: str = None, **context):
        """Initialize real-time analytics error.
        
        Args:
            message: Error message  
            camera_id: Camera ID related to the error
            **context: Additional context
        """
        super().__init__(message, service_name="RealtimeAnalytics", **context)
        self.camera_id = camera_id


class IncidentNotFoundError(AnalyticsServiceError):
    """Exception raised when an incident is not found."""

    def __init__(self, incident_id: str, **context):
        """Initialize incident not found error.
        
        Args:
            incident_id: ID of the incident that was not found
            **context: Additional context
        """
        message = f"Incident not found: {incident_id}"
        super().__init__(message, service_name="IncidentManagement", **context)
        self.incident_id = incident_id


class IncidentManagementError(AnalyticsServiceError):
    """Exception raised when incident management operations fail."""

    def __init__(self, message: str, incident_id: str = None, operation: str = None, **context):
        """Initialize incident management error.
        
        Args:
            message: Error message
            incident_id: ID of the incident related to the error
            operation: Operation that failed (e.g., 'update', 'create', 'delete')
            **context: Additional context
        """
        super().__init__(message, service_name="IncidentManagement", **context)
        self.incident_id = incident_id
        self.operation = operation


class CacheError(AnalyticsServiceError):
    """Exception raised when cache operations fail."""

    def __init__(self, message: str, cache_key: str = None, operation: str = None, **context):
        """Initialize cache error.
        
        Args:
            message: Error message
            cache_key: Cache key related to the error
            operation: Cache operation that failed (e.g., 'get', 'set', 'delete')
            **context: Additional context
        """
        super().__init__(message, service_name="Cache", **context)
        self.cache_key = cache_key
        self.operation = operation


class DatabaseError(AnalyticsServiceError):
    """Exception raised when database operations fail."""

    def __init__(self, message: str, table: str = None, operation: str = None, **context):
        """Initialize database error.
        
        Args:
            message: Error message
            table: Database table related to the error
            operation: Database operation that failed (e.g., 'select', 'insert', 'update')
            **context: Additional context
        """
        super().__init__(message, service_name="Database", **context)
        self.table = table
        self.operation = operation


class ValidationError(AnalyticsServiceError):
    """Exception raised when input validation fails."""

    def __init__(self, message: str, field: str = None, value: str = None, **context):
        """Initialize validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Value that failed validation
            **context: Additional context
        """
        super().__init__(message, service_name="Validation", **context)
        self.field = field
        self.value = value


class ConfigurationError(AnalyticsServiceError):
    """Exception raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: str = None, **context):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that is invalid/missing
            **context: Additional context
        """
        super().__init__(message, service_name="Configuration", **context)
        self.config_key = config_key
