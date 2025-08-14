"""Configuration management for ITS Camera AI system.

Provides centralized configuration management using Pydantic settings
with environment variable support and validation.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """Database configuration settings."""

    url: str = Field(
        default="postgresql+asyncpg://user:pass@localhost/its_camera_ai",
        description="Database connection URL",
    )
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum pool overflow")
    pool_timeout: int = Field(default=30, description="Pool connection timeout")
    echo: bool = Field(default=False, description="Enable SQL query logging")


class RedisConfig(BaseModel):
    """Redis configuration settings."""

    url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )
    max_connections: int = Field(default=20, description="Maximum connections")
    timeout: int = Field(default=30, description="Connection timeout")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")


class RedisQueueConfig(BaseModel):
    """Redis queue configuration settings."""

    url: str = Field(
        default="redis://localhost:6379/0", description="Redis queue connection URL"
    )
    pool_size: int = Field(default=20, description="Connection pool size")
    timeout: int = Field(default=30, description="Connection timeout")
    retry_on_failure: bool = Field(
        default=True, description="Retry on connection failure"
    )

    # Queue settings
    input_queue: str = Field(default="camera_frames", description="Input queue name")
    output_queue: str = Field(
        default="processed_frames", description="Output queue name"
    )
    max_queue_length: int = Field(default=10000, description="Maximum queue length")
    batch_size: int = Field(default=20, description="Batch processing size")

    # Serialization settings
    enable_compression: bool = Field(
        default=True, description="Enable image compression"
    )
    compression_format: str = Field(
        default="jpeg", description="Image compression format"
    )
    compression_quality: int = Field(
        default=85, description="Compression quality (1-100)"
    )


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


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""

    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    enable_profiling: bool = Field(
        default=False, description="Enable performance profiling"
    )
    prometheus_port: int = Field(default=8000, description="Prometheus metrics port")
    jaeger_endpoint: str | None = Field(
        default=None, description="Jaeger tracing endpoint"
    )
    sentry_dsn: str | None = Field(
        default=None, description="Sentry error tracking DSN"
    )


class SecurityConfig(BaseModel):
    """Security configuration settings."""

    enabled: bool = Field(default=True, description="Enable authentication")
    # JWT Settings
    secret_key: str = Field(
        default_factory=lambda: os.getenv("SECRET_KEY") or "change-me-in-production",
        min_length=32,
        description="Secret key for JWT tokens",
    )

    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key strength."""
        weak_secrets = {
            "change-me-in-production",
            "secret",
            "password",
            "admin",
            "test",
            "development"
        }

        if v.lower() in weak_secrets:
            raise ValueError(f"Secret key '{v}' is a default/weak value. Use a cryptographically secure key.")

        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")

        return v
    algorithm: str = Field(
        default="RS256", description="JWT algorithm (RS256 recommended for production)"
    )
    access_token_expire_minutes: int = Field(
        default=15, description="Access token expiration time in minutes"
    )
    refresh_token_expire_days: int = Field(
        default=7, description="Refresh token expiration time in days"
    )

    # CORS Settings
    enable_cors: bool = Field(default=True, description="Enable CORS")
    allowed_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    allow_credentials: bool = Field(default=True, description="Allow credentials")
    allow_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"], description="Allowed CORS methods"
    )
    allow_headers: list[str] = Field(default=["*"], description="Allowed CORS headers")
    # Rate Limiting
    rate_limit_per_minute: int = Field(
        default=100, description="Rate limit per minute per IP"
    )
    auth_rate_limit_per_minute: int = Field(
        default=5, description="Authentication rate limit per minute per IP"
    )

    # Password Policy
    password_min_length: int = Field(default=12, description="Minimum password length")
    password_require_uppercase: bool = Field(
        default=True, description="Require uppercase in password"
    )
    password_require_lowercase: bool = Field(
        default=True, description="Require lowercase in password"
    )
    password_require_digits: bool = Field(
        default=True, description="Require digits in password"
    )
    password_require_special: bool = Field(
        default=True, description="Require special characters in password"
    )
    password_history_size: int = Field(
        default=12, description="Number of previous passwords to remember"
    )

    # MFA Settings
    mfa_issuer_name: str = Field(
        default="ITS Camera AI", description="MFA issuer name for TOTP"
    )
    mfa_totp_window: int = Field(
        default=1, description="TOTP time window (30s intervals)"
    )
    mfa_backup_codes_count: int = Field(
        default=8, description="Number of MFA backup codes to generate"
    )

    # Session Management
    session_timeout_minutes: int = Field(
        default=480, description="Session timeout in minutes (8 hours)"
    )
    max_sessions_per_user: int = Field(
        default=5, description="Maximum concurrent sessions per user"
    )
    session_sliding_expiration: bool = Field(
        default=True, description="Enable sliding session expiration"
    )

    # Brute Force Protection
    max_login_attempts: int = Field(
        default=5, description="Maximum failed login attempts"
    )
    lockout_duration_minutes: int = Field(
        default=15, description="Account lockout duration in minutes"
    )
    attempt_window_minutes: int = Field(
        default=5, description="Time window for counting failed attempts"
    )

    # Security Headers
    enable_security_headers: bool = Field(
        default=True, description="Enable security headers"
    )
    hsts_max_age: int = Field(
        default=31536000, description="HSTS max age in seconds (1 year)"
    )

    # Audit Logging
    enable_audit_logging: bool = Field(
        default=True, description="Enable security audit logging"
    )
    audit_log_retention_days: int = Field(
        default=365, description="Audit log retention in days"
    )
    high_risk_alert_threshold: int = Field(
        default=80, description="Risk score threshold for alerts"
    )

    # Encryption
    enable_at_rest_encryption: bool = Field(
        default=True, description="Enable data encryption at rest"
    )
    enable_in_transit_encryption: bool = Field(
        default=True, description="Enable data encryption in transit"
    )
    encryption_key_rotation_days: int = Field(
        default=90, description="Encryption key rotation period"
    )

    # OAuth2/OIDC
    enable_oauth2: bool = Field(
        default=False, description="Enable OAuth2/OIDC integration"
    )
    oauth2_provider_url: str | None = Field(
        default=None, description="OAuth2 provider URL"
    )
    oauth2_client_id: str | None = Field(default=None, description="OAuth2 client ID")
    oauth2_client_secret: str | None = Field(
        default=None, description="OAuth2 client secret"
    )

    # Zero Trust
    enable_zero_trust: bool = Field(
        default=True, description="Enable zero trust architecture"
    )
    require_device_verification: bool = Field(
        default=False, description="Require device verification"
    )
    enable_continuous_verification: bool = Field(
        default=True, description="Enable continuous security verification"
    )

    # Hosted Proxy
    allowed_hosts: list[str] = Field(
        default=["localhost"], description="List of allowed hosts for the proxy"
    )

    # API Security Enhancement Settings
    enable_api_key_auth: bool = Field(
        default=True, description="Enable API key authentication"
    )
    enable_csrf_protection: bool = Field(
        default=True, description="Enable CSRF protection"
    )
    enable_security_validation: bool = Field(
        default=True, description="Enable comprehensive input validation"
    )
    enable_enhanced_rate_limiting: bool = Field(
        default=True, description="Enable enhanced rate limiting"
    )

    # CSP Settings
    csp_report_uri: str | None = Field(
        default=None, description="CSP violation report URI"
    )

    # File Upload Security
    max_upload_size: int = Field(
        default=100 * 1024 * 1024, description="Maximum upload size in bytes (100MB)"
    )
    allowed_file_types: list[str] = Field(
        default=["image/jpeg", "image/png", "video/mp4", "application/json"],
        description="Allowed file MIME types"
    )

    # Input Validation Settings
    max_json_depth: int = Field(
        default=10, description="Maximum JSON nesting depth"
    )
    max_request_size: int = Field(
        default=50 * 1024 * 1024, description="Maximum request size in bytes (50MB)"
    )

    # Security Monitoring
    enable_security_monitoring: bool = Field(
        default=True, description="Enable security event monitoring"
    )
    security_alert_webhook: str | None = Field(
        default=None, description="Webhook URL for security alerts"
    )


class MinIOConfig(BaseModel):
    """MinIO object storage configuration."""

    # Connection settings
    endpoint: str = Field(default="localhost:9000", description="MinIO server endpoint")
    access_key: str = Field(default="minioadmin", description="MinIO access key")
    secret_key: str = Field(
        default="minioadmin123", description="MinIO secret key", repr=False
    )
    secure: bool = Field(default=False, description="Use HTTPS connection")
    region: str = Field(default="us-east-1", description="MinIO region")

    # Bucket settings
    default_bucket: str = Field(
        default="its-camera-ai", description="Default storage bucket"
    )
    models_bucket: str = Field(
        default="its-models", description="ML model artifacts bucket"
    )
    video_bucket: str = Field(default="its-video", description="Video frames bucket")
    logs_bucket: str = Field(default="its-logs", description="Logs and metadata bucket")
    create_buckets: bool = Field(
        default=True, description="Auto-create buckets on startup"
    )

    # Upload/Download settings
    multipart_threshold: int = Field(
        default=64 * 1024 * 1024,  # 64MB
        description="Multipart upload threshold (bytes)",
    )
    multipart_chunksize: int = Field(
        default=16 * 1024 * 1024,  # 16MB
        description="Multipart chunk size (bytes)",
    )
    max_concurrent_uploads: int = Field(
        default=4, description="Maximum concurrent uploads"
    )

    # Connection pooling
    max_pool_connections: int = Field(
        default=20, description="Maximum pool connections"
    )
    connection_timeout: int = Field(
        default=60, description="Connection timeout (seconds)"
    )
    read_timeout: int = Field(default=300, description="Read timeout (seconds)")

    # Caching
    enable_caching: bool = Field(
        default=True, description="Enable Redis caching for metadata"
    )
    cache_ttl: int = Field(default=3600, description="Cache TTL (seconds)")
    cache_max_size: int = Field(default=1000, description="Max cached objects")

    # Security and performance
    enable_versioning: bool = Field(
        default=True, description="Enable object versioning"
    )
    enable_encryption: bool = Field(
        default=True, description="Enable server-side encryption"
    )
    enable_compression: bool = Field(
        default=True, description="Enable data compression"
    )
    compression_level: int = Field(
        default=6, ge=1, le=9, description="Compression level (1-9)"
    )

    # Monitoring
    enable_metrics: bool = Field(
        default=True, description="Enable MinIO metrics collection"
    )
    metrics_interval: int = Field(
        default=60, description="Metrics collection interval (seconds)"
    )

    # Lifecycle management
    enable_lifecycle: bool = Field(
        default=True, description="Enable object lifecycle management"
    )
    video_retention_days: int = Field(
        default=30, description="Video frame retention (days)"
    )
    retention_days: int = Field(
        default=365, description="Model artifact retention (days)"
    )
    log_retention_days: int = Field(default=90, description="Log retention (days)")


class MLStreamingConfig(BaseModel):
    """ML streaming configuration for AI-annotated video streams."""

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


class CompressionConfig(BaseModel):
    """Compression settings."""

    enabled: bool = Field(default=True, description="Enable compression")
    level: int = Field(default=6, ge=1, le=9, description="Compression level (1-9)")
    min_size: int = Field(
        default=1024, ge=0, description="Minimum size to compress (bytes)"
    )
    formats: list[str] = Field(
        default=["jpeg", "png"], description="Supported compression formats"
    )


class RateLimit(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(default=100, ge=1)
    burst_size: int = Field(default=10, ge=1)


class StreamingConfig(BaseModel):
    """Streaming service configuration."""

    # Core streaming settings
    min_resolution: tuple[int, int] = Field(
        default=(640, 480), description="Minimum frame resolution (width, height)"
    )
    min_quality_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum quality score threshold"
    )
    max_blur_threshold: float = Field(
        default=100.0, ge=0.0, description="Maximum blur threshold"
    )
    max_concurrent_streams: int = Field(
        default=100, ge=1, description="Maximum concurrent camera streams"
    )
    frame_processing_timeout: float = Field(
        default=0.01, ge=0.001, description="Frame processing timeout in seconds (10ms)"
    )

    # gRPC server settings
    grpc_host: str = Field(
        default="127.0.0.1", description="gRPC server host"
    )
    grpc_port: int = Field(
        default=50051, ge=1024, le=65535, description="gRPC server port"
    )

    # Performance settings
    batch_size: int = Field(
        default=10, ge=1, description="Frame batch size for processing"
    )
    queue_max_length: int = Field(
        default=10000, ge=100, description="Maximum queue length"
    )
    worker_threads: int = Field(
        default=4, ge=1, description="Number of worker threads"
    )

    # Health monitoring
    health_check_interval: int = Field(
        default=30, ge=5, description="Health check interval in seconds"
    )
    metrics_collection_interval: int = Field(
        default=10, ge=1, description="Metrics collection interval in seconds"
    )


class PerformanceConfig(BaseModel):
    """Performance optimization configuration."""

    # Performance optimization strategy
    enabled: bool = Field(
        default=True, description="Enable performance optimization"
    )
    strategy: str = Field(
        default="latency_optimized",
        description="Optimization strategy: latency_optimized, memory_optimized, balanced"
    )
    target_latency_ms: float = Field(
        default=100.0, ge=10.0, le=1000.0, description="Target end-to-end latency in ms"
    )
    max_concurrent_streams: int = Field(
        default=100, ge=1, le=1000, description="Maximum concurrent streams to optimize for"
    )

    # GPU optimization
    gpu_optimization_enabled: bool = Field(
        default=True, description="Enable GPU memory optimization"
    )
    gpu_memory_pool_size_gb: float = Field(
        default=8.0, ge=1.0, description="GPU memory pool size in GB"
    )
    enable_tensorrt: bool = Field(
        default=True, description="Enable TensorRT optimization"
    )
    tensorrt_precision: str = Field(
        default="fp16", description="TensorRT precision: fp32, fp16, int8"
    )

    # Caching configuration
    caching_enabled: bool = Field(
        default=True, description="Enable multi-level caching"
    )
    l1_cache_size_mb: int = Field(
        default=512, ge=64, description="L1 cache size in MB"
    )
    l2_cache_enabled: bool = Field(
        default=True, description="Enable L2 Redis cache"
    )
    cache_ttl_seconds: int = Field(
        default=300, ge=60, description="Cache TTL in seconds"
    )

    # Connection pooling
    redis_pool_size: int = Field(
        default=50, ge=5, description="Redis connection pool size"
    )
    database_pool_size: int = Field(
        default=20, ge=5, description="Database connection pool size"
    )
    ffmpeg_pool_size: int = Field(
        default=10, ge=2, description="FFMPEG process pool size"
    )

    # Latency monitoring
    latency_monitoring_enabled: bool = Field(
        default=True, description="Enable latency SLA monitoring"
    )
    latency_sla_ms: float = Field(
        default=100.0, ge=10.0, description="Latency SLA in milliseconds"
    )
    enable_sla_alerts: bool = Field(
        default=True, description="Enable SLA violation alerts"
    )

    # Adaptive quality management
    adaptive_quality_enabled: bool = Field(
        default=True, description="Enable adaptive quality management"
    )
    default_quality: str = Field(
        default="high", description="Default quality level: low, medium, high"
    )
    quality_adjustment_interval: int = Field(
        default=30, ge=5, description="Quality adjustment interval in seconds"
    )
    priority_camera_ids: list[str] = Field(
        default_factory=list, description="Priority camera IDs for quality protection"
    )


class SSEStreamingConfig(BaseModel):
    """SSE (Server-Sent Events) streaming configuration."""

    # Connection management
    max_concurrent_connections: int = Field(
        default=100, ge=1, description="Maximum concurrent SSE connections"
    )
    fragment_duration_ms: int = Field(
        default=2000, ge=500, description="MP4 fragment duration in milliseconds"
    )
    heartbeat_interval: int = Field(
        default=30, ge=5, description="Heartbeat interval in seconds"
    )
    connection_timeout: int = Field(
        default=300, ge=60, description="Connection timeout in seconds"
    )

    # Quality settings
    quality_presets: dict[str, dict[str, int]] = Field(
        default={
            "low": {"bitrate": 500, "jpeg_quality": 60},
            "medium": {"bitrate": 2000, "jpeg_quality": 85},
            "high": {"bitrate": 5000, "jpeg_quality": 95},
        },
        description="Quality presets for different stream qualities"
    )

    # Buffer management
    stream_buffer_size: int = Field(
        default=100, ge=10, description="Stream buffer size per connection"
    )
    fragment_buffer_size: int = Field(
        default=50, ge=5, description="Fragment buffer size"
    )

    # Performance optimization
    enable_compression: bool = Field(
        default=True, description="Enable fragment compression"
    )
    compression_level: int = Field(
        default=6, ge=1, le=9, description="Compression level (1-9)"
    )

    # Monitoring
    enable_metrics: bool = Field(
        default=True, description="Enable SSE metrics collection"
    )
    metrics_interval: int = Field(
        default=10, ge=1, description="Metrics collection interval in seconds"
    )

    # Dual-channel settings
    sync_tolerance_ms: float = Field(
        default=50.0, ge=1.0, description="Channel synchronization tolerance in milliseconds"
    )
    sync_check_interval: float = Field(
        default=1.0, ge=0.1, description="Synchronization check interval in seconds"
    )
    max_sync_violations: int = Field(
        default=10, ge=1, description="Maximum sync violations before marking stream degraded"
    )
    enable_dual_channel: bool = Field(
        default=True, description="Enable dual-channel streaming support"
    )
    channel_switch_timeout: float = Field(
        default=0.1, ge=0.01, description="Channel switching timeout in seconds"
    )


class Settings(BaseSettings):
    """Main application settings.

    All settings can be overridden via environment variables.
    For nested settings, use double underscore notation:
    DATABASE__URL=postgresql://...
    """

    # Application settings
    app_name: str = Field(default="ITS Camera AI", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # API settings
    api_host: str = Field(
        default="0.0.0.0",  # noqa: S104 - Needed for containerized deployment
        description="API server host",
    )
    api_port: int = Field(default=8080, description="API server port")
    api_prefix: str = Field(default="/api/v1", description="API path prefix")
    reload: bool = Field(default=False, description="Enable auto-reload")
    workers: int = Field(default=1, description="Number of worker processes")

    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    redis_queue: RedisQueueConfig = Field(default_factory=RedisQueueConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    ml_streaming: MLStreamingConfig = Field(default_factory=MLStreamingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    minio: MinIOConfig = Field(default_factory=MinIOConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    sse_streaming: SSEStreamingConfig = Field(default_factory=SSEStreamingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")

    # File paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    temp_dir: Path = Field(
        default=Path("temp"), description="Temporary files directory"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_nested_delimiter": "__",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment."""
        valid_envs = {"development", "testing", "staging", "production"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v.lower()

    def get_security_headers(self) -> dict[str, str]:
        """Get security headers configuration."""
        if not self.security.enable_security_headers:
            return {}

        headers = {
            "Strict-Transport-Security": f"max-age={self.security.hsts_max_age}; includeSubDomains",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": self._build_csp(),
            "Permissions-Policy": "geolocation=(), microphone=(), camera=(self)",
        }

        return headers

    def _build_csp(self) -> str:
        """Build Content Security Policy."""
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: https:",
            "connect-src 'self' ws: wss:",
            "media-src 'self' blob:",
            "object-src 'none'",
            "frame-src 'none'",
            "worker-src 'self'",
            "manifest-src 'self'",
            "form-action 'self'",
            "base-uri 'self'"
        ]

        if self.is_production():
            csp_directives.append("upgrade-insecure-requests")

        if self.security.csp_report_uri:
            csp_directives.append(f"report-uri {self.security.csp_report_uri}")

        return "; ".join(csp_directives)

    def create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [self.data_dir, self.logs_dir, self.temp_dir, self.ml.model_path]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_database_url(self, async_driver: bool = True) -> str:
        """Get database URL with proper driver."""
        if async_driver:
            return self.database.url
        # Convert to sync driver for migrations
        return self.database.url.replace("postgresql+asyncpg://", "postgresql://")

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"

    def get_allowed_file_extensions(self) -> set[str]:
        """Get allowed file extensions from MIME types."""
        mime_to_ext = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "video/mp4": ".mp4",
            "video/avi": ".avi",
            "video/mov": ".mov",
            "application/json": ".json",
            "text/plain": ".txt",
            "application/pdf": ".pdf"
        }

        extensions = set()
        for mime_type in self.security.allowed_file_types:
            if mime_type in mime_to_ext:
                extensions.add(mime_to_ext[mime_type])

        return extensions


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings.

    Returns:
        Settings: Application settings instance
    """
    settings = Settings()
    settings.create_directories()
    return settings


def get_settings_for_testing(**overrides: Any) -> Settings:
    """Get settings for testing with optional overrides.

    Args:
        **overrides: Settings to override

    Returns:
        Settings: Test settings instance
    """
    # Clear cache to ensure fresh settings
    get_settings.cache_clear()

    # Set test environment variables
    test_env = {
        "ENVIRONMENT": "testing",
        "DATABASE__URL": "postgresql+asyncpg://test:test@localhost/its_camera_ai_test",
        "REDIS__URL": "redis://localhost:6379/1",
        "DEBUG": "true",
        **{k.upper(): str(v) for k, v in overrides.items()},
    }

    # Temporarily set environment variables
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        settings = Settings()
        settings.create_directories()
        return settings
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
