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

    # JWT Settings
    secret_key: str = Field(
        default="change-me-in-production",
        min_length=32,
        description="Secret key for JWT tokens",
    )
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
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    minio: MinIOConfig = Field(default_factory=MinIOConfig)

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
        "extra": "ignore"
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

        return {
            "Strict-Transport-Security": f"max-age={self.security.hsts_max_age}; includeSubDomains",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }

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
