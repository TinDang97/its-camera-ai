"""Main application settings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from .compression import CompressionConfig
from .database import DatabaseConfig
from .ml import MLConfig, MLStreamingConfig
from .monitoring import MonitoringConfig
from .performance import PerformanceConfig
from .redis import RedisConfig, RedisQueueConfig
from .security import SecurityConfig
from .storage import MinIOConfig
from .streaming import SSEStreamingConfig, StreamingConfig


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
