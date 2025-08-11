"""Configuration management for ITS Camera AI system.

Provides centralized configuration management using Pydantic settings
with environment variable support and validation.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, validator
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


class KafkaConfig(BaseModel):
    """Kafka configuration settings."""

    bootstrap_servers: list[str] = Field(
        default=["localhost:9092"], description="Kafka bootstrap servers"
    )
    consumer_group_id: str = Field(
        default="its-camera-ai", description="Consumer group ID"
    )
    auto_offset_reset: str = Field(
        default="latest", description="Auto offset reset policy"
    )
    enable_auto_commit: bool = Field(default=True, description="Enable auto commit")


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

    secret_key: str = Field(
        default="change-me-in-production",
        min_length=32,
        description="Secret key for JWT tokens",
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration time in minutes"
    )
    enable_cors: bool = Field(default=True, description="Enable CORS")
    allowed_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    rate_limit_per_minute: int = Field(
        default=100, description="Rate limit per minute per IP"
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
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8080, description="API server port")
    api_prefix: str = Field(default="/api/v1", description="API path prefix")
    reload: bool = Field(default=False, description="Enable auto-reload")
    workers: int = Field(default=1, description="Number of worker processes")

    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # File paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    temp_dir: Path = Field(
        default=Path("temp"), description="Temporary files directory"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False

    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment."""
        valid_envs = {"development", "testing", "staging", "production"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v.lower()

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
