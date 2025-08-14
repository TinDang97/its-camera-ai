"""Monitoring and observability configuration."""

from pydantic import BaseModel, Field


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
