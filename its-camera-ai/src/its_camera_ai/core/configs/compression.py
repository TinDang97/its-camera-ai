"""Compression and rate limiting configuration."""

from pydantic import BaseModel, Field


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
