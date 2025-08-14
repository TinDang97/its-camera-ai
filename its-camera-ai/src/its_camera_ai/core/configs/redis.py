"""Redis configuration settings."""

from pydantic import BaseModel, Field


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
