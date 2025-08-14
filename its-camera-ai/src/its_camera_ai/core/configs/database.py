"""Database configuration settings."""

from pydantic import BaseModel, Field


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
