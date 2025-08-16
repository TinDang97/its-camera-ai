"""Database configuration and model definitions.

Defines Pydantic models for database configurations, connection settings,
and response models for database operations.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class DatabaseType(str, Enum):
    """Database type enumeration."""

    POSTGRESQL = "postgresql"
    TIMESCALE = "timescale"
    REDIS = "redis"


class ConnectionStatus(str, Enum):
    """Connection status enumeration."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"


class ReplicationRole(str, Enum):
    """Database replication role."""

    PRIMARY = "primary"
    REPLICA = "replica"
    STANDALONE = "standalone"


class ConnectionPoolConfig(BaseModel):
    """Connection pool configuration."""

    min_connections: int = Field(default=5, ge=1, description="Minimum pool connections")
    max_connections: int = Field(default=20, ge=1, description="Maximum pool connections")
    max_overflow: int = Field(default=10, ge=0, description="Maximum pool overflow")
    pool_timeout: int = Field(default=30, ge=1, description="Pool connection timeout (seconds)")
    pool_recycle: int = Field(default=3600, ge=0, description="Pool recycle time (seconds)")
    pool_pre_ping: bool = Field(default=True, description="Enable connection pre-ping")

    @validator("max_connections")
    def validate_max_connections(cls, v: int, values: dict[str, Any]) -> int:
        """Ensure max_connections >= min_connections."""
        min_connections = values.get("min_connections", 5)
        if v < min_connections:
            raise ValueError("max_connections must be >= min_connections")
        return v


class DatabaseConfig(BaseModel):
    """PostgreSQL database configuration."""

    # Connection settings
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    database: str = Field(description="Database name")
    username: str = Field(description="Database username")
    password: str = Field(description="Database password", repr=False)

    # SSL configuration
    ssl_mode: str = Field(default="prefer", description="SSL mode")
    ssl_cert_path: str | None = Field(default=None, description="SSL certificate path")
    ssl_key_path: str | None = Field(default=None, description="SSL key path")
    ssl_ca_path: str | None = Field(default=None, description="SSL CA path")

    # Connection pooling
    pool_config: ConnectionPoolConfig = Field(default_factory=ConnectionPoolConfig)

    # Query settings
    query_timeout: int = Field(default=30, ge=1, description="Query timeout (seconds)")
    command_timeout: int = Field(default=60, ge=1, description="Command timeout (seconds)")
    statement_cache_size: int = Field(default=1024, ge=0, description="Statement cache size")

    # Replication
    replication_role: ReplicationRole = Field(default=ReplicationRole.STANDALONE)
    read_only: bool = Field(default=False, description="Read-only connection")

    # Retry configuration
    max_retries: int = Field(default=3, ge=0, description="Maximum connection retries")
    retry_delay: float = Field(default=1.0, ge=0, description="Retry delay (seconds)")
    retry_exponential_base: float = Field(default=2.0, ge=1, description="Exponential backoff base")

    def get_connection_url(self, async_driver: bool = True) -> str:
        """Generate database connection URL."""
        driver = "postgresql+asyncpg" if async_driver else "postgresql+psycopg2"
        return (
            f"{driver}://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )

    def get_connection_params(self) -> dict[str, Any]:
        """Get connection parameters for asyncpg."""
        params = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.username,
            "password": self.password,
            "command_timeout": self.command_timeout,
            "statement_cache_size": self.statement_cache_size,
        }

        # Add SSL configuration if provided
        if self.ssl_mode != "disable":
            params["ssl"] = self.ssl_mode
            if self.ssl_cert_path:
                params["ssl_cert"] = self.ssl_cert_path
            if self.ssl_key_path:
                params["ssl_key"] = self.ssl_key_path
            if self.ssl_ca_path:
                params["ssl_ca"] = self.ssl_ca_path

        return params


class TimescaleConfig(DatabaseConfig):
    """TimescaleDB-specific configuration."""

    # TimescaleDB-specific settings
    enable_compression: bool = Field(default=True, description="Enable compression")
    chunk_time_interval: str = Field(default="1 day", description="Chunk time interval")
    retention_policy: str | None = Field(default=None, description="Data retention policy")

    # Continuous aggregates
    enable_continuous_aggregates: bool = Field(default=True, description="Enable continuous aggregates")
    materialized_view_refresh_policy: str = Field(
        default="1 hour", description="Materialized view refresh interval"
    )

    # Batch operations
    batch_size: int = Field(default=1000, ge=1, description="Batch insert size")
    batch_timeout: int = Field(default=5, ge=1, description="Batch timeout (seconds)")

    # Hypertable management
    auto_create_hypertables: bool = Field(default=True, description="Auto-create hypertables")
    default_partition_key: str = Field(default="timestamp", description="Default partition key")


class RedisConfig(BaseModel):
    """Redis configuration."""

    # Connection settings
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    database: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: str | None = Field(default=None, description="Redis password", repr=False)

    # SSL configuration
    ssl: bool = Field(default=False, description="Use SSL connection")
    ssl_cert_reqs: str = Field(default="required", description="SSL certificate requirements")
    ssl_cert_path: str | None = Field(default=None, description="SSL certificate path")
    ssl_key_path: str | None = Field(default=None, description="SSL key path")
    ssl_ca_path: str | None = Field(default=None, description="SSL CA path")

    # Connection pooling
    max_connections: int = Field(default=50, ge=1, description="Maximum pool connections")
    connection_timeout: int = Field(default=5, ge=1, description="Connection timeout (seconds)")
    socket_timeout: int = Field(default=5, ge=1, description="Socket timeout (seconds)")
    socket_keepalive: bool = Field(default=True, description="Enable socket keepalive")
    socket_keepalive_options: dict[str, int] = Field(
        default_factory=lambda: {"TCP_KEEPINTVL": 1, "TCP_KEEPCNT": 3, "TCP_KEEPIDLE": 1}
    )

    # Retry configuration
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    retry_on_error: list[str] = Field(
        default_factory=lambda: ["ConnectionError", "TimeoutError"],
        description="Exception types to retry on"
    )
    max_retries: int = Field(default=3, ge=0, description="Maximum retries")
    retry_delay: float = Field(default=0.5, ge=0, description="Retry delay (seconds)")

    # Cluster configuration
    cluster_mode: bool = Field(default=False, description="Enable cluster mode")
    cluster_nodes: list[str] = Field(default_factory=list, description="Cluster node addresses")
    cluster_require_full_coverage: bool = Field(
        default=True, description="Require full cluster coverage"
    )

    # Sentinel configuration
    sentinel_mode: bool = Field(default=False, description="Enable Sentinel mode")
    sentinel_hosts: list[tuple[str, int]] = Field(
        default_factory=list, description="Sentinel host/port pairs"
    )
    sentinel_service_name: str = Field(default="mymaster", description="Sentinel service name")

    # Stream configuration
    stream_maxlen: int = Field(default=10000, ge=1, description="Maximum stream length")
    stream_approximate: bool = Field(default=True, description="Use approximate stream maxlen")
    consumer_group_timeout: int = Field(default=30000, ge=1, description="Consumer group timeout (ms)")

    # Pub/Sub configuration
    pubsub_timeout: int = Field(default=1, ge=1, description="Pub/Sub timeout (seconds)")
    pubsub_ignore_subscribe_messages: bool = Field(
        default=True, description="Ignore subscribe messages in Pub/Sub"
    )

    def get_connection_url(self) -> str:
        """Generate Redis connection URL."""
        auth = f":{self.password}@" if self.password else ""
        ssl_scheme = "rediss" if self.ssl else "redis"
        return f"{ssl_scheme}://{auth}{self.host}:{self.port}/{self.database}"

    def get_connection_params(self) -> dict[str, Any]:
        """Get connection parameters for Redis client."""
        params = {
            "host": self.host,
            "port": self.port,
            "db": self.database,
            "password": self.password,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.connection_timeout,
            "socket_keepalive": self.socket_keepalive,
            "socket_keepalive_options": self.socket_keepalive_options,
            "retry_on_timeout": self.retry_on_timeout,
            "max_connections": self.max_connections,
        }

        if self.ssl:
            params.update({
                "ssl": self.ssl,
                "ssl_cert_reqs": self.ssl_cert_reqs,
                "ssl_certfile": self.ssl_cert_path,
                "ssl_keyfile": self.ssl_key_path,
                "ssl_ca_certs": self.ssl_ca_path,
            })

        return {k: v for k, v in params.items() if v is not None}


class ConnectionHealth(BaseModel):
    """Connection health status."""

    status: ConnectionStatus = Field(description="Connection status")
    database_type: DatabaseType = Field(description="Database type")
    host: str = Field(description="Database host")
    port: int = Field(description="Database port")
    database: str | None = Field(default=None, description="Database name")

    # Health metrics
    response_time_ms: float | None = Field(default=None, description="Response time (ms)")
    active_connections: int | None = Field(default=None, description="Active connections")
    total_connections: int | None = Field(default=None, description="Total connections")

    # Error information
    error_message: str | None = Field(default=None, description="Error message if unhealthy")
    last_check: datetime = Field(default_factory=datetime.utcnow, description="Last health check")

    # Additional metadata
    version: str | None = Field(default=None, description="Database version")
    replication_role: ReplicationRole | None = Field(default=None, description="Replication role")
    uptime_seconds: int | None = Field(default=None, description="Uptime in seconds")


class QueryResult(BaseModel):
    """Database query result."""

    success: bool = Field(description="Query success status")
    rows_affected: int | None = Field(default=None, description="Number of rows affected")
    execution_time_ms: float | None = Field(default=None, description="Execution time (ms)")

    # Result data
    data: list[dict[str, Any]] | dict[str, Any] | None = Field(
        default=None, description="Query result data"
    )

    # Error information
    error_message: str | None = Field(default=None, description="Error message if failed")
    error_code: str | None = Field(default=None, description="Database error code")

    # Query metadata
    query_hash: str | None = Field(default=None, description="Query hash for caching")
    from_cache: bool = Field(default=False, description="Result retrieved from cache")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Query timestamp")


class BatchOperationResult(BaseModel):
    """Batch operation result."""

    total_items: int = Field(description="Total items processed")
    successful_items: int = Field(description="Successfully processed items")
    failed_items: int = Field(description="Failed items")

    # Timing
    total_time_ms: float = Field(description="Total processing time (ms)")
    average_time_per_item_ms: float = Field(description="Average time per item (ms)")

    # Error details
    errors: list[dict[str, Any]] = Field(default_factory=list, description="Error details")

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        return (self.successful_items / self.total_items) * 100 if self.total_items > 0 else 0.0

    @property
    def has_failures(self) -> bool:
        """Check if there were any failures."""
        return self.failed_items > 0

