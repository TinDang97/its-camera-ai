"""System metrics models for performance monitoring.

Tracks system performance, resource utilization, and operational
metrics for monitoring and alerting integration.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import BaseModel


class MetricType(str, Enum):
    """System metric type enumeration."""

    # Performance metrics
    PROCESSING_TIME = "processing_time"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    QUEUE_LENGTH = "queue_length"

    # Resource utilization
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    GPU_USAGE = "gpu_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"

    # System health
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    UPTIME = "uptime"

    # Camera-specific metrics
    FRAME_RATE = "frame_rate"
    DETECTION_RATE = "detection_rate"
    STREAM_QUALITY = "stream_quality"

    # Storage metrics
    STORAGE_USED = "storage_used"
    STORAGE_AVAILABLE = "storage_available"

    # Custom metrics
    CUSTOM = "custom"


class MetricUnit(str, Enum):
    """Metric unit enumeration."""

    # Time units
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"

    # Rate units
    PER_SECOND = "per_second"
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"

    # Size units
    BYTES = "bytes"
    KILOBYTES = "kilobytes"
    MEGABYTES = "megabytes"
    GIGABYTES = "gigabytes"

    # Percentage
    PERCENT = "percent"
    RATIO = "ratio"

    # Count
    COUNT = "count"
    FRAMES = "frames"
    DETECTIONS = "detections"

    # Dimensionless
    NONE = "none"


class SystemMetrics(BaseModel):
    """System performance and operational metrics.
    
    Designed for time-series data storage with efficient querying
    for monitoring dashboards and alerting systems.
    """

    __tablename__ = "system_metrics"

    # Metric identification
    metric_name: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, comment="Metric name identifier"
    )
    metric_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True, comment="Metric type category"
    )
    metric_unit: Mapped[str] = mapped_column(
        String(20), nullable=False, comment="Metric unit of measurement"
    )

    # Metric value and timestamp
    value: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Metric value"
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="Metric collection timestamp"
    )

    # Source information
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="Source type (camera/system/process)"
    )
    source_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True, comment="Source identifier"
    )
    hostname: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Server hostname"
    )
    service_name: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Service name"
    )

    # Statistical aggregation (for pre-computed rollups)
    aggregation_period: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="Aggregation period (1m/5m/1h/1d)"
    )
    sample_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Number of samples in aggregation"
    )
    min_value: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Minimum value in period"
    )
    max_value: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Maximum value in period"
    )
    avg_value: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Average value in period"
    )
    sum_value: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Sum of values in period"
    )

    # Metadata and context
    labels: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Additional metric labels/tags"
    )
    context: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Additional context information"
    )

    # Alert thresholds (for reference)
    warning_threshold: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Warning threshold value"
    )
    critical_threshold: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Critical threshold value"
    )

    # Optimized indexes for time-series queries
    __table_args__ = (
        # Primary time-series query patterns
        Index("idx_metrics_name_timestamp", "metric_name", "timestamp"),
        Index("idx_metrics_type_timestamp", "metric_type", "timestamp"),
        Index("idx_metrics_source_timestamp", "source_type", "source_id", "timestamp"),

        # Source-specific queries
        Index("idx_metrics_hostname_timestamp", "hostname", "timestamp"),
        Index("idx_metrics_service_timestamp", "service_name", "timestamp"),

        # Aggregation queries
        Index("idx_metrics_aggregation", "metric_name", "aggregation_period", "timestamp"),

        # Value-based queries (for alerting)
        Index("idx_metrics_value_thresholds", "metric_name", "value", "timestamp"),

        # Performance monitoring specific indexes
        Index(
            "idx_metrics_performance",
            "metric_type", "source_id", "timestamp",
            postgresql_where=text(
                "metric_type IN ('processing_time', 'throughput', 'latency', 'frame_rate')"
            )
        ),

        # Resource utilization queries
        Index(
            "idx_metrics_resources",
            "metric_type", "hostname", "timestamp",
            postgresql_where=text(
                "metric_type IN ('cpu_usage', 'memory_usage', 'gpu_usage', 'disk_usage')"
            )
        ),

        # Recent metrics (last 24 hours)
        Index(
            "idx_metrics_recent",
            "metric_name", "timestamp", "value",
            postgresql_where=text("timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'")
        ),

        # GIN index for labels/tags queries
        Index("idx_metrics_labels_gin", "labels", postgresql_using="gin"),

        # Consider partitioning by timestamp for very large datasets
        {"comment": "System metrics optimized for time-series monitoring"}
    )

    def set_thresholds(self, warning: float, critical: float) -> None:
        """Set alert thresholds for the metric.
        
        Args:
            warning: Warning threshold value
            critical: Critical threshold value
        """
        self.warning_threshold = warning
        self.critical_threshold = critical

    def add_label(self, key: str, value: str) -> None:
        """Add a label to the metric.
        
        Args:
            key: Label key
            value: Label value
        """
        if self.labels is None:
            self.labels = {}
        self.labels[key] = value

    def add_context(self, key: str, value: Any) -> None:
        """Add context information to the metric.
        
        Args:
            key: Context key
            value: Context value
        """
        if self.context is None:
            self.context = {}
        self.context[key] = value

    @property
    def is_warning(self) -> bool:
        """Check if metric value exceeds warning threshold."""
        return (
            self.warning_threshold is not None and
            self.value >= self.warning_threshold
        )

    @property
    def is_critical(self) -> bool:
        """Check if metric value exceeds critical threshold."""
        return (
            self.critical_threshold is not None and
            self.value >= self.critical_threshold
        )

    @property
    def alert_level(self) -> str | None:
        """Get alert level based on thresholds."""
        if self.is_critical:
            return "critical"
        elif self.is_warning:
            return "warning"
        return None

    @classmethod
    def create_performance_metric(
        cls,
        name: str,
        value: float,
        source_id: str | None = None,
        unit: MetricUnit = MetricUnit.MILLISECONDS,
        labels: dict[str, str] | None = None
    ) -> "SystemMetrics":
        """Create a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            source_id: Source identifier
            unit: Metric unit
            labels: Additional labels
            
        Returns:
            SystemMetrics instance
        """
        metric = cls(
            metric_name=name,
            metric_type=MetricType.PROCESSING_TIME,
            metric_unit=unit.value,
            value=value,
            source_type="camera",
            source_id=source_id,
            timestamp=datetime.now(UTC)
        )

        if labels:
            metric.labels = labels

        return metric

    @classmethod
    def create_resource_metric(
        cls,
        metric_type: MetricType,
        value: float,
        hostname: str,
        unit: MetricUnit = MetricUnit.PERCENT
    ) -> "SystemMetrics":
        """Create a resource utilization metric.
        
        Args:
            metric_type: Resource metric type
            value: Resource usage value
            hostname: Server hostname
            unit: Metric unit
            
        Returns:
            SystemMetrics instance
        """
        return cls(
            metric_name=metric_type.value,
            metric_type=metric_type.value,
            metric_unit=unit.value,
            value=value,
            source_type="system",
            hostname=hostname,
            timestamp=datetime.now(UTC)
        )

    def __repr__(self) -> str:
        return (
            f"<SystemMetrics("
            f"name={self.metric_name}, "
            f"value={self.value}, "
            f"timestamp={self.timestamp}"
            f")>"
        )


class AggregatedMetrics(BaseModel):
    """Pre-computed aggregated metrics for faster dashboard queries.
    
    Stores hourly, daily, and weekly rollups to avoid expensive
    real-time aggregations on large datasets.
    """

    __tablename__ = "aggregated_metrics"

    # Metric identification
    metric_name: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True, comment="Base metric name"
    )
    metric_type: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="Base metric type"
    )

    # Aggregation configuration
    aggregation_period: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True, comment="Aggregation period (1h/1d/1w)"
    )
    period_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True, comment="Period start time"
    )
    period_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, comment="Period end time"
    )

    # Source aggregation
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False, comment="Source type"
    )
    source_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Source identifier"
    )

    # Aggregated values
    sample_count: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Number of samples"
    )
    min_value: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Minimum value"
    )
    max_value: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Maximum value"
    )
    avg_value: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Average value"
    )
    sum_value: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Sum of values"
    )
    std_deviation: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Standard deviation"
    )

    # Percentiles for performance metrics
    p50_value: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="50th percentile value"
    )
    p95_value: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="95th percentile value"
    )
    p99_value: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="99th percentile value"
    )

    # Indexes for aggregated queries
    __table_args__ = (
        Index("idx_agg_metrics_name_period", "metric_name", "aggregation_period", "period_start"),
        Index("idx_agg_metrics_source_period", "source_type", "source_id", "period_start"),
        Index("idx_agg_metrics_period_range", "aggregation_period", "period_start", "period_end"),
        {"comment": "Pre-computed metric aggregations for dashboard queries"}
    )

    def __repr__(self) -> str:
        return (
            f"<AggregatedMetrics("
            f"name={self.metric_name}, "
            f"period={self.aggregation_period}, "
            f"avg={self.avg_value:.2f}"
            f")>"
        )
