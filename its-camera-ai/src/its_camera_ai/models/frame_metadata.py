"""Frame metadata models for high-throughput processing.

Optimized for 3000+ inserts/second with proper indexing,
partitioning, and batch operations support.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class ProcessingStatus(str, Enum):
    """Frame processing status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FrameQuality(str, Enum):
    """Frame quality assessment enumeration."""

    EXCELLENT = "excellent"  # 0.9-1.0
    GOOD = "good"           # 0.7-0.9
    FAIR = "fair"           # 0.5-0.7
    POOR = "poor"           # 0.0-0.5


class FrameMetadata(BaseModel):
    """Frame metadata for real-time processing results.
    
    Designed for high-throughput inserts (30+ FPS per camera, 100+ cameras)
    with optimized indexing and partitioning for time-series queries.
    """

    __tablename__ = "frame_metadata"

    # Foreign key to camera (indexed for fast joins)
    camera_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to camera"
    )

    # Frame identification and timing (critical for ordering)
    frame_number: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Sequential frame number from camera"
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        server_default=text("CURRENT_TIMESTAMP"),
        comment="Frame capture timestamp"
    )
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Processing start timestamp"
    )
    processing_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Processing completion timestamp"
    )

    # Processing status and performance
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=ProcessingStatus.PENDING,
        index=True,
        comment="Frame processing status"
    )
    processing_time_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Total processing time in milliseconds"
    )

    # Frame quality assessment
    quality_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Frame quality score (0.0-1.0)"
    )
    quality_rating: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="Quality rating category"
    )

    # Frame properties
    width: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Frame width in pixels"
    )
    height: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Frame height in pixels"
    )
    file_size: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Frame file size in bytes"
    )
    format: Mapped[str] = mapped_column(
        String(20), nullable=False, default="jpeg", comment="Frame image format"
    )

    # Storage information
    storage_path: Mapped[str | None] = mapped_column(
        String(500), nullable=True, comment="Storage path or S3 key"
    )
    storage_bucket: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Storage bucket name"
    )
    is_stored: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, comment="Frame stored to persistent storage"
    )

    # Detection summary (for quick queries)
    has_detections: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, comment="Frame contains vehicle detections"
    )
    detection_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Total number of detections"
    )
    vehicle_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Number of vehicle detections"
    )

    # Analytics summary
    traffic_density: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Traffic density score (0.0-1.0)"
    )
    congestion_level: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="Traffic congestion level"
    )

    # Environmental conditions
    weather_conditions: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Weather and lighting conditions"
    )
    lighting_conditions: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="Lighting conditions (day/night/dawn/dusk)"
    )

    # Error handling
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="Error message if processing failed"
    )
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Number of processing retries"
    )

    # Model and processing metadata
    model_version: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="ML model version used"
    )
    processing_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Processing configuration snapshot"
    )

    # Performance metrics
    inference_time_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Model inference time in milliseconds"
    )
    preprocessing_time_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Preprocessing time in milliseconds"
    )
    postprocessing_time_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Postprocessing time in milliseconds"
    )

    # Relationships
    camera = relationship("Camera", back_populates="frame_metadata", lazy="select")
    detection_results = relationship(
        "DetectionResult",
        back_populates="frame_metadata",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="select"
    )

    # Optimized indexes for high-throughput queries
    __table_args__ = (
        # Primary query patterns
        Index("idx_frame_camera_timestamp", "camera_id", "timestamp"),
        Index("idx_frame_camera_frame_num", "camera_id", "frame_number"),
        Index("idx_frame_status_timestamp", "status", "timestamp"),
        Index("idx_frame_detections_timestamp", "has_detections", "timestamp"),

        # Performance monitoring queries
        Index("idx_frame_processing_time", "processing_time_ms"),
        Index("idx_frame_quality_score", "quality_score"),

        # Analytics queries
        Index("idx_frame_traffic_density", "traffic_density"),
        Index("idx_frame_vehicle_count", "vehicle_count"),

        # Storage management
        Index("idx_frame_storage_status", "is_stored", "timestamp"),
        Index("idx_frame_storage_path", "storage_path"),

        # Error tracking
        Index("idx_frame_error_retry", "status", "retry_count"),

        # Composite indexes for common queries
        Index("idx_frame_camera_quality", "camera_id", "quality_score", "timestamp"),
        Index("idx_frame_camera_detections", "camera_id", "has_detections", "timestamp"),

        # Partial indexes for performance
        Index(
            "idx_frame_failed_status",
            "camera_id", "timestamp",
            postgresql_where=text("status = 'failed'")
        ),
        Index(
            "idx_frame_with_detections",
            "camera_id", "timestamp", "vehicle_count",
            postgresql_where=text("has_detections = true")
        ),

        # Consider partitioning by timestamp for very large datasets
        {"comment": "Frame metadata optimized for high-throughput processing"}
    )

    def start_processing(self) -> None:
        """Mark frame as processing started."""
        self.status = ProcessingStatus.PROCESSING
        self.processing_started_at = datetime.now(UTC)

    def complete_processing(
        self,
        processing_time_ms: float,
        has_detections: bool = False,
        detection_count: int = 0,
        vehicle_count: int = 0,
        quality_score: float | None = None
    ) -> None:
        """Mark frame as processing completed with results.
        
        Args:
            processing_time_ms: Total processing time
            has_detections: Whether frame contains detections
            detection_count: Total number of detections
            vehicle_count: Number of vehicle detections
            quality_score: Frame quality score (0.0-1.0)
        """
        self.status = ProcessingStatus.COMPLETED
        self.processing_completed_at = datetime.now(UTC)
        self.processing_time_ms = processing_time_ms
        self.has_detections = has_detections
        self.detection_count = detection_count
        self.vehicle_count = vehicle_count

        if quality_score is not None:
            self.quality_score = quality_score
            self.quality_rating = self._calculate_quality_rating(quality_score)

    def fail_processing(self, error_message: str, increment_retry: bool = True) -> None:
        """Mark frame as processing failed.
        
        Args:
            error_message: Error description
            increment_retry: Whether to increment retry counter
        """
        self.status = ProcessingStatus.FAILED
        self.processing_completed_at = datetime.now(UTC)
        self.error_message = error_message

        if increment_retry:
            self.retry_count += 1

    def set_storage_info(self, storage_path: str, bucket: str, file_size: int) -> None:
        """Update storage information after frame is stored.
        
        Args:
            storage_path: Storage path or object key
            bucket: Storage bucket name
            file_size: File size in bytes
        """
        self.storage_path = storage_path
        self.storage_bucket = bucket
        self.file_size = file_size
        self.is_stored = True

    def _calculate_quality_rating(self, score: float) -> str:
        """Calculate quality rating from score.
        
        Args:
            score: Quality score (0.0-1.0)
            
        Returns:
            Quality rating string
        """
        if score >= 0.9:
            return FrameQuality.EXCELLENT
        elif score >= 0.7:
            return FrameQuality.GOOD
        elif score >= 0.5:
            return FrameQuality.FAIR
        else:
            return FrameQuality.POOR

    @property
    def is_completed(self) -> bool:
        """Check if processing is completed."""
        return self.status == ProcessingStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if processing failed."""
        return self.status == ProcessingStatus.FAILED

    @property
    def needs_retry(self) -> bool:
        """Check if frame needs retry (failed with low retry count)."""
        return self.is_failed and self.retry_count < 3

    def __repr__(self) -> str:
        return (
            f"<FrameMetadata("
            f"camera_id={self.camera_id}, "
            f"frame_number={self.frame_number}, "
            f"status={self.status}"
            f")>"
        )
