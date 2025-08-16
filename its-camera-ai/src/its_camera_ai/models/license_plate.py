"""
License Plate Recognition database models.

Extended models for license plate detection tracking, watchlist management,
and analytics with optimized queries for high-throughput operations.
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
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseTableModel


class PlateRegion(str, Enum):
    """License plate regions for format validation."""

    US = "us"
    EU = "eu"
    ASIA = "asia"
    CA = "ca"  # Canada
    MX = "mx"  # Mexico
    BR = "br"  # Brazil
    AU = "au"  # Australia
    UK = "uk"  # United Kingdom
    AUTO = "auto"


class AlertType(str, Enum):
    """Types of license plate alerts."""

    STOLEN = "stolen"
    WANTED = "wanted"
    AMBER_ALERT = "amber_alert"
    BOLO = "bolo"  # Be On the Lookout
    TOLL_VIOLATION = "toll_violation"
    PARKING_VIOLATION = "parking_violation"
    EXPIRED_REGISTRATION = "expired_registration"
    UNINSURED = "uninsured"
    CUSTOM = "custom"


class AlertPriority(str, Enum):
    """Priority levels for license plate alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PlateDetection(BaseTableModel):
    """Extended license plate detection tracking.
    
    Comprehensive tracking of license plate detections with
    confidence scoring, quality metrics, and validation status.
    """

    # Reference to original detection result
    detection_result_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("detection_result.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to detection result"
    )

    # Camera and location info
    camera_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("camera.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Camera that captured the plate"
    )

    # License plate information
    plate_text: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Recognized license plate text"
    )
    plate_text_normalized: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Normalized plate text for matching"
    )
    plate_region: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default=PlateRegion.AUTO.value,
        comment="License plate region/format"
    )

    # Confidence and quality metrics
    overall_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Overall detection confidence (0.0-1.0)"
    )
    ocr_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="OCR recognition confidence (0.0-1.0)"
    )
    plate_quality_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Image quality score (0.0-1.0)"
    )

    # Character-level confidence
    character_confidences: Mapped[list[float]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Per-character confidence scores"
    )

    # Bounding box coordinates (relative to original image)
    vehicle_bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    vehicle_bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    vehicle_bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    vehicle_bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)

    plate_bbox_x1: Mapped[float] = mapped_column(Float, nullable=False)
    plate_bbox_y1: Mapped[float] = mapped_column(Float, nullable=False)
    plate_bbox_x2: Mapped[float] = mapped_column(Float, nullable=False)
    plate_bbox_y2: Mapped[float] = mapped_column(Float, nullable=False)

    # Processing information
    processing_time_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Total processing time in milliseconds"
    )
    ocr_time_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="OCR processing time in milliseconds"
    )
    detection_time_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Plate detection time in milliseconds"
    )
    engine_used: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="OCR engine used for recognition"
    )

    # Validation and verification
    is_validated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Has been manually validated"
    )
    is_false_positive: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Marked as false positive"
    )
    validation_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        comment="Human validation confidence"
    )
    validated_by: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="User who validated the detection"
    )
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Validation timestamp"
    )

    # Alert information
    triggered_alerts: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=True,
        comment="List of alert IDs triggered by this detection"
    )
    alert_notified: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether alerts have been sent"
    )

    # Additional metadata
    additional_data: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional detection metadata"
    )

    # Relationships
    detection_result = relationship("DetectionResult", back_populates="plate_detections")
    camera = relationship("Camera", back_populates="plate_detections")

    # Optimized indexes for high-performance queries
    __table_args__ = (
        # Primary lookup patterns
        Index("idx_plate_text", "plate_text_normalized"),
        Index("idx_plate_camera_time", "camera_id", "created_at"),
        Index("idx_plate_confidence", "overall_confidence"),

        # Quality and validation filters
        Index("idx_plate_quality", "plate_quality_score"),
        Index("idx_plate_validated", "is_validated", "is_false_positive"),
        Index("idx_plate_alerts", "alert_notified"),

        # Region-based queries
        Index("idx_plate_region", "plate_region"),

        # Performance monitoring
        Index("idx_plate_processing_time", "processing_time_ms"),
        Index("idx_plate_engine", "engine_used"),

        # Composite indexes for common queries
        Index("idx_plate_text_time", "plate_text_normalized", "created_at"),
        Index("idx_plate_camera_confidence", "camera_id", "overall_confidence"),

        # High-confidence detections
        Index(
            "idx_plate_high_confidence",
            "plate_text_normalized",
            "camera_id",
            "created_at",
            postgresql_where=text("overall_confidence >= 0.8 AND is_false_positive = false")
        ),

        # Alert-worthy detections
        Index(
            "idx_plate_alerts_pending",
            "plate_text_normalized",
            "created_at",
            postgresql_where=text("alert_notified = false")
        ),

        {"comment": "License plate detections with optimized indexes"}
    )

    def normalize_plate_text(self) -> None:
        """Normalize plate text for consistent matching."""
        if self.plate_text:
            # Remove spaces, hyphens, and convert to uppercase
            normalized = self.plate_text.upper().replace(" ", "").replace("-", "")
            self.plate_text_normalized = normalized

    def set_vehicle_bbox(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Set vehicle bounding box coordinates."""
        self.vehicle_bbox_x1 = x1
        self.vehicle_bbox_y1 = y1
        self.vehicle_bbox_x2 = x2
        self.vehicle_bbox_y2 = y2

    def set_plate_bbox(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Set plate bounding box coordinates."""
        self.plate_bbox_x1 = x1
        self.plate_bbox_y1 = y1
        self.plate_bbox_x2 = x2
        self.plate_bbox_y2 = y2

    def validate_detection(self, user_id: str, confidence: float = 1.0) -> None:
        """Mark detection as validated by human."""
        self.is_validated = True
        self.validation_confidence = confidence
        self.validated_by = user_id
        self.validated_at = datetime.now(UTC)

    def mark_false_positive(self, user_id: str, reason: str = None) -> None:
        """Mark detection as false positive."""
        self.is_false_positive = True
        self.validated_by = user_id
        self.validated_at = datetime.now(UTC)

        if reason:
            if not self.additional_data:
                self.additional_data = {}
            self.additional_data["false_positive_reason"] = reason

    @property
    def is_high_confidence(self) -> bool:
        """Check if detection has high confidence."""
        return (
            self.overall_confidence >= 0.8 and
            self.ocr_confidence >= 0.8 and
            self.plate_quality_score >= 0.7
        )

    @property
    def is_reliable(self) -> bool:
        """Check if detection is reliable."""
        return (
            self.is_high_confidence and
            not self.is_false_positive and
            (self.is_validated or self.overall_confidence >= 0.9)
        )


class PlateWatchlist(BaseTableModel):
    """License plate watchlist for alerts and monitoring."""

    # Plate information
    plate_number: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="License plate number to watch"
    )
    plate_number_normalized: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Normalized plate number for matching"
    )
    plate_region: Mapped[str] = mapped_column(
        String(10),
        nullable=True,
        comment="Expected plate region"
    )

    # Alert configuration
    alert_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=AlertType.CUSTOM.value,
        comment="Type of alert"
    )
    alert_priority: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=AlertPriority.MEDIUM.value,
        comment="Alert priority level"
    )

    # Status and validity
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether alert is active"
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Alert expiration time"
    )

    # Description and metadata
    description: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        comment="Alert description"
    )
    notes: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
        comment="Additional notes"
    )

    # Contact and notification
    contact_info: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Contact information for alerts"
    )
    notification_channels: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Notification channels (email, sms, webhook)"
    )

    # Owner information
    owner_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        comment="Vehicle owner name"
    )
    owner_info: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional owner information"
    )

    # Agency and jurisdiction
    agency: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Law enforcement agency"
    )
    jurisdiction: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Legal jurisdiction"
    )
    case_number: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="Associated case number"
    )

    # Statistics
    total_detections: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of detections"
    )
    last_detected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last detection timestamp"
    )
    last_detected_camera_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("camera.id", ondelete="SET NULL"),
        nullable=True,
        comment="Last camera that detected this plate"
    )

    # Audit trail
    created_by: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="User who created the watchlist entry"
    )
    updated_by: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        comment="User who last updated the entry"
    )

    # Relationships
    last_detected_camera = relationship("Camera", foreign_keys=[last_detected_camera_id])

    # Indexes for fast lookups
    __table_args__ = (
        # Primary lookup patterns
        Index("idx_watchlist_plate", "plate_number_normalized"),
        Index("idx_watchlist_active", "is_active", "expires_at"),
        Index("idx_watchlist_type", "alert_type"),
        Index("idx_watchlist_priority", "alert_priority"),

        # Expiration and maintenance
        Index("idx_watchlist_expires", "expires_at"),
        Index("idx_watchlist_agency", "agency"),

        # Statistics and monitoring
        Index("idx_watchlist_detections", "total_detections"),
        Index("idx_watchlist_last_seen", "last_detected_at"),

        # Composite indexes for complex queries
        Index("idx_watchlist_active_priority", "is_active", "alert_priority", "alert_type"),

        # Active watchlist entries
        Index(
            "idx_watchlist_active_entries",
            "plate_number_normalized",
            "alert_priority",
            postgresql_where=text("is_active = true AND (expires_at IS NULL OR expires_at > NOW())")
        ),

        # Ensure unique active plates
        UniqueConstraint(
            "plate_number_normalized",
            "alert_type",
            name="uq_watchlist_plate_type"
        ),

        {"comment": "License plate watchlist with alert management"}
    )

    def normalize_plate_number(self) -> None:
        """Normalize plate number for consistent matching."""
        if self.plate_number:
            normalized = self.plate_number.upper().replace(" ", "").replace("-", "")
            self.plate_number_normalized = normalized

    def is_expired(self) -> bool:
        """Check if watchlist entry is expired."""
        if not self.expires_at:
            return False
        return datetime.now(UTC) > self.expires_at

    def increment_detections(self, camera_id: str) -> None:
        """Increment detection count and update last seen info."""
        self.total_detections += 1
        self.last_detected_at = datetime.now(UTC)
        self.last_detected_camera_id = camera_id

    def deactivate(self, reason: str = None) -> None:
        """Deactivate the watchlist entry."""
        self.is_active = False
        if reason and not hasattr(self, 'additional_data'):
            # Store deactivation reason
            if not self.notes:
                self.notes = f"Deactivated: {reason}"
            else:
                self.notes += f"\nDeactivated: {reason}"


class PlateAnalytics(BaseTableModel):
    """Aggregated license plate analytics for reporting."""

    # Time and location dimensions
    camera_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("camera.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Camera ID"
    )
    date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="Analytics date (daily aggregation)"
    )
    hour: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Hour of day (0-23)"
    )

    # Detection metrics
    total_vehicles: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total vehicles detected"
    )
    total_plate_detections: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Total plate detections attempted"
    )
    successful_plate_reads: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Successful plate recognitions"
    )
    unique_plates: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of unique plates detected"
    )

    # Quality metrics
    avg_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Average detection confidence"
    )
    avg_quality_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Average plate quality score"
    )
    avg_processing_time_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Average processing time"
    )

    # Performance metrics
    success_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="Plate recognition success rate (0.0-1.0)"
    )
    false_positive_rate: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="False positive rate (0.0-1.0)"
    )

    # Alert metrics
    alerts_triggered: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of alerts triggered"
    )
    alerts_by_type: Mapped[dict[str, int]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Alert counts by type"
    )

    # Regional distribution
    plates_by_region: Mapped[dict[str, int]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Plate counts by region"
    )

    # Engine performance
    engines_used: Mapped[dict[str, int]] = mapped_column(
        JSONB,
        nullable=True,
        comment="OCR engine usage statistics"
    )

    # Additional metrics
    additional_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional analytics data"
    )

    # Relationships
    camera = relationship("Camera", back_populates="plate_analytics")

    # Indexes for time-series queries
    __table_args__ = (
        # Time-based queries
        Index("idx_analytics_date", "date"),
        Index("idx_analytics_camera_date", "camera_id", "date"),
        Index("idx_analytics_hour", "hour"),

        # Performance monitoring
        Index("idx_analytics_success_rate", "success_rate"),
        Index("idx_analytics_processing_time", "avg_processing_time_ms"),

        # Composite indexes for reporting
        Index("idx_analytics_camera_hour", "camera_id", "date", "hour"),

        # Ensure unique daily records per camera
        UniqueConstraint(
            "camera_id",
            "date",
            "hour",
            name="uq_analytics_camera_date_hour"
        ),

        {"comment": "License plate analytics aggregated by camera and time"}
    )

    def calculate_success_rate(self) -> None:
        """Calculate and update success rate."""
        if self.total_plate_detections > 0:
            self.success_rate = self.successful_plate_reads / self.total_plate_detections
        else:
            self.success_rate = 0.0

    def update_metrics(
        self,
        vehicle_count: int,
        plate_detections: int,
        successful_reads: int,
        unique_count: int,
        avg_conf: float,
        avg_quality: float,
        avg_time: float
    ) -> None:
        """Update all metrics in one call."""
        self.total_vehicles = vehicle_count
        self.total_plate_detections = plate_detections
        self.successful_plate_reads = successful_reads
        self.unique_plates = unique_count
        self.avg_confidence = avg_conf
        self.avg_quality_score = avg_quality
        self.avg_processing_time_ms = avg_time

        self.calculate_success_rate()


# Add relationships to existing models
def add_plate_relationships():
    """Add relationships to existing models (called during model setup)."""

    # This would be imported and called during model initialization
    # to add the relationships to existing Camera and DetectionResult models

    # For Camera model:
    # plate_detections = relationship("PlateDetection", back_populates="camera")
    # plate_analytics = relationship("PlateAnalytics", back_populates="camera")

    # For DetectionResult model:
    # plate_detections = relationship("PlateDetection", back_populates="detection_result")

    pass
