"""Detection result models for vehicle and object detection.

Optimized for storing bounding boxes, classifications, and tracking
information with efficient spatial and temporal queries.
"""

from datetime import datetime
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
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


class DetectionClass(str, Enum):
    """Vehicle and object detection classes."""

    # Vehicles
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"

    # Emergency vehicles
    AMBULANCE = "ambulance"
    FIRE_TRUCK = "fire_truck"
    POLICE = "police"

    # Other objects
    PERSON = "person"
    TRAFFIC_LIGHT = "traffic_light"
    STOP_SIGN = "stop_sign"
    UNKNOWN = "unknown"


class VehicleType(str, Enum):
    """Specific vehicle type classifications."""

    SEDAN = "sedan"
    SUV = "suv"
    HATCHBACK = "hatchback"
    PICKUP = "pickup"
    VAN = "van"
    TRUCK_SMALL = "truck_small"
    TRUCK_LARGE = "truck_large"
    BUS_CITY = "bus_city"
    BUS_SCHOOL = "bus_school"
    MOTORCYCLE_SPORT = "motorcycle_sport"
    MOTORCYCLE_CRUISER = "motorcycle_cruiser"
    SCOOTER = "scooter"
    BICYCLE_ROAD = "bicycle_road"
    BICYCLE_MOUNTAIN = "bicycle_mountain"
    UNKNOWN = "unknown"


class DetectionResult(BaseModel):
    """Individual detection result within a frame.
    
    Stores bounding box coordinates, classification confidence,
    tracking information, and vehicle-specific attributes.
    """

    __tablename__ = "detection_results"

    # Foreign key to frame metadata
    frame_metadata_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("frame_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to frame metadata"
    )

    # Detection sequence within frame
    detection_id: Mapped[int] = mapped_column(
        Integer, nullable=False, comment="Detection ID within frame"
    )

    # Bounding box coordinates (normalized 0.0-1.0 or absolute pixels)
    bbox_x1: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Bounding box top-left X coordinate"
    )
    bbox_y1: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Bounding box top-left Y coordinate"
    )
    bbox_x2: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Bounding box bottom-right X coordinate"
    )
    bbox_y2: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Bounding box bottom-right Y coordinate"
    )

    # Bounding box properties
    bbox_width: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Bounding box width"
    )
    bbox_height: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Bounding box height"
    )
    bbox_area: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Bounding box area"
    )

    # Detection classification
    class_name: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True, comment="Primary object class"
    )
    class_confidence: Mapped[float] = mapped_column(
        Float, nullable=False, comment="Classification confidence (0.0-1.0)"
    )

    # Vehicle-specific classification
    vehicle_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Specific vehicle type"
    )
    vehicle_confidence: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Vehicle type confidence (0.0-1.0)"
    )

    # Object tracking information
    track_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True, index=True, comment="Object tracking ID"
    )
    track_confidence: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Tracking confidence (0.0-1.0)"
    )

    # Motion and trajectory
    velocity_x: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Velocity in X direction (pixels/frame)"
    )
    velocity_y: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Velocity in Y direction (pixels/frame)"
    )
    velocity_magnitude: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Velocity magnitude"
    )
    direction: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Movement direction in degrees (0-360)"
    )

    # Object attributes
    color_primary: Mapped[str | None] = mapped_column(
        String(30), nullable=True, comment="Primary object color"
    )
    color_secondary: Mapped[str | None] = mapped_column(
        String(30), nullable=True, comment="Secondary object color"
    )
    color_confidence: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Color detection confidence"
    )

    # License plate information (if detected)
    license_plate: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="License plate text"
    )
    license_plate_confidence: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="License plate recognition confidence"
    )
    license_plate_region: Mapped[dict[str, float] | None] = mapped_column(
        JSONB, nullable=True, comment="License plate bounding box coordinates"
    )

    # Vehicle characteristics (estimated)
    estimated_length: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Estimated vehicle length (meters)"
    )
    estimated_width: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Estimated vehicle width (meters)"
    )
    estimated_height: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Estimated vehicle height (meters)"
    )

    # Detection zone information
    detection_zone: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Detection zone name"
    )
    zone_entry_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Zone entry timestamp"
    )
    zone_exit_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="Zone exit timestamp"
    )

    # Quality and reliability metrics
    detection_quality: Mapped[float] = mapped_column(
        Float, nullable=False, default=1.0, comment="Overall detection quality score"
    )
    occlusion_ratio: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Object occlusion ratio (0.0-1.0)"
    )
    blur_score: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Detection blur score (0.0-1.0)"
    )

    # Model and processing information
    model_version: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Model version used for detection"
    )
    processing_time_ms: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Detection processing time in milliseconds"
    )

    # Additional attributes (flexible storage)
    additional_attributes: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Additional detection attributes"
    )

    # Flags for special cases
    is_false_positive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, comment="Marked as false positive"
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, comment="Human verified detection"
    )
    is_anomaly: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, comment="Anomalous detection"
    )

    # Relationships
    frame_metadata = relationship("FrameMetadata", back_populates="detection_results")

    # Optimized indexes for detection queries
    __table_args__ = (
        # Primary query patterns
        Index("idx_detection_frame_id", "frame_metadata_id"),
        Index("idx_detection_class", "class_name"),
        Index("idx_detection_track_id", "track_id"),

        # Confidence and quality filters
        Index("idx_detection_confidence", "class_confidence"),
        Index("idx_detection_quality", "detection_quality"),

        # Spatial queries (bounding box)
        Index("idx_detection_bbox_center", "bbox_x1", "bbox_y1"),
        Index("idx_detection_bbox_area", "bbox_area"),

        # Tracking and motion
        Index("idx_detection_velocity", "velocity_magnitude"),
        Index("idx_detection_direction", "direction"),

        # Vehicle-specific queries
        Index("idx_detection_vehicle_type", "vehicle_type"),
        Index("idx_detection_color", "color_primary"),

        # License plate queries
        Index("idx_detection_license_plate", "license_plate"),

        # Zone-based queries
        Index("idx_detection_zone", "detection_zone"),
        Index("idx_detection_zone_entry", "detection_zone", "zone_entry_time"),

        # Quality flags
        Index("idx_detection_flags", "is_false_positive", "is_verified"),

        # Composite indexes for common queries
        Index("idx_detection_class_confidence", "class_name", "class_confidence"),
        Index("idx_detection_track_time", "track_id", "created_at"),

        # High-performance vehicle detection queries
        Index(
            "idx_detection_vehicles",
            "frame_metadata_id", "class_name", "class_confidence",
            postgresql_where=text(
                "class_name IN ('car', 'truck', 'bus', 'motorcycle', 'bicycle') "
                "AND class_confidence >= 0.5"
            )
        ),

        # License plate detections
        Index(
            "idx_detection_with_plates",
            "frame_metadata_id", "license_plate", "license_plate_confidence",
            postgresql_where=text("license_plate IS NOT NULL")
        ),

        {"comment": "Detection results with spatial and classification indexes"}
    )

    def set_bounding_box(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Set bounding box coordinates and calculate derived properties.
        
        Args:
            x1: Top-left X coordinate
            y1: Top-left Y coordinate  
            x2: Bottom-right X coordinate
            y2: Bottom-right Y coordinate
        """
        self.bbox_x1 = x1
        self.bbox_y1 = y1
        self.bbox_x2 = x2
        self.bbox_y2 = y2
        self.bbox_width = x2 - x1
        self.bbox_height = y2 - y1
        self.bbox_area = self.bbox_width * self.bbox_height

    def set_velocity(self, vx: float, vy: float) -> None:
        """Set velocity components and calculate magnitude and direction.
        
        Args:
            vx: Velocity in X direction
            vy: Velocity in Y direction
        """
        self.velocity_x = vx
        self.velocity_y = vy
        self.velocity_magnitude = (vx ** 2 + vy ** 2) ** 0.5

        # Calculate direction in degrees (0-360)
        import math
        self.direction = (math.atan2(vy, vx) * 180 / math.pi) % 360

    def set_license_plate(
        self,
        plate_text: str,
        confidence: float,
        bbox: dict[str, float] | None = None
    ) -> None:
        """Set license plate information.
        
        Args:
            plate_text: License plate text
            confidence: Recognition confidence
            bbox: License plate bounding box coordinates
        """
        self.license_plate = plate_text
        self.license_plate_confidence = confidence
        if bbox:
            self.license_plate_region = bbox

    def mark_false_positive(self, reason: str | None = None) -> None:
        """Mark detection as false positive.
        
        Args:
            reason: Optional reason for false positive marking
        """
        self.is_false_positive = True
        if reason and self.additional_attributes is None:
            self.additional_attributes = {}
        if reason:
            self.additional_attributes["false_positive_reason"] = reason

    def verify_detection(self, verified_by: str) -> None:
        """Mark detection as human-verified.
        
        Args:
            verified_by: Username or ID of verifier
        """
        self.is_verified = True
        if self.additional_attributes is None:
            self.additional_attributes = {}
        self.additional_attributes.update({
            "verified_by": verified_by,
            "verified_at": datetime.utcnow().isoformat()
        })

    @property
    def bbox_center_x(self) -> float:
        """Calculate bounding box center X coordinate."""
        return (self.bbox_x1 + self.bbox_x2) / 2

    @property
    def bbox_center_y(self) -> float:
        """Calculate bounding box center Y coordinate."""
        return (self.bbox_y1 + self.bbox_y2) / 2

    @property
    def is_vehicle(self) -> bool:
        """Check if detection is a vehicle."""
        vehicle_classes = {
            DetectionClass.CAR, DetectionClass.TRUCK, DetectionClass.BUS,
            DetectionClass.MOTORCYCLE, DetectionClass.BICYCLE
        }
        return self.class_name in {cls.value for cls in vehicle_classes}

    @property
    def is_high_confidence(self) -> bool:
        """Check if detection has high confidence."""
        return self.class_confidence >= 0.8

    @property
    def is_reliable(self) -> bool:
        """Check if detection is reliable (high confidence, good quality)."""
        return (
            self.is_high_confidence and
            self.detection_quality >= 0.7 and
            not self.is_false_positive
        )

    def __repr__(self) -> str:
        return (
            f"<DetectionResult("
            f"class={self.class_name}, "
            f"confidence={self.class_confidence:.2f}, "
            f"bbox=({self.bbox_x1:.1f},{self.bbox_y1:.1f},{self.bbox_x2:.1f},{self.bbox_y2:.1f})"
            f")>"
        )
