"""Common Pydantic schemas used across multiple API endpoints."""

from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class SuccessResponse(BaseModel):
    """Standard success response schema."""

    success: bool = Field(True, description="Indicates successful operation")
    message: str = Field(description="Success message")
    data: dict[str, Any] | None = Field(None, description="Optional response data")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Response timestamp"
    )


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    code: str = Field(description="Error code for programmatic handling")
    details: dict[str, Any] | None = Field(None, description="Additional error details")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Error timestamp"
    )


class PaginatedResponse[T](BaseModel):
    """Paginated response schema for list endpoints."""

    items: list[T] = Field(description="List of items")
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page number", ge=1)
    size: int = Field(description="Number of items per page", ge=1, le=100)
    pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there are more pages")
    has_prev: bool = Field(description="Whether there are previous pages")

    @classmethod
    def create(
        cls,
        items: list[T],
        total: int,
        page: int,
        size: int,
    ) -> "PaginatedResponse[T]":
        """Create paginated response with calculated fields."""
        pages = (total + size - 1) // size  # Ceiling division
        return cls(
            items=items,
            total=total,
            page=page,
            size=size,
            pages=pages,
            has_next=page < pages,
            has_prev=page > 1,
        )


class HealthStatus(BaseModel):
    """Health check status schema."""

    status: str = Field(description="Service status (healthy/unhealthy)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Health check timestamp"
    )
    version: str = Field(description="Service version")
    uptime: float = Field(description="Service uptime in seconds")
    dependencies: dict[str, str] = Field(description="Status of external dependencies")


class Coordinates(BaseModel):
    """Geographic coordinates schema."""

    latitude: float = Field(description="Latitude coordinate", ge=-90, le=90)
    longitude: float = Field(description="Longitude coordinate", ge=-180, le=180)
    altitude: float | None = Field(None, description="Altitude in meters")


class TimeRange(BaseModel):
    """Time range filter schema."""

    start_time: datetime = Field(description="Start timestamp")
    end_time: datetime = Field(description="End timestamp")

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> "TimeRange":
        """Validate that end_time is after start_time."""
        instance = super().model_validate(obj, **kwargs)
        if instance.end_time <= instance.start_time:
            raise ValueError("end_time must be after start_time")
        return instance
