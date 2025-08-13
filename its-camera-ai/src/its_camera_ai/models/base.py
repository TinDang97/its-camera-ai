"""Base database model with common functionality.

Provides base SQLAlchemy model with common fields and methods.
"""

import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class BaseTableModel(DeclarativeBase):
    """Base model class with common fields and functionality.

    All models should inherit from this class to get:
    - UUID primary key
    - Created/updated timestamps
    - Common query methods
    """

    # Primary key as UUID
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        server_default=text("gen_random_uuid()"),
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    @declared_attr.directive
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        # Convert CamelCase to snake_case
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
        return name

    def to_dict(self, exclude: set | None = None) -> dict[str, Any]:
        """Convert model instance to dictionary.

        Args:
            exclude: Set of field names to exclude

        Returns:
            Dict representation of the model
        """
        exclude = exclude or set()
        result = {}

        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                # Handle datetime serialization
                if isinstance(value, datetime):
                    value = value.isoformat()
                result[column.name] = value

        return result

    def update_from_dict(self, data: dict[str, Any]) -> None:
        """Update model instance from dictionary.

        Args:
            data: Dictionary of field values to update
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Update timestamp
        self.updated_at = datetime.now(UTC)

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"<{self.__class__.__name__}(id={self.id})>"
