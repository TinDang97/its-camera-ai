"""User model for authentication and authorization."""

from typing import List, Optional

from sqlalchemy import Boolean, Column, String, Table, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel


# Association table for many-to-many relationship between users and roles
user_roles = Table(
    "user_roles",
    BaseModel.metadata,
    Column("user_id", String, ForeignKey("user.id"), primary_key=True),
    Column("role_id", String, ForeignKey("role.id"), primary_key=True),
)


class User(BaseModel):
    """User model for authentication."""
    
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(128), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    roles: Mapped[List["Role"]] = relationship(
        "Role",
        secondary=user_roles,
        back_populates="users"
    )
    
    def __repr__(self) -> str:
        return f"<User(username={self.username})>"


class Role(BaseModel):
    """Role model for authorization."""
    
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Relationships
    users: Mapped[List[User]] = relationship(
        "User",
        secondary=user_roles,
        back_populates="roles"
    )
    
    def __repr__(self) -> str:
        return f"<Role(name={self.name})>"
