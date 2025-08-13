"""User model for authentication and authorization."""

from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseModel

# Association tables for many-to-many relationships
user_roles = Table(
    "user_roles",
    BaseModel.metadata,
    Column("user_id", String, ForeignKey("user.id"), primary_key=True),
    Column("role_id", String, ForeignKey("role.id"), primary_key=True),
)

role_permissions = Table(
    "role_permissions",
    BaseModel.metadata,
    Column("role_id", String, ForeignKey("role.id"), primary_key=True),
    Column("permission_id", String, ForeignKey("permission.id"), primary_key=True),
)


class User(BaseModel):
    """User model for authentication."""

    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(100))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)

    # MFA fields
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    mfa_secret: Mapped[str | None] = mapped_column(String(255))  # Encrypted TOTP secret
    mfa_backup_codes: Mapped[str | None] = mapped_column(
        Text
    )  # JSON array of backup codes

    # Security fields
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_login: Mapped[datetime | None] = mapped_column(DateTime)
    last_password_change: Mapped[datetime | None] = mapped_column(DateTime)
    password_history: Mapped[str | None] = mapped_column(
        Text
    )  # JSON array of password hashes
    account_locked_until: Mapped[datetime | None] = mapped_column(DateTime)

    # Audit fields
    last_login_ip: Mapped[str | None] = mapped_column(String(45))  # IPv6 support
    last_login_device: Mapped[str | None] = mapped_column(String(255))
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime)

    # Relationships
    roles: Mapped[list["Role"]] = relationship(
        "Role", secondary=user_roles, back_populates="users"
    )

    def __repr__(self) -> str:
        return f"<User(username={self.username})>"


class Role(BaseModel):
    """Role model for authorization."""

    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(255))

    # Relationships
    users: Mapped[list[User]] = relationship(
        "User", secondary=user_roles, back_populates="roles"
    )
    permissions: Mapped[list["Permission"]] = relationship(
        "Permission", secondary=role_permissions, back_populates="roles"
    )

    def __repr__(self) -> str:
        return f"<Role(name={self.name})>"




class Permission(BaseModel):
    """Permission model for fine-grained access control."""

    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    resource: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(String(255))

    # Relationships
    roles: Mapped[list[Role]] = relationship(
        "Role", secondary=role_permissions, back_populates="permissions"
    )

    def __repr__(self) -> str:
        return f"<Permission(name={self.name}, resource={self.resource}, action={self.action})>"


class SecurityAuditLog(BaseModel):
    """Security audit log for compliance and monitoring."""

    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String, ForeignKey("user.id"))
    username: Mapped[str | None] = mapped_column(String(50))
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(500))
    session_id: Mapped[str | None] = mapped_column(String(255))
    resource: Mapped[str | None] = mapped_column(String(100))
    action: Mapped[str | None] = mapped_column(String(50))
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)
    details: Mapped[str | None] = mapped_column(Text)  # JSON details
    risk_score: Mapped[int] = mapped_column(Integer, default=0)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )

    # Relationships
    user: Mapped[User | None] = relationship("User", backref="audit_logs")

    def __repr__(self) -> str:
        return f"<SecurityAuditLog(event_type={self.event_type}, user={self.username}, timestamp={self.timestamp})>"


class UserSession(BaseModel):
    """User session model for tracking active sessions."""

    session_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("user.id"), nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(500))
    device_fingerprint: Mapped[str | None] = mapped_column(String(255))
    mfa_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    last_activity: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(UTC)
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    user: Mapped[User] = relationship("User", backref="sessions")

    def __repr__(self) -> str:
        return f"<UserSession(session_id={self.session_id}, user={self.user_id}, expires_at={self.expires_at})>"
