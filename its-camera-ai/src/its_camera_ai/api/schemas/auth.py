"""Authentication and authorization schemas."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, SecretStr, field_validator


class RegisterRequest(BaseModel):
    """User registration request schema."""

    username: str = Field(min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(description="User email address")
    password: SecretStr = Field(min_length=8, description="User password")
    full_name: str | None = Field(None, max_length=100, description="User's full name")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format."""
        if not v.isalnum():
            raise ValueError("Username must contain only alphanumeric characters")
        return v.lower()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: SecretStr) -> SecretStr:
        """Validate password strength."""
        password = v.get_secret_value()
        if not any(c.isupper() for c in password):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in password):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in password):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    """User login request schema."""

    username: str = Field(description="Username or email")
    password: SecretStr = Field(description="User password")
    remember_me: bool = Field(False, description="Remember login for extended period")


class LoginResponse(BaseModel):
    """User login response schema."""

    access_token: str = Field(description="JWT access token")
    refresh_token: str = Field(description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(description="Token expiration time in seconds")
    user: "UserResponse" = Field(description="User information")


class RefreshTokenRequest(BaseModel):
    """Token refresh request schema."""

    refresh_token: str = Field(description="JWT refresh token")


class TokenRefreshResponse(BaseModel):
    """Token refresh response schema."""

    access_token: str = Field(description="New JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(description="Token expiration time in seconds")


class PasswordResetRequest(BaseModel):
    """Password reset request schema."""

    email: EmailStr = Field(description="User email address")


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""

    token: str = Field(description="Password reset token")
    new_password: SecretStr = Field(min_length=8, description="New password")

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: SecretStr) -> SecretStr:
        """Validate password strength."""
        password = v.get_secret_value()
        if not any(c.isupper() for c in password):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in password):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in password):
            raise ValueError("Password must contain at least one digit")
        return v


class VerificationRequest(BaseModel):
    """Email verification request schema."""

    token: str = Field(description="Email verification token")


class UserResponse(BaseModel):
    """User response schema."""

    id: str = Field(description="User ID")
    username: str = Field(description="Username")
    email: str = Field(description="Email address")
    full_name: str | None = Field(None, description="Full name")
    is_active: bool = Field(description="Whether user is active")
    is_verified: bool = Field(description="Whether email is verified")
    is_superuser: bool = Field(description="Whether user is superuser")
    roles: list[str] = Field(description="User roles")
    created_at: datetime = Field(description="User creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")

    class Config:
        from_attributes = True


class UserProfile(BaseModel):
    """User profile update schema."""

    full_name: str | None = Field(None, max_length=100, description="Full name")
    email: EmailStr | None = Field(None, description="Email address")


class PasswordChangeRequest(BaseModel):
    """Password change request schema."""

    current_password: SecretStr = Field(description="Current password")
    new_password: SecretStr = Field(min_length=8, description="New password")

    @field_validator("new_password")
    @classmethod
    def validate_password(cls, v: SecretStr) -> SecretStr:
        """Validate password strength."""
        password = v.get_secret_value()
        if not any(c.isupper() for c in password):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in password):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in password):
            raise ValueError("Password must contain at least one digit")
        return v


class MFASetupRequest(BaseModel):
    """Multi-factor authentication setup request."""

    method: str = Field(description="MFA method (totp, sms)")
    phone_number: str | None = Field(None, description="Phone number for SMS")

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate MFA method."""
        if v not in ["totp", "sms"]:
            raise ValueError("Method must be 'totp' or 'sms'")
        return v


class MFASetupResponse(BaseModel):
    """Multi-factor authentication setup response."""

    secret: str | None = Field(None, description="TOTP secret for QR code")
    backup_codes: list[str] = Field(description="Backup recovery codes")
    qr_code_url: str | None = Field(None, description="QR code URL for TOTP")


class MFAVerifyRequest(BaseModel):
    """Multi-factor authentication verification request."""

    code: str = Field(description="MFA verification code")
    backup_code: str | None = Field(None, description="Backup recovery code")


class ApiKeyCreateRequest(BaseModel):
    """API key creation request schema."""

    name: str = Field(max_length=100, description="API key name")
    permissions: list[str] = Field(description="List of permissions")
    expires_in_days: int | None = Field(
        None, description="Expiration in days (null for no expiration)", ge=1
    )


class ApiKeyResponse(BaseModel):
    """API key response schema."""

    id: str = Field(description="API key ID")
    name: str = Field(description="API key name")
    key: str | None = Field(None, description="API key (only on creation)")
    permissions: list[str] = Field(description="List of permissions")
    is_active: bool = Field(description="Whether key is active")
    created_at: datetime = Field(description="Creation timestamp")
    expires_at: datetime | None = Field(None, description="Expiration timestamp")
    last_used_at: datetime | None = Field(None, description="Last usage timestamp")

    class Config:
        from_attributes = True
