"""Security configuration settings."""

import os

from pydantic import BaseModel, Field, field_validator


class SecurityConfig(BaseModel):
    """Security configuration settings."""

    enabled: bool = Field(default=True, description="Enable authentication")
    # JWT Settings
    secret_key: str = Field(
        default_factory=lambda: os.getenv("SECRET_KEY") or "change-me-in-production",
        min_length=32,
        description="Secret key for JWT tokens",
    )

    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key strength."""
        weak_secrets = {
            "change-me-in-production",
            "secret",
            "password",
            "admin",
            "test",
            "development"
        }

        if v.lower() in weak_secrets:
            raise ValueError(f"Secret key '{v}' is a default/weak value. Use a cryptographically secure key.")

        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")

        return v
    algorithm: str = Field(
        default="RS256", description="JWT algorithm (RS256 recommended for production)"
    )
    access_token_expire_minutes: int = Field(
        default=15, description="Access token expiration time in minutes"
    )
    refresh_token_expire_days: int = Field(
        default=7, description="Refresh token expiration time in days"
    )

    # CORS Settings
    enable_cors: bool = Field(default=True, description="Enable CORS")
    allowed_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    allow_credentials: bool = Field(default=True, description="Allow credentials")
    allow_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"], description="Allowed CORS methods"
    )
    allow_headers: list[str] = Field(default=["*"], description="Allowed CORS headers")
    # Rate Limiting
    rate_limit_per_minute: int = Field(
        default=100, description="Rate limit per minute per IP"
    )
    auth_rate_limit_per_minute: int = Field(
        default=5, description="Authentication rate limit per minute per IP"
    )

    # Password Policy
    password_min_length: int = Field(default=12, description="Minimum password length")
    password_require_uppercase: bool = Field(
        default=True, description="Require uppercase in password"
    )
    password_require_lowercase: bool = Field(
        default=True, description="Require lowercase in password"
    )
    password_require_digits: bool = Field(
        default=True, description="Require digits in password"
    )
    password_require_special: bool = Field(
        default=True, description="Require special characters in password"
    )
    password_history_size: int = Field(
        default=12, description="Number of previous passwords to remember"
    )

    # MFA Settings
    mfa_issuer_name: str = Field(
        default="ITS Camera AI", description="MFA issuer name for TOTP"
    )
    mfa_totp_window: int = Field(
        default=1, description="TOTP time window (30s intervals)"
    )
    mfa_backup_codes_count: int = Field(
        default=8, description="Number of MFA backup codes to generate"
    )

    # Session Management
    session_timeout_minutes: int = Field(
        default=480, description="Session timeout in minutes (8 hours)"
    )
    max_sessions_per_user: int = Field(
        default=5, description="Maximum concurrent sessions per user"
    )
    session_sliding_expiration: bool = Field(
        default=True, description="Enable sliding session expiration"
    )

    # Brute Force Protection
    max_login_attempts: int = Field(
        default=5, description="Maximum failed login attempts"
    )
    lockout_duration_minutes: int = Field(
        default=15, description="Account lockout duration in minutes"
    )
    attempt_window_minutes: int = Field(
        default=5, description="Time window for counting failed attempts"
    )

    # Security Headers
    enable_security_headers: bool = Field(
        default=True, description="Enable security headers"
    )
    hsts_max_age: int = Field(
        default=31536000, description="HSTS max age in seconds (1 year)"
    )

    # Audit Logging
    enable_audit_logging: bool = Field(
        default=True, description="Enable security audit logging"
    )
    audit_log_retention_days: int = Field(
        default=365, description="Audit log retention in days"
    )
    high_risk_alert_threshold: int = Field(
        default=80, description="Risk score threshold for alerts"
    )

    # Encryption
    enable_at_rest_encryption: bool = Field(
        default=True, description="Enable data encryption at rest"
    )
    enable_in_transit_encryption: bool = Field(
        default=True, description="Enable data encryption in transit"
    )
    encryption_key_rotation_days: int = Field(
        default=90, description="Encryption key rotation period"
    )

    # OAuth2/OIDC
    enable_oauth2: bool = Field(
        default=False, description="Enable OAuth2/OIDC integration"
    )
    oauth2_provider_url: str | None = Field(
        default=None, description="OAuth2 provider URL"
    )
    oauth2_client_id: str | None = Field(default=None, description="OAuth2 client ID")
    oauth2_client_secret: str | None = Field(
        default=None, description="OAuth2 client secret"
    )

    # Zero Trust
    enable_zero_trust: bool = Field(
        default=True, description="Enable zero trust architecture"
    )
    require_device_verification: bool = Field(
        default=False, description="Require device verification"
    )
    enable_continuous_verification: bool = Field(
        default=True, description="Enable continuous security verification"
    )

    # Hosted Proxy
    allowed_hosts: list[str] = Field(
        default=["localhost"], description="List of allowed hosts for the proxy"
    )

    # API Security Enhancement Settings
    enable_api_key_auth: bool = Field(
        default=True, description="Enable API key authentication"
    )
    enable_csrf_protection: bool = Field(
        default=True, description="Enable CSRF protection"
    )
    enable_security_validation: bool = Field(
        default=True, description="Enable comprehensive input validation"
    )
    enable_enhanced_rate_limiting: bool = Field(
        default=True, description="Enable enhanced rate limiting"
    )

    # CSP Settings
    csp_report_uri: str | None = Field(
        default=None, description="CSP violation report URI"
    )

    # File Upload Security
    max_upload_size: int = Field(
        default=100 * 1024 * 1024, description="Maximum upload size in bytes (100MB)"
    )
    allowed_file_types: list[str] = Field(
        default=["image/jpeg", "image/png", "video/mp4", "application/json"],
        description="Allowed file MIME types"
    )

    # Input Validation Settings
    max_json_depth: int = Field(
        default=10, description="Maximum JSON nesting depth"
    )
    max_request_size: int = Field(
        default=50 * 1024 * 1024, description="Maximum request size in bytes (50MB)"
    )

    # Security Monitoring
    enable_security_monitoring: bool = Field(
        default=True, description="Enable security event monitoring"
    )
    security_alert_webhook: str | None = Field(
        default=None, description="Webhook URL for security alerts"
    )
