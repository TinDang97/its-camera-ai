"""Security configuration and hardening for ITS Camera AI.

Provides comprehensive security configuration including:
- JWT security settings
- File operation security
- Network binding security
- Authentication flow configuration
- Security middleware settings
"""

import os
import secrets
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, validator


class SecurityLevel(str, Enum):
    """Security level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NetworkBindingConfig(BaseModel):
    """Network binding security configuration."""

    production_hosts: list[str] = Field(
        default=["127.0.0.1", "localhost"],
        description="Allowed hosts for production environment"
    )
    development_hosts: list[str] = Field(
        default=["0.0.0.0", "127.0.0.1", "localhost"],
        description="Allowed hosts for development environment"
    )
    default_production_host: str = Field(
        default="127.0.0.1",
        description="Default host for production binding"
    )
    default_development_host: str = Field(
        default="0.0.0.0",
        description="Default host for development binding"
    )

    @validator('production_hosts')
    def validate_production_hosts(cls, v):
        """Ensure production hosts are secure."""
        if "0.0.0.0" in v:
            raise ValueError("Production should not bind to 0.0.0.0 (all interfaces)")
        return v


class FileSecurityConfig(BaseModel):
    """File operation security configuration."""

    temp_file_permissions: int = Field(
        default=0o600,
        description="Permissions for temporary files (owner read/write only)"
    )
    secure_temp_directory: str | None = Field(
        default=None,
        description="Secure directory for temporary files"
    )
    max_upload_size_mb: int = Field(
        default=100,
        description="Maximum upload size in MB"
    )
    allowed_extensions: list[str] = Field(
        default=[".pt", ".pth", ".onnx", ".pb", ".h5", ".pkl", ".joblib", ".mp4", ".avi", ".jpg", ".png"],
        description="Allowed file extensions for uploads"
    )

    @validator('temp_file_permissions')
    def validate_permissions(cls, v):
        """Ensure secure file permissions."""
        if v & 0o077:  # Check if group/other have any permissions
            raise ValueError("Temporary files should only be accessible by owner (0o600)")
        return v


class JWTSecurityConfig(BaseModel):
    """JWT security configuration."""

    algorithm: str = Field(
        default="RS256",
        description="JWT signing algorithm (RS256 recommended for production)"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )
    refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiration time in days"
    )
    require_audience: bool = Field(
        default=True,
        description="Require audience claim in JWT tokens"
    )
    require_issuer: bool = Field(
        default=True,
        description="Require issuer claim in JWT tokens"
    )

    @validator('algorithm')
    def validate_algorithm(cls, v):
        """Ensure secure JWT algorithm."""
        secure_algorithms = ["RS256", "ES256", "PS256"]
        if v not in secure_algorithms:
            raise ValueError(f"JWT algorithm must be one of {secure_algorithms} for security")
        return v


class PasswordPolicyConfig(BaseModel):
    """Password policy configuration."""

    min_length: int = Field(
        default=12,
        description="Minimum password length"
    )
    require_uppercase: bool = Field(
        default=True,
        description="Require uppercase letters"
    )
    require_lowercase: bool = Field(
        default=True,
        description="Require lowercase letters"
    )
    require_numbers: bool = Field(
        default=True,
        description="Require numbers"
    )
    require_special_chars: bool = Field(
        default=True,
        description="Require special characters"
    )
    max_age_days: int = Field(
        default=90,
        description="Maximum password age in days"
    )
    history_count: int = Field(
        default=5,
        description="Number of previous passwords to remember"
    )

    @validator('min_length')
    def validate_min_length(cls, v):
        """Ensure secure minimum length."""
        if v < 8:
            raise ValueError("Minimum password length should be at least 8 characters")
        return v


class SecurityAuditConfig(BaseModel):
    """Security audit and logging configuration."""

    log_failed_logins: bool = Field(
        default=True,
        description="Log failed login attempts"
    )
    log_successful_logins: bool = Field(
        default=True,
        description="Log successful login attempts"
    )
    log_authorization_failures: bool = Field(
        default=True,
        description="Log authorization failures"
    )
    log_suspicious_activity: bool = Field(
        default=True,
        description="Log suspicious activity"
    )
    max_failed_attempts: int = Field(
        default=5,
        description="Maximum failed login attempts before lockout"
    )
    lockout_duration_minutes: int = Field(
        default=15,
        description="Account lockout duration in minutes"
    )
    session_timeout_minutes: int = Field(
        default=60,
        description="Session timeout in minutes"
    )


class SecurityMiddlewareConfig(BaseModel):
    """Security middleware configuration."""

    enable_cors: bool = Field(
        default=True,
        description="Enable CORS middleware"
    )
    cors_allow_origins: list[str] = Field(
        default=[],
        description="Allowed CORS origins (empty = no origins allowed)"
    )
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting middleware"
    )
    rate_limit_requests_per_minute: int = Field(
        default=60,
        description="Rate limit: requests per minute per IP"
    )
    enable_security_headers: bool = Field(
        default=True,
        description="Enable security headers middleware"
    )
    content_security_policy: str = Field(
        default="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';",
        description="Content Security Policy header"
    )


class ComprehensiveSecurityConfig(BaseModel):
    """Comprehensive security configuration for ITS Camera AI."""

    security_level: SecurityLevel = Field(
        default=SecurityLevel.HIGH,
        description="Overall security level"
    )
    network_binding: NetworkBindingConfig = Field(
        default_factory=NetworkBindingConfig,
        description="Network binding security settings"
    )
    file_security: FileSecurityConfig = Field(
        default_factory=FileSecurityConfig,
        description="File operation security settings"
    )
    jwt_security: JWTSecurityConfig = Field(
        default_factory=JWTSecurityConfig,
        description="JWT security settings"
    )
    password_policy: PasswordPolicyConfig = Field(
        default_factory=PasswordPolicyConfig,
        description="Password policy settings"
    )
    audit_config: SecurityAuditConfig = Field(
        default_factory=SecurityAuditConfig,
        description="Security audit and logging settings"
    )
    middleware_config: SecurityMiddlewareConfig = Field(
        default_factory=SecurityMiddlewareConfig,
        description="Security middleware settings"
    )

    def get_secure_host_for_environment(self, is_production: bool, explicit_host: str | None = None) -> str:
        """Get secure host binding for environment."""
        if explicit_host:
            return explicit_host

        if is_production:
            return self.network_binding.default_production_host
        else:
            return self.network_binding.default_development_host

    def validate_host_for_environment(self, host: str, is_production: bool) -> bool:
        """Validate host binding for environment."""
        allowed_hosts = (
            self.network_binding.production_hosts if is_production
            else self.network_binding.development_hosts
        )
        return host in allowed_hosts

    def get_secure_temp_file_config(self) -> dict[str, any]:
        """Get secure temporary file configuration."""
        return {
            "permissions": self.file_security.temp_file_permissions,
            "directory": self.file_security.secure_temp_directory,
            "max_size_mb": self.file_security.max_upload_size_mb,
        }

    def generate_secure_secret_key(self, length: int = 32) -> str:
        """Generate cryptographically secure secret key."""
        return secrets.token_urlsafe(length)

    def is_file_extension_allowed(self, filename: str) -> bool:
        """Check if file extension is allowed."""
        file_ext = Path(filename).suffix.lower()
        return file_ext in self.file_security.allowed_extensions

    def get_jwt_config_for_environment(self, is_production: bool) -> dict[str, any]:
        """Get JWT configuration for environment."""
        config = self.jwt_security.dict()

        # Shorter token lifetime in production
        if is_production:
            config["access_token_expire_minutes"] = min(config["access_token_expire_minutes"], 15)

        return config


def get_security_config() -> ComprehensiveSecurityConfig:
    """Get security configuration with environment-specific overrides."""
    config = ComprehensiveSecurityConfig()

    # Environment-specific overrides
    environment = os.getenv("ENVIRONMENT", "development").lower()

    if environment == "production":
        # Production security hardening
        config.security_level = SecurityLevel.CRITICAL
        config.jwt_security.access_token_expire_minutes = 15
        config.audit_config.max_failed_attempts = 3
        config.audit_config.lockout_duration_minutes = 30
        config.middleware_config.rate_limit_requests_per_minute = 30

    elif environment == "staging":
        # Staging security
        config.security_level = SecurityLevel.HIGH
        config.jwt_security.access_token_expire_minutes = 20
        config.audit_config.max_failed_attempts = 4

    elif environment == "development":
        # Development security (relaxed but still secure)
        config.security_level = SecurityLevel.MEDIUM
        config.jwt_security.access_token_expire_minutes = 60
        config.audit_config.max_failed_attempts = 10
        config.middleware_config.rate_limit_requests_per_minute = 120

    return config


def validate_security_configuration() -> list[str]:
    """Validate security configuration and return any issues."""
    issues = []
    config = get_security_config()
    environment = os.getenv("ENVIRONMENT", "development").lower()

    # Check environment-specific requirements
    if environment == "production":
        if config.security_level != SecurityLevel.CRITICAL:
            issues.append("Production environment should use CRITICAL security level")

        if config.jwt_security.algorithm not in ["RS256", "ES256", "PS256"]:
            issues.append("Production should use RS256, ES256, or PS256 for JWT signing")

        if "0.0.0.0" in config.network_binding.production_hosts:
            issues.append("Production should not allow binding to 0.0.0.0")

    # Check file security
    if config.file_security.temp_file_permissions & 0o077:
        issues.append("Temporary files should only be accessible by owner (0o600)")

    # Check password policy
    if config.password_policy.min_length < 12 and environment == "production":
        issues.append("Production should require minimum 12 character passwords")

    return issues


# Security utility functions
def secure_compare_secrets(secret1: str, secret2: str) -> bool:
    """Securely compare secrets to prevent timing attacks."""
    return secrets.compare_digest(secret1, secret2)


def generate_secure_filename(original_filename: str, prefix: str = "") -> str:
    """Generate secure filename with random component."""
    file_extension = Path(original_filename).suffix
    random_part = secrets.token_hex(8)
    safe_name = "".join(c for c in Path(original_filename).stem if c.isalnum() or c in "_-")[:20]
    return f"{prefix}{safe_name}_{random_part}{file_extension}"


def create_secure_temp_file(suffix: str = "", permissions: int = 0o600) -> Path:
    """Create temporary file with secure permissions."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_path = Path(tmp.name)

    # Set secure permissions
    os.chmod(temp_path, permissions)

    return temp_path
