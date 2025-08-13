"""
Authentication API routes for ITS Camera AI system.

Provides comprehensive authentication endpoints:
- User authentication (login/logout)
- JWT token management (refresh, validation)
- Multi-factor authentication (setup, verify)
- User registration and management
- Password management (change, reset)
- Session management
- Security audit endpoints
"""

from datetime import datetime, timezone

import redis.asyncio as redis
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config import get_settings
from ...core.database import get_database_session
from ...core.exceptions import AuthenticationError
from ...services.auth_service import (
    AuthenticationService,
    AuthenticationStatus,
    MFAEnrollment,
    MFAMethod,
    MFAVerification,
    TokenPair,
    UserCredentials,
)

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Security scheme
security = HTTPBearer(auto_error=False)


# ============================
# Request/Response Models
# ============================


class LoginRequest(BaseModel):
    """Login request model."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=1, max_length=128)
    mfa_code: str | None = Field(None, min_length=6, max_length=8)
    remember_me: bool = Field(default=False)

    model_config = {
        "json_schema_extra": {
            "example": {
                "username": "john.doe",
                "password": "SecurePass123!",
                "mfa_code": "123456",
                "remember_me": False,
            }
        }
    }


class LoginResponse(BaseModel):
    """Login response model."""

    success: bool
    message: str
    access_token: str | None = None
    refresh_token: str | None = None
    token_type: str = "Bearer"
    expires_in: int | None = None
    user: dict | None = None
    mfa_required: bool = False
    mfa_methods: list[str] = []

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "message": "Login successful",
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "token_type": "Bearer",
                "expires_in": 900,
                "user": {
                    "id": "123e4567-e89b-12d3-a456-426614174000",
                    "username": "john.doe",
                    "email": "john@example.com",
                    "roles": ["viewer"],
                },
            }
        }


class RegisterRequest(BaseModel):
    """User registration request model."""

    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=12, max_length=128)
    full_name: str | None = Field(None, max_length=100)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        """Validate password strength."""
        from ...services.auth_service import PasswordPolicy

        validation = PasswordPolicy.validate_password(v)
        if not validation["valid"]:
            raise ValueError(
                f"Password validation failed: {', '.join(validation['errors'])}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "username": "john.doe",
                "email": "john@example.com",
                "password": "SecureP@ssw0rd2024!",
                "full_name": "John Doe",
            }
        }


class RegisterResponse(BaseModel):
    """User registration response model."""

    success: bool
    message: str
    user_id: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "message": "User registered successfully",
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
            }
        }


class RefreshTokenRequest(BaseModel):
    """Token refresh request model."""

    refresh_token: str

    model_config = {
        "json_schema_extra": {
            "example": {"refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."}
        }


class ChangePasswordRequest(BaseModel):
    """Password change request model."""

    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=12, max_length=128)

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v):
        """Validate new password strength."""
        from ...services.auth_service import PasswordPolicy

        validation = PasswordPolicy.validate_password(v)
        if not validation["valid"]:
            raise ValueError(
                f"Password validation failed: {', '.join(validation['errors'])}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "current_password": "OldPassword123!",
                "new_password": "NewSecureP@ssw0rd2024!",
            }
        }


class MFASetupRequest(BaseModel):
    """MFA setup request model."""

    method: MFAMethod = Field(default=MFAMethod.TOTP)

    model_config = {
        "json_schema_extra": {"example": {"method": "totp"}}


class MFAVerifyRequest(BaseModel):
    """MFA verification request model."""

    code: str = Field(..., min_length=6, max_length=8)
    method: MFAMethod = Field(default=MFAMethod.TOTP)

    model_config = {
        "json_schema_extra": {"example": {"code": "123456", "method": "totp"}}


class UserProfileResponse(BaseModel):
    """User profile response model."""

    id: str
    username: str
    email: str
    full_name: str | None
    is_active: bool
    is_verified: bool
    mfa_enabled: bool
    roles: list[str]
    permissions: list[str]
    last_login: datetime | None
    created_at: datetime

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "username": "john.doe",
                "email": "john@example.com",
                "full_name": "John Doe",
                "is_active": True,
                "is_verified": True,
                "mfa_enabled": False,
                "roles": ["viewer"],
                "permissions": ["cameras:read", "analytics:read"],
                "last_login": "2024-01-15T10:30:00Z",
                "created_at": "2024-01-01T00:00:00Z",
            }
        }


# ============================
# Dependencies
# ============================


async def get_auth_service(
    db: AsyncSession = Depends(get_database_session),
) -> AuthenticationService:
    """Dependency to get authentication service."""
    settings = get_settings()
    redis_client = redis.from_url(settings.redis.url)
    return AuthenticationService(db, redis_client, settings.security)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    auth_service: AuthenticationService = Depends(get_auth_service),
) -> dict:
    """Dependency to get current authenticated user."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        validation = await auth_service.verify_token(credentials.credentials)
        if not validation.valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=validation.error_message or "Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return {
            "user_id": validation.user_id,
            "session_id": validation.session_id,
            "roles": validation.roles,
            "permissions": validation.permissions,
        }

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_permission(permission: str):
    """Dependency factory to require specific permission."""

    async def check_permission(current_user: dict = Depends(get_current_user)) -> dict:
        if permission not in current_user["permissions"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required",
            )
        return current_user

    return check_permission


# ============================
# Authentication Endpoints
# ============================


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    http_request: Request,
    response: Response,
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """
    Authenticate user and return JWT tokens.

    Supports:
    - Username/password authentication
    - Multi-factor authentication
    - Remember me functionality
    - Session management
    """
    try:
        # Get client information
        client_ip = http_request.headers.get(
            "X-Forwarded-For", http_request.client.host
        )
        user_agent = http_request.headers.get("User-Agent", "")

        # Create credentials
        credentials = UserCredentials(
            username=request.username,
            password=request.password,
            mfa_code=request.mfa_code,
            ip_address=client_ip,
            device_fingerprint=user_agent,  # Simplified fingerprint
        )

        # Authenticate user
        auth_result = await auth_service.authenticate(credentials)

        if auth_result.success:
            # Set secure cookies for tokens
            if auth_result.access_token:
                response.set_cookie(
                    key="access_token",
                    value=auth_result.access_token,
                    httponly=True,
                    secure=True,
                    samesite="strict",
                    max_age=auth_result.expires_in,
                )

            if auth_result.refresh_token and request.remember_me:
                response.set_cookie(
                    key="refresh_token",
                    value=auth_result.refresh_token,
                    httponly=True,
                    secure=True,
                    samesite="strict",
                    max_age=7 * 24 * 60 * 60,  # 7 days
                )

            # Get user information
            user_info = None
            if auth_result.user_id:
                user = await auth_service._get_user_with_roles(auth_result.user_id)
                if user:
                    user_info = {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                        "full_name": user.full_name,
                        "roles": [role.name for role in user.roles],
                    }

            return LoginResponse(
                success=True,
                message="Login successful",
                access_token=auth_result.access_token,
                refresh_token=auth_result.refresh_token
                if not request.remember_me
                else None,
                expires_in=auth_result.expires_in,
                user=user_info,
            )

        else:
            # Handle different failure scenarios
            if auth_result.status == AuthenticationStatus.MFA_REQUIRED:
                return LoginResponse(
                    success=False,
                    message="Multi-factor authentication required",
                    mfa_required=True,
                    mfa_methods=[method.value for method in auth_result.mfa_methods],
                )

            elif auth_result.status == AuthenticationStatus.BLOCKED:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=auth_result.error_message,
                )

            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=auth_result.error_message or "Authentication failed",
                )

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication",
        )


@router.post("/register", response_model=RegisterResponse)
async def register(
    request: RegisterRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """
    Register a new user account.

    Creates a new user with:
    - Username and email validation
    - Password strength validation
    - Default viewer role assignment
    """
    try:
        # Check if user already exists
        existing_user = await auth_service._get_user_by_username(request.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists",
            )

        # Hash password
        import bcrypt

        hashed_password = bcrypt.hashpw(
            request.password.encode(), bcrypt.gensalt()
        ).decode()

        # Create user
        user = await auth_service.user_service.create(
            username=request.username,
            email=request.email,
            hashed_password=hashed_password,
            full_name=request.full_name,
            is_active=True,
            is_verified=False,  # Require email verification
        )

        # Send verification email
        from ...services.email_service import EmailService
        
        settings = get_settings()
        email_service = EmailService(settings)
        
        # Generate verification token
        import secrets
        verification_token = secrets.token_urlsafe(32)
        
        # Store verification token in cache
        cache_key = f"email_verification:{verification_token}"
        verification_data = {
            "user_id": user.id,
            "email": request.email,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Store for 24 hours
        redis_client = redis.from_url(settings.redis.url)
        cache = CacheService(redis_client)
        await cache.set_json(cache_key, verification_data, 24 * 3600)
        
        # Send email in background
        import asyncio
        asyncio.create_task(
            email_service.send_verification_email(
                to_email=request.email,
                username=request.username,
                verification_token=verification_token,
                full_name=request.full_name
            )
        )

        return RegisterResponse(
            success=True,
            message="User registered successfully. Please verify your email.",
            user_id=user.id,
        )

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during registration",
        )


@router.post("/refresh", response_model=TokenPair)
async def refresh_token(
    request: RefreshTokenRequest,
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """
    Refresh JWT access token using refresh token.

    Returns:
    - New access token
    - New refresh token
    - Token expiration time
    """
    try:
        token_pair = await auth_service.refresh_token(request.refresh_token)
        return token_pair

    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed",
        )


@router.post("/logout")
async def logout(
    response: Response,
    current_user: dict = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """
    Logout user and invalidate session.

    - Invalidates current session
    - Clears authentication cookies
    - Logs security event
    """
    try:
        # Logout session
        session_id = current_user.get("session_id")
        if session_id:
            await auth_service.logout(session_id)

        # Clear cookies
        response.delete_cookie("access_token")
        response.delete_cookie("refresh_token")

        return {"success": True, "message": "Logout successful"}

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Logout failed"
        )


# ============================
# User Profile Endpoints
# ============================


@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(
    current_user: dict = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """
    Get current user profile information.

    Returns comprehensive user information including:
    - Basic profile data
    - Role and permission assignments
    - Security settings
    """
    try:
        user = await auth_service._get_user_with_roles(current_user["user_id"])
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        # Get user permissions
        permissions = await auth_service._get_user_permissions(user)

        return UserProfileResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            mfa_enabled=getattr(user, "mfa_enabled", False),
            roles=[role.name for role in user.roles],
            permissions=permissions,
            last_login=user.last_login,
            created_at=user.created_at,
        )

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile",
        )


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """
    Change user password.

    Requires:
    - Current password verification
    - New password strength validation
    - Session invalidation for security
    """
    try:
        success = await auth_service.change_password(
            current_user["user_id"], request.current_password, request.new_password
        )

        if success:
            return {"success": True, "message": "Password changed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed",
        )


# ============================
# MFA Endpoints
# ============================


@router.post("/mfa/setup", response_model=MFAEnrollment)
async def setup_mfa(
    request: MFASetupRequest,
    current_user: dict = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """
    Set up multi-factor authentication.

    Supports:
    - TOTP (Time-based One-Time Password)
    - SMS (future enhancement)
    - Backup recovery codes
    """
    try:
        result = await auth_service.enroll_mfa(current_user["user_id"], request.method)
        return result

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MFA setup failed"
        )


@router.post("/mfa/verify", response_model=MFAVerification)
async def verify_mfa(
    request: MFAVerifyRequest,
    current_user: dict = Depends(get_current_user),
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """
    Verify MFA code.

    Used for:
    - Initial MFA setup verification
    - Administrative MFA verification
    - Testing MFA configuration
    """
    try:
        result = await auth_service.verify_mfa(
            current_user["user_id"], request.code, request.method
        )
        return result

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA verification failed",
        )


# ============================
# Token Validation Endpoints
# ============================


@router.get("/validate")
async def validate_token(current_user: dict = Depends(get_current_user)):
    """
    Validate current JWT token.

    Returns token validation information:
    - Token validity status
    - User information
    - Permissions and roles
    - Token expiration
    """
    return {
        "valid": True,
        "user_id": current_user["user_id"],
        "roles": current_user["roles"],
        "permissions": current_user["permissions"],
    }


# ============================
# Admin Endpoints (Require Special Permissions)
# ============================


@router.get("/users", dependencies=[Depends(require_permission("users:read"))])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    auth_service: AuthenticationService = Depends(get_auth_service),
):
    """List all users (admin only)."""
    try:
        users = await auth_service.user_service.get_all(limit=limit, offset=skip)

        return {
            "users": [
                {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "is_active": user.is_active,
                    "is_verified": user.is_verified,
                    "roles": [role.name for role in user.roles] if user.roles else [],
                    "created_at": user.created_at,
                    "last_login": user.last_login,
                }
                for user in users
            ],
            "total": len(users),
            "skip": skip,
            "limit": limit,
        }

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users",
        )
