"""Authentication and authorization endpoints.

Provides user registration, login/logout, password management,
MFA setup, and profile management functionality.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ...core.config import Settings, get_settings
from ...core.logging import get_logger
from ...models.user import User
from ...services.auth import AuthService
from ...services.cache import CacheService
from ..dependencies import (
    RateLimiterDI,
    get_auth_service,
    get_cache_service,
    get_current_user,
    get_database_session,
)
from ..schemas.auth import (
    LoginRequest,
    LoginResponse,
    MFASetupRequest,
    MFASetupResponse,
    MFAVerifyRequest,
    PasswordChangeRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    RefreshTokenRequest,
    RegisterRequest,
    TokenRefreshResponse,
    UserProfile,
    UserResponse,
    VerificationRequest,
)
from ..schemas.common import SuccessResponse

logger = get_logger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Rate limiters for different endpoints
register_rate_limit = RateLimiterDI(calls=5, period=3600)  # 5 registrations per hour
login_rate_limit = RateLimiterDI(calls=10, period=900)  # 10 login attempts per 15 min
password_reset_rate_limit = RateLimiterDI(calls=3, period=3600)  # 3 resets per hour


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(
    data: dict[str, Any], settings: Settings, expires_delta: timedelta | None = None
) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(
            minutes=settings.security.access_token_expire_minutes
        )

    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(
        to_encode, settings.security.secret_key, algorithm=settings.security.algorithm
    )
    return encoded_jwt


def create_refresh_token(data: dict[str, Any], settings: Settings) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(
        days=settings.security.refresh_token_expire_days
    )
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(
        to_encode, settings.security.secret_key, algorithm=settings.security.algorithm
    )
    return encoded_jwt


async def send_verification_email(
    email: str, token: str, username: str, full_name: str | None = None
) -> None:
    """Send email verification message.

    Args:
        email: User email address
        token: Verification token
        username: Username
        full_name: User's full name
    """
    from ...core.config import get_settings
    from ...services.email_service import EmailService

    settings = get_settings()
    email_service = EmailService(settings)

    success = await email_service.send_verification_email(
        to_email=email, username=username, verification_token=token, full_name=full_name
    )

    if success:
        logger.info(
            "Verification email sent successfully", email=email, username=username
        )
    else:
        logger.error(
            "Failed to send verification email", email=email, username=username
        )


async def send_password_reset_email(
    email: str, token: str, username: str, full_name: str | None = None
) -> None:
    """Send password reset email.

    Args:
        email: User email address
        token: Reset token
        username: Username
        full_name: User's full name
    """
    from ...core.config import get_settings
    from ...services.email_service import EmailService

    settings = get_settings()
    email_service = EmailService(settings)

    success = await email_service.send_password_reset_email(
        to_email=email, username=username, reset_token=token, full_name=full_name
    )

    if success:
        logger.info(
            "Password reset email sent successfully", email=email, username=username
        )
    else:
        logger.error(
            "Failed to send password reset email", email=email, username=username
        )


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email verification.",
)
async def register(
    request: Request,
    user_data: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_database_session),
    _cache: CacheService = Depends(get_cache_service),
    settings: Settings = Depends(get_settings),
    _rate_limit: None = Depends(register_rate_limit),
) -> UserResponse:
    """Register a new user account.

    Args:
        user_data: User registration data
        background_tasks: Background task manager
        db: Database session
        cache: Cache service
        settings: Application settings
        _rate_limit: Rate limiting dependency

    Returns:
        UserResponse: Created user information

    Raises:
        HTTPException: If registration fails
    """
    try:
        # Check if username already exists
        existing_user = await db.execute(
            select(User).where(User.username == user_data.username)
        )
        if existing_user.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered",
            )

        # Check if email already exists
        existing_email = await db.execute(
            select(User).where(User.email == user_data.email)
        )
        if existing_email.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Create new user
        hashed_password = get_password_hash(user_data.password.get_secret_value())
        user = User(
            username=user_data.username,
            email=str(user_data.email),
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            is_active=True,
            is_verified=False,  # Require email verification
            is_superuser=False,
        )

        db.add(user)
        await db.commit()
        await db.refresh(user)

        # Generate verification token
        verification_token = create_access_token(
            data={"sub": user.id, "purpose": "email_verification"},
            settings=settings,
            expires_delta=timedelta(hours=24),
        )

        # Send verification email in background
        background_tasks.add_task(
            send_verification_email,
            str(user_data.email),
            verification_token,
            user.username,
            user.full_name,
        )

        logger.info(
            "User registered successfully", user_id=user.id, username=user.username
        )

        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            is_superuser=user.is_superuser,
            roles=[],  # New users have no roles initially
            created_at=user.created_at,
            updated_at=user.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration failed", error=str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed",
        ) from e


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="User login",
    description="Authenticate user and return access tokens.",
)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_database_session),
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
    _rate_limit: None = Depends(login_rate_limit),
) -> LoginResponse:
    """Authenticate user and return tokens.

    Args:
        login_data: Login credentials
        db: Database session
        auth_service: Authentication service
        settings: Application settings
        _rate_limit: Rate limiting dependency

    Returns:
        LoginResponse: Access and refresh tokens with user info

    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Get user by username or email
        user = await auth_service.get_user_by_username(db, login_data.username)
        if not user:
            # Try email
            user_by_email = await db.execute(
                select(User)
                .options(selectinload(User.roles))
                .where(User.email == login_data.username)
            )
            user = user_by_email.scalar_one_or_none()

        # Verify credentials
        if not user or not verify_password(
            login_data.password.get_secret_value(), user.hashed_password
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated",
            )

        # Create tokens
        token_data = {"sub": user.id}
        access_token = create_access_token(token_data, settings)
        refresh_token = create_refresh_token(token_data, settings)

        # Calculate token expiration
        expires_in = settings.security.access_token_expire_minutes * 60

        logger.info(
            "User logged in successfully", user_id=user.id, username=user.username
        )

        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",  # noqa: S106
            expires_in=expires_in,
            user=UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                is_verified=user.is_verified,
                is_superuser=user.is_superuser,
                roles=[role.name for role in user.roles],
                created_at=user.created_at,
                updated_at=user.updated_at,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed", error=str(e), username=login_data.username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed",
        ) from e


@router.post(
    "/refresh",
    response_model=TokenRefreshResponse,
    summary="Refresh access token",
    description="Get a new access token using refresh token.",
)
async def refresh_token(
    request: Request,
    token_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_database_session),
    auth_service: AuthService = Depends(get_auth_service),
    settings: Settings = Depends(get_settings),
) -> TokenRefreshResponse:
    """Refresh access token using refresh token.

    Args:
        token_data: Refresh token data
        db: Database session
        auth_service: Authentication service
        settings: Application settings

    Returns:
        TokenRefreshResponse: New access token

    Raises:
        HTTPException: If token refresh fails
    """
    try:
        # Decode refresh token
        payload = jwt.decode(
            token_data.refresh_token,
            settings.security.secret_key,
            algorithms=[settings.security.algorithm],
        )

        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")

        if user_id is None or token_type != "refresh":  # noqa: S105
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        ) from None

    # Verify user still exists and is active
    user = await auth_service.get_user_by_id(db, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Create new access token
    new_token_data = {"sub": user.id}
    access_token = create_access_token(new_token_data, settings)
    expires_in = settings.security.access_token_expire_minutes * 60

    logger.info("Token refreshed successfully", user_id=user.id)

    return TokenRefreshResponse(
        access_token=access_token,
        token_type="bearer",  # noqa: S106
        expires_in=expires_in,
    )


@router.post(
    "/logout",
    response_model=SuccessResponse,
    summary="User logout",
    description="Logout user and invalidate tokens.",
)
async def logout(
    request: Request,
    current_user: User = Depends(get_current_user),
    cache: CacheService = Depends(get_cache_service),
    settings: Settings = Depends(get_settings),
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    db: AsyncSession = Depends(get_database_session),
    auth_service: AuthService = Depends(get_auth_service),
) -> SuccessResponse:
    """Logout user and invalidate tokens.

    Args:
        request: HTTP request object
        current_user: Currently authenticated user
        cache: Cache service
        settings: Application settings
        credentials: Bearer token credentials
        db: Database session
        auth_service: Authentication service

    Returns:
        SuccessResponse: Logout confirmation
    """
    from ...services.token_service import TokenService

    try:
        token_service = TokenService(cache, settings.security)

        # Blacklist current token if provided
        if credentials and credentials.credentials:
            await token_service.blacklist_token(
                credentials.credentials, reason="user_logout"
            )

        # Get client information for audit log
        client_ip = request.headers.get("X-Forwarded-For", request.client.host)
        user_agent = request.headers.get("User-Agent")

        # Create security audit log
        await auth_service.create_security_audit_log(
            db,
            event_type="user_logout",
            user_id=current_user.id,
            username=current_user.username,
            ip_address=client_ip,
            user_agent=user_agent,
            success=True,
            details={"logout_type": "user_initiated"},
        )

        logger.info(
            "User logged out successfully",
            user_id=current_user.id,
            username=current_user.username,
            ip_address=client_ip,
        )

        return SuccessResponse(
            success=True,
            message="Logged out successfully",
            data={"user_id": current_user.id},
        )

    except Exception as e:
        logger.error("Logout failed", user_id=current_user.id, error=str(e))
        # Still return success to avoid information leakage
        return SuccessResponse(
            success=True,
            message="Logged out successfully",
            data={"user_id": current_user.id},
        )


@router.post(
    "/verify-email",
    response_model=SuccessResponse,
    summary="Verify email address",
    description="Verify user email using verification token.",
)
async def verify_email(
    request: Request,
    verification_data: VerificationRequest,
    db: AsyncSession = Depends(get_database_session),
    settings: Settings = Depends(get_settings),
) -> SuccessResponse:
    """Verify user email address.

    Args:
        verification_data: Email verification data
        db: Database session
        settings: Application settings

    Returns:
        SuccessResponse: Verification confirmation

    Raises:
        HTTPException: If verification fails
    """
    try:
        # Decode verification token
        payload = jwt.decode(
            verification_data.token,
            settings.security.secret_key,
            algorithms=[settings.security.algorithm],
        )

        user_id = payload.get("sub")
        purpose = payload.get("purpose")

        if user_id is None or purpose != "email_verification":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification token",
            )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token",
        ) from None

    # Get user
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Mark as verified
    user.is_verified = True
    await db.commit()

    logger.info("Email verified successfully", user_id=user.id)

    return SuccessResponse(
        success=True,
        message="Email verified successfully",
        data={"user_id": user.id},
    )


@router.post(
    "/password-reset",
    response_model=SuccessResponse,
    summary="Request password reset",
    description="Send password reset email to user.",
)
async def request_password_reset(
    request: Request,
    reset_data: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_database_session),
    settings: Settings = Depends(get_settings),
    _rate_limit: None = Depends(password_reset_rate_limit),
) -> SuccessResponse:
    """Request password reset.

    Args:
        reset_data: Password reset request data
        background_tasks: Background task manager
        db: Database session
        settings: Application settings
        _rate_limit: Rate limiting dependency

    Returns:
        SuccessResponse: Reset request confirmation
    """
    # Find user by email
    user_query = await db.execute(
        select(User).where(User.email == str(reset_data.email))
    )
    user = user_query.scalar_one_or_none()

    # Always return success to prevent email enumeration
    if user and user.is_active:
        # Generate reset token
        reset_token = create_access_token(
            data={"sub": user.id, "purpose": "password_reset"},
            settings=settings,
            expires_delta=timedelta(hours=1),  # 1 hour expiry
        )

        # Send reset email in background
        background_tasks.add_task(
            send_password_reset_email,
            str(reset_data.email),
            reset_token,
            user.username,
            user.full_name,
        )

        logger.info(
            "Password reset requested", user_id=user.id, email=str(reset_data.email)
        )
    else:
        logger.warning(
            "Password reset requested for unknown email", email=str(reset_data.email)
        )

    return SuccessResponse(
        success=True,
        message="If the email exists, a password reset link has been sent",
        data=None,
    )


@router.post(
    "/password-reset/confirm",
    response_model=SuccessResponse,
    summary="Confirm password reset",
    description="Reset password using reset token.",
)
async def confirm_password_reset(
    request: Request,
    reset_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_database_session),
    settings: Settings = Depends(get_settings),
) -> SuccessResponse:
    """Confirm password reset.

    Args:
        reset_data: Password reset confirmation data
        db: Database session
        settings: Application settings

    Returns:
        SuccessResponse: Password reset confirmation

    Raises:
        HTTPException: If reset fails
    """
    try:
        # Decode reset token
        payload = jwt.decode(
            reset_data.token,
            settings.security.secret_key,
            algorithms=[settings.security.algorithm],
        )

        user_id = payload.get("sub")
        purpose = payload.get("purpose")

        if user_id is None or purpose != "password_reset":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token",
            )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        ) from None

    # Get user
    user = await db.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or inactive",
        )

    # Update password
    user.hashed_password = get_password_hash(reset_data.new_password.get_secret_value())
    await db.commit()

    logger.info("Password reset completed", user_id=user.id)

    return SuccessResponse(
        success=True,
        message="Password reset successfully",
        data={"user_id": user.id},
    )


@router.get(
    "/profile",
    response_model=UserResponse,
    summary="Get user profile",
    description="Get current user profile information.",
)
async def get_profile(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """Get current user profile.

    Args:
        current_user: Currently authenticated user

    Returns:
        UserResponse: User profile information
    """
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        is_superuser=current_user.is_superuser,
        roles=[role.name for role in current_user.roles],
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
    )


@router.put(
    "/profile",
    response_model=UserResponse,
    summary="Update current user profile",
    description="Update current user profile information.",
)
async def update_profile(
    request: Request,
    profile_data: UserProfile,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
) -> UserResponse:
    """Update current user profile.

    Args:
        profile_data: Profile update data
        current_user: Currently authenticated user
        db: Database session
        _rate_limit: Rate limiting dependency

    Returns:
        UserResponse: Updated user profile

    Raises:
        HTTPException: If update fails
    """
    try:
        # Update fields if provided
        if profile_data.full_name is not None:
            current_user.full_name = profile_data.full_name

        if profile_data.email is not None:
            # Check if email is already taken by another user
            existing_email = await db.execute(
                select(User).where(
                    User.email == str(profile_data.email),
                    User.id != current_user.id,
                )
            )
            if existing_email.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already taken",
                )

            current_user.email = str(profile_data.email)
            current_user.is_verified = False  # Require re-verification

        await db.commit()
        await db.refresh(current_user)

        logger.info("Profile updated successfully", user_id=current_user.id)

        return UserResponse(
            id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            full_name=current_user.full_name,
            is_active=current_user.is_active,
            is_verified=current_user.is_verified,
            is_superuser=current_user.is_superuser,
            roles=[role.name for role in current_user.roles],
            created_at=current_user.created_at,
            updated_at=current_user.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Profile update failed", user_id=current_user.id, error=str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed",
        ) from e


@router.post(
    "/change-password",
    response_model=SuccessResponse,
    summary="Change password",
    description="Change user password.",
)
async def change_password(
    request: Request,
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
) -> SuccessResponse:
    """Change user password.

    Args:
        password_data: Password change data
        current_user: Currently authenticated user
        db: Database session
        _rate_limit: Rate limiting dependency

    Returns:
        SuccessResponse: Password change confirmation

    Raises:
        HTTPException: If password change fails
    """
    # Verify current password
    if not verify_password(
        password_data.current_password.get_secret_value(),
        current_user.hashed_password,
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Update password
    current_user.hashed_password = get_password_hash(
        password_data.new_password.get_secret_value()
    )

    await db.commit()

    logger.info("Password changed successfully", user_id=current_user.id)

    return SuccessResponse(
        success=True,
        message="Password changed successfully",
        data={"user_id": current_user.id},
    )


# MFA endpoints (basic implementation)
@router.post(
    "/mfa/setup",
    response_model=MFASetupResponse,
    summary="Setup MFA",
    description="Setup multi-factor authentication for user.",
)
async def setup_mfa(
    request: Request,
    mfa_data: MFASetupRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
    cache: CacheService = Depends(get_cache_service),
    settings: Settings = Depends(get_settings),
) -> MFASetupResponse:
    """Setup multi-factor authentication.

    Args:
        mfa_data: MFA setup data
        current_user: Currently authenticated user
        db: Database session
        cache: Cache service
        settings: Application settings
        _rate_limit: Rate limiting dependency

    Returns:
        MFASetupResponse: MFA setup information
    """
    from ...services.mfa_service import MFAService

    try:
        mfa_service = MFAService(cache, settings.security)

        if current_user.mfa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA is already enabled for this user",
            )

        if mfa_data.method == "totp":
            setup_info = await mfa_service.setup_totp(db, current_user)

            logger.info(
                "TOTP MFA setup initiated",
                user_id=current_user.id,
                username=current_user.username,
            )

            return MFASetupResponse(
                secret=setup_info["secret"],
                backup_codes=setup_info["backup_codes"],
                qr_code_url=setup_info["qr_code_url"],
            )

        elif mfa_data.method == "sms":
            # SMS MFA setup (placeholder for future implementation)
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="SMS MFA not yet implemented",
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported MFA method: {mfa_data.method}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "MFA setup failed",
            user_id=current_user.id,
            method=mfa_data.method,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="MFA setup failed"
        ) from e


@router.post(
    "/mfa/verify",
    response_model=SuccessResponse,
    summary="Verify MFA code",
    description="Verify multi-factor authentication code.",
)
async def verify_mfa(
    request: Request,
    mfa_data: MFAVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
    cache: CacheService = Depends(get_cache_service),
    settings: Settings = Depends(get_settings),
) -> SuccessResponse:
    """Verify MFA code.

    Args:
        mfa_data: MFA verification data
        current_user: Currently authenticated user
        db: Database session
        cache: Cache service
        settings: Application settings

    Returns:
        SuccessResponse: MFA verification confirmation
    """
    from ...services.mfa_service import MFAService

    try:
        mfa_service = MFAService(cache, settings.security)

        if not current_user.mfa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA is not enabled for this user",
            )

        is_valid = False

        # Try backup code first if provided
        if mfa_data.backup_code:
            is_valid = await mfa_service.verify_backup_code(
                db, current_user, mfa_data.backup_code
            )
            verification_method = "backup_code"
        else:
            # Verify TOTP code
            is_valid = await mfa_service.verify_totp(current_user, mfa_data.code)
            verification_method = "totp"

        if is_valid:
            logger.info(
                "MFA verification successful",
                user_id=current_user.id,
                method=verification_method,
            )

            return SuccessResponse(
                success=True,
                message="MFA verified successfully",
                data={"user_id": current_user.id, "method": verification_method},
            )
        else:
            logger.warning(
                "MFA verification failed",
                user_id=current_user.id,
                method=verification_method,
            )

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid MFA code"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("MFA verification error", user_id=current_user.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA verification failed",
        ) from e


@router.post(
    "/mfa/complete-setup",
    response_model=SuccessResponse,
    summary="Complete MFA setup",
    description="Complete MFA setup by verifying initial code.",
)
async def complete_mfa_setup(
    request: Request,
    verification_data: MFAVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
    cache: CacheService = Depends(get_cache_service),
    settings: Settings = Depends(get_settings),
) -> SuccessResponse:
    """Complete MFA setup by verifying the initial code.

    Args:
        verification_data: MFA verification data
        current_user: Currently authenticated user
        db: Database session
        cache: Cache service
        settings: Application settings

    Returns:
        SuccessResponse: Setup completion confirmation
    """
    from ...services.email_service import EmailService
    from ...services.mfa_service import MFAService

    try:
        mfa_service = MFAService(cache, settings.security)

        if current_user.mfa_enabled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA is already enabled for this user",
            )

        # Verify setup code
        is_valid = await mfa_service.verify_totp_setup(
            db, current_user, verification_data.code
        )

        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification code",
            )

        # Send notification email
        email_service = EmailService(settings)
        await email_service.send_mfa_enabled_email(
            to_email=current_user.email,
            username=current_user.username,
            mfa_method="TOTP",
            setup_time=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            backup_codes_count=settings.security.mfa_backup_codes_count,
            full_name=current_user.full_name,
        )

        logger.info(
            "MFA setup completed successfully",
            user_id=current_user.id,
            username=current_user.username,
        )

        return SuccessResponse(
            success=True,
            message="MFA enabled successfully",
            data={"user_id": current_user.id},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "MFA setup completion failed", user_id=current_user.id, error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA setup completion failed",
        ) from e
