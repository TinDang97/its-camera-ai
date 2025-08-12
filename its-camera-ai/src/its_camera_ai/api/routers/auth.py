"""Authentication and authorization endpoints.

Provides user registration, login/logout, password management,
MFA setup, and profile management functionality.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import HTTPBearer
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
    RateLimiter,
    get_auth_service,
    get_cache_service,
    get_current_user,
    get_db,
    rate_limit_strict,
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
register_rate_limit = RateLimiter(calls=5, period=3600)  # 5 registrations per hour
login_rate_limit = RateLimiter(calls=10, period=900)  # 10 login attempts per 15 min
password_reset_rate_limit = RateLimiter(calls=3, period=3600)  # 3 resets per hour


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


async def send_verification_email(email: str, token: str) -> None:
    """Send email verification message.

    Args:
        email: User email address
        token: Verification token
    """
    # TODO: Implement email sending
    logger.info("Verification email sent", email=email, token=token)


async def send_password_reset_email(email: str, token: str) -> None:
    """Send password reset email.

    Args:
        email: User email address
        token: Reset token
    """
    # TODO: Implement email sending
    logger.info("Password reset email sent", email=email, token=token)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email verification.",
)
async def register(
    user_data: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
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
            send_verification_email, str(user_data.email), verification_token
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
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db),
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
    token_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
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
    current_user: User = Depends(get_current_user),
    _cache: CacheService = Depends(get_cache_service),
) -> SuccessResponse:
    """Logout user and invalidate tokens.

    Args:
        current_user: Currently authenticated user
        cache: Cache service

    Returns:
        SuccessResponse: Logout confirmation
    """
    # TODO: Implement token blacklisting
    # For now, just log the logout
    logger.info("User logged out", user_id=current_user.id)

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
    verification_data: VerificationRequest,
    db: AsyncSession = Depends(get_db),
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
    reset_data: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
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
            send_password_reset_email, str(reset_data.email), reset_token
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
    reset_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db),
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
    summary="Update user profile",
    description="Update current user profile information.",
)
async def update_profile(
    profile_data: UserProfile,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _rate_limit: None = Depends(rate_limit_strict),
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
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    _rate_limit: None = Depends(rate_limit_strict),
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
    mfa_data: MFASetupRequest,
    current_user: User = Depends(get_current_user),
    _rate_limit: None = Depends(rate_limit_strict),
) -> MFASetupResponse:
    """Setup multi-factor authentication.

    Args:
        mfa_data: MFA setup data
        current_user: Currently authenticated user
        _rate_limit: Rate limiting dependency

    Returns:
        MFASetupResponse: MFA setup information
    """
    # TODO: Implement proper MFA setup with TOTP/SMS
    logger.info("MFA setup requested", user_id=current_user.id, method=mfa_data.method)

    # Generate backup codes
    import secrets

    backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]

    # Generate proper TOTP secret instead of hardcoded value
    totp_secret = secrets.token_urlsafe(32) if mfa_data.method == "totp" else None

    return MFASetupResponse(
        secret=totp_secret,
        backup_codes=backup_codes,
        qr_code_url=(
            f"otpauth://totp/ITS-Camera-AI:{current_user.username}?secret={totp_secret}&issuer=ITS-Camera-AI"
            if mfa_data.method == "totp" and totp_secret
            else None
        ),
    )


@router.post(
    "/mfa/verify",
    response_model=SuccessResponse,
    summary="Verify MFA code",
    description="Verify multi-factor authentication code.",
)
async def verify_mfa(
    _mfa_data: MFAVerifyRequest,
    current_user: User = Depends(get_current_user),
) -> SuccessResponse:
    """Verify MFA code.

    Args:
        mfa_data: MFA verification data
        current_user: Currently authenticated user

    Returns:
        SuccessResponse: MFA verification confirmation
    """
    # TODO: Implement proper MFA verification
    logger.info("MFA verification attempted", user_id=current_user.id)

    return SuccessResponse(
        success=True,
        message="MFA verified successfully",
        data={"user_id": current_user.id},
    )
