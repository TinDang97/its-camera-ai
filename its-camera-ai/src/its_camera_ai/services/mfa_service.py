"""Multi-Factor Authentication (MFA) service with TOTP and SMS support."""

import base64
import json
import secrets
import string
from datetime import UTC, datetime
from typing import Any

import pyotp
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import SecurityConfig
from ..core.logging import get_logger
from ..models.user import User
from ..services.cache import CacheService

logger = get_logger(__name__)


class MFAService:
    """Service for multi-factor authentication operations."""

    def __init__(self, cache_service: CacheService, security_config: SecurityConfig):
        self.cache = cache_service
        self.security_config = security_config

        # Initialize encryption for storing MFA secrets
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for MFA secrets."""
        # In production, this should be stored securely (e.g., environment variable)
        # For now, derive from security config
        import hashlib
        key_material = self.security_config.secret_key.encode()
        key_hash = hashlib.sha256(key_material).digest()
        return base64.urlsafe_b64encode(key_hash)

    def _encrypt_secret(self, secret: str) -> str:
        """Encrypt MFA secret for storage."""
        return self.cipher.encrypt(secret.encode()).decode()

    def _decrypt_secret(self, encrypted_secret: str) -> str:
        """Decrypt MFA secret from storage."""
        return self.cipher.decrypt(encrypted_secret.encode()).decode()

    def _generate_backup_codes(self, count: int = None) -> list[str]:
        """Generate backup recovery codes.
        
        Args:
            count: Number of codes to generate
            
        Returns:
            List of backup codes
        """
        if count is None:
            count = self.security_config.mfa_backup_codes_count

        codes = []
        for _ in range(count):
            # Generate 8-character alphanumeric codes
            code = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            codes.append(code)

        return codes

    def _hash_backup_code(self, code: str) -> str:
        """Hash backup code for storage."""
        import hashlib
        return hashlib.sha256(code.encode()).hexdigest()

    async def setup_totp(self, db: AsyncSession, user: User) -> dict[str, Any]:
        """Set up TOTP for a user.
        
        Args:
            db: Database session
            user: User to set up TOTP for
            
        Returns:
            Setup information including secret and QR code URL
        """
        try:
            # Generate TOTP secret
            secret = pyotp.random_base32()

            # Create TOTP instance
            totp = pyotp.TOTP(secret)

            # Generate backup codes
            backup_codes = self._generate_backup_codes()
            hashed_codes = [self._hash_backup_code(code) for code in backup_codes]

            # Store encrypted secret and hashed backup codes in database
            user.mfa_secret = self._encrypt_secret(secret)
            user.mfa_backup_codes = json.dumps(hashed_codes)
            user.mfa_enabled = False  # Will be enabled after verification

            await db.commit()

            # Generate QR code URL
            issuer_name = self.security_config.mfa_issuer_name
            account_name = f"{user.username}@{issuer_name}"

            provisioning_uri = totp.provisioning_uri(
                name=account_name,
                issuer_name=issuer_name
            )

            # Store setup session for verification
            setup_key = f"mfa_setup:{user.id}"
            setup_data = {
                "user_id": user.id,
                "secret": secret,  # Store temporarily for verification
                "backup_codes": backup_codes,
                "setup_time": datetime.now(UTC).isoformat(),
                "method": "totp"
            }

            # Store for 10 minutes
            await self.cache.set_json(setup_key, setup_data, 600)

            logger.info("TOTP setup initiated", user_id=user.id, username=user.username)

            return {
                "secret": secret,
                "qr_code_url": provisioning_uri,
                "backup_codes": backup_codes,
                "issuer": issuer_name,
                "account_name": account_name
            }

        except Exception as e:
            logger.error("TOTP setup failed", user_id=user.id, error=str(e))
            await db.rollback()
            raise

    async def verify_totp_setup(self, db: AsyncSession, user: User, code: str) -> bool:
        """Verify TOTP setup code.
        
        Args:
            db: Database session
            user: User verifying setup
            code: TOTP code to verify
            
        Returns:
            True if verification successful
        """
        try:
            setup_key = f"mfa_setup:{user.id}"
            setup_data = await self.cache.get_json(setup_key)

            if not setup_data:
                logger.warning("No MFA setup session found", user_id=user.id)
                return False

            secret = setup_data.get('secret')
            if not secret:
                return False

            # Verify TOTP code
            totp = pyotp.TOTP(secret)

            if totp.verify(code, valid_window=self.security_config.mfa_totp_window):
                # Enable MFA
                user.mfa_enabled = True
                await db.commit()

                # Clean up setup session
                await self.cache.delete(setup_key)

                logger.info("TOTP setup completed", user_id=user.id, username=user.username)
                return True

            logger.warning("Invalid TOTP code during setup", user_id=user.id)
            return False

        except Exception as e:
            logger.error("TOTP setup verification failed", user_id=user.id, error=str(e))
            return False

    async def verify_totp(self, user: User, code: str) -> bool:
        """Verify TOTP code for authentication.
        
        Args:
            user: User to verify
            code: TOTP code to verify
            
        Returns:
            True if verification successful
        """
        try:
            if not user.mfa_enabled or not user.mfa_secret:
                return False

            # Decrypt secret
            secret = self._decrypt_secret(user.mfa_secret)

            # Create TOTP instance
            totp = pyotp.TOTP(secret)

            # Verify code with window
            is_valid = totp.verify(code, valid_window=self.security_config.mfa_totp_window)

            if is_valid:
                logger.info("TOTP verification successful", user_id=user.id)
            else:
                logger.warning("TOTP verification failed", user_id=user.id)

            return is_valid

        except Exception as e:
            logger.error("TOTP verification error", user_id=user.id, error=str(e))
            return False

    async def verify_backup_code(self, db: AsyncSession, user: User, code: str) -> bool:
        """Verify backup recovery code.
        
        Args:
            db: Database session
            user: User to verify
            code: Backup code to verify
            
        Returns:
            True if verification successful
        """
        try:
            if not user.mfa_enabled or not user.mfa_backup_codes:
                return False

            # Get stored backup codes
            stored_codes = json.loads(user.mfa_backup_codes)
            code_hash = self._hash_backup_code(code.upper().strip())

            if code_hash in stored_codes:
                # Remove used code
                stored_codes.remove(code_hash)
                user.mfa_backup_codes = json.dumps(stored_codes)
                await db.commit()

                logger.info(
                    "Backup code used successfully",
                    user_id=user.id,
                    remaining_codes=len(stored_codes)
                )

                # Warn if running low on codes
                if len(stored_codes) <= 2:
                    logger.warning(
                        "User running low on backup codes",
                        user_id=user.id,
                        remaining_codes=len(stored_codes)
                    )

                return True

            logger.warning("Invalid backup code", user_id=user.id)
            return False

        except Exception as e:
            logger.error("Backup code verification error", user_id=user.id, error=str(e))
            return False

    async def generate_new_backup_codes(self, db: AsyncSession, user: User) -> list[str]:
        """Generate new backup codes for user.
        
        Args:
            db: Database session
            user: User to generate codes for
            
        Returns:
            List of new backup codes
        """
        try:
            if not user.mfa_enabled:
                raise ValueError("MFA not enabled for user")

            # Generate new codes
            backup_codes = self._generate_backup_codes()
            hashed_codes = [self._hash_backup_code(code) for code in backup_codes]

            # Store in database
            user.mfa_backup_codes = json.dumps(hashed_codes)
            await db.commit()

            logger.info(
                "New backup codes generated",
                user_id=user.id,
                code_count=len(backup_codes)
            )

            return backup_codes

        except Exception as e:
            logger.error("Failed to generate backup codes", user_id=user.id, error=str(e))
            raise

    async def disable_mfa(self, db: AsyncSession, user: User) -> bool:
        """Disable MFA for user.
        
        Args:
            db: Database session
            user: User to disable MFA for
            
        Returns:
            True if successfully disabled
        """
        try:
            user.mfa_enabled = False
            user.mfa_secret = None
            user.mfa_backup_codes = None
            await db.commit()

            # Clean up any setup sessions
            setup_key = f"mfa_setup:{user.id}"
            await self.cache.delete(setup_key)

            logger.info("MFA disabled", user_id=user.id, username=user.username)
            return True

        except Exception as e:
            logger.error("Failed to disable MFA", user_id=user.id, error=str(e))
            await db.rollback()
            return False

    async def get_mfa_status(self, user: User) -> dict[str, Any]:
        """Get MFA status for user.
        
        Args:
            user: User to check
            
        Returns:
            MFA status information
        """
        try:
            backup_codes_count = 0
            if user.mfa_backup_codes:
                backup_codes = json.loads(user.mfa_backup_codes)
                backup_codes_count = len(backup_codes)

            return {
                "enabled": user.mfa_enabled,
                "method": "totp" if user.mfa_secret else None,
                "backup_codes_remaining": backup_codes_count,
                "backup_codes_low": backup_codes_count <= 2 if user.mfa_enabled else False
            }

        except Exception as e:
            logger.error("Failed to get MFA status", user_id=user.id, error=str(e))
            return {
                "enabled": False,
                "method": None,
                "backup_codes_remaining": 0,
                "backup_codes_low": False
            }

    async def send_mfa_code_sms(self, user: User, phone_number: str) -> bool:
        """Send MFA code via SMS (placeholder for future implementation).
        
        Args:
            user: User to send code to
            phone_number: Phone number to send to
            
        Returns:
            True if SMS sent successfully
        """
        # Placeholder for SMS implementation
        # In production, integrate with SMS service (Twilio, AWS SNS, etc.)

        # Generate 6-digit code
        code = ''.join(secrets.choice(string.digits) for _ in range(6))

        # Store code in cache for verification
        sms_key = f"sms_mfa:{user.id}"
        sms_data = {
            "code": code,
            "phone_number": phone_number,
            "created_at": datetime.now(UTC).isoformat(),
            "attempts": 0
        }

        # Store for 5 minutes
        await self.cache.set_json(sms_key, sms_data, 300)

        logger.info(
            "SMS MFA code generated (not sent - implementation pending)",
            user_id=user.id,
            phone=phone_number[-4:]  # Log only last 4 digits
        )

        # Return True to indicate "sent" (placeholder)
        return True

    async def verify_sms_code(self, user: User, code: str) -> bool:
        """Verify SMS MFA code (placeholder for future implementation).
        
        Args:
            user: User to verify
            code: SMS code to verify
            
        Returns:
            True if verification successful
        """
        try:
            sms_key = f"sms_mfa:{user.id}"
            sms_data = await self.cache.get_json(sms_key)

            if not sms_data:
                logger.warning("No SMS MFA session found", user_id=user.id)
                return False

            # Check attempts limit
            if sms_data.get('attempts', 0) >= 3:
                logger.warning("Too many SMS MFA attempts", user_id=user.id)
                await self.cache.delete(sms_key)
                return False

            # Verify code
            if sms_data.get('code') == code:
                # Clean up
                await self.cache.delete(sms_key)
                logger.info("SMS MFA verification successful", user_id=user.id)
                return True
            else:
                # Increment attempts
                sms_data['attempts'] += 1
                await self.cache.set_json(sms_key, sms_data, 300)
                logger.warning("Invalid SMS MFA code", user_id=user.id)
                return False

        except Exception as e:
            logger.error("SMS MFA verification error", user_id=user.id, error=str(e))
            return False

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired MFA setup sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        # This would typically be handled by Redis TTL
        # Manual cleanup could be implemented if needed
        return 0
