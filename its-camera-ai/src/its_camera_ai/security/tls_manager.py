"""
TLS/SSL Configuration Manager for ITS Camera AI.

Provides comprehensive TLS configuration for production security:
- SSL context creation and management
- Certificate validation and rotation
- Cipher suite configuration
- HSTS and security headers
- Certificate monitoring and alerts
"""

import ssl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class TLSConfiguration:
    """TLS configuration and certificate management."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.cert_path = Path("/etc/ssl/certs/its-camera-ai.crt")
        self.key_path = Path("/etc/ssl/private/its-camera-ai.key")
        self.ca_path = Path("/etc/ssl/certs/ca-certificates.crt")

        # Security-focused cipher suites (prioritize forward secrecy)
        self.cipher_suites = [
            "ECDHE-ECDSA-AES256-GCM-SHA384",
            "ECDHE-RSA-AES256-GCM-SHA384",
            "ECDHE-ECDSA-CHACHA20-POLY1305",
            "ECDHE-RSA-CHACHA20-POLY1305",
            "ECDHE-ECDSA-AES128-GCM-SHA256",
            "ECDHE-RSA-AES128-GCM-SHA256",
            "DHE-RSA-AES256-GCM-SHA384",
            "DHE-RSA-AES128-GCM-SHA256",
        ]

    def create_ssl_context(self, purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH) -> ssl.SSLContext:
        """Create secure SSL context for HTTPS connections."""
        try:
            # Create SSL context with secure defaults
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if purpose == ssl.Purpose.SERVER_AUTH else ssl.PROTOCOL_TLS_CLIENT)

            # Set minimum TLS version
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.maximum_version = ssl.TLSVersion.TLSv1_3

            # Configure cipher suites for security
            context.set_ciphers(":".join(self.cipher_suites))

            # Security options
            context.options |= ssl.OP_NO_SSLv2
            context.options |= ssl.OP_NO_SSLv3
            context.options |= ssl.OP_NO_TLSv1
            context.options |= ssl.OP_NO_TLSv1_1
            context.options |= ssl.OP_NO_COMPRESSION
            context.options |= ssl.OP_CIPHER_SERVER_PREFERENCE
            context.options |= ssl.OP_SINGLE_DH_USE
            context.options |= ssl.OP_SINGLE_ECDH_USE

            # Certificate verification settings
            if purpose == ssl.Purpose.CLIENT_AUTH:
                context.verify_mode = ssl.CERT_REQUIRED
                context.check_hostname = True
            else:
                context.verify_mode = ssl.CERT_NONE

            # Load certificates if they exist
            if self.cert_path.exists() and self.key_path.exists():
                context.load_cert_chain(str(self.cert_path), str(self.key_path))
                logger.info("SSL certificates loaded successfully")
            else:
                if self.settings.is_production():
                    raise FileNotFoundError("SSL certificates not found in production environment")
                else:
                    logger.warning("SSL certificates not found - running in development mode")

            # Load CA certificates
            if self.ca_path.exists():
                context.load_verify_locations(str(self.ca_path))

            logger.info("SSL context created successfully",
                       min_version=context.minimum_version.name,
                       max_version=context.maximum_version.name)

            return context

        except Exception as e:
            logger.error("Failed to create SSL context", error=str(e))
            raise

    def validate_certificate(self, cert_path: Path | None = None) -> dict[str, Any]:
        """Validate SSL certificate and return details."""
        cert_file = cert_path or self.cert_path

        if not cert_file.exists():
            return {"valid": False, "error": "Certificate file not found"}

        try:
            with open(cert_file, "rb") as f:
                cert_data = f.read()

            cert = x509.load_pem_x509_certificate(cert_data)

            # Extract certificate information
            subject = cert.subject
            issuer = cert.issuer
            not_before = cert.not_valid_before
            not_after = cert.not_valid_after

            # Calculate days until expiry
            days_until_expiry = (not_after - datetime.now()).days

            # Get subject alternative names
            san_extension = None
            try:
                san_extension = cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
                san_names = [name.value for name in san_extension.value]
            except x509.ExtensionNotFound:
                san_names = []

            # Validate certificate status
            is_expired = datetime.now() > not_after
            is_not_yet_valid = datetime.now() < not_before
            expires_soon = days_until_expiry <= 30  # Warning if expires within 30 days

            validation_result = {
                "valid": not (is_expired or is_not_yet_valid),
                "subject": subject.rfc4514_string(),
                "issuer": issuer.rfc4514_string(),
                "not_before": not_before.isoformat(),
                "not_after": not_after.isoformat(),
                "days_until_expiry": days_until_expiry,
                "san_names": san_names,
                "is_expired": is_expired,
                "is_not_yet_valid": is_not_yet_valid,
                "expires_soon": expires_soon,
                "serial_number": str(cert.serial_number),
                "signature_algorithm": cert.signature_algorithm_oid._name,
            }

            # Log warnings for certificate issues
            if is_expired:
                logger.error("SSL certificate has expired", expiry_date=not_after.isoformat())
            elif expires_soon:
                logger.warning("SSL certificate expires soon",
                             days_remaining=days_until_expiry,
                             expiry_date=not_after.isoformat())

            return validation_result

        except Exception as e:
            logger.error("Certificate validation failed", error=str(e))
            return {"valid": False, "error": str(e)}

    def generate_self_signed_certificate(self,
                                       domains: list[str] | None = None,
                                       key_size: int = 2048,
                                       valid_days: int = 365) -> tuple[bytes, bytes]:
        """Generate self-signed certificate for development/testing."""
        if domains is None:
            domains = ["localhost", "127.0.0.1", "its-camera-ai.local"]

        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )

        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "ITS Camera AI"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Development"),
            x509.NameAttribute(NameOID.COMMON_NAME, domains[0]),
        ])

        # Create certificate with SAN extension
        san_list = []
        for domain in domains:
            try:
                # Try to parse as IP address
                import ipaddress
                ip = ipaddress.ip_address(domain)
                san_list.append(x509.IPAddress(ip))
            except ValueError:
                # Not an IP, treat as DNS name
                san_list.append(x509.DNSName(domain))

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.now()
        ).not_valid_after(
            datetime.now() + timedelta(days=valid_days)
        ).add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False,
        ).add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        ).add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        ).add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
            ]),
            critical=True,
        ).sign(private_key, hashes.SHA256())

        # Serialize certificate and private key
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        logger.info("Self-signed certificate generated",
                   domains=domains,
                   valid_days=valid_days,
                   key_size=key_size)

        return cert_pem, key_pem

    def setup_development_certificates(self) -> bool:
        """Set up self-signed certificates for development."""
        try:
            # Create certificate directories
            self.cert_path.parent.mkdir(parents=True, exist_ok=True)
            self.key_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate certificates
            cert_pem, key_pem = self.generate_self_signed_certificate()

            # Write certificate files
            with open(self.cert_path, "wb") as f:
                f.write(cert_pem)

            with open(self.key_path, "wb") as f:
                f.write(key_pem)

            # Set secure permissions
            self.cert_path.chmod(0o644)
            self.key_path.chmod(0o600)

            logger.info("Development certificates created successfully",
                       cert_path=str(self.cert_path),
                       key_path=str(self.key_path))

            return True

        except Exception as e:
            logger.error("Failed to setup development certificates", error=str(e))
            return False

    def monitor_certificate_expiry(self) -> dict[str, Any]:
        """Monitor certificate expiry and return status."""
        validation = self.validate_certificate()

        if not validation["valid"]:
            return validation

        days_until_expiry = validation["days_until_expiry"]

        # Determine alert level
        if days_until_expiry <= 7:
            alert_level = "critical"
        elif days_until_expiry <= 30:
            alert_level = "warning"
        elif days_until_expiry <= 60:
            alert_level = "info"
        else:
            alert_level = "ok"

        monitoring_result = {
            **validation,
            "alert_level": alert_level,
            "monitoring_timestamp": datetime.now().isoformat(),
            "recommendation": self._get_renewal_recommendation(days_until_expiry)
        }

        # Log appropriate level message
        if alert_level == "critical":
            logger.error("SSL certificate expires very soon", **monitoring_result)
        elif alert_level == "warning":
            logger.warning("SSL certificate expires soon", **monitoring_result)
        elif alert_level == "info":
            logger.info("SSL certificate expires in 60 days", **monitoring_result)

        return monitoring_result

    def _get_renewal_recommendation(self, days_until_expiry: int) -> str:
        """Get certificate renewal recommendation."""
        if days_until_expiry <= 7:
            return "Renew certificate immediately to avoid service disruption"
        elif days_until_expiry <= 30:
            return "Schedule certificate renewal within the next week"
        elif days_until_expiry <= 60:
            return "Plan certificate renewal for the coming month"
        else:
            return "Certificate is valid and does not require immediate attention"

    def get_tls_security_headers(self) -> dict[str, str]:
        """Get TLS-related security headers."""
        headers = {}

        if self.settings.is_production():
            # Strict Transport Security (force HTTPS)
            headers["Strict-Transport-Security"] = f"max-age={self.settings.security.hsts_max_age}; includeSubDomains; preload"

            # Expect-CT header for certificate transparency
            headers["Expect-CT"] = "max-age=86400, enforce"

        return headers

    @staticmethod
    def validate_tls_configuration(ssl_context: ssl.SSLContext) -> dict[str, Any]:
        """Validate TLS configuration for security compliance."""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }

        try:
            # Check minimum TLS version
            if ssl_context.minimum_version < ssl.TLSVersion.TLSv1_2:
                validation_result["errors"].append("Minimum TLS version should be 1.2 or higher")
                validation_result["valid"] = False

            # Check if SSLv2/SSLv3 are disabled
            insecure_options = [ssl.OP_NO_SSLv2, ssl.OP_NO_SSLv3, ssl.OP_NO_TLSv1, ssl.OP_NO_TLSv1_1]
            for option in insecure_options:
                if not (ssl_context.options & option):
                    validation_result["warnings"].append(f"Insecure protocol not explicitly disabled: {option}")

            # Check compression
            if not (ssl_context.options & ssl.OP_NO_COMPRESSION):
                validation_result["warnings"].append("SSL compression not disabled (CRIME attack vulnerability)")

            # Security recommendations
            validation_result["recommendations"].extend([
                "Regularly update SSL certificates before expiry",
                "Monitor certificate transparency logs",
                "Use HSTS headers to enforce HTTPS",
                "Implement certificate pinning for critical connections"
            ])

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"TLS validation error: {str(e)}")

        return validation_result


# Singleton TLS manager instance
_tls_manager: TLSConfiguration | None = None


def get_tls_manager() -> TLSConfiguration:
    """Get the global TLS configuration manager."""
    global _tls_manager
    if _tls_manager is None:
        _tls_manager = TLSConfiguration()
    return _tls_manager


def create_secure_ssl_context() -> ssl.SSLContext:
    """Create a secure SSL context using the TLS manager."""
    return get_tls_manager().create_ssl_context()
