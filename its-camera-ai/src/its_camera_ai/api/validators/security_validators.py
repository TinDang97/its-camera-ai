"""
Security validators for comprehensive input validation.

Provides validation for:
- Email validation with DNS checking
- URL validation with blacklist
- File upload validation (type, size, content)
- JSON schema validation
- Phone number validation
- Credit card validation (if needed)
- IP address validation
- Domain validation
"""

import json
import re
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import dns.resolver
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Validation result model."""
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sanitized_value: Any = None


class EmailValidator:
    """Comprehensive email validation."""

    # RFC 5322 compliant email regex (simplified but secure)
    EMAIL_REGEX = re.compile(
        r'^[a-zA-Z0-9.!#$%&\'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$'
    )

    # Disposable email domains to block
    DISPOSABLE_DOMAINS = {
        '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
        'mailinator.com', 'yopmail.com', 'temp-mail.org',
        'throwaway.email', 'getnada.com', 'maildrop.cc'
    }

    # Suspicious patterns
    SUSPICIOUS_PATTERNS = [
        re.compile(r'\+.*\+.*@'),  # Multiple plus signs
        re.compile(r'\.{2,}'),     # Multiple consecutive dots
        re.compile(r'^\.'),        # Starts with dot
        re.compile(r'\.$'),        # Ends with dot
    ]

    @staticmethod
    def validate(email: str, check_dns: bool = True, allow_disposable: bool = False) -> ValidationResult:
        """
        Validate email address.

        Args:
            email: Email address to validate
            check_dns: Whether to check DNS MX records
            allow_disposable: Whether to allow disposable email domains

        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(valid=True, sanitized_value=email.lower().strip())

        if not email:
            result.valid = False
            result.errors.append("Email is required")
            return result

        # Normalize email
        email = email.lower().strip()

        # Basic format validation
        if not EmailValidator.EMAIL_REGEX.match(email):
            result.valid = False
            result.errors.append("Invalid email format")
            return result

        # Length validation
        if len(email) > 254:  # RFC 5321 limit
            result.valid = False
            result.errors.append("Email address too long")
            return result

        # Split into local and domain parts
        try:
            local_part, domain = email.rsplit('@', 1)
        except ValueError:
            result.valid = False
            result.errors.append("Invalid email format")
            return result

        # Local part validation
        if len(local_part) > 64:  # RFC 5321 limit
            result.valid = False
            result.errors.append("Local part too long")
            return result

        # Check for suspicious patterns
        for pattern in EmailValidator.SUSPICIOUS_PATTERNS:
            if pattern.search(email):
                result.warnings.append("Email contains suspicious patterns")
                break

        # Check for disposable domains
        if not allow_disposable and domain in EmailValidator.DISPOSABLE_DOMAINS:
            result.valid = False
            result.errors.append("Disposable email addresses are not allowed")
            return result

        # DNS validation
        if check_dns:
            dns_result = EmailValidator._check_mx_record(domain)
            if not dns_result:
                result.valid = False
                result.errors.append("Email domain does not exist or has no mail server")

        result.sanitized_value = email
        return result

    @staticmethod
    def _check_mx_record(domain: str) -> bool:
        """Check if domain has MX record."""
        try:
            mx_records = dns.resolver.resolve(domain, 'MX')
            return len(mx_records) > 0
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, Exception):
            # Fall back to A record check
            try:
                dns.resolver.resolve(domain, 'A')
                return True
            except Exception:
                return False


class URLValidator:
    """Comprehensive URL validation with security checks."""

    # Allowed schemes
    ALLOWED_SCHEMES = {'http', 'https', 'ftp', 'ftps'}

    # Blocked domains/IPs
    BLOCKED_DOMAINS = {
        'localhost', '127.0.0.1', '0.0.0.0',
        'metadata.google.internal',  # GCP metadata
        '169.254.169.254',           # AWS/Azure metadata
        'example.com', 'example.org', 'example.net'
    }

    # Private IP ranges (RFC 1918)
    PRIVATE_IP_RANGES = [
        (10, 0, 0, 0, 8),           # 10.0.0.0/8
        (172, 16, 0, 0, 12),        # 172.16.0.0/12
        (192, 168, 0, 0, 16),       # 192.168.0.0/16
        (127, 0, 0, 0, 8),          # 127.0.0.0/8 (loopback)
    ]

    @staticmethod
    def validate(url: str, allow_private_ips: bool = False, require_https: bool = False) -> ValidationResult:
        """
        Validate URL with security checks.

        Args:
            url: URL to validate
            allow_private_ips: Whether to allow private IP addresses
            require_https: Whether to require HTTPS scheme

        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(valid=True, sanitized_value=url.strip())

        if not url:
            result.valid = False
            result.errors.append("URL is required")
            return result

        url = url.strip()

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            result.valid = False
            result.errors.append(f"Invalid URL format: {str(e)}")
            return result

        # Check scheme
        if parsed.scheme.lower() not in URLValidator.ALLOWED_SCHEMES:
            result.valid = False
            result.errors.append(f"Scheme '{parsed.scheme}' is not allowed")
            return result

        # Check HTTPS requirement
        if require_https and parsed.scheme.lower() != 'https':
            result.valid = False
            result.errors.append("HTTPS is required")
            return result

        # Check hostname
        if not parsed.hostname:
            result.valid = False
            result.errors.append("URL must have a hostname")
            return result

        hostname = parsed.hostname.lower()

        # Check blocked domains
        if hostname in URLValidator.BLOCKED_DOMAINS:
            result.valid = False
            result.errors.append(f"Domain '{hostname}' is blocked")
            return result

        # Check for IP address
        if URLValidator._is_ip_address(hostname):
            if not allow_private_ips and URLValidator._is_private_ip(hostname):
                result.valid = False
                result.errors.append("Private IP addresses are not allowed")
                return result

        # Additional security checks
        if '..' in url:
            result.warnings.append("URL contains path traversal patterns")

        if len(url) > 2048:
            result.warnings.append("URL is unusually long")

        result.sanitized_value = url
        return result

    @staticmethod
    def _is_ip_address(hostname: str) -> bool:
        """Check if hostname is an IP address."""
        try:
            socket.inet_aton(hostname)
            return True
        except OSError:
            return False

    @staticmethod
    def _is_private_ip(ip_str: str) -> bool:
        """Check if IP address is in private range."""
        try:
            ip_parts = [int(part) for part in ip_str.split('.')]
            if len(ip_parts) != 4:
                return False

            for range_start in URLValidator.PRIVATE_IP_RANGES:
                a, b, c, d, prefix_len = range_start
                mask = (0xffffffff << (32 - prefix_len)) & 0xffffffff

                network = (a << 24) | (b << 16) | (c << 8) | d
                ip_int = (ip_parts[0] << 24) | (ip_parts[1] << 16) | (ip_parts[2] << 8) | ip_parts[3]

                if (ip_int & mask) == (network & mask):
                    return True

            return False

        except (ValueError, IndexError):
            return False


class FileValidator:
    """File upload validation."""

    # Safe MIME types
    SAFE_MIME_TYPES = {
        'image/jpeg', 'image/png', 'image/gif', 'image/webp',
        'image/bmp', 'image/svg+xml', 'image/tiff',
        'video/mp4', 'video/avi', 'video/mov', 'video/wmv',
        'video/webm', 'video/mkv', 'video/flv',
        'text/plain', 'text/csv', 'application/json',
        'application/xml', 'application/pdf',
        'application/zip', 'application/x-zip-compressed'
    }

    # Dangerous file extensions
    DANGEROUS_EXTENSIONS = {
        '.exe', '.bat', '.cmd', '.com', '.pif', '.scr',
        '.vbs', '.vbe', '.js', '.jse', '.jar', '.ws',
        '.wsf', '.wsc', '.wsh', '.ps1', '.ps1xml',
        '.ps2', '.ps2xml', '.psc1', '.psc2', '.msh',
        '.msh1', '.msh2', '.mshxml', '.msh1xml',
        '.msh2xml', '.scf', '.lnk', '.inf', '.reg'
    }

    # Magic numbers for file type detection
    MAGIC_NUMBERS = {
        b'\xFF\xD8\xFF': 'image/jpeg',
        b'\x89\x50\x4E\x47': 'image/png',
        b'\x47\x49\x46\x38': 'image/gif',
        b'\x52\x49\x46\x46': 'video/avi',  # Also WAV
        b'\x00\x00\x00\x18\x66\x74\x79\x70': 'video/mp4',
        b'\x25\x50\x44\x46': 'application/pdf',
        b'\x50\x4B\x03\x04': 'application/zip',
        b'\x50\x4B\x05\x06': 'application/zip',
        b'\x50\x4B\x07\x08': 'application/zip'
    }

    @staticmethod
    def validate(
        filename: str,
        content: bytes,
        declared_mime_type: str,
        max_size: int = 100 * 1024 * 1024  # 100MB
    ) -> ValidationResult:
        """
        Validate file upload.

        Args:
            filename: Original filename
            content: File content as bytes
            declared_mime_type: MIME type from HTTP header
            max_size: Maximum file size in bytes

        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(valid=True)

        # Filename validation
        if not filename:
            result.valid = False
            result.errors.append("Filename is required")
            return result

        # Sanitize filename
        sanitized_filename = FileValidator._sanitize_filename(filename)
        result.sanitized_value = sanitized_filename

        # Check dangerous extensions
        file_ext = Path(filename).suffix.lower()
        if file_ext in FileValidator.DANGEROUS_EXTENSIONS:
            result.valid = False
            result.errors.append(f"File extension '{file_ext}' is not allowed")
            return result

        # Size validation
        if len(content) > max_size:
            result.valid = False
            result.errors.append(f"File size exceeds maximum of {max_size // (1024*1024)}MB")
            return result

        # MIME type validation
        if declared_mime_type not in FileValidator.SAFE_MIME_TYPES:
            result.valid = False
            result.errors.append(f"MIME type '{declared_mime_type}' is not allowed")
            return result

        # Magic number validation
        detected_mime = FileValidator._detect_mime_type(content)
        if detected_mime and detected_mime != declared_mime_type:
            result.warnings.append(f"Declared MIME type '{declared_mime_type}' doesn't match detected type '{detected_mime}'")

        # Content validation
        if FileValidator._contains_malicious_patterns(content):
            result.valid = False
            result.errors.append("File contains potentially malicious content")
            return result

        return result

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent directory traversal."""
        # Remove path components
        filename = Path(filename).name

        # Remove or replace dangerous characters
        filename = re.sub(r'[<>:"|?*]', '_', filename)
        filename = re.sub(r'\.{2,}', '.', filename)
        filename = filename.strip('. ')

        # Ensure filename is not empty
        if not filename:
            filename = 'unnamed_file'

        return filename

    @staticmethod
    def _detect_mime_type(content: bytes) -> str | None:
        """Detect MIME type from file content magic numbers."""
        for magic_bytes, mime_type in FileValidator.MAGIC_NUMBERS.items():
            if content.startswith(magic_bytes):
                return mime_type
        return None

    @staticmethod
    def _contains_malicious_patterns(content: bytes) -> bool:
        """Check for malicious patterns in file content."""
        try:
            # Convert to string for pattern matching (first 1KB)
            text_content = content[:1024].decode('utf-8', errors='ignore').lower()

            # Malicious patterns
            malicious_patterns = [
                '<script', 'javascript:', 'vbscript:',
                'onload=', 'onerror=', 'onclick=',
                '<?php', '<%', 'eval(', 'exec(',
                'shell_exec', 'system(', 'passthru(',
                'file_get_contents', 'file_put_contents'
            ]

            for pattern in malicious_patterns:
                if pattern in text_content:
                    return True

        except Exception:
            pass

        return False


class JSONValidator:
    """JSON validation with security checks."""

    @staticmethod
    def validate(
        json_str: str,
        max_depth: int = 10,
        max_length: int = 1024 * 1024,  # 1MB
        schema: dict | None = None
    ) -> ValidationResult:
        """
        Validate JSON string.

        Args:
            json_str: JSON string to validate
            max_depth: Maximum nesting depth
            max_length: Maximum JSON string length
            schema: JSON schema for validation (optional)

        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(valid=True)

        if not json_str:
            result.valid = False
            result.errors.append("JSON is required")
            return result

        # Length check
        if len(json_str) > max_length:
            result.valid = False
            result.errors.append(f"JSON too large (max {max_length} bytes)")
            return result

        # Parse JSON
        try:
            data = json.loads(json_str)
            result.sanitized_value = data
        except json.JSONDecodeError as e:
            result.valid = False
            result.errors.append(f"Invalid JSON: {str(e)}")
            return result

        # Depth check
        actual_depth = JSONValidator._calculate_depth(data)
        if actual_depth > max_depth:
            result.valid = False
            result.errors.append(f"JSON too deeply nested (max depth {max_depth})")
            return result

        # Schema validation
        if schema:
            try:
                import jsonschema
                jsonschema.validate(data, schema)
            except ImportError:
                result.warnings.append("jsonschema not available for schema validation")
            except jsonschema.ValidationError as e:
                result.valid = False
                result.errors.append(f"Schema validation failed: {str(e)}")
            except Exception as e:
                result.warnings.append(f"Schema validation error: {str(e)}")

        return result

    @staticmethod
    def _calculate_depth(obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of JSON object."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(JSONValidator._calculate_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(JSONValidator._calculate_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth


class PhoneValidator:
    """Phone number validation."""

    # International phone regex (E.164 format)
    PHONE_REGEX = re.compile(r'^\+?[1-9]\d{1,14}$')

    # US phone regex
    US_PHONE_REGEX = re.compile(r'^(\+1)?[-.\s]?(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})$')

    @staticmethod
    def validate(phone: str, country_code: str = 'US') -> ValidationResult:
        """
        Validate phone number.

        Args:
            phone: Phone number to validate
            country_code: Country code for validation

        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(valid=True)

        if not phone:
            result.valid = False
            result.errors.append("Phone number is required")
            return result

        # Clean phone number
        cleaned_phone = re.sub(r'[^\d+]', '', phone)
        result.sanitized_value = cleaned_phone

        # Country-specific validation
        if country_code == 'US':
            if not PhoneValidator.US_PHONE_REGEX.match(phone):
                result.valid = False
                result.errors.append("Invalid US phone number format")
                return result
        else:
            # International format
            if not PhoneValidator.PHONE_REGEX.match(cleaned_phone):
                result.valid = False
                result.errors.append("Invalid international phone number format")
                return result

        return result


class CreditCardValidator:
    """Credit card validation (if needed)."""

    @staticmethod
    def validate(card_number: str) -> ValidationResult:
        """
        Validate credit card number using Luhn algorithm.

        Args:
            card_number: Credit card number

        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(valid=True)

        if not card_number:
            result.valid = False
            result.errors.append("Card number is required")
            return result

        # Clean card number
        cleaned_number = re.sub(r'\D', '', card_number)
        result.sanitized_value = '****-****-****-' + cleaned_number[-4:]

        # Length check
        if len(cleaned_number) < 13 or len(cleaned_number) > 19:
            result.valid = False
            result.errors.append("Invalid card number length")
            return result

        # Luhn algorithm
        if not CreditCardValidator._luhn_check(cleaned_number):
            result.valid = False
            result.errors.append("Invalid card number")
            return result

        return result

    @staticmethod
    def _luhn_check(card_number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        def luhn_digit(n):
            return sum(divmod(n * 2, 10))

        digits = [int(d) for d in card_number]
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]

        checksum = sum(odd_digits) + sum(luhn_digit(d) for d in even_digits)
        return checksum % 10 == 0


class IPAddressValidator:
    """IP address validation."""

    @staticmethod
    def validate(ip_address: str, allow_private: bool = True) -> ValidationResult:
        """
        Validate IP address.

        Args:
            ip_address: IP address to validate
            allow_private: Whether to allow private IP addresses

        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(valid=True, sanitized_value=ip_address.strip())

        if not ip_address:
            result.valid = False
            result.errors.append("IP address is required")
            return result

        ip_address = ip_address.strip()

        try:
            import ipaddress
            ip_obj = ipaddress.ip_address(ip_address)

            # Check if private
            if not allow_private and ip_obj.is_private:
                result.valid = False
                result.errors.append("Private IP addresses are not allowed")
                return result

            # Additional checks
            if ip_obj.is_loopback:
                result.warnings.append("Loopback IP address")

            if ip_obj.is_multicast:
                result.warnings.append("Multicast IP address")

        except ValueError as e:
            result.valid = False
            result.errors.append(f"Invalid IP address: {str(e)}")

        return result


# Convenience function for common validations
def validate_input(input_type: str, value: Any, **kwargs) -> ValidationResult:
    """
    Validate input based on type.

    Args:
        input_type: Type of validation ('email', 'url', 'phone', etc.)
        value: Value to validate
        **kwargs: Additional validation parameters

    Returns:
        ValidationResult
    """
    validators = {
        'email': EmailValidator.validate,
        'url': URLValidator.validate,
        'phone': PhoneValidator.validate,
        'credit_card': CreditCardValidator.validate,
        'ip_address': IPAddressValidator.validate,
        'json': JSONValidator.validate,
    }

    validator = validators.get(input_type)
    if not validator:
        return ValidationResult(
            valid=False,
            errors=[f"Unknown validation type: {input_type}"]
        )

    return validator(value, **kwargs)
