"""
Security validation middleware for comprehensive input validation and attack prevention.

Provides protection against:
- SQL injection attacks
- XSS (Cross-Site Scripting) attacks
- Path traversal attacks
- Command injection attacks
- SSRF (Server-Side Request Forgery) attacks
- XXE (XML External Entity) attacks
- Input size and content validation
"""

import re
from collections.abc import Callable
from typing import Any
from xml.etree.ElementTree import ParseError

import defusedxml.ElementTree as ET
from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ...core.config import get_settings
from ...core.logging import get_logger

logger = get_logger(__name__)


class SecurityValidationMiddleware(BaseHTTPMiddleware):
    """Security validation middleware for comprehensive input sanitization."""

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        re.compile(r"\b(union|select|insert|update|delete|drop|create|alter|exec|execute)\b", re.IGNORECASE),
        re.compile(r"['\";]", re.IGNORECASE),
        re.compile(r"(\-\-|\/\*|\*\/)", re.IGNORECASE),
        re.compile(r"\b(or|and)\s+\d+\s*=\s*\d+", re.IGNORECASE),
        re.compile(r"0x[0-9a-f]+", re.IGNORECASE),
        re.compile(r"\bcast\s*\(", re.IGNORECASE),
        re.compile(r"\bconvert\s*\(", re.IGNORECASE),
        re.compile(r"\bchar\s*\(", re.IGNORECASE),
        re.compile(r"\bnchar\s*\(", re.IGNORECASE),
        re.compile(r"\bascii\s*\(", re.IGNORECASE),
        re.compile(r"\bsubstring\s*\(", re.IGNORECASE),
        re.compile(r"\blen\s*\(", re.IGNORECASE),
        re.compile(r"\bwaitfor\s+delay", re.IGNORECASE),
        re.compile(r"\bbenchmark\s*\(", re.IGNORECASE),
        re.compile(r"\bsleep\s*\(", re.IGNORECASE),
        re.compile(r"\bpg_sleep\s*\(", re.IGNORECASE),
    ]

    # XSS patterns
    XSS_PATTERNS = [
        re.compile(r"<\s*script[^>]*>", re.IGNORECASE),
        re.compile(r"</\s*script\s*>", re.IGNORECASE),
        re.compile(r"javascript\s*:", re.IGNORECASE),
        re.compile(r"vbscript\s*:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<\s*iframe[^>]*>", re.IGNORECASE),
        re.compile(r"<\s*object[^>]*>", re.IGNORECASE),
        re.compile(r"<\s*embed[^>]*>", re.IGNORECASE),
        re.compile(r"<\s*form[^>]*>", re.IGNORECASE),
        re.compile(r"<\s*img[^>]*src\s*=\s*[\"']?javascript:", re.IGNORECASE),
        re.compile(r"<\s*link[^>]*href\s*=\s*[\"']?javascript:", re.IGNORECASE),
        re.compile(r"expression\s*\(", re.IGNORECASE),
        re.compile(r"@import", re.IGNORECASE),
        re.compile(r"url\s*\(\s*[\"']?javascript:", re.IGNORECASE),
        re.compile(r"<\s*meta[^>]*refresh", re.IGNORECASE),
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        re.compile(r"\.\./", re.IGNORECASE),
        re.compile(r"\.\\", re.IGNORECASE),
        re.compile(r"%2e%2e%2f", re.IGNORECASE),
        re.compile(r"%2e%2e/", re.IGNORECASE),
        re.compile(r"\.%2e/", re.IGNORECASE),
        re.compile(r"%2e\./", re.IGNORECASE),
        re.compile(r"%252e%252e%252f", re.IGNORECASE),
        re.compile(r"..\\", re.IGNORECASE),
        re.compile(r"\.\.\\", re.IGNORECASE),
        re.compile(r"%c0%af", re.IGNORECASE),
        re.compile(r"%c1%9c", re.IGNORECASE),
    ]

    # Command injection patterns
    COMMAND_INJECTION_PATTERNS = [
        re.compile(r"[;&|`]", re.IGNORECASE),
        re.compile(r"\$\(", re.IGNORECASE),
        re.compile(r"`.*`", re.IGNORECASE),
        re.compile(r"\|\s*(ls|pwd|id|whoami|cat|more|less|head|tail)", re.IGNORECASE),
        re.compile(r";\s*(ls|pwd|id|whoami|cat|more|less|head|tail)", re.IGNORECASE),
        re.compile(r"&\s*(ls|pwd|id|whoami|cat|more|less|head|tail)", re.IGNORECASE),
        re.compile(r"\b(curl|wget|nc|netcat|telnet|ssh|ftp)\b", re.IGNORECASE),
        re.compile(r"\b(rm|mv|cp|chmod|chown|kill|ps|top)\b", re.IGNORECASE),
        re.compile(r"\b(python|perl|ruby|bash|sh|cmd|powershell)\b", re.IGNORECASE),
    ]

    # SSRF patterns for URLs
    SSRF_PATTERNS = [
        re.compile(r"\b(localhost|127\.0\.0\.1|0\.0\.0\.0)\b", re.IGNORECASE),
        re.compile(r"\b(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)\b", re.IGNORECASE),
        re.compile(r"\b(file|gopher|dict|ftp|sftp|ldap|tftp)://", re.IGNORECASE),
        re.compile(r"\b0x[0-9a-f]{8}\b", re.IGNORECASE),  # Hex encoded IPs
        re.compile(r"\b[0-9]{10}\b", re.IGNORECASE),  # Decimal encoded IPs
    ]

    # Dangerous file extensions
    DANGEROUS_EXTENSIONS = {
        ".exe", ".bat", ".cmd", ".com", ".pif", ".scr", ".vbs", ".js", ".jar",
        ".php", ".php3", ".php4", ".php5", ".phtml", ".asp", ".aspx", ".jsp",
        ".py", ".rb", ".pl", ".sh", ".bash", ".ps1", ".dll", ".so", ".dylib"
    }

    # Safe content types for uploads
    SAFE_CONTENT_TYPES = {
        "image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp",
        "text/plain", "text/csv", "application/json", "application/xml",
        "application/pdf", "video/mp4", "video/avi", "video/mov", "video/webm"
    }

    def __init__(self, app, settings=None):
        super().__init__(app)
        self.settings = settings or get_settings()

        # Security validation settings
        self.max_request_size = 50 * 1024 * 1024  # 50MB
        self.max_query_params = 100
        self.max_headers = 50
        self.max_header_length = 8192
        self.max_url_length = 2048
        self.max_json_depth = 10

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate request for security threats."""
        try:
            # Validate request size
            if await self._validate_request_size(request):
                return self._create_security_error("Request size exceeds limit")

            # Validate URL
            if self._validate_url(request):
                return self._create_security_error("Suspicious URL detected")

            # Validate headers
            if self._validate_headers(request):
                return self._create_security_error("Suspicious headers detected")

            # Validate query parameters
            if self._validate_query_params(request):
                return self._create_security_error("Suspicious query parameters detected")

            # Validate request body for non-multipart requests
            if request.method in ["POST", "PUT", "PATCH"]:
                content_type = request.headers.get("content-type", "").lower()

                if "application/json" in content_type:
                    if await self._validate_json_body(request):
                        return self._create_security_error("Suspicious JSON content detected")

                elif "application/xml" in content_type or "text/xml" in content_type:
                    if await self._validate_xml_body(request):
                        return self._create_security_error("Suspicious XML content detected")

                elif "multipart/form-data" in content_type:
                    # Note: File validation would be done at the endpoint level
                    pass

            # Log security validation passed
            await self._log_security_event(request, "validation_passed", {"result": "clean"})

            return await call_next(request)

        except Exception as e:
            logger.error("Security validation error", error=str(e), path=request.url.path)
            return self._create_security_error("Security validation failed")

    async def _validate_request_size(self, request: Request) -> bool:
        """Validate request size."""
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            await self._log_security_event(
                request, "request_size_violation",
                {"size": content_length, "limit": self.max_request_size}
            )
            return True
        return False

    def _validate_url(self, request: Request) -> bool:
        """Validate URL for path traversal and other attacks."""
        url = str(request.url)
        path = request.url.path

        # Check URL length
        if len(url) > self.max_url_length:
            return True

        # Check for path traversal
        if self._contains_patterns(path, self.PATH_TRAVERSAL_PATTERNS):
            return True

        # Check for SSRF in URL
        return bool(self._contains_patterns(url, self.SSRF_PATTERNS))

    def _validate_headers(self, request: Request) -> bool:
        """Validate HTTP headers."""
        headers = dict(request.headers)

        # Check number of headers
        if len(headers) > self.max_headers:
            return True

        for _name, value in headers.items():
            # Check header length
            if len(value) > self.max_header_length:
                return True

            # Check for XSS in headers
            if self._contains_patterns(value, self.XSS_PATTERNS):
                return True

            # Check for command injection in headers
            if self._contains_patterns(value, self.COMMAND_INJECTION_PATTERNS):
                return True

        return False

    def _validate_query_params(self, request: Request) -> bool:
        """Validate query parameters."""
        query_params = dict(request.query_params)

        # Check number of parameters
        if len(query_params) > self.max_query_params:
            return True

        for _param, value in query_params.items():
            if isinstance(value, str):
                # Check for SQL injection
                if self._contains_patterns(value, self.SQL_INJECTION_PATTERNS):
                    return True

                # Check for XSS
                if self._contains_patterns(value, self.XSS_PATTERNS):
                    return True

                # Check for path traversal
                if self._contains_patterns(value, self.PATH_TRAVERSAL_PATTERNS):
                    return True

                # Check for command injection
                if self._contains_patterns(value, self.COMMAND_INJECTION_PATTERNS):
                    return True

        return False

    async def _validate_json_body(self, request: Request) -> bool:
        """Validate JSON request body."""
        try:
            # Get raw body to avoid consuming the stream
            body = await request.body()
            if not body:
                return False

            # Check for suspicious patterns in raw JSON
            body_str = body.decode('utf-8', errors='ignore')

            # Check for SQL injection
            if self._contains_patterns(body_str, self.SQL_INJECTION_PATTERNS):
                return True

            # Check for XSS
            if self._contains_patterns(body_str, self.XSS_PATTERNS):
                return True

            # Check for command injection
            if self._contains_patterns(body_str, self.COMMAND_INJECTION_PATTERNS):
                return True

            # Check for path traversal
            return bool(self._contains_patterns(body_str, self.PATH_TRAVERSAL_PATTERNS))

        except Exception as e:
            logger.warning("JSON validation error", error=str(e))
            return True

    async def _validate_xml_body(self, request: Request) -> bool:
        """Validate XML request body and prevent XXE attacks."""
        try:
            body = await request.body()
            if not body:
                return False

            body_str = body.decode('utf-8', errors='ignore')

            # Check for XXE patterns
            xxe_patterns = [
                re.compile(r"<!ENTITY", re.IGNORECASE),
                re.compile(r"SYSTEM\s+[\"']", re.IGNORECASE),
                re.compile(r"PUBLIC\s+[\"']", re.IGNORECASE),
                re.compile(r"<!DOCTYPE.*\[", re.IGNORECASE | re.DOTALL),
            ]

            if self._contains_patterns(body_str, xxe_patterns):
                return True

            # Try to parse with defusedxml to catch XXE attempts
            try:
                ET.fromstring(body_str)
            except (ParseError, ET.ParseError):
                # XML parsing failed - suspicious
                return True

            return False

        except Exception as e:
            logger.warning("XML validation error", error=str(e))
            return True

    def _contains_patterns(self, text: str, patterns: list[re.Pattern]) -> bool:
        """Check if text contains any of the given patterns."""
        return any(pattern.search(text) for pattern in patterns)

    def _create_security_error(self, message: str) -> JSONResponse:
        """Create a security error response."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "security_validation_failed",
                "message": message,
                "details": "Request blocked by security validation"
            }
        )

    async def _log_security_event(self, request: Request, event_type: str, details: dict[str, Any]) -> None:
        """Log security events for monitoring."""
        client_ip = self._get_client_ip(request)

        logger.warning(
            "Security validation event",
            event_type=event_type,
            client_ip=client_ip,
            path=request.url.path,
            method=request.method,
            user_agent=request.headers.get("user-agent", ""),
            details=details
        )

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        forwarded = request.headers.get("x-forwarded")
        if forwarded:
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()

        return request.client.host if request.client else "unknown"

    @staticmethod
    def validate_file_upload(filename: str, content_type: str, file_size: int) -> tuple[bool, str]:
        """
        Validate file upload for security.

        Args:
            filename: Name of the uploaded file
            content_type: MIME type of the file
            file_size: Size of the file in bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        max_file_size = 100 * 1024 * 1024  # 100MB

        # Check file size
        if file_size > max_file_size:
            return False, f"File size exceeds limit of {max_file_size // (1024*1024)}MB"

        # Check file extension
        if "." in filename:
            ext = "." + filename.rsplit(".", 1)[1].lower()
            if ext in SecurityValidationMiddleware.DANGEROUS_EXTENSIONS:
                return False, f"File extension '{ext}' is not allowed"

        # Check content type
        if content_type not in SecurityValidationMiddleware.SAFE_CONTENT_TYPES:
            return False, f"Content type '{content_type}' is not allowed"

        # Check for suspicious filenames
        suspicious_patterns = [
            re.compile(r"\.\./", re.IGNORECASE),
            re.compile(r"\.\\", re.IGNORECASE),
            re.compile(r"[<>:\"|?*]", re.IGNORECASE),
        ]

        for pattern in suspicious_patterns:
            if pattern.search(filename):
                return False, "Filename contains suspicious characters"

        return True, ""

    @staticmethod
    def sanitize_html(text: str) -> str:
        """
        Sanitize HTML content to prevent XSS.

        Args:
            text: Input text that may contain HTML

        Returns:
            Sanitized text with dangerous elements removed
        """
        import html

        # HTML encode the text
        sanitized = html.escape(text, quote=True)

        # Additional sanitization for specific patterns
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
        sanitized = re.sub(r'data:', '', sanitized, flags=re.IGNORECASE)

        return sanitized
