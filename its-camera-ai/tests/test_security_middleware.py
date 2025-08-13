"""
Tests for security middleware components.

Tests the comprehensive API security hardening implementation.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from src.its_camera_ai.api.middleware import (
    APIKeyAuthMiddleware,
    CSRFProtectionMiddleware, 
    EnhancedRateLimitMiddleware,
    SecurityHeadersMiddleware,
    SecurityValidationMiddleware,
)
from src.its_camera_ai.api.validators import (
    EmailValidator,
    URLValidator,
    FileValidator,
    ValidationResult,
)
from src.its_camera_ai.core.config import get_settings


class TestSecurityValidationMiddleware:
    """Test security validation middleware."""
    
    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        # Create test app with middleware
        app = FastAPI()
        app.add_middleware(SecurityValidationMiddleware)
        
        @app.get("/test")
        async def test_endpoint(q: str = ""):
            return {"query": q}
        
        client = TestClient(app)
        
        # Test malicious SQL injection attempts
        malicious_queries = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "UNION SELECT * FROM passwords",
            "'; exec xp_cmdshell('dir'); --"
        ]
        
        for query in malicious_queries:
            response = client.get(f"/test?q={query}")
            assert response.status_code == 400
            assert "security_validation_failed" in response.json()["error"]

    def test_xss_detection(self):
        """Test XSS pattern detection."""
        app = FastAPI()
        app.add_middleware(SecurityValidationMiddleware)
        
        @app.post("/test")
        async def test_endpoint(data: dict):
            return data
        
        client = TestClient(app)
        
        # Test XSS attempts
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<iframe src='javascript:alert(1)'></iframe>",
            "<img src=x onerror=alert('xss')>"
        ]
        
        for payload in xss_payloads:
            response = client.post("/test", json={"input": payload})
            assert response.status_code == 400

    def test_path_traversal_detection(self):
        """Test path traversal detection."""
        app = FastAPI()
        app.add_middleware(SecurityValidationMiddleware)
        
        @app.get("/file/{filename}")
        async def get_file(filename: str):
            return {"filename": filename}
        
        client = TestClient(app)
        
        # Test path traversal attempts
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ]
        
        for attempt in traversal_attempts:
            response = client.get(f"/file/{attempt}")
            assert response.status_code == 400

    def test_safe_requests_pass(self):
        """Test that legitimate requests pass through."""
        app = FastAPI()
        app.add_middleware(SecurityValidationMiddleware)
        
        @app.get("/test")
        async def test_endpoint(q: str = ""):
            return {"query": q}
        
        @app.post("/test")
        async def post_test(data: dict):
            return data
        
        client = TestClient(app)
        
        # Test safe requests
        response = client.get("/test?q=normal_query")
        assert response.status_code == 200
        
        response = client.post("/test", json={"name": "John Doe", "age": 30})
        assert response.status_code == 200


class TestSecurityHeaders:
    """Test security headers middleware."""
    
    def test_security_headers_added(self):
        """Test that security headers are added to responses."""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        
        assert "Content-Security-Policy" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "Referrer-Policy" in response.headers

    def test_docs_csp_override(self):
        """Test that docs pages get relaxed CSP."""
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)
        
        @app.get("/docs")
        async def docs_endpoint():
            return {"message": "docs"}
        
        client = TestClient(app)
        response = client.get("/docs")
        
        # Check that docs get more permissive CSP
        csp = response.headers.get("Content-Security-Policy", "")
        assert "cdn.jsdelivr.net" in csp or "unsafe-inline" in csp


class TestEmailValidator:
    """Test email validation."""
    
    def test_valid_emails(self):
        """Test valid email addresses."""
        valid_emails = [
            "user@example.com",
            "test.email@domain.co.uk", 
            "user+tag@example.org",
            "firstname.lastname@company.com"
        ]
        
        for email in valid_emails:
            result = EmailValidator.validate(email, check_dns=False)
            assert result.valid, f"Email {email} should be valid"
            assert result.sanitized_value == email.lower().strip()

    def test_invalid_emails(self):
        """Test invalid email addresses."""
        invalid_emails = [
            "invalid.email",
            "@example.com",
            "user@",
            "user..double.dot@example.com",
            ".leading.dot@example.com",
            "trailing.dot.@example.com"
        ]
        
        for email in invalid_emails:
            result = EmailValidator.validate(email, check_dns=False)
            assert not result.valid, f"Email {email} should be invalid"
            assert len(result.errors) > 0

    def test_disposable_email_detection(self):
        """Test disposable email domain detection."""
        disposable_emails = [
            "user@10minutemail.com",
            "test@tempmail.org", 
            "spam@guerrillamail.com"
        ]
        
        for email in disposable_emails:
            result = EmailValidator.validate(email, check_dns=False, allow_disposable=False)
            assert not result.valid
            assert "disposable" in result.errors[0].lower()


class TestURLValidator:
    """Test URL validation."""
    
    def test_valid_urls(self):
        """Test valid URLs."""
        valid_urls = [
            "https://example.com",
            "http://subdomain.example.org/path?param=value",
            "https://example.com:8080/secure/path",
            "ftp://files.example.com/public/file.txt"
        ]
        
        for url in valid_urls:
            result = URLValidator.validate(url)
            assert result.valid, f"URL {url} should be valid"

    def test_blocked_urls(self):
        """Test blocked URLs."""
        blocked_urls = [
            "http://localhost/admin",
            "https://127.0.0.1/config",
            "http://metadata.google.internal/",
            "https://169.254.169.254/latest/meta-data/"
        ]
        
        for url in blocked_urls:
            result = URLValidator.validate(url, allow_private_ips=False)
            assert not result.valid, f"URL {url} should be blocked"

    def test_https_requirement(self):
        """Test HTTPS requirement."""
        http_url = "http://example.com"
        result = URLValidator.validate(http_url, require_https=True)
        assert not result.valid
        assert "https" in result.errors[0].lower()
        
        https_url = "https://example.com"
        result = URLValidator.validate(https_url, require_https=True)
        assert result.valid


class TestFileValidator:
    """Test file validation."""
    
    def test_safe_file_types(self):
        """Test safe file type validation."""
        safe_files = [
            ("image.jpg", b"\xFF\xD8\xFF", "image/jpeg"),
            ("document.pdf", b"%PDF-1.4", "application/pdf"),
            ("data.json", b'{"valid": "json"}', "application/json")
        ]
        
        for filename, content, mime_type in safe_files:
            result = FileValidator.validate(filename, content, mime_type)
            assert result.valid, f"File {filename} should be valid"

    def test_dangerous_file_extensions(self):
        """Test dangerous file extension detection."""
        dangerous_files = [
            ("malware.exe", b"fake_content", "application/octet-stream"),
            ("script.bat", b"@echo off", "text/plain"),
            ("virus.scr", b"screen_saver", "application/octet-stream")
        ]
        
        for filename, content, mime_type in dangerous_files:
            result = FileValidator.validate(filename, content, mime_type)
            assert not result.valid, f"File {filename} should be rejected"

    def test_file_size_limits(self):
        """Test file size validation."""
        large_content = b"x" * (200 * 1024 * 1024)  # 200MB
        
        result = FileValidator.validate(
            "large.jpg", 
            large_content, 
            "image/jpeg",
            max_size=100 * 1024 * 1024  # 100MB limit
        )
        assert not result.valid
        assert "size exceeds" in result.errors[0].lower()

    def test_malicious_content_detection(self):
        """Test malicious content pattern detection."""
        malicious_content = b'<script>alert("xss")</script>'
        
        result = FileValidator.validate(
            "suspicious.txt",
            malicious_content,
            "text/plain"
        )
        assert not result.valid
        assert "malicious" in result.errors[0].lower()


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = MagicMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.incr = AsyncMock(return_value=1)
    mock_redis.expire = AsyncMock(return_value=True)
    return mock_redis


class TestEnhancedRateLimit:
    """Test enhanced rate limiting middleware."""
    
    def test_rate_limit_headers(self, mock_redis):
        """Test that rate limit headers are added."""
        app = FastAPI()
        app.add_middleware(EnhancedRateLimitMiddleware, redis_client=mock_redis)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}
        
        client = TestClient(app)
        response = client.get("/test")
        
        # Check rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

    def test_different_limits_for_endpoints(self, mock_redis):
        """Test different rate limits for different endpoints."""
        # This would require more complex setup with Redis mocking
        # For now, just ensure the middleware can be instantiated
        app = FastAPI()
        middleware = EnhancedRateLimitMiddleware(app, redis_client=mock_redis)
        
        # Test that different rules exist for different endpoint types
        assert "auth" in middleware.default_rules
        assert "general" in middleware.default_rules
        assert "upload" in middleware.default_rules


def test_integration_security_stack():
    """Test that all security middleware can be stacked together."""
    app = FastAPI()
    
    # Add all security middleware
    app.add_middleware(SecurityValidationMiddleware)
    app.add_middleware(CSRFProtectionMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    client = TestClient(app)
    
    # Test that a legitimate request works
    response = client.get("/test")
    
    # Should have security headers
    assert "X-Content-Type-Options" in response.headers
    assert "Content-Security-Policy" in response.headers
    
    # Should have CSRF token for future requests
    assert "X-CSRF-Token" in response.headers or "csrf_token" in response.cookies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])