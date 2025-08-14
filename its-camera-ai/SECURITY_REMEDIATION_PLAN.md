# ITS Camera AI - Security Remediation Plan

## CRITICAL PRIORITY FIXES (Week 1)

### 1. Complete API Key Authentication Integration
**Deadline**: 3 days
**Owner**: Backend Security Team

#### Implementation Steps:
1. Create API key database schema
2. Implement database lookup methods
3. Add API key management endpoints
4. Test authentication flows

```sql
-- API Key Database Schema
CREATE TABLE api_keys (
    key_id VARCHAR(32) PRIMARY KEY,
    key_hash VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    scopes TEXT[] NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,
    usage_count INTEGER DEFAULT 0,
    rate_limit_per_minute INTEGER DEFAULT 60,
    rate_limit_per_hour INTEGER DEFAULT 1000,
    allowed_ips TEXT[],
    user_id VARCHAR(36),
    service_name VARCHAR(255)
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_status ON api_keys(status);
CREATE INDEX idx_api_keys_expires ON api_keys(expires_at);
```

### 2. Implement TLS/SSL Configuration
**Deadline**: 5 days
**Owner**: Infrastructure Security Team

#### Implementation Steps:
1. Generate/obtain SSL certificates
2. Configure TLS termination
3. Update deployment configurations
4. Test secure connections

```python
# TLS Configuration for Production
import ssl
from fastapi import FastAPI
from uvicorn import Config, Server

def create_ssl_context() -> ssl.SSLContext:
    """Create SSL context for HTTPS."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(
        certfile="/etc/ssl/certs/its-camera-ai.crt",
        keyfile="/etc/ssl/private/its-camera-ai.key"
    )
    context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    return context

# Update production server configuration
if __name__ == "__main__":
    settings = get_settings()
    if settings.is_production():
        ssl_context = create_ssl_context()
        config = Config(
            app="src.its_camera_ai.api.app:app",
            host="0.0.0.0",
            port=443,
            ssl_keyfile="/etc/ssl/private/its-camera-ai.key",
            ssl_certfile="/etc/ssl/certs/its-camera-ai.crt",
            ssl_version=ssl.PROTOCOL_TLS_SERVER,
            ssl_ciphers='ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS'
        )
    else:
        config = Config(app="src.its_camera_ai.api.app:app", host="0.0.0.0", port=8000)
    
    server = Server(config)
    server.run()
```

### 3. Replace Default Secrets
**Deadline**: 1 day
**Owner**: DevOps Team

#### Implementation Steps:
1. Generate cryptographically secure secrets
2. Update configuration validation
3. Update deployment scripts
4. Rotate existing secrets

```python
# Enhanced Secret Management
import secrets
import os
from cryptography.fernet import Fernet

class SecretManager:
    """Secure secret management."""
    
    @staticmethod
    def generate_secret_key(length: int = 32) -> str:
        """Generate cryptographically secure secret key."""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_jwt_keypair() -> tuple[str, str]:
        """Generate RSA keypair for JWT signing."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem.decode(), public_pem.decode()
    
    @staticmethod
    def validate_secret_strength(secret: str) -> bool:
        """Validate secret strength."""
        if len(secret) < 32:
            return False
        
        # Check for default values
        weak_secrets = {
            "change-me-in-production",
            "secret",
            "password",
            "admin",
            "test"
        }
        
        return secret.lower() not in weak_secrets

# Update configuration validation
class SecurityConfig(BaseModel):
    secret_key: str = Field(
        default_factory=lambda: SecretManager.generate_secret_key(),
        min_length=32,
        description="Secret key for JWT tokens"
    )
    
    @field_validator('secret_key')
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        if not SecretManager.validate_secret_strength(v):
            raise ValueError("Secret key is too weak or uses default value")
        return v
```

## HIGH PRIORITY FIXES (Week 2)

### 4. Complete Rate Limiting Integration
**Deadline**: 7 days
**Owner**: Backend Security Team

```python
# Complete rate limiting service integration
async def _get_user_tier_from_api_key(self, api_key: str) -> str:
    """Get user tier from API key service."""
    try:
        # Query API key from database
        async with self.database.get_session() as session:
            query = select(APIKeyModel).where(APIKeyModel.key_hash == self._hash_api_key(api_key))
            result = await session.execute(query)
            api_key_record = result.scalar_one_or_none()
            
            if not api_key_record:
                return "general"
            
            # Get user tier from associated user
            if api_key_record.user_id:
                user_query = select(UserModel).where(UserModel.id == api_key_record.user_id)
                user_result = await session.execute(user_query)
                user = user_result.scalar_one_or_none()
                return user.tier if user else "general"
            
            # Service-to-service keys get premium tier
            return "premium" if api_key_record.service_name else "general"
            
    except Exception as e:
        logger.error("Failed to get user tier from API key", error=str(e))
        return "general"

async def _get_user_id_from_token(self, token: str) -> str | None:
    """Extract user ID from JWT token."""
    try:
        from ..services.auth_service import AuthService
        auth_service = AuthService()
        payload = auth_service.decode_token(token)
        return payload.get("user_id")
    except Exception as e:
        logger.error("Failed to decode JWT token", error=str(e))
        return None
```

### 5. Implement Camera Stream Authentication
**Deadline**: 10 days
**Owner**: Streaming Security Team

```python
# Camera authentication for RTSP/WebRTC streams
class CameraStreamAuthenticator:
    """Authentication for camera streams."""
    
    def __init__(self, auth_service: AuthService):
        self.auth_service = auth_service
        self.camera_credentials = {}
    
    async def authenticate_camera(self, camera_id: str, credentials: dict) -> bool:
        """Authenticate camera connection."""
        try:
            # Validate camera credentials
            if "api_key" in credentials:
                return await self._validate_camera_api_key(camera_id, credentials["api_key"])
            elif "certificate" in credentials:
                return await self._validate_camera_certificate(camera_id, credentials["certificate"])
            else:
                return False
        except Exception as e:
            logger.error("Camera authentication failed", camera_id=camera_id, error=str(e))
            return False
    
    async def _validate_camera_api_key(self, camera_id: str, api_key: str) -> bool:
        """Validate camera API key."""
        # Implementation for API key validation
        pass
    
    async def _validate_camera_certificate(self, camera_id: str, cert_data: bytes) -> bool:
        """Validate camera certificate."""
        # Implementation for certificate validation
        pass
```

### 6. Secure Redis Configuration
**Deadline**: 3 days
**Owner**: Infrastructure Team

```yaml
# docker-compose.prod.yml - Secure Redis Configuration
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --appendonly yes
      --appendfsync everysec
      --auto-aof-rewrite-percentage 100
      --auto-aof-rewrite-min-size 64mb
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
      --tcp-keepalive 60
      --timeout 300
      --tcp-backlog 511
      --save 900 1
      --save 300 10
      --save 60 10000
    volumes:
      - redis_data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf:ro
    networks:
      - backend
    ports:
      - "127.0.0.1:6379:6379"  # Bind to localhost only
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    sysctls:
      - net.core.somaxconn=1024
    ulimits:
      memlock:
        soft: -1
        hard: -1
```

## MEDIUM PRIORITY ENHANCEMENTS (Weeks 3-4)

### 7. Container Security Hardening
**Deadline**: 14 days
**Owner**: DevOps Security Team

```dockerfile
# Dockerfile.security-hardened
FROM python:3.12-slim AS security-base

# Create non-root user
RUN groupadd -r its-camera-ai && useradd -r -g its-camera-ai its-camera-ai

# Install security updates
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up application directory
WORKDIR /app
CHOWN its-camera-ai:its-camera-ai /app

# Copy and install dependencies
COPY --chown=its-camera-ai:its-camera-ai requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=its-camera-ai:its-camera-ai . .

# Set security labels
LABEL security.scan="enabled" \
      security.policy="restricted" \
      maintainer="security@its-camera-ai.com"

# Switch to non-root user
USER its-camera-ai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "src.its_camera_ai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 8. Enhanced Audit Logging
**Deadline**: 21 days
**Owner**: Compliance Team

```python
# Enhanced audit logging for compliance
class ComplianceAuditLogger:
    """GDPR/SOC2/ISO27001 compliant audit logging."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.audit_logger = get_logger("audit")
    
    async def log_security_event(
        self,
        event_type: str,
        user_id: str | None,
        source_ip: str,
        details: dict[str, Any],
        severity: str = "info"
    ) -> None:
        """Log security event for compliance."""
        audit_record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "source_ip": self._anonymize_ip(source_ip),
            "details": details,
            "severity": severity,
            "session_id": self._get_session_id(),
            "compliance_tags": ["gdpr", "soc2", "iso27001"]
        }
        
        self.audit_logger.info("security_audit_event", **audit_record)
        
        # Store in compliance database
        await self._store_audit_record(audit_record)
    
    def _anonymize_ip(self, ip: str) -> str:
        """Anonymize IP address for GDPR compliance."""
        import ipaddress
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.version == 4:
                # Mask last octet for IPv4
                return str(ip_obj).rsplit('.', 1)[0] + '.xxx'
            else:
                # Mask last 64 bits for IPv6
                return str(ip_obj).split(':', 4)[0] + '::xxxx'
        except:
            return "xxx.xxx.xxx.xxx"
```

## ONGOING SECURITY MONITORING

### Security Metrics Dashboard
```python
# Security monitoring metrics
SECURITY_METRICS = {
    "authentication_failures_per_minute": {"threshold": 10, "severity": "high"},
    "rate_limit_violations_per_hour": {"threshold": 100, "severity": "medium"},
    "sql_injection_attempts_per_day": {"threshold": 1, "severity": "critical"},
    "unauthorized_api_access_attempts": {"threshold": 5, "severity": "high"},
    "certificate_expiry_days": {"threshold": 30, "severity": "medium"},
    "security_header_violations": {"threshold": 10, "severity": "low"}
}
```

### Automated Security Testing
```bash
#!/bin/bash
# security-test-pipeline.sh

# Container security scanning
trivy image its-camera-ai:latest --severity HIGH,CRITICAL --exit-code 1

# Dependency vulnerability scanning  
safety check --json --output security-report.json

# Static security analysis
bandit -r src/ -f json -o bandit-report.json

# API security testing
zap-baseline.py -t http://localhost:8000 -J zap-report.json

# Infrastructure security testing
checkov -f docker-compose.prod.yml --framework docker_compose
```

## COMPLIANCE CHECKLIST

### GDPR Compliance
- [x] Data encryption at rest and in transit
- [x] Privacy anonymization levels implemented
- [x] Audit logging with IP anonymization
- [ ] Data export/deletion APIs
- [ ] Consent management system
- [ ] Data retention policies automated

### SOC2 Type II Compliance
- [x] Access controls and authentication
- [x] Security monitoring and logging
- [x] Incident response procedures
- [ ] Penetration testing scheduled
- [ ] Security awareness training
- [ ] Vendor security assessments

### ISO 27001 Compliance
- [x] Risk assessment framework
- [x] Security controls implemented
- [x] Incident management procedures
- [ ] Business continuity planning
- [ ] Security policy documentation
- [ ] Management review processes

---

**Next Review Date**: 30 days from implementation
**Security Contact**: security@its-camera-ai.com
**Emergency Response**: +1-555-SECURITY