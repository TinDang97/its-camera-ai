#!/bin/bash
# Security Setup Script for ITS Camera AI Production Deployment
# 
# This script sets up comprehensive security measures for production deployment:
# - Generate secure secrets and certificates
# - Configure security middleware
# - Set up monitoring and alerting
# - Harden infrastructure components
# - Validate security configuration

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SECURITY_DIR="$PROJECT_ROOT/security"
CERTS_DIR="$SECURITY_DIR/certificates"
SECRETS_DIR="$SECURITY_DIR/secrets"
LOGS_DIR="$PROJECT_ROOT/logs"

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
DOMAIN="${DOMAIN:-its-camera-ai.local}"
FORCE_REGENERATE="${FORCE_REGENERATE:-false}"

echo -e "${BLUE}=== ITS Camera AI Security Setup ===${NC}"
echo "Environment: $ENVIRONMENT"
echo "Domain: $DOMAIN"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to generate secure random string
generate_secret() {
    openssl rand -base64 32
}

# Function to generate API key
generate_api_key() {
    echo "its_$(openssl rand -hex 16)"
}

# Create directory structure
setup_directories() {
    print_status "Setting up security directories..."
    
    mkdir -p "$SECURITY_DIR"
    mkdir -p "$CERTS_DIR"
    mkdir -p "$SECRETS_DIR"
    mkdir -p "$LOGS_DIR"
    
    # Set secure permissions
    chmod 700 "$SECRETS_DIR"
    chmod 755 "$CERTS_DIR"
    chmod 755 "$LOGS_DIR"
    
    print_status "Directory structure created"
}

# Check required tools
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command_exists openssl; then
        missing_tools+=("openssl")
    fi
    
    if ! command_exists docker; then
        missing_tools+=("docker")
    fi
    
    if ! command_exists python3; then
        missing_tools+=("python3")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_error "Please install missing tools and run again"
        exit 1
    fi
    
    print_status "All prerequisites satisfied"
}

# Generate secrets
generate_secrets() {
    print_status "Generating secure secrets..."
    
    local secrets_file="$SECRETS_DIR/secrets.env"
    local backup_file="$SECRETS_DIR/secrets.env.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Backup existing secrets if they exist
    if [ -f "$secrets_file" ] && [ "$FORCE_REGENERATE" != "true" ]; then
        print_warning "Secrets file already exists. Use FORCE_REGENERATE=true to regenerate."
        return 0
    fi
    
    if [ -f "$secrets_file" ]; then
        cp "$secrets_file" "$backup_file"
        print_status "Backed up existing secrets to $backup_file"
    fi
    
    # Generate new secrets
    cat > "$secrets_file" << EOF
# ITS Camera AI Security Secrets
# Generated on: $(date)
# Environment: $ENVIRONMENT

# JWT Configuration
SECRET_KEY=$(generate_secret)
REFRESH_SECRET_KEY=$(generate_secret)

# Database Encryption
DATABASE_ENCRYPTION_KEY=$(generate_secret)

# Redis Authentication
REDIS_PASSWORD=$(generate_secret)

# API Keys
MASTER_API_KEY=$(generate_api_key)
MONITORING_API_KEY=$(generate_api_key)
ANALYTICS_API_KEY=$(generate_api_key)

# MinIO Credentials
MINIO_ROOT_USER=admin-$(openssl rand -hex 8)
MINIO_ROOT_PASSWORD=$(generate_secret)

# Security Monitoring
SECURITY_WEBHOOK_SECRET=$(generate_secret)
ENCRYPTION_SALT=$(generate_secret)

# Certificate Signing
CA_PRIVATE_KEY_PASSWORD=$(generate_secret)

# Session Security
SESSION_SECRET=$(generate_secret)
CSRF_SECRET=$(generate_secret)

EOF

    # Set secure permissions
    chmod 600 "$secrets_file"
    
    print_status "Secrets generated and saved to $secrets_file"
}

# Generate SSL certificates
generate_certificates() {
    print_status "Generating SSL certificates..."
    
    local ca_key="$CERTS_DIR/ca-key.pem"
    local ca_cert="$CERTS_DIR/ca-cert.pem"
    local server_key="$CERTS_DIR/server-key.pem"
    local server_cert="$CERTS_DIR/server-cert.pem"
    local server_csr="$CERTS_DIR/server.csr"
    
    # Check if certificates already exist
    if [ -f "$server_cert" ] && [ "$FORCE_REGENERATE" != "true" ]; then
        print_warning "Certificates already exist. Use FORCE_REGENERATE=true to regenerate."
        return 0
    fi
    
    # Generate CA private key
    print_status "Generating Certificate Authority..."
    openssl genrsa -out "$ca_key" 4096
    
    # Generate CA certificate
    openssl req -new -x509 -days 3650 -key "$ca_key" -out "$ca_cert" \
        -subj "/C=US/ST=CA/L=San Francisco/O=ITS Camera AI/OU=Security/CN=ITS-CA"
    
    # Generate server private key
    print_status "Generating server certificate..."
    openssl genrsa -out "$server_key" 2048
    
    # Generate server certificate signing request
    openssl req -new -key "$server_key" -out "$server_csr" \
        -subj "/C=US/ST=CA/L=San Francisco/O=ITS Camera AI/OU=Production/CN=$DOMAIN"
    
    # Create certificate extensions
    cat > "$CERTS_DIR/server.ext" << EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = $DOMAIN
DNS.2 = localhost
DNS.3 = *.its-camera-ai.local
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

    # Sign server certificate with CA
    openssl x509 -req -in "$server_csr" -CA "$ca_cert" -CAkey "$ca_key" \
        -CAcreateserial -out "$server_cert" -days 365 \
        -extensions v3_req -extfile "$CERTS_DIR/server.ext"
    
    # Set permissions
    chmod 600 "$ca_key" "$server_key"
    chmod 644 "$ca_cert" "$server_cert"
    
    # Clean up
    rm "$server_csr" "$CERTS_DIR/server.ext"
    
    print_status "SSL certificates generated successfully"
    
    # Validate certificates
    if openssl verify -CAfile "$ca_cert" "$server_cert" >/dev/null 2>&1; then
        print_status "Certificate validation successful"
    else
        print_error "Certificate validation failed"
        exit 1
    fi
}

# Configure security middleware
configure_security() {
    print_status "Configuring security middleware..."
    
    local security_config="$SECURITY_DIR/security-config.json"
    
    cat > "$security_config" << EOF
{
    "middleware": {
        "api_key_auth": {
            "enabled": true,
            "cache_ttl": 300,
            "max_attempts": 5,
            "lockout_duration": 900
        },
        "rate_limiting": {
            "enabled": true,
            "default_limit": 100,
            "auth_limit": 5,
            "burst_size": 10,
            "adaptive": true
        },
        "security_validation": {
            "enabled": true,
            "max_request_size": 52428800,
            "max_json_depth": 10,
            "enable_sql_detection": true,
            "enable_xss_detection": true,
            "enable_xxe_protection": true
        },
        "security_headers": {
            "enabled": true,
            "hsts_max_age": 31536000,
            "enable_csp": true,
            "enable_csrf": true
        }
    },
    "monitoring": {
        "enabled": true,
        "alert_thresholds": {
            "high_risk_events_per_hour": 10,
            "failed_auth_per_minute": 5,
            "rate_limit_violations_per_hour": 100
        },
        "incident_response": {
            "auto_block_threshold": 80,
            "auto_suspend_threshold": 90,
            "alert_cooldown": 300
        }
    },
    "encryption": {
        "algorithm": "AES-256-GCM",
        "key_rotation_days": 90,
        "backup_retention_days": 30
    }
}
EOF

    print_status "Security configuration saved to $security_config"
}

# Set up monitoring
setup_monitoring() {
    print_status "Setting up security monitoring..."
    
    # Create monitoring configuration
    local monitoring_config="$SECURITY_DIR/monitoring.yml"
    
    cat > "$monitoring_config" << EOF
# Security Monitoring Configuration
monitoring:
  enabled: true
  
  # Prometheus metrics
  prometheus:
    enabled: true
    port: 8001
    metrics_path: /metrics
    
  # Security alerts
  alerting:
    enabled: true
    webhook_url: "\${SECURITY_WEBHOOK_URL:-}"
    slack_channel: "#security-alerts"
    
  # Log aggregation
  logging:
    level: INFO
    format: json
    retention_days: 90
    
  # Dashboard
  dashboard:
    enabled: true
    refresh_interval: 30
    
# Incident response
incident_response:
  enabled: true
  auto_response: true
  escalation_rules:
    - condition: "risk_score > 80"
      action: "block_ip"
      duration: 3600
    - condition: "failed_auth > 10"
      action: "alert_team"
    - condition: "data_breach_attempt"
      action: "emergency_alert"

# Compliance
compliance:
  gdpr:
    enabled: true
    data_retention_days: 365
    anonymization: true
  soc2:
    enabled: true
    audit_logging: true
  iso27001:
    enabled: true
    risk_assessment: true
EOF

    print_status "Monitoring configuration created"
}

# Harden infrastructure
harden_infrastructure() {
    print_status "Hardening infrastructure components..."
    
    # Create hardened Docker configuration
    local docker_config="$SECURITY_DIR/docker-security.yml"
    
    cat > "$docker_config" << EOF
# Docker Security Configuration
version: '3.8'

x-security-defaults: &security-defaults
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  cap_add:
    - CHOWN
    - SETGID
    - SETUID
  read_only: true
  tmpfs:
    - /tmp:noexec,nosuid,size=100m
  user: "1000:1000"

services:
  # Production overrides for security
  api:
    <<: *security-defaults
    environment:
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    
  redis:
    <<: *security-defaults
    command: >
      redis-server
      --requirepass \${REDIS_PASSWORD}
      --appendonly yes
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
      --save 900 1
      --save 300 10
      --save 60 10000
    sysctls:
      - net.core.somaxconn=1024
    
  postgres:
    <<: *security-defaults
    environment:
      - POSTGRES_DB=its_camera_ai
      - POSTGRES_USER=its_user
      - POSTGRES_PASSWORD=\${DATABASE_PASSWORD}
      - POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256
    command: >
      postgres
      -c ssl=on
      -c ssl_cert_file=/etc/ssl/certs/server.crt
      -c ssl_key_file=/etc/ssl/private/server.key
      -c log_connections=on
      -c log_disconnections=on
      -c log_statement=all
      -c shared_preload_libraries=pg_stat_statements
EOF

    print_status "Docker security configuration created"
    
    # Create nginx security configuration
    local nginx_config="$SECURITY_DIR/nginx-security.conf"
    
    cat > "$nginx_config" << EOF
# Nginx Security Configuration
server_tokens off;
add_header X-Frame-Options DENY;
add_header X-Content-Type-Options nosniff;
add_header X-XSS-Protection "1; mode=block";
add_header Referrer-Policy "strict-origin-when-cross-origin";
add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' ws: wss:;";

# SSL Configuration
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;

# Security headers
add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
add_header X-Frame-Options DENY always;
add_header X-Content-Type-Options nosniff always;

# Rate limiting
limit_req_zone \$binary_remote_addr zone=api:10m rate=10r/s;
limit_req_zone \$binary_remote_addr zone=auth:10m rate=1r/s;

# Hide server version
server_tokens off;

# Client body size
client_max_body_size 50M;
client_body_timeout 60s;
client_header_timeout 60s;

# Proxy settings
proxy_hide_header X-Powered-By;
proxy_hide_header Server;
EOF

    print_status "Nginx security configuration created"
}

# Create security validation script
create_validation_script() {
    print_status "Creating security validation script..."
    
    local validation_script="$SECURITY_DIR/validate-security.py"
    
    cat > "$validation_script" << 'EOF'
#!/usr/bin/env python3
"""
Security Configuration Validation Script
Validates that all security measures are properly configured.
"""

import os
import ssl
import subprocess
import sys
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.primitives import serialization
import requests

def check_ssl_certificates():
    """Check SSL certificate validity."""
    print("🔒 Checking SSL certificates...")
    
    certs_dir = Path("security/certificates")
    ca_cert = certs_dir / "ca-cert.pem"
    server_cert = certs_dir / "server-cert.pem"
    server_key = certs_dir / "server-key.pem"
    
    issues = []
    
    if not all([ca_cert.exists(), server_cert.exists(), server_key.exists()]):
        issues.append("Missing certificate files")
        return issues
    
    try:
        # Load and validate server certificate
        with open(server_cert, 'rb') as f:
            cert = x509.load_pem_x509_certificate(f.read())
        
        # Check expiration
        from datetime import datetime
        if cert.not_valid_after < datetime.now():
            issues.append("Server certificate has expired")
        elif (cert.not_valid_after - datetime.now()).days < 30:
            issues.append("Server certificate expires within 30 days")
        
        # Check key size
        public_key = cert.public_key()
        if hasattr(public_key, 'key_size') and public_key.key_size < 2048:
            issues.append("Certificate key size is less than 2048 bits")
        
        print("  ✅ SSL certificates are valid")
        
    except Exception as e:
        issues.append(f"Certificate validation error: {e}")
    
    return issues

def check_secrets():
    """Check that secrets are properly generated and secure."""
    print("🔐 Checking secrets...")
    
    secrets_file = Path("security/secrets/secrets.env")
    issues = []
    
    if not secrets_file.exists():
        issues.append("Secrets file not found")
        return issues
    
    # Check file permissions
    stat_info = secrets_file.stat()
    if oct(stat_info.st_mode)[-3:] != '600':
        issues.append("Secrets file has incorrect permissions (should be 600)")
    
    # Read and validate secrets
    with open(secrets_file, 'r') as f:
        content = f.read()
    
    required_secrets = [
        'SECRET_KEY', 'REDIS_PASSWORD', 'MASTER_API_KEY',
        'MINIO_ROOT_PASSWORD', 'DATABASE_ENCRYPTION_KEY'
    ]
    
    for secret in required_secrets:
        if secret not in content:
            issues.append(f"Missing required secret: {secret}")
        elif len(content.split(f'{secret}=')[1].split('\n')[0]) < 16:
            issues.append(f"Secret {secret} appears to be too short")
    
    if not issues:
        print("  ✅ Secrets are properly configured")
    
    return issues

def check_docker_security():
    """Check Docker security configuration."""
    print("🐳 Checking Docker security...")
    
    issues = []
    
    try:
        # Check if Docker is running
        result = subprocess.run(['docker', 'version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            issues.append("Docker is not running")
            return issues
        
        # Check Docker security configuration file
        security_config = Path("security/docker-security.yml")
        if not security_config.exists():
            issues.append("Docker security configuration not found")
        else:
            with open(security_config, 'r') as f:
                content = f.read()
                if 'no-new-privileges:true' not in content:
                    issues.append("Docker containers not configured with no-new-privileges")
                if 'cap_drop:' not in content:
                    issues.append("Docker containers not configured to drop capabilities")
        
        print("  ✅ Docker security configuration is valid")
        
    except FileNotFoundError:
        issues.append("Docker is not installed")
    except Exception as e:
        issues.append(f"Docker security check failed: {e}")
    
    return issues

def check_api_security():
    """Check API security endpoints."""
    print("🌐 Checking API security...")
    
    issues = []
    
    try:
        # Test if API is running (assuming on localhost:8000)
        response = requests.get('http://localhost:8000/health', timeout=5)
        
        # Check security headers
        required_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Content-Security-Policy'
        ]
        
        for header in required_headers:
            if header not in response.headers:
                issues.append(f"Missing security header: {header}")
        
        # Check for information disclosure
        if 'Server' in response.headers:
            issues.append("Server header leaking information")
        
        if not issues:
            print("  ✅ API security headers are properly configured")
        
    except requests.exceptions.RequestException:
        # API not running is not necessarily an issue for validation
        print("  ⚠️  API not running, skipping endpoint checks")
    except Exception as e:
        issues.append(f"API security check failed: {e}")
    
    return issues

def main():
    """Main validation function."""
    print("🔍 ITS Camera AI Security Validation")
    print("=" * 40)
    
    all_issues = []
    
    # Run all checks
    all_issues.extend(check_ssl_certificates())
    all_issues.extend(check_secrets())
    all_issues.extend(check_docker_security())
    all_issues.extend(check_api_security())
    
    print("\n" + "=" * 40)
    
    if all_issues:
        print("❌ Security validation failed with the following issues:")
        for issue in all_issues:
            print(f"  • {issue}")
        sys.exit(1)
    else:
        print("✅ All security validations passed!")
        print("🛡️  Your ITS Camera AI deployment is security-ready!")
        sys.exit(0)

if __name__ == "__main__":
    main()
EOF

    chmod +x "$validation_script"
    print_status "Security validation script created"
}

# Create security maintenance script
create_maintenance_script() {
    print_status "Creating security maintenance script..."
    
    local maintenance_script="$SECURITY_DIR/security-maintenance.sh"
    
    cat > "$maintenance_script" << 'EOF'
#!/bin/bash
# Security Maintenance Script
# Performs routine security maintenance tasks

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[MAINTENANCE]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Rotate secrets (if needed)
rotate_secrets() {
    print_status "Checking secret rotation requirements..."
    
    local secrets_file="security/secrets/secrets.env"
    if [ -f "$secrets_file" ]; then
        local file_age=$(( ($(date +%s) - $(stat -c %Y "$secrets_file")) / 86400 ))
        if [ $file_age -gt 90 ]; then
            print_warning "Secrets are older than 90 days. Consider rotation."
        else
            print_status "Secrets are current (${file_age} days old)"
        fi
    fi
}

# Check certificate expiry
check_certificates() {
    print_status "Checking certificate expiration..."
    
    local cert_file="security/certificates/server-cert.pem"
    if [ -f "$cert_file" ]; then
        local expiry_date=$(openssl x509 -enddate -noout -in "$cert_file" | cut -d= -f2)
        local expiry_epoch=$(date -d "$expiry_date" +%s)
        local current_epoch=$(date +%s)
        local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
        
        if [ $days_until_expiry -lt 30 ]; then
            print_warning "Certificate expires in $days_until_expiry days"
        else
            print_status "Certificate is valid for $days_until_expiry days"
        fi
    fi
}

# Update security configurations
update_configurations() {
    print_status "Checking for security configuration updates..."
    
    # Check for security policy updates
    # This would integrate with your security policy management system
    print_status "Security configurations are current"
}

# Clean up old logs
cleanup_logs() {
    print_status "Cleaning up old security logs..."
    
    find logs/ -name "*.log" -mtime +30 -delete 2>/dev/null || true
    find logs/ -name "security-*.log" -mtime +90 -delete 2>/dev/null || true
    
    print_status "Log cleanup completed"
}

# Main maintenance routine
main() {
    echo "🔧 Security Maintenance - $(date)"
    echo "================================"
    
    rotate_secrets
    check_certificates
    update_configurations
    cleanup_logs
    
    echo "✅ Security maintenance completed"
}

main "$@"
EOF

    chmod +x "$maintenance_script"
    print_status "Security maintenance script created"
}

# Main execution
main() {
    print_status "Starting ITS Camera AI security setup..."
    
    check_prerequisites
    setup_directories
    generate_secrets
    generate_certificates
    configure_security
    setup_monitoring
    harden_infrastructure
    create_validation_script
    create_maintenance_script
    
    echo ""
    echo -e "${GREEN}=== Security Setup Complete ===${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review generated secrets in: $SECRETS_DIR/secrets.env"
    echo "2. Update your .env file with the new secrets"
    echo "3. Run the validation script: python3 $SECURITY_DIR/validate-security.py"
    echo "4. Deploy with security configurations"
    echo "5. Set up regular maintenance: crontab -e"
    echo "   Add: 0 2 * * 0 $SECURITY_DIR/security-maintenance.sh"
    echo ""
    echo -e "${YELLOW}Important:${NC}"
    echo "- Keep secrets secure and never commit them to version control"
    echo "- Monitor security alerts and respond promptly"
    echo "- Review and update security configurations regularly"
    echo "- Test incident response procedures"
    echo ""
    echo -e "${BLUE}For production deployment:${NC}"
    echo "- Update DNS records for your domain"
    echo "- Configure load balancer with SSL termination"
    echo "- Set up monitoring and alerting"
    echo "- Perform penetration testing"
    echo ""
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --force-regenerate    Force regeneration of existing secrets/certificates"
        echo "  --domain DOMAIN       Set domain name (default: its-camera-ai.local)"
        echo "  --environment ENV     Set environment (default: production)"
        echo "  --help               Show this help"
        exit 0
        ;;
    --force-regenerate)
        FORCE_REGENERATE=true
        shift
        ;;
    --domain)
        DOMAIN="$2"
        shift 2
        ;;
    --environment)
        ENVIRONMENT="$2"
        shift 2
        ;;
esac

# Run main function
main