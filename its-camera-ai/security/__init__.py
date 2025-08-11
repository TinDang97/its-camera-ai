"""
ITS Camera AI Security Module.

Comprehensive security architecture for production traffic monitoring system.

Security Components:
- Zero Trust Architecture with encryption and privacy controls
- Vulnerability scanning and security hardening
- Incident response and threat detection
- Compliance automation (GDPR/CCPA/SOC2/ISO27001)
- Production security hardening

Usage:
    from security import create_zero_trust_security_system
    from security.vulnerability_scanner import run_security_assessment
    from security.incident_response import incident_response_system
    from security.production_hardening import run_production_hardening
"""

from .zero_trust_architecture import (
    create_zero_trust_security_system,
    SecurityContext,
    SecurityLevel,
    EncryptionManager,
    PrivacyEngine,
    MultiFactorAuthenticator,
    RoleBasedAccessControl,
    ThreatDetectionEngine,
    SecurityAuditLogger,
    create_security_middleware
)

from .vulnerability_scanner import (
    run_security_assessment,
    VulnerabilityManager,
    VulnerabilitySeverity,
    SecurityScanResult
)

from .incident_response import (
    incident_response_system,
    IncidentSeverity,
    IncidentType,
    SecurityIncident
)

from .production_hardening import (
    run_production_hardening,
    production_hardening,
    HardeningLevel,
    ComplianceFramework
)

__version__ = "1.0.0"
__author__ = "ITS Camera AI Security Team"

# Security module metadata
SECURITY_MODULES = {
    'zero_trust': 'Zero Trust Security Architecture',
    'vulnerability_scanner': 'Vulnerability Assessment and Management',
    'incident_response': 'Security Incident Response System',
    'production_hardening': 'Production Security Hardening'
}

# Security compliance frameworks supported
COMPLIANCE_FRAMEWORKS = [
    'CIS',
    'NIST',
    'SOC2',
    'ISO27001',
    'GDPR',
    'CCPA'
]

# Production security checklist
PRODUCTION_SECURITY_CHECKLIST = [
    "Zero Trust architecture implemented",
    "Multi-factor authentication enabled",
    "Role-based access control configured",
    "Data encryption at rest and in transit",
    "Privacy-preserving video processing",
    "Threat detection and monitoring",
    "Incident response procedures",
    "Vulnerability scanning automated",
    "Security hardening applied",
    "Compliance requirements met",
    "Security audit logging enabled",
    "Network segmentation implemented"
]

def get_security_status():
    """Get overall security module status."""
    return {
        'version': __version__,
        'modules': SECURITY_MODULES,
        'compliance_frameworks': COMPLIANCE_FRAMEWORKS,
        'production_ready': True
    }