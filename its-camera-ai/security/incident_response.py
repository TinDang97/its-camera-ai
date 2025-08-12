"""
Security Incident Response System for ITS Camera AI Traffic Monitoring Platform.

Automated incident detection, response, and recovery capabilities:
- Real-time threat detection and classification
- Automated containment and mitigation procedures
- Forensics data collection and preservation
- Communication and notification workflows
- Recovery and lessons learned processes

Incident Response Workflow:
1. Detection and Analysis
2. Containment and Eradication
3. Recovery and Post-Incident Activities
4. Communication and Reporting
5. Lessons Learned and Improvement
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels based on NIST guidelines."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident status tracking."""

    DETECTED = "detected"
    ANALYZING = "analyzing"
    CONTAINING = "containing"
    ERADICATING = "eradicating"
    RECOVERING = "recovering"
    RESOLVED = "resolved"
    CLOSED = "closed"


class IncidentType(Enum):
    """Types of security incidents."""

    DATA_BREACH = "data_breach"
    MALWARE = "malware"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DENIAL_OF_SERVICE = "denial_of_service"
    INSIDER_THREAT = "insider_threat"
    PRIVACY_VIOLATION = "privacy_violation"
    SYSTEM_COMPROMISE = "system_compromise"
    COMPLIANCE_VIOLATION = "compliance_violation"


@dataclass
class IncidentEvidence:
    """Evidence collected during incident investigation."""

    evidence_id: str
    evidence_type: str
    source: str
    timestamp: datetime
    data: dict[str, Any]
    hash_value: str
    chain_of_custody: list[str] = field(default_factory=list)

    def add_to_custody_chain(self, handler: str):
        """Add handler to chain of custody."""
        self.chain_of_custody.append(f"{handler}:{datetime.now().isoformat()}")


@dataclass
class SecurityIncident:
    """Security incident data structure."""

    incident_id: str
    incident_type: IncidentType
    severity: IncidentSeverity
    status: IncidentStatus
    title: str
    description: str
    detected_at: datetime
    source_system: str
    affected_systems: list[str]
    indicators: dict[str, Any]
    evidence: list[IncidentEvidence] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    assigned_to: str | None = None
    resolved_at: datetime | None = None
    root_cause: str | None = None
    lessons_learned: list[str] = field(default_factory=list)

    def add_action(self, action: str):
        """Add action to incident timeline."""
        timestamp = datetime.now().isoformat()
        self.actions_taken.append(f"{timestamp}: {action}")

    def update_status(self, new_status: IncidentStatus):
        """Update incident status."""
        self.status = new_status
        self.add_action(f"Status updated to {new_status.value}")


class IncidentDetector:
    """Real-time incident detection system."""

    def __init__(self):
        self.detection_rules = self._load_detection_rules()
        self.active_incidents: dict[str, SecurityIncident] = {}
        self.detection_threshold = {
            IncidentSeverity.CRITICAL: 1,
            IncidentSeverity.HIGH: 3,
            IncidentSeverity.MEDIUM: 10,
            IncidentSeverity.LOW: 50,
        }
        logger.info("Incident detector initialized")

    def _load_detection_rules(self) -> list[dict[str, Any]]:
        """Load incident detection rules."""
        return [
            {
                "rule_id": "multiple_failed_auth",
                "name": "Multiple Failed Authentication Attempts",
                "type": IncidentType.UNAUTHORIZED_ACCESS,
                "severity": IncidentSeverity.HIGH,
                "pattern": "auth_failures > 10 in 5 minutes",
                "indicators": ["failed_login_count", "source_ip", "user_account"],
            },
            {
                "rule_id": "unusual_data_access",
                "name": "Unusual Data Access Pattern",
                "type": IncidentType.DATA_BREACH,
                "severity": IncidentSeverity.CRITICAL,
                "pattern": "data_volume > 1GB in 10 minutes",
                "indicators": ["data_volume", "user_id", "access_pattern"],
            },
            {
                "rule_id": "privilege_escalation",
                "name": "Privilege Escalation Attempt",
                "type": IncidentType.SYSTEM_COMPROMISE,
                "severity": IncidentSeverity.CRITICAL,
                "pattern": "admin_access from non-admin user",
                "indicators": ["user_role", "requested_permissions", "source_system"],
            },
            {
                "rule_id": "gdpr_violation",
                "name": "GDPR Compliance Violation",
                "type": IncidentType.PRIVACY_VIOLATION,
                "severity": IncidentSeverity.HIGH,
                "pattern": "personal_data accessed without consent",
                "indicators": ["data_type", "consent_status", "processing_purpose"],
            },
        ]

    async def analyze_security_event(
        self, event: dict[str, Any]
    ) -> SecurityIncident | None:
        """Analyze security event for potential incidents."""
        for rule in self.detection_rules:
            if await self._evaluate_detection_rule(rule, event):
                incident = await self._create_incident(rule, event)
                self.active_incidents[incident.incident_id] = incident

                logger.warning(
                    "Security incident detected",
                    incident_id=incident.incident_id,
                    type=incident.incident_type.value,
                    severity=incident.severity.value,
                )

                return incident

        return None

    async def _evaluate_detection_rule(
        self, rule: dict[str, Any], event: dict[str, Any]
    ) -> bool:
        """Evaluate if event matches detection rule."""
        # Simplified rule evaluation - in production, implement complex pattern matching
        rule_id = rule["rule_id"]

        if rule_id == "multiple_failed_auth":
            return (
                event.get("event_type") == "auth_failure"
                and event.get("failure_count", 0) > 10
            )

        elif rule_id == "unusual_data_access":
            return (
                event.get("event_type") == "data_access"
                and event.get("data_volume_mb", 0) > 1000
            )

        elif rule_id == "privilege_escalation":
            return (
                event.get("event_type") == "privilege_request"
                and event.get("user_role") != "admin"
                and "admin" in event.get("requested_permissions", [])
            )

        elif rule_id == "gdpr_violation":
            return (
                event.get("event_type") == "data_processing"
                and event.get("personal_data", False)
                and not event.get("consent_given", False)
            )

        return False

    async def _create_incident(
        self, rule: dict[str, Any], event: dict[str, Any]
    ) -> SecurityIncident:
        """Create security incident from detected event."""
        incident_id = f"INC-{datetime.now().strftime('%Y%m%d')}-{int(time.time())}"

        # Collect initial evidence
        evidence = IncidentEvidence(
            evidence_id=f"EVD-{incident_id}-001",
            evidence_type="security_event",
            source=event.get("source", "unknown"),
            timestamp=datetime.now(),
            data=event,
            hash_value=self._calculate_hash(json.dumps(event, sort_keys=True)),
        )
        evidence.add_to_custody_chain("incident_detector")

        incident = SecurityIncident(
            incident_id=incident_id,
            incident_type=rule["type"],
            severity=rule["severity"],
            status=IncidentStatus.DETECTED,
            title=rule["name"],
            description=f"Detected: {rule['pattern']}",
            detected_at=datetime.now(),
            source_system=event.get("source", "its-camera-ai"),
            affected_systems=event.get(
                "affected_systems", [event.get("source", "unknown")]
            ),
            indicators={key: event.get(key) for key in rule["indicators"]},
            evidence=[evidence],
        )

        incident.add_action("Incident detected and created")
        return incident

    def _calculate_hash(self, data: str) -> str:
        """Calculate hash for evidence integrity."""
        import hashlib

        return hashlib.sha256(data.encode()).hexdigest()


class IncidentResponder:
    """Automated incident response system."""

    def __init__(self):
        self.response_playbooks = self._load_response_playbooks()
        self.containment_actions = {
            IncidentType.UNAUTHORIZED_ACCESS: self._contain_unauthorized_access,
            IncidentType.DATA_BREACH: self._contain_data_breach,
            IncidentType.MALWARE: self._contain_malware,
            IncidentType.DENIAL_OF_SERVICE: self._contain_dos_attack,
            IncidentType.SYSTEM_COMPROMISE: self._contain_system_compromise,
            IncidentType.PRIVACY_VIOLATION: self._contain_privacy_violation,
        }
        logger.info("Incident responder initialized")

    def _load_response_playbooks(self) -> dict[IncidentType, list[str]]:
        """Load incident response playbooks."""
        return {
            IncidentType.UNAUTHORIZED_ACCESS: [
                "Block suspicious IP addresses",
                "Disable compromised user accounts",
                "Force password reset for affected users",
                "Review access logs",
                "Enable enhanced monitoring",
            ],
            IncidentType.DATA_BREACH: [
                "Isolate affected systems",
                "Preserve forensic evidence",
                "Assess data exposure scope",
                "Notify data protection officer",
                "Prepare breach notification",
            ],
            IncidentType.PRIVACY_VIOLATION: [
                "Stop unauthorized data processing",
                "Assess GDPR compliance impact",
                "Document privacy violation details",
                "Notify data protection authorities",
                "Implement corrective measures",
            ],
            IncidentType.SYSTEM_COMPROMISE: [
                "Isolate compromised systems",
                "Collect forensic images",
                "Analyze attack vectors",
                "Remove malicious artifacts",
                "Rebuild affected systems",
            ],
        }

    async def respond_to_incident(self, incident: SecurityIncident) -> bool:
        """Execute automated response to security incident."""
        try:
            # Update incident status
            incident.update_status(IncidentStatus.CONTAINING)

            # Execute containment actions
            await self._execute_containment(incident)

            # Collect additional evidence
            await self._collect_forensic_evidence(incident)

            # Execute eradication
            incident.update_status(IncidentStatus.ERADICATING)
            await self._execute_eradication(incident)

            # Begin recovery
            incident.update_status(IncidentStatus.RECOVERING)
            await self._execute_recovery(incident)

            # Mark as resolved
            incident.update_status(IncidentStatus.RESOLVED)
            incident.resolved_at = datetime.now()

            logger.info(
                "Incident response completed",
                incident_id=incident.incident_id,
                duration_minutes=(datetime.now() - incident.detected_at).total_seconds()
                / 60,
            )

            return True

        except Exception as e:
            logger.error(
                "Incident response failed",
                incident_id=incident.incident_id,
                error=str(e),
            )
            incident.add_action(f"Response failed: {str(e)}")
            return False

    async def _execute_containment(self, incident: SecurityIncident):
        """Execute containment actions for incident."""
        containment_func = self.containment_actions.get(incident.incident_type)

        if containment_func:
            await containment_func(incident)
        else:
            # Generic containment
            incident.add_action("Executed generic containment procedures")
            logger.info(
                "Generic containment executed", incident_id=incident.incident_id
            )

    async def _contain_unauthorized_access(self, incident: SecurityIncident):
        """Contain unauthorized access incident."""
        # Block suspicious IP addresses
        source_ip = incident.indicators.get("source_ip")
        if source_ip:
            await self._block_ip_address(source_ip)
            incident.add_action(f"Blocked IP address: {source_ip}")

        # Disable compromised accounts
        user_account = incident.indicators.get("user_account")
        if user_account:
            await self._disable_user_account(user_account)
            incident.add_action(f"Disabled user account: {user_account}")

        logger.info("Unauthorized access contained", incident_id=incident.incident_id)

    async def _contain_data_breach(self, incident: SecurityIncident):
        """Contain data breach incident."""
        # Isolate affected systems
        for system in incident.affected_systems:
            await self._isolate_system(system)
            incident.add_action(f"Isolated system: {system}")

        # Stop data processing
        await self._stop_data_processing(incident.affected_systems)
        incident.add_action("Stopped data processing on affected systems")

        logger.info("Data breach contained", incident_id=incident.incident_id)

    async def _contain_privacy_violation(self, incident: SecurityIncident):
        """Contain privacy violation incident."""
        # Stop unauthorized data processing
        await self._stop_data_processing(incident.affected_systems)
        incident.add_action("Stopped unauthorized data processing")

        # Enable privacy protection mode
        await self._enable_privacy_mode()
        incident.add_action("Enabled enhanced privacy protection")

        logger.info("Privacy violation contained", incident_id=incident.incident_id)

    async def _contain_malware(self, incident: SecurityIncident):
        """Contain malware incident."""
        # Isolate infected systems
        for system in incident.affected_systems:
            await self._isolate_system(system)
            incident.add_action(f"Isolated infected system: {system}")

        logger.info("Malware contained", incident_id=incident.incident_id)

    async def _contain_dos_attack(self, incident: SecurityIncident):
        """Contain denial of service attack."""
        # Enable DDoS protection
        await self._enable_ddos_protection()
        incident.add_action("Enabled DDoS protection")

        # Block attack sources
        source_ip = incident.indicators.get("source_ip")
        if source_ip:
            await self._block_ip_address(source_ip)
            incident.add_action(f"Blocked attack source: {source_ip}")

        logger.info("DoS attack contained", incident_id=incident.incident_id)

    async def _contain_system_compromise(self, incident: SecurityIncident):
        """Contain system compromise incident."""
        # Isolate compromised systems
        for system in incident.affected_systems:
            await self._isolate_system(system)
            incident.add_action(f"Isolated compromised system: {system}")

        # Preserve evidence
        await self._preserve_system_state(incident.affected_systems)
        incident.add_action("Preserved system state for forensic analysis")

        logger.info("System compromise contained", incident_id=incident.incident_id)

    async def _collect_forensic_evidence(self, incident: SecurityIncident):
        """Collect additional forensic evidence."""
        # Collect system logs
        for system in incident.affected_systems:
            log_evidence = await self._collect_system_logs(system)
            if log_evidence:
                incident.evidence.append(log_evidence)
                incident.add_action(f"Collected logs from {system}")

        # Collect network traffic
        network_evidence = await self._collect_network_traffic()
        if network_evidence:
            incident.evidence.append(network_evidence)
            incident.add_action("Collected network traffic analysis")

        logger.info("Forensic evidence collected", incident_id=incident.incident_id)

    async def _execute_eradication(self, incident: SecurityIncident):
        """Execute eradication actions."""
        # Remove malicious artifacts
        for system in incident.affected_systems:
            await self._clean_system(system)
            incident.add_action(f"Cleaned system: {system}")

        # Update security controls
        await self._update_security_controls()
        incident.add_action("Updated security controls")

        logger.info("Eradication completed", incident_id=incident.incident_id)

    async def _execute_recovery(self, incident: SecurityIncident):
        """Execute recovery actions."""
        # Restore systems from clean backups
        for system in incident.affected_systems:
            await self._restore_system(system)
            incident.add_action(f"Restored system: {system}")

        # Verify system integrity
        integrity_check = await self._verify_system_integrity(incident.affected_systems)
        incident.add_action(f"System integrity verification: {integrity_check}")

        logger.info("Recovery completed", incident_id=incident.incident_id)

    # Containment action implementations (simplified for demo)
    async def _block_ip_address(self, ip_address: str):
        """Block IP address in firewall."""
        logger.info("IP address blocked", ip=ip_address)

    async def _disable_user_account(self, username: str):
        """Disable user account."""
        logger.info("User account disabled", username=username)

    async def _isolate_system(self, system: str):
        """Isolate system from network."""
        logger.info("System isolated", system=system)

    async def _stop_data_processing(self, systems: list[str]):
        """Stop data processing on systems."""
        logger.info("Data processing stopped", systems=systems)

    async def _enable_privacy_mode(self):
        """Enable enhanced privacy protection."""
        logger.info("Privacy mode enabled")

    async def _enable_ddos_protection(self):
        """Enable DDoS protection."""
        logger.info("DDoS protection enabled")

    async def _preserve_system_state(self, systems: list[str]):
        """Preserve system state for forensics."""
        logger.info("System state preserved", systems=systems)

    async def _collect_system_logs(self, system: str) -> IncidentEvidence | None:
        """Collect system logs as evidence."""
        evidence = IncidentEvidence(
            evidence_id=f"EVD-LOGS-{system}-{int(time.time())}",
            evidence_type="system_logs",
            source=system,
            timestamp=datetime.now(),
            data={"log_entries": f"logs_from_{system}"},
            hash_value="sample_hash_value",
        )
        evidence.add_to_custody_chain("incident_responder")
        return evidence

    async def _collect_network_traffic(self) -> IncidentEvidence | None:
        """Collect network traffic analysis."""
        evidence = IncidentEvidence(
            evidence_id=f"EVD-NETWORK-{int(time.time())}",
            evidence_type="network_traffic",
            source="network_monitor",
            timestamp=datetime.now(),
            data={"traffic_analysis": "network_pcap_data"},
            hash_value="sample_hash_value",
        )
        evidence.add_to_custody_chain("incident_responder")
        return evidence

    async def _clean_system(self, system: str):
        """Clean malicious artifacts from system."""
        logger.info("System cleaned", system=system)

    async def _update_security_controls(self):
        """Update security controls based on incident."""
        logger.info("Security controls updated")

    async def _restore_system(self, system: str):
        """Restore system from clean backup."""
        logger.info("System restored", system=system)

    async def _verify_system_integrity(self, systems: list[str]) -> str:
        """Verify system integrity after recovery."""
        logger.info("System integrity verified", systems=systems)
        return "VERIFIED"


class IncidentCommunicator:
    """Incident communication and notification system."""

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or self._default_config()
        self.notification_channels = {
            "email": self._send_email_notification,
            "slack": self._send_slack_notification,
            "pagerduty": self._send_pagerduty_alert,
            "sms": self._send_sms_notification,
        }
        logger.info("Incident communicator initialized")

    def _default_config(self) -> dict[str, Any]:
        """Default communication configuration."""
        return {
            "email": {
                "smtp_server": "smtp.company.com",
                "smtp_port": 587,
                "username": "security-alerts@company.com",
                "password": "secure_password",
                "recipients": ["security-team@company.com", "ops-team@company.com"],
            },
            "slack": {
                "webhook_url": "https://hooks.slack.com/services/...",
                "channel": "#security-alerts",
            },
            "pagerduty": {"integration_key": "your-pagerduty-integration-key"},
        }

    async def notify_incident_created(self, incident: SecurityIncident):
        """Send notification when incident is created."""
        message = self._format_incident_message(incident, "INCIDENT DETECTED")

        # Send notifications based on severity
        if incident.severity in [IncidentSeverity.CRITICAL, IncidentSeverity.HIGH]:
            await self._send_notifications(["email", "slack", "pagerduty"], message)
        else:
            await self._send_notifications(["email", "slack"], message)

        logger.info(
            "Incident creation notification sent", incident_id=incident.incident_id
        )

    async def notify_incident_resolved(self, incident: SecurityIncident):
        """Send notification when incident is resolved."""
        message = self._format_incident_message(incident, "INCIDENT RESOLVED")
        await self._send_notifications(["email", "slack"], message)

        logger.info(
            "Incident resolution notification sent", incident_id=incident.incident_id
        )

    async def send_status_update(self, incident: SecurityIncident):
        """Send incident status update."""
        message = self._format_incident_message(incident, "INCIDENT UPDATE")
        await self._send_notifications(["slack"], message)

        logger.info("Incident status update sent", incident_id=incident.incident_id)

    def _format_incident_message(
        self, incident: SecurityIncident, alert_type: str
    ) -> dict[str, Any]:
        """Format incident message for notifications."""
        return {
            "alert_type": alert_type,
            "incident_id": incident.incident_id,
            "title": incident.title,
            "severity": incident.severity.value,
            "status": incident.status.value,
            "type": incident.incident_type.value,
            "detected_at": incident.detected_at.isoformat(),
            "affected_systems": incident.affected_systems,
            "description": incident.description,
        }

    async def _send_notifications(self, channels: list[str], message: dict[str, Any]):
        """Send notifications to specified channels."""
        for channel in channels:
            if channel in self.notification_channels:
                try:
                    await self.notification_channels[channel](message)
                except Exception as e:
                    logger.error(f"Failed to send {channel} notification", error=str(e))

    async def _send_email_notification(self, message: dict[str, Any]):
        """Send email notification."""
        # In production: implement actual email sending
        logger.info("Email notification sent", message=message["alert_type"])

    async def _send_slack_notification(self, message: dict[str, Any]):
        """Send Slack notification."""
        # In production: implement actual Slack webhook
        logger.info("Slack notification sent", message=message["alert_type"])

    async def _send_pagerduty_alert(self, message: dict[str, Any]):
        """Send PagerDuty alert."""
        # In production: implement actual PagerDuty integration
        logger.info("PagerDuty alert sent", message=message["alert_type"])

    async def _send_sms_notification(self, message: dict[str, Any]):
        """Send SMS notification."""
        # In production: implement actual SMS sending
        logger.info("SMS notification sent", message=message["alert_type"])


class IncidentReportGenerator:
    """Generate comprehensive incident reports."""

    def __init__(self):
        self.template_path = "incident_report_template.html"
        logger.info("Incident report generator initialized")

    async def generate_incident_report(
        self, incident: SecurityIncident
    ) -> dict[str, Any]:
        """Generate comprehensive incident report."""
        report = {
            "report_id": f"RPT-{incident.incident_id}",
            "generated_at": datetime.now().isoformat(),
            "incident_summary": self._generate_summary(incident),
            "timeline": self._generate_timeline(incident),
            "evidence_summary": self._generate_evidence_summary(incident),
            "impact_assessment": self._assess_impact(incident),
            "root_cause_analysis": self._analyze_root_cause(incident),
            "lessons_learned": incident.lessons_learned,
            "recommendations": self._generate_recommendations(incident),
        }

        logger.info("Incident report generated", incident_id=incident.incident_id)
        return report

    def _generate_summary(self, incident: SecurityIncident) -> dict[str, Any]:
        """Generate incident summary."""
        duration = None
        if incident.resolved_at:
            duration = (
                incident.resolved_at - incident.detected_at
            ).total_seconds() / 3600

        return {
            "incident_id": incident.incident_id,
            "type": incident.incident_type.value,
            "severity": incident.severity.value,
            "status": incident.status.value,
            "detected_at": incident.detected_at.isoformat(),
            "resolved_at": (
                incident.resolved_at.isoformat() if incident.resolved_at else None
            ),
            "duration_hours": duration,
            "affected_systems": incident.affected_systems,
            "assigned_to": incident.assigned_to,
        }

    def _generate_timeline(self, incident: SecurityIncident) -> list[dict[str, str]]:
        """Generate incident timeline."""
        timeline = []

        # Add detection event
        timeline.append(
            {
                "timestamp": incident.detected_at.isoformat(),
                "event": "Incident detected",
                "description": incident.description,
            }
        )

        # Add actions taken
        for action in incident.actions_taken:
            if ":" in action:
                timestamp, description = action.split(":", 1)
                timeline.append(
                    {
                        "timestamp": timestamp.strip(),
                        "event": "Action taken",
                        "description": description.strip(),
                    }
                )

        # Add resolution event
        if incident.resolved_at:
            timeline.append(
                {
                    "timestamp": incident.resolved_at.isoformat(),
                    "event": "Incident resolved",
                    "description": "Incident successfully resolved",
                }
            )

        return timeline

    def _generate_evidence_summary(
        self, incident: SecurityIncident
    ) -> list[dict[str, Any]]:
        """Generate evidence summary."""
        evidence_summary = []

        for evidence in incident.evidence:
            evidence_summary.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "type": evidence.evidence_type,
                    "source": evidence.source,
                    "timestamp": evidence.timestamp.isoformat(),
                    "hash": evidence.hash_value,
                    "custody_chain": evidence.chain_of_custody,
                }
            )

        return evidence_summary

    def _assess_impact(self, incident: SecurityIncident) -> dict[str, Any]:
        """Assess incident impact."""
        return {
            "systems_affected": len(incident.affected_systems),
            "severity_level": incident.severity.value,
            "business_impact": self._determine_business_impact(incident),
            "data_impact": self._assess_data_impact(incident),
            "compliance_impact": self._assess_compliance_impact(incident),
        }

    def _analyze_root_cause(self, incident: SecurityIncident) -> dict[str, Any]:
        """Analyze root cause of incident."""
        return {
            "identified_cause": incident.root_cause or "Under investigation",
            "contributing_factors": self._identify_contributing_factors(incident),
            "vulnerability_exploited": self._identify_exploited_vulnerability(incident),
            "attack_vector": self._identify_attack_vector(incident),
        }

    def _generate_recommendations(self, incident: SecurityIncident) -> list[str]:
        """Generate recommendations based on incident."""
        recommendations = [
            "Implement enhanced monitoring for similar attack patterns",
            "Review and update incident response procedures",
            "Conduct security awareness training",
            "Perform vulnerability assessment of affected systems",
        ]

        # Add specific recommendations based on incident type
        if incident.incident_type == IncidentType.UNAUTHORIZED_ACCESS:
            recommendations.extend(
                [
                    "Implement multi-factor authentication",
                    "Review access control policies",
                    "Enhance login monitoring",
                ]
            )
        elif incident.incident_type == IncidentType.DATA_BREACH:
            recommendations.extend(
                [
                    "Implement data loss prevention controls",
                    "Review data classification and handling procedures",
                    "Enhance data encryption",
                ]
            )

        return recommendations

    def _determine_business_impact(self, incident: SecurityIncident) -> str:
        """Determine business impact level."""
        if incident.severity == IncidentSeverity.CRITICAL:
            return "HIGH"
        elif incident.severity == IncidentSeverity.HIGH:
            return "MEDIUM"
        else:
            return "LOW"

    def _assess_data_impact(self, incident: SecurityIncident) -> str:
        """Assess data impact."""
        if incident.incident_type in [
            IncidentType.DATA_BREACH,
            IncidentType.PRIVACY_VIOLATION,
        ]:
            return "CONFIRMED"
        else:
            return "NONE"

    def _assess_compliance_impact(self, incident: SecurityIncident) -> list[str]:
        """Assess compliance impact."""
        impacts = []

        if incident.incident_type == IncidentType.PRIVACY_VIOLATION:
            impacts.extend(["GDPR", "CCPA"])

        if incident.incident_type == IncidentType.DATA_BREACH:
            impacts.extend(["Data Breach Notification Laws", "Industry Regulations"])

        return impacts

    def _identify_contributing_factors(self, _incident: SecurityIncident) -> list[str]:
        """Identify contributing factors."""
        return ["Configuration weakness", "Inadequate monitoring", "User error"]

    def _identify_exploited_vulnerability(self, _incident: SecurityIncident) -> str:
        """Identify exploited vulnerability."""
        return "Authentication bypass vulnerability"

    def _identify_attack_vector(self, _incident: SecurityIncident) -> str:
        """Identify attack vector."""
        return "Network-based attack"


class IncidentResponseOrchestrator:
    """Orchestrates the complete incident response workflow."""

    def __init__(self):
        self.detector = IncidentDetector()
        self.responder = IncidentResponder()
        self.communicator = IncidentCommunicator()
        self.report_generator = IncidentReportGenerator()
        self.active_incidents: dict[str, SecurityIncident] = {}
        logger.info("Incident response orchestrator initialized")

    async def handle_security_event(
        self, event: dict[str, Any]
    ) -> SecurityIncident | None:
        """Handle incoming security event."""
        # Detect potential incident
        incident = await self.detector.analyze_security_event(event)

        if incident:
            # Store incident
            self.active_incidents[incident.incident_id] = incident

            # Send initial notification
            await self.communicator.notify_incident_created(incident)

            # Execute automated response
            response_success = await self.responder.respond_to_incident(incident)

            if response_success:
                # Send resolution notification
                await self.communicator.notify_incident_resolved(incident)

                # Generate incident report
                await self.report_generator.generate_incident_report(incident)

                # Close incident
                incident.update_status(IncidentStatus.CLOSED)

                logger.info(
                    "Incident response completed successfully",
                    incident_id=incident.incident_id,
                    total_time=(datetime.now() - incident.detected_at).total_seconds(),
                )

            return incident

        return None

    async def get_incident_status(self, incident_id: str) -> dict[str, Any] | None:
        """Get current status of incident."""
        incident = self.active_incidents.get(incident_id)

        if incident:
            return {
                "incident_id": incident.incident_id,
                "status": incident.status.value,
                "severity": incident.severity.value,
                "type": incident.incident_type.value,
                "detected_at": incident.detected_at.isoformat(),
                "resolved_at": (
                    incident.resolved_at.isoformat() if incident.resolved_at else None
                ),
                "actions_count": len(incident.actions_taken),
                "evidence_count": len(incident.evidence),
            }

        return None

    async def list_active_incidents(self) -> list[dict[str, Any]]:
        """List all active incidents."""
        active = []

        for incident in self.active_incidents.values():
            if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                active.append(
                    {
                        "incident_id": incident.incident_id,
                        "title": incident.title,
                        "severity": incident.severity.value,
                        "status": incident.status.value,
                        "detected_at": incident.detected_at.isoformat(),
                        "affected_systems": incident.affected_systems,
                    }
                )

        return active


# Initialize incident response system
incident_response_system = IncidentResponseOrchestrator()


async def simulate_security_incident():
    """Simulate a security incident for testing."""
    # Simulate unauthorized access attempt
    security_event = {
        "event_type": "auth_failure",
        "source": "auth_service",
        "user_account": "test_user",
        "source_ip": "192.168.1.100",
        "failure_count": 15,
        "timestamp": datetime.now().isoformat(),
        "affected_systems": ["auth_service", "user_database"],
    }

    print("🚨 Simulating Security Incident...")
    incident = await incident_response_system.handle_security_event(security_event)

    if incident:
        print(f"✅ Incident {incident.incident_id} handled successfully")
        print(f"📊 Type: {incident.incident_type.value}")
        print(f"🔥 Severity: {incident.severity.value}")
        print(f"📍 Status: {incident.status.value}")
        print(f"⏱️  Actions taken: {len(incident.actions_taken)}")
    else:
        print("❌ No incident detected")


if __name__ == "__main__":
    asyncio.run(simulate_security_incident())
