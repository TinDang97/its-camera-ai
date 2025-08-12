"""Security and authentication commands for ITS Camera AI.

Commands for managing security policies, authentication, and access control.
"""

import asyncio
import json
import time
from pathlib import Path

import typer
from rich.table import Table
from rich.tree import Tree

from ..utils import (
    confirm_action,
    console,
    create_progress,
    handle_async_command,
    print_error,
    print_info,
    print_success,
    print_warning,
)

app = typer.Typer(help="🔒 Security and authentication management")


@app.command()
@handle_async_command
async def users(
    action: str = typer.Argument(
        "list", help="Action: list, create, update, delete, activate, deactivate"
    ),
    username: str | None = typer.Option(
        None, "--username", "-u", help="Username for create/update/delete operations"
    ),
    role: str | None = typer.Option(
        None, "--role", "-r", help="User role (admin, operator, viewer)"
    ),
    email: str | None = typer.Option(
        None, "--email", "-e", help="User email address"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force operation without confirmation"
    ),
) -> None:
    """👥 Manage system users.
    
    Create, update, delete, and manage user accounts with role-based access control.
    """
    valid_actions = ["list", "create", "update", "delete", "activate", "deactivate"]
    if action not in valid_actions:
        print_error(f"Invalid action. Must be one of: {', '.join(valid_actions)}")
        return

    if action == "list":
        await _list_users()
    elif action == "create":
        if not username:
            print_error("Username is required for create operation")
            return
        await _create_user(username, role, email)
    elif action == "update":
        if not username:
            print_error("Username is required for update operation")
            return
        await _update_user(username, role, email)
    elif action == "delete":
        if not username:
            print_error("Username is required for delete operation")
            return
        await _delete_user(username, force)
    elif action in ["activate", "deactivate"]:
        if not username:
            print_error(f"Username is required for {action} operation")
            return
        await _toggle_user_status(username, action == "activate")


async def _list_users() -> None:
    """
    List all system users.
    """
    print_info("Fetching user accounts...")

    # Simulate user data
    users_data = [
        {
            "username": "admin",
            "email": "admin@its-camera-ai.com",
            "role": "admin",
            "status": "active",
            "last_login": "2024-01-16 14:30",
            "created": "2024-01-01",
            "mfa_enabled": True,
            "failed_attempts": 0,
        },
        {
            "username": "operator1",
            "email": "operator1@its-camera-ai.com",
            "role": "operator",
            "status": "active",
            "last_login": "2024-01-16 10:15",
            "created": "2024-01-02",
            "mfa_enabled": True,
            "failed_attempts": 1,
        },
        {
            "username": "viewer1",
            "email": "viewer1@its-camera-ai.com",
            "role": "viewer",
            "status": "inactive",
            "last_login": "2024-01-10 16:45",
            "created": "2024-01-05",
            "mfa_enabled": False,
            "failed_attempts": 3,
        },
    ]

    # Display users table
    table = Table(title="System Users")
    table.add_column("Username", style="cyan")
    table.add_column("Email", style="green")
    table.add_column("Role", style="yellow")
    table.add_column("Status", style="blue")
    table.add_column("Last Login", style="white")
    table.add_column("MFA", style="magenta")
    table.add_column("Failed", style="red")

    for user in users_data:
        status_style = "green" if user["status"] == "active" else "red"
        mfa_status = "✓" if user["mfa_enabled"] else "✗"
        mfa_style = "green" if user["mfa_enabled"] else "red"

        table.add_row(
            user["username"],
            user["email"],
            user["role"],
            f"[{status_style}]{user['status']}[/{status_style}]",
            user["last_login"],
            f"[{mfa_style}]{mfa_status}[/{mfa_style}]",
            str(user["failed_attempts"]),
        )

    console.print(table)
    print_info(f"Found {len(users_data)} users")


async def _create_user(
    username: str, role: str | None, email: str | None
) -> None:
    """
    Create a new user.
    
    Args:
        username: Username
        role: User role
        email: User email
    """
    print_info(f"Creating user: {username}")

    # Validate role
    valid_roles = ["admin", "operator", "viewer"]
    if role and role not in valid_roles:
        print_error(f"Invalid role. Must be one of: {', '.join(valid_roles)}")
        return

    # Use defaults if not provided
    role = role or "viewer"
    email = email or f"{username}@its-camera-ai.com"

    # Display user configuration
    config = {
        "Username": username,
        "Email": email,
        "Role": role,
        "MFA Required": "Yes" if role in ["admin", "operator"] else "No",
        "Initial Status": "Active",
    }

    config_table = Table(title="New User Configuration")
    config_table.add_column("Property", style="cyan")
    config_table.add_column("Value", style="green")

    for key, value in config.items():
        config_table.add_row(key, str(value))

    console.print(config_table)

    # Simulate user creation
    with create_progress() as progress:
        task = progress.add_task("Creating user...", total=4)

        steps = [
            "Validating username",
            "Creating user account",
            "Setting up permissions",
            "Sending welcome email",
        ]

        for step in steps:
            progress.update(task, description=step)
            await asyncio.sleep(0.5)
            progress.advance(task)

    print_success(f"User {username} created successfully!")
    print_info("Temporary password sent to user's email")


async def _update_user(
    username: str, role: str | None, email: str | None
) -> None:
    """
    Update an existing user.
    
    Args:
        username: Username
        role: New role
        email: New email
    """
    print_info(f"Updating user: {username}")

    if not role and not email:
        print_error("At least one field (role or email) must be provided for update")
        return

    # Simulate user update
    updates = []
    if role:
        updates.append(f"Role: {role}")
    if email:
        updates.append(f"Email: {email}")

    print_info(f"Updates: {', '.join(updates)}")

    with create_progress() as progress:
        task = progress.add_task("Updating user...", total=len(updates) + 1)

        for update in updates:
            progress.update(task, description=f"Applying {update}")
            await asyncio.sleep(0.3)
            progress.advance(task)

        progress.update(task, description="Notifying user")
        await asyncio.sleep(0.3)
        progress.advance(task)

    print_success(f"User {username} updated successfully!")


async def _delete_user(username: str, force: bool) -> None:
    """
    Delete a user.
    
    Args:
        username: Username
        force: Force deletion without confirmation
    """
    if not force:
        if not confirm_action(f"Are you sure you want to delete user '{username}'?"):
            print_info("User deletion cancelled")
            return

    print_warning(f"Deleting user: {username}")

    with create_progress() as progress:
        task = progress.add_task("Deleting user...", total=3)

        steps = [
            "Revoking access permissions",
            "Removing user data",
            "Updating audit logs",
        ]

        for step in steps:
            progress.update(task, description=step)
            await asyncio.sleep(0.4)
            progress.advance(task)

    print_success(f"User {username} deleted successfully!")


async def _toggle_user_status(username: str, activate: bool) -> None:
    """
    Activate or deactivate a user.
    
    Args:
        username: Username
        activate: True to activate, False to deactivate
    """
    action = "Activating" if activate else "Deactivating"
    status = "active" if activate else "inactive"

    print_info(f"{action} user: {username}")

    # Simulate status change
    await asyncio.sleep(0.5)

    print_success(f"User {username} is now {status}")


@app.command()
@handle_async_command
async def permissions(
    username: str | None = typer.Option(
        None, "--username", "-u", help="Username to check permissions for"
    ),
    role: str | None = typer.Option(
        None, "--role", "-r", help="Role to check permissions for"
    ),
    resource: str | None = typer.Option(
        None, "--resource", help="Specific resource to check"
    ),
) -> None:
    """🔑 View and manage permissions.
    
    Display permissions for users, roles, or specific resources.
    """
    if username:
        await _show_user_permissions(username)
    elif role:
        await _show_role_permissions(role)
    else:
        await _show_all_permissions(resource)


async def _show_user_permissions(username: str) -> None:
    """
    Show permissions for a specific user.
    
    Args:
        username: Username
    """
    print_info(f"Fetching permissions for user: {username}")

    # Simulate user permissions
    user_permissions = {
        "role": "operator",
        "permissions": [
            "camera:read",
            "camera:write",
            "analytics:read",
            "streams:read",
            "streams:control",
        ],
        "resources": [
            "cameras/*",
            "streams/*",
            "analytics/reports",
        ],
        "restrictions": [
            "No admin panel access",
            "Cannot delete users",
            "Cannot modify system config",
        ],
    }

    # Create permissions tree
    tree = Tree(f"[bold cyan]Permissions for {username}[/bold cyan]")

    role_branch = tree.add(f"[yellow]Role: {user_permissions['role']}[/yellow]")

    perms_branch = role_branch.add("[green]Permissions[/green]")
    for perm in user_permissions["permissions"]:
        perms_branch.add(f"[blue]✓[/blue] {perm}")

    resources_branch = role_branch.add("[green]Resources[/green]")
    for resource in user_permissions["resources"]:
        resources_branch.add(f"[blue]✓[/blue] {resource}")

    restrictions_branch = role_branch.add("[red]Restrictions[/red]")
    for restriction in user_permissions["restrictions"]:
        restrictions_branch.add(f"[red]✗[/red] {restriction}")

    console.print(tree)


async def _show_role_permissions(role: str) -> None:
    """
    Show permissions for a specific role.
    
    Args:
        role: Role name
    """
    print_info(f"Fetching permissions for role: {role}")

    # Simulate role permissions
    role_permissions = {
        "admin": {
            "description": "Full system administrator",
            "permissions": ["*"],
            "resources": ["*"],
            "can_do": [
                "Manage all users",
                "Configure system settings",
                "Access all data",
                "Deploy models",
                "View security logs",
            ],
        },
        "operator": {
            "description": "System operator with limited admin rights",
            "permissions": [
                "camera:read", "camera:write",
                "analytics:read", "streams:read", "streams:control"
            ],
            "resources": ["cameras/*", "streams/*", "analytics/reports"],
            "can_do": [
                "Control camera streams",
                "View analytics reports",
                "Restart services",
                "Monitor system health",
            ],
        },
        "viewer": {
            "description": "Read-only access to analytics",
            "permissions": ["analytics:read"],
            "resources": ["analytics/reports", "analytics/dashboards"],
            "can_do": [
                "View analytics reports",
                "Access dashboards",
                "Export reports",
            ],
        },
    }

    if role not in role_permissions:
        print_error(f"Unknown role: {role}")
        return

    role_data = role_permissions[role]

    # Display role information
    info_table = Table(title=f"Role: {role.title()}")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")

    info_table.add_row("Description", role_data["description"])
    info_table.add_row("Permissions", ", ".join(role_data["permissions"]))
    info_table.add_row("Resources", ", ".join(role_data["resources"]))

    console.print(info_table)

    # Display capabilities
    capabilities_table = Table(title="Capabilities")
    capabilities_table.add_column("Action", style="green")

    for action in role_data["can_do"]:
        capabilities_table.add_row(f"✓ {action}")

    console.print(capabilities_table)


async def _show_all_permissions(resource_filter: str | None) -> None:
    """
    Show all permissions matrix.
    
    Args:
        resource_filter: Optional resource filter
    """
    print_info("Fetching permissions matrix...")

    # Simulate permissions matrix
    permissions_matrix = [
        {"resource": "cameras", "admin": "✓", "operator": "✓", "viewer": "✗"},
        {"resource": "streams", "admin": "✓", "operator": "✓", "viewer": "✗"},
        {"resource": "analytics", "admin": "✓", "operator": "R", "viewer": "R"},
        {"resource": "users", "admin": "✓", "operator": "✗", "viewer": "✗"},
        {"resource": "system", "admin": "✓", "operator": "R", "viewer": "✗"},
        {"resource": "models", "admin": "✓", "operator": "R", "viewer": "✗"},
    ]

    # Filter by resource if specified
    if resource_filter:
        permissions_matrix = [
            p for p in permissions_matrix
            if resource_filter.lower() in p["resource"].lower()
        ]

    # Display permissions matrix
    table = Table(title="Permissions Matrix")
    table.add_column("Resource", style="cyan")
    table.add_column("Admin", style="green")
    table.add_column("Operator", style="yellow")
    table.add_column("Viewer", style="blue")

    for perm in permissions_matrix:
        table.add_row(
            perm["resource"],
            perm["admin"],
            perm["operator"],
            perm["viewer"],
        )

    console.print(table)

    # Legend
    legend_table = Table(title="Legend")
    legend_table.add_column("Symbol", style="cyan")
    legend_table.add_column("Meaning", style="green")

    legend_table.add_row("✓", "Full access (read/write/delete)")
    legend_table.add_row("R", "Read-only access")
    legend_table.add_row("✗", "No access")

    console.print(legend_table)


@app.command()
@handle_async_command
async def audit(
    start_date: str | None = typer.Option(
        None, "--start", "-s", help="Start date (YYYY-MM-DD)"
    ),
    end_date: str | None = typer.Option(
        None, "--end", "-e", help="End date (YYYY-MM-DD)"
    ),
    user: str | None = typer.Option(
        None, "--user", "-u", help="Filter by username"
    ),
    event_type: str | None = typer.Option(
        None, "--event-type", "-t", help="Filter by event type"
    ),
    limit: int = typer.Option(
        50, "--limit", "-l", help="Maximum number of events to show"
    ),
    export: Path | None = typer.Option(
        None, "--export", help="Export audit log to file"
    ),
) -> None:
    """📋 View security audit logs.
    
    Display security audit logs with filtering and export capabilities.
    """
    print_info("Fetching audit logs...")

    # Simulate audit log data
    audit_logs = [
        {
            "timestamp": "2024-01-16 14:30:15",
            "event_id": "audit_001",
            "event_type": "USER_LOGIN",
            "username": "admin",
            "ip_address": "192.168.1.100",
            "result": "SUCCESS",
            "details": "MFA verified",
        },
        {
            "timestamp": "2024-01-16 14:25:30",
            "event_id": "audit_002",
            "event_type": "PERMISSION_DENIED",
            "username": "viewer1",
            "ip_address": "192.168.1.101",
            "result": "DENIED",
            "details": "Attempted to access admin panel",
        },
        {
            "timestamp": "2024-01-16 14:20:45",
            "event_id": "audit_003",
            "event_type": "MODEL_DEPLOYMENT",
            "username": "operator1",
            "ip_address": "192.168.1.102",
            "result": "SUCCESS",
            "details": "Deployed yolo11n_v2.0 to staging",
        },
        {
            "timestamp": "2024-01-16 14:15:12",
            "event_id": "audit_004",
            "event_type": "FAILED_LOGIN",
            "username": "unknown",
            "ip_address": "203.0.113.1",
            "result": "FAILED",
            "details": "Invalid credentials (attempt 3)",
        },
        {
            "timestamp": "2024-01-16 14:10:33",
            "event_id": "audit_005",
            "event_type": "CONFIG_CHANGE",
            "username": "admin",
            "ip_address": "192.168.1.100",
            "result": "SUCCESS",
            "details": "Updated security policy",
        },
    ]

    # Apply filters
    filtered_logs = audit_logs[:]

    if user:
        filtered_logs = [log for log in filtered_logs if log["username"] == user]

    if event_type:
        filtered_logs = [
            log for log in filtered_logs
            if event_type.upper() in log["event_type"]
        ]

    # Limit results
    filtered_logs = filtered_logs[:limit]

    # Display audit logs table
    table = Table(title="Security Audit Logs")
    table.add_column("Timestamp", style="cyan")
    table.add_column("Event Type", style="green")
    table.add_column("Username", style="yellow")
    table.add_column("IP Address", style="blue")
    table.add_column("Result", style="white")
    table.add_column("Details", style="dim")

    for log in filtered_logs:
        result_style = {
            "SUCCESS": "green",
            "FAILED": "red",
            "DENIED": "red",
        }.get(log["result"], "white")

        table.add_row(
            log["timestamp"],
            log["event_type"],
            log["username"],
            log["ip_address"],
            f"[{result_style}]{log['result']}[/{result_style}]",
            log["details"],
        )

    console.print(table)

    # Show summary
    summary_stats = {
        "Total Events": len(filtered_logs),
        "Success Events": len([l for l in filtered_logs if l["result"] == "SUCCESS"]),
        "Failed Events": len([l for l in filtered_logs if l["result"] in ["FAILED", "DENIED"]]),
        "Unique Users": len(set(l["username"] for l in filtered_logs)),
        "Event Types": len(set(l["event_type"] for l in filtered_logs)),
    }

    summary_table = Table(title="Audit Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Count", style="green")

    for metric, count in summary_stats.items():
        summary_table.add_row(metric, str(count))

    console.print(summary_table)

    # Export if requested
    if export:
        await _export_audit_logs(filtered_logs, export)


async def _export_audit_logs(logs: list[dict], export_path: Path) -> None:
    """
    Export audit logs to file.
    
    Args:
        logs: Audit log entries
        export_path: Export file path
    """
    print_info(f"Exporting audit logs to {export_path}...")

    try:
        with open(export_path, "w") as f:
            json.dump(logs, f, indent=2)

        print_success(f"Audit logs exported to {export_path}")
        print_info(f"Exported {len(logs)} log entries")

    except Exception as e:
        print_error(f"Failed to export audit logs: {e}")


@app.command()
@handle_async_command
async def threats(
    action: str = typer.Argument(
        "list", help="Action: list, details, dismiss, block"
    ),
    threat_id: str | None = typer.Option(
        None, "--id", help="Threat ID for details/dismiss/block operations"
    ),
    severity: str | None = typer.Option(
        None, "--severity", "-s", help="Filter by severity (low, medium, high, critical)"
    ),
    active_only: bool = typer.Option(
        True, "--active-only", help="Show only active threats"
    ),
) -> None:
    """⚠️ Monitor and manage security threats.
    
    View active security threats, investigate details, and take action.
    """
    valid_actions = ["list", "details", "dismiss", "block"]
    if action not in valid_actions:
        print_error(f"Invalid action. Must be one of: {', '.join(valid_actions)}")
        return

    if action == "list":
        await _list_threats(severity, active_only)
    elif action == "details":
        if not threat_id:
            print_error("Threat ID is required for details operation")
            return
        await _show_threat_details(threat_id)
    elif action == "dismiss":
        if not threat_id:
            print_error("Threat ID is required for dismiss operation")
            return
        await _dismiss_threat(threat_id)
    elif action == "block":
        if not threat_id:
            print_error("Threat ID is required for block operation")
            return
        await _block_threat(threat_id)


async def _list_threats(severity_filter: str | None, active_only: bool) -> None:
    """
    List security threats.
    
    Args:
        severity_filter: Optional severity filter
        active_only: Show only active threats
    """
    print_info("Fetching security threats...")

    # Simulate threat data
    threats_data = [
        {
            "id": "threat_001",
            "title": "Multiple Failed Login Attempts",
            "severity": "high",
            "status": "active",
            "source_ip": "203.0.113.1",
            "detected": "2024-01-16 14:15:12",
            "count": 5,
            "action_taken": "rate_limited",
        },
        {
            "id": "threat_002",
            "title": "Unusual API Access Pattern",
            "severity": "medium",
            "status": "active",
            "source_ip": "198.51.100.1",
            "detected": "2024-01-16 14:10:30",
            "count": 150,
            "action_taken": "monitoring",
        },
        {
            "id": "threat_003",
            "title": "Privileged Access Anomaly",
            "severity": "critical",
            "status": "dismissed",
            "source_ip": "192.168.1.105",
            "detected": "2024-01-16 13:45:15",
            "count": 1,
            "action_taken": "investigated",
        },
    ]

    # Apply filters
    filtered_threats = threats_data[:]

    if active_only:
        filtered_threats = [t for t in filtered_threats if t["status"] == "active"]

    if severity_filter:
        filtered_threats = [
            t for t in filtered_threats
            if t["severity"] == severity_filter.lower()
        ]

    # Display threats table
    table = Table(title="Security Threats")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Severity", style="yellow")
    table.add_column("Status", style="blue")
    table.add_column("Source IP", style="white")
    table.add_column("Detected", style="dim")
    table.add_column("Count", style="magenta")

    for threat in filtered_threats:
        severity_style = {
            "low": "green",
            "medium": "yellow",
            "high": "red",
            "critical": "bold red",
        }.get(threat["severity"], "white")

        status_style = {
            "active": "red",
            "dismissed": "green",
            "blocked": "blue",
        }.get(threat["status"], "white")

        table.add_row(
            threat["id"],
            threat["title"],
            f"[{severity_style}]{threat['severity'].upper()}[/{severity_style}]",
            f"[{status_style}]{threat['status']}[/{status_style}]",
            threat["source_ip"],
            threat["detected"],
            str(threat["count"]),
        )

    console.print(table)

    # Show threat summary
    active_threats = len([t for t in threats_data if t["status"] == "active"])
    critical_threats = len([t for t in threats_data if t["severity"] == "critical" and t["status"] == "active"])

    if active_threats > 0:
        if critical_threats > 0:
            print_error(f"{critical_threats} critical threats require immediate attention!")
        else:
            print_warning(f"{active_threats} active threats detected")
    else:
        print_success("No active threats detected")


async def _show_threat_details(threat_id: str) -> None:
    """
    Show detailed information about a threat.
    
    Args:
        threat_id: Threat ID
    """
    print_info(f"Fetching details for threat: {threat_id}")

    # Simulate threat details
    threat_details = {
        "id": threat_id,
        "title": "Multiple Failed Login Attempts",
        "description": "Multiple authentication failures detected from single IP address",
        "severity": "high",
        "status": "active",
        "rule_id": "multiple_failed_logins",
        "source_ip": "203.0.113.1",
        "target_accounts": ["admin", "operator1", "test"],
        "first_seen": "2024-01-16 14:10:00",
        "last_seen": "2024-01-16 14:15:12",
        "event_count": 5,
        "actions_taken": [
            "Rate limited source IP",
            "Notified security team",
            "Increased monitoring",
        ],
        "recommended_actions": [
            "Block source IP",
            "Review affected accounts",
            "Check for privilege escalation",
        ],
    }

    # Display threat details
    details_table = Table(title=f"Threat Details: {threat_id}")
    details_table.add_column("Property", style="cyan")
    details_table.add_column("Value", style="green")

    basic_info = {
        "Title": threat_details["title"],
        "Description": threat_details["description"],
        "Severity": threat_details["severity"].upper(),
        "Status": threat_details["status"],
        "Rule ID": threat_details["rule_id"],
        "Source IP": threat_details["source_ip"],
        "Event Count": str(threat_details["event_count"]),
        "First Seen": threat_details["first_seen"],
        "Last Seen": threat_details["last_seen"],
        "Target Accounts": ", ".join(threat_details["target_accounts"]),
    }

    for key, value in basic_info.items():
        details_table.add_row(key, str(value))

    console.print(details_table)

    # Display actions taken
    actions_table = Table(title="Actions Taken")
    actions_table.add_column("Action", style="green")

    for action in threat_details["actions_taken"]:
        actions_table.add_row(f"✓ {action}")

    console.print(actions_table)

    # Display recommended actions
    recommendations_table = Table(title="Recommended Actions")
    recommendations_table.add_column("Recommendation", style="yellow")

    for recommendation in threat_details["recommended_actions"]:
        recommendations_table.add_row(f"⚠️ {recommendation}")

    console.print(recommendations_table)


async def _dismiss_threat(threat_id: str) -> None:
    """
    Dismiss a threat.
    
    Args:
        threat_id: Threat ID
    """
    print_info(f"Dismissing threat: {threat_id}")

    # Simulate dismissal
    await asyncio.sleep(0.5)

    print_success(f"Threat {threat_id} has been dismissed")
    print_info("Threat marked as false positive in security logs")


async def _block_threat(threat_id: str) -> None:
    """
    Block a threat source.
    
    Args:
        threat_id: Threat ID
    """
    print_info(f"Blocking threat source for: {threat_id}")

    with create_progress() as progress:
        task = progress.add_task("Blocking threat...", total=3)

        steps = [
            "Adding IP to block list",
            "Updating firewall rules",
            "Notifying security team",
        ]

        for step in steps:
            progress.update(task, description=step)
            await asyncio.sleep(0.5)
            progress.advance(task)

    print_success(f"Threat source for {threat_id} has been blocked")
    print_info("IP address added to permanent block list")


@app.command()
def policies(
    action: str = typer.Argument(
        "list", help="Action: list, show, update, reset"
    ),
    policy_name: str | None = typer.Option(
        None, "--policy", "-p", help="Policy name"
    ),
    value: str | None = typer.Option(
        None, "--value", "-v", help="New policy value"
    ),
) -> None:
    """📋 Manage security policies.
    
    View and update security policies including password requirements,
    session timeouts, and access controls.
    """
    valid_actions = ["list", "show", "update", "reset"]
    if action not in valid_actions:
        print_error(f"Invalid action. Must be one of: {', '.join(valid_actions)}")
        return

    if action == "list":
        _list_policies()
    elif action == "show":
        if not policy_name:
            print_error("Policy name is required for show operation")
            return
        _show_policy(policy_name)
    elif action == "update":
        if not policy_name or not value:
            print_error("Policy name and value are required for update operation")
            return
        _update_policy(policy_name, value)
    elif action == "reset":
        if not policy_name:
            print_error("Policy name is required for reset operation")
            return
        _reset_policy(policy_name)


def _list_policies() -> None:
    """
    List all security policies.
    """
    print_info("Fetching security policies...")

    # Simulate security policies
    policies = [
        {
            "name": "password_min_length",
            "description": "Minimum password length",
            "value": "12",
            "default": "8",
            "type": "integer",
        },
        {
            "name": "password_require_special",
            "description": "Require special characters in passwords",
            "value": "true",
            "default": "true",
            "type": "boolean",
        },
        {
            "name": "session_timeout_minutes",
            "description": "Session timeout in minutes",
            "value": "480",
            "default": "360",
            "type": "integer",
        },
        {
            "name": "max_failed_attempts",
            "description": "Maximum failed login attempts before lockout",
            "value": "5",
            "default": "3",
            "type": "integer",
        },
        {
            "name": "mfa_required_roles",
            "description": "Roles requiring multi-factor authentication",
            "value": "admin,operator",
            "default": "admin",
            "type": "string",
        },
        {
            "name": "audit_log_retention_days",
            "description": "Audit log retention period in days",
            "value": "2555",
            "default": "365",
            "type": "integer",
        },
    ]

    # Display policies table
    table = Table(title="Security Policies")
    table.add_column("Policy Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Current Value", style="yellow")
    table.add_column("Default", style="blue")
    table.add_column("Type", style="white")

    for policy in policies:
        # Highlight modified policies
        value_style = "yellow" if policy["value"] != policy["default"] else "green"

        table.add_row(
            policy["name"],
            policy["description"],
            f"[{value_style}]{policy['value']}[/{value_style}]",
            policy["default"],
            policy["type"],
        )

    console.print(table)

    modified_count = len([p for p in policies if p["value"] != p["default"]])
    print_info(f"Found {len(policies)} policies ({modified_count} modified from defaults)")


def _show_policy(policy_name: str) -> None:
    """
    Show detailed information about a specific policy.
    
    Args:
        policy_name: Policy name
    """
    print_info(f"Fetching policy details: {policy_name}")

    # Simulate policy details
    policy_details = {
        "name": policy_name,
        "description": "Minimum password length requirement",
        "current_value": "12",
        "default_value": "8",
        "type": "integer",
        "valid_range": "8-128",
        "last_modified": "2024-01-15 10:30:00",
        "modified_by": "admin",
        "compliance_note": "Meets NIST recommendations for secure passwords",
    }

    # Display policy details
    details_table = Table(title=f"Policy: {policy_name}")
    details_table.add_column("Property", style="cyan")
    details_table.add_column("Value", style="green")

    for key, value in policy_details.items():
        if key != "name":
            display_key = key.replace("_", " ").title()
            details_table.add_row(display_key, str(value))

    console.print(details_table)


def _update_policy(policy_name: str, value: str) -> None:
    """
    Update a security policy.
    
    Args:
        policy_name: Policy name
        value: New policy value
    """
    print_info(f"Updating policy {policy_name} to: {value}")

    # Simulate policy update
    time.sleep(0.5)

    print_success(f"Policy {policy_name} updated successfully")
    print_info("Policy change will take effect for new sessions")


def _reset_policy(policy_name: str) -> None:
    """
    Reset a policy to its default value.
    
    Args:
        policy_name: Policy name
    """
    if not confirm_action(f"Reset policy '{policy_name}' to default value?"):
        print_info("Policy reset cancelled")
        return

    print_info(f"Resetting policy {policy_name} to default...")

    # Simulate policy reset
    time.sleep(0.3)

    print_success(f"Policy {policy_name} reset to default value")
