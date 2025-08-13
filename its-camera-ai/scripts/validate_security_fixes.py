#!/usr/bin/env python3
"""
Security Validation Script for ITS Camera AI

Validates that all P0 security vulnerabilities have been properly fixed:
1. CWE-377: Insecure Temporary File Creation
2. CWE-605: Network Binding Security
3. Hardcoded Secrets Detection
4. Authentication Flow Validation
"""

import re
from pathlib import Path
from typing import Any

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()
app = typer.Typer(help="Validate security fixes for ITS Camera AI")


class SecurityValidator:
    """Comprehensive security validation for the ITS Camera AI system."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.src_dir = project_root / "src"
        self.scripts_dir = project_root / "scripts"
        self.issues: list[dict[str, Any]] = []
        self.fixes_validated: list[str] = []

    def validate_all(self) -> bool:
        """Run all security validations."""
        rprint("[cyan]Starting comprehensive security validation...[/cyan]\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # 1. Validate temporary file security fixes
            task1 = progress.add_task("Validating temporary file security fixes...", total=1)
            self._validate_temp_file_security()
            progress.update(task1, completed=1)

            # 2. Validate network binding security fixes
            task2 = progress.add_task("Validating network binding security...", total=1)
            self._validate_network_binding_security()
            progress.update(task2, completed=1)

            # 3. Validate secret comparison security fixes
            task3 = progress.add_task("Validating secret comparison security...", total=1)
            self._validate_secret_comparison_security()
            progress.update(task3, completed=1)

            # 4. Validate authentication architecture compliance
            task4 = progress.add_task("Validating authentication architecture...", total=1)
            self._validate_authentication_architecture()
            progress.update(task4, completed=1)

            # 5. Run static security analysis
            task5 = progress.add_task("Running static security analysis...", total=1)
            self._run_static_security_analysis()
            progress.update(task5, completed=1)

        return len(self.issues) == 0

    def _validate_temp_file_security(self):
        """Validate CWE-377: Insecure Temporary File Creation fixes."""
        files_to_check = [
            self.src_dir / "its_camera_ai" / "api" / "routers" / "model_management.py",
            self.src_dir / "its_camera_ai" / "api" / "routers" / "storage.py",
        ]

        for file_path in files_to_check:
            if not file_path.exists():
                self.issues.append({
                    "severity": "HIGH",
                    "category": "CWE-377",
                    "file": str(file_path),
                    "message": "File not found for security validation"
                })
                continue

            content = file_path.read_text()

            # Check for insecure tempfile.mktemp usage
            if "tempfile.mktemp(" in content:
                self.issues.append({
                    "severity": "CRITICAL",
                    "category": "CWE-377",
                    "file": str(file_path),
                    "message": "Insecure tempfile.mktemp() usage detected",
                    "line": self._find_line_number(content, "tempfile.mktemp(")
                })

            # Check for secure NamedTemporaryFile usage
            if "NamedTemporaryFile" in content and "delete=False" in content:
                self.fixes_validated.append(f"Secure temporary file creation in {file_path.name}")
            else:
                self.issues.append({
                    "severity": "HIGH",
                    "category": "CWE-377",
                    "file": str(file_path),
                    "message": "Secure NamedTemporaryFile pattern not found"
                })

            # Check for secure file permissions
            if "os.chmod" in content and "0o600" in content:
                self.fixes_validated.append(f"Secure file permissions in {file_path.name}")
            else:
                self.issues.append({
                    "severity": "MEDIUM",
                    "category": "CWE-377",
                    "file": str(file_path),
                    "message": "Secure file permissions (0o600) not set for temporary files"
                })

    def _validate_network_binding_security(self):
        """Validate CWE-605: Network Binding Security fixes."""
        cli_file = self.src_dir / "its_camera_ai" / "cli.py"

        if not cli_file.exists():
            self.issues.append({
                "severity": "HIGH",
                "category": "CWE-605",
                "file": str(cli_file),
                "message": "CLI file not found for network binding validation"
            })
            return

        content = cli_file.read_text()

        # Check that hardcoded 0.0.0.0 binding is removed from default
        if 'default="0.0.0.0"' in content:
            self.issues.append({
                "severity": "HIGH",
                "category": "CWE-605",
                "file": str(cli_file),
                "message": "Hardcoded 0.0.0.0 binding found in CLI defaults",
                "line": self._find_line_number(content, 'default="0.0.0.0"')
            })

        # Check for environment-specific binding logic
        if "is_production()" in content and "127.0.0.1" in content:
            self.fixes_validated.append("Environment-specific host binding implemented")
        else:
            self.issues.append({
                "severity": "MEDIUM",
                "category": "CWE-605",
                "file": str(cli_file),
                "message": "Environment-specific host binding logic not found"
            })

        # Check for development environment handling
        if "development" in content and "0.0.0.0" in content:
            self.fixes_validated.append("Development environment binding properly handled")

    def _validate_secret_comparison_security(self):
        """Validate secure secret comparison fixes."""
        deploy_script = self.scripts_dir / "deploy_auth_system.py"

        if not deploy_script.exists():
            self.issues.append({
                "severity": "HIGH",
                "category": "Hardcoded Secrets",
                "file": str(deploy_script),
                "message": "Deploy script not found for secret comparison validation"
            })
            return

        content = deploy_script.read_text()

        # Check for insecure direct string comparison
        if 'settings.security.secret_key == "change-me-in-production"' in content:
            self.issues.append({
                "severity": "CRITICAL",
                "category": "Timing Attack",
                "file": str(deploy_script),
                "message": "Insecure direct secret comparison detected (timing attack vulnerability)",
                "line": self._find_line_number(content, 'settings.security.secret_key == "change-me-in-production"')
            })

        # Check for secure comparison using secrets.compare_digest
        if "secrets.compare_digest" in content:
            self.fixes_validated.append("Secure secret comparison using secrets.compare_digest")
        else:
            self.issues.append({
                "severity": "HIGH",
                "category": "Timing Attack",
                "file": str(deploy_script),
                "message": "Secure secret comparison (secrets.compare_digest) not found"
            })

        # Check for secrets module import
        if "import secrets" in content:
            self.fixes_validated.append("Secrets module properly imported")
        else:
            self.issues.append({
                "severity": "MEDIUM",
                "category": "Import",
                "file": str(deploy_script),
                "message": "secrets module not imported for secure comparison"
            })

    def _validate_authentication_architecture(self):
        """Validate authentication architecture compliance."""
        auth_sequence_diagram = self.project_root / "docs" / "diagrams" / "sequence" / "authentication-authorization.puml"

        if not auth_sequence_diagram.exists():
            self.issues.append({
                "severity": "MEDIUM",
                "category": "Architecture",
                "file": str(auth_sequence_diagram),
                "message": "Authentication sequence diagram not found"
            })
            return

        diagram_content = auth_sequence_diagram.read_text()

        # Check for required components in authentication flow
        required_components = [
            "JWT Manager",
            "MFA Service",
            "RBAC Service",
            "Security Audit",
            "Cache Service"
        ]

        for component in required_components:
            if component in diagram_content:
                self.fixes_validated.append(f"Authentication component documented: {component}")
            else:
                self.issues.append({
                    "severity": "LOW",
                    "category": "Architecture",
                    "file": str(auth_sequence_diagram),
                    "message": f"Required authentication component not documented: {component}"
                })

        # Check for security patterns
        security_patterns = [
            "RS256",  # Secure JWT algorithm
            "MFA verification",
            "Security Audit",
            "Permission cache"
        ]

        for pattern in security_patterns:
            if pattern in diagram_content:
                self.fixes_validated.append(f"Security pattern documented: {pattern}")

    def _run_static_security_analysis(self):
        """Run static security analysis patterns."""
        python_files = list(self.src_dir.rglob("*.py"))

        security_patterns = {
            "SQL Injection": r"f['\"].*\{.*\}.*['\"].*execute",
            "Command Injection": r"subprocess\.(call|run|Popen).*shell=True",
            "Path Traversal": r"\.\./",
            "Hardcoded Password": r"password\s*=\s*['\"][^'\"]{1,}",
            "Weak Random": r"random\.(choice|randint|random)",
        }

        for file_path in python_files:
            if file_path.name.startswith("test_"):
                continue  # Skip test files

            try:
                content = file_path.read_text()

                for pattern_name, pattern in security_patterns.items():
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        self.issues.append({
                            "severity": "MEDIUM",
                            "category": "Static Analysis",
                            "file": str(file_path),
                            "message": f"Potential {pattern_name} vulnerability detected",
                            "line": line_number,
                            "code": match.group(0)
                        })

            except UnicodeDecodeError:
                continue  # Skip binary files

    def _find_line_number(self, content: str, search_text: str) -> int:
        """Find line number of text in content."""
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if search_text in line:
                return i
        return 0

    def generate_report(self) -> None:
        """Generate comprehensive security validation report."""
        console.print("\n")
        console.print(Panel(
            "Security Validation Report",
            style="bold blue",
            padding=(1, 2)
        ))

        # Summary table
        summary_table = Table(title="Validation Summary")
        summary_table.add_column("Category", style="cyan")
        summary_table.add_column("Status", style="green")
        summary_table.add_column("Count")

        critical_issues = len([i for i in self.issues if i["severity"] == "CRITICAL"])
        high_issues = len([i for i in self.issues if i["severity"] == "HIGH"])
        medium_issues = len([i for i in self.issues if i["severity"] == "MEDIUM"])
        low_issues = len([i for i in self.issues if i["severity"] == "LOW"])

        summary_table.add_row("Critical Issues", "❌" if critical_issues > 0 else "✅", str(critical_issues))
        summary_table.add_row("High Issues", "❌" if high_issues > 0 else "✅", str(high_issues))
        summary_table.add_row("Medium Issues", "⚠️" if medium_issues > 0 else "✅", str(medium_issues))
        summary_table.add_row("Low Issues", "ℹ️" if low_issues > 0 else "✅", str(low_issues))
        summary_table.add_row("Fixes Validated", "✅", str(len(self.fixes_validated)))

        console.print(summary_table)
        console.print("\n")

        # Fixes validated
        if self.fixes_validated:
            console.print(Panel(
                "✅ Security Fixes Validated",
                style="green",
                padding=(0, 2)
            ))
            for fix in self.fixes_validated:
                console.print(f"  • {fix}")
            console.print("\n")

        # Issues found
        if self.issues:
            console.print(Panel(
                "🚨 Security Issues Found",
                style="red",
                padding=(0, 2)
            ))

            issues_table = Table()
            issues_table.add_column("Severity", style="bold")
            issues_table.add_column("Category")
            issues_table.add_column("File")
            issues_table.add_column("Line")
            issues_table.add_column("Message")

            for issue in sorted(self.issues, key=lambda x: x["severity"]):
                severity_color = {
                    "CRITICAL": "red",
                    "HIGH": "orange1",
                    "MEDIUM": "yellow",
                    "LOW": "blue"
                }.get(issue["severity"], "white")

                issues_table.add_row(
                    f"[{severity_color}]{issue['severity']}[/{severity_color}]",
                    issue["category"],
                    Path(issue["file"]).name,
                    str(issue.get("line", "")),
                    issue["message"]
                )

            console.print(issues_table)
        else:
            console.print(Panel(
                "🎉 No Security Issues Found!",
                style="green",
                padding=(1, 2)
            ))

        # Recommendations
        console.print("\n")
        console.print(Panel(
            "Security Recommendations",
            style="cyan",
            padding=(0, 2)
        ))

        recommendations = [
            "Run security tests regularly with 'pytest tests/test_security_fixes.py'",
            "Use security scanning tools like bandit and safety in CI/CD",
            "Implement security code reviews for all changes",
            "Monitor security audit logs for suspicious activity",
            "Keep dependencies updated for security patches",
            "Use environment-specific configurations for production deployments"
        ]

        for rec in recommendations:
            console.print(f"  • {rec}")


@app.command()
def validate(
    project_root: str = typer.Option(
        "/Users/tindang/workspaces/its/its-camera-ai",
        "--project-root",
        help="Path to project root directory"
    ),
    fail_on_issues: bool = typer.Option(
        True,
        "--fail-on-issues/--no-fail",
        help="Exit with error code if issues found"
    )
):
    """Validate all security fixes and generate comprehensive report."""

    project_path = Path(project_root)
    if not project_path.exists():
        rprint(f"[red]❌ Project root not found: {project_root}[/red]")
        raise typer.Exit(1)

    validator = SecurityValidator(project_path)

    # Run validation
    success = validator.validate_all()

    # Generate report
    validator.generate_report()

    # Exit with appropriate code
    if not success and fail_on_issues:
        critical_or_high = any(
            issue["severity"] in ["CRITICAL", "HIGH"]
            for issue in validator.issues
        )
        if critical_or_high:
            rprint("\n[red]❌ Critical or high severity security issues found![/red]")
            raise typer.Exit(1)

    rprint("\n[green]✅ Security validation completed successfully![/green]")


@app.command()
def check_fixes():
    """Quick check that specific P0 fixes are in place."""
    project_root = Path("/Users/tindang/workspaces/its/its-camera-ai")

    fixes_status = {
        "CWE-377 (Temp Files)": False,
        "CWE-605 (Network Binding)": False,
        "Hardcoded Secrets": False
    }

    # Check temp file fixes
    model_mgmt = project_root / "src/its_camera_ai/api/routers/model_management.py"
    if model_mgmt.exists():
        content = model_mgmt.read_text()
        if "NamedTemporaryFile" in content and "mktemp(" not in content:
            fixes_status["CWE-377 (Temp Files)"] = True

    # Check network binding fixes
    cli_file = project_root / "src/its_camera_ai/cli.py"
    if cli_file.exists():
        content = cli_file.read_text()
        if "is_production()" in content and 'default="0.0.0.0"' not in content:
            fixes_status["CWE-605 (Network Binding)"] = True

    # Check secret comparison fixes
    deploy_script = project_root / "scripts/deploy_auth_system.py"
    if deploy_script.exists():
        content = deploy_script.read_text()
        if "secrets.compare_digest" in content:
            fixes_status["Hardcoded Secrets"] = True

    # Display results
    table = Table(title="P0 Security Fixes Status")
    table.add_column("Vulnerability", style="cyan")
    table.add_column("Status", style="bold")

    for fix, status in fixes_status.items():
        status_text = "[green]✅ FIXED[/green]" if status else "[red]❌ NOT FIXED[/red]"
        table.add_row(fix, status_text)

    console.print(table)

    all_fixed = all(fixes_status.values())
    if all_fixed:
        rprint("\n[green]🎉 All P0 security fixes are in place![/green]")
    else:
        rprint("\n[red]❌ Some P0 security fixes are missing![/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
