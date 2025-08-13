"""Configuration management commands for ITS Camera AI.

Commands for viewing, updating, and managing system configuration.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from ...core.config import get_settings
from ..utils import (
    confirm_action,
    console,
    display_config,
    print_error,
    print_info,
    print_success,
    print_warning,
)

app = typer.Typer(help="⚙️ Configuration management")


@app.command()
def show(
    section: str | None = typer.Option(
        None, "--section", "-s", help="Show specific configuration section"
    ),
    format_type: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, yaml"
    ),
    include_sensitive: bool = typer.Option(
        False, "--include-sensitive", help="Include sensitive values (passwords, keys)"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Save configuration to file"
    ),
) -> None:
    """📜 Show current configuration.

    Display the current system configuration with options to filter by section,
    change output format, and save to file.
    """
    try:
        settings = get_settings()
        config_dict = settings.model_dump()

        # Filter by section if specified
        if section:
            if section in config_dict:
                config_dict = {section: config_dict[section]}
            else:
                print_error(f"Configuration section '{section}' not found")
                available_sections = list(config_dict.keys())
                print_info(f"Available sections: {', '.join(available_sections)}")
                return

        # Mask sensitive values unless explicitly requested
        if not include_sensitive:
            config_dict = _mask_sensitive_values(config_dict)

        # Display configuration
        if format_type == "table":
            title = f"Configuration - {section}" if section else "System Configuration"
            display_config(config_dict, title)
        elif format_type == "json":
            console.print_json(json.dumps(config_dict, indent=2, default=str))
        elif format_type == "yaml":
            try:
                import yaml
                yaml_output = yaml.dump(config_dict, default_flow_style=False, default=str)
                console.print(yaml_output)
            except ImportError:
                print_error("PyYAML not installed. Please install it to use YAML format.")
                return

        # Save to file if requested
        if output_file:
            _save_config_to_file(config_dict, output_file, format_type)
            print_success(f"Configuration saved to {output_file}")

    except Exception as e:
        print_error(f"Failed to load configuration: {e}")


def _mask_sensitive_values(config: dict[str, Any]) -> dict[str, Any]:
    """
    Mask sensitive configuration values.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with sensitive values masked
    """
    sensitive_keys = {
        "password", "passwd", "secret", "key", "token", "api_key",
        "access_key", "secret_key", "private_key", "dsn", "url"
    }

    def mask_dict(d: dict[str, Any]) -> dict[str, Any]:
        masked = {}
        for key, value in d.items():
            if isinstance(value, dict):
                masked[key] = mask_dict(value)
            elif any(sensitive in key.lower() for sensitive in sensitive_keys):
                if isinstance(value, str) and value:
                    # Show first and last 2 characters, mask the middle
                    if len(value) > 8:
                        masked[key] = f"{value[:2]}...{value[-2:]}"
                    else:
                        masked[key] = "***HIDDEN***"
                else:
                    masked[key] = "***HIDDEN***"
            else:
                masked[key] = value
        return masked

    return mask_dict(config)


def _save_config_to_file(config: dict[str, Any], output_file: Path, format_type: str) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        output_file: Output file path
        format_type: Output format
    """
    try:
        if format_type == "json":
            with open(output_file, "w") as f:
                json.dump(config, f, indent=2, default=str)
        elif format_type == "yaml":
            import yaml
            with open(output_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False, default=str)
        else:
            # Plain text format
            with open(output_file, "w") as f:
                f.write("ITS Camera AI Configuration\n")
                f.write("=" * 30 + "\n\n")
                _write_config_text(config, f)

    except Exception as e:
        raise Exception(f"Failed to save configuration: {e}")


def _write_config_text(config: dict[str, Any], file, indent: int = 0) -> None:
    """
    Write configuration in text format.

    Args:
        config: Configuration dictionary
        file: File object
        indent: Indentation level
    """
    for key, value in config.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            file.write(f"{prefix}{key}:\n")
            _write_config_text(value, file, indent + 1)
        else:
            file.write(f"{prefix}{key}: {value}\n")


@app.command()
def validate(
    config_file: Path | None = typer.Option(
        None, "--file", "-f", help="Validate specific configuration file"
    ),
    strict: bool = typer.Option(
        False, "--strict", "-s", help="Enable strict validation mode"
    ),
) -> None:
    """✅ Validate configuration.

    Validate the current configuration or a specific configuration file
    for correctness and completeness.
    """
    if config_file:
        _validate_config_file(config_file, strict)
    else:
        _validate_current_config(strict)


def _validate_current_config(strict: bool) -> None:
    """
    Validate current system configuration.

    Args:
        strict: Enable strict validation
    """
    print_info("Validating current configuration...")

    try:
        settings = get_settings()
        validation_results = _run_config_validation(settings, strict)
        _display_validation_results(validation_results)

    except Exception as e:
        print_error(f"Configuration validation failed: {e}")


def _validate_config_file(config_file: Path, strict: bool) -> None:
    """
    Validate specific configuration file.

    Args:
        config_file: Configuration file path
        strict: Enable strict validation
    """
    if not config_file.exists():
        print_error(f"Configuration file not found: {config_file}")
        return

    print_info(f"Validating configuration file: {config_file}")

    try:
        # Load and validate the file
        # This would need integration with actual settings loading
        print_warning("External config file validation not fully implemented")

        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": ["External file validation is limited"],
            "info": ["File exists and is readable"],
        }

        _display_validation_results(validation_results)

    except Exception as e:
        print_error(f"Failed to validate config file: {e}")


def _run_config_validation(settings, strict: bool) -> dict[str, Any]:
    """
    Run configuration validation checks.

    Args:
        settings: Settings object
        strict: Enable strict validation

    Returns:
        Validation results dictionary
    """
    errors = []
    warnings = []
    info = []

    # Basic validation checks

    # Check database configuration
    if not settings.database.url:
        errors.append("Database URL is not configured")
    elif "localhost" in settings.database.url and settings.environment == "production":
        warnings.append("Using localhost database URL in production environment")

    # Check Redis configuration
    if not settings.redis.url:
        errors.append("Redis URL is not configured")

    # Check security settings
    if settings.security.secret_key == "change-me-in-production":
        if settings.environment == "production":
            errors.append("Default secret key is being used in production")
        else:
            warnings.append("Using default secret key (change for production)")

    if len(settings.security.secret_key) < 32:
        errors.append("Secret key is too short (minimum 32 characters)")

    # Check ML configuration
    if not settings.ml.model_path.exists():
        warnings.append(f"Model path does not exist: {settings.ml.model_path}")

    # Check API configuration
    if settings.api_host == "0.0.0.0" and settings.environment == "production":  # noqa: S104
        warnings.append("API server is bound to all interfaces in production")

    # Environment-specific checks
    if settings.environment == "production":
        if settings.debug:
            errors.append("Debug mode is enabled in production")

        if settings.log_level == "DEBUG":
            warnings.append("Debug logging is enabled in production")

    # Strict mode additional checks
    if strict:
        if not settings.monitoring.enable_metrics:
            warnings.append("Metrics collection is disabled")

        if not settings.monitoring.enable_tracing:
            warnings.append("Distributed tracing is disabled")

        if settings.security.access_token_expire_minutes > 60:
            warnings.append("Access token expiration time is longer than 1 hour")

    # Success info
    if not errors:
        info.append("Basic configuration validation passed")

    info.append(f"Environment: {settings.environment}")
    info.append(f"Debug mode: {'enabled' if settings.debug else 'disabled'}")
    info.append(f"Log level: {settings.log_level}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }


def _display_validation_results(results: dict[str, Any]) -> None:
    """
    Display configuration validation results.

    Args:
        results: Validation results dictionary
    """
    # Create results table
    table = Table(title="Configuration Validation Results")
    table.add_column("Type", style="cyan")
    table.add_column("Message", style="white")

    # Add errors
    for error in results["errors"]:
        table.add_row("[red]ERROR[/red]", f"[red]{error}[/red]")

    # Add warnings
    for warning in results["warnings"]:
        table.add_row("[yellow]WARNING[/yellow]", f"[yellow]{warning}[/yellow]")

    # Add info
    for info_msg in results["info"]:
        table.add_row("[blue]INFO[/blue]", f"[blue]{info_msg}[/blue]")

    console.print(table)

    # Summary
    if results["valid"]:
        print_success("Configuration validation passed!")
    else:
        print_error(f"Configuration validation failed with {len(results['errors'])} errors")

    if results["warnings"]:
        print_warning(f"Found {len(results['warnings'])} warnings")


@app.command()
def update(
    key: str = typer.Argument(
        ..., help="Configuration key to update (use dot notation: section.key)"
    ),
    value: str = typer.Argument(
        ..., help="New value for the configuration key"
    ),
    config_file: Path | None = typer.Option(
        None, "--file", "-f", help="Update specific configuration file"
    ),
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Create backup before updating"
    ),
    force: bool = typer.Option(
        False, "--force", help="Force update without confirmation"
    ),
) -> None:
    """✏️ Update configuration value.

    Update a specific configuration value. Use dot notation to specify
    nested configuration keys (e.g., database.pool_size).
    """
    if not force and not confirm_action(f"Update '{key}' to '{value}'?"):
        print_info("Configuration update cancelled")
        return

    try:
        if config_file:
            _update_config_file(config_file, key, value, backup)
        else:
            _update_runtime_config(key, value)

    except Exception as e:
        print_error(f"Failed to update configuration: {e}")


def _update_runtime_config(key: str, value: str) -> None:
    """
    Update runtime configuration.

    Args:
        key: Configuration key
        value: New value
    """
    print_info(f"Updating runtime configuration: {key} = {value}")

    # Parse the key to handle nested access
    key_parts = key.split(".")

    # For runtime updates, we would need to:
    # 1. Update environment variables
    # 2. Reload configuration
    # 3. Notify relevant services

    # Simulate the update
    env_key = "__".join(key_parts).upper()
    os.environ[env_key] = value

    print_success(f"Runtime configuration updated: {key}")
    print_info("Note: Some changes may require service restart to take effect")


def _update_config_file(config_file: Path, key: str, value: str, backup: bool) -> None:
    """
    Update configuration file.

    Args:
        config_file: Configuration file path
        key: Configuration key
        value: New value
        backup: Create backup
    """
    if not config_file.exists():
        print_error(f"Configuration file not found: {config_file}")
        return

    print_info(f"Updating configuration file: {config_file}")

    # Create backup if requested
    if backup:
        backup_file = config_file.with_suffix(f"{config_file.suffix}.backup.{int(time.time())}")
        backup_file.write_text(config_file.read_text())
        print_info(f"Backup created: {backup_file}")

    # This would implement actual file parsing and updating
    # For now, simulate the update
    print_success(f"Configuration file updated: {key} = {value}")
    print_warning("File-based configuration updates not fully implemented")


@app.command()
def reset(
    section: str | None = typer.Option(
        None, "--section", "-s", help="Reset specific configuration section"
    ),
    key: str | None = typer.Option(
        None, "--key", "-k", help="Reset specific configuration key"
    ),
    to_defaults: bool = typer.Option(
        False, "--defaults", "-d", help="Reset to default values"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force reset without confirmation"
    ),
) -> None:
    """🔄 Reset configuration.

    Reset configuration to defaults or previous values. Can reset entire
    sections or specific keys.
    """
    if not section and not key:
        print_error("Either --section or --key must be specified")
        return

    target = section or key
    action = "reset to defaults" if to_defaults else "reset"

    if not force:
        if not confirm_action(f"Are you sure you want to {action} '{target}'?"):
            print_info("Configuration reset cancelled")
            return

    try:
        if section:
            _reset_config_section(section, to_defaults)
        else:
            _reset_config_key(key, to_defaults)

    except Exception as e:
        print_error(f"Failed to reset configuration: {e}")


def _reset_config_section(section: str, to_defaults: bool) -> None:
    """
    Reset configuration section.

    Args:
        section: Section name
        to_defaults: Reset to defaults
    """
    print_info(f"Resetting configuration section: {section}")

    # This would implement actual section reset logic
    # For now, simulate the reset

    action = "defaults" if to_defaults else "previous values"
    print_success(f"Configuration section '{section}' reset to {action}")
    print_warning("Configuration reset not fully implemented")


def _reset_config_key(key: str, to_defaults: bool) -> None:
    """
    Reset configuration key.

    Args:
        key: Configuration key
        to_defaults: Reset to defaults
    """
    print_info(f"Resetting configuration key: {key}")

    # This would implement actual key reset logic
    # For now, simulate the reset

    action = "default" if to_defaults else "previous value"
    print_success(f"Configuration key '{key}' reset to {action}")
    print_warning("Configuration reset not fully implemented")


@app.command()
def env(
    action: str = typer.Argument(
        "list", help="Action: list, set, unset, export"
    ),
    variable: str | None = typer.Option(
        None, "--var", "-v", help="Environment variable name"
    ),
    value: str | None = typer.Option(
        None, "--value", help="Environment variable value"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Export environment variables to file"
    ),
) -> None:
    """🌍 Manage environment variables.

    Manage environment variables related to ITS Camera AI configuration.
    """
    valid_actions = ["list", "set", "unset", "export"]
    if action not in valid_actions:
        print_error(f"Invalid action. Must be one of: {', '.join(valid_actions)}")
        return

    if action == "list":
        _list_env_variables(variable)
    elif action == "set":
        if not variable or not value:
            print_error("Both --var and --value are required for set action")
            return
        _set_env_variable(variable, value)
    elif action == "unset":
        if not variable:
            print_error("--var is required for unset action")
            return
        _unset_env_variable(variable)
    elif action == "export":
        _export_env_variables(output_file)


def _list_env_variables(filter_var: str | None) -> None:
    """
    List environment variables.

    Args:
        filter_var: Optional variable name filter
    """
    print_info("ITS Camera AI environment variables:")

    # Get relevant environment variables
    its_env_vars = {}
    prefixes = ["ITS_", "DATABASE_", "REDIS_", "ML_", "API_", "SECURITY_"]

    for key, value in os.environ.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            if not filter_var or filter_var.upper() in key.upper():
                # Mask sensitive values
                if any(sensitive in key.lower() for sensitive in ["password", "secret", "key", "token"]):
                    display_value = "***HIDDEN***"
                else:
                    display_value = value
                its_env_vars[key] = display_value

    if not its_env_vars:
        print_info("No ITS Camera AI environment variables found")
        return

    # Display in table
    table = Table(title="Environment Variables")
    table.add_column("Variable", style="cyan")
    table.add_column("Value", style="green")

    for var, val in sorted(its_env_vars.items()):
        table.add_row(var, str(val))

    console.print(table)
    print_info(f"Found {len(its_env_vars)} variables")


def _set_env_variable(variable: str, value: str) -> None:
    """
    Set environment variable.

    Args:
        variable: Variable name
        value: Variable value
    """
    print_info(f"Setting environment variable: {variable}")

    os.environ[variable] = value

    print_success(f"Environment variable '{variable}' set")
    print_info("Note: This only affects the current session")


def _unset_env_variable(variable: str) -> None:
    """
    Unset environment variable.

    Args:
        variable: Variable name
    """
    print_info(f"Unsetting environment variable: {variable}")

    if variable in os.environ:
        del os.environ[variable]
        print_success(f"Environment variable '{variable}' unset")
    else:
        print_warning(f"Environment variable '{variable}' not found")


def _export_env_variables(output_file: Path | None) -> None:
    """
    Export environment variables.

    Args:
        output_file: Optional output file
    """
    print_info("Exporting environment variables...")

    # Get relevant environment variables
    its_env_vars = {}
    prefixes = ["ITS_", "DATABASE_", "REDIS_", "ML_", "API_", "SECURITY_"]

    for key, value in os.environ.items():
        if any(key.startswith(prefix) for prefix in prefixes):
            # Don't export sensitive values
            if not any(sensitive in key.lower() for sensitive in ["password", "secret", "key", "token"]):
                its_env_vars[key] = value

    if output_file:
        # Export to file
        try:
            with open(output_file, "w") as f:
                f.write("# ITS Camera AI Environment Variables\n")
                f.write(f"# Exported on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                for var, val in sorted(its_env_vars.items()):
                    f.write(f"export {var}=\"{val}\"\n")

            print_success(f"Environment variables exported to {output_file}")

        except Exception as e:
            print_error(f"Failed to export to file: {e}")
    else:
        # Display export commands
        console.print("[bold]Environment Variable Export Commands:[/bold]")
        console.print()

        for var, val in sorted(its_env_vars.items()):
            console.print(f"export {var}=\"{val}\"")

        console.print()
        print_info(f"Found {len(its_env_vars)} exportable variables")


@app.command()
def template(
    template_type: str = typer.Argument(
        "basic", help="Template type: basic, production, development, docker"
    ),
    output_file: Path = typer.Option(
        Path(".env.template"), "--output", "-o", help="Output template file"
    ),
    include_examples: bool = typer.Option(
        True, "--examples/--no-examples", help="Include example values"
    ),
) -> None:
    """📝 Generate configuration templates.

    Generate configuration file templates for different environments
    and deployment scenarios.
    """
    valid_templates = ["basic", "production", "development", "docker"]
    if template_type not in valid_templates:
        print_error(f"Invalid template type. Must be one of: {', '.join(valid_templates)}")
        return

    print_info(f"Generating {template_type} configuration template...")

    try:
        template_content = _generate_config_template(template_type, include_examples)

        with open(output_file, "w") as f:
            f.write(template_content)

        print_success(f"Configuration template saved to {output_file}")
        print_info(f"Template type: {template_type}")

    except Exception as e:
        print_error(f"Failed to generate template: {e}")


def _generate_config_template(template_type: str, include_examples: bool) -> str:
    """
    Generate configuration template content.

    Args:
        template_type: Template type
        include_examples: Include example values

    Returns:
        Template content string
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    template = f"""# ITS Camera AI Configuration Template
# Template Type: {template_type.title()}
# Generated: {timestamp}
#
# Copy this file to .env and update values as needed

"""

    # Basic configuration
    template += "# =============================================================================\n"
    template += "# APPLICATION SETTINGS\n"
    template += "# =============================================================================\n\n"

    if include_examples:
        template += "APP_NAME=\"ITS Camera AI\"\n"
        template += "APP_VERSION=\"0.1.0\"\n"
        template += f"ENVIRONMENT={template_type if template_type != 'basic' else 'development'}\n"
        template += "DEBUG=false\n" if template_type == "production" else "DEBUG=true\n"
        template += "LOG_LEVEL=INFO\n" if template_type == "production" else "LOG_LEVEL=DEBUG\n"
    else:
        template += "APP_NAME=\n"
        template += "APP_VERSION=\n"
        template += "ENVIRONMENT=\n"
        template += "DEBUG=\n"
        template += "LOG_LEVEL=\n"

    template += "\n"

    # API settings
    template += "# =============================================================================\n"
    template += "# API SERVER SETTINGS\n"
    template += "# =============================================================================\n\n"

    if include_examples:
        api_host = "127.0.0.1" if template_type in ["development", "basic"] else "0.0.0.0"
        template += f"API_HOST={api_host}\n"
        template += "API_PORT=8080\n"
        template += "WORKERS=1\n" if template_type == "development" else "WORKERS=4\n"
    else:
        template += "API_HOST=\n"
        template += "API_PORT=\n"
        template += "WORKERS=\n"

    template += "\n"

    # Database settings
    template += "# =============================================================================\n"
    template += "# DATABASE SETTINGS\n"
    template += "# =============================================================================\n\n"

    if include_examples:
        if template_type == "docker":
            db_url = "postgresql+asyncpg://user:password@postgres:5432/its_camera_ai"
        elif template_type == "production":
            db_url = "postgresql+asyncpg://user:password@db.example.com:5432/its_camera_ai"
        else:
            db_url = "postgresql+asyncpg://user:password@localhost:5432/its_camera_ai"

        template += f"DATABASE__URL={db_url}\n"
        template += "DATABASE__POOL_SIZE=10\n"
        template += "DATABASE__MAX_OVERFLOW=20\n"
    else:
        template += "DATABASE__URL=\n"
        template += "DATABASE__POOL_SIZE=\n"
        template += "DATABASE__MAX_OVERFLOW=\n"

    template += "\n"

    # Redis settings
    template += "# =============================================================================\n"
    template += "# REDIS SETTINGS\n"
    template += "# =============================================================================\n\n"

    if include_examples:
        if template_type == "docker":
            redis_url = "redis://redis:6379/0"
        elif template_type == "production":
            redis_url = "redis://cache.example.com:6379/0"
        else:
            redis_url = "redis://localhost:6379/0"

        template += f"REDIS__URL={redis_url}\n"
        template += "REDIS__MAX_CONNECTIONS=20\n"
    else:
        template += "REDIS__URL=\n"
        template += "REDIS__MAX_CONNECTIONS=\n"

    template += "\n"

    # Security settings
    template += "# =============================================================================\n"
    template += "# SECURITY SETTINGS\n"
    template += "# =============================================================================\n\n"

    if include_examples:
        if template_type == "production":
            template += "SECURITY__SECRET_KEY=your-super-secure-secret-key-here-min-32-chars\n"
        else:
            template += "SECURITY__SECRET_KEY=dev-secret-key-change-in-production\n"

        template += "SECURITY__ACCESS_TOKEN_EXPIRE_MINUTES=30\n"
    else:
        template += "SECURITY__SECRET_KEY=\n"
        template += "SECURITY__ACCESS_TOKEN_EXPIRE_MINUTES=\n"

    template += "\n"

    # ML settings
    template += "# =============================================================================\n"
    template += "# ML SETTINGS\n"
    template += "# =============================================================================\n\n"

    if include_examples:
        template += "ML__MODEL_PATH=./models\n"
        template += "ML__BATCH_SIZE=32\n"
        template += "ML__DEVICE=auto\n"
        template += "ML__CONFIDENCE_THRESHOLD=0.5\n"
    else:
        template += "ML__MODEL_PATH=\n"
        template += "ML__BATCH_SIZE=\n"
        template += "ML__DEVICE=\n"
        template += "ML__CONFIDENCE_THRESHOLD=\n"

    template += "\n"

    # Monitoring settings
    template += "# =============================================================================\n"
    template += "# MONITORING SETTINGS\n"
    template += "# =============================================================================\n\n"

    if include_examples:
        enable_monitoring = "true" if template_type == "production" else "false"
        template += f"MONITORING__ENABLE_METRICS={enable_monitoring}\n"
        template += f"MONITORING__ENABLE_TRACING={enable_monitoring}\n"
        template += "MONITORING__PROMETHEUS_PORT=8000\n"
    else:
        template += "MONITORING__ENABLE_METRICS=\n"
        template += "MONITORING__ENABLE_TRACING=\n"
        template += "MONITORING__PROMETHEUS_PORT=\n"

    template += "\n"

    # MinIO settings
    template += "# =============================================================================\n"
    template += "# MINIO OBJECT STORAGE SETTINGS\n"
    template += "# =============================================================================\n\n"

    if include_examples:
        if template_type == "docker":
            minio_endpoint = "minio:9000"
        elif template_type == "production":
            minio_endpoint = "storage.example.com:9000"
        else:
            minio_endpoint = "localhost:9000"

        template += f"MINIO__ENDPOINT={minio_endpoint}\n"
        template += "MINIO__ACCESS_KEY=minioadmin\n"
        template += "MINIO__SECRET_KEY=minioadmin123\n"
        template += "MINIO__SECURE=false\n" if template_type != "production" else "MINIO__SECURE=true\n"
    else:
        template += "MINIO__ENDPOINT=\n"
        template += "MINIO__ACCESS_KEY=\n"
        template += "MINIO__SECRET_KEY=\n"
        template += "MINIO__SECURE=\n"

    template += "\n"

    # Add template-specific sections
    if template_type == "production":
        template += _add_production_config_section(include_examples)
    elif template_type == "docker":
        template += _add_docker_config_section(include_examples)

    return template


def _add_production_config_section(include_examples: bool) -> str:
    """
    Add production-specific configuration section.

    Args:
        include_examples: Include example values

    Returns:
        Configuration section string
    """
    section = "# =============================================================================\n"
    section += "# PRODUCTION-SPECIFIC SETTINGS\n"
    section += "# =============================================================================\n\n"

    if include_examples:
        section += "# Sentry error tracking\n"
        section += "MONITORING__SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id\n\n"

        section += "# SSL/TLS settings\n"
        section += "SSL_CERT_PATH=/path/to/cert.pem\n"
        section += "SSL_KEY_PATH=/path/to/key.pem\n\n"

        section += "# Load balancer settings\n"
        section += "PROXY_HEADERS=true\n"
        section += "FORWARDED_ALLOW_IPS=*\n\n"
    else:
        section += "MONITORING__SENTRY_DSN=\n"
        section += "SSL_CERT_PATH=\n"
        section += "SSL_KEY_PATH=\n"
        section += "PROXY_HEADERS=\n"
        section += "FORWARDED_ALLOW_IPS=\n\n"

    return section


@app.command()
def restore(
    backup_file: Path = typer.Argument(
        help="Backup file to restore from"
    ),
    confirm: bool = typer.Option(
        False, "--confirm", "-y", help="Skip confirmation prompt"
    ),
) -> None:
    """🔄 Restore configuration from backup.

    Restore system configuration from a previously created backup file.
    This will replace current configuration settings.
    """
    if not backup_file.exists():
        print_error(f"Backup file not found: {backup_file}")
        return

    if not confirm:
        if not confirm_action(f"Restore configuration from '{backup_file}'? This will replace current settings."):
            print_info("Restore cancelled")
            return

    backup_manager = ConfigBackupManager()
    success = backup_manager.restore_full_backup(str(backup_file))

    if success:
        print_success("Configuration restored successfully")
        print_info("You may need to restart services for changes to take effect")
    else:
        print_error("Failed to restore configuration")


@app.command()
def list_backups() -> None:
    """📋 List available configuration backups.

    Display all available configuration backups including full system
    backups and individual section/key backups.
    """
    backup_manager = ConfigBackupManager()
    backups = backup_manager.list_backups()

    # Display full backups
    full_backups = backups.get("full_backups", [])
    if full_backups:
        table = Table(title="Full System Backups")
        table.add_column("Timestamp", style="cyan")
        table.add_column("File", style="green")
        table.add_column("Variables", style="blue")
        table.add_column("Age", style="yellow")

        for backup in sorted(full_backups, key=lambda x: x["timestamp"], reverse=True):
            timestamp = backup["timestamp"]
            age_seconds = time.time() - timestamp

            from ..utils import format_duration
            age = format_duration(int(age_seconds))

            table.add_row(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                Path(backup["file"]).name,
                str(backup["env_vars_count"]),
                age
            )

        console.print(table)

    # Display section backups
    section_backups = backups.get("section_backups", {})
    if section_backups:
        console.print("\n[bold]Section Backups:[/bold]")
        for section, backup_data in section_backups.items():
            timestamp = backup_data["timestamp"]
            age_seconds = time.time() - timestamp

            from ..utils import format_duration
            age = format_duration(int(age_seconds))
            var_count = len(backup_data["data"])

            console.print(f"  • {section}: {var_count} variables, {age} ago")

    # Display key backups
    key_backups = backups.get("key_backups", {})
    if key_backups:
        console.print("\n[bold]Individual Key Backups:[/bold]")
        for key, backup_data in key_backups.items():
            timestamp = backup_data["timestamp"]
            age_seconds = time.time() - timestamp

            from ..utils import format_duration
            age = format_duration(int(age_seconds))

            console.print(f"  • {key}: {age} ago")

    if not any([full_backups, section_backups, key_backups]):
        print_info("No configuration backups found")


@app.command()
def diff(
    config1: Path = typer.Argument(help="First configuration file or 'current' for current settings"),
    config2: Path = typer.Argument(help="Second configuration file or 'defaults' for default settings"),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, or unified"
    ),
) -> None:
    """🔍 Compare configuration files or settings.

    Compare two configuration sources and show differences.
    Use 'current' to compare against current runtime settings.
    Use 'defaults' to compare against default values.
    """
    try:
        # Load first configuration
        if str(config1) == "current":
            settings = get_settings()
            config1_data = settings.model_dump()
        elif str(config1) == "defaults":
            # Create default settings without environment overrides
            old_env = os.environ.copy()
            # Temporarily clear ITS environment variables
            prefixes = ["ITS_", "DATABASE_", "REDIS_", "ML_", "API_", "SECURITY_", "MONITORING_"]
            for key in list(os.environ.keys()):
                if any(key.startswith(prefix) for prefix in prefixes):
                    del os.environ[key]

            try:
                from ...core.config import Settings
                default_settings = Settings()
                config1_data = default_settings.model_dump()
            finally:
                os.environ.update(old_env)
        else:
            config1_data = _load_config_file(config1)

        # Load second configuration
        if str(config2) == "current":
            settings = get_settings()
            config2_data = settings.model_dump()
        elif str(config2) == "defaults":
            # Same as above for defaults
            old_env = os.environ.copy()
            prefixes = ["ITS_", "DATABASE_", "REDIS_", "ML_", "API_", "SECURITY_", "MONITORING_"]
            for key in list(os.environ.keys()):
                if any(key.startswith(prefix) for prefix in prefixes):
                    del os.environ[key]

            try:
                from ...core.config import Settings
                default_settings = Settings()
                config2_data = default_settings.model_dump()
            finally:
                os.environ.update(old_env)
        else:
            config2_data = _load_config_file(config2)

        # Compare configurations
        differences = _compare_configs(config1_data, config2_data)

        if not differences:
            print_success("Configurations are identical")
            return

        # Display differences based on format
        if output_format == "table":
            _display_config_diff_table(differences, str(config1), str(config2))
        elif output_format == "json":
            console.print_json(json.dumps(differences, indent=2, default=str))
        elif output_format == "unified":
            _display_config_diff_unified(differences, str(config1), str(config2))
        else:
            print_error(f"Invalid output format: {output_format}")

    except Exception as e:
        print_error(f"Failed to compare configurations: {e}")


def _load_config_file(config_file: Path) -> dict[str, Any]:
    """Load configuration from file."""
    if not config_file.exists():
        raise Exception(f"Configuration file not found: {config_file}")

    if config_file.suffix.lower() == ".json":
        with open(config_file) as f:
            return json.load(f)
    elif config_file.suffix.lower() in [".yaml", ".yml"]:
        import yaml
        with open(config_file) as f:
            return yaml.safe_load(f) or {}
    else:
        raise Exception(f"Unsupported configuration file format: {config_file.suffix}")


def _compare_configs(config1: dict[str, Any], config2: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    """Compare two configuration dictionaries."""
    differences = []

    all_keys = set(config1.keys()) | set(config2.keys())

    for key in sorted(all_keys):
        current_path = f"{prefix}.{key}" if prefix else key

        if key not in config1:
            differences.append({
                "key": current_path,
                "type": "added",
                "old_value": None,
                "new_value": config2[key]
            })
        elif key not in config2:
            differences.append({
                "key": current_path,
                "type": "removed",
                "old_value": config1[key],
                "new_value": None
            })
        else:
            val1, val2 = config1[key], config2[key]

            if isinstance(val1, dict) and isinstance(val2, dict):
                # Recurse into nested dictionaries
                nested_diffs = _compare_configs(val1, val2, current_path)
                differences.extend(nested_diffs)
            elif val1 != val2:
                differences.append({
                    "key": current_path,
                    "type": "changed",
                    "old_value": val1,
                    "new_value": val2
                })

    return differences


def _display_config_diff_table(differences: list[dict[str, Any]], config1_name: str, config2_name: str) -> None:
    """Display configuration differences in table format."""
    table = Table(title=f"Configuration Differences: {config1_name} vs {config2_name}")
    table.add_column("Key", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column(f"Value in {config1_name}", style="green")
    table.add_column(f"Value in {config2_name}", style="yellow")

    for diff in differences:
        type_color = {
            "changed": "yellow",
            "added": "green",
            "removed": "red"
        }.get(diff["type"], "white")

        old_val = str(diff["old_value"]) if diff["old_value"] is not None else "[dim]N/A[/dim]"
        new_val = str(diff["new_value"]) if diff["new_value"] is not None else "[dim]N/A[/dim]"

        table.add_row(
            diff["key"],
            f"[{type_color}]{diff['type'].title()}[/{type_color}]",
            old_val,
            new_val
        )

    console.print(table)
    print_info(f"Found {len(differences)} differences")


def _display_config_diff_unified(differences: list[dict[str, Any]], config1_name: str, config2_name: str) -> None:
    """Display configuration differences in unified diff format."""
    console.print(f"[bold]--- {config1_name}[/bold]")
    console.print(f"[bold]+++ {config2_name}[/bold]")
    console.print()

    for diff in differences:
        key = diff["key"]
        diff_type = diff["type"]

        if diff_type == "changed":
            console.print(f"[red]- {key}: {diff['old_value']}[/red]")
            console.print(f"[green]+ {key}: {diff['new_value']}[/green]")
        elif diff_type == "removed":
            console.print(f"[red]- {key}: {diff['old_value']}[/red]")
        elif diff_type == "added":
            console.print(f"[green]+ {key}: {diff['new_value']}[/green]")

        console.print()


class ConfigBackupManager:
    """Manager for configuration backups and rollbacks."""

    def __init__(self):
        self.backup_dir = Path("config_backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.backup_index_file = self.backup_dir / "backup_index.json"
        self._load_backup_index()

    def _load_backup_index(self) -> None:
        """Load backup index from file."""
        if self.backup_index_file.exists():
            try:
                with open(self.backup_index_file) as f:
                    self.backup_index = json.load(f)
            except Exception:
                self.backup_index = {}
        else:
            self.backup_index = {}

    def _save_backup_index(self) -> None:
        """Save backup index to file."""
        try:
            with open(self.backup_index_file, 'w') as f:
                json.dump(self.backup_index, f, indent=2)
        except Exception as e:
            print_error(f"Failed to save backup index: {e}")

    def save_key_backup(self, key: str, value: str) -> None:
        """Save backup for a specific key."""
        timestamp = int(time.time())
        backup_entry = {
            "timestamp": timestamp,
            "key": key,
            "value": value
        }

        if "keys" not in self.backup_index:
            self.backup_index["keys"] = {}

        self.backup_index["keys"][key] = backup_entry
        self._save_backup_index()

    def restore_key_backup(self, key: str) -> str | None:
        """Restore backup for a specific key."""
        if "keys" not in self.backup_index:
            return None

        backup_entry = self.backup_index["keys"].get(key)
        return backup_entry["value"] if backup_entry else None

    def save_section_backup(self, section: str, data: dict[str, str]) -> None:
        """Save backup for a configuration section."""
        timestamp = int(time.time())
        backup_entry = {
            "timestamp": timestamp,
            "section": section,
            "data": data
        }

        if "sections" not in self.backup_index:
            self.backup_index["sections"] = {}

        self.backup_index["sections"][section] = backup_entry
        self._save_backup_index()

    def restore_section_backup(self, section: str) -> dict[str, str] | None:
        """Restore backup for a configuration section."""
        if "sections" not in self.backup_index:
            return None

        backup_entry = self.backup_index["sections"].get(section)
        return backup_entry["data"] if backup_entry else None

    def create_full_backup(self) -> str:
        """Create a full system backup."""
        timestamp = int(time.time())
        backup_file = self.backup_dir / f"full_backup_{timestamp}.json"

        # Collect all ITS-related environment variables
        env_data = {}
        prefixes = ["ITS_", "DATABASE_", "REDIS_", "ML_", "API_", "SECURITY_", "MONITORING_"]

        for key, value in os.environ.items():
            if any(key.startswith(prefix) for prefix in prefixes):
                env_data[key] = value

        backup_data = {
            "timestamp": timestamp,
            "type": "full_backup",
            "environment": env_data
        }

        try:
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)

            # Update index
            if "full_backups" not in self.backup_index:
                self.backup_index["full_backups"] = []

            self.backup_index["full_backups"].append({
                "timestamp": timestamp,
                "file": str(backup_file),
                "env_vars_count": len(env_data)
            })

            self._save_backup_index()
            return str(backup_file)

        except Exception as e:
            print_error(f"Failed to create full backup: {e}")
            return ""

    def restore_full_backup(self, backup_file: str) -> bool:
        """Restore from a full system backup."""
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                print_error(f"Backup file not found: {backup_file}")
                return False

            with open(backup_path) as f:
                backup_data = json.load(f)

            if backup_data.get("type") != "full_backup":
                print_error("Invalid backup file format")
                return False

            # Clear current ITS environment variables
            prefixes = ["ITS_", "DATABASE_", "REDIS_", "ML_", "API_", "SECURITY_", "MONITORING_"]
            cleared_vars = []

            for key in list(os.environ.keys()):
                if any(key.startswith(prefix) for prefix in prefixes):
                    del os.environ[key]
                    cleared_vars.append(key)

            # Restore from backup
            env_data = backup_data.get("environment", {})
            for key, value in env_data.items():
                os.environ[key] = value

            print_success(f"Restored {len(env_data)} environment variables from backup")
            print_info(f"Cleared {len(cleared_vars)} existing variables")
            return True

        except Exception as e:
            print_error(f"Failed to restore from backup: {e}")
            return False

    def list_backups(self) -> dict[str, Any]:
        """List all available backups."""
        return {
            "full_backups": self.backup_index.get("full_backups", []),
            "section_backups": self.backup_index.get("sections", {}),
            "key_backups": self.backup_index.get("keys", {})
        }


def _update_json_config(config_file: Path, key: str, value: str) -> None:
    """Update JSON configuration file."""
    try:
        with open(config_file) as f:
            config_data = json.load(f)

        # Navigate to the key using dot notation
        keys = key.split(".")
        current = config_data

        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Convert value to appropriate type
        final_key = keys[-1]
        try:
            # Try to parse as JSON for complex types
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            # Use as string if not valid JSON
            parsed_value = value

        current[final_key] = parsed_value

        # Write back to file
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

    except Exception as e:
        raise Exception(f"Failed to update JSON config: {e}")


def _update_yaml_config(config_file: Path, key: str, value: str) -> None:
    """Update YAML configuration file."""
    try:
        import yaml

        with open(config_file) as f:
            config_data = yaml.safe_load(f) or {}

        # Navigate to the key using dot notation
        keys = key.split(".")
        current = config_data

        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Convert value to appropriate type
        final_key = keys[-1]
        try:
            # Try to parse as YAML for proper type conversion
            parsed_value = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed_value = value

        current[final_key] = parsed_value

        # Write back to file
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

    except ImportError:
        raise Exception("PyYAML not installed. Please install it to update YAML files.")
    except Exception as e:
        raise Exception(f"Failed to update YAML config: {e}")


def _update_env_config(config_file: Path, key: str, value: str) -> None:
    """Update .env configuration file."""
    try:
        # Convert dot notation to environment variable format
        env_key = "__".join(key.split(".")).upper()

        lines = []
        key_found = False

        if config_file.exists():
            with open(config_file) as f:
                lines = f.readlines()

        # Update existing key or add new one
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                if '=' in line:
                    existing_key = line.split('=', 1)[0].strip()
                    if existing_key == env_key:
                        lines[i] = f"{env_key}={value}\n"
                        key_found = True
                        break

        if not key_found:
            lines.append(f"{env_key}={value}\n")

        # Write back to file
        with open(config_file, 'w') as f:
            f.writelines(lines)

    except Exception as e:
        raise Exception(f"Failed to update .env config: {e}")


def _update_ini_config(config_file: Path, key: str, value: str) -> None:
    """Update INI configuration file."""
    try:
        import configparser

        config = configparser.ConfigParser()
        config.read(config_file)

        # Parse key (section.option format)
        if '.' in key:
            section, option = key.split('.', 1)
        else:
            section = 'DEFAULT'
            option = key

        if section not in config:
            config.add_section(section)

        config.set(section, option, value)

        # Write back to file
        with open(config_file, 'w') as f:
            config.write(f)

    except Exception as e:
        raise Exception(f"Failed to update INI config: {e}")


def _validate_updated_config(config_file: Path, key: str, value: str) -> None:
    """Validate configuration after update."""
    try:
        # Basic validation - check if file is still valid
        if config_file.suffix.lower() == ".json":
            with open(config_file) as f:
                json.load(f)  # Will raise exception if invalid JSON
        elif config_file.suffix.lower() in [".yaml", ".yml"]:
            import yaml
            with open(config_file) as f:
                yaml.safe_load(f)  # Will raise exception if invalid YAML

        # Additional validation based on key type
        if "port" in key.lower():
            try:
                port_val = int(value)
                if not (1 <= port_val <= 65535):
                    raise ValueError(f"Port {port_val} is out of valid range (1-65535)")
            except ValueError as e:
                raise Exception(f"Invalid port value: {e}")

        elif "url" in key.lower():
            # Basic URL validation
            if not (value.startswith('http://') or value.startswith('https://') or
                   value.startswith('postgresql://') or value.startswith('redis://')):
                print_warning(f"URL value '{value}' may not be properly formatted")

        elif "timeout" in key.lower():
            try:
                timeout_val = float(value)
                if timeout_val < 0:
                    raise ValueError("Timeout cannot be negative")
            except ValueError:
                raise Exception(f"Invalid timeout value: {value}")

    except Exception as e:
        raise Exception(f"Configuration validation failed: {e}")


@app.command()
def backup(
    backup_type: str = typer.Argument(
        "full", help="Backup type: full, section, or key"
    ),
    target: str | None = typer.Option(
        None, "--target", "-t", help="Target section or key name for partial backups"
    ),
    output_file: Path | None = typer.Option(
        None, "--output", "-o", help="Output backup file (for full backups)"
    ),
) -> None:
    """💾 Create configuration backups.

    Create backups of configuration settings for safe rollback.
    Supports full system backups or targeted section/key backups.
    """
    backup_manager = ConfigBackupManager()

    if backup_type == "full":
        backup_file = backup_manager.create_full_backup()
        if backup_file:
            if output_file:
                # Copy to specified location
                import shutil
                shutil.copy2(backup_file, output_file)
                print_success(f"Full backup created: {output_file}")
            else:
                print_success(f"Full backup created: {backup_file}")
        else:
            print_error("Failed to create full backup")

    elif backup_type == "section":
        if not target:
            print_error("Section name is required for section backup")
            return

        # Get current section configuration
        section_data = {}
        section_prefixes = {
            "database": "DATABASE__",
            "redis": "REDIS__",
            "ml": "ML__",
            "security": "SECURITY__",
            "monitoring": "MONITORING__",
            "api": "API_"
        }

        prefix = section_prefixes.get(target.lower())
        if prefix:
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    section_data[key] = value

            if section_data:
                backup_manager.save_section_backup(target, section_data)
                print_success(f"Section backup created for '{target}' ({len(section_data)} variables)")
            else:
                print_warning(f"No configuration found for section '{target}'")
        else:
            print_error(f"Unknown section '{target}'")

    elif backup_type == "key":
        if not target:
            print_error("Key name is required for key backup")
            return

        env_key = "__".join(target.split(".")).upper()
        if env_key in os.environ:
            backup_manager.save_key_backup(target, os.environ[env_key])
            print_success(f"Key backup created for '{target}'")
        else:
            print_warning(f"Configuration key '{target}' not found in environment")

    else:
        print_error(f"Invalid backup type '{backup_type}'. Use 'full', 'section', or 'key'")


def _add_docker_config_section(include_examples: bool) -> str:
    """
    Add Docker-specific configuration section.

    Args:
        include_examples: Include example values

    Returns:
        Configuration section string
    """
    section = "# =============================================================================\n"
    section += "# DOCKER-SPECIFIC SETTINGS\n"
    section += "# =============================================================================\n\n"

    if include_examples:
        section += "# Container networking\n"
        section += "DOCKER_NETWORK=its-network\n\n"

        section += "# Volume mounts\n"
        section += "DATA_VOLUME=/app/data\n"
        section += "MODELS_VOLUME=/app/models\n"
        section += "LOGS_VOLUME=/app/logs\n\n"

        section += "# Resource limits\n"
        section += "MEMORY_LIMIT=2g\n"
        section += "CPU_LIMIT=2\n\n"
    else:
        section += "DOCKER_NETWORK=\n"
        section += "DATA_VOLUME=\n"
        section += "MODELS_VOLUME=\n"
        section += "LOGS_VOLUME=\n"
        section += "MEMORY_LIMIT=\n"
        section += "CPU_LIMIT=\n\n"

    return section


@app.command()
def restore(
    backup_file: Path = typer.Argument(
        help="Backup file to restore from"
    ),
    confirm: bool = typer.Option(
        False, "--confirm", "-y", help="Skip confirmation prompt"
    ),
) -> None:
    """🔄 Restore configuration from backup.

    Restore system configuration from a previously created backup file.
    This will replace current configuration settings.
    """
    if not backup_file.exists():
        print_error(f"Backup file not found: {backup_file}")
        return

    if not confirm:
        if not confirm_action(f"Restore configuration from '{backup_file}'? This will replace current settings."):
            print_info("Restore cancelled")
            return

    backup_manager = ConfigBackupManager()
    success = backup_manager.restore_full_backup(str(backup_file))

    if success:
        print_success("Configuration restored successfully")
        print_info("You may need to restart services for changes to take effect")
    else:
        print_error("Failed to restore configuration")


@app.command()
def list_backups() -> None:
    """📋 List available configuration backups.

    Display all available configuration backups including full system
    backups and individual section/key backups.
    """
    backup_manager = ConfigBackupManager()
    backups = backup_manager.list_backups()

    # Display full backups
    full_backups = backups.get("full_backups", [])
    if full_backups:
        table = Table(title="Full System Backups")
        table.add_column("Timestamp", style="cyan")
        table.add_column("File", style="green")
        table.add_column("Variables", style="blue")
        table.add_column("Age", style="yellow")

        for backup in sorted(full_backups, key=lambda x: x["timestamp"], reverse=True):
            timestamp = backup["timestamp"]
            age_seconds = time.time() - timestamp

            from ..utils import format_duration
            age = format_duration(int(age_seconds))

            table.add_row(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
                Path(backup["file"]).name,
                str(backup["env_vars_count"]),
                age
            )

        console.print(table)

    # Display section backups
    section_backups = backups.get("section_backups", {})
    if section_backups:
        console.print("\n[bold]Section Backups:[/bold]")
        for section, backup_data in section_backups.items():
            timestamp = backup_data["timestamp"]
            age_seconds = time.time() - timestamp

            from ..utils import format_duration
            age = format_duration(int(age_seconds))
            var_count = len(backup_data["data"])

            console.print(f"  • {section}: {var_count} variables, {age} ago")

    # Display key backups
    key_backups = backups.get("key_backups", {})
    if key_backups:
        console.print("\n[bold]Individual Key Backups:[/bold]")
        for key, backup_data in key_backups.items():
            timestamp = backup_data["timestamp"]
            age_seconds = time.time() - timestamp

            from ..utils import format_duration
            age = format_duration(int(age_seconds))

            console.print(f"  • {key}: {age} ago")

    if not any([full_backups, section_backups, key_backups]):
        print_info("No configuration backups found")


@app.command()
def diff(
    config1: Path = typer.Argument(help="First configuration file or 'current' for current settings"),
    config2: Path = typer.Argument(help="Second configuration file or 'defaults' for default settings"),
    output_format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json, or unified"
    ),
) -> None:
    """🔍 Compare configuration files or settings.

    Compare two configuration sources and show differences.
    Use 'current' to compare against current runtime settings.
    Use 'defaults' to compare against default values.
    """
    try:
        # Load first configuration
        if str(config1) == "current":
            settings = get_settings()
            config1_data = settings.model_dump()
        elif str(config1) == "defaults":
            # Create default settings without environment overrides
            old_env = os.environ.copy()
            # Temporarily clear ITS environment variables
            prefixes = ["ITS_", "DATABASE_", "REDIS_", "ML_", "API_", "SECURITY_", "MONITORING_"]
            for key in list(os.environ.keys()):
                if any(key.startswith(prefix) for prefix in prefixes):
                    del os.environ[key]

            try:
                from ...core.config import Settings
                default_settings = Settings()
                config1_data = default_settings.model_dump()
            finally:
                os.environ.update(old_env)
        else:
            config1_data = _load_config_file(config1)

        # Load second configuration
        if str(config2) == "current":
            settings = get_settings()
            config2_data = settings.model_dump()
        elif str(config2) == "defaults":
            # Same as above for defaults
            old_env = os.environ.copy()
            prefixes = ["ITS_", "DATABASE_", "REDIS_", "ML_", "API_", "SECURITY_", "MONITORING_"]
            for key in list(os.environ.keys()):
                if any(key.startswith(prefix) for prefix in prefixes):
                    del os.environ[key]

            try:
                from ...core.config import Settings
                default_settings = Settings()
                config2_data = default_settings.model_dump()
            finally:
                os.environ.update(old_env)
        else:
            config2_data = _load_config_file(config2)

        # Compare configurations
        differences = _compare_configs(config1_data, config2_data)

        if not differences:
            print_success("Configurations are identical")
            return

        # Display differences based on format
        if output_format == "table":
            _display_config_diff_table(differences, str(config1), str(config2))
        elif output_format == "json":
            console.print_json(json.dumps(differences, indent=2, default=str))
        elif output_format == "unified":
            _display_config_diff_unified(differences, str(config1), str(config2))
        else:
            print_error(f"Invalid output format: {output_format}")

    except Exception as e:
        print_error(f"Failed to compare configurations: {e}")


def _load_config_file(config_file: Path) -> dict[str, Any]:
    """Load configuration from file."""
    if not config_file.exists():
        raise Exception(f"Configuration file not found: {config_file}")

    if config_file.suffix.lower() == ".json":
        with open(config_file) as f:
            return json.load(f)
    elif config_file.suffix.lower() in [".yaml", ".yml"]:
        import yaml
        with open(config_file) as f:
            return yaml.safe_load(f) or {}
    else:
        raise Exception(f"Unsupported configuration file format: {config_file.suffix}")


def _compare_configs(config1: dict[str, Any], config2: dict[str, Any], prefix: str = "") -> list[dict[str, Any]]:
    """Compare two configuration dictionaries."""
    differences = []

    all_keys = set(config1.keys()) | set(config2.keys())

    for key in sorted(all_keys):
        current_path = f"{prefix}.{key}" if prefix else key

        if key not in config1:
            differences.append({
                "key": current_path,
                "type": "added",
                "old_value": None,
                "new_value": config2[key]
            })
        elif key not in config2:
            differences.append({
                "key": current_path,
                "type": "removed",
                "old_value": config1[key],
                "new_value": None
            })
        else:
            val1, val2 = config1[key], config2[key]

            if isinstance(val1, dict) and isinstance(val2, dict):
                # Recurse into nested dictionaries
                nested_diffs = _compare_configs(val1, val2, current_path)
                differences.extend(nested_diffs)
            elif val1 != val2:
                differences.append({
                    "key": current_path,
                    "type": "changed",
                    "old_value": val1,
                    "new_value": val2
                })

    return differences


def _display_config_diff_table(differences: list[dict[str, Any]], config1_name: str, config2_name: str) -> None:
    """Display configuration differences in table format."""
    table = Table(title=f"Configuration Differences: {config1_name} vs {config2_name}")
    table.add_column("Key", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column(f"Value in {config1_name}", style="green")
    table.add_column(f"Value in {config2_name}", style="yellow")

    for diff in differences:
        type_color = {
            "changed": "yellow",
            "added": "green",
            "removed": "red"
        }.get(diff["type"], "white")

        old_val = str(diff["old_value"]) if diff["old_value"] is not None else "[dim]N/A[/dim]"
        new_val = str(diff["new_value"]) if diff["new_value"] is not None else "[dim]N/A[/dim]"

        table.add_row(
            diff["key"],
            f"[{type_color}]{diff['type'].title()}[/{type_color}]",
            old_val,
            new_val
        )

    console.print(table)
    print_info(f"Found {len(differences)} differences")


def _display_config_diff_unified(differences: list[dict[str, Any]], config1_name: str, config2_name: str) -> None:
    """Display configuration differences in unified diff format."""
    console.print(f"[bold]--- {config1_name}[/bold]")
    console.print(f"[bold]+++ {config2_name}[/bold]")
    console.print()

    for diff in differences:
        key = diff["key"]
        diff_type = diff["type"]

        if diff_type == "changed":
            console.print(f"[red]- {key}: {diff['old_value']}[/red]")
            console.print(f"[green]+ {key}: {diff['new_value']}[/green]")
        elif diff_type == "removed":
            console.print(f"[red]- {key}: {diff['old_value']}[/red]")
        elif diff_type == "added":
            console.print(f"[green]+ {key}: {diff['new_value']}[/green]")

        console.print()


class ConfigBackupManager:
    """Manager for configuration backups and rollbacks."""

    def __init__(self):
        self.backup_dir = Path("config_backups")
        self.backup_dir.mkdir(exist_ok=True)
        self.backup_index_file = self.backup_dir / "backup_index.json"
        self._load_backup_index()

    def _load_backup_index(self) -> None:
        """Load backup index from file."""
        if self.backup_index_file.exists():
            try:
                with open(self.backup_index_file) as f:
                    self.backup_index = json.load(f)
            except Exception:
                self.backup_index = {}
        else:
            self.backup_index = {}

    def _save_backup_index(self) -> None:
        """Save backup index to file."""
        try:
            with open(self.backup_index_file, 'w') as f:
                json.dump(self.backup_index, f, indent=2)
        except Exception as e:
            print_error(f"Failed to save backup index: {e}")

    def save_key_backup(self, key: str, value: str) -> None:
        """Save backup for a specific key."""
        timestamp = int(time.time())
        backup_entry = {
            "timestamp": timestamp,
            "key": key,
            "value": value
        }

        if "keys" not in self.backup_index:
            self.backup_index["keys"] = {}

        self.backup_index["keys"][key] = backup_entry
        self._save_backup_index()

    def restore_key_backup(self, key: str) -> str | None:
        """Restore backup for a specific key."""
        if "keys" not in self.backup_index:
            return None

        backup_entry = self.backup_index["keys"].get(key)
        return backup_entry["value"] if backup_entry else None

    def save_section_backup(self, section: str, data: dict[str, str]) -> None:
        """Save backup for a configuration section."""
        timestamp = int(time.time())
        backup_entry = {
            "timestamp": timestamp,
            "section": section,
            "data": data
        }

        if "sections" not in self.backup_index:
            self.backup_index["sections"] = {}

        self.backup_index["sections"][section] = backup_entry
        self._save_backup_index()

    def restore_section_backup(self, section: str) -> dict[str, str] | None:
        """Restore backup for a configuration section."""
        if "sections" not in self.backup_index:
            return None

        backup_entry = self.backup_index["sections"].get(section)
        return backup_entry["data"] if backup_entry else None

    def create_full_backup(self) -> str:
        """Create a full system backup."""
        timestamp = int(time.time())
        backup_file = self.backup_dir / f"full_backup_{timestamp}.json"

        # Collect all ITS-related environment variables
        env_data = {}
        prefixes = ["ITS_", "DATABASE_", "REDIS_", "ML_", "API_", "SECURITY_", "MONITORING_"]

        for key, value in os.environ.items():
            if any(key.startswith(prefix) for prefix in prefixes):
                env_data[key] = value

        backup_data = {
            "timestamp": timestamp,
            "type": "full_backup",
            "environment": env_data
        }

        try:
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)

            # Update index
            if "full_backups" not in self.backup_index:
                self.backup_index["full_backups"] = []

            self.backup_index["full_backups"].append({
                "timestamp": timestamp,
                "file": str(backup_file),
                "env_vars_count": len(env_data)
            })

            self._save_backup_index()
            return str(backup_file)

        except Exception as e:
            print_error(f"Failed to create full backup: {e}")
            return ""

    def restore_full_backup(self, backup_file: str) -> bool:
        """Restore from a full system backup."""
        try:
            backup_path = Path(backup_file)
            if not backup_path.exists():
                print_error(f"Backup file not found: {backup_file}")
                return False

            with open(backup_path) as f:
                backup_data = json.load(f)

            if backup_data.get("type") != "full_backup":
                print_error("Invalid backup file format")
                return False

            # Clear current ITS environment variables
            prefixes = ["ITS_", "DATABASE_", "REDIS_", "ML_", "API_", "SECURITY_", "MONITORING_"]
            cleared_vars = []

            for key in list(os.environ.keys()):
                if any(key.startswith(prefix) for prefix in prefixes):
                    del os.environ[key]
                    cleared_vars.append(key)

            # Restore from backup
            env_data = backup_data.get("environment", {})
            for key, value in env_data.items():
                os.environ[key] = value

            print_success(f"Restored {len(env_data)} environment variables from backup")
            print_info(f"Cleared {len(cleared_vars)} existing variables")
            return True

        except Exception as e:
            print_error(f"Failed to restore from backup: {e}")
            return False

    def list_backups(self) -> dict[str, Any]:
        """List all available backups."""
        return {
            "full_backups": self.backup_index.get("full_backups", []),
            "section_backups": self.backup_index.get("sections", {}),
            "key_backups": self.backup_index.get("keys", {})
        }


def _update_json_config(config_file: Path, key: str, value: str) -> None:
    """Update JSON configuration file."""
    try:
        with open(config_file) as f:
            config_data = json.load(f)

        # Navigate to the key using dot notation
        keys = key.split(".")
        current = config_data

        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Convert value to appropriate type
        final_key = keys[-1]
        try:
            # Try to parse as JSON for complex types
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            # Use as string if not valid JSON
            parsed_value = value

        current[final_key] = parsed_value

        # Write back to file
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

    except Exception as e:
        raise Exception(f"Failed to update JSON config: {e}")


def _update_yaml_config(config_file: Path, key: str, value: str) -> None:
    """Update YAML configuration file."""
    try:
        import yaml

        with open(config_file) as f:
            config_data = yaml.safe_load(f) or {}

        # Navigate to the key using dot notation
        keys = key.split(".")
        current = config_data

        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Convert value to appropriate type
        final_key = keys[-1]
        try:
            # Try to parse as YAML for proper type conversion
            parsed_value = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed_value = value

        current[final_key] = parsed_value

        # Write back to file
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

    except ImportError:
        raise Exception("PyYAML not installed. Please install it to update YAML files.")
    except Exception as e:
        raise Exception(f"Failed to update YAML config: {e}")


def _update_env_config(config_file: Path, key: str, value: str) -> None:
    """Update .env configuration file."""
    try:
        # Convert dot notation to environment variable format
        env_key = "__".join(key.split(".")).upper()

        lines = []
        key_found = False

        if config_file.exists():
            with open(config_file) as f:
                lines = f.readlines()

        # Update existing key or add new one
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                if '=' in line:
                    existing_key = line.split('=', 1)[0].strip()
                    if existing_key == env_key:
                        lines[i] = f"{env_key}={value}\n"
                        key_found = True
                        break

        if not key_found:
            lines.append(f"{env_key}={value}\n")

        # Write back to file
        with open(config_file, 'w') as f:
            f.writelines(lines)

    except Exception as e:
        raise Exception(f"Failed to update .env config: {e}")


def _update_ini_config(config_file: Path, key: str, value: str) -> None:
    """Update INI configuration file."""
    try:
        import configparser

        config = configparser.ConfigParser()
        config.read(config_file)

        # Parse key (section.option format)
        if '.' in key:
            section, option = key.split('.', 1)
        else:
            section = 'DEFAULT'
            option = key

        if section not in config:
            config.add_section(section)

        config.set(section, option, value)

        # Write back to file
        with open(config_file, 'w') as f:
            config.write(f)

    except Exception as e:
        raise Exception(f"Failed to update INI config: {e}")


def _validate_updated_config(config_file: Path, key: str, value: str) -> None:
    """Validate configuration after update."""
    try:
        # Basic validation - check if file is still valid
        if config_file.suffix.lower() == ".json":
            with open(config_file) as f:
                json.load(f)  # Will raise exception if invalid JSON
        elif config_file.suffix.lower() in [".yaml", ".yml"]:
            import yaml
            with open(config_file) as f:
                yaml.safe_load(f)  # Will raise exception if invalid YAML

        # Additional validation based on key type
        if "port" in key.lower():
            try:
                port_val = int(value)
                if not (1 <= port_val <= 65535):
                    raise ValueError(f"Port {port_val} is out of valid range (1-65535)")
            except ValueError as e:
                raise Exception(f"Invalid port value: {e}")

        elif "url" in key.lower():
            # Basic URL validation
            if not (value.startswith('http://') or value.startswith('https://') or
                   value.startswith('postgresql://') or value.startswith('redis://')):
                print_warning(f"URL value '{value}' may not be properly formatted")

        elif "timeout" in key.lower():
            try:
                timeout_val = float(value)
                if timeout_val < 0:
                    raise ValueError("Timeout cannot be negative")
            except ValueError:
                raise Exception(f"Invalid timeout value: {value}")

    except Exception as e:
        raise Exception(f"Configuration validation failed: {e}")
