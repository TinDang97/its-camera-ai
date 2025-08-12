"""CLI configuration profiles for different environments.

Provides configuration profile management for development, staging, and production
environments with easy switching and validation.
"""

import json
import os
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.table import Table

from .utils import console, display_config, print_error, print_info, print_success


class ProfileManager:
    """Manages CLI configuration profiles."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path.home() / ".its-camera-ai"
        self.config_dir.mkdir(exist_ok=True)
        self.profiles_file = self.config_dir / "profiles.yaml"
        self.current_profile_file = self.config_dir / ".current_profile"

        # Ensure profiles file exists
        if not self.profiles_file.exists():
            self._initialize_default_profiles()

    def _initialize_default_profiles(self) -> None:
        """Initialize default configuration profiles."""
        default_profiles = {
            "development": {
                "description": "Local development environment",
                "api_host": "localhost",
                "api_port": 8000,
                "debug": True,
                "log_level": "DEBUG",
                "database_url": "postgresql://localhost:5432/its_camera_ai_dev",
                "redis_url": "redis://localhost:6379/0",
                "enable_auth": False,
                "ml_backend": "local",
                "gpu_enabled": False,
                "monitoring": {
                    "enabled": False,
                    "prometheus_port": 9090,
                    "grafana_port": 3000,
                },
                "security": {
                    "encrypt_data": False,
                    "require_https": False,
                },
            },
            "staging": {
                "description": "Staging environment for testing",
                "api_host": "0.0.0.0",
                "api_port": 8000,
                "debug": False,
                "log_level": "INFO",
                "database_url": "postgresql://staging-db:5432/its_camera_ai_staging",
                "redis_url": "redis://staging-redis:6379/0",
                "enable_auth": True,
                "ml_backend": "cloud",
                "gpu_enabled": True,
                "monitoring": {
                    "enabled": True,
                    "prometheus_port": 9090,
                    "grafana_port": 3000,
                },
                "security": {
                    "encrypt_data": True,
                    "require_https": True,
                },
            },
            "production": {
                "description": "Production environment",
                "api_host": "0.0.0.0",
                "api_port": 8000,
                "debug": False,
                "log_level": "WARNING",
                "database_url": "${DATABASE_URL}",
                "redis_url": "${REDIS_URL}",
                "enable_auth": True,
                "ml_backend": "distributed",
                "gpu_enabled": True,
                "monitoring": {
                    "enabled": True,
                    "prometheus_port": 9090,
                    "grafana_port": 3000,
                    "alerting": True,
                },
                "security": {
                    "encrypt_data": True,
                    "require_https": True,
                    "enable_audit_logs": True,
                },
            },
            "edge": {
                "description": "Edge device deployment",
                "api_host": "0.0.0.0",
                "api_port": 8000,
                "debug": False,
                "log_level": "INFO",
                "database_url": "sqlite:///./edge.db",
                "redis_url": "redis://localhost:6379/0",
                "enable_auth": True,
                "ml_backend": "edge",
                "gpu_enabled": False,
                "monitoring": {
                    "enabled": True,
                    "lightweight": True,
                },
                "security": {
                    "encrypt_data": True,
                    "require_https": False,  # Self-signed certs on edge
                },
            },
        }

        with open(self.profiles_file, "w") as f:
            yaml.dump(default_profiles, f, default_flow_style=False)

    def list_profiles(self) -> dict[str, dict[str, Any]]:
        """List all available profiles."""
        if not self.profiles_file.exists():
            return {}

        with open(self.profiles_file) as f:
            return yaml.safe_load(f) or {}

    def get_profile(self, name: str) -> dict[str, Any] | None:
        """Get a specific profile configuration."""
        profiles = self.list_profiles()
        return profiles.get(name)

    def create_profile(self, name: str, config: dict[str, Any]) -> bool:
        """Create a new profile."""
        try:
            profiles = self.list_profiles()
            profiles[name] = config

            with open(self.profiles_file, "w") as f:
                yaml.dump(profiles, f, default_flow_style=False)

            return True
        except Exception as e:
            print_error(f"Failed to create profile: {e}")
            return False

    def update_profile(self, name: str, config: dict[str, Any]) -> bool:
        """Update an existing profile."""
        profiles = self.list_profiles()
        if name not in profiles:
            print_error(f"Profile '{name}' does not exist")
            return False

        try:
            profiles[name].update(config)

            with open(self.profiles_file, "w") as f:
                yaml.dump(profiles, f, default_flow_style=False)

            return True
        except Exception as e:
            print_error(f"Failed to update profile: {e}")
            return False

    def delete_profile(self, name: str) -> bool:
        """Delete a profile."""
        profiles = self.list_profiles()
        if name not in profiles:
            print_error(f"Profile '{name}' does not exist")
            return False

        if name == self.get_current_profile():
            print_error("Cannot delete the currently active profile")
            return False

        try:
            del profiles[name]

            with open(self.profiles_file, "w") as f:
                yaml.dump(profiles, f, default_flow_style=False)

            return True
        except Exception as e:
            print_error(f"Failed to delete profile: {e}")
            return False

    def set_current_profile(self, name: str) -> bool:
        """Set the current active profile."""
        profiles = self.list_profiles()
        if name not in profiles:
            print_error(f"Profile '{name}' does not exist")
            return False

        try:
            self.current_profile_file.write_text(name)
            return True
        except Exception as e:
            print_error(f"Failed to set current profile: {e}")
            return False

    def get_current_profile(self) -> str:
        """Get the name of the current active profile."""
        if self.current_profile_file.exists():
            return self.current_profile_file.read_text().strip()
        return "development"  # Default profile

    def get_current_config(self) -> dict[str, Any]:
        """Get the configuration of the current active profile."""
        current = self.get_current_profile()
        config = self.get_profile(current)
        if not config:
            print_warning(f"Current profile '{current}' not found, using default")
            return self.get_profile("development") or {}

        # Resolve environment variables
        return self._resolve_env_vars(config)

    def _resolve_env_vars(self, config: dict[str, Any]) -> dict[str, Any]:
        """Resolve environment variables in configuration values."""
        def resolve_value(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value

        return {k: resolve_value(v) for k, v in config.items()}

    def validate_profile(self, name: str) -> bool:
        """Validate a profile configuration."""
        profile = self.get_profile(name)
        if not profile:
            print_error(f"Profile '{name}' does not exist")
            return False

        required_fields = [
            "api_host",
            "api_port",
            "database_url",
            "redis_url",
        ]

        missing_fields = []
        for field in required_fields:
            if field not in profile:
                missing_fields.append(field)

        if missing_fields:
            print_error(f"Profile '{name}' missing required fields: {', '.join(missing_fields)}")
            return False

        return True

    def export_profile(self, name: str, output_file: Path) -> bool:
        """Export a profile to a file."""
        profile = self.get_profile(name)
        if not profile:
            print_error(f"Profile '{name}' does not exist")
            return False

        try:
            if output_file.suffix.lower() == ".json":
                with open(output_file, "w") as f:
                    json.dump({name: profile}, f, indent=2)
            else:
                with open(output_file, "w") as f:
                    yaml.dump({name: profile}, f, default_flow_style=False)

            return True
        except Exception as e:
            print_error(f"Failed to export profile: {e}")
            return False

    def import_profile(self, input_file: Path) -> bool:
        """Import profiles from a file."""
        if not input_file.exists():
            print_error(f"File does not exist: {input_file}")
            return False

        try:
            if input_file.suffix.lower() == ".json":
                with open(input_file) as f:
                    data = json.load(f)
            else:
                with open(input_file) as f:
                    data = yaml.safe_load(f)

            profiles = self.list_profiles()
            imported_count = 0

            for name, config in data.items():
                profiles[name] = config
                imported_count += 1

            with open(self.profiles_file, "w") as f:
                yaml.dump(profiles, f, default_flow_style=False)

            print_success(f"Imported {imported_count} profiles")
            return True

        except Exception as e:
            print_error(f"Failed to import profiles: {e}")
            return False


# Global profile manager instance
profile_manager = ProfileManager()

app = typer.Typer(help="🎭 Configuration profile management")


@app.command("list")
def list_profiles() -> None:
    """List all available configuration profiles."""
    profiles = profile_manager.list_profiles()
    current = profile_manager.get_current_profile()

    if not profiles:
        print_info("No profiles found")
        return

    table = Table(title="Configuration Profiles")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Environment", style="yellow")
    table.add_column("Current", style="bold blue")

    for name, config in profiles.items():
        description = config.get("description", "No description")
        env_type = "🐛 Dev" if "dev" in name else "🚀 Prod" if "prod" in name else "🧪 Test"
        is_current = "✅" if name == current else ""

        table.add_row(name, description, env_type, is_current)

    console.print(table)
    print_info(f"Current profile: {current}")


@app.command("show")
def show_profile(
    name: str | None = typer.Argument(
        None, help="Profile name (defaults to current profile)"
    ),
) -> None:
    """Show configuration for a specific profile."""
    if not name:
        name = profile_manager.get_current_profile()

    profile = profile_manager.get_profile(name)
    if not profile:
        print_error(f"Profile '{name}' not found")
        return

    print_info(f"Configuration for profile: {name}")
    display_config(profile, f"Profile: {name}")


@app.command("switch")
def switch_profile(
    name: str = typer.Argument(..., help="Profile name to switch to"),
) -> None:
    """Switch to a different configuration profile."""
    if profile_manager.set_current_profile(name):
        print_success(f"Switched to profile: {name}")

        # Show the new profile configuration
        profile = profile_manager.get_profile(name)
        if profile:
            print_info(f"Description: {profile.get('description', 'N/A')}")
    else:
        print_error(f"Failed to switch to profile: {name}")


@app.command("create")
def create_profile(
    name: str = typer.Argument(..., help="New profile name"),
    based_on: str | None = typer.Option(
        None, "--based-on", "-b", help="Base profile to copy from"
    ),
    description: str = typer.Option(
        "", "--description", "-d", help="Profile description"
    ),
) -> None:
    """Create a new configuration profile."""
    # Check if profile already exists
    if profile_manager.get_profile(name):
        print_error(f"Profile '{name}' already exists")
        return

    # Start with base configuration
    if based_on:
        base_config = profile_manager.get_profile(based_on)
        if not base_config:
            print_error(f"Base profile '{based_on}' not found")
            return
        config = base_config.copy()
    else:
        config = profile_manager.get_profile("development").copy()

    # Update description
    if description:
        config["description"] = description

    if profile_manager.create_profile(name, config):
        print_success(f"Created profile: {name}")
        if based_on:
            print_info(f"Based on profile: {based_on}")
    else:
        print_error(f"Failed to create profile: {name}")


@app.command("update")
def update_profile(
    name: str = typer.Argument(..., help="Profile name to update"),
    key: str = typer.Argument(..., help="Configuration key to update"),
    value: str = typer.Argument(..., help="New value"),
) -> None:
    """Update a configuration value in a profile."""
    profile = profile_manager.get_profile(name)
    if not profile:
        print_error(f"Profile '{name}' not found")
        return

    # Support nested keys using dot notation
    keys = key.split(".")
    current = profile

    # Navigate to the nested key
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Convert value to appropriate type
    if value.lower() in ["true", "false"]:
        value = value.lower() == "true"
    elif value.isdigit():
        value = int(value)
    elif value.replace(".", "", 1).isdigit():
        value = float(value)

    # Set the value
    current[keys[-1]] = value

    if profile_manager.update_profile(name, profile):
        print_success(f"Updated {key} in profile '{name}'")
    else:
        print_error(f"Failed to update profile '{name}'")


@app.command("delete")
def delete_profile(
    name: str = typer.Argument(..., help="Profile name to delete"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force deletion without confirmation"
    ),
) -> None:
    """Delete a configuration profile."""
    if not force:
        from .utils import confirm_action
        if not confirm_action(f"Delete profile '{name}'?"):
            print_info("Deletion cancelled")
            return

    if profile_manager.delete_profile(name):
        print_success(f"Deleted profile: {name}")
    else:
        print_error(f"Failed to delete profile: {name}")


@app.command("validate")
def validate_profile(
    name: str | None = typer.Argument(
        None, help="Profile name (defaults to current profile)"
    ),
) -> None:
    """Validate a configuration profile."""
    if not name:
        name = profile_manager.get_current_profile()

    if profile_manager.validate_profile(name):
        print_success(f"Profile '{name}' is valid")
    else:
        print_error(f"Profile '{name}' validation failed")


@app.command("export")
def export_profile(
    name: str = typer.Argument(..., help="Profile name to export"),
    output: Path = typer.Argument(..., help="Output file path"),
) -> None:
    """Export a profile to a file."""
    if profile_manager.export_profile(name, output):
        print_success(f"Exported profile '{name}' to {output}")
    else:
        print_error(f"Failed to export profile '{name}'")


@app.command("import")
def import_profiles(
    input_file: Path = typer.Argument(..., help="Input file path"),
) -> None:
    """Import profiles from a file."""
    if profile_manager.import_profile(input_file):
        print_success(f"Imported profiles from {input_file}")
    else:
        print_error(f"Failed to import profiles from {input_file}")


@app.command("current")
def show_current_profile() -> None:
    """Show the current active profile."""
    current = profile_manager.get_current_profile()
    config = profile_manager.get_current_config()

    print_info(f"Current profile: {current}")
    display_config(config, f"Current Profile: {current}")


if __name__ == "__main__":
    app()
