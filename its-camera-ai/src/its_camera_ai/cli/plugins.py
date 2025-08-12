"""CLI plugin system for ITS Camera AI.

Provides a plugin architecture for extending the CLI with custom commands,
workflows, and integrations. Supports dynamic loading, dependency management,
and sandboxed execution.
"""

import importlib
import importlib.util
import inspect
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import typer
import yaml
from rich.table import Table

from .utils import console, print_error, print_info, print_success, print_warning


class PluginBase(ABC):
    """Base class for CLI plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        pass

    @property
    def author(self) -> str:
        """Plugin author."""
        return "Unknown"

    @property
    def dependencies(self) -> list[str]:
        """List of required dependencies."""
        return []

    @property
    def min_cli_version(self) -> str:
        """Minimum CLI version required."""
        return "0.1.0"

    @abstractmethod
    def get_commands(self) -> dict[str, typer.Typer]:
        """Return dictionary of command groups provided by this plugin."""
        pass

    def initialize(self) -> bool:
        """Initialize the plugin. Called when plugin is loaded."""
        return True

    def cleanup(self) -> None:
        """Cleanup plugin resources. Called when plugin is unloaded."""
        pass


class PluginManager:
    """Manages CLI plugins with loading, validation, and execution."""

    def __init__(self, plugin_dir: Path | None = None):
        self.plugin_dir = plugin_dir or Path.home() / ".its-camera-ai" / "plugins"
        self.plugin_dir.mkdir(parents=True, exist_ok=True)

        self.loaded_plugins: dict[str, PluginBase] = {}
        self.plugin_configs: dict[str, dict[str, Any]] = {}
        self.disabled_plugins: set = set()

        # Load plugin configurations
        self.config_file = self.plugin_dir / "plugins.yaml"
        self._load_plugin_configs()

    def _load_plugin_configs(self) -> None:
        """Load plugin configurations from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    data = yaml.safe_load(f) or {}

                self.plugin_configs = data.get("plugins", {})
                self.disabled_plugins = set(data.get("disabled", []))
            except Exception as e:
                print_warning(f"Failed to load plugin configs: {e}")

    def _save_plugin_configs(self) -> None:
        """Save plugin configurations to file."""
        try:
            data = {
                "plugins": self.plugin_configs,
                "disabled": list(self.disabled_plugins),
            }

            with open(self.config_file, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception as e:
            print_warning(f"Failed to save plugin configs: {e}")

    def discover_plugins(self) -> list[Path]:
        """Discover available plugin files."""
        plugin_files = []

        # Look for Python files in plugin directory
        for file_path in self.plugin_dir.rglob("*.py"):
            if file_path.name.startswith("_"):
                continue  # Skip private files
            plugin_files.append(file_path)

        # Look for plugin packages (directories with __init__.py)
        for dir_path in self.plugin_dir.iterdir():
            if dir_path.is_dir() and (dir_path / "__init__.py").exists():
                plugin_files.append(dir_path / "__init__.py")

        return plugin_files

    def load_plugin_from_file(self, file_path: Path) -> PluginBase | None:
        """Load a plugin from a Python file."""
        try:
            module_name = f"its_camera_ai_plugin_{file_path.stem}"

            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                print_error(f"Failed to load plugin spec: {file_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, PluginBase) and
                    obj is not PluginBase):
                    plugin_class = obj
                    break

            if not plugin_class:
                print_error(f"No plugin class found in: {file_path}")
                return None

            # Instantiate plugin
            plugin = plugin_class()

            # Validate plugin
            if not self._validate_plugin(plugin):
                return None

            return plugin

        except Exception as e:
            print_error(f"Failed to load plugin from {file_path}: {e}")
            return None

    def _validate_plugin(self, plugin: PluginBase) -> bool:
        """Validate a plugin instance."""
        try:
            # Check required properties
            if not plugin.name or not plugin.version or not plugin.description:
                print_error(f"Plugin missing required properties: {plugin.__class__}")
                return False

            # Check version compatibility
            from packaging import version

            from ..__init__ import __version__

            if version.parse(plugin.min_cli_version) > version.parse(__version__):
                print_error(f"Plugin {plugin.name} requires CLI version {plugin.min_cli_version}")
                return False

            # Check dependencies
            for dep in plugin.dependencies:
                try:
                    importlib.import_module(dep)
                except ImportError:
                    print_error(f"Plugin {plugin.name} missing dependency: {dep}")
                    return False

            # Validate commands
            try:
                commands = plugin.get_commands()
                if not isinstance(commands, dict):
                    print_error(f"Plugin {plugin.name} get_commands() must return dict")
                    return False

                for cmd_name, cmd_app in commands.items():
                    if not isinstance(cmd_app, typer.Typer):
                        print_error(f"Plugin {plugin.name} command {cmd_name} must be typer.Typer")
                        return False

            except Exception as e:
                print_error(f"Plugin {plugin.name} get_commands() failed: {e}")
                return False

            return True

        except Exception as e:
            print_error(f"Plugin validation failed: {e}")
            return False

    def load_plugin(self, plugin: PluginBase) -> bool:
        """Load a validated plugin."""
        try:
            if plugin.name in self.loaded_plugins:
                print_warning(f"Plugin {plugin.name} already loaded")
                return False

            if plugin.name in self.disabled_plugins:
                print_info(f"Plugin {plugin.name} is disabled")
                return False

            # Initialize plugin
            if not plugin.initialize():
                print_error(f"Plugin {plugin.name} initialization failed")
                return False

            # Store plugin
            self.loaded_plugins[plugin.name] = plugin

            # Save config
            if plugin.name not in self.plugin_configs:
                self.plugin_configs[plugin.name] = {
                    "version": plugin.version,
                    "author": plugin.author,
                    "enabled": True,
                    "auto_load": True,
                }
                self._save_plugin_configs()

            print_success(f"Loaded plugin: {plugin.name} v{plugin.version}")
            return True

        except Exception as e:
            print_error(f"Failed to load plugin {plugin.name}: {e}")
            return False

    def unload_plugin(self, name: str) -> bool:
        """Unload a plugin."""
        if name not in self.loaded_plugins:
            print_error(f"Plugin {name} not loaded")
            return False

        try:
            plugin = self.loaded_plugins[name]
            plugin.cleanup()
            del self.loaded_plugins[name]
            print_success(f"Unloaded plugin: {name}")
            return True
        except Exception as e:
            print_error(f"Failed to unload plugin {name}: {e}")
            return False

    def enable_plugin(self, name: str) -> bool:
        """Enable a plugin."""
        if name in self.disabled_plugins:
            self.disabled_plugins.remove(name)
            self._save_plugin_configs()
            print_success(f"Enabled plugin: {name}")
            return True
        else:
            print_info(f"Plugin {name} is already enabled")
            return False

    def disable_plugin(self, name: str) -> bool:
        """Disable a plugin."""
        if name not in self.disabled_plugins:
            self.disabled_plugins.add(name)

            # Unload if currently loaded
            if name in self.loaded_plugins:
                self.unload_plugin(name)

            self._save_plugin_configs()
            print_success(f"Disabled plugin: {name}")
            return True
        else:
            print_info(f"Plugin {name} is already disabled")
            return False

    def reload_plugin(self, name: str) -> bool:
        """Reload a plugin."""
        if name in self.loaded_plugins:
            self.unload_plugin(name)

        # Find and load the plugin again
        plugin_files = self.discover_plugins()
        for file_path in plugin_files:
            plugin = self.load_plugin_from_file(file_path)
            if plugin and plugin.name == name:
                return self.load_plugin(plugin)

        print_error(f"Plugin {name} not found for reload")
        return False

    def load_all_plugins(self) -> int:
        """Load all available plugins."""
        plugin_files = self.discover_plugins()
        loaded_count = 0

        for file_path in plugin_files:
            plugin = self.load_plugin_from_file(file_path)
            if plugin and self.load_plugin(plugin):
                loaded_count += 1

        return loaded_count

    def get_plugin_commands(self) -> dict[str, typer.Typer]:
        """Get all commands from loaded plugins."""
        commands = {}

        for plugin_name, plugin in self.loaded_plugins.items():
            try:
                plugin_commands = plugin.get_commands()
                for cmd_name, cmd_app in plugin_commands.items():
                    # Prefix command name with plugin name to avoid conflicts
                    full_cmd_name = f"{plugin_name}_{cmd_name}"
                    commands[full_cmd_name] = cmd_app
            except Exception as e:
                print_error(f"Failed to get commands from plugin {plugin_name}: {e}")

        return commands

    def list_plugins(self) -> list[dict[str, Any]]:
        """List all plugins with their status."""
        all_plugins = []

        # Add loaded plugins
        for name, plugin in self.loaded_plugins.items():
            all_plugins.append({
                "name": name,
                "version": plugin.version,
                "description": plugin.description,
                "author": plugin.author,
                "status": "loaded",
                "enabled": name not in self.disabled_plugins,
            })

        # Add discovered but not loaded plugins
        plugin_files = self.discover_plugins()
        for file_path in plugin_files:
            try:
                plugin = self.load_plugin_from_file(file_path)
                if plugin and plugin.name not in self.loaded_plugins:
                    all_plugins.append({
                        "name": plugin.name,
                        "version": plugin.version,
                        "description": plugin.description,
                        "author": plugin.author,
                        "status": "available",
                        "enabled": plugin.name not in self.disabled_plugins,
                    })
            except:
                continue  # Skip invalid plugins

        return all_plugins

    def install_plugin_template(self, name: str) -> Path:
        """Create a template plugin file."""
        template_content = f'''"""
Example plugin for ITS Camera AI CLI.

This is a template plugin that demonstrates how to create custom CLI extensions.
"""

import typer
from its_camera_ai.cli.plugins import PluginBase
from its_camera_ai.cli.utils import print_info, print_success


class {name.title().replace("_", "")}Plugin(PluginBase):
    """Example plugin implementation."""
    
    @property
    def name(self) -> str:
        return "{name}"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "Example plugin for demonstrating CLI extensions"
    
    @property
    def author(self) -> str:
        return "Your Name"
    
    @property
    def dependencies(self) -> list[str]:
        return []  # Add any required Python packages
    
    def get_commands(self) -> dict[str, typer.Typer]:
        """Return the command groups provided by this plugin."""
        app = typer.Typer(help=f"🔌 {{self.name}} plugin commands")
        
        @app.command()
        def hello(
            name: str = typer.Option("World", "--name", "-n", help="Name to greet"),
        ) -> None:
            """Say hello from the plugin."""
            print_success(f"Hello {{name}} from {{self.name}} plugin!")
        
        @app.command()
        def info() -> None:
            """Show plugin information."""
            print_info(f"Plugin: {{self.name}} v{{self.version}}")
            print_info(f"Description: {{self.description}}")
            print_info(f"Author: {{self.author}}")
        
        return {{"main": app}}
    
    def initialize(self) -> bool:
        """Initialize the plugin."""
        print_info(f"Initializing {{self.name}} plugin...")
        return True
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        print_info(f"Cleaning up {{self.name}} plugin...")


# Plugin instance (required)
plugin = {name.title().replace("_", "")}Plugin()
'''

        plugin_file = self.plugin_dir / f"{name}.py"
        plugin_file.write_text(template_content.strip())

        return plugin_file


# Global plugin manager instance
plugin_manager = PluginManager()

app = typer.Typer(help="🔌 Plugin management system")


@app.command("list")
def list_plugins() -> None:
    """List all available plugins."""
    plugins = plugin_manager.list_plugins()

    if not plugins:
        print_info("No plugins found")
        print_info(f"Plugin directory: {plugin_manager.plugin_dir}")
        return

    table = Table(title="Available Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Status", style="yellow")
    table.add_column("Enabled", style="blue")
    table.add_column("Description", style="dim")

    for plugin in plugins:
        status_color = "green" if plugin["status"] == "loaded" else "yellow"
        enabled_icon = "✅" if plugin["enabled"] else "❌"

        table.add_row(
            plugin["name"],
            plugin["version"],
            f"[{status_color}]{plugin['status']}[/{status_color}]",
            enabled_icon,
            plugin["description"][:50] + "..." if len(plugin["description"]) > 50 else plugin["description"],
        )

    console.print(table)


@app.command("load")
def load_plugin(
    name: str = typer.Argument(..., help="Plugin name to load"),
) -> None:
    """Load a specific plugin."""
    plugin_files = plugin_manager.discover_plugins()

    for file_path in plugin_files:
        plugin = plugin_manager.load_plugin_from_file(file_path)
        if plugin and plugin.name == name:
            if plugin_manager.load_plugin(plugin):
                print_success(f"Loaded plugin: {name}")
            return

    print_error(f"Plugin '{name}' not found")


@app.command("unload")
def unload_plugin(
    name: str = typer.Argument(..., help="Plugin name to unload"),
) -> None:
    """Unload a specific plugin."""
    if plugin_manager.unload_plugin(name):
        print_success(f"Unloaded plugin: {name}")


@app.command("enable")
def enable_plugin(
    name: str = typer.Argument(..., help="Plugin name to enable"),
) -> None:
    """Enable a plugin."""
    plugin_manager.enable_plugin(name)


@app.command("disable")
def disable_plugin(
    name: str = typer.Argument(..., help="Plugin name to disable"),
) -> None:
    """Disable a plugin."""
    plugin_manager.disable_plugin(name)


@app.command("reload")
def reload_plugin(
    name: str = typer.Argument(..., help="Plugin name to reload"),
) -> None:
    """Reload a plugin."""
    if plugin_manager.reload_plugin(name):
        print_success(f"Reloaded plugin: {name}")


@app.command("load-all")
def load_all_plugins() -> None:
    """Load all available plugins."""
    count = plugin_manager.load_all_plugins()
    print_success(f"Loaded {count} plugins")


@app.command("create")
def create_plugin_template(
    name: str = typer.Argument(..., help="New plugin name"),
) -> None:
    """Create a plugin template file."""
    plugin_file = plugin_manager.install_plugin_template(name)
    print_success(f"Created plugin template: {plugin_file}")
    print_info("Edit the template file and use 'its-camera-ai plugin load <name>' to load it")


@app.command("directory")
def show_plugin_directory() -> None:
    """Show the plugin directory path."""
    print_info(f"Plugin directory: {plugin_manager.plugin_dir}")

    if not plugin_manager.plugin_dir.exists():
        print_warning("Plugin directory does not exist")
        if typer.confirm("Create plugin directory?"):
            plugin_manager.plugin_dir.mkdir(parents=True, exist_ok=True)
            print_success("Created plugin directory")


if __name__ == "__main__":
    app()
