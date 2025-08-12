"""Command history and favorites management for ITS Camera AI CLI.

Provides persistent command history, favorites management, and quick access
to frequently used commands with intelligent suggestions and analytics.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from .utils import console, print_error, print_info, print_success, print_warning


class CommandHistory:
    """Manages persistent command history with analytics."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path.home() / ".its-camera-ai"
        self.config_dir.mkdir(exist_ok=True)
        self.history_file = self.config_dir / "history.json"
        self.favorites_file = self.config_dir / "favorites.json"

        self.history: list[dict[str, Any]] = []
        self.favorites: list[dict[str, Any]] = []

        self._load_data()

    def _load_data(self) -> None:
        """Load history and favorites from files."""
        try:
            if self.history_file.exists():
                with open(self.history_file) as f:
                    self.history = json.load(f)
        except Exception as e:
            print_warning(f"Failed to load command history: {e}")
            self.history = []

        try:
            if self.favorites_file.exists():
                with open(self.favorites_file) as f:
                    self.favorites = json.load(f)
        except Exception as e:
            print_warning(f"Failed to load favorites: {e}")
            self.favorites = []

    def _save_data(self) -> None:
        """Save history and favorites to files."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            print_warning(f"Failed to save command history: {e}")

        try:
            with open(self.favorites_file, "w") as f:
                json.dump(self.favorites, f, indent=2, default=str)
        except Exception as e:
            print_warning(f"Failed to save favorites: {e}")

    def add_command(self, command: str, args: list[str] = None, success: bool = True) -> None:
        """Add a command to history."""
        args = args or []

        entry = {
            "command": command,
            "args": args,
            "full_command": f"{command} {' '.join(args)}".strip(),
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "duration": None,  # Could be added later
        }

        # Avoid duplicate consecutive entries
        if self.history and self.history[-1].get("full_command") == entry["full_command"]:
            self.history[-1]["timestamp"] = entry["timestamp"]
            self.history[-1]["success"] = success
        else:
            self.history.append(entry)

        # Keep only last 1000 entries
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        self._save_data()

    def get_recent_commands(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent commands from history."""
        return self.history[-limit:]

    def get_command_stats(self) -> dict[str, Any]:
        """Get command usage statistics."""
        if not self.history:
            return {}

        command_counts = {}
        success_count = 0
        total_commands = len(self.history)

        # Count command usage
        for entry in self.history:
            cmd = entry["command"]
            command_counts[cmd] = command_counts.get(cmd, 0) + 1
            if entry.get("success", True):
                success_count += 1

        # Get most used commands
        most_used = sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Get recent activity
        last_24h = datetime.now() - timedelta(days=1)
        recent_commands = [
            entry for entry in self.history
            if datetime.fromisoformat(entry["timestamp"]) > last_24h
        ]

        return {
            "total_commands": total_commands,
            "success_rate": (success_count / total_commands) * 100 if total_commands > 0 else 0,
            "most_used": most_used,
            "commands_24h": len(recent_commands),
            "unique_commands": len(command_counts),
        }

    def search_history(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search command history."""
        query_lower = query.lower()
        matches = []

        for entry in reversed(self.history):  # Search newest first
            if (query_lower in entry["full_command"].lower() or
                query_lower in entry["command"].lower()):
                matches.append(entry)
                if len(matches) >= limit:
                    break

        return matches

    def add_favorite(self, command: str, alias: str = "", description: str = "") -> bool:
        """Add a command to favorites."""
        # Check if already in favorites
        for fav in self.favorites:
            if fav["command"] == command:
                print_warning(f"Command already in favorites: {command}")
                return False

        favorite = {
            "command": command,
            "alias": alias or command.replace(" ", "_"),
            "description": description,
            "created": datetime.now().isoformat(),
            "usage_count": 0,
            "last_used": None,
        }

        self.favorites.append(favorite)
        self._save_data()
        return True

    def remove_favorite(self, command: str) -> bool:
        """Remove a command from favorites."""
        original_length = len(self.favorites)
        self.favorites = [fav for fav in self.favorites if fav["command"] != command]

        if len(self.favorites) < original_length:
            self._save_data()
            return True
        return False

    def use_favorite(self, command: str) -> None:
        """Mark a favorite as used."""
        for fav in self.favorites:
            if fav["command"] == command:
                fav["usage_count"] = fav.get("usage_count", 0) + 1
                fav["last_used"] = datetime.now().isoformat()
                self._save_data()
                break

    def get_favorites(self) -> list[dict[str, Any]]:
        """Get all favorites sorted by usage."""
        return sorted(self.favorites, key=lambda x: x.get("usage_count", 0), reverse=True)

    def suggest_commands(self, current_input: str = "", limit: int = 5) -> list[str]:
        """Suggest commands based on history and favorites."""
        suggestions = []

        if current_input:
            # Find commands that start with the current input
            for entry in reversed(self.history):
                cmd = entry["full_command"]
                if cmd.startswith(current_input) and cmd not in suggestions:
                    suggestions.append(cmd)
                    if len(suggestions) >= limit:
                        break
        else:
            # Suggest most frequently used commands
            stats = self.get_command_stats()
            for cmd, _ in stats.get("most_used", []):
                if cmd not in suggestions:
                    suggestions.append(cmd)
                    if len(suggestions) >= limit:
                        break

        return suggestions

    def export_data(self, output_file: Path) -> bool:
        """Export history and favorites to a file."""
        try:
            data = {
                "history": self.history,
                "favorites": self.favorites,
                "exported": datetime.now().isoformat(),
                "stats": self.get_command_stats(),
            }

            with open(output_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            return True
        except Exception as e:
            print_error(f"Failed to export data: {e}")
            return False

    def import_data(self, input_file: Path, merge: bool = True) -> bool:
        """Import history and favorites from a file."""
        try:
            with open(input_file) as f:
                data = json.load(f)

            imported_history = data.get("history", [])
            imported_favorites = data.get("favorites", [])

            if merge:
                # Merge with existing data
                existing_commands = {entry["full_command"] for entry in self.history}
                for entry in imported_history:
                    if entry["full_command"] not in existing_commands:
                        self.history.append(entry)

                existing_favorites = {fav["command"] for fav in self.favorites}
                for fav in imported_favorites:
                    if fav["command"] not in existing_favorites:
                        self.favorites.append(fav)
            else:
                # Replace existing data
                self.history = imported_history
                self.favorites = imported_favorites

            self._save_data()
            print_success(f"Imported {len(imported_history)} history entries and {len(imported_favorites)} favorites")
            return True

        except Exception as e:
            print_error(f"Failed to import data: {e}")
            return False

    def clear_history(self, keep_favorites: bool = True) -> None:
        """Clear command history."""
        self.history = []
        if not keep_favorites:
            self.favorites = []

        self._save_data()


# Global history manager
history_manager = CommandHistory()

app = typer.Typer(help="📚 Command history and favorites management")


@app.command()
def show(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of commands to show"),
    command_filter: str | None = typer.Option(None, "--filter", "-f", help="Filter commands"),
) -> None:
    """Show recent command history."""
    recent = history_manager.get_recent_commands(limit)

    if command_filter:
        recent = [
            entry for entry in recent
            if command_filter.lower() in entry["full_command"].lower()
        ]

    if not recent:
        print_info("No commands in history")
        return

    table = Table(title="Command History")
    table.add_column("Time", style="dim", width=20)
    table.add_column("Command", style="cyan")
    table.add_column("Status", style="green", width=10)

    for entry in recent:
        timestamp = datetime.fromisoformat(entry["timestamp"])
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        status = "✅" if entry.get("success", True) else "❌"

        table.add_row(time_str, entry["full_command"], status)

    console.print(table)


@app.command()
def stats() -> None:
    """Show command usage statistics."""
    stats = history_manager.get_command_stats()

    if not stats:
        print_info("No command statistics available")
        return

    # Overall stats
    overview_table = Table(title="Command Statistics Overview")
    overview_table.add_column("Metric", style="cyan")
    overview_table.add_column("Value", style="green")

    overview_table.add_row("Total Commands", str(stats["total_commands"]))
    overview_table.add_row("Unique Commands", str(stats["unique_commands"]))
    overview_table.add_row("Success Rate", f"{stats['success_rate']:.1f}%")
    overview_table.add_row("Commands (24h)", str(stats["commands_24h"]))

    console.print(overview_table)

    # Most used commands
    if stats["most_used"]:
        usage_table = Table(title="Most Used Commands")
        usage_table.add_column("Command", style="cyan")
        usage_table.add_column("Count", style="green")
        usage_table.add_column("Percentage", style="yellow")

        total = stats["total_commands"]
        for command, count in stats["most_used"]:
            percentage = (count / total) * 100 if total > 0 else 0
            usage_table.add_row(command, str(count), f"{percentage:.1f}%")

        console.print(usage_table)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of results"),
) -> None:
    """Search command history."""
    results = history_manager.search_history(query, limit)

    if not results:
        print_info(f"No commands found matching: {query}")
        return

    table = Table(title=f"Search Results: {query}")
    table.add_column("Time", style="dim", width=20)
    table.add_column("Command", style="cyan")
    table.add_column("Status", style="green", width=10)

    for entry in results:
        timestamp = datetime.fromisoformat(entry["timestamp"])
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        status = "✅" if entry.get("success", True) else "❌"

        table.add_row(time_str, entry["full_command"], status)

    console.print(table)


@app.command()
def favorites() -> None:
    """Show favorite commands."""
    favorites = history_manager.get_favorites()

    if not favorites:
        print_info("No favorite commands")
        return

    table = Table(title="Favorite Commands")
    table.add_column("Alias", style="cyan", width=15)
    table.add_column("Command", style="green")
    table.add_column("Usage", style="yellow", width=8)
    table.add_column("Description", style="dim")

    for fav in favorites:
        usage_count = fav.get("usage_count", 0)
        description = fav.get("description", "")

        table.add_row(
            fav["alias"],
            fav["command"],
            str(usage_count),
            description
        )

    console.print(table)


@app.command()
def add_favorite(
    command: str = typer.Argument(..., help="Command to add to favorites"),
    alias: str = typer.Option("", "--alias", "-a", help="Alias for the command"),
    description: str = typer.Option("", "--description", "-d", help="Command description"),
) -> None:
    """Add a command to favorites."""
    if history_manager.add_favorite(command, alias, description):
        print_success(f"Added '{command}' to favorites")
        if alias:
            print_info(f"Alias: {alias}")
    else:
        print_error("Failed to add command to favorites")


@app.command()
def remove_favorite(
    command: str = typer.Argument(..., help="Command to remove from favorites"),
) -> None:
    """Remove a command from favorites."""
    if history_manager.remove_favorite(command):
        print_success(f"Removed '{command}' from favorites")
    else:
        print_error(f"Command not found in favorites: {command}")


@app.command()
def suggest(
    input_text: str = typer.Option("", "--input", "-i", help="Current input text"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of suggestions"),
) -> None:
    """Get command suggestions based on history."""
    suggestions = history_manager.suggest_commands(input_text, limit)

    if not suggestions:
        print_info("No suggestions available")
        return

    table = Table(title="Command Suggestions")
    table.add_column("Suggestion", style="cyan")

    for suggestion in suggestions:
        table.add_row(suggestion)

    console.print(table)


@app.command()
def export(
    output_file: Path = typer.Argument(..., help="Output file path"),
) -> None:
    """Export history and favorites to a file."""
    if history_manager.export_data(output_file):
        print_success(f"Exported data to {output_file}")
    else:
        print_error("Failed to export data")


@app.command()
def import_data(
    input_file: Path = typer.Argument(..., help="Input file path"),
    merge: bool = typer.Option(True, "--merge/--replace", help="Merge with existing data"),
) -> None:
    """Import history and favorites from a file."""
    if not input_file.exists():
        print_error(f"File not found: {input_file}")
        return

    if history_manager.import_data(input_file, merge):
        action = "merged with" if merge else "replaced"
        print_success(f"Data {action} existing history and favorites")
    else:
        print_error("Failed to import data")


@app.command()
def clear(
    keep_favorites: bool = typer.Option(True, "--keep-favorites/--clear-all", help="Keep favorites"),
    force: bool = typer.Option(False, "--force", "-f", help="Force without confirmation"),
) -> None:
    """Clear command history."""
    if not force:
        action = "history" if keep_favorites else "history and favorites"
        if not typer.confirm(f"Clear {action}?"):
            print_info("Operation cancelled")
            return

    history_manager.clear_history(keep_favorites)

    if keep_favorites:
        print_success("Command history cleared (favorites preserved)")
    else:
        print_success("Command history and favorites cleared")


if __name__ == "__main__":
    app()
