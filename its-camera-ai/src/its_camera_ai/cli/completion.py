"""Shell completion support for ITS Camera AI CLI.

Provides shell completion for bash, zsh, fish, and PowerShell with dynamic
completion for subcommands, options, and contextual values.
"""

import os
import sys
from pathlib import Path

import typer
from rich.panel import Panel
from rich.syntax import Syntax

from .utils import console, print_error, print_info, print_success

# Shell completion configurations
COMPLETION_SCRIPTS = {
    "bash": """
# ITS Camera AI bash completion
_its_camera_ai_completion() {
    local cur opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    
    if [[ $COMP_CWORD -eq 1 ]]; then
        opts="service ml data security monitor config info version status dashboard interactive"
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
    
    case "${COMP_WORDS[1]}" in
        service)
            opts="start stop restart status logs scale health"
            ;;
        ml)
            opts="train inference models deploy experiments benchmark optimize"
            ;;
        data)
            opts="pipeline stream batch validate export"
            ;;
        security)
            opts="auth users roles audit encrypt decrypt"
            ;;
        monitor)
            opts="metrics alerts health dashboard logs"
            ;;
        config)
            opts="get set list validate profile"
            ;;
        *)
            ;;
    esac
    
    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
}

complete -F _its_camera_ai_completion its-camera-ai
""",
    "zsh": """
#compdef its-camera-ai

_its_camera_ai() {
    local context state line
    
    _arguments -C \\
        '1: :_its_camera_ai_commands' \\
        '*::arg:->args'
    
    case $state in
        args)
            case $words[1] in
                service)
                    _arguments \\
                        '1: :(start stop restart status logs scale health)'
                    ;;
                ml)
                    _arguments \\
                        '1: :(train inference models deploy experiments benchmark optimize)'
                    ;;
                data)
                    _arguments \\
                        '1: :(pipeline stream batch validate export)'
                    ;;
                security)
                    _arguments \\
                        '1: :(auth users roles audit encrypt decrypt)'
                    ;;
                monitor)
                    _arguments \\
                        '1: :(metrics alerts health dashboard logs)'
                    ;;
                config)
                    _arguments \\
                        '1: :(get set list validate profile)'
                    ;;
            esac
            ;;
    esac
}

_its_camera_ai_commands() {
    local commands
    commands=(
        'service:Service management commands'
        'ml:ML operations and model management'
        'data:Data pipeline control'
        'security:Security and authentication'
        'monitor:System monitoring and health'
        'config:Configuration management'
        'info:Show system information'
        'version:Show version information'
        'status:Show quick status overview'
        'dashboard:Launch interactive dashboard'
        'interactive:Start interactive mode'
    )
    _describe 'commands' commands
}

_its_camera_ai "$@"
""",
    "fish": """
# ITS Camera AI fish completion

complete -c its-camera-ai -n '__fish_use_subcommand' -a 'service' -d 'Service management commands'
complete -c its-camera-ai -n '__fish_use_subcommand' -a 'ml' -d 'ML operations and model management'
complete -c its-camera-ai -n '__fish_use_subcommand' -a 'data' -d 'Data pipeline control'
complete -c its-camera-ai -n '__fish_use_subcommand' -a 'security' -d 'Security and authentication'
complete -c its-camera-ai -n '__fish_use_subcommand' -a 'monitor' -d 'System monitoring and health'
complete -c its-camera-ai -n '__fish_use_subcommand' -a 'config' -d 'Configuration management'
complete -c its-camera-ai -n '__fish_use_subcommand' -a 'info' -d 'Show system information'
complete -c its-camera-ai -n '__fish_use_subcommand' -a 'version' -d 'Show version information'
complete -c its-camera-ai -n '__fish_use_subcommand' -a 'status' -d 'Show quick status overview'
complete -c its-camera-ai -n '__fish_use_subcommand' -a 'dashboard' -d 'Launch interactive dashboard'
complete -c its-camera-ai -n '__fish_use_subcommand' -a 'interactive' -d 'Start interactive mode'

# Service subcommands
complete -c its-camera-ai -n '__fish_seen_subcommand_from service' -a 'start stop restart status logs scale health'

# ML subcommands
complete -c its-camera-ai -n '__fish_seen_subcommand_from ml' -a 'train inference models deploy experiments benchmark optimize'

# Data subcommands
complete -c its-camera-ai -n '__fish_seen_subcommand_from data' -a 'pipeline stream batch validate export'

# Security subcommands
complete -c its-camera-ai -n '__fish_seen_subcommand_from security' -a 'auth users roles audit encrypt decrypt'

# Monitor subcommands
complete -c its-camera-ai -n '__fish_seen_subcommand_from monitor' -a 'metrics alerts health dashboard logs'

# Config subcommands
complete -c its-camera-ai -n '__fish_seen_subcommand_from config' -a 'get set list validate profile'
""",
    "powershell": """
# ITS Camera AI PowerShell completion

Register-ArgumentCompleter -Native -CommandName its-camera-ai -ScriptBlock {
    param($wordToComplete, $commandAst, $cursorPosition)
    
    $commands = @{
        'service' = @('start', 'stop', 'restart', 'status', 'logs', 'scale', 'health')
        'ml' = @('train', 'inference', 'models', 'deploy', 'experiments', 'benchmark', 'optimize')
        'data' = @('pipeline', 'stream', 'batch', 'validate', 'export')
        'security' = @('auth', 'users', 'roles', 'audit', 'encrypt', 'decrypt')
        'monitor' = @('metrics', 'alerts', 'health', 'dashboard', 'logs')
        'config' = @('get', 'set', 'list', 'validate', 'profile')
    }
    
    $tokens = $commandAst.ToString() -split '\\s+'
    
    if ($tokens.Length -eq 2) {
        # Complete main commands
        $mainCommands = @('service', 'ml', 'data', 'security', 'monitor', 'config', 'info', 'version', 'status', 'dashboard', 'interactive')
        $mainCommands | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
        }
    } elseif ($tokens.Length -eq 3 -and $commands.ContainsKey($tokens[1])) {
        # Complete subcommands
        $commands[$tokens[1]] | Where-Object { $_ -like "$wordToComplete*" } | ForEach-Object {
            [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
        }
    }
}
"""
}

# Installation paths for different shells
COMPLETION_PATHS = {
    "bash": [
        "~/.bash_completion.d/its-camera-ai",
        "/usr/local/etc/bash_completion.d/its-camera-ai",
        "/etc/bash_completion.d/its-camera-ai",
    ],
    "zsh": [
        "~/.zsh/completions/_its-camera-ai",
        "~/.oh-my-zsh/completions/_its-camera-ai",
        "/usr/local/share/zsh/site-functions/_its-camera-ai",
    ],
    "fish": [
        "~/.config/fish/completions/its-camera-ai.fish",
        "/usr/local/share/fish/completions/its-camera-ai.fish",
    ],
    "powershell": [
        "$PROFILE",
    ],
}


def detect_shell() -> str | None:
    """Detect the current shell from environment variables."""
    shell = os.environ.get("SHELL", "")
    if "bash" in shell:
        return "bash"
    elif "zsh" in shell:
        return "zsh"
    elif "fish" in shell:
        return "fish"
    elif sys.platform == "win32":
        return "powershell"
    return None


def get_completion_path(shell: str) -> Path | None:
    """Get the appropriate completion installation path for the shell."""
    if shell not in COMPLETION_PATHS:
        return None

    paths = COMPLETION_PATHS[shell]

    for path_str in paths:
        path = Path(path_str).expanduser()

        # For PowerShell, we append to the profile
        if shell == "powershell":
            return path

        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Check if we can write to this location
        try:
            if path.parent.is_dir() and os.access(path.parent, os.W_OK):
                return path
        except OSError:
            continue

    return None


def install_completion(shell: str) -> bool:
    """Install shell completion for the specified shell."""
    if shell not in COMPLETION_SCRIPTS:
        print_error(f"Unsupported shell: {shell}")
        return False

    completion_path = get_completion_path(shell)
    if not completion_path:
        print_error(f"Could not find writable completion path for {shell}")
        return False

    try:
        script = COMPLETION_SCRIPTS[shell].strip()

        if shell == "powershell":
            # Append to PowerShell profile
            with open(completion_path, "a", encoding="utf-8") as f:
                f.write(f"\n# ITS Camera AI completion\n{script}\n")
        else:
            # Write to completion file
            with open(completion_path, "w", encoding="utf-8") as f:
                f.write(script)

        print_success(f"Shell completion installed for {shell}")
        print_info(f"Completion file: {completion_path}")

        # Provide shell-specific activation instructions
        if shell == "bash":
            print_info("Run: source ~/.bashrc or start a new terminal session")
        elif shell == "zsh":
            print_info("Run: source ~/.zshrc or start a new terminal session")
        elif shell == "fish":
            print_info("Completions will be available in new fish sessions")
        elif shell == "powershell":
            print_info("Run: . $PROFILE or start a new PowerShell session")

        return True

    except Exception as e:
        print_error(f"Failed to install completion: {e}")
        return False


def uninstall_completion(shell: str) -> bool:
    """Uninstall shell completion for the specified shell."""
    completion_path = get_completion_path(shell)
    if not completion_path or not completion_path.exists():
        print_info(f"No completion found for {shell}")
        return True

    try:
        if shell == "powershell":
            # Remove from PowerShell profile
            if completion_path.exists():
                content = completion_path.read_text(encoding="utf-8")
                # Remove ITS Camera AI completion section
                lines = content.split("\n")
                filtered_lines = []
                skip = False

                for line in lines:
                    if "# ITS Camera AI completion" in line:
                        skip = True
                    elif skip and line.strip() == "":
                        skip = False
                    elif not skip:
                        filtered_lines.append(line)

                completion_path.write_text("\n".join(filtered_lines), encoding="utf-8")
        else:
            completion_path.unlink()

        print_success(f"Shell completion uninstalled for {shell}")
        return True

    except Exception as e:
        print_error(f"Failed to uninstall completion: {e}")
        return False


def show_completion_status() -> None:
    """Show the status of shell completions."""
    from rich.table import Table

    table = Table(title="Shell Completion Status")
    table.add_column("Shell", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Path", style="dim")

    for shell in COMPLETION_SCRIPTS:
        completion_path = get_completion_path(shell)
        if completion_path and completion_path.exists():
            status = "[green]Installed[/green]"
            path_str = str(completion_path)
        else:
            status = "[red]Not installed[/red]"
            path_str = "N/A"

        table.add_row(shell.title(), status, path_str)

    console.print(table)


def generate_completion_script(shell: str) -> str:
    """Generate completion script for the specified shell."""
    if shell not in COMPLETION_SCRIPTS:
        raise ValueError(f"Unsupported shell: {shell}")

    return COMPLETION_SCRIPTS[shell].strip()


app = typer.Typer(help="🔧 Shell completion management")


@app.command()
def install(
    shell: str | None = typer.Option(
        None, "--shell", "-s", help="Shell type (bash, zsh, fish, powershell)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force reinstall if already installed"
    ),
) -> None:
    """Install shell completion for the current or specified shell."""
    if not shell:
        shell = detect_shell()
        if not shell:
            print_error("Could not detect shell. Please specify with --shell")
            return

    shell = shell.lower()

    print_info(f"Installing shell completion for {shell}...")

    completion_path = get_completion_path(shell)
    if completion_path and completion_path.exists() and not force:
        print_info(f"Completion already installed for {shell}")
        print_info("Use --force to reinstall")
        return

    if install_completion(shell):
        print_success("Shell completion installed successfully!")
        print_info("You may need to restart your terminal or reload your shell configuration")
    else:
        print_error("Failed to install shell completion")


@app.command()
def uninstall(
    shell: str | None = typer.Option(
        None, "--shell", "-s", help="Shell type (bash, zsh, fish, powershell)"
    ),
) -> None:
    """Uninstall shell completion for the specified shell."""
    if not shell:
        shell = detect_shell()
        if not shell:
            print_error("Could not detect shell. Please specify with --shell")
            return

    shell = shell.lower()

    print_info(f"Uninstalling shell completion for {shell}...")

    if uninstall_completion(shell):
        print_success("Shell completion uninstalled successfully!")
    else:
        print_error("Failed to uninstall shell completion")


@app.command()
def status() -> None:
    """Show shell completion installation status."""
    show_completion_status()


@app.command()
def generate(
    shell: str = typer.Argument(..., help="Shell type (bash, zsh, fish, powershell)"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
) -> None:
    """Generate completion script for a specific shell."""
    shell = shell.lower()

    try:
        script = generate_completion_script(shell)

        if output:
            output.write_text(script, encoding="utf-8")
            print_success(f"Completion script saved to {output}")
        else:
            syntax = Syntax(script, shell, theme="monokai", line_numbers=False)
            panel = Panel(
                syntax,
                title=f"{shell.title()} Completion Script",
                border_style="blue",
            )
            console.print(panel)

    except ValueError as e:
        print_error(str(e))


@app.command()
def supported() -> None:
    """List supported shells."""
    from rich.table import Table

    table = Table(title="Supported Shells")
    table.add_column("Shell", style="cyan")
    table.add_column("Description", style="green")

    descriptions = {
        "bash": "GNU Bash shell with programmable completion",
        "zsh": "Z shell with extended completion system",
        "fish": "Fish shell with smart autocompletion",
        "powershell": "PowerShell with argument completion",
    }

    for shell in COMPLETION_SCRIPTS:
        table.add_row(shell.title(), descriptions.get(shell, ""))

    console.print(table)
