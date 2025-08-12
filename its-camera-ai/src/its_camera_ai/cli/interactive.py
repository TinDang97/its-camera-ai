"""Interactive mode for ITS Camera AI CLI.

Provides an interactive mode with prompts, wizards, and guided workflows
for complex operations using inquirer and rich for enhanced UX.
"""

import asyncio
from typing import Any

import inquirer
import typer
from inquirer import themes
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from .utils import (
    console,
    print_error,
    print_info,
    print_success,
)


# Custom theme for inquirer
class CustomTheme(themes.Theme):
    def __init__(self):
        super().__init__()
        self.Question.mark_color = themes.term.cyan
        self.Question.brackets_color = themes.term.normal
        self.Question.default_color = themes.term.normal
        self.Checkbox.selection_color = themes.term.blue
        self.Checkbox.selection_icon = "❯"
        self.Checkbox.selected_icon = "◉"
        self.Checkbox.unselected_icon = "◯"
        self.List.selection_color = themes.term.blue
        self.List.selection_cursor = "❯"

CUSTOM_THEME = CustomTheme()


class InteractiveSession:
    """Interactive session manager for complex workflows."""

    def __init__(self):
        self.history: list[dict[str, Any]] = []
        self.favorites: list[str] = []
        self.current_context: dict[str, Any] = {}

    def add_to_history(self, command: str, parameters: dict[str, Any]) -> None:
        """Add a command to the session history."""
        entry = {
            'command': command,
            'parameters': parameters,
            'timestamp': asyncio.get_event_loop().time(),
        }
        self.history.append(entry)

    def add_favorite(self, command: str) -> None:
        """Add a command to favorites."""
        if command not in self.favorites:
            self.favorites.append(command)

    def remove_favorite(self, command: str) -> None:
        """Remove a command from favorites."""
        if command in self.favorites:
            self.favorites.remove(command)


class InteractiveWizard:
    """Base class for interactive wizards."""

    def __init__(self, session: InteractiveSession):
        self.session = session
        self.steps: list[dict[str, Any]] = []
        self.results: dict[str, Any] = {}

    def add_step(self, name: str, prompt_type: str, **kwargs) -> None:
        """Add a step to the wizard."""
        step = {
            'name': name,
            'type': prompt_type,
            **kwargs
        }
        self.steps.append(step)

    async def run(self) -> dict[str, Any]:
        """Run the interactive wizard."""
        console.print(Panel(
            Text("Interactive Wizard", style="bold cyan"),
            title="🧙‍♂️ Wizard Mode",
            border_style="blue",
        ))

        for step in self.steps:
            result = await self._execute_step(step)
            self.results[step['name']] = result

        return self.results

    async def _execute_step(self, step: dict[str, Any]) -> Any:
        """Execute a single wizard step."""
        step_type = step['type']
        name = step['name']

        if step_type == 'list':
            question = inquirer.List(
                name,
                message=step.get('message', f"Select {name}"),
                choices=step.get('choices', []),
            )
        elif step_type == 'checkbox':
            question = inquirer.Checkbox(
                name,
                message=step.get('message', f"Select {name}"),
                choices=step.get('choices', []),
            )
        elif step_type == 'text':
            question = inquirer.Text(
                name,
                message=step.get('message', f"Enter {name}"),
                default=step.get('default', ''),
            )
        elif step_type == 'password':
            question = inquirer.Password(
                name,
                message=step.get('message', f"Enter {name}"),
            )
        elif step_type == 'confirm':
            question = inquirer.Confirm(
                name,
                message=step.get('message', f"Confirm {name}"),
                default=step.get('default', False),
            )
        else:
            raise ValueError(f"Unknown step type: {step_type}")

        answer = inquirer.prompt([question], theme=CUSTOM_THEME)
        return answer[name] if answer else None


class TrainingWizard(InteractiveWizard):
    """Interactive wizard for model training configuration."""

    def __init__(self, session: InteractiveSession):
        super().__init__(session)
        self._setup_steps()

    def _setup_steps(self) -> None:
        """Setup wizard steps for training configuration."""
        self.add_step(
            'model_type',
            'list',
            message="Select model architecture:",
            choices=[
                'yolo11n (Nano - Fast, lower accuracy)',
                'yolo11s (Small - Balanced)',
                'yolo11m (Medium - Good accuracy)',
                'yolo11l (Large - High accuracy)',
                'yolo11x (Extra Large - Best accuracy)',
            ]
        )

        self.add_step(
            'dataset',
            'text',
            message="Enter dataset path or name:",
            default="traffic_dataset"
        )

        self.add_step(
            'training_mode',
            'list',
            message="Select training mode:",
            choices=[
                'quick (Fast training, basic settings)',
                'standard (Recommended settings)',
                'advanced (Custom configuration)',
            ]
        )

        self.add_step(
            'device',
            'list',
            message="Select training device:",
            choices=['auto', 'cpu', 'cuda', 'mps (Apple Silicon)']
        )

        self.add_step(
            'use_pretrained',
            'confirm',
            message="Use pretrained weights?",
            default=True
        )


class DeploymentWizard(InteractiveWizard):
    """Interactive wizard for model deployment."""

    def __init__(self, session: InteractiveSession):
        super().__init__(session)
        self._setup_steps()

    def _setup_steps(self) -> None:
        """Setup wizard steps for deployment configuration."""
        self.add_step(
            'model_name',
            'text',
            message="Enter model name to deploy:"
        )

        self.add_step(
            'environment',
            'list',
            message="Select target environment:",
            choices=['development', 'staging', 'production']
        )

        self.add_step(
            'deployment_strategy',
            'list',
            message="Select deployment strategy:",
            choices=[
                'rolling (Gradual replacement)',
                'blue-green (Zero downtime)',
                'canary (Gradual traffic shift)',
            ]
        )

        self.add_step(
            'scaling',
            'list',
            message="Select scaling configuration:",
            choices=[
                'manual (Fixed replicas)',
                'auto (Auto-scaling enabled)',
                'spot (Use spot instances)',
            ]
        )

        self.add_step(
            'monitoring',
            'checkbox',
            message="Select monitoring features:",
            choices=[
                'performance_metrics',
                'error_tracking',
                'custom_alerts',
                'log_aggregation',
            ]
        )


class SystemSetupWizard(InteractiveWizard):
    """Interactive wizard for system setup and configuration."""

    def __init__(self, session: InteractiveSession):
        super().__init__(session)
        self._setup_steps()

    def _setup_steps(self) -> None:
        """Setup wizard steps for system configuration."""
        self.add_step(
            'setup_type',
            'list',
            message="Select setup type:",
            choices=[
                'development (Local development environment)',
                'production (Production deployment)',
                'edge (Edge device deployment)',
            ]
        )

        self.add_step(
            'components',
            'checkbox',
            message="Select components to configure:",
            choices=[
                'database (PostgreSQL, Redis)',
                'message_queue (Kafka, Redis)',
                'monitoring (Prometheus, Grafana)',
                'security (Authentication, encryption)',
                'ml_pipeline (Training, inference)',
            ]
        )

        self.add_step(
            'storage_backend',
            'list',
            message="Select storage backend:",
            choices=['local', 's3', 'gcs', 'azure', 'minio']
        )

        self.add_step(
            'enable_gpu',
            'confirm',
            message="Enable GPU support?",
            default=False
        )


async def interactive_main_menu(session: InteractiveSession) -> str:
    """Display the main interactive menu."""
    choices = [
        '🚀 Quick Start (Setup wizard)',
        '🤖 Train Model (Training wizard)',
        '🚀 Deploy Model (Deployment wizard)',
        '📊 View Status (System dashboard)',
        '🔧 Configuration (Settings)',
        '📈 Monitoring (Metrics and logs)',
        '🔒 Security (Authentication)',
        '📚 Recent Commands (History)',
        '⭐ Favorites (Quick access)',
        '❌ Exit',
    ]

    if session.favorites:
        favorites_display = f"⭐ Favorites ({len(session.favorites)} items)"
        choices[8] = favorites_display

    question = inquirer.List(
        'action',
        message="What would you like to do?",
        choices=choices,
    )

    answer = inquirer.prompt([question], theme=CUSTOM_THEME)
    return answer['action'] if answer else 'exit'


async def show_history(session: InteractiveSession) -> None:
    """Display command history."""
    if not session.history:
        print_info("No commands in history")
        return

    table = Table(title="Command History")
    table.add_column("Time", style="dim")
    table.add_column("Command", style="cyan")
    table.add_column("Parameters", style="green")

    # Show last 10 commands
    recent_history = session.history[-10:]

    for entry in recent_history:
        # Convert timestamp to readable format
        time_str = f"{int(entry['timestamp'])}s ago"
        params_str = ", ".join(f"{k}={v}" for k, v in entry['parameters'].items())

        table.add_row(
            time_str,
            entry['command'],
            params_str[:50] + "..." if len(params_str) > 50 else params_str
        )

    console.print(table)

    # Allow user to rerun a command
    if Confirm.ask("Would you like to rerun a command?"):
        choices = [f"{i}: {entry['command']}" for i, entry in enumerate(recent_history)]

        question = inquirer.List(
            'command',
            message="Select command to rerun:",
            choices=choices,
        )

        answer = inquirer.prompt([question], theme=CUSTOM_THEME)
        if answer:
            index = int(answer['command'].split(':')[0])
            selected_entry = recent_history[index]
            print_info(f"Rerunning: {selected_entry['command']}")
            # Here you would implement command rerun logic


async def show_favorites(session: InteractiveSession) -> None:
    """Display and manage favorite commands."""
    if not session.favorites:
        print_info("No favorite commands")

        if Confirm.ask("Would you like to add some favorites?"):
            available_commands = [
                'its-camera-ai status',
                'its-camera-ai ml models',
                'its-camera-ai monitor metrics',
                'its-camera-ai service health',
                'its-camera-ai config get',
            ]

            question = inquirer.Checkbox(
                'commands',
                message="Select commands to add to favorites:",
                choices=available_commands,
            )

            answer = inquirer.prompt([question], theme=CUSTOM_THEME)
            if answer and answer['commands']:
                for cmd in answer['commands']:
                    session.add_favorite(cmd)
                print_success(f"Added {len(answer['commands'])} favorites")

        return

    table = Table(title="Favorite Commands")
    table.add_column("Index", style="dim")
    table.add_column("Command", style="cyan")

    for i, fav in enumerate(session.favorites):
        table.add_row(str(i + 1), fav)

    console.print(table)

    # Favorite management options
    action = inquirer.List(
        'action',
        message="What would you like to do?",
        choices=[
            'Run a favorite command',
            'Add new favorite',
            'Remove favorite',
            'Back to main menu',
        ],
    )

    answer = inquirer.prompt([action], theme=CUSTOM_THEME)
    if not answer:
        return

    if answer['action'] == 'Run a favorite command':
        cmd_question = inquirer.List(
            'command',
            message="Select command to run:",
            choices=session.favorites,
        )

        cmd_answer = inquirer.prompt([cmd_question], theme=CUSTOM_THEME)
        if cmd_answer:
            print_info(f"Running: {cmd_answer['command']}")
            # Here you would implement command execution logic

    elif answer['action'] == 'Add new favorite':
        new_fav = Prompt.ask("Enter command to add to favorites")
        if new_fav:
            session.add_favorite(new_fav)
            print_success(f"Added '{new_fav}' to favorites")

    elif answer['action'] == 'Remove favorite':
        if session.favorites:
            remove_question = inquirer.List(
                'command',
                message="Select command to remove:",
                choices=session.favorites,
            )

            remove_answer = inquirer.prompt([remove_question], theme=CUSTOM_THEME)
            if remove_answer:
                session.remove_favorite(remove_answer['command'])
                print_success(f"Removed '{remove_answer['command']}' from favorites")


async def show_quick_status() -> None:
    """Show a quick system status overview."""
    with console.status("Fetching system status...", spinner="dots"):
        await asyncio.sleep(1)  # Simulate API calls

    # Create status dashboard
    status_table = Table(title="System Status Overview")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Details", style="dim")

    # Simulate status data
    components = [
        ("API Server", "✅ Running", "Port 8000, 2 workers"),
        ("ML Pipeline", "✅ Active", "3 models deployed"),
        ("Database", "✅ Connected", "PostgreSQL 14.2"),
        ("Redis Cache", "✅ Connected", "6.2.7, 45% memory"),
        ("Message Queue", "⚠️ Warning", "High message backlog"),
        ("GPU Resources", "✅ Available", "CUDA 11.8, 8GB VRAM"),
    ]

    for component, status, details in components:
        status_table.add_row(component, status, details)

    console.print(status_table)

    # Show recent metrics
    metrics_table = Table(title="Key Metrics (Last 5 minutes)")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    metrics_table.add_column("Trend", style="yellow")

    metrics = [
        ("Inference Rate", "125 req/sec", "📈 +12%"),
        ("Model Accuracy", "89.2%", "📊 Stable"),
        ("Response Time", "45ms avg", "📉 -8%"),
        ("Error Rate", "0.1%", "📊 Stable"),
    ]

    for metric, value, trend in metrics:
        metrics_table.add_row(metric, value, trend)

    console.print(metrics_table)


async def interactive_mode() -> None:
    """Main interactive mode loop."""
    session = InteractiveSession()

    # Welcome banner
    welcome_text = Text()
    welcome_text.append("🎥 ITS Camera AI Interactive Mode\n", style="bold blue")
    welcome_text.append("Navigate complex workflows with guided wizards and menus", style="italic")

    console.print(Panel(
        welcome_text,
        title="Welcome",
        border_style="blue",
        padding=(1, 2),
    ))

    while True:
        try:
            action = await interactive_main_menu(session)

            if '❌ Exit' in action:
                print_info("Goodbye!")
                break

            elif '🚀 Quick Start' in action:
                wizard = SystemSetupWizard(session)
                print_info("Starting system setup wizard...")
                results = await wizard.run()
                session.add_to_history('setup_wizard', results)
                print_success("Setup wizard completed!")

            elif '🤖 Train Model' in action:
                wizard = TrainingWizard(session)
                print_info("Starting training wizard...")
                results = await wizard.run()
                session.add_to_history('training_wizard', results)
                print_success("Training wizard completed!")

            elif '🚀 Deploy Model' in action:
                wizard = DeploymentWizard(session)
                print_info("Starting deployment wizard...")
                results = await wizard.run()
                session.add_to_history('deployment_wizard', results)
                print_success("Deployment wizard completed!")

            elif '📊 View Status' in action:
                await show_quick_status()

            elif '📚 Recent Commands' in action:
                await show_history(session)

            elif '⭐ Favorites' in action or 'Favorites' in action:
                await show_favorites(session)

            elif '🔧 Configuration' in action:
                print_info("Configuration menu coming soon...")

            elif '📈 Monitoring' in action:
                print_info("Monitoring dashboard coming soon...")

            elif '🔒 Security' in action:
                print_info("Security management coming soon...")

            # Pause before showing menu again
            if not Confirm.ask("\nContinue to main menu?", default=True):
                break

        except KeyboardInterrupt:
            print_info("\nGoodbye!")
            break
        except Exception as e:
            print_error(f"An error occurred: {e}")
            if not Confirm.ask("Continue?", default=True):
                break


app = typer.Typer(help="🧙‍♂️ Interactive mode and wizards")


@app.command()
def start() -> None:
    """Start interactive mode with guided wizards and menus."""
    try:
        asyncio.run(interactive_mode())
    except KeyboardInterrupt:
        print_info("Interactive mode cancelled")
    except Exception as e:
        print_error(f"Failed to start interactive mode: {e}")


@app.command()
def wizard(
    wizard_type: str = typer.Argument(
        ...,
        help="Wizard type (training, deployment, setup)"
    )
) -> None:
    """Start a specific wizard directly."""
    session = InteractiveSession()

    async def run_wizard():
        if wizard_type == "training":
            wizard = TrainingWizard(session)
        elif wizard_type == "deployment":
            wizard = DeploymentWizard(session)
        elif wizard_type == "setup":
            wizard = SystemSetupWizard(session)
        else:
            print_error(f"Unknown wizard type: {wizard_type}")
            return

        print_info(f"Starting {wizard_type} wizard...")
        results = await wizard.run()

        # Display results
        console.print(Panel(
            f"Wizard completed successfully!\n\nResults: {results}",
            title="✅ Success",
            border_style="green",
        ))

    try:
        asyncio.run(run_wizard())
    except KeyboardInterrupt:
        print_info("Wizard cancelled")
    except Exception as e:
        print_error(f"Wizard failed: {e}")


if __name__ == "__main__":
    app()
