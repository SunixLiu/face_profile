from enum import Enum, auto

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.text import Text

console = Console()


class Command(Enum):
    PROFILE = auto()
    HELP = auto()
    QUIT = auto()
    NONE = auto()


def parse_command(user_input: str) -> tuple[Command, str]:
    """Parse special commands from user input. Returns (command, remaining_text)."""
    text = user_input.strip()
    if text in ("/quit", "/exit", "/q"):
        return Command.QUIT, ""
    if text == "/help":
        return Command.HELP, ""
    if text == "/profile":
        return Command.PROFILE, ""
    return Command.NONE, text


def show_welcome():
    console.print(
        Panel.fit(
            "[bold cyan]Face Profile Chat[/bold cyan]\n"
            "Type [yellow]/help[/yellow] for commands, "
            "[yellow]/quit[/yellow] to exit.\n"
            "Say [yellow]请建立我的个人档案[/yellow] to create a profile.",
            border_style="cyan",
        )
    )


def show_help():
    console.print(
        Panel.fit(
            "[bold]Commands:[/bold]\n"
            "  [yellow]/help[/yellow]    - Show this help\n"
            "  [yellow]/profile[/yellow] - Show current profile info\n"
            "  [yellow]/quit[/yellow]    - Exit the program\n\n"
            "[bold]Special phrases:[/bold]\n"
            "  [yellow]请建立我的个人档案[/yellow] - Create a new profile via webcam",
            title="Help",
            border_style="green",
        )
    )


def show_profile_info(profile):
    console.print(
        Panel.fit(
            f"[bold]Name:[/bold] {profile.name}\n"
            f"[bold]ID:[/bold] {profile.id[:8]}...\n"
            f"[bold]Language:[/bold] {profile.language}\n"
            f"[bold]Tone:[/bold] {profile.tone}\n"
            f"[bold]Topics:[/bold] {', '.join(profile.topics) if profile.topics else '(learning...)'}\n"
            f"[bold]Created:[/bold] {profile.created_at[:10]}",
            title="Profile",
            border_style="blue",
        )
    )


def show_system_message(text: str):
    console.print(f"[dim italic]{text}[/dim italic]")


def show_user_message(name: str, text: str):
    console.print(f"[bold green]{name}[/bold green]: {text}")


def show_agent_message(name: str):
    console.print(f"[bold blue]AI[/bold blue]: ", end="")


def show_error(text: str):
    console.print(f"[red bold]Error:[/red bold] {text}")


def prompt_input(prompt_text: str = "> ") -> str:
    return Prompt.ask(prompt_text)


def show_prompt(text: str):
    """Show a prompt/question to the user."""
    console.print(f"[yellow]{text}[/yellow]")


def stream_token(text: str):
    """Stream a single token to the console."""
    console.print(text, end="", markup=False)
