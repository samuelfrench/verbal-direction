"""TUI dashboard using Textual."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Header, Footer, Static, Label, DataTable, RichLog


class SessionCard(Static):
    """A card showing a single session's status."""

    def __init__(self, name: str = "", status: str = "Idle", question: str = "", cost: str = "$0.00") -> None:
        super().__init__()
        self._name = name
        self._status = status
        self._question = question
        self._cost = cost

    def compose(self) -> ComposeResult:
        yield Static(f"[bold]{self._name}[/bold]", classes="session-name")
        yield Static(f"Status: {self._status}", classes="session-status")
        if self._question:
            yield Static(f"Q: {self._question[:80]}", classes="session-question")
        yield Static(f"Cost: {self._cost}", classes="session-cost")

    DEFAULT_CSS = """
    SessionCard {
        border: solid $primary;
        padding: 1;
        margin: 1;
        height: auto;
        min-height: 5;
    }
    .session-name {
        color: $text;
        text-style: bold;
    }
    .session-status {
        color: $success;
    }
    .session-question {
        color: $warning;
    }
    .session-cost {
        color: $text-muted;
    }
    """


class VoiceStatus(Static):
    """Shows current voice pipeline status."""

    is_listening = reactive(True)
    last_heard = reactive("")

    def render(self) -> str:
        status = "[green]Listening[/green]" if self.is_listening else "[red]Stopped[/red]"
        heard = f"  Last: \"{self.last_heard}\"" if self.last_heard else ""
        return f"Voice: {status}{heard}"


class VoiceQueue(Static):
    """Shows pending questions waiting for voice response."""

    def compose(self) -> ComposeResult:
        yield Static("[bold]Pending Questions[/bold]")
        yield Static("  (none)", classes="queue-empty")

    DEFAULT_CSS = """
    VoiceQueue {
        border: solid $secondary;
        padding: 1;
        margin: 1;
        height: auto;
        min-height: 4;
    }
    .queue-empty {
        color: $text-muted;
    }
    """


class VDDashboard(App):
    """Main TUI dashboard for verbal-direction."""

    CSS = """
    #main-container {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
    }
    #sessions-panel {
        column-span: 1;
        row-span: 2;
    }
    #voice-panel {
        column-span: 1;
    }
    #log-panel {
        column-span: 2;
        height: 10;
        border: solid $primary;
        margin: 1;
    }
    """

    TITLE = "verbal-direction"
    SUB_TITLE = "Voice-Controlled Claude Code Manager"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("d", "toggle_dark", "Dark/Light"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            with Vertical(id="sessions-panel"):
                yield Static("[bold]Sessions[/bold]")
                yield SessionCard(
                    name="(no active sessions)",
                    status="Launch with: vd launch <name> <dir>",
                )
            with Vertical(id="voice-panel"):
                yield VoiceStatus()
                yield VoiceQueue()
        yield RichLog(id="log-panel", highlight=True, markup=True)
        yield Footer()

    def action_refresh(self) -> None:
        """Refresh the dashboard."""
        log = self.query_one(RichLog)
        log.write("[dim]Refreshed[/dim]")

    def action_toggle_dark(self) -> None:
        """Toggle dark mode."""
        self.theme = "textual-light" if self.theme == "textual-dark" else "textual-dark"
