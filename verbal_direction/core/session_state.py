"""Session state machine for Claude Code sessions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class SessionStatus(Enum):
    IDLE = auto()
    WORKING = auto()
    WAITING_FOR_INPUT = auto()
    WAITING_FOR_PERMISSION = auto()
    PAUSED = auto()
    COMPLETED = auto()
    ERROR = auto()


@dataclass
class PendingQuestion:
    """A question from Claude that needs a voice response."""

    session_name: str
    text: str
    category: str  # "question", "permission", "error"
    timestamp: float = field(default_factory=time.time)
    spoken: bool = False


@dataclass
class SessionState:
    """Tracks the state of a single Claude Code session."""

    name: str
    directory: str
    status: SessionStatus = SessionStatus.IDLE
    current_output: str = ""
    pending_question: PendingQuestion | None = None
    last_activity: float = field(default_factory=time.time)
    total_cost_usd: float = 0.0
    session_id: str | None = None
    voice_paused: bool = False

    def transition(self, new_status: SessionStatus) -> None:
        """Transition to a new status, updating last_activity."""
        self.status = new_status
        self.last_activity = time.time()

    def set_pending_question(self, text: str, category: str) -> None:
        """Set a pending question that needs voice response."""
        self.pending_question = PendingQuestion(
            session_name=self.name,
            text=text,
            category=category,
        )
        if category == "permission":
            self.transition(SessionStatus.WAITING_FOR_PERMISSION)
        else:
            self.transition(SessionStatus.WAITING_FOR_INPUT)

    def clear_pending_question(self) -> None:
        """Clear the pending question after it's been answered."""
        self.pending_question = None
        self.transition(SessionStatus.WORKING)

    @property
    def is_waiting(self) -> bool:
        return self.status in (
            SessionStatus.WAITING_FOR_INPUT,
            SessionStatus.WAITING_FOR_PERMISSION,
        )

    @property
    def status_display(self) -> str:
        return self.status.name.replace("_", " ").title()
