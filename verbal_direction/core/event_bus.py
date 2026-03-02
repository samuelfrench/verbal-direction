"""Async event bus for inter-component communication."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    # Output events (from sessions)
    SESSION_OUTPUT = auto()       # Raw output from a session
    SESSION_QUESTION = auto()     # Classified as needing input
    SESSION_PERMISSION = auto()   # Permission request detected
    SESSION_ERROR = auto()        # Error detected
    SESSION_INFO = auto()         # Informational output
    SESSION_COMPLETE = auto()     # Session finished

    # Voice events
    VOICE_TRANSCRIPTION = auto()  # STT produced text
    VOICE_ROUTED = auto()         # Text routed to a session
    VOICE_MIC_LEVEL = auto()      # Mic audio level (0-100)
    VOICE_TTS_STATUS = auto()     # TTS speaking/idle status
    VOICE_STREAM_STATUS = auto()  # Audio stream health

    # Control events
    SESSION_LAUNCHED = auto()
    SESSION_KILLED = auto()
    SESSION_PAUSED = auto()
    SESSION_RESUMED = auto()


@dataclass
class Event:
    type: EventType
    session_name: str
    data: Any = None
    timestamp: float = field(default_factory=lambda: __import__("time").time())


class EventBus:
    """Central event bus using asyncio queues."""

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[asyncio.Queue[Event]]] = {}
        self._global_subscribers: list[asyncio.Queue[Event]] = []

    def subscribe(self, *event_types: EventType) -> asyncio.Queue[Event]:
        """Subscribe to specific event types. Returns a queue to read from."""
        queue: asyncio.Queue[Event] = asyncio.Queue()
        for event_type in event_types:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(queue)
        return queue

    def subscribe_all(self) -> asyncio.Queue[Event]:
        """Subscribe to all events."""
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._global_subscribers.append(queue)
        return queue

    async def publish(self, event: Event) -> None:
        """Publish an event to all relevant subscribers."""
        # Notify type-specific subscribers
        for queue in self._subscribers.get(event.type, []):
            await queue.put(event)

        # Notify global subscribers
        for queue in self._global_subscribers:
            await queue.put(event)

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        """Remove a queue from all subscriptions."""
        for subs in self._subscribers.values():
            if queue in subs:
                subs.remove(queue)
        if queue in self._global_subscribers:
            self._global_subscribers.remove(queue)
