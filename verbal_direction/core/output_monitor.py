"""Output monitor — classifies session output and routes to attention filter."""

from __future__ import annotations

import asyncio
import logging

from verbal_direction.core.event_bus import EventBus, Event, EventType
from verbal_direction.intelligence.attention_filter import AttentionFilter

logger = logging.getLogger(__name__)


class OutputMonitor:
    """Monitors session output events and classifies them via Ollama."""

    def __init__(self, event_bus: EventBus, attention_filter: AttentionFilter) -> None:
        self._event_bus = event_bus
        self._attention_filter = attention_filter
        self._queue = event_bus.subscribe(EventType.SESSION_OUTPUT)
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start monitoring output events."""
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        """Main monitoring loop."""
        while True:
            event = await self._queue.get()
            text = event.data.get("text", "") if event.data else ""
            if not text or len(text.strip()) < 5:
                continue

            try:
                classification = await self._attention_filter.classify(text)

                if classification == "question":
                    await self._event_bus.publish(Event(
                        type=EventType.SESSION_QUESTION,
                        session_name=event.session_name,
                        data={"text": text, "classification": classification},
                    ))
                elif classification == "permission":
                    # Permission events are already handled by the session manager's
                    # can_use_tool callback, so we just log here
                    logger.debug("Permission output from %s: %s", event.session_name, text[:100])
                elif classification == "error":
                    await self._event_bus.publish(Event(
                        type=EventType.SESSION_ERROR,
                        session_name=event.session_name,
                        data={"text": text, "classification": classification},
                    ))
                else:
                    await self._event_bus.publish(Event(
                        type=EventType.SESSION_INFO,
                        session_name=event.session_name,
                        data={"text": text, "classification": classification},
                    ))
            except Exception as e:
                logger.error("Classification error for %s: %s", event.session_name, e)

    async def stop(self) -> None:
        """Stop monitoring."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._event_bus.unsubscribe(self._queue)
