"""Response dispatcher — routes voice responses to the correct session."""

from __future__ import annotations

import asyncio
import logging

from verbal_direction.core.event_bus import EventBus, Event, EventType
from verbal_direction.core.session_manager import SessionManager

logger = logging.getLogger(__name__)


class ResponseDispatcher:
    """Dispatches voice-transcribed responses to the correct Claude session."""

    def __init__(self, event_bus: EventBus, session_manager: SessionManager) -> None:
        self._event_bus = event_bus
        self._session_manager = session_manager
        self._queue = event_bus.subscribe(EventType.VOICE_ROUTED)
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start dispatching responses."""
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        """Main dispatch loop."""
        while True:
            event = await self._queue.get()
            session_name = event.data.get("target_session", "") if event.data else ""
            text = event.data.get("text", "") if event.data else ""

            if not session_name or not text:
                logger.warning("Invalid routed event: %s", event)
                continue

            session = self._session_manager.get(session_name)
            if not session:
                logger.warning("Session '%s' not found for response", session_name)
                continue

            # Handle permission responses
            if session.state.status.name == "WAITING_FOR_PERMISSION":
                text_lower = text.lower().strip()
                approved = text_lower in (
                    "yes", "yeah", "yep", "sure", "approve", "allow",
                    "go ahead", "ok", "okay", "do it", "proceed",
                )
                session.resolve_permission(approved)
                logger.info(
                    "Permission %s for session '%s'",
                    "approved" if approved else "denied",
                    session_name,
                )
            else:
                # Send as a regular response
                try:
                    await session.send_response(text)
                    logger.info("Dispatched response to '%s': %s", session_name, text[:50])
                except Exception as e:
                    logger.error("Failed to dispatch to '%s': %s", session_name, e)

    async def stop(self) -> None:
        """Stop dispatching."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._event_bus.unsubscribe(self._queue)
