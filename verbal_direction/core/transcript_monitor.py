"""Monitor Claude Code transcript files for new assistant messages."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from verbal_direction.core.event_bus import EventBus, Event, EventType
from verbal_direction.core.process_discovery import DiscoveredSession
from verbal_direction.intelligence.attention_filter import AttentionFilter

logger = logging.getLogger(__name__)


class TranscriptMonitor:
    """Watches .jsonl transcript files for new assistant messages."""

    def __init__(
        self,
        event_bus: EventBus,
        attention_filter: AttentionFilter,
        poll_interval: float = 0.5,
    ) -> None:
        self._event_bus = event_bus
        self._attention_filter = attention_filter
        self._poll_interval = poll_interval

        # Track file positions per transcript
        self._file_positions: dict[Path, int] = {}
        # Track sessions we're monitoring
        self._sessions: dict[str, DiscoveredSession] = {}
        self._task: asyncio.Task | None = None

    def set_sessions(self, sessions: list[DiscoveredSession]) -> None:
        """Update the set of sessions to monitor."""
        self._sessions = {s.label: s for s in sessions if s.transcript_path}

        # Initialize file positions to end-of-file for new sessions
        for session in sessions:
            if session.transcript_path and session.transcript_path not in self._file_positions:
                try:
                    self._file_positions[session.transcript_path] = session.transcript_path.stat().st_size
                except OSError:
                    self._file_positions[session.transcript_path] = 0

    async def start(self) -> None:
        """Start monitoring transcripts."""
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        """Main polling loop."""
        while True:
            await asyncio.sleep(self._poll_interval)

            for label, session in list(self._sessions.items()):
                path = session.transcript_path
                if not path or not path.exists():
                    continue

                try:
                    current_size = path.stat().st_size
                except OSError:
                    continue

                last_pos = self._file_positions.get(path, 0)
                if current_size <= last_pos:
                    continue

                # Read new content
                new_lines = self._read_new_lines(path, last_pos)
                self._file_positions[path] = current_size

                for line_data in new_lines:
                    await self._process_message(label, session, line_data)

    def _read_new_lines(self, path: Path, from_pos: int) -> list[dict]:
        """Read new JSONL lines from a transcript file."""
        results = []
        try:
            with open(path, "r") as f:
                f.seek(from_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.debug("Error reading %s: %s", path, e)
        return results

    async def _process_message(
        self,
        label: str,
        session: DiscoveredSession,
        data: dict,
    ) -> None:
        """Process a new transcript message."""
        msg_type = data.get("type")

        if msg_type != "assistant":
            return

        # Extract text from assistant message content blocks
        message = data.get("message", {})
        content = message.get("content", [])

        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))

        text = "\n".join(text_parts).strip()
        if not text or len(text) < 5:
            return

        # Classify with attention filter
        try:
            classification = await self._attention_filter.classify(text)
        except Exception as e:
            logger.error("Classification failed for %s: %s", label, e)
            return

        if classification == "question":
            await self._event_bus.publish(Event(
                type=EventType.SESSION_QUESTION,
                session_name=label,
                data={
                    "text": text,
                    "classification": classification,
                    "session": session,
                },
            ))
        elif classification == "error":
            await self._event_bus.publish(Event(
                type=EventType.SESSION_ERROR,
                session_name=label,
                data={
                    "text": text,
                    "classification": classification,
                    "session": session,
                },
            ))
        else:
            await self._event_bus.publish(Event(
                type=EventType.SESSION_INFO,
                session_name=label,
                data={
                    "text": text[:200],
                    "classification": classification,
                },
            ))

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
