"""Session manager — creates and tracks ClaudeSDKClient instances."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny, ToolPermissionContext

from verbal_direction.core.event_bus import EventBus, Event, EventType
from verbal_direction.core.session_state import SessionState, SessionStatus

logger = logging.getLogger(__name__)


class ManagedSession:
    """A single managed Claude Code session."""

    def __init__(
        self,
        name: str,
        directory: str,
        event_bus: EventBus,
        initial_prompt: str | None = None,
    ) -> None:
        self.name = name
        self.state = SessionState(name=name, directory=directory)
        self._event_bus = event_bus
        self._initial_prompt = initial_prompt
        self._client: ClaudeSDKClient | None = None
        self._monitor_task: asyncio.Task[None] | None = None
        self._permission_future: asyncio.Future[bool] | None = None

    async def _permission_handler(
        self, tool_name: str, input_data: dict[str, Any], context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Handle permission requests by routing to voice."""
        desc = f"Tool: {tool_name}"
        if "command" in input_data:
            desc += f" — {input_data['command']}"
        elif "file_path" in input_data:
            desc += f" — {input_data['file_path']}"

        self.state.set_pending_question(
            text=f"Permission needed: {desc}",
            category="permission",
        )
        await self._event_bus.publish(Event(
            type=EventType.SESSION_PERMISSION,
            session_name=self.name,
            data={"tool": tool_name, "input": input_data, "description": desc},
        ))

        # Wait for voice response
        self._permission_future = asyncio.get_event_loop().create_future()
        try:
            approved = await self._permission_future
        finally:
            self._permission_future = None

        self.state.clear_pending_question()

        if approved:
            return PermissionResultAllow(updated_input=input_data)
        return PermissionResultDeny(message="Denied by voice command")

    def resolve_permission(self, approved: bool) -> None:
        """Resolve a pending permission request."""
        if self._permission_future and not self._permission_future.done():
            self._permission_future.set_result(approved)

    async def start(self) -> None:
        """Start the session and begin monitoring."""
        options = ClaudeAgentOptions(
            cwd=self.state.directory,
            permission_mode="default",
            can_use_tool=self._permission_handler,
            setting_sources=["user", "project", "local"],
        )

        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()

        self.state.transition(SessionStatus.WORKING)
        await self._event_bus.publish(Event(
            type=EventType.SESSION_LAUNCHED,
            session_name=self.name,
        ))

        if self._initial_prompt:
            await self._client.query(self._initial_prompt)

        # Start the output monitor
        self._monitor_task = asyncio.create_task(self._monitor_output())

    async def send_response(self, text: str) -> None:
        """Send a user response to this session."""
        if not self._client:
            return

        self.state.clear_pending_question()
        await self._client.query(text)

    async def _monitor_output(self) -> None:
        """Monitor session output and publish events."""
        if not self._client:
            return

        try:
            async for message in self._client.receive_messages():
                msg_type = type(message).__name__
                data = {"message": message, "type": msg_type}

                # Extract text content from AssistantMessages
                text = ""
                if hasattr(message, "content"):
                    for block in message.content:
                        if hasattr(block, "text"):
                            text += block.text

                if text:
                    data["text"] = text
                    self.state.current_output = text

                await self._event_bus.publish(Event(
                    type=EventType.SESSION_OUTPUT,
                    session_name=self.name,
                    data=data,
                ))

                # Check for result message (turn complete)
                if msg_type == "ResultMessage":
                    cost = getattr(message, "cost_usd", 0) or 0
                    self.state.total_cost_usd += cost
                    if not self.state.is_waiting:
                        self.state.transition(SessionStatus.IDLE)

        except Exception as e:
            logger.error("Session %s monitor error: %s", self.name, e)
            self.state.transition(SessionStatus.ERROR)
            await self._event_bus.publish(Event(
                type=EventType.SESSION_ERROR,
                session_name=self.name,
                data={"error": str(e)},
            ))

    async def kill(self) -> None:
        """Kill this session."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        if self._client:
            await self._client.disconnect()
            self._client = None

        self.state.transition(SessionStatus.COMPLETED)
        await self._event_bus.publish(Event(
            type=EventType.SESSION_KILLED,
            session_name=self.name,
        ))


class SessionManager:
    """Manages multiple Claude Code sessions."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._sessions: dict[str, ManagedSession] = {}

    @property
    def sessions(self) -> dict[str, ManagedSession]:
        return self._sessions

    async def launch(
        self,
        name: str,
        directory: str,
        initial_prompt: str | None = None,
    ) -> ManagedSession:
        """Launch a new Claude session."""
        if name in self._sessions:
            raise ValueError(f"Session '{name}' already exists")

        session = ManagedSession(
            name=name,
            directory=directory,
            event_bus=self._event_bus,
            initial_prompt=initial_prompt,
        )
        self._sessions[name] = session
        await session.start()
        return session

    def get(self, name: str) -> ManagedSession | None:
        return self._sessions.get(name)

    async def kill(self, name: str) -> None:
        """Kill a session by name."""
        session = self._sessions.pop(name, None)
        if session:
            await session.kill()

    async def kill_all(self) -> None:
        """Kill all sessions."""
        names = list(self._sessions.keys())
        for name in names:
            await self.kill(name)

    def list_sessions(self) -> list[SessionState]:
        """List all session states."""
        return [s.state for s in self._sessions.values()]

    def get_waiting_sessions(self) -> list[ManagedSession]:
        """Get sessions that are waiting for input."""
        return [s for s in self._sessions.values() if s.state.is_waiting]
