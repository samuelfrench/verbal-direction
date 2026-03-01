"""Discover running Claude Code sessions and map them to terminal windows."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


@dataclass
class DiscoveredSession:
    """A running Claude Code session discovered on the system."""

    pid: int
    tty: str  # e.g. "/dev/pts/1"
    cwd: str  # working directory
    project_key: str  # e.g. "-home-sam-claude-workspace-coffee-explorer"
    transcript_path: Path | None  # path to .jsonl file
    session_id: str | None  # UUID from transcript
    slug: str  # human-readable name like "sharded-petting-wand"
    label: str  # short name derived from project path

    @property
    def pts_number(self) -> str:
        """Extract pts number like '1' from '/dev/pts/1'."""
        return self.tty.split("/")[-1] if self.tty else ""


def discover_sessions() -> list[DiscoveredSession]:
    """Find all running Claude Code processes and map them to sessions."""
    sessions = []

    # Find claude processes
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,tty,comm"],
            capture_output=True,
            text=True,
        )
    except Exception as e:
        logger.error("Failed to run ps: %s", e)
        return []

    claude_pids = []
    for line in result.stdout.strip().split("\n")[1:]:
        parts = line.split()
        if len(parts) >= 3 and parts[2] == "claude":
            pid = int(parts[0])
            tty = parts[1]
            if tty != "?":
                claude_pids.append((pid, f"/dev/{tty}"))

    for pid, tty in claude_pids:
        session = _build_session(pid, tty)
        if session:
            sessions.append(session)

    return sessions


def _build_session(pid: int, tty: str) -> DiscoveredSession | None:
    """Build a DiscoveredSession from a PID."""
    # Read CWD
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None

    # Derive project key (same format Claude uses)
    project_key = cwd.replace("/", "-")
    if project_key.startswith("-"):
        project_key = project_key  # keep leading dash

    # Find transcript file
    project_dir = CLAUDE_PROJECTS_DIR / project_key
    transcript_path = None
    session_id = None
    slug = ""

    if project_dir.exists():
        # Find the most recently modified .jsonl in this project dir
        jsonl_files = sorted(
            project_dir.glob("*.jsonl"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        for f in jsonl_files:
            # Read the last line to get session info
            info = _read_last_session_info(f)
            if info:
                transcript_path = f
                session_id = info.get("sessionId")
                slug = info.get("slug", "")
                break

    # Derive a short label from the project path
    label = Path(cwd).name or "unknown"

    return DiscoveredSession(
        pid=pid,
        tty=tty,
        cwd=cwd,
        project_key=project_key,
        transcript_path=transcript_path,
        session_id=session_id,
        slug=slug,
        label=label,
    )


def _read_last_session_info(path: Path) -> dict | None:
    """Read the last few lines of a JSONL transcript to extract session info."""
    try:
        # Read last 4KB to find the last complete JSON line
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            read_size = min(size, 4096)
            f.seek(size - read_size)
            data = f.read().decode("utf-8", errors="replace")

        # Find the last complete line
        lines = data.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "sessionId" in obj:
                    return obj
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logger.debug("Failed to read %s: %s", path, e)

    return None
