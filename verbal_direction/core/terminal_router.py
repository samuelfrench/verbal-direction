"""Terminal router — inject typed responses into terminal windows via PTY."""

from __future__ import annotations

import fcntl
import logging
import os
import struct
import termios

from verbal_direction.core.process_discovery import DiscoveredSession

logger = logging.getLogger(__name__)


def inject_text(session: DiscoveredSession, text: str) -> bool:
    """Type text into a terminal by writing to its PTY device.

    Uses TIOCSTI ioctl to simulate keyboard input character by character.
    Falls back to direct PTY write if TIOCSTI is not available.

    Args:
        session: The discovered session with TTY info.
        text: Text to type, followed by Enter.

    Returns:
        True if successful.
    """
    tty = session.tty
    if not tty:
        logger.error("No TTY for session %s", session.label)
        return False

    # Add newline to submit the response
    full_text = text + "\n"

    try:
        fd = os.open(tty, os.O_WRONLY)
        try:
            # Try TIOCSTI (Terminal I/O Control - Simulate Terminal Input)
            # This pushes characters into the terminal's input queue
            for char in full_text:
                try:
                    fcntl.ioctl(fd, termios.TIOCSTI, struct.pack("B", ord(char)))
                except OSError:
                    # TIOCSTI might be disabled (kernel 6.2+ security)
                    # Fall back to direct write
                    logger.debug("TIOCSTI not available, falling back to direct write")
                    os.write(fd, full_text.encode())
                    logger.info("Injected into %s via direct write: %s", tty, text[:50])
                    return True
        finally:
            os.close(fd)

        logger.info("Injected into %s via TIOCSTI: %s", tty, text[:50])
        return True

    except PermissionError:
        logger.error("Permission denied writing to %s", tty)
        return False
    except OSError as e:
        logger.error("Failed to write to %s: %s", tty, e)
        return False


def inject_text_xdotool(session: DiscoveredSession, text: str) -> bool:
    """Fallback: use xdotool to type into the terminal window.

    This activates the terminal window, switches to the correct tab,
    and types the text. Less precise than PTY injection but works
    when PTY write is blocked.
    """
    import subprocess

    # Find the terminal window containing this process
    try:
        # Walk up the process tree to find the terminal emulator
        pid = session.pid
        for _ in range(5):
            ppid = _get_ppid(pid)
            if ppid is None or ppid <= 1:
                break

            result = subprocess.run(
                ["xdotool", "search", "--pid", str(ppid)],
                capture_output=True,
                text=True,
            )
            windows = result.stdout.strip().split("\n")
            if windows and windows[0]:
                window_id = windows[0]

                # Activate window
                subprocess.run(["xdotool", "windowactivate", window_id])

                # Type the text + Enter
                subprocess.run(["xdotool", "type", "--delay", "10", text])
                subprocess.run(["xdotool", "key", "Return"])

                logger.info("Injected via xdotool into window %s: %s", window_id, text[:50])
                return True

            pid = ppid

    except Exception as e:
        logger.error("xdotool injection failed: %s", e)

    return False


def _get_ppid(pid: int) -> int | None:
    """Get parent PID from /proc."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
            return int(fields[3])
    except (OSError, IndexError, ValueError):
        return None
