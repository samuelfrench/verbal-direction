"""Terminal router — inject typed responses into terminal sessions via xdotool."""

from __future__ import annotations

import logging
import os
import subprocess
import time

from verbal_direction.core.process_discovery import DiscoveredSession

logger = logging.getLogger(__name__)

# Cache window IDs to avoid slow xdotool searches every time
_window_cache: dict[int, str] = {}  # pid -> window_id


def inject_text_xdotool(session: DiscoveredSession, text: str) -> bool:
    """Inject text by emulating keyboard input via xdotool.

    Finds the terminal window, activates it, types the text,
    then restores the previously active window.
    """
    # Check cache first
    window_id = _window_cache.get(session.pid)

    if not window_id or not _window_exists(window_id):
        # Cache miss or stale — search by title first (fast), then PID walk (slow)
        window_id = _find_window_by_title(session)
        if not window_id:
            window_id = _find_window_for_session(session)
        if window_id:
            _window_cache[session.pid] = window_id
            logger.info("Cached window %s for session %s", window_id, session.label)

    if not window_id:
        logger.error("Could not find terminal window for %s (PID=%s)", session.label, session.pid)
        return False

    try:
        # Save current active window
        result = subprocess.run(
            ["xdotool", "getactivewindow"],
            capture_output=True, text=True,
        )
        original_window = result.stdout.strip() if result.returncode == 0 else None

        # Activate target window so keystrokes go to it
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id],
                       capture_output=True)
        time.sleep(0.05)

        # Type the text + Enter (simulates real keyboard input)
        subprocess.run(["xdotool", "type", "--clearmodifiers", "--delay", "8", text],
                       capture_output=True)
        time.sleep(0.15)  # let Claude Code process typed chars before Enter
        subprocess.run(["xdotool", "key", "--clearmodifiers", "Return"],
                       capture_output=True)

        # Restore original window
        if original_window and original_window != window_id:
            time.sleep(0.05)
            subprocess.run(["xdotool", "windowactivate", original_window],
                           capture_output=True)

        logger.info("Injected via xdotool into window %s: %s", window_id, text[:50])
        return True

    except Exception as e:
        logger.error("xdotool typing failed: %s", e)
        return False


def _window_exists(window_id: str) -> bool:
    """Check if an X window still exists."""
    result = subprocess.run(
        ["xdotool", "getwindowname", window_id],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def inject_text(session: DiscoveredSession, text: str) -> bool:
    """Fallback: inject text via PTY master write (unreliable with TUI apps)."""
    tty = session.tty
    if not tty:
        return False

    pts_num = session.pts_number
    if not pts_num:
        return False

    master_fd_path = _find_pty_master(int(pts_num))
    if not master_fd_path:
        return False

    try:
        fd = os.open(master_fd_path, os.O_WRONLY | os.O_NOCTTY)
        try:
            os.write(fd, (text + "\n").encode())
        finally:
            os.close(fd)
        logger.info("Injected into pts/%s via PTY master: %s", pts_num, text[:50])
        return True
    except OSError as e:
        logger.error("PTY master write failed: %s", e)
        return False


def _find_pty_master(target_pts: int) -> str | None:
    """Find the PTY master fd path in a terminal emulator process."""
    terminal_pids = _find_terminal_emulator_pids()

    for pid in terminal_pids:
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue

        for fd_name in fds:
            fd_path = f"{fd_dir}/{fd_name}"
            try:
                link = os.readlink(fd_path)
            except OSError:
                continue

            if "ptmx" not in link:
                continue

            try:
                with open(f"/proc/{pid}/fdinfo/{fd_name}") as f:
                    for line in f:
                        if line.startswith("tty-index:"):
                            tty_index = int(line.split(":")[1].strip())
                            if tty_index == target_pts:
                                return fd_path
            except (OSError, ValueError):
                continue

    return None


def _find_terminal_emulator_pids() -> list[int]:
    """Find PIDs of terminal emulator processes."""
    terminal_names = {
        "gnome-terminal-", "gnome-terminal-server",
        "konsole", "xterm", "xfce4-terminal",
        "mate-terminal", "tilix", "terminator",
        "alacritty", "kitty", "wezterm-gui",
    }

    pids = []
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/comm") as f:
                    comm = f.read().strip()
                    if any(comm.startswith(name) for name in terminal_names):
                        pids.append(int(entry))
            except OSError:
                continue
    except OSError:
        pass

    return pids


def _find_window_for_session(session: DiscoveredSession) -> str | None:
    """Find X window by walking up the process tree."""
    pid = session.pid
    for _ in range(5):
        ppid = _get_ppid(pid)
        if ppid is None or ppid <= 1:
            break

        result = subprocess.run(
            ["xdotool", "search", "--pid", str(ppid)],
            capture_output=True, text=True,
        )
        windows = [w.strip() for w in result.stdout.strip().split("\n") if w.strip()]
        if windows:
            return windows[0]

        pid = ppid

    return None


def _find_window_by_title(session: DiscoveredSession) -> str | None:
    """Find X window by searching terminal titles."""
    # Search "Claude Code" first — terminal title when Claude is running
    for search_term in ["Claude Code", session.label, session.cwd]:
        result = subprocess.run(
            ["xdotool", "search", "--name", search_term],
            capture_output=True, text=True,
        )
        windows = [w.strip() for w in result.stdout.strip().split("\n") if w.strip()]
        if windows:
            # If multiple matches, prefer the one with "Claude Code" in the name
            for w in windows:
                try:
                    name_result = subprocess.run(
                        ["xdotool", "getwindowname", w],
                        capture_output=True, text=True,
                    )
                    if "Claude Code" in name_result.stdout:
                        return w
                except Exception:
                    pass
            return windows[0]

    return None


def _get_ppid(pid: int) -> int | None:
    """Get parent PID from /proc."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
            return int(fields[3])
    except (OSError, IndexError, ValueError):
        return None
