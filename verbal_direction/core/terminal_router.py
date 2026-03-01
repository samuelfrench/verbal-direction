"""Terminal router — inject typed responses into terminal sessions via PTY master."""

from __future__ import annotations

import logging
import os
import subprocess

from verbal_direction.core.process_discovery import DiscoveredSession

logger = logging.getLogger(__name__)


def inject_text(session: DiscoveredSession, text: str) -> bool:
    """Inject text into a terminal session by writing to its PTY master.

    Finds the terminal emulator's PTY master fd for this session's PTS
    and writes to it, simulating keyboard input. This works on all
    kernel versions (unlike TIOCSTI which is disabled on kernel 6.2+).
    """
    tty = session.tty
    if not tty:
        logger.error("No TTY for session %s", session.label)
        return False

    pts_num = session.pts_number
    if not pts_num:
        logger.error("Cannot determine PTS number from %s", tty)
        return False

    # Find the terminal emulator's PTY master fd for this PTS
    master_fd_path = _find_pty_master(int(pts_num))
    if not master_fd_path:
        logger.debug("PTY master not found for pts/%s, will try xdotool", pts_num)
        return False

    full_text = text + "\n"

    try:
        fd = os.open(master_fd_path, os.O_WRONLY | os.O_NOCTTY)
        try:
            os.write(fd, full_text.encode())
        finally:
            os.close(fd)

        logger.info("Injected into pts/%s via PTY master: %s", pts_num, text[:50])
        return True

    except PermissionError:
        logger.error("Permission denied writing to PTY master %s", master_fd_path)
        return False
    except OSError as e:
        logger.error("Failed to write to PTY master %s: %s", master_fd_path, e)
        return False


def _find_pty_master(target_pts: int) -> str | None:
    """Find the PTY master fd path in a terminal emulator process.

    Searches all processes that hold /dev/ptmx fds and checks their
    fdinfo tty-index to find the master side of the target PTS.
    """
    # Find terminal emulator processes that hold ptmx fds
    # Common terminal emulators: gnome-terminal-server, konsole, xterm, etc.
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

            # Check the tty-index in fdinfo
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


def inject_text_xdotool(session: DiscoveredSession, text: str) -> bool:
    """Fallback: use xdotool to type into the terminal window.

    Finds the terminal window, activates it, types the text,
    then restores the previously active window.
    """
    window_id = _find_window_for_session(session)

    if not window_id:
        window_id = _find_window_by_title(session)

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

        # Activate target window
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id])

        # Type the text + Enter
        subprocess.run(["xdotool", "type", "--clearmodifiers", "--delay", "5", text])
        subprocess.run(["xdotool", "key", "--clearmodifiers", "Return"])

        # Restore original window
        if original_window and original_window != window_id:
            subprocess.run(["xdotool", "windowactivate", original_window])

        logger.info("Injected via xdotool into window %s: %s", window_id, text[:50])
        return True

    except Exception as e:
        logger.error("xdotool typing failed: %s", e)
        return False


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
    for search_term in [session.label, session.cwd, "Claude Code"]:
        result = subprocess.run(
            ["xdotool", "search", "--name", search_term],
            capture_output=True, text=True,
        )
        windows = [w.strip() for w in result.stdout.strip().split("\n") if w.strip()]
        if windows:
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
