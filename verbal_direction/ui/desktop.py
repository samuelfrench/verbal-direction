"""PyQt6 desktop dashboard for verbal-direction."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QColor, QPainter, QIcon, QAction
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QLineEdit,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QSplitter,
    QProgressBar,
    QSystemTrayIcon,
    QMenu,
    QSizePolicy,
)

from verbal_direction.core.event_bus import EventBus, Event, EventType
from verbal_direction.core.session_manager import SessionManager
from verbal_direction.core.session_state import SessionStatus

logger = logging.getLogger(__name__)

# Session status -> (color, label)
STATUS_STYLES = {
    SessionStatus.IDLE: ("#6b7280", "Idle"),
    SessionStatus.WORKING: ("#3b82f6", "Working"),
    SessionStatus.WAITING_FOR_INPUT: ("#f59e0b", "Waiting"),
    SessionStatus.WAITING_FOR_PERMISSION: ("#ef4444", "Permission"),
    SessionStatus.PAUSED: ("#8b5cf6", "Paused"),
    SessionStatus.COMPLETED: ("#10b981", "Done"),
    SessionStatus.ERROR: ("#ef4444", "Error"),
}

DARK_THEME = """
QMainWindow, QWidget {
    background-color: #0f1419;
    color: #e6e9ed;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

QLabel {
    color: #e6e9ed;
}

#title-bar {
    background-color: #131920;
    border-bottom: 1px solid #1e2a36;
    padding: 8px 16px;
}

#title-label {
    font-size: 16px;
    font-weight: bold;
    color: #7dd3fc;
}

#subtitle-label {
    font-size: 11px;
    color: #64748b;
}

#sessions-panel {
    background-color: #131920;
    border-right: 1px solid #1e2a36;
    min-width: 280px;
    max-width: 320px;
}

#sessions-header {
    font-size: 12px;
    font-weight: bold;
    color: #64748b;
    text-transform: uppercase;
    padding: 12px 16px 8px 16px;
    letter-spacing: 1px;
}

#output-panel {
    background-color: #0f1419;
}

#output-header {
    font-size: 12px;
    font-weight: bold;
    color: #64748b;
    padding: 12px 16px 8px 16px;
    letter-spacing: 1px;
}

#output-log {
    background-color: #0a0e13;
    color: #c9d1d9;
    border: 1px solid #1e2a36;
    border-radius: 6px;
    padding: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 12px;
    selection-background-color: #264f78;
}

#voice-bar {
    background-color: #131920;
    border-top: 1px solid #1e2a36;
    padding: 10px 16px;
    min-height: 60px;
}

#voice-status-label {
    font-size: 13px;
    font-weight: bold;
}

#voice-detail-label {
    font-size: 11px;
    color: #8b949e;
}

#mic-level {
    min-height: 6px;
    max-height: 6px;
    border-radius: 3px;
    background-color: #1e2a36;
}

#mic-level::chunk {
    background-color: #22c55e;
    border-radius: 3px;
}

QPushButton {
    background-color: #1e2a36;
    color: #e6e9ed;
    border: 1px solid #2d3a47;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #2d3a47;
    border-color: #3d4a57;
}

QPushButton:pressed {
    background-color: #3d4a57;
}

#launch-btn {
    background-color: #1d4ed8;
    border-color: #2563eb;
    color: white;
    font-weight: bold;
    padding: 8px 16px;
}

#launch-btn:hover {
    background-color: #2563eb;
}

QDialog {
    background-color: #131920;
    color: #e6e9ed;
}

QLineEdit {
    background-color: #0a0e13;
    color: #e6e9ed;
    border: 1px solid #2d3a47;
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 13px;
}

QLineEdit:focus {
    border-color: #3b82f6;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #0f1419;
    width: 8px;
}

QScrollBar::handle:vertical {
    background-color: #2d3a47;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #3d4a57;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


class SessionCard(QFrame):
    """A card showing a single session's status."""

    pause_clicked = pyqtSignal(str)
    kill_clicked = pyqtSignal(str)
    card_clicked = pyqtSignal(str)

    def __init__(self, name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._name = name
        self._status = SessionStatus.IDLE
        self._cost = 0.0
        self._question = ""
        self._paused = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(self._card_style())
        self._setup_ui()

    def _card_style(self, selected: bool = False) -> str:
        border_color = "#3b82f6" if selected else "#1e2a36"
        bg = "#1a2332" if selected else "#151d27"
        return f"""
            SessionCard {{
                background-color: {bg};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 12px;
                margin: 4px 8px;
            }}
            SessionCard:hover {{
                background-color: #1a2332;
                border-color: #2d3a47;
            }}
        """

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 10, 12, 10)

        # Top row: name + status badge
        top = QHBoxLayout()
        self._name_label = QLabel(self._name)
        self._name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #e6e9ed;")
        top.addWidget(self._name_label)

        self._status_badge = QLabel("Idle")
        self._update_status_badge()
        top.addWidget(self._status_badge)
        top.addStretch()
        layout.addLayout(top)

        # Question line (hidden by default)
        self._question_label = QLabel()
        self._question_label.setStyleSheet("font-size: 11px; color: #f59e0b; padding: 2px 0;")
        self._question_label.setWordWrap(True)
        self._question_label.hide()
        layout.addWidget(self._question_label)

        # Bottom row: cost + buttons
        bottom = QHBoxLayout()
        self._cost_label = QLabel("$0.00")
        self._cost_label.setStyleSheet("font-size: 11px; color: #64748b;")
        bottom.addWidget(self._cost_label)
        bottom.addStretch()

        self._pause_btn = QPushButton("⏸")
        self._pause_btn.setFixedSize(28, 28)
        self._pause_btn.setToolTip("Pause voice monitoring")
        self._pause_btn.setStyleSheet("font-size: 14px; padding: 0; border-radius: 4px;")
        self._pause_btn.clicked.connect(lambda: self.pause_clicked.emit(self._name))
        bottom.addWidget(self._pause_btn)

        self._kill_btn = QPushButton("✕")
        self._kill_btn.setFixedSize(28, 28)
        self._kill_btn.setToolTip("Kill session")
        self._kill_btn.setStyleSheet(
            "font-size: 14px; padding: 0; border-radius: 4px; color: #ef4444;"
        )
        self._kill_btn.clicked.connect(lambda: self.kill_clicked.emit(self._name))
        bottom.addWidget(self._kill_btn)

        layout.addLayout(bottom)

    def _update_status_badge(self) -> None:
        color, label = STATUS_STYLES.get(self._status, ("#6b7280", "Unknown"))
        self._status_badge.setText(label)
        self._status_badge.setStyleSheet(
            f"font-size: 11px; font-weight: bold; color: {color}; "
            f"background-color: {color}22; border-radius: 4px; padding: 2px 8px;"
        )

    def update_state(
        self,
        status: SessionStatus,
        cost: float = 0.0,
        question: str = "",
        paused: bool = False,
    ) -> None:
        self._status = status
        self._cost = cost
        self._question = question
        self._paused = paused

        self._update_status_badge()
        self._cost_label.setText(f"${cost:.2f}")
        self._pause_btn.setText("▶" if paused else "⏸")
        self._pause_btn.setToolTip("Resume" if paused else "Pause voice monitoring")

        if question:
            self._question_label.setText(f"Q: {question[:120]}")
            self._question_label.show()
        else:
            self._question_label.hide()

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(self._card_style(selected))

    def mousePressEvent(self, event) -> None:
        self.card_clicked.emit(self._name)
        super().mousePressEvent(event)


class LaunchDialog(QDialog):
    """Dialog to launch a new session."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Launch Session")
        self.setMinimumWidth(420)
        self.setStyleSheet("""
            QDialog { background-color: #131920; }
            QLabel { color: #e6e9ed; font-size: 13px; }
        """)

        layout = QFormLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. auth, frontend, api")
        layout.addRow("Session name:", self.name_input)

        dir_row = QHBoxLayout()
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("/path/to/project")
        dir_row.addWidget(self.dir_input)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse)
        dir_row.addWidget(browse_btn)
        layout.addRow("Directory:", dir_row)

        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("e.g. implement JWT authentication")
        layout.addRow("Initial prompt:", self.prompt_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.setStyleSheet("""
            QPushButton { min-width: 80px; }
        """)
        layout.addRow(buttons)

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if d:
            self.dir_input.setText(d)

    def get_values(self) -> tuple[str, str, str]:
        return (
            self.name_input.text().strip(),
            self.dir_input.text().strip(),
            self.prompt_input.text().strip(),
        )


class VDDesktopApp(QMainWindow):
    """Main desktop dashboard window."""

    def __init__(self, event_bus: EventBus, session_manager: SessionManager) -> None:
        super().__init__()
        self._event_bus = event_bus
        self._session_manager = session_manager
        self._session_cards: dict[str, SessionCard] = {}
        self._selected_session: str | None = None
        self._output_lines: list[tuple[str, str]] = []  # (session_name, text)

        self.setWindowTitle("verbal-direction")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)

        self._setup_ui()
        self._setup_tray()
        self._setup_refresh_timer()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # Title bar
        title_bar = QWidget()
        title_bar.setObjectName("title-bar")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(16, 8, 16, 8)

        title = QLabel("verbal-direction")
        title.setObjectName("title-label")
        title_layout.addWidget(title)

        subtitle = QLabel("Voice-Controlled Claude Code Manager")
        subtitle.setObjectName("subtitle-label")
        title_layout.addWidget(subtitle)
        title_layout.addStretch()
        root.addWidget(title_bar)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: sessions panel
        sessions_widget = QWidget()
        sessions_widget.setObjectName("sessions-panel")
        sessions_layout = QVBoxLayout(sessions_widget)
        sessions_layout.setContentsMargins(0, 0, 0, 0)
        sessions_layout.setSpacing(0)

        sessions_header = QLabel("SESSIONS")
        sessions_header.setObjectName("sessions-header")
        sessions_layout.addWidget(sessions_header)

        # Scrollable session cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._cards_container = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_container)
        self._cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._cards_layout.setSpacing(4)
        self._cards_layout.setContentsMargins(4, 4, 4, 4)

        self._empty_label = QLabel("No active sessions")
        self._empty_label.setStyleSheet("color: #4b5563; padding: 20px; font-style: italic;")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cards_layout.addWidget(self._empty_label)

        scroll.setWidget(self._cards_container)
        sessions_layout.addWidget(scroll, stretch=1)

        # Launch button
        launch_btn = QPushButton("+ Launch Session")
        launch_btn.setObjectName("launch-btn")
        launch_btn.clicked.connect(self._on_launch)
        launch_btn.setFixedHeight(40)
        btn_container = QWidget()
        btn_layout = QHBoxLayout(btn_container)
        btn_layout.setContentsMargins(12, 8, 12, 12)
        btn_layout.addWidget(launch_btn)
        sessions_layout.addWidget(btn_container)

        splitter.addWidget(sessions_widget)

        # Right: output panel
        output_widget = QWidget()
        output_widget.setObjectName("output-panel")
        output_layout = QVBoxLayout(output_widget)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(0)

        output_header = QLabel("LIVE OUTPUT")
        output_header.setObjectName("output-header")
        output_layout.addWidget(output_header)

        self._output_log = QTextEdit()
        self._output_log.setObjectName("output-log")
        self._output_log.setReadOnly(True)
        output_log_container = QWidget()
        olc_layout = QVBoxLayout(output_log_container)
        olc_layout.setContentsMargins(12, 4, 12, 12)
        olc_layout.addWidget(self._output_log)
        output_layout.addWidget(output_log_container, stretch=1)

        splitter.addWidget(output_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 800])

        root.addWidget(splitter, stretch=1)

        # Bottom: voice status bar
        voice_bar = QWidget()
        voice_bar.setObjectName("voice-bar")
        voice_layout = QVBoxLayout(voice_bar)
        voice_layout.setSpacing(6)
        voice_layout.setContentsMargins(16, 10, 16, 10)

        voice_top = QHBoxLayout()
        self._voice_icon = QLabel("🎤")
        self._voice_icon.setStyleSheet("font-size: 18px;")
        voice_top.addWidget(self._voice_icon)

        self._voice_status = QLabel("Listening")
        self._voice_status.setObjectName("voice-status-label")
        self._voice_status.setStyleSheet("color: #22c55e; font-size: 13px; font-weight: bold;")
        voice_top.addWidget(self._voice_status)

        self._mic_level = QProgressBar()
        self._mic_level.setObjectName("mic-level")
        self._mic_level.setRange(0, 100)
        self._mic_level.setValue(0)
        self._mic_level.setTextVisible(False)
        self._mic_level.setFixedHeight(6)
        self._mic_level.setMinimumWidth(120)
        self._mic_level.setMaximumWidth(200)
        voice_top.addWidget(self._mic_level)

        voice_top.addStretch()

        self._pending_label = QLabel("Pending: 0")
        self._pending_label.setStyleSheet("color: #64748b; font-size: 12px;")
        voice_top.addWidget(self._pending_label)
        voice_layout.addLayout(voice_top)

        voice_bottom = QHBoxLayout()
        self._last_heard_label = QLabel("Last heard: —")
        self._last_heard_label.setObjectName("voice-detail-label")
        voice_bottom.addWidget(self._last_heard_label)

        voice_bottom.addStretch()

        self._tts_queue_label = QLabel("TTS queue: empty")
        self._tts_queue_label.setObjectName("voice-detail-label")
        voice_bottom.addWidget(self._tts_queue_label)
        voice_layout.addLayout(voice_bottom)

        root.addWidget(voice_bar)

    def _setup_tray(self) -> None:
        """Setup system tray icon."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        self._tray = QSystemTrayIcon(self)
        # Use a simple built-in icon
        self._tray.setIcon(self.style().standardIcon(
            self.style().StandardPixmap.SP_ComputerIcon
        ))
        self._tray.setToolTip("verbal-direction")

        tray_menu = QMenu()
        show_action = QAction("Show", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(QApplication.instance().quit)
        tray_menu.addAction(quit_action)

        self._tray.setContextMenu(tray_menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()

    def closeEvent(self, event) -> None:
        """Minimize to tray instead of closing."""
        if hasattr(self, "_tray") and self._tray.isVisible():
            self.hide()
            event.ignore()
        else:
            event.accept()

    def _setup_refresh_timer(self) -> None:
        """Poll session states every 500ms."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(500)

    def _refresh(self) -> None:
        """Refresh session cards and voice status."""
        sessions = self._session_manager.list_sessions()
        current_names = {s.name for s in sessions}
        existing_names = set(self._session_cards.keys())

        # Add new sessions
        for state in sessions:
            if state.name not in existing_names:
                self._add_session_card(state.name)

        # Remove gone sessions
        for name in existing_names - current_names:
            self._remove_session_card(name)

        # Show/hide empty label
        self._empty_label.setVisible(len(sessions) == 0)

        # Update cards
        for state in sessions:
            card = self._session_cards.get(state.name)
            if card:
                q_text = state.pending_question.text if state.pending_question else ""
                card.update_state(
                    status=state.status,
                    cost=state.total_cost_usd,
                    question=q_text,
                    paused=state.voice_paused,
                )

        # Update pending count
        waiting = [s for s in sessions if s.is_waiting]
        self._pending_label.setText(f"Pending: {len(waiting)}")

    def _add_session_card(self, name: str) -> None:
        card = SessionCard(name)
        card.pause_clicked.connect(self._on_pause)
        card.kill_clicked.connect(self._on_kill)
        card.card_clicked.connect(self._on_card_click)
        self._session_cards[name] = card
        self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)

    def _remove_session_card(self, name: str) -> None:
        card = self._session_cards.pop(name, None)
        if card:
            self._cards_layout.removeWidget(card)
            card.deleteLater()

    def _on_card_click(self, name: str) -> None:
        self._selected_session = name
        for n, card in self._session_cards.items():
            card.set_selected(n == name)

    def _on_launch(self) -> None:
        dialog = LaunchDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            name, directory, prompt = dialog.get_values()
            if not name or not directory:
                return
            asyncio.ensure_future(self._do_launch(name, directory, prompt or None))

    async def _do_launch(self, name: str, directory: str, prompt: str | None) -> None:
        try:
            await self._session_manager.launch(name, directory, prompt)
            self._append_output("system", f"Launched session '{name}' in {directory}")
        except Exception as e:
            self._append_output("system", f"Failed to launch '{name}': {e}")

    def _on_pause(self, name: str) -> None:
        session = self._session_manager.get(name)
        if session:
            session.state.voice_paused = not session.state.voice_paused
            state = "paused" if session.state.voice_paused else "resumed"
            self._append_output("system", f"Voice {state} for '{name}'")

    def _on_kill(self, name: str) -> None:
        asyncio.ensure_future(self._do_kill(name))

    async def _do_kill(self, name: str) -> None:
        await self._session_manager.kill(name)
        self._append_output("system", f"Killed session '{name}'")

    def append_output(self, session_name: str, text: str) -> None:
        """Public method to append output from event bus."""
        self._append_output(session_name, text)

    def _append_output(self, session_name: str, text: str) -> None:
        """Append a line to the output log."""
        SESSION_COLORS = [
            "#7dd3fc", "#a78bfa", "#34d399", "#fbbf24", "#f87171",
            "#fb923c", "#c084fc", "#22d3ee", "#a3e635", "#f472b6",
        ]

        if session_name == "system":
            color = "#64748b"
        else:
            idx = hash(session_name) % len(SESSION_COLORS)
            color = SESSION_COLORS[idx]

        timestamp = time.strftime("%H:%M:%S")
        # Escape HTML in text
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = (
            f'<span style="color: #4b5563;">{timestamp}</span> '
            f'<span style="color: {color}; font-weight: bold;">[{session_name}]</span> '
            f'<span style="color: #c9d1d9;">{safe_text}</span>'
        )
        self._output_log.append(html)

        # Auto-scroll to bottom
        scrollbar = self._output_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_voice_listening(self, listening: bool) -> None:
        if listening:
            self._voice_status.setText("Listening")
            self._voice_status.setStyleSheet("color: #22c55e; font-size: 13px; font-weight: bold;")
            self._voice_icon.setText("🎤")
        else:
            self._voice_status.setText("Stopped")
            self._voice_status.setStyleSheet("color: #ef4444; font-size: 13px; font-weight: bold;")
            self._voice_icon.setText("🔇")

    def set_last_heard(self, text: str) -> None:
        self._last_heard_label.setText(f'Last heard: "{text}"')

    def set_mic_level(self, level: int) -> None:
        self._mic_level.setValue(min(100, max(0, level)))


class GUIEventBridge:
    """Bridges EventBus events to the GUI."""

    def __init__(self, event_bus: EventBus, gui: VDDesktopApp) -> None:
        self._event_bus = event_bus
        self._gui = gui
        self._queue = event_bus.subscribe_all()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                self._handle_event(event)
            except Exception as e:
                logger.error("GUI bridge error: %s", e)

    def _handle_event(self, event: Event) -> None:
        if event.type == EventType.SESSION_OUTPUT:
            text = event.data.get("text", "") if event.data else ""
            if text:
                self._gui.append_output(event.session_name, text[:500])

        elif event.type == EventType.SESSION_QUESTION:
            text = event.data.get("text", "") if event.data else ""
            self._gui.append_output(event.session_name, f"❓ {text}")

        elif event.type == EventType.SESSION_ERROR:
            text = event.data.get("text", "") if event.data else ""
            self._gui.append_output(event.session_name, f"❌ {text}")

        elif event.type == EventType.SESSION_PERMISSION:
            desc = event.data.get("description", "") if event.data else ""
            self._gui.append_output(event.session_name, f"🔐 {desc}")

        elif event.type == EventType.VOICE_TRANSCRIPTION:
            text = event.data.get("text", "") if event.data else ""
            if text:
                self._gui.set_last_heard(text)
                self._gui.append_output("voice", f"Heard: {text}")

        elif event.type == EventType.VOICE_ROUTED:
            target = event.data.get("target_session", "") if event.data else ""
            text = event.data.get("text", "") if event.data else ""
            self._gui.append_output("voice", f"→ {target}: {text}")

        elif event.type == EventType.SESSION_LAUNCHED:
            self._gui.append_output("system", f"Session '{event.session_name}' launched")

        elif event.type == EventType.SESSION_KILLED:
            self._gui.append_output("system", f"Session '{event.session_name}' killed")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._event_bus.unsubscribe(self._queue)


def run_desktop_app(
    event_bus: EventBus | None = None,
    session_manager: SessionManager | None = None,
) -> None:
    """Launch the desktop GUI application."""
    import qasync

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("verbal-direction")
    app.setStyleSheet(DARK_THEME)

    if event_bus is None:
        event_bus = EventBus()
    if session_manager is None:
        session_manager = SessionManager(event_bus)

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = VDDesktopApp(event_bus, session_manager)
    window.show()

    # Bridge events to GUI
    bridge = GUIEventBridge(event_bus, window)

    async def start_bridge():
        await bridge.start()

    loop.create_task(start_bridge())

    with loop:
        loop.run_forever()
