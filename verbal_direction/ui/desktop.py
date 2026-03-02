"""PyQt6 desktop dashboard for verbal-direction."""

from __future__ import annotations

import asyncio
import logging
import sys
import time

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QAction
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
    QFrame,
    QSplitter,
    QProgressBar,
    QSystemTrayIcon,
    QMenu,
    QComboBox,
    QGroupBox,
)

from verbal_direction.core.event_bus import EventBus, EventType

logger = logging.getLogger(__name__)

DARK_THEME = """
QMainWindow, QWidget {
    background-color: #0f1419;
    color: #e6e9ed;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
QLabel { color: #e6e9ed; }

#title-bar {
    background-color: #131920;
    border-bottom: 1px solid #1e2a36;
    padding: 8px 16px;
}
#title-label { font-size: 16px; font-weight: bold; color: #7dd3fc; }
#subtitle-label { font-size: 11px; color: #64748b; }

#sessions-panel {
    background-color: #131920;
    border-right: 1px solid #1e2a36;
    min-width: 280px;
    max-width: 360px;
}
#sessions-header {
    font-size: 12px; font-weight: bold; color: #64748b;
    padding: 12px 16px 8px 16px; letter-spacing: 1px;
}

#output-panel { background-color: #0f1419; }
#output-header {
    font-size: 12px; font-weight: bold; color: #64748b;
    padding: 12px 16px 8px 16px; letter-spacing: 1px;
}
#output-log {
    background-color: #0a0e13; color: #c9d1d9;
    border: 1px solid #1e2a36; border-radius: 6px; padding: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 12px;
    selection-background-color: #264f78;
}

#voice-bar {
    background-color: #131920;
    border-top: 1px solid #1e2a36;
    padding: 10px 16px; min-height: 60px;
}
#voice-status-label { font-size: 13px; font-weight: bold; }
#voice-detail-label { font-size: 11px; color: #8b949e; }
#mic-level {
    min-height: 6px; max-height: 6px; border-radius: 3px;
    background-color: #1e2a36;
}
#mic-level::chunk { background-color: #22c55e; border-radius: 3px; }

QPushButton {
    background-color: #1e2a36; color: #e6e9ed;
    border: 1px solid #2d3a47; border-radius: 6px;
    padding: 6px 14px; font-size: 12px; font-weight: 500;
}
QPushButton:hover { background-color: #2d3a47; border-color: #3d4a57; }
QPushButton:pressed { background-color: #3d4a57; }

QComboBox {
    background-color: #0a0e13; color: #e6e9ed;
    border: 1px solid #2d3a47; border-radius: 4px;
    padding: 5px 10px; font-size: 12px; min-height: 28px;
}
QComboBox:hover { border-color: #3d4a57; }
QComboBox::drop-down {
    border: none; width: 24px;
}
QComboBox::down-arrow { image: none; border: none; }
QComboBox QAbstractItemView {
    background-color: #131920; color: #e6e9ed;
    border: 1px solid #2d3a47; selection-background-color: #1e2a36;
    padding: 4px;
}

QGroupBox {
    color: #64748b; font-size: 11px; font-weight: bold;
    border: 1px solid #1e2a36; border-radius: 6px;
    margin-top: 8px; padding-top: 16px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 12px; padding: 0 4px;
}

QScrollArea { border: none; background-color: transparent; }
QScrollBar:vertical { background-color: #0f1419; width: 8px; }
QScrollBar::handle:vertical {
    background-color: #2d3a47; border-radius: 4px; min-height: 30px;
}
QScrollBar::handle:vertical:hover { background-color: #3d4a57; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
"""


class SessionCard(QFrame):
    """A card showing a discovered Claude session."""

    card_clicked = pyqtSignal(str)

    def __init__(self, label: str, pid: int, tty: str, cwd: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label = label
        self._pid = pid
        self._tty = tty
        self._cwd = cwd
        self._has_question = False

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(self._card_style())
        self._setup_ui()

    def _card_style(self, selected: bool = False) -> str:
        border = "#3b82f6" if selected else "#1e2a36"
        bg = "#1a2332" if selected else "#151d27"
        return f"""
            SessionCard {{
                background-color: {bg}; border: 1px solid {border};
                border-radius: 8px; padding: 12px; margin: 4px 8px;
            }}
            SessionCard:hover {{ background-color: #1a2332; border-color: #2d3a47; }}
        """

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(12, 10, 12, 10)

        # Top: name + status
        top = QHBoxLayout()
        self._name_label = QLabel(self._label)
        self._name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #e6e9ed;")
        top.addWidget(self._name_label)

        self._status_badge = QLabel("Active")
        self._status_badge.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #22c55e; "
            "background-color: #22c55e22; border-radius: 4px; padding: 2px 8px;"
        )
        top.addWidget(self._status_badge)
        top.addStretch()
        layout.addLayout(top)

        # Details
        details = QLabel(f"PID {self._pid}  ·  {self._tty}")
        details.setStyleSheet("font-size: 11px; color: #64748b;")
        layout.addWidget(details)

        cwd_label = QLabel(self._cwd)
        cwd_label.setStyleSheet("font-size: 11px; color: #4b5563;")
        cwd_label.setWordWrap(True)
        layout.addWidget(cwd_label)

        # Question (hidden by default)
        self._question_label = QLabel()
        self._question_label.setStyleSheet("font-size: 11px; color: #f59e0b; padding: 4px 0 0 0;")
        self._question_label.setWordWrap(True)
        self._question_label.hide()
        layout.addWidget(self._question_label)

    def set_question(self, text: str) -> None:
        if text:
            self._question_label.setText(f"Q: {text[:150]}")
            self._question_label.show()
            self._has_question = True
            self._status_badge.setText("Waiting")
            self._status_badge.setStyleSheet(
                "font-size: 11px; font-weight: bold; color: #f59e0b; "
                "background-color: #f59e0b22; border-radius: 4px; padding: 2px 8px;"
            )
        else:
            self._question_label.hide()
            self._has_question = False
            self._status_badge.setText("Active")
            self._status_badge.setStyleSheet(
                "font-size: 11px; font-weight: bold; color: #22c55e; "
                "background-color: #22c55e22; border-radius: 4px; padding: 2px 8px;"
            )

    def set_selected(self, selected: bool) -> None:
        self.setStyleSheet(self._card_style(selected))

    def mousePressEvent(self, event) -> None:
        self.card_clicked.emit(self._label)
        super().mousePressEvent(event)


class AudioSettingsPanel(QWidget):
    """Audio device selection and TTS settings panel."""

    device_changed = pyqtSignal(str, object)  # kind, device_index_or_none
    tts_mode_changed = pyqtSignal(str)  # "questions" or "all"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 8)
        layout.setSpacing(8)

        # TTS mode
        tts_group = QGroupBox("VOICE OUTPUT")
        tts_layout = QVBoxLayout(tts_group)
        tts_layout.setSpacing(6)

        tts_label = QLabel("Read aloud")
        tts_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #94a3b8;")
        tts_layout.addWidget(tts_label)

        self._tts_mode_combo = QComboBox()
        self._tts_mode_combo.addItem("Questions only", "questions")
        self._tts_mode_combo.addItem("All messages", "all")
        self._tts_mode_combo.currentIndexChanged.connect(self._on_tts_mode_changed)
        tts_layout.addWidget(self._tts_mode_combo)

        layout.addWidget(tts_group)

        # Audio devices
        group = QGroupBox("AUDIO DEVICES")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(8)

        # Input device
        in_label = QLabel("Microphone")
        in_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #94a3b8;")
        group_layout.addWidget(in_label)

        self._input_combo = QComboBox()
        self._input_combo.currentIndexChanged.connect(self._on_input_changed)
        group_layout.addWidget(self._input_combo)

        # Output device
        out_label = QLabel("Speaker")
        out_label.setStyleSheet("font-size: 12px; font-weight: bold; color: #94a3b8;")
        group_layout.addWidget(out_label)

        self._output_combo = QComboBox()
        self._output_combo.currentIndexChanged.connect(self._on_output_changed)
        group_layout.addWidget(self._output_combo)

        # Refresh button
        refresh_btn = QPushButton("Refresh Devices")
        refresh_btn.setStyleSheet("padding: 4px 10px; font-size: 11px;")
        refresh_btn.clicked.connect(self.refresh_devices)
        group_layout.addWidget(refresh_btn)

        layout.addWidget(group)
        self.refresh_devices()

    def refresh_devices(self) -> None:
        """Reload audio device list."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
        except Exception:
            return

        self._input_combo.blockSignals(True)
        self._output_combo.blockSignals(True)

        self._input_combo.clear()
        self._output_combo.clear()

        self._input_combo.addItem("System Default", None)
        self._output_combo.addItem("System Default", None)

        for i, dev in enumerate(devices):
            name = dev["name"]
            if dev["max_input_channels"] > 0:
                self._input_combo.addItem(f"{name} ({dev['max_input_channels']}ch)", i)
            if dev["max_output_channels"] > 0:
                self._output_combo.addItem(f"{name} ({dev['max_output_channels']}ch)", i)

        self._input_combo.blockSignals(False)
        self._output_combo.blockSignals(False)

    def _on_tts_mode_changed(self, index: int) -> None:
        mode = self._tts_mode_combo.currentData()
        self.tts_mode_changed.emit(mode)

    def _on_input_changed(self, index: int) -> None:
        device = self._input_combo.currentData()
        self.device_changed.emit("input", device)

    def _on_output_changed(self, index: int) -> None:
        device = self._output_combo.currentData()
        self.device_changed.emit("output", device)


class VDDesktopApp(QMainWindow):
    """Main desktop dashboard window."""

    def __init__(self) -> None:
        super().__init__()
        self._session_cards: dict[str, SessionCard] = {}
        self._selected_session: str | None = None
        self._pending_questions: dict[str, str] = {}

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

        # Left: sessions + audio panel
        left_widget = QWidget()
        left_widget.setObjectName("sessions-panel")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        sessions_header = QLabel("SESSIONS")
        sessions_header.setObjectName("sessions-header")
        left_layout.addWidget(sessions_header)

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
        left_layout.addWidget(scroll, stretch=1)

        # Audio settings panel
        self._audio_panel = AudioSettingsPanel()
        self._audio_panel.device_changed.connect(self._on_device_changed)
        self._audio_panel.tts_mode_changed.connect(self._on_tts_mode_changed)
        left_layout.addWidget(self._audio_panel)

        splitter.addWidget(left_widget)

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
        olc = QWidget()
        olc_layout = QVBoxLayout(olc)
        olc_layout.setContentsMargins(12, 4, 12, 12)
        olc_layout.addWidget(self._output_log)
        output_layout.addWidget(olc, stretch=1)

        splitter.addWidget(output_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 780])

        root.addWidget(splitter, stretch=1)

        # Bottom: voice status bar
        voice_bar = QWidget()
        voice_bar.setObjectName("voice-bar")
        vbl = QVBoxLayout(voice_bar)
        vbl.setSpacing(6)
        vbl.setContentsMargins(16, 10, 16, 10)

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
        vbl.addLayout(voice_top)

        voice_bottom = QHBoxLayout()
        self._last_heard_label = QLabel("Last heard: —")
        self._last_heard_label.setObjectName("voice-detail-label")
        voice_bottom.addWidget(self._last_heard_label)
        voice_bottom.addStretch()
        self._tts_queue_label = QLabel("TTS queue: empty")
        self._tts_queue_label.setObjectName("voice-detail-label")
        voice_bottom.addWidget(self._tts_queue_label)
        vbl.addLayout(voice_bottom)

        root.addWidget(voice_bar)

    def _setup_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        self._tray = QSystemTrayIcon(self)
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
        if hasattr(self, "_tray") and self._tray.isVisible():
            self.hide()
            event.ignore()
        else:
            event.accept()

    def _setup_refresh_timer(self) -> None:
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_sessions)
        self._timer.start(2000)
        # Initial refresh
        QTimer.singleShot(100, self._refresh_sessions)

    def _refresh_sessions(self) -> None:
        """Discover sessions and update cards."""
        import os
        from verbal_direction.core.process_discovery import discover_sessions

        own_tty = os.ttyname(0) if os.isatty(0) else None
        sessions = [s for s in discover_sessions() if s.tty != own_tty]
        current_labels = {s.label for s in sessions}
        existing_labels = set(self._session_cards.keys())

        # Add new
        for s in sessions:
            if s.label not in existing_labels:
                self._add_session_card(s.label, s.pid, s.tty, s.cwd)

        # Remove gone
        for label in existing_labels - current_labels:
            self._remove_session_card(label)

        self._empty_label.setVisible(len(sessions) == 0)
        self._pending_label.setText(f"Pending: {len(self._pending_questions)}")

    def _add_session_card(self, label: str, pid: int, tty: str, cwd: str) -> None:
        card = SessionCard(label, pid, tty, cwd)
        card.card_clicked.connect(self._on_card_click)
        self._session_cards[label] = card
        self._cards_layout.insertWidget(self._cards_layout.count() - 1, card)

    def _remove_session_card(self, label: str) -> None:
        card = self._session_cards.pop(label, None)
        if card:
            self._cards_layout.removeWidget(card)
            card.deleteLater()

    def _on_card_click(self, label: str) -> None:
        self._selected_session = label
        for n, card in self._session_cards.items():
            card.set_selected(n == label)
        self._append_output("system", f"Default target: {label}")

        # Update voice router's default target
        if hasattr(self, "_voice_router") and self._voice_router:
            self._voice_router.set_default_target(label)

    def set_voice_router(self, voice_router) -> None:
        """Set the voice router so GUI can control routing."""
        self._voice_router = voice_router

    def _on_tts_mode_changed(self, mode: str) -> None:
        self._append_output("system", f"TTS mode: {mode}")
        if hasattr(self, "_voice_router") and self._voice_router:
            self._voice_router.set_tts_mode(mode)

    def _on_device_changed(self, kind: str, device_index) -> None:
        try:
            import sounddevice as sd
            dev_name = "System Default"
            if device_index is not None:
                dev_name = sd.query_devices(device_index)["name"]
        except Exception:
            dev_name = str(device_index)

        self._append_output("system", f"Audio {kind} → {dev_name}")

        # Update the live audio manager if we have one
        if hasattr(self, "_audio_manager") and self._audio_manager:
            if kind == "input":
                self._audio_manager.input_device = device_index
                self._append_output("system", "Mic change takes effect on next voice restart")
            elif kind == "output":
                self._audio_manager.output_device = device_index
                self._append_output("system", "Speaker change takes effect on next TTS output")

        # Signal to restart voice router if running
        if hasattr(self, "_restart_voice_callback") and self._restart_voice_callback:
            self._restart_voice_callback(kind, device_index)

    def set_audio_manager(self, audio_manager) -> None:
        """Set the AudioDeviceManager so device changes take effect."""
        self._audio_manager = audio_manager

    def set_restart_voice_callback(self, callback) -> None:
        """Set callback to restart voice router when devices change."""
        self._restart_voice_callback = callback

    # --- Public methods for event bridge ---

    def append_output(self, session_name: str, text: str) -> None:
        self._append_output(session_name, text)

    def set_question(self, session_name: str, text: str) -> None:
        self._pending_questions[session_name] = text
        card = self._session_cards.get(session_name)
        if card:
            card.set_question(text)

    def clear_question(self, session_name: str) -> None:
        self._pending_questions.pop(session_name, None)
        card = self._session_cards.get(session_name)
        if card:
            card.set_question("")

    def set_last_heard(self, text: str) -> None:
        self._last_heard_label.setText(f'Last heard: "{text}"')

    def set_mic_level(self, level: int) -> None:
        self._mic_level.setValue(min(100, max(0, level)))

    def _append_output(self, session_name: str, text: str) -> None:
        colors = [
            "#7dd3fc", "#a78bfa", "#34d399", "#fbbf24", "#f87171",
            "#fb923c", "#c084fc", "#22d3ee", "#a3e635", "#f472b6",
        ]
        color = "#64748b" if session_name == "system" else colors[hash(session_name) % len(colors)]

        ts = time.strftime("%H:%M:%S")
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html = (
            f'<span style="color: #4b5563;">{ts}</span> '
            f'<span style="color: {color}; font-weight: bold;">[{session_name}]</span> '
            f'<span style="color: #c9d1d9;">{safe}</span>'
        )
        self._output_log.append(html)
        sb = self._output_log.verticalScrollBar()
        sb.setValue(sb.maximum())


def run_desktop_app() -> None:
    """Launch the desktop GUI with embedded voice listener."""
    import os
    import qasync
    from verbal_direction.config import Config
    from verbal_direction.core.event_bus import EventBus, Event, EventType
    from verbal_direction.core.process_discovery import discover_sessions
    from verbal_direction.core.transcript_monitor import TranscriptMonitor
    from verbal_direction.intelligence.attention_filter import AttentionFilter
    from verbal_direction.intelligence.response_classifier import ResponseClassifier
    from verbal_direction.voice.audio_device import AudioDeviceManager
    from verbal_direction.voice.tts import TTSEngine
    from verbal_direction.voice.stt import STTEngine
    from verbal_direction.voice.vad import VADDetector
    from verbal_direction.voice.voice_router import VoiceRouter

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("verbal-direction")
    app.setStyleSheet(DARK_THEME)

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    config = Config.load()
    event_bus = EventBus()
    audio = AudioDeviceManager(config.audio)

    window = VDDesktopApp()
    window.set_audio_manager(audio)
    window.show()

    # Wire voice router to GUI after creation (below)
    # so clicking a session card sets the default routing target

    def filter_sessions(sessions):
        return sessions

    # Voice components
    attention_filter = AttentionFilter(config.ollama)
    response_classifier = ResponseClassifier(config.ollama)
    tts = TTSEngine(config.voice, audio)
    stt = STTEngine(config.voice)
    vad = VADDetector(config.voice)

    transcript_monitor = TranscriptMonitor(event_bus, attention_filter)
    voice_router = VoiceRouter(
        event_bus=event_bus, tts=tts, stt=stt, vad=vad,
        audio=audio, response_classifier=response_classifier,
    )

    # Wire voice router to GUI so card clicks set default target
    window.set_voice_router(voice_router)

    # Bridge events to GUI
    event_queue = event_bus.subscribe_all()

    async def event_bridge():
        while True:
            event = await event_queue.get()
            try:
                if event.type == EventType.SESSION_QUESTION:
                    text = event.data.get("text", "") if event.data else ""
                    window.set_question(event.session_name, text)
                    window.append_output(event.session_name, f"? {text[:200]}")
                elif event.type == EventType.SESSION_ERROR:
                    text = event.data.get("text", "") if event.data else ""
                    window.append_output(event.session_name, f"ERR: {text[:200]}")
                elif event.type == EventType.SESSION_INFO:
                    text = event.data.get("text", "") if event.data else ""
                    window.append_output(event.session_name, text[:200])
                elif event.type == EventType.VOICE_TRANSCRIPTION:
                    text = event.data.get("text", "") if event.data else ""
                    if text:
                        window.set_last_heard(text)
                        window.append_output("voice", f"Heard: {text}")
                elif event.type == EventType.VOICE_ROUTED:
                    target = event.data.get("target_session", "") if event.data else ""
                    text = event.data.get("text", "") if event.data else ""
                    window.clear_question(target)
                    window.append_output("voice", f"-> {target}: {text}")
            except Exception as e:
                logger.error("Event bridge error: %s", e)

    # Restart voice router when audio device changes
    voice_task = None
    rescan_task = None

    async def start_voice():
        nonlocal voice_task, rescan_task

        sessions = filter_sessions(discover_sessions())
        if sessions:
            window.append_output("system", f"Found {len(sessions)} Claude session(s)")
            for s in sessions:
                window.append_output("system", f"  {s.label} (PID={s.pid}, {s.tty})")
        else:
            window.append_output("system", "No Claude sessions found — rescanning...")

        transcript_monitor.set_sessions(sessions)
        voice_router.set_sessions(sessions)
        await transcript_monitor.start()

        async def rescan():
            while True:
                await asyncio.sleep(10)
                new_sessions = filter_sessions(discover_sessions())
                transcript_monitor.set_sessions(new_sessions)
                voice_router.set_sessions(new_sessions)

        rescan_task = asyncio.create_task(rescan())
        voice_task = asyncio.create_task(voice_router.start())
        window.append_output("system", "Voice listener started")

    async def stop_voice():
        nonlocal voice_task, rescan_task
        if rescan_task:
            rescan_task.cancel()
        if voice_task:
            voice_task.cancel()
        await transcript_monitor.stop()
        await voice_router.stop()

    def on_device_restart(kind, device_index):
        """Restart voice router with new audio device."""
        async def _restart():
            window.append_output("system", "Restarting voice with new audio device...")
            await stop_voice()
            await asyncio.sleep(0.5)
            await start_voice()
        loop.create_task(_restart())

    window.set_restart_voice_callback(on_device_restart)

    # Start everything
    loop.create_task(event_bridge())
    loop.create_task(start_voice())

    with loop:
        loop.run_forever()
