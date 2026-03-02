"""Voice router — routes transcribed speech to the correct terminal session."""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from verbal_direction.core.event_bus import EventBus, Event, EventType
from verbal_direction.core.process_discovery import DiscoveredSession
from verbal_direction.core.terminal_router import inject_text, inject_text_xdotool
from verbal_direction.intelligence.response_classifier import ResponseClassifier
from verbal_direction.voice.audio_device import AudioDeviceManager
from verbal_direction.voice.recorder import VoiceRecorder
from verbal_direction.voice.stt import STTEngine
from verbal_direction.voice.tts import TTSEngine
from verbal_direction.voice.vad import VADDetector

logger = logging.getLogger(__name__)

CHUNK_SIZE = 512  # samples at 16kHz
AUDIO_WATCHDOG_TIMEOUT = 5.0  # seconds without audio before restart
VAD_MAX_SPEECH_DURATION = 30.0  # max seconds of continuous speech before force-end
TTS_MAX_QUEUE_AGE = 30.0  # drop TTS messages older than this


class VoiceRouter:
    """Listens for speech, transcribes, and routes to the correct terminal."""

    def __init__(
        self,
        event_bus: EventBus,
        tts: TTSEngine,
        stt: STTEngine,
        vad: VADDetector,
        audio: AudioDeviceManager,
        response_classifier: ResponseClassifier,
    ) -> None:
        self._event_bus = event_bus
        self._tts = tts
        self._stt = stt
        self._vad = vad
        self._audio = audio
        self._classifier = response_classifier
        self._recorder = VoiceRecorder(sample_rate=audio.sample_rate)
        self._running = False

        # Discovered sessions (updated externally)
        self._sessions: dict[str, DiscoveredSession] = {}
        # Track which sessions asked questions and when
        self._question_order: list[tuple[str, float]] = []
        # Pending questions per session
        self._pending_questions: dict[str, str] = {}
        # Default target session (set via GUI click)
        self._default_target: str | None = None
        # TTS mode: "questions" = only questions/errors, "all" = all messages, "smart" = meaningful
        self._tts_mode: str = "all"
        # Pause state
        self._paused: bool = False

        # Health tracking
        self._last_audio_chunk_time: float = 0.0
        self._last_speech_time: float = 0.0
        self._tts_speaking: bool = False
        self._stream_restarts: int = 0

        # TTS queue subscribed on start()
        self._tts_queue: asyncio.Queue | None = None
        # Callback to sync GUI pause button
        self._on_pause_callback = None

    def set_sessions(self, sessions: list[DiscoveredSession]) -> None:
        """Update the set of discovered sessions."""
        self._sessions = {s.label: s for s in sessions}

    def set_default_target(self, label: str) -> None:
        """Set the default target session (e.g. via GUI click)."""
        self._default_target = label
        logger.info("Default target set to: %s", label)

    def set_tts_mode(self, mode: str) -> None:
        """Set TTS mode: 'questions', 'all', or 'smart'."""
        self._tts_mode = mode
        logger.info("TTS mode set to: %s", mode)

    def set_paused(self, paused: bool) -> None:
        """Pause or resume voice listening and TTS."""
        self._paused = paused
        logger.info("Voice %s", "paused" if paused else "resumed")

    async def start(self) -> None:
        """Start the voice router."""
        self._running = True
        # Re-subscribe TTS queue in case stop() unsubscribed it
        self._tts_queue = self._event_bus.subscribe(
            EventType.SESSION_QUESTION,
            EventType.SESSION_ERROR,
            EventType.SESSION_INFO,
        )
        self._last_audio_chunk_time = time.time()
        await asyncio.gather(
            self._tts_loop(),
            self._listen_loop(),
            self._health_monitor_loop(),
        )

    async def stop(self) -> None:
        self._running = False
        if self._tts_queue:
            self._event_bus.unsubscribe(self._tts_queue)

    async def _health_monitor_loop(self) -> None:
        """Periodic health check — publishes stream status to GUI."""
        while self._running:
            await asyncio.sleep(2.0)

            now = time.time()
            audio_age = now - self._last_audio_chunk_time if self._last_audio_chunk_time else 0
            speech_age = now - self._last_speech_time if self._last_speech_time else 0

            # Determine stream health
            if self._paused:
                status = "paused"
            elif audio_age > AUDIO_WATCHDOG_TIMEOUT:
                status = "no_audio"
                logger.warning("No audio data for %.1fs — stream may be dead", audio_age)
            else:
                status = "healthy"

            await self._event_bus.publish(Event(
                type=EventType.VOICE_STREAM_STATUS,
                session_name="",
                data={
                    "status": status,
                    "audio_age": audio_age,
                    "speech_age": speech_age,
                    "tts_speaking": self._tts_speaking,
                    "tts_queue_size": self._tts_queue.qsize() if self._tts_queue else 0,
                    "stream_restarts": self._stream_restarts,
                },
            ))

    async def _tts_loop(self) -> None:
        """Speak questions/errors/info from Claude sessions based on TTS mode."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._tts_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            text = event.data.get("text", "") if event.data else ""
            if not text:
                continue

            # Skip if paused
            if self._paused:
                continue

            # Drop stale messages (older than TTS_MAX_QUEUE_AGE)
            if time.time() - event.timestamp > TTS_MAX_QUEUE_AGE:
                logger.debug("Dropping stale TTS message (%.0fs old)", time.time() - event.timestamp)
                continue

            # In "questions" mode, skip all non-question/error messages
            if self._tts_mode == "questions" and event.type == EventType.SESSION_INFO:
                continue

            # In "smart" mode, skip plain informational but allow meaningful results
            if self._tts_mode == "smart" and event.type == EventType.SESSION_INFO:
                classification = event.data.get("classification", "") if event.data else ""
                if classification != "meaningful":
                    continue

            # Track question order for routing (only for questions)
            if event.type == EventType.SESSION_QUESTION:
                self._question_order.append((event.session_name, time.time()))
                self._question_order = self._question_order[-20:]
                self._pending_questions[event.session_name] = text

            # Truncate long text for speech
            speak_text = text[:300] if len(text) > 300 else text
            logger.info("TTS speaking (%s): %s", event.type.name, speak_text[:80])

            # Update TTS status
            self._tts_speaking = True
            await self._event_bus.publish(Event(
                type=EventType.VOICE_TTS_STATUS,
                session_name="",
                data={"speaking": True, "text": speak_text[:80]},
            ))

            try:
                await self._tts.speak_async(speak_text, session_name=event.session_name)
            except Exception as e:
                logger.error("TTS speak failed: %s", e)
            finally:
                self._tts_speaking = False
                await self._event_bus.publish(Event(
                    type=EventType.VOICE_TTS_STATUS,
                    session_name="",
                    data={"speaking": False, "text": ""},
                ))

    async def _listen_loop(self) -> None:
        """Listen for speech, transcribe, and inject into terminals."""
        while self._running:
            try:
                await self._listen_once()
            except Exception as e:
                logger.error("Listen loop error: %s — restarting in 2s", e)
                self._stream_restarts += 1
                await self._event_bus.publish(Event(
                    type=EventType.VOICE_STREAM_STATUS,
                    session_name="",
                    data={"status": "restarting", "error": str(e), "stream_restarts": self._stream_restarts},
                ))
                await asyncio.sleep(2.0)

    async def _listen_once(self) -> None:
        """Single listen session — restarts on error."""
        import queue as thread_queue
        audio_queue: thread_queue.Queue[np.ndarray] = thread_queue.Queue()
        speech_buffer: list[np.ndarray] = []
        speech_start_time: float = 0.0

        def audio_callback(indata: np.ndarray, frames: int, time_info: dict, status: int) -> None:
            if status:
                logger.warning("Audio input status: %s", status)
            audio_queue.put_nowait(indata.copy().flatten())

        import sounddevice as sd
        stream = sd.InputStream(
            callback=audio_callback,
            blocksize=CHUNK_SIZE,
            **self._audio.get_input_stream_kwargs(),
        )

        with stream:
            logger.info("Voice listener started — mic active")
            self._last_audio_chunk_time = time.time()

            while self._running:
                # When paused, still listen for "resume" voice command
                # but drain the audio queue to prevent buildup
                if self._paused:
                    try:
                        chunk = audio_queue.get_nowait()
                        self._last_audio_chunk_time = time.time()
                        result = self._vad.process_chunk(chunk)
                        if result["is_speech"] or self._vad._is_speaking:
                            speech_buffer.append(chunk)
                        if result["speech_started"]:
                            speech_buffer.clear()
                            speech_start_time = time.time()
                        if result["speech_ended"] and speech_buffer:
                            audio_data = np.concatenate(speech_buffer)
                            speech_buffer.clear()
                            speech_start_time = 0.0
                            self._vad.reset()
                            text = await self._stt.transcribe_async(audio_data)
                            if text and len(text.strip()) >= 2:
                                logger.info("Heard (paused): %s", text)
                                await self._handle_voice_command(text)
                    except thread_queue.Empty:
                        pass
                    await asyncio.sleep(0.02)
                    continue

                # Poll the thread-safe queue from asyncio
                try:
                    chunk = audio_queue.get_nowait()
                except thread_queue.Empty:
                    # Check for audio stream timeout
                    if time.time() - self._last_audio_chunk_time > AUDIO_WATCHDOG_TIMEOUT * 2:
                        logger.error("Audio stream appears dead — forcing restart")
                        return  # Will be caught by _listen_loop and restarted
                    await asyncio.sleep(0.02)
                    continue

                self._last_audio_chunk_time = time.time()

                # Publish mic level for GUI (every ~10 chunks to avoid flooding)
                peak = float(np.max(np.abs(chunk)))
                level = min(100, int(peak * 500))  # scale to 0-100
                await self._event_bus.publish(Event(
                    type=EventType.VOICE_MIC_LEVEL,
                    session_name="",
                    data={"level": level, "peak": peak},
                ))

                result = self._vad.process_chunk(chunk)

                if result["speech_started"]:
                    speech_buffer.clear()
                    speech_start_time = time.time()

                if result["is_speech"] or self._vad._is_speaking:
                    speech_buffer.append(chunk)

                    # VAD timeout — force end if speaking too long
                    if speech_start_time and (time.time() - speech_start_time) > VAD_MAX_SPEECH_DURATION:
                        logger.warning("VAD timeout — forcing speech end after %.0fs", VAD_MAX_SPEECH_DURATION)
                        result["speech_ended"] = True

                if result["speech_ended"] and speech_buffer:
                    audio_data = np.concatenate(speech_buffer)
                    speech_buffer.clear()
                    speech_start_time = 0.0
                    self._vad.reset()
                    self._last_speech_time = time.time()

                    # Transcribe
                    text = await self._stt.transcribe_async(audio_data)
                    if not text or len(text.strip()) < 2:
                        continue

                    logger.info("Heard: %s", text)

                    # Check for voice commands before routing
                    if await self._handle_voice_command(text):
                        continue

                    # Determine target session
                    target = self._determine_target(text)

                    # Save audio for training
                    self._recorder.save_segment(
                        audio=audio_data,
                        transcription=text,
                        session_name=target or "",
                    )

                    await self._event_bus.publish(Event(
                        type=EventType.VOICE_TRANSCRIPTION,
                        session_name="",
                        data={"text": text},
                    ))

                    if target and target in self._sessions:
                        session = self._sessions[target]

                        # Strip session name prefix if used for routing
                        clean_text = text
                        for prefix in [f"{target}:", f"{target},"]:
                            if clean_text.lower().startswith(prefix.lower()):
                                clean_text = clean_text[len(prefix):].strip()
                                break

                        # Inject into the terminal via keyboard emulation (xdotool)
                        success = await asyncio.get_event_loop().run_in_executor(
                            None, inject_text_xdotool, session, clean_text
                        )
                        if not success:
                            # Fallback to PTY master write
                            success = await asyncio.get_event_loop().run_in_executor(
                                None, inject_text, session, clean_text
                            )

                        if success:
                            # Clear pending question
                            self._pending_questions.pop(target, None)
                            await self._event_bus.publish(Event(
                                type=EventType.VOICE_ROUTED,
                                session_name=target,
                                data={"text": clean_text, "target_session": target},
                            ))
                        else:
                            logger.error("Failed to inject response into %s", target)
                    else:
                        logger.warning("No target session for: %s", text)

    async def _handle_voice_command(self, text: str) -> bool:
        """Check for voice commands. Returns True if handled (don't route to terminal)."""
        cmd = text.strip().lower().rstrip(".,!?")

        if cmd in ("pause", "pause listening", "stop listening"):
            logger.info("Voice command: PAUSE")
            self.set_paused(True)
            await self._event_bus.publish(Event(
                type=EventType.VOICE_TRANSCRIPTION,
                session_name="",
                data={"text": "[voice command: pause]"},
            ))
            # Notify GUI to update pause button state
            if self._on_pause_callback:
                self._on_pause_callback(True)
            return True

        if cmd in ("resume", "resume listening", "start listening"):
            logger.info("Voice command: RESUME")
            self.set_paused(False)
            await self._event_bus.publish(Event(
                type=EventType.VOICE_TRANSCRIPTION,
                session_name="",
                data={"text": "[voice command: resume]"},
            ))
            if self._on_pause_callback:
                self._on_pause_callback(False)
            return True

        return False

    def set_pause_callback(self, callback) -> None:
        """Set callback to sync GUI pause button state with voice commands."""
        self._on_pause_callback = callback

    def _determine_target(self, text: str) -> str | None:
        """Determine which session a voice response targets."""
        if not self._sessions:
            return None

        # Check for explicit session name prefix (works with or without pending questions)
        text_lower = text.lower()
        for label in self._sessions:
            if text_lower.startswith(f"{label.lower()}:"):
                return label
            if text_lower.startswith(f"{label.lower()},"):
                return label

        # If there are pending questions, prioritize those sessions
        if self._pending_questions:
            if len(self._pending_questions) == 1:
                return next(iter(self._pending_questions))

            # Last-asked-first priority
            for name, _ts in reversed(self._question_order):
                if name in self._pending_questions:
                    return name

            return next(iter(self._pending_questions))

        # No pending questions — still route if possible
        if len(self._sessions) == 1:
            return next(iter(self._sessions))

        # Use default target set via GUI click
        if self._default_target and self._default_target in self._sessions:
            return self._default_target

        # Multiple sessions, no pending questions, no default — log available targets
        logger.info(
            "Multiple sessions, no pending question. Prefix with session name or click a session: %s",
            ", ".join(self._sessions.keys()),
        )
        return None
