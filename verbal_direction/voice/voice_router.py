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

        # Subscribe to question/error events for TTS
        self._tts_queue = event_bus.subscribe(
            EventType.SESSION_QUESTION,
            EventType.SESSION_ERROR,
        )

    def set_sessions(self, sessions: list[DiscoveredSession]) -> None:
        """Update the set of discovered sessions."""
        self._sessions = {s.label: s for s in sessions}

    async def start(self) -> None:
        """Start the voice router."""
        self._running = True
        await asyncio.gather(
            self._tts_loop(),
            self._listen_loop(),
        )

    async def stop(self) -> None:
        self._running = False
        self._event_bus.unsubscribe(self._tts_queue)

    async def _tts_loop(self) -> None:
        """Speak questions/errors from Claude sessions."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._tts_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            text = event.data.get("text", "") if event.data else ""
            if not text:
                continue

            # Track question order for routing
            self._question_order.append((event.session_name, time.time()))
            self._question_order = self._question_order[-20:]

            # Store pending question
            self._pending_questions[event.session_name] = text

            # Truncate long text for speech
            speak_text = text[:300] if len(text) > 300 else text
            await self._tts.speak_async(speak_text, session_name=event.session_name)

    async def _listen_loop(self) -> None:
        """Listen for speech, transcribe, and inject into terminals."""
        import queue as thread_queue
        # Use a thread-safe queue since sounddevice callbacks run in a separate thread
        audio_queue: thread_queue.Queue[np.ndarray] = thread_queue.Queue()
        speech_buffer: list[np.ndarray] = []

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
            while self._running:
                # Poll the thread-safe queue from asyncio
                try:
                    chunk = audio_queue.get_nowait()
                except thread_queue.Empty:
                    await asyncio.sleep(0.02)
                    continue

                result = self._vad.process_chunk(chunk)

                if result["speech_started"]:
                    speech_buffer.clear()

                if result["is_speech"] or self._vad._is_speaking:
                    speech_buffer.append(chunk)

                if result["speech_ended"] and speech_buffer:
                    audio_data = np.concatenate(speech_buffer)
                    speech_buffer.clear()
                    self._vad.reset()

                    # Transcribe
                    text = await self._stt.transcribe_async(audio_data)
                    if not text or len(text.strip()) < 2:
                        continue

                    logger.info("Heard: %s", text)

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

                        # Inject into the terminal
                        success = await asyncio.get_event_loop().run_in_executor(
                            None, inject_text, session, clean_text
                        )
                        if not success:
                            # Fallback to xdotool
                            success = await asyncio.get_event_loop().run_in_executor(
                                None, inject_text_xdotool, session, clean_text
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

        # Multiple sessions, no pending questions — log available targets
        logger.info(
            "Multiple sessions, no pending question. Prefix with session name: %s",
            ", ".join(self._sessions.keys()),
        )
        return None
