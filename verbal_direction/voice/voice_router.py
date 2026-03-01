"""Voice router — routes transcribed speech to the correct session."""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from verbal_direction.core.event_bus import EventBus, Event, EventType
from verbal_direction.core.session_manager import SessionManager
from verbal_direction.intelligence.response_classifier import ResponseClassifier
from verbal_direction.voice.audio_device import AudioDeviceManager
from verbal_direction.voice.recorder import VoiceRecorder
from verbal_direction.voice.stt import STTEngine
from verbal_direction.voice.tts import TTSEngine
from verbal_direction.voice.vad import VADDetector

logger = logging.getLogger(__name__)

CHUNK_DURATION = 0.032  # 32ms chunks for VAD
CHUNK_SIZE = 512  # samples at 16kHz


class VoiceRouter:
    """Main voice I/O loop: listens for speech, transcribes, and routes to sessions."""

    def __init__(
        self,
        event_bus: EventBus,
        session_manager: SessionManager,
        tts: TTSEngine,
        stt: STTEngine,
        vad: VADDetector,
        audio: AudioDeviceManager,
        response_classifier: ResponseClassifier,
    ) -> None:
        self._event_bus = event_bus
        self._session_manager = session_manager
        self._tts = tts
        self._stt = stt
        self._vad = vad
        self._audio = audio
        self._classifier = response_classifier
        self._recorder = VoiceRecorder(sample_rate=audio.sample_rate)
        self._running = False

        # Track which sessions asked questions and when
        self._question_order: list[tuple[str, float]] = []

        # Subscribe to question/permission/error events for TTS
        self._tts_queue = event_bus.subscribe(
            EventType.SESSION_QUESTION,
            EventType.SESSION_PERMISSION,
            EventType.SESSION_ERROR,
        )

    async def start(self) -> None:
        """Start the voice router."""
        self._running = True
        await asyncio.gather(
            self._tts_loop(),
            self._listen_loop(),
        )

    async def stop(self) -> None:
        """Stop the voice router."""
        self._running = False
        self._event_bus.unsubscribe(self._tts_queue)

    async def _tts_loop(self) -> None:
        """Read question/error events and speak them."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._tts_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            session = self._session_manager.get(event.session_name)
            if session and session.state.voice_paused:
                continue

            text = event.data.get("text", "") if event.data else ""
            if not text:
                continue

            # Track question order for routing
            self._question_order.append((event.session_name, time.time()))
            # Keep only last 20 entries
            self._question_order = self._question_order[-20:]

            await self._tts.speak_async(text, session_name=event.session_name)

    async def _listen_loop(self) -> None:
        """Listen for speech, transcribe, and route."""
        audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
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
            logger.info("Voice listener started")
            while self._running:
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                result = self._vad.process_chunk(chunk)

                if result["speech_started"]:
                    speech_buffer.clear()
                    logger.debug("Speech started")

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

                    # Route the response
                    target = await self._determine_target(text)

                    # Save audio segment for training
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

                    if target:
                        await self._event_bus.publish(Event(
                            type=EventType.VOICE_ROUTED,
                            session_name=target,
                            data={"text": text, "target_session": target},
                        ))
                    else:
                        logger.warning("Could not route response: %s", text)

    async def _determine_target(self, text: str) -> str | None:
        """Determine which session a voice response targets."""
        waiting = self._session_manager.get_waiting_sessions()
        if not waiting:
            return None

        # Check for explicit session name prefix (e.g., "frontend: use tailwind")
        text_lower = text.lower()
        for session in waiting:
            if text_lower.startswith(f"{session.name.lower()}:"):
                return session.name
            if text_lower.startswith(f"{session.name.lower()},"):
                return session.name

        # If only one session is waiting, route there
        if len(waiting) == 1:
            return waiting[0].name

        # Use last-asked-first priority
        for name, _ts in reversed(self._question_order):
            for session in waiting:
                if session.name == name:
                    return session.name

        # Fall back to Ollama classification
        sessions_map = {}
        for session in waiting:
            q = session.state.pending_question
            if q:
                sessions_map[session.name] = q.text

        if sessions_map:
            result = await self._classifier.route_response(text, sessions_map)
            if result:
                return result

        # Last resort: first waiting session
        return waiting[0].name
