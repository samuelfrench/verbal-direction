"""Voice Activity Detection using Silero VAD."""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from verbal_direction.config import VoiceConfig

logger = logging.getLogger(__name__)


class VADDetector:
    """Silero VAD wrapper for hands-free voice activation."""

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self._config = config or VoiceConfig()
        self._model = None
        self._initialized = False
        # Ring buffer for audio chunks
        self._buffer: deque[np.ndarray] = deque(maxlen=100)
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0

    def _ensure_initialized(self) -> None:
        """Lazy-initialize Silero VAD model."""
        if self._initialized:
            return

        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._model = model
            self._get_speech_timestamps = utils[0]
            self._initialized = True
            logger.info("Silero VAD initialized")
        except Exception as e:
            logger.error("Silero VAD initialization failed: %s", e)
            raise

    def process_chunk(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> dict:
        """Process an audio chunk and detect speech activity.

        Args:
            audio_chunk: Float32 audio data, mono, 16kHz.
            sample_rate: Sample rate (default 16000).

        Returns:
            Dict with keys:
                - is_speech: bool — whether speech is detected in this chunk
                - speech_started: bool — True on the frame speech begins
                - speech_ended: bool — True on the frame speech ends
        """
        self._ensure_initialized()

        import torch

        # Convert to tensor
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        tensor = torch.from_numpy(audio_chunk)
        speech_prob = self._model(tensor, sample_rate).item()

        is_speech = speech_prob > self._config.vad_threshold
        speech_started = False
        speech_ended = False

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0

            if not self._is_speaking and self._speech_frames >= 3:
                self._is_speaking = True
                speech_started = True
        else:
            self._silence_frames += 1
            self._speech_frames = 0

            if self._is_speaking and self._silence_frames >= 15:
                self._is_speaking = False
                speech_ended = True

        self._buffer.append(audio_chunk)

        return {
            "is_speech": is_speech,
            "speech_started": speech_started,
            "speech_ended": speech_ended,
            "probability": speech_prob,
        }

    def get_speech_audio(self) -> np.ndarray:
        """Get the buffered audio from the current/last speech segment."""
        if not self._buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(list(self._buffer))

    def reset(self) -> None:
        """Reset the VAD state."""
        self._buffer.clear()
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        if self._model is not None:
            self._model.reset_states()
