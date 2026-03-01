"""STT wrapper — faster-whisper with CUDA support."""

from __future__ import annotations

import logging

import numpy as np

from verbal_direction.config import VoiceConfig

logger = logging.getLogger(__name__)


class STTEngine:
    """Speech-to-text engine using faster-whisper."""

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self._config = config or VoiceConfig()
        self._model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-initialize the Whisper model."""
        if self._initialized:
            return

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._config.stt_model,
                device="cuda",
                compute_type="float16",
            )
            self._initialized = True
            logger.info("faster-whisper initialized: %s (CUDA)", self._config.stt_model)
        except Exception as e:
            logger.warning("CUDA init failed, falling back to CPU: %s", e)
            try:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self._config.stt_model,
                    device="cpu",
                    compute_type="int8",
                )
                self._initialized = True
                logger.info("faster-whisper initialized: %s (CPU)", self._config.stt_model)
            except Exception as e2:
                logger.error("faster-whisper initialization failed: %s", e2)
                raise

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text.

        Args:
            audio: Float32 audio data, mono.
            sample_rate: Sample rate (default 16000).

        Returns:
            Transcribed text.
        """
        self._ensure_initialized()
        if not self._model:
            return ""

        segments, _info = self._model.transcribe(
            audio,
            language="en",
            vad_filter=True,
        )

        text = " ".join(segment.text.strip() for segment in segments)
        logger.debug("Transcribed: %s", text)
        return text

    async def transcribe_async(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Async wrapper for transcribe."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe, audio, sample_rate)
