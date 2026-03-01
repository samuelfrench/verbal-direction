"""TTS wrapper — Piper TTS with CUDA support."""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path

import numpy as np

from verbal_direction.config import VoiceConfig
from verbal_direction.voice.audio_device import AudioDeviceManager

logger = logging.getLogger(__name__)

DEFAULT_VOICE_DIR = Path.home() / ".local" / "share" / "piper-voices"


class TTSEngine:
    """Text-to-speech engine using Piper TTS."""

    def __init__(
        self,
        config: VoiceConfig | None = None,
        audio: AudioDeviceManager | None = None,
    ) -> None:
        self._config = config or VoiceConfig()
        self._audio = audio or AudioDeviceManager()
        self._voice = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy-initialize the Piper voice model."""
        if self._initialized:
            return

        try:
            from piper import PiperVoice

            model_path = DEFAULT_VOICE_DIR / f"{self._config.tts_model}.onnx"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Piper voice model not found at {model_path}. "
                    f"Download it from https://huggingface.co/rhasspy/piper-voices"
                )

            # Try CUDA first, fall back to CPU
            try:
                self._voice = PiperVoice.load(str(model_path), use_cuda=True)
                logger.info("Piper TTS initialized with model: %s (CUDA)", self._config.tts_model)
            except Exception:
                self._voice = PiperVoice.load(str(model_path), use_cuda=False)
                logger.info("Piper TTS initialized with model: %s (CPU)", self._config.tts_model)

            self._initialized = True
        except ImportError:
            logger.error("piper-tts not installed. Run: pip install piper-tts")
            raise
        except Exception as e:
            logger.error("Piper TTS initialization failed: %s", e)
            raise

    def speak(self, text: str, session_name: str | None = None) -> None:
        """Speak text through the headset.

        Args:
            text: Text to speak.
            session_name: If provided, prefix with session name.
        """
        self._ensure_initialized()
        if not self._voice:
            logger.error("TTS not initialized")
            return

        if session_name:
            text = f"{session_name} asks: {text}"

        from piper.config import SynthesisConfig

        syn_config = SynthesisConfig(
            length_scale=1.0 / self._config.tts_speed,
        )

        # Synthesize to audio chunks
        chunks = list(self._voice.synthesize(text, syn_config))
        if not chunks:
            logger.warning("TTS produced no audio")
            return

        # Concatenate all audio chunks
        audio_data = np.concatenate([
            np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            for chunk in chunks
        ])
        audio_array = audio_data.astype(np.float32) / 32768.0

        # Get sample rate from voice config
        sample_rate = self._voice.config.sample_rate

        # Play through output device
        import sounddevice as sd
        sd.play(
            audio_array,
            samplerate=sample_rate,
            device=self._audio.output_device,
        )
        sd.wait()

    async def speak_async(self, text: str, session_name: str | None = None) -> None:
        """Async wrapper for speak — runs in executor to avoid blocking."""
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.speak, text, session_name)
