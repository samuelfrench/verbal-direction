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

            self._voice = PiperVoice.load(str(model_path), use_cuda=True)
            self._initialized = True
            logger.info("Piper TTS initialized with model: %s (CUDA)", self._config.tts_model)
        except ImportError:
            logger.error("piper-tts not installed. Run: pip install piper-tts")
            raise
        except Exception as e:
            logger.warning("Piper CUDA init failed, trying CPU: %s", e)
            try:
                from piper import PiperVoice

                model_path = DEFAULT_VOICE_DIR / f"{self._config.tts_model}.onnx"
                self._voice = PiperVoice.load(str(model_path), use_cuda=False)
                self._initialized = True
                logger.info("Piper TTS initialized with model: %s (CPU)", self._config.tts_model)
            except Exception as e2:
                logger.error("Piper TTS initialization failed: %s", e2)
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

        # Synthesize to WAV bytes
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            self._voice.synthesize(text, wav_file, length_scale=1.0 / self._config.tts_speed)

        # Read the WAV data
        wav_buffer.seek(0)
        with wave.open(wav_buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            audio_data = wav_file.readframes(n_frames)

        # Convert to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

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
