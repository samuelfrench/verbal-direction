"""Voice audio recorder — saves all speech segments for future training."""

from __future__ import annotations

import logging
import time
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RECORDINGS_DIR = Path.home() / ".local" / "share" / "verbal-direction" / "recordings"


class VoiceRecorder:
    """Records all detected speech segments to WAV files for training data."""

    def __init__(
        self,
        recordings_dir: Path | str | None = None,
        sample_rate: int = 16000,
    ) -> None:
        self._recordings_dir = Path(recordings_dir or DEFAULT_RECORDINGS_DIR)
        self._sample_rate = sample_rate
        self._session_dir: Path | None = None
        self._segment_count = 0
        self.enabled = True

        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Create the recordings directory structure."""
        # Each run gets a timestamped subdirectory
        session_id = time.strftime("%Y%m%d_%H%M%S")
        self._session_dir = self._recordings_dir / session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Voice recordings will be saved to: %s", self._session_dir)

    def save_segment(
        self,
        audio: np.ndarray,
        transcription: str = "",
        session_name: str = "",
    ) -> Path:
        """Save a speech segment to a WAV file.

        Args:
            audio: Float32 mono audio data.
            transcription: The STT transcription (saved to sidecar .txt).
            session_name: Which Claude session this was directed to.

        Returns:
            Path to the saved WAV file.
        """
        if not self.enabled:
            return Path("/dev/null")

        self._segment_count += 1
        timestamp = time.strftime("%H%M%S")
        basename = f"{timestamp}_{self._segment_count:04d}"
        wav_path = self._session_dir / f"{basename}.wav"

        # Convert float32 to int16 for WAV
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio_int16.tobytes())

        # Save metadata sidecar
        meta_path = self._session_dir / f"{basename}.txt"
        meta_lines = []
        if transcription:
            meta_lines.append(f"transcription: {transcription}")
        if session_name:
            meta_lines.append(f"session: {session_name}")
        meta_lines.append(f"duration: {len(audio) / self._sample_rate:.2f}s")
        meta_lines.append(f"samples: {len(audio)}")
        meta_path.write_text("\n".join(meta_lines) + "\n")

        logger.debug("Saved speech segment: %s (%.1fs)", wav_path.name, len(audio) / self._sample_rate)
        return wav_path

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def segment_count(self) -> int:
        return self._segment_count
