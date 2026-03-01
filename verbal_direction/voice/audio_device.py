"""Audio device management — select mic/speaker for headset."""

from __future__ import annotations

import logging

from verbal_direction.config import AudioConfig

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
except OSError:
    sd = None
    logger.warning("PortAudio not found — audio features unavailable. Install: sudo apt install libportaudio2")


def list_devices() -> list[dict]:
    """List all available audio devices."""
    if sd is None:
        logger.error("sounddevice unavailable — install libportaudio2")
        return []
    devices = sd.query_devices()
    result = []
    for i, dev in enumerate(devices):
        result.append({
            "index": i,
            "name": dev["name"],
            "max_input_channels": dev["max_input_channels"],
            "max_output_channels": dev["max_output_channels"],
            "default_samplerate": dev["default_samplerate"],
            "is_input": dev["max_input_channels"] > 0,
            "is_output": dev["max_output_channels"] > 0,
        })
    return result


def resolve_device(device: str | int, kind: str = "input") -> int | None:
    """Resolve a device name or index to a device index.

    Args:
        device: Device name substring or index. "default" uses system default.
        kind: "input" or "output".

    Returns:
        Device index or None for system default.
    """
    if device == "default":
        return None

    if isinstance(device, int):
        return device

    if sd is None:
        return None

    # Search by name substring
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if kind == "input" and dev["max_input_channels"] == 0:
            continue
        if kind == "output" and dev["max_output_channels"] == 0:
            continue
        if device.lower() in dev["name"].lower():
            logger.info("Resolved %s device '%s' -> %d (%s)", kind, device, i, dev["name"])
            return i

    logger.warning("Device '%s' not found, using system default", device)
    return None


class AudioDeviceManager:
    """Manages audio device selection for input (mic) and output (speaker)."""

    def __init__(self, config: AudioConfig | None = None) -> None:
        self._config = config or AudioConfig()
        self.input_device = resolve_device(self._config.input_device, "input")
        self.output_device = resolve_device(self._config.output_device, "output")
        self.sample_rate = self._config.sample_rate

    def get_input_stream_kwargs(self) -> dict:
        """Get kwargs for creating an input stream."""
        return {
            "device": self.input_device,
            "samplerate": self.sample_rate,
            "channels": 1,
            "dtype": "float32",
        }

    def get_output_stream_kwargs(self) -> dict:
        """Get kwargs for creating an output stream."""
        return {
            "device": self.output_device,
            "samplerate": self.sample_rate,
            "channels": 1,
            "dtype": "float32",
        }
