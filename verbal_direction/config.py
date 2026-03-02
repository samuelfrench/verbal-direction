"""Configuration loader for verbal-direction."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VoiceConfig:
    tts_model: str = "en_US-amy-medium"
    tts_speed: float = 1.4
    stt_model: str = "base.en"
    vad_threshold: float = 0.5


@dataclass
class OllamaConfig:
    model: str = "llama3.2"
    host: str = "http://localhost:11434"


@dataclass
class AudioConfig:
    input_device: str | int = "default"
    output_device: str | int = "default"
    sample_rate: int = 16000


@dataclass
class Config:
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Load config from TOML file, falling back to defaults."""
        if path is None:
            path = Path.home() / ".config" / "verbal-direction" / "config.toml"

        config = cls()

        if not path.exists():
            return config

        with open(path, "rb") as f:
            data = tomllib.load(f)

        if "voice" in data:
            for key, value in data["voice"].items():
                if hasattr(config.voice, key):
                    setattr(config.voice, key, value)

        if "ollama" in data:
            for key, value in data["ollama"].items():
                if hasattr(config.ollama, key):
                    setattr(config.ollama, key, value)

        if "audio" in data:
            for key, value in data["audio"].items():
                if hasattr(config.audio, key):
                    setattr(config.audio, key, value)

        return config
