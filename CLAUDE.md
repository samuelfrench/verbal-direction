# verbal-direction

Voice-controlled Claude Code session manager.

## Quick Start
- Read `TODO.md` at the start of each session
- This is a Python project using `pyproject.toml` (setuptools)
- Install with `pip install -e .`
- CLI entry point: `vd` (maps to `verbal_direction.__main__:cli`)

## Architecture
- `verbal_direction/core/` — session management, event bus, output monitor, response dispatcher
- `verbal_direction/voice/` — TTS (Piper), STT (faster-whisper), VAD (Silero), voice routing
- `verbal_direction/intelligence/` — Ollama-based attention filter and response classifier
- `verbal_direction/ui/` — Textual TUI dashboard

## Key Dependencies
- `claude-agent-sdk` — ClaudeSDKClient for programmatic Claude Code control
- `ollama` — local LLM for output classification and response routing
- `piper-tts` — local neural TTS (CUDA)
- `faster-whisper` — local STT (CUDA)
- `silero-vad` — voice activity detection
- `sounddevice` — audio I/O (requires system `libportaudio2`)
- `textual` — TUI framework
- `click` — CLI framework

## System Requirements
- PortAudio: `sudo apt install libportaudio2`
- Ollama running locally with `llama3.2` model
- Piper voice model in `~/.local/share/piper-voices/`
