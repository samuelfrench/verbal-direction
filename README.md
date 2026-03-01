# verbal-direction

Voice-controlled Claude Code session manager. Manage multiple Claude Code terminal sessions hands-free using a headset.

## What it does

Run multiple Claude Code sessions simultaneously and control them all by voice:

1. Claude asks a question in any session → it's spoken to your headset via TTS
2. You respond verbally → speech is transcribed and routed to the correct session
3. Claude continues working — no keyboard needed

## Architecture

```
  CLI (vd launch/list/kill)
        |
  Session Manager (ClaudeSDKClient per session)
        |
  Event Bus (asyncio queues)
        |
  ┌─────┼─────────┬──────────┐
  │     │         │          │
Output  Attention Voice I/O  TUI
Monitor Filter    (Piper +   Dashboard
        (Ollama)  Whisper)   (Textual)
```

- **Session Manager** — wraps `ClaudeSDKClient` instances from the official Python SDK
- **Output Monitor** — watches each session for questions, errors, permission requests
- **Attention Filter** — uses local Ollama (llama3.2) to classify what needs your attention
- **Voice I/O** — Piper TTS (CUDA) for output, faster-whisper (CUDA) for input, Silero VAD for hands-free activation
- **Voice Router** — routes your spoken responses to the correct session
- **TUI Dashboard** — Textual-based terminal UI showing all session statuses

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (for TTS/STT acceleration)
- [Ollama](https://ollama.ai) running locally with `llama3.2` model
- A headset with microphone

## Installation

```bash
git clone https://github.com/samuelfrench/verbal-direction.git
cd verbal-direction
pip install -e .
```

Download a Piper TTS voice model:

```bash
mkdir -p ~/.local/share/piper-voices
# Download a voice (e.g., en_US-lessac-medium)
wget -O ~/.local/share/piper-voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget -O ~/.local/share/piper-voices/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Pull the Ollama model:

```bash
ollama pull llama3.2
```

## Usage

```bash
# Start a Claude session
vd launch auth ~/my-project -- "implement JWT authentication"

# Start another session
vd launch frontend ~/my-project -- "build the login page"

# Start the voice listener (main loop)
vd

# List active sessions
vd list

# Show TUI dashboard
vd status

# Pause/resume voice monitoring for a session
vd pause auth
vd resume auth

# Kill a session
vd kill auth
```

## How voice routing works

1. When Claude asks a question, the session name is prefixed: *"Auth asks: Which JWT library should I use?"*
2. The most recently asking session gets priority for your response
3. To target a specific session, say its name: *"Frontend: use Tailwind CSS"*
4. For ambiguous cases, Ollama helps determine the best match

## Configuration

Create `~/.config/verbal-direction/config.toml`:

```toml
[voice]
tts_model = "en_US-lessac-medium"
tts_speed = 1.0
stt_model = "base.en"
vad_threshold = 0.5

[ollama]
model = "llama3.2"
host = "http://localhost:11434"

[audio]
input_device = "default"
output_device = "default"
sample_rate = 16000
```

## License

MIT
