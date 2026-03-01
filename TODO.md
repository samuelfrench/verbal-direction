# TODO — verbal-direction

## Pending
- [ ] Config file support (~/.config/verbal-direction/config.toml)
- [ ] IPC for cross-process session management (vd list/kill from different terminal)
- [ ] End-to-end integration test with real Claude session
- [ ] Demo recording / GIF for README

## Completed
- [x] GitHub repo created (samuelfrench/verbal-direction)
- [x] Project scaffolding (pyproject.toml, directory structure)
- [x] README.md with architecture, installation, usage docs
- [x] Core session management (SessionManager, SessionState, EventBus)
- [x] Output monitor with async event routing
- [x] Response dispatcher for voice→session routing
- [x] Attention filter (Ollama + heuristic fallback)
- [x] Response classifier (Ollama-based ambiguous routing)
- [x] Voice I/O pipeline (Piper TTS, faster-whisper STT, Silero VAD)
- [x] Audio device management with graceful PortAudio fallback
- [x] Voice router with priority-based session targeting
- [x] CLI interface (vd launch/list/kill/pause/resume/status/listen/devices)
- [x] TUI dashboard (Textual) with session cards, voice status, log panel
- [x] pip install -e . installs cleanly
- [x] All module imports verified
