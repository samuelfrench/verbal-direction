"""CLI entry point for verbal-direction."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import click

from verbal_direction.config import Config


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group(invoke_without_command=True)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.option("-c", "--config", "config_path", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, config_path: str | None) -> None:
    """verbal-direction — Voice-controlled Claude Code session manager.

    Run with no subcommand to start the voice listener.
    """
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config.load(Path(config_path) if config_path else None)

    if ctx.invoked_subcommand is None:
        ctx.invoke(listen)


@cli.command("list")
def list_sessions() -> None:
    """List running Claude Code sessions detected on this machine."""
    from verbal_direction.core.process_discovery import discover_sessions

    sessions = discover_sessions()
    if not sessions:
        click.echo("No running Claude Code sessions found.")
        return

    click.echo(f"Found {len(sessions)} Claude Code session(s):\n")
    for s in sessions:
        click.echo(f"  {s.label:<25} PID={s.pid}  TTY={s.tty}  CWD={s.cwd}")
        if s.slug:
            click.echo(f"  {'':25} slug={s.slug}")
        click.echo()


@cli.command()
@click.pass_context
def listen(ctx: click.Context) -> None:
    """Start voice listener — monitors existing Claude sessions."""
    config = ctx.obj["config"]

    async def _run() -> None:
        from verbal_direction.core.event_bus import EventBus
        from verbal_direction.core.process_discovery import discover_sessions
        from verbal_direction.core.transcript_monitor import TranscriptMonitor
        from verbal_direction.intelligence.attention_filter import AttentionFilter
        from verbal_direction.intelligence.response_classifier import ResponseClassifier
        from verbal_direction.voice.audio_device import AudioDeviceManager
        from verbal_direction.voice.tts import TTSEngine
        from verbal_direction.voice.stt import STTEngine
        from verbal_direction.voice.vad import VADDetector
        from verbal_direction.voice.voice_router import VoiceRouter

        event_bus = EventBus()
        audio = AudioDeviceManager(config.audio)
        attention_filter = AttentionFilter(config.ollama)
        response_classifier = ResponseClassifier(config.ollama)
        tts = TTSEngine(config.voice, audio)
        stt = STTEngine(config.voice)
        vad = VADDetector(config.voice)

        transcript_monitor = TranscriptMonitor(event_bus, attention_filter)
        voice_router = VoiceRouter(
            event_bus=event_bus,
            tts=tts,
            stt=stt,
            vad=vad,
            audio=audio,
            response_classifier=response_classifier,
        )

        # Detect our own TTY so we can exclude it from monitoring
        own_tty = os.ttyname(0) if os.isatty(0) else None

        def _filter_sessions(sessions: list) -> list:
            """Exclude our own terminal from the session list."""
            return [s for s in sessions if s.tty != own_tty]

        # Discover running sessions
        all_sessions = discover_sessions()
        sessions = _filter_sessions(all_sessions)
        if sessions:
            click.echo(f"Monitoring {len(sessions)} Claude session(s):")
            for s in sessions:
                click.echo(f"  {s.label} (PID={s.pid}, TTY={s.tty})")
        else:
            click.echo("No Claude sessions found — will re-scan periodically.")
        if len(all_sessions) > len(sessions):
            click.echo(f"  (excluded own terminal {own_tty})")

        transcript_monitor.set_sessions(sessions)
        voice_router.set_sessions(sessions)

        click.echo("\nListening for speech. Press Ctrl+C to stop.")
        click.echo("Voice recordings saved to: ~/.local/share/verbal-direction/recordings/\n")

        await transcript_monitor.start()

        # Periodically re-discover sessions
        async def _rescan() -> None:
            while True:
                await asyncio.sleep(10)
                new_sessions = _filter_sessions(discover_sessions())
                transcript_monitor.set_sessions(new_sessions)
                voice_router.set_sessions(new_sessions)

        rescan_task = asyncio.create_task(_rescan())

        try:
            await voice_router.start()
        except KeyboardInterrupt:
            pass
        finally:
            rescan_task.cancel()
            await transcript_monitor.stop()
            await voice_router.stop()
            click.echo("\nStopped.")

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        click.echo("\nStopped.")


@cli.command()
@click.pass_context
def gui(ctx: click.Context) -> None:
    """Launch the desktop GUI dashboard."""
    from verbal_direction.ui.desktop import run_desktop_app
    run_desktop_app()


@cli.command()
def devices() -> None:
    """List available audio devices."""
    from verbal_direction.voice.audio_device import list_devices

    devs = list_devices()
    click.echo("Audio devices:")
    for dev in devs:
        kind = ""
        if dev["is_input"] and dev["is_output"]:
            kind = "[IN/OUT]"
        elif dev["is_input"]:
            kind = "[IN]"
        elif dev["is_output"]:
            kind = "[OUT]"
        click.echo(f"  {dev['index']:3d}: {dev['name']} {kind}")


if __name__ == "__main__":
    cli()
