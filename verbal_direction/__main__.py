"""CLI entry point for verbal-direction."""

from __future__ import annotations

import asyncio
import logging
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
    """verbal-direction — Voice-controlled Claude Code session manager."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config.load(Path(config_path) if config_path else None)

    # If no subcommand, start the voice listener
    if ctx.invoked_subcommand is None:
        ctx.invoke(listen)


@cli.command()
@click.argument("name")
@click.argument("directory", type=click.Path(exists=True))
@click.option("--prompt", "-p", help="Initial prompt for the session")
@click.argument("initial_prompt", nargs=-1)
@click.pass_context
def launch(ctx: click.Context, name: str, directory: str, prompt: str | None, initial_prompt: tuple[str, ...]) -> None:
    """Launch a new Claude Code session.

    Usage: vd launch <name> <directory> -- "initial prompt"
    """
    # Combine prompt sources
    full_prompt = prompt or " ".join(initial_prompt) if initial_prompt else prompt

    async def _launch() -> None:
        from verbal_direction.core.event_bus import EventBus
        from verbal_direction.core.session_manager import SessionManager

        config = ctx.obj["config"]
        event_bus = EventBus()
        manager = SessionManager(event_bus)

        session = await manager.launch(
            name=name,
            directory=str(Path(directory).resolve()),
            initial_prompt=full_prompt,
        )
        click.echo(f"Launched session '{name}' in {directory}")
        if full_prompt:
            click.echo(f"Initial prompt: {full_prompt}")
        click.echo(f"Status: {session.state.status_display}")

    asyncio.run(_launch())


@cli.command("list")
@click.pass_context
def list_sessions(ctx: click.Context) -> None:
    """List active Claude Code sessions."""
    # In practice this reads from a shared state file/socket
    # For now, show a placeholder
    click.echo("Active sessions:")
    click.echo("  (No active sessions — launch one with 'vd launch <name> <dir>')")


@cli.command()
@click.argument("name")
@click.pass_context
def kill(ctx: click.Context, name: str) -> None:
    """Kill a Claude Code session."""
    click.echo(f"Killing session '{name}'...")


@cli.command()
@click.argument("name")
@click.pass_context
def pause(ctx: click.Context, name: str) -> None:
    """Pause voice monitoring for a session."""
    click.echo(f"Paused voice monitoring for '{name}'")


@cli.command()
@click.argument("name")
@click.pass_context
def resume(ctx: click.Context, name: str) -> None:
    """Resume voice monitoring for a session."""
    click.echo(f"Resumed voice monitoring for '{name}'")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show TUI dashboard."""
    from verbal_direction.ui.dashboard import VDDashboard

    app = VDDashboard()
    app.run()


@cli.command()
@click.pass_context
def listen(ctx: click.Context) -> None:
    """Start the voice listener (main loop)."""
    config = ctx.obj["config"]

    async def _run() -> None:
        from verbal_direction.core.event_bus import EventBus
        from verbal_direction.core.session_manager import SessionManager
        from verbal_direction.core.output_monitor import OutputMonitor
        from verbal_direction.core.response_dispatcher import ResponseDispatcher
        from verbal_direction.intelligence.attention_filter import AttentionFilter
        from verbal_direction.intelligence.response_classifier import ResponseClassifier
        from verbal_direction.voice.audio_device import AudioDeviceManager
        from verbal_direction.voice.tts import TTSEngine
        from verbal_direction.voice.stt import STTEngine
        from verbal_direction.voice.vad import VADDetector
        from verbal_direction.voice.voice_router import VoiceRouter

        # Initialize components
        event_bus = EventBus()
        session_manager = SessionManager(event_bus)
        audio = AudioDeviceManager(config.audio)
        attention_filter = AttentionFilter(config.ollama)
        response_classifier = ResponseClassifier(config.ollama)
        tts = TTSEngine(config.voice, audio)
        stt = STTEngine(config.voice)
        vad = VADDetector(config.voice)

        output_monitor = OutputMonitor(event_bus, attention_filter)
        response_dispatcher = ResponseDispatcher(event_bus, session_manager)
        voice_router = VoiceRouter(
            event_bus=event_bus,
            session_manager=session_manager,
            tts=tts,
            stt=stt,
            vad=vad,
            audio=audio,
            response_classifier=response_classifier,
        )

        click.echo("verbal-direction voice listener starting...")
        click.echo("Listening for speech. Press Ctrl+C to stop.")

        await output_monitor.start()
        await response_dispatcher.start()

        try:
            await voice_router.start()
        except KeyboardInterrupt:
            pass
        finally:
            await voice_router.stop()
            await output_monitor.stop()
            await response_dispatcher.stop()
            await session_manager.kill_all()
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
