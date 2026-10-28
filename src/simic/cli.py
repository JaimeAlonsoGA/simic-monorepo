import shutil
import subprocess
from importlib.metadata import version as _pkg_version
from pathlib import Path

import click

from simic import _aec, const
from simic._background import last_status, process_in_background
from simic._cli_gui import tray, ui
from simic._watch import CaptureWatch, EchoWatch, SilenceWatch, wait_while_recording
from simic.recorder import (
    NoActiveRecording,
    Recorder,
    detect_devices,
    session_dir,
    session_root,
)
from simic.settings import Settings, get_settings


def _echo_status(msg: str) -> None:
    """Status callback that prints to stderr via click."""
    click.echo(msg, err=True)


def _echo_remote_stt_notice(settings: Settings, *, live: bool = False) -> None:
    """Warn once when meeting audio will leave the machine via remote STT."""
    if settings.stt_backend == "local":
        return
    if live:
        msg = (
            f"Remote STT enabled: meeting audio will be uploaded to {settings.stt_backend} "
            "during the meeting and again after stop."
        )
    else:
        msg = f"Remote STT enabled: meeting audio will be uploaded to {settings.stt_backend}."
    click.echo(msg, err=True)


_WAIT_OPTION = click.option(
    "--wait",
    is_flag=True,
    help="Transcribe in the foreground once stopped (overrides SIMIC_BACKGROUND)",
)


@click.group()
@click.version_option(_pkg_version("si-mic"), prog_name="simic")
def cli() -> None:
    """Simic — Super Intelligent Mic. Meetings into Obsidian notes.

    Records system audio + microphone via PipeWire/PulseAudio,
    transcribes locally with Whisper, identifies speakers with pyannote,
    saves Markdown notes to your Obsidian vault.

    Works with any video call platform (Meet, Zoom, Teams, Telegram, Discord).

    \b
    Quick start:
      simic doctor         # check that a meeting can be recorded
      simic start          # record (Ctrl+C to stop and transcribe)
      simic status         # what is recording and being transcribed
      simic process a.mp3  # transcribe existing file
      simic ui             # the same in a window, with settings

    \b
    Configuration:
      All settings via SIMIC_* env vars or ~/.config/simic/.env or .env file.
      See: https://github.com/JaimeAlonsoGA/simic-monorepo#configuration-reference
    """


@cli.command()
@click.argument("name", required=False)
@click.option("--no-diarize", is_flag=True, help="Skip speaker diarization")
@click.option("--no-summarize", is_flag=True, help="Skip LLM summarization")
@click.option("--no-live", is_flag=True, help="Disable live transcription (overrides SIMIC_LIVE)")
@_WAIT_OPTION
def start(
    name: str | None, no_diarize: bool, no_summarize: bool, no_live: bool, wait: bool
) -> None:
    """Start recording monitor source + microphone.

    Records system audio (what you hear) and microphone (what you say),
    one compressed file per channel. Press Ctrl+C (or run 'simic stop')
    to stop: the recording is then transcribed in the background and the
    note saved to the vault, with a desktop notification when it is ready.
    'simic status' shows the progress; --wait transcribes in the foreground.

    \b
    Transcription runs after recording stops. Set SIMIC_LIVE=true to also
    transcribe in the background during recording (mid-meeting markdown).
    Live mode is off by default because it competes with the post-recording
    pipeline for GPU memory on small cards.

    \b
    Optionally provide a NAME for the output file:
      simic start "weekly-standup"
    """
    settings = get_settings()

    live_active = settings.live and not no_live
    if live_active and settings.audio_format == "mp3":
        raise click.ClickException(
            "Live transcription reads a growing WAV. Use --no-live, or set SIMIC_AUDIO_FORMAT=wav."
        )

    recorder = Recorder()
    _warn_if_disk_is_short(settings)

    monitor, mic = detect_devices(settings)
    session_name = recorder.start(settings, session_name=name)
    session_path = session_dir(settings, session_name)

    click.echo(f"Recording started: {session_name}", err=True)
    click.echo(f"Monitor: {monitor}", err=True)
    click.echo(f"Mic: {mic}", err=True)
    unavailable = (recorder.read_session() or {}).get("echo_cancel_unavailable")
    click.echo(_aec.status_line(settings.echo_cancel, unavailable), err=True)

    mic_path = session_path / f"{const.CHANNEL_MIC}.{settings.audio_format}"
    monitor_path = session_path / f"{const.CHANNEL_MONITOR}.{settings.audio_format}"

    live_transcriber = None
    if live_active:
        from simic.live import LiveTranscriber

        live_transcriber = LiveTranscriber(settings, session_name, mic_path, monitor_path)
        live_transcriber.start()
        click.echo(f"Live transcript: {live_transcriber.live_md_path}", err=True)

    _echo_remote_stt_notice(settings, live=live_active)

    click.echo("Run 'simic stop' to finish and transcribe.", err=True)
    click.echo("Or press Ctrl+C to stop and transcribe now.", err=True)

    interrupted = False
    watch = CaptureWatch({const.CHANNEL_MONITOR: monitor_path, const.CHANNEL_MIC: mic_path})
    try:
        silence = SilenceWatch({const.CHANNEL_MONITOR: monitor_path})
        wait_while_recording(recorder, watch, silence, EchoWatch(mic_path, monitor_path))
    except KeyboardInterrupt:
        interrupted = True

    # `simic stop` from another terminal claimed the session and owns the pipeline.
    if not recorder.has_session():
        return
    stopped = recorder.stopped_channels()
    if not interrupted and stopped:
        click.echo(
            f"\nA recorder stopped on its own ({', '.join(stopped)}). "
            "Transcribing what was captured.",
            err=True,
        )

    click.echo("\nStopping...", err=True)
    from simic.pipeline import stop_recording

    try:
        session = stop_recording(
            recorder, live_transcriber=live_transcriber, on_status=_echo_status
        )
    except NoActiveRecording:
        # Peer `simic stop` claimed the session between has_session() and stop().
        return
    _process_stopped(
        session,
        settings,
        wait=wait,
        diarize=not no_diarize,
        do_summarize=not no_summarize,
    )


def _process_stopped(
    session: Path, settings: Settings, *, wait: bool, diarize: bool, do_summarize: bool
) -> None:
    """Turn a stopped recording into a note: here, or in the background."""
    if wait or not settings.background:
        from simic.pipeline import process_session

        try:
            process_session(
                session,
                settings,
                diarize=diarize,
                do_summarize=do_summarize,
                on_status=_echo_status,
            )
        except KeyboardInterrupt:
            click.echo(
                f"\nAborted during processing. The recording is kept: "
                f"run 'simic recover {session.name}' to finish it.",
                err=True,
            )
        return

    process_in_background(session, diarize=diarize, do_summarize=do_summarize)
    announced = " A notification will say when the note is ready." if settings.notify else ""
    click.echo(
        f"Transcribing {session.name} in the background.{announced}\n"
        "'simic status' shows how far it has got.",
        err=True,
    )


def _warn_if_disk_is_short(settings: Settings) -> None:
    """Say so before the meeting when it may not fit, not when the disk fills."""
    free = shutil.disk_usage(session_root(settings)).free
    if free < const.BYTES_PER_MEETING_HOUR * const.MIN_FREE_MEETING_HOURS:
        minutes = free * 60 // const.BYTES_PER_MEETING_HOUR
        click.echo(
            f"Warning: only {free / const.BYTES_PER_GB:.1f} GB free for recordings — about "
            f"{minutes} minutes of meeting can be recorded and transcribed.",
            err=True,
        )


@cli.command()
@_WAIT_OPTION
def stop(wait: bool) -> None:
    """Stop recording from another terminal.

    Stops a running 'simic start' — or a recording whose 'start' is gone —
    and transcribes it like Ctrl+C would.
    """
    settings = get_settings()
    recorder = Recorder()

    from simic.pipeline import stop_recording

    try:
        session = stop_recording(recorder, on_status=_echo_status)
    except NoActiveRecording as exc:
        hint = ""
        if recorder.leftover_sessions(settings):
            hint = " Recordings not yet transcribed are finished with 'simic recover'."
        raise click.ClickException(f"{exc}{hint}") from None
    _echo_remote_stt_notice(settings)
    _process_stopped(session, settings, wait=wait, diarize=True, do_summarize=True)


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--name", default=None, help="Session name for output file")
@click.option("--no-diarize", is_flag=True, help="Skip speaker diarization")
@click.option("--no-summarize", is_flag=True, help="Skip LLM summarization")
def process(audio_file: str, name: str | None, no_diarize: bool, no_summarize: bool) -> None:
    """Process an existing audio file (mp3, m4a, ogg, wav).

    Transcribes the file, identifies speakers, generates an LLM summary,
    and saves everything as a Markdown note in your vault.

    \b
    Stereo WAV files (from simic start) use the dual-channel pipeline
    with per-channel transcription and speaker attribution.
    All other files use mono processing.

    \b
    Examples:
      simic process meeting.mp3
      simic process call.wav --name "client-call" --no-diarize
    """
    settings = get_settings()

    _echo_remote_stt_notice(settings)

    from simic.pipeline import process_file

    process_file(
        Path(audio_file),
        settings,
        name=name,
        diarize=not no_diarize,
        do_summarize=not no_summarize,
        on_status=_echo_status,
    )


@cli.command()
@click.argument("name", required=False)
@click.option("--no-diarize", is_flag=True, help="Skip speaker diarization")
@click.option("--no-summarize", is_flag=True, help="Skip LLM summarization")
def recover(name: str | None, no_diarize: bool, no_summarize: bool) -> None:
    """Transcribe recordings whose processing failed or was interrupted.

    A meeting is never thrown away before its note exists: when processing
    fails, is aborted, or the machine goes down, the recording stays behind.
    This turns it into a note, then removes it, like a normal stop.

    \b
    Without NAME, every leftover recording is processed, oldest first.
    'simic status' lists them.
      simic recover
      simic recover 2026-10-12_09-00-00
    """
    settings = get_settings()
    leftovers = Recorder().leftover_sessions(settings)
    if name is not None:
        leftovers = [path for path in leftovers if path.name == name]
        if not leftovers:
            raise click.ClickException(f"No leftover recording named {name!r}.")
    if not leftovers:
        click.echo("Nothing to recover.", err=True)
        return

    _echo_remote_stt_notice(settings)

    from simic.pipeline import process_session

    failed = []
    for session in leftovers:
        click.echo(f"Recovering {session.name}...", err=True)
        try:
            process_session(
                session,
                settings,
                diarize=not no_diarize,
                do_summarize=not no_summarize,
                on_status=_echo_status,
            )
        except Exception as exc:
            # One broken recording must not hold back the others. Its run record has
            # the traceback; the recording itself stays for another attempt.
            click.echo(f"Failed to recover {session.name}: {exc}", err=True)
            failed.append(session.name)
    if failed:
        raise click.ClickException(f"Not recovered: {', '.join(failed)}.")


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "groq", "gemini", "openrouter", "deepseek", "qwen"]),
    default=None,
    help="Override LLM provider",
)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--show-masked",
    is_flag=True,
    help="Print what would be sent to the provider and exit, without sending it",
)
def summarize(file: str, provider: str | None, model: str | None, show_masked: bool) -> None:
    """Summarize an existing transcript markdown file.

    Sends the transcript to an LLM and adds a summary section with
    brief overview, action items, and key decisions.

    \b
    Requires the llm extra: uv pip install si-mic[llm]
    and an API key — set SIMIC_LLM_API_KEY or a provider-specific
    env var (ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.).

    \b
    Examples:
      simic summarize vault/meetings/2026-03-26.md
      simic summarize transcript.md --provider gemini
      simic summarize transcript.md --show-masked   # check SIMIC_MASK_PII
    """
    settings = get_settings()

    overrides: dict[str, str] = {}
    if provider:
        overrides["llm_provider"] = provider
    if model:
        overrides["llm_model"] = model
    if overrides:
        settings = settings.model_copy(update=overrides)

    from simic.summarizer import (
        extract_transcript_from_markdown,
        format_summary_markdown,
        inject_summary_into_markdown,
        masked_preview,
    )
    from simic.summarizer import summarize as do_summarize

    path = Path(file)
    md_content = path.read_text()

    transcript = extract_transcript_from_markdown(md_content)
    if not transcript.strip():
        click.echo("Error: No transcript content found in file.", err=True)
        raise SystemExit(1)

    if show_masked:
        # Report to stderr and the payload to stdout, so it can be piped or diffed.
        masked, counts = masked_preview(transcript, settings)
        if not settings.mask_pii:
            click.echo("Masking is off (SIMIC_MASK_PII) — sent verbatim:", err=True)
        else:
            tally = ", ".join(f"{label} {n}" for label, n in sorted(counts.items()))
            click.echo(f"Masked: {tally or 'nothing matched'}", err=True)
        click.echo(masked)
        return

    click.echo("Summarizing...", err=True)
    summary = do_summarize(transcript, settings)
    summary_md = format_summary_markdown(summary)
    new_content = inject_summary_into_markdown(md_content, summary_md)
    path.write_text(new_content)
    click.echo(f"Summary added to {path}", err=True)


@cli.command()
def status() -> None:
    """Show current recording status and settings.

    Displays whether a recording is in progress, vault path,
    Whisper model, device, and available audio sources.
    """
    settings = get_settings()

    recorder = Recorder()
    session = recorder.get_session_info()

    if session:
        click.echo(f"Recording in progress: {session['session_name']}")
        click.echo(f"Started at: {session['started_at']}")
    elif recorder.has_session():
        click.echo("A recording was interrupted. Run 'simic stop' to transcribe it.")
    else:
        click.echo("Not recording.")

    in_progress = recorder.sessions_in_progress(settings)
    if in_progress:
        click.echo("\nBeing transcribed:")
        for path in in_progress:
            progress = last_status(path)
            click.echo(f"  {path.name}  {progress}" if progress else f"  {path.name}")

    leftovers = recorder.leftover_sessions(settings)
    if leftovers:
        click.echo("\nNot yet transcribed (run 'simic recover'):")
        for path in leftovers:
            click.echo(f"  {path.name}")

    click.echo(f"\nVault: {settings.vault_path}")
    click.echo(f"STT backend: {settings.stt_backend}")
    click.echo(f"STT model: {settings.stt_model}")
    if settings.stt_backend == "local":
        click.echo(f"Device: {settings.device}")
    click.echo(f"Language: {settings.language}")

    if shutil.which("pactl"):
        click.echo("\nAudio sources:")
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            click.echo(result.stdout)


_DOCTOR_MARKS = {"ok": "✓", "warn": "!", "fail": "✗"}


@cli.command()
def doctor() -> None:
    """Check that everything a meeting needs is in place.

    Tools, the sound server and its sources, the vault, disk space for
    recordings, the Whisper model, and recordings still waiting to be
    transcribed. Exits 1 when a meeting could not be recorded or saved.
    """
    from simic._doctor import run_checks

    checks = run_checks(get_settings())
    width = max(len(check.name) for check in checks)
    for check in checks:
        click.echo(f"{_DOCTOR_MARKS[check.outcome]} {check.name:<{width}}  {check.detail}")
    if any(check.outcome == "fail" for check in checks):
        raise SystemExit(1)


cli.add_command(ui)
cli.add_command(tray)
