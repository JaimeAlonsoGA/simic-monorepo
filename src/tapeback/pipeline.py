"""Processing pipeline shared between CLI and tray."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tapeback.live import LiveTranscriber

from tapeback import const
from tapeback._gpu import free_gpu_memory, sample_gpu, telemetry_enabled
from tapeback._lazy import load_transcriber
from tapeback._mono import process_mono_file
from tapeback._notify import note_failed, note_ready
from tapeback._runlog import run_log
from tapeback._stt_caps import allows_local_diarize
from tapeback._timing import stage_timer
from tapeback.audio import gate_wav_inactive, is_stereo, merge_channels, split_channels_16k
from tapeback.channel import (
    classify_segment_by_channel,
    filter_silent_segments,
    load_stereo_channels,
    split_on_silence,
)
from tapeback.diarizer import (
    Diarizer,
    assign_speakers,
    diarization_available,
    merge_channel_segments,
    merge_similar_speakers,
)
from tapeback.formatter import TranscriptMeta, format_markdown
from tapeback.models import Segment
from tapeback.recorder import (
    Recorder,
    SessionBusy,
    processing,
    session_channels,
    transcription_turn,
    validate_session_name,
)
from tapeback.settings import Settings
from tapeback.summarizer import maybe_summarize
from tapeback.vault import remove_live_markdown, save_audio_to_vault, save_markdown_to_vault

StatusCallback = Callable[[str], None]

_WAITING_FOR_TURN = "Waiting for another recording to finish transcribing..."


def _noop_status(msg: str) -> None:
    pass


def stop_recording(
    recorder: Recorder,
    *,
    live_transcriber: LiveTranscriber | None = None,
    on_status: StatusCallback = _noop_status,
) -> Path:
    """Stop capturing and return the session directory, ready to be processed.

    If a live_transcriber is active, stops it first to free GPU memory
    before the full pipeline creates its own Whisper model.
    """
    if live_transcriber is not None:
        on_status("Stopping live transcription...")
        live_transcriber.stop()

    on_status("Stopping recording...")
    monitor_path, _mic_path = recorder.stop()
    return monitor_path.parent


def stop_and_process(
    recorder: Recorder,
    settings: Settings,
    *,
    live_transcriber: LiveTranscriber | None = None,
    diarize: bool = True,
    do_summarize: bool = True,
    on_status: StatusCallback = _noop_status,
) -> Path:
    """Stop recording and turn the session into a note. Returns the note's path."""
    session = stop_recording(recorder, live_transcriber=live_transcriber, on_status=on_status)
    return process_session(
        session,
        settings,
        diarize=diarize,
        do_summarize=do_summarize,
        on_status=on_status,
    )


def process_session(
    session: Path,
    settings: Settings,
    *,
    diarize: bool = True,
    do_summarize: bool = True,
    on_status: StatusCallback = _noop_status,
) -> Path:
    """Transcribe a recorded session directory into a note. Returns the note's path.

    The directory is removed once the note is written. A run that fails leaves it in
    place, so `tapeback recover` can process it again. Raises SessionBusy when another
    process is already at it.
    """
    session_name = session.name
    try:
        md_path = _process_session(
            session, settings, diarize=diarize, do_summarize=do_summarize, on_status=on_status
        )
    except SessionBusy:
        # Not a failure: whoever holds the session announces its note.
        raise
    except Exception as exc:
        note_failed(
            settings,
            session_name,
            exc,
            f"The recording is kept: run 'tapeback recover {session_name}'.",
        )
        raise
    note_ready(settings, md_path)
    return md_path


def _process_session(
    session: Path,
    settings: Settings,
    *,
    diarize: bool,
    do_summarize: bool,
    on_status: StatusCallback,
) -> Path:
    session_name = session.name
    monitor_path, mic_path = session_channels(session)

    with (
        processing(session),
        run_log(session_name, settings, on_status) as report,
        transcription_turn(lambda: report(_WAITING_FOR_TURN)),
    ):
        report("Merging audio channels...")
        with stage_timer("merge", report):
            stereo_path = merge_channels(monitor_path, mic_path, session)

        audio_rel_path = _archive_audio(stereo_path, settings, session_name, report)

        segments, info, raw_segments = process_stereo_file(
            stereo_path, session, settings, diarize=diarize, on_status=report
        )

        md_path = _write_note(segments, info, raw_segments, settings, session_name, audio_rel_path)
        report(f"Saved: {md_path}")
        # A live transcript of this meeting is superseded by the note.
        remove_live_markdown(settings, session_name)

        if do_summarize:
            _maybe_summarize(md_path, settings, report)

        shutil.rmtree(session, ignore_errors=True)
    return md_path


def process_file(
    audio_path: Path,
    settings: Settings,
    *,
    name: str | None = None,
    diarize: bool = True,
    do_summarize: bool = True,
    on_status: StatusCallback = _noop_status,
) -> Path:
    """Process an existing audio file. Returns path to saved markdown."""
    if name is None:
        name = audio_path.stem
    validate_session_name(name)
    try:
        md_path = _process_file(
            audio_path,
            settings,
            name,
            diarize=diarize,
            do_summarize=do_summarize,
            on_status=on_status,
        )
    except Exception as exc:
        note_failed(settings, name, exc, f"Run 'tapeback process {audio_path}' again.")
        raise
    note_ready(settings, md_path)
    return md_path


def _process_file(
    audio_path: Path,
    settings: Settings,
    name: str,
    *,
    diarize: bool,
    do_summarize: bool,
    on_status: StatusCallback,
) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="tapeback_"))

    with (
        run_log(name, settings, on_status) as report,
        transcription_turn(lambda: report(_WAITING_FOR_TURN)),
    ):
        audio_rel_path = _archive_audio(audio_path, settings, name, report)

        if is_stereo(audio_path):
            report("Stereo file detected, using dual-channel pipeline...")
            segments, info, raw_segments = process_stereo_file(
                audio_path, tmp_dir, settings, diarize=diarize, on_status=report
            )
        else:
            segments, info, raw_segments = process_mono_file(
                audio_path, tmp_dir, settings, diarize=diarize, on_status=report
            )

        md_path = _write_note(segments, info, raw_segments, settings, name, audio_rel_path)
        report(f"Saved: {md_path}")

        if do_summarize:
            _maybe_summarize(md_path, settings, report)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return md_path


def _archive_audio(
    audio_path: Path, settings: Settings, name: str, report: StatusCallback
) -> str | None:
    """Copy the recording into the vault when it is kept. Returns its vault-relative
    link for the note, or None when the audio is not kept."""
    if not settings.keep_audio:
        return None
    audio_dest = save_audio_to_vault(audio_path, settings, name)
    report(f"Audio saved: {audio_dest}")
    return audio_dest.resolve().relative_to(settings.vault_path.resolve()).as_posix()


def _write_note(
    segments: list[Segment],
    info: dict[str, str | float],
    raw_segments: list[Segment] | None,
    settings: Settings,
    name: str,
    audio_rel_path: str | None,
) -> Path:
    markdown = format_markdown(
        segments=segments,
        meta=TranscriptMeta(
            session_name=name,
            audio_rel_path=audio_rel_path,
            duration_seconds=float(info.get("duration", 0.0)),
            language=str(info.get("language", settings.language)),
            partial=bool(info.get("partial")),
        ),
        raw_segments=raw_segments,
    )
    return save_markdown_to_vault(markdown, settings, name)


def process_stereo_file(
    stereo_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    diarize: bool,
    on_status: StatusCallback = _noop_status,
) -> tuple[list[Segment], dict[str, str | float], list[Segment] | None]:
    """Process a stereo WAV through the dual-channel pipeline.

    Returns (diarized_segments, info, raw_segments).
    raw_segments have basic You/Other attribution (channel-based, no diarization).
    """
    with stage_timer("load channels", on_status):
        mic_raw, monitor_raw, raw_sr = load_stereo_channels(stereo_path)

    on_status("Splitting channels...")
    with stage_timer("split", on_status):
        mic_16k, monitor_16k = split_channels_16k(stereo_path, output_dir)

    if settings.gate_mic_silence:
        # Silence the mic where the user only listens, so Whisper doesn't loop on it.
        with stage_timer("gate mic", on_status):
            gate_wav_inactive(mic_16k, mic_raw, monitor_raw, raw_sr)

    on_status("Transcribing (this may take a few minutes)...")
    with stage_timer("load model", on_status):
        transcriber = load_transcriber(settings)
    on_status(transcriber.describe())
    try:
        with sample_gpu(on_status, enabled=telemetry_enabled(settings)):
            mic_segments, monitor_segments, info = transcriber.transcribe_stereo(
                mic_16k, monitor_16k, on_status=on_status
            )
    finally:
        # Release VRAM even when the stage raised, so a failure here does not starve
        # the diarizer that runs next.
        del transcriber
        free_gpu_memory()

    mic_segments = split_on_silence(
        mic_segments,
        mic_raw,
        raw_sr,
        settings.pause_threshold,
        monitor_samples=monitor_raw,
    )

    mic_segments = filter_silent_segments(mic_segments, mic_raw, raw_sr)
    monitor_segments = filter_silent_segments(monitor_segments, monitor_raw, raw_sr)

    # Drop mic segments where monitor is louder (headphone bleed, not real speech)
    mic_segments = [
        s
        for s in mic_segments
        if classify_segment_by_channel(s.start, s.end, mic_raw, monitor_raw, raw_sr) != "monitor"
    ]

    # Raw transcript: basic channel attribution only (You vs Other)
    raw_monitor = [
        Segment(start=s.start, end=s.end, text=s.text, words=s.words, speaker=const.SPEAKER_OTHER)
        for s in monitor_segments
    ]
    raw_segments = merge_channel_segments(mic_segments, raw_monitor)

    want_diarize = diarize and settings.diarize
    if want_diarize and not allows_local_diarize(settings.stt_backend, settings.stt_model):
        on_status(
            "Skipping local diarization — remote STT model "
            f"'{settings.stt_model}' provides text only or remote speakers."
        )
        want_diarize = False

    diarized = False
    if want_diarize and settings.hf_token.get_secret_value():
        if not diarization_available():
            on_status(
                "Warning: pyannote-audio not installed, skipping diarization. "
                "Install with: uv pip install tapeback[diarize]"
            )
        else:
            on_status("Diarizing speakers...")
            with stage_timer("diarize", on_status):
                diarizer = Diarizer(settings)
                diarization_segments = diarizer.diarize(monitor_16k)
                if settings.spectral_merge_threshold > 0:
                    diarization_segments = merge_similar_speakers(
                        diarization_segments,
                        monitor_raw,
                        raw_sr,
                        similarity_threshold=settings.spectral_merge_threshold,
                    )
                monitor_segments = assign_speakers(monitor_segments, diarization_segments)
            diarized = True

    # Remote diarize already set speaker labels; channel fallback only when unset.
    if not diarized and monitor_segments and monitor_segments[0].speaker is None:
        monitor_segments = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text,
                words=s.words,
                speaker=const.SPEAKER_OTHER,
            )
            for s in monitor_segments
        ]

    segments = merge_channel_segments(mic_segments, monitor_segments)
    # Without diarization, raw_segments == segments — skip the duplicate section.
    return segments, info, raw_segments if diarized else None


def _maybe_summarize(md_path: Path, settings: Settings, on_status: StatusCallback) -> None:
    """Run summarization if available."""
    on_status("Summarizing...")
    with stage_timer("summarize", on_status):
        maybe_summarize(md_path, settings)
