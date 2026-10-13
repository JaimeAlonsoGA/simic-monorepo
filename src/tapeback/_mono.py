"""The single-channel pipeline: `tapeback process` on a file that is not a stereo WAV
from `tapeback start` — a call recorded by another app, a voice memo."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tapeback._gpu import free_gpu_memory, sample_gpu, telemetry_enabled
from tapeback._lazy import load_transcriber
from tapeback._stt_caps import allows_local_diarize
from tapeback._timing import stage_timer
from tapeback.audio import convert_to_mono16k, is_stereo
from tapeback.channel import identify_user_speaker
from tapeback.diarizer import Diarizer, assign_speakers, diarization_available
from tapeback.models import Segment
from tapeback.settings import Settings

# Same shape as pipeline.StatusCallback; pipeline imports this module, not the other
# way round.
StatusCallback = Callable[[str], None]


def _noop_status(msg: str) -> None:
    pass


def process_mono_file(
    audio_path: Path,
    output_dir: Path,
    settings: Settings,
    *,
    diarize: bool,
    on_status: StatusCallback = _noop_status,
) -> tuple[list[Segment], dict[str, str | float], list[Segment] | None]:
    """Process a mono/non-stereo audio file through the single-channel pipeline.

    Returns (diarized_segments, info, raw_segments).
    """
    on_status("Converting audio...")
    with stage_timer("convert", on_status):
        mono_16k_path = convert_to_mono16k(audio_path, output_dir)

    on_status("Transcribing (this may take a few minutes)...")
    with stage_timer("load model", on_status):
        transcriber = load_transcriber(settings)
    on_status(transcriber.describe())
    try:
        with (
            sample_gpu(on_status, enabled=telemetry_enabled(settings)),
            stage_timer("transcribe", on_status),
        ):
            segments, info = transcriber.transcribe(mono_16k_path, on_status=on_status)
    finally:
        # Release VRAM even when the stage raised — see the stereo path.
        del transcriber
        free_gpu_memory()

    # Raw transcript before diarization
    raw_segments = list(segments)

    stereo_for_attribution = audio_path if is_stereo(audio_path) else None
    segments_before = segments
    segments = _maybe_diarize_segments(
        segments,
        settings,
        mono_16k_path,
        stereo_for_attribution,
        diarize=diarize,
        on_status=on_status,
    )

    # _maybe_diarize_segments returns the same list reference when it skips
    # diarization — use identity to tell whether the raw section is a duplicate.
    diarized = segments is not segments_before
    return segments, info, raw_segments if diarized else None


def _maybe_diarize_segments(
    segments: list[Segment],
    settings: Settings,
    mono_16k_path: Path,
    stereo_path: Path | None,
    *,
    diarize: bool,
    on_status: StatusCallback = _noop_status,
) -> list[Segment]:
    """Run diarization if enabled, configured, and token available."""
    if not diarize or not settings.diarize:
        return segments

    if not allows_local_diarize(settings.stt_backend, settings.stt_model):
        on_status(
            "Skipping local diarization — remote STT model "
            f"'{settings.stt_model}' provides text only or remote speakers."
        )
        return segments

    if not settings.hf_token.get_secret_value():
        on_status(
            "Warning: TAPEBACK_HF_TOKEN not set, skipping diarization. "
            "See README for setup instructions."
        )
        return segments

    if not diarization_available():
        on_status(
            "Warning: pyannote-audio not installed, skipping diarization. "
            "Install with: uv pip install tapeback[diarize]"
        )
        return segments

    on_status("Diarizing speakers...")
    with stage_timer("diarize", on_status):
        diarizer = Diarizer(settings)
        diarization_segments = diarizer.diarize(mono_16k_path)

        user_speaker = None
        if stereo_path is not None:
            user_speaker = identify_user_speaker(diarization_segments, stereo_path)

        result = assign_speakers(segments, diarization_segments, user_speaker, stereo_path)
    return result
