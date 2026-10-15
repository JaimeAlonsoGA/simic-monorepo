"""`simic doctor`: everything recording, transcribing and saving a meeting depends
on, checked before a meeting rather than discovered during one.

Each check reports one of three outcomes. "fail" means a meeting cannot be recorded or
turned into a note as configured; "warn" means it can, with something worth knowing
(a first run that downloads the model, a vault that does not exist yet).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from faster_whisper.utils import download_model
from huggingface_hub.errors import LocalEntryNotFoundError

from simic import _aec, _capture, const
from simic.recorder import Recorder, detect_devices, session_root
from simic.settings import Settings
from simic.summarizer import available_providers

type Outcome = Literal["ok", "warn", "fail"]

# What the first transcription downloads for the default model.
DEFAULT_MODEL_DOWNLOAD = "about 1.6 GB"


@dataclass(frozen=True)
class Check:
    name: str
    outcome: Outcome
    detail: str


def _tools(settings: Settings) -> Check:
    needed = ["parecord", "pactl", "ffmpeg"]
    missing = [tool for tool in needed if not shutil.which(tool)]
    if missing:
        return Check(
            "Tools",
            "fail",
            f"missing {', '.join(missing)} — install pulseaudio-utils (or pipewire-pulse) "
            "and ffmpeg",
        )
    if settings.audio_format == "mp3":
        encoders = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
        ).stdout
        if "libmp3lame" not in encoders:
            return Check(
                "Tools",
                "fail",
                "this ffmpeg cannot encode MP3 (no libmp3lame) — install a full ffmpeg, "
                "or set SIMIC_AUDIO_FORMAT=wav",
            )
        return Check("Tools", "ok", "parecord, pactl, ffmpeg with MP3")
    return Check("Tools", "ok", "parecord, pactl, ffmpeg")


def _sound_server(_settings: Settings) -> Check:
    if not shutil.which("pactl"):
        return Check("Sound server", "fail", "pactl not found")
    result = subprocess.run(["pactl", "info"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return Check(
            "Sound server",
            "fail",
            "no PulseAudio/PipeWire server answers — is the session's audio service running?",
        )
    for line in result.stdout.splitlines():
        if line.startswith("Server Name:"):
            return Check("Sound server", "ok", line.split(":", 1)[1].strip())
    return Check("Sound server", "ok", "reachable")


def _sources(settings: Settings) -> Check:
    try:
        monitor, mic = detect_devices(settings)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        return Check("Sources", "fail", str(exc))
    return Check("Sources", "ok", f"monitor {monitor}, mic {mic}")


def _echo_cancel(settings: Settings) -> Check:
    """Loads the canceller for a moment, as `start` would: only that shows the module and
    its WebRTC plugin are there."""
    if not settings.echo_cancel:
        return Check("Echo cancellation", "ok", "off — on speakers the mic records the call too")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            monitor, mic = detect_devices(settings)
            canceller = _aec.start(mic, monitor, Path(tmp) / "echo-cancel.log")
        # EchoCancelUnavailable is a RuntimeError, as detect_devices' own failures are.
        except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
            return Check("Echo cancellation", "warn", f"{exc} — on speakers, use headphones")
    _capture.terminate(canceller.pid)
    _capture.wait_and_kill([canceller.pid], _capture.RECORDER_STOP_SECONDS)
    return Check("Echo cancellation", "ok", "PipeWire WebRTC canceller on the mic")


def _vault(settings: Settings) -> Check:
    vault = settings.vault_path
    if not vault.exists():
        return Check("Vault", "warn", f"{vault} does not exist yet; the first note creates it")
    if not vault.is_dir() or not os.access(vault, os.W_OK):
        return Check("Vault", "fail", f"{vault} is not a writable directory")
    return Check("Vault", "ok", str(vault))


def _disk(settings: Settings) -> Check:
    try:
        root = session_root(settings)
    except RuntimeError as exc:
        return Check("Recordings", "fail", str(exc))
    free = shutil.disk_usage(root).free
    hours = free // const.BYTES_PER_MEETING_HOUR
    detail = f"{root}, {free / const.BYTES_PER_GB:.0f} GB free (about {hours} hours of meetings)"
    if hours < const.MIN_FREE_MEETING_HOURS:
        return Check("Recordings", "warn", f"{detail} — a longer meeting would not fit")
    return Check("Recordings", "ok", detail)


def _model(settings: Settings) -> Check:
    if settings.stt_backend != "local":
        return Check("Model", "ok", f"remote STT ({settings.stt_backend}): nothing to download")
    name = settings.stt_model
    try:
        path = (
            Path(name) if Path(name).is_dir() else Path(download_model(name, local_files_only=True))
        )
    except LocalEntryNotFoundError:
        return Check(
            "Model",
            "warn",
            f"{name} is not downloaded yet; the first transcription fetches it "
            f"({DEFAULT_MODEL_DOWNLOAD} for the default model)",
        )
    except ValueError as exc:
        return Check("Model", "fail", str(exc))
    if not (path / "model.bin").exists():
        return Check(
            "Model",
            "warn",
            f"{name} was only partly downloaded; the next transcription completes it",
        )
    return Check("Model", "ok", f"{name} on {settings.device}")


def _pending(settings: Settings) -> Check:
    try:
        leftovers = Recorder().leftover_sessions(settings)
    except RuntimeError as exc:
        return Check("Pending", "fail", str(exc))
    if leftovers:
        names = ", ".join(path.name for path in leftovers)
        return Check("Pending", "warn", f"not yet transcribed: {names} — run 'simic recover'")
    return Check("Pending", "ok", "nothing waiting")


def _installed(module: str) -> bool:
    """Whether an optional extra's package is there, without importing it: pyannote
    pulls in torch, seconds of startup for a check. A dotted name imports its parent,
    which raises rather than returning None when the parent is missing."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def _speakers(settings: Settings) -> Check:
    if not settings.diarize:
        return Check("Speakers", "ok", "off")
    if not _installed("pyannote.audio"):
        return Check("Speakers", "warn", "pyannote is not installed — install si-mic[diarize]")
    if not settings.hf_token.get_secret_value():
        return Check("Speakers", "warn", "no Hugging Face token — set SIMIC_HF_TOKEN")
    return Check("Speakers", "ok", "pyannote")


def _summary(settings: Settings) -> Check:
    if not settings.summarize:
        return Check("Summary", "ok", "off")
    providers = available_providers(settings)
    if not providers:
        return Check(
            "Summary",
            "warn",
            f"no API key for {settings.llm_provider} — notes are saved without a summary",
        )
    package = "anthropic" if providers[0] == "anthropic" else "openai"
    if not _installed(package):
        return Check("Summary", "warn", f"{package} is not installed — install si-mic[llm]")
    return Check("Summary", "ok", ", then ".join(providers))


CHECKS: tuple[Callable[[Settings], Check], ...] = (
    _tools,
    _sound_server,
    _sources,
    _echo_cancel,
    _vault,
    _disk,
    _model,
    _speakers,
    _summary,
    _pending,
)


def run_checks(settings: Settings) -> list[Check]:
    return [check(settings) for check in CHECKS]
