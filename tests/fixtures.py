import contextlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import wave
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from click.testing import CliRunner
from pydantic import SecretStr

from simic import _watch
from simic.recorder import Recorder
from simic.settings import Settings
from simic.summarizer import _PROVIDER_ENV_VARS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyannote_available() -> bool:
    """Whether pyannote is installed, without importing it. The import pulls in torch —
    six seconds on a small laptop — and every process that imports this module paid it,
    including the `simic start` that ctrl_c_in_terminal runs: its Ctrl+C then landed
    in the middle of the import. A dotted name imports its parent, which raises rather
    than returning None when the parent is missing."""
    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except ModuleNotFoundError:
        return False


requires_pyannote = pytest.mark.skipif(
    not _pyannote_available(),
    reason="pyannote-audio not installed (install si-mic[diarize])",
)


requires_dbus = pytest.mark.skipif(
    importlib.util.find_spec("dbus_fast") is None,
    reason="dbus-fast not installed (install si-mic[tray])",
)

# --- pytest fixtures ---


@pytest.fixture(autouse=True)
def isolate_user_dirs(monkeypatch, tmp_path_factory):
    """Keep everything the suite writes out of the developer's home.

    Session state and recordings default to XDG_STATE_HOME, run records and the resume
    cache to XDG_DATA_HOME, all resolved at call time. Without this every test run wrote
    into the real ~/.local/state and ~/.local/share/simic — and run records and the
    resume cache are bounded (200 entries each), so the suite's entries evicted the
    developer's real ones. XDG variables rather than SIMIC_* paths:
    isolate_settings_sources removes every SIMIC_ variable.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path_factory.mktemp("state")))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path_factory.mktemp("data")))


@pytest.fixture(autouse=True)
def isolate_settings_sources(monkeypatch):
    """Cut Settings off from the developer's real configuration.

    Settings reads ~/.config/simic/.env and ./.env, so without this every test
    that constructs Settings() inherits whatever the machine happens to have
    configured. That makes results machine-dependent: a developer with
    SIMIC_CHUNK_LENGTH=2 in their user config sees different values than CI.
    Isolation is by construction — no test should depend on the ambient config.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", ())
    for name in [key for key in os.environ if key.startswith("SIMIC_")]:
        monkeypatch.delenv(name, raising=False)
    # Provider keys are read under their own names, not the SIMIC_ prefix, so the
    # sweep above never touched them: a developer with ANTHROPIC_API_KEY exported gave
    # every test a live provider chain, and one forgotten mock would bill the vendor.
    # Driven off the production mapping on purpose — a new provider must be isolated
    # the moment it is added, not the next time someone remembers this list.
    for env_var in _PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)
    # Same rule applied to the machine's thermal state: without this every test that
    # builds a Transcriber on "cuda" polls the real GPU and can sit in the clamp wait,
    # so the suite's runtime would depend on how hot the developer's laptop is.
    monkeypatch.setenv("SIMIC_THERMAL_CLAMP_CHECK", "false")
    # Likewise for process isolation: a test that mocks WhisperModel would otherwise
    # spawn a real worker, which loads a real model and ignores the mock entirely.
    # Tests for the isolated path opt back in and mock the subprocess instead.
    monkeypatch.setenv("SIMIC_ISOLATE_TRANSCRIPTION", "false")
    # And for the desktop: a developer's real notify-send would announce every note the
    # suite writes. Tests of notifications opt back in with a fake notify-send.
    monkeypatch.setenv("SIMIC_NOTIFY", "false")
    # And for processing: a `start` or `stop` in the background would hand the note to
    # a real `python -m simic` child that none of the test's fakes reach. Tests of
    # the background path opt back in.
    monkeypatch.setenv("SIMIC_BACKGROUND", "false")


@pytest.fixture
def tmp_vault(tmp_path):
    """Temporary Obsidian vault for tests."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def vault_env(tmp_vault, monkeypatch):
    """tmp_vault + SIMIC_VAULT_PATH env var — for CLI tests that call get_settings()."""
    monkeypatch.setenv("SIMIC_VAULT_PATH", str(tmp_vault))
    return tmp_vault


@pytest.fixture
def settings(tmp_vault):
    """Settings with temporary vault."""
    return Settings(vault_path=tmp_vault)


@pytest.fixture
def session_wavs(tmp_path):
    """Factory: create session_dir with mic.wav + monitor.wav sine WAVs.

    Returns callable(session_name, duration=2.0) -> (session_dir, monitor_wav, mic_wav).
    """

    def _create(session_name: str, duration: float = 2.0) -> tuple[Path, Path, Path]:
        session_dir = tmp_path / session_name
        session_dir.mkdir()
        monitor_wav = session_dir / "monitor.wav"
        mic_wav = session_dir / "mic.wav"
        create_mono_wav(monitor_wav, duration=duration, sample_rate=48000, amplitude=0.5)
        create_mono_wav(mic_wav, duration=duration, sample_rate=48000, amplitude=0.5)
        return session_dir, monitor_wav, mic_wav

    return _create


@pytest.fixture
def recorder(tmp_path):
    """Recorder with temporary state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return Recorder(state_dir=state_dir)


@pytest.fixture
def session_file(recorder):
    """Path to session file."""
    return recorder.session_file


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def stereo_wav(tmp_path):
    """Factory fixture: returns callable(segments_spec) -> Path to stereo WAV."""

    def _create(segments_spec, sample_rate=16000):
        path = tmp_path / "stereo.wav"
        create_stereo_wav_segments(path, sample_rate, segments_spec)
        return path

    return _create


@pytest.fixture
def e2e_settings(tmp_path):
    """Settings for e2e tests, with the HF token from HF_TOKEN.

    Only HF_TOKEN: isolate_settings_sources has already removed every SIMIC_
    variable by the time this runs, so SIMIC_HF_TOKEN could never reach it.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    return Settings(vault_path=vault, hf_token=SecretStr(os.environ.get("HF_TOKEN", "")))


@pytest.fixture
def summarize_settings(tmp_vault):
    """Settings pre-configured for summarizer tests."""
    return Settings(
        vault_path=tmp_vault,
        llm_provider="anthropic",
        llm_api_key=SecretStr("sk-ant-test-key"),
    )


@pytest.fixture
def tray_app(settings):
    """TrayApp with mocked SNI / DBusMenu / recorder — exercises state machine
    without ever touching the session bus."""
    from simic.tray import TrayApp  # noqa: PLC0415 — optional dep

    app = TrayApp(settings)
    app._sni = MagicMock()
    app._menu = MagicMock()
    app._recorder = MagicMock()
    app._recorder.is_recording.return_value = False
    app._recorder.get_session_info.return_value = None
    return app


def _click_quit(menu) -> None:
    """Click Quit the way a tray host does: find it in the layout, send `clicked`."""
    _revision, (_root, _props, children) = menu.GetLayout.__wrapped__(menu, 0, -1, [])
    quit_id = next(c.value[0] for c in children if c.value[1].get("label", "").value == "Quit")
    menu.Event.__wrapped__(menu, quit_id, "clicked", None, 0)


@pytest.fixture
def session_bus(monkeypatch):
    """The desktop's D-Bus session bus, for `simic tray` to run against. The
    StatusNotifierWatcher is there — the GNOME AppIndicator extension or Plasma
    provides it — unless the test sets `watcher_gone`. Once the tray has tried to
    register, the user clicks Quit in its menu. Returns the fake bus; `registered` is
    the watcher's RegisterStatusNotifierItem."""
    bus = MagicMock(unique_name=":1.42", watcher_gone=False)
    watcher = AsyncMock()
    bus.get_proxy_object.return_value.get_interface.return_value = watcher
    bus.registered = watcher.call_register_status_notifier_item
    exported: dict[str, object] = {}
    bus.export.side_effect = exported.__setitem__

    async def introspect(_name, _path):
        _click_quit(exported["/MenuBar"])
        if bus.watcher_gone:
            # What the bus answers when nothing owns the watcher's name.
            raise RuntimeError("org.freedesktop.DBus.Error.ServiceUnknown")
        return MagicMock()

    bus.introspect = introspect
    connect = AsyncMock(return_value=bus)
    monkeypatch.setattr("simic.tray.MessageBus", lambda **_kw: MagicMock(connect=connect))
    return bus


@pytest.fixture
def e2e_output_dir(tmp_path):
    """Temporary directory for intermediate pipeline files."""
    d = tmp_path / "e2e_output"
    d.mkdir()
    return d


@pytest.fixture
def release_tree(tmp_path):
    """A tree where every copy of the version agrees on 1.2.3 — what
    scripts/release_from_tag.py accepts. Tests break one copy at a time."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "si-mic"\nversion = "1.2.3"\n')
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "click"\nversion = "8.1.7"\n\n'
        '[[package]]\nname = "si-mic"\nversion = "1.2.3"\nsource = { editable = "." }\n'
    )
    for directory in ("packaging", "packaging/simic-llm"):
        (tmp_path / directory).mkdir(exist_ok=True)
        (tmp_path / directory / "PKGBUILD").write_text("pkgname=simic\npkgver=1.2.3\npkgrel=1\n")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [1.2.3] — 2026-10-05\n\n### Fixed\n"
        "- The fix.\n\n## [1.2.2] — 2026-10-01\n\n### Added\n- The feature.\n"
    )
    (tmp_path / "README.md").write_text(
        "wget https://github.com/JaimeAlonsoGA/simic-monorepo/releases/download/v1.2.3/simic_1.2.3_amd64.deb\n"
        "sudo apt install ./simic_1.2.3_amd64.deb\n"
        "sudo apt install ./simic-tray_1.2.3_all.deb   # tray icon\n"
    )
    return tmp_path


# `parecord` as simic meets it: called with --device=... and an output path, it writes
# a WAV there and records until SIGTERM. Each call logs its PID, its device and the mode
# of the directory it was told to write into — after the audio is in place, so a logged
# call is a recorder that has captured something. The traps come first: bash runs them
# only between commands, so a SIGTERM during the copy still leaves a whole file. Each
# logs the signal that stopped the recorder, as real parecord exits on either. The
# loop rather than one long `sleep`, which as a background job would outlive the script.
_FAKE_PARECORD = """#!/usr/bin/env bash
trap 'echo "$$ TERM" >> "$FAKE_PARECORD_LOG.stops"; exit 0' TERM
trap 'echo "$$ INT" >> "$FAKE_PARECORD_LOG.stops"; exit 130' INT
device="" out="" rate=48000 raw=0
for arg in "$@"; do
  case "$arg" in
    --device=*) device="${arg#--device=}" ;;
    --rate=*) rate="${arg#--rate=}" ;;
    --raw) raw=1 ;;
    -*) ;;
    *) out="$arg" ;;
  esac
done
src="$FAKE_PARECORD_AUDIO/$device.wav"
if [ "$raw" = 1 ]; then
  ffmpeg -v error -i "$src" -f s16le -ar "$rate" -ac 1 pipe:1
  dir="$(pwd)"
else
  cp "$src" "$out"
  dir="$(dirname "$out")"
fi
printf '%s %s %s\\n' "$$" "$device" "$(stat -c %a "$dir")" >> "$FAKE_PARECORD_LOG"
while :; do sleep 0.05; done
"""


@dataclass(frozen=True)
class ParecordCall:
    pid: int
    device: str
    dir_mode: str
    # The signal that stopped it ("TERM", "INT"); None while it runs or when SIGKILLed.
    stopped_by: str | None


@pytest.fixture
def fake_parecord(tmp_path, monkeypatch):
    """Record a meeting without PulseAudio: a fake `parecord` on PATH, and the two
    sources pointed at it. The mic speaks for the first 2 s, the other side for the next
    2 s, at the recording rate. Returns a callable listing the recorders started."""
    audio = tmp_path / "fake-audio"
    audio.mkdir()
    create_mono_wav_segments(audio / "fake.mic.wav", 48000, [(2.0, 0.8), (2.0, 0.0)])
    create_mono_wav_segments(audio / "fake.monitor.wav", 48000, [(2.0, 0.0), (2.0, 0.8)])
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    script = bin_dir / "parecord"
    script.write_text(_FAKE_PARECORD)
    script.chmod(0o755)
    log = tmp_path / "parecord.log"
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_PARECORD_AUDIO", str(audio))
    monkeypatch.setenv("FAKE_PARECORD_LOG", str(log))
    # Named sources skip the @DEFAULT_MONITOR@ probe, which needs a sound server.
    monkeypatch.setenv("SIMIC_MONITOR_SOURCE", "fake.monitor")
    monkeypatch.setenv("SIMIC_MIC_SOURCE", "fake.mic")

    stops = Path(f"{log}.stops")

    def calls() -> list[ParecordCall]:
        lines = log.read_text().splitlines() if log.exists() else []
        stop_lines = stops.read_text().splitlines() if stops.exists() else []
        stopped = dict(line.split() for line in stop_lines)
        return [
            ParecordCall(int(pid), device, mode, stopped.get(pid))
            for pid, device, mode in (line.split() for line in lines)
        ]

    return calls


# `notify-send` as a desktop has it: each call appends its arguments, one per line,
# then a separator — a body may itself span lines.
_FAKE_NOTIFY_SEND = """#!/usr/bin/env bash
printf '%s\\n' "$@" >> "$FAKE_NOTIFY_LOG"
echo '---' >> "$FAKE_NOTIFY_LOG"
"""


@pytest.fixture
def fake_notify_send(tmp_path, monkeypatch):
    """Notifications on, delivered to a fake notify-send. Returns a callable listing
    them as (title, body)."""
    bin_dir = tmp_path / "notify-bin"
    bin_dir.mkdir()
    script = bin_dir / "notify-send"
    script.write_text(_FAKE_NOTIFY_SEND)
    script.chmod(0o755)
    log = tmp_path / "notify.log"
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_NOTIFY_LOG", str(log))
    monkeypatch.setenv("SIMIC_NOTIFY", "true")

    def sent() -> list[tuple[str, str]]:
        calls = log.read_text().split("---\n")[:-1] if log.exists() else []
        result = []
        for call in calls:
            app_name, icon, title, *body = call.rstrip("\n").split("\n")
            assert app_name == "--app-name=Simic"
            assert icon.endswith("/icons/hicolor/256x256/apps/io.github.jaimealonsoga.simic.png")
            assert Path(icon.removeprefix("--icon=")).is_file()
            result.append((title, "\n".join(body)))
        return result

    return sent


@pytest.fixture
def system_timezone():
    """Set the machine's time zone for one test: `system_timezone("Asia/Tokyo")`.

    The C library reads TZ only on tzset(), so the zone is applied with it — and
    restored with it too: a monkeypatched TZ would be put back after this fixture's
    teardown, too late for the tzset that has to follow it."""
    original = os.environ.get("TZ")

    def _set(zone: str) -> None:
        os.environ["TZ"] = zone
        time.tzset()

    yield _set
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    time.tzset()


@pytest.fixture
def cpu_only(monkeypatch):
    """No GPU for a flow whose model is a stand-in: CUDA would only add a real nvidia-smi
    poll, and make the run depend on the machine it happens to be on."""
    monkeypatch.setenv("SIMIC_DEVICE", "cpu")
    monkeypatch.setenv("SIMIC_GPU_TELEMETRY", "false")


@pytest.fixture
def interpreter_without_spawn_setsid(monkeypatch):
    """os.posix_spawn as python-build-standalone has it — the interpreter of the .deb and
    of uv. It is built against a glibc older than POSIX_SPAWN_SETSID, so `setsid=True`
    raises NotImplementedError; everything else spawns for real."""
    real_spawn = os.posix_spawn

    def spawn(*args, setsid=False, **kwargs):
        if setsid:
            raise NotImplementedError("setsid is not supported on this platform")
        return real_spawn(*args, **kwargs)

    monkeypatch.setattr(os, "posix_spawn", spawn)


# How long the simulated meeting may take to get both recorders writing. Reached in
# milliseconds on an idle machine; the bound is for a loaded one — a gate running the
# Docker stands beside the suite once took longer than 5 s.
_RECORDERS_START_SECONDS = 30.0


_RECORDERS_LATE = f"the fake recorders did not start in {_RECORDERS_START_SECONDS} s"


def _wait_for_recorders(fake_parecord, sleep) -> bool:
    """Whether both fake recorders captured their audio in time. A meeting stopped
    before that is a different scenario, and its flow fails somewhere that says nothing
    of why — so a late start is reported by name. Not raised from here: the stop still
    has to happen, because simic's own stop is what ends the recorders."""
    deadline = time.monotonic() + _RECORDERS_START_SECONDS
    while len(fake_parecord()) < 2:
        if time.monotonic() >= deadline:
            return False
        sleep(0.01)
    return True


@pytest.fixture
def ctrl_c_while_recording(monkeypatch, fake_parecord):
    """The user presses Ctrl+C while `simic start` waits for the meeting to end.

    The interrupt arrives where the real one would — in the command's wait loop — and
    only once both recorders have captured their audio: a person stops a meeting that
    has been recorded, not one that has not started yet. Only simic._watch's `time` is
    replaced; the recorder's own waits stay real."""
    real_sleep = time.sleep
    late: list[bool] = []

    def _interrupt(_seconds: float) -> None:
        if not _wait_for_recorders(fake_parecord, real_sleep):
            late.append(True)
        raise KeyboardInterrupt

    monkeypatch.setattr(_watch, "time", SimpleNamespace(sleep=_interrupt))
    yield
    if late:
        pytest.fail(_RECORDERS_LATE)


@pytest.fixture
def mic_recorder_dies(monkeypatch, fake_parecord):
    """The microphone's recorder exits mid-meeting (a headset unplugged, PipeWire
    restarted) while `simic start` waits. It is SIGKILLed, so no handler of its own
    runs, once both recorders have captured their audio."""
    real_sleep = time.sleep
    late: list[bool] = []

    def _kill_mic(seconds: float) -> None:
        if not _wait_for_recorders(fake_parecord, real_sleep):
            late.append(True)
            raise KeyboardInterrupt
        mic = next(call for call in fake_parecord() if call.device == "fake.mic")
        with contextlib.suppress(ProcessLookupError):
            os.kill(mic.pid, signal.SIGKILL)
        real_sleep(seconds)

    monkeypatch.setattr(_watch, "time", SimpleNamespace(sleep=_kill_mic))
    yield
    if late:
        pytest.fail(_RECORDERS_LATE)


# `simic start` as a terminal runs it, in a process of its own: the in-process
# runner cannot receive a signal the way a terminal delivers one. The Whisper model is
# the flow's two-sentence meeting, and Settings is cut off from the developer's .env
# files the way isolate_settings_sources cuts off this process.
_START_IN_TERMINAL = """
import sys
from unittest.mock import patch

sys.path.insert(0, sys.argv[1])

from simic.cli import cli
from simic.settings import Settings
from tests.fixtures import mock_whisper_by_channel

Settings.model_config["env_file"] = ()
model = mock_whisper_by_channel(
    mic_segments=[(0.0, 2.0, "I will send the report.")],
    monitor_segments=[(2.0, 4.0, "Thanks, see you Friday.")],
)
with patch("simic.transcriber.WhisperModel", return_value=model):
    cli(sys.argv[2:])
"""

# Bounds a run that should finish in seconds; never asserted on.
_TERMINAL_RUN_SECONDS = 60.0


@dataclass(frozen=True)
class TerminalRun:
    returncode: int
    output: str


@pytest.fixture
def ctrl_c_in_terminal(tmp_path, fake_parecord, cpu_only):
    """Press Ctrl+C in the terminal running `simic start ARGS`.

    A terminal sends SIGINT to its whole foreground process group, not to simic
    alone: every child that shares the group receives it too. So the command runs as a
    session leader, as a shell's foreground job is, and the signal goes to the group
    once both recorders have captured their audio."""

    def run(args: list[str]) -> TerminalRun:
        proc = subprocess.Popen(
            [sys.executable, "-c", _START_IN_TERMINAL, str(REPO_ROOT), *args],
            cwd=tmp_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        started = _wait_for_recorders(fake_parecord, time.sleep)
        os.killpg(proc.pid, signal.SIGINT)
        try:
            output, _ = proc.communicate(timeout=_TERMINAL_RUN_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output, _ = proc.communicate()
        if not started:
            pytest.fail(f"{_RECORDERS_LATE}\n{output}")
        return TerminalRun(proc.returncode, output)

    return run


@pytest.fixture
def run_gate(tmp_path):
    """Run the real scripts/gate.sh with every command it calls stubbed on PATH.

    Returns a callable: `run_gate(*args, changed=[...], fail="uv run ty check")` runs
    the gate on a branch whose diff against origin/main is `changed` (None: no merge
    base at all), makes the `fail` command exit 1, and gives back the finished process
    and every command the gate invoked, in order. The repository root reads `<root>`.
    """
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    gate = scripts / "gate.sh"
    shutil.copy2(REPO_ROOT / "scripts" / "gate.sh", gate)
    logging_stub = (
        "#!/usr/bin/env bash\n"
        'printf "%s %s\\n" "${0##*/}" "$*" >> "$GATE_LOG"\n'
        'if [ "${0##*/} $*" = "$GATE_FAIL" ]; then exit 1; fi\n'
    )
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for path in (stubs / "uv", stubs / "python3", stubs / "docker", scripts / "deb-smoke.sh"):
        path.write_text(logging_stub)
        path.chmod(0o755)
    # git answers only the two questions the gate asks it.
    git = stubs / "git"
    git.write_text(
        "#!/usr/bin/env bash\n"
        'case "$*" in\n'
        '  *merge-base*) [ -n "$GATE_NO_BASE" ] && exit 1; echo base ;;\n'
        '  *"diff --name-only"*) printf "%s" "$GATE_CHANGED" ;;\n'
        "esac\n"
    )
    git.chmod(0o755)
    log = tmp_path / "commands.log"

    def run(
        *args: str, changed: Sequence[str] | None = (), fail: str = ""
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        env = {
            "PATH": f"{stubs}:{os.environ['PATH']}",
            "GATE_LOG": str(log),
            "GATE_FAIL": fail,
            "GATE_NO_BASE": "1" if changed is None else "",
            "GATE_CHANGED": "".join(f"{name}\n" for name in changed or ()),
        }
        result = subprocess.run(
            [gate, *args],
            env=os.environ | env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        lines = log.read_text().splitlines() if log.exists() else []
        return result, [line.replace(str(tmp_path), "<root>").rstrip() for line in lines]

    return run


# --- WAV file helpers ---


def create_silent_wav(path: Path, duration: float = 1.0, sample_rate: int = 48000) -> None:
    """Create a silent (zero-amplitude) WAV file for testing."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        n_frames = int(duration * sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def create_mono_wav(
    path: Path, duration: float = 1.0, sample_rate: int = 48000, amplitude: float = 0.5
) -> None:
    """Create a mono WAV with a 440Hz sine wave at given amplitude."""
    n_frames = int(duration * sample_rate)
    t = np.linspace(0, duration, n_frames, dtype=np.float32)
    samples = (amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def create_stereo_wav(path, duration, sample_rate, left_amplitude, right_amplitude):
    """Create a stereo WAV with sine wave on each channel at given amplitudes."""
    n_frames = int(duration * sample_rate)
    t = np.linspace(0, duration, n_frames, dtype=np.float32)

    left = (left_amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (right_amplitude * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    stereo = np.empty(n_frames * 2, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


def create_mono_wav_segments(path, sample_rate, segments_spec):
    """Create a mono WAV of 440 Hz sections; segments_spec: list of (duration, amplitude).
    An amplitude of 0 is digital silence."""
    samples = np.concatenate(
        [
            (
                amp * np.sin(2 * np.pi * 440 * np.linspace(0, dur, int(dur * sample_rate))) * 32767
            ).astype(np.int16)
            for dur, amp in segments_spec
        ]
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def create_stereo_wav_segments(path, sample_rate, segments_spec):
    """Create a stereo WAV with different amplitudes per time segment.

    segments_spec: list of (duration, left_amp, right_amp)
    """
    all_left = []
    all_right = []

    for duration, left_amp, right_amp in segments_spec:
        n_frames = int(duration * sample_rate)
        t = np.linspace(0, duration, n_frames, dtype=np.float32)
        all_left.append((left_amp * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16))
        all_right.append((right_amp * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16))

    left = np.concatenate(all_left)
    right = np.concatenate(all_right)

    stereo = np.empty(len(left) * 2, dtype=np.int16)
    stereo[0::2] = left
    stereo[1::2] = right

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(stereo.tobytes())


# --- Test data constants ---


SAMPLE_MD = """\
---
date: 2026-03-20
time: "14:30"
duration: "00:10:00"
language: en
---

# Meeting 2026-03-20 14:30

[00:00:01] **You:** Hello there.
[00:00:05] **Speaker 1:** Hi, let's discuss the plan.
"""

SAMPLE_MD_WITH_SUMMARY = """\
---
date: 2026-03-20
time: "14:30"
duration: "00:10:00"
language: en
---

## Summary

Old summary here.

---

# Meeting 2026-03-20 14:30

[00:00:01] **You:** Hello there.
"""

SAMPLE_TRANSCRIPT_MD = """\
---
date: 2026-03-20
time: "14:30"
duration: "00:10:00"
language: en
tags:
  - meeting
  - transcript
audio: "[[attachments/audio/2026-03-20_14-30-00.wav]]"
---

# Meeting 2026-03-20 14:30

[00:00:01] **You:** Hello there.
[00:00:05] **Speaker 1:** Hi, let's discuss the plan.
"""

VALID_LLM_RESPONSE = json.dumps(
    {
        "brief": "Discussed the project plan and assigned tasks.",
        "action_items": [
            {"assignee": "You", "action": "Send the report", "deadline": "Friday"},
            {"assignee": "Speaker 1", "action": "Review the code", "deadline": None},
        ],
        "key_decisions": ["Use PostgreSQL instead of MongoDB"],
        "is_trivial": False,
    }
)

VALID_LLM_RESPONSE_MINIMAL = json.dumps(
    {
        "brief": "Test summary.",
        "action_items": [],
        "key_decisions": [],
        "is_trivial": False,
    }
)


class HttpError(Exception):
    """Exception with status_code attribute, mimicking SDK exceptions (anthropic, openai)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


# --- Test helpers ---


def load_script(name: str) -> ModuleType:
    """Load `scripts/<name>.py` by path — `scripts/` is not an importable package."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clear_all_provider_env_vars(monkeypatch) -> None:
    """Remove all provider-specific API key env vars.

    Redundant since isolate_settings_sources does the same for every test; kept because
    call sites read as an explicit statement of what the test depends on.
    """
    for env_var in _PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)


def mock_anthropic_response(text: str) -> MagicMock:
    """Create a mock anthropic Messages.create response."""
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    return response


def voice_signal(duration: float, sr: int, fundamental: float) -> np.ndarray:
    """Generate a synthetic voice signal with harmonics."""
    t = np.linspace(0, duration, int(duration * sr), dtype=np.float32)
    return (
        np.sin(2 * np.pi * fundamental * t)
        + 0.5 * np.sin(2 * np.pi * fundamental * 2 * t)
        + 0.25 * np.sin(2 * np.pi * fundamental * 3 * t)
    ) * 10000


# --- Session file helpers ---


def create_session_file(session_file, **overrides):
    """Write a session.json file with sensible defaults, overridden by kwargs."""
    data = {
        "pid_monitor": 99998,
        "pid_mic": 99999,
        "session_name": "test_session",
        "monitor_path": "/run/user/1000/simic/test_session/monitor.wav",
        "mic_path": "/run/user/1000/simic/test_session/mic.wav",
        "started_at": "2026-03-17T14:30:00",
        **overrides,
    }
    session_file.write_text(json.dumps(data))
    return data


# --- Mock helpers ---


def mock_whisper_transcribe(segments_data):
    """Create a mock WhisperModel that returns given segments.

    segments_data: list of (start, end, text) tuples
    """
    mock_model = MagicMock()

    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.language_probability = 0.99
    mock_info.duration = 10.0

    mock_segments = []
    for start, end, text in segments_data:
        seg = MagicMock()
        seg.start = start
        seg.end = end
        seg.text = f" {text} "
        seg.words = []
        mock_segments.append(seg)

    # Each call to transcribe() returns a fresh iterator
    mock_model.transcribe.side_effect = lambda *a, **kw: (iter(list(mock_segments)), mock_info)

    return mock_model


def mock_pyannote_annotation(tracks):
    """Create a mock pyannote annotation from (start, end, speaker) tuples."""
    mock_annotation = MagicMock()
    itertracks_result = []
    for start, end, speaker in tracks:
        turn = MagicMock()
        turn.start = start
        turn.end = end
        itertracks_result.append((turn, None, speaker))
    mock_annotation.itertracks.return_value = itertracks_result
    return mock_annotation


def mock_whisper_by_channel(mic_segments, monitor_segments):
    """A WhisperModel stand-in that answers per channel, going by the 16 kHz file it is
    handed — so a test can tell who said what — and, like faster-whisper, reports that
    file's real duration. Segments: (start, end, text) tuples."""

    def transcribe(audio, **kwargs):
        segments = mic_segments if "mic_16k" in str(audio) else monitor_segments
        result, info = mock_whisper_transcribe(segments).transcribe(audio, **kwargs)
        with wave.open(str(audio), "rb") as wf:
            info.duration = wf.getnframes() / wf.getframerate()
        return result, info

    model = MagicMock()
    model.transcribe.side_effect = transcribe
    return model


def process_state(pid: int) -> str | None:
    """The kernel's one-letter state for a process (R, S, Z, ...); None once it is gone,
    reaped included. A zombie is "Z": exited, but its parent has not collected it."""
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        # ProcessLookupError: reaped between opening the file and reading it.
        return None
    # The state follows the command name, which is parenthesised and may hold spaces.
    return stat_line.rsplit(")", 1)[1].split()[0]
