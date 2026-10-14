import contextlib
import datetime
import fcntl
import json
import os
import re
import shutil
import stat
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NotRequired, TypedDict
from zoneinfo import ZoneInfo

from simic import _capture, const
from simic.settings import Settings


class NoActiveRecording(RuntimeError):
    """Raised when stop() is called with no session in progress."""


class SessionBusy(RuntimeError):
    """Raised when another process is already turning a session into a note."""


class SessionData(TypedDict):
    pid_monitor: int
    pid_mic: int
    session_name: str
    monitor_path: str
    mic_path: str
    started_at: str
    # ffmpeg pids when audio_format is mp3. Absent for a WAV capture.
    pid_monitor_encode: NotRequired[int]
    pid_mic_encode: NotRequired[int]


_SESSION_NAME_RE = re.compile(r"^[\w-]+$")

_SESSIONS_DIR_NAME = "sessions"


def _state_dir() -> Path:
    """simic's XDG state directory — ~/.local/state/simic unless XDG_STATE_HOME
    says otherwise. Holds session.json and, under `sessions/`, the recordings it points
    at. A relative XDG_STATE_HOME is ignored, as the XDG spec requires: it would put the
    state wherever the command happened to be run from."""
    xdg_state_home = os.environ.get("XDG_STATE_HOME", "")
    base = Path(xdg_state_home)
    if not base.is_absolute():
        base = Path.home() / ".local" / "state"
    return base / "simic"


def _ensure_private_dir(path: Path) -> None:
    """Create the directory as 0700, or refuse one that is not privately ours.

    `mkdir(mode=...)` sets the mode only when it actually creates the directory, so an
    existing one has to be checked rather than trusted. lstat, not stat: mkdir with
    exist_ok follows a symlink to a directory and reports success, and a symlink is
    exactly what an attacker would plant. Refuse rather than chmod — repairing the mode
    cannot undo the window in which someone else already held the directory open.
    """
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeError(f"Refusing to use {path}: it is a symlink, not a directory")
    if info.st_uid != os.geteuid():
        raise RuntimeError(f"Refusing to use {path}: owned by uid {info.st_uid}, not this user")
    if info.st_mode & 0o077:
        raise RuntimeError(
            f"Refusing to use {path}: reachable by other users "
            f"(mode {stat.S_IMODE(info.st_mode):04o}, expected 0700). "
            f"If it is yours, run: chmod 700 {path}"
        )


def session_root(settings: Settings) -> Path:
    """Directory holding recordings that are still in progress. Created if missing.

    `SIMIC_SESSIONS_DIR` when set, otherwise `sessions/` in the XDG state directory
    (~/.local/state/simic/sessions). Wherever it is, it is verified private before
    use. Rejected locations, and why:

    - /tmp/simic: /tmp is world-writable, so any other local user could pre-create
      the path and quietly collect every meeting recording, raw microphone and system
      audio both, captured long before any masking applies.
    - $XDG_RUNTIME_DIR: private, but a tmpfs of 10% of RAM — 780 MB on an 8 GB laptop,
      while an hour of recording needs about 1.6 GB once the merge and the 16 kHz copies
      land next to the raw channels. The XDG spec itself asks applications not to put
      large files there. /tmp has the same problem on systemd 258+, where it is a tmpfs
      with a per-user quota (issue #13).
    - Both of those are emptied at reboot or logout. A recording that was interrupted
      before it was transcribed should survive that, so it can still be processed.

    The state directory is on disk under the user's home: private by construction, as
    the runtime directory was, without its size limit or its lifetime. A session is
    removed once it has been processed; one that failed stays to be recovered.
    """
    root = settings.sessions_dir or _state_dir() / _SESSIONS_DIR_NAME
    _ensure_private_dir(root)
    return root


def session_dir(settings: Settings, session_name: str) -> Path:
    """Directory holding one session's raw channels. Not created here — `start` does
    that; callers that only need the path (live transcription, status messages) should
    not have a side effect."""
    validate_session_name(session_name)
    return session_root(settings) / session_name


def validate_session_name(session_name: str) -> None:
    """Reject session names that could traverse the filesystem.

    Only alphanumerics, dashes, and underscores are allowed — this keeps
    names safe to use as path components in the vault.
    """
    if not _SESSION_NAME_RE.match(session_name):
        raise ValueError(
            f"Invalid session name: {session_name!r}. "
            "Only alphanumerics, dashes, and underscores are allowed."
        )


def detect_devices(settings: Settings) -> tuple[str, str]:
    """Return (monitor_source, mic_source).

    If settings.monitor_source == "auto":
        Use @DEFAULT_MONITOR@ (PulseAudio/PipeWire dynamic reference that
        follows the current default sink — survives device switches).
        Falls back to pactl info -> default_sink + ".monitor" when
        @DEFAULT_MONITOR@ is not supported.
    If settings.mic_source == "auto":
        Use @DEFAULT_SOURCE@ (follows the current default source).
        Falls back to pactl info -> default_source.

    Raises RuntimeError if pactl is not available or devices not found.
    """
    monitor = settings.monitor_source
    mic = settings.mic_source

    if monitor == "auto" or mic == "auto":
        if not shutil.which("pactl"):
            raise RuntimeError("pactl not found. Install: sudo apt install pulseaudio-utils")

        # Try dynamic references first (PipeWire / PulseAudio 14+).
        # They follow the current default device, so recording survives
        # hot-switching between speakers, headphones, etc.
        if monitor == "auto":
            if _probe_source(const.PA_DEFAULT_MONITOR):
                monitor = const.PA_DEFAULT_MONITOR
            else:
                monitor = _resolve_monitor_via_pactl()

        if mic == "auto":
            if _probe_source(const.PA_DEFAULT_SOURCE):
                mic = const.PA_DEFAULT_SOURCE
            else:
                mic = _resolve_source_via_pactl()

    return monitor, mic


def _probe_source(source_name: str) -> bool:
    """Return True if parecord can open the given source (quick 0.2s test)."""
    if not shutil.which("parecord"):
        return False
    try:
        proc = subprocess.Popen(
            [
                "parecord",
                f"--device={source_name}",
                "--format=s16le",
                "--rate=16000",
                "--channels=1",
                "/dev/null",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(0.2)
        proc.terminate()
        proc.wait(timeout=2)
        # If parecord ran for 0.2s without crashing, the source exists
        return proc.returncode is not None
    except Exception:
        return False


def _resolve_monitor_via_pactl() -> str:
    """Resolve monitor source from pactl info (legacy fallback)."""
    result = subprocess.run(
        ["pactl", "--format=json", "info"],
        capture_output=True,
        text=True,
        check=True,
    )
    info: dict[str, str] = json.loads(result.stdout)
    default_sink = info.get("default_sink_name") or info.get("default_sink", "")
    if not default_sink:
        raise RuntimeError(
            "No default sink found. Run 'pactl list sources short' to check devices."
        )
    return f"{default_sink}{const.PA_MONITOR_SUFFIX}"


def _resolve_source_via_pactl() -> str:
    """Resolve default source from pactl info (legacy fallback)."""
    result = subprocess.run(
        ["pactl", "--format=json", "info"],
        capture_output=True,
        text=True,
        check=True,
    )
    info: dict[str, str] = json.loads(result.stdout)
    default_source = info.get("default_source_name") or info.get("default_source", "")
    if not default_source:
        raise RuntimeError(
            "No default source found. Run 'pactl list sources short' to check devices."
        )
    return default_source


def _meeting_time(settings: Settings, now: datetime.datetime) -> datetime.datetime:
    """`now` on the meeting's clock: SIMIC_TIMEZONE, else the machine's own zone.

    The session name becomes the note's file name and its date and time, so it has to
    read as the user's clock. UTC filed a late meeting under the next day (issue #15).
    """
    if settings.timezone:
        return now.astimezone(ZoneInfo(settings.timezone))
    return now.astimezone()


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _channel_pids(session: SessionData) -> dict[str, list[int]]:
    """The processes capturing each channel: its recorder, then the encoder behind it
    when the format is compressed."""
    pids = {
        const.CHANNEL_MONITOR: [session["pid_monitor"], session.get("pid_monitor_encode")],
        const.CHANNEL_MIC: [session["pid_mic"], session.get("pid_mic_encode")],
    }
    return {channel: [pid for pid in found if pid is not None] for channel, found in pids.items()}


def _stopped_channels(session: SessionData) -> list[str]:
    return [
        channel
        for channel, pids in _channel_pids(session).items()
        if not all(_capture.running(pid) for pid in pids)
    ]


def session_channels(directory: Path) -> tuple[Path, Path]:
    """(monitor, mic) recorded into a session directory, whatever format captured them."""
    found = []
    for channel in (const.CHANNEL_MONITOR, const.CHANNEL_MIC):
        files = sorted(directory.glob(f"{channel}.*"))
        if len(files) != 1:
            raise RuntimeError(f"No {channel} recording in {directory}")
        found.append(files[0])
    return found[0], found[1]


@contextlib.contextmanager
def processing(directory: Path) -> Iterator[None]:
    """Hold a session directory while it becomes a note, or raise SessionBusy.

    Once stopped, a session is no longer named in session.json, so nothing else tells
    a `simic stop` at work apart from a recording whose processing failed: without
    the lock, `simic recover` in another terminal would transcribe it a second time.
    flock, because the kernel releases it when its holder exits — a crash included.
    """
    fd = os.open(directory / const.FILE_PROCESSING_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SessionBusy(f"{directory.name} is already being processed.") from None
        yield
    finally:
        os.close(fd)


@contextlib.contextmanager
def transcription_turn(on_wait: Callable[[], None]) -> Iterator[None]:
    """Wait for any other transcription to finish, then hold the turn until this one does.

    Back-to-back meetings processed in the background would otherwise each load a
    Whisper model at once, and a laptop that fits one runs out of memory with two —
    the kernel kills both. `on_wait` is called once, only when there is a wait.
    """
    # Next to session.json, which is no more private: the lock holds no meeting data.
    state = _state_dir()
    state.mkdir(parents=True, exist_ok=True)
    fd = os.open(state / const.FILE_TRANSCRIBE_LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            on_wait()
            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _being_processed(directory: Path) -> bool:
    try:
        with processing(directory):
            return False
    except SessionBusy:
        return True


class Recorder:
    def __init__(
        self,
        state_dir: Path | None = None,
        clock: Callable[[], datetime.datetime] = _utc_now,
    ) -> None:
        self._state_dir = state_dir or _state_dir()
        self._session_file = self._state_dir / const.FILE_SESSION
        # The current instant; a test fixes it to name a session deterministically.
        self._clock = clock

    @property
    def session_file(self) -> Path:
        return self._session_file

    def start(self, settings: Settings, session_name: str | None = None) -> str:
        """Start capturing the monitor source and the microphone.

        Creates a private session directory (see `session_root`) holding one file per
        channel in `settings.audio_format`, and saves state to session.json. A session
        left by a `start` that was killed is stopped first; its audio stays in its
        directory for `simic recover`. Returns session_name.
        """
        if self.is_recording():
            raise RuntimeError(
                "Recording already in progress. Run 'simic stop' to finish it first."
            )

        if not shutil.which("parecord"):
            raise RuntimeError("parecord not found. Install: sudo apt install pulseaudio-utils")
        if settings.audio_format == "mp3" and not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not found. Install: sudo apt install ffmpeg")

        if self.has_session():
            with contextlib.suppress(NoActiveRecording):
                self.stop()

        monitor_source, mic_source = detect_devices(settings)

        started = _meeting_time(settings, self._clock())
        if session_name is None:
            session_name = started.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            validate_session_name(session_name)

        tmp_dir = session_dir(settings, session_name)
        _ensure_private_dir(tmp_dir)

        monitor_path = tmp_dir / f"{const.CHANNEL_MONITOR}.{settings.audio_format}"
        mic_path = tmp_dir / f"{const.CHANNEL_MIC}.{settings.audio_format}"

        monitor_rec, monitor_enc = _capture.spawn(monitor_source, monitor_path, settings, tmp_dir)
        mic_rec, mic_enc = _capture.spawn(mic_source, mic_path, settings, tmp_dir)

        self._state_dir.mkdir(parents=True, exist_ok=True)
        session_data: SessionData = {
            "pid_monitor": monitor_rec,
            "pid_mic": mic_rec,
            "session_name": session_name,
            "monitor_path": str(monitor_path),
            "mic_path": str(mic_path),
            "started_at": started.isoformat(),
        }
        if monitor_enc is not None and mic_enc is not None:
            session_data["pid_monitor_encode"] = monitor_enc
            session_data["pid_mic_encode"] = mic_enc
        self._session_file.write_text(json.dumps(session_data, indent=2))

        return session_name

    def stop(self) -> tuple[Path, Path]:
        """Stop the capture and return the (monitor, mic) files it wrote.

        The session is claimed before anything is signalled: unlinking is atomic, so of
        a `simic stop` and a `simic start` stopping on Ctrl+C at the same moment,
        exactly one owns the recording and the other gets NoActiveRecording. Then the
        recorders get SIGTERM, and the encoders the time to flush once their input ends.
        """
        session = self.read_session()
        if session is None:
            raise NoActiveRecording("No recording in progress.")
        try:
            self._session_file.unlink()
        except FileNotFoundError:
            raise NoActiveRecording("No recording in progress.") from None

        recorders = [session["pid_monitor"], session["pid_mic"]]
        encoders = [
            session[key] for key in ("pid_monitor_encode", "pid_mic_encode") if key in session
        ]
        for pid in recorders:
            _capture.terminate(pid)
        _capture.wait_and_kill(recorders, _capture.RECORDER_STOP_SECONDS)
        if encoders:
            _capture.wait_and_kill(encoders, _capture.ENCODER_FLUSH_SECONDS)

        return Path(session["monitor_path"]), Path(session["mic_path"])

    def has_session(self) -> bool:
        """A recording was started and nobody has stopped it yet. Its recorders may
        have exited on their own: it still holds a meeting to transcribe."""
        return self._session_file.exists()

    def is_recording(self) -> bool:
        """Every capture process of the session is running.

        Only asks. It used to delete the session as soon as one recorder was gone, and
        `simic start` then read the missing session as a peer `simic stop` and
        returned without transcribing anything.
        """
        session = self.read_session()
        return session is not None and not _stopped_channels(session)

    def stopped_channels(self) -> list[str]:
        """Channels whose capture exited on its own. Empty without a session."""
        session = self.read_session()
        return [] if session is None else _stopped_channels(session)

    def get_session_info(self) -> SessionData | None:
        """Return session info dict if recording, else None."""
        return self.read_session() if self.is_recording() else None

    def leftover_sessions(self, settings: Settings) -> list[Path]:
        """Session directories nothing owns: not recording, not being processed — the
        meetings whose processing failed or was interrupted. Oldest name first."""
        session = self.read_session()
        active = session["session_name"] if session else None
        root = session_root(settings)
        return sorted(
            p for p in root.iterdir() if p.is_dir() and p.name != active and not _being_processed(p)
        )

    def sessions_in_progress(self, settings: Settings) -> list[Path]:
        """Session directories some process is turning into a note right now, in the
        background or not. Oldest name first."""
        root = session_root(settings)
        return sorted(p for p in root.iterdir() if p.is_dir() and _being_processed(p))

    def read_session(self) -> SessionData | None:
        """The session as started, whether or not its recorders still run."""
        try:
            session: SessionData = json.loads(self._session_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return session
