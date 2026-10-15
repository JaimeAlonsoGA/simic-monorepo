"""The processes that capture a meeting: one `parecord` per channel and, for a
compressed format, the ffmpeg encoder it feeds. Starting them, telling whether they
still run, and stopping them in order."""

import contextlib
import os
import signal
import subprocess
import time
from pathlib import Path

from simic.settings import Settings

# How long a recorder gets to exit on SIGTERM before SIGKILL.
RECORDER_STOP_SECONDS = 5.0
# How long an encoder gets to flush the end of its file once its recorder has exited.
ENCODER_FLUSH_SECONDS = 15.0


def spawn(device: str, dest: Path, settings: Settings, work_dir: Path) -> tuple[int, int | None]:
    """Start capturing one source. Returns (parecord pid, ffmpeg pid or None for WAV).

    Every process starts a session of its own. A terminal delivers Ctrl+C to its whole
    foreground process group, so recorders in `simic start`'s group died of the same
    keystroke that asked simic to stop them — before simic could, and in no
    particular order — and the start command took the vanished recorders for a peer
    `simic stop`. Outside the group, only simic hears Ctrl+C, and it stops them
    itself: the recorder first, so the encoder reads end-of-file and finishes its file.

    A compressed capture is parecord's raw PCM piped into ffmpeg. The parent closes its
    copy of ffmpeg's stdin so the encoder finishes when parecord exits.

    --latency-msec=100 makes parecord flush about ten times a second. At 16 kHz its
    default fragment is two seconds, so stopping used to drop the tail.
    """
    if settings.audio_format == "wav":
        proc = subprocess.Popen(
            [
                "parecord",
                "--format=s16le",
                f"--rate={settings.sample_rate}",
                "--channels=1",
                "--latency-msec=100",
                "--file-format=wav",
                f"--device={device}",
                str(dest),
            ],
            cwd=work_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        return proc.pid, None

    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(settings.sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libmp3lame",
            "-b:a",
            f"{settings.mp3_bitrate_k}k",
            # Write each frame as it is encoded. ffmpeg otherwise held the first tens of
            # seconds in its buffer: the file sat at 0 bytes, the stall warning fired on a
            # healthy recording, and the live checks had nothing to read.
            "-flush_packets",
            "1",
            str(dest),
        ],
        cwd=work_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    if encoder.stdin is None:
        encoder.kill()
        raise RuntimeError("ffmpeg did not provide a stdin pipe")
    try:
        recorder = subprocess.Popen(
            [
                "parecord",
                "--raw",
                "--format=s16le",
                f"--rate={settings.sample_rate}",
                "--channels=1",
                "--latency-msec=100",
                f"--device={device}",
            ],
            cwd=work_dir,
            stdout=encoder.stdin,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except Exception:
        encoder.kill()
        raise
    finally:
        encoder.stdin.close()
    return recorder.pid, encoder.pid


def running(pid: int) -> bool:
    """Whether a capture process is still running.

    kill(pid, 0) alone cannot tell: a process that has exited stays a zombie until its
    parent reaps it, and kill still succeeds on a zombie. `simic start` and the tray
    spawn parecord themselves and never wait() on it, and `simic stop` runs in another
    process altogether — so a stopped recorder kept reading as alive, and every stop
    waited out the whole kill timeout before SIGKILLing processes already dead.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if not _is_zombie(pid):
        return True
    # Exited. Reap it if it is ours; another process's zombie is its parent's to collect.
    _reap(pid, os.WNOHANG)
    return False


def _reap(pid: int, flags: int) -> None:
    """Collect an exited child of this process, so it does not linger as a zombie.
    A pid that is not our child raises ChildProcessError, which is the expected case for
    `simic stop` run against another terminal's recording."""
    with contextlib.suppress(ChildProcessError):
        os.waitpid(pid, flags)


def _is_zombie(pid: int) -> bool:
    """An exited process another parent has not reaped yet. Reads /proc: Linux only,
    as simic is."""
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    # The state follows the command name, which is parenthesised and may hold spaces.
    return stat_line.rsplit(")", 1)[1].split()[0] == "Z"


def terminate(pid: int) -> None:
    """Send SIGTERM to a single process, ignore if already dead."""
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)


def wait_and_kill(pids: list[int], timeout: float) -> None:
    """Wait for processes to exit, then SIGKILL survivors."""
    alive = set(pids)
    deadline = time.monotonic() + timeout

    while alive and time.monotonic() < deadline:
        alive = {pid for pid in alive if running(pid)}
        if alive:
            time.sleep(0.1)

    for pid in alive:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
            # SIGKILL cannot be ignored, so waiting for our own child cannot hang.
            _reap(pid, 0)
