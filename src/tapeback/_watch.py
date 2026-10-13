"""A recording in progress as the person running it sees it: how long it has run, how
big each channel is, and whether each is still being written.

An hour-long `tapeback start` used to print three lines and then nothing, so a capture
that had quietly stopped — a sound server that suspended its source, a recorder stuck
on a vanished device — looked exactly like one that was working. A compressed channel
grows even through silence (the encoder still writes frames), so a file that stops
growing means the capture stopped, not that the room went quiet.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click

from tapeback._timing import format_clock

if TYPE_CHECKING:
    from tapeback.recorder import Recorder

# How long a channel's file may stay the same size before it counts as stalled. parecord
# flushes about ten times a second and the MP3 encoder writes a frame every 72 ms at
# 16 kHz, so a healthy file grows many times within it.
STALL_SECONDS = 15.0

BYTES_PER_MB = 1_000_000


@dataclass(frozen=True)
class CaptureStatus:
    elapsed: float
    # Bytes written so far, per channel.
    sizes: dict[str, int]
    # Channels whose file has not grown for STALL_SECONDS.
    stalled: list[str]


class CaptureWatch:
    def __init__(
        self,
        files: dict[str, Path],
        clock: Callable[[], float] = time.monotonic,
        stall_seconds: float = STALL_SECONDS,
    ) -> None:
        self._files = files
        self._clock = clock
        self._stall_seconds = stall_seconds
        self._started = clock()
        self._last_size = dict.fromkeys(files, 0)
        self._last_growth = dict.fromkeys(files, self._started)

    def poll(self) -> CaptureStatus:
        now = self._clock()
        sizes = {channel: _size(path) for channel, path in self._files.items()}
        for channel, size in sizes.items():
            if size > self._last_size[channel]:
                self._last_size[channel] = size
                self._last_growth[channel] = now
        stalled = [
            channel
            for channel in self._files
            if now - self._last_growth[channel] >= self._stall_seconds
        ]
        return CaptureStatus(now - self._started, sizes, stalled)


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def format_status(status: CaptureStatus) -> str:
    """One line for a terminal: `● REC 12:34  monitor 5.8 MB  mic 5.8 MB`."""
    channels = "  ".join(
        f"{channel} {size / BYTES_PER_MB:.1f} MB" for channel, size in status.sizes.items()
    )
    return f"● REC {format_clock(status.elapsed)}  {channels}"


# --- the other side going quiet ------------------------------------------------------
#
# A channel that grows can still be capturing nothing: when the call plays on an output
# tapeback is not recording (headphones that took over, a browser tab routed elsewhere),
# the monitor channel records an hour of digital silence and the meeting is lost all
# the same. Every LEVEL_CHECK_SECONDS the end of each watched channel is decoded and its
# peak measured; one that stays below SILENT_PEAK_DB for SILENCE_WARN_SECONDS is reported.

LEVEL_CHECK_SECONDS = 60.0
# How much of the end of the file each check decodes.
LEVEL_WINDOW_SECONDS = 30.0
# A peak this low is no sound at all, not a quiet room: speech peaks above -30 dBFS, and
# even a far, faint voice above -50. Digital silence measures -91 or -inf.
SILENT_PEAK_DB = -60.0
SILENCE_WARN_SECONDS = 180.0
_MAX_VOLUME_RE = re.compile(r"max_volume: (-?(?:inf|[\d.]+)) dB")


def peak_db(path: Path, window: float) -> float | None:
    """The loudest sample in the last `window` seconds of `path`, in dBFS. None when it
    cannot be measured — a file still too short, or a format ffmpeg cannot seek in while
    it is being written — which is no evidence of silence either way."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-sseof",
                f"-{window:.0f}",
                "-i",
                str(path),
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=window,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = _MAX_VOLUME_RE.search(result.stderr)
    return float(match.group(1)) if match else None


class SilenceWatch:
    def __init__(
        self,
        files: dict[str, Path],
        measure: Callable[[Path, float], float | None] = peak_db,
        clock: Callable[[], float] = time.monotonic,
        check_every: float = LEVEL_CHECK_SECONDS,
        silent_after: float = SILENCE_WARN_SECONDS,
    ) -> None:
        self._files = files
        self._measure = measure
        self._clock = clock
        self._check_every = check_every
        self._silent_after = silent_after
        self._last_check = clock()
        self._silent_since: dict[str, float] = {}

    def poll(self) -> dict[str, float]:
        """Channels silent for at least `silent_after`, with how long they have been."""
        now = self._clock()
        if now - self._last_check >= self._check_every:
            self._last_check = now
            for channel, path in self._files.items():
                peak = self._measure(path, LEVEL_WINDOW_SECONDS)
                if peak is None:
                    continue
                if peak < SILENT_PEAK_DB:
                    # The window just measured was silent from its start.
                    self._silent_since.setdefault(channel, now - LEVEL_WINDOW_SECONDS)
                else:
                    self._silent_since.pop(channel, None)
        return {
            channel: now - since
            for channel, since in self._silent_since.items()
            if now - since >= self._silent_after
        }


# --- the loop `tapeback start` waits in ---------------------------------------------


def wait_while_recording(recorder: Recorder, watch: CaptureWatch, silence: SilenceWatch) -> None:
    """Block until the recording ends, showing it on a terminal as it goes.

    A status line rewritten in place when stderr is a terminal; a warning, on any
    output, when a channel stops being written or the other side goes silent — and a
    line when it comes back.
    """
    status_line = sys.stderr.isatty()
    warned: set[str] = set()
    quiet = False
    while recorder.is_recording():
        silent = silence.poll()
        if silent and not quiet:
            quiet = True
            minutes = max(silent.values()) // 60
            click.echo(
                f"\nWarning: no sound from the other side for {minutes:.0f} min. If the "
                "call is still going, its audio may be playing on an output tapeback is "
                "not recording — check 'tapeback status'.",
                err=True,
            )
        elif quiet and not silent:
            quiet = False
            click.echo("\nSound from the other side again.", err=True)
        status = watch.poll()
        for channel in status.stalled:
            if channel not in warned:
                warned.add(channel)
                click.echo(
                    f"\nWarning: nothing has been written to the {channel} channel for "
                    f"{STALL_SECONDS:.0f}s. Check the device with 'tapeback status'.",
                    err=True,
                )
        for channel in sorted(warned - set(status.stalled)):
            warned.discard(channel)
            click.echo(f"\nThe {channel} channel is being written again.", err=True)
        if status_line:
            click.echo(f"\r{format_status(status)}", nl=False, err=True)
        time.sleep(1)
    if status_line:
        click.echo(err=True)
