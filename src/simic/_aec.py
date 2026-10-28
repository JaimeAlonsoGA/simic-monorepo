"""Echo cancellation at capture: the microphone without what the speakers play.

On a call without headphones the microphone hears the other side from the speakers. The
call app cancels that echo in what it sends, but simic records the microphone itself,
before any app has touched it: the other side was transcribed twice, once as "Other"
from the monitor and again, a moment later, as "You". A 1 h 24 min meeting came out with
77 % of what the other person said repeated under the user's name.

PipeWire ships WebRTC's echo canceller as a module. In monitor mode it takes the
speakers' monitor as the reference and gives a new source: the microphone minus that
reference. No sink is added and nothing is rerouted, so the call app's audio is left
alone. Measured on a laptop with its speakers at 76 %: the echo 15 to 40 dB lower, while
sound that is not on the speakers passes within 2 dB.

The module lives in a `pw-cli` process of its own, started before the recorders and
stopped after them: it exists exactly as long as the recording, like the recorders, and
a `simic stop` from another terminal ends it by pid the same way.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from simic import const

# The source the recorder reads. One recording at a time, so one fixed name; a fake
# parecord in the tests reads the same name.
SOURCE_NAME = "simic.echo-cancel"
# How long the module may take to load. It answers in tens of milliseconds.
START_SECONDS = 5.0
_POLL_SECONDS = 0.05
_LOADED_RE = re.compile(r"^\d+ = @module:\d+", re.MULTILINE)
_FAILED_RE = re.compile(r'^Error: "Could not load module"', re.MULTILINE)
# What the canceller does besides cancelling. Gain control off: it would pump the level
# of the recording, which Whisper does not need and the loudness pass already evens out.
_WEBRTC_ARGS = "webrtc.gain_control = false webrtc.extended_filter = true"


class EchoCancelUnavailable(RuntimeError):
    """PipeWire's echo canceller could not be started: no pw-cli, or no module."""


@dataclass(frozen=True)
class EchoCanceller:
    pid: int
    source: str


def _target(device: str, suffix: str = "") -> str:
    """`target.object = ...` for a named device; nothing for a dynamic reference such as
    @DEFAULT_SOURCE@, so the stream follows the default device as the recorder would."""
    if device.startswith("@"):
        return ""
    return f"target.object = {json.dumps(device.removesuffix(suffix))}"


def _props(*parts: str) -> str:
    return "{ " + " ".join(part for part in parts if part) + " }"


def module_command(mic_source: str, monitor_source: str) -> str:
    """The pw-cli line that loads the canceller between these two devices."""
    capture = _props(
        f"node.name = {json.dumps(SOURCE_NAME + '.capture')} node.passive = true",
        _target(mic_source),
    )
    reference = _props(
        f"node.name = {json.dumps(SOURCE_NAME + '.reference')}",
        _target(monitor_source, const.PA_MONITOR_SUFFIX),
    )
    # priority.session 0: the new source must never become the desktop's default mic.
    source = _props(
        f"node.name = {json.dumps(SOURCE_NAME)}",
        'node.description = "Simic echo-cancelled mic" priority.session = 0',
    )
    return (
        "load-module libpipewire-module-echo-cancel { "
        "library.name = aec/libspa-aec-webrtc monitor.mode = true "
        f"node.name = {json.dumps(SOURCE_NAME + '.aec')} "
        f"capture.props = {capture} sink.props = {reference} source.props = {source} "
        f"aec.args = {_props(_WEBRTC_ARGS)} }}\n"
    )


def start(mic_source: str, monitor_source: str, log: Path) -> EchoCanceller:
    """Load the canceller and return the source to record the microphone from.

    pw-cli writes its answer, then every change to the graph for as long as it runs, to
    `log`: a file rather than a pipe, which nobody would drain once the recording is
    under way. Raises EchoCancelUnavailable, with the process and the log gone, when it
    does not load.
    """
    if not shutil.which("pw-cli"):
        raise EchoCancelUnavailable("pw-cli not found (PipeWire's command-line tool)")
    with log.open("w") as out:
        proc = subprocess.Popen(
            ["pw-cli"],
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    if proc.stdin is None:
        proc.kill()
        raise EchoCancelUnavailable("pw-cli did not provide a stdin pipe")
    # pw-cli keeps running after its input ends; the module goes with the process.
    proc.stdin.write(module_command(mic_source, monitor_source))
    proc.stdin.close()

    deadline = time.monotonic() + START_SECONDS
    while time.monotonic() < deadline:
        answer = log.read_text(errors="replace")
        if _LOADED_RE.search(answer):
            return EchoCanceller(pid=proc.pid, source=SOURCE_NAME)
        if _FAILED_RE.search(answer) or proc.poll() is not None:
            break
        time.sleep(_POLL_SECONDS)
    proc.kill()
    proc.wait()
    # The recording goes on without it; nothing in its directory should suggest otherwise.
    log.unlink(missing_ok=True)
    raise EchoCancelUnavailable(
        "PipeWire's echo-cancel module did not load (is the WebRTC AEC plugin installed?)"
    )


def status_line(enabled: bool, unavailable: str | None) -> str:
    """What `simic start` says of the mic's echo cancellation."""
    if not enabled:
        return "Echo cancellation: off (SIMIC_ECHO_CANCEL)"
    if unavailable:
        return (
            f"Warning: echo cancellation unavailable — {unavailable}. On speakers, the mic "
            "will also record the other side: use headphones."
        )
    return "Echo cancellation: on"
