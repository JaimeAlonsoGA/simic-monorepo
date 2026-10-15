"""Turning a stopped recording into a note without holding the terminal.

An hour of meeting takes most of an hour to transcribe on a CPU, and `simic start`
used to keep the terminal for all of it — a Ctrl+C there, meant for the next command,
aborted the transcription. The work now runs as `simic recover NAME` in a process of
its own: in a new session, so the terminal's Ctrl+C and hangup never reach it, and with
its output in the session directory, where `simic status` reads how far it has got.
A note that is ready, or could not be made, is announced on the desktop (`_notify`).

Nothing new can be lost this way: until its note exists the recording stays in its
directory, so a background run that dies is a leftover for `simic recover` like any
other.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from simic import const

_PRIVATE = 0o600


def process_in_background(session: Path, *, diarize: bool, do_summarize: bool) -> int:
    """Start transcribing a stopped session in a detached process. Returns its pid.

    Popen with start_new_session runs setsid() in the child itself, which every CPython
    can do. posix_spawn(setsid=True) needs it compiled in, and python-build-standalone —
    the interpreter of the .deb, of `uv tool install` and of uv-made venvs — is built
    without it, so `stop` failed there with NotImplementedError.

    The Popen object is dropped while the child runs, as a detached child should be —
    the same as the recorders in _capture. CPython keeps it and reaps it at the next
    Popen, so a parent that lives on (the window) piles up no zombies. Dropping it
    raises a ResourceWarning, hidden by default; silencing it here would mean
    warnings.catch_warnings(), which swaps the filters of the whole process, and the
    window calls this from a worker thread.

    The child gets the parent's import path. A wrapped install (Nix, a bundled
    interpreter) puts simic on sys.path from its launcher script, not from the
    interpreter, so a bare `python -m simic` would not find the code now running.
    """
    argv = [sys.executable, "-m", "simic", "recover", session.name]
    if not diarize:
        argv.append("--no-diarize")
    if not do_summarize:
        argv.append("--no-summarize")
    log = os.open(
        session / const.FILE_PROCESSING_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _PRIVATE
    )
    try:
        child = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)},
            start_new_session=True,
        )
    finally:
        os.close(log)
    return child.pid


def last_status(session: Path) -> str:
    """The latest line a background run printed — its stage and progress. Empty when
    the session is not processed in the background, or has printed nothing yet."""
    try:
        lines = (session / const.FILE_PROCESSING_LOG).read_text().splitlines()
    except FileNotFoundError:
        return ""
    return next((line.strip() for line in reversed(lines) if line.strip()), "")
