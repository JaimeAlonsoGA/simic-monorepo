"""Turning a stopped recording into a note without holding the terminal.

An hour of meeting takes most of an hour to transcribe on a CPU, and `tapeback start`
used to keep the terminal for all of it — a Ctrl+C there, meant for the next command,
aborted the transcription. The work now runs as `tapeback recover NAME` in a process of
its own: in a new session, so the terminal's Ctrl+C and hangup never reach it, and with
its output in the session directory, where `tapeback status` reads how far it has got.
A note that is ready, or could not be made, is announced on the desktop (`_notify`).

Nothing new can be lost this way: until its note exists the recording stays in its
directory, so a background run that dies is a leftover for `tapeback recover` like any
other.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from tapeback import const


def process_in_background(session: Path, *, diarize: bool, do_summarize: bool) -> int:
    """Start transcribing a stopped session in a detached process. Returns its pid.

    posix_spawn rather than Popen: the parent exits long before the child, and a Popen
    object would only be left to warn that its process is still running.

    The child gets the parent's import path. A wrapped install (Nix, a bundled
    interpreter) puts tapeback on sys.path from its launcher script, not from the
    interpreter, so a bare `python -m tapeback` would not find the code now running.
    """
    argv = [sys.executable, "-m", "tapeback", "recover", session.name]
    if not diarize:
        argv.append("--no-diarize")
    if not do_summarize:
        argv.append("--no-summarize")
    log = session / const.FILE_PROCESSING_LOG
    return os.posix_spawn(
        sys.executable,
        argv,
        {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)},
        file_actions=[
            (os.POSIX_SPAWN_OPEN, 0, os.devnull, os.O_RDONLY, 0),
            (os.POSIX_SPAWN_OPEN, 1, str(log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600),
            (os.POSIX_SPAWN_DUP2, 1, 2),
        ],
        setsid=True,
    )


def last_status(session: Path) -> str:
    """The latest line a background run printed — its stage and progress. Empty when
    the session is not processed in the background, or has printed nothing yet."""
    try:
        lines = (session / const.FILE_PROCESSING_LOG).read_text().splitlines()
    except FileNotFoundError:
        return ""
    return next((line.strip() for line in reversed(lines) if line.strip()), "")
