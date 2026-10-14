"""Desktop notifications for the end of processing.

An hour of meeting takes most of an hour to transcribe on a CPU. Nobody watches a
terminal for that long, so the moment the note exists — or the moment it could not be
made — is announced where the person actually is: the desktop. `notify-send` is the
freedesktop.org client every Linux desktop ships; without it, or without a session
to show it in, nothing happens and nothing breaks.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from pathlib import Path

from simic.settings import Settings

# A notification daemon that does not answer must not hold up the end of a run.
NOTIFY_TIMEOUT_SECONDS = 5.0


def notify(settings: Settings, title: str, body: str) -> None:
    if not settings.notify or not shutil.which("notify-send"):
        return
    with contextlib.suppress(subprocess.TimeoutExpired):
        subprocess.run(
            ["notify-send", "--app-name=simic", title, body],
            capture_output=True,
            check=False,
            timeout=NOTIFY_TIMEOUT_SECONDS,
        )


def note_ready(settings: Settings, md_path: Path) -> None:
    notify(settings, "Meeting note ready", md_path.name)


def note_failed(settings: Settings, name: str, error: BaseException, hint: str) -> None:
    notify(settings, f"{name} was not transcribed", f"{error}\n{hint}")
