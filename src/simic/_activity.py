"""What simic is doing now, and the three things `simic ui` asks of it.

The window holds no state of its own: everything it shows is read back from where the
CLI keeps it — session.json for the recording, the session directories for what is
being transcribed or waits for `recover`, the vault for the last note. A recording
started in a terminal shows up in the window, and the other way round.

Stopping hands the recording to the same detached `simic recover NAME` as `simic
stop`, so closing the window never loses a transcription in progress.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

from simic._background import last_status, process_in_background
from simic.pipeline import stop_recording
from simic.recorder import Recorder
from simic.settings import Settings

_LIVE_NOTE_SUFFIX = "_live"
# GTK applies the system locale, so strftime("%b") would mix it into an English window.
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class Recording:
    name: str
    started_at: datetime.datetime
    # A recorder exited on its own. What it captured is still there to transcribe.
    stalled: bool


@dataclass(frozen=True)
class Job:
    name: str
    status: str


@dataclass(frozen=True)
class Note:
    path: Path
    written_at: datetime.datetime


@dataclass(frozen=True)
class Activity:
    recording: Recording | None
    transcribing: tuple[Job, ...]
    pending: tuple[str, ...]
    # The newest notes, newest first, as many as were asked for.
    notes: tuple[Note, ...]
    # Older notes exist beyond those.
    more_notes: bool


def recent_notes(settings: Settings, limit: int) -> tuple[tuple[Note, ...], bool]:
    """The newest `limit` notes in the vault's transcripts folder, and whether there are
    older ones. A live transcript is a draft being rewritten, not a note."""
    folder = settings.vault_path / settings.meetings_dir
    if not folder.is_dir():
        return (), False
    notes = sorted(
        (
            Note(path, datetime.datetime.fromtimestamp(path.stat().st_mtime).astimezone())
            for path in folder.glob("*.md")
            if not path.stem.endswith(_LIVE_NOTE_SUFFIX)
        ),
        key=lambda note: note.written_at,
        reverse=True,
    )
    return tuple(notes[:limit]), len(notes) > limit


def when(written_at: datetime.datetime, now: datetime.datetime) -> str:
    """ "Today, 17:05", "Yesterday, 09:30", "11 Oct, 09:00", "31 Dec 2025, 09:00"."""
    clock = f"{written_at:%H:%M}"
    days = (now.date() - written_at.date()).days
    if days == 0:
        return f"Today, {clock}"
    if days == 1:
        return f"Yesterday, {clock}"
    day = f"{written_at.day} {_MONTHS[written_at.month - 1]}"
    if written_at.year != now.year:
        day = f"{day} {written_at.year}"
    return f"{day}, {clock}"


def read_activity(settings: Settings, recorder: Recorder, note_limit: int) -> Activity:
    recording = None
    notes, more_notes = recent_notes(settings, note_limit)
    session = recorder.read_session()
    if session is not None:
        recording = Recording(
            name=session["session_name"],
            started_at=datetime.datetime.fromisoformat(session["started_at"]),
            stalled=not recorder.is_recording(),
        )
    return Activity(
        recording=recording,
        transcribing=tuple(
            Job(path.name, last_status(path)) for path in recorder.sessions_in_progress(settings)
        ),
        pending=tuple(path.name for path in recorder.leftover_sessions(settings)),
        notes=notes,
        more_notes=more_notes,
    )


def start(settings: Settings, recorder: Recorder, name: str | None) -> str:
    """Start recording. Returns the session's name."""
    return recorder.start(settings, session_name=name or None)


def stop(recorder: Recorder) -> str:
    """Stop recording and transcribe in the background. Returns the session's name."""
    session = stop_recording(recorder)
    process_in_background(session, diarize=True, do_summarize=True)
    return session.name


def recover(settings: Settings, recorder: Recorder, name: str) -> None:
    """Transcribe a recording whose processing failed or was interrupted, in the
    background. One that is no longer waiting — done meanwhile, or being processed —
    is left alone."""
    for session in recorder.leftover_sessions(settings):
        if session.name == name:
            process_in_background(session, diarize=True, do_summarize=True)
