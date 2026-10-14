"""What the window shows of a meeting, and what its buttons do — through the real
Recorder, its session directory and a fake `parecord`, as `tapeback start` runs them."""

import datetime
import os
import shutil
import time

import pytest

from tapeback import _activity
from tapeback._activity import Activity, Job
from tapeback.recorder import Recorder, processing, session_root
from tapeback.settings import Settings
from tests.fixtures import _RECORDERS_START_SECONDS, create_session_file, process_state

SESSION = "2026-10-13_10-00-00"


@pytest.fixture
def spawned(monkeypatch):
    """The detached `tapeback recover NAME`, recorded rather than run: tests/
    test_background.py runs a real one."""
    calls: list[tuple[str, dict[str, bool]]] = []
    monkeypatch.setattr(
        "tapeback._activity.process_in_background",
        lambda session, **flags: calls.append((session.name, flags)),
    )
    return calls


def test_nothing_going_on(tmp_vault):
    settings = Settings(vault_path=tmp_vault)
    assert _activity.read_activity(settings, Recorder(), 5) == Activity(
        None, (), (), (), more_notes=False
    )


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_record_then_stop_hands_the_meeting_to_a_background_transcription(
    tmp_vault, fake_parecord, spawned
):
    settings = Settings(vault_path=tmp_vault)
    recorder = Recorder()

    name = _activity.start(settings, recorder, "")
    session = recorder.read_session()
    assert session is not None
    assert session["session_name"] == name
    shown = _activity.read_activity(settings, Recorder(), 5)
    assert shown.recording is not None
    assert shown.recording.stalled is False
    assert shown.recording.started_at.tzinfo is not None
    # A person stops a meeting that has been recorded, not one still starting.
    deadline = time.monotonic() + _RECORDERS_START_SECONDS
    while len(fake_parecord()) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert _activity.stop(Recorder()) == name
    assert spawned == [(name, {"diarize": True, "do_summarize": True})]
    assert [process_state(call.pid) for call in fake_parecord()] == [None, None]
    after = _activity.read_activity(settings, Recorder(), 5)
    assert after.recording is None
    # The spawn is recorded, not run: nothing holds the session, so it waits.
    assert after.pending == (name,)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_a_named_meeting_keeps_its_name(tmp_vault, fake_parecord, spawned):
    settings = Settings(vault_path=tmp_vault)
    assert _activity.start(settings, Recorder(), "weekly-sync") == "weekly-sync"
    assert _activity.stop(Recorder()) == "weekly-sync"


def test_a_recording_whose_recorders_died_is_shown_stalled(tmp_vault):
    """Its audio is still there: the window keeps offering Stop."""
    recorder = Recorder()
    recorder.session_file.parent.mkdir(parents=True)
    create_session_file(
        recorder.session_file,
        session_name=SESSION,
        pid_monitor=2**22 + 1,
        pid_mic=2**22 + 2,
        started_at="2026-10-13T10:00:00+02:00",
    )

    shown = _activity.read_activity(Settings(vault_path=tmp_vault), recorder, 5)

    assert shown.recording == _activity.Recording(
        name=SESSION,
        started_at=datetime.datetime.fromisoformat("2026-10-13T10:00:00+02:00"),
        stalled=True,
    )


def test_a_transcription_shows_its_latest_line(tmp_vault):
    settings = Settings(vault_path=tmp_vault)
    session = session_root(settings) / SESSION
    session.mkdir()
    (session / "processing.log").write_text("Transcribing...\n  transcribe mic: 40%\n")

    with processing(session):
        shown = _activity.read_activity(settings, Recorder(), 5)

    assert shown.transcribing == (Job(SESSION, "transcribe mic: 40%"),)
    assert shown.pending == ()


def test_recover_transcribes_only_a_meeting_still_waiting(tmp_vault, spawned):
    settings = Settings(vault_path=tmp_vault)
    (session_root(settings) / SESSION).mkdir()

    _activity.recover(settings, Recorder(), "2026-10-13_11-00-00")
    assert spawned == []

    _activity.recover(settings, Recorder(), SESSION)
    assert spawned == [(SESSION, {"diarize": True, "do_summarize": True})]


def _notes(folder, stamps):
    folder.mkdir(parents=True, exist_ok=True)
    for name, mtime in stamps:
        (folder / name).write_text("")
        os.utime(folder / name, (mtime, mtime))


def test_notes_are_the_newest_first_and_never_a_live_draft(tmp_vault):
    settings = Settings(vault_path=tmp_vault)
    meetings = tmp_vault / "meetings"
    _notes(meetings, [("old.md", 100), ("new.md", 200), ("new_live.md", 300)])
    (meetings / "notes.txt").write_text("")

    notes, more = _activity.recent_notes(settings, 5)

    assert [note.path for note in notes] == [meetings / "new.md", meetings / "old.md"]
    assert notes[0].written_at == datetime.datetime.fromtimestamp(200).astimezone()
    assert more is False


@pytest.mark.parametrize(("limit", "shown", "more"), [(2, 2, True), (3, 3, False), (4, 3, False)])
def test_notes_stop_at_the_limit_and_say_there_are_more(tmp_vault, limit, shown, more):
    _notes(tmp_vault / "meetings", [("a.md", 100), ("b.md", 200), ("c.md", 300)])

    notes, has_more = _activity.recent_notes(Settings(vault_path=tmp_vault), limit)

    assert [note.path.name for note in notes] == ["c.md", "b.md", "a.md"][:shown]
    assert has_more is more


def test_notes_come_from_the_folder_in_the_settings(tmp_vault):
    _notes(tmp_vault / "meetings", [("meeting.md", 100)])
    _notes(tmp_vault / "Voice notes", [("memo.md", 100)])

    notes, _more = _activity.recent_notes(
        Settings(vault_path=tmp_vault, meetings_dir="Voice notes"), 5
    )

    assert [note.path.name for note in notes] == ["memo.md"]


def test_no_notes_before_the_first_transcript(tmp_vault):
    assert _activity.recent_notes(Settings(vault_path=tmp_vault), 5) == ((), False)
    (tmp_vault / "meetings").mkdir()
    assert _activity.recent_notes(Settings(vault_path=tmp_vault), 5) == ((), False)


NOW = datetime.datetime(2026, 10, 13, 17, 30).astimezone()


@pytest.mark.parametrize(
    ("written_at", "label"),
    [
        (datetime.datetime(2026, 10, 13, 0, 0), "Today, 00:00"),
        (datetime.datetime(2026, 10, 13, 17, 5), "Today, 17:05"),
        (datetime.datetime(2026, 10, 12, 23, 59), "Yesterday, 23:59"),
        (datetime.datetime(2026, 10, 11, 9, 0), "11 Oct, 09:00"),
        (datetime.datetime(2026, 1, 1, 9, 0), "1 Jan, 09:00"),
        (datetime.datetime(2025, 12, 31, 9, 0), "31 Dec 2025, 09:00"),
    ],
)
def test_when_a_note_was_written_reads_as_a_person_says_it(written_at, label):
    assert _activity.when(written_at.astimezone(), NOW) == label
