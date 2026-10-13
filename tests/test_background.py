"""Transcription after the terminal is given back: the detached `tapeback recover`, what
`tapeback status` shows of it, and the queue that keeps two from running at once."""

import os
import threading

from tapeback._background import last_status, process_in_background
from tapeback.cli import cli
from tapeback.recorder import processing, session_root, transcription_turn
from tapeback.settings import Settings

SESSION = "2026-10-13_09-00-00"


def test_the_detached_process_is_tapeback_recover_and_writes_its_log(
    tmp_path, vault_env, monkeypatch
):
    """A real child, run as the user's would be. A session without audio fails at its
    first step, before any model loads, which is enough to see the whole handoff: the
    command the child runs, where its output goes, and that a failure keeps the
    recording. The child reads .env from its working directory and ~/.config, so both
    point into the test's own tmp dir."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    session = session_root(Settings()) / SESSION
    session.mkdir()

    pid = process_in_background(session, diarize=False, do_summarize=False)
    _, wait_status = os.waitpid(pid, 0)

    assert os.waitstatus_to_exitcode(wait_status) == 1
    log = session / "processing.log"
    assert oct(log.stat().st_mode & 0o777) == "0o600"
    assert log.read_text().splitlines() == [
        f"Recovering {SESSION}...",
        f"Failed to recover {SESSION}: No monitor recording in {session}",
        f"Error: Not recovered: {SESSION}.",
    ]
    assert session.is_dir()


def test_the_detached_process_runs_in_a_session_of_its_own(tmp_path, vault_env, monkeypatch):
    """Ctrl+C in the terminal goes to its foreground process group, and closing the
    terminal hangs up its session: a child left in either dies with the meeting."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    session = session_root(Settings()) / SESSION
    session.mkdir()

    pid = process_in_background(session, diarize=True, do_summarize=True)
    child_session = os.getsid(pid)
    os.waitpid(pid, 0)

    assert child_session == pid
    assert child_session != os.getsid(0)


def test_last_status_is_the_latest_line_printed():
    """Progress lines are indented and the log may end in a blank line."""
    session = session_root(Settings()) / SESSION
    session.mkdir()
    (session / "processing.log").write_text(
        f"Recovering {SESSION}...\n  transcribe mic: 40% (0:24 / 1:00), about 1 min left\n\n"
    )

    assert last_status(session) == "transcribe mic: 40% (0:24 / 1:00), about 1 min left"


def test_last_status_is_empty_without_a_log():
    session = session_root(Settings()) / SESSION
    session.mkdir()

    assert last_status(session) == ""


def test_status_shows_how_far_a_background_transcription_has_got(runner, vault_env):
    session = session_root(Settings()) / SESSION
    session.mkdir()
    (session / "processing.log").write_text(
        f"Recovering {SESSION}...\n  transcribe mic: 40% (24:00 / 1:00:00), about 18 min left\n"
    )

    with processing(session):
        output = runner.invoke(cli, ["status"]).output

    assert (
        f"Being transcribed:\n  {SESSION}  transcribe mic: 40% (24:00 / 1:00:00), "
        "about 18 min left\n" in output
    )
    assert "Not yet transcribed" not in output


def test_status_names_a_foreground_transcription_without_progress(runner, vault_env):
    """`tapeback start --wait` prints its progress to its own terminal, not to a log."""
    session = session_root(Settings()) / SESSION
    session.mkdir()

    with processing(session):
        output = runner.invoke(cli, ["status"]).output

    assert f"Being transcribed:\n  {SESSION}\n" in output


def test_status_lists_nothing_being_transcribed_when_nothing_is(runner, vault_env):
    assert "Being transcribed" not in runner.invoke(cli, ["status"]).output


def test_a_free_turn_is_taken_without_waiting():
    waits: list[str] = []

    with transcription_turn(lambda: waits.append("waited")):
        pass

    assert waits == []


def test_a_second_transcription_waits_for_the_first_to_finish():
    """Two Whisper models at once do not fit on a laptop that holds one: the kernel
    kills both. The second waits its turn and says so."""
    events: list[str] = []
    second_waiting = threading.Event()

    def second() -> None:
        def on_wait() -> None:
            events.append("second waits")
            second_waiting.set()

        with transcription_turn(on_wait):
            events.append("second runs")

    with transcription_turn(lambda: events.append("first waits")):
        thread = threading.Thread(target=second)
        thread.start()
        assert second_waiting.wait(timeout=5)
        events.append("first done")
    thread.join(timeout=5)

    assert events == ["second waits", "first done", "second runs"]
