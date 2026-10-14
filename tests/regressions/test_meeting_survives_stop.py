"""Regression: a meeting that ended in any way but `simic stop` was never transcribed.

Bug 1 — Ctrl+C. A terminal sends SIGINT to its whole foreground process group, and
the recorders shared `simic start`'s group, so they died of the same keystroke.
By the time the wait loop asked is_recording(), they were gone; the command took that
for a peer `simic stop` that had already processed the meeting, and returned 0. No
note, no message, the audio orphaned in the sessions directory. The in-process flow
test raises KeyboardInterrupt without signalling anyone, so it never saw this.

Bug 2 — a recorder dying mid-meeting. is_recording() deleted the session file as soon
as one recorder was gone, and `start` returned silently the same way: an hour of
meeting lost to an unplugged headset.

Bug 3 — a session left by a `start` that was killed. Its file was deleted by the next
is_recording() and any recorder still alive kept recording, unowned, until logout.
"""

import json
import subprocess
from unittest.mock import patch

import pytest

from simic.cli import cli
from simic.recorder import Recorder
from simic.settings import Settings
from tests.fixtures import mock_whisper_by_channel, process_state

pytestmark = pytest.mark.usefixtures("cpu_only")

SESSION = "2026-10-12_09-00-00"
START = ["start", SESSION, "--no-diarize", "--no-summarize", "--no-live"]
MEETING = (
    "[00:00:00] **You:** I will send the report.\n\n[00:00:02] **Other:** Thanks, see you Friday.\n"
)


def test_ctrl_c_in_a_terminal_transcribes_the_meeting(vault_env, fake_parecord, ctrl_c_in_terminal):
    run = ctrl_c_in_terminal(START)

    assert run.returncode == 0, run.output
    assert "Stopping..." in run.output
    assert (vault_env / "meetings" / f"{SESSION}.md").read_text().endswith(MEETING)
    # The keystroke reached simic alone; simic stopped the recorders itself, in
    # order, so each encoder read its input to the end.
    assert [call.stopped_by for call in fake_parecord()] == ["TERM", "TERM"]
    assert [process_state(call.pid) for call in fake_parecord()] == [None, None]


@pytest.mark.usefixtures("mic_recorder_dies")
def test_a_recorder_dying_mid_meeting_still_transcribes_what_was_captured(
    runner, vault_env, fake_parecord
):
    model = mock_whisper_by_channel(
        mic_segments=[(0.0, 2.0, "I will send the report.")],
        monitor_segments=[(2.0, 4.0, "Thanks, see you Friday.")],
    )
    with patch("simic.transcriber.WhisperModel", return_value=model):
        result = runner.invoke(cli, START)

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "A recorder stopped on its own (mic). Transcribing what was captured." in result.output
    assert (vault_env / "meetings" / f"{SESSION}.md").read_text().endswith(MEETING)
    assert [process_state(call.pid) for call in fake_parecord()] == [None, None]


def test_a_new_recording_stops_what_a_killed_start_left_running(tmp_vault, fake_parecord):
    survivor = subprocess.Popen(["sleep", "100"])
    gone = subprocess.Popen(["true"])
    gone.wait()
    recorder = Recorder()
    recorder.session_file.parent.mkdir(parents=True, exist_ok=True)
    recorder.session_file.write_text(
        json.dumps(
            {
                "pid_monitor": survivor.pid,
                "pid_mic": gone.pid,
                "session_name": "killed",
                "monitor_path": "unused",
                "mic_path": "unused",
                "started_at": "2026-10-12T09:00:00+02:00",
            }
        )
    )

    try:
        assert recorder.start(Settings(vault_path=tmp_vault), "next") == "next"
        assert process_state(survivor.pid) is None
        assert json.loads(recorder.session_file.read_text())["session_name"] == "next"
    finally:
        survivor.kill()
        recorder.stop()
