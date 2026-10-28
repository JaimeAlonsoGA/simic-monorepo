"""A meeting recorded the way a user records one: `simic start`, Ctrl+C, a note.

Everything inside simic is real — the Recorder, the session directory and its
privacy check, the stop signals, ffmpeg, the pipeline, the vault. Only what sits
outside it is faked, at its boundary: `parecord` (a script on PATH writing real WAVs),
the Whisper model (answering per channel), and the user's Ctrl+C. The checks this
replaces mocked simic's own parts, so no test ever ran this path end to end.

Writing it exposed a stop that always waited out its five-second kill timeout. A flow
cannot see a wait — spawning ffmpeg afterwards reaps the leftover processes anyway —
so that check lives in tests/regressions/test_stop_reaps_recorders.py.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from simic._watch import CaptureWatch, EchoWatch, SilenceWatch
from simic.cli import cli
from simic.recorder import processing
from tests.fixtures import mock_whisper_by_channel, process_state

pytestmark = pytest.mark.usefixtures("cpu_only")

SESSION = "2026-10-06_10-30-00"
START = ["start", SESSION, "--no-diarize", "--no-summarize", "--no-live"]


def _meeting():
    return mock_whisper_by_channel(
        mic_segments=[(0.0, 2.0, "I will send the report.")],
        monitor_segments=[(2.0, 4.0, "Thanks, see you Friday.")],
    )


def _sessions_root() -> Path:
    return Path(os.environ["XDG_STATE_HOME"]) / "simic" / "sessions"


@pytest.mark.usefixtures("ctrl_c_while_recording")
@pytest.mark.parametrize("audio_format", ["mp3", "wav"])
def test_start_then_ctrl_c_records_transcribes_and_cleans_up(
    runner, vault_env, fake_parecord, fake_notify_send, monkeypatch, audio_format
):
    monkeypatch.setenv("SIMIC_AUDIO_FORMAT", audio_format)
    with patch("simic.transcriber.WhisperModel", return_value=_meeting()):
        result = runner.invoke(cli, START)

    assert result.exit_code == 0, result.output + repr(result.exception)

    # Both channels were recorded into one private directory under the XDG state dir.
    recorders = fake_parecord()
    assert sorted(call.device for call in recorders) == ["fake.mic", "fake.monitor"]
    assert [call.dir_mode for call in recorders] == ["700", "700"]
    # Nothing is left recording once the meeting is processed.
    assert [process_state(call.pid) for call in recorders] == [None, None]

    # The note says who said what, and when.
    note = (vault_env / "meetings" / f"{SESSION}.md").read_text()
    assert "date: 2026-10-06\n" in note
    assert 'time: "10:30"\n' in note
    assert 'duration: "00:00:04"\n' in note
    assert note.endswith(
        "[00:00:00] **You:** I will send the report.\n\n"
        "[00:00:02] **Other:** Thanks, see you Friday.\n"
    )

    # The note does not keep a link, and the vault copy is gone.
    assert "audio:" not in note
    assert not (vault_env / "attachments").exists()

    # The desktop hears about it: nobody watches a terminal through a long transcription.
    assert fake_notify_send() == [("Meeting note ready", f"{SESSION}.md")]

    # Processed, so nothing is left to recover and nothing is still recording.
    assert list(_sessions_root().iterdir()) == []
    assert not (Path(os.environ["XDG_STATE_HOME"]) / "simic" / "session.json").exists()


@pytest.mark.usefixtures("ctrl_c_while_recording")
def test_start_in_the_background_gives_the_terminal_back_and_recover_makes_the_note(
    runner, vault_env, fake_parecord, fake_notify_send, monkeypatch
):
    """Ctrl+C stops the recording and returns at once; the note is made by
    `simic recover NAME` in a process of its own (tests/test_background.py spawns a
    real one). Here the spawn is recorded and its command run in-process, where the
    Whisper fake reaches it."""
    monkeypatch.setenv("SIMIC_BACKGROUND", "true")
    spawned = []
    monkeypatch.setattr(
        "simic.cli.process_in_background", lambda session, **flags: spawned.append(flags)
    )

    result = runner.invoke(cli, START)

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert spawned == [{"diarize": False, "do_summarize": False}]
    assert result.stderr.endswith(
        f"Transcribing {SESSION} in the background. A notification will say when the "
        "note is ready.\n'simic status' shows how far it has got.\n"
    )
    assert [process_state(call.pid) for call in fake_parecord()] == [None, None]
    assert not (vault_env / "meetings").exists()
    assert sorted(p.name for p in (_sessions_root() / SESSION).iterdir()) == [
        "mic.mp3",
        "monitor.mp3",
    ]

    with patch("simic.transcriber.WhisperModel", return_value=_meeting()):
        recovered = runner.invoke(cli, ["recover", SESSION, "--no-diarize", "--no-summarize"])

    assert recovered.exit_code == 0, recovered.output + repr(recovered.exception)
    assert (
        (vault_env / "meetings" / f"{SESSION}.md")
        .read_text()
        .endswith(
            "[00:00:00] **You:** I will send the report.\n\n"
            "[00:00:02] **Other:** Thanks, see you Friday.\n"
        )
    )
    assert fake_notify_send() == [("Meeting note ready", f"{SESSION}.md")]
    assert list(_sessions_root().iterdir()) == []


@pytest.mark.usefixtures("ctrl_c_while_recording")
def test_start_with_wait_transcribes_in_the_foreground(runner, vault_env, monkeypatch):
    monkeypatch.setenv("SIMIC_BACKGROUND", "true")
    spawned = []
    monkeypatch.setattr(
        "simic.cli.process_in_background", lambda session, **flags: spawned.append(flags)
    )

    with patch("simic.transcriber.WhisperModel", return_value=_meeting()):
        result = runner.invoke(cli, [*START, "--wait"])

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert spawned == []
    assert "in the background" not in result.stderr
    assert (vault_env / "meetings" / f"{SESSION}.md").exists()
    assert list(_sessions_root().iterdir()) == []


def test_failed_processing_keeps_the_recording_and_recover_finishes_it(
    runner, vault_env, fake_parecord, ctrl_c_while_recording, fake_notify_send
):
    """Sessions outlive a reboot so that a meeting whose processing failed can still be
    processed. That promise is only worth something if a failure leaves the audio, and
    one command turns it into the note."""
    model = MagicMock()
    model.transcribe.side_effect = RuntimeError("decoder exploded")

    with patch("simic.transcriber.WhisperModel", return_value=model):
        result = runner.invoke(cli, START)

    assert result.exit_code == 1
    assert str(result.exception) == "decoder exploded"
    assert fake_notify_send() == [
        (
            f"{SESSION} was not transcribed",
            f"decoder exploded\nThe recording is kept: run 'simic recover {SESSION}'.",
        )
    ]
    session = _sessions_root() / SESSION
    assert sorted(p.name for p in session.iterdir()) == [
        ".processing.lock",
        "mic.mp3",
        "mic_16k.wav",
        "monitor.mp3",
        "monitor_16k.wav",
        "stereo.wav",
    ]
    assert [process_state(call.pid) for call in fake_parecord()] == [None, None]
    assert not (vault_env / "meetings").exists()
    assert not (vault_env / "attachments").exists()

    status = runner.invoke(cli, ["status"])
    assert f"Not yet transcribed (run 'simic recover'):\n  {SESSION}\n" in status.output

    # While another process turns it into a note, it is neither pending nor recoverable.
    with processing(session):
        busy = runner.invoke(cli, ["status"]).output
        assert f"Being transcribed:\n  {SESSION}\n" in busy
        assert "Not yet transcribed" not in busy
        assert runner.invoke(cli, ["recover"]).stderr == "Nothing to recover.\n"

    with patch("simic.transcriber.WhisperModel", return_value=_meeting()):
        recovered = runner.invoke(cli, ["recover", "--no-diarize", "--no-summarize"])

    assert recovered.exit_code == 0, recovered.output + repr(recovered.exception)
    assert (
        (vault_env / "meetings" / f"{SESSION}.md")
        .read_text()
        .endswith(
            "[00:00:00] **You:** I will send the report.\n\n"
            "[00:00:02] **Other:** Thanks, see you Friday.\n"
        )
    )
    assert list(_sessions_root().iterdir()) == []
    assert fake_notify_send()[-1] == ("Meeting note ready", f"{SESSION}.md")
    assert runner.invoke(cli, ["recover"]).stderr == "Nothing to recover.\n"


@pytest.mark.usefixtures("ctrl_c_while_recording")
def test_kept_audio_lands_in_the_vault_and_the_note_links_it(
    runner, vault_env, fake_parecord, monkeypatch
):
    monkeypatch.setenv("SIMIC_KEEP_AUDIO", "true")

    with patch("simic.transcriber.WhisperModel", return_value=_meeting()):
        result = runner.invoke(cli, START)

    assert result.exit_code == 0, result.output + repr(result.exception)
    note = (vault_env / "meetings" / f"{SESSION}.md").read_text()
    assert f'audio: "[[attachments/audio/{SESSION}.mp3]]"\n' in note
    assert [p.name for p in (vault_env / "attachments" / "audio").iterdir()] == [f"{SESSION}.mp3"]
    assert list(_sessions_root().iterdir()) == []


@pytest.mark.usefixtures("ctrl_c_while_recording")
@pytest.mark.parametrize("mode", [0o700, 0o755], ids=["private dir records", "shared dir refused"])
def test_start_records_only_into_a_directory_no_one_else_can_reach(
    runner, vault_env, fake_parecord, monkeypatch, tmp_path, mode
):
    """The same command on the same directory, one permission bit apart: the refusal is
    the privacy check and nothing else, and it happens before anything is recorded."""
    chosen = tmp_path / "sessions"
    chosen.mkdir()
    chosen.chmod(mode)
    monkeypatch.setenv("SIMIC_SESSIONS_DIR", str(chosen))

    with patch("simic.transcriber.WhisperModel", return_value=_meeting()):
        result = runner.invoke(cli, START)

    if mode == 0o700:
        assert result.exit_code == 0, result.output + repr(result.exception)
        assert [call.dir_mode for call in fake_parecord()] == ["700", "700"]
        assert (vault_env / "meetings" / f"{SESSION}.md").exists()
    else:
        assert result.exit_code == 1
        assert str(result.exception) == (
            f"Refusing to use {chosen}: reachable by other users (mode 0755, expected 0700). "
            f"If it is yours, run: chmod 700 {chosen}"
        )
        assert fake_parecord() == []
        assert list(chosen.iterdir()) == []


@pytest.mark.usefixtures("ctrl_c_while_recording")
def test_start_warns_when_a_channel_is_not_being_written(
    runner, vault_env, fake_parecord, monkeypatch
):
    """The fake recorders write their audio at once and then nothing more: with the
    threshold at zero, both channels read as stalled at the first look."""
    monkeypatch.setattr(
        "simic.cli.CaptureWatch",
        lambda files: CaptureWatch(files, stall_seconds=0.0),
    )

    with patch("simic.transcriber.WhisperModel", return_value=_meeting()):
        result = runner.invoke(cli, START)

    assert result.exit_code == 0, result.output + repr(result.exception)
    for channel in ("monitor", "mic"):
        assert result.stderr.count(f"nothing has been written to the {channel} channel") == 1


@pytest.mark.usefixtures("ctrl_c_while_recording")
def test_start_warns_when_the_other_side_is_silent(runner, vault_env, fake_parecord, monkeypatch):
    """Checked at every look and silent from the first: the warning comes once."""
    monkeypatch.setattr(
        "simic.cli.SilenceWatch",
        lambda files: SilenceWatch(
            files, measure=lambda _p, _w: -91.0, check_every=0.0, silent_after=0.0
        ),
    )

    with patch("simic.transcriber.WhisperModel", return_value=_meeting()):
        result = runner.invoke(cli, START)

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert result.stderr.count("Warning: no sound from the other side for 0 min.") == 1


@pytest.mark.usefixtures("ctrl_c_while_recording")
def test_start_warns_when_the_mic_hears_the_speakers(runner, vault_env, fake_parecord, monkeypatch):
    """The Ctrl+C comes after one look, so one check is made to be enough here; the
    two-in-a-row rule has its own tests in test_watch.py."""
    monkeypatch.setattr("simic._watch.ECHO_CONFIRM_CHECKS", 1)
    monkeypatch.setattr(
        "simic.cli.EchoWatch",
        lambda mic, monitor: EchoWatch(mic, monitor, measure=lambda _m, _o: 0.84, check_every=0.0),
    )

    with patch("simic.transcriber.WhisperModel", return_value=_meeting()):
        result = runner.invoke(cli, START)

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert (
        result.stderr.count(
            "Warning: your mic is picking up the other side from the speakers, so the note would "
            "repeat what they say as yours. Use headphones, or turn the speakers down."
        )
        == 1
    )


# --- the mic through PipeWire's echo canceller --------------------------------------


@pytest.mark.usefixtures("ctrl_c_while_recording")
def test_the_mic_is_recorded_through_the_echo_canceller_and_it_goes_with_the_recording(
    runner, vault_env, fake_parecord, fake_pw_cli
):
    with patch("simic.transcriber.WhisperModel", return_value=_meeting()):
        result = runner.invoke(cli, START)

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "Echo cancellation: on\n" in result.output
    # The mic is read from the canceller's source; the monitor as before.
    assert sorted(call.device for call in fake_parecord()) == ["fake.monitor", "simic.echo-cancel"]
    # The canceller lived exactly as long as the recording: stopped with it, and gone.
    [canceller] = fake_pw_cli()
    assert canceller.stopped_by == "TERM"
    assert process_state(canceller.pid) is None
    note = (vault_env / "meetings" / f"{SESSION}.md").read_text()
    assert note.endswith(
        "[00:00:00] **You:** I will send the report.\n\n"
        "[00:00:02] **Other:** Thanks, see you Friday.\n"
    )


@pytest.mark.usefixtures("ctrl_c_while_recording")
def test_without_a_canceller_the_plain_mic_is_recorded_and_the_user_told(
    runner, vault_env, fake_parecord
):
    """isolate_pipewire's pw-cli: PipeWire without the WebRTC plugin."""
    with patch("simic.transcriber.WhisperModel", return_value=_meeting()):
        result = runner.invoke(cli, START)

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert (
        "Warning: echo cancellation unavailable — PipeWire's echo-cancel module did not load "
        "(is the WebRTC AEC plugin installed?). On speakers, the mic will also record the "
        "other side: use headphones.\n"
    ) in result.output
    assert sorted(call.device for call in fake_parecord()) == ["fake.mic", "fake.monitor"]
    assert (vault_env / "meetings" / f"{SESSION}.md").exists()


@pytest.mark.usefixtures("ctrl_c_while_recording")
def test_echo_cancellation_turned_off_records_the_plain_mic(
    runner, vault_env, fake_parecord, fake_pw_cli, monkeypatch
):
    monkeypatch.setenv("SIMIC_ECHO_CANCEL", "false")
    with patch("simic.transcriber.WhisperModel", return_value=_meeting()):
        result = runner.invoke(cli, START)

    assert result.exit_code == 0, result.output + repr(result.exception)
    assert "Echo cancellation: off (SIMIC_ECHO_CANCEL)\n" in result.output
    assert fake_pw_cli() == []
    assert sorted(call.device for call in fake_parecord()) == ["fake.mic", "fake.monitor"]
