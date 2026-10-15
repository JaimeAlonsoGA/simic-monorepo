"""Regression: a call on the speakers was transcribed twice — as "Other" from the
monitor and again as "You" from the mic, which heard the speakers.

A 1 h 24 min meeting recorded on a laptop's speakers, its mic at +52 dB: 77 % of what
the other person said came back under "You" a second or two later. The mic was loud
enough through every filter on the way: the gate silences the mic only where it has
under 30 % of the monitor's energy, and a segment is dropped only where the monitor is
twice as loud. The echo is now recognised by its words — the same run of words in the
same seconds — and taken out of the mic's transcript.
"""

import shutil
from unittest.mock import patch

import pytest

from simic.cli import cli
from tests.fixtures import create_stereo_wav_segments, mock_whisper_by_channel

THEIRS = "Thanks, see you on Friday then."


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_the_other_side_heard_by_the_mic_is_not_the_users(runner, tmp_path, vault_env):
    audio = tmp_path / "2026-10-15_12-41-20.wav"
    # (seconds, mic, monitor): the user speaks; then the other side, which the mic
    # hears from the speakers at 60 % of its level.
    create_stereo_wav_segments(audio, 48000, [(2.0, 0.5, 0.0), (2.0, 0.3, 0.5)])
    model = mock_whisper_by_channel(
        mic_segments=[(0.0, 2.0, "I will send the report."), (2.2, 4.0, THEIRS)],
        monitor_segments=[(2.0, 4.0, THEIRS)],
    )

    with patch("simic.transcriber.WhisperModel", return_value=model):
        result = runner.invoke(cli, ["process", str(audio), "--no-diarize", "--no-summarize"])

    assert result.exit_code == 0, result.output + repr(result.exception)
    note = (vault_env / "meetings" / "2026-10-15_12-41-20.md").read_text()
    assert note.endswith(
        "[00:00:00] **You:** I will send the report.\n\n"
        "[00:00:02] **Other:** Thanks, see you on Friday then.\n"
    )
