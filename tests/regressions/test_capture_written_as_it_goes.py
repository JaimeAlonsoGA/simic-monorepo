"""Regression: an MP3 channel stayed at 0 bytes for the first tens of seconds of every
recording, and `simic start` warned that nothing was being written to either channel.

parecord delivered audio all along; ffmpeg held the encoded frames in its output buffer
and wrote them out much later. So the stall warning fired on a healthy recording, the
live checks that read the end of each file (the other side silent, the mic hearing the
speakers) had nothing to read, and a recording cut short would have lost what the
encoder still held. The encoder now writes each frame as it is made.
"""

import shutil
import time
from pathlib import Path

import pytest

from simic.recorder import Recorder
from simic.settings import Settings

# Bounds a wait for files that grow within a second; never asserted on.
_GROW_SECONDS = 10.0


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_an_mp3_channel_is_written_while_it_records(tmp_path, vault_env, fake_parecord):
    recorder = Recorder(state_dir=tmp_path / "state")
    recorder.start(Settings(audio_format="mp3"), session_name="2026-10-15_16-40-00")
    session = recorder.read_session()
    assert session is not None
    files = [Path(session["monitor_path"]), Path(session["mic_path"])]
    try:
        deadline = time.monotonic() + _GROW_SECONDS
        sizes = [0, 0]
        while time.monotonic() < deadline and 0 in sizes:
            time.sleep(0.1)
            sizes = [path.stat().st_size if path.exists() else 0 for path in files]
        assert 0 not in sizes, f"channel files still empty while recording: {sizes}"
    finally:
        recorder.stop()
