"""Regression: live transcription assumed a 48 kHz capture.

Bug: each chunk was decimated by a fixed factor of 3, written for parecord's old 48 kHz
default. Captured at 16 kHz, Whisper's own rate, one second of speech reached the model
as a third of a second played three times too fast.
"""

import wave
from unittest.mock import MagicMock

import numpy as np
import pytest

from simic.live import LiveTranscriber
from simic.settings import Settings


@pytest.mark.parametrize("sample_rate", [16000, 48000])
def test_one_second_of_capture_reaches_whisper_as_one_second(tmp_path, tmp_vault, sample_rate):
    settings = Settings(vault_path=tmp_vault, audio_format="wav", sample_rate=sample_rate)
    live = LiveTranscriber(settings, "live-rate", tmp_path / "mic.wav", tmp_path / "monitor.wav")
    frames: list[tuple[int, int]] = []

    def transcribe(path):
        with wave.open(str(path), "rb") as wf:
            frames.append((wf.getnframes(), wf.getframerate()))
        return [], {}

    transcriber = MagicMock()
    transcriber.transcribe.side_effect = transcribe
    one_second = np.zeros(sample_rate, dtype=np.int16).tobytes()

    live._transcribe_chunk(transcriber, one_second, len(one_second), 0, is_mic=True)

    assert frames == [(16000, 16000)]
