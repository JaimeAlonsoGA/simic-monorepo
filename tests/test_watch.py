"""The live view of a recording in progress: elapsed time, sizes, stalled and silent
channels."""

import shutil
import subprocess

import pytest

from tapeback._watch import CaptureStatus, CaptureWatch, SilenceWatch, format_status, peak_db
from tests.fixtures import create_mono_wav


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _grow(path, size: int) -> None:
    path.write_bytes(b"\0" * size)


def test_a_growing_channel_is_never_stalled(tmp_path):
    clock = _Clock()
    mic = tmp_path / "mic.mp3"
    watch = CaptureWatch({"mic": mic}, clock=clock, stall_seconds=15.0)

    for size in (1000, 2000, 3000):
        clock.now += 10.0
        _grow(mic, size)
        status = watch.poll()

    assert status == CaptureStatus(elapsed=30.0, sizes={"mic": 3000}, stalled=[])


def test_a_channel_stalls_at_exactly_the_threshold(tmp_path):
    clock = _Clock()
    monitor, mic = tmp_path / "monitor.mp3", tmp_path / "mic.mp3"
    watch = CaptureWatch({"monitor": monitor, "mic": mic}, clock=clock, stall_seconds=15.0)
    _grow(monitor, 500)
    _grow(mic, 500)
    watch.poll()

    clock.now += 14.9
    _grow(monitor, 900)
    assert watch.poll().stalled == []

    clock.now += 0.1
    assert watch.poll().stalled == ["mic"]


def test_a_stalled_channel_recovers_when_it_grows_again(tmp_path):
    clock = _Clock()
    mic = tmp_path / "mic.mp3"
    watch = CaptureWatch({"mic": mic}, clock=clock, stall_seconds=15.0)
    _grow(mic, 500)
    watch.poll()
    clock.now += 20.0
    assert watch.poll().stalled == ["mic"]

    _grow(mic, 800)
    assert watch.poll().stalled == []


def test_a_file_not_created_yet_counts_as_empty(tmp_path):
    clock = _Clock()
    watch = CaptureWatch({"mic": tmp_path / "mic.mp3"}, clock=clock, stall_seconds=15.0)
    clock.now += 15.0

    assert watch.poll() == CaptureStatus(elapsed=15.0, sizes={"mic": 0}, stalled=["mic"])


def test_status_line():
    status = CaptureStatus(
        elapsed=3754.0, sizes={"monitor": 30_100_000, "mic": 29_960_000}, stalled=[]
    )
    assert format_status(status) == "● REC 1:02:34  monitor 30.1 MB  mic 30.0 MB"


# --- silence ---


def _mp3(tmp_path, name, amplitude):
    wav = tmp_path / f"{name}.wav"
    create_mono_wav(wav, duration=40.0, sample_rate=16000, amplitude=amplitude)
    mp3 = tmp_path / f"{name}.mp3"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-b:a", "64k", str(mp3)], check=True)
    return mp3


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_peak_of_silence_and_of_sound(tmp_path):
    silent = peak_db(_mp3(tmp_path, "silent", 0.0), 30.0)
    loud = peak_db(_mp3(tmp_path, "loud", 0.5), 30.0)

    assert silent is not None and silent < -60.0
    assert loud is not None and -7.0 < loud < -5.0


def test_peak_of_a_missing_file_is_unknown(tmp_path):
    assert peak_db(tmp_path / "nowhere.mp3", 30.0) is None


class _Levels:
    """A channel's peak at each check, in order."""

    def __init__(self, peaks):
        self._peaks = iter(peaks)

    def __call__(self, _path, _window):
        return next(self._peaks)


def test_silence_is_reported_once_it_has_lasted(tmp_path):
    clock = _Clock()
    watch = SilenceWatch(
        {"monitor": tmp_path / "monitor.mp3"},
        measure=_Levels([-91.0, -91.0, -91.0]),
        clock=clock,
        check_every=60.0,
        silent_after=180.0,
    )
    clock.now += 60.0  # silent since 30 s before this check
    assert watch.poll() == {}
    clock.now += 60.0
    assert watch.poll() == {}
    clock.now += 60.0  # 150 s silent
    assert watch.poll() == {}
    clock.now += 30.0  # between checks: 180 s, exactly the threshold
    assert watch.poll() == {"monitor": 180.0}


def test_sound_resets_the_silence(tmp_path):
    clock = _Clock()
    watch = SilenceWatch(
        {"monitor": tmp_path / "monitor.mp3"},
        measure=_Levels([-91.0, -12.0]),
        clock=clock,
        check_every=60.0,
        silent_after=30.0,
    )
    clock.now += 60.0
    assert watch.poll() == {"monitor": 30.0}
    clock.now += 60.0
    assert watch.poll() == {}


def test_an_unmeasurable_check_changes_nothing(tmp_path):
    clock = _Clock()
    watch = SilenceWatch(
        {"monitor": tmp_path / "monitor.mp3"},
        measure=_Levels([-91.0, None]),
        clock=clock,
        check_every=60.0,
        silent_after=60.0,
    )
    clock.now += 60.0
    watch.poll()
    clock.now += 60.0
    assert watch.poll() == {"monitor": 90.0}
