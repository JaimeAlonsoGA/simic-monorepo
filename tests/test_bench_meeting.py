"""scripts/make_bench_meeting.py: a two-sided meeting whose references match its audio."""

import json
import shutil
import wave

import pytest

from tests.fixtures import create_mono_wav, load_script

make_bench_meeting = load_script("make_bench_meeting")

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")


def _fleurs(tmp_path):
    """Six distinct sentences of one second each; sentence 1 is read twice, as in FLEURS."""
    fleurs = tmp_path / "fleurs"
    (fleurs / "test").mkdir(parents=True)
    rows = []
    for n in range(1, 7):
        create_mono_wav(fleurs / "test" / f"{n}.wav", duration=1.0, sample_rate=16000)
        rows.append(f"{n}\t{n}.wav\tFrase número {n}.\tfrase número {n}\t-\t16000\tFEMALE")
    create_mono_wav(fleurs / "test" / "1b.wav", duration=1.0, sample_rate=16000)
    rows.append("1\t1b.wav\tFrase número 1.\tfrase número 1\t-\t16000\tMALE")
    (fleurs / "test.tsv").write_text("\n".join(rows) + "\n")
    return fleurs


def test_each_sentence_is_spoken_once_by_one_side(tmp_path):
    manifest = make_bench_meeting.build(
        _fleurs(tmp_path), minutes=4 / 60, out=tmp_path / "m", seed=1
    )

    mic, monitor = json.loads(manifest.read_text())
    assert mic["path"].endswith("mic_16k.wav")
    assert monitor["path"].endswith("monitor_16k.wav")
    spoken = mic["reference"].split(". ") + monitor["reference"].split(". ")
    numbers = sorted(sentence.rstrip(".").removeprefix("Frase número ") for sentence in spoken)
    assert len(numbers) == len(set(numbers))
    assert set(numbers) <= {"1", "2", "3", "4", "5", "6"}
    # The conversation opens on the mic, so the mic always has something to say.
    assert mic["reference"]
    for entry in (mic, monitor):
        with wave.open(entry["path"], "rb") as wf:
            assert (wf.getnchannels(), wf.getframerate()) == (1, 16000)


def test_the_same_seed_builds_the_same_meeting(tmp_path):
    fleurs = _fleurs(tmp_path)
    first = make_bench_meeting.build(fleurs, minutes=4 / 60, out=tmp_path / "a", seed=7)
    second = make_bench_meeting.build(fleurs, minutes=4 / 60, out=tmp_path / "b", seed=7)

    assert [e["reference"] for e in json.loads(first.read_text())] == [
        e["reference"] for e in json.loads(second.read_text())
    ]
    assert (tmp_path / "a" / "meeting.wav").read_bytes() == (
        tmp_path / "b" / "meeting.wav"
    ).read_bytes()


def test_too_few_sentences_for_the_length_asked(tmp_path):
    with pytest.raises(SystemExit, match="ran out of sentences"):
        make_bench_meeting.build(_fleurs(tmp_path), minutes=1.0, out=tmp_path / "m", seed=1)
