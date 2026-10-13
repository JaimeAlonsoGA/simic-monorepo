#!/usr/bin/env python
"""Build a benchmark meeting with a reference transcript, from public speech.

Word error rate needs a transcript known to be right, and no one has one for their own
meetings. FLEURS (Google, CC-BY-4.0) has read sentences with their text in 102
languages. This script strings them into a two-sided conversation — turns that
alternate between "you" (the mic, left) and "the others" (the monitor, right), with
the pauses of a conversation between them — and then runs it through tapeback's own
channel split and mic gate, so the benchmark is handed exactly what Whisper gets in
production.

Read speech is cleaner than a meeting: no crosstalk, no codec, no room. The corpus
ranks configurations against each other; it does not predict a real meeting's WER.

Download one language's test split first (about 580 MB):

    lang=es_419; base=https://huggingface.co/datasets/google/fleurs/resolve/main/data
    mkdir -p tmp/fleurs/$lang
    curl -L -o tmp/fleurs/$lang/test.tsv $base/$lang/test.tsv
    curl -L $base/$lang/audio/test.tar.gz | tar -xz -C tmp/fleurs/$lang

Then:

    uv run python scripts/make_bench_meeting.py --fleurs tmp/fleurs/es_419 \\
        --minutes 15 --out tmp/bench-es
    uv run python scripts/bench_transcribe.py --manifest tmp/bench-es/manifest.json \\
        --device cpu --compute-types int8 --language es
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tapeback import const
from tapeback.audio import gate_wav_inactive, split_channels_16k
from tapeback.channel import load_stereo_channels

# FLEURS test.tsv columns: id, file name, raw transcription, normalised transcription,
# characters, sample count, speaker gender. One sentence is read by several speakers.
_COL_ID, _COL_FILE, _COL_TEXT = 0, 1, 2

# A turn is a few sentences by one side; the gaps are a conversation's, not a lecture's.
TURN_SENTENCES = (1, 3)
GAP_WITHIN_TURN = (0.2, 0.6)
GAP_BETWEEN_TURNS = (0.4, 1.5)
# A microphone is never digitally silent: room tone at about -55 dBFS.
MIC_NOISE_RMS = 0.0018


def _read_sentences(fleurs: Path, rng: random.Random) -> list[tuple[Path, str]]:
    """(audio, text) for each distinct sentence, in a seeded random order."""
    seen: set[str] = set()
    sentences = []
    with (fleurs / "test.tsv").open(newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row[_COL_ID] in seen:
                continue
            seen.add(row[_COL_ID])
            sentences.append((fleurs / "test" / row[_COL_FILE], row[_COL_TEXT]))
    rng.shuffle(sentences)
    return sentences


def _load(path: Path) -> np.ndarray:
    """16 kHz mono int16 samples. Through ffmpeg: FLEURS stores float32 WAVs, which the
    standard library's wave module cannot read."""
    pcm = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(const.SAMPLE_RATE_16K),
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(pcm, dtype=np.int16)


def build(fleurs: Path, minutes: float, out: Path, seed: int) -> Path:
    """Write the meeting and its manifest into `out`. Returns the manifest path."""
    # Seeds a reproducible corpus; nothing here is secret.
    rng = random.Random(seed)  # noqa: S311
    sentences = iter(_read_sentences(fleurs, rng))
    target = int(minutes * 60 * const.SAMPLE_RATE_16K)

    # Each side's audio as (start sample, samples); its text in speaking order.
    placed: dict[str, list[tuple[int, np.ndarray]]] = {"mic": [], "monitor": []}
    texts: dict[str, list[str]] = {"mic": [], "monitor": []}
    cursor = 0
    side = "monitor"
    while cursor < target:
        side = "mic" if side == "monitor" else "monitor"
        for index in range(rng.randint(*TURN_SENTENCES)):
            try:
                audio_path, text = next(sentences)
            except StopIteration:
                raise SystemExit("ran out of sentences — ask for fewer minutes") from None
            if index:
                cursor += int(rng.uniform(*GAP_WITHIN_TURN) * const.SAMPLE_RATE_16K)
            samples = _load(audio_path)
            placed[side].append((cursor, samples))
            texts[side].append(text)
            cursor += len(samples)
        cursor += int(rng.uniform(*GAP_BETWEEN_TURNS) * const.SAMPLE_RATE_16K)

    channels = {}
    for name, clips in placed.items():
        channel = np.zeros(cursor, dtype=np.float32)
        for start, samples in clips:
            channel[start : start + len(samples)] += samples / 32768.0
        channels[name] = channel
    noise = np.random.default_rng(seed).normal(0.0, MIC_NOISE_RMS, cursor)
    channels["mic"] += noise.astype(np.float32)

    out.mkdir(parents=True, exist_ok=True)
    stereo = out / "meeting.wav"
    interleaved = np.empty(cursor * 2, dtype=np.int16)
    interleaved[0::2] = np.clip(channels["mic"] * 32767, -32768, 32767).astype(np.int16)
    interleaved[1::2] = np.clip(channels["monitor"] * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(stereo), "wb") as wf:
        wf.setnchannels(const.STEREO_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(const.SAMPLE_RATE_16K)
        wf.writeframes(interleaved.tobytes())

    # What the pipeline hands Whisper: normalised 16 kHz channels, the mic gated.
    mic_16k, monitor_16k = split_channels_16k(stereo, out)
    mic_raw, monitor_raw, raw_sr = load_stereo_channels(stereo)
    gate_wav_inactive(mic_16k, mic_raw, monitor_raw, raw_sr)

    manifest = out / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {"path": str(mic_16k.resolve()), "reference": " ".join(texts["mic"])},
                {"path": str(monitor_16k.resolve()), "reference": " ".join(texts["monitor"])},
            ],
            indent=2,
            ensure_ascii=False,
        )
    )
    spoken = {name: sum(len(s) for _, s in clips) for name, clips in placed.items()}
    print(
        f"{cursor / const.SAMPLE_RATE_16K / 60:.1f} min meeting: "
        f"mic speaks {spoken['mic'] / const.SAMPLE_RATE_16K / 60:.1f} min "
        f"({len(texts['mic'])} sentences), "
        f"monitor {spoken['monitor'] / const.SAMPLE_RATE_16K / 60:.1f} min "
        f"({len(texts['monitor'])} sentences)"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleurs", type=Path, required=True, help="one language's directory")
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    print(f"Wrote {build(args.fleurs, args.minutes, args.out, args.seed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
