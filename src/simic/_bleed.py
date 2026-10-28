"""The other side heard by the user's mic: measuring it while recording, and taking it
out of the transcript afterwards.

Echo cancellation (_aec) keeps most of it out of the recording. What is left — no
PipeWire canceller, a canceller still converging, a loud room — reaches Whisper as a
second, worse copy of the other person: in one meeting 77 % of what they said came back
under "You", a second or two later, with a Whisper loop where the copy was faintest.

Measuring: a mic that hears the speakers follows the monitor. While the other side
speaks, the loudness of the mic rises and falls with theirs; without echo it does not,
because the user is listening. Measured on one laptop's speakers: an envelope
correlation of 0.84 with echo, at most 0.36 for unrelated audio, 0.17 through the
canceller.

Removing: an echo is the same words in the same seconds. A run of mic words that
repeats a run of monitor words within ECHO_MAX_LAG_SEC is the other side's, and goes;
what the user said around it stays.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from itertools import pairwise

import numpy as np

from simic.models import Segment, Word

# --- measuring -------------------------------------------------------------------

# Loudness is followed in frames this long, and the mic may lag the monitor by up to
# ECHO_MAX_DELAY_SEC: the speakers' output latency and the room.
ENVELOPE_FRAME_SEC = 0.05
ECHO_MAX_DELAY_SEC = 0.5
# A monitor frame louder than this has the other side speaking in it (dBFS; speech
# sits around -30, a quiet room below -60).
ACTIVE_FRAME_DB = -45.0
# The other side must speak for this many frames of the window for it to tell anything.
MIN_ACTIVE_FRAMES = 40
# Correlation above which the mic is taken to hear the speakers: between the 0.84 of a
# laptop's speakers and the 0.36 of unrelated audio.
ECHO_CORRELATION = 0.6
_POWER_FLOOR = 1e-10


def _envelope_db(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(1, int(ENVELOPE_FRAME_SEC * sample_rate))
    count = len(samples) // frame
    frames = samples[: count * frame].astype(np.float64).reshape(count, frame) / 32768.0
    return 10 * np.log10(np.mean(frames**2, axis=1) + _POWER_FLOOR)


def echo_correlation(mic: np.ndarray, monitor: np.ndarray, sample_rate: int) -> float | None:
    """How closely the mic's loudness follows the monitor's while the other side speaks,
    at the best offset within ECHO_MAX_DELAY_SEC: near 1 when the mic hears the speakers.
    None when the other side spoke too little to tell. int16 samples at one rate."""
    mic_db, monitor_db = _envelope_db(mic, sample_rate), _envelope_db(monitor, sample_rate)
    frames = min(len(mic_db), len(monitor_db))
    mic_db, monitor_db = mic_db[:frames], monitor_db[:frames]
    best: float | None = None
    # Both ways: read live, the end of one channel's file may trail the other's.
    reach = int(ECHO_MAX_DELAY_SEC / ENVELOPE_FRAME_SEC)
    for delay in range(-reach, reach + 1):
        heard = mic_db[max(delay, 0) : frames + min(delay, 0)]
        played = monitor_db[max(-delay, 0) : frames - max(delay, 0)]
        active = played > ACTIVE_FRAME_DB
        if active.sum() < MIN_ACTIVE_FRAMES or np.ptp(heard[active]) == 0:
            continue
        correlation = float(np.corrcoef(heard[active], played[active])[0, 1])
        best = correlation if best is None else max(best, correlation)
    return best


# --- removing --------------------------------------------------------------------

# Whisper places a word within a second or two of where the other channel places the
# same word; an echo itself lags by well under a second.
ECHO_MAX_LAG_SEC = 3.0
# Fewer words in a row than this can be the user agreeing ("sí, sí, claro").
ECHO_MIN_WORDS = 4
# Words between two echoed runs are the same echo transcribed a little differently.
ECHO_MAX_GAP_WORDS = 3
_TOKEN_RE = re.compile(r"\w+")


def _token(word: str) -> str:
    return " ".join(_TOKEN_RE.findall(word.lower()))


def _timed_words(segment: Segment) -> list[Word]:
    """The segment's words; spread evenly over it when the STT gave none."""
    if segment.words:
        return segment.words
    tokens = segment.text.split()
    step = (segment.end - segment.start) / max(1, len(tokens))
    return [
        Word(segment.start + i * step, segment.start + (i + 1) * step, f" {t}", 1.0)
        for i, t in enumerate(tokens)
    ]


def _echoed(words: list[Word], monitor_words: list[Word]) -> list[bool]:
    """Which of `words` repeat the monitor's words in the same seconds."""
    mine = [_token(w.word) for w in words]
    near = [
        w
        for w in monitor_words
        if words[0].start - ECHO_MAX_LAG_SEC <= w.start <= words[-1].end + ECHO_MAX_LAG_SEC
    ]
    theirs = [_token(w.word) for w in near]
    echoed = [False] * len(words)
    runs: list[tuple[int, int]] = []
    for block in SequenceMatcher(None, mine, theirs, autojunk=False).get_matching_blocks():
        if block.size < ECHO_MIN_WORDS:
            continue
        lags = [abs(words[block.a + k].start - near[block.b + k].start) for k in range(block.size)]
        if sorted(lags)[len(lags) // 2] <= ECHO_MAX_LAG_SEC:
            runs.append((block.a, block.a + block.size))
    gaps = [
        (end, start) for (_, end), (start, _) in pairwise(runs) if start - end <= ECHO_MAX_GAP_WORDS
    ]
    for start, end in runs + gaps:
        echoed[start:end] = [True] * (end - start)
    return echoed


def drop_echoed_words(mic: list[Segment], monitor: list[Segment]) -> list[Segment]:
    """The mic's segments without the runs of words that repeat the other side. What
    is left of a segment is split where the echo was, each piece keeping its words."""
    monitor_words = [w for s in monitor for w in _timed_words(s)]
    result: list[Segment] = []
    for segment in mic:
        words = _timed_words(segment)
        if not words or not monitor_words:
            result.append(segment)
            continue
        echoed = _echoed(words, monitor_words)
        if not any(echoed):
            result.append(segment)
            continue
        piece: list[Word] = []
        for word, echo in [*zip(words, echoed, strict=True), (None, True)]:
            if word is not None and not echo:
                piece.append(word)
            elif piece:
                if any(_token(w.word) for w in piece):
                    result.append(
                        Segment(
                            start=piece[0].start,
                            end=piece[-1].end,
                            text="".join(w.word for w in piece).strip(),
                            words=piece if segment.words else None,
                            speaker=segment.speaker,
                        )
                    )
                piece = []
    return result
