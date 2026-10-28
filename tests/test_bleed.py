"""The other side heard by the user's mic: telling it from the loudness of both
channels, and taking it out of the mic's transcript."""

import numpy as np

from simic._bleed import drop_echoed_words, echo_correlation
from simic.models import Segment, Word

RATE = 16000
FRAME = 800  # 50 ms at 16 kHz: one envelope frame


def _speech(seconds: float, seed: int) -> np.ndarray:
    """Noise whose loudness rises and falls frame by frame, as speech does."""
    rng = np.random.default_rng(seed)
    frames = int(seconds * RATE) // FRAME
    loudness = rng.uniform(0.05, 0.5, frames).repeat(FRAME)
    return (rng.standard_normal(frames * FRAME) * loudness * 8000).astype(np.int16)


# --- measuring ---------------------------------------------------------------------


def _correlation(mic: np.ndarray, monitor: np.ndarray) -> float:
    correlation = echo_correlation(mic, monitor, RATE)
    assert correlation is not None
    return correlation


def test_a_mic_playing_back_the_speakers_follows_them_exactly():
    monitor = _speech(10, seed=1)
    delay = 4 * FRAME  # 0.2 s: the speakers' latency and the room
    mic = np.concatenate([np.zeros(delay, np.int16), monitor[:-delay] // 4])

    assert round(_correlation(mic, monitor), 6) == 1.0


def test_a_mic_file_trailing_the_monitor_file_is_found_too():
    """Read live, the mic's file may end before the monitor's."""
    monitor = _speech(10, seed=1)
    lead = 4 * FRAME
    mic = np.concatenate([monitor[lead:] // 4, np.zeros(lead, np.int16)])

    assert round(_correlation(mic, monitor), 6) == 1.0


def test_a_mic_with_its_own_speech_does_not_follow_the_monitor():
    monitor, mic = _speech(10, seed=1), _speech(10, seed=2)

    assert round(_correlation(mic, monitor), 3) == 0.168


def test_too_little_of_the_other_side_tells_nothing():
    """40 frames of the other side speaking measure; 39 do not."""
    mic = _speech(10, seed=1)
    monitor = np.zeros(10 * RATE, np.int16)
    monitor[: 40 * FRAME] = mic[: 40 * FRAME]
    assert echo_correlation(mic, monitor, RATE) is not None

    monitor[39 * FRAME : 40 * FRAME] = 0
    assert echo_correlation(mic, monitor, RATE) is None


# --- removing ----------------------------------------------------------------------


def _words(text: str, start: float, step: float = 0.4) -> list[Word]:
    return [
        Word(start + i * step, start + (i + 1) * step, f" {w}", 0.9)
        for i, w in enumerate(text.split())
    ]


def _segment(words: list[Word], speaker: str) -> Segment:
    return Segment(
        start=words[0].start,
        end=words[-1].end,
        text="".join(w.word for w in words).strip(),
        words=words,
        speaker=speaker,
    )


THEIRS = "el dinero mueve el mundo, ¿sabes?"


def _other(start: float = 10.0) -> list[Segment]:
    return [_segment(_words(THEIRS, start), "Other")]


def test_the_other_sides_words_leave_the_mic_and_the_users_stay():
    """A segment of the user's words, the echo, and the user again: two pieces."""
    words = _words("yo creo que sí", 0.0) + _words(THEIRS, 10.5) + _words("pero no siempre", 14.0)
    mic = [_segment(words, "You")]

    kept = drop_echoed_words(mic, _other())

    assert [(s.text, s.start, s.speaker) for s in kept] == [
        ("yo creo que sí", 0.0, "You"),
        ("pero no siempre", 14.0, "You"),
    ]
    assert [w.word for w in kept[1].words or []] == [" pero", " no", " siempre"]


def test_four_words_in_a_row_are_an_echo_and_three_are_the_user_agreeing():
    other = [_segment(_words("sí claro eso es justo", 10.0), "Other")]

    four = drop_echoed_words([_segment(_words("sí claro eso es", 10.2), "You")], other)
    three = drop_echoed_words([_segment(_words("sí claro eso", 10.2), "You")], other)

    assert four == []
    assert [s.text for s in three] == ["sí claro eso"]


def test_the_same_words_three_seconds_apart_are_an_echo_and_later_they_are_not():
    at_limit = drop_echoed_words([_segment(_words(THEIRS, 13.0), "You")], _other())
    beyond = drop_echoed_words([_segment(_words(THEIRS, 13.1), "You")], _other())

    assert at_limit == []
    assert [s.text for s in beyond] == ["el dinero mueve el mundo, ¿sabes?"]


def test_a_few_words_heard_differently_inside_an_echo_go_with_it():
    """Three words between two echoed runs are the same echo transcribed differently;
    four are the user's."""
    other = [_segment(_words("uno dos tres cuatro cinco seis siete ocho", 10.0), "Other")]
    three = _words("uno dos tres cuatro", 10.0) + _words("a b c", 11.6)
    three += _words("cinco seis siete ocho", 12.8)
    four = _words("uno dos tres cuatro", 10.0) + _words("a b c d", 11.6)
    four += _words("cinco seis siete ocho", 13.2)

    assert drop_echoed_words([_segment(three, "You")], other) == []
    assert [s.text for s in drop_echoed_words([_segment(four, "You")], other)] == ["a b c d"]


def test_a_transcript_without_word_times_loses_the_echo_too():
    """Remote STT may give segments without words: they are spread over the segment."""
    mic = [Segment(start=10.2, end=12.6, text=THEIRS, speaker="You")]

    assert drop_echoed_words(mic, _other()) == []


def test_without_the_other_side_the_mic_is_left_as_it_is():
    mic = [_segment(_words(THEIRS, 10.0), "You")]

    assert drop_echoed_words(mic, []) is not mic
    assert drop_echoed_words(mic, []) == mic
