"""Tests for whole-sentence encoding and decoding."""
import numpy as np
import pytest

from src.codec import constants as C
from src.codec.chord_table import CHORD_BY_SYMBOL, SPACE, SPARE_CHORDS, SYMBOL_BY_CHORD
from src.codec.message import UNDECODABLE, decode, encode
from src.codec.spectrum import detect_chord, frame_peaks
from src.codec.waveform import chord_clip, frame_amplitudes, radius_profile


def test_frame_count_includes_two_sentinels():
    assert encode("AB").frames.shape == ((2 + 2) * C.FRAMES_PER_CHAR, C.N_BINS)


def test_message_opens_with_the_sentinel():
    peak_frame = encode("A").frames[:C.ACTIVE_FRAMES][3]
    assert detect_chord(peak_frame)[0] == C.SENTINEL_CHORD


def test_unsupported_characters_are_reported():
    assert encode("A@B").dropped == ['@']


def test_normalized_text_is_returned():
    """Normalization is lossy, so callers must be able to see what was encoded
    rather than re-deriving it."""
    result = encode("hello@")
    assert result.text == "HELLO"
    assert result.dropped == ['@']


def test_interior_space_is_an_excited_ring_like_any_other():
    """Space stopped being the absence of a signal and became a signal."""
    frames = encode("A B").frames
    space_clip = frames[2 * C.FRAMES_PER_CHAR:3 * C.FRAMES_PER_CHAR]
    assert not np.allclose(space_clip, C.REST_RADIUS)
    assert detect_chord(space_clip[3])[0] == CHORD_BY_SYMBOL[SPACE]


def test_empty_input_encodes_to_bare_sentinels():
    """Nothing representable survives normalization, so no character clip is
    emitted between the two sentinels."""
    result = encode("@@@")
    assert result.text == ""
    assert result.frames.shape == (2 * C.FRAMES_PER_CHAR, C.N_BINS)


def test_decodes_a_single_character():
    assert decode(encode("A").frames) == "A"


def test_decodes_a_sentence():
    assert decode(encode("HELLO WORLD").frames) == "HELLO WORLD"


def test_decodes_consecutive_spaces():
    """Each space is its own excited ring, so a run of them must not collapse."""
    assert decode(encode("X  Y").frames) == "X  Y"


def test_decodes_digits_and_punctuation():
    assert decode(encode("MEET ME AT 8PM!").frames) == "MEET ME AT 8PM!"


def test_sentinels_are_stripped_from_output():
    """The sentinel chord brackets every message but belongs to no character,
    so a two-character message must decode to exactly two characters."""
    assert len(decode(encode("AB").frames)) == 2
    assert C.SENTINEL_CHORD not in SYMBOL_BY_CHORD


def test_empty_message_decodes_to_empty_string():
    """Two adjacent sentinels with nothing between them."""
    assert decode(encode("").frames) == ""


def test_missing_sentinels_raise():
    with pytest.raises(ValueError):
        decode(chord_clip((3, 7)))


def test_degenerate_ring_is_not_decoded_as_a_letter():
    """A single-mode ring is the loudest shape the format can produce, yet
    detect_chord reports it as a valid-looking pair - mode 5 comes back as
    (4, 5), which means 'H'. Only the confidence gate catches it.
    """
    degenerate = np.stack([
        radius_profile((5, 5), a) for a in frame_amplitudes() * C.AMPLITUDE
    ])
    frames = np.concatenate([
        chord_clip(C.SENTINEL_CHORD), degenerate, chord_clip(C.SENTINEL_CHORD)
    ])
    assert decode(frames) == UNDECODABLE


def test_one_corrupt_frame_does_not_destroy_the_message():
    """A failed contour is what the image decoder will produce. One bad frame
    must cost at most one character, never the whole message."""
    frames = encode("HI").frames.copy()
    frames[5, :] = 0.0
    assert decode(frames) in {"HI", "H" + UNDECODABLE, UNDECODABLE + "I"}


def test_a_one_dimensional_input_is_rejected_clearly():
    """Without the guard this fails deep inside runs_of with a TypeError."""
    with pytest.raises(ValueError, match="n_frames, n_bins"):
        decode(radius_profile((3, 7), C.AMPLITUDE))


def _muddied_sentinel_clip(ratio: float = 0.9) -> np.ndarray:
    """A sentinel clip with a third mode just below the chord's own two.

    The interfering mode rides the same per-frame envelope as the chord, so the
    clip still starts and ends silent and segments exactly like a clean one.
    Modes 2 and 12 stay the two strongest, so `detect_chord` still reports
    (2, 12) -- but with mode 7 close behind, confidence collapses to 1/ratio.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, C.N_BINS, endpoint=False)
    low, high = C.SENTINEL_CHORD
    shape = (np.cos(low * theta) + np.cos(high * theta)) / 2.0
    shape = shape + ratio * 0.5 * np.cos(7 * theta)
    return np.stack([
        C.REST_RADIUS * (1.0 + amplitude * shape)
        for amplitude in frame_amplitudes() * C.AMPLITUDE
    ])


def test_a_muddied_sentinel_still_reports_the_sentinel_chord():
    """Guards the test below: it is only meaningful if the chord really is the
    sentinel pair, so that confidence is the sole thing rejecting it."""
    clip = _muddied_sentinel_clip()
    strongest = clip[int(np.argmax(frame_peaks(clip)))]
    chord, confidence = detect_chord(strongest)
    assert chord == C.SENTINEL_CHORD
    assert confidence < C.MIN_CONFIDENCE


def test_a_low_confidence_sentinel_does_not_anchor_the_message():
    """Sentinel matching is held to the same confidence bar as every other
    chord. A ring that reports (2, 12) but cannot be trusted must not bracket a
    message, or noise would anchor the decode to the wrong frames."""
    clip = _muddied_sentinel_clip()
    with pytest.raises(ValueError, match="sentinel"):
        decode(np.concatenate([clip, chord_clip((3, 7)), clip]))


def test_spare_chords_are_not_decoded_as_letters():
    """The twelve spare pairs belong to no character."""
    frames = np.concatenate([
        chord_clip(C.SENTINEL_CHORD),
        chord_clip(SPARE_CHORDS[0]),
        chord_clip(C.SENTINEL_CHORD),
    ])
    assert decode(frames) == UNDECODABLE
