"""Tests for whole-sentence encoding and decoding."""
import numpy as np
import pytest

from src.codec import constants as C
from src.codec.chord_table import SPARE_CHORDS, SYMBOL_BY_CHORD
from src.codec.message import UNDECODABLE, decode, encode
from src.codec.spectrum import detect_chord
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


def test_interior_space_produces_an_unexcited_clip():
    """Note this uses an interior space: normalize strips leading and trailing
    whitespace, so encode(" ") has no space clip at all."""
    frames = encode("A B").frames
    space_clip = frames[2 * C.FRAMES_PER_CHAR:3 * C.FRAMES_PER_CHAR]
    assert np.allclose(space_clip, C.REST_RADIUS)


def test_empty_input_encodes_to_bare_sentinels():
    result = encode("   ")
    assert result.text == ""
    assert result.frames.shape == (2 * C.FRAMES_PER_CHAR, C.N_BINS)


def test_decodes_a_single_character():
    assert decode(encode("A").frames) == "A"


def test_decodes_a_sentence():
    assert decode(encode("HELLO WORLD").frames) == "HELLO WORLD"


def test_decodes_consecutive_spaces():
    """Spaces come from quiet-run length, so runs of them must not collapse."""
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


def test_spare_chords_are_not_decoded_as_letters():
    """The twelve spare pairs belong to no character."""
    frames = np.concatenate([
        chord_clip(C.SENTINEL_CHORD),
        chord_clip(SPARE_CHORDS[0]),
        chord_clip(C.SENTINEL_CHORD),
    ])
    assert decode(frames) == UNDECODABLE
