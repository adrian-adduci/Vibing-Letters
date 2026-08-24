"""Tests for whole-sentence encoding and decoding."""
import numpy as np

from src.codec import constants as C
from src.codec.message import encode
from src.codec.spectrum import detect_chord


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
