"""Tests for codec constants."""
from src.codec import constants as C


def test_mode_range_excludes_translation_mode():
    """Mode 1 is a rigid translation, not a deformation, so it must be excluded."""
    assert C.MIN_MODE == 2
    assert C.MAX_MODE == 12


def test_frames_per_char_is_active_plus_gap():
    assert C.FRAMES_PER_CHAR == C.ACTIVE_FRAMES + C.GAP_FRAMES


def test_sentinel_uses_extreme_modes():
    """The sentinel must be maximally separated so it is never confused."""
    assert C.SENTINEL_CHORD == (C.MIN_MODE, C.MAX_MODE)
