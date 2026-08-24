"""Tests for codec constants."""
from src.codec import constants as C


def test_mode_range_excludes_translation_mode():
    """Mode 1 is a rigid translation, not a deformation, so it must be excluded."""
    assert C.MIN_MODE == 2
    assert C.MAX_MODE == 12


def test_frames_per_char_is_active_plus_gap():
    assert C.FRAMES_PER_CHAR == C.ACTIVE_FRAMES + C.TRAILING_SILENCE_FRAMES


def test_sentinel_uses_extreme_modes():
    """The sentinel must be maximally separated so it is never confused."""
    assert C.SENTINEL_CHORD == (C.MIN_MODE, C.MAX_MODE)


def test_modes_fit_within_the_transform():
    """mode_band slices magnitude[MIN_MODE:MAX_MODE+1]; MAX_MODE must stay in range."""
    assert C.MAX_MODE < C.N_BINS // 2


def test_quiet_threshold_is_below_the_expected_peak():
    """A chord at AMPLITUDE produces peaks near AMPLITUDE/2. If the threshold
    reaches that, excited frames read as silent and characters vanish."""
    assert C.QUIET_THRESHOLD < C.AMPLITUDE / 2


def test_wire_format_values_are_pinned():
    """Explicit lock on values that cannot change without breaking old messages."""
    assert (C.MIN_MODE, C.MAX_MODE) == (2, 12)
    assert C.SENTINEL_CHORD == (2, 12)
    assert C.FRAMES_PER_CHAR == 15
    assert C.ACTIVE_FRAMES == 12
    assert C.TRAILING_SILENCE_FRAMES == 3
    assert C.OSCILLATIONS == 2
