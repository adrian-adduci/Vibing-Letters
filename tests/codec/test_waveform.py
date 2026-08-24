"""Tests for radius profile and envelope generation."""
import numpy as np

from src.codec import constants as C
from src.codec.waveform import radius_profile


def test_profile_has_one_value_per_bin():
    assert radius_profile((3, 7), 1.0).shape == (C.N_BINS,)


def test_zero_amplitude_gives_a_perfect_circle():
    profile = radius_profile((3, 7), 0.0)
    assert np.allclose(profile, C.REST_RADIUS)


def test_mean_radius_is_preserved():
    """Modes 2 and above integrate to zero, so they cannot change mean radius."""
    profile = radius_profile((3, 7), 1.0)
    assert np.isclose(profile.mean(), C.REST_RADIUS)


def test_lobe_count_matches_the_lower_mode():
    """A single mode n produces exactly n maxima around the ring."""
    profile = radius_profile((5, 5), 1.0)
    above = profile > profile.mean()
    transitions = np.sum(above != np.roll(above, 1))
    assert transitions == 2 * 5
