"""Tests for radius profile and envelope generation."""
import numpy as np

from src.codec import constants as C
from src.codec.waveform import (
    chord_clip,
    envelope,
    frame_amplitudes,
    radius_profile,
)


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


def test_envelope_starts_and_ends_at_zero():
    """This is what makes clips loop and splice without a visible seam."""
    env = envelope()
    assert np.isclose(env[0], 0.0)
    assert np.isclose(env[-1], 0.0)


def test_envelope_never_exceeds_unity():
    """AMPLITUDE scales this envelope, so a peak above 1.0 would silently
    exceed the configured radial modulation."""
    assert envelope().max() <= 1.0


def test_envelope_reaches_unity_where_the_attack_point_is_sampled():
    """The continuous envelope peaks at exactly 1.0 at t == ATTACK, where the
    rise and decay branches meet. The default 12-frame grid steps by 1/11 and
    never lands on 0.25, so it peaks near 0.94 instead. A 5-frame grid samples
    t = 0, 0.25, 0.5, 0.75, 1.0 and does hit it exactly.
    """
    assert np.isclose(envelope(5).max(), 1.0)


def test_envelope_attacks_faster_than_it_decays():
    """A pluck rises sharply and relaxes slowly."""
    env = envelope()
    peak = int(np.argmax(env))
    assert peak < len(env) - peak


def test_frame_amplitudes_include_trailing_silence():
    amps = frame_amplitudes()
    assert len(amps) == C.FRAMES_PER_CHAR
    assert np.allclose(amps[-C.TRAILING_SILENCE_FRAMES:], 0.0)


def test_frame_amplitudes_change_sign():
    """The standing wave oscillates; lobes swap in and out."""
    amps = frame_amplitudes()
    assert amps.min() < 0 < amps.max()


def test_chord_clip_shape():
    assert chord_clip((3, 7)).shape == (C.FRAMES_PER_CHAR, C.N_BINS)


def test_chord_clip_opens_and_closes_on_a_circle():
    """Required for seamless looping and splicing."""
    clip = chord_clip((3, 7))
    assert np.allclose(clip[0], C.REST_RADIUS)
    assert np.allclose(clip[-1], C.REST_RADIUS)


def test_chord_clip_actually_deforms_in_between():
    clip = chord_clip((3, 7))
    assert clip[1:-1].std() > 0.0
