"""Tests for recovering a chord from a radius profile."""

import numpy as np
import pytest

from src.codec import constants as C
from src.codec.spectrum import (
    active_mask,
    close_short_gaps,
    detect_chord,
    mode_band,
    runs_of,
)
from src.codec.waveform import chord_clip, radius_profile


def test_recovers_the_encoded_chord():
    chord, _ = detect_chord(radius_profile((3, 7), C.AMPLITUDE))
    assert chord == (3, 7)


@pytest.mark.parametrize("chord", [(2, 3), (4, 9), (5, 12), (2, 12), (11, 12)])
def test_recovers_every_shape_of_chord(chord):
    detected, _ = detect_chord(radius_profile(chord, C.AMPLITUDE))
    assert detected == chord


def test_a_still_circle_decodes_to_nothing():
    """A space carries no signal and must not be reported as a character."""
    chord, _ = detect_chord(np.full(C.N_BINS, C.REST_RADIUS))
    assert chord is None


def test_detection_is_rotation_invariant():
    """Magnitude spectrum discards phase, so rotating the ring changes nothing."""
    profile = radius_profile((4, 9), C.AMPLITUDE)
    rotated = np.roll(profile, C.N_BINS // 7)
    assert detect_chord(rotated)[0] == detect_chord(profile)[0]


def test_detection_is_scale_invariant():
    """Normalizing by mean radius makes resizing irrelevant."""
    profile = radius_profile((4, 9), C.AMPLITUDE)
    assert detect_chord(profile * 37.5)[0] == (4, 9)


def test_sign_flip_does_not_change_the_chord():
    """The wave inverts each half cycle; both halves must decode the same."""
    positive = radius_profile((5, 8), C.AMPLITUDE)
    negative = radius_profile((5, 8), -C.AMPLITUDE)
    assert detect_chord(positive)[0] == detect_chord(negative)[0]


def test_mode_band_peaks_at_the_encoded_modes():
    band = mode_band(radius_profile((3, 7), C.AMPLITUDE))
    assert int(np.argmax(band)) + C.MIN_MODE in (3, 7)


def test_mode_band_reports_modulation_in_amplitude_units():
    """QUIET_THRESHOLD is only interpretable if peaks equal AMPLITUDE / 2."""
    band = mode_band(radius_profile((3, 7), C.AMPLITUDE))
    assert len(band) == C.MAX_MODE - C.MIN_MODE + 1
    assert band[3 - C.MIN_MODE] == pytest.approx(C.AMPLITUDE / 2)
    assert band[7 - C.MIN_MODE] == pytest.approx(C.AMPLITUDE / 2)
    others = np.delete(band, [3 - C.MIN_MODE, 7 - C.MIN_MODE])
    assert others.max() < 1e-12


def test_mode_band_accepts_a_stack_of_profiles():
    """Tasks 9 and 10 pass whole clips; slicing frames instead of modes would
    be silent, so pin the shape contract."""
    band = mode_band(chord_clip((3, 7)))
    assert band.shape == (C.FRAMES_PER_CHAR, C.MAX_MODE - C.MIN_MODE + 1)


def test_mode_band_rejects_an_undersampled_profile():
    """Below 2*MAX_MODE+1 samples, aliasing returns confidently wrong chords."""
    with pytest.raises(ValueError, match="at least"):
        mode_band(radius_profile((9, 12), C.AMPLITUDE, n_bins=16))


def test_a_character_splits_without_gap_closing():
    """The standing wave crosses zero mid-clip, so raw runs over-segment."""
    mask = active_mask(chord_clip((3, 7)))
    assert sum(1 for is_active, _, _ in runs_of(mask) if is_active) > 1


def test_gap_closing_reunites_one_character():
    """Closing short quiet runs is what makes a character a single segment."""
    mask = close_short_gaps(active_mask(chord_clip((3, 7))))
    assert sum(1 for is_active, _, _ in runs_of(mask) if is_active) == 1


def test_gap_closing_does_not_merge_across_characters():
    clip = np.concatenate([chord_clip((3, 7)), chord_clip((4, 9))])
    mask = close_short_gaps(active_mask(clip))
    assert sum(1 for is_active, _, _ in runs_of(mask) if is_active) == 2


def test_runs_cover_the_whole_sequence():
    mask = active_mask(chord_clip((3, 7)))
    assert sum(stop - start for _, start, stop in runs_of(mask)) == len(mask)


def test_boundary_gap_matches_the_constant():
    """BOUNDARY_GAP_FRAMES is a measured consequence of seven other constants,
    not an independent value. The round-trip test cannot catch it drifting:
    encoder and decoder read the same constants and desynchronize in lockstep,
    staying green while real messages decode wrong. So measure it directly.
    """
    clip = np.concatenate([chord_clip((3, 7)), chord_clip((4, 9))])
    mask = close_short_gaps(active_mask(clip))
    interior_gaps = [
        stop - start
        for is_active, start, stop in runs_of(mask)
        if not is_active and start > 0 and stop < len(mask)
    ]
    assert interior_gaps == [C.BOUNDARY_GAP_FRAMES]
