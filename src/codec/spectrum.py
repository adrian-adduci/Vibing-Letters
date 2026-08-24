"""Chord recovery from a radius profile.

Decoding is an angular FFT. Dividing by mean radius makes it scale invariant;
taking magnitude and discarding phase makes it rotation invariant. Both
properties fall out of the transform rather than being engineered.
"""

import numpy as np

from . import constants as C


def mode_band(profile: np.ndarray) -> np.ndarray:
    """Spectral magnitude for modes MIN_MODE..MAX_MODE, in modulation units.

    Values are the fractional radial modulation contributed by each mode, so a
    chord encoded at amplitude a produces peaks of roughly a/2. This makes the
    threshold interpretable and independent of ring size.

    Accepts either a single profile of shape (n_bins,) or a stack of them of
    shape (..., n_bins); the transform runs along the last axis, so a whole
    clip can be reduced in one call and the leading axes are preserved.

    Args:
        profile: Radii of shape (..., n_bins), sampled at equal angles.

    Returns:
        np.ndarray: Modal magnitudes of shape (..., MAX_MODE - MIN_MODE + 1).

    Raises:
        ValueError: If the profile has too few samples to resolve MAX_MODE, or
            if any profile has a non-positive mean radius.
    """
    values = np.asarray(profile, dtype=float)

    if values.shape[-1] < 2 * C.MAX_MODE + 1:
        raise ValueError(
            f"Profile needs at least {2 * C.MAX_MODE + 1} samples to resolve "
            f"mode {C.MAX_MODE}; got {values.shape[-1]}"
        )

    mean_radius = values.mean(axis=-1, keepdims=True)
    if np.any(mean_radius <= 0.0):
        raise ValueError("Profile must have a positive mean radius")

    normalized = values / mean_radius - 1.0
    magnitude = np.abs(np.fft.rfft(normalized, axis=-1)) / (values.shape[-1] / 2.0)
    return magnitude[..., C.MIN_MODE:C.MAX_MODE + 1]


def detect_chord(profile: np.ndarray) -> tuple[tuple[int, int] | None, float]:
    """Recover the chord from a single radius profile.

    Args:
        profile: Radii of shape (n_bins,), sampled at equal angles.

    Returns:
        tuple: (chord or None if no mode clears the quiet threshold, confidence)
        Confidence is the ratio of the weaker peak to the strongest non-peak
        mode. Values near 1.0 mean the answer is barely distinguishable from
        noise; large values mean the two peaks stand well clear. A ring that is
        loud but malformed -- a single-mode ring, say -- reports a chord with
        low confidence rather than None, so the caller can tell an unexcited
        ring apart from an unrecognizable one.
    """
    band = mode_band(profile)
    ranked = np.argsort(band, kind='stable')[::-1]
    strongest, second = int(ranked[0]), int(ranked[1])

    if band[strongest] < C.QUIET_THRESHOLD:
        return None, 0.0

    remainder = np.delete(band, [strongest, second])
    noise_floor = float(remainder.max()) if remainder.size else 0.0
    confidence = float(band[second] / noise_floor) if noise_floor > 0 else float('inf')

    chord = (min(strongest, second) + C.MIN_MODE, max(strongest, second) + C.MIN_MODE)
    return chord, confidence
