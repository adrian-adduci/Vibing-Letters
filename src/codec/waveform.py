"""Radius profiles for the vibrating string.

A profile is the radius of the ring sampled at N_BINS angles. All geometry is
expressed in normalized units where the rest radius is 1.0, so nothing here
depends on eventual render size.
"""

import numpy as np

from . import constants as C

# Angles are sampled without the endpoint: theta=0 and theta=2*pi are the same
# point on a closed loop, and including both would double-weight it in the FFT.
_THETA = np.linspace(0.0, 2.0 * np.pi, C.N_BINS, endpoint=False)


def radius_profile(
    chord: tuple[int, int],
    amplitude: float,
    n_bins: int = C.N_BINS,
) -> np.ndarray:
    """Build the radius profile for a chord at a given amplitude.

    Args:
        chord: Pair of mode numbers (n1, n2).
        amplitude: Signed radial modulation as a fraction of rest radius.
        n_bins: Number of angular samples.

    Returns:
        np.ndarray: Radii of shape (n_bins,).
    """
    low, high = chord
    theta = _THETA if n_bins == C.N_BINS else np.linspace(
        0.0, 2.0 * np.pi, n_bins, endpoint=False
    )
    shape = (np.cos(low * theta) + np.cos(high * theta)) / 2.0
    return C.REST_RADIUS * (1.0 + amplitude * shape)


def envelope(n_frames: int = C.ACTIVE_FRAMES) -> np.ndarray:
    """Pluck envelope: fast attack, slow decay, zero at both ends.

    Pinning both ends to zero does three jobs at once. Clips loop seamlessly,
    any clip can follow any other without a jump cut, and the silence between
    characters becomes the delimiter the decoder segments on.
    """
    t = np.linspace(0.0, 1.0, n_frames)
    rise = t / C.ATTACK
    fall = ((1.0 - t) / (1.0 - C.ATTACK)) ** C.DECAY_POWER
    return np.where(t < C.ATTACK, rise, fall)


def frame_amplitudes(
    n_active: int = C.ACTIVE_FRAMES,
    n_gap: int = C.TRAILING_SILENCE_FRAMES,
) -> np.ndarray:
    """Signed amplitude for each frame of one character clip.

    The envelope shapes the pluck; the cosine term is the standing wave
    oscillating. Amplitude is signed because the wave inverts each half cycle.
    Trailing silent frames give the segmenter an unambiguous character boundary.
    """
    t = np.linspace(0.0, 1.0, n_active)
    active = envelope(n_active) * np.cos(2.0 * np.pi * C.OSCILLATIONS * t)
    return np.concatenate([active, np.zeros(n_gap)])


def chord_clip(chord: tuple[int, int], n_bins: int = C.N_BINS) -> np.ndarray:
    """Build every frame of one excited character clip.

    Returns:
        np.ndarray: Radii of shape (FRAMES_PER_CHAR, n_bins).
    """
    amplitudes = frame_amplitudes() * C.AMPLITUDE
    return np.stack([radius_profile(chord, a, n_bins) for a in amplitudes])


def quiet_clip(n_bins: int = C.N_BINS) -> np.ndarray:
    """Build an unexcited clip: a still circle, which encodes a space."""
    return np.full((C.FRAMES_PER_CHAR, n_bins), C.REST_RADIUS)
