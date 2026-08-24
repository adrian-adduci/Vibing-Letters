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
