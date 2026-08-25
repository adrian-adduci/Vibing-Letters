"""Turn one styled still into every frame of its clip.

A curated still shows the chord at its loudest. The other fourteen frames are
the same picture breathing: the pluck envelope carries the modulation from zero
up to that peak and back down. Every shape here is star-shaped about a known
centre, so the deformation is a per-angle radial scale and one `cv2.remap` does
it, with no mesh and no thin-plate spline. The glow scales along with the ring
because it is the same map.

The one decision worth stating is where the shape comes from. The obvious source
is the analytic profile for the chord, but that assumes the still sits at the
same angular phase as the control image -- and if the model rotated it even
slightly, the warp would fight the picture instead of driving it. So the shape
is measured *from the still*:

    r_still(theta) = mean * (1 + a_peak * s(theta))

is solved for `s`, and every frame is rebuilt as `mean * (1 + a_t * s(theta))`.
Phase never enters the arithmetic. Whatever the model actually drew -- rotated,
embellished, slightly off -- is what gets animated, and the acceptance gate is
left to judge whether that thing was the right chord in the first place.
"""

from typing import NamedTuple

import cv2
import numpy as np

from ..codec import constants as C
from ..codec.waveform import frame_amplitudes
from ..vision import ring
from .contour import peak_amplitude

# Amplitude of every frame in one clip, in rest-radius units.
FRAME_AMPLITUDES = frame_amplitudes() * C.AMPLITUDE


class WarpField(NamedTuple):
    """Everything about a still that does not change between its frames."""

    shape: np.ndarray      # s(theta) per pixel, dimensionless, roughly [-1, 1]
    centre: tuple[float, float]
    xs: np.ndarray         # column index per pixel
    ys: np.ndarray         # row index per pixel


def measure(still: np.ndarray, n_bins: int = C.N_BINS) -> WarpField:
    """Recover the still's own shape function, sampled at every pixel.

    Args:
        still: The curated peak-excitation image, grayscale or colour.
        n_bins: Angular resolution of the profile measurement.

    Returns:
        WarpField: Precomputed fields shared by all frames of this still.

    Raises:
        ValueError: If the still carries no measurable modulation, which means
            it is a plain circle and there is no chord in it to animate.
    """
    ink = ring.ink_map(still)
    centre = ring.locate_ring(ink, n_bins)
    profile = ring.profile_about(ink, centre, n_bins)

    mean_radius = float(profile.mean())
    shape_samples = (profile / mean_radius - 1.0) / peak_amplitude()
    if np.abs(shape_samples).max() < 0.1:
        raise ValueError(
            "Still is effectively a circle; there is no excitation to animate"
        )

    height, width = still.shape[:2]
    ys, xs = np.indices((height, width), dtype=np.float32)
    # Rows grow downward, so y is negated to match the orientation the contour
    # is drawn in and the extractor reads in.
    theta = np.arctan2(-(ys - centre[1]), xs - centre[0]) % (2.0 * np.pi)

    bins = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=False)
    shape = np.interp(
        theta.ravel(), bins, shape_samples, period=2.0 * np.pi
    ).reshape(height, width).astype(np.float32)

    return WarpField(shape=shape, centre=centre, xs=xs, ys=ys)


def at_amplitude(still: np.ndarray, field: WarpField, amplitude: float) -> np.ndarray:
    """Rebuild the still with its modulation rescaled to `amplitude`.

    A destination pixel at (theta, rho) is filled from the source at
    (theta, rho * r_peak(theta) / r_t(theta)), and because both radii share the
    same mean the ratio collapses to a function of the shape alone. That leaves
    one multiply per pixel: the offset from the centre, scaled.

    Args:
        still: The image `field` was measured from.
        field: Output of `measure`.
        amplitude: Target modulation in rest-radius units. Zero yields a circle.

    Returns:
        np.ndarray: A warped copy of `still`, same shape and dtype.

    Raises:
        ValueError: If the target profile would pass through zero radius, which
            would make the map singular. Reachable only with an amplitude far
            outside anything the envelope produces.
    """
    denominator = 1.0 + amplitude * field.shape
    if denominator.min() <= 0.0:
        raise ValueError(
            f"Amplitude {amplitude} collapses the ring to zero radius somewhere"
        )

    scale = (1.0 + peak_amplitude() * field.shape) / denominator
    cx, cy = field.centre
    map_x = (cx + (field.xs - cx) * scale).astype(np.float32)
    map_y = (cy + (field.ys - cy) * scale).astype(np.float32)

    # Replicating the border keeps the surrounding background intact where the
    # map reaches outside the canvas; the ring's margin means that only ever
    # happens in background.
    return cv2.remap(still, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def envelope_clip(still: np.ndarray, n_bins: int = C.N_BINS) -> np.ndarray:
    """Every frame of one character clip, warped from a single still.

    Args:
        still: The curated peak-excitation image.
        n_bins: Angular resolution of the shape measurement.

    Returns:
        np.ndarray: Frames stacked on a leading axis, length FRAMES_PER_CHAR.
        The trailing silent frames are true circles, which is what lets any
        clip follow any other without a jump cut.
    """
    field = measure(still, n_bins)
    return np.stack([at_amplitude(still, field, a) for a in FRAME_AMPLITUDES])
