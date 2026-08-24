"""Recover a radius profile from a rendered ring.

The codec reads `r(theta)` sampled at equal angles. Images are what ship. This
module converts one into the other, and it is the only place in the project that
knows about both.

Three decisions carry the accuracy:

* **Ink is distance from the background, not darkness.** The background is the
  median intensity -- a ring is a thin stroke, so most of the canvas is
  background by construction. Measuring |value - median| makes dark-on-light
  line art and a glowing ring on black behave identically, with no polarity flag
  to get wrong.
* **Radius is the ink-weighted centroid along each ray, not the first crossing.**
  A crossing test reports the near edge of the stroke and therefore drifts with
  stroke width and with glow. A weighted centroid reports the middle of the
  stroke, which is the curve itself, to sub-pixel precision.
* **The centre is solved for, not assumed.** Sampling about the wrong centre
  injects a mode-1 term into `r(theta)`, and that term states the error exactly:
  its size is the offset and its phase is the direction. Two or three rounds of
  reading it and stepping there converge. This is the same fact that bars mode 1
  from the alphabet, used constructively.
"""

import cv2
import numpy as np

from ..codec import constants as C

# Radial samples per pixel of ring radius. Two is enough to place a two-pixel
# stroke to a fraction of a pixel once the weighting is applied.
SAMPLES_PER_PIXEL = 2.0

# Per-ray relative cutoff. Everything below this fraction of the ray's own peak
# is discarded before the weighted centroid. Without it, a wide soft glow drags
# the centroid toward the middle of the sampled range; with it, only the
# stroke's core votes. Relative rather than absolute so it survives any exposure.
INK_FLOOR = 0.25

# Rays whose strongest ink falls below this fraction of the image-wide peak are
# treated as having missed the ring entirely and are filled from their
# neighbours rather than reporting a meaningless radius.
RAY_PRESENCE = 0.1

# Rounds of centre refinement. The mode-1 correction is a first-order estimate,
# so it is iterated; in practice the second round moves the centre by well under
# a pixel.
REFINEMENTS = 3


def ink_map(image: np.ndarray) -> np.ndarray:
    """Convert an image to ink weight in [0, 1], 1 where the ring is.

    Args:
        image: Grayscale or colour image of any dtype.

    Returns:
        np.ndarray: Float32 ink weights of the same height and width.

    Raises:
        ValueError: If the image is empty or carries no contrast at all.
    """
    values = np.asarray(image)
    if values.size == 0:
        raise ValueError("Image is empty")

    if values.ndim == 3:
        values = cv2.cvtColor(values.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif values.ndim != 2:
        raise ValueError(f"Expected a 2D or 3D image, got shape {values.shape}")

    gray = values.astype(np.float32)
    departure = np.abs(gray - float(np.median(gray)))
    peak = float(departure.max())
    if peak == 0.0:
        raise ValueError("Image is uniform; there is no ring to find")
    return departure / peak


def _polar(ink: np.ndarray, centre: tuple[float, float], n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample ink into (angle, radius) about `centre`.

    Returns:
        tuple: (samples of shape (n_bins, n_radii), the radii they sit at)
    """
    cx, cy = centre
    height, width = ink.shape
    # Stay inside the canvas: sampling past the edge would read the border
    # value and invent ink that is not there.
    reach = min(cx, cy, width - 1.0 - cx, height - 1.0 - cy)
    if reach <= 1.0:
        raise ValueError(f"Centre {centre} leaves no room to sample")

    radii = np.linspace(0.0, reach, max(8, int(reach * SAMPLES_PER_PIXEL)))
    theta = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=False)

    # y is negated to match the orientation `assets.contour.to_pixels` draws in.
    map_x = (cx + np.outer(np.cos(theta), radii)).astype(np.float32)
    map_y = (cy - np.outer(np.sin(theta), radii)).astype(np.float32)
    samples = cv2.remap(
        ink, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    return samples, radii


def profile_about(
    ink: np.ndarray,
    centre: tuple[float, float],
    n_bins: int = C.N_BINS,
) -> np.ndarray:
    """Radius of the ring at each angle, measured from a given centre.

    Args:
        ink: Ink weights from `ink_map`.
        centre: (x, y) in pixels to measure from.
        n_bins: Number of angular samples.

    Returns:
        np.ndarray: Radii in pixels, shape (n_bins,).

    Raises:
        ValueError: If no ray finds the ring.
    """
    samples, radii = _polar(ink, centre, n_bins)

    ray_peaks = samples.max(axis=1)
    found = ray_peaks >= RAY_PRESENCE * float(ray_peaks.max())
    if not found.any():
        raise ValueError("No ray crossed the ring")

    weights = np.where(samples >= INK_FLOOR * ray_peaks[:, None], samples, 0.0)
    totals = weights.sum(axis=1)
    found &= totals > 0.0

    profile = np.full(n_bins, np.nan)
    profile[found] = (weights[found] @ radii) / totals[found]

    if not found.all():
        # A closed curve should never leave a gap. When one appears -- a stroke
        # broken by compression, say -- interpolating across it costs a little
        # accuracy at those angles, where refusing would cost the whole ring.
        angles = np.arange(n_bins)
        profile = np.interp(
            angles, angles[found], profile[found], period=n_bins,
        )
    return profile


def locate_ring(
    ink: np.ndarray,
    n_bins: int = C.N_BINS,
    refinements: int = REFINEMENTS,
) -> tuple[float, float]:
    """Solve for the centre the ring is star-shaped about.

    Starts from the ink centroid, then repeatedly reads the mode-1 component of
    the measured profile -- which is exactly the sampling offset -- and steps
    there.

    Args:
        ink: Ink weights from `ink_map`.
        n_bins: Number of angular samples used while refining.
        refinements: Correction rounds after the initial centroid.

    Returns:
        tuple: (x, y) of the centre in pixels.
    """
    total = float(ink.sum())
    if total <= 0.0:
        raise ValueError("Image contains no ink")

    rows, cols = np.indices(ink.shape, dtype=np.float32)
    centre = (float((cols * ink).sum() / total), float((rows * ink).sum() / total))

    for _ in range(refinements):
        profile = profile_about(ink, centre, n_bins)
        first = np.fft.rfft(profile - profile.mean())[1]
        # r(theta) measured about centre + delta gains a term -delta . u(theta),
        # so bin 1 carries the negated offset. See the derivation in the module
        # docstring: F[1] = (N/2)(dx - i*dy) for a dx*cos + dy*sin component.
        dx = 2.0 * float(first.real) / n_bins
        dy = -2.0 * float(first.imag) / n_bins
        # dy is subtracted because canvas rows grow downward.
        centre = (centre[0] + dx, centre[1] - dy)

    return centre


def radius_profile(image: np.ndarray, n_bins: int = C.N_BINS) -> np.ndarray:
    """Radius profile of the ring in an image, ready for the codec.

    Args:
        image: A rendered frame, grayscale or colour.
        n_bins: Number of angular samples.

    Returns:
        np.ndarray: Radii in pixels, shape (n_bins,). The codec normalizes by
        mean radius, so the pixel scale carries no meaning downstream.
    """
    ink = ink_map(image)
    return profile_about(ink, locate_ring(ink, n_bins), n_bins)
