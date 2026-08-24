"""The exact peak contour, and the line art that conditions the image model.

This is the first stage of the asset pipeline and the only one that is pure
mathematics. It answers one question -- what shape does chord {n1, n2} make at
its loudest moment -- and draws that answer precisely enough that an
edge-conditioned image model has no room to invent a different one.

Two properties are load-bearing:

* **One scale for every chord.** Pixel radii are derived from a single constant
  bound, not from each chord's own maximum, so all 44 characters share a rest
  radius. The warp reads r_peak in pixels; a per-chord scale would quietly
  change what those numbers mean.
* **A closed curve.** The extractor walks rays outward from a centre and expects
  to cross ink exactly once. A seam at the theta wrap would break that walk, and
  a canny-conditioned model would reproduce the notch faithfully.
"""

import cv2
import numpy as np

from ..codec import constants as C
from ..codec.waveform import frame_amplitudes, radius_profile

# ---------------------------------------------------------------------------
# Render parameters. Not wire format: decoding normalizes by mean radius and by
# bin count, so these choose a convenient picture rather than a format.
# ---------------------------------------------------------------------------

# Blank border, as a fraction of the canvas edge. Keeping it proportional is
# what makes the pixel mapping scale exactly with `size`.
MARGIN_FRACTION = 0.06

# Stroke width in pixels at the 1024 px reference size, scaled with the canvas.
LINE_WIDTH_AT_1024 = 3

# Never thinner than this. A one-pixel anti-aliased curve never saturates -- at
# 256 px the darkest pixel measured 22/255 and barely 0.05% of the canvas fell
# below a quarter intensity -- which is faint line art to condition an edge
# model on. Two pixels always reach full black somewhere along the stroke.
MIN_LINE_WIDTH = 2

# cv2 fixed-point fractional bits for sub-pixel polyline endpoints. Rounding the
# contour to whole pixels would inject broadband error into exactly the spectrum
# the decoder reads.
_SHIFT = 4

# The loudest frame of every clip. The pluck peaks a quarter of the way in,
# where the standing wave has swung to its trough, so this frame carries a
# negative amplitude -- the largest excursion, not the largest signed value.
PEAK_FRAME = int(np.argmax(np.abs(frame_amplitudes())))


def peak_amplitude() -> float:
    """Signed radial modulation at the loudest frame, in rest-radius units."""
    return float(frame_amplitudes()[PEAK_FRAME] * C.AMPLITUDE)


# Every chord's radius is bounded by this, because the shape term
# (cos(n1*theta) + cos(n2*theta)) / 2 never leaves [-1, 1]. Most chords never
# reach the bound, which is the point: the slack is what keeps one scale valid
# for all of them.
_MAX_RADIUS = C.REST_RADIUS * (1.0 + abs(peak_amplitude()))


def peak_profile(chord: tuple[int, int], n_bins: int = C.N_BINS) -> np.ndarray:
    """Radius profile of `chord` at its loudest frame, in rest-radius units."""
    return radius_profile(chord, peak_amplitude(), n_bins)


def pixel_scale(size: int) -> float:
    """Pixels per unit radius on a `size` x `size` canvas."""
    return (size / 2.0 - MARGIN_FRACTION * size) / _MAX_RADIUS


def to_pixels(profile: np.ndarray, size: int) -> np.ndarray:
    """Map a radius profile onto canvas coordinates.

    Args:
        profile: Radii of shape (n_bins,) in rest-radius units.
        size: Canvas edge in pixels.

    Returns:
        np.ndarray: Points of shape (n_bins, 2) as (x, y), origin top-left.
    """
    radii = np.asarray(profile, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, len(radii), endpoint=False)
    centre = size / 2.0
    scaled = radii * pixel_scale(size)
    # y is negated because canvas rows grow downward while theta grows
    # counter-clockwise. The choice is cosmetic -- reflecting a profile leaves
    # its magnitude spectrum untouched -- but it keeps the picture and the maths
    # oriented the same way.
    return np.stack([centre + scaled * np.cos(theta),
                     centre - scaled * np.sin(theta)], axis=1)


def render_control_image(
    chord: tuple[int, int],
    size: int = 1024,
    n_bins: int = C.N_BINS,
    line_width: int | None = None,
) -> np.ndarray:
    """Draw the peak contour as black line art on white.

    The output is ready to use directly as an edge control image: it is already
    the edge map, so a model that preprocesses to canny internally will recover
    it unchanged rather than re-detecting something approximate.

    Args:
        chord: Pair of mode numbers (n1, n2).
        size: Canvas edge in pixels.
        n_bins: Number of contour vertices.
        line_width: Stroke width in pixels; scaled from the reference size when
            omitted.

    Returns:
        np.ndarray: Grayscale uint8 image of shape (size, size).
    """
    if line_width is None:
        line_width = max(MIN_LINE_WIDTH, round(LINE_WIDTH_AT_1024 * size / 1024))

    points = to_pixels(peak_profile(chord, n_bins), size)
    fixed = np.round(points * (1 << _SHIFT)).astype(np.int32)

    canvas = np.full((size, size), 255, dtype=np.uint8)
    cv2.polylines(
        canvas,
        [fixed.reshape(-1, 1, 2)],
        isClosed=True,
        color=0,
        thickness=line_width,
        lineType=cv2.LINE_AA,
        shift=_SHIFT,
    )
    return canvas
