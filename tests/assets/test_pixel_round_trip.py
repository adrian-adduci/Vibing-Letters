"""The claim that makes the whole pipeline possible: chords survive pixels.

`tests/codec/test_round_trip.py` proves decode(encode(s)) == s over arrays of
floats. That says nothing about images. These tests close the gap -- render a
chord to actual pixels, read it back with nothing but the extractor, and require
the original chord out. Everything downstream (styling, warping, emitting) is a
transformation of pixels that this contract already covers.
"""
import cv2
import numpy as np
import pytest

from src.assets import contour
from src.codec import constants as C
from src.codec.chord_table import CHORD_BY_SYMBOL, SYMBOL_BY_CHORD
from src.codec.spectrum import detect_chord
from src.vision import ring

ALL_CHORDS = sorted(set(CHORD_BY_SYMBOL.values()) | {C.SENTINEL_CHORD})


def read_back(image: np.ndarray) -> tuple[tuple[int, int] | None, float]:
    return detect_chord(ring.radius_profile(image))


class TestEveryChordSurvivesPixels:
    @pytest.mark.parametrize("chord", ALL_CHORDS)
    def test_chord_round_trips_through_a_control_image(self, chord):
        found, confidence = read_back(contour.render_control_image(chord, size=1024))
        assert found == chord
        assert confidence > C.MIN_CONFIDENCE

    @pytest.mark.parametrize("chord", ALL_CHORDS)
    def test_chord_round_trips_at_the_gif_fallback_size(self, chord):
        """256 px is the smallest size the design commits to shipping.

        Mode 12 has 24 lobes; at 256 px that is roughly 47 px of circumference
        each, so the sampling is not close to marginal. Asserting it here means
        a future size change cannot quietly cross the line.
        """
        found, _ = read_back(contour.render_control_image(chord, size=256))
        assert found == chord


class TestInvariances:
    """Scale and rotation invariance are claimed by the design; through pixels
    they are no longer free, because rasterizing and resampling are lossy."""

    @pytest.mark.parametrize("size", [256, 384, 512, 768, 1024])
    def test_scale_invariant(self, size):
        assert read_back(contour.render_control_image((5, 11), size=size))[0] == (5, 11)

    @pytest.mark.parametrize("degrees", [7, 45, 90, 137, 180, 271])
    def test_rotation_invariant(self, degrees):
        """Runtime rotates every ring instance by a hash-seeded offset.

        The FFT magnitude spectrum does not see rotation, but the pixel grid
        does: rotating resamples the whole image. If that resampling injected
        enough broadband error to matter, it would show up here.
        """
        image = contour.render_control_image((3, 9), size=512)
        matrix = cv2.getRotationMatrix2D((256.0, 256.0), degrees, 1.0)
        rotated = cv2.warpAffine(image, matrix, (512, 512),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        assert read_back(rotated)[0] == (3, 9)

    @pytest.mark.parametrize("shift", [(30, 0), (0, -25), (-40, 35)])
    def test_translation_invariant(self, shift):
        """A styled ring will not be perfectly centred in its frame."""
        image = contour.render_control_image((4, 7), size=512)
        matrix = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        moved = cv2.warpAffine(image, matrix, (512, 512),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        assert read_back(moved)[0] == (4, 7)

    @pytest.mark.parametrize("width", [2, 5, 9, 17])
    def test_stroke_width_invariant(self, width):
        """Whatever the model draws with, the curve underneath is the same."""
        image = contour.render_control_image((6, 10), size=512, line_width=width)
        assert read_back(image)[0] == (6, 10)


class TestDegradation:
    def test_jpeg_compression_survives(self):
        image = contour.render_control_image((2, 9), size=512)
        ok, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])
        assert ok
        assert read_back(cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE))[0] == (2, 9)

    def test_blur_survives(self):
        """Glow is blur. The styled assets will have plenty of it."""
        image = contour.render_control_image((7, 9), size=512)
        assert read_back(cv2.GaussianBlur(image, (15, 15), 0))[0] == (7, 9)

    def test_a_broken_arc_still_decodes(self):
        """One damaged sector costs accuracy at those angles, not the chord.

        This is also what makes the ray-presence floor load-bearing: raise it
        and most rays are discarded as misses, the interpolation fills in from
        the survivors, and the waviness the chord lives in is flattened away.
        """
        image = contour.render_control_image((3, 8), size=512)
        image[180:230, 60:150] = 255
        assert read_back(image)[0] == (3, 8)


    def test_a_sector_of_pure_noise_does_not_invent_a_chord(self):
        """The dangerous damage is faint, not absent.

        Where a sector is erased to clean background, no ray finds ink and the
        gap is bridged. Where the sector is instead filled with faint speckle,
        every ray finds *something*, and because the ink floor is relative to
        each ray's own peak, the speckle passes it and reports a radius drawn
        from noise. Measured: this reads chord (2, 4) -- the letter E -- at
        confidence 1.13, which is a wrong answer stated calmly. The absolute
        presence floor is what rejects those rays so the gap is bridged instead.
        """
        rng = np.random.default_rng(0)
        image = contour.render_control_image((3, 8), size=512)
        wedge = np.zeros((512, 512), np.uint8)
        cv2.ellipse(wedge, (256, 256), (260, 260), 0, 20, 70, 255, -1)
        speckle = rng.integers(240, 252, size=(512, 512)).astype(np.uint8)
        image[wedge > 0] = speckle[wedge > 0]

        found, confidence = read_back(image)
        assert found == (3, 8)
        assert confidence > C.MIN_CONFIDENCE


class TestSpelling:
    def test_a_word_of_control_images_reads_back(self):
        """End to end at the character level, purely through pixels."""
        word = "HELLO WORLD"
        rendered = [contour.render_control_image(CHORD_BY_SYMBOL[c], size=512)
                    for c in word]
        recovered = ''.join(SYMBOL_BY_CHORD[read_back(i)[0]] for i in rendered)
        assert recovered == word
