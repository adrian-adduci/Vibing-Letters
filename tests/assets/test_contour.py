"""The peak contour, and the line art that conditions the image model.

Everything here is about one claim: the pixels we hand the image model carry the
chord exactly. If the control image is wrong, every downstream stage inherits the
error and the acceptance gate can only report it, never repair it.
"""
import numpy as np
import pytest

from src.assets import contour
from src.codec import constants as C
from src.codec.chord_table import CHORD_BY_SYMBOL
from src.codec.spectrum import detect_chord
from src.codec.waveform import frame_amplitudes, radius_profile

ALL_CHORDS = sorted(set(CHORD_BY_SYMBOL.values()) | {C.SENTINEL_CHORD})


class TestPeakFrame:
    def test_peak_frame_is_the_largest_excursion_of_the_clip(self):
        """Largest *magnitude*, not largest signed value.

        The pluck peaks a quarter of the way in, where cos(2*pi*OSCILLATIONS*t)
        has swung to -1, so the loudest frame of every clip is a trough rather
        than a crest. Picking the signed maximum would select a visibly weaker
        shape.
        """
        amplitudes = frame_amplitudes()
        assert contour.PEAK_FRAME == int(np.argmax(np.abs(amplitudes)))
        assert amplitudes[contour.PEAK_FRAME] < 0.0

    def test_peak_frame_lands_inside_the_active_run(self):
        assert 0 <= contour.PEAK_FRAME < C.ACTIVE_FRAMES

    def test_peak_amplitude_matches_that_frame(self):
        expected = frame_amplitudes()[contour.PEAK_FRAME] * C.AMPLITUDE
        assert contour.peak_amplitude() == pytest.approx(expected)


class TestPeakProfile:
    @pytest.mark.parametrize("chord", ALL_CHORDS)
    def test_peak_profile_decodes_to_its_own_chord(self, chord):
        assert detect_chord(contour.peak_profile(chord))[0] == chord

    @pytest.mark.parametrize("chord", ALL_CHORDS)
    def test_peak_profile_is_the_peak_frame_of_the_clip(self, chord):
        expected = radius_profile(chord, contour.peak_amplitude())
        assert contour.peak_profile(chord) == pytest.approx(expected)

    def test_profile_stays_strictly_positive(self):
        """A radius that reaches zero is not a star-shaped ring any more.

        The warp maps every pixel by rho * r(theta,t) / r_peak(theta); a zero in
        the denominator would blow that map up. AMPLITUDE is far from this, but
        the property is what keeps the warp well defined, so it is asserted
        rather than assumed.
        """
        for chord in ALL_CHORDS:
            assert contour.peak_profile(chord).min() > 0.0


class TestPixelMapping:
    def test_every_chord_shares_one_scale(self):
        """Rest radius must not drift between characters.

        Normalizing each chord by its own maximum radius would fit each shape to
        the frame individually, and a message would then pulse in overall size
        from character to character. Worse, the warp reads r_peak in pixels, so a
        per-chord scale would silently change the meaning of those numbers.
        """
        size = 512
        radii = [
            np.hypot(*(contour.to_pixels(contour.peak_profile(c), size) - size / 2.0).T).max()
            for c in ALL_CHORDS
        ]
        assert max(radii) - min(radii) < 0.05 * max(radii)

    def test_points_stay_inside_the_canvas(self):
        for chord in ALL_CHORDS:
            points = contour.to_pixels(contour.peak_profile(chord), 512)
            assert points.min() >= 0.0
            assert points.max() <= 511.0

    def test_mapping_is_centred(self):
        size = 512
        points = contour.to_pixels(contour.peak_profile((2, 5)), size)
        assert points.mean(axis=0) == pytest.approx(size / 2.0, abs=0.5)

    def test_scale_is_proportional_to_size(self):
        small = contour.to_pixels(contour.peak_profile((3, 7)), 256) - 128.0
        large = contour.to_pixels(contour.peak_profile((3, 7)), 1024) - 512.0
        assert large == pytest.approx(small * 4.0, abs=1e-4)


class TestControlImage:
    def test_control_image_is_black_line_on_white(self):
        image = contour.render_control_image((4, 9), size=256)
        assert image.dtype == np.uint8
        assert image.shape == (256, 256)
        assert image.max() == 255            # background survives untouched
        assert image.min() == 0              # the stroke is fully black
        assert (image < 128).mean() < 0.15   # a line, not a filled disc

    def test_the_curve_is_closed(self):
        """No seam where theta wraps.

        Drawing 512 open segments leaves a gap between the last point and the
        first. A gap breaks the star-shaped ray walk the extractor depends on,
        and a canny-conditioned model would faithfully reproduce the notch.
        """
        image = contour.render_control_image((2, 12), size=512)
        ink = image < 128
        # A closed curve separates inside from outside: flood filling from a
        # corner must not reach the centre.
        import cv2
        flooded = ink.astype(np.uint8).copy()
        mask = np.zeros((flooded.shape[0] + 2, flooded.shape[1] + 2), np.uint8)
        cv2.floodFill(flooded, mask, (0, 0), 1)
        assert flooded[256, 256] == 0

    def test_interior_is_untouched_background(self):
        image = contour.render_control_image((5, 8), size=512)
        assert image[256, 256] == 255

    @pytest.mark.parametrize("size", [256, 512, 1024])
    def test_renders_at_any_size(self, size):
        image = contour.render_control_image((3, 6), size=size)
        assert image.shape == (size, size)
