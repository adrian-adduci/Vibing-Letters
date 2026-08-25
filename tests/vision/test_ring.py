"""Recovering r(theta) from pixels.

These tests cover the extractor in isolation. Whether the recovered profile
actually decodes to the right chord is a separate, stronger claim, tested in
tests/assets/test_pixel_round_trip.py.
"""
import cv2
import numpy as np
import pytest

from src.assets import contour
from src.codec import constants as C
from src.vision import ring


def draw_circle(size=512, radius=180, centre=None, line_width=3,
                fg=0, bg=255) -> np.ndarray:
    cx, cy = centre if centre else (size / 2.0, size / 2.0)
    canvas = np.full((size, size), bg, dtype=np.uint8)
    cv2.circle(canvas, (round(cx * 16), round(cy * 16)), round(radius * 16),
               fg, line_width, cv2.LINE_AA, shift=4)
    return canvas


class TestInkMap:
    def test_dark_line_on_light_becomes_ink(self):
        ink = ring.ink_map(draw_circle())
        assert ink.max() == pytest.approx(1.0)
        assert ink[0, 0] == pytest.approx(0.0)      # corner is background

    def test_light_line_on_dark_becomes_the_same_ink(self):
        """Polarity is not a flag the caller has to get right.

        Styled assets may come back as a glowing ring on black; control images
        are black on white. Measuring departure from the median background makes
        both read identically, so nothing downstream branches on appearance.
        """
        dark = ring.ink_map(draw_circle(fg=0, bg=255))
        light = ring.ink_map(draw_circle(fg=255, bg=0))
        assert light == pytest.approx(dark, abs=0.02)

    def test_colour_input_is_accepted(self):
        gray = draw_circle()
        colour = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        assert ring.ink_map(colour) == pytest.approx(ring.ink_map(gray))

    def test_uniform_image_is_rejected(self):
        with pytest.raises(ValueError, match="uniform"):
            ring.ink_map(np.full((64, 64), 200, dtype=np.uint8))

    def test_empty_image_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            ring.ink_map(np.zeros((0, 0), dtype=np.uint8))


class TestLocateRing:
    def test_finds_the_centre_of_a_centred_circle(self):
        ink = ring.ink_map(draw_circle(size=512))
        assert ring.locate_ring(ink) == pytest.approx((256.0, 256.0), abs=0.5)

    def test_finds_the_centre_of_an_offset_circle(self):
        ink = ring.ink_map(draw_circle(size=512, centre=(300.0, 220.0)))
        assert ring.locate_ring(ink) == pytest.approx((300.0, 220.0), abs=0.5)

    @pytest.mark.parametrize("chord", [(2, 3), (4, 9), (2, 12), (7, 8)])
    def test_finds_the_centre_of_a_wavy_ring(self, chord):
        """The waviness must not pull the centre off.

        Every chord is built from modes 2 and above, which contribute nothing to
        the mode-1 term. That is the whole reason mode 1 is barred from the
        alphabet, and it is what lets the same correction that finds the centre
        also leave a correctly centred ring alone.
        """
        image = contour.render_control_image(chord, size=512)
        ink = ring.ink_map(image)
        assert ring.locate_ring(ink) == pytest.approx((256.0, 256.0), abs=0.5)

    def test_correction_is_applied_in_the_right_vertical_direction(self):
        """The sign of the y correction has to be tested deliberately.

        For any symmetric ring the ink centroid is already the true centre, so
        the correction is zero and a sign error is invisible. Only ink that is
        lopsided along y exercises it: here the top arc is thickened, which
        drags the starting centroid 35 px upward and demands a downward step.
        """
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        cv2.circle(canvas, (256, 256), 180, 0, 3, cv2.LINE_AA)
        cv2.ellipse(canvas, (256, 256), (180, 180), 0, -120, -60, 0, 11, cv2.LINE_AA)
        ink = ring.ink_map(canvas)

        total = ink.sum()
        rows, cols = np.indices(ink.shape, dtype=np.float32)
        start_y = float((rows * ink).sum() / total)

        assert start_y < 250.0                       # genuinely dragged upward
        assert ring.locate_ring(ink) == pytest.approx((256.0, 256.0), abs=0.6)

    def test_refinement_beats_the_bare_centroid(self):
        """The starting centroid is not already the answer.

        If it were, the refinement loop would be dead code that no test could
        distinguish from a correct one. An asymmetric ink distribution -- here a
        stroke that thickens on one side -- separates them.
        """
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        cv2.circle(canvas, (256, 256), 180, 0, 3, cv2.LINE_AA)
        # Thicken one flank without moving the curve.
        cv2.ellipse(canvas, (256, 256), (180, 180), 0, -60, 60, 0, 9, cv2.LINE_AA)
        ink = ring.ink_map(canvas)

        total = ink.sum()
        rows, cols = np.indices(ink.shape, dtype=np.float32)
        centroid = ((cols * ink).sum() / total, (rows * ink).sum() / total)
        refined = ring.locate_ring(ink)

        assert abs(centroid[0] - 256.0) > 1.0        # centroid is genuinely off
        assert refined == pytest.approx((256.0, 256.0), abs=0.6)

    def test_blank_ink_is_rejected(self):
        with pytest.raises(ValueError, match="no ink"):
            ring.locate_ring(np.zeros((64, 64), dtype=np.float32))


class TestProfile:
    def test_a_circle_has_a_flat_profile(self):
        profile = ring.radius_profile(draw_circle(size=512, radius=180))
        assert profile.mean() == pytest.approx(180.0, abs=1.0)
        assert profile.std() < 0.5

    def test_a_circle_carries_no_chord(self):
        """A pure circle is the rest state and must read as silence."""
        from src.codec.spectrum import detect_chord
        chord, _ = detect_chord(ring.radius_profile(draw_circle()))
        assert chord is None

    def test_radius_is_the_stroke_centre_not_its_edge(self):
        """Measured radius must not drift with stroke width.

        A first-crossing test would report the inner edge and shrink as the
        stroke fattens. The weighted centroid reports the middle of the stroke,
        which is the curve, whatever its width.
        """
        thin = ring.radius_profile(draw_circle(radius=180, line_width=3)).mean()
        fat = ring.radius_profile(draw_circle(radius=180, line_width=15)).mean()
        assert thin == pytest.approx(fat, abs=0.7)

    def test_a_faint_companion_halo_does_not_drag_the_radius_out(self):
        """Generated art will not be clean line work.

        A styled ring arrives with a glow around it, and glow that sits further
        out than the stroke pulls an unfiltered centroid outward with it. Here a
        faint ring at radius 200 shifts the measurement to 185 without the
        per-ray floor and leaves it at 180 with it -- a 5 px error, which at
        mode 12 is most of a lobe.
        """
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        cv2.circle(canvas, (256 * 16, 256 * 16), 200 * 16, 205, 7, cv2.LINE_AA, shift=4)
        cv2.circle(canvas, (256 * 16, 256 * 16), 180 * 16, 0, 3, cv2.LINE_AA, shift=4)
        assert ring.radius_profile(canvas).mean() == pytest.approx(180.0, abs=1.0)

    def test_radius_tracks_sub_pixel_changes_smoothly(self):
        """Ink weighting buys sub-pixel resolution; thresholding does not.

        Sliding the true radius across one pixel and watching the measurement
        follow is what separates a weighted centroid from a count of
        above-threshold samples. Measured wobble is 0.035 px weighted against
        0.106 px for the binary alternative, so the bound below fails if the
        weighting is ever dropped.
        """
        errors = []
        for radius in np.arange(180.0, 181.01, 0.125):
            canvas = np.full((512, 512), 255, dtype=np.uint8)
            cv2.circle(canvas, (256 * 16, 256 * 16), round(radius * 16), 0, 3,
                       cv2.LINE_AA, shift=4)
            errors.append(ring.radius_profile(canvas).mean() - radius)
        assert max(errors) - min(errors) < 0.06

    def test_profile_length_follows_n_bins(self):
        for n_bins in (64, 128, 512):
            assert len(ring.radius_profile(draw_circle(), n_bins)) == n_bins

    def test_off_centre_sampling_is_rejected_when_it_leaves_no_room(self):
        ink = ring.ink_map(draw_circle(size=512))
        with pytest.raises(ValueError, match="no room"):
            ring.profile_about(ink, (0.5, 256.0))


class TestGaps:
    def test_a_broken_stroke_is_bridged_rather_than_refused(self):
        """One damaged arc costs accuracy at those angles, not the whole ring."""
        canvas = draw_circle(size=512, radius=180)
        canvas[150:250, 60:130] = 255            # erase a chunk of the stroke
        profile = ring.radius_profile(canvas)
        assert np.isfinite(profile).all()
        assert profile.mean() == pytest.approx(180.0, abs=6.0)
