"""Animating a still through the pluck envelope.

The warp is where a curated picture becomes a clip, and it is the last stage
that can silently corrupt the signal: every later stage only copies frames
around. So these tests check the arithmetic of the deformation, not just that
frames come out.
"""
import cv2
import numpy as np
import pytest

from src.assets import contour, warp
from src.codec import constants as C
from src.codec.spectrum import detect_chord, mode_band
from src.vision import ring


def still_for(chord, size=512):
    """A stand-in for a curated asset: the exact peak contour, drawn."""
    return contour.render_control_image(chord, size=size)


class TestMeasure:
    def test_shape_recovers_the_analytic_profile(self):
        """s(theta) read off the picture must match the maths it was drawn from.

        This is the claim the whole warp rests on. If the measured shape were
        even mildly wrong, every frame would be wrong in the same way and the
        error would be invisible in any single frame.
        """
        chord = (3, 8)
        field = warp.measure(still_for(chord, size=1024))
        cx, cy = field.centre

        theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        sampled = field.shape[
            np.round(cy - 300 * np.sin(theta)).astype(int),
            np.round(cx + 300 * np.cos(theta)).astype(int),
        ]
        expected = (np.cos(chord[0] * theta) + np.cos(chord[1] * theta)) / 2.0
        assert sampled == pytest.approx(expected, abs=0.05)

    def test_a_plain_circle_is_refused(self):
        """There is nothing to animate, and pretending otherwise would divide a
        measurement of noise by a small number and amplify it into the frames."""
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        cv2.circle(canvas, (256, 256), 180, 0, 3, cv2.LINE_AA)
        with pytest.raises(ValueError, match="circle"):
            warp.measure(canvas)


class TestSingleFrame:
    def test_the_peak_amplitude_is_the_identity(self):
        """Warping to the amplitude the still already has must change nothing.

        Any drift here would mean the curated frame is not actually in its own
        clip, and the peak frame is the one the acceptance gate judges.
        """
        still = still_for((4, 9))
        field = warp.measure(still)
        result = warp.at_amplitude(still, field, contour.peak_amplitude())
        assert result == pytest.approx(still, abs=1)

    def test_zero_amplitude_yields_a_circle(self):
        """The rest state. Both ends of the envelope land here, which is what
        makes clips loop and lets any clip follow any other."""
        still = still_for((2, 11))
        field = warp.measure(still)
        flat = warp.at_amplitude(still, field, 0.0)

        profile = ring.radius_profile(flat)
        assert profile.std() / profile.mean() < 0.01
        assert detect_chord(profile)[0] is None

    def test_a_rotated_still_flattens_just_as_well(self):
        """Phase independence, which is the reason the shape is measured at all.

        Every analytic contour is a sum of cosines and therefore mirror
        symmetric, so a shape field read with a flipped vertical axis is
        indistinguishable from a correct one on synthetic stills -- and a real
        styled asset, which is not symmetric, would then be warped against its
        own geometry. Rotating first breaks the symmetry: measured this way, the
        correct orientation flattens to a coefficient of variation of 0.00025
        and the flipped one to 0.072, a factor of 287.
        """
        still = still_for((3, 8))
        matrix = cv2.getRotationMatrix2D((256.0, 256.0), 23.0, 1.0)
        rotated = cv2.warpAffine(still, matrix, (512, 512),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        field = warp.measure(rotated)
        flat = warp.at_amplitude(rotated, field, 0.0)
        profile = ring.radius_profile(flat)

        assert profile.std() / profile.mean() < 0.005
        assert detect_chord(ring.radius_profile(rotated))[0] == (3, 8)

    def test_inverted_amplitude_flips_the_lobes(self):
        """Half the frames of every clip are on the far side of zero.

        The standing wave inverts each half cycle, so a bulge becomes a dent.
        The chord is unchanged -- the magnitude spectrum does not see the sign --
        but the picture must genuinely turn inside out, not merely sit still.
        """
        still = still_for((3, 7))
        field = warp.measure(still)
        peak = contour.peak_amplitude()

        forward = ring.radius_profile(warp.at_amplitude(still, field, peak))
        reverse = ring.radius_profile(warp.at_amplitude(still, field, -peak))

        assert detect_chord(reverse)[0] == (3, 7)
        centred_f = forward / forward.mean() - 1.0
        centred_r = reverse / reverse.mean() - 1.0
        assert centred_r == pytest.approx(-centred_f, abs=0.02)

    def test_a_singular_amplitude_is_refused(self):
        still = still_for((2, 12))
        field = warp.measure(still)
        with pytest.raises(ValueError, match="zero radius"):
            warp.at_amplitude(still, field, -2.0)


class TestClip:
    def test_clip_has_one_frame_per_frame_of_the_format(self):
        assert len(warp.envelope_clip(still_for((5, 6)))) == C.FRAMES_PER_CHAR

    def test_trailing_frames_are_silent(self):
        clip = warp.envelope_clip(still_for((6, 9)))
        for frame in clip[C.ACTIVE_FRAMES:]:
            assert detect_chord(ring.radius_profile(frame))[0] is None

    def test_measured_modulation_tracks_the_envelope(self):
        """Not just 'frames differ' -- they must differ by the right amount.

        A chord at amplitude a puts roughly a/2 into each of its two modal bins.
        Reading those bins back out of the warped pixels and comparing against
        the envelope checks the deformation quantitatively, which no assertion
        about the decoded chord can do: the chord is right for any amplitude
        large enough to detect.
        """
        chord = (4, 7)
        clip = warp.envelope_clip(still_for(chord, size=768))

        measured = np.array([
            mode_band(ring.radius_profile(f))[[c - C.MIN_MODE for c in chord]].mean()
            for f in clip
        ])
        expected = np.abs(warp.FRAME_AMPLITUDES) / 2.0

        # Not a slack bound: across seven chords at two render sizes the worst
        # residual measured 1.0e-4, so this leaves a factor of five and no more.
        # Peak modulation is 0.056, making 5e-4 a tenth of one percent of it.
        assert measured == pytest.approx(expected, abs=5e-4)

    def test_no_frame_ever_reports_a_different_chord(self):
        """Silence is acceptable. A wrong answer is not.

        Frames near the envelope's zeros carry no signal and correctly decode to
        nothing. What must never happen is a frame confidently naming some other
        character, because runtime concatenates these frames untouched.
        """
        for chord in [(2, 3), (3, 8), (7, 10), (2, 12), (6, 12)]:
            clip = warp.envelope_clip(still_for(chord, size=512))
            for index, frame in enumerate(clip):
                found, confidence = detect_chord(ring.radius_profile(frame))
                if found is not None and confidence > C.MIN_CONFIDENCE:
                    assert found == chord, f"{chord} frame {index} read as {found}"

    def test_the_peak_frame_decodes_confidently(self):
        for chord in [(2, 3), (5, 9), (8, 9), (2, 12)]:
            clip = warp.envelope_clip(still_for(chord, size=512))
            found, confidence = detect_chord(ring.radius_profile(clip[contour.PEAK_FRAME]))
            assert found == chord
            assert confidence > C.MIN_CONFIDENCE
