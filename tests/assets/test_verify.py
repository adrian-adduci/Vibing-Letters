"""The gate that decides what is allowed into the repository.

Two kinds of test here. The rule itself is checked against synthetic readings,
where every case -- including ones no renderer would produce -- can be
constructed exactly. Then the whole path is checked against real pixels, to make
sure the rule is being fed what it thinks it is.
"""
import cv2
import numpy as np
import pytest

from src.assets import contour, verify, warp
from src.assets.verify import FrameReading
from src.codec import constants as C
from src.codec.chord_table import CHORD_BY_SYMBOL

CHORD = (4, 7)
OTHER = (3, 9)


def reading(index, chord, confidence):
    return FrameReading(index, chord, confidence)


def clip_of(*readings):
    return tuple(readings)


def good_clip(chord=CHORD, length=C.FRAMES_PER_CHAR):
    """Readings for a clip that should pass: loud peak, silent tail."""
    return tuple(
        reading(i, chord, 40.0) if i == contour.PEAK_FRAME else reading(i, None, 0.0)
        for i in range(length)
    )


class TestTheRule:
    def test_a_loud_peak_and_silence_elsewhere_passes(self):
        verdict = verify.judge(good_clip(), CHORD)
        assert verdict.accepted
        assert verdict.reasons == ()
        assert bool(verdict) is True

    def test_silence_everywhere_else_is_not_a_defect(self):
        """The envelope crosses zero mid-character by design.

        Requiring every frame to decode would reject every correct asset, which
        is why the rule is asymmetric rather than uniform.
        """
        verdict = verify.judge(good_clip(), CHORD)
        assert sum(r.is_silent for r in verdict.readings) == C.FRAMES_PER_CHAR - 1
        assert verdict.accepted

    def test_a_silent_peak_frame_fails(self):
        readings = tuple(reading(i, None, 0.0) for i in range(C.FRAMES_PER_CHAR))
        verdict = verify.judge(readings, CHORD)
        assert not verdict
        assert "silent" in verdict.reasons[0]

    def test_a_peak_frame_with_the_wrong_chord_fails(self):
        readings = list(good_clip())
        readings[contour.PEAK_FRAME] = reading(contour.PEAK_FRAME, OTHER, 40.0)
        verdict = verify.judge(tuple(readings), CHORD)
        assert not verdict
        # Reported once, in the curator's terms. The general sweep skips the
        # peak frame precisely so the commonest failure -- a candidate that came
        # back as the wrong shape -- is not also restated anonymously.
        assert verdict.reasons == (
            f"peak frame {contour.PEAK_FRAME} decodes to {OTHER}, not {CHORD}",
        )

    def test_a_peak_frame_below_the_confidence_gate_fails(self):
        """A degenerate ring names a plausible chord with no conviction.

        Excited in a single mode, it comes back as a valid-looking pair, so a
        table lookup alone sees nothing wrong. Confidence is the only thing that
        separates that from a real chord, and the gate has to apply it too.
        """
        readings = list(good_clip())
        readings[contour.PEAK_FRAME] = reading(
            contour.PEAK_FRAME, CHORD, C.MIN_CONFIDENCE - 0.01
        )
        verdict = verify.judge(tuple(readings), CHORD)
        assert not verdict
        assert "confidence" in verdict.reasons[0]

    def test_any_frame_confidently_naming_another_chord_fails(self):
        """The failure that matters. Runtime ships these frames untouched, so a
        stray frame reading as some other letter becomes a corrupted message."""
        readings = list(good_clip())
        readings[2] = reading(2, OTHER, 30.0)
        verdict = verify.judge(tuple(readings), CHORD)
        assert not verdict
        assert any("frame 2" in reason for reason in verdict.reasons)

    def test_a_diverging_frame_below_the_confidence_gate_is_tolerated(self):
        """Only confident disagreement is a defect.

        Below MIN_CONFIDENCE the decoder reports the segment as undecodable
        rather than guessing, so such a frame cannot put a wrong letter into a
        message. Rejecting on it would discard usable assets for a fault that
        cannot reach the reader.
        """
        readings = list(good_clip())
        readings[2] = reading(2, OTHER, C.MIN_CONFIDENCE - 0.5)
        assert verify.judge(tuple(readings), CHORD).accepted

    def test_every_offending_frame_is_named_not_just_the_first(self):
        """The curator is choosing between candidates and wants the whole
        picture, not one symptom at a time."""
        readings = list(good_clip())
        readings[1] = reading(1, OTHER, 30.0)
        readings[5] = reading(5, (2, 6), 22.0)
        verdict = verify.judge(tuple(readings), CHORD)
        assert len(verdict.reasons) == 2

    def test_a_clip_that_never_goes_quiet_is_rejected(self):
        """The hole the first full asset set fell through.

        The gate said silence was acceptable and never checked it was present,
        but quiet frames are the delimiters. Three clips of forty-four came back
        permanently excited -- the model had drawn the ring as two parallel
        filaments, which breaks the single-valued r(theta) the warp assumes --
        and the character after each one merged into the same segment and
        vanished. `MEET ME AT 8PM!` decoded as `MEET ME AT 8M!` with every
        individual frame perfectly legal.
        """
        readings = list(good_clip())
        readings[-1] = reading(len(readings) - 1, CHORD, 30.0)
        verdict = verify.judge(tuple(readings), CHORD)
        assert not verdict
        assert "never returns to rest" in verdict.reasons[0]

    def test_the_rest_rule_looks_at_the_right_chord_too(self):
        """Ending loud is a defect even when it is loud with the *correct*
        chord. Nothing about the delimiter cares which letter is playing."""
        readings = list(good_clip())
        readings[-1] = reading(len(readings) - 1, CHORD, 99.0)
        assert not verify.judge(tuple(readings), CHORD)

    def test_a_single_still_is_not_asked_to_be_at_rest(self):
        """A still is all peak. Applying the rule to one would reject every
        candidate that was correctly excited, which is all of them."""
        readings = (reading(0, CHORD, 40.0),)
        assert verify.judge(readings, CHORD, peak_frame=0, require_rest=False)
        assert not verify.judge(readings, CHORD, peak_frame=0)

    def test_an_empty_clip_is_rejected_outright(self):
        with pytest.raises(ValueError, match="no frames"):
            verify.judge((), CHORD)

    def test_a_peak_frame_past_the_end_is_rejected(self):
        with pytest.raises(ValueError, match="outside"):
            verify.judge(good_clip(length=4), CHORD, peak_frame=9)


class TestAgainstRealPixels:
    @pytest.mark.parametrize("chord", [(2, 3), (4, 7), (2, 12), (8, 9), (6, 12)])
    def test_a_warped_clip_from_an_exact_contour_is_accepted(self, chord):
        clip = warp.envelope_clip(contour.render_control_image(chord, size=512))
        verdict = verify.accept(clip, chord)
        assert verdict.accepted, verdict.reasons
        assert verdict.peak.chord == chord

    def test_a_clip_is_rejected_against_the_wrong_chord(self):
        """The gate must be capable of saying no to real input, not only to
        hand-built readings."""
        clip = warp.envelope_clip(contour.render_control_image((4, 7), size=512))
        assert not verify.accept(clip, (3, 9))

    def test_a_single_still_can_be_judged_before_warping(self):
        still = contour.render_control_image((5, 9), size=512)
        assert verify.accept_still(still, (5, 9))
        assert not verify.accept_still(still, (5, 10))

    def test_a_circle_still_is_rejected(self):
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        cv2.circle(canvas, (256, 256), 180, 0, 3, cv2.LINE_AA)
        verdict = verify.accept_still(canvas, (5, 9))
        assert not verdict
        assert "silent" in verdict.reasons[0]

    def test_an_unreadable_frame_is_a_defect_not_a_crash(self):
        """A blank frame cannot be measured at all. The caller asked for a
        verdict on a clip, so it gets one rather than an exception from the
        middle of the extractor."""
        clip = warp.envelope_clip(contour.render_control_image((3, 8), size=512))
        clip[contour.PEAK_FRAME] = 255
        verdict = verify.accept(clip, (3, 8))
        assert not verdict
        assert verdict.peak.chord is None

    def test_a_spliced_in_foreign_frame_is_caught(self):
        """The exact defect the gate exists for.

        Nothing about a clip's own frames guarantees they all came from the same
        still. Dropping one frame of a different character in must be detected,
        because runtime will happily ship it.
        """
        clip = warp.envelope_clip(contour.render_control_image((3, 8), size=512))
        intruder = warp.envelope_clip(contour.render_control_image((5, 11), size=512))
        clip[2] = intruder[contour.PEAK_FRAME]

        verdict = verify.accept(clip, (3, 8))
        assert not verdict
        assert any("frame 2" in reason for reason in verdict.reasons)


class TestWholeAlphabet:
    def test_every_character_produces_an_acceptable_clip(self):
        """The pipeline has to work for all 43 characters plus the sentinel, not
        for a sample. Anything that fails here is unshippable by definition."""
        chords = sorted(set(CHORD_BY_SYMBOL.values()) | {C.SENTINEL_CHORD})
        rejected = []
        for chord in chords:
            clip = warp.envelope_clip(contour.render_control_image(chord, size=384))
            verdict = verify.accept(clip, chord)
            if not verdict:
                rejected.append((chord, verdict.reasons))
        assert rejected == []
