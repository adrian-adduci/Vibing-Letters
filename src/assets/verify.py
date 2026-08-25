"""The gate every asset must pass before it is allowed into the repository.

The obvious rule -- every frame must decode -- is wrong. Frames where the pluck
envelope passes through zero are, by construction, near-perfect circles carrying
no signal at all, and demanding a chord from them would reject correct assets.
The rule that is actually right is asymmetric:

* the **loudest frame must decode to the intended chord**, confidently,
* **no frame may decode to a different one**, and
* **the clip must return to silence**.

Silence is acceptable. A wrong answer is not. Runtime concatenates these frames
untouched and never regenerates anything, so a defect that gets past this gate
gets shipped, and the only place it will surface is in someone's decoded message
reading a letter that was never written.

The third rule was added after the first full asset set was built. The gate said
silence was *acceptable* and never checked that it was *present*, which is a hole
the size of the format: quiet frames are the delimiters. Three clips of
forty-four came back permanently excited -- the model had drawn the ring as two
parallel filaments, which breaks the single-valued r(theta) the warp assumes, so
flattening the measured profile did not flatten the picture. Their tails never
dropped below the quiet threshold, the character after them merged into the same
segment, and `MEET ME AT 8PM!` decoded as `MEET ME AT 8M!` -- a letter lost, with
every individual frame perfectly legal.
"""

from typing import NamedTuple

import numpy as np

from ..codec import constants as C
from ..codec.spectrum import detect_chord
from ..vision import ring
from .contour import PEAK_FRAME


class FrameReading(NamedTuple):
    """What the decoder makes of one frame."""

    index: int
    chord: tuple[int, int] | None
    confidence: float

    @property
    def is_silent(self) -> bool:
        """No mode cleared the quiet threshold: a rest-state circle."""
        return self.chord is None

    @property
    def is_confident(self) -> bool:
        return self.chord is not None and self.confidence > C.MIN_CONFIDENCE


class Verdict(NamedTuple):
    """The gate's decision, with enough detail to record and to act on."""

    accepted: bool
    chord: tuple[int, int]
    peak: FrameReading
    readings: tuple[FrameReading, ...]
    reasons: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.accepted


def read_frames(clip: np.ndarray, n_bins: int = C.N_BINS) -> tuple[FrameReading, ...]:
    """Decode every frame of a clip independently.

    Args:
        clip: Frames stacked on a leading axis.
        n_bins: Angular resolution of the extraction.

    Returns:
        tuple: One FrameReading per frame, in order.
    """
    readings = []
    for index, frame in enumerate(clip):
        try:
            chord, confidence = detect_chord(ring.radius_profile(frame, n_bins))
        except ValueError:
            # An unreadable frame is a defect, not an exception to propagate:
            # the caller wants a verdict on the whole clip, and a frame that
            # cannot be measured is recorded as carrying nothing.
            chord, confidence = None, 0.0
        readings.append(FrameReading(index, chord, float(confidence)))
    return tuple(readings)


def judge(
    readings: tuple[FrameReading, ...],
    chord: tuple[int, int],
    peak_frame: int = PEAK_FRAME,
    require_rest: bool = True,
) -> Verdict:
    """Apply the acceptance rule to already-decoded frames.

    Separated from `read_frames` so the rule can be tested against synthetic
    readings without rendering anything, and so a caller that already has
    readings does not pay to extract them twice.

    Args:
        readings: One reading per frame, from `read_frames`.
        chord: The chord the clip is supposed to carry.
        peak_frame: Index of the loudest frame.
        require_rest: Require the clip to end silent. Off for a single still,
            which is all peak and has no rest state to check.

    Returns:
        Verdict: Accepted only if the peak frame is confidently `chord`, no
        frame confidently claims anything else, and the clip ends at rest.

    Raises:
        ValueError: If `readings` is empty or does not reach `peak_frame`.
    """
    if not readings:
        raise ValueError("A clip with no frames cannot be judged")
    if not 0 <= peak_frame < len(readings):
        raise ValueError(
            f"Peak frame {peak_frame} is outside a clip of {len(readings)} frames"
        )

    peak = readings[peak_frame]
    reasons: list[str] = []

    if peak.chord is None:
        reasons.append(f"peak frame {peak.index} is silent; expected {chord}")
    elif peak.chord != chord:
        reasons.append(
            f"peak frame {peak.index} decodes to {peak.chord}, not {chord}"
        )
    elif not peak.is_confident:
        reasons.append(
            f"peak frame {peak.index} decodes to {chord} but confidence "
            f"{peak.confidence:.2f} is below {C.MIN_CONFIDENCE}"
        )

    # A frame that names another chord is only a defect when it says so
    # confidently. Below MIN_CONFIDENCE the decoder already reports the segment
    # as undecodable rather than guessing, so such a frame cannot put a wrong
    # letter into anyone's message.
    #
    # The peak frame is skipped because it was judged above, against a stricter
    # rule. Including it would report the single most common failure -- a
    # candidate that simply came back as the wrong shape -- twice, once in the
    # curator's terms and once anonymously.
    for reading in readings:
        if reading.index == peak_frame:
            continue
        if reading.is_confident and reading.chord != chord:
            reasons.append(
                f"frame {reading.index} decodes to {reading.chord} at "
                f"confidence {reading.confidence:.2f}"
            )

    # A clip that never goes quiet has no delimiter, and runtime concatenates
    # clips without inserting one. The character that follows merges into the
    # same segment and is lost -- not misread, lost -- while every individual
    # frame remains perfectly legal.
    if require_rest and not readings[-1].is_silent:
        reasons.append(
            f"clip never returns to rest: final frame {readings[-1].index} "
            f"still reads {readings[-1].chord}"
        )

    return Verdict(
        accepted=not reasons,
        chord=chord,
        peak=peak,
        readings=readings,
        reasons=tuple(reasons),
    )


def accept(
    clip: np.ndarray,
    chord: tuple[int, int],
    n_bins: int = C.N_BINS,
    peak_frame: int = PEAK_FRAME,
) -> Verdict:
    """Decode a clip and judge it in one call.

    Args:
        clip: Frames stacked on a leading axis.
        chord: The chord the clip is supposed to carry.
        n_bins: Angular resolution of the extraction.
        peak_frame: Index of the loudest frame.

    Returns:
        Verdict: See `judge`.
    """
    return judge(read_frames(clip, n_bins), chord, peak_frame)


def accept_still(
    still: np.ndarray,
    chord: tuple[int, int],
    n_bins: int = C.N_BINS,
) -> Verdict:
    """Judge a single curated still before spending the warp on it.

    Curation is the one manual step in the pipeline, so it is worth telling the
    curator immediately that a candidate is unusable rather than after fifteen
    frames have been built from it.

    Args:
        still: A candidate peak-excitation image.
        chord: The chord it is supposed to carry.
        n_bins: Angular resolution of the extraction.

    Returns:
        Verdict: Over a single frame, which is therefore also the peak frame.
        The rest-state rule is not applied: a still is all peak and has no rest
        state to check. That is exactly why a still passing here is not evidence
        its clip will -- only `accept` can tell you that.
    """
    return judge(read_frames(still[None, ...], n_bins), chord, peak_frame=0,
                 require_rest=False)
