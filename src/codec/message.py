"""Whole-sentence encoding and decoding.

The boundary of the codec: text in, radius profiles out, and back again. No
image handling lives here or anywhere below it.
"""

from typing import NamedTuple

import numpy as np

from . import constants as C
from .chord_table import CHORD_BY_SYMBOL, SPACE, SYMBOL_BY_CHORD, normalize
from .spectrum import active_mask, close_short_gaps, detect_chord, frame_peaks, runs_of
from .waveform import chord_clip, quiet_clip


class Encoded(NamedTuple):
    """The result of encoding a sentence.

    `text` is the normalized form actually encoded, which may differ from the
    caller's input: normalization uppercases, drops unrepresentable characters,
    strips leading and trailing whitespace, and can even lengthen the string
    through Unicode case expansion. Returning it means a caller never has to
    re-derive what was encoded, and it is the correct right-hand side of the
    round-trip property: decode(result.frames) == result.text.
    """

    frames: np.ndarray
    text: str
    dropped: list[str]


def encode(sentence: str, strict: bool = False) -> Encoded:
    """Encode a sentence as a sequence of radius profiles.

    The message is bracketed by sentinel clips so the decoder can find its
    boundaries even when leading or trailing frames are lost.

    Args:
        sentence: Text to encode.
        strict: Raise on unsupported characters instead of dropping them.

    Returns:
        Encoded: frames of shape (n_frames, N_BINS), the normalized text, and
        the characters that were dropped.
    """
    text, dropped = normalize(sentence, strict=strict)

    clips = [chord_clip(C.SENTINEL_CHORD)]
    for symbol in text:
        clips.append(quiet_clip() if symbol == SPACE else chord_clip(CHORD_BY_SYMBOL[symbol]))
    clips.append(chord_clip(C.SENTINEL_CHORD))

    return Encoded(np.concatenate(clips), text, dropped)


# Stands in for a ring that was loud enough to segment but decoded to a chord no
# character owns. Using the Unicode replacement character keeps decode total: a
# damaged ring costs one character rather than the whole message.
UNDECODABLE = '�'


def _symbol_for(chord: tuple[int, int] | None, confidence: float) -> str:
    """Map a detected chord to its character, or mark it undecodable.

    Two distinct failures land here, and neither can be caught by a table
    lookup alone.

    A chord no character owns -- one of the twelve spare pairs -- fails the
    lookup outright.

    A degenerate ring, excited in a single mode, is the subtler case.
    `detect_chord` always returns two distinct modes, so a one-mode ring comes
    back as its real peak paired with whatever float-rounding bin ranked second:
    a ring at mode 5 reports (4, 5), which is a perfectly valid entry meaning
    'H'. The lookup cannot see anything wrong. Only confidence separates them.

    That separation is narrower than it looks, and depends on which frame the
    caller hands over. At the argmax frame of a degenerate clip -- the frame
    `decode` selects -- confidence tops out well clear of the gate. Across every
    above-threshold frame of the same clip it reaches within 1.2% of it. Anyone
    who changes how a segment's representative frame is chosen, or any image
    decoder that lands on a frame other than the loudest, gives up most of that
    margin. See MIN_CONFIDENCE in constants for the measured populations; the
    numbers are kept in one place deliberately.

    The encoder never emits either case, so the round-trip property is
    unaffected; both arise only from corrupted or externally supplied rings.
    """
    if chord is None or confidence < C.MIN_CONFIDENCE:
        return UNDECODABLE
    return SYMBOL_BY_CHORD.get(chord, UNDECODABLE)


def _spaces_in_gap(length: int) -> int:
    """Convert a quiet-run length into a count of spaces.

    A space is a still circle, so unlike every other symbol it produces no
    active segment to count. It is recovered instead from how much longer the
    silence ran than a plain character boundary.

    The rounding gives generous slack: any gap in [0, 17] yields zero spaces and
    any in [18, 32] yields one, and the clamp means a short gap can never
    fabricate a space.
    """
    return max(0, round((length - C.BOUNDARY_GAP_FRAMES) / C.FRAMES_PER_CHAR))


def _tolerant_peaks(frames: np.ndarray) -> np.ndarray:
    """Peak modal magnitude per frame, treating an unreadable frame as quiet.

    `frame_peaks` transforms the whole stack in one call, which is what makes
    decoding cheap, but it also means a single unusable profile takes the batch
    down with it: `mode_band` rejects a non-positive mean radius, and one
    all-zero row is enough. A failed contour is exactly what an image decoder
    produces, and one bad ring out of a thousand must not cost the message.

    So: try the batch, and only on failure fall back to a per-frame pass where
    an unreadable profile scores 0.0 and segments away into a gap. If every
    frame fails the input is malformed as a whole -- too few angular samples to
    resolve MAX_MODE, say -- and the original error is re-raised rather than
    silently returning an empty decode.
    """
    try:
        return frame_peaks(frames)
    except ValueError:
        pass

    peaks = np.zeros(len(frames), dtype=float)
    failures: list[ValueError] = []
    for index, frame in enumerate(frames):
        try:
            peaks[index] = frame_peaks(frame)
        except ValueError as error:
            failures.append(error)
    if len(failures) == len(frames):
        raise failures[0]
    return peaks


def decode(frames: np.ndarray) -> str:
    """Recover a sentence from a sequence of radius profiles.

    Args:
        frames: Array of shape (n_frames, n_bins).

    Returns:
        str: The decoded sentence, with sentinels removed. Rings that cannot be
        identified become UNDECODABLE rather than raising, so one damaged
        character costs one character.

    Raises:
        ValueError: If `frames` is not two-dimensional, if no frame can be
            transformed at all, or if fewer than two sentinel markers are found.

    Note:
        The two axes are not equally forgiving, and the asymmetry is the main
        trap when wiring contour extraction into this function.

        The angular axis is fully invariant. Detection divides by mean radius
        and by bin count, so profiles sampled at 32, 64, 128, 256, 512 or 1024
        bins all decode identically, as do profiles at any positive scale or any
        rotation. Callers may resample it freely.

        The time axis is rigid. `_spaces_in_gap` measures silence in frames and
        compares it against FRAMES_PER_CHAR and BOUNDARY_GAP_FRAMES, so frames
        must arrive at the cadence the encoder emitted them. A timebase mismatch
        corrupts the output silently instead of failing: on encode("A B"),
        feeding frames at 2x rate decodes as ' A   B ', 3x as
        '� AA    BB �', and 0.5x as 'AB' with the space lost entirely.
    """
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 2:
        raise ValueError(
            f"Expected frames of shape (n_frames, n_bins); got {frames.ndim}-D "
            f"array of shape {frames.shape}"
        )

    # One FFT pass over the whole message. The peaks feed both the active/quiet
    # decision and the per-segment argmax, so nothing is transformed twice.
    peaks = _tolerant_peaks(frames)
    mask = close_short_gaps(active_mask(peaks))

    tokens: list[tuple[str, object]] = []
    for is_active, start, stop in runs_of(mask):
        if not is_active:
            tokens.append(('gap', stop - start))
            continue
        strongest = start + int(np.argmax(peaks[start:stop]))
        try:
            chord, confidence = detect_chord(frames[strongest])
        except ValueError:
            chord, confidence = None, 0.0
        tokens.append(('chord', (chord, confidence)))

    # Sentinels are held to the same standard _symbol_for applies to every other
    # chord. Without the confidence clause a noise frame that happens to rank
    # modes 2 and 12 highest would anchor the message: 54 of 3000 heavy-noise
    # frames reported (2, 12) at confidence 1.0 to 1.2.
    sentinels = [
        index for index, (kind, value) in enumerate(tokens)
        if kind == 'chord'
        and value[0] == C.SENTINEL_CHORD
        and value[1] >= C.MIN_CONFIDENCE
    ]
    if len(sentinels) < 2:
        raise ValueError("Message is missing its sentinel markers")

    body = tokens[sentinels[0] + 1:sentinels[-1]]

    decoded = []
    for kind, value in body:
        if kind == 'chord':
            decoded.append(_symbol_for(*value))
        else:
            decoded.append(SPACE * _spaces_in_gap(value))
    return ''.join(decoded)
