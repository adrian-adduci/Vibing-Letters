"""Whole-sentence encoding and decoding.

The boundary of the codec: text in, radius profiles out, and back again. No
image handling lives here or anywhere below it.
"""

from typing import NamedTuple

import numpy as np

from . import constants as C
from .chord_table import CHORD_BY_SYMBOL, SPACE, normalize
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
