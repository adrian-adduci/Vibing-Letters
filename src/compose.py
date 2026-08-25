"""Turn a sentence into one file.

This is the entire runtime. No image generation, no geometry, no model calls: a
lookup and a concatenation. Every frame it emits came out of an asset that
already passed the gate, which is why runtime cannot produce an undecodable
file -- there is no step here capable of inventing one.

Rotation is the only thing done to the frames. Each character instance is turned
by an angle seeded from the sentence and the position, so a message does not
read as the same ring stamped repeatedly, while the same sentence always
produces a byte-identical file. Rotation cannot affect decoding: the magnitude
spectrum does not see it.
"""

import hashlib
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from .assets import emit
from .codec import constants as C
from .codec.chord_table import CHORD_BY_SYMBOL, normalize

SENTINEL_NAME = "SENTINEL"


class Composed(NamedTuple):
    """A composed message and what had to be done to the caller's text."""

    frames: np.ndarray
    text: str
    dropped: list[str]


def symbol_name(symbol: str) -> str:
    """A filesystem-safe name for a symbol, matching what the build wrote."""
    if symbol == ' ':
        return 'SPACE'
    if symbol.isalnum():
        return symbol
    return f"U{ord(symbol):04X}"


def rotation_for(sentence: str, index: int) -> float:
    """The angle for one character instance, in degrees.

    Seeded from a real hash rather than `hash()`, which is salted per process
    and would make the same sentence produce a different file on every run. The
    point of seeding at all is that it is reproducible.
    """
    key = f"{sentence}\x00{index}".encode('utf-8')
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'big') % 360 / 1.0


def rotate(frames: np.ndarray, degrees: float) -> np.ndarray:
    """Turn every frame of a clip by the same angle.

    The same angle for all of them, not a different one per frame: the ring is
    meant to vibrate, not to spin.
    """
    if degrees == 0.0:
        return frames
    height, width = frames.shape[1:3]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), degrees, 1.0)
    return np.stack([
        cv2.warpAffine(frame, matrix, (width, height),
                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        for frame in frames
    ])


def load_clips(clips_dir: str | Path) -> dict[str, np.ndarray]:
    """Read every published clip into memory, keyed by symbol name.

    Args:
        clips_dir: Directory the publish step wrote to.

    Returns:
        dict: Symbol name to frames.

    Raises:
        FileNotFoundError: If the directory has no clips at all.
    """
    directory = Path(clips_dir)
    clips = {path.stem: emit.read_clip(path)
             for path in sorted(directory.glob("*.webp"))}
    if not clips:
        raise FileNotFoundError(f"No clips found in {directory}")
    return clips


def compose(
    sentence: str,
    clips: dict[str, np.ndarray],
    strict: bool = False,
) -> Composed:
    """Build the frames for a message.

    Args:
        sentence: The text to encode.
        clips: Loaded assets, from `load_clips`.
        strict: Raise on unrepresentable characters instead of dropping them.

    Returns:
        Composed: Frames, the normalized text they actually carry, and whatever
        normalization had to drop.

    Raises:
        KeyError: If a needed asset is missing. Composing part of a message and
            leaving the rest out would produce a file that decodes cleanly to
            the wrong words, which is worse than refusing.
    """
    text, dropped = normalize(sentence, strict=strict)

    needed = {SENTINEL_NAME} | {symbol_name(s) for s in text}
    missing = sorted(needed - set(clips))
    if missing:
        raise KeyError(f"No published asset for: {', '.join(missing)}")

    parts = [clips[SENTINEL_NAME]]
    for index, symbol in enumerate(text):
        parts.append(rotate(clips[symbol_name(symbol)],
                            rotation_for(text, index)))
    parts.append(clips[SENTINEL_NAME])

    return Composed(np.concatenate(parts), text, dropped)


def write(
    sentence: str,
    path: str | Path,
    clips: dict[str, np.ndarray],
    strict: bool = False,
    url: str = emit.DECODER_URL,
) -> Composed:
    """Compose a message and write it, choosing the format from the suffix.

    Args:
        sentence: The text to encode.
        path: Destination; `.gif` writes a GIF, anything else a WebP.
        clips: Loaded assets, from `load_clips`.
        strict: Raise on unrepresentable characters instead of dropping them.
        url: Decoder link to embed.

    Returns:
        Composed: As `compose`, after the file has been written.
    """
    result = compose(sentence, clips, strict=strict)
    destination = Path(path)
    writer = (emit.write_gif if destination.suffix.lower() == '.gif'
              else emit.write_webp)
    writer(result.frames, destination, url=url)
    return result
