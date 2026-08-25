"""Write a clip to a shippable file, with a link home that survived the write.

Two formats, for the reason the design gives: WebP at 512 px is canonical, GIF
at 256 px is the fallback for anywhere WebP is not accepted. Both carry the
decoder URL in metadata and nothing visible on the artwork.

The link is written and then **read back out of the saved file**. Pillow's
metadata support varies by format and by version, and a link that silently
failed to attach looks exactly like one that attached fine until someone tries
to follow it. Verified here, once, rather than assumed everywhere.

**Frame counts are not preserved, and must not be relied on.** Both encoders
merge runs of byte-identical consecutive frames into one longer-lived frame,
neither offers a way to switch it off, and animated WebP does not report the
merged duration back through Pillow at all. The envelope is pinned to zero at
both ends, so a clip's last active frame is already a rest circle before the
silent frames begin: TRAILING_SILENCE_FRAMES + 1 identical circles collapse to
one and every clip leaves here as twelve frames rather than fifteen.

This is safe, and the reason is worth stating because it is not obvious. What
the decoder measures is the *length of a quiet run* against MIN_CLOSABLE_GAP,
and a character boundary is not made of identical frames: it spans two
different stills, whose rest-state circles are different pictures. Measured
across the whole alphabet written as one message, boundary runs come back at
seven frames and the mid-character zero crossings that MIN_CLOSABLE_GAP exists
to close come back at one, against a threshold of three. Three-way separation,
not a margin. Losing identical frames is lossless by definition; only the
counting could have broken, and it does not.
"""

from pathlib import Path
from xml.etree import ElementTree
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

from ..codec import constants as C

# PROVISIONAL. The design lists "where is the decoder hosted, and what is the
# final URL?" as an open question, so this is a placeholder shaped like the
# answer rather than the answer. Every emitted file embeds it, so changing it
# later means re-emitting -- it is worth settling before a batch is published.
DECODER_URL = "https://adrian-adduci.github.io/Vibing-Letters/"

# ~33 ms a frame puts one character at just under half a second, per the design.
# GIF stores delays in hundredths of a second, so its frames land on 30 ms; the
# decoder is invariant to frame cadence between 0.5x and 3.0x, so the 10% drift
# between the two formats changes nothing that gets read.
FRAME_MS = 33

WEBP_SIZE = 512
GIF_SIZE = 256

# WebP quality. Lossless is the wrong default here and the measurement is not
# close: on real generated artwork one character came out at 3353 KB lossless
# against 172 KB at quality 90, which is 39 MB versus 2.0 MB for a twelve
# character message. The design budgets 2-3 MB, so lossless misses it by more
# than an order of magnitude.
#
# The compression is free in the only currency that matters. Decode confidence
# was 34.5 either way -- identical, not merely adequate -- because the contour
# is a high-contrast edge and what WebP discards is the smooth glow around it.
# Quality 30 still passed at 35.1, so 90 is chosen to preserve the artwork
# rather than to protect the signal, which needs no protecting.
WEBP_QUALITY = 90

_DC = "http://purl.org/dc/elements/1.1/"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

_XMP_TEMPLATE = (
    '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
    f'<rdf:RDF xmlns:rdf="{_RDF}">'
    f'<rdf:Description rdf:about="" xmlns:dc="{_DC}">'
    '<dc:source>{url}</dc:source>'
    '</rdf:Description>'
    '</rdf:RDF>'
    '</x:xmpmeta>'
    '<?xpacket end="w"?>'
)


def xmp_packet(url: str) -> bytes:
    """Wrap a URL in a minimal, well-formed XMP packet.

    Pillow will happily store a bare string in the XMP slot, but nothing else
    will read it: XMP is defined as an RDF document, and a metadata tool that
    finds a bare URL there sees a corrupt packet. Wrapping costs a few hundred
    bytes per file and makes the link visible to anything that speaks XMP.
    """
    return _XMP_TEMPLATE.format(url=escape(url)).encode('utf-8')


def _url_from_xmp(packet: bytes) -> str | None:
    try:
        text = packet.decode('utf-8')
    except UnicodeDecodeError:
        return None
    # Trim the xpacket processing instructions, which are not part of the XML
    # document proper and make a strict parser refuse the whole packet.
    start, end = text.find('<x:xmpmeta'), text.rfind('</x:xmpmeta>')
    if start < 0 or end < 0:
        return None
    try:
        root = ElementTree.fromstring(text[start:end + len('</x:xmpmeta>')])
    except ElementTree.ParseError:
        return None
    node = root.find(f'.//{{{_DC}}}source')
    return node.text if node is not None else None


def _as_pil(frames: np.ndarray, size: int) -> list[Image.Image]:
    """Convert a stacked clip to RGB PIL frames at the target size."""
    array = np.asarray(frames)
    if array.ndim not in (3, 4):
        raise ValueError(
            f"Expected a stack of frames, got an array of shape {array.shape}"
        )
    if len(array) == 0:
        raise ValueError("Cannot write a clip with no frames")

    images = []
    for frame in array:
        image = Image.fromarray(frame.astype(np.uint8))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if image.size != (size, size):
            image = image.resize((size, size), Image.LANCZOS)
        images.append(image)
    return images


def read_link(path: str | Path) -> str | None:
    """Read the decoder URL back out of a saved file.

    Args:
        path: A WebP or GIF written by this module.

    Returns:
        str | None: The URL, or None if the file carries no readable link.
    """
    with Image.open(path) as image:
        if 'xmp' in image.info:
            return _url_from_xmp(image.info['xmp'])
        comment = image.info.get('comment')
        if isinstance(comment, bytes):
            return comment.decode('utf-8', errors='replace')
        return comment


def _verify_link(path: Path, url: str) -> None:
    recovered = read_link(path)
    if recovered != url:
        raise OSError(
            f"{path.name} did not keep its decoder link: wrote {url!r}, "
            f"read back {recovered!r}"
        )


def write_webp(
    frames: np.ndarray,
    path: str | Path,
    url: str = DECODER_URL,
    size: int = WEBP_SIZE,
    frame_ms: int = FRAME_MS,
    quality: int = WEBP_QUALITY,
) -> Path:
    """Write the canonical animated WebP, link included and confirmed.

    Args:
        frames: Frames stacked on a leading axis.
        path: Destination file.
        url: Decoder link to embed.
        size: Output edge in pixels.
        frame_ms: Per-frame delay.
        quality: WebP quality, 0-100. See WEBP_QUALITY for why this is lossy.

    Raises:
        OSError: If the link did not survive the write.
    """
    destination = Path(path)
    images = _as_pil(frames, size)
    images[0].save(
        destination, format='WEBP', save_all=True, append_images=images[1:],
        duration=frame_ms, loop=0, quality=quality, xmp=xmp_packet(url),
    )
    _verify_link(destination, url)
    return destination


def write_gif(
    frames: np.ndarray,
    path: str | Path,
    url: str = DECODER_URL,
    size: int = GIF_SIZE,
    frame_ms: int = FRAME_MS,
) -> Path:
    """Write the fallback animated GIF, link included and confirmed.

    Raises:
        OSError: If the link did not survive the write.
    """
    destination = Path(path)
    images = _as_pil(frames, size)
    images[0].save(
        destination, format='GIF', save_all=True, append_images=images[1:],
        duration=frame_ms, loop=0, optimize=False, comment=url.encode('utf-8'),
    )
    _verify_link(destination, url)
    return destination


def read_clip(path: str | Path, n_bins: int = C.N_BINS) -> np.ndarray:
    """Read every frame of a saved clip back as grayscale.

    The counterpart to the writers, and the reason they can be trusted: an
    asset is only verified if what comes out of the file decodes, not what went
    into it.

    Args:
        path: A WebP or GIF written by this module.
        n_bins: Unused; present so callers can pass decoder settings uniformly.

    Returns:
        np.ndarray: Frames stacked on a leading axis, grayscale uint8.
    """
    frames = []
    with Image.open(path) as image:
        for index in range(getattr(image, 'n_frames', 1)):
            image.seek(index)
            frames.append(np.array(image.convert('L')))
    return np.stack(frames)
