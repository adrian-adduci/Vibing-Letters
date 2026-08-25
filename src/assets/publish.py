"""Turn chosen candidates into the clips runtime actually ships.

The build picks a still per symbol; this warps it, writes it, and then reads it
back out of the file and gates it again. That second gate is the point. Every
earlier check ran on arrays in memory, and the artefact that ships is a
compressed file that has been through a resize, a quality pass and frame
merging. Verifying the array and shipping the file would be verifying something
else.

Assets that fail are not written. A symbol with no publishable clip is recorded
as missing, because a runtime that finds a file assumes it decodes.
"""

import hashlib
import json
from pathlib import Path
from typing import Iterable, NamedTuple

import cv2

from ..codec import constants as C
from . import emit, verify, warp


class Published(NamedTuple):
    """One symbol's shippable clip, or the reason there isn't one."""

    name: str
    chord: tuple[int, int]
    path: Path | None
    digest: str | None
    confidence: float
    reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.path is not None

    def as_record(self) -> dict:
        return {
            "name": self.name,
            "chord": list(self.chord),
            "clip": self.path.name if self.path else None,
            "sha256": self.digest,
            "confidence": round(self.confidence, 2),
            "verified": self.ok,
            "reasons": list(self.reasons),
        }


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_one(
    name: str,
    chord: tuple[int, int],
    still_path: Path,
    out_dir: Path,
    url: str = emit.DECODER_URL,
    size: int = emit.WEBP_SIZE,
    quality: int = emit.WEBP_QUALITY,
) -> Published:
    """Warp one chosen still into a clip, write it, and verify the written file.

    Args:
        name: Filesystem-safe symbol name.
        chord: The chord this symbol carries.
        still_path: The curated peak-excitation image.
        out_dir: Where the clip goes.
        url: Decoder link to embed.
        size: Output edge in pixels.
        quality: WebP quality.

    Returns:
        Published: With a path and digest if the written file passed the gate,
        and with the gate's reasons if it did not. A rejected clip is deleted
        rather than left on disk, because runtime treats the presence of a file
        as a promise that it decodes.
    """
    still = cv2.imread(str(still_path), cv2.IMREAD_COLOR)
    if still is None:
        return Published(name, chord, None, None, 0.0,
                         (f"still {still_path.name} could not be read",))

    try:
        clip = warp.envelope_clip(still)
    except ValueError as error:
        return Published(name, chord, None, None, 0.0, (str(error),))

    destination = out_dir / f"{name}.webp"
    emit.write_webp(clip, destination, url=url, size=size, quality=quality)

    # The gate that counts: run on what came back out of the file, not on the
    # array that went in.
    verdict = verify.accept(emit.read_clip(destination), chord)
    if not verdict.accepted:
        destination.unlink()
        return Published(name, chord, None, None,
                         verdict.peak.confidence, verdict.reasons)

    return Published(name, chord, destination, digest_of(destination),
                     verdict.peak.confidence, ())


def publish_best(
    name: str,
    chord: tuple[int, int],
    still_paths: Iterable[Path],
    out_dir: Path,
    url: str = emit.DECODER_URL,
) -> Published:
    """Try candidates in order and keep the first whose *clip* passes.

    A still passing the gate is not evidence its clip will. The still-level
    check has no rest state to look at, so it cannot see the one defect that
    matters most here: a ring the model drew as two parallel filaments breaks
    the single-valued r(theta) the warp assumes, and the clip never returns to
    silence. Selecting on the still and publishing the clip would pick such a
    candidate whenever it happened to be the most confident one.

    Args:
        name: Filesystem-safe symbol name.
        chord: The chord this symbol carries.
        still_paths: Candidates, best first.
        out_dir: Where the clip goes.
        url: Decoder link to embed.

    Returns:
        Published: The first success, or the last failure if none succeeded.
    """
    last = None
    for still_path in still_paths:
        last = publish_one(name, chord, Path(still_path), out_dir, url=url)
        if last.ok:
            return last
    if last is None:
        return Published(name, chord, None, None, 0.0, ("no candidates offered",))
    return last


def chosen_from_manifest(
    manifest_path: str | Path,
    candidates_dir: str | Path | None = None,
) -> list[tuple[str, tuple[int, int], Path]]:
    """Read the build's manifest and return what it selected.

    Symbols the build could not satisfy are skipped rather than raising, so a
    partial build still publishes what it has; `publish_all` will report the
    shortfall in its own manifest.

    Args:
        manifest_path: The build's manifest.json.
        candidates_dir: Where the candidate files live; the manifest's own
            directory by default.

    Returns:
        list: (name, chord, still path) triples ready for `publish_all`.
    """
    path = Path(manifest_path)
    directory = Path(candidates_dir) if candidates_dir else path.parent
    manifest = json.loads(path.read_text())
    return [
        (entry["name"], tuple(entry["chord"]), directory / entry["chosen"])
        for entry in manifest["symbols"]
        if entry["chosen"]
    ]


def ranked_from_manifest(
    manifest_path: str | Path,
    candidates_dir: str | Path | None = None,
) -> list[tuple[str, tuple[int, int], list[Path]]]:
    """Every accepted candidate per symbol, most confident first.

    `chosen_from_manifest` returns only the build's single pick, which is chosen
    on still-level evidence alone. This returns the whole shortlist so
    `publish_best` can fall through to the next one when a clip fails.
    """
    path = Path(manifest_path)
    directory = Path(candidates_dir) if candidates_dir else path.parent
    manifest = json.loads(path.read_text())

    ranked = []
    for entry in manifest["symbols"]:
        usable = sorted(
            (c for c in entry["candidates"] if c["accepted"]),
            key=lambda c: c["confidence"], reverse=True,
        )
        ranked.append((entry["name"], tuple(entry["chord"]),
                       [directory / c["file"] for c in usable]))
    return ranked


def publish_all(
    chosen: Iterable[tuple[str, tuple[int, int], Path]],
    out_dir: str | Path,
    url: str = emit.DECODER_URL,
) -> list[Published]:
    """Publish every symbol that has a chosen still.

    Args:
        chosen: (name, chord, still path) triples.
        out_dir: Directory for the clips and the asset manifest.
        url: Decoder link to embed.

    Returns:
        list[Published]: One per input, in order.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results = [publish_one(name, chord, Path(still), directory, url)
               for name, chord, still in chosen]
    write_manifest(results, directory / "assets.json", url)
    return results


def write_manifest(results: Iterable[Published], path: str | Path,
                   url: str = emit.DECODER_URL) -> Path:
    """Record what shipped, with hashes, so a later change is visible."""
    results = list(results)
    destination = Path(path)
    destination.write_text(json.dumps({
        "decoder_url": url,
        "frames_per_char": C.FRAMES_PER_CHAR,
        "min_confidence": C.MIN_CONFIDENCE,
        "complete": all(r.ok for r in results),
        "symbols": [r.as_record() for r in results],
    }, indent=2) + '\n')
    return destination


def report(results: Iterable[Published]) -> str:
    results = list(results)
    good = [r for r in results if r.ok]
    lines = [f"{len(good)}/{len(results)} clips published and verified from file"]
    for bad in (r for r in results if not r.ok):
        lines.append(f"  FAILED {bad.name:<8} {bad.chord}  "
                     f"{bad.reasons[0] if bad.reasons else 'rejected'}")
    if good:
        total = sum(r.path.stat().st_size for r in good) / 1024
        worst = min(good, key=lambda r: r.confidence)
        lines.append(f"  {total:.0f} KB total, {total / len(good):.0f} KB average")
        lines.append(f"  lowest confidence: {worst.name} at {worst.confidence:.1f} "
                     f"(gate is {C.MIN_CONFIDENCE})")
    return '\n'.join(lines)
