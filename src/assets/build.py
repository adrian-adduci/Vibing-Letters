"""Build the whole asset set: 43 characters, one sentinel, four candidates each.

This is the offline pipeline's entry point. It runs once, costs money, and
produces everything runtime will ever need, so it is written to be resumable and
to record what it did rather than to be fast.

Three properties matter more than speed:

* **Resumable.** A candidate already on disk that still passes the gate is not
  regenerated. A run interrupted halfway costs nothing to finish.
* **Self-reporting.** Every candidate's verdict goes into the manifest, failures
  included. A chord where nothing passed is a fact about the prompt, and it
  should be visible without rerunning anything.
* **Nothing unverified is ever selected.** The chosen candidate for a symbol is
  the highest-confidence one that passed the gate. If none passed, the symbol
  has no choice recorded and the manifest says so.

Generation is parallel because it is entirely network-bound; the gate and the
warp are not, and run inline where each candidate lands.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import cv2

from ..codec import constants as C
from ..codec.chord_table import CHORD_BY_SYMBOL
from . import stylize, verify

# fal is network-bound and tolerant of concurrency; this is well short of
# anything that has drawn a rate limit, and the run is minutes rather than hours.
WORKERS = 8

SEEDS = (1, 2, 3, 4)

SENTINEL_NAME = "SENTINEL"


def symbol_name(symbol: str) -> str:
    """A filesystem-safe name for a symbol.

    The alphabet includes a space, a comma, a full stop, an apostrophe and a
    question mark, none of which belong in a filename, and several of which are
    illegal on some platforms. Naming by code point is ugly and unambiguous,
    which is the right trade for a build artefact.
    """
    if symbol == ' ':
        return 'SPACE'
    if symbol.isalnum():
        return symbol
    return f"U{ord(symbol):04X}"


def targets() -> list[tuple[str, tuple[int, int]]]:
    """Every symbol to build, plus the sentinel, in a stable order."""
    entries = sorted(CHORD_BY_SYMBOL.items())
    return [(symbol_name(s), c) for s, c in entries] + [
        (SENTINEL_NAME, C.SENTINEL_CHORD)
    ]


class Built(NamedTuple):
    """Everything one symbol's build produced."""

    name: str
    chord: tuple[int, int]
    candidates: list[stylize.Candidate]

    @property
    def chosen(self) -> stylize.Candidate | None:
        """The best candidate that actually passed, or None if none did."""
        usable = [c for c in self.candidates if c.usable]
        if not usable:
            return None
        return max(usable, key=lambda c: c.verdict.peak.confidence)

    def as_record(self) -> dict:
        chosen = self.chosen
        return {
            "name": self.name,
            "chord": list(self.chord),
            "chosen": chosen.path.name if chosen else None,
            "confidence": (
                round(chosen.verdict.peak.confidence, 2) if chosen else None
            ),
            "candidates": [
                {
                    "file": c.path.name,
                    "seed": c.seed,
                    "accepted": c.verdict.accepted,
                    "read_as": (
                        list(c.verdict.peak.chord) if c.verdict.peak.chord else None
                    ),
                    "confidence": round(c.verdict.peak.confidence, 2),
                    "reasons": list(c.verdict.reasons),
                }
                for c in self.candidates
            ],
        }


def _existing(directory: Path, chord: tuple[int, int], seeds: Iterable[int]
              ) -> dict[int, stylize.Candidate]:
    """Candidates already on disk, re-judged rather than trusted.

    Re-running the gate costs a fraction of a second and means a resumed run
    cannot inherit a verdict from a version of the gate that no longer exists.
    """
    stem = f"{chord[0]:02d}_{chord[1]:02d}"
    found = {}
    for seed in seeds:
        path = directory / f"cand_{stem}_s{seed}.png"
        if not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        found[seed] = stylize.Candidate(path, seed, verify.accept_still(image, chord))
    return found


def build_symbol(
    name: str,
    chord: tuple[int, int],
    out_dir: Path,
    seeds: Iterable[int] = SEEDS,
    generate: Callable[..., list[stylize.Candidate]] = stylize.stylize,
) -> Built:
    """Generate and judge every candidate for one symbol, skipping any on disk."""
    seeds = tuple(seeds)
    have = _existing(out_dir, chord, seeds)
    missing = tuple(s for s in seeds if s not in have)

    if missing:
        for candidate in generate(chord, out_dir, seeds=missing):
            have[candidate.seed] = candidate

    return Built(name, chord, [have[s] for s in seeds if s in have])


def build_all(
    out_dir: str | Path,
    seeds: Iterable[int] = SEEDS,
    workers: int = WORKERS,
    generate: Callable[..., list[stylize.Candidate]] = stylize.stylize,
    on_done: Callable[[Built], None] | None = None,
) -> list[Built]:
    """Build every symbol, in parallel, and write the manifest.

    Args:
        out_dir: Directory for control images, candidates and the manifest.
        seeds: Seeds to generate per symbol.
        workers: Concurrent generations.
        generate: Generation call, injectable for testing.
        on_done: Called with each Built as it completes, for progress reporting.

    Returns:
        list[Built]: One per symbol, in the order `targets` gives.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    work = targets()

    def one(entry):
        result = build_symbol(*entry, directory, seeds=seeds, generate=generate)
        if on_done is not None:
            on_done(result)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, work))

    write_manifest(results, directory / "manifest.json")
    return results


def write_manifest(results: Iterable[Built], path: str | Path) -> Path:
    """Record what the build produced, failures included."""
    results = list(results)
    destination = Path(path)
    destination.write_text(json.dumps({
        "alphabet_size": len(CHORD_BY_SYMBOL),
        "frames_per_char": C.FRAMES_PER_CHAR,
        "min_confidence": C.MIN_CONFIDENCE,
        "complete": all(r.chosen is not None for r in results),
        "symbols": [r.as_record() for r in results],
    }, indent=2) + '\n')
    return destination


def report(results: Iterable[Built]) -> str:
    """A short human summary: what is ready, and what is not."""
    results = list(results)
    missing = [r for r in results if r.chosen is None]
    lines = [
        f"{len(results) - len(missing)}/{len(results)} symbols have a verified asset"
    ]
    if missing:
        lines.append("no usable candidate for:")
        lines += [f"  {r.name:<8} {r.chord}" for r in missing]
    else:
        worst = min(results, key=lambda r: r.chosen.verdict.peak.confidence)
        lines.append(
            f"lowest confidence: {worst.name} at "
            f"{worst.chosen.verdict.peak.confidence:.1f} "
            f"(gate is {C.MIN_CONFIDENCE})"
        )
    return '\n'.join(lines)
