"""Turn exact line art into artwork, without letting it stop being exact.

This is the one stage that calls out to a model, and the one stage whose output
nobody can predict. Everything it produces is therefore run straight through the
acceptance gate before a human ever looks at it, so curation is a choice between
candidates that are already known to decode -- not a choice that might silently
introduce one that does not.

Two findings from probing the endpoint, both of which simplify this a lot:

* **Control image polarity does not matter.** Black line on white and white line
  on black produced pixel-identical output. Canny does not care which side of an
  edge is dark, so both preprocess to the same edge map.
* **The `preprocess_depth` flag does not matter either.** Its description is
  copied verbatim from the depth variant of the same endpoint, and toggling it
  changed nothing: all four combinations of polarity and flag returned the same
  image to the pixel. The endpoint re-detects edges from the supplied image
  regardless. That would be a problem if it doubled a thick stroke into two
  contours, but at the widths used here the doubling is sub-pixel and adherence
  measured 22.93 confidence against a gate of 5.0.

So the control image goes up as drawn, the flag is left at its default, and
neither is a knob worth exposing.
"""

import os
import urllib.request
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import cv2
import numpy as np

from ..codec import constants as C
from . import verify
from .contour import render_control_image

ENDPOINT = "fal-ai/flux-control-lora-canny"

# The aesthetic the reference GIFs in input/ already establish: a single bright
# filament against black, glowing, with no lettering anywhere near it.
PROMPT = (
    "a single glowing filament of light bent into a closed wavy ring, "
    "luminous cyan and magenta energy, thin bright line, deep black background, "
    "long exposure light painting, sharp, high detail, no text"
)

# 1024 px is where the endpoint is priced by the megapixel and where a mode-12
# lobe is still ~130 px of circumference. Assets are rendered here and
# downsampled at emit time, never the other way round.
SIZE = 1024

CONTROL_STRENGTH = 1.0
STEPS = 28
GUIDANCE = 3.5


class Candidate(NamedTuple):
    """One generated still and what the gate made of it."""

    path: Path
    seed: int
    verdict: verify.Verdict

    @property
    def usable(self) -> bool:
        return self.verdict.accepted


def load_key(start: Path | None = None) -> str:
    """Find and load FAL_KEY, searching upward for a .env.

    Worktrees do not share the repository root's untracked files, so a key
    written once at the top of the checkout has to be found from wherever this
    runs rather than assumed to sit alongside it.

    Args:
        start: Directory to search upward from; this file's own directory by
            default.

    Returns:
        str: The key.

    Raises:
        RuntimeError: If no key can be found.
    """
    if os.environ.get('FAL_KEY'):
        return os.environ['FAL_KEY']

    here = (start or Path(__file__).resolve().parent)
    for directory in [here, *here.parents]:
        candidate = directory / '.env'
        if candidate.is_file():
            from dotenv import load_dotenv
            load_dotenv(candidate)
            if os.environ.get('FAL_KEY'):
                return os.environ['FAL_KEY']

    raise RuntimeError(
        "FAL_KEY is not set and no .env containing it was found above "
        f"{here}. Write it to a .env at the repository root."
    )


def _submit(endpoint: str, arguments: dict) -> list[str]:
    """Run one generation and return the URLs it produced.

    Isolated so that everything above it can be tested without a network or an
    account, which matters because this is the only paid call in the project.
    """
    import fal_client

    load_key()
    result = fal_client.subscribe(endpoint, arguments=arguments)
    return [image["url"] for image in result["images"]]


def _fetch(url: str, destination: Path) -> Path:
    urllib.request.urlretrieve(url, destination)
    return destination


def stylize(
    chord: tuple[int, int],
    out_dir: str | Path,
    seeds: Sequence[int] = (1, 2, 3, 4),
    prompt: str = PROMPT,
    size: int = SIZE,
    endpoint: str = ENDPOINT,
    submit: Callable[[str, dict], list[str]] = _submit,
    fetch: Callable[[str, Path], Path] = _fetch,
    upload: Callable[[Path], str] | None = None,
) -> list[Candidate]:
    """Generate styled candidates for one chord and judge every one.

    Args:
        chord: The chord to render.
        out_dir: Directory for the control image and the candidates.
        seeds: One generation per seed. Distinct seeds are what produce variety;
            the same seed returns the same picture.
        prompt: Style description.
        size: Render size in pixels, square.
        endpoint: fal model id.
        submit: Generation call, injectable for testing.
        fetch: Download call, injectable for testing.
        upload: Control-image upload, injectable for testing.

    Returns:
        list[Candidate]: One per seed, in seed order, each carrying the gate's
        verdict. Failures are returned rather than dropped: a chord where every
        candidate fails is a signal about the prompt, and silently returning an
        empty list would hide it.

    Raises:
        ValueError: If no seeds are given.
    """
    if not seeds:
        raise ValueError("Generating candidates needs at least one seed")

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{chord[0]:02d}_{chord[1]:02d}"

    control_path = directory / f"control_{stem}.png"
    cv2.imwrite(str(control_path), render_control_image(chord, size=size))

    control_url = (upload or upload_file)(control_path)

    candidates = []
    for seed in seeds:
        urls = submit(endpoint, {
            "prompt": prompt,
            "control_lora_image_url": control_url,
            "control_lora_strength": CONTROL_STRENGTH,
            "image_size": {"width": size, "height": size},
            "num_inference_steps": STEPS,
            "guidance_scale": GUIDANCE,
            "seed": seed,
            "num_images": 1,
            "output_format": "png",
        })
        path = fetch(urls[0], directory / f"cand_{stem}_s{seed}.png")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Generated candidate at {path} could not be read back")
        candidates.append(Candidate(path, seed, verify.accept_still(image, chord)))

    return candidates


def upload_file(path: Path) -> str:
    """Put a local file where the endpoint can read it."""
    import fal_client

    load_key()
    return fal_client.upload_file(str(path))


def summarize(candidates: Sequence[Candidate]) -> str:
    """One line per candidate, for the curator to read before looking at any.

    The gate has already decided which are usable; this says why the rest are
    not, so a run where everything failed reads as a prompt problem rather than
    as a mystery.
    """
    lines = []
    for candidate in candidates:
        verdict = candidate.verdict
        if verdict.accepted:
            detail = f"chord {verdict.peak.chord} at {verdict.peak.confidence:.1f}"
        else:
            detail = verdict.reasons[0] if verdict.reasons else "rejected"
        lines.append(
            f"  seed {candidate.seed:<5} "
            f"{'OK  ' if verdict.accepted else 'FAIL'}  "
            f"{candidate.path.name:<24} {detail}"
        )
    usable = sum(c.usable for c in candidates)
    lines.append(f"  {usable}/{len(candidates)} usable")
    return '\n'.join(lines)
