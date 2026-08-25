# Handoff

**State:** the cipher works end to end. A sentence becomes an animated file, and
that file decodes back to the sentence from its own pixels. All 44 assets are
generated, verified and committed.

**Branch:** `main` at `9b5add2`. Nothing outstanding, no open PRs, no worktrees.

**Tests:** 634 passing.

---

## Setup

Python 3.13. A virtualenv is already at `.venv/`; if it is gone:

```bash
uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q          # expect 634 passed
```

Every `python` below means `.venv/bin/python`, from the repository root.

---

## Try it in one minute

```bash
.venv/bin/python -c "
from src import compose
clips = compose.load_clips('assets/clips')
compose.write('HELLO WORLD', 'out.webp', clips)
"
```

Then read it back:

```bash
.venv/bin/python -c "
import numpy as np
from src.assets import emit
from src.codec.message import decode
from src.vision import ring
frames = emit.read_clip('out.webp')
print(decode(np.stack([ring.radius_profile(f) for f in frames])))
"
```

`output/demo/hello.webp` is the same thing, already built.

---

## What the thing is

Each character is a closed string standing wave. The character is identified by
*which two modes are ringing* — an unordered pair drawn from modes 2 to 12, which
gives C(11,2) = 55 pairs, spent as 43 characters, 1 sentinel and 11 spare.

Decoding is an angular FFT of the ring's radius profile. Normalizing by mean
radius gives scale invariance; taking magnitude and discarding phase gives
rotation invariance. Both fall out of the transform rather than being engineered.

Mode 1 is excluded by necessity, not preference: `R + A·cos(θ)` is a circle
translated sideways, which moves the centroid the decoder depends on while
changing no shape. That same fact is used constructively in the extractor — see
`src/vision/ring.py`.

The full rationale, with the costs of each decision, is in
`docs/plans/2026-08-23-vibrating-string-cipher-design.md`. It is the design of
record. Read it before changing anything in `src/codec/constants.py`.

---

## Layout

| path | role |
|---|---|
| `src/codec/` | the cipher itself, pure maths on arrays. No pixels. |
| `src/vision/ring.py` | pixels → `r(θ)`. The bridge. **The browser decoder must mirror this.** |
| `src/assets/contour.py` | chord → exact peak contour → line-art control image |
| `src/assets/stylize.py` | fal generation; every candidate pre-judged |
| `src/assets/warp.py` | one still → all 15 frames, via one `cv2.remap` |
| `src/assets/verify.py` | the acceptance gate |
| `src/assets/emit.py` | WebP 512 / GIF 256, with the decoder link |
| `src/assets/build.py` | generate the whole set: parallel, resumable |
| `src/assets/publish.py` | chosen stills → shippable clips |
| `src/compose.py` | the entire runtime: a lookup and a concatenation |
| `assets/clips/` | 44 verified clips, 8.4 MB, committed |

---

## Where things stand

**Assets.** 44/44 published and verified *from the written file* rather than from
the array that produced it. 192 KB average. Confidence min 12.1, median 35.4,
max 99.1, against a gate of 5.0.

**Generation.** 176 candidates, 176 accepted at still level, 12.5 minutes,
$7.04 on fal. Total project spend $7.32. `assets/candidates-manifest.json` is
the committed record of what was generated and how each candidate scored.

**The raw candidate stills no longer exist.** They lived in `assets/candidates/`,
which is gitignored, inside a git worktree that was removed after merging —
`git worktree remove --force` takes ignored files with it. Roughly 180 MB and
$7.04 of generated artwork, gone.

This turned out not to matter, and the reason is worth knowing: **a clip's
loudest frame is its still.** Warping to the amplitude a still already has is the
identity, so the peak frame of a published clip is the curated picture, at output
resolution and one compression generation older. `publish.stills_from_clips`
recovers the whole set from `assets/clips/`, which makes the shipped clips
self-sufficient — the alphabet can be republished at a different size, quality or
decoder URL without candidates, without a network, and without spending anything.
Verified over all 44: still 44/44, lowest confidence moved 12.1 → 11.8.

It is not free. Each recovery round costs one more compression generation and the
output shrinks (122 KB average from recovered stills against 192 KB from the
1024 px originals). So it is a recovery path, not the normal one. **If you
regenerate, keep `assets/candidates/` somewhere outside the repo** before
removing any worktree.

**Verified round trips through real artwork:** the pangram, the full alphabet,
`0123456789`, `HELLO WORLD`, `MEET ME AT 8PM!`, `  IT'S 42, ISN'T IT?  `,
`8PM`, `BIG GB 88`.

---

## Next, in order

1. **Write the browser decoder.** This is the only thing standing between the
   project and being usable by anyone else. It is a direct port of
   `src/vision/ring.py` plus `src/codec/spectrum.py` — locate the ring, solve for
   its centre, walk rays, FFT, look up the chord. Runs entirely client-side;
   nothing is uploaded. Everything it needs to reproduce is already covered by
   tests you can port alongside it.

2. **Settle `emit.DECODER_URL`.** Currently
   `https://adrian-adduci.github.io/Vibing-Letters/`, a placeholder.

   This is a one-line change and nothing else is required. A composed message
   takes its link from `compose.write`, which defaults to `emit.DECODER_URL`, so
   editing that constant is enough — the URLs embedded in the individual clips
   are never read or propagated by runtime, which uses clips only for their
   frames.

   Republishing the clips to carry the new URL is optional tidiness. If you do
   want it, it costs nothing and needs no network, because the stills come back
   out of the clips themselves:

   ```bash
   .venv/bin/python -c "
   from src.assets import publish
   stills = publish.stills_from_clips('assets/clips', '/tmp/recovered-stills')
   print(publish.report(publish.publish_all(stills, 'assets/clips')))
   "
   ```

   Do that at most once — every pass costs a compression generation. If
   `assets/candidates/` has been regenerated and still exists, use
   `ranked_from_manifest` + `publish_best` against it instead: that path starts
   from the 1024 px originals and loses nothing.

3. **Delete or rewrite `README.md` and `PLANS.md`.** Both describe the retired
   letterform-morphing project — Procrustes alignment, Perlin vibration, 31
   easing types. None of that is what this repo does, and §7 of the design doc
   establishes it never did: the A–Z GIFs in `input/` were always vibrating
   rings, not glyph morphs. Anyone picking this up from the README will be
   misled. The legacy drivers at the repository root (`vibing_letter_generator.py`,
   `collapse_O.py`, `build_string.py`, `string_builder.py`, `generate_letter.py`,
   `batch_generate.py`) are retired by the design and can go with them.

4. **Decide the remaining open questions** in §8 of the design doc: whether a
   transparent-background variant is wanted, and whether the sentinel should also
   carry a codec version so future table changes stay decodable.

---

## Things that will bite you

**Frame counts are not preserved, by design of the encoders.** Both WebP and GIF
merge runs of byte-identical consecutive frames and neither offers a way to stop
it. A clip ships as 12 frames, not `FRAMES_PER_CHAR` (15), because the envelope is
pinned to zero at both ends — the last active frame is already a rest circle, so
four identical circles collapse to one. **Never index a composed message by a
fixed 15-frame stride.** This is safe for decoding: boundary quiet runs come back
at 7 frames and mid-character zero crossings at 1, against a `MIN_CLOSABLE_GAP`
of 3. Three-way separation, not a margin.

**A still passing the gate is not evidence its clip will.** A still is all peak
and has no rest state, so still-level evidence is structurally incapable of
seeing the worst defect. Use `verify.accept` on a clip, not `verify.accept_still`,
for anything that ships.

**Doubled contours are the failure mode to watch.** The fal endpoint re-runs
canny on whatever it is given, so it sometimes renders the ring as two parallel
filaments. That breaks the single-valued `r(θ)` the warp assumes, the flattened
frames are not flat, the clip never falls silent, and *the next character merges
into it and disappears* — with every individual frame perfectly legal. This cost
a lost `P` in `MEET ME AT 8PM!` before the gate learned to require silence. If you
regenerate assets, keep that rule and keep the fall-through in `publish_best`.

**`preprocess_depth` and control-image polarity do nothing.** All four
combinations returned pixel-identical output. Do not spend money rediscovering
this; it is documented at the top of `src/assets/stylize.py`.

**`FAL_KEY` lives in `.env` at the repository root** (gitignored). `stylize.load_key`
searches upward for it, so it works from a worktree. An `export` in your shell
does *not* reach a subprocess spawned from a fresh profile.

**Guards are mutation-checked.** Every constant and correction sign in the
extractor, every branch of the gate, the warp arithmetic, and the build's
selection logic have been verified to turn a test red when broken. If you change
one and nothing fails, the test is wrong, not the change.

---

## Regenerating assets

Only needed if you change the aesthetic or the prompt — **not** to change the
decoder URL, which step 2 handles for free. Costs about $7 and 13 minutes.
Resumable: candidates already on disk are reused, though they are re-judged
rather than trusted, so a stale verdict cannot survive a change to the gate.

Needs `FAL_KEY` in `.env` at the repository root.

```bash
.venv/bin/python -c "
from src.assets import build
print(build.report(build.build_all('assets/candidates')))
"
```

Then publish from the fresh candidates, which starts at 1024 px and loses
nothing:

```bash
.venv/bin/python -c "
from pathlib import Path
from src.assets import publish
ranked = publish.ranked_from_manifest('assets/candidates/manifest.json')
out = Path('assets/clips')
results = [publish.publish_best(n, c, p, out) for n, c, p in ranked]
publish.write_manifest(results, out / 'assets.json')
print(publish.report(results))
"
cp assets/candidates/manifest.json assets/candidates-manifest.json
```

Use `publish_best`, not `publish_all`: it falls through candidates until a
*clip* passes, which is the only thing that catches a doubled contour. And copy
`assets/candidates/` somewhere outside the repository before you delete any
worktree — see the note above about how the first set was lost.
