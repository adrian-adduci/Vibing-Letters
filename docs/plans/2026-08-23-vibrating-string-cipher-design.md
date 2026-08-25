# Vibing Letters — Vibrating String Cipher

**Date:** 2026-08-23
**Status:** Design approved, not yet implemented
**Supersedes:** the letterform-morph direction described in `PLANS.md` and `README.md`

---

## Summary

Encode a sentence as a sequence of vibrating circular strings. Each character is a
closed loop that begins as a perfect circle, is "plucked" into a standing wave whose
mode content identifies the character, and decays back to a circle. The animation is
not decorative — the shape *is* the message, and a decoder recovers the original
sentence from the pixels alone.

A link back to the decoder is written into the file's metadata. Nothing is rendered
over the artwork.

## What changed from the original project

The original direction morphed a circle into a readable letter glyph. This design
drops legibility entirely: the ring never becomes a letter. The character is carried
by *how the string vibrates*, not by what shape it settles into.

This is a real trade. The output is unreadable without the decoder. In exchange, the
piece becomes an actual cipher rather than an illustration, and every message is
recoverable from the image itself with no external lookup.

---

## Decision record

| # | Decision | Chosen | Rationale |
|---|---|---|---|
| 1 | Legibility | Pure cipher; ring never becomes a glyph | Start and end frames are identical circles, so clips loop and splice seamlessly |
| 2 | Decode path | True recovery from pixels | The artwork carries the message; the link is only a pointer, not a payload |
| 3 | Alphabet | A–Z, 0–9, space, `. , ! ? ' -` (43 symbols) | Covers short messages, names, dates without needing case |
| 4 | Carrier | Chord of two modes from 2–12 | Coherent alphabet, generous slack, matches the string metaphor |
| 5 | Layout | Sequential in time, 0.5s per character | Full canvas per ring; maximum decode margin |
| 6 | Assets | Pre-generated, spliced at runtime | Moves all uncertainty offline behind a hard verification gate |
| 7 | Link | Metadata only, invisible | Artwork stays unmarked |
| 8 | Format | WebP canonical + GIF fallback | Quality and size vs. universal compatibility |
| 9 | Asset generation | One styled still per chord, warped through the envelope | No temporal flicker; 38→44 curated model calls total |
| 10 | Decoder | Client-side in browser | No upload, no server, message never leaves the device |
| 11 | Space | A chord like any other, `{2,3}` | Reverses an earlier choice; see below |

**Decision 11 was made during implementation, and reverses an earlier one.** Space was
originally the *absence* of excitation, recovered from the length of the quiet run
between characters. Building the codec showed that made space the only symbol
depending on frame count rather than image content — contradicting decision #2 — and a
platform resampling the timebase silently corrupted the text. Space became a chord;
the gap arithmetic and its constant were deleted entirely.

**Known cost of #7:** metadata is stripped by most platforms on re-encode, and a viewer
has no visual cue that a decoder exists. Accepted deliberately in favour of an
unmarked piece.

**Known cost of #5:** a screenshot captures one ring, i.e. one character. The full
message requires the full file.

---

## 1. The encoding

### Symbol space

Modes 2–12 give 11 modes and C(11,2) = **55 unordered pairs**. Of these:

- **43** are assigned to characters
- **1** is reserved as a start/end sentinel
- **11** are held spare for curation and future expansion

**Space is a character like any other**, and takes the calmest chord in the table.

An earlier draft made space the *absence* of excitation — a still circle, recovered
from the length of the quiet run between characters. That was reversed during
implementation, because it made space the only symbol depending on frame count rather
than on image content. Every other axis is invariant: rescaling, rotation, angular
resampling from 32 to 1024 bins. But a platform that resamples the timebase silently
turned `"A B"` into `" A   B "`, and frame-rate change is among the most common things
re-encoding does. That contradicted decision #2 — that the artwork itself carries the
message.

Space is also the most frequent character in English text, more frequent than `E`, so
under the table's own "frequent characters get the calmest rings" rule it earns the
lowest-sum pair. It is still visually the quietest ring: `{2,3}` is a slow two-lobe
wobble. The silence did not disappear from the piece either — every clip still opens
and closes on a perfect circle. It simply stopped being load-bearing for decoding.

Mode 1 is excluded by necessity, not preference: `R + A·cos(θ)` is not a deformation
but a *translation* of the circle. It would shift the centroid the decoder relies on
while producing no measurable shape change.

### Ring geometry

```
r(θ, t) = R + A · E(t) · cos(2πft) · [cos(n₁θ) + cos(n₂θ)] / 2
```

| Symbol | Meaning | Default |
|---|---|---|
| `R` | rest radius | 180 px (on 512² canvas) |
| `A` | peak amplitude | ~0.12 · R |
| `f` | oscillation rate | ~2 cycles per clip |
| `E(t)` | pluck envelope | fast attack, slow decay |

`E(t)` is pinned to **E(0) = E(T) = 0**. This single constraint does three jobs:

1. Every clip loops seamlessly.
2. Any clip can follow any other with no jump-cut.
3. The quiet circles between characters become **self-describing delimiters**.

Point 3 is the important one. The decoder locates character boundaries by finding
frames where radial variance falls to zero. It needs no frame counts, no fixed clip
length, and no metadata — so segmentation survives platform re-encoding.

### Canonical chord table

Pairs are ordered by `(n₁ + n₂, n₁)` ascending and assigned to characters in English
frequency order, so common letters get the calmest rings and typical messages read as
visually quieter. The sentinel takes `{2,12}` — maximum mode separation, so it can
never be confused with a character.

| | | | | | |
|---|---|---|---|---|---|
| `SPACE` {2,3} | `A` {3,4} | `B` {2,11} | `C` {3,8} | `D` {3,7} | `E` {2,4} |
| `F` {5,6} | `G` {4,8} | `H` {2,8} | `I` {3,5} | `J` {3,11} | `K` {4,9} |
| `L` {4,6} | `M` {4,7} | `N` {2,7} | `O` {2,6} | `P` {5,7} | `Q` {6,7} |
| `R` {4,5} | `S` {3,6} | `T` {2,5} | `U` {2,9} | `V` {3,10} | `W` {3,9} |
| `X` {5,8} | `Y` {2,10} | `Z` {4,10} | `0` {5,9} | `1` {6,8} | `2` {3,12} |
| `3` {4,11} | `4` {5,10} | `5` {6,9} | `6` {7,8} | `7` {4,12} | `8` {5,11} |
| `9` {6,10} | `!` {8,9} | `'` {6,11} | `,` {5,12} | `-` {7,10} | `.` {7,9} |
| `?` {6,12} | | | | | |

**Sentinel:** {2,12}
**Spare:** {7,11} {8,10} {7,12} {8,11} {9,10} {8,12} {9,11} {9,12} {10,11} {10,12} {11,12}

This table is data, not logic. It lives in a versioned file so curation can reassign
any character to a spare pair without touching code. Any change is a breaking change
to the format and must bump the codec version.

### Defaults

| Parameter | Value |
|---|---|
| Canvas | 512 × 512 (WebP), 256 × 256 (GIF fallback) |
| Frames per character | 15 |
| Frame duration | ~33 ms |
| Clip duration | 0.5 s |
| Angular sampling | 512 bins |

---

## 2. Asset pipeline (build time)

Runs once, offline. Produces 44 clips: 43 characters (space included) and 1 sentinel.

Per symbol:

1. **Compute** the exact peak contour for chord {n₁, n₂} — pure math, 512 points.
2. **Style** it. The image model renders that shape, conditioned on the contour
   (canny/depth), using `reference/` for the target aesthetic. Generate N candidates.
3. **Curate.** Pick the best by eye. The only manual step.
4. **Warp** the chosen still through the pluck envelope to produce all 15 frames.
5. **Verify.** FFT-decode every frame; reject the asset on failure.

### The warp

Every shape here is star-shaped about a known centre, so in polar coordinates the
deformation is a **per-angle radial scale**: a pixel at `(θ, ρ)` maps to
`ρ · r(θ,t) / r_peak(θ)`. One `cv2.remap` with a precomputed map. No mesh, no
thin-plate spline. Exact on the contour, smooth elsewhere, and the surrounding glow
scales along with the ring.

### Acceptance gate

The obvious formulation — "every frame must decode" — is wrong. Frames near
`E(t) → 0` are nearly perfect circles carrying no signal. The correct assertion is:

- The **maximum-excitation frame decodes to {n₁, n₂}**, and
- **no frame decodes to a different chord.**

Silence is acceptable. A wrong answer is not.

### Output

`assets/` plus a manifest recording symbol → chord → file hash → verification status.
Assets are committed. Building a message never requires regeneration or a model call.

---

## 3. Runtime composition

No image generation, no geometry, no model calls. A lookup and a concatenation.

1. **Normalize** — uppercase, validate against the 43-symbol alphabet.
   Unsupported characters are **stripped with an explicit report** of what was
   removed; `--strict` errors instead. (The current `string_builder.py:11` drops them
   silently, which breaks round-trip fidelity with no indication to the user.)
2. **Compose** — sentinel, then each character's 15 frames in order, then sentinel.
3. **Rotate** — each ring instance gets a rotation offset seeded from
   `hash(sentence + index)`. Varied across the message, but deterministic, so the
   same sentence always produces byte-identical files. Rotation cannot affect decoding:
   the FFT magnitude spectrum is rotation-invariant.
4. **Emit** — WebP at 512 px (canonical) and GIF at 256 px (fallback).
   256 px still gives ~47 px per lobe at mode 12, far above the decode threshold.
5. **Stamp** — decoder URL into WebP XMP and GIF Comment Extension, then **read back
   from the saved file and assert**. Pillow's metadata support varies by format and
   version; this must be verified, not assumed.

A 12-character message: 180 frames, roughly 2–3 MB WebP.

Every uncertain step was resolved offline against a hard gate. Runtime touches only
verified assets, so it cannot emit an undecodable file.

---

## 4. Decoder

Runs entirely client-side in the browser. Drag a file in; decode locally; nothing
uploaded.

**Per frame** — locate the ring, find its centroid, walk the contour, resample radius
as `r(θ)` into 512 uniform angular bins.

**Segment** — compute radial variance per frame. Runs of near-zero variance are the
neutral circles between characters.

**Per segment** — take the maximum-excitation frame, normalize `r(θ)` by its mean
radius, subtract the mean, FFT. The two largest magnitude peaks in bins 2–12 are
{n₁, n₂}. Look up the chord — space included, since it is a character like any other.
Sentinels at the ends are stripped.

A segment whose chord belongs to no character, or whose confidence falls below
`MIN_CONFIDENCE`, becomes a replacement character rather than raising, so one damaged
ring costs one character instead of the whole message. The confidence gate is
load-bearing: a *degenerate* ring excited in a single mode returns a valid-looking
pair — mode 5 comes back as {4,5}, meaning `R` — so a table lookup alone cannot see
anything wrong.

**Output** — the sentence, plus per-character confidence: the ratio of peak magnitude
to the largest non-peak bin.

Normalizing by mean radius gives scale invariance. Using magnitude and discarding
phase gives rotation invariance. Both fall out for free.

---

## 5. Verification

**The central test is a round-trip property test:** for randomly generated sentences,
`decode(encode(s)) == s`. Encoding and decoding are exact mathematical inverses, so
this exercises the whole message space rather than sampling it.

Supporting suites:

- **Chord table** — all 43 chords unique, all modes in 2–12, sentinel distinct from
  every character.
- **Asset gate** (build time) — peak frame decodes correctly; no frame decodes wrong.
- **Format parity** — WebP and GIF of the same sentence decode identically.
- **Robustness** — decode survives resize, rotation, re-compression, moderate crop.
- **Metadata** — URL written, read back from the saved file, asserted, per format.

### Measured tolerances

Numbers the asset pipeline should design against. Measured against the built
codec, not estimated.

| Distortion | Tolerance |
|---|---|
| Additive noise on the radius profile | **σ ≤ 0.03.** First failure 0.032; ~50% at 0.045; total loss at 0.060 |
| Out-of-band noise (modes above 12) | **Effectively unbounded** — correct decode at amplitude 1e12 |
| Angular resampling | **Fully invariant** — 32 to 1024 bins decode identically |
| Rescaling | **Fully invariant** — verified 0.25× to 100× |
| Rotation | **Fully invariant** — magnitude spectrum discards phase |
| Frame cadence | **Tolerant** — 0.5× to 3.0× resampling decodes correctly |

Three consequences worth carrying into asset work.

**Perlin texture is unconstrained in strength.** Section 6 below assumed
band-limited noise was merely *safe*. Measurement says the band limit is the
entire budget: out-of-band texture at roughly 1e13× the chord amplitude still
decodes correctly. Push the aesthetics as hard as the look wants — only the
*frequency* of the texture matters, never its depth.

**Failures announce themselves.** Below σ=0.045 every failure surfaces as the
replacement character rather than a wrong letter, and the dominant mode (76%) is
*fabrication* — noise lifting a single quiet frame over the threshold, which
then becomes its own segment. Silent corruption begins only at σ=0.045, and even
there it is a dropped character rather than a substituted one.

**Both axes are now largely free.** Contour extraction may sample angles at any
resolution, and frames resample from 0.5× to 3.0× without error. This was not
true of the earlier design, where space was recovered from gap length: at 2×
frame rate `"A B"` decoded as `" A   B "`, silently. Making space a chord
removed that dependency.

One frame-count dependency survives, in the decoder rather than in the wire.
`close_short_gaps` closes quiet runs shorter than `MIN_CLOSABLE_GAP = 3` — the
one-frame patch where the standing wave crosses zero mid-character. Duplicate
frames at 3× or more by nearest-neighbour and that patch outgrows the threshold,
splitting every character into two segments that each decode: `"X  Y"` becomes
`"�XX    YY�"`. The constant scales with cadence if it ever needs to, so this is
a tuning limit rather than information lost.

---

## 6. Existing code disposition

**Reused:** `contour_extractor.py`, `easing_curve.py` (the pluck envelope),
`gif_builder.py`, `validators.py`, `logger.py`.

**Replaced:** `morph_engine.py`'s general morphing gives way to the polar radial remap.

**Constrained:** `perlin_vibrator.py`'s role inverts. Perlin noise was the aesthetic
centrepiece; here it is *adversarial* — arbitrary contour noise injects spurious FFT
energy that can outvote a real mode peak. It remains usable only if **band-limited
above bin 12**, confined to spatial frequencies the decoder ignores.

**Retired:** the letterform-morph path — `vibing_letter_generator.py`, `collapse_O.py`,
and the CLI drivers that depend on it.

**Kept as reference:** `input/*.gif` moves to `reference/`. See below.

---

## 7. Repository findings

Two discoveries during design that correct the existing documentation.

### The A–Z GIFs are already vibrating rings, not glyph morphs

`PLANS.md` and `README.md` describe circle-to-letter morphing. The assets do not do
this. `A.gif` opens on a rough black circle and develops ~28 smooth lobes; `O.gif`
develops ~25 sharp spikes. Neither ever becomes a letter. They are already vibrating
circular strings on an iridescent holographic ground.

**Specs:** 1024 × 1024, 9 frames, 0.32 s (already inside the 0.5 s budget), ~5.3 MB
each. `_B`–`_F` are 512 × 512, 3-frame tests on flat grey.

They cannot ship as-is — their lobe counts are arbitrary and encode nothing, and
`A`'s ~28 lobes sit far outside the 2–12 range. But they define the target aesthetic
precisely, so they are **kept in `reference/`** to condition the image model.

That aesthetic is fortunate: a hard black line on a bright ground is the
highest-contrast contour available, so the decoder can threshold on luminance rather
than tune edge detection. A soft glowing ring on a dark field would have made
sub-pixel contour localization the hardest part of the project.

Note the iridescent background is baked in, so WebP's alpha channel buys nothing
unless transparent-ground variants are wanted later.

### The `src/morphing/` refactor has never run

`vibing_letter_generator.py:15` and `batch_generate.py:35` require `_Static_O.png`
and `clean_background.png`. Neither was ever committed; no `.png` exists in the repo
or anywhere in its history. `README.md:49` also references `input/A.png`, but `input/`
contains only `.gif`.

The 6,661 lines added in `32c4f19` are therefore unexercised end-to-end. The three
modules with unit tests — `contour_extractor`, `easing_curve`, `validators` — are the
trustworthy core; the rest is unproven and should be treated as draft when reused.

---

## 8. Open questions

- Which image model, and via what call path? No image-generation MCP is connected in
  the current session.
- Where is the decoder hosted, and what is the final URL?
- Is a transparent-background variant wanted, given the reference aesthetic bakes in
  its ground?
- Should the sentinel also encode a codec version, so future table changes stay
  decodable?
