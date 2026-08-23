# Vibrating String Codec Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the pure-math codec that turns a sentence into a sequence of radius
profiles and recovers the sentence back from them, with no image handling at all.

**Architecture:** Four small modules under `src/codec/`. `chord_table` owns the
symbol↔chord mapping and text normalization. `waveform` turns a chord into radius
profiles over time. `spectrum` recovers a chord from a single profile via FFT.
`message` composes and segments whole sentences. The boundary of this work is
`sentence ↔ np.ndarray of shape (n_frames, n_bins)`. Pixels, contours, and asset
generation are explicitly out of scope — the codec must be provably correct before
any of that exists.

**Tech Stack:** Python 3.13, numpy, pytest. No OpenCV, no Pillow, no image I/O.

**Design reference:** `docs/plans/2026-08-23-vibrating-string-cipher-design.md`,
sections 1, 4, 5.

**Baseline:** worktree `.worktrees/feat-codec`, branch `feat/codec`, 80 tests passing.
Run tests with `.venv/bin/python -m pytest`.

---

## Design refinements discovered during planning

Three corrections to design sections 1 and 4. All were validated by prototype before
this plan was written; the constants below are measured, not guessed.

**1. The quiet-frame threshold must be relative, not absolute.**
Design section 4 says a segment is a space when variance "never clears the noise
floor." Measured: with amplitude 0.12, the spectral peak of an excited frame is only
`0.06 × |frame_amplitude|`. A fixed threshold of 0.01 marks only 4 of 12 active frames
as excited. The threshold is therefore defined in **radial-modulation units** — a
frame is active when its strongest mode exceeds 1% radial modulation — which is
scale-free and directly interpretable.

**2. A character splits into two active runs, and the segmenter must close the gap.**
The `cos(2πft)` term passes through zero *mid-character*, so each character produces
two runs of active frames separated by one quiet frame. Naive run-splitting decodes
every character twice. Fix: close any quiet run shorter than `MIN_CLOSABLE_GAP = 3`
before segmenting. Measured margin — interior quiet runs are 1 frame, boundary runs
are 10 or more. Comfortable.

**3. Spaces cannot be recovered as segments; they are recovered from gap length.**
A space is a still circle, so it produces *no* active run at all — there is nothing to
count. Spaces are instead read from the length of the quiet run:

```
spaces = round((run_length - BOUNDARY_GAP_FRAMES) / FRAMES_PER_CHAR)
```

Measured: a plain character boundary is 10 frames; each space adds 15. Runs of 10, 25,
40 map to 0, 1, 2 spaces — cleanly separable.

> **Note for later:** this is the least robust part of the codec, because it depends on
> frame counts surviving intact rather than on the image content. Twelve spare chords
> exist; assigning one to space would make it a segment like any other and remove the
> arithmetic entirely, at the cost of the "silence is a still circle" idea. Flagged,
> not decided.

---

## Constants

All live in `src/codec/constants.py`. Every value below was verified by prototype.

| Constant | Value | Meaning |
|---|---|---|
| `MIN_MODE` | 2 | Lowest usable mode (mode 1 is translation, not deformation) |
| `MAX_MODE` | 12 | Highest mode; C(11,2)=55 pairs |
| `SENTINEL_CHORD` | `(2, 12)` | Start/end marker; maximum mode separation |
| `N_BINS` | 512 | Angular samples per profile |
| `AMPLITUDE` | 0.12 | Peak radial modulation as a fraction of rest radius |
| `REST_RADIUS` | 1.0 | Codec works in normalized radius units |
| `ACTIVE_FRAMES` | 12 | Frames carrying the pluck |
| `TRAILING_SILENCE_FRAMES` | 3 | Trailing silent frames per character |
| `FRAMES_PER_CHAR` | 15 | `ACTIVE_FRAMES + TRAILING_SILENCE_FRAMES` |
| `OSCILLATIONS` | 2 | Cycles of the standing wave per clip |
| `ATTACK` | 0.25 | Fraction of the clip spent rising |
| `DECAY_POWER` | 2.0 | Decay curve exponent |
| `QUIET_THRESHOLD` | 0.01 | Active if strongest mode exceeds 1% radial modulation |
| `MIN_CLOSABLE_GAP` | 3 | Quiet runs shorter than this are closed |
| `BOUNDARY_GAP_FRAMES` | 10 | Quiet-run length at a plain character boundary |

---

### Task 1: Package scaffold and constants

> **Status: complete.** Implemented in `6beb77d`, revised after review in `563aa5c`.
> The authoritative content is `src/codec/constants.py` and
> `tests/codec/test_constants.py` — read those, not the blocks below, which are
> kept only to show what was originally specified. Review added three renames
> (`TRAILING_SILENCE_FRAMES`, `MIN_CLOSABLE_GAP`, `BOUNDARY_GAP_FRAMES`), split
> the constants into wire-format / render-parameter / decoder-tuning groups, and
> added five relational invariant tests. 88 tests pass.

**Files:**
- Create: `src/codec/__init__.py`
- Create: `src/codec/constants.py`
- Test: `tests/codec/__init__.py`, `tests/codec/test_constants.py`

**Step 1: Write the failing test**

```python
# tests/codec/test_constants.py
"""Tests for codec constants."""
from src.codec import constants as C


def test_mode_range_excludes_translation_mode():
    """Mode 1 is a rigid translation, not a deformation, so it must be excluded."""
    assert C.MIN_MODE == 2
    assert C.MAX_MODE == 12


def test_frames_per_char_is_active_plus_gap():
    assert C.FRAMES_PER_CHAR == C.ACTIVE_FRAMES + C.TRAILING_SILENCE_FRAMES


def test_sentinel_uses_extreme_modes():
    """The sentinel must be maximally separated so it is never confused."""
    assert C.SENTINEL_CHORD == (C.MIN_MODE, C.MAX_MODE)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/codec/test_constants.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.codec'`

**Step 3: Write minimal implementation**

Create `src/codec/__init__.py`:

```python
"""Vibrating string cipher codec.

Encodes text as sequences of radius profiles and recovers it again. This
package handles no images; it operates purely on numpy arrays of radii.
"""
```

Create `tests/codec/__init__.py` (empty file).

Create `src/codec/constants.py`:

```python
"""Codec constants.

Every value here is part of the wire format. Changing any of them breaks
compatibility with previously generated messages.
"""

# Mode range. Mode 1 is excluded by necessity, not preference: R + A*cos(theta)
# is a circle translated sideways, which shifts the centroid the decoder relies
# on while producing no measurable change in shape.
MIN_MODE = 2
MAX_MODE = 12

# Start/end marker, using the widest available mode separation.
SENTINEL_CHORD = (MIN_MODE, MAX_MODE)

# Geometry
N_BINS = 512
AMPLITUDE = 0.12
REST_RADIUS = 1.0

# Timing
ACTIVE_FRAMES = 12
TRAILING_SILENCE_FRAMES = 3
FRAMES_PER_CHAR = ACTIVE_FRAMES + TRAILING_SILENCE_FRAMES
OSCILLATIONS = 2
ATTACK = 0.25
DECAY_POWER = 2.0

# Segmentation. QUIET_THRESHOLD is in radial-modulation units: a frame counts as
# active when its strongest mode exceeds 1% modulation of the rest radius.
QUIET_THRESHOLD = 0.01
MIN_CLOSABLE_GAP = 3
BOUNDARY_GAP_FRAMES = 10
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/codec/test_constants.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/codec tests/codec
git commit -m "Add codec package scaffold and wire-format constants"
```

---

### Task 2: Chord table

**Files:**
- Create: `src/codec/chord_table.py`
- Test: `tests/codec/test_chord_table.py`

**Step 1: Write the failing test**

```python
# tests/codec/test_chord_table.py
"""Tests for the symbol-to-chord assignment."""
import pytest

from src.codec import constants as C
from src.codec.chord_table import (
    ALPHABET,
    CHORD_BY_SYMBOL,
    SPACE,
    SPARE_CHORDS,
    SYMBOL_BY_CHORD,
)


def test_all_42_characters_assigned():
    """26 letters + 10 digits + 6 punctuation marks."""
    assert len(CHORD_BY_SYMBOL) == 42


def test_every_chord_is_unique():
    assert len(set(CHORD_BY_SYMBOL.values())) == len(CHORD_BY_SYMBOL)


def test_no_character_uses_the_sentinel():
    """A character colliding with the sentinel would truncate every message."""
    assert C.SENTINEL_CHORD not in CHORD_BY_SYMBOL.values()


def test_all_modes_within_range():
    for low, high in CHORD_BY_SYMBOL.values():
        assert C.MIN_MODE <= low < high <= C.MAX_MODE


def test_reverse_lookup_round_trips():
    for symbol, chord in CHORD_BY_SYMBOL.items():
        assert SYMBOL_BY_CHORD[chord] == symbol


def test_space_is_in_alphabet_but_has_no_chord():
    """Space is the absence of excitation, so it needs no chord."""
    assert SPACE in ALPHABET
    assert SPACE not in CHORD_BY_SYMBOL


def test_spare_chords_remain_for_curation():
    """55 pairs total, minus 42 characters and 1 sentinel."""
    assert len(SPARE_CHORDS) == 12


def test_common_letters_get_calmer_chords():
    """E is the most frequent letter and should get the lowest-sum chord."""
    assert sum(CHORD_BY_SYMBOL['E']) < sum(CHORD_BY_SYMBOL['Z'])
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/codec/test_chord_table.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.codec.chord_table'`

**Step 3: Write minimal implementation**

```python
# src/codec/chord_table.py
"""Symbol-to-chord assignment.

Each character maps to an unordered pair of vibration modes. The pairing is
generated deterministically so the table can be regenerated and audited, but it
is part of the wire format: changing it invalidates every existing message.
"""

from itertools import combinations

from . import constants as C

SPACE = ' '

# Characters in descending English frequency, then digits, then punctuation.
# Pairs are handed out lowest-sum first, so frequent letters get the calmest
# rings and a typical message reads as visually quieter.
_ASSIGNMENT_ORDER = "ETAOINSRHDLUCMFYWGPBVKXQJZ0123456789.,'-!?"


def _build_table() -> tuple[dict[str, tuple[int, int]], list[tuple[int, int]]]:
    """Assign chords to characters, returning the table and the unused pairs."""
    pairs = [
        pair
        for pair in combinations(range(C.MIN_MODE, C.MAX_MODE + 1), 2)
        if pair != C.SENTINEL_CHORD
    ]
    pairs.sort(key=lambda pair: (pair[0] + pair[1], pair[0]))
    table = dict(zip(_ASSIGNMENT_ORDER, pairs))
    return table, pairs[len(_ASSIGNMENT_ORDER):]


CHORD_BY_SYMBOL, SPARE_CHORDS = _build_table()
SYMBOL_BY_CHORD = {chord: symbol for symbol, chord in CHORD_BY_SYMBOL.items()}
ALPHABET = frozenset(CHORD_BY_SYMBOL) | {SPACE}
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/codec/test_chord_table.py -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add src/codec/chord_table.py tests/codec/test_chord_table.py
git commit -m "Add symbol-to-chord assignment table"
```

---

### Task 3: Text normalization

**Files:**
- Modify: `src/codec/chord_table.py` (append `normalize`)
- Modify: `tests/codec/test_chord_table.py` (append tests)

**Step 1: Write the failing test**

```python
# append to tests/codec/test_chord_table.py
from src.codec.chord_table import normalize


def test_lowercase_is_uppercased():
    assert normalize("hello") == ("HELLO", [])


def test_unsupported_characters_are_reported_not_silently_dropped():
    """Silent dropping breaks round-trip fidelity with no signal to the user."""
    text, dropped = normalize("A@B#C")
    assert text == "ABC"
    assert dropped == ['@', '#']


def test_strict_mode_raises_on_unsupported():
    with pytest.raises(ValueError, match="unsupported"):
        normalize("A@B", strict=True)


def test_spaces_are_preserved():
    assert normalize("A B") == ("A B", [])
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/codec/test_chord_table.py -v`
Expected: FAIL — `ImportError: cannot import name 'normalize'`

**Step 3: Write minimal implementation**

```python
# append to src/codec/chord_table.py


def normalize(text: str, strict: bool = False) -> tuple[str, list[str]]:
    """Uppercase text and remove characters the alphabet cannot represent.

    Args:
        text: Raw input text.
        strict: Raise instead of dropping unsupported characters.

    Returns:
        tuple: (normalized text, list of dropped characters in order)

    Raises:
        ValueError: If strict is True and any character is unsupported.
    """
    upper = text.upper()
    kept = [char for char in upper if char in ALPHABET]
    dropped = [char for char in upper if char not in ALPHABET]

    if dropped and strict:
        raise ValueError(f"Text contains unsupported characters: {sorted(set(dropped))}")

    return ''.join(kept), dropped
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/codec/test_chord_table.py -v`
Expected: PASS (12 tests)

**Step 5: Commit**

```bash
git add src/codec/chord_table.py tests/codec/test_chord_table.py
git commit -m "Add text normalization with explicit dropped-character reporting"
```

---

### Task 4: Radius profile

**Files:**
- Create: `src/codec/waveform.py`
- Test: `tests/codec/test_waveform.py`

**Step 1: Write the failing test**

```python
# tests/codec/test_waveform.py
"""Tests for radius profile and envelope generation."""
import numpy as np

from src.codec import constants as C
from src.codec.waveform import radius_profile


def test_profile_has_one_value_per_bin():
    assert radius_profile((3, 7), 1.0).shape == (C.N_BINS,)


def test_zero_amplitude_gives_a_perfect_circle():
    profile = radius_profile((3, 7), 0.0)
    assert np.allclose(profile, C.REST_RADIUS)


def test_mean_radius_is_preserved():
    """Modes 2 and above integrate to zero, so they cannot change mean radius."""
    profile = radius_profile((3, 7), 1.0)
    assert np.isclose(profile.mean(), C.REST_RADIUS)


def test_lobe_count_matches_the_lower_mode():
    """A single mode n produces exactly n maxima around the ring."""
    profile = radius_profile((5, 5), 1.0)
    above = profile > profile.mean()
    transitions = np.sum(above != np.roll(above, 1))
    assert transitions == 2 * 5
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/codec/test_waveform.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.codec.waveform'`

**Step 3: Write minimal implementation**

```python
# src/codec/waveform.py
"""Radius profiles for the vibrating string.

A profile is the radius of the ring sampled at N_BINS angles. All geometry is
expressed in normalized units where the rest radius is 1.0, so nothing here
depends on eventual render size.
"""

import numpy as np

from . import constants as C

# Angles are sampled without the endpoint: theta=0 and theta=2*pi are the same
# point on a closed loop, and including both would double-weight it in the FFT.
_THETA = np.linspace(0.0, 2.0 * np.pi, C.N_BINS, endpoint=False)


def radius_profile(
    chord: tuple[int, int],
    amplitude: float,
    n_bins: int = C.N_BINS,
) -> np.ndarray:
    """Build the radius profile for a chord at a given amplitude.

    Args:
        chord: Pair of mode numbers (n1, n2).
        amplitude: Signed radial modulation as a fraction of rest radius.
        n_bins: Number of angular samples.

    Returns:
        np.ndarray: Radii of shape (n_bins,).
    """
    low, high = chord
    theta = _THETA if n_bins == C.N_BINS else np.linspace(
        0.0, 2.0 * np.pi, n_bins, endpoint=False
    )
    shape = (np.cos(low * theta) + np.cos(high * theta)) / 2.0
    return C.REST_RADIUS * (1.0 + amplitude * shape)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/codec/test_waveform.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/codec/waveform.py tests/codec/test_waveform.py
git commit -m "Add radius profile generation for mode chords"
```

---

### Task 5: Pluck envelope and frame amplitudes

**Files:**
- Modify: `src/codec/waveform.py`
- Modify: `tests/codec/test_waveform.py`

**Step 1: Write the failing test**

```python
# append to tests/codec/test_waveform.py
from src.codec.waveform import envelope, frame_amplitudes


def test_envelope_starts_and_ends_at_zero():
    """This is what makes clips loop and splice without a visible seam."""
    env = envelope()
    assert np.isclose(env[0], 0.0)
    assert np.isclose(env[-1], 0.0)


def test_envelope_peaks_at_one():
    assert np.isclose(envelope().max(), 1.0)


def test_envelope_attacks_faster_than_it_decays():
    """A pluck rises sharply and relaxes slowly."""
    env = envelope()
    peak = int(np.argmax(env))
    assert peak < len(env) - peak


def test_frame_amplitudes_include_trailing_silence():
    amps = frame_amplitudes()
    assert len(amps) == C.FRAMES_PER_CHAR
    assert np.allclose(amps[-C.TRAILING_SILENCE_FRAMES:], 0.0)


def test_frame_amplitudes_change_sign():
    """The standing wave oscillates; lobes swap in and out."""
    amps = frame_amplitudes()
    assert amps.min() < 0 < amps.max()
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/codec/test_waveform.py -v`
Expected: FAIL — `ImportError: cannot import name 'envelope'`

**Step 3: Write minimal implementation**

```python
# append to src/codec/waveform.py


def envelope(n_frames: int = C.ACTIVE_FRAMES) -> np.ndarray:
    """Pluck envelope: fast attack, slow decay, zero at both ends.

    Pinning both ends to zero does three jobs at once. Clips loop seamlessly,
    any clip can follow any other without a jump cut, and the silence between
    characters becomes the delimiter the decoder segments on.
    """
    t = np.linspace(0.0, 1.0, n_frames)
    rise = t / C.ATTACK
    fall = ((1.0 - t) / (1.0 - C.ATTACK)) ** C.DECAY_POWER
    return np.where(t < C.ATTACK, rise, fall)


def frame_amplitudes(
    n_active: int = C.ACTIVE_FRAMES,
    n_gap: int = C.TRAILING_SILENCE_FRAMES,
) -> np.ndarray:
    """Signed amplitude for each frame of one character clip.

    The envelope shapes the pluck; the cosine term is the standing wave
    oscillating. Amplitude is signed because the wave inverts each half cycle.
    Trailing silent frames give the segmenter an unambiguous character boundary.
    """
    t = np.linspace(0.0, 1.0, n_active)
    active = envelope(n_active) * np.cos(2.0 * np.pi * C.OSCILLATIONS * t)
    return np.concatenate([active, np.zeros(n_gap)])
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/codec/test_waveform.py -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add src/codec/waveform.py tests/codec/test_waveform.py
git commit -m "Add pluck envelope and per-frame amplitudes"
```

---

### Task 6: Character clips

**Files:**
- Modify: `src/codec/waveform.py`
- Modify: `tests/codec/test_waveform.py`

**Step 1: Write the failing test**

```python
# append to tests/codec/test_waveform.py
from src.codec.waveform import chord_clip, quiet_clip


def test_chord_clip_shape():
    assert chord_clip((3, 7)).shape == (C.FRAMES_PER_CHAR, C.N_BINS)


def test_chord_clip_opens_and_closes_on_a_circle():
    """Required for seamless looping and splicing."""
    clip = chord_clip((3, 7))
    assert np.allclose(clip[0], C.REST_RADIUS)
    assert np.allclose(clip[-1], C.REST_RADIUS)


def test_chord_clip_actually_deforms_in_between():
    clip = chord_clip((3, 7))
    assert clip[1:-1].std() > 0.0


def test_quiet_clip_never_deforms():
    """Space is the absence of excitation."""
    assert np.allclose(quiet_clip(), C.REST_RADIUS)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/codec/test_waveform.py -v`
Expected: FAIL — `ImportError: cannot import name 'chord_clip'`

**Step 3: Write minimal implementation**

```python
# append to src/codec/waveform.py


def chord_clip(chord: tuple[int, int], n_bins: int = C.N_BINS) -> np.ndarray:
    """Build every frame of one excited character clip.

    Returns:
        np.ndarray: Radii of shape (FRAMES_PER_CHAR, n_bins).
    """
    amplitudes = frame_amplitudes() * C.AMPLITUDE
    return np.stack([radius_profile(chord, a, n_bins) for a in amplitudes])


def quiet_clip(n_bins: int = C.N_BINS) -> np.ndarray:
    """Build an unexcited clip: a still circle, which encodes a space."""
    return np.full((C.FRAMES_PER_CHAR, n_bins), C.REST_RADIUS)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/codec/test_waveform.py -v`
Expected: PASS (13 tests)

**Step 5: Commit**

```bash
git add src/codec/waveform.py tests/codec/test_waveform.py
git commit -m "Add character clip generation"
```

---

### Task 7: Spectrum and chord detection

**Files:**
- Create: `src/codec/spectrum.py`
- Test: `tests/codec/test_spectrum.py`

**Step 1: Write the failing test**

```python
# tests/codec/test_spectrum.py
"""Tests for recovering a chord from a radius profile."""
import numpy as np
import pytest

from src.codec import constants as C
from src.codec.spectrum import detect_chord, mode_band
from src.codec.waveform import radius_profile


def test_recovers_the_encoded_chord():
    chord, _ = detect_chord(radius_profile((3, 7), C.AMPLITUDE))
    assert chord == (3, 7)


@pytest.mark.parametrize("chord", [(2, 3), (4, 9), (5, 12), (2, 12), (11, 12)])
def test_recovers_every_shape_of_chord(chord):
    detected, _ = detect_chord(radius_profile(chord, C.AMPLITUDE))
    assert detected == chord


def test_a_still_circle_decodes_to_nothing():
    """A space carries no signal and must not be reported as a character."""
    chord, _ = detect_chord(np.full(C.N_BINS, C.REST_RADIUS))
    assert chord is None


def test_detection_is_rotation_invariant():
    """Magnitude spectrum discards phase, so rotating the ring changes nothing."""
    profile = radius_profile((4, 9), C.AMPLITUDE)
    rotated = np.roll(profile, C.N_BINS // 7)
    assert detect_chord(rotated)[0] == detect_chord(profile)[0]


def test_detection_is_scale_invariant():
    """Normalizing by mean radius makes resizing irrelevant."""
    profile = radius_profile((4, 9), C.AMPLITUDE)
    assert detect_chord(profile * 37.5)[0] == (4, 9)


def test_sign_flip_does_not_change_the_chord():
    """The wave inverts each half cycle; both halves must decode the same."""
    positive = radius_profile((5, 8), C.AMPLITUDE)
    negative = radius_profile((5, 8), -C.AMPLITUDE)
    assert detect_chord(positive)[0] == detect_chord(negative)[0]


def test_mode_band_peaks_at_the_encoded_modes():
    band = mode_band(radius_profile((3, 7), C.AMPLITUDE))
    assert int(np.argmax(band)) + C.MIN_MODE in (3, 7)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/codec/test_spectrum.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.codec.spectrum'`

**Step 3: Write minimal implementation**

```python
# src/codec/spectrum.py
"""Chord recovery from a radius profile.

Decoding is an angular FFT. Dividing by mean radius makes it scale invariant;
taking magnitude and discarding phase makes it rotation invariant. Both
properties fall out of the transform rather than being engineered.
"""

import numpy as np

from . import constants as C


def mode_band(profile: np.ndarray) -> np.ndarray:
    """Spectral magnitude for modes MIN_MODE..MAX_MODE, in modulation units.

    Values are the fractional radial modulation contributed by each mode, so a
    chord encoded at amplitude a produces peaks of roughly a/2. This makes the
    threshold interpretable and independent of ring size.
    """
    values = np.asarray(profile, dtype=float)
    mean_radius = values.mean()
    if mean_radius <= 0.0:
        raise ValueError("Profile must have a positive mean radius")

    normalized = values / mean_radius - 1.0
    magnitude = np.abs(np.fft.rfft(normalized)) / (len(values) / 2.0)
    return magnitude[C.MIN_MODE:C.MAX_MODE + 1]


def detect_chord(profile: np.ndarray) -> tuple[tuple[int, int] | None, float]:
    """Recover the chord from a single radius profile.

    Returns:
        tuple: (chord or None if the ring is unexcited, confidence)
        Confidence is the ratio of the weaker peak to the strongest non-peak
        mode. Values near 1.0 mean the answer is barely distinguishable from
        noise; large values mean the two peaks stand well clear.
    """
    band = mode_band(profile)
    ranked = np.argsort(band)[::-1]
    strongest, second = int(ranked[0]), int(ranked[1])

    if band[second] < C.QUIET_THRESHOLD:
        return None, 0.0

    remainder = np.delete(band, [strongest, second])
    noise_floor = float(remainder.max()) if remainder.size else 0.0
    confidence = float(band[second] / noise_floor) if noise_floor > 0 else float('inf')

    chord = (min(strongest, second) + C.MIN_MODE, max(strongest, second) + C.MIN_MODE)
    return chord, confidence
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/codec/test_spectrum.py -v`
Expected: PASS (12 tests)

**Step 5: Commit**

```bash
git add src/codec/spectrum.py tests/codec/test_spectrum.py
git commit -m "Add FFT chord detection with scale and rotation invariance"
```

---

### Task 8: Message encoding

**Files:**
- Create: `src/codec/message.py`
- Test: `tests/codec/test_message.py`

**Step 1: Write the failing test**

```python
# tests/codec/test_message.py
"""Tests for whole-sentence encoding and decoding."""
import numpy as np
import pytest

from src.codec import constants as C
from src.codec.message import encode
from src.codec.spectrum import detect_chord


def test_frame_count_includes_two_sentinels():
    frames, _ = encode("AB")
    assert frames.shape == ((2 + 2) * C.FRAMES_PER_CHAR, C.N_BINS)


def test_message_opens_with_the_sentinel():
    frames, _ = encode("A")
    peak_frame = frames[:C.ACTIVE_FRAMES][3]
    assert detect_chord(peak_frame)[0] == C.SENTINEL_CHORD


def test_unsupported_characters_are_reported():
    _, dropped = encode("A@B")
    assert dropped == ['@']


def test_space_produces_an_unexcited_clip():
    frames, _ = encode(" ")
    space_clip = frames[C.FRAMES_PER_CHAR:2 * C.FRAMES_PER_CHAR]
    assert np.allclose(space_clip, C.REST_RADIUS)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/codec/test_message.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.codec.message'`

**Step 3: Write minimal implementation**

```python
# src/codec/message.py
"""Whole-sentence encoding and decoding.

The boundary of the codec: text in, radius profiles out, and back again. No
image handling lives here or anywhere below it.
"""

import numpy as np

from . import constants as C
from .chord_table import CHORD_BY_SYMBOL, SPACE, normalize
from .waveform import chord_clip, quiet_clip


def encode(sentence: str, strict: bool = False) -> tuple[np.ndarray, list[str]]:
    """Encode a sentence as a sequence of radius profiles.

    The message is bracketed by sentinel clips so the decoder can find its
    boundaries even when leading or trailing frames are lost.

    Args:
        sentence: Text to encode.
        strict: Raise on unsupported characters instead of dropping them.

    Returns:
        tuple: (frames of shape (n_frames, N_BINS), dropped characters)
    """
    text, dropped = normalize(sentence, strict=strict)

    clips = [chord_clip(C.SENTINEL_CHORD)]
    for symbol in text:
        clips.append(quiet_clip() if symbol == SPACE else chord_clip(CHORD_BY_SYMBOL[symbol]))
    clips.append(chord_clip(C.SENTINEL_CHORD))

    return np.concatenate(clips), dropped
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/codec/test_message.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/codec/message.py tests/codec/test_message.py
git commit -m "Add sentence encoding with sentinel bracketing"
```

---

### Task 9: Frame segmentation

**Files:**
- Modify: `src/codec/spectrum.py`
- Modify: `tests/codec/test_spectrum.py`

**Step 1: Write the failing test**

```python
# append to tests/codec/test_spectrum.py
from src.codec.spectrum import active_mask, close_short_gaps, runs_of
from src.codec.waveform import chord_clip


def test_a_character_splits_without_gap_closing():
    """The standing wave crosses zero mid-clip, so raw runs over-segment."""
    mask = active_mask(chord_clip((3, 7)))
    assert sum(1 for is_active, _, _ in runs_of(mask) if is_active) > 1


def test_gap_closing_reunites_one_character():
    """Closing short quiet runs is what makes a character a single segment."""
    mask = close_short_gaps(active_mask(chord_clip((3, 7))))
    assert sum(1 for is_active, _, _ in runs_of(mask) if is_active) == 1


def test_gap_closing_does_not_merge_across_characters():
    clip = np.concatenate([chord_clip((3, 7)), chord_clip((4, 9))])
    mask = close_short_gaps(active_mask(clip))
    assert sum(1 for is_active, _, _ in runs_of(mask) if is_active) == 2


def test_runs_cover_the_whole_sequence():
    mask = active_mask(chord_clip((3, 7)))
    assert sum(stop - start for _, start, stop in runs_of(mask)) == len(mask)


def test_boundary_gap_matches_the_constant():
    """BOUNDARY_GAP_FRAMES is a measured consequence of seven other constants,
    not an independent value. The round-trip test cannot catch it drifting:
    encoder and decoder read the same constants and desynchronize in lockstep,
    staying green while real messages decode wrong. So measure it directly.
    """
    clip = np.concatenate([chord_clip((3, 7)), chord_clip((4, 9))])
    mask = close_short_gaps(active_mask(clip))
    interior_gaps = [
        stop - start
        for is_active, start, stop in runs_of(mask)
        if not is_active and start > 0 and stop < len(mask)
    ]
    assert interior_gaps == [C.BOUNDARY_GAP_FRAMES]
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/codec/test_spectrum.py -v`
Expected: FAIL — `ImportError: cannot import name 'active_mask'`

**Step 3: Write minimal implementation**

```python
# append to src/codec/spectrum.py


def active_mask(frames: np.ndarray) -> np.ndarray:
    """Flag which frames carry enough modulation to decode."""
    return np.array([mode_band(frame).max() >= C.QUIET_THRESHOLD for frame in frames])


def runs_of(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    """Split a boolean mask into (value, start, stop) runs."""
    runs: list[tuple[bool, int, int]] = []
    start = 0
    while start < len(mask):
        stop = start
        while stop < len(mask) and mask[stop] == mask[start]:
            stop += 1
        runs.append((bool(mask[start]), start, stop))
        start = stop
    return runs


def close_short_gaps(mask: np.ndarray, min_gap: int = C.MIN_CLOSABLE_GAP) -> np.ndarray:
    """Fill quiet runs too short to be character boundaries.

    The standing wave passes through zero mid-character, leaving a one-frame
    quiet patch that would otherwise split one character into two segments and
    decode it twice. Real boundaries are an order of magnitude longer, so the
    two cases separate cleanly on length.
    """
    closed = mask.copy()
    for is_active, start, stop in runs_of(mask):
        interior = start > 0 and stop < len(mask)
        if not is_active and interior and (stop - start) < min_gap:
            closed[start:stop] = True
    return closed
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/codec/test_spectrum.py -v`
Expected: PASS (16 tests)

**Step 5: Commit**

```bash
git add src/codec/spectrum.py tests/codec/test_spectrum.py
git commit -m "Add frame segmentation with short-gap closing"
```

---

### Task 10: Message decoding

**Files:**
- Modify: `src/codec/message.py`
- Modify: `tests/codec/test_message.py`

**Step 1: Write the failing test**

```python
# append to tests/codec/test_message.py
from src.codec.message import decode


def test_decodes_a_single_character():
    frames, _ = encode("A")
    assert decode(frames) == "A"


def test_decodes_a_sentence():
    frames, _ = encode("HELLO WORLD")
    assert decode(frames) == "HELLO WORLD"


def test_decodes_consecutive_spaces():
    """Spaces come from quiet-run length, so runs of them must not collapse."""
    frames, _ = encode("X  Y")
    assert decode(frames) == "X  Y"


def test_decodes_digits_and_punctuation():
    frames, _ = encode("MEET ME AT 8PM!")
    assert decode(frames) == "MEET ME AT 8PM!"


def test_sentinels_are_stripped_from_output():
    frames, _ = encode("A")
    assert decode(frames) == "A"
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/codec/test_message.py -v`
Expected: FAIL — `ImportError: cannot import name 'decode'`

**Step 3: Write minimal implementation**

```python
# append to src/codec/message.py
from .chord_table import SYMBOL_BY_CHORD
from .spectrum import active_mask, close_short_gaps, detect_chord, mode_band, runs_of


def _spaces_in_gap(length: int) -> int:
    """Convert a quiet-run length into a count of spaces.

    A space is a still circle, so unlike every other symbol it produces no
    active segment to count. It is recovered instead from how much longer the
    silence ran than a plain character boundary.
    """
    return max(0, round((length - C.BOUNDARY_GAP_FRAMES) / C.FRAMES_PER_CHAR))


def decode(frames: np.ndarray) -> str:
    """Recover a sentence from a sequence of radius profiles.

    Args:
        frames: Array of shape (n_frames, n_bins).

    Returns:
        str: The decoded sentence, with sentinels removed.

    Raises:
        ValueError: If no sentinel pair is found.
    """
    frames = np.asarray(frames, dtype=float)
    mask = close_short_gaps(active_mask(frames))

    tokens: list[tuple[str, object]] = []
    for is_active, start, stop in runs_of(mask):
        if not is_active:
            tokens.append(('gap', stop - start))
            continue
        segment = frames[start:stop]
        strongest = int(np.argmax([mode_band(frame).max() for frame in segment]))
        chord, _ = detect_chord(segment[strongest])
        tokens.append(('chord', chord))

    sentinels = [
        index for index, (kind, value) in enumerate(tokens)
        if kind == 'chord' and value == C.SENTINEL_CHORD
    ]
    if len(sentinels) < 2:
        raise ValueError("Message is missing its sentinel markers")

    body = tokens[sentinels[0] + 1:sentinels[-1]]

    decoded = []
    for kind, value in body:
        if kind == 'chord':
            decoded.append(SYMBOL_BY_CHORD[value])
        else:
            decoded.append(SPACE * _spaces_in_gap(value))
    return ''.join(decoded)
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/codec/test_message.py -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add src/codec/message.py tests/codec/test_message.py
git commit -m "Add sentence decoding with gap-length space recovery"
```

---

### Task 11: Round-trip property test

**Files:**
- Create: `tests/codec/test_round_trip.py`

This is the central test of the whole project. Encoding and decoding are exact
mathematical inverses, so this exercises the entire message space rather than
sampling a few cases.

**Step 1: Write the failing test**

```python
# tests/codec/test_round_trip.py
"""The central correctness property: decode(encode(s)) == s."""
import random

import pytest

from src.codec.chord_table import CHORD_BY_SYMBOL, SPACE
from src.codec.message import decode, encode

_SYMBOLS = sorted(CHORD_BY_SYMBOL) + [SPACE]


@pytest.mark.parametrize("symbol", sorted(CHORD_BY_SYMBOL))
def test_every_character_round_trips(symbol):
    """Exhaustive over the alphabet, not a sample."""
    frames, _ = encode(symbol)
    assert decode(frames) == symbol


@pytest.mark.parametrize("seed", range(50))
def test_random_sentences_round_trip(seed):
    rng = random.Random(seed)
    length = rng.randint(1, 12)
    sentence = ''.join(rng.choice(_SYMBOLS) for _ in range(length))
    # Leading and trailing spaces have no active segment to anchor them and are
    # not recoverable by design; the encoder's own normalization keeps them, so
    # compare against the trimmed form.
    sentence = sentence.strip() or 'A'
    frames, _ = encode(sentence)
    assert decode(frames) == sentence


def test_long_sentence_round_trips():
    sentence = "THE QUICK BROWN FOX JUMPS OVER 13 LAZY DOGS, TWICE!"
    frames, _ = encode(sentence)
    assert decode(frames) == sentence
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/codec/test_round_trip.py -v`
Expected: PASS if tasks 1-10 are correct. If any case fails, fix the codec — do
not weaken the test.

**Step 3: Handle leading and trailing spaces explicitly**

The test above sidesteps leading/trailing spaces. That is a real limitation, not a
test artifact: a leading space is silence adjacent to the sentinel's own silence, so
its length is ambiguous. Document it in the module docstring of
`src/codec/message.py`:

```python
# append to the encode() docstring
    Note:
        Leading and trailing spaces are not recoverable. Their silence merges
        with the sentinel's own trailing silence, leaving no unambiguous length
        to measure. Interior spaces, including runs of them, round-trip exactly.
```

**Step 4: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 80 pre-existing plus the new codec tests

**Step 5: Commit**

```bash
git add tests/codec/test_round_trip.py src/codec/message.py
git commit -m "Add round-trip property test over the full alphabet"
```

---

### Task 12: Noise robustness

**Files:**
- Create: `tests/codec/test_robustness.py`

Design section 6 says Perlin vibration may remain only if band-limited above the
decoder's mode range. This task turns that claim into a test.

**Step 1: Write the failing test**

```python
# tests/codec/test_robustness.py
"""Decoding must survive the distortions real assets will introduce."""
import numpy as np
import pytest

from src.codec import constants as C
from src.codec.message import decode, encode
from src.codec.spectrum import detect_chord
from src.codec.waveform import radius_profile

SENTENCE = "VIBING LETTERS"


def _high_frequency_noise(n_bins: int, amplitude: float, seed: int) -> np.ndarray:
    """Noise confined to modes above MAX_MODE, as design section 6 requires."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=False)
    modes = range(C.MAX_MODE + 1, C.MAX_MODE + 20)
    return amplitude * sum(
        rng.normal() * np.cos(mode * theta + rng.uniform(0, 2 * np.pi))
        for mode in modes
    )


def test_band_limited_noise_does_not_break_decoding():
    """Perlin-style texture is safe if it stays out of the decoder's band."""
    frames, _ = encode(SENTENCE)
    noisy = frames + _high_frequency_noise(C.N_BINS, 0.02, seed=1)
    assert decode(noisy) == SENTENCE


def test_in_band_noise_does_break_decoding():
    """The band limit is load-bearing, not decorative."""
    profile = radius_profile((3, 7), C.AMPLITUDE)
    theta = np.linspace(0.0, 2.0 * np.pi, C.N_BINS, endpoint=False)
    corrupted = profile + 0.5 * C.AMPLITUDE * np.cos(5 * theta)
    assert detect_chord(corrupted)[0] != (3, 7)


@pytest.mark.parametrize("scale", [0.25, 1.0, 7.5, 100.0])
def test_decoding_survives_rescaling(scale):
    frames, _ = encode(SENTENCE)
    assert decode(frames * scale) == SENTENCE


def test_decoding_survives_rotation():
    frames, _ = encode(SENTENCE)
    rotated = np.roll(frames, C.N_BINS // 5, axis=1)
    assert decode(rotated) == SENTENCE


def test_confidence_is_high_for_clean_signal():
    _, confidence = detect_chord(radius_profile((3, 7), C.AMPLITUDE))
    assert confidence > 10.0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/codec/test_robustness.py -v`
Expected: Some may pass immediately. Any failure is real information about the
codec's limits — investigate before adjusting thresholds.

**Step 3: Record measured limits**

Add the noise amplitude at which decoding starts to fail to
`docs/plans/2026-08-23-vibrating-string-cipher-design.md` under section 5, so the
asset pipeline has a concrete budget to design against.

**Step 4: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/codec/test_robustness.py docs/plans/
git commit -m "Add robustness tests and record measured noise tolerance"
```

---

## Definition of done

- [ ] `.venv/bin/python -m pytest -q` passes, including the 80 pre-existing tests
- [ ] Every character in the alphabet round-trips, verified exhaustively
- [ ] Interior spaces, including consecutive runs, round-trip
- [ ] Decoding is invariant to rotation and rescaling, verified by test
- [ ] Band-limited noise tolerance is measured and written into the design doc
- [ ] `src/codec/` imports nothing from `src/morphing/`, `PIL`, or `cv2`

## Out of scope

Contour extraction from images, asset generation, image-model styling, the polar
warp, WebP/GIF writing, metadata stamping, and the browser decoder. The codec must
stand on its own first.
