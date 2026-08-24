"""Codec constants.

The values here fall into three groups with very different stability
guarantees, so they are kept in separate sections below:

* Wire format -- changing these invalidates every previously generated message.
* Render parameters -- not part of the wire format; decoding is invariant to
  them, so they may be changed freely to suit an output medium.
* Decoder tuning -- does not change what an encoder emits, but does change how
  existing messages are read.
"""

# ---------------------------------------------------------------------------
# Wire format (changing these invalidates existing messages)
# ---------------------------------------------------------------------------

# Mode 1 is excluded by necessity, not preference: R + A*cos(theta) is a circle
# translated sideways, which shifts the centroid the decoder relies on while
# producing no measurable change in shape.
MIN_MODE = 2

# Modes MIN_MODE..MAX_MODE give C(11, 2) = 55 unordered pairs. That budget is
# spent as 42 characters + 1 sentinel + 12 spare. Raising MAX_MODE would buy
# more pairs but renumber nothing, so it invalidates existing messages.
MAX_MODE = 12

# Start/end marker, using the widest available mode separation.
SENTINEL_CHORD = (MIN_MODE, MAX_MODE)

# Timing
ACTIVE_FRAMES = 12
TRAILING_SILENCE_FRAMES = 3
FRAMES_PER_CHAR = ACTIVE_FRAMES + TRAILING_SILENCE_FRAMES
OSCILLATIONS = 2
ATTACK = 0.25
DECAY_POWER = 2.0

# ---------------------------------------------------------------------------
# Render parameters (not wire format; decoding is invariant to them)
# ---------------------------------------------------------------------------
# Detection normalizes by mean radius and by bin count, so radius profiles at
# any positive scale or any bin count decode identically. These values pick a
# convenient rendering, not a format.
#
# AMPLITUDE is the exception to "change freely". Decoding is invariant to the
# *scale* of a profile, but AMPLITUDE is not a scale: it sets how far a pluck
# rises above QUIET_THRESHOLD, and therefore how many frames at the head and
# tail of a character fall below it. Raising it lengthens the active run and
# shortens the quiet one, which shifts BOUNDARY_GAP_FRAMES -- measured at 11
# frames for AMPLITUDE = 0.06, 10 at the current 0.12, and 7 at 0.24, ranging
# from 14 down to 7 across the usable band. Space recovery rounds the measured
# gap to the nearest multiple of FRAMES_PER_CHAR and so absorbs roughly +/- 7
# frames of drift; every value in that band still recovers zero spaces at a
# plain boundary. A change here is therefore safe for decoding but leaves
# BOUNDARY_GAP_FRAMES stale until it is re-measured.
#
# Below roughly 0.04 no frame clears QUIET_THRESHOLD at all and characters stop
# being detected entirely, which is a floor on AMPLITUDE rather than a drift.

N_BINS = 512

# The codec works in normalized radius units; a rest radius of 1.0 means every
# other radial quantity reads directly as a fraction of the ring size.
REST_RADIUS = 1.0

# Peak radial modulation as a fraction of the rest radius.
AMPLITUDE = 0.12

# ---------------------------------------------------------------------------
# Decoder tuning (changes how existing messages are read)
# ---------------------------------------------------------------------------

# In radial-modulation units: a frame counts as active when its strongest mode
# exceeds 1% modulation of the rest radius.
QUIET_THRESHOLD = 0.01

# Quiet runs shorter than this are closed rather than treated as boundaries.
MIN_CLOSABLE_GAP = 3

# Quiet-run length observed at a plain character boundary. This is not an
# independent knob: it is a measured consequence of ACTIVE_FRAMES,
# TRAILING_SILENCE_FRAMES, OSCILLATIONS, ATTACK, DECAY_POWER, AMPLITUDE, and
# QUIET_THRESHOLD, since those seven together decide how many frames at the tail
# of a character fall below the quiet threshold. 10 is the value measured at the
# current defaults; tuning any of the seven desynchronizes space recovery and
# this number must be re-measured. A test in a later task asserts the measured
# value against the generated waveform.
BOUNDARY_GAP_FRAMES = 10

# Below this confidence, a segment is reported undecodable rather than guessed.
# This separates two measured populations. A degenerate single-mode ring scores
# 1.09 to 1.46, because detect_chord always returns two distinct modes and the
# second is float rounding noise. A genuine chord scores ~4e15 clean and no
# lower than 11.1 under sigma=0.02 noise, already past the point where
# confidence degrades faster than accuracy does. 5.0 sits between them.
MIN_CONFIDENCE = 5.0
