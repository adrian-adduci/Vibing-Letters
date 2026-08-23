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
GAP_FRAMES = 3
FRAMES_PER_CHAR = ACTIVE_FRAMES + GAP_FRAMES
OSCILLATIONS = 2
ATTACK = 0.25
DECAY_POWER = 2.0

# Segmentation. QUIET_THRESHOLD is in radial-modulation units: a frame counts as
# active when its strongest mode exceeds 1% modulation of the rest radius.
QUIET_THRESHOLD = 0.01
MIN_GAP_FRAMES = 3
BASE_GAP_FRAMES = 10
