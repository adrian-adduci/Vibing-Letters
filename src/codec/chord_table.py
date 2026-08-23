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
