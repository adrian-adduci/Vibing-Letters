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


def _build_table() -> tuple[dict[str, tuple[int, int]], tuple[tuple[int, int], ...]]:
    """Assign chords to characters, returning the table and the unused pairs."""
    pairs = [
        pair
        for pair in combinations(range(C.MIN_MODE, C.MAX_MODE + 1), 2)
        if pair != C.SENTINEL_CHORD
    ]
    pairs.sort(key=lambda pair: (pair[0] + pair[1], pair[0]))

    assert len(set(_ASSIGNMENT_ORDER)) == len(_ASSIGNMENT_ORDER), \
        "_ASSIGNMENT_ORDER contains a duplicate character"
    assert SPACE not in _ASSIGNMENT_ORDER, \
        "space is encoded as stillness, not a chord"
    assert len(_ASSIGNMENT_ORDER) <= len(pairs), \
        f"{len(_ASSIGNMENT_ORDER)} characters but only {len(pairs)} chords available"

    table = dict(zip(_ASSIGNMENT_ORDER, pairs))
    return table, tuple(pairs[len(_ASSIGNMENT_ORDER):])


CHORD_BY_SYMBOL, SPARE_CHORDS = _build_table()
SYMBOL_BY_CHORD = {chord: symbol for symbol, chord in CHORD_BY_SYMBOL.items()}
ALPHABET = frozenset(CHORD_BY_SYMBOL) | {SPACE}


def normalize(text: str, strict: bool = False) -> tuple[str, list[str]]:
    """Uppercase text and remove characters the alphabet cannot represent.

    Args:
        text: Raw input text.
        strict: Raise instead of dropping unsupported characters.

    Returns:
        tuple: (normalized text, list of dropped characters in order)

    Raises:
        ValueError: If strict is True and any character is unsupported.

    Note:
        Normalization is lossy in more ways than dropping characters. Full
        Unicode case mapping can expand one character into several
        ('ss' from the German sharp s, 'FI' from the fi ligature), so the
        result may be longer than the input and `dropped` tracks the
        uppercased text rather than the original. Leading and trailing
        whitespace is stripped, because the decoder cannot recover it.
        `strict` guards against characters the alphabet cannot represent; it
        does not guarantee the output matches the input character for
        character. The normalized text, not the raw input, is what round-trips.
    """
    upper = text.upper()
    kept = [char for char in upper if char in ALPHABET]
    dropped = [char for char in upper if char not in ALPHABET]

    if dropped and strict:
        raise ValueError(f"Text contains unsupported characters: {sorted(set(dropped))}")

    return ''.join(kept).strip(), dropped
