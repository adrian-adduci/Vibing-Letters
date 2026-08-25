"""Tests for the symbol-to-chord assignment."""
import pytest

from src.codec import constants as C
from src.codec.chord_table import (
    ALPHABET,
    CHORD_BY_SYMBOL,
    SPACE,
    SPARE_CHORDS,
    SYMBOL_BY_CHORD,
    normalize,
)

# Transcribed from the canonical table in
# docs/plans/2026-08-23-vibrating-string-cipher-design.md, section 1.
# This IS the wire format. A diff here is a breaking change to every message
# ever generated and must bump the codec version. Never "fix" this table to
# match new generator output -- fix the generator.
_GOLDEN_TABLE = {
    ' ': (2, 3), 'A': (3, 4), 'B': (2, 11), 'C': (3, 8), 'D': (3, 7), 'E': (2, 4),
    'F': (5, 6), 'G': (4, 8), 'H': (2, 8), 'I': (3, 5), 'J': (3, 11), 'K': (4, 9),
    'L': (4, 6), 'M': (4, 7), 'N': (2, 7), 'O': (2, 6), 'P': (5, 7), 'Q': (6, 7),
    'R': (4, 5), 'S': (3, 6), 'T': (2, 5), 'U': (2, 9), 'V': (3, 10), 'W': (3, 9),
    'X': (5, 8), 'Y': (2, 10), 'Z': (4, 10), '0': (5, 9), '1': (6, 8), '2': (3, 12),
    '3': (4, 11), '4': (5, 10), '5': (6, 9), '6': (7, 8), '7': (4, 12), '8': (5, 11),
    '9': (6, 10), '!': (8, 9), "'": (6, 11), ',': (5, 12), '-': (7, 10), '.': (7, 9),
    '?': (6, 12),
}

_GOLDEN_SPARE = (
    (7, 11), (8, 10), (7, 12), (8, 11), (9, 10), (8, 12),
    (9, 11), (9, 12), (10, 11), (10, 12), (11, 12),
)


def test_table_matches_the_canonical_wire_format():
    """The chord table IS the wire format. Changing it breaks every message."""
    assert CHORD_BY_SYMBOL == _GOLDEN_TABLE


def test_spare_chords_match_the_canonical_wire_format():
    assert SPARE_CHORDS == _GOLDEN_SPARE


def test_all_43_characters_assigned():
    """26 letters + 10 digits + 6 punctuation marks + space."""
    assert len(CHORD_BY_SYMBOL) == 43


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


def test_space_is_a_character_with_the_calmest_chord():
    """Space is the most frequent character in English text, so under the
    table's own frequency rule it takes the lowest-sum chord."""
    assert SPACE in CHORD_BY_SYMBOL
    assert sum(CHORD_BY_SYMBOL[SPACE]) == min(sum(c) for c in CHORD_BY_SYMBOL.values())


def test_space_is_in_the_alphabet():
    assert SPACE in ALPHABET


def test_spare_chords_remain_for_curation():
    """55 pairs total, minus 43 characters and 1 sentinel."""
    assert len(SPARE_CHORDS) == 11


def test_common_letters_get_calmer_chords():
    """E is the most frequent letter and should get the lowest-sum chord."""
    assert sum(CHORD_BY_SYMBOL['E']) < sum(CHORD_BY_SYMBOL['Z'])


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


def test_case_mapping_may_expand_characters():
    """Full case mapping is not length-preserving; the codec accepts this."""
    assert normalize("straße") == ("STRASSE", [])


def test_normalization_is_idempotent():
    """encode/decode round-trips normalized text, so normalizing twice must
    equal normalizing once."""
    once, _ = normalize("straße café")
    twice, _ = normalize(once)
    assert twice == once


def test_leading_and_trailing_spaces_are_preserved():
    """Space is now a decodable character, so there is no reason to discard it."""
    assert normalize("  A  ") == ("  A  ", [])


def test_interior_spaces_are_preserved():
    """Interior runs of spaces round-trip exactly and must survive."""
    assert normalize("X  Y") == ("X  Y", [])
