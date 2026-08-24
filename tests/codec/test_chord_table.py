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
    'A': (2, 5), 'B': (5, 7), 'C': (2, 9), 'D': (2, 8), 'E': (2, 3), 'F': (4, 7),
    'G': (3, 9), 'H': (4, 5), 'I': (2, 6), 'J': (6, 7), 'K': (3, 10), 'L': (3, 7),
    'M': (3, 8), 'N': (3, 5), 'O': (3, 4), 'P': (4, 8), 'Q': (5, 8), 'R': (3, 6),
    'S': (2, 7), 'T': (2, 4), 'U': (4, 6), 'V': (2, 11), 'W': (2, 10), 'X': (4, 9),
    'Y': (5, 6), 'Z': (3, 11), '0': (4, 10), '1': (5, 9), '2': (6, 8), '3': (3, 12),
    '4': (4, 11), '5': (5, 10), '6': (6, 9), '7': (7, 8), '8': (4, 12), '9': (5, 11),
    '!': (7, 10), "'": (5, 12), ',': (7, 9), '-': (6, 11), '.': (6, 10), '?': (8, 9),
}

_GOLDEN_SPARE = (
    (6, 12), (7, 11), (8, 10), (7, 12), (8, 11), (9, 10),
    (8, 12), (9, 11), (9, 12), (10, 11), (10, 12), (11, 12),
)


def test_table_matches_the_canonical_wire_format():
    """The chord table IS the wire format. Changing it breaks every message."""
    assert CHORD_BY_SYMBOL == _GOLDEN_TABLE


def test_spare_chords_match_the_canonical_wire_format():
    assert SPARE_CHORDS == _GOLDEN_SPARE


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


def test_leading_and_trailing_spaces_are_stripped():
    """The decoder cannot recover them, so normalize must not produce them."""
    assert normalize("  A  ") == ("A", [])


def test_interior_spaces_are_preserved():
    """Interior runs of spaces round-trip exactly and must survive."""
    assert normalize("X  Y") == ("X  Y", [])
