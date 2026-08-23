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
