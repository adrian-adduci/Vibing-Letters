"""The central correctness property: decode(encode(s)) == s."""
import random

import pytest

from src.codec.chord_table import CHORD_BY_SYMBOL, SPACE
from src.codec.message import decode, encode

_SYMBOLS = sorted(CHORD_BY_SYMBOL) + [SPACE]


@pytest.mark.parametrize("symbol", sorted(CHORD_BY_SYMBOL))
def test_every_character_round_trips(symbol):
    """Exhaustive over the alphabet, not a sample."""
    result = encode(symbol)
    assert decode(result.frames) == result.text


@pytest.mark.parametrize("seed", range(50))
def test_random_sentences_round_trip(seed):
    rng = random.Random(seed)
    length = rng.randint(1, 12)
    sentence = ''.join(rng.choice(_SYMBOLS) for _ in range(length))
    result = encode(sentence)
    assert decode(result.frames) == result.text


def test_long_sentence_round_trips():
    result = encode("THE QUICK BROWN FOX JUMPS OVER 13 LAZY DOGS, TWICE!")
    assert decode(result.frames) == result.text


@pytest.mark.parametrize("raw", [
    "  hello world  ",     # lowercase plus surrounding whitespace
    "Café 8pm!",           # unsupported accented character
    "\tX  Y\n",            # tabs, newline, interior double space
    "   ",                 # whitespace only
    "@@@",                 # nothing representable at all
    "",                    # empty
])
def test_property_holds_for_arbitrary_raw_input(raw):
    """The property is TOTAL - no input class is excluded. That is only true
    because normalize already removed the unrecoverable cases (leading and
    trailing whitespace), so the right-hand side is the normalized text rather
    than the caller's original string."""
    result = encode(raw)
    assert decode(result.frames) == result.text
