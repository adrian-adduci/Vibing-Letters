"""The entire runtime: a lookup and a concatenation.

The tests that matter are the ones that put a sentence in and get the same
sentence back out of a written file, because that is the only claim a user
cares about. The rest guard the two things runtime could get wrong -- shipping
a message with a character silently missing, and producing a different file
every time it is asked for the same one.
"""
import cv2
import numpy as np
import pytest

from src import compose
from src.assets import contour, publish
from src.codec import constants as C
from src.codec.chord_table import CHORD_BY_SYMBOL
from src.codec.message import decode
from src.vision import ring


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    """A published asset set, built from exact contours rather than artwork.

    Real assets cost money and a network; exact contours exercise the same code
    path and are what the artwork is conditioned on anyway.
    """
    root = tmp_path_factory.mktemp("assets")
    stills, out = root / "stills", root / "clips"
    stills.mkdir()

    wanted = "ABC HELO WRD?"
    entries = []
    for symbol in dict.fromkeys(wanted):
        chord = CHORD_BY_SYMBOL[symbol]
        path = stills / f"{compose.symbol_name(symbol)}.png"
        cv2.imwrite(str(path), contour.render_control_image(chord, size=384))
        entries.append((compose.symbol_name(symbol), chord, path))

    sentinel = stills / "SENTINEL.png"
    cv2.imwrite(str(sentinel),
                contour.render_control_image(C.SENTINEL_CHORD, size=384))
    entries.append((compose.SENTINEL_NAME, C.SENTINEL_CHORD, sentinel))

    results = publish.publish_all(entries, out)
    assert all(r.ok for r in results), publish.report(results)
    return compose.load_clips(out)


def read_back(path) -> str:
    from src.assets import emit
    frames = emit.read_clip(path)
    return decode(np.stack([ring.radius_profile(f) for f in frames]))


class TestRoundTrip:
    @pytest.mark.parametrize("sentence", ["A", "ABC", "HELLO WORLD", "  A  "])
    def test_a_sentence_written_to_a_file_reads_back(self, sentence, clips, tmp_path):
        result = compose.write(sentence, tmp_path / "m.webp", clips)
        assert read_back(tmp_path / "m.webp") == result.text

    def test_it_round_trips_through_gif_too(self, clips, tmp_path):
        result = compose.write("HELLO", tmp_path / "m.gif", clips)
        assert read_back(tmp_path / "m.gif") == result.text

    def test_lowercase_is_upper_cased_and_reported_as_such(self, clips, tmp_path):
        result = compose.write("hello world", tmp_path / "m.webp", clips)
        assert result.text == "HELLO WORLD"
        assert read_back(tmp_path / "m.webp") == "HELLO WORLD"

    def test_spaces_are_carried_exactly(self, clips, tmp_path):
        """Space is a chord like any other, so leading and trailing ones are
        not trimmed and not inferred from a gap length."""
        result = compose.write("  A B  ", tmp_path / "m.webp", clips)
        assert result.text == "  A B  "
        assert read_back(tmp_path / "m.webp") == "  A B  "


class TestDeterminism:
    def test_the_same_sentence_gives_a_byte_identical_file(self, clips, tmp_path):
        """The rotation is seeded, and the seed has to be reproducible.

        Python's built-in hash is salted per process, so using it would give a
        different file on every run and defeat the reason for seeding at all.
        """
        a = compose.write("HELLO", tmp_path / "a.webp", clips)
        b = compose.write("HELLO", tmp_path / "b.webp", clips)
        assert (tmp_path / "a.webp").read_bytes() == (tmp_path / "b.webp").read_bytes()
        assert a.text == b.text

    def test_repeated_characters_are_rotated_differently(self, clips, tmp_path):
        """Otherwise a word like HELLO reads as the same ring stamped twice.

        The slice arithmetic uses the clips' actual length rather than
        FRAMES_PER_CHAR. A published clip is shorter than the format's fifteen
        frames because both encoders merge the identical trailing circles, and
        slicing by fifteen would compare misaligned windows that differ whether
        or not anything was rotated -- which is to say, it would pass for the
        wrong reason.
        """
        head, body = len(clips[compose.SENTINEL_NAME]), len(clips['L'])
        result = compose.compose("LL", clips)
        first = result.frames[head:head + body]
        second = result.frames[head + body:head + 2 * body]

        assert np.array_equal(first, clips['L']) is False  # something happened
        assert not np.array_equal(first, second)

    def test_rotation_does_not_change_what_is_read(self, clips, tmp_path):
        """Claimed by the design and free in the maths, but rotation resamples
        the pixel grid, so through an image it has to be demonstrated."""
        compose.write("LLL", tmp_path / "m.webp", clips)
        assert read_back(tmp_path / "m.webp") == "LLL"

    def test_different_sentences_rotate_differently(self, clips):
        assert compose.rotation_for("AB", 0) != compose.rotation_for("BA", 0)
        assert compose.rotation_for("AB", 0) != compose.rotation_for("AB", 1)


class TestRefusing:
    def test_a_missing_asset_refuses_rather_than_omitting(self, clips):
        """Composing part of a message and leaving the rest out yields a file
        that decodes cleanly to the wrong words. Refusing is the lesser harm."""
        with pytest.raises(KeyError, match="No published asset"):
            compose.compose("AZ", clips)          # Z was never published

    def test_a_missing_sentinel_refuses(self, clips):
        without = {k: v for k, v in clips.items() if k != compose.SENTINEL_NAME}
        with pytest.raises(KeyError, match="SENTINEL"):
            compose.compose("A", without)

    def test_unrepresentable_characters_are_reported_when_dropped(self, clips):
        result = compose.compose("AéB", clips)
        assert result.text == "AB"
        assert result.dropped

    def test_strict_mode_raises_instead_of_dropping(self, clips):
        with pytest.raises(ValueError):
            compose.compose("AéB", clips, strict=True)

    def test_an_empty_clip_directory_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No clips"):
            compose.load_clips(tmp_path)


class TestNames:
    def test_names_match_what_the_build_writes(self):
        from src.assets import build
        for symbol in CHORD_BY_SYMBOL:
            assert compose.symbol_name(symbol) == build.symbol_name(symbol)
