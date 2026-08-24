"""Writing a clip out, and getting the same message back from the file.

The strongest test here is not that a file appears. It is that a clip written
to disk, compressed, quantized and read back by an ordinary image library still
decodes to the chord it started as. Everything before this stage worked on
arrays that never left memory.
"""
import numpy as np
import pytest
from PIL import Image

from src.assets import contour, emit, verify, warp
from src.codec import constants as C
from src.codec.chord_table import CHORD_BY_SYMBOL
from src.codec.message import decode
from src.codec.spectrum import active_mask, frame_peaks, runs_of
from src.vision import ring


@pytest.fixture(scope="module")
def clip():
    return warp.envelope_clip(contour.render_control_image((3, 8), size=512))


class TestTheLink:
    def test_webp_keeps_the_link(self, clip, tmp_path):
        path = emit.write_webp(clip, tmp_path / "a.webp")
        assert emit.read_link(path) == emit.DECODER_URL

    def test_gif_keeps_the_link(self, clip, tmp_path):
        path = emit.write_gif(clip, tmp_path / "a.gif")
        assert emit.read_link(path) == emit.DECODER_URL

    def test_a_custom_url_round_trips(self, clip, tmp_path):
        url = "https://example.org/decode?v=2&from=here"
        emit.write_webp(clip, tmp_path / "b.webp", url=url)
        emit.write_gif(clip, tmp_path / "b.gif", url=url)
        assert emit.read_link(tmp_path / "b.webp") == url
        assert emit.read_link(tmp_path / "b.gif") == url

    def test_the_xmp_is_a_real_packet_not_a_bare_string(self, clip, tmp_path):
        """Pillow will store anything in the XMP slot; nothing else will read it.

        XMP is defined as an RDF document. A bare URL parked there reads as a
        corrupt packet to every other tool, so the link would be present and
        invisible at the same time -- the exact failure the read-back assertion
        exists to prevent, one level down.
        """
        path = emit.write_webp(clip, tmp_path / "c.webp")
        with Image.open(path) as image:
            packet = image.info['xmp']
        assert b'<x:xmpmeta' in packet
        assert b'dc:source' in packet
        assert emit.DECODER_URL.encode() in packet

    def test_xml_significant_characters_are_escaped(self):
        """A URL with a query string carries ampersands, which are markup."""
        packet = emit.xmp_packet("https://example.org/d?a=1&b=2")
        assert b'&amp;' in packet
        assert emit._url_from_xmp(packet) == "https://example.org/d?a=1&b=2"

    def test_a_link_that_did_not_survive_is_an_error_not_a_shrug(
        self, clip, tmp_path, monkeypatch
    ):
        """The whole point of reading back. A silently unlinked file looks
        exactly like a correctly linked one until someone tries to follow it."""
        monkeypatch.setattr(emit, 'xmp_packet', lambda url: b'not a packet')
        with pytest.raises(OSError, match="decoder link"):
            emit.write_webp(clip, tmp_path / "d.webp")

    def test_a_file_with_no_link_reads_as_none(self, clip, tmp_path):
        images = emit._as_pil(clip, 128)
        path = tmp_path / "bare.gif"
        images[0].save(path, save_all=True, append_images=images[1:])
        assert emit.read_link(path) is None


class TestTheClipSurvivesTheFile:
    def test_webp_round_trips_the_chord(self, clip, tmp_path):
        path = emit.write_webp(clip, tmp_path / "e.webp")
        assert verify.accept(emit.read_clip(path), (3, 8)).accepted

    def test_gif_round_trips_the_chord(self, clip, tmp_path):
        """GIF is the hard case: 256 px and a 256-colour palette.

        Quantizing a glowing gradient to a palette is exactly the kind of damage
        that could smear a contour, and the fallback format is the one most
        likely to be re-encoded by whatever carries it.
        """
        path = emit.write_gif(clip, tmp_path / "e.gif")
        assert verify.accept(emit.read_clip(path), (3, 8)).accepted

    @pytest.mark.parametrize("writer", [emit.write_webp, emit.write_gif])
    def test_identical_frames_are_merged_by_both_encoders(self, clip, writer, tmp_path):
        """Documented, not worked around. Frame counts are not preserved.

        Both encoders collapse runs of byte-identical consecutive frames and
        neither offers a way to stop them.

        The run is one frame longer than the trailing silence alone, because the
        envelope is pinned to zero at *both* ends: the last active frame is
        already a rest circle before the silent frames begin. So
        TRAILING_SILENCE_FRAMES + 1 identical circles collapse to one, and
        exactly TRAILING_SILENCE_FRAMES frames are lost. Asserting the count
        stops a future reader assuming the frames line up, and would catch an
        encoder that started merging something else as well.
        """
        written = writer(clip, tmp_path / "f.out")
        assert len(emit.read_clip(written)) == \
            C.FRAMES_PER_CHAR - C.TRAILING_SILENCE_FRAMES

    def test_quiet_runs_stay_clear_of_the_gap_threshold_after_merging(self, tmp_path):
        """Merging is only safe because of what it cannot reach.

        A character boundary is not a run of identical frames: it spans two
        different stills, whose rest circles are different pictures. So the
        boundary survives merging while the mid-character zero crossing -- which
        MIN_CLOSABLE_GAP exists to close -- does not need to. Written as one
        message, the whole alphabet comes back with boundaries at seven frames
        and crossings at one, against a threshold of three.
        """
        chords = sorted(set(CHORD_BY_SYMBOL.values()) | {C.SENTINEL_CHORD})
        frames = np.concatenate([
            warp.envelope_clip(contour.render_control_image(c, size=256))
            for c in chords
        ])
        recovered = emit.read_clip(emit.write_gif(frames, tmp_path / "all.gif"))
        profiles = np.stack([ring.radius_profile(f) for f in recovered])

        mask = active_mask(frame_peaks(profiles))
        interior = {stop - start for live, start, stop in runs_of(mask)
                    if not live and start > 0 and stop < len(mask)}

        assert min(interior) < C.MIN_CLOSABLE_GAP   # crossings, correctly closed
        assert min(g for g in interior if g >= C.MIN_CLOSABLE_GAP) > C.MIN_CLOSABLE_GAP

    def test_output_lands_at_the_declared_size(self, clip, tmp_path):
        assert emit.read_clip(emit.write_webp(clip, tmp_path / "g.webp")).shape[1:] \
            == (emit.WEBP_SIZE, emit.WEBP_SIZE)
        assert emit.read_clip(emit.write_gif(clip, tmp_path / "g.gif")).shape[1:] \
            == (emit.GIF_SIZE, emit.GIF_SIZE)

    @pytest.mark.parametrize("chord", [(2, 3), (5, 9), (2, 12), (7, 10), (6, 12)])
    def test_a_spread_of_chords_survives_gif(self, chord, tmp_path):
        source = warp.envelope_clip(contour.render_control_image(chord, size=512))
        path = emit.write_gif(source, tmp_path / f"{chord}.gif")
        assert verify.accept(emit.read_clip(path), chord).accepted


def compose(text: str, size: int = 256) -> np.ndarray:
    """Build a full message the way runtime will: sentinel, characters, sentinel."""
    chords = ([C.SENTINEL_CHORD]
              + [CHORD_BY_SYMBOL[s] for s in text]
              + [C.SENTINEL_CHORD])
    return np.concatenate([
        warp.envelope_clip(contour.render_control_image(c, size=size))
        for c in chords
    ])


def read_message(path) -> str:
    return decode(np.stack([ring.radius_profile(f) for f in emit.read_clip(path)]))


class TestMessages:
    """The end of the line: text in, file on disk, text back out.

    Every earlier test works on arrays that never left memory. These are the
    only ones that put the message through an actual image file -- compression,
    palette quantization, frame merging and all -- and ask for it back.
    """

    @pytest.mark.parametrize("writer", [emit.write_webp, emit.write_gif])
    def test_a_word_survives_being_written_and_read(self, writer, tmp_path):
        assert read_message(writer(compose("OK 42"), tmp_path / "w.out")) == "OK 42"

    @pytest.mark.parametrize("writer", [emit.write_webp, emit.write_gif])
    def test_the_whole_alphabet_survives(self, writer, tmp_path):
        """Every character the format can express, in one file.

        At 256 px this is the smallest, most quantized form anything ships in,
        and it is exhaustive rather than sampled: if a character cannot make it
        through here, it cannot be shipped at all.
        """
        alphabet = ''.join(sorted(CHORD_BY_SYMBOL))
        path = writer(compose(alphabet), tmp_path / "all.out")
        assert read_message(path) == alphabet

    def test_leading_and_trailing_spaces_survive(self, tmp_path):
        """Space is a chord like any other now, so this must be exact."""
        assert read_message(emit.write_gif(compose("  A B  "), tmp_path / "s.gif")) \
            == "  A B  "


class TestRejections:
    def test_an_empty_clip_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="no frames"):
            emit.write_webp(np.zeros((0, 8, 8), dtype=np.uint8), tmp_path / "x.webp")

    def test_a_single_image_is_not_a_clip(self, tmp_path):
        with pytest.raises(ValueError, match="stack of frames"):
            emit.write_webp(np.zeros((8, 8), dtype=np.uint8), tmp_path / "y.webp")
