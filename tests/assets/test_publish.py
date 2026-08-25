"""Producing the clips that actually ship.

The distinguishing claim of this stage is that verification happens on the
*written file*, after compression, resizing and frame merging, rather than on
the array that went in. Most of these tests exist to make sure that stays true.
"""
import json

import cv2
import numpy as np
import pytest

from src.assets import contour, emit, publish, verify
from src.codec import constants as C
from src.codec.chord_table import CHORD_BY_SYMBOL
from src.codec.message import decode
from src.vision import ring

CHORD = (3, 8)


@pytest.fixture
def still(tmp_path):
    path = tmp_path / "still.png"
    cv2.imwrite(str(path), contour.render_control_image(CHORD, size=512))
    return path


class TestPublishingOne:
    def test_a_good_still_produces_a_verified_clip(self, still, tmp_path):
        result = publish.publish_one("X", CHORD, still, tmp_path)
        assert result.ok
        assert result.path.is_file()
        assert result.confidence > C.MIN_CONFIDENCE

    def test_the_clip_decodes_from_the_file(self, still, tmp_path):
        result = publish.publish_one("X", CHORD, still, tmp_path)
        frames = emit.read_clip(result.path)
        assert verify_chord(frames) == CHORD

    def test_the_digest_is_of_the_file_that_shipped(self, still, tmp_path):
        result = publish.publish_one("X", CHORD, still, tmp_path)
        assert result.digest == publish.digest_of(result.path)
        assert len(result.digest) == 64

    def test_the_link_is_embedded(self, still, tmp_path):
        result = publish.publish_one("X", CHORD, still, tmp_path,
                                     url="https://example.org/d")
        assert emit.read_link(result.path) == "https://example.org/d"


class TestRefusing:
    def test_a_still_of_the_wrong_shape_is_not_published(self, tmp_path):
        path = tmp_path / "wrong.png"
        cv2.imwrite(str(path), contour.render_control_image((5, 11), size=512))
        result = publish.publish_one("X", CHORD, path, tmp_path)
        assert not result.ok
        assert result.reasons

    def test_a_rejected_clip_leaves_no_file_behind(self, tmp_path):
        """Runtime treats the presence of a file as a promise that it decodes.

        Leaving a rejected clip on disk would turn a caught failure into a
        shipped one, which is worse than never having checked.
        """
        path = tmp_path / "wrong.png"
        cv2.imwrite(str(path), contour.render_control_image((5, 11), size=512))
        publish.publish_one("X", CHORD, path, tmp_path)
        assert not (tmp_path / "X.webp").exists()

    def test_a_plain_circle_is_refused_before_warping(self, tmp_path):
        """There is no excitation to animate, and the warp says so rather than
        dividing a measurement of noise by a small number."""
        canvas = np.full((512, 512), 255, dtype=np.uint8)
        cv2.circle(canvas, (256, 256), 180, 0, 3, cv2.LINE_AA)
        path = tmp_path / "circle.png"
        cv2.imwrite(str(path), canvas)

        result = publish.publish_one("X", CHORD, path, tmp_path)
        assert not result.ok
        assert "circle" in result.reasons[0]

    def test_the_gate_runs_on_the_file_not_on_the_array(self, still, tmp_path):
        """The distinguishing claim of this whole stage.

        Every earlier check ran on arrays in memory; what ships is a compressed
        file that has been resized, quality-passed and frame-merged. Verifying
        the array and shipping the file would be verifying something else.

        The format turns out to be remarkably tough -- it still decodes at 64 px
        and quality 1 -- so making the two disagree takes real abuse. At 24 px
        and quality 1 the in-memory clip still passes while the written file
        reads back as chord (2, 3), which is SPACE. A wrong letter, from an
        asset whose array was fine.
        """
        result = publish.publish_one("X", CHORD, still, tmp_path,
                                     size=24, quality=1)
        assert not result.ok
        assert not (tmp_path / "X.webp").exists()

        # The array it was built from is unimpaired, which is the whole point.
        from src.assets import warp
        still_image = cv2.imread(str(still), cv2.IMREAD_COLOR)
        assert verify.accept(warp.envelope_clip(still_image), CHORD).accepted

    def test_an_unreadable_still_is_reported_not_raised(self, tmp_path):
        path = tmp_path / "broken.png"
        path.write_bytes(b"not an image")
        result = publish.publish_one("X", CHORD, path, tmp_path)
        assert not result.ok
        assert "could not be read" in result.reasons[0]


class TestPublishingAll:
    @pytest.fixture
    def chosen(self, tmp_path):
        """Three real symbols, plus one that will fail."""
        stills = tmp_path / "stills"
        stills.mkdir()
        entries = []
        for symbol in "ABC":
            chord = CHORD_BY_SYMBOL[symbol]
            path = stills / f"{symbol}.png"
            cv2.imwrite(str(path), contour.render_control_image(chord, size=512))
            entries.append((symbol, chord, path))

        bad = stills / "D.png"
        cv2.imwrite(str(bad), contour.render_control_image((5, 11), size=512))
        entries.append(("D", CHORD_BY_SYMBOL["D"], bad))
        return entries

    def test_good_symbols_publish_and_bad_ones_do_not(self, chosen, tmp_path):
        results = publish.publish_all(chosen, tmp_path / "clips")
        assert [r.ok for r in results] == [True, True, True, False]

    def test_the_manifest_records_hashes_and_failures(self, chosen, tmp_path):
        publish.publish_all(chosen, tmp_path / "clips")
        manifest = json.loads((tmp_path / "clips" / "assets.json").read_text())

        assert manifest["complete"] is False
        assert manifest["decoder_url"] == emit.DECODER_URL
        published = [s for s in manifest["symbols"] if s["verified"]]
        assert len(published) == 3
        assert all(len(s["sha256"]) == 64 for s in published)
        assert [s for s in manifest["symbols"] if not s["verified"]][0]["clip"] is None

    def test_the_report_names_what_failed(self, chosen, tmp_path):
        text = publish.report(publish.publish_all(chosen, tmp_path / "clips"))
        assert "3/4" in text
        assert "FAILED D" in text

    def test_published_clips_concatenate_into_a_readable_message(
        self, chosen, tmp_path
    ):
        """The end of the whole pipeline.

        Three published files, read off disk, concatenated with sentinels, and
        decoded as a message. Nothing here touches an in-memory array that did
        not come out of a file first.
        """
        results = publish.publish_all(chosen, tmp_path / "clips")
        good = {r.name: r for r in results if r.ok}

        sentinel_still = tmp_path / "sentinel.png"
        cv2.imwrite(str(sentinel_still),
                    contour.render_control_image(C.SENTINEL_CHORD, size=512))
        sentinel = publish.publish_one("S", C.SENTINEL_CHORD, sentinel_still,
                                       tmp_path / "clips")

        order = [sentinel.path] + [good[s].path for s in "ABC"] + [sentinel.path]
        frames = np.concatenate([emit.read_clip(p) for p in order])
        profiles = np.stack([ring.radius_profile(f) for f in frames])
        assert decode(profiles) == "ABC"


def verify_chord(frames):
    from src.codec.spectrum import detect_chord
    peak = max(frames, key=lambda f: ring.radius_profile(f).std())
    return detect_chord(ring.radius_profile(peak))[0]
