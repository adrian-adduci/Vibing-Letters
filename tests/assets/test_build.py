"""Orchestrating the whole build.

The units this drives are tested elsewhere; what matters here is the
orchestration itself -- that every symbol is covered exactly once, that a
resumed run does not pay twice, that selection can only ever pick something the
gate passed, and that failures reach the manifest instead of vanishing.

No network. Generation is the injected seam.
"""
import json

import cv2
import numpy as np
import pytest

from src.assets import build, contour, stylize, verify
from src.codec import constants as C
from src.codec.chord_table import CHORD_BY_SYMBOL

CHORD = (3, 8)


def fake_still(chord, seed=1, size=256):
    """A candidate whose quality *improves* with the seed.

    Two things are deliberate. Candidates must differ at all -- a stand-in that
    returned one picture for every seed gives every candidate identical
    confidence, and nothing about how selection ranks them could be tested. And
    the best one must not be the first one, or "pick the most confident" is
    indistinguishable from "pick whichever arrived first". Noise falling as
    24/seed arranges both: the last seed wins.
    """
    base = contour.render_control_image(chord, size=size)
    noise = np.random.default_rng(seed).normal(0.0, 24.0 / seed, base.shape)
    return np.clip(base.astype(float) + noise, 0, 255).astype(np.uint8)


def make_generator(shape_for=lambda chord, seed: None, log=None):
    """A stand-in for `stylize.stylize` that writes files and judges them.

    `shape_for` returns the chord a given (chord, seed) should actually come
    back as, so a candidate can be made to fail exactly the way a real one would
    -- by being the wrong shape, not by being marked wrong.
    """
    def generate(chord, out_dir, seeds=(1,), **kwargs):
        if log is not None:
            log.append((chord, tuple(seeds)))
        stem = f"{chord[0]:02d}_{chord[1]:02d}"
        out = []
        for seed in seeds:
            actual = shape_for(chord, seed) or chord
            path = out_dir / f"cand_{stem}_s{seed}.png"
            cv2.imwrite(str(path), fake_still(actual, seed))
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            out.append(stylize.Candidate(path, seed,
                                         verify.accept_still(image, chord)))
        return out
    return generate


class TestCoverage:
    def test_every_symbol_and_the_sentinel_are_targeted(self):
        entries = build.targets()
        assert len(entries) == len(CHORD_BY_SYMBOL) + 1
        assert entries[-1][1] == C.SENTINEL_CHORD

    def test_no_chord_is_built_twice(self):
        """Two symbols sharing a chord would make the alphabet ambiguous, and
        the build is the last place that could notice."""
        chords = [chord for _, chord in build.targets()]
        assert len(set(chords)) == len(chords)

    def test_names_are_safe_to_put_in_a_filename(self):
        """The alphabet contains a space, a comma, a full stop, an apostrophe
        and a question mark. None of those belong in a path."""
        names = [name for name, _ in build.targets()]
        assert build.symbol_name(' ') == 'SPACE'
        assert build.symbol_name('A') == 'A'
        assert build.symbol_name('7') == '7'
        assert build.symbol_name('?') == 'U003F'
        assert all(name.isalnum() or name.isidentifier() for name in names)
        assert len(set(names)) == len(names)


class TestResuming:
    def test_a_candidate_already_on_disk_is_not_regenerated(self, tmp_path):
        """An interrupted run should cost nothing to finish. Every regenerated
        candidate is money."""
        log = []
        generate = make_generator(log=log)

        build.build_symbol('X', CHORD, tmp_path, seeds=(1, 2), generate=generate)
        assert log == [(CHORD, (1, 2))]

        build.build_symbol('X', CHORD, tmp_path, seeds=(1, 2), generate=generate)
        assert log == [(CHORD, (1, 2))]          # nothing new was asked for

    def test_only_the_missing_seeds_are_requested(self, tmp_path):
        log = []
        generate = make_generator(log=log)
        build.build_symbol('X', CHORD, tmp_path, seeds=(1, 2), generate=generate)
        build.build_symbol('X', CHORD, tmp_path, seeds=(1, 2, 3, 4),
                           generate=generate)
        assert log[-1] == (CHORD, (3, 4))

    def test_existing_candidates_are_rejudged_not_trusted(self, tmp_path):
        """A resumed run must not inherit a verdict from a gate that has since
        changed. Re-judging costs a fraction of a second."""
        generate = make_generator()
        build.build_symbol('X', CHORD, tmp_path, seeds=(1,), generate=generate)

        # Replace the file on disk with the wrong shape entirely.
        path = tmp_path / f"cand_{CHORD[0]:02d}_{CHORD[1]:02d}_s1.png"
        cv2.imwrite(str(path), fake_still((5, 11), 1))

        result = build.build_symbol('X', CHORD, tmp_path, seeds=(1,),
                                    generate=generate)
        assert not result.candidates[0].usable
        assert result.chosen is None

    def test_an_unreadable_file_on_disk_is_regenerated(self, tmp_path):
        log = []
        generate = make_generator(log=log)
        path = tmp_path / f"cand_{CHORD[0]:02d}_{CHORD[1]:02d}_s1.png"
        path.write_bytes(b"truncated")
        build.build_symbol('X', CHORD, tmp_path, seeds=(1,), generate=generate)
        assert log == [(CHORD, (1,))]


class TestSelection:
    def test_the_most_confident_passing_candidate_wins(self, tmp_path):
        result = build.build_symbol('X', CHORD, tmp_path, seeds=(1, 2, 3),
                                    generate=make_generator())
        confidences = [c.verdict.peak.confidence for c in result.candidates]
        # Distinct, and the winner is deliberately last. Without both of those
        # the test cannot tell picking the best from picking the worst, or from
        # picking whichever candidate happened to arrive first.
        assert len(set(confidences)) == 3
        assert confidences.index(max(confidences)) == len(confidences) - 1
        assert result.chosen is result.candidates[-1]

    def test_a_failing_candidate_can_never_be_chosen(self, tmp_path):
        """Selection ranks by confidence, and a wrong shape can be confident.

        Ranking alone would happily pick a candidate the gate rejected, so the
        filter has to come first.
        """
        generate = make_generator(
            shape_for=lambda chord, seed: (5, 11) if seed == 1 else None)
        result = build.build_symbol('X', CHORD, tmp_path, seeds=(1, 2),
                                    generate=generate)
        assert not result.candidates[0].usable
        assert result.chosen is result.candidates[1]

    def test_no_choice_is_recorded_when_nothing_passed(self, tmp_path):
        generate = make_generator(shape_for=lambda chord, seed: (5, 11))
        result = build.build_symbol('X', CHORD, tmp_path, seeds=(1, 2),
                                    generate=generate)
        assert result.chosen is None


class TestManifest:
    def test_the_manifest_records_every_symbol(self, tmp_path):
        results = build.build_all(tmp_path, seeds=(1,),
                                  generate=make_generator())
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert len(manifest["symbols"]) == len(CHORD_BY_SYMBOL) + 1
        assert manifest["complete"] is True

    def test_failures_reach_the_manifest(self, tmp_path):
        """A chord where nothing passed is a fact about the prompt. It has to be
        visible without rerunning the build."""
        generate = make_generator(
            shape_for=lambda chord, seed: (5, 11) if chord == (2, 3) else None)
        build.build_all(tmp_path, seeds=(1,), generate=generate)

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["complete"] is False
        broken = [s for s in manifest["symbols"] if s["chord"] == [2, 3]]
        assert broken[0]["chosen"] is None
        assert broken[0]["candidates"][0]["read_as"] == [5, 11]
        assert broken[0]["candidates"][0]["reasons"]

    def test_the_manifest_is_valid_json_with_the_format_pinned(self, tmp_path):
        build.build_all(tmp_path, seeds=(1,), generate=make_generator())
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["frames_per_char"] == C.FRAMES_PER_CHAR
        assert manifest["min_confidence"] == C.MIN_CONFIDENCE
        assert manifest["alphabet_size"] == len(CHORD_BY_SYMBOL)


class TestReport:
    def test_a_complete_build_reports_its_weakest_link(self, tmp_path):
        results = build.build_all(tmp_path, seeds=(1,), generate=make_generator())
        text = build.report(results)
        assert "44/44" in text
        assert "lowest confidence" in text

    def test_an_incomplete_build_names_what_is_missing(self, tmp_path):
        generate = make_generator(
            shape_for=lambda chord, seed: (5, 11) if chord == (2, 3) else None)
        text = build.report(build.build_all(tmp_path, seeds=(1,),
                                            generate=generate))
        assert "43/44" in text
        assert "SPACE" in text
