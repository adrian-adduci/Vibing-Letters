"""Generating candidates, and refusing to trust any of them.

Nothing here touches the network. The generation and download calls are the two
injected seams in `stylize`, which exists partly so that the only paid call in
the project is not also the only untestable one. What is tested is everything
around it: that the request is shaped correctly, that every returned image is
judged before a human sees it, and that failures are surfaced rather than
quietly dropped.
"""
import cv2
import numpy as np
import pytest

from src.assets import contour, stylize

CHORD = (3, 8)


class FakeEndpoint:
    """Stands in for fal: records the requests, returns images on demand."""

    def __init__(self, produce=None):
        self.requests = []
        self.uploads = []
        # By default every seed comes back as a faithful render of the chord.
        self.produce = produce or (lambda seed: contour.render_control_image(
            CHORD, size=256))

    def upload(self, path):
        self.uploads.append(path)
        return f"https://cdn.example/{path.name}"

    def submit(self, endpoint, arguments):
        self.requests.append((endpoint, arguments))
        return [f"https://cdn.example/out_{arguments['seed']}.png"]

    def fetch(self, url, destination):
        seed = int(url.rsplit('_', 1)[1].split('.')[0])
        cv2.imwrite(str(destination), self.produce(seed))
        return destination

    def run(self, tmp_path, **kwargs):
        return stylize.stylize(
            CHORD, tmp_path, submit=self.submit, fetch=self.fetch,
            upload=self.upload, **kwargs,
        )


class TestTheRequest:
    def test_the_control_image_is_written_and_uploaded(self, tmp_path):
        endpoint = FakeEndpoint()
        endpoint.run(tmp_path, seeds=(1,))

        control = tmp_path / "control_03_08.png"
        assert control.is_file()
        assert endpoint.uploads == [control]

        written = cv2.imread(str(control), cv2.IMREAD_GRAYSCALE)
        assert written.shape == (stylize.SIZE, stylize.SIZE)

    def test_the_control_image_is_uploaded_once_for_all_seeds(self, tmp_path):
        """Four candidates are four generations, not four uploads."""
        endpoint = FakeEndpoint()
        endpoint.run(tmp_path, seeds=(1, 2, 3, 4))
        assert len(endpoint.uploads) == 1
        assert len(endpoint.requests) == 4

    def test_each_seed_is_sent_verbatim(self, tmp_path):
        """Distinct seeds are the only source of variety here, so a request that
        dropped or reused them would return four copies of one picture."""
        endpoint = FakeEndpoint()
        endpoint.run(tmp_path, seeds=(11, 22, 33))
        assert [args["seed"] for _, args in endpoint.requests] == [11, 22, 33]

    def test_the_request_carries_the_uploaded_control_image(self, tmp_path):
        endpoint = FakeEndpoint()
        endpoint.run(tmp_path, seeds=(1,))
        _, arguments = endpoint.requests[0]
        assert arguments["control_lora_image_url"] == endpoint.uploads[0].name.join(
            ["https://cdn.example/", ""]
        )
        assert arguments["image_size"] == {"width": stylize.SIZE, "height": stylize.SIZE}
        assert arguments["output_format"] == "png"

    def test_polarity_and_preprocessing_are_not_knobs(self, tmp_path):
        """Both were probed and both changed nothing.

        Black-on-white and white-on-black returned pixel-identical output,
        because canny does not care which side of an edge is dark, and toggling
        `preprocess_depth` changed nothing either. Sending the flag would imply
        it does something.
        """
        endpoint = FakeEndpoint()
        endpoint.run(tmp_path, seeds=(1,))
        assert "preprocess_depth" not in endpoint.requests[0][1]

    def test_the_prompt_is_overridable(self, tmp_path):
        endpoint = FakeEndpoint()
        endpoint.run(tmp_path, seeds=(1,), prompt="something else entirely")
        assert endpoint.requests[0][1]["prompt"] == "something else entirely"


class TestJudging:
    def test_a_faithful_candidate_is_usable(self, tmp_path):
        candidate, = FakeEndpoint().run(tmp_path, seeds=(1,))
        assert candidate.usable
        assert candidate.verdict.peak.chord == CHORD
        assert candidate.seed == 1

    def test_a_candidate_of_the_wrong_shape_is_marked_unusable(self, tmp_path):
        """The failure the gate exists for: something beautiful and wrong."""
        endpoint = FakeEndpoint(
            produce=lambda seed: contour.render_control_image((5, 11), size=256))
        candidate, = endpoint.run(tmp_path, seeds=(1,))
        assert not candidate.usable
        assert candidate.verdict.peak.chord == (5, 11)

    def test_a_plain_circle_is_marked_unusable(self, tmp_path):
        def circle(seed):
            canvas = np.full((256, 256), 255, dtype=np.uint8)
            cv2.circle(canvas, (128, 128), 90, 0, 2, cv2.LINE_AA)
            return canvas

        candidate, = FakeEndpoint(produce=circle).run(tmp_path, seeds=(1,))
        assert not candidate.usable
        assert "silent" in candidate.verdict.reasons[0]

    def test_failures_are_returned_not_dropped(self, tmp_path):
        """A run where everything failed is a signal about the prompt.

        Returning only the usable ones would turn that into an empty list, which
        reads as 'nothing to choose from' rather than 'the prompt is wrong'.
        """
        endpoint = FakeEndpoint(
            produce=lambda seed: contour.render_control_image(
                CHORD if seed % 2 else (5, 11), size=256))
        candidates = endpoint.run(tmp_path, seeds=(1, 2, 3, 4))
        assert len(candidates) == 4
        assert [c.usable for c in candidates] == [True, False, True, False]

    def test_candidates_are_saved_under_distinct_names(self, tmp_path):
        candidates = FakeEndpoint().run(tmp_path, seeds=(1, 2, 3))
        paths = [c.path for c in candidates]
        assert len({p.name for p in paths}) == 3
        assert all(p.is_file() for p in paths)


class TestRefusals:
    def test_no_seeds_is_an_error(self, tmp_path):
        with pytest.raises(ValueError, match="at least one seed"):
            FakeEndpoint().run(tmp_path, seeds=())

    def test_an_unreadable_download_is_an_error(self, tmp_path):
        """A truncated download must not be judged as though it were an image."""
        def broken(url, destination):
            destination.write_bytes(b"not an image")
            return destination

        endpoint = FakeEndpoint()
        with pytest.raises(OSError, match="could not be read back"):
            stylize.stylize(CHORD, tmp_path, seeds=(1,),
                            submit=endpoint.submit, fetch=broken,
                            upload=endpoint.upload)


class TestKeyLookup:
    def test_an_existing_environment_key_wins(self, monkeypatch):
        monkeypatch.setenv('FAL_KEY', 'already-set')
        assert stylize.load_key() == 'already-set'

    def test_a_key_is_found_by_walking_upward(self, tmp_path, monkeypatch):
        """Worktrees do not share the checkout's untracked files.

        A key written once at the top of the repository has to be findable from
        a worktree several directories down, which is exactly where this runs.
        """
        monkeypatch.delenv('FAL_KEY', raising=False)
        (tmp_path / '.env').write_text('FAL_KEY=from-a-parent\n')
        deep = tmp_path / 'a' / 'b' / 'c'
        deep.mkdir(parents=True)
        assert stylize.load_key(start=deep) == 'from-a-parent'

    def test_a_missing_key_says_what_to_do(self, tmp_path, monkeypatch):
        monkeypatch.delenv('FAL_KEY', raising=False)
        with pytest.raises(RuntimeError, match=r"\.env"):
            stylize.load_key(start=tmp_path)


class TestSummary:
    def test_the_summary_names_the_failure(self, tmp_path):
        endpoint = FakeEndpoint(
            produce=lambda seed: contour.render_control_image(
                CHORD if seed == 1 else (5, 11), size=256))
        text = stylize.summarize(endpoint.run(tmp_path, seeds=(1, 2)))
        assert "1/2 usable" in text
        assert "OK" in text and "FAIL" in text
        assert "(5, 11)" in text
