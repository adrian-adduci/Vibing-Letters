"""Decoding must survive the distortions real assets will introduce."""
import numpy as np
import pytest

from src.codec import constants as C
from src.codec.message import decode, encode
from src.codec.spectrum import detect_chord
from src.codec.waveform import radius_profile

SENTENCE = "VIBING LETTERS"


def _high_frequency_noise(n_bins: int, amplitude: float, seed: int) -> np.ndarray:
    """Noise confined to modes above MAX_MODE, as design section 6 requires."""
    rng = np.random.default_rng(seed)
    theta = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=False)
    modes = range(C.MAX_MODE + 1, C.MAX_MODE + 20)
    return amplitude * sum(
        rng.normal() * np.cos(mode * theta + rng.uniform(0, 2 * np.pi))
        for mode in modes
    )


def test_band_limited_noise_does_not_break_decoding():
    """Perlin-style texture is safe if it stays out of the decoder's band."""
    frames = encode(SENTENCE).frames + _high_frequency_noise(C.N_BINS, 0.02, seed=1)
    assert decode(frames) == SENTENCE


def test_in_band_noise_destroys_confidence():
    """The band limit is load-bearing, not decorative.

    Injecting a third mode at the same strength as the chord's two produces an
    exact three-way tie. Note the chord returned is still (3, 7) - argsort has
    to break the tie somehow - so asserting on the chord would pass or fail by
    luck. Confidence is the honest signal: it collapses to 1.0, meaning the
    answer is indistinguishable from the runner-up.
    """
    profile = radius_profile((3, 7), C.AMPLITUDE)
    theta = np.linspace(0.0, 2.0 * np.pi, C.N_BINS, endpoint=False)
    corrupted = profile + 0.5 * C.AMPLITUDE * np.cos(5 * theta)
    assert detect_chord(corrupted)[1] == pytest.approx(1.0)


@pytest.mark.parametrize("scale", [0.25, 1.0, 7.5, 100.0])
def test_decoding_survives_rescaling(scale):
    assert decode(encode(SENTENCE).frames * scale) == SENTENCE


def test_decoding_survives_rotation():
    rotated = np.roll(encode(SENTENCE).frames, C.N_BINS // 5, axis=1)
    assert decode(rotated) == SENTENCE


def test_in_band_noise_does_not_fabricate_a_character_in_a_boundary():
    """The hazard the band-limited test above misses entirely.

    Every character, space included, is an excited ring with a chord's worth of
    margin. The quiet frames delimiting them have none: they carry no signal at
    all, so any in-band energy can push one over QUIET_THRESHOLD and invent a
    character mid-message. `active_mask` is the only thing between that and a
    corrupted decode, which makes this the one place the quiet threshold is
    load-bearing rather than merely convenient.
    """
    rng = np.random.default_rng(3)
    frames = encode("A B").frames.copy()
    quiet = np.all(np.isclose(frames, C.REST_RADIUS), axis=1)
    frames[quiet] += rng.normal(0.0, 0.002, frames[quiet].shape)
    assert quiet.sum() > 0
    assert decode(frames) == "A B"


def test_confidence_survives_realistic_noise():
    """On clean input confidence is ~4e15, so asserting > 10 there constrains
    nothing. Measured under additive Gaussian noise, 200 trials per sigma:
    0.005 -> median 83, 0.01 -> 43, 0.02 -> median 21 / min 11. Chord accuracy
    stayed perfect throughout, so confidence degrades well before correctness
    does. sigma=0.02 is therefore the point where this assertion actually bites.
    """
    rng = np.random.default_rng(0)
    noisy = radius_profile((3, 7), C.AMPLITUDE) + rng.normal(0.0, 0.02, C.N_BINS)
    chord, confidence = detect_chord(noisy)
    assert chord == (3, 7)
    assert confidence > 10.0
