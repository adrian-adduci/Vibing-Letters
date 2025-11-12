"""Perlin noise-based vibration effects for Vibing Letters.

This module provides the PerlinVibrator class for generating smooth,
organic vibration effects using Perlin noise instead of random jitter.
"""

import numpy as np
from perlin_noise import PerlinNoise
from typing import Optional

from ..utils.logger import get_logger


logger = get_logger(__name__)


class PerlinVibrator:
    """Generates smooth vibration effects using Perlin noise.

    Perlin noise creates continuous, natural-looking variations that are
    more organic than random jitter. This class applies Perlin noise to
    shape coordinates to create a "vibrating string" effect.
    """

    def __init__(
        self,
        octaves: int = 4,
        persistence: float = 0.5,
        scale: float = 0.3,
        seed: Optional[int] = None
    ):
        """Initialize the Perlin vibrator.

        Args:
            octaves: Number of octaves for Perlin noise (default: 4)
                    Higher values add finer detail
            persistence: Amplitude decrease per octave (default: 0.5)
                        Controls how quickly detail diminishes
            scale: Scale factor for noise values (default: 0.3)
                  Controls vibration amplitude in pixels
            seed: Random seed for reproducibility (default: None for random)

        Raises:
            ValueError: If parameters are invalid
        """
        if octaves < 1:
            raise ValueError(f"octaves must be >= 1, got {octaves}")
        if not 0.0 <= persistence <= 1.0:
            raise ValueError(f"persistence must be in [0, 1], got {persistence}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")

        self.octaves = octaves
        self.persistence = persistence
        self.scale = scale
        self.seed = seed

        # Create two Perlin noise generators (one for x, one for y)
        self.noise_x = PerlinNoise(octaves=octaves, seed=seed)
        self.noise_y = PerlinNoise(octaves=octaves, seed=(seed + 1) if seed is not None else None)

        logger.debug(
            f"PerlinVibrator initialized: octaves={octaves}, "
            f"persistence={persistence}, scale={scale}, seed={seed}"
        )

    def vibrate(
        self,
        points: np.ndarray,
        time: float,
        frequency: float = 2.5,
        amplitude_override: Optional[float] = None
    ) -> np.ndarray:
        """Apply Perlin noise vibration to points.

        Args:
            points: Input points as Nx2 array of (x, y) coordinates
            time: Time parameter for animation (controls noise sampling)
            frequency: Frequency multiplier for vibration (default: 2.5)
                      Higher values create faster vibration
            amplitude_override: Override the scale parameter (default: None uses self.scale)

        Returns:
            np.ndarray: Vibrated points as Nx2 array

        Raises:
            ValueError: If points are invalid
        """
        if points is None or points.size == 0:
            raise ValueError("points is None or empty")

        if len(points.shape) != 2 or points.shape[1] != 2:
            raise ValueError(f"points must be Nx2 array, got shape {points.shape}")

        amplitude = amplitude_override if amplitude_override is not None else self.scale

        logger.debug(
            f"Applying vibration to {len(points)} points: "
            f"time={time:.3f}, frequency={frequency:.2f}, amplitude={amplitude:.3f}"
        )

        # Generate noise offsets for each point
        vibrated = points.copy()
        n_points = len(points)

        for i in range(n_points):
            # Use point index and time to sample different parts of noise space
            # This ensures each point vibrates differently but smoothly over time
            noise_x_coord = i / n_points + time * frequency
            noise_y_coord = i / n_points + time * frequency + 100.0  # Offset for independence

            # Sample Perlin noise (-1 to 1 range)
            offset_x = self.noise_x([noise_x_coord]) * amplitude
            offset_y = self.noise_y([noise_y_coord]) * amplitude

            # Apply offsets
            vibrated[i, 0] += offset_x
            vibrated[i, 1] += offset_y

        return vibrated

    def vibrate_sequence(
        self,
        points: np.ndarray,
        n_frames: int,
        frequency: float = 2.5,
        amplitude_override: Optional[float] = None,
        start_time: float = 0.0
    ) -> list[np.ndarray]:
        """Generate a sequence of vibrated frames.

        Args:
            points: Input points as Nx2 array
            n_frames: Number of frames to generate
            frequency: Frequency multiplier for vibration (default: 2.5)
            amplitude_override: Override the scale parameter (default: None)
            start_time: Starting time value (default: 0.0)

        Returns:
            list[np.ndarray]: List of vibrated point arrays

        Raises:
            ValueError: If parameters are invalid
        """
        if n_frames < 1:
            raise ValueError(f"n_frames must be >= 1, got {n_frames}")

        logger.info(f"Generating vibration sequence: {n_frames} frames")

        frames = []
        for frame_idx in range(n_frames):
            time = start_time + frame_idx / n_frames
            vibrated = self.vibrate(points, time, frequency, amplitude_override)
            frames.append(vibrated)

        logger.info(f"Generated {len(frames)} vibrated frames")
        return frames

    def create_wave_pattern(
        self,
        points: np.ndarray,
        n_cycles: int = 3,
        frames_per_cycle: int = 10,
        frequency: float = 2.5,
        amplitude_override: Optional[float] = None
    ) -> list[np.ndarray]:
        """Create a repeating wave pattern of vibration.

        Args:
            points: Input points as Nx2 array
            n_cycles: Number of vibration cycles (default: 3)
            frames_per_cycle: Frames per cycle (default: 10)
            frequency: Frequency multiplier (default: 2.5)
            amplitude_override: Override the scale parameter (default: None)

        Returns:
            list[np.ndarray]: List of vibrated point arrays forming a wave pattern
        """
        total_frames = n_cycles * frames_per_cycle

        logger.info(
            f"Creating wave pattern: {n_cycles} cycles, "
            f"{frames_per_cycle} frames/cycle, total={total_frames} frames"
        )

        return self.vibrate_sequence(
            points,
            n_frames=total_frames,
            frequency=frequency,
            amplitude_override=amplitude_override
        )

    def blend_vibration(
        self,
        points: np.ndarray,
        time: float,
        blend_factor: float,
        frequency: float = 2.5,
        amplitude_override: Optional[float] = None
    ) -> np.ndarray:
        """Apply partial vibration with blending.

        Useful for gradually introducing or fading out vibration effects.

        Args:
            points: Input points as Nx2 array
            time: Time parameter for animation
            blend_factor: Blending factor in [0, 1] (0=no vibration, 1=full vibration)
            frequency: Frequency multiplier (default: 2.5)
            amplitude_override: Override the scale parameter (default: None)

        Returns:
            np.ndarray: Blended vibrated points

        Raises:
            ValueError: If blend_factor is out of range
        """
        if not 0.0 <= blend_factor <= 1.0:
            raise ValueError(f"blend_factor must be in [0, 1], got {blend_factor}")

        # Generate fully vibrated points
        vibrated = self.vibrate(points, time, frequency, amplitude_override)

        # Blend between original and vibrated
        blended = (1.0 - blend_factor) * points + blend_factor * vibrated

        return blended

    def reset_noise_generators(self, seed: Optional[int] = None):
        """Reset the Perlin noise generators with a new seed.

        Args:
            seed: New random seed (default: None for random)
        """
        self.seed = seed
        self.noise_x = PerlinNoise(octaves=self.octaves, seed=seed)
        self.noise_y = PerlinNoise(octaves=self.octaves, seed=(seed + 1) if seed is not None else None)

        logger.debug(f"Noise generators reset with seed={seed}")


class HybridVibrator:
    """Combines Perlin noise with random jitter for hybrid vibration effects.

    This class allows mixing smooth Perlin noise with sharp random jitter
    to create varied vibration styles.
    """

    def __init__(
        self,
        perlin_vibrator: PerlinVibrator,
        jitter_strength: float = 1.0,
        perlin_weight: float = 0.7
    ):
        """Initialize the hybrid vibrator.

        Args:
            perlin_vibrator: PerlinVibrator instance to use
            jitter_strength: Strength of random jitter in pixels (default: 1.0)
            perlin_weight: Weight of Perlin noise vs jitter in [0, 1] (default: 0.7)
                          0.0 = pure jitter, 1.0 = pure Perlin
        """
        if not 0.0 <= perlin_weight <= 1.0:
            raise ValueError(f"perlin_weight must be in [0, 1], got {perlin_weight}")

        self.perlin_vibrator = perlin_vibrator
        self.jitter_strength = jitter_strength
        self.perlin_weight = perlin_weight
        self.jitter_weight = 1.0 - perlin_weight

        logger.debug(
            f"HybridVibrator initialized: jitter_strength={jitter_strength}, "
            f"perlin_weight={perlin_weight}"
        )

    def vibrate(
        self,
        points: np.ndarray,
        time: float,
        frequency: float = 2.5
    ) -> np.ndarray:
        """Apply hybrid vibration (Perlin + jitter) to points.

        Args:
            points: Input points as Nx2 array
            time: Time parameter for animation
            frequency: Frequency multiplier (default: 2.5)

        Returns:
            np.ndarray: Vibrated points
        """
        # Get Perlin noise vibration
        perlin_vibrated = self.perlin_vibrator.vibrate(points, time, frequency)

        # Generate random jitter
        jitter = np.random.normal(0, self.jitter_strength, size=points.shape)
        jitter_vibrated = points + jitter

        # Blend both effects
        hybrid = (
            self.perlin_weight * perlin_vibrated +
            self.jitter_weight * jitter_vibrated
        )

        return hybrid
