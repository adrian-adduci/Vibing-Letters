"""Morph engine for orchestrating shape morphing in Vibing Letters.

This module provides the MorphEngine class which orchestrates all morphing
operations including alignment, interpolation, vibration, and easing.
"""

import numpy as np
from typing import Optional, List

from .contour_extractor import ContourExtractor
from .procrustes_aligner import ProcrustesAligner
from .perlin_vibrator import PerlinVibrator
from .easing_curve import EasingCurve
from ..config.morph_config import MorphConfig
from ..utils.logger import get_logger, PerformanceTimer


logger = get_logger(__name__)


class MorphEngine:
    """Orchestrates the complete morphing pipeline.

    This class coordinates ContourExtractor, ProcrustesAligner, PerlinVibrator,
    and EasingCurve to create smooth morphing animations with vibration effects.
    """

    def __init__(
        self,
        config: Optional[MorphConfig] = None,
        contour_extractor: Optional[ContourExtractor] = None,
        aligner: Optional[ProcrustesAligner] = None,
        vibrator: Optional[PerlinVibrator] = None,
        easing: Optional[EasingCurve] = None
    ):
        """Initialize the morph engine.

        Args:
            config: Morphing configuration (uses default if None)
            contour_extractor: Custom ContourExtractor (creates default if None)
            aligner: Custom ProcrustesAligner (creates default if None)
            vibrator: Custom PerlinVibrator (creates default if None)
            easing: Custom EasingCurve (creates default if None)
        """
        from ..config.morph_config import DEFAULT_CONFIG

        self.config = config or DEFAULT_CONFIG

        # Initialize components with dependency injection
        self.contour_extractor = contour_extractor or ContourExtractor(
            n_points=self.config.n_points,
            threshold=127
        )

        self.aligner = aligner or ProcrustesAligner(
            allow_scaling=self.config.procrustes_scaling
        )

        self.vibrator = vibrator or PerlinVibrator(
            octaves=self.config.noise_octaves,
            persistence=self.config.noise_persistence,
            scale=self.config.noise_scale
        )

        self.easing = easing or EasingCurve(
            easing_type=self.config.easing_type
        )

        logger.info(f"MorphEngine initialized with config: {self.config.easing_type}")

    def create_morph_sequence(
        self,
        base_shape: np.ndarray,
        target_shape: np.ndarray,
        n_steps: int,
        use_alignment: Optional[bool] = None
    ) -> List[np.ndarray]:
        """Create a morphing sequence from base to target shape.

        Args:
            base_shape: Starting shape as Nx2 array
            target_shape: Target shape as Nx2 array
            n_steps: Number of interpolation steps
            use_alignment: Override config's use_procrustes setting (None uses config)

        Returns:
            List[np.ndarray]: List of morphed shapes

        Raises:
            ValueError: If shapes are invalid
        """
        if base_shape.shape != target_shape.shape:
            raise ValueError(
                f"Shapes must have same dimensions: "
                f"{base_shape.shape} != {target_shape.shape}"
            )

        if n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {n_steps}")

        use_alignment = use_alignment if use_alignment is not None else self.config.use_procrustes

        logger.info(
            f"Creating morph sequence: {n_steps} steps, "
            f"alignment={'enabled' if use_alignment else 'disabled'}"
        )

        with PerformanceTimer(logger, "Morph sequence generation"):
            # Align target to base if requested
            if use_alignment:
                aligned_target = self.aligner.align(target_shape, base_shape)
                logger.debug("Applied Procrustes alignment")
            else:
                aligned_target = target_shape

            # Generate interpolation sequence using easing
            sequence = self.easing.generate_interpolation_sequence(
                base_shape,
                aligned_target,
                n_steps
            )

        logger.info(f"Generated {len(sequence)} morphed frames")
        return sequence

    def create_overshoot_sequence(
        self,
        base_shape: np.ndarray,
        target_shape: np.ndarray,
        overshoot_values: Optional[List[float]] = None,
        use_alignment: Optional[bool] = None
    ) -> List[np.ndarray]:
        """Create a morphing sequence with overshoot effect.

        The sequence goes from base → target+overshoot → target, creating
        a bouncy, dynamic effect.

        Args:
            base_shape: Starting shape as Nx2 array
            target_shape: Target shape as Nx2 array
            overshoot_values: List of t values including overshoot (None uses config)
            use_alignment: Override config's use_procrustes setting

        Returns:
            List[np.ndarray]: List of morphed shapes with overshoot

        Example:
            overshoot_values = [0.0, 1.1, 1.0] creates:
            - Frame 0: base (t=0.0)
            - Frame 1: 110% toward target (t=1.1)
            - Frame 2: target (t=1.0)
        """
        overshoot_values = overshoot_values or self.config.overshoot_values
        use_alignment = use_alignment if use_alignment is not None else self.config.use_procrustes

        logger.info(f"Creating overshoot sequence with values: {overshoot_values}")

        # Align if requested
        if use_alignment:
            aligned_target = self.aligner.align(target_shape, base_shape)
        else:
            aligned_target = target_shape

        # Generate frames for each overshoot value
        sequence = []
        for t in overshoot_values:
            t_eased = self.easing.ease(np.clip(t, 0.0, 1.0))

            # Allow overshoot by using raw t value for interpolation
            morphed = (1.0 - t) * base_shape + t * aligned_target
            sequence.append(morphed)

        logger.info(f"Generated {len(sequence)} overshoot frames")
        return sequence

    def create_vibration_sequence(
        self,
        shape: np.ndarray,
        n_cycles: Optional[int] = None,
        frequency: Optional[float] = None
    ) -> List[np.ndarray]:
        """Create a vibration sequence for a shape.

        Args:
            shape: Shape to vibrate as Nx2 array
            n_cycles: Number of vibration cycles (None uses config)
            frequency: Vibration frequency (None uses config)

        Returns:
            List[np.ndarray]: List of vibrated shapes
        """
        n_cycles = n_cycles or self.config.vibration_cycles
        frequency = frequency or self.config.vibration_frequency

        logger.info(f"Creating vibration sequence: {n_cycles} cycles, frequency={frequency}")

        sequence = self.vibrator.create_wave_pattern(
            shape,
            n_cycles=n_cycles,
            frames_per_cycle=1,  # One frame per cycle
            frequency=frequency
        )

        logger.info(f"Generated {len(sequence)} vibration frames")
        return sequence

    def create_full_animation_sequence(
        self,
        base_shape: np.ndarray,
        target_shape: np.ndarray
    ) -> tuple[List[np.ndarray], List[int]]:
        """Create a complete animation sequence with all effects.

        This creates the full animation including:
        1. Static base shape (start)
        2. Morph with overshoot to target
        3. Vibration cycles
        4. Reverse morph back to base
        5. Static base shape (end)
        6. Blank frame (pause)

        Args:
            base_shape: Starting shape as Nx2 array
            target_shape: Target shape as Nx2 array

        Returns:
            tuple[List[np.ndarray], List[int]]:
                - List of shapes for each frame
                - List of frame durations in milliseconds
        """
        logger.info("Creating full animation sequence")

        frames = []
        durations = []

        with PerformanceTimer(logger, "Full animation generation"):
            # 1. Static start frames
            for _ in range(self.config.static_start_frames):
                frames.append(base_shape.copy())
                durations.append(self.config.frame_duration_ms)
            logger.debug(f"Added {self.config.static_start_frames} static start frames")

            # 2. Morph with overshoot
            overshoot_frames = self.create_overshoot_sequence(base_shape, target_shape)
            frames.extend(overshoot_frames)
            durations.extend([self.config.frame_duration_ms] * len(overshoot_frames))
            logger.debug(f"Added {len(overshoot_frames)} overshoot morph frames")

            # 3. Vibration cycles
            vibration_frames = self.create_vibration_sequence(target_shape)
            frames.extend(vibration_frames)
            durations.extend([self.config.frame_duration_ms] * len(vibration_frames))
            logger.debug(f"Added {len(vibration_frames)} vibration frames")

            # 4. Reverse morph (target back to base)
            reverse_frames = self.create_morph_sequence(target_shape, base_shape, n_steps=2)
            frames.extend(reverse_frames)
            durations.extend([self.config.frame_duration_ms] * len(reverse_frames))
            logger.debug(f"Added {len(reverse_frames)} reverse morph frames")

            # 5. Static end frames
            for _ in range(self.config.static_end_frames):
                frames.append(base_shape.copy())
                durations.append(self.config.frame_duration_ms)
            logger.debug(f"Added {self.config.static_end_frames} static end frames")

            # 6. Blank pause (represented by base shape with longer duration)
            frames.append(base_shape.copy())
            durations.append(self.config.blank_pause_duration_ms)
            logger.debug(f"Added blank pause frame ({self.config.blank_pause_duration_ms}ms)")

        logger.info(f"Generated complete animation: {len(frames)} frames, "
                   f"total duration={sum(durations)}ms")

        return frames, durations

    def update_config(self, **config_overrides):
        """Update configuration with new parameters.

        This recreates components that depend on the changed configuration.

        Args:
            **config_overrides: Configuration parameters to override
        """
        logger.info(f"Updating configuration with overrides: {config_overrides}")

        # Create new config with overrides
        self.config = self.config.copy(**config_overrides)

        # Recreate components if their parameters changed
        if any(key in config_overrides for key in ['n_points']):
            self.contour_extractor = ContourExtractor(
                n_points=self.config.n_points,
                threshold=127
            )
            logger.debug("Recreated ContourExtractor")

        if 'procrustes_scaling' in config_overrides:
            self.aligner = ProcrustesAligner(
                allow_scaling=self.config.procrustes_scaling
            )
            logger.debug("Recreated ProcrustesAligner")

        if any(key in config_overrides for key in ['noise_octaves', 'noise_persistence', 'noise_scale']):
            self.vibrator = PerlinVibrator(
                octaves=self.config.noise_octaves,
                persistence=self.config.noise_persistence,
                scale=self.config.noise_scale
            )
            logger.debug("Recreated PerlinVibrator")

        if 'easing_type' in config_overrides:
            self.easing = EasingCurve(easing_type=self.config.easing_type)
            logger.debug("Recreated EasingCurve")

        logger.info("Configuration update complete")

    def get_animation_info(self, base_shape: np.ndarray, target_shape: np.ndarray) -> dict:
        """Get information about the animation that would be generated.

        Args:
            base_shape: Starting shape
            target_shape: Target shape

        Returns:
            dict: Animation information including frame count, duration, etc.
        """
        # Calculate frame counts
        n_static_start = self.config.static_start_frames
        n_overshoot = len(self.config.overshoot_values)
        n_vibration = self.config.vibration_cycles
        n_reverse = 2
        n_static_end = self.config.static_end_frames
        n_blank = 1

        total_frames = n_static_start + n_overshoot + n_vibration + n_reverse + n_static_end + n_blank

        # Calculate duration
        regular_frame_duration = (
            n_static_start + n_overshoot + n_vibration + n_reverse + n_static_end
        ) * self.config.frame_duration_ms

        total_duration = regular_frame_duration + self.config.blank_pause_duration_ms

        info = {
            'total_frames': total_frames,
            'total_duration_ms': total_duration,
            'total_duration_seconds': total_duration / 1000.0,
            'frame_breakdown': {
                'static_start': n_static_start,
                'overshoot_morph': n_overshoot,
                'vibration': n_vibration,
                'reverse_morph': n_reverse,
                'static_end': n_static_end,
                'blank_pause': n_blank,
            },
            'config': {
                'easing_type': self.config.easing_type,
                'vibration_cycles': self.config.vibration_cycles,
                'n_points': self.config.n_points,
                'use_procrustes': self.config.use_procrustes,
            }
        }

        return info
