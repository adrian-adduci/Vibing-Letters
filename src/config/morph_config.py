"""Global morphing configuration for Vibing Letters.

This module provides the MorphConfig class which holds default parameters
for the morphing engine, easing functions, vibration effects, and frame generation.
"""

from dataclasses import dataclass, field
from typing import Literal

# Type aliases for clarity
EasingType = Literal[
    'linear', 'ease_in_quad', 'ease_out_quad', 'ease_in_out_quad',
    'ease_in_cubic', 'ease_out_cubic', 'ease_in_out_cubic',
    'ease_in_quart', 'ease_out_quart', 'ease_in_out_quart',
    'ease_in_quint', 'ease_out_quint', 'ease_in_out_quint',
    'ease_in_sine', 'ease_out_sine', 'ease_in_out_sine',
    'ease_in_expo', 'ease_out_expo', 'ease_in_out_expo',
    'ease_in_circ', 'ease_out_circ', 'ease_in_out_circ',
    'ease_in_back', 'ease_out_back', 'ease_in_out_back',
    'ease_in_elastic', 'ease_out_elastic', 'ease_in_out_elastic',
    'ease_in_bounce', 'ease_out_bounce', 'ease_in_out_bounce'
]


@dataclass
class MorphConfig:
    """Configuration for morphing parameters.

    This class holds all default parameters for the morphing engine.
    Individual letters can override these values via LetterConfig.

    Attributes:
        n_points: Number of points to resample contours to (default: 120)
        frame_duration_ms: Duration of each frame in milliseconds (default: 20)
        blank_pause_duration_ms: Pause duration after animation in milliseconds (default: 60)

        # Morphing parameters
        easing_type: Type of easing function to use (default: 'ease_in_out_cubic')
        overshoot_values: List of overshoot multipliers for morph sequence (default: [0.0, 1.1, 1.0])

        # Vibration parameters
        vibration_cycles: Number of vibration cycles (default: 3)
        noise_octaves: Number of Perlin noise octaves (default: 4)
        noise_persistence: Perlin noise persistence value (default: 0.5)
        noise_scale: Perlin noise scale factor (default: 0.3)
        vibration_frequency: Vibration frequency multiplier (default: 2.5)
        jitter_strength: Random jitter strength in pixels (default: 1.5)

        # Procrustes alignment parameters
        use_procrustes: Whether to use Procrustes alignment (default: True)
        procrustes_scaling: Whether to allow scaling in Procrustes (default: True)

        # Frame generation parameters
        static_start_frames: Number of static frames at start (default: 3)
        static_end_frames: Number of static frames at end (default: 2)

        # Image parameters
        background_color: RGB tuple for background (default: (255, 255, 255))
        line_color: RGB tuple for line drawing (default: (0, 0, 0))
        line_thickness: Thickness of drawn lines in pixels (default: 2)
        image_size: Tuple of (width, height) for output images (default: (512, 512))
    """

    # Contour parameters
    n_points: int = 120

    # Timing parameters
    frame_duration_ms: int = 20
    blank_pause_duration_ms: int = 60

    # Easing parameters
    easing_type: EasingType = 'ease_in_out_cubic'
    overshoot_values: list[float] = field(default_factory=lambda: [0.0, 1.1, 1.0])

    # Vibration parameters
    vibration_cycles: int = 3
    noise_octaves: int = 4
    noise_persistence: float = 0.5
    noise_scale: float = 0.3
    vibration_frequency: float = 2.5
    jitter_strength: float = 1.5

    # Procrustes alignment parameters
    use_procrustes: bool = True
    procrustes_scaling: bool = True

    # Frame generation parameters
    static_start_frames: int = 3
    static_end_frames: int = 2

    # Image parameters
    background_color: tuple[int, int, int] = (255, 255, 255)
    line_color: tuple[int, int, int] = (0, 0, 0)
    line_thickness: int = 2
    image_size: tuple[int, int] = (512, 512)

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate()

    def _validate(self):
        """Validate all configuration parameters.

        Raises:
            ValueError: If any parameter is invalid
        """
        if self.n_points < 3:
            raise ValueError(f"n_points must be >= 3, got {self.n_points}")

        if self.frame_duration_ms <= 0:
            raise ValueError(f"frame_duration_ms must be positive, got {self.frame_duration_ms}")

        if self.blank_pause_duration_ms < 0:
            raise ValueError(f"blank_pause_duration_ms must be non-negative, got {self.blank_pause_duration_ms}")

        if self.vibration_cycles < 0:
            raise ValueError(f"vibration_cycles must be non-negative, got {self.vibration_cycles}")

        if self.noise_octaves < 1:
            raise ValueError(f"noise_octaves must be >= 1, got {self.noise_octaves}")

        if not 0.0 <= self.noise_persistence <= 1.0:
            raise ValueError(f"noise_persistence must be in [0, 1], got {self.noise_persistence}")

        if self.noise_scale <= 0:
            raise ValueError(f"noise_scale must be positive, got {self.noise_scale}")

        if self.vibration_frequency <= 0:
            raise ValueError(f"vibration_frequency must be positive, got {self.vibration_frequency}")

        if self.jitter_strength < 0:
            raise ValueError(f"jitter_strength must be non-negative, got {self.jitter_strength}")

        if self.static_start_frames < 0:
            raise ValueError(f"static_start_frames must be non-negative, got {self.static_start_frames}")

        if self.static_end_frames < 0:
            raise ValueError(f"static_end_frames must be non-negative, got {self.static_end_frames}")

        if self.line_thickness <= 0:
            raise ValueError(f"line_thickness must be positive, got {self.line_thickness}")

        if len(self.image_size) != 2 or any(d <= 0 for d in self.image_size):
            raise ValueError(f"image_size must be (width, height) with positive values, got {self.image_size}")

        # Validate color tuples
        for color_name, color_value in [
            ('background_color', self.background_color),
            ('line_color', self.line_color)
        ]:
            if len(color_value) != 3 or any(not 0 <= c <= 255 for c in color_value):
                raise ValueError(f"{color_name} must be RGB tuple with values in [0, 255], got {color_value}")

    def copy(self, **overrides):
        """Create a copy of this config with optional parameter overrides.

        Args:
            **overrides: Keyword arguments to override in the copy

        Returns:
            MorphConfig: New configuration instance
        """
        # Get current values as dict
        current_values = {
            'n_points': self.n_points,
            'frame_duration_ms': self.frame_duration_ms,
            'blank_pause_duration_ms': self.blank_pause_duration_ms,
            'easing_type': self.easing_type,
            'overshoot_values': self.overshoot_values.copy(),
            'vibration_cycles': self.vibration_cycles,
            'noise_octaves': self.noise_octaves,
            'noise_persistence': self.noise_persistence,
            'noise_scale': self.noise_scale,
            'vibration_frequency': self.vibration_frequency,
            'jitter_strength': self.jitter_strength,
            'use_procrustes': self.use_procrustes,
            'procrustes_scaling': self.procrustes_scaling,
            'static_start_frames': self.static_start_frames,
            'static_end_frames': self.static_end_frames,
            'background_color': self.background_color,
            'line_color': self.line_color,
            'line_thickness': self.line_thickness,
            'image_size': self.image_size,
        }

        # Apply overrides
        current_values.update(overrides)

        return MorphConfig(**current_values)


# Default global configuration instance
DEFAULT_CONFIG = MorphConfig()
