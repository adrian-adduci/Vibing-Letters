"""Per-letter configuration management for Vibing Letters.

This module provides the LetterConfigManager class which manages custom
configurations for individual letters, allowing fine-tuning of morphing
parameters for each letter of the alphabet.
"""

from typing import Dict, Optional
from .morph_config import MorphConfig, DEFAULT_CONFIG


class LetterConfigManager:
    """Manages per-letter configuration overrides.

    This class allows defining custom parameters for individual letters
    while falling back to default values for unspecified parameters.
    """

    def __init__(self, default_config: Optional[MorphConfig] = None):
        """Initialize the letter configuration manager.

        Args:
            default_config: Base configuration to use for all letters.
                          If None, uses the global DEFAULT_CONFIG.
        """
        self._default_config = default_config or DEFAULT_CONFIG
        self._letter_overrides: Dict[str, Dict] = {}

    def set_letter_config(self, letter: str, **overrides):
        """Set configuration overrides for a specific letter.

        Args:
            letter: The letter to configure (case-insensitive, will be uppercased)
            **overrides: Configuration parameters to override

        Raises:
            ValueError: If letter is not a single character A-Z

        Example:
            manager.set_letter_config('A',
                                     easing_type='bounce',
                                     noise_scale=0.4,
                                     vibration_frequency=3.0)
        """
        letter = letter.upper()
        if len(letter) != 1 or not letter.isalpha():
            raise ValueError(f"Letter must be a single alphabetic character, got: {letter}")

        self._letter_overrides[letter] = overrides

    def get_config(self, letter: str) -> MorphConfig:
        """Get the configuration for a specific letter.

        Args:
            letter: The letter to get configuration for (case-insensitive)

        Returns:
            MorphConfig: Configuration with letter-specific overrides applied

        Raises:
            ValueError: If letter is not a single character A-Z
        """
        letter = letter.upper()
        if len(letter) != 1 or not letter.isalpha():
            raise ValueError(f"Letter must be a single alphabetic character, got: {letter}")

        # Get overrides for this letter (if any)
        overrides = self._letter_overrides.get(letter, {})

        # Return a copy of default config with overrides applied
        return self._default_config.copy(**overrides)

    def clear_letter_config(self, letter: str):
        """Clear configuration overrides for a specific letter.

        Args:
            letter: The letter to clear configuration for (case-insensitive)
        """
        letter = letter.upper()
        self._letter_overrides.pop(letter, None)

    def clear_all_configs(self):
        """Clear all letter-specific configuration overrides."""
        self._letter_overrides.clear()

    def get_all_configured_letters(self) -> list[str]:
        """Get a list of all letters with custom configurations.

        Returns:
            list[str]: Sorted list of letters with overrides
        """
        return sorted(self._letter_overrides.keys())

    def has_config(self, letter: str) -> bool:
        """Check if a letter has custom configuration.

        Args:
            letter: The letter to check (case-insensitive)

        Returns:
            bool: True if the letter has custom configuration
        """
        return letter.upper() in self._letter_overrides

    def load_preset(self, preset_name: str):
        """Load a predefined preset configuration.

        Args:
            preset_name: Name of the preset to load

        Raises:
            ValueError: If preset_name is not recognized
        """
        if preset_name == 'default':
            self._load_default_preset()
        elif preset_name == 'bouncy':
            self._load_bouncy_preset()
        elif preset_name == 'smooth':
            self._load_smooth_preset()
        elif preset_name == 'energetic':
            self._load_energetic_preset()
        else:
            raise ValueError(f"Unknown preset: {preset_name}. "
                           f"Available presets: default, bouncy, smooth, energetic")

    def _load_default_preset(self):
        """Load the default preset with balanced parameters."""
        self.clear_all_configs()
        # No overrides - use global defaults for all letters

    def _load_bouncy_preset(self):
        """Load the bouncy preset with elastic, playful motion."""
        self.clear_all_configs()

        # All letters get bouncy easing with increased vibration
        base_bouncy = {
            'easing_type': 'ease_out_bounce',
            'overshoot_values': [0.0, 1.2, 1.0],
            'vibration_frequency': 3.5,
            'noise_scale': 0.4,
        }

        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.set_letter_config(letter, **base_bouncy)

    def _load_smooth_preset(self):
        """Load the smooth preset with gentle, flowing motion."""
        self.clear_all_configs()

        # All letters get smooth easing with reduced vibration
        base_smooth = {
            'easing_type': 'ease_in_out_sine',
            'overshoot_values': [0.0, 1.05, 1.0],
            'vibration_frequency': 1.5,
            'noise_scale': 0.2,
            'vibration_cycles': 2,
        }

        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.set_letter_config(letter, **base_smooth)

    def _load_energetic_preset(self):
        """Load the energetic preset with fast, intense motion."""
        self.clear_all_configs()

        # All letters get elastic easing with high vibration
        base_energetic = {
            'easing_type': 'ease_out_elastic',
            'overshoot_values': [0.0, 1.3, 1.0],
            'vibration_frequency': 4.5,
            'noise_scale': 0.5,
            'vibration_cycles': 4,
        }

        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.set_letter_config(letter, **base_energetic)

    def export_config(self, letter: str) -> dict:
        """Export configuration for a letter as a dictionary.

        Args:
            letter: The letter to export configuration for

        Returns:
            dict: Configuration parameters (only overrides, not defaults)
        """
        letter = letter.upper()
        return self._letter_overrides.get(letter, {}).copy()

    def import_config(self, letter: str, config_dict: dict):
        """Import configuration for a letter from a dictionary.

        Args:
            letter: The letter to import configuration for
            config_dict: Dictionary of configuration parameters
        """
        self.set_letter_config(letter, **config_dict)


# Example configurations for specific letters
EXAMPLE_LETTER_CONFIGS = {
    'A': {
        'easing_type': 'ease_out_bounce',
        'noise_scale': 0.3,
        'vibration_frequency': 2.5,
        'overshoot_values': [0.0, 1.15, 1.0],
    },
    'B': {
        'easing_type': 'ease_in_out_back',
        'noise_scale': 0.35,
        'vibration_frequency': 2.8,
        'overshoot_values': [0.0, 1.12, 1.0],
    },
    'C': {
        'easing_type': 'ease_in_out_cubic',
        'noise_scale': 0.25,
        'vibration_frequency': 2.2,
        'overshoot_values': [0.0, 1.08, 1.0],
    },
    'O': {
        'easing_type': 'ease_in_out_sine',
        'noise_scale': 0.2,
        'vibration_frequency': 1.8,
        'overshoot_values': [0.0, 1.05, 1.0],
    },
    'S': {
        'easing_type': 'ease_out_elastic',
        'noise_scale': 0.4,
        'vibration_frequency': 3.2,
        'overshoot_values': [0.0, 1.2, 1.0],
    },
}


def create_default_letter_manager() -> LetterConfigManager:
    """Create a LetterConfigManager with example configurations loaded.

    Returns:
        LetterConfigManager: Manager with example configs for A, B, C, O, S
    """
    manager = LetterConfigManager()

    for letter, config in EXAMPLE_LETTER_CONFIGS.items():
        manager.set_letter_config(letter, **config)

    return manager
