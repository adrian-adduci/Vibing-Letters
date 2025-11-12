"""Configuration management for Vibing Letters.

This module provides configuration classes for morphing parameters,
per-letter customization, and global settings.
"""

from .morph_config import MorphConfig
from .letter_config import LetterConfigManager

__all__ = ['MorphConfig', 'LetterConfigManager']
