"""Morphing engine for Vibing Letters.

This module provides classes for contour extraction, shape alignment,
vibration effects, easing curves, and frame generation.
"""

from .contour_extractor import ContourExtractor
from .procrustes_aligner import ProcrustesAligner
from .perlin_vibrator import PerlinVibrator
from .easing_curve import EasingCurve
from .morph_engine import MorphEngine
from .frame_generator import FrameGenerator
from .gif_builder import GifBuilder

__all__ = [
    'ContourExtractor',
    'ProcrustesAligner',
    'PerlinVibrator',
    'EasingCurve',
    'MorphEngine',
    'FrameGenerator',
    'GifBuilder'
]
