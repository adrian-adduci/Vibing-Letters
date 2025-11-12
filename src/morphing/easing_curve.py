"""Easing curve functions for Vibing Letters.

This module provides the EasingCurve class for applying non-linear
interpolation curves to morphing animations, creating more natural motion.
"""

import numpy as np
from easing_functions import (
    LinearInOut,
    QuadEaseIn, QuadEaseOut, QuadEaseInOut,
    CubicEaseIn, CubicEaseOut, CubicEaseInOut,
    QuarticEaseIn, QuarticEaseOut, QuarticEaseInOut,
    QuinticEaseIn, QuinticEaseOut, QuinticEaseInOut,
    SineEaseIn, SineEaseOut, SineEaseInOut,
    CircularEaseIn, CircularEaseOut, CircularEaseInOut,
    ExponentialEaseIn, ExponentialEaseOut, ExponentialEaseInOut,
    ElasticEaseIn, ElasticEaseOut, ElasticEaseInOut,
    BackEaseIn, BackEaseOut, BackEaseInOut,
    BounceEaseIn, BounceEaseOut, BounceEaseInOut
)
from typing import Union, Callable

from ..utils.logger import get_logger


logger = get_logger(__name__)


class EasingCurve:
    """Provides easing curve functions for smooth animation.

    Easing curves modify the timing of animations to create more natural,
    non-linear motion. This class wraps the easing-functions library and
    provides a consistent interface for all easing types.
    """

    # Map easing names to classes
    EASING_FUNCTIONS = {
        'linear': LinearInOut,
        # Quadratic
        'ease_in_quad': QuadEaseIn,
        'ease_out_quad': QuadEaseOut,
        'ease_in_out_quad': QuadEaseInOut,
        # Cubic
        'ease_in_cubic': CubicEaseIn,
        'ease_out_cubic': CubicEaseOut,
        'ease_in_out_cubic': CubicEaseInOut,
        # Quartic
        'ease_in_quart': QuarticEaseIn,
        'ease_out_quart': QuarticEaseOut,
        'ease_in_out_quart': QuarticEaseInOut,
        # Quintic
        'ease_in_quint': QuinticEaseIn,
        'ease_out_quint': QuinticEaseOut,
        'ease_in_out_quint': QuinticEaseInOut,
        # Sine
        'ease_in_sine': SineEaseIn,
        'ease_out_sine': SineEaseOut,
        'ease_in_out_sine': SineEaseInOut,
        # Circular
        'ease_in_circ': CircularEaseIn,
        'ease_out_circ': CircularEaseOut,
        'ease_in_out_circ': CircularEaseInOut,
        # Exponential
        'ease_in_expo': ExponentialEaseIn,
        'ease_out_expo': ExponentialEaseOut,
        'ease_in_out_expo': ExponentialEaseInOut,
        # Elastic
        'ease_in_elastic': ElasticEaseIn,
        'ease_out_elastic': ElasticEaseOut,
        'ease_in_out_elastic': ElasticEaseInOut,
        # Back
        'ease_in_back': BackEaseIn,
        'ease_out_back': BackEaseOut,
        'ease_in_out_back': BackEaseInOut,
        # Bounce
        'ease_in_bounce': BounceEaseIn,
        'ease_out_bounce': BounceEaseOut,
        'ease_in_out_bounce': BounceEaseInOut,
    }

    def __init__(self, easing_type: str = 'ease_in_out_cubic'):
        """Initialize the easing curve.

        Args:
            easing_type: Type of easing function to use (default: 'ease_in_out_cubic')

        Raises:
            ValueError: If easing_type is not recognized
        """
        if easing_type not in self.EASING_FUNCTIONS:
            available = ', '.join(sorted(self.EASING_FUNCTIONS.keys()))
            raise ValueError(
                f"Unknown easing type: {easing_type}. "
                f"Available types: {available}"
            )

        self.easing_type = easing_type
        self.easing_function = self.EASING_FUNCTIONS[easing_type]()

        logger.debug(f"EasingCurve initialized with type: {easing_type}")

    def ease(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply easing function to time value(s).

        Args:
            t: Time value(s) in range [0, 1]
               Can be a scalar or numpy array

        Returns:
            Union[float, np.ndarray]: Eased value(s) in range [0, 1]

        Raises:
            ValueError: If t is outside [0, 1] range
        """
        # Handle scalar
        if isinstance(t, (int, float)):
            if not 0.0 <= t <= 1.0:
                raise ValueError(f"t must be in [0, 1], got {t}")
            return self.easing_function(t)

        # Handle array
        if isinstance(t, np.ndarray):
            if not np.all((t >= 0.0) & (t <= 1.0)):
                raise ValueError(f"All t values must be in [0, 1], got min={t.min()}, max={t.max()}")
            return np.array([self.easing_function(val) for val in t.flat]).reshape(t.shape)

        raise TypeError(f"t must be float or np.ndarray, got {type(t)}")

    def generate_curve(self, n_steps: int) -> np.ndarray:
        """Generate an array of eased values.

        Args:
            n_steps: Number of steps to generate

        Returns:
            np.ndarray: Array of eased values from 0.0 to 1.0

        Raises:
            ValueError: If n_steps < 2
        """
        if n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {n_steps}")

        # Generate linear time values
        t_linear = np.linspace(0.0, 1.0, n_steps)

        # Apply easing
        t_eased = self.ease(t_linear)

        logger.debug(f"Generated easing curve with {n_steps} steps")

        return t_eased

    def interpolate(
        self,
        start: np.ndarray,
        end: np.ndarray,
        t: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Interpolate between start and end using easing.

        Args:
            start: Starting values (any shape)
            end: Ending values (same shape as start)
            t: Time value(s) in [0, 1]

        Returns:
            np.ndarray: Interpolated values

        Raises:
            ValueError: If shapes don't match or t is invalid
        """
        if start.shape != end.shape:
            raise ValueError(f"start and end must have same shape: {start.shape} != {end.shape}")

        # Apply easing to time
        t_eased = self.ease(t)

        # Interpolate
        if isinstance(t_eased, (int, float)):
            result = (1.0 - t_eased) * start + t_eased * end
        else:
            # Handle array of t values
            t_eased = t_eased.reshape(-1, 1, 1)  # Reshape for broadcasting
            result = (1.0 - t_eased) * start + t_eased * end

        return result

    def generate_interpolation_sequence(
        self,
        start: np.ndarray,
        end: np.ndarray,
        n_steps: int
    ) -> list[np.ndarray]:
        """Generate a sequence of interpolated values.

        Args:
            start: Starting values
            end: Ending values
            n_steps: Number of interpolation steps

        Returns:
            list[np.ndarray]: List of interpolated arrays

        Raises:
            ValueError: If parameters are invalid
        """
        if n_steps < 2:
            raise ValueError(f"n_steps must be >= 2, got {n_steps}")

        logger.debug(f"Generating interpolation sequence: {n_steps} steps")

        # Generate easing curve
        t_values = self.generate_curve(n_steps)

        # Interpolate for each t value
        sequence = []
        for t in t_values:
            interpolated = self.interpolate(start, end, t)
            sequence.append(interpolated)

        logger.debug(f"Generated {len(sequence)} interpolated frames")

        return sequence

    @classmethod
    def get_available_easings(cls) -> list[str]:
        """Get a list of all available easing function names.

        Returns:
            list[str]: Sorted list of easing function names
        """
        return sorted(cls.EASING_FUNCTIONS.keys())

    @classmethod
    def get_easing_description(cls, easing_type: str) -> str:
        """Get a description of an easing function.

        Args:
            easing_type: Name of the easing function

        Returns:
            str: Description of the easing function

        Raises:
            ValueError: If easing_type is not recognized
        """
        if easing_type not in cls.EASING_FUNCTIONS:
            raise ValueError(f"Unknown easing type: {easing_type}")

        descriptions = {
            'linear': 'Constant speed, no acceleration',
            'ease_in_quad': 'Slow start, quadratic acceleration',
            'ease_out_quad': 'Fast start, quadratic deceleration',
            'ease_in_out_quad': 'Slow start and end, quadratic',
            'ease_in_cubic': 'Slow start, cubic acceleration',
            'ease_out_cubic': 'Fast start, cubic deceleration',
            'ease_in_out_cubic': 'Slow start and end, cubic (smooth)',
            'ease_in_quart': 'Slow start, quartic acceleration',
            'ease_out_quart': 'Fast start, quartic deceleration',
            'ease_in_out_quart': 'Slow start and end, quartic',
            'ease_in_quint': 'Slow start, quintic acceleration',
            'ease_out_quint': 'Fast start, quintic deceleration',
            'ease_in_out_quint': 'Slow start and end, quintic',
            'ease_in_sine': 'Slow start, sinusoidal',
            'ease_out_sine': 'Fast start, sinusoidal',
            'ease_in_out_sine': 'Slow start and end, sinusoidal (very smooth)',
            'ease_in_circ': 'Slow start, circular',
            'ease_out_circ': 'Fast start, circular',
            'ease_in_out_circ': 'Slow start and end, circular',
            'ease_in_expo': 'Very slow start, exponential',
            'ease_out_expo': 'Very fast start, exponential',
            'ease_in_out_expo': 'Very slow start and end, exponential',
            'ease_in_elastic': 'Elastic bounce at start',
            'ease_out_elastic': 'Elastic bounce at end (springy)',
            'ease_in_out_elastic': 'Elastic bounce at start and end',
            'ease_in_back': 'Pull back before starting',
            'ease_out_back': 'Overshoot then settle',
            'ease_in_out_back': 'Pull back at start, overshoot at end',
            'ease_in_bounce': 'Bounce at start',
            'ease_out_bounce': 'Bounce at end (ball dropping)',
            'ease_in_out_bounce': 'Bounce at start and end',
        }

        return descriptions.get(easing_type, 'No description available')


def create_custom_easing(func: Callable[[float], float]) -> EasingCurve:
    """Create a custom easing curve from a function.

    Args:
        func: Function that maps [0, 1] to [0, 1]

    Returns:
        EasingCurve: Custom easing curve instance

    Example:
        def my_ease(t):
            return t * t  # Quadratic
        custom = create_custom_easing(my_ease)
    """
    # Create a temporary class that wraps the function
    class CustomEasing:
        def __init__(self):
            self.func = func

        def __call__(self, t):
            return self.func(t)

    # Create an EasingCurve instance and override its function
    easing = EasingCurve('linear')  # Use linear as base
    easing.easing_function = CustomEasing()
    easing.easing_type = 'custom'

    return easing
