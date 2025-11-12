"""Unit tests for EasingCurve class.

Tests cover easing functions, interpolation, and curve generation.
"""

import pytest
import numpy as np

from src.morphing.easing_curve import EasingCurve, create_custom_easing


class TestEasingCurveInit:
    """Test suite for EasingCurve initialization."""

    def test_default_initialization(self):
        """Test EasingCurve with default easing type."""
        easing = EasingCurve()
        assert easing.easing_type == 'ease_in_out_cubic'

    def test_custom_initialization(self):
        """Test EasingCurve with custom easing type."""
        easing = EasingCurve('ease_out_bounce')
        assert easing.easing_type == 'ease_out_bounce'

    def test_invalid_easing_type(self):
        """Test initialization fails with invalid easing type."""
        with pytest.raises(ValueError, match="Unknown easing type"):
            EasingCurve('invalid_easing')

    def test_all_easing_types_valid(self):
        """Test that all available easing types can be initialized."""
        for easing_type in EasingCurve.get_available_easings():
            easing = EasingCurve(easing_type)
            assert easing.easing_type == easing_type


class TestEase:
    """Test suite for ease method."""

    def test_ease_scalar_at_boundaries(self):
        """Test easing at boundary values (0 and 1)."""
        easing = EasingCurve('linear')

        # At t=0, result should be 0
        result_start = easing.ease(0.0)
        assert abs(result_start - 0.0) < 0.01

        # At t=1, result should be 1
        result_end = easing.ease(1.0)
        assert abs(result_end - 1.0) < 0.01

    def test_ease_scalar_midpoint(self):
        """Test easing at midpoint."""
        easing = EasingCurve('linear')

        result = easing.ease(0.5)
        assert 0.0 <= result <= 1.0

    def test_ease_array(self):
        """Test easing with numpy array input."""
        easing = EasingCurve('linear')
        t_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        results = easing.ease(t_values)

        assert isinstance(results, np.ndarray)
        assert len(results) == len(t_values)
        assert np.all(results >= 0.0)
        assert np.all(results <= 1.0)

    def test_ease_out_of_range_raises_error(self):
        """Test that out-of-range t values raise error."""
        easing = EasingCurve('linear')

        with pytest.raises(ValueError, match="must be in"):
            easing.ease(-0.1)

        with pytest.raises(ValueError, match="must be in"):
            easing.ease(1.5)

    def test_ease_different_types(self):
        """Test different easing types produce different results."""
        t = 0.5

        linear = EasingCurve('linear').ease(t)
        ease_in = EasingCurve('ease_in_quad').ease(t)
        ease_out = EasingCurve('ease_out_quad').ease(t)

        # Linear should be exactly 0.5
        assert abs(linear - 0.5) < 0.01

        # ease_in should be less than linear at t=0.5
        assert ease_in < linear

        # ease_out should be greater than linear at t=0.5
        assert ease_out > linear


class TestGenerateCurve:
    """Test suite for generate_curve method."""

    def test_generate_curve_basic(self):
        """Test basic curve generation."""
        easing = EasingCurve('linear')
        curve = easing.generate_curve(n_steps=10)

        assert len(curve) == 10
        assert curve[0] == pytest.approx(0.0, abs=0.01)
        assert curve[-1] == pytest.approx(1.0, abs=0.01)
        assert np.all(np.diff(curve) >= 0)  # Should be monotonically increasing

    def test_generate_curve_invalid_n_steps(self):
        """Test that invalid n_steps raises error."""
        easing = EasingCurve('linear')

        with pytest.raises(ValueError, match="n_steps must be >= 2"):
            easing.generate_curve(n_steps=1)

    def test_generate_curve_different_lengths(self):
        """Test curve generation with different lengths."""
        easing = EasingCurve('linear')

        curve_10 = easing.generate_curve(n_steps=10)
        curve_20 = easing.generate_curve(n_steps=20)

        assert len(curve_10) == 10
        assert len(curve_20) == 20


class TestInterpolate:
    """Test suite for interpolate method."""

    def test_interpolate_scalars(self):
        """Test interpolation between scalar start and end values."""
        easing = EasingCurve('linear')
        start = np.array([0.0])
        end = np.array([100.0])

        result_start = easing.interpolate(start, end, 0.0)
        result_mid = easing.interpolate(start, end, 0.5)
        result_end = easing.interpolate(start, end, 1.0)

        assert result_start == pytest.approx(0.0, abs=0.1)
        assert result_mid == pytest.approx(50.0, abs=0.1)
        assert result_end == pytest.approx(100.0, abs=0.1)

    def test_interpolate_arrays(self):
        """Test interpolation between array start and end values."""
        easing = EasingCurve('linear')
        start = np.array([[0, 0], [10, 10]])
        end = np.array([[100, 100], [110, 110]])

        result = easing.interpolate(start, end, 0.5)

        expected = np.array([[50, 50], [60, 60]])
        assert np.allclose(result, expected, atol=0.1)

    def test_interpolate_mismatched_shapes_raises_error(self):
        """Test that mismatched shapes raise error."""
        easing = EasingCurve('linear')
        start = np.array([0, 0])
        end = np.array([100, 100, 100])

        with pytest.raises(ValueError, match="must have same shape"):
            easing.interpolate(start, end, 0.5)

    def test_interpolate_with_nonlinear_easing(self):
        """Test that non-linear easing affects interpolation."""
        start = np.array([0.0])
        end = np.array([100.0])

        linear = EasingCurve('linear').interpolate(start, end, 0.5)
        ease_in = EasingCurve('ease_in_quad').interpolate(start, end, 0.5)
        ease_out = EasingCurve('ease_out_quad').interpolate(start, end, 0.5)

        # Linear should be at 50
        assert linear == pytest.approx(50.0, abs=0.1)

        # ease_in should be less than 50
        assert ease_in < linear

        # ease_out should be greater than 50
        assert ease_out > linear


class TestGenerateInterpolationSequence:
    """Test suite for generate_interpolation_sequence method."""

    def test_generate_sequence_basic(self):
        """Test basic interpolation sequence generation."""
        easing = EasingCurve('linear')
        start = np.array([[0, 0]])
        end = np.array([[100, 100]])

        sequence = easing.generate_interpolation_sequence(start, end, n_steps=5)

        assert len(sequence) == 5
        assert np.allclose(sequence[0], start, atol=0.1)
        assert np.allclose(sequence[-1], end, atol=0.1)

    def test_generate_sequence_invalid_n_steps(self):
        """Test that invalid n_steps raises error."""
        easing = EasingCurve('linear')
        start = np.array([[0, 0]])
        end = np.array([[100, 100]])

        with pytest.raises(ValueError, match="n_steps must be >= 2"):
            easing.generate_interpolation_sequence(start, end, n_steps=1)

    def test_generate_sequence_preserves_shape(self):
        """Test that sequence preserves array shape."""
        easing = EasingCurve('linear')
        start = np.array([[0, 0], [10, 10], [20, 20]])
        end = np.array([[100, 100], [110, 110], [120, 120]])

        sequence = easing.generate_interpolation_sequence(start, end, n_steps=10)

        assert len(sequence) == 10
        for frame in sequence:
            assert frame.shape == start.shape


class TestGetAvailableEasings:
    """Test suite for get_available_easings method."""

    def test_get_available_easings(self):
        """Test that available easings list is returned."""
        easings = EasingCurve.get_available_easings()

        assert isinstance(easings, list)
        assert len(easings) > 0
        assert 'linear' in easings
        assert 'ease_in_out_cubic' in easings
        assert 'ease_out_bounce' in easings


class TestGetEasingDescription:
    """Test suite for get_easing_description method."""

    def test_get_description_valid_easing(self):
        """Test getting description for valid easing."""
        description = EasingCurve.get_easing_description('linear')

        assert isinstance(description, str)
        assert len(description) > 0

    def test_get_description_invalid_easing(self):
        """Test that invalid easing raises error."""
        with pytest.raises(ValueError, match="Unknown easing type"):
            EasingCurve.get_easing_description('invalid_easing')


class TestCreateCustomEasing:
    """Test suite for create_custom_easing function."""

    def test_create_custom_easing(self):
        """Test creating a custom easing function."""
        def quadratic(t):
            return t * t

        custom = create_custom_easing(quadratic)

        assert custom.easing_type == 'custom'

        # Test that custom function works
        result = custom.ease(0.5)
        assert result == pytest.approx(0.25, abs=0.01)

    def test_custom_easing_interpolation(self):
        """Test that custom easing works with interpolation."""
        def identity(t):
            return t

        custom = create_custom_easing(identity)
        start = np.array([0.0])
        end = np.array([100.0])

        result = custom.interpolate(start, end, 0.5)
        assert result == pytest.approx(50.0, abs=0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
