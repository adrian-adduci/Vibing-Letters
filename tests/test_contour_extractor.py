"""Unit tests for ContourExtractor class.

Tests cover contour extraction, resampling, and property calculations.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from src.morphing.contour_extractor import ContourExtractor


class TestContourExtractorInit:
    """Test suite for ContourExtractor initialization."""

    def test_default_initialization(self):
        """Test ContourExtractor with default parameters."""
        extractor = ContourExtractor()
        assert extractor.n_points == 120
        assert extractor.threshold == 127

    def test_custom_initialization(self):
        """Test ContourExtractor with custom parameters."""
        extractor = ContourExtractor(n_points=200, threshold=100)
        assert extractor.n_points == 200
        assert extractor.threshold == 100

    def test_invalid_n_points(self):
        """Test initialization fails with invalid n_points."""
        with pytest.raises(ValueError, match="n_points must be >= 3"):
            ContourExtractor(n_points=2)

    def test_invalid_threshold(self):
        """Test initialization fails with invalid threshold."""
        with pytest.raises(ValueError, match="threshold must be in"):
            ContourExtractor(threshold=300)


class TestExtractFromImage:
    """Test suite for extract_from_image method."""

    def test_extract_circle_contour(self):
        """Test extraction of circular contour."""
        # Create a simple image with a circle
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(image, (100, 100), 50, (0, 0, 0), -1)

        extractor = ContourExtractor()
        contour = extractor.extract_from_image(image)

        assert contour is not None
        assert len(contour) > 0
        assert contour.shape[1] == 2  # Nx2 array

    def test_extract_rectangle_contour(self):
        """Test extraction of rectangular contour."""
        # Create a simple image with a rectangle
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (50, 50), (150, 150), (0, 0, 0), -1)

        extractor = ContourExtractor()
        contour = extractor.extract_from_image(image)

        assert contour is not None
        assert len(contour) >= 4  # At least 4 points for rectangle

    def test_no_contours_found(self):
        """Test error when no contours found."""
        # Create a blank image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255

        extractor = ContourExtractor()

        with pytest.raises(ValueError, match="No contours found"):
            extractor.extract_from_image(image)

    def test_invalid_image_none(self):
        """Test error for None image."""
        extractor = ContourExtractor()

        with pytest.raises(ValueError, match="Invalid image"):
            extractor.extract_from_image(None)

    def test_invalid_image_empty(self):
        """Test error for empty image."""
        extractor = ContourExtractor()
        empty_image = np.array([])

        with pytest.raises(ValueError, match="Invalid image"):
            extractor.extract_from_image(empty_image)


class TestResample:
    """Test suite for resample method."""

    def test_resample_circle_contour(self):
        """Test resampling a circular contour."""
        # Create a circular contour
        angles = np.linspace(0, 2*np.pi, 100, endpoint=False)
        contour = np.column_stack([
            50 * np.cos(angles) + 100,
            50 * np.sin(angles) + 100
        ])

        extractor = ContourExtractor(n_points=120)
        resampled = extractor.resample(contour)

        assert len(resampled) == 120
        assert resampled.shape == (120, 2)

    def test_resample_preserves_shape(self):
        """Test that resampling preserves overall shape."""
        # Create a square contour
        contour = np.array([
            [0, 0], [100, 0], [100, 100], [0, 100]
        ], dtype=np.float32)

        extractor = ContourExtractor(n_points=40)
        resampled = extractor.resample(contour)

        # Check that resampled points are roughly within the square bounds
        assert np.all(resampled[:, 0] >= -1)
        assert np.all(resampled[:, 0] <= 101)
        assert np.all(resampled[:, 1] >= -1)
        assert np.all(resampled[:, 1] <= 101)

    def test_resample_custom_n_points(self):
        """Test resampling with custom n_points."""
        contour = np.array([
            [0, 0], [100, 0], [100, 100], [0, 100]
        ], dtype=np.float32)

        extractor = ContourExtractor(n_points=120)
        resampled = extractor.resample(contour, n_points=60)

        assert len(resampled) == 60

    def test_resample_invalid_contour(self):
        """Test resampling with invalid contour."""
        extractor = ContourExtractor()

        with pytest.raises(ValueError, match="Invalid contour"):
            extractor.resample(None)

        with pytest.raises(ValueError, match="Invalid contour"):
            extractor.resample(np.array([]))

    def test_resample_invalid_shape(self):
        """Test resampling with wrong contour shape."""
        extractor = ContourExtractor()
        invalid_contour = np.array([1, 2, 3])  # Not Nx2

        with pytest.raises(ValueError, match="Contour must be Nx2"):
            extractor.resample(invalid_contour)


class TestExtractAndResample:
    """Test suite for extract_and_resample method."""

    def test_extract_and_resample_from_array(self):
        """Test extracting and resampling from image array."""
        # Create a simple image with a circle
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(image, (100, 100), 50, (0, 0, 0), -1)

        extractor = ContourExtractor(n_points=120)
        contour = extractor.extract_and_resample(image)

        assert len(contour) == 120
        assert contour.shape == (120, 2)

    def test_extract_and_resample_with_custom_n_points(self):
        """Test extracting and resampling with custom n_points."""
        # Create a simple image
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.circle(image, (100, 100), 50, (0, 0, 0), -1)

        extractor = ContourExtractor(n_points=120)
        contour = extractor.extract_and_resample(image, n_points=80)

        assert len(contour) == 80

    def test_invalid_image_type(self):
        """Test error with invalid image type."""
        extractor = ContourExtractor()

        with pytest.raises(ValueError, match="must be np.ndarray, str, or Path"):
            extractor.extract_and_resample(12345)


class TestGetContourProperties:
    """Test suite for get_contour_properties method."""

    def test_circle_properties(self):
        """Test properties of a circular contour."""
        # Create a circular contour
        angles = np.linspace(0, 2*np.pi, 100, endpoint=False)
        contour = np.column_stack([
            50 * np.cos(angles) + 100,
            50 * np.sin(angles) + 100
        ]).astype(np.float32)

        extractor = ContourExtractor()
        props = extractor.get_contour_properties(contour)

        # Check that properties exist
        assert 'area' in props
        assert 'perimeter' in props
        assert 'centroid' in props
        assert 'bounding_box' in props
        assert 'n_points' in props

        # Check centroid is near (100, 100)
        cx, cy = props['centroid']
        assert abs(cx - 100) < 5
        assert abs(cy - 100) < 5

        # Check area is positive
        assert props['area'] > 0

        # Check n_points matches
        assert props['n_points'] == 100

    def test_square_properties(self):
        """Test properties of a square contour."""
        contour = np.array([
            [0, 0], [100, 0], [100, 100], [0, 100]
        ], dtype=np.float32)

        extractor = ContourExtractor()
        props = extractor.get_contour_properties(contour)

        # Check centroid is at (50, 50)
        cx, cy = props['centroid']
        assert abs(cx - 50) < 1
        assert abs(cy - 50) < 1

        # Check bounding box
        bbox = props['bounding_box']
        assert abs(bbox['width'] - 100) < 1
        assert abs(bbox['height'] - 100) < 1

    def test_invalid_contour_for_properties(self):
        """Test error with invalid contour for properties."""
        extractor = ContourExtractor()

        with pytest.raises(ValueError, match="Invalid contour"):
            extractor.get_contour_properties(None)

        with pytest.raises(ValueError, match="Invalid contour"):
            extractor.get_contour_properties(np.array([]))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
