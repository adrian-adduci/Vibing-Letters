"""Contour extraction and resampling for Vibing Letters.

This module provides the ContourExtractor class for extracting contours
from images and resampling them to a fixed number of points for morphing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional

from ..utils.logger import get_logger
from ..utils.validators import validate_image_file, ValidationError


logger = get_logger(__name__)


class ContourExtractor:
    """Extracts and resamples contours from images.

    This class handles contour extraction using OpenCV and provides
    methods for resampling contours to a fixed number of evenly-spaced points.
    """

    def __init__(self, n_points: int = 120, threshold: int = 127):
        """Initialize the contour extractor.

        Args:
            n_points: Number of points to resample contours to (default: 120)
            threshold: Binary threshold value for contour detection (default: 127)

        Raises:
            ValueError: If parameters are invalid
        """
        if n_points < 3:
            raise ValueError(f"n_points must be >= 3, got {n_points}")
        if not 0 <= threshold <= 255:
            raise ValueError(f"threshold must be in [0, 255], got {threshold}")

        self.n_points = n_points
        self.threshold = threshold

        logger.debug(f"ContourExtractor initialized with n_points={n_points}, threshold={threshold}")

    def extract_from_image(self, image: np.ndarray) -> np.ndarray:
        """Extract the largest contour from an image.

        Args:
            image: Input image (BGR format from cv2.imread)

        Returns:
            np.ndarray: Contour points as Nx2 array of (x, y) coordinates

        Raises:
            ValueError: If no contours found or image is invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid image: image is None or empty")

        logger.debug(f"Extracting contour from image of shape {image.shape}")

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply binary threshold (invert so object is white)
            _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                raise ValueError("No contours found in image")

            # Get the largest contour by area
            contour = max(contours, key=cv2.contourArea)

            # Reshape to Nx2 array (remove middle dimension)
            contour_points = contour[:, 0, :]

            logger.debug(f"Extracted contour with {len(contour_points)} points, "
                        f"area={cv2.contourArea(contour):.2f}")

            return contour_points

        except cv2.error as e:
            logger.error(f"OpenCV error during contour extraction: {e}")
            raise ValueError(f"Failed to extract contour: {e}")

    def extract_from_file(
        self,
        file_path: Union[str, Path],
        base_directory: Optional[Path] = None
    ) -> np.ndarray:
        """Extract the largest contour from an image file.

        Args:
            file_path: Path to the image file
            base_directory: Optional base directory for path validation

        Returns:
            np.ndarray: Contour points as Nx2 array

        Raises:
            ValidationError: If file validation fails
            ValueError: If contour extraction fails
        """
        # Validate file path
        try:
            validated_path = validate_image_file(
                file_path,
                allowed_formats={'png', 'jpg', 'jpeg', 'bmp'},
                base_directory=base_directory
            )
        except ValidationError as e:
            logger.error(f"File validation failed: {e}")
            raise

        logger.info(f"Loading image from {validated_path}")

        # Load image
        image = cv2.imread(str(validated_path))
        if image is None:
            raise ValueError(f"Failed to load image from {validated_path}")

        # Extract contour
        return self.extract_from_image(image)

    def resample(self, contour: np.ndarray, n_points: Optional[int] = None) -> np.ndarray:
        """Resample contour to a fixed number of evenly-spaced points.

        This method distributes points evenly along the contour perimeter
        based on arc length, ensuring smooth morphing between shapes.

        Args:
            contour: Input contour as Nx2 array of (x, y) coordinates
            n_points: Number of points to resample to (uses self.n_points if None)

        Returns:
            np.ndarray: Resampled contour as n_points x 2 array

        Raises:
            ValueError: If contour is invalid
        """
        if contour is None or contour.size == 0:
            raise ValueError("Invalid contour: contour is None or empty")

        if len(contour.shape) != 2 or contour.shape[1] != 2:
            raise ValueError(f"Contour must be Nx2 array, got shape {contour.shape}")

        if n_points is None:
            n_points = self.n_points

        if n_points < 3:
            raise ValueError(f"n_points must be >= 3, got {n_points}")

        logger.debug(f"Resampling contour from {len(contour)} to {n_points} points")

        # Calculate cumulative distances along contour
        distances = [0.0]
        for i in range(1, len(contour)):
            d = np.linalg.norm(contour[i] - contour[i-1])
            distances.append(d)

        cumulative_distances = np.cumsum(distances)
        total_perimeter = cumulative_distances[-1]

        if total_perimeter == 0:
            logger.warning("Contour has zero perimeter, returning duplicate points")
            return np.tile(contour[0], (n_points, 1))

        # Sample points evenly along perimeter
        resampled = []
        target_distances = np.linspace(0, total_perimeter, n_points, endpoint=False)

        for target_dist in target_distances:
            # Find segment containing this distance
            idx = np.searchsorted(cumulative_distances, target_dist)
            idx = min(idx, len(contour) - 2)  # Ensure we don't go past the last segment

            p1 = contour[idx]
            p2 = contour[idx + 1]

            segment_length = np.linalg.norm(p2 - p1)

            if segment_length == 0:
                # Degenerate segment, use p1
                resampled.append(p1)
            else:
                # Interpolate along segment
                segment_start_dist = cumulative_distances[idx]
                ratio = (target_dist - segment_start_dist) / segment_length
                ratio = np.clip(ratio, 0.0, 1.0)  # Ensure ratio is in [0, 1]
                new_point = (1 - ratio) * p1 + ratio * p2
                resampled.append(new_point)

        resampled_array = np.array(resampled, dtype=np.float32)

        logger.debug(f"Resampled contour to {len(resampled_array)} points, "
                    f"perimeter={total_perimeter:.2f}")

        return resampled_array

    def extract_and_resample(
        self,
        image: Union[np.ndarray, str, Path],
        n_points: Optional[int] = None,
        base_directory: Optional[Path] = None
    ) -> np.ndarray:
        """Extract and resample a contour in one operation.

        Args:
            image: Image array, file path, or Path object
            n_points: Number of points to resample to (uses self.n_points if None)
            base_directory: Optional base directory for path validation

        Returns:
            np.ndarray: Resampled contour as n_points x 2 array

        Raises:
            ValueError: If extraction or resampling fails
        """
        # Determine if image is array or file path
        if isinstance(image, (str, Path)):
            contour = self.extract_from_file(image, base_directory)
        elif isinstance(image, np.ndarray):
            contour = self.extract_from_image(image)
        else:
            raise ValueError(f"image must be np.ndarray, str, or Path, got {type(image)}")

        return self.resample(contour, n_points)

    def get_contour_properties(self, contour: np.ndarray) -> dict:
        """Get geometric properties of a contour.

        Args:
            contour: Contour as Nx2 array

        Returns:
            dict: Properties including area, perimeter, centroid, bounding box
        """
        if contour is None or contour.size == 0:
            raise ValueError("Invalid contour: contour is None or empty")

        # Reshape for OpenCV functions (needs Nx1x2)
        contour_cv = contour.reshape((-1, 1, 2)).astype(np.float32)

        # Calculate properties
        area = cv2.contourArea(contour_cv)
        perimeter = cv2.arcLength(contour_cv, closed=True)

        # Calculate centroid
        moments = cv2.moments(contour_cv)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
        else:
            # Fallback to mean if moment is zero
            cx, cy = np.mean(contour, axis=0)

        # Bounding box
        x_min, y_min = contour.min(axis=0)
        x_max, y_max = contour.max(axis=0)

        properties = {
            'area': area,
            'perimeter': perimeter,
            'centroid': (cx, cy),
            'bounding_box': {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max,
                'width': x_max - x_min,
                'height': y_max - y_min,
            },
            'n_points': len(contour),
        }

        return properties
