"""Procrustes shape alignment for Vibing Letters.

This module provides the ProcrustesAligner class for optimally aligning
two shapes using Procrustes analysis (translation, rotation, and optional scaling).
"""

import numpy as np
from scipy.spatial import procrustes
from typing import Tuple, Optional

from ..utils.logger import get_logger


logger = get_logger(__name__)


class ProcrustesAligner:
    """Aligns shapes using Procrustes analysis.

    Procrustes analysis finds the optimal alignment between two shapes
    by minimizing the sum of squared distances between corresponding points.
    It can perform translation, rotation, and optional scaling.
    """

    def __init__(self, allow_scaling: bool = True):
        """Initialize the Procrustes aligner.

        Args:
            allow_scaling: Whether to allow scaling during alignment (default: True)
        """
        self.allow_scaling = allow_scaling
        logger.debug(f"ProcrustesAligner initialized with allow_scaling={allow_scaling}")

    def align(
        self,
        source: np.ndarray,
        target: np.ndarray,
        return_disparity: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """Align source shape to target shape using Procrustes analysis.

        Args:
            source: Source shape as Nx2 array of (x, y) coordinates
            target: Target shape as Nx2 array of (x, y) coordinates
            return_disparity: If True, also return the disparity value

        Returns:
            np.ndarray or Tuple[np.ndarray, float]:
                - Aligned source shape (Nx2 array)
                - Disparity value (only if return_disparity=True)

        Raises:
            ValueError: If shapes are invalid or have mismatched dimensions
        """
        # Validate inputs
        if source is None or source.size == 0:
            raise ValueError("source is None or empty")
        if target is None or target.size == 0:
            raise ValueError("target is None or empty")

        if len(source.shape) != 2 or source.shape[1] != 2:
            raise ValueError(f"source must be Nx2 array, got shape {source.shape}")
        if len(target.shape) != 2 or target.shape[1] != 2:
            raise ValueError(f"target must be Nx2 array, got shape {target.shape}")

        if source.shape[0] != target.shape[0]:
            raise ValueError(
                f"source and target must have same number of points: "
                f"{source.shape[0]} != {target.shape[0]}"
            )

        logger.debug(f"Aligning shapes with {len(source)} points")

        try:
            if self.allow_scaling:
                # Use scipy's procrustes (includes scaling)
                _, aligned_source, disparity = procrustes(target, source)
            else:
                # Perform Procrustes without scaling
                aligned_source, disparity = self._procrustes_no_scaling(source, target)

            logger.debug(f"Alignment complete, disparity={disparity:.6f}")

            if return_disparity:
                return aligned_source, disparity
            else:
                return aligned_source

        except Exception as e:
            logger.error(f"Procrustes alignment failed: {e}")
            raise ValueError(f"Alignment failed: {e}")

    def _procrustes_no_scaling(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Perform Procrustes alignment without scaling.

        This implementation translates and rotates the source to match the target
        without changing the scale.

        Args:
            source: Source shape as Nx2 array
            target: Target shape as Nx2 array

        Returns:
            Tuple[np.ndarray, float]: (aligned_source, disparity)
        """
        # Center both shapes
        source_centered = source - source.mean(axis=0)
        target_centered = target - target.mean(axis=0)

        # Compute optimal rotation using SVD
        # We want R such that source_centered @ R ≈ target_centered
        # Solve using SVD of cross-covariance matrix
        H = source_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation (det(R) = 1, not -1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Apply rotation and translation
        aligned_source = source_centered @ R + target.mean(axis=0)

        # Compute disparity (sum of squared distances)
        disparity = np.sum((aligned_source - target) ** 2)

        return aligned_source, disparity

    def align_multiple(
        self,
        source: np.ndarray,
        targets: list[np.ndarray]
    ) -> list[np.ndarray]:
        """Align a source shape to multiple target shapes.

        Args:
            source: Source shape as Nx2 array
            targets: List of target shapes (each Nx2 array)

        Returns:
            list[np.ndarray]: List of aligned source shapes

        Raises:
            ValueError: If any alignment fails
        """
        if not targets:
            raise ValueError("targets list is empty")

        logger.info(f"Aligning source to {len(targets)} targets")

        aligned_shapes = []
        for i, target in enumerate(targets):
            try:
                aligned = self.align(source, target, return_disparity=False)
                aligned_shapes.append(aligned)
            except ValueError as e:
                logger.error(f"Failed to align to target {i}: {e}")
                raise

        logger.info(f"Successfully aligned {len(aligned_shapes)} shapes")
        return aligned_shapes

    def compute_alignment_transform(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> dict:
        """Compute the alignment transform parameters.

        Returns the translation, rotation, and scaling parameters
        that align source to target.

        Args:
            source: Source shape as Nx2 array
            target: Target shape as Nx2 array

        Returns:
            dict: Transform parameters with keys:
                - 'translation': (tx, ty) translation vector
                - 'rotation_angle': Rotation angle in radians
                - 'scale': Scale factor (1.0 if scaling disabled)
                - 'disparity': Alignment error
        """
        # Validate inputs
        if source is None or source.size == 0:
            raise ValueError("source is None or empty")
        if target is None or target.size == 0:
            raise ValueError("target is None or empty")

        # Center both shapes
        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)

        source_centered = source - source_mean
        target_centered = target - target_mean

        # Compute scale if allowed
        if self.allow_scaling:
            source_scale = np.sqrt(np.sum(source_centered ** 2))
            target_scale = np.sqrt(np.sum(target_centered ** 2))
            scale = target_scale / source_scale if source_scale > 0 else 1.0

            # Normalize for rotation computation
            source_normalized = source_centered / source_scale if source_scale > 0 else source_centered
            target_normalized = target_centered / target_scale if target_scale > 0 else target_centered
        else:
            scale = 1.0
            source_normalized = source_centered
            target_normalized = target_centered

        # Compute optimal rotation using SVD
        H = source_normalized.T @ target_normalized
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure proper rotation
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Extract rotation angle
        rotation_angle = np.arctan2(R[1, 0], R[0, 0])

        # Translation is the difference in centroids (after rotation and scaling)
        translation = target_mean - source_mean

        # Apply full transform to compute disparity
        transformed = (source_centered * scale) @ R + target_mean
        disparity = np.sum((transformed - target) ** 2)

        transform = {
            'translation': tuple(translation),
            'rotation_angle': rotation_angle,
            'rotation_degrees': np.degrees(rotation_angle),
            'scale': scale,
            'disparity': disparity,
            'rotation_matrix': R.tolist(),
        }

        logger.debug(
            f"Transform: translation={translation}, "
            f"rotation={np.degrees(rotation_angle):.2f}°, "
            f"scale={scale:.3f}, disparity={disparity:.6f}"
        )

        return transform

    def apply_transform(
        self,
        source: np.ndarray,
        transform: dict
    ) -> np.ndarray:
        """Apply a previously computed transform to a shape.

        Args:
            source: Source shape as Nx2 array
            transform: Transform dictionary from compute_alignment_transform()

        Returns:
            np.ndarray: Transformed shape
        """
        # Extract transform parameters
        translation = np.array(transform['translation'])
        rotation_matrix = np.array(transform['rotation_matrix'])
        scale = transform['scale']

        # Center shape
        source_centered = source - source.mean(axis=0)

        # Apply scale, rotation, and translation
        transformed = (source_centered * scale) @ rotation_matrix + source.mean(axis=0) + translation

        return transformed
