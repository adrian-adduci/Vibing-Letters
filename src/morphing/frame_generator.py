"""Frame generation for Vibing Letters animations.

This module provides the FrameGenerator class for rendering shape contours
onto background images to create animation frames.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union, List

from ..config.morph_config import MorphConfig
from ..utils.logger import get_logger, PerformanceTimer, CounterLogger
from ..utils.validators import validate_image_file, ValidationError


logger = get_logger(__name__)


class FrameGenerator:
    """Generates image frames from shape contours.

    This class handles rendering contours onto background images
    with configurable colors, line thickness, and image size.
    """

    def __init__(
        self,
        config: Optional[MorphConfig] = None,
        background_image: Optional[np.ndarray] = None
    ):
        """Initialize the frame generator.

        Args:
            config: Morphing configuration (uses default if None)
            background_image: Background image to use (creates blank if None)
        """
        from ..config.morph_config import DEFAULT_CONFIG

        self.config = config or DEFAULT_CONFIG
        self._background_image = background_image
        self._counter = CounterLogger(logger)

        logger.info(
            f"FrameGenerator initialized: image_size={self.config.image_size}, "
            f"line_thickness={self.config.line_thickness}"
        )

    @property
    def background_image(self) -> np.ndarray:
        """Get the background image, creating it if necessary.

        Returns:
            np.ndarray: Background image (BGR format)
        """
        if self._background_image is None:
            self._background_image = self._create_blank_background()
        return self._background_image

    def _create_blank_background(self) -> np.ndarray:
        """Create a blank background image.

        Returns:
            np.ndarray: Blank image with configured size and background color
        """
        height, width = self.config.image_size[1], self.config.image_size[0]
        image = np.full((height, width, 3), self.config.background_color, dtype=np.uint8)

        logger.debug(f"Created blank background: {width}x{height}")
        return image

    def set_background_image(self, image: Union[np.ndarray, str, Path]):
        """Set a custom background image.

        Args:
            image: Background image as array or path to image file

        Raises:
            ValidationError: If file validation fails
            ValueError: If image is invalid
        """
        if isinstance(image, (str, Path)):
            # Load from file
            try:
                validated_path = validate_image_file(image)
                loaded_image = cv2.imread(str(validated_path))
                if loaded_image is None:
                    raise ValueError(f"Failed to load image from {validated_path}")
                self._background_image = loaded_image
                logger.info(f"Loaded background image from {validated_path}")
            except ValidationError as e:
                logger.error(f"Background image validation failed: {e}")
                raise
        elif isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Background image is empty")
            self._background_image = image.copy()
            logger.info(f"Set background image: shape={image.shape}")
        else:
            raise TypeError(f"image must be np.ndarray, str, or Path, got {type(image)}")

    def generate_frame(
        self,
        contour: np.ndarray,
        background: Optional[np.ndarray] = None,
        line_color: Optional[tuple] = None,
        line_thickness: Optional[int] = None
    ) -> np.ndarray:
        """Generate a single frame with the contour drawn on background.

        Args:
            contour: Contour points as Nx2 array
            background: Custom background (uses default if None)
            line_color: Custom line color (uses config if None)
            line_thickness: Custom line thickness (uses config if None)

        Returns:
            np.ndarray: Rendered frame (BGR format)

        Raises:
            ValueError: If contour is invalid
        """
        if contour is None or contour.size == 0:
            raise ValueError("Contour is None or empty")

        if len(contour.shape) != 2 or contour.shape[1] != 2:
            raise ValueError(f"Contour must be Nx2 array, got shape {contour.shape}")

        # Use defaults if not specified
        if background is None:
            background = self.background_image
        if line_color is None:
            line_color = self.config.line_color
        if line_thickness is None:
            line_thickness = self.config.line_thickness

        # Copy background to avoid modifying original
        frame = background.copy()

        # Reshape contour for cv2.polylines (needs Nx1x2)
        contour_cv = contour.reshape((-1, 1, 2)).astype(np.int32)

        # Draw contour
        cv2.polylines(
            frame,
            [contour_cv],
            isClosed=True,
            color=line_color,
            thickness=line_thickness
        )

        self._counter.increment('frames_generated')

        return frame

    def generate_frames(
        self,
        contours: List[np.ndarray],
        background: Optional[np.ndarray] = None,
        line_color: Optional[tuple] = None,
        line_thickness: Optional[int] = None
    ) -> List[np.ndarray]:
        """Generate multiple frames from a list of contours.

        Args:
            contours: List of contour arrays (each Nx2)
            background: Custom background (uses default if None)
            line_color: Custom line color (uses config if None)
            line_thickness: Custom line thickness (uses config if None)

        Returns:
            List[np.ndarray]: List of rendered frames

        Raises:
            ValueError: If contours list is empty
        """
        if not contours:
            raise ValueError("Contours list is empty")

        logger.info(f"Generating {len(contours)} frames")

        with PerformanceTimer(logger, f"Frame generation ({len(contours)} frames)"):
            frames = []
            for i, contour in enumerate(contours):
                try:
                    frame = self.generate_frame(contour, background, line_color, line_thickness)
                    frames.append(frame)
                except ValueError as e:
                    logger.error(f"Failed to generate frame {i}: {e}")
                    raise

        logger.info(f"Successfully generated {len(frames)} frames")
        return frames

    def convert_frames_to_rgb(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """Convert frames from BGR to RGB color space.

        Args:
            frames: List of frames in BGR format

        Returns:
            List[np.ndarray]: List of frames in RGB format
        """
        logger.debug(f"Converting {len(frames)} frames from BGR to RGB")

        rgb_frames = []
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frames.append(rgb_frame)

        return rgb_frames

    def save_frame(
        self,
        frame: np.ndarray,
        output_path: Union[str, Path]
    ) -> Path:
        """Save a single frame to file.

        Args:
            frame: Frame to save (BGR format)
            output_path: Path to save the image

        Returns:
            Path: Path where frame was saved

        Raises:
            ValueError: If frame save fails
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(output_path), frame)
        if not success:
            raise ValueError(f"Failed to save frame to {output_path}")

        logger.debug(f"Saved frame to {output_path}")
        return output_path

    def save_frames(
        self,
        frames: List[np.ndarray],
        output_dir: Union[str, Path],
        prefix: str = "frame",
        extension: str = "png"
    ) -> List[Path]:
        """Save multiple frames to files.

        Args:
            frames: List of frames to save (BGR format)
            output_dir: Directory to save frames
            prefix: Filename prefix (default: "frame")
            extension: File extension (default: "png")

        Returns:
            List[Path]: List of paths where frames were saved
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {len(frames)} frames to {output_dir}")

        saved_paths = []
        for i, frame in enumerate(frames):
            filename = f"{prefix}_{i:04d}.{extension}"
            output_path = output_dir / filename
            self.save_frame(frame, output_path)
            saved_paths.append(output_path)

        logger.info(f"Saved {len(saved_paths)} frames to {output_dir}")
        return saved_paths

    def create_debug_frame(
        self,
        contour: np.ndarray,
        title: str = "Debug Frame"
    ) -> np.ndarray:
        """Create a debug frame with contour and annotations.

        Args:
            contour: Contour to visualize
            title: Title to display on frame

        Returns:
            np.ndarray: Debug frame with annotations
        """
        # Generate base frame
        frame = self.generate_frame(contour)

        # Add title
        cv2.putText(
            frame,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),  # Red
            2
        )

        # Draw points
        for i, point in enumerate(contour):
            x, y = point.astype(int)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)  # Blue dots

            # Label every 10th point
            if i % 10 == 0:
                cv2.putText(
                    frame,
                    str(i),
                    (x + 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1
                )

        # Add contour info
        info_text = f"Points: {len(contour)}"
        cv2.putText(
            frame,
            info_text,
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),  # Green
            1
        )

        return frame

    def get_statistics(self) -> dict:
        """Get frame generation statistics.

        Returns:
            dict: Statistics including frame count
        """
        return {
            'frames_generated': self._counter.get_counter('frames_generated'),
        }

    def reset_statistics(self):
        """Reset frame generation statistics."""
        self._counter.reset()
        logger.debug("Frame generation statistics reset")
