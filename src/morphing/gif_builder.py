"""GIF building and optimization for Vibing Letters.

This module provides the GifBuilder class for creating optimized GIF
animations from frame sequences.
"""

from PIL import Image
import numpy as np
from pathlib import Path
from typing import Union, List, Optional

from ..config.morph_config import MorphConfig
from ..utils.logger import get_logger, PerformanceTimer
from ..utils.validators import sanitize_filename, ValidationError


logger = get_logger(__name__)


class GifBuilder:
    """Builds and optimizes GIF animations.

    This class handles converting frame sequences to GIF format with
    optimization options for file size reduction.
    """

    def __init__(
        self,
        config: Optional[MorphConfig] = None,
        optimize: bool = True,
        colors: int = 256,
        loop: int = 0
    ):
        """Initialize the GIF builder.

        Args:
            config: Morphing configuration (uses default if None)
            optimize: Whether to optimize GIF (default: True)
            colors: Number of colors in palette (default: 256, max: 256)
            loop: Number of loops (0 = infinite, default: 0)

        Raises:
            ValueError: If parameters are invalid
        """
        from ..config.morph_config import DEFAULT_CONFIG

        self.config = config or DEFAULT_CONFIG

        if not 2 <= colors <= 256:
            raise ValueError(f"colors must be in [2, 256], got {colors}")

        if loop < 0:
            raise ValueError(f"loop must be non-negative, got {loop}")

        self.optimize = optimize
        self.colors = colors
        self.loop = loop

        logger.info(
            f"GifBuilder initialized: optimize={optimize}, "
            f"colors={colors}, loop={loop}"
        )

    def frames_to_pil_images(
        self,
        frames: List[np.ndarray],
        from_bgr: bool = False
    ) -> List[Image.Image]:
        """Convert numpy frames to PIL Images.

        Args:
            frames: List of frames as numpy arrays
            from_bgr: If True, convert from BGR to RGB (default: False)

        Returns:
            List[Image.Image]: List of PIL Images

        Raises:
            ValueError: If frames are invalid
        """
        if not frames:
            raise ValueError("Frames list is empty")

        logger.debug(f"Converting {len(frames)} numpy frames to PIL Images")

        pil_images = []
        for i, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                raise ValueError(f"Frame {i} is None or empty")

            # Convert BGR to RGB if needed
            if from_bgr:
                import cv2
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(frame)
            pil_images.append(pil_image)

        logger.debug(f"Converted {len(pil_images)} frames to PIL Images")
        return pil_images

    def create_gif(
        self,
        frames: Union[List[np.ndarray], List[Image.Image]],
        output_path: Union[str, Path],
        durations: Optional[List[int]] = None,
        from_bgr: bool = False,
        optimize: Optional[bool] = None,
        colors: Optional[int] = None,
        loop: Optional[int] = None
    ) -> Path:
        """Create a GIF from frame sequence.

        Args:
            frames: List of frames (numpy arrays or PIL Images)
            output_path: Path to save GIF
            durations: List of frame durations in ms (uses config if None)
            from_bgr: If True and frames are numpy, convert from BGR to RGB
            optimize: Override optimize setting (uses instance setting if None)
            colors: Override colors setting (uses instance setting if None)
            loop: Override loop setting (uses instance setting if None)

        Returns:
            Path: Path where GIF was saved

        Raises:
            ValueError: If parameters are invalid
            ValidationError: If filename sanitization fails
        """
        if not frames:
            raise ValueError("Frames list is empty")

        # Use instance settings if not overridden
        optimize = optimize if optimize is not None else self.optimize
        colors = colors if colors is not None else self.colors
        loop = loop if loop is not None else self.loop

        # Validate and sanitize output path
        output_path = Path(output_path)
        try:
            sanitized_filename = sanitize_filename(output_path.name)
            output_path = output_path.parent / sanitized_filename
        except ValidationError as e:
            logger.error(f"Filename sanitization failed: {e}")
            raise

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating GIF: {len(frames)} frames → {output_path}")

        with PerformanceTimer(logger, "GIF creation"):
            # Convert numpy frames to PIL if needed
            if frames and isinstance(frames[0], np.ndarray):
                pil_frames = self.frames_to_pil_images(frames, from_bgr=from_bgr)
            else:
                pil_frames = frames

            # Set up durations
            if durations is None:
                durations = [self.config.frame_duration_ms] * len(pil_frames)
            elif len(durations) != len(pil_frames):
                raise ValueError(
                    f"durations length ({len(durations)}) must match "
                    f"frames length ({len(pil_frames)})"
                )

            # Quantize colors if needed
            if colors < 256:
                logger.debug(f"Quantizing to {colors} colors")
                pil_frames = self._quantize_frames(pil_frames, colors)

            # Save GIF
            pil_frames[0].save(
                str(output_path),
                save_all=True,
                append_images=pil_frames[1:],
                duration=durations,
                loop=loop,
                optimize=optimize
            )

        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"GIF created: {output_path} "
            f"({len(pil_frames)} frames, {file_size_mb:.2f} MB)"
        )

        return output_path

    def _quantize_frames(
        self,
        frames: List[Image.Image],
        colors: int
    ) -> List[Image.Image]:
        """Quantize frames to reduce color palette.

        Args:
            frames: List of PIL Images
            colors: Number of colors to use

        Returns:
            List[Image.Image]: Quantized frames
        """
        quantized_frames = []

        for frame in frames:
            # Convert to palette mode with specified colors
            quantized = frame.convert('P', palette=Image.ADAPTIVE, colors=colors)
            # Convert back to RGB for consistency
            quantized = quantized.convert('RGB')
            quantized_frames.append(quantized)

        return quantized_frames

    def optimize_gif(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        colors: Optional[int] = None
    ) -> Path:
        """Optimize an existing GIF file.

        Args:
            input_path: Path to input GIF
            output_path: Path for optimized GIF (overwrites input if None)
            colors: Number of colors to use (uses instance setting if None)

        Returns:
            Path: Path to optimized GIF

        Raises:
            ValueError: If input file is invalid
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise ValueError(f"Input file does not exist: {input_path}")

        if output_path is None:
            output_path = input_path
        else:
            output_path = Path(output_path)

        colors = colors if colors is not None else self.colors

        logger.info(f"Optimizing GIF: {input_path} → {output_path}")

        # Get original file size
        original_size_mb = input_path.stat().st_size / (1024 * 1024)

        with PerformanceTimer(logger, "GIF optimization"):
            # Load GIF
            with Image.open(input_path) as img:
                frames = []
                durations = []

                # Extract all frames and durations
                try:
                    while True:
                        frames.append(img.copy().convert('RGB'))
                        durations.append(img.info.get('duration', 100))
                        img.seek(img.tell() + 1)
                except EOFError:
                    pass  # End of frames

                logger.debug(f"Loaded {len(frames)} frames from {input_path}")

                # Get loop setting
                loop = img.info.get('loop', 0)

                # Quantize if needed
                if colors < 256:
                    frames = self._quantize_frames(frames, colors)

                # Save optimized GIF
                frames[0].save(
                    str(output_path),
                    save_all=True,
                    append_images=frames[1:],
                    duration=durations,
                    loop=loop,
                    optimize=True
                )

        # Get optimized file size
        optimized_size_mb = output_path.stat().st_size / (1024 * 1024)
        reduction = (1 - optimized_size_mb / original_size_mb) * 100 if original_size_mb > 0 else 0

        logger.info(
            f"GIF optimized: {original_size_mb:.2f} MB → {optimized_size_mb:.2f} MB "
            f"({reduction:.1f}% reduction)"
        )

        return output_path

    def create_gif_from_files(
        self,
        image_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        durations: Optional[List[int]] = None
    ) -> Path:
        """Create a GIF from image files.

        Args:
            image_paths: List of paths to image files
            output_path: Path to save GIF
            durations: List of frame durations in ms (uses config if None)

        Returns:
            Path: Path where GIF was saved

        Raises:
            ValueError: If image files are invalid
        """
        if not image_paths:
            raise ValueError("image_paths list is empty")

        logger.info(f"Creating GIF from {len(image_paths)} image files")

        # Load images
        frames = []
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert('RGB')
                frames.append(img)
            except Exception as e:
                logger.error(f"Failed to load image {i} from {path}: {e}")
                raise ValueError(f"Failed to load image from {path}: {e}")

        # Create GIF
        return self.create_gif(frames, output_path, durations)

    def batch_create_gifs(
        self,
        frames_list: List[List[np.ndarray]],
        output_dir: Union[str, Path],
        filenames: Optional[List[str]] = None,
        durations_list: Optional[List[List[int]]] = None
    ) -> List[Path]:
        """Create multiple GIFs from frame sequences.

        Args:
            frames_list: List of frame sequences
            output_dir: Directory to save GIFs
            filenames: List of output filenames (auto-generated if None)
            durations_list: List of duration lists (uses config if None)

        Returns:
            List[Path]: List of paths where GIFs were saved
        """
        if not frames_list:
            raise ValueError("frames_list is empty")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Batch creating {len(frames_list)} GIFs in {output_dir}")

        saved_paths = []

        for i, frames in enumerate(frames_list):
            # Generate filename if not provided
            if filenames and i < len(filenames):
                filename = filenames[i]
            else:
                filename = f"animation_{i:03d}.gif"

            output_path = output_dir / filename

            # Get durations for this sequence
            if durations_list and i < len(durations_list):
                durations = durations_list[i]
            else:
                durations = None

            # Create GIF
            try:
                saved_path = self.create_gif(frames, output_path, durations, from_bgr=True)
                saved_paths.append(saved_path)
            except Exception as e:
                logger.error(f"Failed to create GIF {i}: {e}")
                raise

        logger.info(f"Created {len(saved_paths)} GIFs in {output_dir}")
        return saved_paths

    def get_gif_info(self, gif_path: Union[str, Path]) -> dict:
        """Get information about a GIF file.

        Args:
            gif_path: Path to GIF file

        Returns:
            dict: GIF information including frames, size, duration

        Raises:
            ValueError: If file is invalid
        """
        gif_path = Path(gif_path)

        if not gif_path.exists():
            raise ValueError(f"GIF file does not exist: {gif_path}")

        with Image.open(gif_path) as img:
            frames = []
            durations = []

            try:
                while True:
                    durations.append(img.info.get('duration', 0))
                    img.seek(img.tell() + 1)
                    frames.append(None)  # Don't store actual frames, just count
            except EOFError:
                pass

            file_size_mb = gif_path.stat().st_size / (1024 * 1024)
            total_duration_ms = sum(durations)

            info = {
                'path': str(gif_path),
                'n_frames': len(frames),
                'file_size_mb': file_size_mb,
                'file_size_bytes': gif_path.stat().st_size,
                'total_duration_ms': total_duration_ms,
                'total_duration_seconds': total_duration_ms / 1000.0,
                'average_frame_duration_ms': total_duration_ms / len(frames) if frames else 0,
                'loop': img.info.get('loop', 0),
                'size': img.size,
            }

        return info
