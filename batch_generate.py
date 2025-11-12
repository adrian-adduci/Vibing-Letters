"""Batch generate letter animations from all images in input directory.

This script processes all images in the input directory and generates
animated GIFs for each one.

Usage:
    python batch_generate.py [options]

Example:
    python batch_generate.py --preset bouncy --base _Static_O.png
"""

import sys
import argparse
from pathlib import Path
from typing import List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.morphing import ContourExtractor, MorphEngine, FrameGenerator, GifBuilder
from src.config import MorphConfig, LetterConfigManager
from src.utils.logger import setup_logging, get_logger, CounterLogger


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch generate letter animations'
    )

    parser.add_argument(
        '--base-image',
        type=Path,
        default=Path('_Static_O.png'),
        help='Base shape image (default: _Static_O.png)'
    )

    parser.add_argument(
        '--background',
        type=Path,
        default=Path('clean_background.png'),
        help='Background image (default: clean_background.png)'
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('input'),
        help='Input directory with letter images (default: input/)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Output directory (default: output/)'
    )

    parser.add_argument(
        '--preset',
        type=str,
        choices=['default', 'bouncy', 'smooth', 'energetic'],
        default='default',
        help='Animation preset to use'
    )

    parser.add_argument(
        '--file-pattern',
        type=str,
        default='*.png',
        help='File pattern to match (default: *.png)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    return parser.parse_args()


def find_input_images(input_dir: Path, pattern: str, logger) -> List[Path]:
    """Find all input images matching pattern.

    Args:
        input_dir: Directory to search
        pattern: File pattern (e.g., '*.png')
        logger: Logger instance

    Returns:
        List[Path]: List of found image paths
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    images = sorted(input_dir.glob(pattern))

    logger.info(f"Found {len(images)} images in {input_dir} matching '{pattern}'")

    return images


def batch_generate_animations(
    base_image_path: Path,
    target_images: List[Path],
    output_dir: Path,
    background_path: Path = None,
    config: MorphConfig = None,
    logger = None
) -> List[Path]:
    """Batch generate animations for multiple target images.

    Args:
        base_image_path: Base shape image
        target_images: List of target image paths
        output_dir: Output directory
        background_path: Background image path
        config: Morph configuration
        logger: Logger instance

    Returns:
        List[Path]: List of generated GIF paths
    """
    if logger is None:
        logger = get_logger(__name__)

    counter = CounterLogger(logger)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components (reuse across all generations)
    logger.info("Initializing morphing pipeline...")
    extractor = ContourExtractor(n_points=config.n_points if config else 120)
    morph_engine = MorphEngine(config=config)
    frame_generator = FrameGenerator(config=config)
    gif_builder = GifBuilder(config=config, optimize=True)

    # Set background
    if background_path and background_path.exists():
        frame_generator.set_background_image(background_path)

    # Extract base contour once (reuse for all)
    logger.info(f"Extracting base contour from {base_image_path}")
    base_contour = extractor.extract_and_resample(base_image_path)

    # Process each target image
    generated_gifs = []
    failed = []

    for i, target_path in enumerate(target_images, 1):
        try:
            logger.info(f"[{i}/{len(target_images)}] Processing {target_path.name}...")

            # Extract target contour
            target_contour = extractor.extract_and_resample(target_path)

            # Generate animation sequence
            shape_sequence, durations = morph_engine.create_full_animation_sequence(
                base_contour, target_contour
            )

            # Render frames
            frames = frame_generator.generate_frames(shape_sequence)

            # Save GIF
            output_name = f"{target_path.stem}_animated.gif"
            output_path = output_dir / output_name

            gif_builder.create_gif(
                frames,
                output_path,
                durations=durations,
                from_bgr=True
            )

            generated_gifs.append(output_path)
            counter.increment('gifs_created')

            # Get file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Created: {output_path.name} ({file_size_mb:.2f} MB)")

        except Exception as e:
            logger.error(f"  Failed to process {target_path.name}: {e}")
            failed.append(target_path)
            counter.increment('failures')

    # Summary
    logger.info("=" * 60)
    logger.info(f"Batch processing complete:")
    logger.info(f"  Successful: {len(generated_gifs)}/{len(target_images)}")
    logger.info(f"  Failed: {len(failed)}/{len(target_images)}")

    if failed:
        logger.warning("Failed files:")
        for path in failed:
            logger.warning(f"  - {path.name}")

    counter.log_summary()

    return generated_gifs


def main():
    """Main entry point."""
    args = parse_arguments()

    # Setup logging
    setup_logging(level=args.log_level, log_to_console=True)
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("Vibing Letters - Batch Animation Generator")
    logger.info("=" * 60)

    try:
        # Validate base image
        if not args.base_image.exists():
            logger.error(f"Base image not found: {args.base_image}")
            sys.exit(1)

        # Find input images
        target_images = find_input_images(args.input_dir, args.file_pattern, logger)

        if not target_images:
            logger.error("No input images found")
            sys.exit(1)

        # Setup configuration
        config_manager = LetterConfigManager()

        if args.preset != 'default':
            logger.info(f"Loading preset: {args.preset}")
            config_manager.load_preset(args.preset)

        # Use default config for batch processing
        config = config_manager._default_config

        # Check background
        background_path = args.background if args.background.exists() else None
        if background_path:
            logger.info(f"Using background: {background_path}")
        else:
            logger.info("Using default white background")

        # Batch generate
        generated_gifs = batch_generate_animations(
            args.base_image,
            target_images,
            args.output_dir,
            background_path,
            config,
            logger
        )

        logger.info("=" * 60)
        logger.info(f"SUCCESS: Generated {len(generated_gifs)} animations in {args.output_dir}")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        return 130

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
