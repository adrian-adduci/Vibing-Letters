"""Generate individual letter animations using the new Vibing Letters architecture.

This script replaces vibing_letter_generator.py with a clean, maintainable
implementation using the refactored morphing engine.

Usage:
    python generate_letter.py <base_image> <target_image> [output_name]

Example:
    python generate_letter.py _Static_O.png input/A.png output/A_animated.gif
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.morphing import (
    ContourExtractor,
    MorphEngine,
    FrameGenerator,
    GifBuilder
)
from src.config import MorphConfig, LetterConfigManager
from src.utils.logger import setup_logging, get_logger, CounterLogger


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Generate letter animation from base and target images'
    )

    parser.add_argument(
        'base_image',
        type=Path,
        help='Path to base shape image (e.g., _Static_O.png)'
    )

    parser.add_argument(
        'target_image',
        type=Path,
        help='Path to target letter image'
    )

    parser.add_argument(
        'output',
        type=Path,
        nargs='?',
        default=None,
        help='Output GIF path (default: auto-generated)'
    )

    parser.add_argument(
        '--background',
        type=Path,
        default=None,
        help='Background image path (default: clean_background.png)'
    )

    parser.add_argument(
        '--letter',
        type=str,
        default=None,
        help='Letter name for custom configuration (e.g., A)'
    )

    parser.add_argument(
        '--preset',
        type=str,
        choices=['default', 'bouncy', 'smooth', 'energetic'],
        default='default',
        help='Animation preset to use'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Output directory (default: output/)'
    )

    return parser.parse_args()


def generate_letter_animation(
    base_image_path: Path,
    target_image_path: Path,
    output_path: Path,
    background_image_path: Path = None,
    letter_config: MorphConfig = None,
    logger = None
) -> Path:
    """Generate a letter animation.

    Args:
        base_image_path: Path to base shape image
        target_image_path: Path to target letter image
        output_path: Path for output GIF
        background_image_path: Optional background image path
        letter_config: Configuration for this letter
        logger: Logger instance

    Returns:
        Path: Path to generated GIF

    Raises:
        Exception: If generation fails
    """
    if logger is None:
        logger = get_logger(__name__)

    counter = CounterLogger(logger)

    try:
        logger.info(f"Generating animation: {base_image_path.name} → {target_image_path.name}")

        # Initialize components
        logger.info("Initializing morphing engine...")
        extractor = ContourExtractor(
            n_points=letter_config.n_points if letter_config else 120
        )
        morph_engine = MorphEngine(config=letter_config)
        frame_generator = FrameGenerator(config=letter_config)
        gif_builder = GifBuilder(config=letter_config, optimize=True)

        # Set background if provided
        if background_image_path and background_image_path.exists():
            logger.info(f"Loading background: {background_image_path}")
            frame_generator.set_background_image(background_image_path)
        else:
            logger.info("Using default white background")

        # Extract and resample contours
        logger.info("Extracting base shape contour...")
        base_contour = extractor.extract_and_resample(base_image_path)
        counter.increment('contours_extracted')

        logger.info("Extracting target shape contour...")
        target_contour = extractor.extract_and_resample(target_image_path)
        counter.increment('contours_extracted')

        # Generate animation sequence
        logger.info("Generating animation sequence...")
        shape_sequence, durations = morph_engine.create_full_animation_sequence(
            base_contour,
            target_contour
        )
        counter.set_counter('frames_in_sequence', len(shape_sequence))

        # Generate frames
        logger.info(f"Rendering {len(shape_sequence)} frames...")
        frames = frame_generator.generate_frames(shape_sequence)
        counter.set_counter('frames_rendered', len(frames))

        # Create GIF
        logger.info(f"Creating GIF: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result_path = gif_builder.create_gif(
            frames,
            output_path,
            durations=durations,
            from_bgr=True
        )

        # Get GIF info
        gif_info = gif_builder.get_gif_info(result_path)
        logger.info(
            f"GIF created successfully: {gif_info['file_size_mb']:.2f} MB, "
            f"{gif_info['n_frames']} frames, "
            f"{gif_info['total_duration_seconds']:.2f}s"
        )

        counter.increment('gifs_created')

        return result_path

    except Exception as e:
        logger.error(f"Failed to generate animation: {e}", exc_info=True)
        raise

    finally:
        counter.log_summary()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Setup logging
    setup_logging(level=args.log_level, log_to_console=True)
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("Vibing Letters - Letter Animation Generator")
    logger.info("=" * 60)

    try:
        # Validate inputs
        if not args.base_image.exists():
            logger.error(f"Base image not found: {args.base_image}")
            sys.exit(1)

        if not args.target_image.exists():
            logger.error(f"Target image not found: {args.target_image}")
            sys.exit(1)

        # Determine background image
        if args.background:
            background_path = args.background
        else:
            background_path = Path('clean_background.png')
            if not background_path.exists():
                logger.warning(f"Default background not found: {background_path}")
                background_path = None

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Auto-generate output name
            target_name = args.target_image.stem
            output_name = f"{target_name}_animated.gif"
            output_path = args.output_dir / output_name

        # Setup configuration
        config_manager = LetterConfigManager()

        # Load preset
        if args.preset != 'default':
            logger.info(f"Loading preset: {args.preset}")
            config_manager.load_preset(args.preset)

        # Get letter-specific config if letter provided
        if args.letter:
            letter_config = config_manager.get_config(args.letter)
            logger.info(f"Using configuration for letter: {args.letter}")
        else:
            letter_config = None
            logger.info("Using default configuration")

        # Generate animation
        result_path = generate_letter_animation(
            args.base_image,
            args.target_image,
            output_path,
            background_path,
            letter_config,
            logger
        )

        logger.info("=" * 60)
        logger.info(f"SUCCESS: Animation saved to {result_path}")
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
