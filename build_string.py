"""Build text strings from pre-generated letter GIFs.

This script replaces string_builder.py with improved error handling,
logging, and configuration options.

Usage:
    python build_string.py "TEXT" [output_name]

Example:
    python build_string.py "HELLO" output/hello.gif
"""

import sys
import argparse
from pathlib import Path
from PIL import Image

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logging, get_logger, CounterLogger
from src.utils.validators import sanitize_filename, validate_letter


def parse_arguments():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Build text string from letter GIF animations'
    )

    parser.add_argument(
        'text',
        type=str,
        help='Text string to create'
    )

    parser.add_argument(
        'output',
        type=Path,
        nargs='?',
        default=None,
        help='Output GIF path (default: output/<text>.gif)'
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('input'),
        help='Directory containing letter GIFs (default: input/)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('output'),
        help='Output directory (default: output/)'
    )

    parser.add_argument(
        '--suffix',
        type=str,
        default='_animated',
        help='Suffix for letter GIF files (default: _animated)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )

    return parser.parse_args()


def load_letter_gif(
    letter: str,
    input_dir: Path,
    suffix: str = '_animated',
    logger = None
) -> tuple[list[Image.Image], list[int]]:
    """Load a letter GIF and extract frames and durations.

    Args:
        letter: Letter to load (A-Z)
        input_dir: Directory containing letter GIFs
        suffix: Filename suffix before .gif
        logger: Logger instance

    Returns:
        tuple: (list of PIL Images, list of frame durations)

    Raises:
        FileNotFoundError: If letter GIF not found
        ValueError: If GIF cannot be loaded
    """
    if logger is None:
        logger = get_logger(__name__)

    # Validate letter
    letter = validate_letter(letter)

    # Construct filename
    filename = f"{letter}{suffix}.gif"
    filepath = input_dir / filename

    if not filepath.exists():
        # Try without suffix
        filename_no_suffix = f"{letter}.gif"
        filepath = input_dir / filename_no_suffix

        if not filepath.exists():
            raise FileNotFoundError(
                f"Letter GIF not found: {input_dir / filename} or {input_dir / filename_no_suffix}"
            )

    logger.debug(f"Loading letter '{letter}' from {filepath}")

    try:
        frames = []
        durations = []

        with Image.open(filepath) as img:
            # Extract all frames
            frame_count = 0
            try:
                while True:
                    frames.append(img.copy().convert('RGB'))
                    durations.append(img.info.get('duration', 100))
                    img.seek(img.tell() + 1)
                    frame_count += 1
            except EOFError:
                pass  # End of frames

            logger.debug(f"Loaded {frame_count} frames for letter '{letter}'")

        return frames, durations

    except Exception as e:
        raise ValueError(f"Failed to load GIF {filepath}: {e}")


def build_text_animation(
    text: str,
    input_dir: Path,
    output_path: Path,
    suffix: str = '_animated',
    logger = None
) -> Path:
    """Build a text string animation from letter GIFs.

    Args:
        text: Text string to create
        input_dir: Directory containing letter GIFs
        output_path: Path for output GIF
        suffix: Filename suffix for letter GIFs
        logger: Logger instance

    Returns:
        Path: Path to created GIF

    Raises:
        ValueError: If text is empty or contains invalid characters
        FileNotFoundError: If letter GIFs are missing
    """
    if logger is None:
        logger = get_logger(__name__)

    counter = CounterLogger(logger)

    try:
        # Validate text
        text = text.upper().strip()

        if not text:
            raise ValueError("Text string is empty")

        # Filter to only alphabetic characters
        letters = [c for c in text if c.isalpha()]

        if not letters:
            raise ValueError("Text contains no alphabetic characters")

        logger.info(f"Building animation for text: {text} ({len(letters)} letters)")

        # Load all letter GIFs
        all_frames = []
        all_durations = []

        for letter in letters:
            try:
                frames, durations = load_letter_gif(letter, input_dir, suffix, logger)
                all_frames.extend(frames)
                all_durations.extend(durations)
                counter.increment('letters_loaded')
                logger.info(f"  Loaded letter '{letter}': {len(frames)} frames")
            except FileNotFoundError as e:
                logger.error(f"  Missing letter '{letter}': {e}")
                raise
            except ValueError as e:
                logger.error(f"  Error loading letter '{letter}': {e}")
                raise

        logger.info(f"Total frames: {len(all_frames)}, total duration: {sum(all_durations)}ms")
        counter.set_counter('total_frames', len(all_frames))

        # Create output GIF
        logger.info(f"Creating output GIF: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_frames[0].save(
            str(output_path),
            save_all=True,
            append_images=all_frames[1:],
            duration=all_durations,
            loop=0,
            optimize=True
        )

        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"GIF created: {file_size_mb:.2f} MB")

        counter.increment('gifs_created')

        return output_path

    except Exception as e:
        logger.error(f"Failed to build text animation: {e}", exc_info=True)
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
    logger.info("Vibing Letters - Text String Builder")
    logger.info("=" * 60)

    try:
        # Validate input directory
        if not args.input_dir.exists():
            logger.error(f"Input directory not found: {args.input_dir}")
            sys.exit(1)

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            # Auto-generate output name
            sanitized_text = sanitize_filename(args.text.lower())
            output_name = f"{sanitized_text}.gif"
            output_path = args.output_dir / output_name

        # Build animation
        result_path = build_text_animation(
            args.text,
            args.input_dir,
            output_path,
            args.suffix,
            logger
        )

        logger.info("=" * 60)
        logger.info(f"SUCCESS: Text animation saved to {result_path}")
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
