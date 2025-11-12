"""Input validation and sanitization for Vibing Letters.

This module provides validation functions for file paths, images,
numeric ranges, and configuration dictionaries to prevent security
vulnerabilities and improve error handling.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union
import imghdr


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = True,
    must_be_file: bool = True,
    allowed_extensions: Optional[set[str]] = None,
    base_directory: Optional[Path] = None
) -> Path:
    """Validate and sanitize a file path.

    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist (default: True)
        must_be_file: Whether the path must be a file, not directory (default: True)
        allowed_extensions: Set of allowed file extensions (e.g., {'.png', '.jpg'})
        base_directory: If provided, ensure path is within this directory (prevents path traversal)

    Returns:
        Path: Validated and resolved absolute path

    Raises:
        ValidationError: If validation fails
    """
    try:
        # Convert to Path object
        path = Path(file_path)

        # Resolve to absolute path (handles .., symlinks, etc.)
        path = path.resolve()

        # Check if within base directory (path traversal prevention)
        if base_directory is not None:
            base_directory = Path(base_directory).resolve()
            try:
                path.relative_to(base_directory)
            except ValueError:
                raise ValidationError(
                    f"Path {path} is outside allowed directory {base_directory}"
                )

        # Check existence
        if must_exist and not path.exists():
            raise ValidationError(f"Path does not exist: {path}")

        # Check if file (not directory)
        if must_exist and must_be_file and not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")

        # Check file extension
        if allowed_extensions is not None:
            if path.suffix.lower() not in allowed_extensions:
                raise ValidationError(
                    f"File extension {path.suffix} not in allowed extensions: {allowed_extensions}"
                )

        return path

    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Invalid file path: {e}")


def validate_image_file(
    file_path: Union[str, Path],
    allowed_formats: Optional[set[str]] = None,
    max_size_mb: float = 50.0,
    base_directory: Optional[Path] = None
) -> Path:
    """Validate an image file for security and format.

    Args:
        file_path: Path to the image file
        allowed_formats: Set of allowed image formats (e.g., {'png', 'jpeg'})
                        If None, allows: png, jpeg, jpg, bmp, gif
        max_size_mb: Maximum file size in megabytes (default: 50.0)
        base_directory: If provided, ensure path is within this directory

    Returns:
        Path: Validated image file path

    Raises:
        ValidationError: If validation fails
    """
    # Default allowed formats
    if allowed_formats is None:
        allowed_formats = {'png', 'jpeg', 'jpg', 'bmp', 'gif'}

    # Convert format names to extensions for validate_file_path
    allowed_extensions = {f'.{fmt}' for fmt in allowed_formats}

    # Validate basic file path
    path = validate_file_path(
        file_path,
        must_exist=True,
        must_be_file=True,
        allowed_extensions=allowed_extensions,
        base_directory=base_directory
    )

    # Check file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        raise ValidationError(
            f"Image file too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)"
        )

    # Validate actual image format using magic bytes (not just extension)
    detected_format = imghdr.what(path)
    if detected_format is None:
        raise ValidationError(f"File is not a valid image: {path}")

    if detected_format not in allowed_formats:
        raise ValidationError(
            f"Image format {detected_format} not in allowed formats: {allowed_formats}"
        )

    return path


def sanitize_filename(
    filename: str,
    max_length: int = 255,
    allowed_chars: Optional[str] = None
) -> str:
    """Sanitize a filename for safe filesystem use.

    Args:
        filename: Filename to sanitize
        max_length: Maximum filename length (default: 255)
        allowed_chars: Regex pattern of allowed characters
                      Default: alphanumeric, underscore, hyphen, dot

    Returns:
        str: Sanitized filename

    Raises:
        ValidationError: If filename becomes empty after sanitization
    """
    if allowed_chars is None:
        allowed_chars = r'[^a-zA-Z0-9._-]'

    # Remove path separators (prevent directory traversal)
    filename = filename.replace('/', '').replace('\\', '')

    # Replace disallowed characters with underscore
    filename = re.sub(allowed_chars, '_', filename)

    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')

    # Limit length
    if len(filename) > max_length:
        # Preserve extension if present
        name, ext = os.path.splitext(filename)
        max_name_length = max_length - len(ext)
        filename = name[:max_name_length] + ext

    # Ensure filename is not empty
    if not filename:
        raise ValidationError("Filename is empty after sanitization")

    # Prevent reserved names on Windows
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    name_without_ext = os.path.splitext(filename)[0].upper()
    if name_without_ext in reserved_names:
        filename = f"_{filename}"

    return filename


def validate_numeric_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    value_name: str = "value"
) -> Union[int, float]:
    """Validate that a numeric value is within a specified range.

    Args:
        value: Value to validate
        min_value: Minimum allowed value (inclusive), None for no minimum
        max_value: Maximum allowed value (inclusive), None for no maximum
        value_name: Name of the value for error messages

    Returns:
        Union[int, float]: The validated value

    Raises:
        ValidationError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{value_name} must be numeric, got {type(value).__name__}")

    if min_value is not None and value < min_value:
        raise ValidationError(f"{value_name} must be >= {min_value}, got {value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{value_name} must be <= {max_value}, got {value}")

    return value


def validate_color_tuple(
    color: tuple,
    value_name: str = "color"
) -> tuple[int, int, int]:
    """Validate an RGB color tuple.

    Args:
        color: Tuple to validate
        value_name: Name of the value for error messages

    Returns:
        tuple[int, int, int]: Validated RGB tuple

    Raises:
        ValidationError: If color tuple is invalid
    """
    if not isinstance(color, tuple):
        raise ValidationError(f"{value_name} must be a tuple, got {type(color).__name__}")

    if len(color) != 3:
        raise ValidationError(f"{value_name} must have 3 values (R, G, B), got {len(color)}")

    for i, component in enumerate(color):
        if not isinstance(component, int):
            raise ValidationError(
                f"{value_name}[{i}] must be int, got {type(component).__name__}"
            )
        if not 0 <= component <= 255:
            raise ValidationError(
                f"{value_name}[{i}] must be in range [0, 255], got {component}"
            )

    return tuple(color)


def validate_config_dict(
    config: Dict[str, Any],
    required_keys: Optional[set[str]] = None,
    allowed_keys: Optional[set[str]] = None,
    validators: Optional[Dict[str, callable]] = None
) -> Dict[str, Any]:
    """Validate a configuration dictionary.

    Args:
        config: Configuration dictionary to validate
        required_keys: Set of required keys (None to skip check)
        allowed_keys: Set of allowed keys (None to allow any keys)
        validators: Dict mapping keys to validator functions

    Returns:
        Dict[str, Any]: Validated configuration dictionary

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Config must be a dictionary, got {type(config).__name__}")

    # Check required keys
    if required_keys is not None:
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ValidationError(f"Missing required config keys: {missing_keys}")

    # Check allowed keys
    if allowed_keys is not None:
        extra_keys = set(config.keys()) - allowed_keys
        if extra_keys:
            raise ValidationError(f"Unknown config keys: {extra_keys}")

    # Run validators
    if validators is not None:
        for key, validator_func in validators.items():
            if key in config:
                try:
                    config[key] = validator_func(config[key])
                except Exception as e:
                    raise ValidationError(f"Validation failed for config key '{key}': {e}")

    return config


def validate_image_dimensions(
    width: int,
    height: int,
    min_width: int = 1,
    min_height: int = 1,
    max_width: int = 4096,
    max_height: int = 4096
) -> tuple[int, int]:
    """Validate image dimensions.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        min_width: Minimum allowed width (default: 1)
        min_height: Minimum allowed height (default: 1)
        max_width: Maximum allowed width (default: 4096)
        max_height: Maximum allowed height (default: 4096)

    Returns:
        tuple[int, int]: Validated (width, height)

    Raises:
        ValidationError: If dimensions are invalid
    """
    validate_numeric_range(width, min_width, max_width, "width")
    validate_numeric_range(height, min_height, max_height, "height")

    return (width, height)


def validate_letter(letter: str) -> str:
    """Validate that a string is a single alphabetic letter.

    Args:
        letter: String to validate

    Returns:
        str: Validated letter (uppercased)

    Raises:
        ValidationError: If not a single alphabetic letter
    """
    if not isinstance(letter, str):
        raise ValidationError(f"Letter must be a string, got {type(letter).__name__}")

    if len(letter) != 1:
        raise ValidationError(f"Letter must be a single character, got: {letter}")

    if not letter.isalpha():
        raise ValidationError(f"Letter must be alphabetic, got: {letter}")

    return letter.upper()
