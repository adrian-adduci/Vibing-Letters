"""Unit tests for validators module.

Tests cover validation functions for file paths, images, filenames,
numeric ranges, and configuration dictionaries.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from src.utils.validators import (
    ValidationError,
    validate_file_path,
    validate_image_file,
    sanitize_filename,
    validate_numeric_range,
    validate_color_tuple,
    validate_config_dict,
    validate_image_dimensions,
    validate_letter
)


class TestValidateFilePath:
    """Test suite for validate_file_path function."""

    def test_valid_existing_file(self, tmp_path):
        """Test validation of a valid existing file."""
        # Create a temporary file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        # Should not raise exception
        result = validate_file_path(test_file, must_exist=True)
        assert result.exists()
        assert result.is_file()

    def test_nonexistent_file_with_must_exist(self, tmp_path):
        """Test validation fails for nonexistent file when must_exist=True."""
        nonexistent = tmp_path / "nonexistent.txt"

        with pytest.raises(ValidationError, match="does not exist"):
            validate_file_path(nonexistent, must_exist=True)

    def test_nonexistent_file_without_must_exist(self, tmp_path):
        """Test validation succeeds for nonexistent file when must_exist=False."""
        nonexistent = tmp_path / "nonexistent.txt"

        # Should not raise exception
        result = validate_file_path(nonexistent, must_exist=False)
        assert isinstance(result, Path)

    def test_path_traversal_prevention(self, tmp_path):
        """Test prevention of path traversal attacks."""
        # Create a file outside the base directory
        base_dir = tmp_path / "safe"
        base_dir.mkdir()

        unsafe_path = tmp_path / "safe" / ".." / "unsafe.txt"

        with pytest.raises(ValidationError, match="outside allowed directory"):
            validate_file_path(unsafe_path, must_exist=False, base_directory=base_dir)

    def test_allowed_extensions_valid(self, tmp_path):
        """Test validation with allowed extensions."""
        test_file = tmp_path / "test.png"
        test_file.write_text("test")

        result = validate_file_path(
            test_file,
            must_exist=True,
            allowed_extensions={'.png', '.jpg'}
        )
        assert result.exists()

    def test_allowed_extensions_invalid(self, tmp_path):
        """Test validation fails with disallowed extension."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValidationError, match="not in allowed extensions"):
            validate_file_path(
                test_file,
                must_exist=True,
                allowed_extensions={'.png', '.jpg'}
            )


class TestSanitizeFilename:
    """Test suite for sanitize_filename function."""

    def test_alphanumeric_unchanged(self):
        """Test that alphanumeric filenames are unchanged."""
        filename = "test_file_123.txt"
        result = sanitize_filename(filename)
        assert result == filename

    def test_special_chars_replaced(self):
        """Test that special characters are replaced with underscores."""
        filename = "test<file>name.txt"
        result = sanitize_filename(filename)
        assert result == "test_file_name.txt"

    def test_path_separators_removed(self):
        """Test that path separators are removed (path traversal prevention)."""
        filename = "../../../etc/passwd"
        result = sanitize_filename(filename)
        assert ".." not in result
        assert "/" not in result
        assert "\\" not in result

    def test_max_length_truncation(self):
        """Test that long filenames are truncated."""
        filename = "a" * 300 + ".txt"
        result = sanitize_filename(filename, max_length=255)
        assert len(result) <= 255
        assert result.endswith(".txt")  # Extension preserved

    def test_empty_after_sanitization_raises_error(self):
        """Test that empty filename after sanitization raises error."""
        filename = "<<>>"  # All special chars

        with pytest.raises(ValidationError, match="empty after sanitization"):
            sanitize_filename(filename)


class TestValidateNumericRange:
    """Test suite for validate_numeric_range function."""

    def test_value_within_range(self):
        """Test validation of value within range."""
        result = validate_numeric_range(50, min_value=0, max_value=100)
        assert result == 50

    def test_value_below_minimum(self):
        """Test validation fails for value below minimum."""
        with pytest.raises(ValidationError, match="must be >="):
            validate_numeric_range(5, min_value=10, max_value=100)

    def test_value_above_maximum(self):
        """Test validation fails for value above maximum."""
        with pytest.raises(ValidationError, match="must be <="):
            validate_numeric_range(150, min_value=0, max_value=100)

    def test_boundary_values(self):
        """Test validation at boundary values."""
        # Minimum boundary
        result_min = validate_numeric_range(0, min_value=0, max_value=100)
        assert result_min == 0

        # Maximum boundary
        result_max = validate_numeric_range(100, min_value=0, max_value=100)
        assert result_max == 100

    def test_non_numeric_value(self):
        """Test validation fails for non-numeric value."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_numeric_range("not a number", min_value=0, max_value=100)


class TestValidateColorTuple:
    """Test suite for validate_color_tuple function."""

    def test_valid_rgb_tuple(self):
        """Test validation of valid RGB tuple."""
        color = (255, 128, 0)
        result = validate_color_tuple(color)
        assert result == color

    def test_invalid_tuple_length(self):
        """Test validation fails for wrong tuple length."""
        with pytest.raises(ValidationError, match="must have 3 values"):
            validate_color_tuple((255, 128))

    def test_out_of_range_values(self):
        """Test validation fails for out-of-range values."""
        with pytest.raises(ValidationError, match="must be in range"):
            validate_color_tuple((255, 300, 0))

        with pytest.raises(ValidationError, match="must be in range"):
            validate_color_tuple((255, -10, 0))

    def test_non_integer_values(self):
        """Test validation fails for non-integer values."""
        with pytest.raises(ValidationError, match="must be int"):
            validate_color_tuple((255.5, 128, 0))

    def test_boundary_values(self):
        """Test validation at boundary values."""
        # Minimum boundaries
        result_min = validate_color_tuple((0, 0, 0))
        assert result_min == (0, 0, 0)

        # Maximum boundaries
        result_max = validate_color_tuple((255, 255, 255))
        assert result_max == (255, 255, 255)


class TestValidateConfigDict:
    """Test suite for validate_config_dict function."""

    def test_valid_config(self):
        """Test validation of valid configuration."""
        config = {'key1': 'value1', 'key2': 'value2'}
        result = validate_config_dict(config)
        assert result == config

    def test_missing_required_keys(self):
        """Test validation fails for missing required keys."""
        config = {'key1': 'value1'}
        required = {'key1', 'key2', 'key3'}

        with pytest.raises(ValidationError, match="Missing required config keys"):
            validate_config_dict(config, required_keys=required)

    def test_extra_keys_not_allowed(self):
        """Test validation fails for extra keys when not allowed."""
        config = {'key1': 'value1', 'key2': 'value2', 'extra': 'value'}
        allowed = {'key1', 'key2'}

        with pytest.raises(ValidationError, match="Unknown config keys"):
            validate_config_dict(config, allowed_keys=allowed)

    def test_custom_validators(self):
        """Test validation with custom validator functions."""
        config = {'age': 25, 'name': 'Test'}

        validators = {
            'age': lambda x: validate_numeric_range(x, 0, 120, 'age'),
            'name': lambda x: x if isinstance(x, str) else ValueError("name must be string")
        }

        result = validate_config_dict(config, validators=validators)
        assert result == config

    def test_custom_validator_failure(self):
        """Test validation fails when custom validator fails."""
        config = {'age': 150}

        validators = {
            'age': lambda x: validate_numeric_range(x, 0, 120, 'age')
        }

        with pytest.raises(ValidationError, match="Validation failed"):
            validate_config_dict(config, validators=validators)


class TestValidateImageDimensions:
    """Test suite for validate_image_dimensions function."""

    def test_valid_dimensions(self):
        """Test validation of valid dimensions."""
        result = validate_image_dimensions(800, 600)
        assert result == (800, 600)

    def test_dimensions_below_minimum(self):
        """Test validation fails for dimensions below minimum."""
        with pytest.raises(ValidationError):
            validate_image_dimensions(0, 600, min_width=1, min_height=1)

    def test_dimensions_above_maximum(self):
        """Test validation fails for dimensions above maximum."""
        with pytest.raises(ValidationError):
            validate_image_dimensions(5000, 600, max_width=4096, max_height=4096)

    def test_boundary_dimensions(self):
        """Test validation at boundary dimensions."""
        result = validate_image_dimensions(
            4096, 4096,
            min_width=1, min_height=1,
            max_width=4096, max_height=4096
        )
        assert result == (4096, 4096)

    def test_minimum_dimensions(self):
        """Test validation at minimum dimensions."""
        result = validate_image_dimensions(1, 1, min_width=1, min_height=1)
        assert result == (1, 1)


class TestValidateLetter:
    """Test suite for validate_letter function."""

    def test_valid_letter(self):
        """Test validation of valid single letter."""
        result = validate_letter('A')
        assert result == 'A'

        result = validate_letter('z')
        assert result == 'Z'  # Should be uppercased

    def test_multiple_characters(self):
        """Test validation fails for multiple characters."""
        with pytest.raises(ValidationError, match="must be a single character"):
            validate_letter('AB')

    def test_non_alphabetic(self):
        """Test validation fails for non-alphabetic character."""
        with pytest.raises(ValidationError, match="must be alphabetic"):
            validate_letter('1')

        with pytest.raises(ValidationError, match="must be alphabetic"):
            validate_letter('!')

    def test_empty_string(self):
        """Test validation fails for empty string."""
        with pytest.raises(ValidationError, match="must be a single character"):
            validate_letter('')

    def test_non_string_input(self):
        """Test validation fails for non-string input."""
        with pytest.raises(ValidationError, match="must be a string"):
            validate_letter(123)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
