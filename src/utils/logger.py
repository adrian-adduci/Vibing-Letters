"""Centralized logging configuration for Vibing Letters.

This module provides structured logging with consistent formatting,
log levels, and performance tracking capabilities.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


# Global logger cache
_loggers = {}


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[Path] = None,
    log_to_console: bool = True,
    log_format: Optional[str] = None
) -> None:
    """Set up the root logger configuration.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path to write logs to
        log_to_console: Whether to log to console (default: True)
        log_format: Custom log format string (uses default if None)
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Default format with timestamp, level, module, and message
    if log_format is None:
        log_format = '%(asctime)s - %(levelname)-8s - %(name)s - %(message)s'

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Name for the logger (typically __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger
    return _loggers[name]


class PerformanceTimer:
    """Context manager for measuring and logging performance metrics.

    Example:
        with PerformanceTimer(logger, "Processing image"):
            # ... do work ...
        # Logs: "Processing image completed in 0.123s"
    """

    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        """Initialize the performance timer.

        Args:
            logger: Logger instance to use
            operation: Description of the operation being timed
            level: Logging level to use (default: INFO)
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.logger.log(self.level, f"{self.operation} - started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer and log the elapsed time."""
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time

        if exc_type is not None:
            self.logger.log(self.level, f"{self.operation} - failed after {elapsed:.3f}s")
        else:
            self.logger.log(self.level, f"{self.operation} - completed in {elapsed:.3f}s")

        return False  # Don't suppress exceptions

    @property
    def elapsed(self) -> float:
        """Get the elapsed time in seconds.

        Returns:
            float: Elapsed time, or 0 if timer hasn't finished
        """
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time


@contextmanager
def log_operation(logger: logging.Logger, operation: str, **context):
    """Context manager for logging operations with context.

    Args:
        logger: Logger instance to use
        operation: Description of the operation
        **context: Additional context to include in logs

    Example:
        with log_operation(logger, "Extracting contour", image_path=path, n_points=120):
            contour = extract_contour(image, n_points)
    """
    # Format context as key=value pairs
    context_str = ", ".join(f"{k}={v}" for k, v in context.items())
    full_message = f"{operation} [{context_str}]" if context else operation

    logger.info(f"Starting: {full_message}")
    start_time = time.perf_counter()

    try:
        yield
        elapsed = time.perf_counter() - start_time
        logger.info(f"Completed: {full_message} (took {elapsed:.3f}s)")
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"Failed: {full_message} (after {elapsed:.3f}s) - {type(e).__name__}: {e}")
        raise


class CounterLogger:
    """Logger with built-in counters for tracking metrics.

    Example:
        counter = CounterLogger(logger)
        counter.increment('frames_generated')
        counter.log_summary()  # Logs all counters
    """

    def __init__(self, logger: logging.Logger):
        """Initialize the counter logger.

        Args:
            logger: Base logger to use
        """
        self.logger = logger
        self.counters = {}
        self.timers = {}

    def increment(self, counter_name: str, amount: int = 1):
        """Increment a counter.

        Args:
            counter_name: Name of the counter
            amount: Amount to increment by (default: 1)
        """
        self.counters[counter_name] = self.counters.get(counter_name, 0) + amount

    def set_counter(self, counter_name: str, value: int):
        """Set a counter to a specific value.

        Args:
            counter_name: Name of the counter
            value: Value to set
        """
        self.counters[counter_name] = value

    def record_time(self, timer_name: str, seconds: float):
        """Record a time measurement.

        Args:
            timer_name: Name of the timer
            seconds: Time in seconds
        """
        if timer_name not in self.timers:
            self.timers[timer_name] = []
        self.timers[timer_name].append(seconds)

    def get_counter(self, counter_name: str) -> int:
        """Get the current value of a counter.

        Args:
            counter_name: Name of the counter

        Returns:
            int: Current counter value (0 if not set)
        """
        return self.counters.get(counter_name, 0)

    def get_timer_stats(self, timer_name: str) -> dict:
        """Get statistics for a timer.

        Args:
            timer_name: Name of the timer

        Returns:
            dict: Statistics including count, total, average, min, max
        """
        if timer_name not in self.timers or not self.timers[timer_name]:
            return {'count': 0, 'total': 0.0, 'average': 0.0, 'min': 0.0, 'max': 0.0}

        times = self.timers[timer_name]
        return {
            'count': len(times),
            'total': sum(times),
            'average': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
        }

    def log_summary(self, level: int = logging.INFO):
        """Log a summary of all counters and timers.

        Args:
            level: Logging level to use (default: INFO)
        """
        self.logger.log(level, "=" * 60)
        self.logger.log(level, "Performance Summary")
        self.logger.log(level, "=" * 60)

        if self.counters:
            self.logger.log(level, "Counters:")
            for name, value in sorted(self.counters.items()):
                self.logger.log(level, f"  {name}: {value}")

        if self.timers:
            self.logger.log(level, "Timers:")
            for name in sorted(self.timers.keys()):
                stats = self.get_timer_stats(name)
                self.logger.log(level,
                    f"  {name}: count={stats['count']}, "
                    f"total={stats['total']:.3f}s, "
                    f"avg={stats['average']:.3f}s, "
                    f"min={stats['min']:.3f}s, "
                    f"max={stats['max']:.3f}s"
                )

        self.logger.log(level, "=" * 60)

    def reset(self):
        """Reset all counters and timers."""
        self.counters.clear()
        self.timers.clear()


# Initialize default logging on module import
setup_logging(level='INFO', log_to_console=True)
