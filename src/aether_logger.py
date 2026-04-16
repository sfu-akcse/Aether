import atexit
import logging
import os
from pathlib import Path


# Guard to ensure logging.shutdown() is registered with atexit only once,
# even if setup_logger() is called multiple times across different modules.
_LOGGING_SHUTDOWN_REGISTERED = False


def _resolve_level(level_name: str) -> int:
    # Convert a level name string (e.g. "DEBUG") to its logging int constant.
    # Falls back to INFO if the name is unrecognized.
    return getattr(logging, level_name.upper(), logging.INFO)


def setup_logger(log_filename: str, logger_name: str) -> logging.Logger:
    global _LOGGING_SHUTDOWN_REGISTERED

    # Return the existing logger if it has already been configured.
    # This prevents duplicate handlers when setup_logger() is called again
    # with the same logger_name (e.g. on module reload).
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    # Allow log level to be overridden at runtime via environment variable.
    log_level = _resolve_level(os.getenv("AETHER_LOG_LEVEL", "INFO"))

    # Resolve the logs/ directory relative to the project root, regardless of
    # where the process is launched from.
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / log_filename

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    # Write logs to file for post-run analysis.
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # Mirror logs to console for real-time monitoring.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.setLevel(log_level)
    # Disable propagation to the root logger to avoid duplicate log entries
    # when other libraries also attach handlers to the root.
    logger.propagate = False
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Flush and close all handlers on program exit so no log records are lost.
    if not _LOGGING_SHUTDOWN_REGISTERED:
        atexit.register(logging.shutdown)
        _LOGGING_SHUTDOWN_REGISTERED = True

    logger.info("Logging initialized. Writing to %s", log_path)
    return logger
