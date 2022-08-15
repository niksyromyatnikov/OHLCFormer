import logging
import os
import sys
import threading
from typing import Optional

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None
_default_logging_level = logging.WARNING
_default_logging_formatter = logging.Formatter('%(pathname)s:%(lineno)s: %(levelname)s: %(message)s')

logging_levels = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}


def set_logging_level(log_level: Optional[str] = None):
    global _default_logging_level

    log_level = os.getenv("OHLCFORMER_VERBOSITY", None) if log_level is None else log_level

    if log_level and log_level in logging_levels:
        _default_logging_level = logging_levels[log_level]
    else:
        logging.getLogger().warning(
            f"Unknown logging level = {log_level}, expected one of: {', '.join(logging_levels.keys())}."
        )


def set_logging_formatting(log_formatting: Optional[str] = None):
    global _default_logging_formatter

    log_formatting = os.getenv("OHLCFORMER_LOG_FORMATTING", None) if log_formatting is None else log_formatting

    try:
        if log_formatting:
            _default_logging_formatter = logging.Formatter(log_formatting)
    except (ValueError, TypeError) as e:
        logging.getLogger().error(e, exc_info=True)


def _get_lib_name() -> str:
    return __name__.split(".")[0]


def _configure_lib_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            return

        _default_handler = logging.StreamHandler()
        _default_handler.flush = sys.stderr.flush
        _default_handler.setFormatter(_default_logging_formatter)

        library_root_logger = logging.getLogger(_get_lib_name())
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_default_logging_level)
        library_root_logger.propagate = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        name = _get_lib_name()

    _configure_lib_root_logger()

    return logging.getLogger(name)
