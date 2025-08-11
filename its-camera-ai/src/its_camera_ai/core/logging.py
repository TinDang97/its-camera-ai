"""Logging configuration for ITS Camera AI system.

Provides structured logging with JSON formatting, correlation IDs,
and integration with monitoring systems.
"""

import logging
import logging.config
import sys
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import Settings

# Global logger instance
logger = structlog.get_logger()


def setup_logging(
    settings: Settings,
    enable_json: bool | None = None,
    enable_rich: bool | None = None,
) -> None:
    """Configure application logging.

    Args:
        settings: Application settings
        enable_json: Force JSON formatting (None = auto-detect from env)
        enable_rich: Force Rich formatting (None = auto-detect from env)
    """
    # Determine formatting based on environment
    if enable_json is None:
        enable_json = settings.is_production()
    if enable_rich is None:
        enable_rich = settings.is_development() and not enable_json

    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    handlers = []

    if enable_rich and not enable_json:
        # Rich handler for development
        console = Console(stderr=True)
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        rich_handler.setLevel(settings.log_level)
        handlers.append(rich_handler)
    else:
        # Standard stream handler
        stream_handler = logging.StreamHandler(sys.stdout)
        if enable_json:
            formatter = logging.Formatter(
                fmt="%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(settings.log_level)
        handlers.append(stream_handler)

    # File handler for persistent logging
    if settings.logs_dir:
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = settings.logs_dir / "its-camera-ai.log"

        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=10_000_000,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )

        if enable_json:
            file_formatter = logging.Formatter(
                fmt="%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            file_formatter = logging.Formatter(
                fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(settings.log_level)
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)
    root_logger.handlers.clear()

    for handler in handlers:
        root_logger.addHandler(handler)

    # Configure third-party loggers
    _configure_third_party_loggers(settings.log_level)

    logger.info(
        "Logging configured",
        log_level=settings.log_level,
        json_logging=enable_json,
        rich_logging=enable_rich,
        log_file=(
            str(settings.logs_dir / "its-camera-ai.log") if settings.logs_dir else None
        ),
    )


def _configure_third_party_loggers(log_level: str) -> None:
    """Configure third-party library loggers."""
    # Reduce verbosity of noisy loggers
    noisy_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3",
        "asyncio",
        "kafka",
        "aiokafka",
        "aioredis",
        "sqlalchemy.engine",
        "sqlalchemy.pool",
        "uvicorn.access",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(
            max(logging.WARNING, getattr(logging, log_level))
        )

    # Special handling for specific loggers
    if log_level != "DEBUG":
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("fastapi").setLevel(logging.INFO)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)


class ContextualLogger:
    """Logger with automatic context management."""

    def __init__(self, name: str, **context: Any) -> None:
        self._logger = structlog.get_logger(name)
        self._context = context

    def bind(self, **new_context: Any) -> "ContextualLogger":
        """Create a new logger with additional context."""
        combined_context = {**self._context, **new_context}
        return ContextualLogger(self._logger.name, **combined_context)

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Internal logging method with context."""
        combined_kwargs = {**self._context, **kwargs}
        getattr(self._logger, level)(message, **combined_kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log("error", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log("critical", message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._log("exception", message, **kwargs)


def get_logger(name: str, **context: Any) -> ContextualLogger:
    """Get a contextual logger instance.

    Args:
        name: Logger name
        **context: Additional context to include in all log messages

    Returns:
        ContextualLogger: Configured logger instance
    """
    return ContextualLogger(name, **context)


@contextmanager
def log_context(**context: Any) -> Generator[None, None, None]:
    """Context manager for temporary logging context.

    Args:
        **context: Context to add to all log messages within this block
    """
    token = structlog.contextvars.bind_contextvars(**context)
    try:
        yield
    finally:
        structlog.contextvars.unbind_contextvars(token)


class LoggingMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> ContextualLogger:
        """Get logger for this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(
                self.__class__.__module__ + "." + self.__class__.__name__
            )
        return self._logger


# Performance logging utilities
def log_execution_time(
    logger_instance: ContextualLogger, operation: str, level: str = "info"
) -> Any:
    """Decorator to log execution time of functions.

    Args:
        logger_instance: Logger to use
        operation: Name of the operation being timed
        level: Log level to use

    Returns:
        Decorator function
    """

    def decorator(func: Any) -> Any:
        import functools
        import time

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                getattr(logger_instance, level)(
                    f"{operation} completed",
                    operation=operation,
                    execution_time_seconds=execution_time,
                    success=True,
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger_instance.error(
                    f"{operation} failed",
                    operation=operation,
                    execution_time_seconds=execution_time,
                    error=str(e),
                    success=False,
                )
                raise

        return wrapper

    return decorator
