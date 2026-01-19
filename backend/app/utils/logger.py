"""
Logging Configuration and Utilities

Provides structured logging with JSON support for production environments.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional
from app.config import settings, LOGGING_CONFIG


def setup_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Setup and configure logger with proper formatting.
    
    Args:
        name: Logger name (usually __name__ of the module)
    
    Returns:
        Configured logger instance
    
    Example:
        logger = setup_logger(__name__)
        logger.info("Application started")
    """
    # Apply logging configuration
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Get logger
    logger = logging.getLogger(name or __name__)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Logger instance
    
    Example:
        from app.utils.logger import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


# Create default logger
logger = setup_logger(__name__)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    
    Example:
        class MyClass(LoggerMixin):
            def process(self):
                self.logger.info("Processing...")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return get_logger(self.__class__.__name__)


if __name__ == "__main__":
    # Test logging
    test_logger = setup_logger("test")
    
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    
    print("✅ Logger configured successfully!")
