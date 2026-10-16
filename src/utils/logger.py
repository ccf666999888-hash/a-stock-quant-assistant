from __future__ import annotations

import sys

from loguru import logger


def configure_logger(level: str = "INFO"):
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True)
    return logger

