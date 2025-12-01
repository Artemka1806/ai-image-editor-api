import logging
from logging import Logger
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: int = logging.INFO) -> Logger:
    logging.basicConfig(format=LOG_FORMAT, level=level)
    logger = logging.getLogger("ai-image-editor")
    logger.setLevel(level)
    logger.debug("Logging configured", extra={"level": level})
    return logger


def get_logger(name: Optional[str] = None) -> Logger:
    return logging.getLogger(name or "ai-image-editor")
