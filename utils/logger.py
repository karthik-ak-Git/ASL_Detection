import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


_DEF_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logger(name: str,
                 logfile: Optional[str] = None,
                 level: int = logging.INFO,
                 max_bytes: int = 1_000_000,
                 backup_count: int = 3) -> logging.Logger:
    """
    Create or get a logger with console and optional rotating file handler.

    - name: logger name
    - logfile: path to file inside logs/ by default; if None, only console.
    - level: logging level
    - max_bytes/backup_count: rotation settings
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured
        logger.setLevel(level)
        return logger

    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(_DEF_FMT))
    logger.addHandler(ch)

    # File handler
    if logfile:
        # Ensure directory exists
        logdir = os.path.dirname(logfile)
        if logdir:
            os.makedirs(logdir, exist_ok=True)
        fh = RotatingFileHandler(
            logfile, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_DEF_FMT))
        logger.addHandler(fh)

    logger.propagate = False
    return logger
