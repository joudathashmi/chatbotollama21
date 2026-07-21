from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
from typing import Optional


# ---------------------------------------------------------------------------
# Requirements addressed here:
# - All logs stored in logs/YYYY-MM/YYYY-MM-DD.log
# - Auto create month folder + daily log file
# - Use stdlib logging only
# - UTF-8 encoding
# - Prevent duplicate handlers
# - Support DEBUG/INFO/WARNING/ERROR/CRITICAL + EXCEPTION stack traces
# - Log format includes: timestamp | level | filename | function | line | message
# ---------------------------------------------------------------------------

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(filename)s | %(funcName)s:%(lineno)d | %(message)s"
)
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class LogPaths:
    logs_dir: str


class _DailyPathTimedRotatingFileHandler(TimedRotatingFileHandler):
    """TimedRotatingFileHandler that writes to:

        logs/YYYY-MM/YYYY-MM-DD.log

    It updates its baseFilename at midnight so the daily file is named
    exactly as required.

    Why not rely purely on TimedRotatingFileHandler's `suffix`?
    Because the requirement includes a monthly directory (YYYY-MM) AND a
    fixed daily filename (YYYY-MM-DD.log) under that directory.
    """

    def __init__(
        self,
        *,
        log_paths: LogPaths,
        level: int,
        when: str = "midnight",
        interval: int = 1,
        backupCount: int = 0,
    ):
        self._log_paths = log_paths
        os.makedirs(self._log_paths.logs_dir, exist_ok=True)

        # Determine initial baseFilename for *today*.
        today_filename = self._compute_daily_logfile(datetime.now(timezone.utc))

        # We still let TimedRotatingFileHandler manage rollover timing.
        super().__init__(
            filename=today_filename,
            when=when,
            interval=interval,
            backupCount=backupCount,
            utc=True,
            encoding="utf-8",
            delay=True,  # don't touch filesystem until first write
        )
        self.setLevel(level)

        # Ensure month dir exists for the initial file.
        self._ensure_parent_dir(today_filename)

    def _compute_daily_logfile(self, now_utc: datetime) -> str:
        year_month = now_utc.strftime("%Y-%m")
        year_month_day = now_utc.strftime("%Y-%m-%d")
        month_dir = os.path.join(self._log_paths.logs_dir, year_month)
        filename = f"{year_month_day}.log"
        return os.path.join(month_dir, filename)

    @staticmethod
    def _ensure_parent_dir(path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def doRollover(self) -> None:
        """At rollover, switch baseFilename to the new day's file.

        We intentionally do NOT call the parent's rollover naming logic
        because we want a deterministic YYYY-MM-DD.log with no timestamp
        suffix.
        """
        # Close current stream
        try:
            if self.stream:
                self.stream.close()
                self.stream = None
        except Exception:
            pass

        # Set baseFilename for the new day
        now_utc = datetime.now(timezone.utc)
        new_filename = self._compute_daily_logfile(now_utc)
        self._ensure_parent_dir(new_filename)
        self.baseFilename = new_filename

        # Advance rolloverAt to the next midnight from NOW.
        # Using time.time() (just-after-midnight) gives ~24 hours ahead.
        # The previous code used (rolloverAt - 1) which resolved to the
        # same midnight, so rolloverAt never advanced and doRollover()
        # was called on every subsequent log write.
        self.rolloverAt = self.computeRollover(int(time.time()))

    def emit(self, record: logging.LogRecord) -> None:
        """Ensure correct daily file on the log record's time.

        In case the process runs across midnight and `emit()` is called
        before TimedRotatingFileHandler triggers doRollover(), this keeps
        the filename correct.
        """
        now_utc = datetime.now(timezone.utc)
        expected = self._compute_daily_logfile(now_utc)
        if os.path.abspath(self.baseFilename) != os.path.abspath(expected):
            try:
                # Switch stream to new daily file.
                self.doRollover()
            except Exception:
                # Fallback: attempt to write anyway.
                self._ensure_parent_dir(expected)
                self.baseFilename = expected
        super().emit(record)


# Global guard to avoid duplicate handlers (important with reload).
_CONFIG_LOCK = threading.Lock()
_CONFIGURED = False


def configure_logging(
    *,
    level: int = logging.INFO,
    logs_dir: str = "logs",
    file_level: Optional[int] = None,
    console_level: Optional[int] = None,
    backupCount: int = 0,
) -> Logger:
    """Configure production-ready logging for both console and files.

    Safe to call multiple times; will only configure once.
    """
    global _CONFIGURED
    with _CONFIG_LOCK:
        if _CONFIGURED:
            return logging.getLogger("app")

        # Root-ish format.
        formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FMT)

        # Prevent propagation duplication.
        logger = logging.getLogger("app")
        logger.propagate = False
        logger.setLevel(level)

        # Clear existing handlers if they were added by previous imports
        # within the same interpreter (still avoids duplicates).
        # Note: we keep any already-configured file/console handlers out
        # by checking types.
        existing = list(logger.handlers)
        for h in existing:
            logger.removeHandler(h)

        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level if console_level is not None else level)
        logger.addHandler(console_handler)

        file_level_to_use = file_level if file_level is not None else level
        log_paths = LogPaths(logs_dir=os.path.abspath(logs_dir))

        file_handler = _DailyPathTimedRotatingFileHandler(
            log_paths=log_paths,
            level=file_level_to_use,
            when="midnight",
            interval=1,
            backupCount=backupCount,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        _CONFIGURED = True
        return logger


# Public logger instance for app-wide usage.
logger = configure_logging()


# Example usage (required by spec):
# logger.info("Server started")
# logger.error("Database connection failed")

