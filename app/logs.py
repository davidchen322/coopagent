"""Application logging.

Named logs.py, not logging.py, so it can't shadow the standard library.

Two levels, split on privacy rather than verbosity:

  INFO  (default)  metadata only — timings, chunk counts, citation tags,
                   notice topics, redactions. Enough to see what the app did.
  DEBUG            adds the resident's question text and previews of the
                   retrieved passages.

That split is deliberate. Residents ask about disputes, arrears and medical
accommodations, and DEBUG writes those questions — plus the co-op document text
that answered them — to disk. The co-op's documents are gitignored and kept
off the network by design; a log file is the easiest way to undo that without
noticing. Turn DEBUG on to diagnose something, then turn it back off.

Set LOG_LEVEL and (optionally) LOG_FILE in .env. If you set LOG_FILE, put it
somewhere already gitignored — data/ is.
"""
from __future__ import annotations

import logging
import sys

from app.config import settings

FORMAT = "%(asctime)s %(levelname)-7s %(name)-16s %(message)s"
DATEFMT = "%H:%M:%S"

_configured = False


def setup_logging() -> None:
    """Configure the 'coopagent' logger tree once. Safe to call repeatedly."""
    global _configured
    if _configured:
        return

    root = logging.getLogger("coopagent")
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))
    # Don't propagate to the root logger — uvicorn installs a handler there and
    # every line would print twice.
    root.propagate = False

    formatter = logging.Formatter(FORMAT, DATEFMT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"coopagent.{name}")
