"""Logging setup for entry points (scripts, future CI jobs).

Library modules never call this — they just use logging.getLogger(__name__).
This is also the seam where future retry/backoff logging hooks in: each
adapter's low-level HTTP call is the natural place for a @retry_with_backoff
decorator to log attempts, once that's built.
"""

from __future__ import annotations

import logging


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
