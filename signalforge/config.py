"""Environment-backed application settings.

Loading never validates or raises: a Kraken-only fetch shouldn't require
OANDA_API_TOKEN to be set. Missing-credential errors surface lazily, inside
the adapter that actually needs the value.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    oanda_api_token: str | None
    oanda_account_id: str | None
    oanda_environment: str
    db_path: Path
    log_level: str


def load_settings() -> Settings:
    load_dotenv()  # no-op if .env is absent, e.g. CI with GitHub Secrets already in env

    db_path_raw = os.environ.get("SIGNALFORGE_DB_PATH", "data/signalforge.db")
    db_path = Path(db_path_raw)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path

    return Settings(
        oanda_api_token=os.environ.get("OANDA_API_TOKEN") or None,
        oanda_account_id=os.environ.get("OANDA_ACCOUNT_ID") or None,
        oanda_environment=os.environ.get("OANDA_ENVIRONMENT", "practice"),
        db_path=db_path,
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
