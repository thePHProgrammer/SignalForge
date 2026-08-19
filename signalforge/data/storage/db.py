"""The only module that touches sqlite3 — adapters never import it directly."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from signalforge.data.models import Candle

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA_PATH.read_text())
    conn.commit()


def upsert_candles(conn: sqlite3.Connection, candles: list[Candle]) -> int:
    conn.executemany(
        """
        INSERT INTO ohlcv (symbol, asset_class, timeframe, timestamp, open, high, low, close, volume, source)
        VALUES (:symbol, :asset_class, :timeframe, :timestamp, :open, :high, :low, :close, :volume, :source)
        ON CONFLICT (symbol, asset_class, timeframe, timestamp) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            volume = excluded.volume,
            source = excluded.source
        """,
        [
            {
                "symbol": c.symbol,
                "asset_class": c.asset_class,
                "timeframe": c.timeframe,
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
                "source": c.source,
            }
            for c in candles
        ],
    )
    conn.commit()
    return len(candles)


def query_candles(
    conn: sqlite3.Connection,
    symbol: str,
    asset_class: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> list[Candle]:
    rows = conn.execute(
        """
        SELECT symbol, asset_class, timeframe, timestamp, open, high, low, close, volume, source
        FROM ohlcv
        WHERE symbol = ? AND asset_class = ? AND timeframe = ?
          AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp ASC
        """,
        (symbol, asset_class, timeframe, int(start.timestamp()), int(end.timestamp())),
    ).fetchall()
    return [
        Candle(
            symbol=r[0],
            asset_class=r[1],
            timeframe=r[2],
            timestamp=r[3],
            open=r[4],
            high=r[5],
            low=r[6],
            close=r[7],
            volume=r[8],
            source=r[9],
        )
        for r in rows
    ]
