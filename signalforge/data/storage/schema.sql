CREATE TABLE IF NOT EXISTS ohlcv (
    id          INTEGER PRIMARY KEY,
    symbol      TEXT    NOT NULL,
    asset_class TEXT    NOT NULL CHECK (asset_class IN ('crypto', 'forex')),
    timeframe   TEXT    NOT NULL,
    timestamp   INTEGER NOT NULL,   -- unix epoch seconds, UTC, candle open time
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      REAL,               -- nullable: CoinGecko OHLC has no volume
    source      TEXT    NOT NULL,   -- 'coingecko' | 'oanda'
    ingested_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE (symbol, asset_class, timeframe, timestamp)
);
