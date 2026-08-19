# SignalForge

Personal-use quant trading signal generator across crypto and forex, backed
by backtesting and eventually ML, running on free infrastructure (local
machine + GitHub Actions).

**Not investment advice.** Engineering exercise only. No strategy here is
guaranteed profitable — paper trade extensively before risking real capital.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # fill in OANDA_API_TOKEN from your OANDA practice account
python -m pytest
```

Commands are run from the repo root via `python -m` (e.g. `python -m
scripts.fetch_candles`, `python -m pytest`) — this puts the repo root on
`sys.path` so `signalforge` is importable without an editable install.

## Manually fetching candles

```bash
python -m scripts.fetch_candles --asset-class crypto --symbol BTC/USD --timeframe 4h --start 2026-08-01 --end 2026-08-19
python -m scripts.fetch_candles --asset-class forex  --symbol EUR/USD --timeframe 1h --start 2026-08-15 --end 2026-08-19
```

Candles are written to the SQLite database at `data/signalforge.db` (path
configurable via `SIGNALFORGE_DB_PATH` in `.env`).

## Generating signals

Reads candles already stored by `fetch_candles.py`, computes RSI/MACD/Bollinger
Bands/ATR, and prints a buy/sell/hold signal with a confidence score
(0-1, agreement strength across the three directional indicators — not a
probability):

```bash
python -m scripts.generate_signals --asset-class crypto --symbol BTC/USD --timeframe 1h --start 2026-08-01 --end 2026-08-19
```

Add `--confirm-timeframe` to require a faster timeframe to agree before a
signal fires (e.g. a 5m primary signal confirmed by 1m) — both timeframes
need their candles fetched independently first:

```bash
python -m scripts.fetch_candles --asset-class crypto --symbol BTC/USD --timeframe 5m --start 2026-08-15 --end 2026-08-19
python -m scripts.fetch_candles --asset-class crypto --symbol BTC/USD --timeframe 1m --start 2026-08-15 --end 2026-08-19
python -m scripts.generate_signals --asset-class crypto --symbol BTC/USD --timeframe 5m --confirm-timeframe 1m --start 2026-08-15 --end 2026-08-19
```

## Walk-forward backtesting

Simulates the signal generator over stored candles with next-bar-open fills,
realistic per-asset-class fees/slippage, and rolling out-of-sample test
windows (no parameter fitting happens yet — that's Phase 4; this evaluates
the current fixed-vote strategy across multiple historical periods rather
than a single full-history run, so it catches a strategy that only looks
good on one period):

```bash
python -m scripts.run_backtest --asset-class crypto --symbol BTC/USD --timeframe 1h \
    --start 2024-01-01 --end 2026-08-01 --train-days 90 --test-days 30
```

Crypto is always long-only (Kraken is spot-only in this codebase). Forex
defaults to `--position-mode long_short` (OANDA supports shorting), overridable
to `long_only`. A combined "stitched" out-of-sample summary is only shown
when test windows are contiguous (`--step-days` unset or equal to
`--test-days`) — overlapping or gapped windows would corrupt a naive
combined calculation, so only per-window results are shown in that case.
Fee/slippage defaults are printed up front when they're an unverified
estimate rather than a researched figure (see `signalforge/backtest/costs.py`).

## Status

Phase 0 (setup), Phase 1 (Kraken + OANDA data adapters), Phase 2
(indicator/signal layer with optional multi-timeframe confirmation), and
Phase 3 (walk-forward backtesting) are in place. See the project plan for
the full phased roadmap (ML, fundamentals, paper trading, live execution).
