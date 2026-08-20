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
cp .env.example .env   # fill in TWELVE_DATA_API_KEY (free signup at twelvedata.com)
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

Crypto always uses Kraken's public API (no key/account needed). Forex
defaults to **Twelve Data** (free API-key signup, no account-approval or
country restriction). **OANDA** is also supported — pass `--forex-provider
oanda` — but requires an OANDA account, which isn't available in every
country; kept mainly for possible future live-execution use, since that
needs OANDA's account-scoped endpoints regardless of where historical data
comes from.

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
windows — this evaluates the current fixed-vote strategy across multiple
historical periods rather than a single full-history run, so it catches a
strategy that only looks good on one period:

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

## ML vs. rule-based comparison

Fits a fresh LightGBM classifier per walk-forward window (train range strictly
before its test range, so purging happens by construction — training data
never reaches into the test period) and compares it against the same
rule-based baseline over the identical test windows, same cost model, same
`simulate_trades()`/`compute_metrics()` Phase 3 already established:

```bash
python -m scripts.run_ml_backtest --asset-class crypto --symbol BTC/USD --timeframe 1h \
    --start 2024-01-01 --end 2026-08-01 --train-days 60 --test-days 14
```

**The ML side is always long-only**, with a fixed `--horizon`-bar holding
period realized mechanically through the signal sequence: a "buy" fires when
`P(up)` crosses `--buy-threshold`, and the position is force-exited exactly
`--horizon` bars later regardless of that bar's own prediction — a binary
up/down label can't support principled short decisions (class 0 conflates
"flat" with "crashed"), so ML shorting is out of scope. `--rule-position-mode`
governs *only* the rule-based baseline, which can still run `long_short` for
forex — a deliberate, documented asymmetry: the rule-based side's shorting
logic comes from real bearish indicator readings, and forcing it to
long-only just to make the comparison symmetric would understate capability
it genuinely has. The CLI prints this plainly before the results table.

Features are engineered from price/indicators only by default (`rsi`,
normalized MACD/Bollinger/ATR, momentum returns, a volatility-regime ratio —
see `signalforge/ml/features.py`); pass `--include-rule-based-features` to
additionally give the model the rule-based generator's own `vote_sum`/
`confidence` as features. Reported per window: trading metrics (Sharpe, win
rate, profit factor) alongside probabilistic prediction-quality metrics
(ROC-AUC, Brier score, log loss) — these answer different questions ("does
it predict" vs. "does it make money after costs") and can disagree. A window
is skipped for ML (with a logged reason — insufficient post-purge training
rows, or single-class labels) while the rule-based side still reports
normally. A combined "stitched" summary is shown for each side independently,
and for the ML side only when *every* window in the range produced a fitted
model — a gapped ML history stitched as if contiguous would misrepresent it,
the same trap Phase 3's design avoided for overlapping/gapped test windows.

**Methodology note**: repeatedly re-running this against the same date range
while tuning `--horizon`/`--buy-threshold`/features is itself a form of
overfitting — you become the hyperparameter search. Reserve a final,
most-recent date range and don't touch it until every other decision is
locked in from earlier, disjoint ranges, then run it there once as a genuine
final check. Not enforced in code — discipline only.

## Status

Phase 0 (setup), Phase 1 (Kraken + OANDA data adapters), Phase 2
(indicator/signal layer with optional multi-timeframe confirmation), Phase 3
(walk-forward backtesting), and Phase 4 (LightGBM ML layer, compared against
the rule-based baseline) are in place. See the project plan for the full
phased roadmap (fundamentals, paper trading, live execution).
