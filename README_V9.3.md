# Adaptive Trading AI V9.3 — Price-Action-Aware Ranker

V9.3 is a research-only evolution of the V8.27 ranker.

## Architecture

`V8.27 multi-asset features (127)`

`+`

`64 causal BTC price-action / market-structure features`

`-> XGBRanker`

`-> existing locked execution policy`

The V9.2.2 adaptive failure learner is intentionally **OFF** in the V9.3 isolation test.

## New feature families

- Candle geometry
- Candle structure patterns
- Prior-range position and breakout strength
- Higher-high / higher-low / lower-high / lower-low structure
- Trend efficiency / directional pressure
- Relative candle-range regime

All rolling market-structure thresholds use prior candles (`shift(1)`), so a current candle cannot create its own breakout threshold.

## Run

```bash
python3 -m app.v923_train \
  --data data/processed/v821_common_multiasset.csv.gz \
  --ohlc data/processed/btcusd_5m_ohlc.csv.gz \
  --output-dir logs/v93_final
```

Run a single fold:

```bash
python3 -m app.v923_train \
  --data data/processed/v821_common_multiasset.csv.gz \
  --ohlc data/processed/btcusd_5m_ohlc.csv.gz \
  --fold 1 \
  --output-dir logs/v93_final
```

## Validation

```bash
python3 -m pytest -q
```

Final package test result: **6 passed**.

## Safety / research status

- No exchange order placement
- No live execution
- No paper-trading promotion logic
- Historical backtest only
