# Adaptive Trading AI V12.2 — Live V9.3 Signal Observer

V12.2 connects the frozen V9.3 Fold-3 ranker to current Delta public market data.

## Safety

- Public market data only.
- No API credentials.
- No order submission code/path.
- Signal observation only.

## Pipeline

Public REST warmup (400 completed 5m bars per asset) → synchronized BTC/ETH/SOL/BNB/XRP history → V9.3 191-feature builder → frozen Fold-3 LONG/SHORT rankers → score percentiles → signal log.

The observer emits LONG/SHORT/BOTH/NO_SIGNAL and writes `logs/v12/signal/live_signals.csv`.

The warmup is required because V9.3 contains lookbacks up to 288 5-minute bars and cross-asset rolling features.

## Run

```bash
python -m pytest -q
python run_v12_live_signal.py --seconds 900 --warmup-bars 400
```

This phase does not simulate fills and does not place real orders.
