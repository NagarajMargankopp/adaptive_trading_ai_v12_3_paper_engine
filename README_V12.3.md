# Adaptive Trading AI V12.3 — Live Paper-Position Engine

V12.3 connects the verified V12.2 live signal observer to a deterministic paper-position engine.

- Public Delta market-data only
- No API credentials
- No private endpoints
- No order submission
- V9.3 Fold-3 model remains frozen
- Five-asset 5-minute warmup and live feed
- Signal -> next BTC 5m candle open (1-bar latency) simulated fill
- 0.02% adverse slippage per side by default (research assumption)
- 0.14% round-trip cost assumption
- 0.5% risk budget, 0.5% stop, 1.0% target, 12-bar maximum hold
- SL-first convention when SL and TP are both hit inside the same candle

This is a paper/research system, not proof of future trading performance.
