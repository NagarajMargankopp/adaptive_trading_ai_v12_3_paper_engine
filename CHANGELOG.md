# Changelog

## V9.0
- Added causal online Bayesian failure learner.
- Learner updates only after completed trades.
- Learner uses context aging and hierarchical shrinkage to avoid reacting to a single loss.
- Learner may reduce risk or suppress repeated high-loss contexts; it never increases risk.
- Added validation-only promotion gate with automatic baseline fallback.
- Added isolated per-fold subprocess execution to prevent resource accumulation.
- Preserved V8.27 daily ranker and actual OHLC intrabar execution.
- Added causality, aging and suppression tests.
## V12.2.2
- Fixed missing `import pandas as pd` in `run_v12_live_signal.py`.
- Live orders remain disabled.
