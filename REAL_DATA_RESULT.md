# V9.0 Real-Data Test Report

Dataset:
- User-provided `v821_common_multiasset.csv.gz`
- User-provided Delta BTCUSD 5-minute OHLC `btcusd_5m_ohlc.csv.gz`

Software checks:
- Unit tests: 4 passed
- Causal update / aging / repeated-loss suppression tests: passed
- OHLC coverage: passed
- Signal/execution alignment: passed

## Isolated real-data walk-forward runs

V9.0 uses the V8.27 ranker and actual OHLC execution, with an online Bayesian failure learner that updates only after completed trades.

### Fold 1
- Validation: +0.110619 net return, 91 trades
- Test: +0.087828 net return, 91 trades
- Self-correction suppressed: 0
- Self-correction scaled: 0

### Fold 2
- Validation: +0.174194 net return, 92 trades
- Test: +0.250212 net return, 93 trades
- Self-correction suppressed: 0
- Self-correction scaled: 0

### Fold 3
- Validation: +0.237117 net return, 93 trades
- Test: +0.156461 net return, 72 trades
- Blind future: +0.069433 net return, 38 trades
- Future bootstrap 95% CI for mean net return: +0.000400 to +0.003202
- Self-correction suppressed: 0
- Self-correction scaled: 0

## Interpretation

The self-correction mechanism is functional and causally updates memory after completed trades, but it did not activate on the real-data folds under the balanced evidence threshold. Therefore V9.0 did not improve or change the trade set in this particular historical run.

This is intentional: the promotion logic should not force a correction when the evidence is insufficient.

The underlying V8.27 ranker remains the baseline. V9.0 is research/paper-only and has no live order execution.
