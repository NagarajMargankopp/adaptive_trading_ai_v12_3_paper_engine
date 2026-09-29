# Adaptive Trading AI V9.0

V9.0 is a **paper-only self-correcting decision layer** on top of the V8.27 daily ranker and real 5-minute OHLC execution model.

The online failure learner stores completed outcomes by side, momentum state, volatility state and score-tail bucket. It is strictly causal: the current trade outcome is never known before the decision. Repeated high-loss contexts can be suppressed or risk-reduced. The promotion gate selects a correction only from validation folds; otherwise the system automatically keeps the baseline.

The core XGBoost signal model is not rewritten after every loss. The first self-correction stage is the safer decision/risk layer; model retraining/challenger promotion can be added later after paper-trading data exists.

## Run

```bash
python3 -m app.v90_train \
  --data data/processed/v821_common_multiasset.csv.gz \
  --ohlc data/processed/btcusd_5m_ohlc.csv.gz
```

No live orders are implemented or enabled.
