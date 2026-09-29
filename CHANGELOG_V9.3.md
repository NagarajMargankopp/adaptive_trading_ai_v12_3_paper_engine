# V9.3 Change Log

## V9.3 — Price-Action-Aware Ranker

Added 64 causal BTC price-action / market-structure features while preserving the original 127 V8.27 features.

The adaptive failure learner is disabled in this release so the effect of price-action information can be evaluated independently.

Added tests covering feature presence and future-data causality.

Validated with the existing three expanding walk-forward folds and the final reserved future block.
