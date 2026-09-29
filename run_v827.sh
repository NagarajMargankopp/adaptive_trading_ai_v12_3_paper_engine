#!/bin/bash
set -euo pipefail
python3 -m app.v827_train \
  --data data/processed/v821_common_multiasset.csv.gz \
  --download-ohlc
