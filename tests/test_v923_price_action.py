from pathlib import Path
import numpy as np

from app.v923_train import build_price_action_features
from app.v827_train import load_dataset
from app.delta_ohlc import load_ohlc

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/v821_common_multiasset.csv.gz"
OHLC = ROOT / "data/processed/btcusd_5m_ohlc.csv.gz"


def test_price_action_features_are_present():
    df = load_dataset(DATA).iloc[:2000].copy()
    ohlc = load_ohlc(OHLC)
    aligned = ohlc.set_index("timestamp").reindex(df.timestamp)[["open", "high", "low", "close"]]
    pa_df = df.copy()
    pa_df["btcusd_open"] = aligned["open"].to_numpy(float)
    pa_df["btcusd_high"] = aligned["high"].to_numpy(float)
    pa_df["btcusd_low"] = aligned["low"].to_numpy(float)
    pa_df["btcusd_close_ohlc"] = aligned["close"].to_numpy(float)
    pa_df["btcusd_close"] = pa_df["btcusd_close_ohlc"]
    f, names = build_price_action_features(pa_df, ohlc)
    pa_names = [n for n in names if n.startswith("pa_")]
    assert len(pa_names) == 64
    assert all(n in f.columns for n in pa_names)


def test_price_action_features_are_causal():
    df = load_dataset(DATA).iloc[:2500].copy()
    ohlc = load_ohlc(OHLC)
    aligned = ohlc.set_index("timestamp").reindex(df.timestamp)[["open", "high", "low", "close"]]

    def make_frame(a):
        x = a.copy()
        x["btcusd_open"] = aligned["open"].to_numpy(float)
        x["btcusd_high"] = aligned["high"].to_numpy(float)
        x["btcusd_low"] = aligned["low"].to_numpy(float)
        x["btcusd_close_ohlc"] = aligned["close"].to_numpy(float)
        x["btcusd_close"] = x["btcusd_close_ohlc"]
        return x

    _, names_a = build_price_action_features(make_frame(df), ohlc)
    future_idx = 2200
    altered = df.copy()
    # Change only one future candle. Features through future_idx-1 must not change.
    altered_ohlc = ohlc.copy()
    j = altered_ohlc.index[altered_ohlc["timestamp"] == df.loc[future_idx, "timestamp"]]
    if len(j) != 1:
        raise AssertionError("test timestamp not found in OHLC")
    altered_ohlc.loc[j[0], "open"] *= 1.20
    altered_ohlc.loc[j[0], "high"] *= 1.20
    altered_ohlc.loc[j[0], "low"] *= 1.20
    altered_ohlc.loc[j[0], "close"] *= 1.20

    altered_aligned = altered_ohlc.set_index("timestamp").reindex(df.timestamp)[["open", "high", "low", "close"]]
    x2 = altered.copy()
    x2["btcusd_open"] = altered_aligned["open"].to_numpy(float)
    x2["btcusd_high"] = altered_aligned["high"].to_numpy(float)
    x2["btcusd_low"] = altered_aligned["low"].to_numpy(float)
    x2["btcusd_close_ohlc"] = altered_aligned["close"].to_numpy(float)
    x2["btcusd_close"] = x2["btcusd_close_ohlc"]

    a, _ = build_price_action_features(make_frame(df), ohlc)
    b, _ = build_price_action_features(x2, altered_ohlc)
    cols = [c for c in names_a if c.startswith("pa_")]
    assert np.allclose(
        a.loc[:future_idx - 1, cols].to_numpy(float),
        b.loc[:future_idx - 1, cols].to_numpy(float),
        equal_nan=True,
    )
