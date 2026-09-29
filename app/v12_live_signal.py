from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import joblib
import websocket

from app.v12_market_feed import Candle, CandleStore, PUBLIC_WS_URL, DEFAULT_CHANNEL
from app.v923_train import build_price_action_features

DEFAULT_SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD")
REST_URL = "https://api.india.delta.exchange/v2/history/candles"
RESOLUTION = "5m"
WARMUP_BARS = 400
KEEP_BARS = 400
REQUEST_TIMEOUT = 20


def fetch_history(symbol: str, bars: int = WARMUP_BARS) -> pd.DataFrame:
    """Fetch recent completed 5m candles for one symbol from Delta public REST."""
    now = int(time.time())
    current_start = (now // 300) * 300
    end = current_start - 300
    start = end - (bars + 20) * 300
    r = requests.get(
        REST_URL,
        params={"resolution": RESOLUTION, "symbol": symbol, "start": start, "end": end + 1},
        headers={"Accept": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("result")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"No historical candles returned for {symbol}: {payload}")
    out = pd.DataFrame(rows)
    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"Missing fields for {symbol}: {missing}")
    out = out[required].copy()
    out["timestamp"] = pd.to_datetime(out["time"], unit="s", utc=True)
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    out = out[(out["timestamp"].astype("int64") // 10**9) <= end]
    out = out.drop_duplicates("timestamp").sort_values("timestamp").tail(bars).reset_index(drop=True)
    if len(out) < min(300, bars):
        raise RuntimeError(f"Insufficient warmup for {symbol}: {len(out)} rows")
    return out


def build_raw_from_history(hist: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged: pd.DataFrame | None = None
    for sym, x in hist.items():
        q = x[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        q = q.rename(columns={"close": f"{sym.lower()}_close", "volume": f"{sym.lower()}_volume",
                              "open": f"{sym.lower()}_open", "high": f"{sym.lower()}_high",
                              "low": f"{sym.lower()}_low"})
        keep = ["timestamp", f"{sym.lower()}_close", f"{sym.lower()}_volume"]
        if merged is None:
            merged = q[keep]
        else:
            merged = merged.merge(q[keep], on="timestamp", how="inner")
    if merged is None or len(merged) < 300:
        raise RuntimeError("Unable to build synchronized multi-asset history")
    merged = merged.sort_values("timestamp").tail(KEEP_BARS).reset_index(drop=True)
    btc = hist["BTCUSD"].copy()[["timestamp", "open", "high", "low", "close"]]
    btc = btc.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close"})
    btc = merged[["timestamp"]].merge(btc, on="timestamp", how="inner")
    return merged, btc


def percentile(ref: np.ndarray, value: float) -> float:
    if ref.size == 0 or not np.isfinite(value):
        return 0.0
    return float(np.searchsorted(ref, value, side="right") / ref.size)


class LiveSignalEngine:
    """V9.3 Fold-3 inference only. Never sends orders."""
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.imputer = joblib.load(model_dir / "v93_fold3_imputer.joblib")
        self.long_model = joblib.load(model_dir / "v93_fold3_L_ranker.joblib")
        self.short_model = joblib.load(model_dir / "v93_fold3_S_ranker.joblib")
        self.long_ref = np.load(model_dir / "v93_fold3_L_score_reference.npy")
        self.short_ref = np.load(model_dir / "v93_fold3_S_score_reference.npy")
        self.feature_names = json.loads((model_dir / "feature_names.json").read_text())
        self.taken_by_day: set[tuple[str, str]] = set()

    def predict(self, raw_df: pd.DataFrame, btc_ohlc: pd.DataFrame) -> dict:
        features, names = build_price_action_features(raw_df, btc_ohlc)
        if names != self.feature_names:
            raise RuntimeError(f"Feature mismatch: runtime={len(names)} saved={len(self.feature_names)}")
        X = features[self.feature_names].replace([np.inf, -np.inf], np.nan)
        X = self.imputer.transform(X).astype(np.float32)
        long_score = float(self.long_model.predict(X[-1:])[0])
        short_score = float(self.short_model.predict(X[-1:])[0])
        lp = percentile(self.long_ref, long_score)
        sp = percentile(self.short_ref, short_score)
        ts = str(raw_df.iloc[-1]["timestamp"])
        day = pd.Timestamp(raw_df.iloc[-1]["timestamp"]).strftime("%Y-%m-%d")
        long_ok = lp >= 0.90 and (day, "LONG") not in self.taken_by_day
        short_ok = sp >= 0.90 and (day, "SHORT") not in self.taken_by_day
        if long_ok:
            self.taken_by_day.add((day, "LONG"))
        if short_ok:
            self.taken_by_day.add((day, "SHORT"))
        if long_ok and short_ok:
            signal = "BOTH"
        elif long_ok:
            signal = "LONG"
        elif short_ok:
            signal = "SHORT"
        else:
            signal = "NO_SIGNAL"
        return {
            "timestamp": ts,
            "btc_close": float(raw_df.iloc[-1]["btcusd_close"]),
            "long_score": long_score,
            "short_score": short_score,
            "long_percentile": lp,
            "short_percentile": sp,
            "long_candidate": bool(long_ok),
            "short_candidate": bool(short_ok),
            "signal": signal,
        }
