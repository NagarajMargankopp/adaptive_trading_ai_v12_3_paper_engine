from __future__ import annotations

import time as time_mod
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

API_URL = "https://api.india.delta.exchange/v2/history/candles"
RESOLUTION_SECONDS = 300
MAX_CANDLES_PER_REQUEST = 2000
DEFAULT_TIMEOUT = 30


def _chunks(start: int, end: int, span: int = MAX_CANDLES_PER_REQUEST) -> Iterable[tuple[int, int]]:
    step = (span - 1) * RESOLUTION_SECONDS
    cur = int(start)
    while cur <= end:
        nxt = min(cur + step, int(end))
        yield cur, nxt
        cur = nxt + RESOLUTION_SECONDS


def _request(session: requests.Session, symbol: str, start: int, end: int, retries: int = 6) -> list[dict]:
    params = {"resolution": "5m", "symbol": symbol, "start": int(start), "end": int(end)}
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(API_URL, params=params, headers={"Accept": "application/json"}, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 429:
                wait = min(30.0, 2.0 ** attempt)
                time_mod.sleep(wait)
                continue
            resp.raise_for_status()
            payload = resp.json()
            if not payload.get("success", False):
                raise RuntimeError(f"Delta API returned success=false: {payload}")
            result = payload.get("result", [])
            if not isinstance(result, list):
                raise RuntimeError("Unexpected Delta candle response shape.")
            return result
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_err = exc
            if attempt == retries - 1:
                break
            time_mod.sleep(min(20.0, 1.5 ** attempt))
    raise RuntimeError(f"Failed Delta OHLC request {symbol} {start}-{end}") from last_err


def download_ohlc(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, pause_seconds: float = 0.15) -> pd.DataFrame:
    start = int(pd.Timestamp(start_ts).timestamp())
    end = int(pd.Timestamp(end_ts).timestamp())
    if end <= start:
        raise ValueError("end_ts must be after start_ts")

    rows: list[dict] = []
    with requests.Session() as session:
        total = 0
        for total, (a, b) in enumerate(_chunks(start, end), start=1):
            batch = _request(session, symbol, a, b)
            for x in batch:
                rows.append(x)
            if pause_seconds > 0:
                time_mod.sleep(pause_seconds)
            print(f"Downloaded {symbol} batch {total}: {len(batch)} candles", flush=True)

    if not rows:
        raise RuntimeError(f"Delta returned no candles for {symbol}.")

    out = pd.DataFrame(rows)
    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Delta response missing columns: {missing}")
    out = out[required].copy()
    # Delta returns epoch seconds for historical candles.
    out["timestamp"] = pd.to_datetime(out["time"], unit="s", utc=True, errors="coerce")
    out = out.drop(columns=["time"])
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    out = out[["timestamp", "open", "high", "low", "close", "volume"]]
    validate_ohlc(out)
    return out


def validate_ohlc(df: pd.DataFrame) -> dict:
    req = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC data missing required columns: {missing}")
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if ts.isna().any() or not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError("OHLC timestamps must be valid, strictly increasing, and unique.")
    bad_prices = (
        (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (~df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1))
    )
    if bad_prices.any():
        raise ValueError(f"Found {int(bad_prices.sum())} invalid OHLC rows.")
    if (df["volume"] < 0).any():
        raise ValueError("OHLC volume cannot be negative.")
    delta = ts.diff().dropna()
    return {
        "rows": int(len(df)),
        "start": str(ts.min()),
        "end": str(ts.max()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "non_5m_deltas": int((delta != pd.Timedelta(minutes=5)).sum()),
        "max_gap_minutes": float(delta.max().total_seconds() / 60.0) if len(delta) else 0.0,
    }


def save_ohlc(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="infer")


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    validate_ohlc(df)
    return df
