from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd

from app.v12_market_feed import CandleStore
from app.v12_multi_market_feed import DEFAULT_SYMBOLS, DeltaMultiCandleStreamer
from app.v12_live_signal import LiveSignalEngine, fetch_history, build_raw_from_history, DEFAULT_SYMBOLS


def main() -> int:
    ap = argparse.ArgumentParser(description="V12.2 live V9.3 signal observer — NO ORDERS")
    ap.add_argument("--seconds", type=int, default=900)
    ap.add_argument("--warmup-bars", type=int, default=400)
    ap.add_argument("--output-dir", default="logs/v12/signal")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    model_dir = Path("models/v93_fold3")
    engine = LiveSignalEngine(model_dir)

    print("=== V12.2 LIVE V9.3 SIGNAL OBSERVER — PAPER ONLY ===")
    print("Public Delta REST + websocket | NO API credentials | NO orders")
    print(f"Warmup: {args.warmup_bars} completed 5m bars per asset")

    history = {}
    for sym in DEFAULT_SYMBOLS:
        print(f"Fetching warmup: {sym}...", flush=True)
        history[sym] = fetch_history(sym, args.warmup_bars)
        print(f"  {sym}: {len(history[sym])} bars | {history[sym].timestamp.iloc[0]} -> {history[sym].timestamp.iloc[-1]}", flush=True)

    raw_df, btc_ohlc = build_raw_from_history(history)
    print(f"Synchronized warmup rows: {len(raw_df)}", flush=True)
    print(f"Warmup end: {raw_df.timestamp.iloc[-1]}", flush=True)

    # Prime stores with warmup only to keep the live window synchronized.
    stores_dir = out / "bars"; stores_dir.mkdir(parents=True, exist_ok=True)
    stores = {sym: CandleStore(stores_dir / f"{sym}_5m.csv") for sym in DEFAULT_SYMBOLS}

    signal_path = out / "live_signals.csv"
    fieldnames = ["timestamp", "btc_close", "long_score", "short_score", "long_percentile", "short_percentile",
                  "long_candidate", "short_candidate", "signal"]
    with signal_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    latest = {sym: history[sym].iloc[-1].to_dict() for sym in DEFAULT_SYMBOLS}
    finalized = set()

    def on_finalized(row: dict) -> None:
        sym = row["finalized_symbol"]
        # CandleStore serializes finalized timestamps as ISO strings; warmup history
        # uses pandas UTC Timestamps. Normalize immediately to avoid mixed-type
        # sorting/comparison errors when a live candle is appended.
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        latest[sym] = {
            "timestamp": ts, "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]),
            "close": float(row["close"]), "volume": float(row["volume"])
        }
        # Keep the rolling history synchronized with newly finalized live candles.
        x = history[sym].copy()
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
        x = x[x["timestamp"] != ts]
        x = pd.concat([x, pd.DataFrame([latest[sym]])], ignore_index=True)
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
        history[sym] = x.sort_values("timestamp").tail(args.warmup_bars).reset_index(drop=True)

        # Only infer once every asset has the same finalized timestamp.
        if not all(s in latest and pd.Timestamp(latest[s]["timestamp"]) == ts for s in DEFAULT_SYMBOLS):
            return
        if ts in finalized:
            return
        finalized.add(ts)
        raw = None
        for s in DEFAULT_SYMBOLS:
            x = history[s][["timestamp", "close", "volume"]].copy()
            x = x.rename(columns={"close": f"{s.lower()}_close", "volume": f"{s.lower()}_volume"})
            raw = x if raw is None else raw.merge(x, on="timestamp", how="inner")
        raw = raw.sort_values("timestamp").tail(400).reset_index(drop=True)
        # Update BTC OHLC from saved history/current row. Older OHLC remains from REST history.
        bh = history["BTCUSD"][["timestamp", "open", "high", "low", "close"]].copy()
        btc = raw[["timestamp"]].merge(bh, on="timestamp", how="inner").sort_values("timestamp").tail(400).reset_index(drop=True)
        try:
            sig = engine.predict(raw, btc)
        except Exception as exc:
            print(f"SIGNAL_ERROR {ts}: {type(exc).__name__}: {exc}", flush=True)
            return
        with signal_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(sig)
        print(
            f"SIGNAL {sig['timestamp']} | BTC={sig['btc_close']} | "
            f"L pctl={sig['long_percentile']:.3f} S pctl={sig['short_percentile']:.3f} | {sig['signal']}", flush=True
        )

    streamer = DeltaMultiCandleStreamer(stores=stores, symbols=DEFAULT_SYMBOLS, on_finalized=on_finalized)
    streamer.run(seconds=args.seconds)
    print("=== V12.2 SIGNAL OBSERVER STOPPED ===")
    print(f"Signals: {signal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
