from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd

from app.v12_market_feed import CandleStore
from app.v12_multi_market_feed import DEFAULT_SYMBOLS, DeltaMultiCandleStreamer
from app.v12_live_signal import LiveSignalEngine, fetch_history, build_raw_from_history
from app.v12_paper_engine import PaperPositionEngine


def main() -> int:
    ap = argparse.ArgumentParser(description="V12.3 live paper-position engine — NO ORDERS")
    ap.add_argument("--seconds", type=int, default=1800)
    ap.add_argument("--warmup-bars", type=int, default=400)
    ap.add_argument("--output-dir", default="logs/v12/paper")
    ap.add_argument("--slippage-per-side-pct", type=float, default=0.02)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    engine = LiveSignalEngine(Path("models/v93_fold3"))
    paper = PaperPositionEngine(out, slippage_per_side=args.slippage_per_side_pct / 100.0)

    print("=== V12.3 LIVE PAPER-POSITION ENGINE ===")
    print("Public Delta REST + websocket | NO API credentials | NO orders")
    print(f"Warmup: {args.warmup_bars} completed 5m bars per asset")
    print(f"Paper execution: next-candle-open fill | slippage={args.slippage_per_side_pct:.3f}%/side | cost={paper.cost_rt*100:.3f}% RT")

    history = {}
    for sym in DEFAULT_SYMBOLS:
        print(f"Fetching warmup: {sym}...", flush=True)
        history[sym] = fetch_history(sym, args.warmup_bars)
        print(f"  {sym}: {len(history[sym])} bars | {history[sym].timestamp.iloc[0]} -> {history[sym].timestamp.iloc[-1]}", flush=True)

    raw_df, _ = build_raw_from_history(history)
    print(f"Synchronized warmup rows: {len(raw_df)}", flush=True)
    print(f"Warmup end: {raw_df.timestamp.iloc[-1]}", flush=True)

    stores_dir = out / "bars"; stores_dir.mkdir(parents=True, exist_ok=True)
    stores = {sym: CandleStore(stores_dir / f"{sym}_5m.csv") for sym in DEFAULT_SYMBOLS}
    signal_path = out / "paper_signals.csv"
    signal_fields = ["timestamp", "btc_close", "long_score", "short_score", "long_percentile", "short_percentile",
                     "long_candidate", "short_candidate", "signal"]
    with signal_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=signal_fields).writeheader()

    latest = {sym: history[sym].iloc[-1].to_dict() for sym in DEFAULT_SYMBOLS}
    finalized = set()
    last_heartbeat = time.time()

    def current_btc() -> dict | None:
        st = stores["BTCUSD"]
        if st.current_start_us is None:
            return None
        row = st.rows.get(st.current_start_us)
        return dict(row) if row else None

    def on_finalized(row: dict) -> None:
        nonlocal last_heartbeat
        sym = row["finalized_symbol"]
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        latest[sym] = {
            "timestamp": ts,
            "open": float(row["open"]), "high": float(row["high"]),
            "low": float(row["low"]), "close": float(row["close"]), "volume": float(row["volume"])
        }

        if sym == "BTCUSD":
            paper.process_completed_btc_bar(latest[sym])

        x = history[sym].copy()
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
        x = x[x["timestamp"] != ts]
        x = pd.concat([x, pd.DataFrame([latest[sym]])], ignore_index=True)
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
        history[sym] = x.sort_values("timestamp").tail(args.warmup_bars).reset_index(drop=True)

        if not all(s in latest and pd.Timestamp(latest[s]["timestamp"]) == ts for s in DEFAULT_SYMBOLS):
            return
        if ts in finalized:
            return
        finalized.add(ts)

        merged = None
        for s in DEFAULT_SYMBOLS:
            q = history[s][["timestamp", "close", "volume"]].copy()
            q = q.rename(columns={"close": f"{s.lower()}_close", "volume": f"{s.lower()}_volume"})
            merged = q if merged is None else merged.merge(q, on="timestamp", how="inner")
        raw = merged.sort_values("timestamp").tail(400).reset_index(drop=True)
        btc = history["BTCUSD"][["timestamp", "open", "high", "low", "close"]].copy()
        btc = raw[["timestamp"]].merge(btc, on="timestamp", how="inner").sort_values("timestamp").tail(400).reset_index(drop=True)

        try:
            sig = engine.predict(raw, btc)
        except Exception as exc:
            print(f"SIGNAL_ERROR {ts}: {type(exc).__name__}: {exc}", flush=True)
            return

        with signal_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=signal_fields).writerow(sig)

        filled = paper.maybe_fill_from_signal(sig, current_btc())
        suffix = " | PAPER_ENTRY" if filled else ""
        print(
            f"SIGNAL {sig['timestamp']} | BTC={sig['btc_close']} | "
            f"L pctl={sig['long_percentile']:.3f} S pctl={sig['short_percentile']:.3f} | {sig['signal']}{suffix}",
            flush=True,
        )

        if time.time() - last_heartbeat >= 60:
            s = paper.summary()
            print(f"PAPER STATUS | equity={s.get('current_equity', paper.equity):.2f} | trades={s.get('realized_trades',0)} | open={s.get('open_position') is not None}", flush=True)
            last_heartbeat = time.time()

    streamer = DeltaMultiCandleStreamer(stores=stores, symbols=DEFAULT_SYMBOLS, on_finalized=on_finalized)
    try:
        streamer.run(seconds=args.seconds)
    finally:
        summary = paper.summary()
        (out / "paper_config.json").write_text(json.dumps({
            "version": "V12.3",
            "live_orders": False,
            "latency_bars": 1,
            "slippage_per_side_pct": args.slippage_per_side_pct,
            "round_trip_cost_pct": paper.cost_rt * 100.0,
            "stop_pct": paper.stop_pct * 100.0,
            "target_pct": paper.target_pct * 100.0,
            "max_hold_bars": paper.max_hold_bars,
            "risk_per_trade_pct": paper.risk_per_trade * 100.0,
            "warmup_bars": args.warmup_bars,
        }, indent=2), encoding="utf-8")
        print("=== V12.3 PAPER ENGINE STOPPED ===")
        print(json.dumps(summary, indent=2, default=str))
        print(f"Signals: {signal_path}")
        print(f"Trades: {paper.trade_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
