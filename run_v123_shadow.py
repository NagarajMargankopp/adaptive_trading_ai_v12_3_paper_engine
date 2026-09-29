from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.v12_market_feed import CandleStore
from app.v12_multi_market_feed import DEFAULT_SYMBOLS, DeltaMultiCandleStreamer
from app.v12_live_signal import fetch_history, build_raw_from_history
from app.v12_paper_engine import PaperPositionEngine
from app.telegram_notifier import TelegramNotifier
from app.v12_cloud_state import (
    load_engine_state,
    load_shadow_taken,
    save_engine_state,
    save_shadow_taken,
)
from app.v923_train import build_price_action_features


THRESHOLDS = [0.90, 0.85, 0.80, 0.75, 0.70]
MODEL_DIR = Path("models/v93_fold3")


def percentile(ref: np.ndarray, value: float) -> float:
    if ref.size == 0 or not np.isfinite(value):
        return 0.0
    return float(np.searchsorted(ref, value, side="right") / ref.size)


class ShadowSignalEngine:
    def __init__(self, model_dir: Path):
        self.imputer = joblib.load(model_dir / "v93_fold3_imputer.joblib")
        self.long_model = joblib.load(model_dir / "v93_fold3_L_ranker.joblib")
        self.short_model = joblib.load(model_dir / "v93_fold3_S_ranker.joblib")
        self.long_ref = np.load(model_dir / "v93_fold3_L_score_reference.npy")
        self.short_ref = np.load(model_dir / "v93_fold3_S_score_reference.npy")
        self.feature_names = json.loads(
            (model_dir / "feature_names.json").read_text()
        )
        self.taken_by_day = {
            t: set() for t in THRESHOLDS
        }

    def predict(self, raw_df: pd.DataFrame, btc_ohlc: pd.DataFrame) -> dict:
        features, names = build_price_action_features(raw_df, btc_ohlc)

        if names != self.feature_names:
            raise RuntimeError(
                f"Feature mismatch: runtime={len(names)} saved={len(self.feature_names)}"
            )

        X = features[self.feature_names].replace(
            [np.inf, -np.inf], np.nan
        )
        X = self.imputer.transform(X).astype(np.float32)

        long_score = float(self.long_model.predict(X[-1:])[0])
        short_score = float(self.short_model.predict(X[-1:])[0])

        lp = percentile(self.long_ref, long_score)
        sp = percentile(self.short_ref, short_score)

        ts = str(raw_df.iloc[-1]["timestamp"])
        day = pd.Timestamp(
            raw_df.iloc[-1]["timestamp"]
        ).strftime("%Y-%m-%d")

        return {
            "timestamp": ts,
            "btc_close": float(raw_df.iloc[-1]["btcusd_close"]),
            "long_score": long_score,
            "short_score": short_score,
            "long_percentile": lp,
            "short_percentile": sp,
            "day": day,
        }


def make_signal(base: dict, threshold: float) -> dict:
    day = base["day"]

    long_ok = (
        base["long_percentile"] >= threshold
        and (day, "LONG") not in shadow_taken[threshold]
    )

    short_ok = (
        base["short_percentile"] >= threshold
        and (day, "SHORT") not in shadow_taken[threshold]
    )

    if long_ok:
        shadow_taken[threshold].add((day, "LONG"))

    if short_ok:
        shadow_taken[threshold].add((day, "SHORT"))

    if long_ok and short_ok:
        signal = "BOTH"
    elif long_ok:
        signal = "LONG"
    elif short_ok:
        signal = "SHORT"
    else:
        signal = "NO_SIGNAL"

    return {
        "timestamp": base["timestamp"],
        "btc_close": base["btc_close"],
        "long_score": base["long_score"],
        "short_score": base["short_score"],
        "long_percentile": base["long_percentile"],
        "short_percentile": base["short_percentile"],
        "long_candidate": bool(long_ok),
        "short_candidate": bool(short_ok),
        "signal": signal,
    }


shadow_taken = {t: set() for t in THRESHOLDS}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="V12.3 live threshold shadow engine — NO ORDERS"
    )
    ap.add_argument("--seconds", type=int, default=21600)
    ap.add_argument("--warmup-bars", type=int, default=400)
    ap.add_argument(
        "--output-dir",
        default="logs/v12/shadow",
    )
    ap.add_argument(
        "--state-dir",
        default=None,
        help="Directory for persistent cloud paper state",
    )
    ap.add_argument(
        "--slippage-per-side-pct",
        type=float,
        default=0.02,
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is not None:
        state_dir.mkdir(parents=True, exist_ok=True)
        shadow_taken.clear()
        shadow_taken.update(
            load_shadow_taken(state_dir / "shadow_taken.json", THRESHOLDS)
        )
        print("Persistent cloud state: enabled", flush=True)
    else:
        print("Persistent cloud state: disabled", flush=True)
    telegram = TelegramNotifier()
    print(f"Telegram notifications: {telegram.enabled}", flush=True)

    print("=== V12.3 LIVE THRESHOLD SHADOW ENGINE ===")
    print("BTC/ETH/SOL/BNB/XRP live data")
    print("Thresholds:", THRESHOLDS)
    print("NO API credentials | NO live orders")
    print(
        f"Execution: next-candle-open | "
        f"slippage={args.slippage_per_side_pct:.3f}%/side | "
        f"cost=0.140% RT"
    )

    history = {}

    for sym in DEFAULT_SYMBOLS:
        print(f"Fetching warmup: {sym}...", flush=True)
        history[sym] = fetch_history(sym, args.warmup_bars)
        print(
            f"  {sym}: {len(history[sym])} bars | "
            f"{history[sym].timestamp.iloc[0]} -> "
            f"{history[sym].timestamp.iloc[-1]}",
            flush=True,
        )

    raw_df, _ = build_raw_from_history(history)

    print(
        f"Synchronized warmup rows: {len(raw_df)}",
        flush=True,
    )
    print(
        f"Warmup end: {raw_df.timestamp.iloc[-1]}",
        flush=True,
    )

    stores_dir = out / "bars"
    stores_dir.mkdir(parents=True, exist_ok=True)

    stores = {
        sym: CandleStore(stores_dir / f"{sym}_5m.csv")
        for sym in DEFAULT_SYMBOLS
    }

    engines = {}
    signal_files = {}
    signal_writers = {}

    for threshold in THRESHOLDS:
        tag = f"p{int(threshold * 100):02d}"
        d = out / tag
        d.mkdir(parents=True, exist_ok=True)

        engines[threshold] = PaperPositionEngine(
            d,
            slippage_per_side=args.slippage_per_side_pct / 100.0,
        )

        if state_dir is not None:
            state_path = state_dir / f"paper_{int(threshold * 100):02d}.json"
            if state_path.exists():
                loaded = load_engine_state(engines[threshold], state_path)
                print(
                    f"Loaded paper state P{int(threshold * 100):02d}: "
                    f"equity=${engines[threshold].equity:.2f} "
                    f"trades={len(engines[threshold].trades)} "
                    f"open_position={engines[threshold].position is not None}",
                    flush=True,
                )

        signal_path = d / "paper_signals.csv"
        f = signal_path.open("w", newline="")
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "btc_close",
                "long_score",
                "short_score",
                "long_percentile",
                "short_percentile",
                "long_candidate",
                "short_candidate",
                "signal",
            ],
        )
        writer.writeheader()

        signal_files[threshold] = f
        signal_writers[threshold] = writer

    latest = {
        sym: history[sym].iloc[-1].to_dict()
        for sym in DEFAULT_SYMBOLS
    }

    finalized = set()

    def current_btc():
        st = stores["BTCUSD"]
        if st.current_start_us is None:
            return None
        row = st.rows.get(st.current_start_us)
        return dict(row) if row else None

    model = ShadowSignalEngine(MODEL_DIR)

    def save_state() -> None:
        if state_dir is None:
            return

        for threshold in THRESHOLDS:
            save_engine_state(
                engines[threshold],
                state_dir / f"paper_{int(threshold * 100):02d}.json",
            )

        save_shadow_taken(
            shadow_taken,
            state_dir / "shadow_taken.json",
        )

    def on_finalized(row: dict):
        sym = row["finalized_symbol"]

        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        latest[sym] = {
            "timestamp": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }

        if sym == "BTCUSD":
            for threshold in THRESHOLDS:
                trade = engines[threshold].process_completed_btc_bar(
                    latest[sym]
                )

                if trade is not None:
                    telegram.send(
                        f"✅ PAPER EXIT | P{int(threshold * 100):02d}\n"
                        f"Side: {trade['side']}\n"
                        f"Reason: {trade['reason']}\n"
                        f"Entry: {trade['entry_price']:.2f}\n"
                        f"Exit: {trade['exit_price']:.2f}\n"
                        f"Net Return: {trade['net_return'] * 100:.3f}%\n"
                        f"Net R: {trade['net_r']:.3f}\n"
                        f"Hold: {trade['hold_bars']} bars\n"
                        f"Equity: ${trade['equity_after']:.2f}\n"
                        f"Time: {trade['exit_timestamp']}"
                    )

        x = history[sym].copy()
        x["timestamp"] = pd.to_datetime(
            x["timestamp"], utc=True
        )
        x = x[x["timestamp"] != ts]
        x = pd.concat(
            [x, pd.DataFrame([latest[sym]])],
            ignore_index=True,
        )
        x["timestamp"] = pd.to_datetime(
            x["timestamp"], utc=True
        )
        history[sym] = (
            x.sort_values("timestamp")
            .tail(args.warmup_bars)
            .reset_index(drop=True)
        )

        if not all(
            s in latest
            and pd.Timestamp(latest[s]["timestamp"]) == ts
            for s in DEFAULT_SYMBOLS
        ):
            return

        if ts in finalized:
            return

        finalized.add(ts)

        merged = None

        for s in DEFAULT_SYMBOLS:
            q = history[s][
                ["timestamp", "close", "volume"]
            ].copy()

            q = q.rename(
                columns={
                    "close": f"{s.lower()}_close",
                    "volume": f"{s.lower()}_volume",
                }
            )

            merged = (
                q if merged is None
                else merged.merge(
                    q,
                    on="timestamp",
                    how="inner",
                )
            )

        raw = (
            merged.sort_values("timestamp")
            .tail(400)
            .reset_index(drop=True)
        )

        btc = history["BTCUSD"][
            ["timestamp", "open", "high", "low", "close"]
        ].copy()

        btc = (
            raw[["timestamp"]]
            .merge(btc, on="timestamp", how="inner")
            .sort_values("timestamp")
            .tail(400)
            .reset_index(drop=True)
        )

        try:
            base = model.predict(raw, btc)
        except Exception as exc:
            print(
                f"SIGNAL_ERROR {ts}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return

        btc_current = current_btc()

        for threshold in THRESHOLDS:
            sig = make_signal(base, threshold)

            signal_writers[threshold].writerow(sig)
            signal_files[threshold].flush()

            filled = engines[threshold].maybe_fill_from_signal(
                sig,
                btc_current,
            )

            if filled:
                position = engines[threshold].position

                if position is not None:
                    telegram.send(
                        f"📈 PAPER ENTRY | P{int(threshold * 100):02d}\n"
                        f"Side: {position.side}\n"
                        f"BTC: {position.signal_reference_price:.2f}\n"
                        f"Entry: {position.entry_price:.2f}\n"
                        f"SL: {position.sl:.2f}\n"
                        f"TP: {position.tp:.2f}\n"
                        f"Percentile: {position.score_percentile:.3f}\n"
                        f"Signal Time: {position.signal_timestamp}\n"
                        f"Fill Time: {position.fill_timestamp}"
                    )

            tag = f"{threshold:.2f}"

            print(
                f"SHADOW {tag} | {sig['timestamp']} | "
                f"BTC={sig['btc_close']} | "
                f"L={sig['long_percentile']:.3f} "
                f"S={sig['short_percentile']:.3f} | "
                f"{sig['signal']}"
                f"{' | ENTRY' if filled else ''}",
                flush=True,
            )

        save_state()

    streamer = DeltaMultiCandleStreamer(
        stores=stores,
        symbols=DEFAULT_SYMBOLS,
        on_finalized=on_finalized,
    )

    try:
        streamer.run(seconds=args.seconds)
    finally:
        save_state()

        for threshold in THRESHOLDS:
            summary = engines[threshold].summary()

            tag = f"p{int(threshold * 100):02d}"
            d = out / tag

            (d / "shadow_config.json").write_text(
                json.dumps(
                    {
                        "version": "V12.3-shadow",
                        "threshold": threshold,
                        "live_orders": False,
                        "latency_bars": 1,
                        "slippage_per_side_pct":
                            args.slippage_per_side_pct,
                        "round_trip_cost_pct": 0.14,
                        "stop_pct": 0.5,
                        "target_pct": 1.0,
                        "risk_per_trade_pct": 0.5,
                        "max_hold_bars": 12,
                        "warmup_bars": args.warmup_bars,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                f"FINAL {threshold:.2f} | "
                f"equity={summary['current_equity']:.2f} | "
                f"trades={summary['realized_trades']}",
                flush=True,
            )

        for f in signal_files.values():
            f.close()

        print("=== SHADOW ENGINE STOPPED ===", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
