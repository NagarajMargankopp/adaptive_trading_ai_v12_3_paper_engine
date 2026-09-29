from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from app.v923_train import build_price_action_features
from app.v827_train import load_dataset, make_walk_forward_folds, PURGE

ROOT = Path(".")
DATA = ROOT / "data" / "processed"
MODEL = ROOT / "models" / "v93_fold3"
OUT = ROOT / "logs" / "v12" / "threshold_study"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_EQUITY = 10000.0
RISK_PER_TRADE = 0.005
STOP_PCT = 0.005
TARGET_PCT = 0.010
COST_RT = 0.0014
SLIPPAGE = 0.0002
MAX_HOLD_BARS = 12
DAILY_LOSS_LIMIT = 0.02

THRESHOLDS = [0.90, 0.85, 0.80, 0.75, 0.70]


def score_percentile(ref: np.ndarray, value: float) -> float:
    if ref.size == 0 or not np.isfinite(value):
        return 0.0
    return float(np.searchsorted(ref, value, side="right") / ref.size)


def simulate(
    df: pd.DataFrame,
    ohlc: pd.DataFrame,
    long_scores: np.ndarray,
    short_scores: np.ndarray,
    long_ref: np.ndarray,
    short_ref: np.ndarray,
    start: int,
    end: int,
    threshold: float,
):
    ts = pd.to_datetime(df["timestamp"], utc=True)
    day = ts.dt.strftime("%Y-%m-%d").to_numpy()

    o = ohlc.set_index("timestamp").reindex(ts)
    opens = o["open"].to_numpy(float)
    highs = o["high"].to_numpy(float)
    lows = o["low"].to_numpy(float)
    closes = o["close"].to_numpy(float)

    # Need enough bars for next-open entry + 12 completed exit bars.
    last_signal = min(end - (MAX_HOLD_BARS + 1), len(df) - (MAX_HOLD_BARS + 1))
    if last_signal <= start:
        return pd.DataFrame(), {
            "candidate_signals": 0,
            "long_candidates": 0,
            "short_candidates": 0,
        }

    equity = INITIAL_EQUITY
    peak_equity = INITIAL_EQUITY
    current_day = None
    day_start_equity = INITIAL_EQUITY

    taken_by_day: set[tuple[str, str]] = set()
    next_free = start
    trades = []

    candidate_signals = 0
    long_candidates = 0
    short_candidates = 0

    # Match V11 candidate selection:
    # for each UTC day and each side, select the highest-scoring
    # qualifying timestamp above the percentile threshold.
    day_candidates = {}

    for i in range(start, last_signal):
        if not np.isfinite(long_scores[i]) and not np.isfinite(short_scores[i]):
            continue

        lp = score_percentile(long_ref, long_scores[i])
        sp = score_percentile(short_ref, short_scores[i])

        if np.isfinite(long_scores[i]) and lp >= threshold:
            key = (day[i], "LONG")
            prev = day_candidates.get(key)
            if prev is None or long_scores[i] > long_scores[prev] or (long_scores[i] == long_scores[prev] and i > prev):
                day_candidates[key] = i

        if np.isfinite(short_scores[i]) and sp >= threshold:
            key = (day[i], "SHORT")
            prev = day_candidates.get(key)
            if prev is None or short_scores[i] > short_scores[prev] or (short_scores[i] == short_scores[prev] and i > prev):
                day_candidates[key] = i

    selected = []

    for (d, side_name), i in day_candidates.items():
        selected.append(
            (
                i,
                1 if side_name == "LONG" else -1,
                long_scores[i] if side_name == "LONG" else short_scores[i],
            )
        )

    selected.sort(key=lambda x: x[0])

    candidate_signals = len(selected)
    long_candidates = sum(1 for _, side, _ in selected if side == 1)
    short_candidates = sum(1 for _, side, _ in selected if side == -1)

    for i, side, score in selected:
        # Only one paper position at a time.
        if i < next_free:
            continue

        entry_index = i + 1
        if entry_index >= len(df):
            continue

        fill_day = day[entry_index]
        if fill_day != current_day:
            current_day = fill_day
            day_start_equity = equity

        daily_loss = equity / day_start_equity - 1.0
        if daily_loss <= -DAILY_LOSS_LIMIT:
            continue

        open_px = opens[entry_index]
        if not np.isfinite(open_px) or open_px <= 0:
            continue

        entry = (
            open_px * (1.0 + SLIPPAGE)
            if side == 1
            else open_px * (1.0 - SLIPPAGE)
        )

        sl = (
            entry * (1.0 - STOP_PCT)
            if side == 1
            else entry * (1.0 + STOP_PCT)
        )
        tp = (
            entry * (1.0 + TARGET_PCT)
            if side == 1
            else entry * (1.0 - TARGET_PCT)
        )

        exit_index = None
        exit_price = None
        reason = None
        ambiguous = False

        # Keep V12.3 behavior: do not evaluate the entry candle itself.
        first_exit_bar = entry_index + 1
        last_exit_bar = entry_index + MAX_HOLD_BARS

        for j in range(first_exit_bar, last_exit_bar + 1):
            if j >= len(df):
                break

            op = opens[j]
            hi = highs[j]
            lo = lows[j]
            cl = closes[j]

            if not np.isfinite([op, hi, lo, cl]).all():
                continue

            if side == 1:
                open_sl = op <= sl
                open_tp = op >= tp
                sl_hit = lo <= sl
                tp_hit = hi >= tp
            else:
                open_sl = op >= sl
                open_tp = op <= tp
                sl_hit = hi >= sl
                tp_hit = lo <= tp

            if open_sl and open_tp:
                exit_price = (
                    op * (1.0 - SLIPPAGE)
                    if side == 1
                    else op * (1.0 + SLIPPAGE)
                )
                exit_index = j
                reason = "SL_FIRST_GAP_AMBIGUITY"
                ambiguous = True
                break

            if open_sl:
                exit_price = (
                    op * (1.0 - SLIPPAGE)
                    if side == 1
                    else op * (1.0 + SLIPPAGE)
                )
                exit_index = j
                reason = "SL_GAP"
                break

            if open_tp:
                exit_price = (
                    op * (1.0 - SLIPPAGE)
                    if side == 1
                    else op * (1.0 + SLIPPAGE)
                )
                exit_index = j
                reason = "TP_GAP"
                break

            if sl_hit and tp_hit:
                exit_price = (
                    sl * (1.0 - SLIPPAGE)
                    if side == 1
                    else sl * (1.0 + SLIPPAGE)
                )
                exit_index = j
                reason = "SL_FIRST"
                ambiguous = True
                break

            if sl_hit:
                exit_price = (
                    sl * (1.0 - SLIPPAGE)
                    if side == 1
                    else sl * (1.0 + SLIPPAGE)
                )
                exit_index = j
                reason = "SL"
                break

            if tp_hit:
                exit_price = (
                    tp * (1.0 - SLIPPAGE)
                    if side == 1
                    else tp * (1.0 + SLIPPAGE)
                )
                exit_index = j
                reason = "TP"
                break

        if exit_index is None:
            exit_index = entry_index + MAX_HOLD_BARS
            if exit_index >= len(df):
                continue

            cl = closes[exit_index]
            exit_price = (
                cl * (1.0 - SLIPPAGE)
                if side == 1
                else cl * (1.0 + SLIPPAGE)
            )
            reason = "TIME"

        if side == 1:
            gross = exit_price / entry - 1.0
        else:
            gross = (entry - exit_price) / entry

        net_return = gross - COST_RT
        net_r = net_return / STOP_PCT

        equity_before = equity
        equity_after = equity * (1.0 + RISK_PER_TRADE * net_r)
        peak_equity = max(peak_equity, equity_after)
        dd = equity_after / peak_equity - 1.0

        trades.append({
            "signal_index": i,
            "signal_time": str(ts.iloc[i]),
            "entry_index": entry_index,
            "entry_time": str(ts.iloc[entry_index]),
            "exit_index": exit_index,
            "exit_time": str(ts.iloc[exit_index]),
            "side": "LONG" if side == 1 else "SHORT",
            "signal_reference_price": float(closes[i]),
            "intended_open": float(open_px),
            "entry_price": float(entry),
            "exit_price": float(exit_price),
            "long_percentile": float(
                score_percentile(long_ref, long_scores[i])
            ),
            "short_percentile": float(
                score_percentile(short_ref, short_scores[i])
            ),
            "gross_return": float(gross),
            "net_return": float(net_return),
            "net_r": float(net_r),
            "reason": reason,
            "same_bar_ambiguity": ambiguous,
            "hold_bars": int(exit_index - entry_index),
            "equity_before": float(equity_before),
            "equity_after": float(equity_after),
            "drawdown_from_peak": float(dd),
        })

        equity = equity_after
        next_free = exit_index + 1
    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        stats = {
            "candidate_signals": candidate_signals,
            "long_candidates": long_candidates,
            "short_candidates": short_candidates,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "profit_factor": None,
            "equity_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }
    else:
        r = trades_df["net_return"].to_numpy(float)
        gains = float(r[r > 0].sum())
        losses = float(-r[r < 0].sum())
        final_equity = float(trades_df.iloc[-1]["equity_after"])

        stats = {
            "candidate_signals": candidate_signals,
            "long_candidates": long_candidates,
            "short_candidates": short_candidates,
            "trades": int(len(trades_df)),
            "wins": int((r > 0).sum()),
            "losses": int((r <= 0).sum()),
            "win_rate_pct": float((r > 0).mean() * 100.0),
            "profit_factor": float(gains / losses) if losses > 0 else None,
            "equity_return_pct": float((final_equity / INITIAL_EQUITY - 1.0) * 100.0),
            "max_drawdown_pct": float(
                trades_df["equity_after"]
                .div(trades_df["equity_after"].cummax())
                .sub(1.0)
                .min()
                * 100.0
            ),
        }

    return trades_df, stats


def main():
    print("=== V12.3 HISTORICAL THRESHOLD STUDY ===")
    print("Frozen V9.3 Fold-3 model | No retraining")
    print("Execution: next-candle-open | 0.02% slip/side | 0.14% RT cost")
    print("Risk: 0.5% | SL=0.5% | TP=1.0% | Max hold=12 bars")
    print()

    df = load_dataset(DATA / "v821_common_multiasset.csv.gz")

    ohlc = pd.read_csv(DATA / "btcusd_5m_ohlc.csv.gz", compression="infer")
    ohlc["timestamp"] = pd.to_datetime(ohlc["timestamp"], utc=True)
    ohlc = ohlc.sort_values("timestamp").reset_index(drop=True)

    aligned = (
        ohlc.set_index("timestamp")
        .reindex(df["timestamp"])[["open", "high", "low", "close"]]
    )

    pa_df = df.copy()
    pa_df["btcusd_open"] = aligned["open"].to_numpy(float)
    pa_df["btcusd_high"] = aligned["high"].to_numpy(float)
    pa_df["btcusd_low"] = aligned["low"].to_numpy(float)
    pa_df["btcusd_close_ohlc"] = aligned["close"].to_numpy(float)
    pa_df["btcusd_close"] = pa_df["btcusd_close_ohlc"]

    features, names = build_price_action_features(pa_df, ohlc)

    saved_names = json.loads((MODEL / "feature_names.json").read_text())
    if names != saved_names:
        raise RuntimeError(
            f"Feature mismatch: runtime={len(names)} saved={len(saved_names)}"
        )

    imputer = joblib.load(MODEL / "v93_fold3_imputer.joblib")
    long_model = joblib.load(MODEL / "v93_fold3_L_ranker.joblib")
    short_model = joblib.load(MODEL / "v93_fold3_S_ranker.joblib")
    long_ref = np.load(MODEL / "v93_fold3_L_score_reference.npy")
    short_ref = np.load(MODEL / "v93_fold3_S_score_reference.npy")

    X = features[names].replace([np.inf, -np.inf], np.nan)
    X = imputer.transform(X).astype(np.float32)

    print(f"Rows: {len(df)}")
    print(f"Features: {len(names)}")
    print("Scoring dataset...", flush=True)

    long_scores = long_model.predict(X).astype(float)
    short_scores = short_model.predict(X).astype(float)

    folds = make_walk_forward_folds(len(df))
    fold3 = folds[2]

    splits = {
        "validation": fold3["validation"],
        "test": fold3["test"],
        "blind_future": (
            int(len(df) * 0.95) + PURGE,
            len(df),
        ),
    }

    all_rows = []
    all_trades = []

    for threshold in THRESHOLDS:
        print(f"\n=== THRESHOLD {threshold:.2f} ===")

        for split_name, (lo, hi) in splits.items():
            trades, stats = simulate(
                df,
                ohlc,
                long_scores,
                short_scores,
                long_ref,
                short_ref,
                lo,
                hi,
                threshold,
            )

            row = {
                "threshold": threshold,
                "split": split_name,
                **stats,
            }
            all_rows.append(row)

            if not trades.empty:
                trades = trades.copy()
                trades["threshold"] = threshold
                trades["split"] = split_name
                all_trades.append(trades)

            print(
                f"{split_name:12s} | "
                f"candidates={stats['candidate_signals']:3d} | "
                f"trades={stats['trades']:3d} | "
                f"win={stats['win_rate_pct']:6.2f}% | "
                f"PF={stats['profit_factor']} | "
                f"return={stats['equity_return_pct']:7.3f}% | "
                f"DD={stats['max_drawdown_pct']:7.3f}%"
            )

    summary = pd.DataFrame(all_rows)
    summary_path = OUT / "threshold_study_summary.csv"
    summary.to_csv(summary_path, index=False)

    if all_trades:
        trade_df = pd.concat(all_trades, ignore_index=True)
    else:
        trade_df = pd.DataFrame()

    trade_path = OUT / "threshold_study_trades.csv"
    trade_df.to_csv(trade_path, index=False)

    print("\n=== SAVED ===")
    print(summary_path)
    print(trade_path)


if __name__ == "__main__":
    main()
