from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRanker

from app.v827_train import (
    COST_RT,
    HORIZON,
    LOCKED_POLICY,
    PURGE,
    RANDOM_SEED,
    make_targets,
    make_walk_forward_folds,
    sample_rank_training_indices,
    execute_policy,
    metrics,
    bootstrap_mean_ci,
    load_dataset,
    build_features,
)
from app.delta_ohlc import load_ohlc, validate_ohlc


def build_price_action_features(df: pd.DataFrame, ohlc: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Preserve V8.27 features exactly and append causal BTC price-action features."""
    base, base_names = build_features(df)
    aligned = ohlc.set_index("timestamp").reindex(df["timestamp"])[["open", "high", "low", "close"]]
    if aligned.isna().any().any():
        raise ValueError("Missing OHLC for price-action feature construction")

    # Align OHLC feature series to the signal dataframe positional index.
    o = pd.Series(aligned["open"].to_numpy(float), index=df.index)
    h = pd.Series(aligned["high"].to_numpy(float), index=df.index)
    l = pd.Series(aligned["low"].to_numpy(float), index=df.index)
    c = pd.Series(aligned["close"].to_numpy(float), index=df.index)
    pc = c.shift(1)
    eps = 1e-12

    additions: dict[str, pd.Series] = {}
    names: list[str] = []
    candle_range = (h - l).clip(lower=0.0)
    body = c - o
    abs_body = body.abs()
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

    pa = {
        "pa_body_pct": body / (o.abs() + eps),
        "pa_abs_body_pct": abs_body / (o.abs() + eps),
        "pa_range_pct": candle_range / (c.abs() + eps),
        "pa_upper_wick_pct": upper_wick.clip(lower=0) / (c.abs() + eps),
        "pa_lower_wick_pct": lower_wick.clip(lower=0) / (c.abs() + eps),
        "pa_body_to_range": abs_body / (candle_range + eps),
        "pa_close_location": (c - l) / (candle_range + eps),
        "pa_wick_imbalance": (lower_wick.clip(lower=0) - upper_wick.clip(lower=0)) / (candle_range + eps),
        "pa_gap_pct": (o - pc) / (pc.abs() + eps),
    }
    for k, s in pa.items():
        additions[k] = s.replace([np.inf, -np.inf], np.nan)
        names.append(k)

    prev_o, prev_c, prev_h, prev_l = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    bullish = c > o
    bearish = c < o
    prev_bullish = prev_c > prev_o
    prev_bearish = prev_c < prev_o
    additions["pa_bullish_engulfing"] = (bullish & prev_bearish & (o <= prev_c) & (c >= prev_o)).astype(float)
    additions["pa_bearish_engulfing"] = (bearish & prev_bullish & (o >= prev_c) & (c <= prev_o)).astype(float)
    additions["pa_inside_bar"] = ((h < prev_h) & (l > prev_l)).astype(float)
    additions["pa_doji"] = ((abs_body / (candle_range + eps)) <= 0.15).astype(float)
    additions["pa_hammer"] = ((lower_wick >= 2.0 * abs_body) & (upper_wick <= abs_body) & (candle_range > 0)).astype(float)
    additions["pa_shooting_star"] = ((upper_wick >= 2.0 * abs_body) & (lower_wick <= abs_body) & (candle_range > 0)).astype(float)
    names += ["pa_bullish_engulfing", "pa_bearish_engulfing", "pa_inside_bar", "pa_doji", "pa_hammer", "pa_shooting_star"]

    for w in (12, 24, 48, 96):
        prior_hh = h.shift(1).rolling(w, min_periods=max(6, w // 2)).max()
        prior_ll = l.shift(1).rolling(w, min_periods=max(6, w // 2)).min()
        span = (prior_hh - prior_ll).abs() + eps
        additions[f"pa_pos_{w}"] = (c - prior_ll) / span
        additions[f"pa_dist_prev_high_{w}"] = (c - prior_hh) / (c.abs() + eps)
        additions[f"pa_dist_prev_low_{w}"] = (c - prior_ll) / (c.abs() + eps)
        additions[f"pa_breakout_up_{w}"] = (c > prior_hh).astype(float)
        additions[f"pa_breakout_down_{w}"] = (c < prior_ll).astype(float)
        additions[f"pa_breakout_strength_up_{w}"] = ((c - prior_hh) / span).clip(-5, 5)
        additions[f"pa_breakout_strength_down_{w}"] = ((prior_ll - c) / span).clip(-5, 5)
        names += [
            f"pa_pos_{w}", f"pa_dist_prev_high_{w}", f"pa_dist_prev_low_{w}",
            f"pa_breakout_up_{w}", f"pa_breakout_down_{w}",
            f"pa_breakout_strength_up_{w}", f"pa_breakout_strength_down_{w}"
        ]

    for w in (12, 24, 48):
        abs_moves = c.diff().abs().rolling(w, min_periods=max(6, w // 2)).sum()
        additions[f"pa_efficiency_{w}"] = (c - c.shift(w)).abs() / (abs_moves + eps)
        additions[f"pa_up_fraction_{w}"] = (c.diff() > 0).astype(float).rolling(w, min_periods=max(6, w // 2)).mean()
        additions[f"pa_higher_high_fraction_{w}"] = (h > h.shift(1)).astype(float).rolling(w, min_periods=max(6, w // 2)).mean()
        additions[f"pa_higher_low_fraction_{w}"] = (l > l.shift(1)).astype(float).rolling(w, min_periods=max(6, w // 2)).mean()
        additions[f"pa_lower_high_fraction_{w}"] = (h < h.shift(1)).astype(float).rolling(w, min_periods=max(6, w // 2)).mean()
        additions[f"pa_lower_low_fraction_{w}"] = (l < l.shift(1)).astype(float).rolling(w, min_periods=max(6, w // 2)).mean()
        names += [
            f"pa_efficiency_{w}", f"pa_up_fraction_{w}", f"pa_higher_high_fraction_{w}",
            f"pa_higher_low_fraction_{w}", f"pa_lower_high_fraction_{w}", f"pa_lower_low_fraction_{w}"
        ]

    for w in (24, 48, 96):
        rp = pa["pa_range_pct"]
        mu = rp.rolling(w, min_periods=max(8, w // 2)).mean()
        sd = rp.rolling(w, min_periods=max(8, w // 2)).std()
        additions[f"pa_range_z_{w}"] = (rp - mu) / sd.replace(0, np.nan)
        names.append(f"pa_range_z_{w}")

    pa_df = pd.DataFrame(additions, index=df.index)
    out = pd.concat([base, pa_df], axis=1)
    names = list(dict.fromkeys([n for n in base_names if n in out.columns] + [n for n in names if n in out.columns]))
    assert len(base_names) == 127, f"Expected 127 V8.27 base features, got {len(base_names)}"
    assert len(names) == 191, f"Expected 191 total features, got {len(names)}"
    return out, names


def train_models(X_df: pd.DataFrame, df: pd.DataFrame, targets: dict[str, np.ndarray], train_hi: int, seed_offset: int):
    train_end = train_hi - PURGE
    train_idx = np.arange(0, train_end, dtype=np.int64)
    day = df["timestamp"].dt.floor("D").astype(str).to_numpy()
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_df.iloc[:train_end]).astype(np.float32)
    X_all = imputer.transform(X_df).astype(np.float32)
    scores, refs = {}, {}
    for side, seed in (("L", 8261 + seed_offset), ("S", 8262 + seed_offset)):
        y = targets[side]
        idx, groups = sample_rank_training_indices(y, day, train_idx)
        if len(idx) == 0 or len(groups) == 0:
            raise RuntimeError(f"No sampled training rows for {side}.")
        model = XGBRanker(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.80, reg_lambda=3.0,
            min_child_weight=5.0, objective="rank:pairwise",
            eval_metric="ndcg@10", tree_method="hist", n_jobs=4,
            random_state=seed,
        )
        model.fit(X_train[idx], y[idx], group=groups)
        scores[side] = model.predict(X_all).astype(float)
        refs[side] = np.sort(scores[side][idx][np.isfinite(scores[side][idx])])
    return scores, refs


def run_one_fold(df, o, f, feature_names, targets, fold_no, od, write_summary=True):
    fold = make_walk_forward_folds(len(df))[fold_no - 1]
    X = f[feature_names].replace([np.inf, -np.inf], np.nan)
    print(f"--- V9.3 FOLD {fold_no} ---", flush=True)
    scores, refs = train_models(X, df, targets, fold["train"][1], fold_no * 100)
    rows = []
    for split in ("validation", "test"):
        lo, hi = fold[split]
        t = execute_policy(df, o, scores, LOCKED_POLICY, lo, hi, refs)
        m = metrics(t)
        m.update({"fold": fold_no, "split": split, "train_end_index": fold["train"][1], "eval_start_index": lo, "eval_end_index": hi})
        rows.append(m)
        t.to_csv(od / f"fold_{fold_no}_{split}_trades.csv", index=False)
        print(f"{split}: trades={m['trades']} | win={m['win_rate_pct']:.2f}% | raw_net={m['total_net_return']:+.6f} | equity={m['equity_return_pct']:+.4f}% | PF={m['profit_factor']:.3f} | MDD={m['max_drawdown_return']:.4%}", flush=True)
    future_m = None; future_ci = None
    if fold_no == 3:
        future_lo = int(len(df) * 0.95) + PURGE
        future_hi = len(df)
        ft = execute_policy(df, o, scores, LOCKED_POLICY, future_lo, future_hi, refs)
        future_m = metrics(ft); future_ci = bootstrap_mean_ci(ft, RANDOM_SEED + 903)
        ft.to_csv(od / "wf3_blind_future_trades.csv", index=False)
        print(f"future: trades={future_m['trades']} | win={future_m['win_rate_pct']:.2f}% | raw_net={future_m['total_net_return']:+.6f} | equity={future_m['equity_return_pct']:+.4f}% | PF={future_m['profit_factor']:.3f}", flush=True)
    summary = {"fold": fold_no, "features": len(feature_names), "validation": rows[0], "test": rows[1], "future": future_m, "future_ci": future_ci}
    (od / f"fold_{fold_no}_result.json").write_text(json.dumps(summary, indent=2, default=str))
    del scores, refs, X
    gc.collect()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ohlc", required=True)
    ap.add_argument("--output-dir", default="logs/v93")
    ap.add_argument("--fold", type=int, choices=[1, 2, 3], default=None)
    args = ap.parse_args()
    od = Path(args.output_dir); od.mkdir(parents=True, exist_ok=True)
    df = load_dataset(Path(args.data)); o = load_ohlc(Path(args.ohlc)); validate_ohlc(o)
    aligned = o.set_index("timestamp").reindex(df.timestamp)
    miss = int(aligned[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    if miss: raise ValueError(f"OHLC missing {miss} signal timestamps")

    f, feature_names = build_price_action_features(df, o)
    targets = make_targets(df.btcusd_close.to_numpy(float))
    print("=== ADAPTIVE TRADING AI V9.3 — PRICE ACTION AWARE RANKER ===")
    print(f"Rows: {len(df):,} | Features: {len(feature_names)} | Base V8.27 features preserved: 127")
    print("Added: 64 causal BTC price-action / market-structure features")
    print(f"Horizon: {HORIZON} bars | Cost: {COST_RT:.4%} | Policy: {asdict(LOCKED_POLICY)}")
    print("Adaptive failure learner: OFF (isolated price-action test)")

    folds = [args.fold] if args.fold else [1, 2, 3]
    results = [run_one_fold(df, o, f, feature_names, targets, n, od) for n in folds]
    if len(results) == 3:
        tests = [r["test"] for r in results]; vals = [r["validation"] for r in results]
        fold_test_comp = float(np.prod([1.0 + r["equity_return_pct"] / 100.0 for r in tests]) - 1.0)
        summary = {
            "version": "V9.3",
            "purpose": "isolate the effect of causal BTC price-action / market-structure features on V8.27",
            "feature_count": 191,
            "base_features": 127,
            "added_price_action_features": 64,
            "adaptive_failure_learner": "OFF",
            "validation_positive_folds": int(sum(r["total_net_return"] > 0 for r in vals)),
            "test_positive_folds": int(sum(r["total_net_return"] > 0 for r in tests)),
            "test_trade_count": int(sum(r["trades"] for r in tests)),
            "test_raw_net_sum": float(sum(r["total_net_return"] for r in tests)),
            "test_descriptive_fold_compounded": fold_test_comp,
            "blind_future": results[2]["future"],
            "blind_future_ci": results[2]["future_ci"],
            "walk_forward": {str(r["fold"]): r for r in results},
            "status": "RESEARCH_ONLY — NO LIVE EXECUTION",
        }
        (od / "v93_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        pd.DataFrame([
            {"fold": r["fold"], "validation_trades": r["validation"]["trades"], "validation_equity_pct": r["validation"]["equity_return_pct"], "validation_raw_net": r["validation"]["total_net_return"], "test_trades": r["test"]["trades"], "test_equity_pct": r["test"]["equity_return_pct"], "test_raw_net": r["test"]["total_net_return"], "test_pf": r["test"]["profit_factor"], "test_win_rate": r["test"]["win_rate_pct"]}
            for r in results
        ]).to_csv(od / "v93_walk_forward.csv", index=False)
        print("=== V9.3 FINAL ===", flush=True)
        print(json.dumps({k: summary[k] for k in ("validation_positive_folds","test_positive_folds","test_trade_count","test_raw_net_sum","test_descriptive_fold_compounded","blind_future","blind_future_ci")}, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
