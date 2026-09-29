from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from xgboost import XGBRanker

SYMBOLS = ["btcusd", "ethusd", "solusd", "bnbusd", "xrpusd"]
HORIZON = 12                 # 12 x 5 minutes = 60 minutes
COST_RT = 0.0014             # 0.14% round trip
PURGE = 48                   # evaluation purge between chronological blocks

# Locked research policy. It is not optimized on test/future data.
LOCKED_POLICY = None
RISK_PER_TRADE = 0.005       # 0.5% equity risk budget at the stop distance
DAILY_LOSS_LIMIT = 0.02      # stop opening new positions after -2% realized daily loss

INITIAL_EQUITY = 10_000.0
RANDOM_SEED = 82525


@dataclass(frozen=True)
class Policy:
    k_per_side_per_day: int = 1
    score_floor_pct: float = 0.90
    stop_pct: float = 0.005
    target_pct: float = 0.010


LOCKED_POLICY = Policy()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer")
    required = ["timestamp", "btcusd_close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for c in [c for c in df.columns if c != "timestamp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df["timestamp"].duplicated().any():
        df = df.loc[~df["timestamp"].duplicated(keep="first")].reset_index(drop=True)
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not strictly increasing after sorting/deduplication.")
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build backward-looking features and deterministically remove duplicate names."""
    base = df.copy()
    additions: dict[str, pd.Series] = {}
    feature_names: list[str] = []

    for s in SYMBOLS:
        close_col = f"{s}_close"
        vol_col = f"{s}_volume"
        if close_col not in base.columns or vol_col not in base.columns:
            raise ValueError(f"Dataset must contain {close_col} and {vol_col}")
        p = base[close_col]
        v = base[vol_col]
        r1 = p.pct_change()

        for n in (1, 3, 6, 12, 24, 48, 96):
            name = f"{s}_ret_{n}"
            additions[name] = p.pct_change(n)
            feature_names.append(name)

        for w in (12, 24, 48, 96, 288):
            name = f"{s}_vol_{w}"
            additions[name] = r1.rolling(w, min_periods=max(4, w // 2)).std()
            feature_names.append(name)

        lv = np.log1p(v.clip(lower=0))
        dv = lv.diff()
        for w in (24, 48, 96):
            name = f"{s}_volume_z_{w}"
            mu = dv.rolling(w, min_periods=max(6, w // 2)).mean()
            sd = dv.rolling(w, min_periods=max(6, w // 2)).std()
            additions[name] = (dv - mu) / sd.replace(0, np.nan)
            feature_names.append(name)

    btc_r1 = additions["btcusd_ret_1"]
    for s in SYMBOLS[1:]:
        for n in (1, 3, 6, 12, 24, 48, 96):
            name = f"btc_vs_{s}_ret_{n}"
            additions[name] = additions[f"btcusd_ret_{n}"] - additions[f"{s}_ret_{n}"]
            feature_names.append(name)
        for w in (24, 48, 96):
            name = f"corr_btc_{s}_{w}"
            additions[name] = btc_r1.rolling(w, min_periods=max(6, w // 2)).corr(additions[f"{s}_ret_1"])
            feature_names.append(name)

    for h in (1, 3, 6, 12, 24, 48, 96):
        name = f"btc_momentum_accel_{h}"
        additions[name] = additions[f"btcusd_ret_{h}"].diff()
        feature_names.append(name)

    ts = base["timestamp"]
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    additions["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    additions["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    additions["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7.0)
    additions["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7.0)
    additions["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)
    feature_names += ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]

    additions_df = pd.DataFrame(additions, index=base.index)
    out = pd.concat([base, additions_df], axis=1)
    out = out.loc[:, ~out.columns.duplicated(keep="first")].copy()
    feature_names = list(dict.fromkeys(n for n in feature_names if n in out.columns))
    return out, feature_names


def make_splits(n: int) -> dict[str, tuple[int, int]]:
    a = int(n * 0.70)
    b = int(n * 0.80)
    c = int(n * 0.90)
    d = int(n * 0.95)
    return {
        "train": (0, a),
        "validation_1": (a + PURGE, b),
        "validation_2": (b + PURGE, c),
        "test_holdout": (c + PURGE, d),
        "reserved_future": (d + PURGE, n),
    }


def validate_time_axis(df: pd.DataFrame) -> dict:
    delta = df["timestamp"].diff().dropna()
    five = pd.Timedelta(minutes=5)
    bad = delta[delta != five]
    return {
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "rows": int(len(df)),
        "non_5m_deltas": int((delta != five).sum()),
        "max_gap_minutes": float(delta.max().total_seconds() / 60.0) if len(delta) else 0.0,
        "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "first_non_5m_examples": [str(x) for x in bad.head(5).tolist()],
    }


def make_targets(close: np.ndarray, horizon: int = HORIZON, cost_rt: float = COST_RT) -> dict[str, np.ndarray]:
    long_net = np.full(len(close), np.nan, dtype=float)
    short_net = np.full(len(close), np.nan, dtype=float)
    if len(close) > horizon:
        long_net[:-horizon] = close[horizon:] / close[:-horizon] - 1.0 - cost_rt
        short_net[:-horizon] = (close[:-horizon] - close[horizon:]) / close[:-horizon] - cost_rt
    return {"L": long_net, "S": short_net}


def fit_ranker(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> XGBRanker:
    model = XGBRanker(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_lambda=3.0,
        objective="rank:pairwise",
        eval_metric="ndcg@10",
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
    )
    model.fit(X, y, group=groups)
    return model


def score_percentile(sorted_reference: np.ndarray, value: float) -> float:
    if len(sorted_reference) == 0 or not np.isfinite(value):
        return 0.0
    return float(np.searchsorted(sorted_reference, value, side="right") / len(sorted_reference))


def top_candidates_per_day(
    scores: np.ndarray,
    day: np.ndarray,
    start: int,
    end: int,
    k: int,
    floor_pct: float,
    floor_reference: np.ndarray,
) -> list[int]:
    """Return at most k highest-score timestamps per UTC day, above a training-score floor."""
    if end <= start:
        return []
    selected: list[int] = []
    sub_scores = scores[start:end]
    sub_days = day[start:end]
    boundaries = np.flatnonzero(sub_days[1:] != sub_days[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(sub_days)]
    for a, b in zip(starts, ends):
        local = np.arange(a, b, dtype=np.int64)
        local = local[np.isfinite(sub_scores[local])]
        if len(local) == 0:
            continue
        take = min(k, len(local))
        order = np.argsort(sub_scores[local], kind="mergesort")[-take:]
        for j in local[order]:
            idx = start + int(j)
            if score_percentile(floor_reference, scores[idx]) >= floor_pct:
                selected.append(idx)
    return selected


def execute_policy(
    df: pd.DataFrame,
    scores: dict[str, np.ndarray],
    policy: Policy,
    start: int,
    end: int,
    train_score_refs: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Sequential close-only simulator with a real daily-loss block."""
    close = df["btcusd_close"].to_numpy(float)
    day = df["timestamp"].dt.floor("D").astype(str).to_numpy()

    # Strict isolation: candidate + HORIZON forward bars must stay inside this split.
    eval_end = min(end - HORIZON, len(df) - HORIZON)
    if eval_end <= start:
        return pd.DataFrame()

    longs = top_candidates_per_day(
        scores["L"], day, start, eval_end,
        policy.k_per_side_per_day, policy.score_floor_pct, train_score_refs["L"]
    )
    shorts = top_candidates_per_day(
        scores["S"], day, start, eval_end,
        policy.k_per_side_per_day, policy.score_floor_pct, train_score_refs["S"]
    )
    candidates = sorted(
        [(i, 1, scores["L"][i]) for i in longs] +
        [(i, -1, scores["S"][i]) for i in shorts],
        key=lambda x: x[0],
    )

    def pct(side: int, score: float) -> float:
        return score_percentile(train_score_refs["L" if side == 1 else "S"], score)

    # Resolve LONG/SHORT collision at the same timestamp without repeated full-list scans.
    by_index: dict[int, list[tuple[int, int, float]]] = {}
    for row in candidates:
        by_index.setdefault(row[0], []).append(row)
    resolved: list[tuple[int, int, float]] = []
    for i in sorted(by_index):
        same = by_index[i]
        resolved.append(max(same, key=lambda x: pct(x[1], x[2])))

    trades: list[dict] = []
    next_free = start
    equity = INITIAL_EQUITY
    current_day = None
    day_start_equity = INITIAL_EQUITY
    daily_block_count = 0

    for i, side, score in resolved:
        if i < next_free or i >= eval_end:
            continue
        if not np.isfinite(close[i]) or close[i] <= 0:
            continue

        entry_day = day[i]
        if entry_day != current_day:
            current_day = entry_day
            day_start_equity = equity

        day_loss = equity / day_start_equity - 1.0
        if day_loss <= -DAILY_LOSS_LIMIT:
            daily_block_count += 1
            continue

        future = close[i + 1 : i + HORIZON + 1]
        if len(future) != HORIZON or not np.isfinite(future).all():
            continue

        entry = close[i]
        path = (future / entry - 1.0) if side == 1 else ((entry - future) / entry)
        sl_hits = np.flatnonzero(path <= -policy.stop_pct)
        tp_hits = np.flatnonzero(path >= policy.target_pct)
        first_sl = int(sl_hits[0]) if len(sl_hits) else 10**9
        first_tp = int(tp_hits[0]) if len(tp_hits) else 10**9

        if first_sl <= first_tp and first_sl < 10**9:
            exit_index = i + 1 + first_sl
            gross = float(path[first_sl])
            reason = "SL"
        elif first_tp < 10**9:
            exit_index = i + 1 + first_tp
            gross = float(path[first_tp])
            reason = "TP"
        else:
            exit_index = i + HORIZON
            gross = float(path[-1])
            reason = "TIME"

        net_return = gross - COST_RT
        net_r = net_return / policy.stop_pct
        equity_before = equity
        equity *= 1.0 + RISK_PER_TRADE * net_r

        trades.append({
            "entry_index": i,
            "exit_index": exit_index,
            "entry_time": str(df.loc[i, "timestamp"]),
            "exit_time": str(df.loc[exit_index, "timestamp"]),
            "side": "LONG" if side == 1 else "SHORT",
            "reason": reason,
            "score": float(score),
            "score_percentile": pct(side, score),
            "gross_return": gross,
            "net_return": net_return,
            "net_r": net_r,
            "equity_before": equity_before,
            "equity_after": equity,
            "day_loss_before_entry": day_loss,
        })
        next_free = exit_index + 1

    out = pd.DataFrame(trades)
    if not out.empty:
        out.attrs["daily_block_count"] = daily_block_count
    return out


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": float("nan"),
            "total_net_return": 0.0,
            "mean_net_return": float("nan"),
            "median_net_return": float("nan"),
            "profit_factor": float("nan"),
            "max_drawdown_return": 0.0,
            "equity_return_pct": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "tp_trades": 0,
            "sl_trades": 0,
            "time_trades": 0,
        }
    r = trades["net_return"].to_numpy(float)
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    eq = np.cumprod(1.0 + np.clip(RISK_PER_TRADE * trades["net_r"].to_numpy(float), -0.999999, None))
    dd = eq / np.maximum.accumulate(eq) - 1.0
    return {
        "trades": int(len(r)),
        "win_rate_pct": float(np.mean(r > 0) * 100.0),
        "total_net_return": float(r.sum()),
        "mean_net_return": float(r.mean()),
        "median_net_return": float(np.median(r)),
        "profit_factor": float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else float("nan")),
        "max_drawdown_return": float(dd.min()),
        "equity_return_pct": float((eq[-1] - 1.0) * 100.0),
        "long_trades": int((trades["side"] == "LONG").sum()),
        "short_trades": int((trades["side"] == "SHORT").sum()),
        "tp_trades": int((trades["reason"] == "TP").sum()),
        "sl_trades": int((trades["reason"] == "SL").sum()),
        "time_trades": int((trades["reason"] == "TIME").sum()),
    }


def score_diagnostics(scores: np.ndarray, y: np.ndarray, start: int, end: int) -> dict:
    # Only diagnostics, never used to select the locked policy.
    eval_end = min(end - HORIZON, len(scores) - HORIZON)
    if eval_end <= start:
        return {"rows": 0, "spearman": float("nan")}
    s = scores[start:eval_end]
    t = y[start:eval_end]
    mask = np.isfinite(s) & np.isfinite(t)
    if mask.sum() < 3:
        return {"rows": int(mask.sum()), "spearman": float("nan")}
    rho = spearmanr(s[mask], t[mask]).statistic
    return {"rows": int(mask.sum()), "spearman": float(rho) if np.isfinite(rho) else float("nan")}


def bootstrap_mean_ci(trades: pd.DataFrame, seed: int, n_boot: int = 2000) -> tuple[float, float]:
    if trades.empty:
        return (float("nan"), float("nan"))
    r = trades["net_return"].to_numpy(float)
    rng = np.random.default_rng(seed)
    sample = rng.choice(r, size=(n_boot, len(r)), replace=True)
    means = sample.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def chunk_audit(trades: pd.DataFrame, start: int, end: int, chunks: int = 16) -> pd.DataFrame:
    edges = np.linspace(start, end, chunks + 1, dtype=int)
    rows = []
    for i in range(chunks):
        a, b = int(edges[i]), int(edges[i + 1])
        if trades.empty:
            x = trades
        else:
            x = trades[(trades["entry_index"] >= a) & (trades["entry_index"] < b)]
        rows.append({
            "chunk": i + 1,
            "start_index": a,
            "end_index": b,
            "trades": int(len(x)),
            "net_return": float(x["net_return"].sum()) if len(x) else 0.0,
            "mean_net_return": float(x["net_return"].mean()) if len(x) else float("nan"),
        })
    return pd.DataFrame(rows)


def random_baseline(df: pd.DataFrame, start: int, end: int, policy: Policy, seed: int = 12345) -> pd.DataFrame:
    """Random daily entry baseline with the same close-only execution assumptions."""
    rng = np.random.default_rng(seed)
    day = df["timestamp"].dt.floor("D").astype(str).to_numpy()
    close = df["btcusd_close"].to_numpy(float)
    eval_end = min(end - HORIZON, len(df) - HORIZON)
    if eval_end <= start:
        return pd.DataFrame()

    candidates: list[tuple[int, int]] = []
    sub_days = day[start:eval_end]
    boundaries = np.flatnonzero(sub_days[1:] != sub_days[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(sub_days)]
    for a, b in zip(starts, ends):
        local = np.arange(a, b, dtype=np.int64)
        if len(local) == 0:
            continue
        # One random long and one random short opportunity per UTC day.
        for side in (1, -1):
            candidates.append((start + int(local[rng.integers(len(local))]), side))

    trades: list[dict] = []
    next_free = start
    for i, side in sorted(candidates):
        if i < next_free or i >= eval_end or not np.isfinite(close[i]) or close[i] <= 0:
            continue
        future = close[i + 1 : i + HORIZON + 1]
        if len(future) != HORIZON or not np.isfinite(future).all():
            continue
        entry = close[i]
        path = (future / entry - 1.0) if side == 1 else ((entry - future) / entry)
        sl_hits = np.flatnonzero(path <= -policy.stop_pct)
        tp_hits = np.flatnonzero(path >= policy.target_pct)
        sl = int(sl_hits[0]) if len(sl_hits) else 10**9
        tp = int(tp_hits[0]) if len(tp_hits) else 10**9
        if sl <= tp and sl < 10**9:
            exit_idx = i + 1 + sl
            gross = float(path[sl])
            reason = "SL"
        elif tp < 10**9:
            exit_idx = i + 1 + tp
            gross = float(path[tp])
            reason = "TP"
        else:
            exit_idx = i + HORIZON
            gross = float(path[-1])
            reason = "TIME"
        net_return = gross - COST_RT
        trades.append({
            "entry_index": i,
            "exit_index": exit_idx,
            "side": "LONG" if side == 1 else "SHORT",
            "reason": reason,
            "net_return": net_return,
            "net_r": net_return / policy.stop_pct,
        })
        next_free = exit_idx + 1
    return pd.DataFrame(trades)


def development_check(
    validation_1: dict,
    validation_2: dict,
) -> tuple[bool, str]:
    eligible = (
        validation_1.get("trades", 0) >= 50 and
        validation_2.get("trades", 0) >= 50 and
        validation_1.get("mean_net_return", np.nan) > 0 and
        validation_2.get("mean_net_return", np.nan) > 0
    )
    return eligible, ("STABLE_POSITIVE_DEVELOPMENT" if eligible else "NOT_STABLE_POSITIVE_DEVELOPMENT")


def run(args: argparse.Namespace) -> int:
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    features_df, features = build_features(df)
    if len(features) == 0:
        raise ValueError("No features were constructed.")

    splits = make_splits(len(df))
    train_lo, train_hi = splits["train"]

    print("=== ADAPTIVE TRADING AI V8.25 — DAILY LEARNING-TO-RANK + RISK ENGINE ===")
    print(f"Rows: {len(df):,} | Features: {len(features)} | Horizon: {HORIZON} bars (60 minutes)")
    print(f"Round-trip cost: {COST_RT:.4%} | Purge: {PURGE} bars")
    print(f"Locked policy: {asdict(LOCKED_POLICY)}")
    print("Training only on the chronological TRAIN block; last 48 TRAIN rows are purged.")
    print("Validation_1 + Validation_2 are development checks only; TEST_HOLDOUT and RESERVED_FUTURE are locked.")
    print("Execution: top-1 opportunity per side per UTC day, 90th-percentile training score floor, close-only SL/TP.")
    print("Risk: 0.5% equity risk budget per trade, 2% realized daily loss block.")
    print("No threshold fitting on test/future. No live execution.")

    time_diag = validate_time_axis(df)
    print(f"Time axis: {time_diag}")

    X_df = features_df[features].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_train_full = imputer.fit_transform(X_df.iloc[train_lo:train_hi])
    X_all = imputer.transform(X_df).astype(np.float32)

    targets = make_targets(features_df["btcusd_close"].to_numpy(float))
    day = features_df["timestamp"].dt.floor("D").astype(str).to_numpy()

    scores: dict[str, np.ndarray] = {}
    refs: dict[str, np.ndarray] = {}
    importances: list[dict] = []
    diagnostics: list[dict] = []

    # Model training ends before the evaluation regions, so all model fitting is pre-test.
    train_model_end = train_hi - PURGE
    train_indices = np.arange(train_lo, train_model_end, dtype=np.int64)

    for side, seed in (("L", 8251), ("S", 8252)):
        y = targets[side]
        idx = train_indices[np.isfinite(y[train_indices])]
        groups = pd.Series(day[idx]).groupby(pd.Series(day[idx]), sort=False).size().to_numpy()
        if len(idx) == 0 or len(groups) == 0:
            raise RuntimeError(f"No training rows for side {side}.")
        print(f"Training {side} ranker on {len(idx):,} rows / {len(groups):,} UTC-day groups...", flush=True)
        model = fit_ranker(X_all[idx], y[idx], groups, seed)
        scores[side] = model.predict(X_all).astype(float)
        refs[side] = np.sort(scores[side][idx][np.isfinite(scores[side][idx])])
        for name, val in sorted(zip(features, model.feature_importances_), key=lambda z: z[1], reverse=True)[:25]:
            importances.append({"side": side, "feature": name, "importance": float(val)})
        print(f"{side} ranker complete.", flush=True)

        for split_name in ("validation_1", "validation_2", "test_holdout", "reserved_future"):
            lo, hi = splits[split_name]
            d = score_diagnostics(scores[side], y, lo, hi)
            d.update({"side": side, "split": split_name})
            diagnostics.append(d)

    # Evaluate the fixed policy. Nothing after Validation_2 can modify it.
    trade_tables: dict[str, pd.DataFrame] = {}
    result_rows: list[dict] = []
    for split_name in ("validation_1", "validation_2", "test_holdout", "reserved_future"):
        lo, hi = splits[split_name]
        trades = execute_policy(df, scores, LOCKED_POLICY, lo, hi, refs)
        trade_tables[split_name] = trades
        m = metrics(trades)
        m["split"] = split_name
        m["daily_loss_blocks"] = int(trades.attrs.get("daily_block_count", 0)) if not trades.empty else 0
        result_rows.append(m)

    dev1 = next(x for x in result_rows if x["split"] == "validation_1")
    dev2 = next(x for x in result_rows if x["split"] == "validation_2")
    dev_ok, dev_status = development_check(dev1, dev2)

    test_trades = trade_tables["test_holdout"]
    future_trades = trade_tables["reserved_future"]
    test_ci = bootstrap_mean_ci(test_trades, seed=RANDOM_SEED)
    future_ci = bootstrap_mean_ci(future_trades, seed=RANDOM_SEED + 1)
    test_chunks = chunk_audit(test_trades, *splits["test_holdout"])
    future_chunks = chunk_audit(future_trades, *splits["reserved_future"])
    baseline_test = random_baseline(df, *splits["test_holdout"], LOCKED_POLICY, seed=RANDOM_SEED)
    baseline_future = random_baseline(df, *splits["reserved_future"], LOCKED_POLICY, seed=RANDOM_SEED + 1)

    result_df = pd.DataFrame(result_rows)
    baseline_df = pd.DataFrame([
        {"split": "test_holdout", "baseline": "random_daily", **metrics(baseline_test)},
        {"split": "reserved_future", "baseline": "random_daily", **metrics(baseline_future)},
    ])

    # Audit for overlap/leakage in generated trades.
    overlap_audit = []
    for split_name, trades in trade_tables.items():
        if trades.empty:
            overlap_audit.append({"split": split_name, "sequential": True, "target_in_split": True, "nan_rows": 0})
            continue
        sequential = bool((trades["entry_index"].to_numpy()[1:] > trades["exit_index"].to_numpy()[:-1]).all())
        lo, hi = splits[split_name]
        target_in_split = bool((trades["exit_index"] < hi).all() and (trades["entry_index"] >= lo).all())
        nan_rows = int(trades.isna().any(axis=1).sum())
        overlap_audit.append({"split": split_name, "sequential": sequential, "target_in_split": target_in_split, "nan_rows": nan_rows})

    # Save all artifacts.
    pd.DataFrame(diagnostics).to_csv(outdir / "v825_score_diagnostics.csv", index=False)
    result_df.to_csv(outdir / "v825_selected_results.csv", index=False)
    baseline_df.to_csv(outdir / "v825_random_baseline.csv", index=False)
    test_chunks.to_csv(outdir / "v825_test_chunks.csv", index=False)
    future_chunks.to_csv(outdir / "v825_future_chunks.csv", index=False)
    pd.DataFrame(importances).to_csv(outdir / "v825_feature_importance.csv", index=False)
    pd.DataFrame(overlap_audit).to_csv(outdir / "v825_execution_audit.csv", index=False)
    for name, trades in trade_tables.items():
        trades.to_csv(outdir / f"v825_trades_{name}.csv", index=False)

    policy_row = {
        **asdict(LOCKED_POLICY),
        "development_status": dev_status,
        "selected_using_test": False,
        "selected_using_future": False,
        "selection_method": "fixed_locked_research_policy",
    }
    pd.DataFrame([policy_row]).to_csv(outdir / "v825_locked_policy.csv", index=False)

    summary = {
        "version": "V8.25",
        "rows": int(len(df)),
        "features": int(len(features)),
        "feature_names": features,
        "horizon_bars": HORIZON,
        "round_trip_cost": COST_RT,
        "purge_bars": PURGE,
        "locked_policy": asdict(LOCKED_POLICY),
        "development_status": dev_status,
        "development_eligible": bool(dev_ok),
        "time_axis": time_diag,
        "selected_results": result_rows,
        "random_test_metrics": metrics(baseline_test),
        "random_future_metrics": metrics(baseline_future),
        "test_bootstrap_mean_net_return_ci95": test_ci,
        "future_bootstrap_mean_net_return_ci95": future_ci,
        "execution_audit": overlap_audit,
        "limitations": [
            "Supplied dataset contains close/volume rather than high/low; SL/TP are therefore evaluated on 5-minute closes only.",
            "The dataset contains a small number of non-5-minute timestamp gaps, so a nominal 12-bar horizon is not guaranteed to equal exactly 60 wall-clock minutes across every gap.",
            "This is a research backtest. It does not place live exchange orders.",
        ],
        "status": "RESEARCH_ONLY — DO NOT DEPLOY LIVE",
    }
    (outdir / "v825_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== V8.25 LOCKED POLICY DEVELOPMENT CHECK ===")
    print(pd.DataFrame([dev1, dev2]).to_string(index=False))
    print(f"Development status: {dev_status}")
    print("\n=== V8.25 RESULTS ===")
    print(result_df.to_string(index=False))
    print("\n=== RANDOM BASELINE ===")
    print(baseline_df.to_string(index=False))
    print("\n=== ROBUSTNESS ===")
    print(f"Test bootstrap mean-net-return 95% CI: {test_ci}")
    print(f"Future bootstrap mean-net-return 95% CI: {future_ci}")
    print("\n=== EXECUTION AUDIT ===")
    print(pd.DataFrame(overlap_audit).to_string(index=False))
    print("\n=== OUTPUT ===")
    print(f"Selected results: {outdir / 'v825_selected_results.csv'}")
    print(f"Test trades: {outdir / 'v825_trades_test_holdout.csv'}")
    print(f"Future trades: {outdir / 'v825_trades_reserved_future.csv'}")
    print(f"Summary: {outdir / 'v825_summary.json'}")
    print("Status: RESEARCH ONLY — DO NOT DEPLOY LIVE")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="V8.25 adaptive intraday ranking AI with locked research policy")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("logs/v825"))
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
