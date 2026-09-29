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
RANDOM_SEED = 82625
TRAIN_GROUP_SAMPLE_TOP_BOTTOM = 32


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


def sample_rank_training_indices(y: np.ndarray, day: np.ndarray, train_indices: np.ndarray, k: int = TRAIN_GROUP_SAMPLE_TOP_BOTTOM) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the top/bottom k labels per UTC day to reduce pairwise-ranking cost."""
    selected: list[int] = []
    groups: list[int] = []
    sub_days = day[train_indices]
    boundaries = np.flatnonzero(sub_days[1:] != sub_days[:-1]) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(sub_days)]
    for a, b in zip(starts, ends):
        local = train_indices[a:b]
        local = local[np.isfinite(y[local])]
        if len(local) == 0:
            continue
        kk = min(k, len(local) // 2)
        if kk <= 0:
            chosen = local
        else:
            order = np.argsort(y[local], kind="mergesort")
            chosen = np.concatenate([local[order[:kk]], local[order[-kk:]]])
        if len(chosen) == 0:
            continue
        selected.extend(chosen.tolist())
        groups.append(len(chosen))
    return np.asarray(selected, dtype=np.int64), np.asarray(groups, dtype=np.int32)


def fit_ranker(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> XGBRanker:
    model = XGBRanker(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_lambda=3.0,
        min_child_weight=5.0,
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



def make_walk_forward_folds(n: int) -> list[dict[str, tuple[int, int]]]:
    folds = []
    for i, (train_frac, test_end_frac) in enumerate(((0.55, 0.75), (0.65, 0.85), (0.75, 0.95)), start=1):
        train_end = int(n * train_frac)
        dev_end = int(n * (train_frac + 0.10))
        test_end = int(n * test_end_frac)
        folds.append({
            "name": f"wf_{i}",
            "train": (0, train_end),
            "validation": (train_end + PURGE, dev_end),
            "test": (dev_end + PURGE, test_end),
        })
    return folds


def train_fold_models(X_df: pd.DataFrame, df: pd.DataFrame, targets: dict[str, np.ndarray], train_hi: int, seed_offset: int):
    train_end = train_hi - PURGE
    train_idx = np.arange(0, train_end, dtype=np.int64)
    day = df["timestamp"].dt.floor("D").astype(str).to_numpy()
    imputer = SimpleImputer(strategy="median")
    X_train_all = imputer.fit_transform(X_df.iloc[:train_end]).astype(np.float32)
    X_all = imputer.transform(X_df).astype(np.float32)
    scores, refs = {}, {}
    for side, seed in (("L", 8261 + seed_offset), ("S", 8262 + seed_offset)):
        y = targets[side]
        idx, groups = sample_rank_training_indices(y, day, train_idx)
        if len(idx) == 0 or len(groups) == 0:
            raise RuntimeError(f"No sampled training rows for {side}.")
        model = fit_ranker(X_train_all[idx], y[idx], groups, seed)
        scores[side] = model.predict(X_all).astype(float)
        refs[side] = np.sort(scores[side][idx][np.isfinite(scores[side][idx])])
    return scores, refs


def run_walk_forward(df, X_df, targets):
    folds = make_walk_forward_folds(len(df))
    rows, trades_map = [], {}
    last_scores, last_refs = None, None
    for fold_no, fold in enumerate(folds, 1):
        scores, refs = train_fold_models(X_df, df, targets, fold["train"][1], fold_no * 100)
        last_scores, last_refs = scores, refs
        for key, label in (("validation", "validation"), ("test", "test")):
            lo, hi = fold[key]
            trades = execute_policy(df, scores, LOCKED_POLICY, lo, hi, refs)
            trades_map[f"wf{fold_no}_{label}"] = trades
            m = metrics(trades)
            m.update({"fold": fold_no, "split": label, "train_end_index": fold["train"][1], "eval_start_index": lo, "eval_end_index": hi})
            rows.append(m)
    wf = pd.DataFrame(rows)
    tests = wf[wf["split"] == "test"]
    positive = int((tests["mean_net_return"] > 0).sum()) if len(tests) else 0
    agg = {
        "test_folds": int(len(tests)),
        "positive_test_folds": positive,
        "positive_test_fold_fraction": float(positive / len(tests)) if len(tests) else 0.0,
        "test_total_net_return": float(sum(metrics(trades_map[f"wf{i}_test"])["total_net_return"] for i in range(1, len(folds)+1))),
        "test_total_trades": int(sum(len(trades_map[f"wf{i}_test"]) for i in range(1, len(folds)+1))),
        "status": "POSITIVE_IN_ALL_TEST_FOLDS" if positive == len(tests) and positive > 0 else ("POSITIVE_IN_2_OF_3_TEST_FOLDS" if positive >= 2 else "NOT_STABLE_ACROSS_TEST_FOLDS"),
    }
    return wf, trades_map, agg, last_scores, last_refs

def run(args: argparse.Namespace) -> int:
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(args.data)
    features_df, features = build_features(df)
    X_df = features_df[features].replace([np.inf, -np.inf], np.nan)
    targets = make_targets(features_df["btcusd_close"].to_numpy(float))
    splits = make_splits(len(df))

    print("=== ADAPTIVE TRADING AI V8.26.1 — SAMPLED DAILY RANKER + WALK-FORWARD ===")
    print(f"Rows: {len(df):,} | Features: {len(features)} | Horizon: {HORIZON} bars (60 minutes)")
    print(f"Round-trip cost: {COST_RT:.4%} | Purge: {PURGE} bars")
    print(f"Policy: {asdict(LOCKED_POLICY)} | Training sample: top/bottom {TRAIN_GROUP_SAMPLE_TOP_BOTTOM} labels per UTC day")
    print("Three expanding-train walk-forward folds; final 5% remains reserved future.")
    print("No threshold fitting on test/future. No live execution.")
    print(f"Time axis: {validate_time_axis(df)}")

    print("\n=== V8.26.1 WALK-FORWARD ===", flush=True)
    wf_df, wf_trades, wf_agg, wf3_scores, wf3_refs = run_walk_forward(df, X_df, targets)
    print(wf_df.to_string(index=False))
    print(f"Aggregate: {wf_agg}")

    # Blind final-future stress test uses the WF3 model trained only through 75% of history.
    # The 75%-95% interval remains completely unseen by this model.
    future_lo, future_hi = splits["reserved_future"]
    future_trades = execute_policy(df, wf3_scores, LOCKED_POLICY, future_lo, future_hi, wf3_refs)
    future_m = metrics(future_trades)
    future_m["split"] = "reserved_future_blind_wf3_model"
    future_m["daily_loss_blocks"] = int(future_trades.attrs.get("daily_block_count", 0)) if not future_trades.empty else 0
    future_ci = bootstrap_mean_ci(future_trades, RANDOM_SEED + 1)
    future_chunks = chunk_audit(future_trades, future_lo, future_hi)
    baseline = random_baseline(df, future_lo, future_hi, LOCKED_POLICY, seed=RANDOM_SEED + 1)

    audits = []
    for name, trades in {**wf_trades, "reserved_future_blind_wf3_model": future_trades}.items():
        if trades.empty:
            audits.append({"split": name, "sequential": True, "target_in_split": True, "nan_rows": 0})
            continue
        if name.startswith("wf"):
            fno = int(name[2])
            typ = name.split("_")[-1]
            f = make_walk_forward_folds(len(df))[fno - 1]
            lo, hi = f[typ]
        else:
            lo, hi = future_lo, future_hi
        sequential = bool((trades["entry_index"].to_numpy()[1:] > trades["exit_index"].to_numpy()[:-1]).all())
        target_in_split = bool((trades["entry_index"] >= lo).all() and (trades["exit_index"] < hi).all())
        audits.append({"split": name, "sequential": sequential, "target_in_split": target_in_split, "nan_rows": int(trades.isna().any(axis=1).sum())})

    wf_df.to_csv(outdir / "v8261_walk_forward_results.csv", index=False)
    pd.DataFrame([future_m]).to_csv(outdir / "v8261_blind_future_result.csv", index=False)
    pd.DataFrame([{"split": "reserved_future_blind_wf3_model", "baseline": "random_daily", **metrics(baseline)}]).to_csv(outdir / "v8261_random_baseline.csv", index=False)
    future_chunks.to_csv(outdir / "v8261_future_chunks.csv", index=False)
    pd.DataFrame(audits).to_csv(outdir / "v8261_execution_audit.csv", index=False)
    for name, trades in wf_trades.items():
        trades.to_csv(outdir / f"v8261_trades_{name}.csv", index=False)
    future_trades.to_csv(outdir / "v8261_trades_reserved_future_blind_wf3_model.csv", index=False)

    summary = {
        "version": "V8.26.1",
        "rows": int(len(df)), "features": int(len(features)), "feature_names": features,
        "horizon_bars": HORIZON, "round_trip_cost": COST_RT, "purge_bars": PURGE,
        "policy": asdict(LOCKED_POLICY),
        "training_method": "XGBRanker rank:pairwise, top/bottom 32 labels retained per UTC-day group, 80 trees",
        "walk_forward": {"folds": make_walk_forward_folds(len(df)), "aggregate": wf_agg, "results": wf_df.to_dict(orient="records")},
        "blind_reserved_future": {"training_fraction": 0.75, "result": future_m, "bootstrap_ci95": future_ci},
        "random_baseline": metrics(baseline),
        "execution_audit": audits,
        "time_axis": validate_time_axis(df),
        "environment": {"python": __import__("platform").python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": __import__("scipy").__version__, "scikit_learn": __import__("sklearn").__version__, "xgboost": __import__("xgboost").__version__},
        "limitations": [
            "Close-only SL/TP because supplied data lacks high/low.",
            "Non-5-minute timestamp gaps remain in supplied dataset.",
            "Earlier V8 research used the same overall dataset, so evaluation blocks are not pristine independent research.",
            "The blind future test intentionally uses the WF3 model trained through 75%; it is a stress test, not a production retrain.",
        ],
        "status": "RESEARCH_ONLY — DO NOT DEPLOY LIVE",
    }
    (outdir / "v8261_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== V8.26.1 BLIND RESERVED FUTURE (WF3 MODEL) ===")
    print(pd.DataFrame([future_m]).to_string(index=False))
    print("\n=== RANDOM BASELINE ===")
    print(pd.DataFrame([{"split": "reserved_future_blind_wf3_model", "baseline": "random_daily", **metrics(baseline)}]).to_string(index=False))
    print("\n=== ROBUSTNESS ===")
    print(f"Future bootstrap mean-net-return 95% CI: {future_ci}")
    print("\n=== EXECUTION AUDIT ===")
    print(pd.DataFrame(audits).to_string(index=False))
    print("\n=== OUTPUT ===")
    print(f"Walk-forward: {outdir / 'v8261_walk_forward_results.csv'}")
    print(f"Blind future: {outdir / 'v8261_blind_future_result.csv'}")
    print(f"Summary: {outdir / 'v8261_summary.json'}")
    print("Status: RESEARCH ONLY — DO NOT DEPLOY LIVE")
    return 0

def main() -> int:
    ap = argparse.ArgumentParser(description="V8.26.1 adaptive intraday ranking AI with locked research policy")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("logs/v826"))
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
