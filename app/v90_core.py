from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from app.v827_train import (
    LOCKED_POLICY,
    HORIZON,
    COST_RT,
    RISK_PER_TRADE,
    DAILY_LOSS_LIMIT,
    INITIAL_EQUITY,
    score_percentile,
    top_candidates_per_day,
)


@dataclass(frozen=True)
class LearnerConfig:
    name: str
    min_obs: int
    min_losses: int
    moderate_prob: float
    strong_prob: float
    moderate_action: float
    strong_action: float
    activation_gap: float = 0.10
    max_age_days: int = 90
    prior_strength: float = 8.0


CONFIGS = (
    LearnerConfig("OFF", 10_000, 10_000, 1.0, 1.0, 1.0, 1.0),
    LearnerConfig("BALANCED", 8, 4, 0.62, 0.75, 0.50, 0.0),
    LearnerConfig("CAUTIOUS", 10, 5, 0.66, 0.80, 0.50, 0.0),
    LearnerConfig("AGGRESSIVE", 6, 3, 0.60, 0.70, 0.50, 0.0),
)


class OnlineFailureLearner:
    """Causal Bayesian failure memory with deterministic hierarchical backoff."""

    def __init__(self, prior_strength: float = 8.0, max_age_days: int = 90):
        self.prior_strength = float(prior_strength)
        self.max_age_days = int(max_age_days)
        self.total_obs = 0
        self.total_losses = 0
        self.context: Dict[tuple, list[tuple[int, bool]]] = {}

    @staticmethod
    def signature(
        side: int,
        ret12: float,
        vol24: float,
        score_pct: float,
        vol_threshold: float,
    ) -> tuple | None:
        if not all(np.isfinite(v) for v in (ret12, vol24, score_pct)):
            return None

        momentum_state = int(side * ret12 >= 0.0)
        volatility_state = int(vol24 >= vol_threshold)
        score_state = 0 if score_pct < 0.94 else (1 if score_pct < 0.975 else 2)

        return (
            int(side),
            momentum_state,
            volatility_state,
            score_state,
        )

    @staticmethod
    def _levels(sig: tuple) -> list[tuple[str, tuple]]:
        """Most-specific to broadest deterministic backoff levels."""
        side, momentum, volatility, score = sig

        return [
            ("full", (side, momentum, volatility, score)),
            ("side_momentum_volatility", (side, momentum, volatility)),
            ("side_volatility", (side, volatility)),
            ("side_momentum", (side, momentum)),
            ("side", (side,)),
        ]

    def _global_loss_rate(self) -> float:
        return float(
            (self.total_losses + 0.5 * self.prior_strength)
            / (self.total_obs + self.prior_strength)
        )

    def _prune(
        self,
        events: list[tuple[int, bool]],
        day_num: int,
    ) -> list[tuple[int, bool]]:
        cutoff = int(day_num) - self.max_age_days
        return [e for e in events if e[0] >= cutoff]

    def predict(
        self,
        sig: tuple | None,
        day_num: int,
        cfg: LearnerConfig,
    ) -> tuple[float, int, int, float, str]:

        global_p = self._global_loss_rate()

        if sig is None:
            return global_p, 0, 0, 1.0, "global"

        selected_level = "global"
        selected_events: list[tuple[int, bool]] = []

        # Use the most specific context that has enough direct evidence.
        # If the full context is sparse, back off to a broader parent context.
        for level_name, key in self._levels(sig):
            events = self._prune(self.context.get(key, []), day_num)
            obs = len(events)
            losses = sum(int(loss) for _, loss in events)

            if obs >= cfg.min_obs and losses >= cfg.min_losses:
                selected_level = level_name
                selected_events = events
                break

        if not selected_events:
            return global_p, 0, 0, 1.0, "global"

        obs = len(selected_events)
        losses = sum(int(loss) for _, loss in selected_events)

        posterior = (
            losses + cfg.prior_strength * global_p
        ) / (
            obs + cfg.prior_strength
        )

        action = 1.0

        if (
            obs >= cfg.min_obs
            and losses >= cfg.min_losses
            and posterior - global_p >= cfg.activation_gap
        ):
            if posterior >= cfg.strong_prob:
                action = cfg.strong_action
            elif posterior >= cfg.moderate_prob:
                action = cfg.moderate_action

        return (
            float(posterior),
            int(obs),
            int(losses),
            float(action),
            selected_level,
        )

    def update(
        self,
        sig: tuple | None,
        day_num: int,
        is_loss: bool,
    ) -> None:

        self.total_obs += 1
        self.total_losses += int(is_loss)

        if sig is None:
            return

        # Store the completed trade in every hierarchy bucket.
        # It is counted once globally, while prediction selects only
        # one level, avoiding double-counting in a single decision.
        for _, key in self._levels(sig):
            events = self._prune(self.context.get(key, []), day_num)
            events.append((int(day_num), bool(is_loss)))
            self.context[key] = events

    def snapshot(self) -> dict:
        exact_contexts = sum(
            1 for key in self.context if len(key) == 4
        )

        return {
            "total_obs": self.total_obs,
            "total_losses": self.total_losses,
            "contexts": exact_contexts,
            "memory_buckets": len(self.context),
        }

def execute_adaptive(
    df: pd.DataFrame,
    ohlc: pd.DataFrame,
    features_df: pd.DataFrame,
    scores: dict[str, np.ndarray],
    refs: dict[str, np.ndarray],
    start: int,
    end: int,
    learner: OnlineFailureLearner,
    cfg: LearnerConfig,
    vol_threshold: float,
) -> tuple[pd.DataFrame, OnlineFailureLearner]:
    close = df["btcusd_close"].to_numpy(float)
    days = df["timestamp"].dt.floor("D").astype(str).to_numpy()
    day_num = (df["timestamp"].dt.floor("D") - df["timestamp"].dt.floor("D").min()).dt.days.to_numpy()
    eval_end = min(end - HORIZON, len(df) - HORIZON)
    out = ohlc.set_index("timestamp").reindex(df["timestamp"])
    if eval_end <= start:
        empty = pd.DataFrame()
        empty.attrs.update({"suppressed": 0, "scaled": 0})
        return empty, learner
    need = out[["open", "high", "low", "close"]].to_numpy(float)[start:eval_end]
    if not np.isfinite(need).all():
        raise ValueError("Missing/non-finite OHLC in evaluation range.")

    longs = top_candidates_per_day(scores["L"], days, start, eval_end, 1, LOCKED_POLICY.score_floor_pct, refs["L"])
    shorts = top_candidates_per_day(scores["S"], days, start, eval_end, 1, LOCKED_POLICY.score_floor_pct, refs["S"])
    by_index: dict[int, list[tuple[int, int, float]]] = {}
    for i in longs:
        by_index.setdefault(i, []).append((i, 1, float(scores["L"][i])))
    for i in shorts:
        by_index.setdefault(i, []).append((i, -1, float(scores["S"][i])))

    def pct(side: int, score: float) -> float:
        return score_percentile(refs["L" if side == 1 else "S"], score)

    resolved = [max(by_index[i], key=lambda x: pct(x[1], x[2])) for i in sorted(by_index)]

    trades = []
    next_free = start
    equity = INITIAL_EQUITY
    current_day = None
    day_start_equity = INITIAL_EQUITY
    suppressed = 0
    scaled = 0
    for i, side, score in resolved:
        if i < next_free or i >= eval_end:
            continue
        entry = float(close[i])
        if not np.isfinite(entry) or entry <= 0:
            continue
        if days[i] != current_day:
            current_day = days[i]
            day_start_equity = equity
        if equity / day_start_equity - 1.0 <= -DAILY_LOSS_LIMIT:
            continue

        score_pct = pct(side, score)
        sig = OnlineFailureLearner.signature(
            side,
            float(features_df["btcusd_ret_12"].iat[i]),
            float(features_df["btcusd_vol_24"].iat[i]),
            score_pct,
            vol_threshold,
        )
        p_loss, obs, losses, action, memory_level = learner.predict(sig, int(day_num[i]), cfg)
        if action == 0.0:
            suppressed += 1
            continue
        if action < 1.0:
            scaled += 1

        future = out.iloc[i + 1 : i + HORIZON + 1]
        if len(future) != HORIZON:
            continue
        sl = entry * (1.0 - LOCKED_POLICY.stop_pct) if side == 1 else entry * (1.0 + LOCKED_POLICY.stop_pct)
        tp = entry * (1.0 + LOCKED_POLICY.target_pct) if side == 1 else entry * (1.0 - LOCKED_POLICY.target_pct)
        reason = "TIME"
        exit_index = i + HORIZON
        exit_price = float(future.iloc[-1]["close"])
        ambiguous = False
        for off, row in enumerate(future.itertuples(index=False), start=0):
            high, low = float(row.high), float(row.low)
            sl_hit = low <= sl if side == 1 else high >= sl
            tp_hit = high >= tp if side == 1 else low <= tp
            if sl_hit and tp_hit:
                ambiguous = True
                reason = "SL"; exit_index = i + 1 + off; exit_price = sl; break
            if sl_hit:
                reason = "SL"; exit_index = i + 1 + off; exit_price = sl; break
            if tp_hit:
                reason = "TP"; exit_index = i + 1 + off; exit_price = tp; break

        gross = (exit_price / entry - 1.0) if side == 1 else (entry - exit_price) / entry
        net_return = gross - COST_RT
        net_r = net_return / LOCKED_POLICY.stop_pct
        equity_before = equity
        equity *= 1.0 + RISK_PER_TRADE * action * net_r
        is_loss = net_return <= 0
        learner.update(sig, int(day_num[exit_index]), is_loss)
        trades.append({
            "entry_index": i,
            "exit_index": int(exit_index),
            "entry_time": str(df.loc[i, "timestamp"]),
            "exit_time": str(df.loc[exit_index, "timestamp"]),
            "side": "LONG" if side == 1 else "SHORT",
            "score": float(score),
            "score_percentile": score_pct,
            "entry_price": entry,
            "exit_price": float(exit_price),
            "reason": reason,
            "gross_return": float(gross),
            "net_return": float(net_return),
            "net_r": float(net_r),
            "risk_multiplier": float(action),
            "memory_level": memory_level,
            "failure_probability_before": float(p_loss),
            "memory_observations_before": int(obs),
            "memory_losses_before": int(losses),
            "memory_context": str(sig),
            "ambiguous_same_bar": bool(ambiguous),
            "equity_before": float(equity_before),
            "equity_after": float(equity),
        })
        next_free = exit_index + 1

    result = pd.DataFrame(trades)
    result.attrs.update({"suppressed": suppressed, "scaled": scaled, "learner_snapshot": learner.snapshot()})
    return result, learner
