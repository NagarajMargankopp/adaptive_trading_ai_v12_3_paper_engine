from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import timezone
from pathlib import Path
import json
import math
from typing import Optional


INITIAL_EQUITY = 10000.0
RISK_PER_TRADE = 0.005
STOP_PCT = 0.005
TARGET_PCT = 0.01
COST_RT = 0.0014
DEFAULT_SLIPPAGE = 0.0002  # 0.02% adverse per side; research assumption
MAX_HOLD_BARS = 12
DAILY_LOSS_LIMIT = 0.02


@dataclass
class PaperPosition:
    side: str
    signal_timestamp: str
    fill_timestamp: str
    signal_reference_price: float
    intended_open: float
    entry_price: float
    sl: float
    tp: float
    risk_budget_usd: float
    notional_usd: float
    quantity_btc: float
    score: float
    score_percentile: float
    hold_bars: int = 0
    entry_bar_skipped: bool = False


class PaperPositionEngine:
    """Deterministic paper-position engine. No private API and no live orders."""

    LIVE_ORDERS = False

    def __init__(
        self,
        output_dir: str | Path,
        initial_equity: float = INITIAL_EQUITY,
        risk_per_trade: float = RISK_PER_TRADE,
        stop_pct: float = STOP_PCT,
        target_pct: float = TARGET_PCT,
        cost_rt: float = COST_RT,
        slippage_per_side: float = DEFAULT_SLIPPAGE,
        max_hold_bars: int = MAX_HOLD_BARS,
        daily_loss_limit: float = DAILY_LOSS_LIMIT,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.initial_equity = float(initial_equity)
        self.equity = float(initial_equity)
        self.peak_equity = float(initial_equity)
        self.risk_per_trade = float(risk_per_trade)
        self.stop_pct = float(stop_pct)
        self.target_pct = float(target_pct)
        self.cost_rt = float(cost_rt)
        self.slippage = float(slippage_per_side)
        self.max_hold_bars = int(max_hold_bars)
        self.daily_loss_limit = float(daily_loss_limit)
        self.position: Optional[PaperPosition] = None
        self.trades: list[dict] = []
        self.events: list[dict] = []
        self.day: Optional[str] = None
        self.day_start_equity = self.equity

        self.trade_path = self.output_dir / "paper_trades.csv"
        self.event_path = self.output_dir / "paper_events.csv"
        self.summary_path = self.output_dir / "paper_summary.json"

    @staticmethod
    def _adverse_entry(open_px: float, side: str, slip: float) -> float:
        return open_px * (1.0 + slip) if side == "LONG" else open_px * (1.0 - slip)

    @staticmethod
    def _adverse_exit(px: float, side: str, slip: float) -> float:
        return px * (1.0 - slip) if side == "LONG" else px * (1.0 + slip)

    def _normalize_day(self, ts: str) -> str:
        return str(ts)[:10]

    def _daily_loss(self) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return self.equity / self.day_start_equity - 1.0

    def _ensure_day(self, ts: str) -> None:
        day = self._normalize_day(ts)
        if day != self.day:
            self.day = day
            self.day_start_equity = self.equity

    def _event(self, event: str, timestamp: str, **kwargs) -> None:
        row = {"event": event, "timestamp": timestamp, **kwargs}
        self.events.append(row)
        with self.event_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def maybe_fill_from_signal(self, signal: dict, current_btc_candle: Optional[dict]) -> bool:
        """Fill a signal at the next BTC candle open (1-bar latency) when allowed."""
        if signal.get("signal") not in {"LONG", "SHORT"}:
            return False
        if self.position is not None:
            self._event("SIGNAL_IGNORED_POSITION_OPEN", signal["timestamp"], signal=signal["signal"])
            return False
        if current_btc_candle is None:
            self._event("SIGNAL_WAITING_FOR_NEXT_BTC_CANDLE", signal["timestamp"], signal=signal["signal"])
            return False

        fill_ts = str(current_btc_candle["timestamp"])
        signal_ts = str(signal["timestamp"])
        if fill_ts == signal_ts:
            self._event("SIGNAL_WAITING_FOR_NEXT_BTC_CANDLE", signal_ts, signal=signal["signal"])
            return False

        self._ensure_day(fill_ts)
        if self._daily_loss() <= -self.daily_loss_limit:
            self._event("ENTRY_BLOCKED_DAILY_LOSS", fill_ts, daily_loss=self._daily_loss())
            return False

        side = str(signal["signal"])
        open_px = float(current_btc_candle["open"])
        if not math.isfinite(open_px) or open_px <= 0:
            self._event("ENTRY_BLOCKED_BAD_OPEN", fill_ts, open=open_px)
            return False
        entry = self._adverse_entry(open_px, side, self.slippage)
        sl = entry * (1.0 - self.stop_pct) if side == "LONG" else entry * (1.0 + self.stop_pct)
        tp = entry * (1.0 + self.target_pct) if side == "LONG" else entry * (1.0 - self.target_pct)
        risk_budget = self.equity * self.risk_per_trade
        notional = risk_budget / self.stop_pct
        qty = notional / entry
        self.position = PaperPosition(
            side=side,
            signal_timestamp=signal_ts,
            fill_timestamp=fill_ts,
            signal_reference_price=float(signal.get("btc_close", open_px)),
            intended_open=open_px,
            entry_price=entry,
            sl=sl,
            tp=tp,
            risk_budget_usd=risk_budget,
            notional_usd=notional,
            quantity_btc=qty,
            score=float(signal.get("long_score") if side == "LONG" else signal.get("short_score")),
            score_percentile=float(signal.get("long_percentile") if side == "LONG" else signal.get("short_percentile")),
        )
        self._event(
            "PAPER_ENTRY",
            fill_ts,
            side=side,
            signal_timestamp=signal_ts,
            intended_open=open_px,
            entry_price=entry,
            sl=sl,
            tp=tp,
            risk_budget_usd=risk_budget,
            notional_usd=notional,
            quantity_btc=qty,
            slippage_per_side=self.slippage,
        )
        return True

    def process_completed_btc_bar(self, bar: dict) -> Optional[dict]:
        """Evaluate SL/TP/TIME on a completed BTC candle.

        The entry candle itself is not evaluated for exits, matching V11's
        1-bar-latency replay convention. Subsequent bars are evaluated.
        """
        if self.position is None:
            return None
        p = self.position
        ts = str(bar["timestamp"])
        self._ensure_day(ts)

        if not p.entry_bar_skipped:
            p.entry_bar_skipped = True
            return None

        p.hold_bars += 1
        op = float(bar["open"]); hi = float(bar["high"]); lo = float(bar["low"]); cl = float(bar["close"])
        side = p.side

        if side == "LONG":
            open_sl = op <= p.sl
            open_tp = op >= p.tp
            sl_hit = lo <= p.sl
            tp_hit = hi >= p.tp
        else:
            open_sl = op >= p.sl
            open_tp = op <= p.tp
            sl_hit = hi >= p.sl
            tp_hit = lo <= p.tp

        exit_px: Optional[float] = None
        reason: Optional[str] = None
        ambiguous = False
        if open_sl and open_tp:
            exit_px = self._adverse_exit(op, side, self.slippage); reason = "SL_FIRST_GAP_AMBIGUITY"; ambiguous = True
        elif open_sl:
            exit_px = self._adverse_exit(op, side, self.slippage); reason = "SL_GAP"
        elif open_tp:
            exit_px = self._adverse_exit(op, side, self.slippage); reason = "TP_GAP"
        elif sl_hit and tp_hit:
            exit_px = self._adverse_exit(p.sl, side, self.slippage); reason = "SL_FIRST"; ambiguous = True
        elif sl_hit:
            exit_px = self._adverse_exit(p.sl, side, self.slippage); reason = "SL"
        elif tp_hit:
            exit_px = self._adverse_exit(p.tp, side, self.slippage); reason = "TP"
        elif p.hold_bars >= self.max_hold_bars:
            exit_px = self._adverse_exit(cl, side, self.slippage); reason = "TIME"

        if exit_px is None:
            return None

        gross = (exit_px / p.entry_price - 1.0) if side == "LONG" else (p.entry_price - exit_px) / p.entry_price
        net_return = gross - self.cost_rt
        net_r = net_return / self.stop_pct
        eq_before = self.equity
        self.equity *= 1.0 + self.risk_per_trade * net_r
        self.peak_equity = max(self.peak_equity, self.equity)
        dd = self.equity / self.peak_equity - 1.0

        trade = {
            "signal_timestamp": p.signal_timestamp,
            "fill_timestamp": p.fill_timestamp,
            "exit_timestamp": ts,
            "side": side,
            "signal_reference_price": p.signal_reference_price,
            "intended_open": p.intended_open,
            "entry_price": p.entry_price,
            "exit_price": float(exit_px),
            "sl": p.sl,
            "tp": p.tp,
            "score": p.score,
            "score_percentile": p.score_percentile,
            "slippage_per_side": self.slippage,
            "gross_return": float(gross),
            "net_return": float(net_return),
            "net_r": float(net_r),
            "reason": reason,
            "same_bar_ambiguity": bool(ambiguous),
            "hold_bars": int(p.hold_bars),
            "risk_budget_usd": p.risk_budget_usd,
            "notional_usd": p.notional_usd,
            "quantity_btc": p.quantity_btc,
            "equity_before": float(eq_before),
            "equity_after": float(self.equity),
            "drawdown_from_peak": float(dd),
            "daily_loss_after": float(self._daily_loss()),
        }
        self.trades.append(trade)
        self._write_trades()
        self._event("PAPER_EXIT", ts, **trade)
        self.position = None
        return trade

    def _write_trades(self) -> None:
        if not self.trades:
            return
        import pandas as pd
        pd.DataFrame(self.trades).to_csv(self.trade_path, index=False)

    def summary(self) -> dict:
        import pandas as pd
        d = pd.DataFrame(self.trades)
        if d.empty:
            s = {
                "live_orders": False,
                "initial_equity": self.initial_equity,
                "current_equity": self.equity,
                "equity_return_pct": (self.equity / self.initial_equity - 1.0) * 100.0,
                "realized_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate_pct": 0.0,
                "profit_factor": None,
                "max_equity_drawdown": 0.0,
                "open_position": asdict(self.position) if self.position else None,
                "slippage_per_side_pct": self.slippage * 100.0,
                "round_trip_cost_pct": self.cost_rt * 100.0,
            }
            self.summary_path.write_text(
                json.dumps(s, indent=2, default=str),
                encoding="utf-8"
            )
            return s
        r = d["net_return"].to_numpy(float)
        gains = float(r[r > 0].sum())
        losses = float(-r[r < 0].sum())
        s = {
            "live_orders": False,
            "initial_equity": self.initial_equity,
            "current_equity": self.equity,
            "equity_return_pct": (self.equity / self.initial_equity - 1.0) * 100.0,
            "realized_trades": int(len(d)),
            "wins": int((r > 0).sum()),
            "losses": int((r <= 0).sum()),
            "win_rate_pct": float((r > 0).mean() * 100.0),
            "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
            "max_equity_drawdown": float((d["equity_after"] / d["equity_after"].cummax() - 1.0).min()),
            "open_position": asdict(self.position) if self.position else None,
            "slippage_per_side_pct": self.slippage * 100.0,
            "round_trip_cost_pct": self.cost_rt * 100.0,
        }
        self.summary_path.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
        return s
