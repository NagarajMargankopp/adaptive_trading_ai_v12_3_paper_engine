from pathlib import Path
from app.v12_paper_engine import PaperPositionEngine


def make_signal(side="LONG"):
    return {
        "timestamp": "2026-09-28 18:00:00+00:00",
        "btc_close": 100.0,
        "long_score": 1.0,
        "short_score": -1.0,
        "long_percentile": 0.95,
        "short_percentile": 0.05,
        "long_candidate": side == "LONG",
        "short_candidate": side == "SHORT",
        "signal": side,
    }


def candle(ts, o, h, l, c):
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": 1.0}


def test_long_entry_and_tp(tmp_path: Path):
    p = PaperPositionEngine(tmp_path, slippage_per_side=0.0)
    assert p.maybe_fill_from_signal(make_signal("LONG"), candle("2026-09-28 18:05:00+00:00", 100, 100, 100, 100))
    assert p.position is not None
    assert p.process_completed_btc_bar(candle("2026-09-28 18:05:00+00:00", 100, 100, 99, 100)) is None
    trade = p.process_completed_btc_bar(candle("2026-09-28 18:10:00+00:00", 100, 101, 100, 100.8))
    assert trade is not None
    assert trade["reason"] == "TP"


def test_long_entry_and_sl(tmp_path: Path):
    p = PaperPositionEngine(tmp_path, slippage_per_side=0.0)
    assert p.maybe_fill_from_signal(make_signal("LONG"), candle("2026-09-28 18:05:00+00:00", 100, 100, 100, 100))
    p.process_completed_btc_bar(candle("2026-09-28 18:05:00+00:00", 100, 100, 99, 100))
    trade = p.process_completed_btc_bar(candle("2026-09-28 18:10:00+00:00", 100, 100, 99.4, 99.8))
    assert trade is not None
    assert trade["reason"] == "SL"


def test_time_exit(tmp_path: Path):
    p = PaperPositionEngine(tmp_path, slippage_per_side=0.0, max_hold_bars=2)
    assert p.maybe_fill_from_signal(make_signal("LONG"), candle("2026-09-28 18:05:00+00:00", 100, 100, 100, 100))
    p.process_completed_btc_bar(candle("2026-09-28 18:05:00+00:00", 100, 100.1, 99.9, 100))
    assert p.process_completed_btc_bar(candle("2026-09-28 18:10:00+00:00", 100, 100.1, 99.9, 100)) is None
    trade = p.process_completed_btc_bar(candle("2026-09-28 18:15:00+00:00", 100, 100.1, 99.9, 100))
    assert trade is not None
    assert trade["reason"] == "TIME"
