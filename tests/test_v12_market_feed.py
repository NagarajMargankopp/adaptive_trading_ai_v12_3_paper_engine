import json
from pathlib import Path

from app.v12_market_feed import CandleStore, parse_candle_message


def msg(start_us=1000, close=101.0):
    return json.dumps({
        "c": close, "cst": start_us, "h": 102.0, "l": 99.0, "o": 100.0,
        "res": "5m", "sy": "BTCUSD", "ts": 2000, "type": "candlestick_5m", "v": 123.0,
    })


def test_parse_candle():
    c = parse_candle_message(msg())
    assert c.symbol == "BTCUSD"
    assert c.start_us == 1000
    assert c.close == 101.0


def test_ignore_other_message():
    assert parse_candle_message('{"type":"heartbeat"}') is None


def test_upsert_same_candle_does_not_finalize(tmp_path: Path):
    s = CandleStore(tmp_path / "candles.csv")
    c = parse_candle_message(msg(1000, 101.0))
    assert s.upsert(c) == (None, False)
    c2 = parse_candle_message(msg(1000, 103.0))
    assert s.upsert(c2) == (None, False)
    assert s.rows[1000]["close"] == 103.0
    assert not s.rows[1000]["finalized"]


def test_new_candle_finalizes_previous(tmp_path: Path):
    s = CandleStore(tmp_path / "candles.csv")
    c1 = parse_candle_message(msg(1000, 101.0))
    c2 = parse_candle_message(msg(1300, 104.0))
    s.upsert(c1)
    finalized, started = s.upsert(c2)
    assert started is True
    assert finalized is not None
    assert finalized["start_us"] == 1000
    assert s.rows[1000]["finalized"] is True
    assert s.rows[1300]["finalized"] is False


def test_persistence(tmp_path: Path):
    p = tmp_path / "candles.csv"
    s = CandleStore(p)
    s.upsert(parse_candle_message(msg(1000, 101.0)))
    s.upsert(parse_candle_message(msg(1300, 104.0)))
    s2 = CandleStore(p)
    assert 1000 in s2.finalized_starts
    assert 1300 in s2.rows
    assert s2.rows[1300]["close"] == 104.0
