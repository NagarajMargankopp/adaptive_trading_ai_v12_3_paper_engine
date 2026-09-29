import json
from pathlib import Path

from app.v12_market_feed import CandleStore
from app.v12_multi_market_feed import DeltaMultiCandleStreamer, parse_multi_candle_message


def msg(symbol="ETHUSD", start_us=1000, close=101.0):
    return json.dumps({
        "c": close, "cst": start_us, "h": 102.0, "l": 99.0, "o": 100.0,
        "res": "5m", "sy": symbol, "ts": 2000, "type": "candlestick_5m", "v": 123.0,
    })


def test_parse_multi_symbol():
    c = parse_multi_candle_message(msg())
    assert c.symbol == "ETHUSD"
    assert c.close == 101.0


def test_multi_streamer_subscribes_all(monkeypatch, tmp_path: Path):
    class FakeWS:
        def __init__(self):
            self.sent = []
            self.closed = False
        def send(self, payload): self.sent.append(json.loads(payload))
        def recv(self):
            if not hasattr(self, "ack"):
                self.ack = True
                return json.dumps({"type": "subscriptions"})
            self.closed = True
            raise RuntimeError("stop")
        def close(self): self.closed = True

    fake = FakeWS()
    monkeypatch.setattr("websocket.create_connection", lambda *a, **k: fake)
    stores = {s: CandleStore(tmp_path / f"{s}.csv") for s in ("BTCUSD", "ETHUSD")}
    streamer = DeltaMultiCandleStreamer(stores=stores, symbols=("BTCUSD", "ETHUSD"))
    try:
        streamer.run(seconds=1)
    except RuntimeError as exc:
        assert str(exc) == "stop"
    payload = fake.sent[0]
    assert payload["payload"]["channels"][0]["symbols"] == ["BTCUSD", "ETHUSD"]
