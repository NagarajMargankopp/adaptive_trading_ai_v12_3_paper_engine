from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from typing import Callable, Optional

import pandas as pd
import websocket


PUBLIC_WS_URL = "wss://public-socket.india.delta.exchange"
DEFAULT_SYMBOL = "BTCUSD"
DEFAULT_CHANNEL = "candlestick_5m"


@dataclass(frozen=True)
class Candle:
    symbol: str
    start_us: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    server_ts_us: int

    @property
    def start(self) -> pd.Timestamp:
        return pd.to_datetime(self.start_us, unit="us", utc=True)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.start.isoformat()
        return d


def parse_candle_message(message: str | bytes) -> Optional[Candle]:
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    payload = json.loads(message)
    if payload.get("type") != DEFAULT_CHANNEL:
        return None
    required = ("sy", "cst", "o", "h", "l", "c", "v", "ts")
    if any(k not in payload for k in required):
        raise ValueError(f"Missing candlestick fields: {required}")
    return Candle(
        symbol=str(payload["sy"]),
        start_us=int(payload["cst"]),
        open=float(payload["o"]),
        high=float(payload["h"]),
        low=float(payload["l"]),
        close=float(payload["c"]),
        volume=float(payload["v"]),
        server_ts_us=int(payload["ts"]),
    )


class CandleStore:
    """Upsert the current forming candle and finalize it when a new candle starts."""

    COLUMNS = [
        "symbol", "start_us", "timestamp", "open", "high", "low", "close",
        "volume", "server_ts_us", "finalized", "local_received_ts"
    ]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.current_start_us: Optional[int] = None
        self.rows: dict[int, dict] = {}
        self.finalized_starts: set[int] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        df = pd.read_csv(self.path)
        if df.empty:
            return
        for row in df.to_dict(orient="records"):
            s = int(row["start_us"])
            value = row["finalized"]
            if isinstance(value, str):
                row["finalized"] = value.strip().lower() in {"true", "1", "yes"}
            else:
                row["finalized"] = bool(value)
            self.rows[s] = row
            if row["finalized"]:
                self.finalized_starts.add(s)
        non_final = [s for s, r in self.rows.items() if not r["finalized"]]
        self.current_start_us = max(non_final) if non_final else None

    def upsert(self, candle: Candle, received_ts: Optional[float] = None) -> tuple[Optional[dict], bool]:
        now = time.time() if received_ts is None else received_ts
        finalized: Optional[dict] = None
        new_candle_started = self.current_start_us is not None and candle.start_us > self.current_start_us
        if new_candle_started:
            old = self.rows.get(self.current_start_us)
            if old is not None and not old["finalized"]:
                old["finalized"] = True
                self.finalized_starts.add(self.current_start_us)
                finalized = dict(old)

        row = candle.to_dict()
        row["finalized"] = False
        row["local_received_ts"] = now
        self.rows[candle.start_us] = row
        self.current_start_us = candle.start_us
        self._flush()
        return finalized, new_candle_started

    def _flush(self) -> None:
        df = pd.DataFrame(list(self.rows.values()))
        if df.empty:
            return
        df = df.reindex(columns=self.COLUMNS).sort_values("start_us")
        df.to_csv(self.path, index=False)

    def finalized_dataframe(self) -> pd.DataFrame:
        rows = [r for r in self.rows.values() if r["finalized"]]
        if not rows:
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.DataFrame(rows).sort_values("start_us").reset_index(drop=True)


class DeltaCandleStreamer:
    def __init__(
        self,
        store: CandleStore,
        symbol: str = DEFAULT_SYMBOL,
        url: str = PUBLIC_WS_URL,
        on_finalized: Optional[Callable[[dict], None]] = None,
    ):
        self.store = store
        self.symbol = symbol
        self.url = url
        self.on_finalized = on_finalized
        self.ws: websocket.WebSocket | None = None

    def connect(self) -> websocket.WebSocket:
        self.ws = websocket.create_connection(self.url, timeout=15)
        self.ws.send(json.dumps({
            "type": "subscribe",
            "payload": {
                "channels": [{"name": DEFAULT_CHANNEL, "symbols": [self.symbol]}]
            },
        }))
        ack = json.loads(self.ws.recv())
        if ack.get("type") != "subscriptions":
            raise RuntimeError(f"Unexpected subscription response: {ack}")
        return self.ws

    def run(self, seconds: Optional[int] = None) -> int:
        ws = self.connect()
        started = time.time()
        count = 0
        try:
            while True:
                if seconds is not None and time.time() - started >= seconds:
                    break
                raw = ws.recv()
                candle = parse_candle_message(raw)
                if candle is None:
                    continue
                finalized, new_candle = self.store.upsert(candle)
                count += 1
                if new_candle and finalized is not None and self.on_finalized:
                    self.on_finalized(finalized)
        finally:
            ws.close()
        return count
