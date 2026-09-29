from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import websocket

from app.v12_market_feed import CandleStore, Candle, PUBLIC_WS_URL, DEFAULT_CHANNEL

DEFAULT_SYMBOLS = ("BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD")


def parse_multi_candle_message(message: str | bytes) -> Optional[Candle]:
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


class DeltaMultiCandleStreamer:
    """Collect 5m candles for the full V9.3 multi-asset feature universe."""

    def __init__(
        self,
        stores: dict[str, CandleStore],
        symbols: tuple[str, ...] = DEFAULT_SYMBOLS,
        url: str = PUBLIC_WS_URL,
        on_finalized: Optional[Callable[[dict], None]] = None,
    ):
        self.stores = stores
        self.symbols = tuple(symbols)
        self.url = url
        self.on_finalized = on_finalized
        self.ws: websocket.WebSocket | None = None

    def connect(self) -> websocket.WebSocket:
        self.ws = websocket.create_connection(self.url, timeout=30)
        self.ws.send(json.dumps({
            "type": "subscribe",
            "payload": {
                "channels": [{"name": DEFAULT_CHANNEL, "symbols": list(self.symbols)}]
            },
        }))
        ack = json.loads(self.ws.recv())
        if ack.get("type") != "subscriptions":
            raise RuntimeError(f"Unexpected subscription response: {ack}")
        return self.ws

    def _close_ws(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            finally:
                self.ws = None

    def _reconnect(self, started: float, seconds: Optional[int]) -> bool:
        delay = 2

        while True:
            if seconds is not None and time.time() - started >= seconds:
                return False

            print(f"[WS] Reconnecting in {delay}s...", flush=True)
            time.sleep(delay)

            try:
                self._close_ws()
                self.connect()
                print("[WS] Reconnected successfully", flush=True)
                return True
            except Exception as exc:
                print(
                    f"[WS] Reconnect failed: {type(exc).__name__}: {exc}",
                    flush=True,
                )
                delay = min(delay * 2, 30)

    def run(self, seconds: Optional[int] = None) -> int:
        started = time.time()
        count = 0

        try:
            self.connect()

            while True:
                if seconds is not None and time.time() - started >= seconds:
                    break

                ws = self.ws
                if ws is None:
                    if not self._reconnect(started, seconds):
                        break
                    continue

                try:
                    raw = ws.recv()
                except (
                    websocket.WebSocketTimeoutException,
                    websocket.WebSocketConnectionClosedException,
                    TimeoutError,
                    OSError,
                ) as exc:
                    print(
                        f"[WS] Connection interrupted: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )

                    if not self._reconnect(started, seconds):
                        break
                    continue

                candle = parse_multi_candle_message(raw)

                if candle is None or candle.symbol not in self.stores:
                    continue

                finalized, new_candle = self.stores[candle.symbol].upsert(candle)
                count += 1

                if new_candle and finalized is not None and self.on_finalized:
                    row = dict(finalized)
                    row["finalized_symbol"] = candle.symbol
                    self.on_finalized(row)

        finally:
            self._close_ws()

        return count
