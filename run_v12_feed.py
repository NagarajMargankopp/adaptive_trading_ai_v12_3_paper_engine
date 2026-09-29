from __future__ import annotations

import argparse
from pathlib import Path

from app.v12_market_feed import CandleStore, DeltaCandleStreamer


def main() -> int:
    ap = argparse.ArgumentParser(description="V12 live BTCUSD 5m public market-data collector — NO ORDERS")
    ap.add_argument("--seconds", type=int, default=90, help="Run duration; use 0 for continuous")
    ap.add_argument("--symbol", default="BTCUSD")
    ap.add_argument("--output", default="logs/v12/live_candles.csv")
    args = ap.parse_args()

    seconds = None if args.seconds == 0 else args.seconds
    store = CandleStore(Path(args.output))

    def on_finalized(row: dict) -> None:
        print(
            f"FINALIZED {row['timestamp']} | O={row['open']} H={row['high']} "
            f"L={row['low']} C={row['close']} V={row['volume']}",
            flush=True,
        )

    print("=== V12 LIVE MARKET DATA — PAPER ONLY ===")
    print("Public Delta websocket: connected data only; NO API credentials; NO orders")
    print(f"Symbol: {args.symbol} | Channel: candlestick_5m | Duration: {args.seconds}s")
    print(f"Output: {args.output}")

    streamer = DeltaCandleStreamer(store=store, symbol=args.symbol, on_finalized=on_finalized)
    updates = streamer.run(seconds=seconds)
    print(f"Updates received: {updates}")
    print(f"Stored candles: {len(store.rows)} | Finalized candles: {len(store.finalized_starts)}")
    print("=== V12 FEED STOPPED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
