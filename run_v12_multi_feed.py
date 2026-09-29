from __future__ import annotations

import argparse
from pathlib import Path

from app.v12_market_feed import CandleStore
from app.v12_multi_market_feed import DEFAULT_SYMBOLS, DeltaMultiCandleStreamer


def main() -> int:
    ap = argparse.ArgumentParser(description="V12.1 live multi-asset 5m public market-data collector — NO ORDERS")
    ap.add_argument("--seconds", type=int, default=90)
    ap.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    ap.add_argument("--output-dir", default="logs/v12/multi")
    args = ap.parse_args()

    symbols = tuple(args.symbols)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stores = {sym: CandleStore(out / f"{sym}_5m.csv") for sym in symbols}

    finalized_counts = {sym: 0 for sym in symbols}

    def on_finalized(row: dict) -> None:
        sym = row["finalized_symbol"]
        finalized_counts[sym] += 1
        print(
            f"FINALIZED {sym} {row['timestamp']} | O={row['open']} H={row['high']} "
            f"L={row['low']} C={row['close']} V={row['volume']}",
            flush=True,
        )

    print("=== V12.1 LIVE MULTI-ASSET MARKET DATA — PAPER ONLY ===")
    print("Public Delta websocket: data only; NO API credentials; NO orders")
    print(f"Symbols: {', '.join(symbols)} | Channel: candlestick_5m | Duration: {args.seconds}s")
    print(f"Output: {out}")

    streamer = DeltaMultiCandleStreamer(stores=stores, symbols=symbols, on_finalized=on_finalized)
    updates = streamer.run(seconds=None if args.seconds == 0 else args.seconds)

    print(f"Updates received: {updates}")
    for sym in symbols:
        store = stores[sym]
        print(
            f"{sym}: stored={len(store.rows)} finalized={len(store.finalized_starts)} "
            f"callbacks={finalized_counts[sym]}"
        )
    print("=== V12.1 FEED STOPPED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
