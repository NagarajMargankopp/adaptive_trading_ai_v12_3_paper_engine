# Adaptive Trading AI V12 — Live Market Data / Paper Only

V12 introduces a real-time public Delta Exchange market-data adapter.

## Scope

- BTCUSD 5-minute candlestick feed
- Repeated updates to the currently-forming candle are upserted
- When a new 5-minute candle appears, the previous candle is marked finalized
- Candles are persisted to `logs/v12/live_candles.csv`
- No API credentials
- No order submission
- No private/account channel

Delta's public WebSocket endpoint is `wss://public-socket.india.delta.exchange` and the `candlestick_5m` channel supplies OHLC updates for BTCUSD.

## Run

```bash
python -m pytest -q
python run_v12_feed.py --seconds 90
```

For continuous collection:

```bash
python run_v12_feed.py --seconds 0
```

The next V12 component will consume finalized candles, maintain 5m/15m/1h context, and only then connect the frozen V9.3 model for paper signals.
