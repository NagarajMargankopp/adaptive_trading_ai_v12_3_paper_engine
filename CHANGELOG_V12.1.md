# V12.1

- Added multi-symbol 5m candlestick collector.
- Collects BTCUSD, ETHUSD, SOLUSD, BNBUSD and XRPUSD in one public websocket connection.
- Keeps a separate persistent candle store per symbol.
- Preserves V12 finalized-candle semantics.
- No credentials and no order execution.
