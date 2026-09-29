# V12 — Live Market Data / Paper Only

## Added
- Delta public WebSocket `candlestick_5m` collector
- Causal forming-candle upsert/finalization logic
- Persistent live candle CSV
- Unit tests for parsing, updates, finalization, and persistence

## Safety
- No API keys
- No private channels
- No order endpoints
- No live execution path
