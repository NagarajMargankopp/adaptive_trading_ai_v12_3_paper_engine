# Adaptive Trading AI V11 — Final Paper-Execution Report

## Status
**RESEARCH-ONLY — PAPER EXECUTION SIMULATION — NO LIVE ORDERS**

## What was changed
V11 replays the **frozen V9.3 trade candidates** with a more realistic execution lifecycle. The V9.3 signal is generated at the signal-bar close; V11 fills on a later bar open, applies adverse per-side slippage, recalculates SL/TP from the actual fill, handles gap-through exits conservatively, applies the existing 0.14% round-trip cost, and prevents overlapping positions based on the simulated exit.

The simulator also records continuous BTC quantity/notional for audit, but does not infer exchange-specific contract/lot rules.

## Data alignment audit
The OHLC series is aligned to the V9.3 signal dataset by timestamp before using `entry_index`. This prevents positional misalignment between the 258,639-row signal data and the larger raw OHLC file.

## V9.3 baseline reference
- Held-out test candidates: **267 trades**
- Descriptive chained equity return across the three non-overlapping test streams: **+78.98%**
- Reserved future: **33 trades**, **+5.91%** equity return

The chained test figure is a sensitivity accounting calculation across separate walk-forward test streams, not a single continuous live account result.

## Primary execution scenario
A reasonable conservative paper scenario for this dataset is **1 bar entry latency + 0.02% adverse slippage per side**. These are simulation assumptions, not observed exchange execution statistics.

### Held-out test by fold
 fold  v93_equity_pct  v11_1bar_0.02_equity_pct  change_pp  trades  win_rate_pct  profit_factor  max_drawdown_pct
    1       15.118348                 13.650078  -1.468270      95     64.210526       1.789095         -2.903326
    2       30.747032                 24.018914  -6.728118     101     67.326733       3.274599         -1.543267
    3       18.909647                 17.504420  -1.405227      71     67.605634       3.868447         -0.964007

Aggregate across the three test streams:
- Trades: **267**
- Descriptive chained equity return: **+65.62%**
- Win rate: **66.29%**
- Profit factor: **2.609**
- Max drawdown: **-2.90%** under the chained simulation
- Relative to the V9.3 stored-entry baseline, the execution assumption reduces the chained equity result by about **13.36 percentage points**.

## Execution sensitivity
### 1-bar latency
| Slippage per side | Test equity return | Test PF |
|---:|---:|---:|
| 0.00% | +84.07% | 3.184 |
| 0.02% | +65.62% | 2.609 |
| 0.05% | +40.41% | 1.903 |
| 0.10% | +6.73% | 1.137 |

### 2-bar latency
| Slippage per side | Test equity return | Test PF |
|---:|---:|---:|
| 0.00% | +36.78% | 1.870 |
| 0.02% | +22.68% | 1.506 |
| 0.05% | +4.53% | 1.097 |
| 0.10% | -20.28% | 0.652 |

### 3-bar latency
| Slippage per side | Test equity return | Test PF |
|---:|---:|---:|
| 0.00% | +17.00% | 1.363 |
| 0.02% | +4.04% | 1.086 |
| 0.05% | -12.53% | 0.780 |
| 0.10% | -36.98% | 0.419 |

## Reserved future block
For the 33-trade reserved future block:
- V9.3 stored-entry baseline: **+5.91%** equity return, PF **2.573**
- 1-bar + 0.02% slippage: **+5.29%**, PF **2.189**
- 1-bar + 0.05% slippage: **+3.29%**, PF **1.641**
- 1-bar + 0.10% slippage: **+0.34%**, PF **1.058**
- 2-bar + 0.02% slippage: **-1.13%**, PF **0.797**
- 2-bar + 0.05% slippage: **-2.86%**, PF **0.542**

The future sample is small, so these values are sensitivity evidence rather than a forecast.

## Fill-friction sensitivity
At 1-bar latency and 0.02% adverse slippage per side, the test chained equity return was:
- 100% fill, 0% rejection: **+65.62%**
- 75% fill, 0% rejection: **+46.10%**
- 50% fill, 0% rejection: **+28.81%**
- 100% fill, 10% rejection: **+53.52%**

These partial-fill and rejection rates are deliberately labeled as **stress assumptions**, not measured exchange behavior.

## Main conclusion
V11 shows that the historical V9.3 edge is **sensitive to entry delay and execution friction**. A 1-bar/0.02% scenario remains positive in this historical replay, while larger delays or slippage can reduce the edge substantially. Because no bid/ask/order-book history was supplied, V11 cannot establish actual Delta Exchange fill quality.

V11 therefore does **not** authorize live trading.

## Next gate
The next stage should be a **real-time paper adapter** that consumes live Delta market data and logs simulated order submission, fill time, fill price, SL/TP state, fees, and P&L **without sending exchange orders**. Its purpose is to compare real observed paper execution against the V11 historical assumptions before considering any live integration.
