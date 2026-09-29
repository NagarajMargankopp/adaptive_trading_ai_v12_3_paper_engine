from pathlib import Path
import json, math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'logs' / 'v93_final'
OHLC_PATH = ROOT / 'data' / 'processed' / 'btcusd_5m_ohlc.csv.gz'
OUT = ROOT / 'logs' / 'v11'
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_EQUITY = 10000.0
RISK_PER_TRADE = 0.005
STOP_PCT = 0.005
TARGET_PCT = 0.01
COST_RT = 0.0014
DAILY_LOSS_LIMIT = 0.02
MAX_HOLD_BARS = 12

# These are scenario assumptions, not exchange facts.
LATENCIES = [1, 2, 3]  # entry fill on 1st/2nd/3rd bar after signal
SLIPPAGES = [0.0000, 0.0002, 0.0005, 0.0010]  # adverse per side
PARTIAL_FILL_FRACTIONS = [1.0, 0.75, 0.50]
REJECTION_RATES = [0.0, 0.05, 0.10]


def adverse_entry(open_px, side, slip):
    return open_px * (1.0 + slip) if side == 'LONG' else open_px * (1.0 - slip)


def adverse_exit(px, side, slip):
    return px * (1.0 - slip) if side == 'LONG' else px * (1.0 + slip)


def load_signals():
    frames = []
    for f in (1, 2, 3):
        p = SRC / f'fold_{f}_test_trades.csv'
        d = pd.read_csv(p)
        d['fold'] = f
        frames.append(d)
    test = pd.concat(frames, ignore_index=True)
    test = test.sort_values(['entry_index', 'fold']).reset_index(drop=True)
    future = pd.read_csv(SRC / 'wf3_blind_future_trades.csv')
    future['fold'] = 3
    future = future.sort_values('entry_index').reset_index(drop=True)
    return test, future


def load_ohlc():
    sig_df = pd.read_csv(ROOT / 'data' / 'processed' / 'v821_common_multiasset.csv.gz', usecols=['timestamp'])
    sig_df['timestamp'] = pd.to_datetime(sig_df['timestamp'], utc=True)
    o = pd.read_csv(OHLC_PATH)
    o['timestamp'] = pd.to_datetime(o['timestamp'], utc=True)
    aligned = o.set_index('timestamp').reindex(sig_df['timestamp'])[['open','high','low','close']].reset_index()
    aligned.rename(columns={'index':'timestamp'}, inplace=True)
    if aligned[['open','high','low','close']].isna().any().any():
        raise ValueError('OHLC alignment has missing signal timestamps')
    return aligned


def simulate(signals, ohlc, latency, slip, partial_fraction=1.0, rejection_rate=0.0, seed=0):
    if signals.empty:
        return pd.DataFrame(), {
            'trades': 0, 'filled_trades': 0, 'rejected_trades': 0,
            'partial_fill_fraction': partial_fraction, 'rejection_rate': rejection_rate,
            'equity_return_pct': 0.0, 'total_net_return': 0.0,
            'win_rate_pct': float('nan'), 'profit_factor': float('nan'),
            'max_drawdown_return': 0.0,
        }
    rng = np.random.default_rng(seed)
    o = ohlc[['open','high','low','close']].to_numpy(float)
    ts = ohlc['timestamp'].astype(str).to_numpy() if 'timestamp' in ohlc.columns else np.arange(len(o)).astype(str)
    eq = INITIAL_EQUITY
    peak = eq
    rows = []
    next_free = 0
    day = None
    day_start = INITIAL_EQUITY
    rej = 0

    for _, s in signals.iterrows():
        i = int(s['entry_index'])
        side = s['side']
        if i < next_free:
            continue
        fill_i = i + latency
        if fill_i >= len(o) - 1:
            continue
        fill_day = str(pd.Timestamp(ohlc.iloc[fill_i]['timestamp']))[:10]
        if fill_day != day:
            day = fill_day
            day_start = eq
        day_loss = eq / day_start - 1.0
        if day_loss <= -DAILY_LOSS_LIMIT:
            continue

        u = rng.random()
        filled_fraction = partial_fraction
        if u < rejection_rate:
            rej += 1
            continue

        raw_open = float(o[fill_i,0])
        if not math.isfinite(raw_open) or raw_open <= 0:
            continue
        entry = adverse_entry(raw_open, side, slip)
        sl = entry * (1.0 - STOP_PCT) if side == 'LONG' else entry * (1.0 + STOP_PCT)
        tp = entry * (1.0 + TARGET_PCT) if side == 'LONG' else entry * (1.0 - TARGET_PCT)

        exit_i = None; exit_px = None; reason = None; ambiguity = False
        last_i = min(fill_i + MAX_HOLD_BARS, len(o)-1)
        for j in range(fill_i + 1, last_i + 1):
            op, hi, lo, cl = map(float, o[j])
            # Gap handling: if the market opens beyond a protective level, fill at adverse open.
            sl_hit = lo <= sl if side == 'LONG' else hi >= sl
            tp_hit = hi >= tp if side == 'LONG' else lo <= tp
            if side == 'LONG':
                open_sl = op <= sl
                open_tp = op >= tp
            else:
                open_sl = op >= sl
                open_tp = op <= tp

            if open_sl and open_tp:
                ambiguity = True
                exit_i, exit_px, reason = j, adverse_exit(op, side, slip), 'SL_FIRST_GAP_AMBIGUITY'
                break
            if open_sl:
                exit_i, exit_px, reason = j, adverse_exit(op, side, slip), 'SL_GAP'
                break
            if open_tp:
                exit_i, exit_px, reason = j, adverse_exit(op, side, slip), 'TP_GAP'
                break
            if sl_hit and tp_hit:
                ambiguity = True
                exit_i, exit_px, reason = j, adverse_exit(sl, side, slip), 'SL_FIRST'
                break
            if sl_hit:
                exit_i, exit_px, reason = j, adverse_exit(sl, side, slip), 'SL'
                break
            if tp_hit:
                exit_i, exit_px, reason = j, adverse_exit(tp, side, slip), 'TP'
                break

        if exit_i is None:
            exit_i = last_i
            exit_px = adverse_exit(float(o[exit_i,3]), side, slip)
            reason = 'TIME'

        gross = (exit_px / entry - 1.0) if side == 'LONG' else (entry - exit_px) / entry
        # Fees/cost apply to executed fraction; slippage is already embedded in gross.
        net_return = (gross - COST_RT) * filled_fraction
        net_r = net_return / STOP_PCT
        eq_before = eq
        risk_budget = eq_before * RISK_PER_TRADE
        notional = risk_budget / STOP_PCT
        quantity_btc = notional / entry
        eq *= 1.0 + RISK_PER_TRADE * net_r
        peak = max(peak, eq)
        rows.append({
            'signal_entry_index': i,
            'fill_index': fill_i,
            'exit_index': int(exit_i),
            'signal_time': str(s.get('entry_time','')),
            'fill_time': str(ts[fill_i]),
            'exit_time': str(ts[exit_i]),
            'side': side,
            'score': float(s.get('score', np.nan)),
            'score_percentile': float(s.get('score_percentile', np.nan)),
            'signal_close_reference': float(s.get('entry_price', np.nan)),
            'intended_fill_open': raw_open,
            'actual_entry_fill': entry,
            'actual_exit_fill': float(exit_px),
            'fill_latency_bars': int(latency),
            'slippage_per_side': float(slip),
            'filled_fraction': float(filled_fraction),
            'risk_budget_usd': float(risk_budget),
            'continuous_notional_usd': float(notional),
            'continuous_btc_quantity': float(quantity_btc),
            'reason': reason,
            'gross_return': float(gross),
            'net_return': float(net_return),
            'net_r': float(net_r),
            'equity_before': float(eq_before),
            'equity_after': float(eq),
            'day_loss_before_entry': float(day_loss),
            'same_bar_ambiguity': bool(ambiguity),
        })
        next_free = int(exit_i) + 1

    d = pd.DataFrame(rows)
    if d.empty:
        m = {'trades': 0, 'filled_trades': 0, 'rejected_trades': rej,
             'partial_fill_fraction': partial_fraction, 'rejection_rate': rejection_rate,
             'equity_return_pct': 0.0, 'total_net_return': 0.0,
             'win_rate_pct': float('nan'), 'profit_factor': float('nan'), 'max_drawdown_return': 0.0}
        return d, m
    r = d['net_return'].to_numpy(float)
    eq_curve = np.cumprod(1.0 + RISK_PER_TRADE * d['net_r'].to_numpy(float))
    dd = eq_curve / np.maximum.accumulate(eq_curve) - 1.0
    gain = float(r[r>0].sum()); loss = float(-r[r<0].sum())
    m = {
        'trades': int(len(d)), 'filled_trades': int(len(d)), 'rejected_trades': int(rej),
        'partial_fill_fraction': float(partial_fraction), 'rejection_rate': float(rejection_rate),
        'equity_return_pct': float((eq_curve[-1]-1.0)*100.0),
        'total_net_return': float(r.sum()), 'mean_net_return': float(r.mean()),
        'win_rate_pct': float((r>0).mean()*100.0),
        'profit_factor': float(gain/loss) if loss>0 else float('inf'),
        'max_drawdown_return': float(dd.min()),
        'tp_count': int(d['reason'].astype(str).str.startswith('TP').sum()),
        'sl_count': int(d['reason'].astype(str).str.startswith('SL').sum()),
        'time_count': int((d['reason']=='TIME').sum()),
        'ambiguous_count': int(d['same_bar_ambiguity'].sum()),
        'end_equity': float(eq_curve[-1]*INITIAL_EQUITY),
    }
    return d, m


def run_grid():
    test, future = load_signals(); ohlc = load_ohlc()
    results=[]
    configs=[]
    for lat in LATENCIES:
        for slip in SLIPPAGES:
            configs.append((lat, slip, 1.0, 0.0, 0))
    # Separate fill-friction sensitivity at a fixed 1-bar latency and 0.02% slippage/side.
    for frac in PARTIAL_FILL_FRACTIONS:
        for rr in REJECTION_RATES:
            configs.append((1, 0.0002, frac, rr, int(7000 + frac*100 + rr*1000)))

    for lat, slip, frac, rr, seed in configs:
        d, m = simulate(test, ohlc, lat, slip, frac, rr, seed)
        results.append({'sample':'test','latency_bars':lat,'slippage_per_side_pct':slip*100,'partial_fill_fraction':frac,'rejection_rate_pct':rr*100,**m})
        d.to_csv(OUT / f'test_lat{lat}_slip{slip:.4f}_frac{frac:.2f}_rej{rr:.2f}.csv', index=False)

    # Reserved future: only the main realistic scenarios, not tuning.
    for lat, slip in [(1,0.0002),(2,0.0002),(1,0.0005),(2,0.0005),(1,0.0010)]:
        d, m = simulate(future, ohlc, lat, slip, 1.0, 0.0, 9031 + lat*10 + int(slip*1e6))
        results.append({'sample':'future','latency_bars':lat,'slippage_per_side_pct':slip*100,'partial_fill_fraction':1.0,'rejection_rate_pct':0.0,**m})
        d.to_csv(OUT / f'future_lat{lat}_slip{slip:.4f}.csv', index=False)

    df = pd.DataFrame(results)
    df.to_csv(OUT/'v11_execution_results.csv', index=False)

    # Baseline comparison uses original V9.3 metrics from the stored trade streams.
    baseline_rows=[]
    for name, sig in [('test',test),('future',future)]:
        r=sig.net_return.to_numpy(float)
        eq=np.cumprod(1+RISK_PER_TRADE*sig.net_r.to_numpy(float))
        gain=float(r[r>0].sum()); loss=float(-r[r<0].sum())
        baseline_rows.append({'sample':name,'latency_bars':0,'slippage_per_side_pct':0.0,
            'partial_fill_fraction':1.0,'rejection_rate_pct':0.0,'trades':len(sig),
            'equity_return_pct':float((eq[-1]-1)*100),'total_net_return':float(r.sum()),
            'win_rate_pct':float((r>0).mean()*100),'profit_factor':float(gain/loss) if loss else float('inf')})
    base=pd.DataFrame(baseline_rows); base.to_csv(OUT/'v11_baseline_v93.csv',index=False)

    summary={
      'version':'V11',
      'base':'V9.3 frozen trade candidates',
      'purpose':'execution-realism paper simulator',
      'assumptions':{'initial_equity':INITIAL_EQUITY,'risk_per_trade':RISK_PER_TRADE,'stop_pct':STOP_PCT,'target_pct':TARGET_PCT,'round_trip_cost':COST_RT,'daily_loss_limit':DAILY_LOSS_LIMIT,'max_hold_bars':MAX_HOLD_BARS},
      'important_limitations':['No bid/ask history is present; spread is therefore not modeled in the primary run.','No exchange-specific quantity lot/contract rules are inferred.','Partial-fill/rejection scenarios are sensitivity assumptions, not observed exchange behavior.','Live API/orders are not used.'],
      'baseline':base.to_dict(orient='records'),
      'results':df.to_dict(orient='records'),
      'status':'RESEARCH_ONLY — PAPER EXECUTION SIMULATION, NO LIVE ORDERS',
    }
    (OUT/'v11_summary.json').write_text(json.dumps(summary,indent=2))
    return df, base

if __name__=='__main__':
    df, base=run_grid()
    print('=== V11 PAPER EXECUTION ===')
    print('\nBaseline V9.3:')
    print(base.to_string(index=False))
    print('\nTest latency/slippage:')
    print(df[(df['sample']=='test') & (df['partial_fill_fraction']==1.0) & (df['rejection_rate_pct']==0)].to_string(index=False))
    print('\nFuture main scenarios:')
    print(df[df['sample']=='future'].to_string(index=False))
