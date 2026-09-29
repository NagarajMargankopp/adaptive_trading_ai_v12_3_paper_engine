from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--ohlc',required=True); ap.add_argument('--output-dir',default='logs/v90'); a=ap.parse_args()
    od=Path(a.output_dir); fd=od/'folds'; fd.mkdir(parents=True,exist_ok=True)
    print('=== ADAPTIVE TRADING AI V9.0 — PAPER SELF-CORRECTING ENGINE ===')
    print('V8.27 ranker + real OHLC execution + causal online Bayesian failure learner.')
    print('The learner updates only after completed trades. It can reduce risk or suppress repeated failure contexts.')
    print('Live orders are disabled.')
    for fold in (1,2,3):
        cmd=[sys.executable,'-m','app.v90_fold','--data',a.data,'--ohlc',a.ohlc,'--fold',str(fold),'--output-dir',str(fd)]
        print(f'\n--- RUN FOLD {fold} ---',flush=True)
        subprocess.run(cmd,check=True)
    rows=[]
    for fold in (1,2,3):
        rows.append(json.loads((fd/f'fold_{fold}_summary.json').read_text()))
    test_net=sum(r['test']['total_net_return'] for r in rows); test_trades=sum(r['test']['trades'] for r in rows)
    pos=sum(r['test']['total_net_return']>0 for r in rows)
    dev_delta=sum(r['validation_delta'] for r in rows); dev_pos=sum(r['validation_delta']>0 for r in rows)
    eligible=dev_pos>=2 and dev_delta>0 and min(r['validation_delta'] for r in rows)>=-0.01
    selected='BALANCED' if eligible else 'OFF'
    future=rows[-1].get('future') or {'trades':0,'total_net_return':0.0}
    summary={'version':'V9.0','selected_correction':selected,'promotion_gate':{'positive_validation_folds':int(dev_pos),'total_validation_delta':float(dev_delta),'eligible':bool(eligible)},'test_aggregate':{'folds':3,'positive_folds':int(pos),'trades':int(test_trades),'total_net_return':float(test_net)},'blind_future':future,'walk_forward':rows,'status':'RESEARCH_ONLY — PAPER LEARNING, NO LIVE EXECUTION'}
    (od/'v90_summary.json').write_text(json.dumps(summary,indent=2,default=str))
    pd.DataFrame([{'fold':r['fold'],'validation_net':r['validation']['total_net_return'],'validation_baseline_net':r['validation_baseline']['total_net_return'],'validation_delta':r['validation_delta'],'test_net':r['test']['total_net_return'],'test_baseline_net':r['test_baseline']['total_net_return'],'test_delta':r['test_delta'],'test_trades':r['test']['trades'],'test_pf':r['test']['profit_factor'],'test_win_rate':r['test']['win_rate_pct'],'test_suppressed':r['test_suppressed'],'test_scaled':r['test_scaled']} for r in rows]).to_csv(od/'v90_walk_forward.csv',index=False)
    print('\n=== V9.0 FINAL ===')
    print(f'Selected correction: {selected}')
    print(f'Validation delta total: {dev_delta:+.6f} | positive folds: {dev_pos}/3')
    print(f'Test: {test_trades} trades | total net {test_net:+.6f} | positive folds {pos}/3')
    print(f'Blind future: {future}')
    print(f'Outputs: {od}/v90_summary.json, {od}/v90_walk_forward.csv')

if __name__=='__main__': main()
