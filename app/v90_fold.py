from __future__ import annotations
import argparse, json, gc
from pathlib import Path
import numpy as np
import pandas as pd
from app.v827_train import load_dataset, build_features, make_targets, make_walk_forward_folds, train_fold_models, execute_policy, metrics, bootstrap_mean_ci, PURGE, RANDOM_SEED, LOCKED_POLICY
from app.delta_ohlc import load_ohlc, validate_ohlc
from app.v90_core import OnlineFailureLearner, LearnerConfig, CONFIGS, execute_adaptive

CFG = None

def audit(t, lo, hi):
    if t.empty:
        return {'sequential': True, 'target_in_split': True, 'nan_rows': 0}
    return {'sequential': bool((t.entry_index.to_numpy()[1:] > t.exit_index.to_numpy()[:-1]).all()),
            'target_in_split': bool((t.entry_index >= lo).all() and (t.exit_index < hi).all()),
            'nan_rows': int(t.isna().any(axis=1).sum())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--ohlc',required=True); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--config',choices=[c.name for c in CONFIGS if c.name != 'OFF'],default='BALANCED'); ap.add_argument('--output-dir',default='logs/v90/folds'); a=ap.parse_args(); global CFG; CFG=next(c for c in CONFIGS if c.name==a.config)
    od=Path(a.output_dir); od.mkdir(parents=True, exist_ok=True)
    df=load_dataset(a.data); o=load_ohlc(a.ohlc); oi=validate_ohlc(o); aligned=o.set_index('timestamp').reindex(df.timestamp)
    miss=int(aligned[['open','high','low','close']].isna().any(axis=1).sum())
    if miss: raise ValueError(f'OHLC missing {miss} signal timestamps')
    f,fn=build_features(df); X=f[fn].replace([np.inf,-np.inf],np.nan); targets=make_targets(df.btcusd_close.to_numpy(float)); folds=make_walk_forward_folds(len(df)); fold=folds[a.fold-1]
    print(f'=== V9.0 FOLD {a.fold} ===',flush=True)
    scores,refs=train_fold_models(X,df,targets,fold['train'][1],a.fold*100)
    vol=float(np.nanmedian(f.btcusd_vol_24.iloc[:fold['train'][1]-PURGE]))
    learner=OnlineFailureLearner(CFG.prior_strength,CFG.max_age_days)
    val,learner=execute_adaptive(df,o,f,scores,refs,*fold['validation'],learner,CFG,vol)
    base_val=execute_policy(df,o,scores,LOCKED_POLICY,*fold['validation'],refs)
    test,learner=execute_adaptive(df,o,f,scores,refs,*fold['test'],learner,CFG,vol)
    base_test=execute_policy(df,o,scores,LOCKED_POLICY,*fold['test'],refs)
    future=pd.DataFrame(); base_future=pd.DataFrame()
    if a.fold==3:
        rlo=int(len(df)*0.95)+PURGE; rhi=len(df)
        future,learner=execute_adaptive(df,o,f,scores,refs,rlo,rhi,learner,CFG,vol)
        base_future=execute_policy(df,o,scores,LOCKED_POLICY,rlo,rhi,refs)
    result={
      'fold':a.fold,
      'validation':metrics(val), 'validation_baseline':metrics(base_val),
      'test':metrics(test), 'test_baseline':metrics(base_test),
      'validation_delta':metrics(val)['portfolio_return']-metrics(base_val)['portfolio_return'],
      'test_delta':metrics(test)['portfolio_return']-metrics(base_test)['portfolio_return'],
      'validation_suppressed':int(val.attrs.get('suppressed',0)), 'test_suppressed':int(test.attrs.get('suppressed',0)),
      'validation_scaled':int(val.attrs.get('scaled',0)), 'test_scaled':int(test.attrs.get('scaled',0)),
      'audit_validation':audit(val,*fold['validation']), 'audit_test':audit(test,*fold['test']),
      'future':metrics(future) if not future.empty else None,
      'future_baseline':metrics(base_future) if not base_future.empty else None,
      'future_ci': bootstrap_mean_ci(future,RANDOM_SEED+900+a.fold) if not future.empty else [float('nan'),float('nan')],
      'learner_after_fold':learner.snapshot(),
    }
    (od/f'fold_{a.fold}_summary.json').write_text(json.dumps(result,indent=2,default=str))
    val.to_csv(od/f'fold_{a.fold}_validation_trades.csv',index=False)
    test.to_csv(od/f'fold_{a.fold}_test_trades.csv',index=False)
    if not future.empty: future.to_csv(od/f'fold_{a.fold}_future_trades.csv',index=False)
    print(json.dumps(result,indent=2,default=str),flush=True)
    del scores,refs,X,f; gc.collect()

if __name__=='__main__': main()
