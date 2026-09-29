from pathlib import Path
import json, numpy as np, pandas as pd
from app.v923_train import build_price_action_features, train_models
from app.v827_train import load_dataset, make_targets, make_walk_forward_folds, execute_policy, metrics, bootstrap_mean_ci, LOCKED_POLICY, PURGE, RANDOM_SEED
from app.delta_ohlc import load_ohlc, validate_ohlc

root=Path('/mnt/data/adaptive_trading_ai_v9_3_price_action')
od=root/'logs/v93'; od.mkdir(parents=True, exist_ok=True)
df=load_dataset(root/'data/processed/v821_common_multiasset.csv.gz')
o=load_ohlc(root/'data/processed/btcusd_5m_ohlc.csv.gz'); validate_ohlc(o)
aligned=o.set_index('timestamp').reindex(df.timestamp)[['open','high','low','close']]
pa_df=df.copy()
pa_df['btcusd_open']=aligned['open'].to_numpy(float); pa_df['btcusd_high']=aligned['high'].to_numpy(float); pa_df['btcusd_low']=aligned['low'].to_numpy(float); pa_df['btcusd_close_ohlc']=aligned['close'].to_numpy(float); pa_df['btcusd_close']=pa_df['btcusd_close_ohlc']
f,names=build_price_action_features(pa_df,o); X=f[names].replace([np.inf,-np.inf],np.nan); targets=make_targets(df.btcusd_close.to_numpy(float)); folds=make_walk_forward_folds(len(df)); fold=folds[2]
print('training fold3', flush=True)
scores,refs=train_models(X,df,targets,fold['train'][1],300)
rows=[]
for split in ('validation','test'):
    lo,hi=fold[split]; t=execute_policy(df,o,scores,LOCKED_POLICY,lo,hi,refs); m=metrics(t); print(split,m,flush=True); t.to_csv(od/f'fold_3_{split}_trades.csv',index=False); rows.append((split,m))
future_lo=int(len(df)*0.95)+PURGE; future_hi=len(df); ft=execute_policy(df,o,scores,LOCKED_POLICY,future_lo,future_hi,refs); fm=metrics(ft); ci=bootstrap_mean_ci(ft,RANDOM_SEED+903); ft.to_csv(od/'wf3_blind_future_trades.csv',index=False); print('future',fm,ci,flush=True)
res={'fold':3,'features':len(names),'validation':rows[0][1],'test':rows[1][1],'future':fm,'future_ci':ci}
(od/'fold_3_result.json').write_text(json.dumps(res,indent=2,default=str))
print('DONE',json.dumps(res,indent=2,default=str),flush=True)
