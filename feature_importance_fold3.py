from pathlib import Path
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRanker
from app.v827_train import load_dataset, make_targets, make_walk_forward_folds, sample_rank_training_indices, PURGE
from app.v923_train import build_price_action_features
from app.delta_ohlc import load_ohlc

root=Path('/mnt/data/adaptive_trading_ai_v9_3_price_action')
df=load_dataset(root/'data/processed/v821_common_multiasset.csv.gz')
o=load_ohlc(root/'data/processed/btcusd_5m_ohlc.csv.gz')
f,names=build_price_action_features(df,o); X=f[names].replace([np.inf,-np.inf],np.nan); folds=make_walk_forward_folds(len(df)); train_hi=folds[2]['train'][1]; train_end=train_hi-PURGE; idx0=np.arange(0,train_end,dtype=np.int64); day=df.timestamp.dt.floor('D').astype(str).to_numpy(); imp=SimpleImputer(strategy='median'); Xtr=imp.fit_transform(X.iloc[:train_end]).astype(np.float32); tar=make_targets(df.btcusd_close.to_numpy(float));
out=[]
for side,seed in [('L',8561),('S',8562)]:
 idx,groups=sample_rank_training_indices(tar[side],day,idx0); model=XGBRanker(n_estimators=80,max_depth=3,learning_rate=.05,subsample=.85,colsample_bytree=.80,reg_lambda=3.0,min_child_weight=5.0,objective='rank:pairwise',eval_metric='ndcg@10',tree_method='hist',n_jobs=4,random_state=seed); model.fit(Xtr[idx],tar[side][idx],group=groups); imps=pd.DataFrame({'feature':names,'importance':model.feature_importances_}).sort_values('importance',ascending=False); imps.to_csv(root/f'logs/v93_final/fold3_{side}_feature_importance.csv',index=False); print('\nSIDE',side,'total PA importance',imps.loc[imps.feature.str.startswith('pa_'),'importance'].sum()); print('Top PA:'); print(imps[imps.feature.str.startswith('pa_')].head(15).to_string(index=False)); print('Top all:'); print(imps.head(15).to_string(index=False))
