"""Step 3. Do label-free signals rank per-cell error on an unseen domain?
Main configuration (GBR, self-referenced). Signals: ensemble disagreement, kNN distance,
rebound fraction, roughness, certificate LB. Metrics: Spearman (pooled and within-domain
median) and AUROC for 'bad' cells (RMSE > 2% and > 5%), with 95% CIs from a
domain-cluster bootstrap (B = 2000)."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from common import OUT, SEED
C = pd.read_csv(f'{OUT}/cells_gbr_self.csv')
SIG = {'ens': 'Ensemble disagreement', 'knn': 'kNN feature distance', 'rebound': 'Rebound fraction',
       'rough': 'Roughness', 'LB': 'Certificate LB'}
doms = C.dom.unique(); rng = np.random.default_rng(SEED); B = 2000


def stats(df, s):
    within = np.nanmedian([spearmanr(g[s], g.rmse).correlation for _, g in df.groupby('dom') if len(g) > 5])
    out = dict(rho=spearmanr(df[s], df.rmse).correlation, rho_within=within)
    for thr in (0.02, 0.05):
        y = df.rmse > thr
        out[f'auc{int(thr*100)}'] = roc_auc_score(y, df[s]) if 0 < y.mean() < 1 else np.nan
    return out


rows = []
for s, name in SIG.items():
    est = stats(C, s); boot = []
    for _ in range(B):
        pick = rng.choice(doms, len(doms), replace=True)
        boot.append(stats(pd.concat([C[C.dom == d] for d in pick]), s))
    bt = pd.DataFrame(boot)
    r = dict(signal=name)
    for k, v in est.items():
        lo, hi = np.nanpercentile(bt[k], [2.5, 97.5]); r[k] = v; r[k + '_lo'] = lo; r[k + '_hi'] = hi
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/tab_signals.csv', index=False)
print(T.round(3).to_string(index=False))
print('bad-cell prevalence: >2pct %.3f, >5pct %.3f' % ((C.rmse > .02).mean(), (C.rmse > .05).mean()))
# error decomposition: share of squared error explained by the per-cell mean bias
C['bias_share'] = C.bias ** 2 / C.rmse ** 2
print('median share of MSE due to constant per-cell bias: %.3f' % C.bias_share.median())
pd.DataFrame({'bias_share_median': [C.bias_share.median()], 'bias_share_mean': [C.bias_share.mean()]}).to_csv(f'{OUT}/tab_bias_share.csv', index=False)
