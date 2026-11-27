"""Step 1. Leave-one-domain-out (LODO) SOH estimation on 13 public domains.
Configurations: 3 learners (GBR, ridge, MLP) x 2 feature references (raw, self-referenced).
Main configuration (GBR, self-referenced) uses a 5-member bootstrap ensemble to give a
disagreement signal and a kNN feature-space distance. Writes:
  outputs/pred_<cfg>.csv.gz  per-cycle predictions on the held-out domain
  outputs/cells_<cfg>.csv    per-cell errors and label-free signals
"""
import warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import NearestNeighbors
from common import load_cells, certificate, rms, OUT, QUICK, SEED
warnings.filterwarnings('ignore')
D, F = load_cells()
STRIDE = 6  # training uses every 6th cycle (memory/time); evaluation uses every cycle
CFGS = [('gbr', 'self'), ('gbr', 'raw'), ('ridge', 'self'), ('ridge', 'raw'), ('mlp', 'self'), ('mlp', 'raw')]
if QUICK: CFGS = CFGS[:2]


def learner(name, s):
    if name == 'gbr': return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=s)
    if name == 'ridge': return Ridge(alpha=1.0)
    return MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, early_stopping=True, n_iter_no_change=10, random_state=s)


def fold(tgt, model, ref):
    X = F if ref == 'raw' else ['d_' + c for c in F]
    tr = D[D.dom != tgt].iloc[::STRIDE]; te = D[D.dom == tgt]
    mu, sd = tr[X].mean(), tr[X].std() + 1e-12
    Ztr, Zte = ((tr[X] - mu) / sd).values, ((te[X] - mu) / sd).values
    main = (model == 'gbr' and ref == 'self')
    n_ens = 5 if main else 1
    P = []
    for s in range(n_ens):
        idx = np.random.default_rng(SEED + s).choice(len(tr), int(0.7 * len(tr)), replace=False) if n_ens > 1 else np.arange(len(tr))
        P.append(learner(model, SEED + s).fit(Ztr[idx], tr.soh.values[idx]).predict(Zte))
    P = np.array(P); te = te[['cell', 'dom', 'dataset', 'k', 'soh']].assign(p=P.mean(0), ens=P.std(0))
    if main:
        nn = NearestNeighbors(n_neighbors=5).fit(Ztr[::10])
        sub = np.arange(0, len(te), 5)
        kd = pd.Series(nn.kneighbors(Zte[sub])[0].mean(1)).groupby(te.cell.values[sub]).mean()
    rows = []
    for c, g in te.groupby('cell', sort=False):
        p, y = g.p.values, g.soh.values
        cert = certificate(p, y)
        sm = pd.Series(p).rolling(15, min_periods=1, center=True).median().values
        inc = np.diff(sm)
        r = dict(cell=c, dom=tgt, dataset=g.dataset.iloc[0], n=len(g), rmse=cert['rmse'], rmse_q=cert['rmse_q'], LB=cert['LB'],
                 bias=float(np.mean(p - y)), rebound=float(inc[inc > 0].sum() / (np.abs(inc).sum() + 1e-12)),
                 rough=float(np.std(np.diff(p)) / (np.std(p) + 1e-12)),
                 viol=rms(y - certificate(y)['q']))
        if main: r.update(ens=float(g.ens.mean()), knn=float(kd[c]))
        rows.append(r)
    return te[['cell', 'k', 'soh', 'p']], rows


for model, ref in CFGS:
    res = Parallel(n_jobs=2)(delayed(fold)(t, model, ref) for t in sorted(D.dom.unique()))
    pd.concat([r[0] for r in res]).to_csv(f'{OUT}/pred_{model}_{ref}.csv.gz', index=False, float_format='%.6f')
    C = pd.DataFrame([x for r in res for x in r[1]]); C.to_csv(f'{OUT}/cells_{model}_{ref}.csv', index=False)
    print(f'{model:5s} {ref:4s}: median cell RMSE {C.rmse.median():.4f}  mean {C.rmse.mean():.4f}  '
          f'after projection {C.rmse_q.median():.4f}  LB<=RMSE {np.mean(C.LB <= C.rmse + 1e-12):.3f}', flush=True)
