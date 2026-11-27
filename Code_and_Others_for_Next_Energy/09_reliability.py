"""Step 9. Reliability of the findings.
(a) Run-to-run stability: the leave-one-domain-out experiment is repeated with 5 random seeds and
    5 different training subsamples (offset of the every-6th-cycle stride) for the main configuration
    (GBR, self-referenced) and the noisiest configuration (MLP, raw). Headline quantities are recomputed.
(b) Statistical uncertainty: domain-cluster bootstrap (B = 2000) 95% CIs for the headline quantities
    of the reported run (outputs/cells_*.csv)."""
import warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from common import load_cells, certificate, OUT, SEED, QUICK
warnings.filterwarnings('ignore')
D, F = load_cells(); doms = sorted(D.dom.unique())
RUNS = [(s, s) for s in range(5)] if not QUICK else [(0, 0), (1, 1)]


def fold(tgt, model, ref, seed, off):
    X = F if ref == 'raw' else ['d_' + c for c in F]
    tr = D[D.dom != tgt].iloc[off::6]; te = D[D.dom == tgt]
    mu, sd = tr[X].mean(), tr[X].std() + 1e-12
    m = (HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=seed) if model == 'gbr'
         else MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, early_stopping=True, n_iter_no_change=10, random_state=seed))
    m.fit((tr[X] - mu) / sd, tr.soh)
    te = te.assign(p=m.predict((te[X] - mu) / sd)); rows = []
    for c, g in te.groupby('cell', sort=False):
        ce = certificate(g.p.values, g.soh.values)
        rows.append(dict(cell=c, dom=tgt, rmse=ce['rmse'], rmse_q=ce['rmse_q'], LB=ce['LB']))
    return rows


def headline(C):
    return dict(med_rmse=C.rmse.median(), med_rmse_q=C.rmse_q.median(), improved=(C.rmse_q <= C.rmse + 1e-12).mean(),
                lb_valid=(C.LB <= C.rmse + 1e-12).mean(), removed=np.median(C.LB ** 2 / C.rmse ** 2),
                rel_red=np.median(1 - C.rmse_q / C.rmse))


rows = []
for model, ref in [('gbr', 'self'), ('mlp', 'raw')]:
    for seed, off in RUNS:
        C = pd.DataFrame([r for o in Parallel(n_jobs=2)(delayed(fold)(t, model, ref, seed, off) for t in doms) for r in o])
        rows.append(dict(cfg=f'{model}_{ref}', seed=seed, offset=off, **headline(C)))
        print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/rel_runs.csv', index=False)
S = R.groupby('cfg')[['med_rmse', 'med_rmse_q', 'improved', 'lb_valid', 'removed', 'rel_red']].agg(['mean', 'std', 'min', 'max'])
S.to_csv(f'{OUT}/tab_reliability_runs.csv'); print(S.round(4).T.to_string())

# (b) domain-cluster bootstrap of the reported run
rng = np.random.default_rng(SEED); out = []
for cfg in ['gbr_self', 'gbr_raw', 'mlp_self', 'mlp_raw', 'ridge_self', 'ridge_raw']:
    try: C = pd.read_csv(f'{OUT}/cells_{cfg}.csv')
    except FileNotFoundError: continue
    est = headline(C); dd = C.dom.unique(); bt = []
    for _ in range(2000):
        pick = rng.choice(dd, len(dd), replace=True); bt.append(headline(pd.concat([C[C.dom == d] for d in pick])))
    bt = pd.DataFrame(bt)
    for k, v in est.items():
        lo, hi = np.percentile(bt[k], [2.5, 97.5]); out.append(dict(cfg=cfg, quantity=k, estimate=v, lo=lo, hi=hi))
B = pd.DataFrame(out); B.to_csv(f'{OUT}/tab_reliability_ci.csv', index=False)
print(B.pivot(index='cfg', columns='quantity', values='estimate').round(4))
