"""Step 11. Generalizability.
(a) Leave-one-dataset-out (LODSO): train on three of the four datasets (XJTU, TJU, HUST, MIT) and test on
    all cells of the fourth - a harder shift than leave-one-domain-out (new laboratory and protocol family).
(b) Training-domain diversity: for every target domain, train on k randomly chosen other domains
    (k = 1, 2, 4, 8, 12; 3 draws) and record unseen-domain error and the certificate quantities.
Main configuration (GBR) with self-referenced and raw features."""
import warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from common import load_cells, certificate, OUT, SEED, QUICK
warnings.filterwarnings('ignore')
D, F = load_cells(); doms = sorted(D.dom.unique())


def run(train_mask, test_mask, ref, seed=SEED):
    X = F if ref == 'raw' else ['d_' + c for c in F]
    tr = D[train_mask].iloc[::6]; te = D[test_mask]
    mu, sd = tr[X].mean(), tr[X].std() + 1e-12
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=seed).fit((tr[X] - mu) / sd, tr.soh)
    te = te.assign(p=m.predict((te[X] - mu) / sd)); rows = []
    for c, g in te.groupby('cell', sort=False):
        ce = certificate(g.p.values, g.soh.values)
        rows.append(dict(cell=c, dom=g.dom.iloc[0], dataset=g.dataset.iloc[0], rmse=ce['rmse'], rmse_q=ce['rmse_q'], LB=ce['LB']))
    return rows


# (a) leave-one-dataset-out
out = []
for ref in ['self', 'raw']:
    res = Parallel(n_jobs=2)(delayed(run)(D.dataset != ds, D.dataset == ds, ref) for ds in sorted(D.dataset.unique()))
    for r in res:
        for x in r: out.append(dict(ref=ref, **x))
L = pd.DataFrame(out); L.to_csv(f'{OUT}/gen_lodso_cells.csv', index=False)
LS = L.groupby(['ref', 'dataset']).agg(cells=('cell', 'size'), med_rmse=('rmse', 'median'), med_rmse_q=('rmse_q', 'median'),
                                       improved=('rmse_q', lambda s: np.nan), lb_valid=('LB', lambda s: np.nan)).reset_index()
LS['improved'] = L.assign(i=L.rmse_q <= L.rmse + 1e-12).groupby(['ref', 'dataset']).i.mean().values
LS['lb_valid'] = L.assign(i=L.LB <= L.rmse + 1e-12).groupby(['ref', 'dataset']).i.mean().values
# compare with leave-one-domain-out on the same cells
for ref in ['self', 'raw']:
    lodo = pd.read_csv(f'{OUT}/cells_gbr_{ref}.csv')
    for ds in LS[LS.ref == ref].dataset:
        LS.loc[(LS.ref == ref) & (LS.dataset == ds), 'med_rmse_lodo'] = lodo[lodo.dataset == ds].rmse.median()
LS.to_csv(f'{OUT}/tab_lodso.csv', index=False); print(LS.round(4).to_string(index=False))

# (b) number of training domains
KS = [1, 2, 4, 8, 12]; DRAWS = 1 if QUICK else 3; rng = np.random.default_rng(SEED); jobs = []
for tgt in doms:
    others = [d for d in doms if d != tgt]
    for k in KS:
        for r in range(DRAWS if k < 12 else 1):
            src = list(rng.choice(others, k, replace=False)); jobs.append((tgt, k, r, src))
def job(tgt, k, r, src):
    rows = run(D.dom.isin(src), D.dom == tgt, 'self')
    C = pd.DataFrame(rows)
    return dict(target=tgt, k=k, draw=r, med_rmse=C.rmse.median(), med_rmse_q=C.rmse_q.median(),
                lb_valid=(C.LB <= C.rmse + 1e-12).mean(), removed=np.median(C.LB ** 2 / C.rmse ** 2))
K = pd.DataFrame(Parallel(n_jobs=2)(delayed(job)(*j) for j in jobs)); K.to_csv(f'{OUT}/gen_ndomains.csv', index=False)
KS_ = K.groupby('k').agg(med_rmse=('med_rmse', 'median'), q25=('med_rmse', lambda s: s.quantile(.25)), q75=('med_rmse', lambda s: s.quantile(.75)),
                         med_rmse_q=('med_rmse_q', 'median'), lb_valid=('lb_valid', 'mean'), removed=('removed', 'median')).reset_index()
KS_.to_csv(f'{OUT}/tab_ndomains.csv', index=False); print(KS_.round(4).to_string(index=False))
