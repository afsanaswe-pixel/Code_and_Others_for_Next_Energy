"""Step 5. Sparse reference capacity checks (RPTs) combined with the projection.
For K checks per cell the residual (y - p) is measured at K cycles, linearly interpolated
(constant for K = 1) and added to p before projection; error is evaluated on non-check cycles.
Placements compared at equal K: periodic (equal cycle spacing), equal predicted-fade spacing
(label-free), largest local departure from monotonicity (label-free)."""
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from common import project, rms, OUT
P = pd.read_csv(f'{OUT}/pred_gbr_self.csv.gz')


def corrected(p, idx, y):
    if len(idx) == 0: return project(p)
    r = y[idx] - p[idx]
    c = np.full(len(p), r[0]) if len(idx) == 1 else np.interp(np.arange(len(p)), idx, r)
    return project(p + c)


rows = []
for c, g in P.groupby('cell', sort=False):
    p, y = g.p.values, g.soh.values; n = len(p); q0 = project(p)
    rec = dict(cell=c, dom=c.split('/')[0], K0=rms(q0 - y), base=rms(p - y))
    for K in range(1, 6):
        place = {'periodic': np.unique(np.linspace(0, n - 1, K + 2)[1:-1].astype(int))}
        fade = 1 - q0; tf = np.linspace(fade.min(), fade.max(), K + 2)[1:-1]
        place['fade'] = np.unique([int(np.argmin(np.abs(fade - f))) for f in tf])
        dev = pd.Series(np.abs(p - q0)).rolling(max(5, n // 20), center=True, min_periods=1).mean().values.copy()
        sel = []
        for _ in range(K):
            j = int(np.argmax(dev)); sel.append(j); dev[max(0, j - n // (K + 1)):j + n // (K + 1)] = -1
        place['departure'] = np.unique(sel)
        for nm, idx in place.items():
            idx = np.sort(idx); m = np.ones(n, bool); m[idx] = False
            rec[f'{nm}{K}'] = rms(corrected(p, idx, y)[m] - y[m])
    rows.append(rec)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/cells_checks.csv', index=False)
summ = []
for K in range(0, 6):
    if K == 0:
        summ.append(dict(K=0, placement='none', median=R.K0.median(), mean=R.K0.mean())); continue
    for nm in ['periodic', 'fade', 'departure']:
        col = f'{nm}{K}'
        d = dict(K=K, placement=nm, median=R[col].median(), mean=R[col].mean())
        if nm != 'periodic':
            d['better_than_periodic'] = float((R[col] < R[f'periodic{K}']).mean())
            d['wilcoxon_p_vs_periodic'] = float(wilcoxon(R[col], R[f'periodic{K}']).pvalue)
        summ.append(d)
S = pd.DataFrame(summ); S.to_csv(f'{OUT}/tab_checks.csv', index=False); print(S.round(4).to_string(index=False))
print('base median %.4f mean %.4f' % (R.base.median(), R.base.mean()))
