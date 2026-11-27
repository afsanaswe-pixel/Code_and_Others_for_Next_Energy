"""Step 4. Can a learned meta-model turn label-free signals into a calibrated per-cell
error estimate on a new domain? Nested leave-one-domain-out: for each target domain the
meta-model (GBR on log signals -> log RMSE) and a split-conformal upper bound (90%) are fit
only on the remaining 12 domains (conformity scores from an inner LODO)."""
import numpy as np, pandas as pd, warnings
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from common import OUT
warnings.filterwarnings('ignore')
C = pd.read_csv(f'{OUT}/cells_gbr_self.csv')
C = C.assign(lLB=np.log(C.LB + 1e-6), lens=np.log(C.ens), lknn=np.log(C.knn), y=np.log(C.rmse))
FE = ['lLB', 'lens', 'lknn', 'rebound', 'rough']
ALPHA = 0.10
mk = {'gbr': lambda: HistGradientBoostingRegressor(max_iter=150, max_depth=3, learning_rate=0.05, min_samples_leaf=10),
      'ridge': lambda: Ridge(1.0)}
res = []
for name, m in mk.items():
    out = []
    for t in C.dom.unique():
        tr, te = C[C.dom != t], C[C.dom == t]
        scores = []
        for u in tr.dom.unique():
            a, b = tr[tr.dom != u], tr[tr.dom == u]
            scores += list(b.y - m().fit(a[FE], a.y).predict(b[FE]))
        q = np.quantile(scores, np.ceil((1 - ALPHA) * (len(scores) + 1)) / len(scores))
        pr = m().fit(tr[FE], tr.y).predict(te[FE])
        out.append(te.assign(pred=np.exp(pr), UB=np.exp(pr + q)))
    O = pd.concat(out)
    cov_dom = O.groupby('dom').apply(lambda g: (g.rmse <= g.UB).mean())
    res.append(dict(meta=name, rho=spearmanr(O.pred, O.rmse).correlation,
                    rho_within=np.nanmedian([spearmanr(g.pred, g.rmse).correlation for _, g in O.groupby('dom')]),
                    coverage=(O.rmse <= O.UB).mean(), cov_min_domain=cov_dom.min(), cov_min_dom_name=cov_dom.idxmin(),
                    n_domains_under_80=int((cov_dom < 0.8).sum()), ub_over_rmse_median=float((O.UB / O.rmse).median()),
                    mae_est=float(np.median(np.abs(O.pred - O.rmse))),
                    mae_constant=float(np.median(np.abs(C.rmse.median() - C.rmse)))))
    O.to_csv(f'{OUT}/cells_meta_{name}.csv', index=False)
    print(name, cov_dom.round(2).to_dict())
T = pd.DataFrame(res); T.to_csv(f'{OUT}/tab_meta.csv', index=False); print(T.round(3).to_string(index=False))
