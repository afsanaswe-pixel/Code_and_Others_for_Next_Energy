"""Step 17. Conformal error bounds that do not assume exchangeability of cells across domains.
Same nested leave-one-domain-out set-up and signals as step 4 (meta-model GBR on log signals -> log RMSE,
conformity score = log RMSE - prediction, scores from an inner leave-one-domain-out on the 12 source domains).
Variants (target alpha = 0.10):
  split          : pooled split conformal (step 4, repeated for comparison);
  weighted       : weighted conformal under covariate shift (Tibshirani et al., 2019); likelihood-ratio weights from a
                   logistic domain classifier that separates source cells from target cells on the label-free signals;
  family         : group-conditional calibration with scores from source domains of the same dataset family
                   (XJTU, TJU, MIT; metadata known at deployment); pooled scores when fewer than two such domains exist;
  hierarchical   : domain-level calibration: the 90% quantile of scores inside each source domain, then the
                   ceil(0.9 (m + 1))-th smallest of these m = 12 values (two-layer hierarchical conformal prediction).
Per-domain coverage with Clopper-Pearson 95% intervals; width = median ratio of the bound to the realised RMSE."""
import numpy as np, pandas as pd, warnings
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import OUT
warnings.filterwarnings('ignore')
C = pd.read_csv(f'{OUT}/cells_gbr_self.csv')
C = C.assign(lLB=np.log(C.LB + 1e-6), lens=np.log(C.ens), lknn=np.log(C.knn), y=np.log(C.rmse))
FE = ['lLB', 'lens', 'lknn', 'rebound', 'rough']; A = 0.10
mk = lambda: HistGradientBoostingRegressor(max_iter=150, max_depth=3, learning_rate=0.05, min_samples_leaf=10)


def wquant(s, w, wt, a):
    """Weighted (1-a) quantile of scores s with weights w and test weight wt (mass at +inf)."""
    o = np.argsort(s); s, w = s[o], w[o]; cw = np.cumsum(w) / (w.sum() + wt)
    i = np.searchsorted(cw, 1 - a)
    return s[i] if i < len(s) else np.inf


out = []
for t in sorted(C.dom.unique()):
    tr, te = C[C.dom != t], C[C.dom == t]
    sc, sdom = [], []
    for u in tr.dom.unique():
        a, b = tr[tr.dom != u], tr[tr.dom == u]
        sc += list(b.y - mk().fit(a[FE], a.y).predict(b[FE])); sdom += [u] * len(b)
    sc, sdom = np.array(sc), np.array(sdom)
    pr = mk().fit(tr[FE], tr.y).predict(te[FE]); n = len(sc)
    q_split = np.quantile(sc, min(1, np.ceil((1 - A) * (n + 1)) / n))
    # weighted
    sx = StandardScaler().fit(pd.concat([tr[FE], te[FE]]))
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(sx.transform(pd.concat([tr[FE], te[FE]])), np.r_[np.zeros(len(tr)), np.ones(len(te))])
    lr = lambda X: (lambda p: p / (1 - p))(np.clip(clf.predict_proba(sx.transform(X))[:, 1], 1e-4, 1 - 1e-4)) * len(tr) / len(te)
    w_src = lr(tr[FE]); w_te = lr(te[FE]); q_w = np.array([wquant(sc, w_src, wt, A) for wt in w_te])
    # family-conditional
    fam = t.split('-')[0]; same = np.array([d.split('-')[0] == fam for d in sdom])
    sf = sc[same] if len(set(sdom[same])) >= 2 else sc
    q_f = np.quantile(sf, min(1, np.ceil((1 - A) * (len(sf) + 1)) / len(sf)))
    # hierarchical (domain-level)
    dq = np.sort([np.quantile(sc[sdom == u], 1 - A) for u in np.unique(sdom)]); m = len(dq)
    k = int(np.ceil((1 - A) * (m + 1))); q_h = dq[k - 1] if k <= m else np.inf
    out.append(te.assign(pred=np.exp(pr), UB_split=np.exp(pr + q_split), UB_weighted=np.exp(pr + q_w),
                         UB_family=np.exp(pr + q_f), UB_hier=np.exp(pr + q_h), wmax=float(np.max(w_te / (w_src.sum() + w_te)))))
O = pd.concat(out); O.to_csv(f'{OUT}/cells_conformal.csv', index=False)


def cp(k, n):
    return beta.ppf(.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(.975, k + 1, n - k) if k < n else 1.0


rows, per = [], []
for v in ['split', 'weighted', 'family', 'hier']:
    cov = O.rmse <= O[f'UB_{v}']; cd = cov.groupby(O.dom).mean()
    rows.append(dict(variant=v, coverage=cov.mean(), domain_mean_coverage=cd.mean(), min_domain=cd.min(), min_domain_name=cd.idxmin(),
                     domains_below_80=int((cd < 0.8).sum()), width_median=float(np.median(np.minimum(O[f'UB_{v}'], 10) / O.rmse)),
                     infinite_bounds=float(np.isinf(O[f'UB_{v}']).mean())))
    for d, g in O.groupby('dom'):
        k = int((g.rmse <= g[f'UB_{v}']).sum()); lo, hi = cp(k, len(g))
        per.append(dict(variant=v, dom=d, cells=len(g), coverage=k / len(g), lo=lo, hi=hi))
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/tab_conformal.csv', index=False); print(T.round(3).to_string(index=False))
Pd = pd.DataFrame(per); Pd.to_csv(f'{OUT}/tab_conformal_domains.csv', index=False)
print(Pd.pivot(index='dom', columns='variant', values='coverage').round(2).to_string())
