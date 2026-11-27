"""Step 10. Explainability of the SOH estimator and of its cross-domain error (TreeSHAP).
For each leave-one-domain-out fold (main configuration: GBR, self-referenced features):
  (a) global importance = mean |SHAP| on the held-out domain; stability of the feature ranking across
      the 13 fold models (Kendall's W, mean pairwise Spearman).
  (b) bias attribution: by SHAP additivity, mean prediction on target minus mean prediction on source
      equals the sum over features of (mean SHAP on target - mean SHAP on source). Each feature's term is
      its contribution to the domain-level shift of the prediction. We compare the total shift with the
      domain's measured bias and identify which features carry it.
  (c) error-consistency: correlation between |SHAP shift| of a feature and the feature's standardised
      mean difference between source and target (does the explanation point to the shifted inputs?)."""
import warnings, numpy as np, pandas as pd, shap
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from common import load_cells, OUT, SEED, QUICK
warnings.filterwarnings('ignore')
D, F = load_cells(); X = ['d_' + c for c in F]; doms = sorted(D.dom.unique())
NS = 600 if QUICK else 2000
rng = np.random.default_rng(SEED)
imp, att, rows = {}, {}, []
for tgt in doms:
    tr = D[D.dom != tgt].iloc[::6]; te = D[D.dom == tgt]
    mu, sd = tr[X].mean(), tr[X].std() + 1e-12
    Ztr, Zte = (tr[X] - mu) / sd, (te[X] - mu) / sd
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=SEED).fit(Ztr, tr.soh)
    ex = shap.TreeExplainer(m)
    s_te = ex.shap_values(Zte.iloc[rng.choice(len(Zte), min(NS, len(Zte)), replace=False)])
    s_tr = ex.shap_values(Ztr.iloc[rng.choice(len(Ztr), NS, replace=False)])
    imp[tgt] = np.abs(s_te).mean(0)
    shift = s_te.mean(0) - s_tr.mean(0); att[tgt] = shift
    p_te = m.predict(Zte); bias = float(np.mean(p_te - te.soh.values))
    smd = (te[X].mean() - tr[X].mean()) / (tr[X].std() + 1e-12)
    rows.append(dict(dom=tgt, bias=bias, pred_shift=float(p_te.mean() - m.predict(Ztr).mean()), shap_shift_sum=float(shift.sum()),
                     true_shift=float(te.soh.mean() - tr.soh.mean()),
                     top_feature=F[int(np.argmax(np.abs(shift)))], top_share=float(np.abs(shift).max() / np.abs(shift).sum()),
                     rho_shift_smd=spearmanr(np.abs(shift), np.abs(smd.values)).correlation))
    print(rows[-1], flush=True)
I = pd.DataFrame(imp, index=F).T; I.to_csv(f'{OUT}/xai_importance.csv')
A = pd.DataFrame(att, index=F).T; A.to_csv(f'{OUT}/xai_shift.csv')
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/xai_domains.csv', index=False)
# ranking stability across fold models
ranks = I.rank(axis=1, ascending=False).values; m_, n_ = ranks.shape
Rj = ranks.sum(0); W = 12 * np.sum((Rj - Rj.mean()) ** 2) / (m_ ** 2 * (n_ ** 3 - n_))
pair = [spearmanr(I.iloc[a], I.iloc[b]).correlation for a, b in combinations(range(m_), 2)]
top = I.mean().sort_values(ascending=False)
# share of total importance in top 3 features; agreement of top-3 sets across folds
top3 = [set(I.columns[np.argsort(-I.iloc[i].values)[:3]]) for i in range(m_)]
jac = np.mean([len(a & b) / len(a | b) for a, b in combinations(top3, 2)])
summ = dict(kendall_W=W, mean_pairwise_spearman=float(np.mean(pair)), min_pairwise_spearman=float(np.min(pair)),
            top3_jaccard=float(jac), top_features=';'.join(top.index[:5]),
            rho_bias_vs_predshift=spearmanr(R.bias, R.pred_shift - R.true_shift).correlation,
            median_rho_shift_smd=float(R.rho_shift_smd.median()))
pd.DataFrame([summ]).to_csv(f'{OUT}/tab_xai.csv', index=False)
print(pd.Series(summ)); print(top.round(4).to_string()); print(R[['dom', 'bias', 'top_feature', 'top_share', 'rho_shift_smd']].round(3).to_string(index=False))
