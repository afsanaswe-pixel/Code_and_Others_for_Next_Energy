"""Step 18. Error-centred explanation and stability null for the main configuration (GBR, self-referenced).
(a) Oracle-difference decomposition of the domain bias. For each held-out domain an oracle model with the same
    hyperparameters is cross-fitted on the target domain itself (3-fold split by cell; labels used for analysis only).
    With path-dependent TreeSHAP, f(x) = phi_0 + sum_j phi_j(x) for each model, so over target cycles
        mean(f_src - f_oracle) = (phi0_src - phi0_oracle) + sum_j [mean phi_j,src - mean phi_j,oracle].
    The first term is the difference of the models' expected outputs (it reflects the different label levels of the
    training data, i.e. label shift); the feature terms show which inputs the deployed model uses differently from a
    model that knows the target domain. Since f_oracle approximates y, the left side approximates the estimator bias.
(b) Stability null: Kendall's W of the 13 leave-one-domain-out importance rankings against (i) a permutation null that
    shuffles feature labels independently per model and (ii) models trained on disjoint halves of the 13 domains
    (20 random splits), whose importance correlation measures stability across genuinely different training data."""
import warnings, numpy as np, pandas as pd, shap
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from common import load_cells, OUT, SEED, QUICK
warnings.filterwarnings('ignore')
D, F = load_cells(); X = ['d_' + c for c in F]; doms = sorted(D.dom.unique())
NS = 600 if QUICK else 1500; rng = np.random.default_rng(SEED)
mk = lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=SEED)
rows, terms = [], {}
for tgt in doms:
    tr = D[D.dom != tgt].iloc[::6]; te = D[D.dom == tgt].reset_index(drop=True)
    mu, sd = tr[X].mean(), tr[X].std() + 1e-12
    Ztr, Zte = (tr[X] - mu) / sd, (te[X] - mu) / sd
    m = mk().fit(Ztr, tr.soh)
    sub = np.sort(rng.choice(len(te), min(NS, len(te)), replace=False))
    ex = shap.TreeExplainer(m); s_src = ex.shap_values(Zte.iloc[sub]); b_src = float(np.ravel(ex.expected_value)[0])
    p_src = m.predict(Zte)
    # cross-fitted oracle on the target domain (split by cell)
    p_or = np.zeros(len(te)); s_or = np.zeros_like(s_src); b_or = np.zeros(len(sub))
    k = min(3, te.cell.nunique())
    for a, b in GroupKFold(k).split(Zte, groups=te.cell):
        mo = mk().fit(Zte.iloc[a][::2], te.soh.values[a][::2]); p_or[b] = mo.predict(Zte.iloc[b])
        eo = shap.TreeExplainer(mo); inb = np.isin(sub, b)
        if inb.any(): s_or[inb] = eo.shap_values(Zte.iloc[sub[inb]]); b_or[inb] = float(np.ravel(eo.expected_value)[0])
    dphi = s_src.mean(0) - s_or.mean(0); base = b_src - b_or.mean()
    bias = float(np.mean(p_src - te.soh.values)); gap = float(np.mean(p_src[sub] - p_or[sub]))
    terms[tgt] = dphi
    # old attribution (prediction shift source -> target) for comparison
    s_trn = ex.shap_values(Ztr.iloc[rng.choice(len(Ztr), NS, replace=False)]); shift = s_src.mean(0) - s_trn.mean(0)
    rows.append(dict(dom=tgt, cells=te.cell.nunique(), bias=bias, oracle_rmse=float(np.sqrt(np.mean((p_or - te.soh.values) ** 2))),
                     gap_src_minus_oracle=gap, base_term=base, feature_terms=float(dphi.sum()),
                     base_share=abs(base) / (abs(base) + np.abs(dphi).sum()),
                     top_error_feature=F[int(np.argmax(np.abs(dphi)))], top_error_share=float(np.abs(dphi).max() / np.abs(dphi).sum()),
                     top_shift_feature=F[int(np.argmax(np.abs(shift)))],
                     rho_error_vs_shift_attr=spearmanr(np.abs(dphi), np.abs(shift)).correlation))
    print(rows[-1], flush=True)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/xai_error_domains.csv', index=False)
pd.DataFrame(terms, index=F).T.to_csv(f'{OUT}/xai_error_terms.csv')

# (b) stability null
I = pd.read_csv(f'{OUT}/xai_importance.csv', index_col=0)


def kendall_w(M):
    r = np.argsort(np.argsort(-M, 1), 1) + 1.0; m_, n_ = r.shape; Rj = r.sum(0)
    return 12 * np.sum((Rj - Rj.mean()) ** 2) / (m_ ** 2 * (n_ ** 3 - n_))


W = kendall_w(I.values)
null = [kendall_w(np.array([rng.permutation(row) for row in I.values])) for _ in range(2000)]
half = []
for s in range(4 if QUICK else 20):
    r2 = np.random.default_rng(100 + s); perm = r2.permutation(doms); A_, B_ = perm[:6], perm[6:]
    imps = []
    for grp in (A_, B_):
        tr = D[D.dom.isin(grp)].iloc[::6]; mu, sd = tr[X].mean(), tr[X].std() + 1e-12; Z = (tr[X] - mu) / sd
        mm = mk().fit(Z, tr.soh); ev = Z.iloc[r2.choice(len(Z), NS, replace=False)]
        imps.append(np.abs(shap.TreeExplainer(mm).shap_values(ev)).mean(0))
    half.append(spearmanr(imps[0], imps[1]).correlation)
lodo_pairs = [spearmanr(I.iloc[a], I.iloc[b]).correlation for a, b in combinations(range(len(I)), 2)]
S = dict(kendall_W=W, W_null_mean=float(np.mean(null)), W_null_p=float((np.sum(np.array(null) >= W) + 1) / (len(null) + 1)),
         rho_lodo_pairs_mean=float(np.mean(lodo_pairs)), rho_disjoint_halves_mean=float(np.mean(half)),
         rho_disjoint_halves_min=float(np.min(half)), rho_disjoint_halves_max=float(np.max(half)),
         median_base_share=float(R.base_share.median()), median_oracle_rmse=float(R.oracle_rmse.median()),
         rho_bias_vs_gap=spearmanr(R.bias, R.gap_src_minus_oracle).correlation,
         top_feature_agreement=float((R.top_error_feature == R.top_shift_feature).mean()),
         median_rho_error_vs_shift=float(R.rho_error_vs_shift_attr.median()))
pd.DataFrame([S]).to_csv(f'{OUT}/tab_xai_error.csv', index=False); print(pd.Series(S).round(4))
print(R[['dom', 'bias', 'oracle_rmse', 'base_term', 'feature_terms', 'base_share', 'top_error_feature', 'top_shift_feature']].round(4).to_string(index=False))
