"""Step 15. Clustering-aware inference (cells are nested in 13 domains and share one fitted model per fold).
For every configuration:
  * domain-weighted summaries (mean of 13 domain medians) next to cell-pooled medians;
  * paired test of projection on the 13 domain medians (exact Wilcoxon signed-rank);
  * mixed-effects model on the per-cell log error ratio log(RMSE_q / RMSE) with a random intercept per domain;
  * domain-cluster bootstrap 95% CI of the median relative reduction and of the median removed share;
  * Clopper-Pearson 95% CI of the bound-validity rate per configuration (387 cells).
Holm correction within each family (offline projection, online projection, reference-check placements)."""
import glob, warnings, numpy as np, pandas as pd
from scipy.stats import wilcoxon, beta
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from common import OUT, SEED
warnings.filterwarnings('ignore')
CFGS = ['gbr_self', 'gbr_raw', 'ridge_self', 'ridge_raw', 'mlp_self', 'mlp_raw', 'gbrtuned_self', 'gbrfull_self', 'cnn_self', 'pinn', 'pinnpub']
CFGS = [c for c in CFGS if glob.glob(f'{OUT}/cells_{c}.csv')]
B = 2000; rng = np.random.default_rng(SEED)


def cp(k, n, a=0.05):
    return (beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0, beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0)


def mixed(df, col):
    d = df.assign(r=np.log(df[col]))
    m = smf.mixedlm('r ~ 1', d, groups=d['dom']).fit(reml=True)
    return float(m.params['Intercept']), float(m.bse['Intercept']), float(m.pvalues['Intercept'])


def cluster_boot(df, stat):
    doms = df.dom.unique(); g = {d: df[df.dom == d] for d in doms}; out = []
    for _ in range(B):
        out.append(stat(pd.concat([g[d] for d in rng.choice(doms, len(doms))])))
    return np.percentile(out, [2.5, 97.5])


rows = []
for c in CFGS:
    C = pd.read_csv(f'{OUT}/cells_{c}.csv').assign(ratio=lambda d: d.rmse_q / d.rmse, removed=lambda d: d.LB ** 2 / d.rmse ** 2)
    dm = C.groupby('dom').agg(rmse=('rmse', 'median'), rmse_q=('rmse_q', 'median'), removed=('removed', 'median'))
    k = int((C.LB <= C.rmse + 1e-12).sum()); lo, hi = cp(k, len(C))
    est, se, p_mixed = mixed(C, 'ratio')
    ci_red = cluster_boot(C, lambda d: np.median(1 - d.ratio)); ci_rem = cluster_boot(C, lambda d: np.median(d.removed))
    rows.append(dict(cfg=c, cells=len(C), domains=C.dom.nunique(),
                     med_rmse_cell=C.rmse.median(), med_rmse_domainw=dm.rmse.mean(),
                     med_rmse_q_cell=C.rmse_q.median(), med_rmse_q_domainw=dm.rmse_q.mean(),
                     removed_cell=C.removed.median(), removed_domainw=dm.removed.mean(), removed_lo=ci_rem[0], removed_hi=ci_rem[1],
                     red_median=float(np.median(1 - C.ratio)), red_lo=ci_red[0], red_hi=ci_red[1],
                     domains_improved=int((dm.rmse_q < dm.rmse).sum()),
                     p_domain_wilcoxon=float(wilcoxon(dm.rmse_q, dm.rmse, method='exact').pvalue),
                     mixed_log_ratio=est, mixed_se=se, p_mixed=p_mixed,
                     lb_valid=k / len(C), lb_valid_lo=lo, lb_valid_hi=hi))
T = pd.DataFrame(rows)
for col in ['p_domain_wilcoxon', 'p_mixed']: T[col + '_holm'] = multipletests(T[col], method='holm')[1]
T.to_csv(f'{OUT}/tab_stats_offline.csv', index=False); print(T.round(4).to_string(index=False))

# online, domain level (cells_online.csv from step 2, which includes every configuration present)
O = pd.read_csv(f'{OUT}/cells_online.csv'); O['dom'] = O.cell.str.split('/').str[0]
on = []
for c, g in O.groupby('cfg'):
    dm = g.groupby('dom')[['rmse_online', 'rmse_online_q']].median()
    est, se, p = mixed(g.assign(ratio=g.rmse_online_q / g.rmse_online), 'ratio')
    on.append(dict(cfg=c, med_online_cell=g.rmse_online.median(), med_online_q_cell=g.rmse_online_q.median(),
                   med_online_domainw=dm.rmse_online.mean(), med_online_q_domainw=dm.rmse_online_q.mean(),
                   domains_improved=int((dm.rmse_online_q < dm.rmse_online).sum()),
                   p_domain_wilcoxon=float(wilcoxon(dm.rmse_online_q, dm.rmse_online, method='exact').pvalue),
                   mixed_log_ratio=est, p_mixed=p))
O2 = pd.DataFrame(on)
for col in ['p_domain_wilcoxon', 'p_mixed']: O2[col + '_holm'] = multipletests(O2[col], method='holm')[1]
O2.to_csv(f'{OUT}/tab_stats_online.csv', index=False); print(O2.round(4).to_string(index=False))

# reference-check placements, domain level, Holm over K x placement
R = pd.read_csv(f'{OUT}/cells_checks.csv'); ck = []
for K in range(1, 6):
    for nm in ['fade', 'departure']:
        dm = R.groupby('dom')[[f'{nm}{K}', f'periodic{K}']].median()
        est, se, p = mixed(R.assign(ratio=R[f'{nm}{K}'] / R[f'periodic{K}']), 'ratio')
        ck.append(dict(K=K, placement=nm, domains_better=int((dm[f'{nm}{K}'] < dm[f'periodic{K}']).sum()),
                       p_domain_wilcoxon=float(wilcoxon(dm[f'{nm}{K}'], dm[f'periodic{K}'], method='exact').pvalue),
                       mixed_log_ratio=est, p_mixed=p))
CK = pd.DataFrame(ck)
for col in ['p_domain_wilcoxon', 'p_mixed']: CK[col + '_holm'] = multipletests(CK[col], method='holm')[1]
CK.to_csv(f'{OUT}/tab_stats_checks.csv', index=False); print(CK.round(4).to_string(index=False))
