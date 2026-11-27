"""Step 19. Additional diagnostics (Supplementary Material).
(a) Data provenance: rows per public file before/after removal of missing or infinite values.
(b) SOH reference sensitivity: main configuration (GBR, self-referenced) re-trained with SOH = capacity / nominal
    capacity (XJTU 2.0 Ah, HUST 1.1 Ah, MIT 1.1 Ah, TJU 3.5/3.5/2.5 Ah, as in the source repository) instead of the
    median of the first five cycles; certificate metrics recomputed.
(c) MLP training diagnostics: iterations used and convergence warnings for every fold of both MLP configurations.
(d) GMP surface with autocorrelated error (AR(1), phi = 0.9) next to the independent-noise surface."""
import warnings, numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from common import load_cells, cell_rows, raw_row_counts, certificate, OUT, DATA, SEED
D, F = load_cells(); XS = ['d_' + c for c in F]; doms = sorted(D.dom.unique())

# (a)
R = raw_row_counts(); R.to_csv(f'{OUT}/data_provenance.csv', index=False)
print('rows raw %d kept %d removed %d in %d files' % (R.rows_raw.sum(), R.rows_kept.sum(), R.removed.sum(), int((R.removed > 0).sum())))

# (b)
NOM = {'XJTU': 2.0, 'HUST': 1.1, 'MIT': 1.1, 'TJU-Dataset_1': 3.5, 'TJU-Dataset_2': 3.5, 'TJU-Dataset_3': 2.5}
D['soh_nom'] = D.capacity / D.dom.map(lambda d: NOM.get(d, NOM.get(d.split('-')[0])))


def fold_nom(tgt):
    warnings.filterwarnings('ignore')
    tr = D[D.dom != tgt].iloc[::6]; te = D[D.dom == tgt]
    mu, sd = tr[XS].mean(), tr[XS].std() + 1e-12
    p = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=SEED).fit((tr[XS] - mu) / sd, tr.soh_nom).predict((te[XS] - mu) / sd)
    return cell_rows(te[['cell', 'dataset', 'k']].assign(soh=te.soh_nom.values, p=p), tgt)


C = pd.DataFrame([r for rs in Parallel(n_jobs=2)(delayed(fold_nom)(t) for t in doms) for r in rs]); C.to_csv(f'{OUT}/cells_gbr_self_nominal.csv', index=False)
B = pd.read_csv(f'{OUT}/cells_gbr_self.csv')
sens = pd.DataFrame([dict(soh_reference=nm, med_rmse=d.rmse.median(), med_rmse_q=d.rmse_q.median(), improved=(d.rmse_q <= d.rmse + 1e-12).mean(),
                          lb_valid=(d.LB <= d.rmse + 1e-12).mean(), removed=np.median(d.LB ** 2 / d.rmse ** 2),
                          max_initial_soh=D.groupby('cell').soh_nom.first().max() if nm == 'nominal' else 1.0)
                     for nm, d in [('first five cycles', B), ('nominal', C)]])
sens.to_csv(f'{OUT}/tab_soh_reference.csv', index=False); print(sens.round(4).to_string(index=False))

# (c)
diag = []
for ref in ['raw', 'self']:
    X = F if ref == 'raw' else XS
    for tgt in doms:
        tr = D[D.dom != tgt].iloc[::6]; mu, sd = tr[X].mean(), tr[X].std() + 1e-12
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', ConvergenceWarning)
            m = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300, early_stopping=True, n_iter_no_change=10, random_state=SEED).fit((tr[X] - mu) / sd, tr.soh)
        diag.append(dict(cfg=f'mlp_{ref}', dom=tgt, n_iter=m.n_iter_, converged_warning=any(issubclass(x.category, ConvergenceWarning) for x in w),
                         best_val_score=m.best_validation_score_))
G = pd.DataFrame(diag); G.to_csv(f'{OUT}/tab_mlp_diag.csv', index=False)
print(G.groupby('cfg').agg(n_iter_min=('n_iter', 'min'), n_iter_max=('n_iter', 'max'), warnings=('converged_warning', 'sum')).to_string())

# (d)
G0 = pd.read_csv(f'{DATA}/GMP.csv'); y = (G0['Capacity(Ah)'] / G0['Capacity(Ah)'].iloc[:5].median()).values
rng = np.random.default_rng(SEED); surf = []
for b in np.linspace(0, 0.05, 11):
    for s in np.linspace(0.0025, 0.03, 12):
        for rep in range(5):
            e = np.zeros(len(y)); z = rng.normal(0, s * np.sqrt(1 - 0.81), len(y)); e[0] = rng.normal(0, s)
            for t in range(1, len(y)): e[t] = 0.9 * e[t - 1] + z[t]
            cc = certificate(y + b + e, y); surf.append(dict(bias=b, noise_sd=s, rep=rep, rmse=cc['rmse'], LB=cc['LB']))
S = pd.DataFrame(surf).groupby(['bias', 'noise_sd'])[['rmse', 'LB']].mean().reset_index(); S['removed'] = S.LB ** 2 / S.rmse ** 2
S.to_csv(f'{OUT}/gmp_surface_ar1.csv', index=False); print('AR(1) surface removed share %.3f-%.3f' % (S.removed.min(), S.removed.max()))
