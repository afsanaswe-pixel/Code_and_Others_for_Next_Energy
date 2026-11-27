"""Step 6. Known-truth and assumption-violation controls.
(a) GMP pack (Mendeley doi:10.17632/k675pxrd83.2): exact monotone ground truth.
    (a1) a learner trained on the operating channels, chronological 70/30 split;
    (a2) controlled error injection: p = y + b + e, e ~ AR(1)(phi) with sd s, on the
         true 3000-cycle trajectory, to show which error component the certificate sees.
(b) NASA B0005/6/7/18: measured capacity with regeneration (truth not monotone);
    leave-one-cell-out with non-proxy discharge features."""
import os, numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from common import certificate, project, rms, DATA, OUT, SEED
warnings.filterwarnings('ignore')
rng = np.random.default_rng(SEED)

# ---------- (a) GMP ----------
gp = os.path.join(DATA, 'GMP.csv')
if not os.path.exists(gp):  # regenerate from the published generator (Ulfat et al., Data in Brief 69 (2026) 113252; seed 42)
    np.random.seed(42); N = 3000; n = np.arange(1, N + 1); cap = 40.0 - 12.0 * (n / N) ** 1.35
    V = np.random.normal(25.60, 0.12, N); I_ = np.random.uniform(-40.0, 40.0, N)
    Tt = np.random.normal(28.0, 1.8, N); Cv = np.random.normal(3.200, 0.008, (N, 8))
    g = pd.DataFrame({'PackID': 1, 'CycleNumber': n, 'Capacity(Ah)': cap, 'SOH(%)': cap / 40 * 100, 'RUL(cycles)': N - n,
                      'Voltage(V)': V, 'Current(A)': I_, 'DoD(%)': n / N * 100, 'Temperature(C)': Tt})
    for i in range(8): g[f'Cell{i+1}_V'] = Cv[:, i]
    g.to_csv(gp, index=False)
G = pd.read_csv(gp); y_g = (G['Capacity(Ah)'] / G['Capacity(Ah)'].iloc[:5].median()).values
ops = ['Voltage(V)', 'Current(A)', 'Temperature(C)'] + [f'Cell{i}_V' for i in range(1, 9)]
cut = int(0.7 * len(G))
m = HistGradientBoostingRegressor(max_iter=200, random_state=SEED).fit(G[ops].iloc[:cut], y_g[:cut])
c = certificate(m.predict(G[ops].iloc[cut:]), y_g[cut:])
a1 = dict(case='GMP learner on operating channels (chronological 70/30)', rmse=c['rmse'], rmse_q=c['rmse_q'], LB=c['LB'],
          viol=rms(y_g - project(y_g)))
inj = []
for b in [0.0, 0.02, 0.05]:
    for s in [0.005, 0.01, 0.02]:
        for phi in [0.0, 0.9]:
            for rep in range(20):
                e = np.zeros(len(y_g)); z = rng.normal(0, s * np.sqrt(1 - phi ** 2), len(y_g)); e[0] = rng.normal(0, s)
                for t in range(1, len(e)): e[t] = phi * e[t - 1] + z[t]
                cc = certificate(y_g + b + e, y_g)
                inj.append(dict(bias=b, noise_sd=s, phi=phi, rep=rep, rmse=cc['rmse'], rmse_q=cc['rmse_q'], LB=cc['LB']))
I = pd.DataFrame(inj); I.to_csv(f'{OUT}/gmp_injection.csv', index=False)
Ig = I.groupby(['bias', 'noise_sd', 'phi'])[['rmse', 'rmse_q', 'LB']].mean().reset_index()
Ig.to_csv(f'{OUT}/tab_gmp_injection.csv', index=False)
print(pd.DataFrame([a1]).round(4).to_string(index=False)); print(Ig.round(4).to_string(index=False))

# ---------- (b) NASA ----------
npth = os.path.join(DATA, 'nasa_cycle_features.csv')
N = pd.read_csv(npth)
X = ['V_mean', 'V_600s', 'T_max', 'T_start', 'I_mean']  # discharge-time proxies excluded
for b_, g in N.groupby('battery'):
    base = g[X].iloc[:5].mean(); N.loc[g.index, [f'd_{x}' for x in X]] = (g[X] - base).values
    N.loc[g.index, 'soh'] = g.capacity / g.capacity.iloc[:5].median()
XD = [f'd_{x}' for x in X]; rows = []
for name, mk in [('gbr', lambda: HistGradientBoostingRegressor(max_iter=200, random_state=SEED)), ('ridge', lambda: Ridge(1.0))]:
    for b_ in N.battery.unique():
        tr, te = N[N.battery != b_], N[N.battery == b_]
        p = mk().fit(tr[XD], tr.soh).predict(te[XD]); yv = te.soh.values; cc = certificate(p, yv)
        dv = rms(yv - project(yv))
        rows.append(dict(learner=name, cell=f'B{int(b_):04d}', rmse=cc['rmse'], rmse_q=cc['rmse_q'], LB=cc['LB'], viol=dv,
                         lb_holds=cc['LB'] <= cc['rmse'], lb_robust=cc['LB'] <= cc['rmse'] + dv,
                         improved=cc['rmse_q'] <= cc['rmse'], improved_robust=cc['rmse_q'] <= cc['rmse'] + 2 * dv))
NN = pd.DataFrame(rows); NN.to_csv(f'{OUT}/tab_nasa.csv', index=False); print(NN.round(4).to_string(index=False))
pd.DataFrame([a1]).to_csv(f'{OUT}/tab_gmp.csv', index=False)

# ---------- (a3) dense bias x noise surface on GMP (for the 3-D panel of Figure 4) ----------
surf = []
for b in np.linspace(0, 0.05, 11):
    for s in np.linspace(0.0025, 0.03, 12):
        for rep in range(5):
            cc = certificate(y_g + b + rng.normal(0, s, len(y_g)), y_g)
            surf.append(dict(bias=b, noise_sd=s, rep=rep, rmse=cc['rmse'], rmse_q=cc['rmse_q'], LB=cc['LB']))
S = pd.DataFrame(surf).groupby(['bias', 'noise_sd'])[['rmse', 'rmse_q', 'LB']].mean().reset_index()
S['removed'] = S.LB ** 2 / S.rmse ** 2; S.to_csv(f'{OUT}/gmp_surface.csv', index=False)
print('surface: removed share range %.3f-%.3f' % (S.removed.min(), S.removed.max()))
