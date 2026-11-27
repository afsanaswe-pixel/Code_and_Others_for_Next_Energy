"""Step 16. Deployable (causal) reference-check schedules and an alarm operating curve for the certificate.
(a) Online evaluation: at cycle t only predictions p_0..p_t and checks performed at cycles <= t are known.
    Online estimate at t = last value of the monotone projection of (p + c)[0..t], where c interpolates the
    measured residuals y_s - p_s at past checks and is held constant after the last check (zero before the first).
    Error = RMS over 40 evenly spaced evaluation cycles that are not check cycles.
    Schedules with a budget of K checks (K = 1..5), none uses the target cell's life length:
      fixed-interval : checks at cycles j * L_src / (K + 1), j = 1..K, with L_src the median life of source-domain cells;
      event-triggered: check when the upward departure of the prediction from its running minimum, averaged over the
                       last w = max(5, L_src / 40) cycles, exceeds tau = 1% SOH, at least L_src / (2(K+1)) cycles after
                       the previous check; a check is forced when L_src / (K + 1) cycles have passed without one;
      periodic-known-life (reference, not deployable): checks at j * n / (K + 1) with the true life n.
(b) Alarm operating curve: raise an alarm for a cell when LB >= theta (offline over the whole life, and online at mid-life using the first half of the trajectory);
    report alarm rate, precision and recall for the events RMSE >= theta and RMSE >= 2%."""
import numpy as np, pandas as pd
from common import project, rms, OUT
P = pd.read_csv(f'{OUT}/pred_gbr_self.csv.gz'); P['dom'] = P.cell.str.split('/').str[0]
life = P.groupby('cell').size(); ldom = P.groupby('cell').dom.first()
TAU = 0.01


def online_err(p, y, checks, ev):
    checks = np.sort(np.asarray(checks, int)); err = []
    for t in ev:
        s = checks[checks <= t]
        if len(s): c = np.interp(np.arange(t + 1), s, y[s] - p[s]); c[:s[0]] = 0.0
        else: c = np.zeros(t + 1)
        err.append(project(p[:t + 1] + c)[-1] - y[t])
    return rms(err)


def event_schedule(p, K, L):
    w = max(5, L // 40); gap_min = max(1, L // (2 * (K + 1))); gap_max = max(1, L // (K + 1))
    dep = pd.Series(p - np.minimum.accumulate(p)).rolling(w, min_periods=1).mean().values
    sel, last = [], 0
    for t in range(1, len(p)):
        if len(sel) >= K: break
        if (dep[t] > TAU and t - last >= gap_min) or t - last >= gap_max: sel.append(t); last = t
    return sel


rows = []
for c, g in P.groupby('cell', sort=False):
    p, y = g.p.values, g.soh.values; n = len(p); d = ldom[c]
    L = int(np.median(life[ldom != d]))
    ev_all = np.unique(np.linspace(10, n - 1, 40).astype(int))
    rec = dict(cell=c, dom=d, n=n, L_src=L)
    rec['none'] = online_err(p, y, [], ev_all)
    for K in range(1, 6):
        sch = {'fixed': [int(j * L / (K + 1)) for j in range(1, K + 1) if j * L / (K + 1) < n],
               'event': event_schedule(p, K, L),
               'periodic_known': list(np.unique(np.linspace(0, n - 1, K + 2)[1:-1].astype(int)))}
        for nm, s in sch.items():
            ev = np.setdiff1d(ev_all, s)
            rec[f'{nm}{K}'] = online_err(p, y, s, ev); rec[f'{nm}{K}_used'] = len(s)
    rows.append(rec)
R = pd.DataFrame(rows); R.to_csv(f'{OUT}/cells_checks_causal.csv', index=False)
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
summ = [dict(K=0, schedule='none', median=R.none.median(), domain_weighted=R.groupby('dom').none.median().mean(), checks_used=0)]
for K in range(1, 6):
    for nm in ['fixed', 'event', 'periodic_known']:
        dm = R.groupby('dom')[list(dict.fromkeys([f'{nm}{K}', f'fixed{K}', 'none']))].median()
        r = dict(K=K, schedule=nm, median=R[f'{nm}{K}'].median(), domain_weighted=dm[f'{nm}{K}'].mean(),
                 checks_used=R[f'{nm}{K}_used'].mean(), domains_better_than_none=int((dm[f'{nm}{K}'] < dm.none).sum()),
                 p_vs_none_domain=float(wilcoxon(dm[f'{nm}{K}'], dm.none, method='exact').pvalue))
        if nm == 'event':
            r['domains_better_than_fixed'] = int((dm[f'event{K}'] < dm[f'fixed{K}']).sum())
            r['p_vs_fixed_domain'] = float(wilcoxon(dm[f'event{K}'], dm[f'fixed{K}'], method='exact').pvalue)
        summ.append(r)
S = pd.DataFrame(summ)
m = S.p_vs_none_domain.notna(); S.loc[m, 'p_vs_none_holm'] = multipletests(S.loc[m, 'p_vs_none_domain'], method='holm')[1]
m = S.p_vs_fixed_domain.notna(); S.loc[m, 'p_vs_fixed_holm'] = multipletests(S.loc[m, 'p_vs_fixed_domain'], method='holm')[1]
S.to_csv(f'{OUT}/tab_checks_causal.csv', index=False); print(S.round(4).to_string(index=False))

# ---------------- (b) alarm operating curve ----------------
C = pd.read_csv(f'{OUT}/cells_gbr_self.csv'); on = []
for c, g in P.groupby('cell', sort=False):
    h = max(10, len(g) // 2); p, y = g.p.values[:h], g.soh.values[:h]; q = project(p); on.append(dict(cell=c, LB_on=rms(p - q), rmse_on=rms(p - y)))
C = C.merge(pd.DataFrame(on), on='cell'); al = []
for mode, lb, err in [('offline', 'LB', 'rmse'), ('online, first half of life', 'LB_on', 'rmse_on')]:
    for th in [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]:
        a = C[lb] >= th
        for ev_name, ev in [('RMSE >= theta', C[err] >= th), ('RMSE >= 2%', C[err] >= 0.02)]:
            al.append(dict(mode=mode, theta=th, event=ev_name, alarm_rate=a.mean(), prevalence=ev.mean(),
                           precision=(ev[a].mean() if a.any() else np.nan), recall=(a[ev].mean() if ev.any() else np.nan)))
A = pd.DataFrame(al); A.to_csv(f'{OUT}/tab_alarm.csv', index=False); print(A.round(3).to_string(index=False))
