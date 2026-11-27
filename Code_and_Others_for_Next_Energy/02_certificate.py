"""Step 2. Certificate and projection: offline (whole life) and online (prefix) evaluation.
Checks the theorem on every cell/configuration, the robust form LB <= RMSE + delta
(delta = RMS distance of the measured SOH from its own monotone projection),
and the Pythagorean gain RMSE_q^2 <= RMSE^2 - LB^2 (+ terms in delta)."""
import glob, warnings, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from common import project, rms, OUT
import os
warnings.filterwarnings('ignore')
# the six main configurations and, when step 14 has been run, the four stronger estimators
ALL = ['gbr_self', 'gbr_raw', 'ridge_self', 'ridge_raw', 'mlp_self', 'mlp_raw', 'gbrtuned_self', 'gbrfull_self', 'cnn_self', 'pinn', 'pinnpub']
CFGS = [c for c in ALL if os.path.exists(f'{OUT}/cells_{c}.csv')]
rows = []
for cfg in CFGS:
    C = pd.read_csv(f'{OUT}/cells_{cfg}.csv')
    holds = C.LB <= C.rmse + 1e-12; robust = C.LB <= C.rmse + C.viol + 1e-12
    pyth = C.rmse_q ** 2 <= C.rmse ** 2 - C.LB ** 2 + 1e-10
    rows.append(dict(cfg=cfg, cells=len(C), med_rmse=C.rmse.median(), med_rmse_q=C.rmse_q.median(),
                     mean_rmse=C.rmse.mean(), mean_rmse_q=C.rmse_q.mean(),
                     rel_red_median=float(np.median(1 - C.rmse_q / C.rmse)),
                     improved=float((C.rmse_q <= C.rmse + 1e-12).mean()),
                     lb_holds=float(holds.mean()), lb_robust=float(robust.mean()), pyth_holds=float(pyth.mean()),
                     wilcoxon_p=float(wilcoxon(C.rmse_q, C.rmse).pvalue)))
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/tab_certificate_offline.csv', index=False)
print(T.round(4).to_string(index=False))

# Online use: at checkpoint t only cycles 0..t are available. Online estimate = last value of
# the projection of the prefix (isotonic regression of the prefix).
on = []
for cfg in CFGS:
    P = pd.read_csv(f'{OUT}/pred_{cfg}.csv.gz')
    for c, g in P.groupby('cell', sort=False):
        p, y = g.p.values, g.soh.values; n = len(p)
        cps = np.unique(np.linspace(10, n - 1, 20).astype(int))
        e_raw, e_q, lbs, pre = [], [], [], []
        for t in cps:
            q = project(p[:t + 1]); e_raw.append(p[t] - y[t]); e_q.append(q[-1] - y[t])
            lbs.append(rms(p[:t + 1] - q)); pre.append(rms(p[:t + 1] - y[:t + 1]))
        on.append(dict(cfg=cfg, cell=c, rmse_online=rms(e_raw), rmse_online_q=rms(e_q),
                       lb_prefix_holds=float(np.mean(np.array(lbs) <= np.array(pre) + 1e-12))))
O = pd.DataFrame(on); O.to_csv(f'{OUT}/cells_online.csv', index=False)
S = O.groupby('cfg').agg(med_raw=('rmse_online', 'median'), med_q=('rmse_online_q', 'median'),
                         improved=('rmse_online_q', lambda s: np.nan), lb_holds=('lb_prefix_holds', 'mean'))
S['improved'] = O.assign(i=O.rmse_online_q <= O.rmse_online).groupby('cfg').i.mean()
S['wilcoxon_p'] = O.groupby('cfg').apply(lambda g: wilcoxon(g.rmse_online_q, g.rmse_online).pvalue)
S.to_csv(f'{OUT}/tab_certificate_online.csv'); print(S.round(4).to_string())
