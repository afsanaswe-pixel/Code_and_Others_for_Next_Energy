"""Step 7. Data figures from outputs/, saved under the Next Energy article numbering:
Figure_2 (unseen-domain error), Figure_3 (certificate), Figure_4 (GMP known truth), Figure_5 (conformal bounds),
Figure_6 (reference checks). Two further plots are not in the article and are saved as not_in_paper_*.png:
the label-free warning signals (values in Table 5) and the offline check placements (values in Table S4)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT, FIG, project
plt.rcParams.update({'font.size': 13, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
                     'axes.titlesize': 14, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                     'legend.fontsize': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
BLUE, ORANGE, GREEN, AMBER, PINK, GREY, INK = '#1f5fbf', '#d9541e', '#138a5e', '#c98a00', '#c2477d', '#8a8a85', '#111111'
import textwrap
def title(a, t):
    w = a.get_position().width * a.figure.get_figwidth()
    a.set_title(textwrap.fill(t, max(20, int(w * 72 / (14 * 0.6)))), loc='left', pad=8)
CFG = ['gbr_self', 'gbr_raw', 'mlp_self', 'mlp_raw', 'ridge_self', 'ridge_raw']
CFG10 = ['gbr_self', 'gbr_raw', 'ridge_self', 'ridge_raw', 'mlp_self', 'mlp_raw', 'gbrtuned_self', 'gbrfull_self', 'cnn_self', 'pinn', 'pinnpub']
LAB = {'gbr_self': 'GBR, self-ref.', 'gbr_raw': 'GBR, raw', 'mlp_self': 'MLP, self-ref.', 'mlp_raw': 'MLP, raw',
       'ridge_self': 'Ridge, self-ref.', 'ridge_raw': 'Ridge, raw', 'gbrtuned_self': 'Tuned GBR, self-ref.',
       'gbrfull_self': 'GBR, all cycles', 'cnn_self': '1D-CNN, self-ref.', 'pinn': 'PINN (Wang et al.)', 'pinnpub': 'PINN, published scaling*'}
COL = {'gbr_self': BLUE, 'gbr_raw': BLUE, 'mlp_self': ORANGE, 'mlp_raw': ORANGE, 'ridge_self': GREEN, 'ridge_raw': GREEN,
       'gbrtuned_self': '#0b2a5b', 'gbrfull_self': '#5b8fd9', 'cnn_self': PINK, 'pinn': AMBER, 'pinnpub': '#8a6d00'}

# ---------- Fig. 2: unseen-domain error by domain and estimator ----------
cells = {c: pd.read_csv(f'{OUT}/cells_{c}.csv') for c in CFG10}
M = pd.DataFrame({LAB[c]: cells[c].groupby('dom').rmse.median() * 100 for c in CFG10})
fig, a = plt.subplots(figsize=(14, 7.6))
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list('l', plt.get_cmap('YlOrBr')(np.linspace(0, .6, 256)))
im = a.imshow(M.values, cmap=cm, aspect='auto', vmin=0, vmax=12)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        a.text(j, i, f'{M.values[i, j]:.1f}', ha='center', va='center', color=INK, fontsize=11)
a.set_xticks(range(M.shape[1])); a.set_xticklabels(M.columns, rotation=30, ha='right'); a.axvline(5.5, color=INK, lw=3)
a.set_yticks(range(M.shape[0])); a.set_yticklabels(M.index)
cb = fig.colorbar(im, ax=a, fraction=0.04); cb.set_label('median per-cell RMSE (% SOH)')
title(a, 'Unseen-domain SOH error varies more across domains than across eleven estimator configurations, including tuned, sequence and physics-informed models')
fig.tight_layout(); fig.savefig(f'{FIG}/Figure_2.png'); plt.close(fig)

# ---------- Fig. 3: certificate geometry and validity ----------
P = pd.read_csv(f'{OUT}/pred_mlp_raw.csv.gz'); Cm = cells['mlp_raw']
cand = Cm[(Cm.viol < 0.001) & (Cm.rmse < 0.08)]; ex = cand.sort_values('LB').iloc[int(0.85 * len(cand))].cell
g = P[P.cell == ex]; p, y = g.p.values, g.soh.values; q = project(p)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
fig = plt.figure(figsize=(15.5, 12))
# (a) 3-D geometry of the certificate for n = 3 cycles
a = fig.add_subplot(2, 2, 1, projection='3d')
u = np.linspace(0.4, 1, 25); U, T = np.meshgrid(u, u); M = T <= U
P1 = np.where(M, U, np.nan); P2 = np.where(M, T, np.nan)
a.plot_surface(P1, P1, P2, color='#a9c6f0', alpha=.30, linewidth=0)          # face q1 = q2 >= q3
a.plot_surface(P1, P2, P2, color='#9fd6b8', alpha=.30, linewidth=0)          # face q1 >= q2 = q3
a.plot([0.4, 1], [0.4, 1], [0.4, 1], color=INK, lw=2)
yv = np.array([0.95, 0.78, 0.50]); pv = np.array([0.62, 0.92, 0.58]); qv = project(pv)
a.plot(*np.c_[pv, qv], color=ORANGE, lw=5, label='p \u2192 q : certificate LB')
a.plot(*np.c_[qv, yv], color=BLUE, lw=5, label='q \u2192 y : error after projection')
a.plot(*np.c_[pv, yv], color=GREY, lw=3, ls='--', label='p \u2192 y : error before')
a.scatter(*pv, s=260, color=ORANGE, depthshade=False, edgecolor=INK, linewidth=1.5, label='prediction p (outside C)')
a.scatter(*qv, s=260, color=BLUE, depthshade=False, edgecolor=INK, linewidth=1.5, label='projection q (on C)')
a.scatter(*yv, s=260, color=INK, depthshade=False, edgecolor='white', linewidth=1.5, label='true y (inside C)')
a.legend(loc='upper left', bbox_to_anchor=(-0.02, 0.93), frameon=False, fontsize=11)
a.set_xlabel('q1', labelpad=8); a.set_ylabel('q2', labelpad=8); a.set_zlabel('q3', labelpad=8)
a.set_xlim(0.4, 1); a.set_ylim(0.4, 1); a.set_zlim(0.4, 1); a.view_init(elev=24, azim=-62)
a.set_title('(a) 3-D geometry for three cycles, C = {q1 \u2265 q2 \u2265 q3}:\n|p\u2212y|\u00b2 \u2265 |p\u2212q|\u00b2 + |q\u2212y|\u00b2, hence LB \u2264 error', loc='left', pad=4)
a = fig.add_subplot(2, 2, 2)
a.plot(g.k, p, color=ORANGE, lw=1.2, label='prediction p'); a.plot(g.k, q, color=BLUE, lw=2.6, label='projection q')
a.plot(g.k, y, color=INK, lw=2, ls='--', label='measured SOH y'); a.set_xlabel('cycle'); a.set_ylabel('SOH'); a.legend(frameon=False)
title(a, f'(b) The projection q removes rebounds of the prediction p (cell from {ex.split("/")[0]})')
a = fig.add_subplot(2, 2, 3)
for c in CFG10:
    d = cells[c]; a.scatter(d.rmse * 100, d.LB * 100, s=9, color=COL[c], alpha=.4, label=LAB[c].split(',')[0] if c in ('gbr_self', 'mlp_self', 'ridge_self', 'gbrtuned_self', 'cnn_self', 'pinn') else None)
mx = max(d.rmse.max() for d in cells.values()) * 100; a.plot([0.05, mx], [0.05, mx], color=INK, lw=2, ls='--'); a.set_xscale('log'); a.set_yscale('log'); a.set_ylim(0.01, mx * 1.5); a.set_xlim(0.1, mx * 1.5)
a.set_xlabel('true per-cell RMSE (% SOH)'); a.set_ylabel('certificate LB (% SOH)'); a.legend(frameon=False, markerscale=2, fontsize=10, ncol=2, loc='lower right')
nv = sum(int((d.LB <= d.rmse + 1e-12).sum()) for d in cells.values()); nt = sum(len(d) for d in cells.values())
title(a, f'(c) LB stayed below the true error in {nv:,} of {nt:,} cell-estimator pairs (log scale)')
a = fig.add_subplot(2, 2, 4)
for c in CFG10:
    d = cells[c]; a.scatter(d.rmse * 100, d.rmse_q * 100, s=9, color=COL[c], alpha=.4)
a.plot([0.1, mx], [0.1, mx], color=INK, lw=2, ls='--'); a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('RMSE before projection (% SOH)'); a.set_ylabel('RMSE after projection (% SOH)')
imp = [(d.rmse_q <= d.rmse + 1e-12).mean() * 100 for d in cells.values()]
title(a, f'(d) Projection lowered the error in {min(imp):.0f}-{max(imp):.0f}% of cells for every estimator (log scale)')
fig.tight_layout(); fig.savefig(f'{FIG}/Figure_3.png'); plt.close(fig)

# ---------- Fig. 4: GMP known-truth injection ----------
I = pd.read_csv(f'{OUT}/tab_gmp_injection.csv'); I0 = I[I.phi == 0]
fig = plt.figure(figsize=(16, 12.5)); ax = [fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)]
a = ax[0]
for b, col, mk in [(0, BLUE, 'o'), (0.02, ORANGE, 's'), (0.05, GREEN, '^')]:
    d = I0[I0.bias == b]; a.plot(d.noise_sd * 100, d.rmse * 100, color=col, marker=mk, lw=2.5, ms=8, label=f'true RMSE, bias {b*100:.0f}%')
    a.plot(d.noise_sd * 100, d.LB * 100, color=col, marker=mk, lw=2.5, ms=8, ls=':', mfc='white')
a.set_xlabel('injected noise SD (% SOH)'); a.set_ylabel('% SOH'); a.legend(frameon=False, title='dotted = certificate LB', loc='center left', bbox_to_anchor=(0.0, 0.62))
title(a, '(a) The certificate measures the noise and is blind to bias')
a = ax[1]; w = 0.27; xs = np.arange(3)
for i, (b, col) in enumerate([(0, BLUE), (0.02, ORANGE), (0.05, GREEN)]):
    d = I0[I0.bias == b]; a.bar(xs + (i - 1) * w, (1 - d.rmse_q / d.rmse).values * 100, w, color=col, label=f'bias {b*100:.0f}%')
a.set_xticks(xs); a.set_xticklabels([f'noise {s*100:.1f}%' for s in sorted(I0.noise_sd.unique())])
a.set_ylabel('RMSE reduction by projection (%)'); a.legend(frameon=False)
title(a, '(b) Projection removes noise, so its gain shrinks as bias dominates')
from matplotlib.colors import LinearSegmentedColormap
cm3 = LinearSegmentedColormap.from_list('l3', plt.get_cmap('YlOrBr')(np.linspace(0.05, .6, 256)))
for k, (fn, lab) in enumerate([('gmp_surface.csv', '(c) Independent noise'), ('gmp_surface_ar1.csv', '(d) Autocorrelated noise, AR(1) \u03c6 = 0.9')]):
    a = fig.add_subplot(2, 2, 3 + k, projection='3d'); Sf = pd.read_csv(f'{OUT}/{fn}')
    Bv = np.sort(Sf.bias.unique()); Nv = np.sort(Sf.noise_sd.unique()); Z = Sf.pivot(index='noise_sd', columns='bias', values='removed').loc[Nv, Bv].values * 100
    BB, NN = np.meshgrid(Bv * 100, Nv * 100)
    a.plot_surface(BB, NN, Z, cmap=cm3, edgecolor='#555555', linewidth=0.3, alpha=0.95, vmin=0, vmax=100)
    a.set_xlabel('bias (% SOH)', labelpad=10); a.set_ylabel('noise SD (% SOH)', labelpad=10); a.set_zlabel('error visible (%)', labelpad=10)
    a.view_init(elev=26, azim=-130); a.set_zlim(0, 100)
    a.set_title(lab + ':\nshare of squared error visible to the certificate (LB\u00b2/RMSE\u00b2)', loc='left', pad=4)
fig.tight_layout(); fig.savefig(f'{FIG}/Figure_4.png'); plt.close(fig)

# ---------- not in the article (values in Table 5): label-free warning signals ----------
S = pd.read_csv(f'{OUT}/tab_signals.csv')
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
yy = np.arange(len(S))[::-1]
for a, (k, lab, ref) in zip(ax, [('auc2', 'AUROC for RMSE > 2% SOH', 0.5), ('rho_within', 'within-domain Spearman with RMSE', 0)]):
    a.errorbar(S[k], yy, xerr=[S[k] - S[k + '_lo'], S[k + '_hi'] - S[k]], fmt='o', color=BLUE, ms=10, lw=3, capsize=6)
    a.axvline(ref, color=INK, ls='--', lw=2); a.set_xlabel(lab); a.set_yticks(yy); a.set_yticklabels(S.signal)
title(ax[0], '(a) Signals separate bad from good cells only with wide uncertainty')
title(ax[1], '(b) Within a domain, ranking of cells by error is weak (95% domain-bootstrap CIs)')
fig.tight_layout(); fig.savefig(f'{FIG}/not_in_paper_warning_signals.png'); plt.close(fig)

# ---------- Fig. 5: conformal error bounds ----------
Q = pd.read_csv(f'{OUT}/tab_conformal_domains.csv'); Cf = pd.read_csv(f'{OUT}/cells_conformal.csv')
order = Q[Q.variant == 'split'].sort_values('coverage').dom.tolist()
fig, ax = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
a = ax[0]; yy = {d: i for i, d in enumerate(order)}
for k, (v, col, mk, lab) in enumerate([('split', ORANGE, 'o', 'split (pooled)'), ('weighted', PINK, 's', 'weighted (covariate shift)'),
                                       ('family', GREEN, 'D', 'group (dataset family)'), ('hier', BLUE, '^', 'hierarchical (domain level)')]):
    d = Q[Q.variant == v].set_index('dom').loc[order]; y = np.arange(len(order)) + (k - 1.5) * 0.18
    a.errorbar(d.coverage * 100, y, xerr=[(d.coverage - d.lo) * 100, (d.hi - d.coverage) * 100], fmt=mk, color=col, ms=7, lw=1.4, capsize=2, label=lab)
a.axvline(90, color=INK, ls='--', lw=2); a.set_yticks(range(len(order))); a.set_yticklabels(order, fontsize=11)
a.set_xlabel('coverage of the 90% upper bound (%, 95% Clopper-Pearson)'); a.set_xlim(-2, 104)
a.legend(frameon=False, fontsize=10.5, loc='upper left')
title(a, '(a) Per-domain coverage: only the weighted variant reaches HUST, the most dissimilar domain')
a = ax[1]; T6 = pd.read_csv(f'{OUT}/tab_conformal.csv').set_index('variant').loc[['split', 'weighted', 'family', 'hier']]
lbl = ['split', 'weighted', 'group', 'hierarchical']; x = np.arange(4)
a.bar(x - 0.2, T6.coverage * 100, 0.4, color=BLUE, label='overall coverage (%)')
a.bar(x + 0.2, T6.infinite_bounds * 100, 0.4, color=ORANGE, label='non-informative (infinite) bounds (%)')
for i, (c_, w_) in enumerate(zip(T6.coverage, T6.width_median)): a.text(i - 0.2, c_ * 100 + 1.5, f'{c_*100:.0f}', ha='center', fontsize=11); a.text(i, -9, f'width {w_:.1f}x', ha='center', fontsize=10.5, color=INK)
a.axhline(90, color=INK, ls='--', lw=2); a.set_xticks(x); a.set_xticklabels(lbl); a.set_ylim(-12, 105); a.legend(frameon=False, fontsize=10.5, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1)
title(a, '(b) Coverage is bought with wider or non-informative bounds')
fig.tight_layout(); fig.savefig(f'{FIG}/Figure_5.png'); plt.close(fig)

# ---------- Fig. 6: deployable (causal) reference-check schedules ----------
T7 = pd.read_csv(f'{OUT}/tab_checks_causal.csv')
fig, a = plt.subplots(figsize=(11, 6.0))
b0 = T7[T7.K == 0]['median'].values[0] * 100
for nm, col, mk, ls, lab in [('fixed', BLUE, 'o', '-', 'fixed interval (source-domain life)'), ('event', ORANGE, '^', '-', 'event-triggered (departure, causal)'),
                             ('periodic_known', GREY, 's', '--', 'periodic with known life (reference)')]:
    d = T7[T7.schedule == nm]; a.plot([0] + list(d.K), [b0] + list(d['median'] * 100), color=col, marker=mk, lw=3, ms=10, ls=ls, label=lab)
a.set_xlabel('reference capacity checks per cell, K (budget)'); a.set_ylabel('median online per-cell RMSE (% SOH)'); a.legend(frameon=False)
title(a, 'Online, a few measured checks lower the error; knowing when to check matters as much as how many')
fig.tight_layout(); fig.savefig(f'{FIG}/Figure_6.png'); plt.close(fig)

# ---------- not in the article (values in Table S4): offline placement comparison ----------
T = pd.read_csv(f'{OUT}/tab_checks.csv')
fig, a = plt.subplots(figsize=(10, 5.8))
base = T[T.K == 0]['median'].values[0] * 100
for nm, col, mk, lab in [('periodic', BLUE, 'o', 'periodic (equal cycles)'), ('fade', AMBER, 's', 'equal predicted fade'),
                         ('departure', ORANGE, '^', 'largest departure from monotone')]:
    d = T[T.placement == nm]; a.plot([0] + list(d.K), [base] + list(d['median'] * 100), color=col, marker=mk, lw=3, ms=10, label=lab)
a.set_xlabel('reference capacity checks per cell, K'); a.set_ylabel('median per-cell RMSE (% SOH)'); a.legend(frameon=False)
title(a, 'Offline placements that use the whole predicted trajectory (not deployable)')
fig.tight_layout(); fig.savefig(f'{FIG}/not_in_paper_offline_check_placements.png'); plt.close(fig)
print('figures done')
