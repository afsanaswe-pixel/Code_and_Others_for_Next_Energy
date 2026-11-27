"""Step 12. Explainability, generalizability and reliability figures, saved under the Next Energy article numbering:
Figure_7 (explainability), Figure_8 (generalizability), Figure_S1 (reliability)."""
import numpy as np, pandas as pd, matplotlib, textwrap
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import OUT, FIG
plt.rcParams.update({'font.size': 13, 'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
                     'axes.titlesize': 14, 'axes.labelsize': 13, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
                     'legend.fontsize': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
BLUE, ORANGE, GREEN, AMBER, PINK, GREY, INK = '#1f5fbf', '#d9541e', '#138a5e', '#c98a00', '#c2477d', '#8a8a85', '#111111'
def title(a, t):
    w = a.get_position().width * a.figure.get_figwidth()
    a.set_title(textwrap.fill(t, max(20, int(w * 72 / (14 * 0.6)))), loc='left', pad=8)
LAB = {'gbr_self': 'GBR, self-ref.', 'gbr_raw': 'GBR, raw', 'mlp_self': 'MLP, self-ref.', 'mlp_raw': 'MLP, raw',
       'ridge_self': 'Ridge, self-ref.', 'ridge_raw': 'Ridge, raw'}

# ---------- Fig. S1 (Supplementary Material): reliability ----------
R = pd.read_csv(f'{OUT}/rel_runs.csv'); B = pd.read_csv(f'{OUT}/tab_reliability_ci.csv')
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
a = ax[0]
for i, cfg in enumerate(['gbr_self', 'mlp_raw']):
    d = R[R.cfg == cfg]
    a.scatter(np.full(len(d), 2 * i) + np.linspace(-.12, .12, len(d)), d.med_rmse * 100, s=90, color=ORANGE, label='before projection' if i == 0 else None)
    a.scatter(np.full(len(d), 2 * i + 0.8) + np.linspace(-.12, .12, len(d)), d.med_rmse_q * 100, s=90, color=BLUE, marker='s', label='after projection' if i == 0 else None)
a.set_xticks([0.4, 2.4]); a.set_xticklabels([LAB['gbr_self'], LAB['mlp_raw']]); a.set_ylabel('median per-cell RMSE (% SOH)')
a.legend(frameon=False); a.set_xlim(-0.6, 3.4)
title(a, '(a) Five seeds and training subsamples: projection removes most run-to-run variation of the noisy learner')
a = ax[1]; d = B[B.quantity == 'rel_red'].reset_index(drop=True); yy = np.arange(len(d))[::-1]
a.errorbar(d.estimate * 100, yy, xerr=[(d.estimate - d.lo) * 100, (d.hi - d.estimate) * 100], fmt='o', color=BLUE, ms=10, lw=3, capsize=6)
a.axvline(0, color=INK, ls='--', lw=2); a.set_yticks(yy); a.set_yticklabels([LAB[c] for c in d.cfg])
a.set_xlabel('median relative RMSE reduction by projection (%)')
title(a, '(b) The gain is positive for every configuration (95% domain-bootstrap CIs)')
fig.tight_layout(); fig.savefig(f'{FIG}/Figure_S1.png'); plt.close(fig)

# ---------- Fig. 7: explainability ----------
I = pd.read_csv(f'{OUT}/xai_importance.csv', index_col=0); A = pd.read_csv(f'{OUT}/xai_shift.csv', index_col=0)
Xd = pd.read_csv(f'{OUT}/xai_domains.csv').set_index('dom')
order = I.mean().sort_values(ascending=False).index
In = I[order].div(I.sum(axis=1), axis=0) * 100
In = pd.concat([In.iloc[:, :8], In.iloc[:, 8:].sum(axis=1).rename('other 8 features')], axis=1)
fig, ax = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw=dict(width_ratios=[1.2, 1]))
a = ax[0]
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list('l', plt.get_cmap('YlOrBr')(np.linspace(0, .6, 256)))
im = a.imshow(In.values, cmap=cm, aspect='auto', vmin=0, vmax=45)
for i in range(In.shape[0]):
    for j in range(In.shape[1]):
        a.text(j, i, f'{In.values[i, j]:.0f}', ha='center', va='center', fontsize=11, color=INK)
a.set_xticks(range(In.shape[1])); a.set_xticklabels(In.columns, rotation=45, ha='right', fontsize=11)
a.set_yticks(range(len(In))); a.set_yticklabels(In.index, fontsize=11)
cb = fig.colorbar(im, ax=a, fraction=0.035); cb.set_label('share of mean |SHAP| (%)')
title(a, '(a) Feature importance on each unseen domain: the two charged-capacity features dominate in all 13 models')
a = ax[1]
E = pd.read_csv(f'{OUT}/xai_error_domains.csv').set_index('dom').loc[In.index]; T_ = pd.read_csv(f'{OUT}/xai_error_terms.csv', index_col=0).loc[In.index]
yy = np.arange(len(E))[::-1]; h = 0.36
a.barh(yy + h / 2, E.base_term * 100, h, color=GREY, label='expected-output difference (label level)')
a.barh(yy - h / 2, E.feature_terms * 100, h, color=BLUE, label='feature-use difference (sum over features)')
a.scatter(E.gap_src_minus_oracle * 100, yy, color=ORANGE, marker='*', s=240, zorder=5, label='deployed minus oracle prediction')
a.axvline(0, color=INK, lw=1.5); a.set_yticks(yy); a.set_yticklabels(E.index, fontsize=11)
a.set_xlabel('contribution to the mean gap from a target-trained oracle (% SOH)')
a.legend(frameon=False, fontsize=10.5, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=1)
title(a, '(b) Error-centred decomposition: deployed model versus a model trained on the target domain')
fig.tight_layout(); fig.savefig(f'{FIG}/Figure_7.png'); plt.close(fig)

# ---------- Fig. 8: generalizability ----------
L = pd.read_csv(f'{OUT}/tab_lodso.csv'); K = pd.read_csv(f'{OUT}/tab_ndomains.csv'); KK = pd.read_csv(f'{OUT}/gen_ndomains.csv')
fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
a = ax[0]; ds = sorted(L.dataset.unique()); x = np.arange(len(ds)); w = 0.2
for i, (ref, col_lodo, col_lods) in enumerate([('self', '#a9c6f0', BLUE), ('raw', '#f5ae8c', ORANGE)]):
    d = L[L.ref == ref].set_index('dataset').loc[ds]
    a.bar(x + (2 * i - 1.5) * w, d.med_rmse_lodo * 100, w, color=col_lodo, edgecolor=col_lods, lw=2, label=f'{ref}: leave-one-domain-out')
    a.bar(x + (2 * i - 0.5) * w, d.med_rmse * 100, w, color=col_lods, label=f'{ref}: leave-one-dataset-out')
a.set_xticks(x); a.set_xticklabels(ds); a.set_ylabel('median per-cell RMSE (% SOH)'); a.legend(frameon=False, fontsize=10)
title(a, '(a) Removing a whole dataset raises error, most for MIT')
a = ax[1]
a.fill_between(K.k, K.q25 * 100, K.q75 * 100, color='#a9c6f0', alpha=.6, label='interquartile range over targets')
a.plot(K.k, K.med_rmse * 100, color=BLUE, marker='o', lw=3, ms=9, label='median, before projection')
a.plot(K.k, K.med_rmse_q * 100, color=GREEN, marker='s', lw=3, ms=9, label='median, after projection')
a.set_xscale('log', base=2); a.set_xticks(K.k); a.set_xticklabels(K.k); a.set_xlabel('number of training domains'); a.set_ylabel('median per-cell RMSE (% SOH)')
a.legend(frameon=False, fontsize=10.5)
title(a, '(b) Error falls as more domains are used for training')
a = ax[2]
a.plot(K.k, K.removed * 100, color=ORANGE, marker='^', lw=3, ms=10)
a.set_xscale('log', base=2); a.set_xticks(K.k); a.set_xticklabels(K.k); a.set_xlabel('number of training domains'); a.set_ylabel('squared error removed by projection (%)')
title(a, '(c) As bias shrinks, a larger share of error becomes visible to the certificate')
fig.tight_layout(); fig.savefig(f'{FIG}/Figure_8.png'); plt.close(fig)
print('trend figures done')
