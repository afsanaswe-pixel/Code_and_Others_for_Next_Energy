"""Figure 1: methodology figure. Layered pipeline whose icons are miniatures of the paper's own
figures and tables, redrawn from outputs/ (run after steps 01-12). Dark bold text on light fills only."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from common import OUT, FIG, project
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.weight': 'bold'})
INK, NAVY, GREY = '#111111', '#0b2a5b', '#8a8a85'
BLUE, ORANGE, GREEN, AMBER, PINK = '#1f5fbf', '#d9541e', '#138a5e', '#c98a00', '#c2477d'
TINT = {BLUE: '#dde8f8', ORANGE: '#fbe4d8', GREEN: '#dff0e8', AMBER: '#f6f1df', PINK: '#f7e9f0', GREY: '#eeeeea'}
CM = LinearSegmentedColormap.from_list('l', plt.get_cmap('YlOrBr')(np.linspace(0, .6, 256)))
W, H = 160, 92
fig = plt.figure(figsize=(16, 9.2), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')


def panel(x, y, w, h, col, head):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.4', fc='white', ec=col, lw=2.6, zorder=1))
    ax.add_patch(FancyBboxPatch((x, y + h - 5.2), w, 5.2, boxstyle='round,pad=0,rounding_size=1.4', fc=TINT[col], ec=col, lw=2.6, zorder=1))
    ax.text(x + w / 2, y + h - 2.6, head, ha='center', va='center', fontsize=13.5, color=INK, zorder=3)


def mini(x, y, w, h, tag=None):
    a = fig.add_axes([x / W, y / H, w / W, h / H]); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_color(GREY); s.set_linewidth(1)
    if tag: ax.text(x + w, y - 0.4, tag, ha='right', va='top', fontsize=9.5, color=NAVY)
    return a


def chip(x, y, t, fs=10.8, c=INK):
    ax.text(x, y, t, ha='center', va='center', fontsize=fs, color=c, zorder=4,
            bbox=dict(boxstyle='round,pad=0.22', fc='#f3f1ea', ec='none'))


def arrow(p, q, c=INK, lw=2.6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=20, lw=lw, color=c, zorder=5))


TOPY, TOPH, PW, GAP, X0 = 33, 57, 29.6, 2.5, 1.5
xs = [X0 + i * (PW + GAP) for i in range(5)]
heads = ['1  Public data', '2  Estimation', '3  Label-free monitoring', '4  Calibration', '5  Trustworthiness']
cols = [GREEN, BLUE, ORANGE, PINK, AMBER]
for x, hd, c in zip(xs, heads, cols): panel(x, TOPY, PW, TOPH, c, hd)
for i in range(4): arrow((xs[i] + PW + 0.2, TOPY + 30), (xs[i + 1] - 0.2, TOPY + 30))

# ---- 1 data: battery icons + mini table ----
x = xs[0]
for k, (nm, c) in enumerate([('XJTU', BLUE), ('TJU', ORANGE), ('HUST', GREEN), ('MIT', PINK)]):
    bx = x + 2.2 + k * 6.6
    ax.add_patch(FancyBboxPatch((bx, 70), 5.0, 10.5, boxstyle='round,pad=0,rounding_size=0.8', fc=TINT[c], ec=c, lw=2))
    ax.add_patch(Rectangle((bx + 1.6, 80.5), 1.8, 1.0, fc=c, ec='none'))
    ax.add_patch(Rectangle((bx + 0.7, 71), 3.6, 6.5 - k * 1.2, fc=c, ec='none', alpha=.8))
    ax.text(bx + 2.5, 68.3, nm, ha='center', fontsize=10, color=INK)
cells = [['domain', 'cells', 'cycles'], ['13', '387', '310,554']]
for r in range(2):
    for cc in range(3):
        ax.add_patch(Rectangle((x + 2.5 + cc * 8.2, 57.5 - r * 4), 8.2, 4, fc='#e7eef7' if r == 0 else 'white', ec=GREY, lw=1))
        ax.text(x + 6.6 + cc * 8.2, 59.5 - r * 4, cells[r][cc], ha='center', va='center', fontsize=10, color=INK)
ax.text(x + 27.1, 52.6, 'Table 2', ha='right', va='top', fontsize=9.5, color=NAVY)
chip(x + PW / 2, 47.5, '16 charge features per cycle')
chip(x + PW / 2, 43.2, 'SOH = C / C(start)')
chip(x + PW / 2, 38.9, 'controls: GMP, NASA')

# ---- 2 estimation: mini heatmap (Fig. 2) ----
x = xs[1]
CFG = ['gbr_self', 'gbr_raw', 'mlp_self', 'mlp_raw', 'ridge_self', 'ridge_raw']
CFG10 = ['gbr_self', 'gbr_raw', 'ridge_self', 'ridge_raw', 'mlp_self', 'mlp_raw', 'gbrtuned_self', 'gbrfull_self', 'cnn_self', 'pinn', 'pinnpub']
M = pd.DataFrame({c: pd.read_csv(f'{OUT}/cells_{c}.csv').groupby('dom').rmse.median() for c in CFG10}).values * 100
a = mini(x + 3, 58, PW - 6, 22.5, 'Fig. 2'); a.imshow(M, cmap=CM, aspect='auto', vmin=0, vmax=12)
chip(x + PW / 2, 52.3, 'GBR · Ridge · MLP, raw / self-ref.')
chip(x + PW / 2, 48.0, 'tuned GBR · 1D-CNN · PINN')
chip(x + PW / 2, 43.7, 'leave-one-domain-out (13)')
chip(x + PW / 2, 39.4, 'leave-one-dataset-out (4)')

# ---- 3 monitoring: trajectory + LB scatter ----
x = xs[2]
P = pd.read_csv(f'{OUT}/pred_mlp_raw.csv.gz'); Cm = pd.read_csv(f'{OUT}/cells_mlp_raw.csv')
cand = Cm[(Cm.viol < 0.001) & (Cm.rmse < 0.08)]; ex = cand.sort_values('LB').iloc[int(0.85 * len(cand))].cell
g = P[P.cell == ex]
a = mini(x + 2.5, 69, PW - 5, 13.5, 'Fig. 3b')
a.plot(g.k, g.p, color=ORANGE, lw=0.6); a.plot(g.k, project(g.p.values), color=BLUE, lw=1.8); a.plot(g.k, g.soh, color=INK, lw=1.2, ls='--')
a = mini(x + 2.5, 50.5, 11.5, 13.5, 'Fig. 3c')
for c, col in zip(CFG, [BLUE, BLUE, ORANGE, ORANGE, GREEN, GREEN]):
    d = pd.read_csv(f'{OUT}/cells_{c}.csv'); a.scatter(d.rmse, d.LB, s=1.2, color=col, alpha=.5)
a.set_xscale('log'); a.set_yscale('log'); a.plot([1e-3, 3], [1e-3, 3], color=INK, lw=1); a.set_xlim(2e-3, 3); a.set_ylim(1e-4, 3)
a.set_xticks([]); a.set_yticks([]); a.minorticks_off()
S = pd.read_csv(f'{OUT}/tab_signals.csv')
a = mini(x + 15.6, 50.5, 11.5, 13.5, 'Table 5'); a.barh(range(len(S)), S.auc2, color=ORANGE); a.axvline(0.5, color=INK, lw=1); a.set_xlim(0.3, 1)
chip(x + PW / 2, 45.0, 'projection onto non-increasing C')
chip(x + PW / 2, 40.7, 'certificate LB = RMS(p − q)')
chip(x + PW / 2, 36.4, '+ 4 warning signals')

# ---- 4 calibration: coverage + checks ----
x = xs[3]
Q = pd.read_csv(f'{OUT}/tab_conformal_domains.csv'); pv = Q.pivot(index='dom', columns='variant', values='coverage').sort_values('split')
a = mini(x + 2.5, 66.5, PW - 5, 16, 'Fig. 5')
for v, col, mk in [('split', ORANGE, 'o'), ('weighted', PINK, 's'), ('hier', BLUE, '^')]: a.scatter(pv[v], range(len(pv)), s=7, color=col, marker=mk)
a.axvline(0.9, color=INK, lw=1.2, ls='--'); a.set_xlim(0, 1.05)
T = pd.read_csv(f'{OUT}/tab_checks_causal.csv'); base = T[T.K == 0]['median'].values[0]
a = mini(x + 2.5, 47.5, PW - 5, 14, 'Fig. 6')
for nm, col in [('fixed', BLUE), ('event', ORANGE), ('periodic_known', GREY)]:
    d = T[T.schedule == nm]; a.plot([0] + list(d.K), [base] + list(d['median']), color=col, lw=2, marker='o', ms=2.5, ls='--' if nm == 'periodic_known' else '-')
chip(x + PW / 2, 43.4, 'conformal 90%: split, weighted,')
chip(x + PW / 2, 39.6, 'group, hierarchical')
chip(x + PW / 2, 35.8, 'K = 1–5 causal checks', fs=10.2)

# ---- 5 trustworthiness: reliability, explainability, generalizability ----
x = xs[4]
R = pd.read_csv(f'{OUT}/rel_runs.csv')
a = mini(x + 2.5, 70.5, 11.5, 11.5, 'Fig. S1')
for i, cfg in enumerate(['gbr_self', 'mlp_raw']):
    d = R[R.cfg == cfg]; a.scatter(np.full(len(d), 2 * i), d.med_rmse, s=8, color=ORANGE); a.scatter(np.full(len(d), 2 * i + 0.8), d.med_rmse_q, s=8, color=BLUE, marker='s')
a.set_xlim(-0.6, 3.4)
I = pd.read_csv(f'{OUT}/xai_importance.csv', index_col=0); I = I[I.mean().sort_values(ascending=False).index]
a = mini(x + 15.6, 70.5, 11.5, 11.5, 'Fig. 7'); a.imshow(I.div(I.sum(axis=1), axis=0).values, cmap=CM, aspect='auto')
K = pd.read_csv(f'{OUT}/tab_ndomains.csv')
a = mini(x + 2.5, 53, PW - 5, 12, 'Fig. 8'); a.fill_between(K.k, K.q25, K.q75, color='#f6e3b4'); a.plot(K.k, K.med_rmse, color=AMBER, lw=2, marker='o', ms=3)
a.plot(K.k, K.med_rmse_q, color=GREEN, lw=2, marker='s', ms=3); a.set_xscale('log', base=2); a.set_xticks([]); a.minorticks_off()
chip(x + PW / 2, 46.8, 'Reliability: 5 seeds, bootstrap')
chip(x + PW / 2, 42.5, 'Explainability: SHAP vs oracle')
chip(x + PW / 2, 38.2, 'Generalizability: unseen data')

# ---- bottom: controls and findings ----
panel(1.5, 1.2, 49.5, 29.5, GREY, 'Controls')
I4 = pd.read_csv(f'{OUT}/tab_gmp_injection.csv'); I0 = I4[I4.phi == 0]
a = mini(4, 4.6, 21, 15.5, 'Fig. 4a')
for b, col in [(0, BLUE), (0.02, ORANGE), (0.05, GREEN)]:
    d = I0[I0.bias == b]; a.plot(d.noise_sd, d.rmse, color=col, lw=2); a.plot(d.noise_sd, d.LB, color=col, lw=2, ls=':')
NA = pd.read_csv(f'{OUT}/tab_nasa.csv')
a = mini(28, 4.6, 20.5, 15.5, 'Table S2'); xx = np.arange(len(NA)); a.bar(xx - 0.2, NA.rmse, 0.4, color=ORANGE); a.bar(xx + 0.2, NA.rmse_q, 0.4, color=BLUE)
ax.text(14.5, 22.4, 'GMP: simulated truth', ha='center', fontsize=10.8, color=NAVY)
ax.text(38.25, 22.4, 'NASA: regeneration', ha='center', fontsize=10.8, color=NAVY)
arrow((26.2, 32.6), (26.2, 31.0), c=GREY)

panel(53, 1.2, 105.5, 29.5, BLUE, 'What a battery operator learns: question \u2192 section \u2192 answer')
_S = pd.read_csv(f'{OUT}/tab_stats_offline.csv')   # median share of squared error seen, range over the 11 configurations (Table 4)
SEEN = f'{100 * _S.removed_cell.min():.1f}\u2013{100 * _S.removed_cell.max():.1f}%'
F = [('Q1', 'Error on a new chemistry or protocol?', 'Sec. 3.1', 'median 2.4\u20138.0% SOH; domain > model'),
     ('Q2', 'Can the error be bounded without labels?', 'Sec. 3.2', f'LB \u2264 RMSE in 4,257/4,257; sees {SEEN}'),
     ('Q3', 'Can risky cells be flagged in advance?', 'Sec. 3.3', 'domains yes; cells no (\u03c1 \u2264 0.33)'),
     ('Q4', 'How many capacity checks are needed?', 'Sec. 3.4', '3 checks: 2.41 \u2192 1.86% SOH'),
     ('Q5', 'Are the findings trustworthy?', 'Sec. 3.5\u20133.7', 'seeds \u00b10.08; W = 0.74; unseen datasets')]
for i_, (o, t, st, m) in enumerate(F):
    yy = 22.2 - i_ * 4.3
    ax.text(55.2, yy, o, ha='left', va='center', fontsize=11.5, color=INK, bbox=dict(boxstyle='round,pad=0.25', fc='#dde8f8', ec='none'))
    ax.text(61.2, yy, t, ha='left', va='center', fontsize=11, color=INK)
    ax.text(101.5, yy, st, ha='left', va='center', fontsize=10.5, color=NAVY)
    ax.text(115.5, yy, m, ha='left', va='center', fontsize=10, color=NAVY)
arrow((118, 32.6), (118, 31.0))
fig.savefig(f'{FIG}/Figure_1.png', facecolor='white'); print('methodology figure done')
