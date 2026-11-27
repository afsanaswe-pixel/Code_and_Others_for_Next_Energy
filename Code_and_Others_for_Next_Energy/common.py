"""Shared data loading and the monotone-projection certificate.

Data: processed per-cycle charge features of 387 public cells (XJTU, TJU, HUST, MIT)
as redistributed in github.com/wang-fujin/PINN4SOH (Wang et al., 2024, Nat. Commun.).
"""
import os, glob, subprocess
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'outputs'); FIG = os.path.join(ROOT, 'figures')
for d in (DATA, OUT, FIG): os.makedirs(d, exist_ok=True)
PINN_REPO = 'https://github.com/wang-fujin/PINN4SOH.git'
PINN_COMMIT = 'cc3cc5053caf38f04e0665f7f88cb109144d035e'  # version used in the paper (2 Sep 2026)
QUICK = os.environ.get('MONO_QUICK', '0') == '1'
SEED = 0


def fetch_public():
    """Clone the public processed datasets once (about 265 MB)."""
    tgt = os.path.join(DATA, 'PINN4SOH')
    if not os.path.isdir(os.path.join(tgt, 'data')):
        subprocess.run(['git', 'clone', '-q', PINN_REPO, tgt], check=True)
        subprocess.run(['git', '-C', tgt, 'checkout', '-q', PINN_COMMIT], check=True)
    head = subprocess.run(['git', '-C', tgt, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if head and head != PINN_COMMIT:
        print(f'WARNING: data repository at {head[:10]}, paper used {PINN_COMMIT[:10]}')
    return os.path.join(tgt, 'data')


def load_cells():
    """Return one long DataFrame: one row per cycle, with columns
    16 raw features, 16 self-referenced features (d_*), soh, k (cycle index), cell, dom, dataset."""
    B = fetch_public() + '/'
    cells = []
    def add(ds, dom, f):
        d = pd.read_csv(f).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        feat = [c for c in d.columns if c != 'capacity']
        d['soh'] = d.capacity / d.capacity.iloc[:5].median()
        base = d[feat].iloc[:10].mean()
        for c in feat:
            d['d_' + c] = d[c] - base[c]
        d['k'] = np.arange(len(d)); d['cell'] = f'{dom}/{os.path.basename(f)}'
        d['dom'] = dom; d['dataset'] = ds
        cells.append(d)
    for f in sorted(glob.glob(B + 'XJTU data/*.csv')): add('XJTU', 'XJTU-' + os.path.basename(f).split('_battery')[0], f)
    for f in sorted(glob.glob(B + 'HUST data/*.csv')): add('HUST', 'HUST', f)
    for f in sorted(glob.glob(B + 'MIT data/*/*.csv')): add('MIT', 'MIT-' + f.split('/')[-2], f)
    for f in sorted(glob.glob(B + 'TJU data/*/*.csv')): add('TJU', 'TJU-' + f.split('/')[-2][:9], f)
    D = pd.concat(cells, ignore_index=True)
    F = [c for c in D.columns if c not in ('capacity', 'soh', 'k', 'cell', 'dom', 'dataset') and not c.startswith('d_')]
    return D, F


_iso = IsotonicRegression(increasing=False)


def project(p):
    """Euclidean projection of a trajectory onto non-increasing sequences (pool-adjacent-violators)."""
    return _iso.fit_transform(np.arange(len(p)), p)


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def certificate(p, y=None):
    """Return dict with projected trajectory q and label-free lower bound LB = RMS(p - q).
    Theorem (Pythagorean inequality for projections onto a closed convex set C):
    if y in C then ||p - y||^2 >= ||p - q||^2 + ||q - y||^2,
    hence LB <= RMSE(p, y) and RMSE(q, y) <= sqrt(RMSE(p, y)^2 - LB^2)."""
    q = project(p); out = dict(q=q, LB=rms(p - q))
    if y is not None:
        out.update(rmse=rms(p - y), rmse_q=rms(q - y))
    return out


def cell_rows(te, tgt, extra=None):
    """Per-cell error and label-free summaries for a held-out domain.
    te must have columns cell, dataset, soh, p (cycle-ordered within cell)."""
    rows = []
    for c, g in te.groupby('cell', sort=False):
        p, y = g.p.values, g.soh.values
        cert = certificate(p, y)
        sm = pd.Series(p).rolling(15, min_periods=1, center=True).median().values
        inc = np.diff(sm)
        r = dict(cell=c, dom=tgt, dataset=g.dataset.iloc[0], n=len(g), rmse=cert['rmse'], rmse_q=cert['rmse_q'], LB=cert['LB'],
                 bias=float(np.mean(p - y)), rebound=float(inc[inc > 0].sum() / (np.abs(inc).sum() + 1e-12)),
                 rough=float(np.std(np.diff(p)) / (np.std(p) + 1e-12)), viol=rms(y - certificate(y)['q']))
        if extra is not None: r.update(extra.get(c, {}))
        rows.append(r)
    return rows


def raw_row_counts():
    """Rows in the public CSV files before and after removal of missing/infinite values (data provenance)."""
    B = fetch_public() + '/'
    fs = glob.glob(B + 'XJTU data/*.csv') + glob.glob(B + 'HUST data/*.csv') + glob.glob(B + 'MIT data/*/*.csv') + glob.glob(B + 'TJU data/*/*.csv')
    out = []
    for f in sorted(fs):
        d = pd.read_csv(f); n0 = len(d); n1 = len(d.replace([np.inf, -np.inf], np.nan).dropna())
        out.append(dict(file=f.split('data/', 1)[1], rows_raw=n0, rows_kept=n1, removed=n0 - n1))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------------------------
# Figure style. The article's figures were drawn in an environment whose matplotlib defaults differed from the
# standard ones in three settings: default font Inter, no font hinting and 200 dpi figures. They are set here
# explicitly, with the fonts shipped in fonts/ (SIL Open Font License), so that every machine (including Colab)
# draws pixel-identical figures. Importing common applies the style.
def apply_figure_style():
    try:
        import matplotlib
        from matplotlib import font_manager
    except ImportError:
        return
    for f in sorted(glob.glob(os.path.join(ROOT, 'fonts', '*.[ot]tf'))):
        font_manager.fontManager.addfont(f)
    matplotlib.rcParams.update({'font.family': ['Inter', 'sans-serif', 'DejaVu Sans'], 'text.hinting': 'no_hinting',
                                'figure.dpi': 200})


apply_figure_style()
