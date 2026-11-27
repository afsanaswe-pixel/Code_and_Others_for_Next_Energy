"""Step 22. Check the regenerated tables and figures against the article.
reference/tables/*.csv hold the tables of the article exactly as printed by 13_tables.py in the reported run, and
reference/figures.json the pixel fingerprint (SHA-256 of the decoded RGBA pixels) of every figure file of the article.
Every table cell and every figure pixel is compared; differences are listed."""
import os, glob, json, hashlib
import numpy as np, pandas as pd
from PIL import Image

REF = 'reference'; ok_t = ok_f = 0; bad = []
tabs = sorted(glob.glob(os.path.join(REF, 'tables', '*.csv')))
for f in tabs:
    name = os.path.basename(f); new = os.path.join('outputs', 'tables', name)
    if not os.path.exists(new):
        bad.append(f'{name}: missing'); continue
    a = pd.read_csv(f, dtype=str, keep_default_na=False); b = pd.read_csv(new, dtype=str, keep_default_na=False)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        bad.append(f'{name}: shape or columns differ {a.shape} vs {b.shape}'); continue
    diff = [(i, c, a.at[i, c], b.at[i, c]) for c in a.columns for i in range(len(a)) if a.at[i, c] != b.at[i, c]]
    if diff:
        bad.append(f'{name}: {len(diff)} cells differ, e.g. row {diff[0][0]} "{diff[0][1]}": article {diff[0][2]} / now {diff[0][3]}')
    else:
        ok_t += 1


def fingerprint(path):
    im = np.asarray(Image.open(path).convert('RGBA'))
    return f'{im.shape[1]}x{im.shape[0]}:' + hashlib.sha256(im.tobytes()).hexdigest()


figs = json.load(open(os.path.join(REF, 'figures.json')))
for rel, fp in figs.items():
    if not os.path.exists(rel):
        bad.append(f'{rel}: missing'); continue
    now = fingerprint(rel)
    if now == fp:
        ok_f += 1
    else:
        bad.append(f'{rel}: pixels differ from the article ({now.split(":")[0]} vs {fp.split(":")[0]})')

print(f'tables identical to the article: {ok_t} of {len(tabs)}')
print(f'figures pixel-identical to the article: {ok_f} of {len(figs)}')
for b in bad:
    print('  DIFFERENT:', b)
if not bad:
    print('ALL TABLES AND FIGURES REPRODUCED EXACTLY')
