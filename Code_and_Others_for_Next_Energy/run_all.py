"""Run the whole pipeline of the Next Energy article and check the result against the article.

    python run_all.py              # reference mode (default, about 10 minutes)
    python run_all.py --full       # full mode: retrain every model from the public data (about 2 hours on 2 CPU cores)

Reference mode restores the per-cycle predictions and per-cell results of the reported run (reference/outputs_part*.zip),
recomputes from them every analysis step that does not train an SOH estimator (certificate, signals, meta-model,
reference checks, controls, statistics, conformal bounds), and redraws every table and figure.
Full mode also retrains all estimators (steps 01, 09, 10, 11, 14, 18, 19) from the public data.
Both modes end with 22_verify.py, which compares every table cell and every figure pixel with the article.

Data: the public XJTU, TJU, HUST and MIT cells are downloaded in full mode (git clone of PINN4SOH, commit cc3cc50);
data/nasa_cycle_features.csv and data/GMP.csv are included. Set MONO_QUICK=1 for a fast smoke test of full mode.
Figures are written to figures/NextEnergy and tables to outputs/tables under the article numbering."""
import glob, os, sys, subprocess, time, zipfile

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
FULL = '--full' in sys.argv

# Same thread layout as the machine that produced the article (2 CPU cores, single-threaded OpenBLAS). Results of
# floating-point training can depend on the number of threads, so a larger machine is restricted to 2 cores.
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
try:
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) > 2:
        os.sched_setaffinity(0, cpus[:2]); print('using 2 of', len(cpus), 'CPU cores, as in the reported run')
    elif len(cpus) < 2:
        print('WARNING: only one CPU core; full mode may differ from the article in the last digits')
except AttributeError:
    print('NOTE: CPU affinity cannot be set on this system; use Linux or Colab for full mode')

# Library versions: the article's numbers were produced with the versions in requirements.txt.
from importlib.metadata import version, PackageNotFoundError
for line in open('requirements.txt'):
    name, want = line.strip().split('==')
    try:
        have = version(name)
    except PackageNotFoundError:
        have = 'not installed'
    if have != want:
        print(f'WARNING: {name} {have} is installed, the article used {want}; results may differ')

TRAIN = ['01_lodo', '14_strong all', '09_reliability', '10_explain', '11_generalize', '18_xai_error', '19_diag']
ANALYSIS = ['02_certificate', '03_signals', '04_meta', '05_checks', '06_controls', '15_stats', '16_checks_causal', '17_conformal']
DRAW = ['07_figures', '12_figures_trends', '13_tables', '08_workflow', '20_nextenergy_assets', '21_graphical_abstract', '22_verify']
if FULL:   # article order: estimators first, then analyses that use their predictions
    steps = ['01_lodo', '14_strong all', '02_certificate', '03_signals', '04_meta', '05_checks', '06_controls', '09_reliability',
             '10_explain', '11_generalize', '15_stats', '16_checks_causal', '17_conformal', '18_xai_error', '19_diag'] + DRAW
else:
    os.makedirs('outputs', exist_ok=True)
    parts = sorted(glob.glob(os.path.join('reference', 'outputs*.zip')))   # two parts, each small enough for GitHub
    if not parts:
        sys.exit('reference/outputs_part1.zip and outputs_part2.zip are missing (see README.md)')
    n = 0
    for part in parts:
        with zipfile.ZipFile(part) as z:
            z.extractall('.'); n += len(z.namelist())
    print('restored', n, 'files of the reported run into outputs/')
    steps = ANALYSIS + DRAW

for s in steps:
    t0 = time.time(); print('===', s, flush=True)
    a = s.split()
    r = subprocess.run([sys.executable, '-u', a[0] + '.py'] + a[1:], capture_output=True, text=True, env=env)
    print('\n'.join(l for l in r.stdout.splitlines() if 'arning' not in l)[-3000:])
    print(f'--- {s} finished in {time.time() - t0:.0f} s', flush=True)
    if r.returncode:
        print(r.stderr[-3000:]); sys.exit(r.returncode)
print('done')
