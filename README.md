# How much can a battery management system know about its state of health error without capacity labels?

**Code and data for the article:** *How much can a battery management system know about its state of health error
without capacity labels? A cross-domain evaluation on 387 lithium-ion cells.* This article was submitted to
*Next Energy* (Elsevier).

**Authors:**
- Afsana Begum
- MST. Asfia Binte Ulfat
- Anoy Saha
- A. H. M. Saifullah Sadi
- Md. Shohel Sayeed (corresponding author, shohel.sayeed@mmu.edu.my)
- Zahereel Ishwar Abdul Khalib
- Md. Mahbubur Rahman

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/afsanaswe-pixel/Code_and_Others_for_Next_Energy/blob/main/NE_Colab_notebook.ipynb)

This repository regenerates every table and every figure (Figs. 1–8, Fig. S1 and the
graphical abstract) of the article. 

## What the study does

A state-of-health (SOH) estimator trained on one set of lithium-ion cells is often deployed on cells of another
chemistry, charging protocol or test bench, where capacity is not measured. The study asks how much an operator can
learn about the estimator's error without capacity labels.

Eleven estimators were evaluated with leave-one-domain-out testing on 387 public cells in 13 domains (310,554 cycles).
The estimators range from ridge regression to a physics-informed neural network.

Capacity does not increase over life. Each predicted SOH trajectory is therefore projected onto non-increasing
sequences, and the distance between the prediction and its projection, `LB = RMS(p − q)`, is a provable lower bound on
the true error.

The study also tests:
- four label-free warning signals;
- four conformal error bounds;
- reference capacity checks with schedules that can run online;
- the reliability, explainability and generalizability of the findings.

## Quick start

### Google Colab (no installation)

1. Click **Open in Colab** above.
2. Choose *Runtime → Run all*.

The notebook installs the exact library versions and copies the data from this repository. It then runs the pipeline
in about 10 minutes and ends with:

```
tables identical to the article: 18 of 18
figures pixel-identical to the article: 11 of 11
ALL TABLES AND FIGURES REPRODUCED EXACTLY
```

### Local computer (Python 3.12 or newer)

```bash
git clone https://github.com/afsanaswe-pixel/Code_and_Others_for_Next_Energy.git
cd Code_and_Others_for_Next_Energy
pip install -r requirements.txt
python run_all.py            # reference mode, about 10 minutes
python run_all.py --full     # full mode, retrains every estimator, about 2 hours on 2 CPU cores
```

Linux is recommended for full mode, because `run_all.py` sets the number of CPU cores there.

## Two modes

**Reference mode (default)** reproduces the article exactly.
- It restores the per-cycle predictions of the reported run from `reference/outputs_part1.zip` and
  `outputs_part2.zip`.
- It then recomputes every analysis that does not train an SOH estimator: certificate, warning signals, meta-model,
  reference checks, GMP and NASA controls, statistics and conformal bounds.
- Finally, it redraws every table and figure.
- This was verified in Google Colab and in a clean Python 3.12 environment on Linux.

**Full mode** also retrains all eleven estimators from the public cell data.
- The cell data are downloaded automatically: a `git clone` of
  [PINN4SOH](https://github.com/wang-fujin/PINN4SOH) at commit `cc3cc50`.
- The reported run used two cores of an x86-64 CPU with AVX-512 (Intel Xeon) and single-threaded OpenBLAS.
- The MLP, 1D-CNN and PINN are trained by long chains of floating-point operations. Their results depend on the
  processor type and the number of cores, so on other hardware their rows can differ from the article.
- `22_verify.py` lists every table cell that differs.

## What makes the results identical

| Source of variation | How it is fixed |
|---|---|
| library versions | exact versions in `requirements.txt` (the notebook installs the same) |
| threads | `run_all.py` sets `OPENBLAS_NUM_THREADS=1` and uses 2 CPU cores, as in the reported run |
| random numbers | seed 0 everywhere |
| figure style | `common.py` sets the font (Inter and Liberation Sans, shipped in `fonts/`), hinting and resolution explicitly, so figures do not depend on local matplotlib settings |
| checking | `22_verify.py` compares every table cell as text and every figure by the SHA-256 of its decoded pixels |


## Analysis steps

| Script | What it does | Article items |
|---|---|---|
| `00_nasa_features.py` | builds `data/nasa_cycle_features.csv` from the original NASA `.mat` files (optional; the file is included) | data for Table S2 |
| `01_lodo.py` | leave-one-domain-out estimation: GBR, ridge and MLP with raw and self-referenced features | Table 4, Fig. 2 |
| `02_certificate.py` | monotone projection and certificate, offline and online | Table 4, Table S1, Fig. 3 |
| `03_signals.py` | four label-free warning signals and the certificate as error rankers | Table 5 |
| `04_meta.py` | meta-model of per-cell error from the signals | input to step 17 |
| `05_checks.py` | offline placements of reference capacity checks | Table S4 |
| `06_controls.py` | GMP simulation with known truth; NASA cells with capacity regeneration | Fig. 4, Table S2 |
| `07_figures.py` | draws Figs. 2–6 | Figs. 2–6 |
| `08_workflow.py` | draws the study design | Fig. 1 |
| `09_reliability.py` | five seeds × five training offsets; domain-cluster bootstrap | Table S3, Fig. S1 |
| `10_explain.py` | TreeSHAP importance and its stability | Table 6, Fig. 7a |
| `11_generalize.py` | leave-one-dataset-out; number of training domains | Table S14, Fig. 8 |
| `12_figures_trends.py` | draws Figs. 7, 8 and S1 | Figs. 7, 8, S1 |
| `13_tables.py` | writes every table under the article numbering | all tables |
| `14_strong.py` | tuned GBR, GBR on all cycles, 1D-CNN, PINN with deployable and with published scaling | Table 4, Table S8 |
| `15_stats.py` | domain-level Wilcoxon tests and mixed models, Holm correction | Tables 4, S1, S4 |
| `16_checks_causal.py` | online (causal) reference-check schedules; alarm operating curve | Table S13, Table S5, Fig. 6 |
| `17_conformal.py` | split, weighted, group and hierarchical conformal bounds | Table S12, Table S10, Fig. 5 |
| `18_xai_error.py` | error-centred decomposition against a target-trained oracle | Table 6, Table S11, Fig. 7b |
| `19_diag.py` | data provenance, SOH reference, MLP diagnostics, autocorrelated noise | Tables S6, S7, S9, Fig. 4d |
| `20_nextenergy_assets.py` | collects the figure files under the article numbering | – |
| `21_graphical_abstract.py` | draws the graphical abstract | graphical abstract |
| `22_verify.py` | compares every table cell and figure pixel with `reference/` | – |

### Outputs

- **Tables** go to `outputs/tables/`: `Table4_*` to `Table6_*` and `TableS1_*` to `TableS14b_*`, all collected in
  `all_tables.html`.
- **Figures** go to `figures/NextEnergy/`: `Figure_1.png` to `Figure_8.png`, `Figure_S1.png`, and
  `Graphical_abstract` in .tif, .pdf, .eps and .png.

## Data sources

| Data | Source | In this repository |
|---|---|---|
| XJTU, TJU, HUST and MIT cells, 13 domains, 387 cells | processed release of Wang et al. (2024), [PINN4SOH](https://github.com/wang-fujin/PINN4SOH), commit `cc3cc50`. Original data: XJTU (Wang et al., 2024), TJU (Zhu et al., 2022), HUST (Ma et al., 2022), MIT (Severson et al., 2019) | no; downloaded in full mode |
| NASA B0005, B0006, B0007, B0018 | B. Saha and K. Goebel, Battery Data Set, NASA Ames Prognostics Center of Excellence (2007) | per-cycle features, `data/nasa_cycle_features.csv` |
| GMP 25.6 V, 40 Ah LiFePO4 pack | Ulfat et al., Mendeley Data, v2 (2026), [doi:10.17632/k675pxrd83.2](https://doi.org/10.17632/k675pxrd83.2); described in *Data in Brief* 69 (2026) 113252, [doi:10.1016/j.dib.2026.113252](https://doi.org/10.1016/j.dib.2026.113252) | `data/GMP.csv` |

**Rebuilding the NASA file.** Put `B0005.mat`, `B0006.mat`, `B0007.mat` and `B0018.mat` from the NASA repository in
`data/nasa_raw/` and run `python 00_nasa_features.py`.

The study uses five discharge features per cycle:
- mean voltage and mean current up to the 2.7 V cut-off;
- voltage at 600 s;
- initial and maximum temperature.

Features that measure discharge duration are written to the file but not used, because they are proxies of
capacity.

## Citation

If you use this code, please cite the article (citation details will be added on publication) and the GMP dataset:

> M.A.B. Ulfat, A. Begum, M.A.A. Mamun, S.N.A.E. Azam, M.M. Hassan, I. Mahmud, A.H.M.S. Sadi, M.S. Sayeed, Industrial
> LiFePO4 battery management system dataset for early degradation analysis and remaining useful life prediction,
> Data Brief 69 (2026) 113252. https://doi.org/10.1016/j.dib.2026.113252

## Acknowledgments

- The authors of PINN4SOH, for distributing the processed XJTU, TJU, HUST and MIT cells.
- The NASA Ames Prognostics Center of Excellence, for the battery aging data.
- GMP Lithium Ltd., Bangladesh, for permission to publish the pack dataset.

