"""Step 13. Export every data table of the Next Energy article as CSV files and one HTML page, under the article's
numbering: main Tables 4, 5 and 6 and Supplementary Tables S1-S14. Text-only tables (1-3 and 7) are in the manuscript."""
import os, numpy as np, pandas as pd
from common import OUT
T = os.path.join(OUT, 'tables'); os.makedirs(T, exist_ok=True)
for f in os.listdir(T): os.remove(os.path.join(T, f))
pc = lambda x: np.round(100 * np.asarray(x, float), 2)
LAB = {'gbr_self': 'GBR, self-ref.', 'gbr_raw': 'GBR, raw', 'mlp_self': 'MLP, self-ref.', 'mlp_raw': 'MLP, raw',
       'ridge_self': 'Ridge, self-ref.', 'ridge_raw': 'Ridge, raw', 'gbrtuned_self': 'Tuned GBR, self-ref.',
       'gbrfull_self': 'GBR, self-ref., all cycles', 'cnn_self': '1D-CNN, self-ref.', 'pinn': 'PINN (Wang et al., 2024)', 'pinnpub': 'PINN, published per-cell scaling (non-deployable)'}
fp = lambda p: '< 0.001' if p < 0.001 else f'{p:.3f}'
tabs = {}

# ---- Table 4: estimators and certificate, with clustering-aware inference
S = pd.read_csv(f'{OUT}/tab_stats_offline.csv'); O = pd.read_csv(f'{OUT}/tab_certificate_offline.csv').set_index('cfg')
tabs['Table4_estimators_certificate'] = pd.DataFrame({
    'Configuration': S.cfg.map(LAB),
    'Median RMSE, cells / domains (%)': [f'{100*a:.2f} / {100*b:.2f}' for a, b in zip(S.med_rmse_cell, S.med_rmse_domainw)],
    'After projection, cells / domains (%)': [f'{100*a:.2f} / {100*b:.2f}' for a, b in zip(S.med_rmse_q_cell, S.med_rmse_q_domainw)],
    'Cells improved (%)': pc(O.loc[S.cfg].improved), 'Domains improved (of 13)': S.domains_improved,
    'LB <= RMSE, % (95% CI)': [f'{100*a:.1f} ({100*b:.1f}-{100*c:.1f})' for a, b, c in zip(S.lb_valid, S.lb_valid_lo, S.lb_valid_hi)],
    'Squared error removed, median % (95% CI)': [f'{100*a:.1f} ({100*b:.1f}-{100*c:.1f})' for a, b, c in zip(S.removed_cell, S.removed_lo, S.removed_hi)],
    'p, domain-level Wilcoxon (Holm)': [fp(p) for p in S.p_domain_wilcoxon_holm], 'p, mixed model (Holm)': [fp(p) for p in S.p_mixed_holm]})

# ---- Table 5: label-free signals
s = pd.read_csv(f'{OUT}/tab_signals.csv'); f2 = lambda a, b, c: [f'{x:.2f} ({y:.2f}, {z:.2f})' for x, y, z in zip(a, b, c)]
tabs['Table5_signals'] = pd.DataFrame({'Signal': s.signal, 'rho pooled': f2(s.rho, s.rho_lo, s.rho_hi), 'rho within domain': f2(s.rho_within, s.rho_within_lo, s.rho_within_hi),
    'AUROC >2%': f2(s.auc2, s.auc2_lo, s.auc2_hi), 'AUROC >5%': f2(s.auc5, s.auc5_lo, s.auc5_hi)})

# ---- Table S12: conformal variants
c = pd.read_csv(f'{OUT}/tab_conformal.csv'); NM = {'split': 'Split (pooled cells)', 'weighted': 'Weighted (covariate shift)', 'family': 'Group (dataset family)', 'hier': 'Hierarchical (domain level)'}
tabs['TableS12_conformal'] = pd.DataFrame({'Calibration': c.variant.map(NM), 'Overall coverage (%)': pc(c.coverage), 'Mean domain coverage (%)': pc(c.domain_mean_coverage),
    'Lowest domain (%)': [f'{100*a:.0f} ({b})' for a, b in zip(c.min_domain, c.min_domain_name)], 'Domains below 80%': c.domains_below_80,
    'Median bound / RMSE': c.width_median.round(2), 'Infinite bounds (%)': pc(c.infinite_bounds)})

# ---- Table S13: causal reference-check schedules
k = pd.read_csv(f'{OUT}/tab_checks_causal.csv'); piv = k.pivot_table(index='K', columns='schedule', values='median')
ev = k[k.schedule == 'event'].set_index('K'); fx = k[k.schedule == 'fixed'].set_index('K')
tabs['TableS13_checks_causal'] = pd.DataFrame({'K': piv.index, 'Fixed interval (%)': pc(piv['fixed'].fillna(piv['none'])),
    'Event-triggered (%)': pc(piv['event'].fillna(piv['none'])), 'Checks used, event (mean)': ev.checks_used.reindex(piv.index).round(2).fillna(0),
    'Periodic, known life (reference, %)': pc(piv['periodic_known'].fillna(piv['none'])),
    'Fixed vs none: domains better, p (Holm)': [('-' if K == 0 else f'{int(fx.loc[K, "domains_better_than_none"])}/13, {fp(fx.loc[K, "p_vs_none_holm"])}') for K in piv.index],
    'Event vs fixed: domains better, p (Holm)': [('-' if K == 0 else f'{int(ev.loc[K, "domains_better_than_fixed"])}/13, {fp(ev.loc[K, "p_vs_fixed_holm"])}') for K in piv.index]}).reset_index(drop=True)

# ---- Table 6: explainability
x = pd.read_csv(f'{OUT}/tab_xai.csv').iloc[0]; xe = pd.read_csv(f'{OUT}/tab_xai_error.csv').iloc[0]
I = pd.read_csv(f'{OUT}/xai_importance.csv', index_col=0); sh = I.div(I.sum(axis=1), axis=0).mean()
E = pd.read_csv(f'{OUT}/xai_error_domains.csv')
tabs['Table6_explainability'] = pd.DataFrame({'Quantity': [
    "Kendall's W of feature ranking, 13 models (permutation null mean; p)", 'Mean pairwise Spearman of importance: leave-one-domain-out models',
    'Mean (range) Spearman of importance: models trained on disjoint halves of the domains', 'Mean importance share: CC Q / CV Q (%)',
    'Oracle RMSE on the target domain, median (%)', 'Share of the deployed-minus-oracle gap from the expected-output (label-level) term, median (%)',
    'Domains in which the top error feature equals the top prediction-shift feature', 'Spearman of error terms with prediction-shift terms, median'],
    'Value': [f'{x.kendall_W:.3f} ({xe.W_null_mean:.3f}; {fp(xe.W_null_p)})', f'{xe.rho_lodo_pairs_mean:.2f}',
              f'{xe.rho_disjoint_halves_mean:.2f} ({xe.rho_disjoint_halves_min:.2f}-{xe.rho_disjoint_halves_max:.2f})', f'{100*sh["CC Q"]:.1f} / {100*sh["CV Q"]:.1f}',
              f'{100*xe.median_oracle_rmse:.2f}', f'{100*xe.median_base_share:.0f}', f'{int(round(xe.top_feature_agreement*len(E)))} of {len(E)}', f'{xe.median_rho_error_vs_shift:.2f}']})

# ---- Table S14: generalizability
l = pd.read_csv(f'{OUT}/tab_lodso.csv'); kk = pd.read_csv(f'{OUT}/tab_ndomains.csv')
tabs['TableS14a_leave_one_dataset_out'] = pd.DataFrame({'Features': l.ref, 'Held-out dataset': l.dataset, 'Cells': l.cells,
    'Leave-one-domain-out (%)': pc(l.med_rmse_lodo), 'Leave-one-dataset-out (%)': pc(l.med_rmse), 'After projection (%)': pc(l.med_rmse_q),
    'Cells improved (%)': pc(l.improved), 'LB valid (%)': pc(l.lb_valid)})
tabs['TableS14b_training_domains'] = pd.DataFrame({'Training domains k': kk.k, 'Median of domain medians (%)': pc(kk.med_rmse), 'IQR (%)': [f'{100*a:.2f}-{100*b:.2f}' for a, b in zip(kk.q25, kk.q75)],
    'After projection (%)': pc(kk.med_rmse_q), 'LB valid (%)': pc(kk.lb_valid), 'Squared error removed (%)': pc(kk.removed)})

# ================= Supplementary =================
n = pd.read_csv(f'{OUT}/tab_certificate_online.csv'); so = pd.read_csv(f'{OUT}/tab_stats_online.csv').set_index('cfg').loc[n.cfg]
tabs['TableS1_online'] = pd.DataFrame({'Configuration': n.cfg.map(LAB), 'Median online RMSE (%)': pc(n.med_raw), 'After prefix projection (%)': pc(n.med_q),
    'Cells improved (%)': pc(n.improved), 'Domains improved (of 13)': so.domains_improved.values, 'Prefix LB holds (%)': pc(n.lb_holds),
    'p, domain-level Wilcoxon (Holm)': [fp(p) for p in so.p_domain_wilcoxon_holm], 'p, mixed model (Holm)': [fp(p) for p in so.p_mixed_holm]})
na = pd.read_csv(f'{OUT}/tab_nasa.csv')
tabs['TableS2_nasa'] = pd.DataFrame({'Learner': na.learner, 'Cell': na.cell, 'RMSE (%)': pc(na.rmse), 'After projection (%)': pc(na.rmse_q),
    'LB (%)': pc(na.LB), 'delta (%)': pc(na.viol), 'Projection helped': np.where(na.improved, 'Yes', 'No')})
r = pd.read_csv(f'{OUT}/rel_runs.csv'); g = r.groupby('cfg')
tabs['TableS3_reliability'] = pd.DataFrame({'Configuration': [LAB[k_] for k_ in g.groups],
    'Median RMSE, mean +/- SD (%)': [f'{100*v.med_rmse.mean():.2f} +/- {100*v.med_rmse.std():.2f}' for _, v in g],
    'After projection, mean +/- SD (%)': [f'{100*v.med_rmse_q.mean():.2f} +/- {100*v.med_rmse_q.std():.2f}' for _, v in g],
    'Cells improved, mean (%)': [pc(v.improved.mean()) for _, v in g], 'LB valid, mean (%)': [pc(v.lb_valid.mean()) for _, v in g],
    'Squared error removed, mean (%)': [pc(v.removed.mean()) for _, v in g]})
c9 = pd.read_csv(f'{OUT}/tab_checks.csv'); pv = c9.pivot_table(index='K', columns='placement', values='median').reindex(columns=['none', 'periodic', 'fade', 'departure'])
st = pd.read_csv(f'{OUT}/tab_stats_checks.csv'); dp = st[st.placement == 'departure'].set_index('K')
tabs['TableS4_checks_offline'] = pd.DataFrame({'K': pv.index, 'Periodic (%)': pc(pv.periodic.fillna(pv['none'])), 'Equal-fade (%)': pc(pv.fade), 'Departure (%)': pc(pv.departure),
    'Departure vs periodic: domains better, p (Holm)': [('-' if K_ == 0 else f'{int(dp.loc[K_, "domains_better"])}/13, {fp(dp.loc[K_, "p_domain_wilcoxon_holm"])}') for K_ in pv.index]}).reset_index(drop=True)
A = pd.read_csv(f'{OUT}/tab_alarm.csv')
tabs['TableS5_alarm'] = pd.DataFrame({'Mode': A['mode'], 'Threshold theta (%)': pc(A.theta), 'Event': A.event, 'Alarm rate (%)': pc(A.alarm_rate),
    'Prevalence (%)': pc(A.prevalence), 'Precision (%)': pc(A.precision), 'Recall (%)': pc(A.recall)})
sr = pd.read_csv(f'{OUT}/tab_soh_reference.csv'); md = pd.read_csv(f'{OUT}/tab_mlp_diag.csv')
tabs['TableS6_soh_reference'] = pd.DataFrame({'SOH reference': sr.soh_reference, 'Median RMSE (%)': pc(sr.med_rmse), 'After projection (%)': pc(sr.med_rmse_q),
    'Cells improved (%)': pc(sr.improved), 'LB valid (%)': pc(sr.lb_valid), 'Squared error removed (%)': pc(sr.removed), 'Largest initial SOH': sr.max_initial_soh.round(3)})
tabs['TableS7_mlp_training'] = md.groupby('cfg').agg(folds=('dom', 'size'), iterations_min=('n_iter', 'min'), iterations_median=('n_iter', 'median'),
    iterations_max=('n_iter', 'max'), convergence_warnings=('converged_warning', 'sum')).reset_index().rename(columns={'cfg': 'Configuration'})
hp = {}
for cfg, cols in [('gbrtuned_self', ['hp_learning_rate', 'hp_max_leaf_nodes', 'hp_max_iter', 'hp_l2_regularization']), ('cnn_self', ['hp_W', 'hp_lr', 'epochs']),
                  ('pinn', ['epochs']), ('pinnpub', ['epochs'])]:
    p_ = f'{OUT}/cells_{cfg}.csv'
    if os.path.exists(p_):
        hp[cfg] = pd.read_csv(p_).groupby('dom')[cols].first()
if hp:   # layout of Table S8: one row per held-out domain
    doms_ = sorted(set().union(*[set(v.index) for v in hp.values()]))
    g_ = lambda cfg, c_: hp[cfg][c_].reindex(doms_).values if cfg in hp else [None] * len(doms_)
    tabs['TableS8_selected_hyperparameters'] = pd.DataFrame({'Held-out domain': doms_,
        'Tuned GBR: learning rate': g_('gbrtuned_self', 'hp_learning_rate'), 'Tuned GBR: max. leaf nodes': g_('gbrtuned_self', 'hp_max_leaf_nodes'),
        'Tuned GBR: max. iterations': g_('gbrtuned_self', 'hp_max_iter'), 'Tuned GBR: L2': g_('gbrtuned_self', 'hp_l2_regularization'),
        '1D-CNN: W (cycles)': g_('cnn_self', 'hp_W'), '1D-CNN: learning rate': g_('cnn_self', 'hp_lr'), '1D-CNN: epochs': g_('cnn_self', 'epochs'),
        'PINN epochs: deployable': g_('pinn', 'epochs'), 'PINN epochs: published scaling': g_('pinnpub', 'epochs')})
pr = pd.read_csv(f'{OUT}/data_provenance.csv')
tabs['TableS9_data_provenance'] = pr[pr.removed > 0].rename(columns={'file': 'File', 'rows_raw': 'Rows in source', 'rows_kept': 'Rows kept', 'removed': 'Removed (missing/infinite)'})
cd_ = pd.read_csv(f'{OUT}/tab_conformal_domains.csv')   # layout of Table S10: coverage (Clopper-Pearson 95% interval) per domain and calibration
cd_['txt'] = [f'{100*a_:.2f} ({100*b_:.2f}-{100*c_:.2f})' for a_, b_, c_ in zip(cd_.coverage, cd_.lo, cd_.hi)]
w_ = cd_.pivot(index='dom', columns='variant', values='txt')
tabs['TableS10_conformal_domains'] = pd.DataFrame({'Domain': w_.index, 'Cells': cd_.groupby('dom').cells.first().reindex(w_.index).values,
    **{NM[v_]: w_[v_].values for v_ in ['split', 'weighted', 'family', 'hier'] if v_ in w_}})
tabs['TableS11_xai_error_domains'] = E.assign(**{c_: pc(E[c_]) for c_ in ['bias', 'oracle_rmse', 'gap_src_minus_oracle', 'base_term', 'feature_terms']}).round(3)

ORDER = ['Table4_estimators_certificate', 'Table5_signals', 'Table6_explainability'] + sorted(
    [k_ for k_ in tabs if k_.startswith('TableS')], key=lambda s_: (int(''.join(ch for ch in s_.split('_')[0][6:] if ch.isdigit())), s_))
tabs = {k_: tabs[k_] for k_ in ORDER}
html = ['<html><head><meta charset="utf-8"><style>body{font-family:Arial;}table{border-collapse:collapse;margin:12px 0 28px}td,th{border:1px solid #999;padding:4px 8px;font-size:13px}th{background:#e7eef7}</style></head><body>']
for name, df in tabs.items():
    df.to_csv(f'{T}/{name}.csv', index=False); html += [f'<h3>{name.replace("_", " ")}</h3>', df.to_html(index=False)]
open(f'{T}/all_tables.html', 'w').write('\n'.join(html + ['</body></html>'])); print('tables written:', list(tabs))
