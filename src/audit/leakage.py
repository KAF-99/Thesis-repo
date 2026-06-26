"""Leakage audit for the stacked feature panel.

Recomputes each sampled feature row from ``df_raw.loc[<=t]`` only, and compares
it to the value stored in the panel (which was built from the full history). A
feature that uses any information from ``> t`` would differ — so an exact match
(within ``1e-6``) certifies the feature is observable at decision time.

Also exposes :func:`timing_label`, the timing-annotation helper used to
describe each feature's observability (shift(1) / contemporaneous /
ffill+shift(1)).
"""

import numpy as np
import pandas as pd

from src.features.security import features_for_security


def leakage_audit(panel: pd.DataFrame, df_raw: pd.DataFrame, n_samples: int = 30,
                  rng_seed: int = 42, skip_first: int = 500) -> bool:
    """Recompute features from past-only data and compare to stored panel values.

    For ``n_samples`` randomly sampled panel rows (after ``skip_first`` warm-up
    rows), rebuild the feature row from ``df_raw`` truncated to ``<= t`` and
    compare against the stored value within a ``1e-6`` tolerance. Prints a
    per-feature FAIL list plus a PASS/FAIL/SKIP summary and returns ``True``
    iff every checked feature is leakage-free.
    """
    _META     = {'date', 'security', 'y', 'level'}
    feat_cols = [c for c in panel.columns if c not in _META]

    eligible = panel.iloc[skip_first:].copy()
    sample   = eligible.sample(n=min(n_samples, len(eligible)), random_state=rng_seed)

    TOL          = 1e-6
    fail_counts  = {c: 0 for c in feat_cols}
    check_counts = {c: 0 for c in feat_cols}

    for _, row in sample.iterrows():
        t       = row['date']
        sec     = row['security']
        country, maturity = sec.rsplit('_', 1)
        past = df_raw.loc[df_raw.index <= t]
        if sec not in past.columns or past[sec].isna().all():
            continue
        feats_at_t = features_for_security(sec, past, country, maturity).iloc[-1]

        for col in feat_cols:
            stored = row[col]
            recomp = feats_at_t.get(col, np.nan)
            check_counts[col] += 1
            if pd.isna(stored) and pd.isna(recomp):
                continue
            if pd.isna(stored) != pd.isna(recomp):
                fail_counts[col] += 1
                continue
            if abs(float(stored) - float(recomp)) > TOL:
                fail_counts[col] += 1

    n_fail = sum(1 for c in feat_cols if fail_counts[c] > 0)
    n_skip = sum(1 for c in feat_cols if check_counts[c] == 0)
    n_pass = len(feat_cols) - n_fail - n_skip

    print(f'=== Leakage audit — {len(feat_cols)} features, '
          f'{n_samples} sample rows ===\n')

    for col in feat_cols:
        if fail_counts[col] > 0:
            n = check_counts[col]
            print(f'  FAIL  {col}  {n-fail_counts[col]}/{n} OK  '
                  f'← {fail_counts[col]} mismatch(es)')

    print(f'\n  Summary: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP')
    if n_fail == 0:
        print(f'\n[PASS] All {n_pass} checked features leakage-free.')
        print(f'  VIX/MOVE/equity/commodity: shift(1) in add_macro_features.')
        print(f'  Cross-market swap rates: shift(1) in features_for_security.')
    else:
        print(f'\n[FAIL] {n_fail} feature(s) have mismatches — fix before full run.')

    return n_fail == 0


def timing_label(fname: str) -> str:
    """Human-readable observability annotation for a feature column name.

    Returns the timing class (shift(1) / contemporaneous / ffill+shift(1)) used
    by the feature-matrix display; ``'?'`` for an unrecognised name.
    """
    if fname.startswith('chg_lag'):
        return 'backward AR lag (chg_1d shifted k days; own security)'
    if fname in ('carry_roll', 'mom_63d', 'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m',
                 'dev_ma_3m', 'dev_ma_12m', 'vol_20d', 'vol_regime', 'vol_zscore',
                 'level_zscore', 'slope_10_1', 'slope_10_5', 'slope_5_1'):
        return 'contemporaneous (own security daily close)'
    if fname in ('ibor_level', 'ibor_chg_1d', 'ibor_mom_1m', 'ibor_term_slope',
                 'swap_ibor_basis', 'sofr_mom_1m', 'eur_3m', 'eur_1w',
                 'nor_ibor_1m', 'nor_ibor_6m'):
        return 'contemporaneous (IBOR / same-session fix)'
    if any(fname.startswith(p) for p in ('move_', 'vix_', 'v2x_', 'vxn_', 'rvx_',
                                          'ovx_', 'gvz_')):
        return 'shift(1): vol index, prior close'
    if any(fname.startswith(p) for p in ('spx_', 'sx5e_', 'mxwo_', 'ndx_')):
        return 'shift(1): equity close, prior day'
    if any(fname.startswith(p) for p in ('oil_', 'copper_', 'opec_', 'natgas_')):
        return 'shift(1): commodity close, prior day'
    if any(fname.startswith(p) for p in ('be5y_', 'be10y_', 'be_slope', 'ig_', 'hy_')):
        return 'shift(1): TIPS/credit market, prior day'
    if any(fname.startswith(p) for p in ('cpi_', 'pce_', 'nfp_', 'pmi_', 'unemp_')):
        return 'ffill+shift(1): monthly release, prior value'
    if fname.startswith('xm_'):
        return 'shift(1): cross-market swap, prior close'
    if fname.startswith('nor_'):
        if any(fname.startswith(p) for p in ('nor_eurnok', 'nor_usdnok', 'nor_i44')):
            return 'shift(1): NOK FX / krone index, prior close'
        if any(fname.startswith(p) for p in ('nor_kpi', 'nor_kpijae', 'nor_unemp')):
            return 'ffill+shift(1): SSB monthly release, prior value'
        return 'contemporaneous (NOK official / administered same-session)'
    if fname == 'swap_govt_spread':
        return 'contemporaneous (swap − govvie, same-session)'
    return '?'
