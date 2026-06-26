"""Shared evaluation harness (single source of truth).

Every definition that governs the evaluation protocol is copied **verbatim** from
``model_htboost_v5.ipynb`` so that the simple-rule notebooks
(``model_mean_reversion.ipynb`` / ``model_momentum.ipynb``) and the linear
benchmark are scored on exactly the same harness as HTBoost v5, and their
long-CSV rows merge with ``v5_metrics_long.csv`` with zero reconciliation.

The HTBoost v5 notebook itself is **not** modified and does **not** import from
this module, so v5's numerical output is unchanged by construction. The metric
maths below (``clark_west``, ``dm_harvey``, ``pesaran_timmermann``,
``_hac_mean_tstat``, ``config_hash``, ``_score``, ``SHARED_COLS``,
``compute_metrics_row``) are byte-for-byte the v5 definitions; provenance fields
(notebook / run_ts / model_kind / is_pooled) are read from ``meta`` rather than
module globals so each notebook can stamp its own ids.

The run-protocol constants (``WF_FOLDS``, ``H_GRID``, ``NOR_TENORS``,
``MIN_TRAIN_OBS``, ``MIN_TEST_OBS``, ``ALPHA_MT``, ``FOLD_NAMES``, ``REGIMES``)
are now imported from :mod:`src.config` instead of being redefined here.
"""

import json
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm
from scipy.stats import t as _t_dist
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.metrics import r2_score

from src.config import (
    WF_FOLDS, FOLD_NAMES, REGIMES, H_GRID, NOR_TENORS,
    MIN_TRAIN_OBS, MIN_TEST_OBS, ALPHA_MT,
)


def OVERLAP_FOR_H(h):
    """Label-overlap purge length (business days). ALWAYS H-1 per horizon (v5 invariant)."""
    return h - 1


def wf_purge_split(panel, test_start, test_end, H, date_col='date'):
    """Causal train/test split for one walk-forward fold, with the v5 label-overlap
    purge applied at the boundary. Copied from v5 ``run_security_v5``:

        purge_ts = ts_ts - BDay(H-1)              # OVERLAP = H-1
        tr = panel[date < purge_ts]
        te = panel[ts_ts <= date <= te_ts]

    Overlapping h-day labels reaching into the test window are removed from train.
    Returns (train_df, test_df).
    """
    ts_ts, te_ts = pd.Timestamp(test_start), pd.Timestamp(test_end)
    purge_ts = ts_ts - pd.tseries.offsets.BDay(H - 1)          # OVERLAP = H-1
    tr = panel[panel[date_col] < purge_ts].copy()
    te = panel[(panel[date_col] >= ts_ts) & (panel[date_col] <= te_ts)].copy()
    return tr, te


# ── Metric functions (copied verbatim from v5 cell 14) ───────────────────────
# RW benchmark forecasts the h-day change as 0 ⇒ rw error = y, mse_rw = mean(y²).
# Overlapping h>1 returns are handled with HAC/Newey–West variance (lag = H−1) in
# the Clark–West and Diebold–Mariano–Harvey tests, and an effective-sample-size
# note (n_eff ≈ n/H) for the binomial / Pesaran–Timmermann directional tests.

def _hac_mean_tstat(z, lag):
    """t-stat of mean(z)=0 with Newey–West HAC variance (lag≥0). Returns (mean, tstat, n)."""
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    n = z.size
    if n < 5 or np.allclose(z, z[0]):
        return (float(np.mean(z)) if n else np.nan, np.nan, n)
    X = np.ones((n, 1))
    try:
        res = sm.OLS(z, X).fit(cov_type='HAC', cov_kwds={'maxlags': max(int(lag), 0)})
        return float(res.params[0]), float(res.tvalues[0]), n
    except Exception:
        return float(np.mean(z)), np.nan, n


def clark_west(y, yhat, H):
    """Clark–West vs RW(=0). f_t = y² − (y−ŷ)² + ŷ²; one-sided p (model better)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    if y.size < 5:
        return (np.nan, np.nan)
    f = y**2 - (y - yhat)**2 + yhat**2
    _, t, _ = _hac_mean_tstat(f, H - 1)
    if not np.isfinite(t):
        return (np.nan, np.nan)
    return (t, float(1 - norm.cdf(t)))           # one-sided: H1 = model beats RW


def dm_harvey(y, yhat, H):
    """Diebold–Mariano with Harvey–Leybourne–Newbold small-sample correction, HAC lag=H−1.
    Loss differential d_t = e_rw² − e_m² = y² − (y−ŷ)²  (d>0 ⇒ model better). One-sided p."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    n = y.size
    if n < 5:
        return (np.nan, np.nan)
    d = y**2 - (y - yhat)**2
    _, dm, _ = _hac_mean_tstat(d, H - 1)
    if not np.isfinite(dm):
        return (np.nan, np.nan)
    h = max(int(H), 1)
    hln = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))   # HLN factor
    dm_star = dm * hln
    return (float(dm_star), float(_t_dist.sf(dm_star, df=n - 1)))      # one-sided: model better


def pesaran_timmermann(y, yhat):
    """Pesaran–Timmermann (1992) directional-accuracy test. One-sided p (predictability)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    n = y.size
    if n < 5:
        return (np.nan, np.nan)
    Dy = (y > 0).astype(float)
    Dx = (yhat > 0).astype(float)
    P = float(np.mean(Dy == Dx))
    Py, Px = float(np.mean(Dy)), float(np.mean(Dx))
    Pstar = Py * Px + (1 - Py) * (1 - Px)
    var_P = Pstar * (1 - Pstar) / n
    var_Pstar = (((2 * Py - 1) ** 2) * Px * (1 - Px) / n
                 + ((2 * Px - 1) ** 2) * Py * (1 - Py) / n
                 + 4 * Py * Px * (1 - Py) * (1 - Px) / (n ** 2))
    denom = var_P - var_Pstar
    if denom <= 0:
        return (np.nan, np.nan)
    pt = (P - Pstar) / np.sqrt(denom)
    return (float(pt), float(1 - norm.cdf(pt)))


def _score(y, yhat):
    """v4-compatible DirAcc/R²/n (kept for back-compat with smoke/Part-C prints)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 5:
        return None
    return {'dir_acc': float(np.mean(np.sign(y[m]) == np.sign(yhat[m]))),
            'r2': float(r2_score(y[m], yhat[m])), 'n_obs': int(m.sum())}


def config_hash(cfg, extra=''):
    payload = json.dumps({k: cfg.get(k) for k in sorted(cfg)} if isinstance(cfg, dict) else cfg,
                         sort_keys=True, default=str) + '|' + str(extra)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


# Frozen list of shared schema columns (the pooled-comparability contract).
# Copied verbatim from v5 — DO NOT rename or drop a column.
SHARED_COLS = [
    'notebook', 'run_ts', 'model_kind', 'is_pooled', 'validation_scheme', 'target_kind',
    'security', 'country', 'tenor', 'horizon', 'fold', 'regime', 'sample',
    'config_hash', 'feature_count',
    'n_obs', 'dir_acc', 'r2_raw', 'mse_model', 'mse_rw', 'ct_r2_oos',
    'cw_stat', 'cw_pval', 'dmh_stat', 'dmh_pval', 'pt_stat', 'pt_pval',
    'binom_pval', 'n_eff',
]

# MTC columns appended to every row after the family-wise correction.
MTC_COLS = ['reject_bonferroni', 'reject_fdr_bh', 'mtc_N', 'mtc_family']


def compute_metrics_row(y, yhat, H, meta):
    """Build ONE shared-schema row. ``meta`` carries the identifier/provenance columns.

    Numerically identical to v5's ``compute_metrics_row``; the only change is that the
    provenance fields (notebook, run_ts, model_kind, is_pooled) are read from ``meta``
    rather than v5 module globals, so the simple notebooks can stamp their own ids while
    producing metric values on exactly the v5 scale.
    """
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    yy, yh = y[m], yhat[m]
    n = int(yy.size)
    # All shared identifier/provenance columns come straight from meta.
    row = {c: meta.get(c) for c in SHARED_COLS if c in meta}
    row['horizon'] = int(H)
    if n < 5:
        for c in ('n_obs', 'dir_acc', 'r2_raw', 'mse_model', 'mse_rw', 'ct_r2_oos',
                  'cw_stat', 'cw_pval', 'dmh_stat', 'dmh_pval', 'pt_stat', 'pt_pval',
                  'binom_pval', 'n_eff'):
            row[c] = np.nan
        row['n_obs'] = n
        return row
    mse_model = float(np.mean((yy - yh) ** 2))
    mse_rw    = float(np.mean(yy ** 2))
    dir_acc   = float(np.mean(np.sign(yy) == np.sign(yh)))
    cw_s, cw_p   = clark_west(yy, yh, H)
    dm_s, dm_p   = dm_harvey(yy, yh, H)
    pt_s, pt_p   = pesaran_timmermann(yy, yh)
    k = int(np.sum(np.sign(yy) == np.sign(yh)))
    binom_p = binomtest(k, n, p=0.5, alternative='greater').pvalue
    row.update({
        'n_obs': n, 'dir_acc': dir_acc, 'r2_raw': float(r2_score(yy, yh)),
        'mse_model': mse_model, 'mse_rw': mse_rw,
        'ct_r2_oos': float(1 - mse_model / mse_rw) if mse_rw > 0 else np.nan,
        'cw_stat': cw_s, 'cw_pval': cw_p, 'dmh_stat': dm_s, 'dmh_pval': dm_p,
        'pt_stat': pt_s, 'pt_pval': pt_p, 'binom_pval': float(binom_p),
        'n_eff': int(max(1, round(n / max(int(H), 1)))),
    })
    return row


# ── Multiple-testing correction (copied verbatim from v5 cell 27) ────────────
WF_MTC_FAMILY = 'walk_forward:{horizon×tenor×regime}'


def apply_mtc_family(df, schemes, label, *, model_kind=None, alpha=ALPHA_MT):
    """Bonferroni + BH-FDR over the OOS binomial p-values within one MTC family.
    Returns (N, n_bonf, n_bh) and writes reject_* / mtc_N / mtc_family in place.

    Two modes (this is the reconciliation of two previously-divergent copies — the
    choice is methodological, NOT a bug, because it changes which results survive
    correction):

    * ``model_kind=None`` (POOLED) — one family over {horizon × tenor × regime}
      with **all** model_kinds pooled. This is the original ``thesis_eval`` /
      v5 behaviour, used by the mean-reversion and momentum notebooks.

    * ``model_kind=<kind>`` (PER-ESTIMATOR) — additionally restricts the family
      to rows with ``df['model_kind'] == model_kind``, giving one family per
      estimator. This is the linear notebook's behaviour (called once per
      estimator, e.g. OLS / ElasticNet), so each estimator's family size is
      comparable to HTBoost's single-estimator family.

    Pooling vs per-estimator changes the family size N and therefore the
    Bonferroni / BH-FDR thresholds, i.e. **which** rows are rejected. Each
    notebook keeps the mode it previously used.
    """
    mask = (df['sample'] == 'oos') & df['validation_scheme'].isin(schemes) \
           & df['binom_pval'].notna()
    if model_kind is not None:
        mask = mask & (df['model_kind'] == model_kind)
    idx = df.index[mask]
    if len(idx) == 0:
        return 0, 0, 0
    pvals = df.loc[idx, 'binom_pval'].to_numpy(float)
    rej_b, _, _, _ = multipletests(pvals, alpha=alpha, method='bonferroni')
    rej_h, _, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    df.loc[idx, 'reject_bonferroni'] = rej_b
    df.loc[idx, 'reject_fdr_bh']     = rej_h
    df.loc[idx, 'mtc_N']             = len(idx)
    df.loc[idx, 'mtc_family']        = label
    return len(idx), int(rej_b.sum()), int(rej_h.sum())


def finalize_long_csv(df_all):
    """Ensure every shared + MTC column exists and order columns: shared first, then
    extras. Mirrors v5 cell 27's column-ordering so the files concatenate cleanly."""
    df_all = df_all.copy()
    for c in SHARED_COLS:
        if c not in df_all.columns:
            df_all[c] = np.nan
    for c in MTC_COLS:
        if c not in df_all.columns:
            df_all[c] = (False if 'reject' in c else (np.nan if c == 'mtc_N' else ''))
    _extra = [c for c in df_all.columns if c not in SHARED_COLS]
    return df_all[[c for c in SHARED_COLS if c in df_all.columns] + _extra]
