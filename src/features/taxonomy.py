"""Shared feature taxonomy — the single source of truth for bucketing features.

``bucket_feature`` is copied **verbatim** from ``model_htboost_v5_clean.ipynb`` /
``model_linear_v5_clean.ipynb`` (both byte-identical). Lifting it here lets every
model — HTBoost, the linear benchmarks, and XGBoost — bucket feature importance,
permutation importance, and SHAP onto the **same** macro-vs-curve taxonomy, so the
cross-model comparison (thesis §3.5 / Ch. 4) is apples-to-apples. This mirrors the
``SHARED_COLS`` "one definition" rule for the metric schema: do not let copies drift.

The PCA components ``xmpca_*`` inherit the ``cross_market`` bucket, so importance
attributed to the compressed cross-market block is still bucketed correctly.
"""

# Mutually-exclusive buckets; every engineered feature maps to exactly one.
BUCKETS = ['curve', 'momentum', 'vol', 'macro', 'credit',
           'cross_market', 'norway', 'carry_roll']


def bucket_feature(name):
    n = str(name)
    if n == 'carry_roll':
        return 'carry_roll'
    if n.startswith('xm_') or n.startswith('xmpca_'):
        return 'cross_market'
    if n.startswith('nor_') or n == 'swap_govt_spread':
        return 'norway'
    if n.startswith('ig_') or n.startswith('hy_'):
        return 'credit'
    if (n.startswith(('vol_', 'move_', 'vix_', 'v2x_', 'vxn_', 'ovx_', 'gvz_', 'rvx_'))):
        return 'vol'
    if n.startswith(('mom_', 'chg_lag')):
        return 'momentum'
    if (n in ('level_zscore', 'swap_ibor_basis') or
            n.startswith(('slope_', 'dev_ma_'))):
        return 'curve'
    # everything else: rates/macro/equity/commodity/breakeven exogenous → macro
    return 'macro'
