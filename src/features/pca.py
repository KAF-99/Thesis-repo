"""Shared cross-market (``xm_*``) PCA compression for the GB notebooks.

Splits the feature matrix into the ``xm_*`` cross-market block vs the rest,
standardises and PCA-fits ONLY on the training rows, then transforms train AND
test with the SAME fitted objects (so test data never touches the PCA fit). The
``xm_*`` columns are replaced by ``k`` components (``xmpca_01..xmpca_kk``), with
``k`` the smallest count explaining ≥ ``XM_PCA_VAR`` of TRAIN variance, capped at
``XM_PCA_KMAX``.

This pair was byte-identical in ``model_htboost_v5_clean.ipynb`` (per-security) and
``model_htboost_pooled_v5.ipynb`` (pooled); both now import it from here. The
``xm_*`` block *construction* stays in each notebook (per-security builds it per
security; pooled builds it once over ``df_raw``) — only this fit/transform pair is
shared. Defaults bind to ``src.config`` so there is one PCA rule.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.config import XM_PCA_VAR, XM_PCA_KMAX, XM_PCA_ENABLE


def _xm_split(cols):
    xm   = [c for c in cols if c.startswith('xm_')]
    rest = [c for c in cols if not c.startswith('xm_')]
    return xm, rest


def fit_transform_xm_pca(x_tr_df, x_te_df, var_target=XM_PCA_VAR, kmax=XM_PCA_KMAX,
                         enable=XM_PCA_ENABLE):
    """Return (x_tr_new, x_te_new, info). info: applied, k, n_xm_in, evr_cum_k, var_target.
    Round-trips through walk-forward: fit on x_tr only, apply to both."""
    cols = list(x_tr_df.columns)
    xm, rest = _xm_split(cols)
    info = {'applied': False, 'k': 0, 'n_xm_in': len(xm),
            'evr_cum_k': np.nan, 'var_target': var_target}
    if (not enable) or len(xm) < 2:
        return x_tr_df, x_te_df, info

    def _to_arr(df):
        return (df[xm].replace([np.inf, -np.inf], np.nan)
                .astype(np.float64).to_numpy())

    Xtr, Xte = _to_arr(x_tr_df), _to_arr(x_te_df)

    # Standardise on TRAIN moments computed over observed (non-NaN) entries only.
    mu = np.nanmean(Xtr, axis=0)
    sd = np.nanstd(Xtr, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)   # guard zero/NaN-variance cols
    mu = np.where(np.isfinite(mu), mu, 0.0)

    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd

    # Zero-fill AFTER standardising = training-mean imputation (text §6.5).
    Ztr = np.nan_to_num(Ztr, nan=0.0, posinf=0.0, neginf=0.0)
    Zte = np.nan_to_num(Zte, nan=0.0, posinf=0.0, neginf=0.0)

    kfit = min(len(xm), Xtr.shape[0])
    pca_full = PCA(n_components=kfit).fit(Ztr)         # fit on TRAIN only
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cum, var_target) + 1)
    k = max(1, min(k, kmax, kfit))

    pca = PCA(n_components=k).fit(Ztr)
    Ctr, Cte = pca.transform(Ztr), pca.transform(Zte)  # SAME fitted PCA on both
    comp_names = [f'xmpca_{i+1:02d}' for i in range(k)]

    tr_new = pd.concat(
        [x_tr_df[rest].reset_index(drop=True),
         pd.DataFrame(Ctr, columns=comp_names)], axis=1)
    tr_new.index = x_tr_df.index
    te_new = pd.concat(
        [x_te_df[rest].reset_index(drop=True),
         pd.DataFrame(Cte, columns=comp_names)], axis=1)
    te_new.index = x_te_df.index

    info.update({'applied': True, 'k': k,
                 'evr_cum_k': float(cum[k - 1])})
    return tr_new, te_new, info
