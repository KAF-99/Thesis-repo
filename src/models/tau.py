"""Canonical weighted-smoothness τ^w diagnostic for HTBoost (paper eq. 19 + fn. 8).

τ^w is the importance-weighted geometric-mean smoothing parameter: per-feature
``avgtau`` capped at 40, then a geometric mean weighted by HTBrelevance importance.
It is a MODEL DIAGNOSTIC (curve smoothness), never an accuracy metric, and is kept
OUT of the shared metrics schema.

This module is the single source of truth, extracted verbatim from the per-security
``model_htboost_v5_clean.ipynb`` approach: read the package scalar directly from the
PINNED field (``gavgtau`` = the geometric importance-weighted mean), with the
candidate-list resolution as a fallback; compute the manual cap-40 importance-weighted
geometric mean ONLY as a cross-check (logging a one-line warning on divergence).

The pooled notebook previously recomputed τ^w itself (its candidate list omitted
``gavgtau``, so it always fell through to the manual geometric mean). Pointing pooled
at :func:`extract_weighted_tau` aligns its reported τ^w to the per-security definition
— the one intended diagnostic change of this consolidation.

``jl`` (juliacall) is imported lazily so the pure-Python helpers (``tau_w_band``,
``_weighted_tau_scalar``, ``_resolve_field``) are importable without a Julia runtime;
only :func:`extract_weighted_tau` actually touches Julia.
"""

import numpy as np
import pandas as pd

try:                                   # available only in the juliacall HTBoost kernel
    from juliacall import Main as jl
except Exception:                      # offline / non-Julia env — pure-Python helpers still import
    jl = None


# ── τ^w smoothness diagnostic constants ──────────────────────────────────────
TAU_W_CAP   = 40.0
# Interpretive bands (HTBoost tutorial thresholds), upper-inclusive cutoffs.
TAU_W_BANDS = ((7.0, 'strong'), (15.0, 'mild'), (25.0, 'weak'))   # >25 → 'none'


def tau_w_band(tau):
    """Map a scalar τ^w to its interpretive smoothness band:
    ≤7 'strong', 7–15 'mild', 15–25 'weak', >25 'none' (≈ hard splits / no smoothness).
    Returns 'na' for a missing/non-finite τ^w."""
    if tau is None or not np.isfinite(tau):
        return 'na'
    for hi, lab in TAU_W_BANDS:
        if tau <= hi:
            return lab
    return 'none'


def _weighted_tau_scalar(importance, avgtau, cap=TAU_W_CAP):
    """Aggregate scalar τ^w per eq. (19) + footnote 8: cap each per-feature avgtau at
    `cap`, then take the importance-weighted GEOMETRIC mean. NaN if undefined."""
    w = np.asarray(importance, dtype=float)
    t = np.minimum(np.asarray(avgtau, dtype=float), cap)
    m = np.isfinite(w) & np.isfinite(t) & (w > 0) & (t > 0)
    if not m.any():
        return float('nan')
    w, t = w[m], t[m]
    return float(np.exp(np.sum(w * np.log(t)) / np.sum(w)))


# τ^w field resolution — the installed HybridTreeBoosting build may name the
# HTBweightedtau columns differently from the tutorial, so we resolve them from the
# REAL returned keys (case-insensitive substring) instead of hard-coding names.
# Pinned below to the installed build's schema (confirmed via the run-once PROBE
# cell); overrides win over auto-resolution. Re-run the PROBE and edit the dict if
# the build ever changes.
HTB_TAU_FIELD_OVERRIDES = {
    # Pinned to the installed HybridTreeBoosting build's HTBweightedtau schema
    # (confirmed via the PROBE cell). gavgtau = geometric importance-weighted mean
    # = paper eq.19 + footnote 8 = the reported τ^w (NOT the arithmetic avgtau).
    'tau_scalar':     'gavgtau',   # scalar τ^w (read directly)
    'tau_table':      'df',        # per-feature DataFrame field
    'feature_col':    'feature',   # column inside df
    'importance_col': 'importance',# column inside df
    'smoothness_col': 'avgtau',    # per-feature smoothness column inside df
}
_TAU_FIELD_CANDIDATES = {
    'feature':    ['feature', 'fname', 'fnames', 'name', 'names', 'var'],
    'importance': ['importance', 'relevance', 'fi', 'weight', 'rel'],
    'smoothness': ['avgtau', 'gavgtau', 'meantau', 'avg_tau', 'tau', 'smooth'],
}
_TAU_SCALAR_CANDIDATES = ['gavgtau', 'tau_w', 'wtau', 'weightedtau', 'taubar', 'avgtau_w']


def _resolve_field(keys, candidates, override=None):
    """Pick the key matching an ordered candidate list. An explicit override wins
    (returned only if actually present). Otherwise: exact case-insensitive match in
    candidate order, then substring match in candidate order, preferring a non-
    'sorted'/'index' variant so per-feature columns stay row-aligned. None if none match."""
    keys = list(keys)
    if override is not None:
        return override if override in keys else None
    low = {k.lower(): k for k in keys}
    for cand in candidates:                                   # exact (case-insensitive)
        if cand.lower() in low:
            return low[cand.lower()]
    matches = []                                              # substring, candidate order
    for cand in candidates:
        for k in keys:
            if cand.lower() in k.lower() and k not in matches:
                matches.append(k)
    if not matches:
        return None
    plain = [k for k in matches
             if not any(s in k.lower() for s in ('sort', 'indx', 'index', 'idx'))]
    return (plain or matches)[0]


def _extract_weightedtau(wt):
    """Resolve (tau_w_scalar, per_feature_DataFrame) from an HTBweightedtau return
    object using the PINNED HTB_TAU_FIELD_OVERRIDES (confirmed against the installed
    build via the PROBE cell). The scalar τ^w is read DIRECTLY from the geometric
    importance-weighted-mean field (`gavgtau` = paper eq.19 + fn.8) — never the
    arithmetic `avgtau`, never recomputed. The per-feature table is pulled from the
    `df` field (a Julia DataFrame). `_weighted_tau_scalar` is used ONLY as a cross-
    check: if the paper-formula reconstruction disagrees with the package scalar, a
    one-line warning is printed (we still report the package value). Raises if the
    scalar field is absent (the caller logs the keys and returns None)."""
    try:
        wt_keys = [str(f) for f in jl.seval('x -> collect(string.(keys(x)))')(wt)]
    except Exception:
        wt_keys = [str(f) for f in jl.seval('x -> collect(string.(propertynames(x)))')(wt)]
    ov = HTB_TAU_FIELD_OVERRIDES
    k_scalar = ov.get('tau_scalar') or _resolve_field(wt_keys, _TAU_SCALAR_CANDIDATES)
    if not k_scalar or k_scalar not in wt_keys:
        raise KeyError(f'τ^w scalar field not found (keys={wt_keys})')
    tau_w = float(getattr(wt, k_scalar))
    tau_table = None
    k_table = ov.get('tau_table')
    f_col = ov.get('feature_col', 'feature')
    i_col = ov.get('importance_col', 'importance')
    s_col = ov.get('smoothness_col', 'avgtau')
    if k_table and k_table in wt_keys:
        _dfj  = getattr(wt, k_table)
        _pull = jl.seval('(d, c) -> collect(d[!, Symbol(c)])')
        feats = [str(x) for x in _pull(_dfj, f_col)]
        imp_w = np.asarray(_pull(_dfj, i_col), dtype=float)
        avgt  = np.asarray(_pull(_dfj, s_col), dtype=float)
        tau_table = pd.DataFrame({'feature': feats, 'importance': imp_w, 'avgtau': avgt})
        _recon = _weighted_tau_scalar(imp_w, avgt)     # cap-40 weighted geo-mean cross-check
        if (np.isfinite(_recon) and np.isfinite(tau_w)
                and abs(_recon - tau_w) > max(0.05, 0.05 * tau_w)):
            print(f'    [warn] τ^w cross-check: package {k_scalar}={tau_w:.3f} vs '
                  f'eq.19 reconstruction={_recon:.3f} disagree (reporting {k_scalar})')
    return tau_w, tau_table


def extract_weighted_tau(output, data):
    """Canonical weighted smoothness τ^w from a fitted HTBoost model (paper eq.19).

    Calls ``HTBweightedtau(output, data)``, then reads the scalar τ^w directly from
    the pinned package field (``gavgtau`` = geometric importance-weighted mean), with
    candidate-list resolution as a fallback; the manual cap-40 importance-weighted
    geometric mean is computed only as a cross-check (one-line warning on divergence).

    Returns ``(tau_w, {feature: avgtau})``. FAIL-SOFT: returns ``(nan, {})`` on any
    error and prints a short warning, so no fit is ever lost.
    """
    try:
        wt = jl.HTBweightedtau(output, data, verbose=False)
        tau_w, tau_table = _extract_weightedtau(wt)
        perfeat = ({str(f): float(t)
                    for f, t in zip(tau_table['feature'], tau_table['avgtau'])}
                   if tau_table is not None else {})
        return tau_w, perfeat
    except Exception as e:
        print(f'    [warn] HTBweightedtau extraction failed: {repr(e)[:120]}')
        return float('nan'), {}
