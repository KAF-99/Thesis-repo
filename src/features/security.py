"""Per-security feature construction.

Builds the full feature matrix for one swap security: own-security momentum /
volatility / level features (contemporaneous), precomputed macro/global columns
(timing baked in by :func:`src.features.macro.add_macro_features`), the
cross-market ``xm_*`` swap block (shift(1) applied here), and the NOR-gated
``nor_*`` block. Curve helpers (annuity, linear interpolation, carry/roll) live
here too. No feature formula is changed; column names are preserved exactly.
"""

import re

import numpy as np
import pandas as pd

from src.config import MATURITY_YEARS, SWAP_PAT


def swap_annuity(S_pct: float, T_years: float) -> float:
    """Par-swap annuity factor for a semiannual fixed leg at rate ``S_pct`` (%).

    Returns ``np.nan`` for missing or degenerate rates (``S_pct <= 0.01``).
    """
    if pd.isna(S_pct) or S_pct <= 0.01:
        return np.nan
    S_semi = S_pct / 200
    n = 2 * T_years
    return (1 - (1 + S_semi) ** (-n)) / (S_pct / 100)


def interp_rate(country: str, T_target: float, df: pd.DataFrame) -> pd.Series:
    """Linearly interpolate the ``country`` swap curve to ``T_target`` years.

    Uses whichever ``{country}_{tenor}`` columns exist; returns a 0.0 series at
    or below the shortest available tenor, the longest tenor's series at or
    above the longest point, and NaN when no curve columns are present.
    """
    pts = {}
    for m, T in MATURITY_YEARS.items():
        col = f'{country}_{m}'
        if col in df.columns:
            pts[T] = df[col]
    if not pts:
        return pd.Series(np.nan, index=df.index)
    ts = sorted(pts.keys())
    if T_target <= 0 or T_target <= ts[0]:
        return pd.Series(0.0, index=df.index)
    if T_target >= ts[-1]:
        return pts[ts[-1]]
    for i in range(len(ts) - 1):
        if ts[i] <= T_target <= ts[i + 1]:
            w = (T_target - ts[i]) / (ts[i + 1] - ts[i])
            return pts[ts[i]] * (1 - w) + pts[ts[i + 1]] * w
    return pd.Series(np.nan, index=df.index)


def carry_roll_spread(sec: str, df: pd.DataFrame, country: str, maturity: str) -> pd.Series:
    """Carry+roll-down spread: current rate minus the (T-1)y-interpolated rate."""
    T = MATURITY_YEARS.get(maturity, np.nan)
    if np.isnan(T):
        return pd.Series(np.nan, index=df.index)
    S_n   = df[sec]
    T_roll = max(0.0, T - 1.0)
    S_nm1  = (pd.Series(0.0, index=df.index) if T_roll == 0.0
              else interp_rate(country, T_roll, df))
    return S_n - S_nm1


def features_for_security(sec: str, df: pd.DataFrame, country: str, maturity: str,
                          target_lags: int = 5) -> pd.DataFrame:
    """Full feature set for one security.

    Reads precomputed columns from df (add_macro_features applied, so exogenous
    series already carry shift(1)), and adds:
      - AR target lags chg_lag1..chg_lag{target_lags}: backward-looking by
        construction (lag-k at t uses chg_1d[t-k]); picked up by leakage audit.
      - Extra vol indices: V2X, OVX, VXN, RVX, GVZ (shift(1) via precomputed)
      - Credit spreads: IG, HY (shift(1) via precomputed)
      - Energy: OPEC prod, natgas (shift(1) via precomputed)
      - Extra IBOR: EUR 1w/3m, NOR 1m/6m (contemporaneous via precomputed)
      - Cross-market swap rates: ALL other swap columns, shift(1) applied here.
        feature[t] = other_swap[t-1] — safe for all market timezones.

    ``target_lags`` is the number of AR lags of chg_1d (formerly the global
    TARGET_LAGS); the default of 5 preserves the original behavior.
    """

    def _get(col):
        return df[col].copy() if col in df.columns else pd.Series(np.nan, index=df.index)

    # ── Own-security features (contemporaneous) ───────────────────────────────
    level    = df[sec]
    chg_1d   = level.diff(1)

    # Explicit autoregressive target lags (lag-1..target_lags of chg_1d).
    # Backward-looking by construction: lag-k at t uses chg_1d at t-k only.
    _tlags = {f'chg_lag{k}': chg_1d.shift(k) for k in range(1, target_lags + 1)}

    mom_1m   = chg_1d.rolling(21,  min_periods=10).sum()
    mom_3m   = chg_1d.rolling(63,  min_periods=30).sum()
    mom_6m   = chg_1d.rolling(126, min_periods=60).sum()
    mom_12m  = chg_1d.rolling(252, min_periods=120).sum()
    mom_63d  = chg_1d.rolling(63,  min_periods=21).mean()
    dev_ma_3m  = level - level.rolling(63,  min_periods=21).mean()
    dev_ma_12m = level - level.rolling(252, min_periods=120).mean()
    rvol_20d   = chg_1d.rolling(20, min_periods=5).std()
    vol_mu     = rvol_20d.rolling(252, min_periods=60).mean()
    vol_sd     = rvol_20d.rolling(252, min_periods=60).std()
    vol_med    = rvol_20d.rolling(252, min_periods=60).median()
    vol_regime = (rvol_20d > vol_med).astype(float)
    vol_zscore = ((rvol_20d - vol_mu) / (vol_sd + 1e-9)).clip(-5, 5)
    lv_mu      = level.rolling(504, min_periods=120).mean()
    lv_sd      = level.rolling(504, min_periods=120).std()
    lv_zscore  = ((level - lv_mu) / (lv_sd + 1e-9)).clip(-5, 5)
    carry_roll = carry_roll_spread(sec, df, country, maturity)

    def _slope(lm, sm):
        lc, sc = f'{country}_{lm}', f'{country}_{sm}'
        if lc in df.columns and sc in df.columns:
            return df[lc] - df[sc]
        return pd.Series(np.nan, index=df.index)

    sofr_mom_1m = (df['SOFR_10Y'].diff(1).rolling(21, min_periods=10).sum()
                   if 'SOFR_10Y' in df.columns
                   else pd.Series(np.nan, index=df.index))

    base = {
        **_tlags,
        'carry_roll'         : carry_roll,
        'mom_63d'            : mom_63d,
        'mom_1m'             : mom_1m,
        'mom_3m'             : mom_3m,
        'mom_6m'             : mom_6m,
        'mom_12m'            : mom_12m,
        'dev_ma_3m'          : dev_ma_3m,
        'dev_ma_12m'         : dev_ma_12m,
        'vol_20d'            : rvol_20d,
        'vol_regime'         : vol_regime,
        'vol_zscore'         : vol_zscore,
        'level_zscore'       : lv_zscore,
        'slope_10_1'         : _slope('10Y', '1Y'),
        'slope_10_5'         : _slope('10Y', '5Y'),
        'slope_5_1'          : _slope('5Y',  '1Y'),
        'sofr_mom_1m'        : sofr_mom_1m,
        # country IBOR (contemporaneous)
        'ibor_level'         : _get(f'{country}_ibor_level'),
        'ibor_chg_1d'        : _get(f'{country}_ibor_chg_1d'),
        'ibor_mom_1m'        : _get(f'{country}_ibor_mom_1m'),
        'ibor_term_slope'    : _get(f'{country}_ibor_term_slope'),
        'swap_ibor_basis'    : level - _get(f'{country}_ibor_level'),
        # global vol (shift(1) baked into precomputed columns)
        'move_level'         : _get('move_level'),
        'move_zscore'        : _get('move_zscore'),
        'vix_level'          : _get('vix_level'),
        'vix_zscore'         : _get('vix_zscore'),
        'v2x_level'          : _get('v2x_level'),
        'vxn_level'          : _get('vxn_level'),
        'ovx_level'          : _get('ovx_level'),
        'gvz_level'          : _get('gvz_level'),
        # equity (shift(1))
        'spx_mom_1m'         : _get('spx_mom_1m'),
        'spx_mom_3m'         : _get('spx_mom_3m'),
        'sx5e_mom_1m'        : _get('sx5e_mom_1m'),
        'sx5e_mom_3m'        : _get('sx5e_mom_3m'),
        'mxwo_mom_1m'        : _get('mxwo_mom_1m'),
        'ndx_mom_1m'         : _get('ndx_mom_1m'),
        # commodities (shift(1))
        'oil_mom_1m'         : _get('oil_mom_1m'),
        'oil_mom_3m'         : _get('oil_mom_3m'),
        'copper_mom_1m'      : _get('copper_mom_1m'),
        'opec_prod'          : _get('opec_prod'),
        'natgas_mom_1m'      : _get('natgas_mom_1m'),
        # breakeven + credit (shift(1))
        'be5y_level'         : _get('breakeven_5Y_level'),
        'be10y_level'        : _get('breakeven_10Y_level'),
        'be_slope'           : _get('breakeven_slope'),
        'be5y_chg_1m'        : _get('breakeven_5Y_chg_1m'),
        'ig_spread'          : _get('ig_spread'),
        'hy_spread'          : _get('hy_spread'),
        'ig_chg_1m'          : _get('ig_chg_1m'),
        'hy_chg_1m'          : _get('hy_chg_1m'),
        # monthly macro (ffill+shift(1))
        'cpi_yoy'            : _get('cpi_yoy'),
        'pce_core_chg'       : _get('pce_core_chg'),
        'nfp_change'         : _get('nfp_change'),
        'pmi_us'             : _get('pmi_us'),
        'pmi_eur'            : _get('pmi_eur'),
        'unemp_eur'          : _get('unemp_eur'),
        'cpi_nsa_level'      : _get('cpi_nsa_level'),
        'pce_core_level'     : _get('pce_core_level'),
        # extra IBOR tenors (contemporaneous)
        'eur_3m'             : _get('eur_3m'),
        'eur_1w'             : _get('eur_1w'),
        'nor_ibor_1m'        : _get('nor_ibor_1m'),
        'nor_ibor_6m'        : _get('nor_ibor_6m'),
    }

    # Cross-market swap rates — ALL other swap cols, shift(1)
    # feature[t] = other_swap_close[t-1], safe for all timezones
    xm = {}
    other_swaps = sorted(c for c in df.columns if SWAP_PAT.match(c) and c != sec)
    for col in other_swaps:
        fn = 'xm_' + re.sub(r'[^a-zA-Z0-9]', '_', col).strip('_')
        xm[fn + '_lv'] = df[col].shift(1)
        xm[fn + '_ch'] = df[col].diff(1).shift(1)

    # ── Norway-specific features — gated to country=='NOR' (Part B) ──────────
    # Read precomputed nor_* columns (timing baked in by add_norway_features_v4)
    # plus the swap-govvie spread for the security's own tenor (contemporaneous).
    nor = {}
    if country == 'NOR':
        for fn in sorted(c for c in df.columns if c.startswith('nor_')):
            nor[fn] = df[fn]
        gcol = f'nor_govt_{maturity.lower()}'
        nor['swap_govt_spread'] = ((level - df[gcol]) if gcol in df.columns
                                   else pd.Series(np.nan, index=df.index))

    return pd.DataFrame({**base, **xm, **nor}, index=df.index)
