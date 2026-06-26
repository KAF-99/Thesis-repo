"""Norway-gated (``nor_*``) feature construction.

Derives the leakage-safe ``nor_*`` feature columns from the raw fetched Norway
series (see :mod:`src.data.norway`). Timing policy, applied exactly as in the
original ``add_norway_features_v4``:

  FX (EUR/NOK, USD/NOK, I-44) ........ shift(1)        [USD-cross & index, prior close]
  policy rate / NOWA / govt yields ... contemporaneous [NOK official same-session]
  ECB / Riksbank policy rates ........ contemporaneous [administered step, ffill]
  monthly macro (KPI, KPI-JAE, unemp)  release-aligned ffill + shift(1)

The monthly release lag (formerly the global ``NOR_RELEASE_LAG_M``) is exposed
as the ``lag_m`` keyword, defaulting to 1 to preserve the original behavior.
"""

import numpy as np
import pandas as pd


def _monthly_released_daily(s_refmonth: pd.Series, target_index: pd.Index,
                            lag_m: int = 1) -> pd.Series:
    """Reference-month value -> stamped at end of month M+lag (>= real SSB release),
    then ffill onto daily target_index. Conservative: never look-ahead."""
    if s_refmonth is None or len(s_refmonth.dropna()) == 0:
        return pd.Series(np.nan, index=target_index)
    sh = s_refmonth.dropna().sort_index().copy()
    sh.index = sh.index + pd.offsets.MonthEnd(lag_m)
    sh = sh[~sh.index.duplicated(keep='last')]
    return sh.reindex(target_index, method='ffill')


def add_norway_features(df: pd.DataFrame, nor_raw: pd.DataFrame = None,
                        *, lag_m: int = 1) -> pd.DataFrame:
    """Leakage-safe NOR-gated feature columns (nor_*) from raw fetched series.
    Timing policy:
      FX (EUR/NOK, USD/NOK, I-44) ........ shift(1)  [USD-cross & index, prior close]
      policy rate / NOWA / govt yields ... contemporaneous  [NOK official same-session]
      ECB / Riksbank policy rates ........ contemporaneous  [administered step, ffill]
      monthly macro (KPI, KPI-JAE, unemp) release-aligned ffill + shift(1)
    Daily series read from df (joined); monthly series from nor_raw (native month index).

    ``lag_m`` is the monthly-release lag (formerly the global NOR_RELEASE_LAG_M);
    the default of 1 preserves the original behavior.
    """
    df  = df.copy()
    idx = df.index

    def _raw(col):
        return df[col].copy() if col in df.columns else pd.Series(np.nan, index=idx)

    # FX — shift(1)
    for srccol, lbl in [('nb_eurnok', 'eurnok'), ('nb_usdnok', 'usdnok'), ('nb_i44', 'i44')]:
        fx = _raw(srccol)
        if fx.notna().sum() == 0:
            continue
        df[f'nor_{lbl}_level']    = fx.shift(1)
        df[f'nor_{lbl}_chg_1d']   = fx.diff().shift(1)
        df[f'nor_{lbl}_mom_1m']   = fx.pct_change().rolling(21, min_periods=10).sum().shift(1)
        df[f'nor_{lbl}_mom_3m']   = fx.pct_change().rolling(63, min_periods=30).sum().shift(1)
        df[f'nor_{lbl}_rvol_20d'] = fx.pct_change().rolling(20, min_periods=5).std().shift(1)

    # Policy rate (styringsrenten) — contemporaneous; days-since-last-change
    pol = _raw('nb_polrate')
    if pol.notna().sum() > 0:
        df['nor_polrate_level']  = pol
        df['nor_polrate_chg_1d'] = pol.diff()
        grp = (pol.diff().fillna(0) != 0).cumsum()
        df['nor_polrate_days_since_chg'] = grp.groupby(grp).cumcount().astype(float)

    # NOWA + NIBOR-NOWA spread — contemporaneous (NOK same-session)
    nowa = _raw('nb_nowa')
    if nowa.notna().sum() > 0:
        df['nor_nowa_level']        = nowa
        df['nor_nibor_nowa_spread'] = _raw('NIBOR3M Index  (L1)') - nowa

    # Government bond yields — contemporaneous (NOK official)
    for ten in ['3y', '5y', '10y']:
        g = _raw(f'nb_govt_{ten}')
        if g.notna().sum() > 0:
            df[f'nor_govt_{ten}'] = g

    # ECB deposit rate / Riksbank policy rate — contemporaneous (administered step)
    ecb = _raw('ecb_dfr').reindex(idx).ffill()
    if ecb.notna().sum() > 0:
        df['nor_ecb_dfr_level'] = ecb
        df['nor_ecb_dfr_chg']   = ecb.diff()
    rb = _raw('rb_polrate').reindex(idx).ffill()
    if rb.notna().sum() > 0:
        df['nor_rb_rate_level'] = rb
        df['nor_rb_rate_chg']   = rb.diff()

    # Monthly macro — release-aligned ffill + shift(1)
    for srccol, dst in [('ssb_kpi_yoy', 'nor_kpi_yoy'),
                        ('ssb_kpijae_yoy', 'nor_kpijae_yoy'),
                        ('ssb_unemp', 'nor_unemp')]:
        mser = nor_raw[srccol] if (nor_raw is not None and srccol in nor_raw.columns) else None
        if mser is not None and mser.dropna().shape[0] > 0:
            df[dst] = _monthly_released_daily(mser, idx, lag_m=lag_m).shift(1)
    return df
