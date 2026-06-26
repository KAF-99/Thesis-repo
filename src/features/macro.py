"""Macro / global cross-market feature construction.

Adds the derived macro, volatility, equity, commodity, credit, inflation and
country-IBOR columns to the raw frame. The timing policy (which series are
shift(1) vs contemporaneous vs ffill+shift(1)) is documented in
:mod:`src.features.timing`; the per-column choices are applied here.

Every derived column name is preserved exactly — downstream feature code
(:func:`src.features.security.features_for_security`) reads these by string.
"""

import numpy as np
import pandas as pd

from src.config import COUNTRY_PRIMARY_IBOR


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with derived macro/global feature columns added.

    Exogenous US-close series carry shift(1); country IBOR rates are
    contemporaneous; monthly macro releases are ffill+shift(1). Column names
    and NaN handling are identical to the original ``add_macro_features_v4``.
    """
    df = df.copy()

    def _safe(col):
        return df[col].copy() if col in df.columns else pd.Series(np.nan, index=df.index)

    def _s1(col):
        return _safe(col).shift(1)

    def _mom_s1(col, w, mp):
        return _safe(col).pct_change().rolling(w, min_periods=mp).sum().shift(1)

    def _diff_roll_s1(col, w, mp):
        return _safe(col).diff(1).rolling(w, min_periods=mp).sum().shift(1)

    # MOVE — shift(1): US rates vol, only final after US close
    move = _safe('MOVE Index  (R3)')
    df['move_level']  = move.shift(1)
    df['move_chg_1d'] = move.diff().shift(1)
    mv_mu = move.rolling(252, min_periods=60).mean()
    mv_sd = move.rolling(252, min_periods=60).std()
    df['move_zscore'] = ((move - mv_mu) / (mv_sd + 1e-9)).clip(-5, 5).shift(1)

    # VIX — shift(1): US 4pm close
    vix = _safe('^VIX')
    df['vix_level']  = vix.shift(1)
    vx_mu = vix.rolling(252, min_periods=60).mean()
    vx_sd = vix.rolling(252, min_periods=60).std()
    df['vix_zscore'] = ((vix - vx_mu) / (vx_sd + 1e-9)).clip(-5, 5).shift(1)

    # Additional vol indices — shift(1)
    for src, dst in [('VXN Index  (R2)', 'vxn_level'),
                     ('RVX Index  (L2)', 'rvx_level'),
                     ('OVX Index  (R3)', 'ovx_level'),
                     ('GVZ Index  (R2)', 'gvz_level'),
                     ('V2X Index  (L2)', 'v2x_level')]:
        df[dst] = _s1(src)

    # Equity — shift(1)
    for src, lbl in [('SPX Index  (R4)',  'spx'),
                     ('SX5E Index  (R4)', 'sx5e'),
                     ('MXWO Index  (R4)', 'mxwo'),
                     ('NDX Index  (L4)',  'ndx')]:
        df[f'{lbl}_mom_1m'] = _mom_s1(src, 21, 10)
        df[f'{lbl}_mom_3m'] = _mom_s1(src, 63, 30)

    # Commodities — shift(1)
    df['oil_mom_1m']    = _mom_s1('CL1 COMB Comdty  (R3)', 21, 10)
    df['oil_mom_3m']    = _mom_s1('CL1 COMB Comdty  (R3)', 63, 30)
    df['copper_mom_1m'] = _mom_s1('C01 Comdty  (R4)', 21, 10)
    df['opec_prod']     = _s1('OPECDALY Index  (R3)')
    df['natgas_mom_1m'] = _mom_s1('MUC1 Comdty  (L2)', 21, 10)

    # Breakeven inflation — shift(1): TIPS market (US)
    for bc in ['breakeven_5Y', 'breakeven_10Y']:
        be = _safe(bc)
        df[f'{bc}_level']  = be.shift(1)
        df[f'{bc}_chg_1m'] = be.diff(21).shift(1)
    df['breakeven_slope'] = (_safe('breakeven_10Y') - _safe('breakeven_5Y')).shift(1)

    # Credit spreads — shift(1): US credit market
    df['ig_spread'] = _s1('IG_spread')
    df['hy_spread'] = _s1('HY_spread')
    df['ig_chg_1m'] = _safe('IG_spread').diff(21).shift(1)
    df['hy_chg_1m'] = _safe('HY_spread').diff(21).shift(1)

    # Extra inflation — shift(1)
    df['cpi_nsa_level']  = _s1('CPURNSA Index  (L3)')
    df['pce_core_level'] = _s1('PCE CORE Index  (L2)')

    # Monthly macro — ffill+shift(1): same logic as v3
    for src, dst in [
        ('CPI YOY Index  (R1)',  'cpi_yoy'),
        ('PCE CRCH Index  (L1)', 'pce_core_chg'),
        ('NFP TCH Index  (L4)',  'nfp_change'),
        ('NAPMPMI Index  (R2)',  'pmi_us'),
        ('ECCPEMUY Index  (R1)', 'pmi_eur'),
        ('UMRTEMU Index  (R1)',  'unemp_eur'),
    ]:
        df[dst] = _safe(src).ffill().shift(1)

    # Country IBOR rates — contemporaneous (own-country daily fix)
    for country, (col3m, col6m) in COUNTRY_PRIMARY_IBOR.items():
        sr3m = _safe(col3m)
        df[f'{country}_ibor_level']      = sr3m
        df[f'{country}_ibor_chg_1d']     = sr3m.diff()
        df[f'{country}_ibor_mom_1m']     = sr3m.diff().rolling(21, min_periods=10).sum()
        sr6m = _safe(col6m) if col6m else pd.Series(np.nan, index=df.index)
        df[f'{country}_ibor_term_slope'] = sr6m - sr3m

    # Additional EURIBOR + NIBOR tenors — contemporaneous
    for src, dst in [
        ('EUR003M Index', 'eur_3m'),
        ('EUR001W Index', 'eur_1w'),
        ('NIBOR1M Index', 'nor_ibor_1m'),
        ('NIBOR6M Index', 'nor_ibor_6m'),
    ]:
        df[dst] = _safe(src)

    return df
