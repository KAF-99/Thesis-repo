"""Swap-universe selection and stacked-panel construction.

Identifies the swap columns in a raw frame, freezes the trading universe on
full-history observation counts, and assembles the stacked per-security feature
panel used for modeling. Behavior is identical to the original ``build_panel_v4``
and the inline universe-construction lines in the notebook.
"""

import pandas as pd

from src.config import SWAP_PAT
from src.features.security import features_for_security

META_COLS = {'date', 'security', 'y', 'level'}


def swap_columns(df: pd.DataFrame) -> list:
    """Return the sorted column names in ``df`` matching the swap pattern
    ``<COUNTRY>_<TENOR>`` (e.g. ``NOR_10Y``)."""
    return sorted(c for c in df.columns if SWAP_PAT.match(c))


def build_universe(df: pd.DataFrame, swap_cols: list, min_obs: int) -> list:
    """Freeze the trading universe on full history: swap columns with at least
    ``min_obs`` non-NaN observations, sorted."""
    obs = df[swap_cols].notna().sum()
    return sorted(obs[obs >= min_obs].index.tolist())


def build_panel(df_raw: pd.DataFrame, securities, h: int) -> pd.DataFrame:
    """Stacked per-security panel. Target: outright h-day change (not residual)."""
    rows = []
    for sec in securities:
        if sec not in df_raw.columns:
            continue
        country, maturity = sec.rsplit('_', 1)
        level = df_raw[sec]
        feats = features_for_security(sec, df_raw, country, maturity)
        y     = level.shift(-h) - level
        sec_df             = feats.reset_index().rename(columns={'index': 'date',
                                                                  df_raw.index.name or 'Date': 'date'})
        # handle named index
        if 'date' not in sec_df.columns:
            sec_df.insert(0, 'date', feats.index)
        sec_df['security'] = sec
        sec_df['y']        = y.values
        sec_df['level']    = level.values
        rows.append(sec_df)

    if not rows:
        return pd.DataFrame()
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna(subset=['y'])
    panel = panel.sort_values(['date', 'security']).reset_index(drop=True)
    return panel
