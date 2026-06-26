"""Project-wide configuration constants.

This module centralises every literal that the data-loading and
feature-engineering code relies on, so behaviour is identical across the
notebooks and the extracted ``src`` package.

Groups
------
DATA_PATH
    Root directory holding the raw CSV exports. Resolved from the
    ``THESIS_DATA_PATH`` environment variable, falling back to a relative
    ``./data/raw`` so no machine-specific absolute path is hard-coded.

MATURITY_NAMES / MATURITY_YEARS
    The nine swap tenors, in the column order produced by the Bloomberg
    exports, and a mapping from each tenor label to its length in years
    (used when interpolating along the maturity axis).

COUNTRY_PRIMARY_IBOR
    For each country/curve code, the primary (and optional secondary)
    Bloomberg short-rate series used as that curve's front-end reference.
    Each value is a ``(primary, secondary)`` tuple; ``secondary`` is
    ``None`` when only one series is available.

SIMPLE_DATE_FILES / SWAP_SKIP_FILES
    ``SIMPLE_DATE_FILES`` are CSVs whose first column is a plain
    ``DD.MM.YYYY`` date with numeric columns after it. ``SWAP_SKIP_FILES``
    are the files that must NOT be treated as per-country swap exports
    (the simple-date files plus the macro-feature file).

SWAP_PAT
    Compiled regex matching a swap column name of the form
    ``<COUNTRY>_<TENOR>`` (e.g. ``NOR_3M``), used to pick swap columns
    out of the merged frame.

Run-protocol constants (evaluation harness)
    Shared single source of truth for the walk-forward / block-CV evaluation
    protocol, previously duplicated in both ``thesis_eval.py`` and the
    benchmark notebooks. Values are copied verbatim from ``thesis_eval.py``
    (the canonical copy) — ``WF_FOLDS`` (expanding-window regime folds),
    ``FOLD_NAMES``, ``REGIMES``, ``H_GRID`` (pre-committed horizons), the
    Norwegian headline universe ``NOR_TENORS``, the fold-acceptance thresholds
    ``MIN_TRAIN_OBS`` / ``MIN_TEST_OBS``, and the multiple-testing alpha
    ``ALPHA_MT``. ``BLOCK_CV_FOLDS`` and ``EMBARGO_FOR_H`` live only in the
    notebooks (not in ``thesis_eval.py``); their values are copied verbatim
    from the v5/linear CONFIG and are unchanged.
"""

import os
import re

DATA_PATH = os.environ.get("THESIS_DATA_PATH", "./data/raw")

MATURITY_NAMES = ["1W", "1M", "3M", "6M", "1Y", "5Y", "10Y", "15Y", "30Y"]

MATURITY_YEARS = {
    "1W": 1 / 52, "1M": 1 / 12, "3M": 0.25, "6M": 0.5,
    "1Y": 1.0, "5Y": 5.0, "10Y": 10.0, "15Y": 15.0, "30Y": 30.0,
}

COUNTRY_PRIMARY_IBOR = {
    "NOR":   ("NIBOR3M Index  (L1)",  "NIBOR6M Index"),
    "SWE":   ("STIB3M Index  (L1)",   None),
    "EUR":   ("EUR006M Index  (L1)",  "EUR003M Index"),
    "SOFR":  ("SOFRRATE Index  (L1)", None),
    "CAN":   ("CAONREPO Index  (L1)", None),
    "AUS":   ("BBSW3M Index  (L1)",   None),
    "POL":   ("WIBR3M Index  (L1)",   None),
    "BRAZ":  ("BZDIOVRA Index  (R2)", None),
    "CHIN":  ("CNRR007 Index  (R2)",  None),
    "TURK":  ("MUTKCALM Index  (R1)", None),
    "SONIA": ("SONIO/N Index  (L1)",  None),
}

SIMPLE_DATE_FILES = {"Interest rates.csv", "Interest Rates 2.csv", "Oil, vol, div.csv"}
SWAP_SKIP_FILES = SIMPLE_DATE_FILES | {"macro_features.csv"}

SWAP_PAT = re.compile(r"^[A-Z]+_\d+[WMY]$")


# ── Run-protocol constants (evaluation harness) ──────────────────────────────
# Copied verbatim from thesis_eval.py (the canonical copy). Do not change values.

# Walk-forward folds: expanding window, test windows map to regimes.
WF_FOLDS = [
    # (name,          test_start,    test_end,      regime)
    ('GFC',           '2010-01-01',  '2012-12-31',  'GFC'),
    ('ZIRP_early',    '2013-01-01',  '2016-12-31',  'ZIRP'),
    ('ZIRP_late',     '2017-01-01',  '2019-12-31',  'ZIRP'),
    ('COVID',         '2020-01-01',  '2021-12-31',  'COVID'),
    ('Hiking',        '2022-01-01',  '2026-12-31',  'Hiking'),
]
FOLD_NAMES = [f[0] for f in WF_FOLDS]
REGIMES = ['GFC', 'ZIRP', 'COVID', 'Hiking']

# Pre-committed horizon grid (mirrors v5 H_GRID): 1d, 1w, 1m, 3m.
H_GRID = [1, 5, 21, 63]

# Headline universe — the SAME Norwegian tenors v5 restricts to.
NOR_TENORS = ['NOR_1Y', 'NOR_5Y', 'NOR_10Y', 'NOR_15Y', 'NOR_30Y']

# Fold-acceptance thresholds (copied from v5 CONFIG).
MIN_TRAIN_OBS = 252         # ≥1 year training per fold
MIN_TEST_OBS  = 20          # ≥20 OOS obs to score a fold
ALPHA_MT      = 0.05        # multiple-testing alpha

# Block-CV / leave-one-regime-out folds + embargo. These live only in the
# notebooks (not in thesis_eval.py); values copied verbatim from the v5/linear
# CONFIG. Full sample (2007+), aligned to regimes; EMBARGO scales with H.
BLOCK_CV_FOLDS = [
    ('Block_GFC',    '2007-01-01', '2012-12-31', 'GFC'),
    ('Block_ZIRP',   '2013-01-01', '2019-12-31', 'ZIRP'),
    ('Block_COVID',  '2020-01-01', '2021-12-31', 'COVID'),
    ('Block_Hiking', '2022-01-01', '2026-12-31', 'Hiking'),
]


def EMBARGO_FOR_H(h):
    """López de Prado embargo (business days); scales with H. Two-sided in block-CV."""
    return max(int(h), 10)


# ── Cross-market xm_* PCA compression (GB notebooks) ─────────────────────────
# Pre-committed rule shared by both gradient-boosting notebooks: replace the xm_*
# block with k principal components, k = smallest #components explaining ≥ XM_PCA_VAR
# of TRAIN variance, capped at XM_PCA_KMAX. Values copied verbatim from the v5/pooled
# CONFIG (identical in both). Consumed by src.features.pca.fit_transform_xm_pca.
XM_PCA_ENABLE = True
XM_PCA_VAR    = 0.95        # explained-variance target (a-priori)
XM_PCA_KMAX   = 12          # hard cap on #components (a-priori)

# ── Julia RNG seed (GB notebooks) ────────────────────────────────────────────
# One number for every determinism knob: the Julia RNG (re-seeded before every
# HTBfit), the leakage-audit subsample, and the pooled PCA-subsample random_state.
# Copied verbatim from the pooled CONFIG / per-security seed cell. Do not change.
JULIA_SEED = 20260619
