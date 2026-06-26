"""htb_metrics.py — backward-compatibility shim.

The shared metrics harness now lives in :mod:`src.evaluation.metrics` (the single
source of truth, which imports its run-protocol constants from :mod:`src.config`).
This module re-exports it so existing ``import htb_metrics`` / ``from htb_metrics
import ...`` callsites in the pooled GB notebook keep working unchanged.

The metric maths (``_hac_mean_tstat``, ``clark_west``, ``dm_harvey``,
``pesaran_timmermann``, ``_score``, ``config_hash``, ``SHARED_COLS``) were verified
byte-identical (logic) to the shared module and are taken from there directly.

The ONE genuine difference was ``compute_metrics_row``: the original htb_metrics
stamped ``notebook``/``run_ts`` from these module-level provenance tags plus the
defaults ``model_kind='per_security'`` / ``is_pooled=False`` (and those always won
over ``meta``), whereas the shared module reads all provenance from ``meta``. To
preserve the existing behaviour — so the pooled notebook's
``htb_metrics.NOTEBOOK_TAG = 'pooled_v5'`` override still flows into every row —
``compute_metrics_row`` below is a thin wrapper that delegates the metric math to
the shared module and then re-applies the htb_metrics provenance defaults. The
defaults are unchanged.
"""

from src.evaluation.metrics import *  # noqa: F401,F403  — clark_west/dm_harvey/SHARED_COLS/...
from src.evaluation.metrics import _hac_mean_tstat, _score  # noqa: F401  — underscored re-exports
import src.evaluation.metrics as _M

# Provenance tags — overridden by the importing notebook (e.g. pooled sets 'pooled_v5').
NOTEBOOK_TAG = 'shared'
RUN_TS = ''


def compute_metrics_row(y, yhat, H, meta):
    """Build ONE shared-schema row, preserving htb_metrics' provenance defaults.

    Metric math is delegated verbatim to :func:`src.evaluation.metrics.compute_metrics_row`;
    the four provenance fields are then re-stamped from this module's globals/defaults
    exactly as the original htb_metrics did (they always win over ``meta``). The pooled
    notebook overrides ``model_kind``/``is_pooled`` on each row afterward, as before.
    """
    row = _M.compute_metrics_row(y, yhat, H, meta)
    row['notebook']   = NOTEBOOK_TAG
    row['run_ts']     = RUN_TS
    row['model_kind'] = 'per_security'
    row['is_pooled']  = False
    return row
