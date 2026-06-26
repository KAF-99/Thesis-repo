"""thesis_eval.py — backward-compatibility shim.

The shared evaluation harness has moved to :mod:`src.evaluation.metrics` (the
single source of truth, which now imports its run-protocol constants from
:mod:`src.config`). This module re-exports everything so existing
``import thesis_eval as te`` callsites keep working unchanged during migration.

New code should import from ``src.evaluation.metrics`` directly.
"""

from src.evaluation.metrics import *  # noqa: F401,F403
from src.evaluation.metrics import (  # noqa: F401  — re-export underscored helpers too
    _hac_mean_tstat,
    _score,
)
