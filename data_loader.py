"""data_loader.py — backward-compatibility shim.

The Bloomberg CSV loader now lives in :mod:`src.data.bloomberg` (extracted from this
module — the loading logic is byte-for-byte identical, with added type hints and
config-driven paths). This shim re-exports it so the model notebooks'
``from data_loader import load_data, MATURITY_NAMES`` keep working — now resolving the
data directory via ``THESIS_DATA_PATH`` (default ``./data/raw``) on any machine,
instead of the old hard-coded absolute path.

New code should import from ``src.data.bloomberg`` / ``src.config`` directly.
"""

from src.config import MATURITY_NAMES, SIMPLE_DATE_FILES, SWAP_SKIP_FILES, DATA_PATH  # noqa: F401
from src.data.bloomberg import (  # noqa: F401
    load_data,
    _parse_bloomberg_date,
    _load_simple_csv,
)
