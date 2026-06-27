"""Loaders for the raw Bloomberg CSV exports.

Extracted verbatim (behaviour-wise) from the original ``data_loader.py``:
per-country swap curves are read with the Bloomberg header/skip layout and
prefixed by country, while the interest-rate and macro/market files are read
as simple date-indexed numeric tables. Constants live in :mod:`src.config`.
"""

import os

import pandas as pd

from src import config


def _parse_bloomberg_date(x: object) -> pd.Timestamp:
    """Parse a Bloomberg date cell into a Timestamp.

    Accepts either an Excel serial day count (relative to the 1899-12-30
    epoch) or a day-first date string; returns ``pd.NaT`` for missing or
    unparseable values.
    """
    if pd.isna(x):
        return pd.NaT
    x = str(x).strip()
    if x.isdigit():
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(int(x), unit="D")
    return pd.to_datetime(x, dayfirst=True, errors="coerce")


def _load_simple_csv(path: str) -> pd.DataFrame:
    """Load a CSV where the first column is a DD.MM.YYYY date and the rest are numeric."""
    # float_precision="round_trip": the default C parser (xstrtod) uses approximate FP
    # arithmetic whose last bit rounds differently across arm64/x64, so high-precision series
    # (e.g. ^VIX in Oil, vol, div.csv) parsed to ULP-different floats on macOS vs Windows,
    # making df_raw's hash non-portable. The precise round-trip parser is platform-independent.
    df = pd.read_csv(path, float_precision="round_trip")
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Date"]).set_index("Date").sort_index()


def _require_data_dir(data_path: str) -> None:
    """Fail early with an actionable message if the data dir is missing or has no CSVs.

    The common cause is ``THESIS_DATA_PATH`` not being inherited by a GUI Jupyter
    kernel (so the default ``./data/raw`` is used); without this guard the failure
    surfaces late and cryptically deep inside :func:`load_data`.
    """
    env = os.environ.get("THESIS_DATA_PATH")
    where = (f"THESIS_DATA_PATH points to '{data_path}'" if env
             else f"THESIS_DATA_PATH is not set, so the default '{data_path}' is used")
    hint = ("Set it to your Bloomberg data directory, e.g.:\n"
            "    export THESIS_DATA_PATH=/path/to/Data")
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"{where}, but that directory does not exist. {hint}")
    if not any(f.endswith(".csv") for f in os.listdir(data_path)):
        raise FileNotFoundError(f"{where}, but it contains no .csv files. {hint}")


def load_data(data_path: str = config.DATA_PATH) -> pd.DataFrame:
    """Load and merge swap CSVs, interest rate files, and macro/market variables."""
    _require_data_dir(data_path)
    # sorted() makes the per-country load order — and therefore df_raw's column order —
    # deterministic across machines. os.listdir is OS-/filesystem-dependent (NTFS-alphabetical
    # on Windows vs APFS order on macOS), which gave identical values in a different column
    # ORDER, changing df_raw's pd.util.hash_pandas_object hash across machines (benign for the
    # name-based fit, but a real non-determinism bug). Sorting fixes the hash portably.
    files = sorted(f for f in os.listdir(data_path) if f.endswith(".csv"))
    swap_files = [f for f in files if f not in config.SWAP_SKIP_FILES]

    df_list = []
    for file in swap_files:
        country = file.replace(".csv", "")
        # float_precision="round_trip" for platform-independent float parsing (see _load_simple_csv).
        df = pd.read_csv(os.path.join(data_path, file), skiprows=5, float_precision="round_trip")
        df = df.iloc[:, :10].copy()
        df.columns = ["Date"] + config.MATURITY_NAMES
        df["Date"] = df["Date"].apply(_parse_bloomberg_date)
        for col in config.MATURITY_NAMES:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date")
        df = df.dropna(axis=1, how="all")
        df = df.add_prefix(f"{country}_")
        df_list.append(df)

    df_swaps = pd.concat(df_list, axis=1).sort_index()

    df_rates  = _load_simple_csv(os.path.join(data_path, "Interest rates.csv"))
    df_rates2 = _load_simple_csv(os.path.join(data_path, "Interest Rates 2.csv"))
    df_macro  = _load_simple_csv(os.path.join(data_path, "Oil, vol, div.csv"))

    df = (df_swaps
          .join(df_rates,  how="outer")
          .join(df_rates2, how="outer")
          .join(df_macro,  how="outer"))

    macro_path = os.path.join(data_path, "macro_features.csv")
    if os.path.exists(macro_path):
        df_extra = _load_simple_csv(macro_path)
        df = df.join(df_extra, how="outer")

    return df
