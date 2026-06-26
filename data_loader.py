import os
import pandas as pd

DATA_PATH = "/Users/knutandersfjellheim/Documents/Master/Masteroppgave/Script/Data"

MATURITY_NAMES = ["1W", "1M", "3M", "6M", "1Y", "5Y", "10Y", "15Y", "30Y"]


def _parse_bloomberg_date(x):
    if pd.isna(x):
        return pd.NaT
    x = str(x).strip()
    if x.isdigit():
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(int(x), unit="D")
    return pd.to_datetime(x, dayfirst=True, errors="coerce")


SIMPLE_DATE_FILES = {"Interest rates.csv", "Interest Rates 2.csv", "Oil, vol, div.csv"}
SWAP_SKIP_FILES   = SIMPLE_DATE_FILES | {"macro_features.csv"}


def _load_simple_csv(path):
    """Load a CSV where the first column is a DD.MM.YYYY date and the rest are numeric."""
    df = pd.read_csv(path)
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Date"]).set_index("Date").sort_index()


def load_data(data_path=DATA_PATH):
    """Load and merge swap CSVs, interest rate files, and macro/market variables."""
    files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    swap_files = [f for f in files if f not in SWAP_SKIP_FILES]

    df_list = []
    for file in swap_files:
        country = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(data_path, file), skiprows=5)
        df = df.iloc[:, :10].copy()
        df.columns = ["Date"] + MATURITY_NAMES
        df["Date"] = df["Date"].apply(_parse_bloomberg_date)
        for col in MATURITY_NAMES:
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
