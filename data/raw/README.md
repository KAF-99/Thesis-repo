# `data/raw/` — licensed raw data (NOT committed)

This directory is intentionally **empty in the repository**. The files below are
proprietary **Bloomberg** exports and are git-ignored; keep them in your local copy
and point the loaders at them via `THESIS_DATA_PATH` (default `./data/raw`). This
file documents each one so the licensed inputs have provenance without being shared.

Inventory and roles are inferred from `src/data/bloomberg.py`
(`load_data`, `SIMPLE_DATE_FILES`, `SWAP_SKIP_FILES`, `COUNTRY_PRIMARY_IBOR`).

## Country swap-curve files (Bloomberg)

Loaded by `load_data` as `{COUNTRY}_{tenor}` columns: header `skiprows=5`, first 10
columns taken as `Date` + the 9 tenors `MATURITY_NAMES = [1W, 1M, 3M, 6M, 1Y, 5Y,
10Y, 15Y, 30Y]`. One file per market; the filename stem is the column prefix.

| File | Market (inferred) | Has primary IBOR mapping? |
|------|-------------------|---------------------------|
| `NOR.csv`   | Norway (NOK)            | yes (NIBOR 3M/6M) |
| `SWE.csv`   | Sweden (SEK)            | yes (STIB3M) |
| `EUR.csv`   | Euro area (EUR)         | yes (EURIBOR 6M/3M) |
| `SOFR.csv`  | United States (USD/SOFR)| yes (SOFR) |
| `CAN.csv`   | Canada (CAD)            | yes (CAONREPO) |
| `AUS.csv`   | Australia (AUD)         | yes (BBSW3M) |
| `POL.csv`   | Poland (PLN)            | yes (WIBOR3M) |
| `BRAZ.csv`  | Brazil (BRL)            | yes (BZDIOVRA) |
| `CHIN.csv`  | China (CNY)             | yes (CNRR007) |
| `TURK.csv`  | Turkey (TRY)            | yes (MUTKCALM) |
| `SONIA.csv` | United Kingdom (GBP/SONIA) | yes (SONIA O/N) |
| `JPY.csv`   | Japan (JPY)             | **no** — not in `COUNTRY_PRIMARY_IBOR` |
| `NEWZ.csv`  | New Zealand (NZD) *(inferred)* | **no** — not in `COUNTRY_PRIMARY_IBOR` |
| `SWZ.csv`   | Switzerland (CHF) *(inferred)* | **no** — not in `COUNTRY_PRIMARY_IBOR` |

> **Flag:** `JPY`, `NEWZ`, `SWZ` are loaded as swap-curve columns but have **no entry
> in `COUNTRY_PRIMARY_IBOR`**, so they receive no IBOR-derived features (they still
> appear as cross-market `xm_*` features). The market identities for `NEWZ`/`SWZ` are
> inferred from the filename — please confirm.

## Simple date-indexed files (Bloomberg)

`SIMPLE_DATE_FILES` — first column is a `DD.MM.YYYY` date, remaining columns numeric:

| File | Role |
|------|------|
| `Interest rates.csv`   | Short-rate / IBOR / policy-rate series (the `COUNTRY_PRIMARY_IBOR` tickers, e.g. NIBOR3M/STIB3M/EUR006M/SOFRRATE/... and additional EURIBOR/NIBOR tenors). Joined into `df_raw`. |
| `Interest Rates 2.csv` | Additional interest-rate series (second batch), joined into `df_raw`. |
| `Oil, vol, div.csv`    | Macro/market block consumed by `add_macro_features` — volatility (MOVE/VIX/VXN/RVX/OVX/GVZ/V2X), equities (SPX/SX5E/MXWO/NDX), commodities (CL1 oil/copper/OPEC/natgas), inflation (CPI/PCE/breakeven) and credit (IG/HY) series. |

## Macro features file (Bloomberg)

| File | Role |
|------|------|
| `macro_features.csv` | In `SWAP_SKIP_FILES` (not treated as a swap curve). Optional extra macro block joined last by `load_data` if present (e.g. `breakeven_5Y/10Y`, `IG_spread`, `HY_spread`). |

## Not loaded by the pipeline

| File | Note |
|------|------|
| `Duration.xlsx` | Bloomberg duration / DV01 export. **Not** read by `load_data`; `notebooks/01_build_dataset.ipynb` explicitly asserts no duration columns enter `df_raw`. Kept for reference only. |

> **Verify before relying on this:** exact Bloomberg tickers and the full column
> inventory of the three simple-date files and `macro_features.csv` are described by
> their consuming code (`COUNTRY_PRIMARY_IBOR` in `src/config.py`, the series names in
> `src/features/macro.py`) rather than parsed from the files themselves — confirm
> against the actual exports if precise ticker provenance is needed.
