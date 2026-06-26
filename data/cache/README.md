# `data/cache/` — Norway public-API cache (committed)

Unlike `data/raw/` (licensed Bloomberg data, never committed), this directory holds
**non-licensed, regenerable** data fetched from public APIs, so it **is** committed —
it is the shared, reproducible Norway panel both collaborators build on.

## `norway_raw_features.csv`

Produced and consumed by `src/data/norway.py::load_norway_raw(start, end, cache_path,
*, live=False)`:

- **`live=False` (default)** — read this cached CSV. No network access; both
  collaborators reproduce the **identical** Norway panel from the committed file.
- **`live=True`** — fetch fresh series from the public APIs below and **overwrite**
  this cache on success; fall back to the cache if a fetch fails or returns empty.

### Sources (all public, no authentication / no licence)

| Provider | Series |
|----------|--------|
| **Norges Bank** (`data.norges-bank.no`) | EXR: EUR/NOK, USD/NOK, I-44 krone index; IR: policy rate (KPRA); SHORT_RATES: NOWA; GOVT_GENERIC_RATES: 3Y/5Y/10Y government yields |
| **SSB** (`data.ssb.no`, PxWebApi) | KPI YoY (table 03013), KPI-JAE YoY (05327), LFS unemployment (13760) |
| **ECB** (`data-api.ecb.europa.eu`) | Deposit facility rate |
| **Riksbank** (`api.riksbank.se`, SWEA) | Policy rate (SECBREPOEFF) |

Connectivity per source is reported as 7-tuples `(label, status, note, freq, n, lo,
hi)`; see `print_connectivity_report`. Brent / WTI / EIA gas are intentionally skipped
(they need an API key) and the pipeline falls back to existing oil/gas columns.

### Provenance of the committed copy

This file was generated from the public sources above (originally cached under
`htboost_results_v5_nor/` in the working tree) and copied here as the canonical shared
panel. Regenerate at any time with `live=True`; because the sources are public it can
be committed without any licensing concern.
