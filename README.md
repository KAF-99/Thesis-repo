# Thesis — Out-of-Sample Forecasting of Interest-Rate Swap Changes

Code for the master's thesis on forecasting daily changes in interest-rate swap
rates (headline universe: Norwegian tenors) and benchmarking model families on a
single, shared evaluation protocol:

- **HTBoost** (gradient boosting via the Julia bridge) — per-security and pooled.
- **Linear** benchmarks — OLS and Elastic Net.
- **Simple-rule** benchmarks — mean reversion (OU/AR(1)) and time-series momentum.

All models are scored by the same walk-forward / block-CV harness so their results
merge into one comparable long-format table.

> **Licensed data is NOT in this repository.** The Bloomberg swap/rate exports are
> proprietary and are git-ignored. See [`data/raw/README.md`](data/raw/README.md)
> for what each file is and where it comes from. The only committed data file is the
> regenerable Norway public-API cache (`data/cache/norway_raw_features.csv`).

## Repository layout

```
src/                     Installable package (import as `from src.xxx import ...`)
  config.py              Shared constants: paths, maturities, folds, seeds, PCA, MTC alpha
  data/                  bloomberg.py (licensed CSV loader), norway.py (public-API, cache-first)
  features/              macro, security, norway, cross-market PCA, timing policy, panel builder
  models/                tau.py — HTBoost weighted-smoothness (tau^w) diagnostic
  audit/                 leakage.py — past-only feature recomputation audit
  evaluation/            metrics.py — Clark-West/DM-Harvey/PT, schema, MTC families
notebooks/               01_build_dataset + the five model notebooks
data/raw/                (git-ignored) drop the licensed Bloomberg CSVs here
data/cache/              Norway public-API cache (committed) + protocol README
thesis_eval.py           Back-compat shim -> src.evaluation.metrics
htb_metrics.py           Back-compat shim -> src.evaluation.metrics (GB notebooks)
thesis_style.py          Shared Matplotlib figure style + save_fig
data_loader.py           Back-compat shim -> src.data.bloomberg (portable, config-driven paths)
```

## Setup

Conda (recommended):

```bash
conda env create -f environment.yml
conda activate thesis
pip install -e .
```

or pip only:

```bash
pip install -e .            # core: numpy/pandas/scipy/scikit-learn/statsmodels/matplotlib
pip install -e ".[notebooks]"   # + jupyter, nbstripout
```

The HTBoost notebooks additionally require a Julia environment with
`HybridTreeBoosting` and the `juliacall` bridge (`pip install -e ".[htboost]"`);
the linear and simple-rule notebooks run on pure Python.

## Data

Point the loaders at your local copy of the licensed data:

```bash
export THESIS_DATA_PATH=/path/to/your/Data    # default: ./data/raw
```

`src/config.py` reads `THESIS_DATA_PATH` (falling back to `./data/raw`), so each
collaborator keeps the licensed CSVs locally and nothing proprietary is committed.

## Reproducibility invariants

- **One Julia seed** (`src.config.JULIA_SEED`), re-seeded before every HTBoost fit.
- **Cache-first Norway data**: `load_norway_raw(..., live=False)` reads the committed
  cache so both collaborators reproduce the identical panel; `live=True` refetches
  from the Norges Bank / SSB / ECB / Riksbank public APIs and overwrites it.
- **Leakage audit**: features are recomputed from past-only data and compared to the
  stored panel within tolerance before any results are trusted.
- **Shared metrics schema** so per-security, pooled, linear and simple-rule rows
  concatenate with zero reconciliation.

### Methodological decisions (do not silently revert)

These are deliberate analysis choices, not incidental implementation details. A
future refactor that "simplifies" them changes published results — keep them explicit.

- **τ^w aggregation reads the package `gavgtau`.** Both gradient-boosting notebooks
  report the HTBoost weighted-smoothness diagnostic τ^w by reading the package scalar
  `gavgtau` (paper eq. 19 + footnote 8) via `src/models/tau.py::extract_weighted_tau`,
  keeping a cap-40 importance-weighted geometric mean only as a logged cross-check. The
  pooled notebook previously **recomputed** that geometric mean (its field list omitted
  `gavgtau`); it was deliberately aligned to the per-security `gavgtau` definition.
  Reverting to a recomputed mean silently changes the reported τ^w.
- **MTC family scope is dual-mode and intentional.** `apply_mtc_family(df, schemes,
  label, *, model_kind=None)` applies Bonferroni + BH-FDR within one family.
  `model_kind=None` **pools all estimators** into a single family over
  {horizon × tenor × regime} (used by the mean-reversion and momentum notebooks);
  `model_kind=<kind>` restricts the family to **one estimator** (used by the linear
  notebook, called once per OLS / Elastic Net). Pooled vs per-estimator changes the
  family size N and therefore which results survive correction — each notebook keeps
  the mode it was written for; do not unify them.

## Note on `data_loader.py`

`data_loader.py` is a thin **backward-compatibility shim** that re-exports
`src.data.bloomberg` (`load_data`, `_parse_bloomberg_date`, `_load_simple_csv`) and
`MATURITY_NAMES`, so the model notebooks' `from data_loader import ...` keep working
while resolving the data directory via `THESIS_DATA_PATH` (default `./data/raw`) on any
machine — **there is no hard-coded path**. New code should import from
`src.data.bloomberg` / `src.config` directly.
