# `results/` — committable derived results

This directory holds **our own derived output** (result CSVs), which is safe to commit
— it is never licensed Bloomberg data. The `.gitignore` `results/` carve-out commits
only result CSVs and this manifest; everything else here is ignored.

## Layout

| Path | Producer | Committed? |
|------|----------|------------|
| `results/pooled_v5/`  | `model_htboost_pooled_v5.ipynb` (`OUT_DIR`)     | CSVs only |
| `results/v5_nor/`     | `model_htboost_v5_clean.ipynb` (`OUT_DIR`) + pooled's `V5_OUT_DIR` | CSVs only |

`results/v5_nor/pooled_metrics_long.csv` is written by the **pooled** notebook (to
`V5_OUT_DIR`) and read by the **per-security** notebook — both resolve to the identical
path, so the cross-notebook comparison merges with zero reconciliation.

## Conventions

- **Per-machine stamping.** The pooled per-horizon files are stamped with `MACHINE_ID`
  (`…/pooled_metrics_wf_H{h}__{MACHINE_ID}.csv`, likewise block-CV and the importance
  pickles), so two machines running the sweep in parallel never clobber each other on
  push. `MACHINE_ID` comes from `src.config` (hostname, or the `THESIS_MACHINE_ID`
  env override).
- **Canonical file.** `pooled_metrics_long.csv` is **unstamped** — a single canonical
  file. Run the final assemble on **one** machine to avoid clobbering it.

## Ignored under `results/` (never committed)

- `*.pkl` — feature-importance pickles (large, regenerable).
- `_pilot/**` — pilot / smoke-test scratch (see the reproducibility handshake), kept
  out so it never mixes with real-run files.
