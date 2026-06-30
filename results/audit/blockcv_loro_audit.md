# §8.6 Audit — Block-CV vs LORO fold partitions

_Read-only. Model **htboost_t** (per-security HTBoost), security `NOR_10Y`, **H=21**. No fit performed. `_blockcv_entries` + the two-sided purge+embargo masking are reproduced verbatim from `run_security_blockcv` (`model_htboost_v5_clean.ipynb`); folds from `config.BLOCK_CV_FOLDS`; date index = canonical NOR_10Y trading days from `load_data()` (df_raw_values.hash16=`798f8764c5f695b1`=canonical). gap=(H-1)+EMBARGO_FOR_H(H)=20+21=41 business days, two-sided._

## Fold definitions (side by side)
```
config.BLOCK_CV_FOLDS:
  Block_GFC    2007-01-01 .. 2012-12-31   regime=GFC
  Block_ZIRP   2013-01-01 .. 2019-12-31   regime=ZIRP
  Block_COVID  2020-01-01 .. 2021-12-31   regime=COVID
  Block_Hiking 2022-01-01 .. 2026-12-31   regime=Hiking
```

`_blockcv_entries(scheme)`:
- **block_cv** → one fold per row; test = that single contiguous window.
- **loro** → one fold per *unique regime*; test = union of that regime’s segments.

Regime→#blocks multiplicity: `{'GFC': 1, 'ZIRP': 1, 'COVID': 1, 'Hiking': 1}` → **1:1 (one contiguous block per regime)**.

## Per-fold train/test set comparison

| regime | block_cv label | loro label | n_test | n_train | test identical | train identical |
|---|---|---|---:|---:|:---:|:---:|
| GFC | Block_GFC | LORO_GFC | 1479 | 3587 | YES | YES |
| ZIRP | Block_ZIRP | LORO_ZIRP | 1808 | 3260 | YES | YES |
| COVID | Block_COVID | LORO_COVID | 542 | 4516 | YES | YES |
| Hiking | Block_Hiking | LORO_Hiking | 1132 | 3967 | YES | YES |

## Verdict

**COINCIDE BY CONSTRUCTION.** `BLOCK_CV_FOLDS` defines exactly **one contiguous calendar block per regime** (1:1 regime↔block), so LORO’s "leave one regime out" groups a single segment — the identical window block-CV holds out. Identical test masks ⇒ identical complement ⇒ identical two-sided purge+embargo ⇒ **identical train and test date-index sets for every fold**; only the fold *label* differs (`Block_X` vs `LORO_X`).

**§8.6 implication:** the near-identical block-CV vs LORO metrics are **expected, not a bug** — same partition. Present them as **one scheme with a methodological note** (LORO ≡ blocked-by-regime when each regime is a single contiguous window), not two independent results. They would only diverge if a regime were split across multiple non-contiguous `BLOCK_CV_FOLDS` entries.


---
FINGERPRINT: Block and LORO partitions are IDENTICAL for htboost_t H21; verdict = by-construction.
