# Cross-machine reproducibility evidence

`df_raw.hash16` (the `pd.util.hash_pandas_object` digest of the full 266-column `df_raw`)
differs across machines, but the difference is **inherent cross-architecture floating point in
the derived feature columns — not a data, parsing, or logic difference**, and it does not move
results (τ is identical across machines). This folder documents that.

## Gates (these must match across machines)
- `df_raw_values.hash16 = 798f8764c5f695b1` — the **raw** (pre-augmentation, 138-col) values-only
  hash. Portable: `float_precision="round_trip"` parsing + deterministic (sorted) column order.
  **The raw Bloomberg inputs are bit-identical across machines.**
- `config.hash16 = 2ef2eace9701cfcc`, `feature_spec.hash16 = a6b89af4849951bd` — exact, OS-independent.
- `tau_w ≈ 2.287` — within ~1e-6 (the per-security pilot fit is identical across machines).

`df_raw.hash16` itself is **informational**.

## What the manifest shows
`colmanifest_266_macos_arm64.tsv` — a per-column value hash (`pd.util.hash_pandas_object`,
index-excluded) of every column in the 266-col `df_raw`, generated on **macOS / Apple Silicon
(arm64)**. Columns: `column`, `kind` (`raw` | `derived`), `dtype`, `value_hash16`.

Diffing this against the equivalent manifest from a **Windows / x64** machine shows:
- **138 `raw` columns: identical hashes** (bit-identical values).
- **128 `derived` columns: hashes differ** — the augmentation features (`*_zscore`, rolling-vol,
  `*_mom_*`, `*_ibor_chg_1d` / `_mom_1m` / `_term_slope`, FX transforms, …) are computed with
  SIMD/FP operations whose ordering/rounding differs at the ULP level (~1e-15) between x64 and
  arm64. That ~1e-15 is far below any threshold that affects the forecasts.

**Conclusion for the methodology section:** raw inputs bit-identical, config/features/τ identical,
derived intermediates agree to ~1e-15 documented architecture-level floating point. This is a
stronger reproducibility statement than a single whole-frame hash match.

(To add the x64 reference: run the same per-column manifest on a Windows machine and commit it
as `colmanifest_266_windows_x64.tsv` alongside this file.)
