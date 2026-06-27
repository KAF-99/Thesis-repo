# AGENT_SETUP_RUNBOOK.md — fresh-machine setup, written for a coding agent

> **You are a coding agent setting up this repo on a brand-new machine.** Execute the steps
> **in order**. After each step, check the **Expect** line. If output does not match,
> **STOP and report** the step + the actual output — do **not** improvise, retry blindly, or
> skip ahead. Several steps are **HUMAN** actions (auth, licensed data): at those, **pause**
> and tell the human exactly what to do; do not attempt them yourself.

Repo: **https://github.com/KAF-99/Thesis-repo** (private).
Canonical reproducibility fingerprint (compare at the end — all must match):
`df_raw.hash16 = e7cbef30f804bae4`, `config.hash16 = 2ef2eace9701cfcc`,
`feature_spec.hash16 = a6b89af4849951bd`, `tau_w ≈ 2.287`.

---

## Step 0 — Detect OS and pick the bootstrap script

- **Windows** → `scripts\setup_machine.ps1` (PowerShell), env vars via `$env:` / `setx`.
- **macOS / Linux** → `scripts/setup_machine.sh` (bash), env vars via `export`.

Use the matching branch throughout. (`python -c "import platform;print(platform.system())"` if unsure: `Windows` / `Darwin` / `Linux`.)

---

## Step 1 — Prerequisites the script can't install (git, conda, Julia)

Install these first, then verify each. **STOP if any verify fails.**

### Windows (PowerShell)
```powershell
winget install -e --id Git.Git                --accept-source-agreements --accept-package-agreements --disable-interactivity
winget install -e --id Anaconda.Miniconda3    --accept-source-agreements --accept-package-agreements --disable-interactivity
winget install -e --id Julialang.Juliaup      --accept-source-agreements --accept-package-agreements --disable-interactivity
winget install -e --id GitHub.cli             --accept-source-agreements --accept-package-agreements --disable-interactivity
```
Then initialise conda for PowerShell and **reopen the shell**:
```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
```

### macOS / Linux
```bash
# git: usually present (Xcode CLT on mac: xcode-select --install). Homebrew shown; installers also fine.
brew install git gh
brew install --cask miniconda   # or the Miniconda installer
brew install juliaup            # or: curl -fsSL https://install.julialang.org | sh
conda init "$(basename "$SHELL")"   # then reopen the shell
```

**Verify (both OSes):**
```text
git --version
julia --version
conda --version
```
**Expect:** three version lines, no "not recognized". **STOP** if `conda`/`julia` is missing — on Windows this usually means PATH hasn't refreshed: reopen PowerShell and ensure `conda init powershell` ran.

> Exact winget IDs (case-sensitive): `Git.Git`, `Anaconda.Miniconda3`, **`Julialang.Juliaup`**, `GitHub.cli`.

---

## Step 2 — HUMAN: GitHub auth (the repo is private)

**Pause.** Tell the human: *"The repo is private — please run `gh auth login` (GitHub.com → HTTPS → browser) or otherwise grant this machine collaborator access, then tell me to continue."*
**Do NOT authenticate to GitHub yourself.** Wait for the human to confirm before cloning.

---

## Step 3 — Clone (Windows: set line-ending policy FIRST)

**Windows only — BEFORE cloning** (prevents CRLF byte-mangling of CSV/notebook bytes):
```powershell
git config --global core.autocrlf input
```
Then clone and enter the repo (both OSes):
```text
gh repo clone KAF-99/Thesis-repo
cd Thesis-repo
```
**Verify:** `git -C . log --oneline -1`
**Expect:** the latest commit prints. **STOP** if clone failed (almost always Step 2 auth).

---

## Step 4 — Run the bootstrap script

From the repo root:

- **Windows:** `powershell -ExecutionPolicy Bypass -File scripts\setup_machine.ps1`
- **macOS / Linux:** `bash scripts/setup_machine.sh`

The script is idempotent and does: (a) accept conda ToS for pkgs/main·r·msys2; (b) create/update
the `thesis` env from `environment.yml`; (c) `pip install -e .` then `pip install -e ".[htboost]"`;
(d) `python scripts/setup_htboost.py`; (e) register the `Python (thesis)` Jupyter kernel; (f) BLAS +
Julia smoke tests.

**Expect (last line):** `SETUP COMPLETE — BLAS OK / HTBoost OK`
**STOP** on `SETUP FAILED at step: <step>` — report the step verbatim and consult the **Troubleshooting** appendix for that step. Do not hand-patch around it.

---

## Step 5 — HUMAN: place the licensed data + set THESIS_DATA_PATH

**Pause.** The licensed Bloomberg data is **NOT in the repo** (proprietary, transferred out-of-band).
Tell the human exactly what's needed:
> *"I need the Bloomberg data folder — the ~18 CSVs: the per-country `<COUNTRY>.csv` files plus
> `Interest rates.csv`, `Interest Rates 2.csv`, `Oil, vol, div.csv`, `macro_features.csv` — copied
> to this machine WITHOUT any re-save/line-ending conversion. Then give me the folder path."*

Once the human gives the path, set the env var (the agent may do this part):
- **macOS/Linux:** `export THESIS_DATA_PATH="/path/to/Data"`
- **Windows:** `$env:THESIS_DATA_PATH = "C:\path\to\Data"`  (and `setx THESIS_DATA_PATH "C:\path\to\Data"` to persist)

**Verify count > 0:**
- macOS/Linux: `ls "$THESIS_DATA_PATH"/*.csv | wc -l`
- Windows: `(Get-ChildItem "$env:THESIS_DATA_PATH\*.csv").Count`

**Do NOT proceed to Step 6/7 until the data path is confirmed.** Leave `NORWAY_LIVE_FETCH` **unset**
(offline cache-first build).

---

## Step 6 — Verify the 266-column offline build

```text
conda run -n thesis python -c "import os; os.environ.pop('NORWAY_LIVE_FETCH',None); from src.data.bloomberg import load_data; import src.config as c; print(load_data(c.DATA_PATH).shape)"
```
**Expect:** `(8183, 138)` here (raw load). The full **266** columns are built inside the notebook
(load → norway cache → macro). If you instead want the full check, it is produced by the handshake
in Step 7 (`df_raw.shape = (8183, 266)`). **STOP** if the raw load raises (a `FileNotFoundError`
naming `THESIS_DATA_PATH` means Step 5 isn't set in this shell).

---

## Step 7 — Run the handshake (captured, no GUI) + report the FINGERPRINT

```text
conda run -n thesis python scripts/run_handshake.py notebooks/model_htboost_v5_clean.ipynb
```
(Offline cache-first; `THESIS_DATA_PATH` set, `NORWAY_LIVE_FETCH` unset. The runner truncates to the
REPRO_HANDSHAKE cell — **no sweep** — and tees the fingerprint under `results/v5_nor/_pilot/`. If it
errors with "kernel not found", add `HANDSHAKE_KERNEL=thesis` to the command.)

**Expect:** a `FINGERPRINT:` block then `PILOT OK (modality=accurate, …)`. **Paste the full
FINGERPRINT block back to the human.** Cross-check against the canonical:

| Field | Must match? |
|---|---|
| `df_raw.hash16 = e7cbef30f804bae4` | **YES, exactly.** Column load order is now sorted/deterministic, so this hash is portable across machines. If it differs while config/feature_spec/τ match, it's a pandas-version datetime difference — see G7. |
| `config.hash16 = 2ef2eace9701cfcc` | **YES, exactly** (OS-independent) |
| `feature_spec.hash16 = a6b89af4849951bd` | **YES, exactly** (OS-independent) |
| `tau_w ≈ 2.287` | within ~1e-6 (cross-OS FP/BLAS) |
| `df_raw.shape = (8183, 266)` | **YES** |

**STOP & report** if `config.hash16` or `feature_spec.hash16` differs (real input/config divergence —
reconcile before any sweep), or if you get `PILOT FAIL: <field>` (report the field). If only
`df_raw.hash16` differs (config/feature_spec/τ all match), it's the pandas-version artifact in G7 —
report it but it is not a blocker.

---

## Explicit STOP conditions (do NOT do these)
- **Never run any notebook cell *below* the REPRO_HANDSHAKE cell** — that launches the multi-hour sweep.
- **Never `pip install numpy`, `scipy`, or `scikit-learn`** — they come from conda (OpenBLAS); a pip BLAS crashes `np.linalg.svd`. `pyproject.toml` deliberately omits them.
- **Never authenticate to GitHub on the human's behalf** (Step 2 is human).
- **Never proceed past Step 5 without the data path confirmed** by the human.
- On any `SETUP FAILED` / mismatch: **stop and report**, don't improvise.

---

## Troubleshooting appendix (G1–G8 — observed gotchas)

- **G1 — `conda activate` not recognized (Windows).** PATH hasn't refreshed; reopen PowerShell after `conda init powershell` (Step 1). The script uses `conda run -n thesis` to avoid activation entirely.
- **G2 — conda ToS error ("Terms of Service have not been accepted").** Step (a) accepts pkgs/main·r·msys2. If it surfaces anyway: `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main` (repeat for r, msys2).
- **G3 — winget Julia not found.** The ID is **`Julialang.Juliaup`** (not "Julia"); add `--accept-source-agreements --accept-package-agreements --disable-interactivity`.
- **G4 — `np.linalg.svd` crash / fatal `0xc06d007f` (Windows) — BLAS clash.** numpy/scipy are on different LAPACKs. Re-create the env from `environment.yml` (pins `libblas=*=*openblas`); never `pip install` numpy/scipy/scikit-learn. The script's step (f) BLAS test catches this.
- **G5 — Julia `Package DataFrames/HybridTreeBoosting not found` — juliacall project not wired.** `python scripts/setup_htboost.py` sets `PYTHON_JULIAPKG_PROJECT` + writes conda activate.d hooks; **reopen the shell** afterward. Confirm `PYTHON_JULIAPKG_PROJECT` points at `…/envs/thesis/julia_env`.
- **G6 — GUI Jupyter kernel doesn't see `THESIS_DATA_PATH`.** Kernels don't inherit shell exports; the runbook uses captured `run_handshake.py` from a shell where the var is set. In a notebook, set `os.environ["THESIS_DATA_PATH"]=...` before importing the loaders.
- **G7 — `df_raw.hash16` ≠ `e7cbef30f804bae4` but config/feature_spec/τ match.** Column load order is now deterministic (sorted in `load_data`), so a remaining mismatch means a pandas-version datetime-unit difference (`datetime64[ns]` vs `[us]`), which changes `pd.util.hash_pandas_object` even with identical values. Re-create the env from `environment.yml` so pandas matches the fleet; report it, but config/feature_spec/τ remain the definitive reproducibility check — not a blocker.
- **G8 — `df_raw.shape = (8183, 138)` instead of 266.** The norway augmentation didn't run (offline cache path). Ensure the clone is current and `NORWAY_LIVE_FETCH` is unset (cache-first builds 266 from the committed `data/cache/norway_raw_features.csv`).
