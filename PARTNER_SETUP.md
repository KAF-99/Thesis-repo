# PARTNER_SETUP.md — onboarding a brand-new machine (Windows & macOS)

Complete, copy-pasteable setup for a collaborator (or a Claude agent acting on their
behalf) to take a **fresh machine** to a **validated REPRO_HANDSHAKE**. Works on
**Windows (PowerShell)** and **macOS (zsh)**. Where commands differ, follow the branch
for your OS. Run steps **in order**.

> **Agent note:** This file is written so a Claude agent in VS Code can execute it
> end-to-end. Run one numbered section at a time, check the stated "expect" output
> before moving on, and stop on the first failure rather than pushing past it.

The repo URL is **https://github.com/KAF-99/Thesis-repo** (private — access required).

---

## 0. What you're installing and why

| Component | Why |
|---|---|
| git + GitHub CLI (`gh`) | clone the private repo |
| Miniconda | the Python env (`environment.yml` → env name **`thesis`**) |
| Julia (juliaup) | the HTBoost gradient-boosting engine runs in Julia via `juliacall` |
| The licensed Bloomberg data | **NOT in the repo** — transferred out-of-band (see §6) |

Two model families need nothing Julia (linear, simple-rule); the two **GB notebooks**
(`model_htboost_v5_clean`, `model_htboost_pooled_v5`) need Julia + HTBoost + data.

---

## 1. Prerequisites from scratch (per OS)

### Windows (PowerShell)

Run PowerShell (no admin needed for `winget --scope user`; use an elevated shell if a
package asks for it).

```powershell
winget install -e --id Git.Git
winget install -e --id GitHub.cli
winget install -e --id Anaconda.Miniconda3
winget install -e --id Julia.Juliaup
```

Close and reopen PowerShell so PATH updates. Then initialise conda for PowerShell once:

```powershell
& "$env:USERPROFILE\miniconda3\Scripts\conda.exe" init powershell
```

Reopen PowerShell again. **Verify:**

```powershell
git --version ; gh --version ; conda --version ; julia --version
```

> If `winget` is unavailable, use the installers: Git → https://git-scm.com/download/win,
> Miniconda → https://docs.conda.io/en/latest/miniconda.html, Julia (juliaup) → the
> Microsoft Store "Julia" app or https://julialang.org/downloads/.

### macOS (zsh)

Install Homebrew if absent (https://brew.sh), then:

```zsh
brew install git gh
brew install --cask miniconda
brew install juliaup
conda init zsh
```

Reopen the terminal. **Verify:**

```zsh
git --version ; gh --version ; conda --version ; julia --version
```

> No Homebrew? Installers: Miniconda → https://docs.conda.io/en/latest/miniconda.html,
> juliaup → `curl -fsSL https://install.julialang.org | sh`.

---

## 2. Clone the private repo (mind line endings)

**Authenticate first** (the repo is private — you must have been granted access):

```text
gh auth login          # choose GitHub.com → HTTPS → login with a browser
```

**Windows only — set line-ending policy BEFORE cloning.** This keeps CSV/notebook
bytes from being mangled (CRLF↔LF), which would change file hashes and break the
data fingerprint:

```powershell
git config --global core.autocrlf input
```

(macOS: no autocrlf change needed — the default is fine.)

Clone (run from your home dir or wherever you keep projects):

```text
gh repo clone KAF-99/Thesis-repo
cd Thesis-repo
```

**Verify:** `git -C . log --oneline -1` shows the latest commit.

---

## 3. Python environment (same on both OSes)

From the repo root:

```text
conda env create -f environment.yml
conda activate thesis
pip install -e .
pip install -e ".[htboost]"
```

- The env is named **`thesis`** (from `environment.yml`).
- `pip install -e .` installs the `src` package (so `from src.config import ...` works anywhere).
- `pip install -e ".[htboost]"` adds the **`juliacall`** Python↔Julia bridge.

> **Shell difference:** the four commands are identical; only how you *activate* differs
> — PowerShell and zsh both use `conda activate thesis` once `conda init` has run (§1).
> If `conda activate` errors on Windows, reopen PowerShell after the `conda init powershell`
> step.

**Verify:** `python -c "import src, numpy, pandas, scipy, sklearn, statsmodels; print('core OK')"`

---

## 4. Install the HTBoost Julia package

`juliacall` keeps an **isolated Julia project per Python env** (`$CONDA_PREFIX/julia_env`),
so the previous step installed only the bridge — not HTBoost. Add it once (idempotent;
pulls **HybridTreeBoosting v0.1.0** from the General registry — no git URL needed):

```text
python scripts/setup_htboost.py
```

First run downloads + precompiles HTBoost and its deps (~minutes). **Verify end-to-end:**

```text
python -c "from juliacall import Main as jl; jl.seval('using HybridTreeBoosting'); print('HTBoost OK')"
```

Expect the last line to be exactly: **`HTBoost OK`**.

---

## 5. Register & select the Jupyter kernel

```text
pip install ipykernel
python -m ipykernel install --user --name thesis --display-name "Python (thesis)"
```

**Verify:** `jupyter kernelspec list` includes `thesis`.

In the GUI (JupyterLab or the VS Code notebook UI), open a notebook and select the
kernel **"Python (thesis)"**. Sanity-check inside the notebook:

```python
import sys; print(sys.executable)   # must point inside .../envs/thesis
```

---

## 6. ⚠️ DATA GAP — the licensed Bloomberg data is NOT in the repo

The proprietary Bloomberg swap/rate exports are **git-ignored and transferred
out-of-band** (secure file transfer from the data owner — never committed, never
emailed in the clear). Without them, only the no-data tier (§7 Tier 1) runs.

**You need a local folder containing** (see `data/raw/README.md` for the full list):
the per-country `<COUNTRY>.csv` swap files **plus** `Interest rates.csv`,
`Interest Rates 2.csv`, `Oil, vol, div.csv`, and `macro_features.csv`.

> **The data folder must be BYTE-IDENTICAL to the source copy.** Do not open/re-save the
> CSVs in Excel, do not let any tool rewrite line endings, do not zip/unzip with a tool
> that converts newlines. A single byte change alters `df_raw.hash16` (see §8).

Point the loaders at your local copy via `THESIS_DATA_PATH`:

### macOS (zsh)

```zsh
export THESIS_DATA_PATH="/Users/you/path/to/Data"
```

### Windows (PowerShell)

```powershell
# current session only:
$env:THESIS_DATA_PATH = "C:\path\to\Data"

# persistent (new shells will inherit it; reopen PowerShell after):
setx THESIS_DATA_PATH "C:\path\to\Data"
```

> `setx` sets it for **future** shells, not the current one — set both (the `$env:` line
> for now, `setx` for later), or reopen the shell after `setx`.

**Verify the path resolves:** (should print the directory and a CSV count > 0)

- macOS: `ls "$THESIS_DATA_PATH"/*.csv | wc -l`
- Windows: `(Get-ChildItem "$env:THESIS_DATA_PATH\*.csv").Count`

If the path is wrong/empty, the loader now fails **early and clearly** with a message
telling you to set `THESIS_DATA_PATH` (instead of a cryptic `./data/raw` error).

---

## 7. Two test tiers

### Tier 1 — env + import sanity (NO data required)

Run from the repo root with the `thesis` env active:

```text
python -c "import src, numpy, pandas, scipy, sklearn, statsmodels; print('core OK')"
python -c "from juliacall import Main as jl; jl.seval('using HybridTreeBoosting'); print('HTBoost OK')"
```

Optional — confirm the data guard is wired (expected to raise a *clear* error, since no
data is set here):

```text
python -c "from src.data.bloomberg import load_data; load_data('./data/raw')"
```

Expect a `FileNotFoundError` whose message names `THESIS_DATA_PATH` and tells you to set
it. That clean message **is** the passing result for this check.

### Tier 2 — REPRO_HANDSHAKE (NEEDS data)

This is the real cross-machine validation. Do it for **both** GB notebooks:
`notebooks/model_htboost_v5_clean.ipynb` and `notebooks/model_htboost_pooled_v5.ipynb`.

1. Make sure `THESIS_DATA_PATH` is set (§6) **in the shell you launch Jupyter from**
   (see §9a about GUI kernels not inheriting env vars).
2. Launch from that shell: `jupyter lab` (or open the notebook in VS Code).
3. Select kernel **"Python (thesis)"**.
4. Run cells **top-to-bottom up to and including** the `REPRO_HANDSHAKE` cell
   (in JupyterLab: select the handshake cell → *Run → Run All Above Selected Cell*,
   then run the handshake cell). This guarantees the data load + all `def` cells ran.
5. The handshake prints a **`FINGERPRINT:`** block (Part A, instant) then runs a
   **~8-minute pilot fit** (Part B) and prints a **`PILOT OK`** (or `PILOT FAIL`) line.
6. **Do NOT run any cell below the handshake** — that would start the full multi-hour sweep.

**Capture and send** (for each GB notebook): the entire `FINGERPRINT:` block **and** the
final `PILOT OK ...` line. Example shape of what to copy:

```text
FINGERPRINT:
  df_raw.shape        = (8183, 138)
  df_raw.hash16       = <16 hex>
  df_raw.index_min    = ...
  df_raw.index_max    = ...
  panel.hash16        = <16 hex>   (or "informational")
  config.hash16       = <16 hex>
  feature_spec.n      = <int>
  feature_spec.hash16 = <16 hex>
  MACHINE_ID          = <your machine id>
  versions            = numpy x / pandas y / scipy z
  julia / HTBoost     = julia 1.x / HybridTreeBoosting 0.1.0
PILOT OK (modality=accurate, ~8m): all output fields present (NOR_10Y H=21 fold=Hiking; ... tau_w=<value>; ...)
```

`PILOT FAIL: ...` names the exact missing field — fix that and re-run the handshake cell.

---

## 8. Cross-check semantics (what must match across machines — and what may not)

Send your `FINGERPRINT` block + `PILOT` line to Knut; he diffs them against his.

| Field | Must match across machines? |
|---|---|
| `config.hash16` | **YES — exactly. OS-independent.** A mismatch means different config constants → reconcile before trusting anything. |
| `feature_spec.hash16` | **YES — exactly. OS-independent.** Mismatch ⇒ different feature set. |
| `df_raw.hash16` | **Only if the data copy is byte-identical.** A mismatch usually means the CSVs differ — most often **CRLF↔LF** mangling on Windows (see §2 `core.autocrlf input` and §6 byte-identical warning), or an Excel re-save. |
| `tau_w` (τ^w) in the PILOT line | **Approximately**, not bit-for-bit. Small Mac↔Windows differences (~`1e-6`) from floating-point / BLAS ordering across OSes are **EXPECTED and acceptable**. |

> **Do not treat a tiny τ^w difference across OSes as a failure.** Demanding bit-identical
> τ across operating systems would be wrong — cross-OS FP/BLAS divergence is normal. The
> tolerance is ~`1e-6`. What must be bit-identical are the **hashes** of config and
> feature_spec (and df_raw given identical data), because those are pure byte digests, not
> floating-point computations.

A green cross-check = identical `config.hash16` + `feature_spec.hash16` on every machine,
identical `df_raw.hash16` wherever the data copy is byte-identical, and `PILOT OK` with a
`tau_w` within tolerance.

---

## 9. Troubleshooting (per OS where relevant)

**(a) GUI Jupyter kernel doesn't see `THESIS_DATA_PATH`.** Jupyter/VS Code kernels do not
always inherit shell exports. Two fixes: **(i)** launch `jupyter lab` from the same shell
where you set `THESIS_DATA_PATH` (§6); or **(ii)** set it in-notebook *before* importing
the loaders:
```python
import os; os.environ["THESIS_DATA_PATH"] = r"C:\path\to\Data"   # macOS: "/Users/you/path/to/Data"
```
On Windows, `setx` (§6) + reopening the launching shell also makes it persistent.

**(b) `NameError` / missing `def`s at the handshake.** You skipped cells. Always run
**top-to-bottom** (*Run All Above* the handshake) so every data-load and `def` cell
executes before the handshake.

**(c) `using HybridTreeBoosting` fails / PILOT can't find HTBoost.** You skipped §4. Run
`python scripts/setup_htboost.py` **before** opening any GB notebook (it installs HTBoost
into this env's `julia_env`). Re-select the kernel / restart it afterward.

**(d) Never run cells *below* the handshake.** Those launch the full sweep (hours). The
pilot in the handshake is the only fit you run for validation.

**(e) Windows path & CRLF gotchas.**
- Use a raw string or double backslashes for Windows paths in Python: `r"C:\path\to\Data"`.
- If `df_raw.hash16` differs only on Windows, suspect line-ending conversion: confirm
  `git config --global core.autocrlf` prints `input` (§2), re-clone if it was set wrong,
  and make sure the data CSVs were copied **without** newline conversion.
- If `conda` or `julia` "isn't recognized", reopen PowerShell (PATH updates after install)
  and ensure `conda init powershell` ran (§1).

---

### Done

Tier 1 green + Tier 2 `PILOT OK` on both GB notebooks, with matching `config.hash16` /
`feature_spec.hash16`, means this machine reproduces the pipeline. Send your two
FINGERPRINT blocks + PILOT lines for the cross-machine diff.
