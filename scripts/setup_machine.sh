#!/usr/bin/env bash
# One-command machine bootstrap (mac/linux). Idempotent: safe to re-run.
# Each step echoes its action and the script stops on the first failure, printing
# "SETUP FAILED at step: <step>". On success it prints "SETUP COMPLETE — BLAS OK / HTBoost OK".
#
# Prereqs NOT handled here (see AGENT_SETUP_RUNBOOK.md): git, Miniconda/Anaconda, Julia (juliaup),
# the cloned repo, and the licensed data. Run from the repo root: bash scripts/setup_machine.sh
set -euo pipefail

STEP="init"
trap 'echo; echo "SETUP FAILED at step: $STEP"; exit 1' ERR

ENVNAME="thesis"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Make conda usable in a non-interactive shell (conda init only touches interactive shells).
if ! command -v conda >/dev/null 2>&1; then
    for base in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "$HOME/mambaforge"; do
        if [ -f "$base/etc/profile.d/conda.sh" ]; then . "$base/etc/profile.d/conda.sh"; break; fi
    done
fi
STEP="pre) conda on PATH"
command -v conda >/dev/null 2>&1 || { echo "conda not found — install Miniconda + 'conda init' first (see runbook)"; false; }
echo "== using conda: $(command -v conda) =="

STEP="a) accept conda ToS (pkgs/main, pkgs/r, pkgs/msys2)"; echo "== $STEP =="
for ch in main r msys2; do
    conda tos accept --override-channels --channel "https://repo.anaconda.com/pkgs/$ch" 2>/dev/null \
        && echo "  ToS accepted: pkgs/$ch" \
        || echo "  (conda tos unavailable or already accepted: pkgs/$ch — continuing)"
done

STEP="b) create/update conda env '$ENVNAME'"; echo "== $STEP =="
if conda env list | awk '{print $1}' | grep -qx "$ENVNAME"; then
    echo "  env '$ENVNAME' exists -> updating from environment.yml (--prune)"
    conda env update -n "$ENVNAME" -f environment.yml --prune
else
    echo "  creating env '$ENVNAME' from environment.yml"
    conda env create -f environment.yml
fi

run() { conda run -n "$ENVNAME" --no-capture-output "$@"; }

STEP="c) pip install -e ."; echo "== $STEP =="
run pip install -e .
STEP="c) pip install -e .[htboost]"; echo "== $STEP =="
run pip install -e ".[htboost]"

STEP="d) setup_htboost.py (HybridTreeBoosting + juliacall project)"; echo "== $STEP =="
run python scripts/setup_htboost.py

STEP="e) register Jupyter kernel 'Python ($ENVNAME)'"; echo "== $STEP =="
run python -m ipykernel install --user --name "$ENVNAME" --display-name "Python ($ENVNAME)"

STEP="f) BLAS smoke test"; echo "== $STEP =="
run python -c "import numpy as np; np.linalg.svd(np.random.rand(64,64)); import scipy.linalg as sl; sl.svd(np.random.rand(8,8)); import sklearn; print('BLAS OK')"

STEP="f) Julia smoke test"; echo "== $STEP =="
run python -c "from juliacall import Main as jl; jl.seval('using HybridTreeBoosting, DataFrames, Distributed, SharedArrays, Dates, Random'); print('HTBoost OK')"

echo
echo "SETUP COMPLETE — BLAS OK / HTBoost OK"
