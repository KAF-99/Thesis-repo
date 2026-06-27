<#
  One-command machine bootstrap (Windows / PowerShell). Idempotent: safe to re-run.
  Each step echoes its action and the script stops on the first failure, printing
  "SETUP FAILED at step: <step>". On success it prints "SETUP COMPLETE - BLAS OK / HTBoost OK".

  Prereqs NOT handled here (see AGENT_SETUP_RUNBOOK.md): git, Miniconda/Anaconda, Julia (juliaup),
  the cloned repo, and the licensed data. Run from the repo root:
      powershell -ExecutionPolicy Bypass -File scripts\setup_machine.ps1
#>
$ErrorActionPreference = 'Stop'
$ENVNAME = 'thesis'
$ROOT = Split-Path -Parent $PSScriptRoot   # scripts\.. = repo root
Set-Location $ROOT
$script:step = 'init'

function Step([string]$name, [scriptblock]$block) {
    $script:step = $name
    Write-Host "== $name =="
    & $block
    if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) { throw "exit code $LASTEXITCODE" }
}

try {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        $script:step = 'pre) conda on PATH'
        throw "conda not found on PATH. Install Miniconda, run 'conda init powershell', and reopen PowerShell (see runbook)."
    }
    Write-Host "== using conda: $((Get-Command conda).Source) =="

    Step 'a) accept conda ToS (pkgs/main, pkgs/r, pkgs/msys2)' {
        foreach ($ch in @('main', 'r', 'msys2')) {
            conda tos accept --override-channels --channel "https://repo.anaconda.com/pkgs/$ch" 2>$null
            if ($LASTEXITCODE -eq 0) { Write-Host "  ToS accepted: pkgs/$ch" }
            else { Write-Host "  (conda tos unavailable or already accepted: pkgs/$ch - continuing)" }
        }
        $global:LASTEXITCODE = 0   # ToS acceptance is best-effort; never fail the run on it
    }

    Step "b) create/update conda env '$ENVNAME'" {
        $exists = (conda env list) | ForEach-Object { ($_ -split '\s+')[0] } | Where-Object { $_ -eq $ENVNAME }
        if ($exists) {
            Write-Host "  env '$ENVNAME' exists -> updating from environment.yml (--prune)"
            conda env update -n $ENVNAME -f environment.yml --prune
        } else {
            Write-Host "  creating env '$ENVNAME' from environment.yml"
            conda env create -f environment.yml
        }
    }

    Step 'c) pip install -e .'           { conda run -n $ENVNAME --no-capture-output pip install -e . }
    Step 'c) pip install -e .[htboost]'  { conda run -n $ENVNAME --no-capture-output pip install -e ".[htboost]" }
    Step 'd) setup_htboost.py (HybridTreeBoosting + juliacall project)' { conda run -n $ENVNAME --no-capture-output python scripts/setup_htboost.py }
    Step "e) register Jupyter kernel 'Python ($ENVNAME)'" { conda run -n $ENVNAME --no-capture-output python -m ipykernel install --user --name $ENVNAME --display-name "Python ($ENVNAME)" }
    Step 'f) BLAS smoke test'  { conda run -n $ENVNAME --no-capture-output python -c "import numpy as np; np.linalg.svd(np.random.rand(64,64)); import scipy.linalg as sl; sl.svd(np.random.rand(8,8)); import sklearn; print('BLAS OK')" }
    Step 'f) Julia smoke test' { conda run -n $ENVNAME --no-capture-output python -c "from juliacall import Main as jl; jl.seval('using HybridTreeBoosting, DataFrames, Distributed, SharedArrays, Dates, Random'); print('HTBoost OK')" }

    Write-Host ''
    Write-Host 'SETUP COMPLETE - BLAS OK / HTBoost OK'
}
catch {
    Write-Host ''
    Write-Host "SETUP FAILED at step: $script:step"
    Write-Host $_.Exception.Message
    exit 1
}
