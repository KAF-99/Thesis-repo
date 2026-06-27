#!/usr/bin/env python
"""One-time HTBoost/juliacall setup for the active conda env.

Does two things, both idempotent:

1. GAP A — pin juliacall's Julia project to this env. juliacall resolves its Julia
   project from PYTHON_JULIAPKG_PROJECT; on Windows/VS Code a fresh shell may not point
   at the env's project, so `using DataFrames` / `using HybridTreeBoosting` fail to
   resolve. We set it for this process AND write conda activate.d hooks (sh/bat/ps1) so
   every future shell exports PYTHON_JULIAPKG_PROJECT=$CONDA_PREFIX/julia_env.

2. GAP / HTBoost — ensure the (registered) HybridTreeBoosting Julia package is present in
   that project and precompile it. juliacall keeps an isolated Julia project per Python
   env, so `pip install -e ".[htboost]"` installs only the bridge — not HybridTreeBoosting.

Run:  python scripts/setup_htboost.py   (thesis conda env activated)
"""
import os
import sys
from pathlib import Path

_PREFIX = os.environ.get("CONDA_PREFIX") or sys.prefix
_JULIA_ENV = os.path.join(_PREFIX, "julia_env")


def _wire_activate_hooks() -> Path:
    """Write conda activate.d hooks so PYTHON_JULIAPKG_PROJECT survives fresh shells
    (PowerShell/cmd/bash) and VS Code-launched kernels. Cross-platform: all three are
    written; each OS's conda activation runs the one it understands."""
    actd = Path(_PREFIX) / "etc" / "conda" / "activate.d"
    actd.mkdir(parents=True, exist_ok=True)
    (actd / "julia_env.sh").write_text(
        'export PYTHON_JULIAPKG_PROJECT="$CONDA_PREFIX/julia_env"\n')
    (actd / "julia_env.bat").write_text(
        '@set "PYTHON_JULIAPKG_PROJECT=%CONDA_PREFIX%\\julia_env"\n')
    (actd / "julia_env.ps1").write_text(
        '$Env:PYTHON_JULIAPKG_PROJECT = "$Env:CONDA_PREFIX\\julia_env"\n')
    return actd


# Point juliacall at this env's project for THIS process too, before juliacall imports.
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", _JULIA_ENV)
_hooks = _wire_activate_hooks()
print(f"PYTHON_JULIAPKG_PROJECT -> {os.environ['PYTHON_JULIAPKG_PROJECT']}")
print(f"  activate.d hooks (sh/bat/ps1) written under {_hooks}")

from juliacall import Main as jl  # noqa: E402  (must follow the env-var wiring above)

jl.seval("import Pkg")
active = jl.seval("Base.active_project()")
if jl.seval('haskey(Pkg.project().dependencies, "HybridTreeBoosting")'):
    print(f"HybridTreeBoosting already present in {active}")
else:
    print(f"Adding HybridTreeBoosting (General registry, v0.1.0) -> {active}")
    jl.seval('Pkg.add("HybridTreeBoosting")')
jl.seval("using HybridTreeBoosting")   # precompile now so the first notebook fit isn't blocked
print("HTBoost ready — `using HybridTreeBoosting` succeeds.")
