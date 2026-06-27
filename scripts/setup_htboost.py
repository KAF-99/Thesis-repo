#!/usr/bin/env python
"""One-time HTBoost/juliacall setup for the active conda env.

Does two things, both idempotent:

1. GAP A — pin juliacall's Julia project to this env. juliacall resolves its Julia
   project from PYTHON_JULIAPKG_PROJECT; on Windows/VS Code a fresh shell may not point
   at the env's project, so `using DataFrames` / `using HybridTreeBoosting` fail to
   resolve. We set it for this process AND write conda activate.d hooks (sh/bat/ps1) so
   every future shell exports PYTHON_JULIAPKG_PROJECT=$CONDA_PREFIX/julia_env.

2. GAP / HTBoost — add the FULL set of direct Julia deps the notebooks `using`
   (HybridTreeBoosting, DataFrames, Distributed, SharedArrays, Dates, Random) to that
   project and precompile them. juliacall keeps an isolated Julia project per Python env, so
   `pip install -e ".[htboost]"` installs only the bridge; and `using X` needs X as a DIRECT
   dep (transitive deps are not importable), so all six must be added explicitly.

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

# Every `using X` in the notebooks (and the bootstrap smoke test) needs X to be a DIRECT
# dependency of this env's Julia project — a transitive dep (e.g. DataFrames pulled in by
# HybridTreeBoosting) is NOT importable. The stdlibs (Distributed/SharedArrays/Dates/Random)
# likewise must be added explicitly to land in [deps]. Pkg.add is idempotent.
print(f"Ensuring Julia direct deps in {active}: "
      "HybridTreeBoosting, DataFrames, Distributed, SharedArrays, Dates, Random")
jl.seval('Pkg.add(["HybridTreeBoosting", "DataFrames", "Distributed", "SharedArrays", "Dates", "Random"])')
jl.seval("using HybridTreeBoosting, DataFrames, Distributed, SharedArrays, Dates, Random")  # precompile all
print("Julia deps ready — `using DataFrames, HybridTreeBoosting, Distributed, SharedArrays, Dates, Random` succeeds.")
