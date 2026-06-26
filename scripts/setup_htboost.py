#!/usr/bin/env python
"""One-time: ensure the (registered) HybridTreeBoosting Julia package is present in the
active conda env's juliacall project ($CONDA_PREFIX/julia_env), then precompile it.

juliacall keeps an isolated Julia project per Python env, so `pip install -e ".[htboost]"`
installs only the bridge — not HybridTreeBoosting. A fresh clone fails at
`using HybridTreeBoosting` until this runs once. Idempotent.

Run:  python scripts/setup_htboost.py   (thesis conda env activated)
"""
from juliacall import Main as jl

jl.seval("import Pkg")
active = jl.seval("Base.active_project()")
if jl.seval('haskey(Pkg.project().dependencies, "HybridTreeBoosting")'):
    print(f"HybridTreeBoosting already present in {active}")
else:
    print(f"Adding HybridTreeBoosting (General registry, v0.1.0) -> {active}")
    jl.seval('Pkg.add("HybridTreeBoosting")')
jl.seval("using HybridTreeBoosting")   # precompile now so the first notebook fit isn't blocked
print("HTBoost ready — `using HybridTreeBoosting` succeeds.")
