# ════════════════════════════════════════════════════════════════════════════════════
# GATE 1 — L2 pooled-HTBoost speed-stack probe: regime/security-STRATIFIED slice + 2 arms
# Paste into model_htboost_pooled_v5.ipynb AFTER the REPRO_HANDSHAKE cell (≈39) — it runs
# after the bridge (≈25), so ALL of these are live by then:
#   build_panel_pooled, feature_columns, _prep_fit_matrix, fit_htboost_pooled,
#   prepare_x_pooled, _to_julia_dates, pooled_metrics_row, jl, JULIA_SEED, UNIVERSE,
#   SCORE_UNIVERSE, FEAT_SPEC, FROZEN_CONFIG, VALID_HTB_FIELDS, WF_FOLDS, REGIMES,
#   AND the fingerprint vars _raw_vfp / _feat / _sha16 / _json (defined by cell ≈39).
# DO NOT RUN until the slice is approved (this is the one place we spend compute).
# Vary ONLY the speed config between arms. loss=:L2, seed, slice, test block, feature_spec
# are IDENTICAL across A and B. No push / commit / stamp — pure validation probe.
# ════════════════════════════════════════════════════════════════════════════════════
import time, json, numpy as np, pandas as pd

# ── 0. FINGERPRINT GATE (POOLED) — executable, RAISES before any fit. Two HARD gates for the
#   pooled notebook: (1) DATA (df_raw_values.hash16) and (2) CONFIG (pooled config.hash16),
#   recomputed/referenced the SAME way as the REPRO_HANDSHAKE. The FEATURE-SPEC half is adjusted
#   for the pooled structure: the pooled handshake builds the panel PER-ARM, so `_feat` is None at
#   the fingerprint point and cannot be hashed here. Per the original pooled fleet-launch precedent
#   (Bob couldn't compute it either; two machines agreeing + identical committed code covered it),
#   we SKIP the feature-spec HARD-assert in pooled and rely on DATA + CONFIG + identical code.
#   Requires the REPRO_HANDSHAKE cell to have RUN first (defines _raw_vfp, _sha16, _json).
_CANON_RAW         = '798f8764c5f695b1'   # pooled df_raw_values.hash16 (HARD data gate)
_CANON_CFG_POOLED  = '0df8d9704167d1e0'   # pooled config.hash16    (HARD config gate; NOT per-sec 2ef2eace…)
_CANON_FEAT_POOLED = 'a5fb6d0229a848f3'   # pooled feature_spec.hash16 (SOFT — only if _feat is available)
_g = globals()
for _v in ('_raw_vfp', '_sha16', '_json'):
    if _v not in _g:
        raise RuntimeError(f'FINGERPRINT GATE: {_v!r} not in scope — run the REPRO_HANDSHAKE '
                           f'cell BEFORE this probe so the fingerprints exist.')
# (1) DATA GATE — HARD. Stays a raise; this is the gate that caught the divergent (dafc5457…) data.
if not (isinstance(_raw_vfp, str) and _raw_vfp == _CANON_RAW):
    raise RuntimeError(f'DATA GATE FAIL: df_raw_values.hash16={_raw_vfp!r} != canonical '
                       f'{_CANON_RAW!r} — NOT the same data. Halting before any fit.')
# (2) CONFIG GATE — HARD. Recomputed exactly as the handshake (cell ~39) does, pooled value.
import src.config as _Cgate
_cfg_gate = {'JULIA_SEED': _Cgate.JULIA_SEED, 'H_GRID': _Cgate.H_GRID, 'WF_FOLDS': _Cgate.WF_FOLDS,
             'BLOCK_CV_FOLDS': _Cgate.BLOCK_CV_FOLDS, 'NOR_TENORS': _Cgate.NOR_TENORS,
             'MIN_TRAIN_OBS': _Cgate.MIN_TRAIN_OBS, 'MIN_TEST_OBS': _Cgate.MIN_TEST_OBS,
             'ALPHA_MT': _Cgate.ALPHA_MT, 'XM_PCA_ENABLE': _Cgate.XM_PCA_ENABLE,
             'XM_PCA_VAR': _Cgate.XM_PCA_VAR, 'XM_PCA_KMAX': _Cgate.XM_PCA_KMAX,
             'FROZEN_CONFIG': _g.get('FROZEN_CONFIG')}
_cfg_hash = _sha16(_json.dumps(_cfg_gate, sort_keys=True, default=str).encode())
if _cfg_hash != _CANON_CFG_POOLED:
    raise RuntimeError(f'CONFIG GATE FAIL: pooled config.hash16={_cfg_hash!r} != canonical '
                       f'{_CANON_CFG_POOLED!r} — config differs. Halting before any fit.')
# (3) FEATURE-SPEC — pooled: _feat is None at handshake (panel built per-arm) → SKIP (logged),
#     covered by the two HARD gates above + identical committed code. If a panel-derived _feat IS
#     available, do a SOFT cross-check against the committed pooled value (warn only, never block).
_feat_probe = _g.get('_feat')
if _feat_probe is None:
    print('feature-spec gate: SKIPPED (pooled — panel not a global at handshake, _feat=None). '
          'Relying on the HARD data+config gates + identical committed code, per pooled fleet-launch precedent.')
else:
    _feat_hash = _sha16(_json.dumps(_feat_probe, sort_keys=True).encode())
    _ok = (_feat_hash == _CANON_FEAT_POOLED)
    print(f'feature-spec gate (soft): feature_spec.hash16={_feat_hash} '
          f'{"== canonical" if _ok else "!= "+_CANON_FEAT_POOLED+" (WARN, not blocking)"}')
print(f'fingerprint gate PASS (HARD data+config): df_raw_values.hash16={_raw_vfp}  '
      f'config.hash16={_cfg_hash}  (canonical pooled)')

PROBE_H        = 5                     # primary horizon (matches the ~440k headline fold); OVERLAP = H-1 = 4
PROBE_FOLD     = 'Hiking'              # WF fold: test 2022-01-01 → 2026-12-31, train = expanding 2007–2021
SLICE_TARGET   = 70_000               # stratified training-slice size (see GATE 1 justification)
INNER_CV_SHARE = 0.20                 # edit 1: subsample-CV fraction → ~14k (clears the 8.7k failure floor)
SEED           = JULIA_SEED            # 20260619 — identical across arms
RNG            = np.random.default_rng(SEED)

# ── 1. Build the FULL Hiking-fold pooled panel + causal tr/te split (exactly like run_pooled_wf) ──
_fold = next(f for f in WF_FOLDS if f[0] == PROBE_FOLD)
_ts_ts, _te_ts = pd.Timestamp(_fold[1]), pd.Timestamp(_fold[2])
_purge_ts = _ts_ts - pd.tseries.offsets.BDay(PROBE_H - 1)            # OVERLAP = H-1 purge
panel      = build_panel_pooled(df_raw, UNIVERSE, PROBE_H)          # full universe (all securities)
numeric, cats = feature_columns(panel, use_categoricals=True)
tr_full = panel[panel['date'] <  _purge_ts].copy()                  # ~440k training rows
te_real = panel[(panel['date'] >= _ts_ts) & (panel['date'] <= _te_ts)].copy()  # REAL Hiking test block (NOT subsampled)

# ── 2. MANDATORY regime × security stratification (GATE-0 finding: package subsamples random-row,
#       so we impose the balance ourselves). Strata = security × training-regime-period. Same
#       per-stratum fraction → preserves the full 88-security cross-section AND the GFC/ZIRP/COVID mix.
def _regime_of(d):
    d = pd.Timestamp(d)
    if d <= pd.Timestamp('2012-12-31'): return 'GFC'
    if d <= pd.Timestamp('2019-12-31'): return 'ZIRP'
    return 'COVID'                                                   # 2020–2021 (Hiking-train tail)

def stratified_slice(df, target, rng):
    df = df.copy()
    df['_regime'] = df['date'].map(_regime_of)
    frac = min(1.0, target / len(df))
    parts = []
    for _, g in df.groupby(['security', '_regime'], sort=False):
        k = max(1, int(round(len(g) * frac)))                       # ≥1 row/stratum → every security kept
        parts.append(g.sample(n=min(k, len(g)), random_state=int(rng.integers(1e9))))
    out = pd.concat(parts).sort_values('date').drop(columns='_regime')
    return out

tr_slice = stratified_slice(tr_full, SLICE_TARGET, RNG)              # ≈70k, all securities, regime mix intact
print(f'slice: {len(tr_slice):,} train rows | securities={tr_slice["security"].nunique()} '
      f'| regimes={ {r:int((tr_slice["date"].map(_regime_of)==r).sum()) for r in ("GFC","ZIRP","COVID")} } '
      f'| test block={len(te_real):,} rows (real, full)')

# PCA-compress xm_* on the SLICE training rows only (v5 rule), reattach categoricals
x_tr, x_te, pca = _prep_fit_matrix(tr_slice, te_real, numeric, cats)

# ── 3. Speed-aware param builder (extends the bridge's _build_param with the edit-3 grid + fixed depth) ──
def build_param_speed(H, *, modality, nfold, depth_coarse_grid=None,
                      depth=None, sparsity_penalization=None, ntrees=None,
                      loss='L2', priortype=FROZEN_CONFIG.get('priortype')):
    kw = dict(loss=loss, modality=modality, nfold=int(nfold),
              randomizecv=False, overlap=int(H - 1), verbose='Off')
    if priortype and 'priortype' in VALID_HTB_FIELDS: kw['priortype'] = str(priortype)
    if depth_coarse_grid is not None:                               # edit 3
        g1, g2 = depth_coarse_grid
        kw['depth_coarse_grid'], kw['depth_coarse_grid2'] = int(g1), int(g2)
    if depth is not None:                 kw['depth'] = int(depth)
    if sparsity_penalization is not None: kw['sparsity_penalization'] = float(sparsity_penalization)
    if ntrees is not None:                kw['ntrees'] = int(ntrees)
    bad = [k for k in kw if k not in VALID_HTB_FIELDS and k != 'verbose']
    assert not bad, f'unknown HTBparam keys: {bad}'                 # depth_coarse_grid etc. confirmed valid in GATE 0
    return jl.HTBparam(**kw)

def _fit_y(y, param, x_train_df, dates, x_test_df):
    x_jl = jl.DataFrame(prepare_x_pooled(x_train_df, cats))
    data = jl.HTBdata(np.asarray(y, float), x_jl, param, _to_julia_dates(dates))
    jl.seval(f'Random.seed!({SEED})')                              # determinism, identical across arms
    out  = jl.HTBfit(data, param)
    yhat_te = np.asarray(jl.HTBpredict(jl.DataFrame(prepare_x_pooled(x_test_df, cats)), out), float)
    return out, yhat_te

def _oos(yhat):
    m = pooled_metrics_row(te_real['y'].to_numpy(float), np.asarray(yhat, float), PROBE_H,
                           {'security': '__POOL__', 'agg_level': 'aggregate', 'sample': 'oos',
                            'validation_scheme': 'walk_forward', 'fold': PROBE_FOLD, 'regime': 'Hiking'})
    return {k: m.get(k) for k in ('mse', 'r2_oos', 'ct_r2_oos', 'dir_acc', 'n_obs') if k in m}

results = {}

# ── ARM A — baseline full-accuracy L2 (default grid, nfold=4) on the FULL slice ──────
tA = time.time()
pA = build_param_speed(PROBE_H, modality='accurate', nfold=4)       # default depth_coarse_grid (5,8,10)
outA, yhatA = _fit_y(tr_slice['y'], pA, x_tr, tr_slice['date'], x_te)
runA = time.time() - tA
depthA  = int(jl.seval('o -> Int(o.bestparam.depth)')(outA))
ntreesA = int(jl.seval('o -> Int(o.ntrees)')(outA))
capA    = int(jl.seval('p -> Int(p.ntrees)')(pA))                   # CHECK 1: ntrees ceiling for this arm
results['A'] = {'runtime_s': runA, 'oos': _oos(yhatA), 'bestparam_depth': depthA,
                'ntrees': ntreesA, 'ntrees_cap': capA, 'early_stopped': ntreesA < capA}
print(f'[A] {runA:.1f}s  depth={depthA}  ntrees={ntreesA}/{capA} early_stopped={ntreesA < capA}  oos={results["A"]["oos"]}')
assert depthA >= 4, (f'DEPTH GUARD: Arm A selected depth={depthA} (<4) → slice under-exercised edit 3; '
                     f'result is INCONCLUSIVE for depth_coarse_grid. Re-run with a richer slice or pin cv_grid=[3,5,6].')

# ── ARM B — stacked: edit 1 (subsample-CV→fast refit) + edit 2 (nfold=2) + edit 3 (grid=(4,5)) ──
tB = time.time()
# B1: edit-1 CV on a stratified 20% subsample (edit 2 nfold=2, edit 3 grid=(4,5)) → bestparam
tr_sub = stratified_slice(tr_slice, int(round(INNER_CV_SHARE * len(tr_slice))), RNG)
x_trs, _x_te_unused, _ = _prep_fit_matrix(tr_sub, te_real, numeric, cats)
pB_cv = build_param_speed(PROBE_H, modality='accurate', nfold=2, depth_coarse_grid=(4, 5))
outB_cv, _ = _fit_y(tr_sub['y'], pB_cv, x_trs, tr_sub['date'], x_te)
bp_depth  = int(jl.seval('o -> Int(o.bestparam.depth)')(outB_cv))
bp_sparse = float(jl.seval('o -> Float64(o.bestparam.sparsity_penalization)')(outB_cv))
# B2: refit ONE :fast on the FULL slice training rows at bestparam (ntrees RESET → early-stop), grid=(4,5)
pB_fin = build_param_speed(PROBE_H, modality='fast', nfold=1, depth=bp_depth,
                           sparsity_penalization=bp_sparse, depth_coarse_grid=(4, 5))
outB, yhatB = _fit_y(tr_slice['y'], pB_fin, x_tr, tr_slice['date'], x_te)
runB = time.time() - tB
ntreesB = int(jl.seval('o -> Int(o.ntrees)')(outB))
capB    = int(jl.seval('p -> Int(p.ntrees)')(pB_fin))              # CHECK 1: did the :fast refit early-stop on 70k?
results['B'] = {'runtime_s': runB, 'oos': _oos(yhatB),
                'bestparam_depth': bp_depth, 'bestparam_sparsity': bp_sparse,
                'inner_cv_rows': len(tr_sub), 'ntrees': ntreesB,
                'ntrees_cap': capB, 'early_stopped': ntreesB < capB}
print(f'[B] {runB:.1f}s  cv-depth={bp_depth} sparsity={bp_sparse:.3f} (inner={len(tr_sub):,})  '
      f'ntrees={ntreesB}/{capB} early_stopped={ntreesB < capB}  oos={results["B"]["oos"]}')
# CHECK 1 (the edit-1 silent-misbehaviour guard): a :fast refit PINNED at the ntrees cap means it
# did NOT early-stop — the bestparam→:fast round-trip likely failed to re-optimise on the 70k slice.
# Surface it cheaply now (70k) before it matters at 440k. Not fatal; flagged for inspection.
if ntreesB >= capB:
    print(f'  ⚠ CHECK-1 FLAG: Arm B :fast refit ntrees={ntreesB} is PINNED at cap={capB} (no early stop). '
          f'Inspect the bestparam→:fast round-trip before trusting Arm B — ntrees did not re-optimise on the slice.')

# ── 4. Decision numbers + EXTRACT (save so we never re-run this probe) ───────────────
# FRAMING (per GATE-2 note):
#   • OOS-EQUIVALENCE is the VALIDATED OUTPUT — it proves the stack is SAFE.
#   • The runtime ratio is DIRECTIONAL ONLY. This probe's inner CV runs at ~14k rows;
#     production inner CV is ~0.20×440k ≈ 88k (~6× larger) on a SUPER-linear curve, so
#     this ratio UNDERSTATES the production speed win and cannot be scaled up cleanly.
#   • Do NOT use a 70k→440k extrapolation as the production per-fold estimate. Production
#     per-fold time MUST come from the FIRST REAL 440k Hiking fold. The edit-4 decision
#     keys off (OOS-equivalence) + (that first real-fold runtime), NOT this probe.
ratio = runA / runB if runB else float('inf')
oosA, oosB = results['A']['oos'], results['B']['oos']
def _d(k):
    return (oosB.get(k) - oosA.get(k)) if (oosA.get(k) is not None and oosB.get(k) is not None) else None
results['summary'] = {
    'runtime_ratio_A_over_B_DIRECTIONAL_ONLY': ratio,
    'oos_delta_B_minus_A': {k: _d(k) for k in ('r2_oos', 'ct_r2_oos', 'dir_acc', 'mse')},
    'inner_cv_rows_probe': results['B'].get('inner_cv_rows'),
    'inner_cv_rows_production_approx': int(round(0.20 * len(tr_full))),   # ~88k → ~6× this probe
    'slice_rows': len(tr_slice), 'full_fold_rows': len(tr_full),
    'production_perfold_time': 'MEASURE on first real 440k fold — NOT extrapolated here',
    'armB_early_stopped': results['B'].get('early_stopped'),
    'inner_cv_noise_caveat': ('A and B see DIFFERENT inner subsamples by construction '
        '(HTBoost subsamples random-row via the subsampling seed; B also CVs on a separate 20% slice). '
        'A small ΔB-A — ESPECIALLY B appearing BETTER — is CV-noise, not a real effect. '
        'Only a SYSTEMATIC B-worse signal across metrics indicates a problem.'),
    'probe_H': PROBE_H, 'fold': PROBE_FOLD, 'seed': SEED}
print('\n── OOS equivalence (PRIMARY / safety — the validated output) ──')
print(f'   A: {oosA}')
print(f'   B: {oosB}')
print(f'   ΔB-A: {results["summary"]["oos_delta_B_minus_A"]}')
print('   ⓘ CHECK-2 caveat: A and B use different inner subsample randomizations by construction, so a '
      'small ΔB-A (especially B better) is CV-noise; only a systematic B-worse across metrics is a real problem.')
print(f'\n── runtime (DIRECTIONAL ONLY) ──  A/B = {ratio:.1f}×  '
      f'(probe inner CV ~{results["B"].get("inner_cv_rows"):,} vs production ~{int(round(0.20*len(tr_full))):,} '
      f'→ ~6× on a super-linear curve; production per-fold time comes from the first real 440k fold, not this).')
with open('htboost_results_v5_nor/_probe_L2_speedstack_gate2.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print('saved: htboost_results_v5_nor/_probe_L2_speedstack_gate2.json  (local probe artifact — NOT pushed)')
