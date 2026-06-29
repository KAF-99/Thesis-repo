#!/usr/bin/env python3
"""XGBoost benchmark — v5 per-security pipeline, Norwegian swap rates.

Methodological anchor: identical to model_htboost_v5_clean.ipynb in every
respect except the model itself — same target, same features (including PCA
compression of the cross-market block), same walk-forward folds, same one-sided
purge, same evaluation harness (SHARED_COLS / compute_metrics_row).

Two known differences are documented here and in the thesis:

  1. LOSS FUNCTION — XGBoost uses squared error (reg:squarederror). HTBoost
     uses a heavy-tailed Student-t loss (LOSS='t'). This means the comparison
     is NOT a perfect isolation of the smooth-split mechanism: residual loss
     differences may contribute to any performance gap. The thesis states this
     explicitly. A pseudo-Huber loss (reg:pseudohubererror) would reduce outlier
     sensitivity but squared error is the conventional regression baseline.

  2. HYPERPARAMETER TUNING — XGBoost tunes learning_rate / max_depth /
     n_estimators (via early stopping) / subsample / colsample_bytree /
     reg_lambda / min_child_weight within each walk-forward training window
     using a time-ordered inner validation split (last INNER_VAL_FRAC of
     training). HTBoost uses a fixed lambda (pre-committed midpoint of LAM_GRID)
     with modality='accurate' internal block-CV early stopping. Both approaches
     select hyperparameters on training data only — no OOS test window ever
     influences hyperparameter choice.

Usage
-----
    # Smoke test: NOR_10Y × H=21, writes to results/xgb_v5_nor/_smoke/
    python scripts/model_xgboost_v5_clean.py --smoke

    # Single tenor / horizon (full walk-forward folds)
    python scripts/model_xgboost_v5_clean.py --tenor NOR_10Y --horizon 21

    # Full sweep (all tenors × all horizons)
    python scripts/model_xgboost_v5_clean.py
"""

import argparse
import hashlib
import itertools
import json
import os
import pickle
import re
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score

# ── Repo root resolution (same pattern as the HTBoost notebook) ───────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Shared modules — imported unchanged, no modifications to any of these ─────
from src.data.bloomberg import load_data
from src.data.norway import load_norway_raw, print_connectivity_report
from src.features.macro import add_macro_features
from src.features.norway import add_norway_features
from src.features.panel import build_panel
from src.features.pca import fit_transform_xm_pca
from src.evaluation.metrics import (
    compute_metrics_row,
    SHARED_COLS,
    MTC_COLS,
    apply_mtc_family,
    finalize_long_csv,
)
import src.config as config
from src.config import MACHINE_ID

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# ════════════════════════════════════════════════════════════════════════════════
# CONFIG — all run parameters live here; nothing is selected on OOS performance
# ════════════════════════════════════════════════════════════════════════════════

NOTEBOOK    = 'xgboost_v5'         # short provenance label (analogous to 'v5' in HTBoost)
MODEL_KIND  = 'per_security'       # same as HTBoost per-security notebook
IS_POOLED   = False
RUN_TS      = datetime.now(tz=timezone.utc).isoformat()

# Horizon grid — identical to HTBoost
H_GRID       = config.H_GRID          # [1, 5, 21, 63]
H_GRID_LONG  = [126, 252]             # 6m, 12m — gated on data length (same rule as HTBoost)

# Walk-forward folds — identical to HTBoost (from shared config)
WF_FOLDS     = config.WF_FOLDS
FOLD_NAMES   = config.FOLD_NAMES

# Universe thresholds — identical to HTBoost
UNIVERSE_MIN_OBS = 500
MIN_TRAIN_OBS    = config.MIN_TRAIN_OBS   # 252
MIN_TEST_OBS     = config.MIN_TEST_OBS    # 20
TARGET_LAGS      = 5                       # AR lags of chg_1d — identical to HTBoost

# PCA compression of the cross-market xm_* block — identical to HTBoost
XM_PCA_ENABLE = config.XM_PCA_ENABLE
XM_PCA_VAR    = config.XM_PCA_VAR    # 0.95 — explained-variance target
XM_PCA_KMAX   = config.XM_PCA_KMAX   # 12   — hard cap on n_components

# FEAT_SPEC: compressed fingerprint of the feature specification; used in config_hash
FEAT_SPEC = f'pca_var{XM_PCA_VAR}_kmax{XM_PCA_KMAX}_tl{TARGET_LAGS}'

# Output directory — separate from HTBoost; never writes to results/v5_nor/
OUT_DIR    = os.path.join(_ROOT, 'results', 'xgb_v5_nor')
SMOKE_DIR  = os.path.join(OUT_DIR, '_smoke')     # smoke-test writes go here

# Norway feature cache (cache-first, same as production — live=False)
NOR_CACHE  = os.path.join(_ROOT, 'data', 'cache', 'norway_raw_features.csv')

# Smoke test defaults
SMOKE_SEC  = 'NOR_10Y'
SMOKE_H    = 21

# Meta columns that separate targets/identifiers from features in the panel
META_COLS  = {'date', 'security', 'y', 'level'}

# ════════════════════════════════════════════════════════════════════════════════
# XGBoost hyperparameter tuning — KNOWN DIFFERENCE FROM HTBOOST (see docstring)
# All values are pre-committed; no OOS fold ever influences this grid.
# ════════════════════════════════════════════════════════════════════════════════

# NOTE: HTBoost uses Student-t loss (LOSS='t'). XGBoost uses squared error.
# This is a documented, known difference — see module docstring.
OBJECTIVE       = 'reg:squarederror'

N_TREES_MAX     = 500     # maximum boosting rounds (early stopping applied in inner CV)
EARLY_STOP_RND  = 20      # patience: rounds without improvement before stopping
INNER_VAL_FRAC  = 0.20    # time-ordered inner validation: last 20% of training window

# Pre-committed hyperparameter grid (tuned within training window only)
XGB_PARAM_GRID = {
    'learning_rate':    [0.01, 0.05, 0.10],
    'max_depth':        [3, 5],
    'subsample':        [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda':       [1.0, 5.0, 10.0],   # L2 regularization
    'min_child_weight': [1, 5],
}
# Grid size: 3 × 2 × 2 × 2 × 3 × 2 = 144 combinations

# ── File-path helpers (analogous to V5_WF_CSV etc. in the HTBoost notebook) ───

def _wf_csv(h, out_dir):      return os.path.join(out_dir, f'xgb_wf_H{h}__{MACHINE_ID}.csv')
def _pred_csv(h, out_dir):    return os.path.join(out_dir, f'xgb_wf_preds_H{h}__{MACHINE_ID}.csv')
def _imp_csv(h, out_dir):     return os.path.join(out_dir, f'xgb_wf_imps_H{h}__{MACHINE_ID}.csv')
def _imp_pkl(h, out_dir):     return os.path.join(out_dir, f'xgb_wf_imps_H{h}__{MACHINE_ID}.pkl')
def _tuning_csv(h, out_dir):  return os.path.join(out_dir, f'xgb_tuning_log_H{h}__{MACHINE_ID}.csv')


# ════════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ════════════════════════════════════════════════════════════════════════════════

def _config_hash(cfg, extra=''):
    """Stable 12-char MD5 fingerprint of a config dict. Identical logic to the HTBoost notebook."""
    payload = (json.dumps({k: cfg.get(k) for k in sorted(cfg)},
                           sort_keys=True, default=str)
               + '|' + str(extra))
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _score(y, yhat):
    """DirAcc / R² / n_obs helper for diagnostic printing."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 5:
        return None
    return {'dir_acc': float(np.mean(np.sign(y[m]) == np.sign(yhat[m]))),
            'r2':      float(r2_score(y[m], yhat[m])),
            'n_obs':   int(m.sum())}


def _prepare_x_xgb(df):
    """Sanitize column names and replace ±inf with NaN.
    XGBoost handles NaN natively for tree-building (no fillna needed).
    The xmpca_* block is already zero-filled by fit_transform_xm_pca."""
    df = df.copy()
    rmap = {c: re.sub(r'[^a-zA-Z0-9_]', '_', str(c)).strip('_') for c in df.columns}
    df = df.rename(columns=rmap)
    # Deduplicate any name collisions created by sanitization (rare with these features)
    seen, new_cols = {}, []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f'{c}_{seen[c]}')
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df.replace([np.inf, -np.inf], np.nan).astype(np.float64)


def _done_secs(csv_path):
    """Securities whose full fold-batch is already written to csv_path."""
    if not os.path.exists(csv_path):
        return set()
    try:
        return set(pd.read_csv(csv_path, usecols=['security'])['security'].astype(str))
    except Exception as e:
        print(f'  [resume] could not read {csv_path} ({repr(e)[:50]}) — treating as empty')
        return set()


def _append_csv(df, path):
    """Append DataFrame rows to a CSV, writing header on first write. Flush + fsync."""
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as fh:
        df.to_csv(fh, header=write_header, index=False)
        fh.flush()
        os.fsync(fh.fileno())


# ════════════════════════════════════════════════════════════════════════════════
# Data loading — macro + Norway features, identical to the HTBoost notebook
# ════════════════════════════════════════════════════════════════════════════════

def load_and_augment_data():
    """Load Bloomberg data and augment with macro + Norway features.

    Replicates the data-preparation steps of the HTBoost notebook in order:
      1. load_data()                     (Bloomberg CSVs)
      2. load_norway_raw()               (cache-first; same cache used by the notebook)
      3. join Norway raw series to df    (nb_* / ecb_dfr / rb_polrate daily columns)
      4. add_norway_features()           (nor_* derived columns)
      5. add_macro_features()            (VIX/MOVE/equity/commodity/credit/IBOR)
    All five steps are byte-identical to what the notebook does.
    """
    print('Loading Bloomberg data ...')
    df_raw = load_data()
    print(f'  df_raw shape: {df_raw.shape}   '
          f'({df_raw.index.min().date()} → {df_raw.index.max().date()})')

    # Norway data (cache-first; live=False preserves reproducibility across machines)
    print('Loading Norway data (cache-first) ...')
    start_str = df_raw.index.min().strftime('%Y-%m-%d')
    end_str   = df_raw.index.max().strftime('%Y-%m-%d')
    nor_raw, nor_report = load_norway_raw(start_str, end_str, NOR_CACHE, live=False)
    print_connectivity_report(nor_report)

    if not nor_raw.empty:
        # Join the raw Norway series (nb_*, ecb_dfr, rb_polrate, ssb_*) to df_raw
        df_raw = df_raw.join(nor_raw, how='left')
        # Derive nor_* feature columns from the joined raw series
        df_raw = add_norway_features(df_raw, nor_raw)
        print(f'  Norway nor_* columns added: '
              f'{sum(1 for c in df_raw.columns if c.startswith("nor_"))}')
    else:
        print('  [WARN] Norway data unavailable — nor_* features will be NaN '
              '(XGBoost handles NaN natively; nor_* columns will have no predictive content)')

    # Macro / global features (VIX, MOVE, equity, commodity, credit, IBOR)
    print('Adding macro features ...')
    df_raw = add_macro_features(df_raw)
    print(f'  df_raw shape after augmentation: {df_raw.shape}')
    return df_raw


# ════════════════════════════════════════════════════════════════════════════════
# XGBoost tuning and fitting
# ════════════════════════════════════════════════════════════════════════════════

def _tune_and_fit_xgb(y_tr, x_tr_df, y_te, x_te_df, H, seed=config.JULIA_SEED):
    """Time-ordered inner-CV grid search + refit on full training window.

    Inner validation: the last INNER_VAL_FRAC of the training rows (time-ordered).
    Grid search over XGB_PARAM_GRID with early stopping on the inner validation set.
    Selected hyperparameters are never exposed to the test window.

    Returns
    -------
    yhat_tr : ndarray       in-sample predictions on full training window
    yhat_te : ndarray       OOS predictions on test window
    best_params : dict      selected hyperparameters
    best_n_est : int        selected n_estimators (from early stopping)
    tuning_rows : list[dict]  per-combination tuning log (for appendix)
    imp_dict : dict         {feature_name: gain_importance} — analogous to HTBrelevance
    """
    n = len(y_tr)
    n_inner_val = max(MIN_TEST_OBS, int(n * INNER_VAL_FRAC))
    n_inner_tr  = n - n_inner_val

    # One-sided inner purge: drop the last H-1 rows of the inner-training block so
    # that none of its labels overlap the inner-validation window. This mirrors the
    # outer walk-forward purge (purge_ts = test_start − BDay(H−1)) and is required
    # because y_{t,h} = r_{t+h} − r_t uses a future rate: the last H−1 inner-train
    # labels reach into the inner-validation period, leaking future information into
    # the early-stopping decision that selects best_n_est.
    n_purge     = max(0, H - 1)
    n_it_purged = n_inner_tr - n_purge

    if n_it_purged >= MIN_TRAIN_OBS:
        # Normal path: purged inner-train block is large enough for early stopping.
        X_it = x_tr_df.iloc[:n_it_purged]
        y_it = y_tr[:n_it_purged]
        use_early_stopping = True
    else:
        # Fallback: purge would shrink inner-train below MIN_TRAIN_OBS (can happen for
        # very small folds at long horizons). Skip early stopping and fit the full
        # training window with N_TREES_MAX rounds; inner-val split is still used for
        # val_mse logging but does not drive n_est selection.
        X_it = x_tr_df.iloc[:n_inner_tr]
        y_it = y_tr[:n_inner_tr]
        use_early_stopping = False

    X_iv = x_tr_df.iloc[n_inner_tr:]
    y_iv = y_tr[n_inner_tr:]

    keys   = list(XGB_PARAM_GRID.keys())
    values = list(XGB_PARAM_GRID.values())

    best_mse    = np.inf
    best_params = None
    best_n_est  = N_TREES_MAX
    tuning_rows = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        if use_early_stopping:
            mdl = xgb.XGBRegressor(
                n_estimators=N_TREES_MAX,
                objective=OBJECTIVE,
                early_stopping_rounds=EARLY_STOP_RND,
                random_state=seed,
                n_jobs=1,
                verbosity=0,
                **params,
            )
            mdl.fit(X_it, y_it,
                    eval_set=[(X_iv, y_iv)],
                    verbose=False)
            # best_iteration is 0-indexed; add 1 for n_estimators of the final model
            n_est = int(getattr(mdl, 'best_iteration', N_TREES_MAX - 1)) + 1
        else:
            mdl = xgb.XGBRegressor(
                n_estimators=N_TREES_MAX,
                objective=OBJECTIVE,
                random_state=seed,
                n_jobs=1,
                verbosity=0,
                **params,
            )
            mdl.fit(X_it, y_it)
            n_est = N_TREES_MAX

        yhat_iv = mdl.predict(X_iv)
        mse_iv  = float(np.mean((y_iv - yhat_iv) ** 2))

        tuning_rows.append({
            **params,
            'val_mse':          mse_iv,
            'best_n_estimators': n_est,
            'selected':          False,
        })

        if mse_iv < best_mse:
            best_mse    = mse_iv
            best_params = params.copy()
            best_n_est  = n_est

    # Flag the selected row in the tuning log
    for row in tuning_rows:
        if all(row[k] == best_params[k] for k in keys):
            row['selected'] = True
            break

    # Refit on the full training window with the selected hyperparameters
    mdl_final = xgb.XGBRegressor(
        n_estimators=best_n_est,
        objective=OBJECTIVE,
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        **best_params,
    )
    mdl_final.fit(x_tr_df, y_tr)

    yhat_tr = mdl_final.predict(x_tr_df)
    yhat_te = mdl_final.predict(x_te_df)

    # Feature importance (gain — analogous to HTBrelevance; fills 0 for unused features)
    imp_dict = {}
    try:
        raw_imp = mdl_final.get_booster().get_score(importance_type='gain')
        feat_names = list(x_tr_df.columns)
        for fn in feat_names:
            imp_dict[fn] = float(raw_imp.get(fn, 0.0))
    except Exception:
        pass

    return yhat_tr, yhat_te, best_params, best_n_est, tuning_rows, imp_dict


# ════════════════════════════════════════════════════════════════════════════════
# Walk-forward runner — identical logic to run_security_v5 except model
# ════════════════════════════════════════════════════════════════════════════════

def _run_security_xgb(sec, df_raw, H, seed=config.JULIA_SEED, verbose=False):
    """Expanding-window walk-forward for one security at horizon H.

    Replicates the fold logic of run_security_v5 exactly:
      - Same WF_FOLDS from src.config
      - Same one-sided purge: purge_ts = test_start − BDay(H−1)
      - Same PCA compression via fit_transform_xm_pca (fit on TRAIN only)
      - Same compute_metrics_row / SHARED_COLS evaluation

    Returns (rows, imp_records, tuning_rows_all, preds_list).
    """
    if sec not in df_raw.columns:
        return [], [], [], []
    panel = build_panel(df_raw, [sec], H)
    if len(panel) == 0:
        return [], [], [], []

    fc = [c for c in panel.columns if c not in META_COLS]
    country, tenor = sec.rsplit('_', 1)

    rows, imp_records, tuning_all, preds_list = [], [], [], []

    for fold_name, test_start, test_end, regime in WF_FOLDS:
        ts_ts = pd.Timestamp(test_start)
        te_ts = pd.Timestamp(test_end)
        # One-sided purge: identical to HTBoost (OVERLAP = H-1)
        purge_ts = ts_ts - pd.tseries.offsets.BDay(H - 1)

        tr = panel[panel['date'] < purge_ts].copy()
        te = panel[(panel['date'] >= ts_ts) & (panel['date'] <= te_ts)].copy()

        if len(tr) < MIN_TRAIN_OBS or len(te) < MIN_TEST_OBS:
            continue

        # PCA compression — fit on TRAIN only, applied identically to train + test
        # Identical to HTBoost: standardise on training moments, zero-fill NaN, fit PCA
        x_tr, x_te, pca_info = fit_transform_xm_pca(tr[fc], te[fc])
        feat_count = x_tr.shape[1]

        # Prepare for XGBoost (sanitize column names; XGBoost handles NaN natively)
        x_tr_xgb = _prepare_x_xgb(x_tr)
        x_te_xgb = _prepare_x_xgb(x_te)

        y_tr_arr = tr['y'].to_numpy(float)
        y_te_arr = te['y'].to_numpy(float)

        # Tune hyperparameters within training window, then refit on full training
        yhat_tr, yhat_te, best_params, best_n_est, fold_tuning, imp_dict = \
            _tune_and_fit_xgb(y_tr_arr, x_tr_xgb, y_te_arr, x_te_xgb, H, seed=seed)

        # Annotate tuning rows with fold metadata for the appendix log
        for trow in fold_tuning:
            trow.update({'security': sec, 'horizon': H,
                         'fold': fold_name, 'regime': regime})
        tuning_all.extend(fold_tuning)

        # Config hash captures the selected hyperparameters + feature spec
        cfg_for_hash = {
            **best_params,
            'n_estimators': best_n_est,
            'objective':     OBJECTIVE,
            'inner_val_frac': INNER_VAL_FRAC,
        }
        meta = dict(
            notebook=NOTEBOOK, run_ts=RUN_TS,
            model_kind=MODEL_KIND, is_pooled=IS_POOLED,
            validation_scheme='walk_forward', target_kind='level_change',
            security=sec, country=country, tenor=tenor,
            fold=fold_name, regime=regime,
            config_hash=_config_hash(cfg_for_hash, extra=FEAT_SPEC),
            feature_count=feat_count,
        )

        row_tr = compute_metrics_row(y_tr_arr, yhat_tr, H, {**meta, 'sample': 'train'})
        row_te = compute_metrics_row(y_te_arr, yhat_te, H, {**meta, 'sample': 'oos'})

        # Extra diagnostic columns (analogous to pca_k / xm_pca_evr / htb_ntrees / htb_depth)
        for row in (row_tr, row_te):
            row['pca_k']         = pca_info['k']
            row['xm_pca_evr']    = pca_info['evr_cum_k']
            row['xgb_n_trees']   = best_n_est
            row['xgb_max_depth'] = best_params.get('max_depth', -1)
            row['xgb_lr']        = best_params.get('learning_rate', -1)

        rows.extend([row_tr, row_te])

        # Per-obs predictions (same schema as v5_wf_preds_H*.csv)
        pred_df = pd.DataFrame({
            'date':     te['date'].values,
            'security': sec,
            'tenor':    tenor,
            'horizon':  int(H),
            'regime':   regime,
            'scheme':   'walk_forward',
            'y_true':   y_te_arr,
            'y_pred':   np.asarray(yhat_te, float),
        })
        preds_list.append(pred_df)

        # Feature importance record (long format: one row per feature)
        if imp_dict:
            imp_records.append({
                'security': sec,
                'horizon':  H,
                'fold':     fold_name,
                'regime':   regime,
                'imp':      imp_dict,
            })

        if verbose:
            s = _score(y_te_arr, yhat_te)
            if s:
                print(f'    {fold_name:12s}  train={len(tr):4d}  OOS={len(te):4d}  '
                      f'DirAcc={s["dir_acc"]:.3f}  R²={s["r2"]:+.4f}  '
                      f'n_trees={best_n_est}  lr={best_params.get("learning_rate")}  '
                      f'depth={best_params.get("max_depth")}')

    return rows, imp_records, tuning_all, preds_list


# ════════════════════════════════════════════════════════════════════════════════
# Horizon support gate — identical logic to _horizon_supported in the HTBoost notebook
# ════════════════════════════════════════════════════════════════════════════════

def _horizon_supported(df_raw, h):
    """Return True if at least one WF fold has ≥ MIN_TRAIN_OBS / MIN_TEST_OBS for SMOKE_SEC."""
    panel = build_panel(df_raw, [SMOKE_SEC], h)
    if len(panel) == 0:
        return False
    for _, ts, te, _ in WF_FOLDS:
        ts_ts = pd.Timestamp(ts)
        te_ts = pd.Timestamp(te)
        purge = ts_ts - pd.tseries.offsets.BDay(h - 1)
        ntr = (panel['date'] < purge).sum()
        nte = ((panel['date'] >= ts_ts) & (panel['date'] <= te_ts)).sum()
        if ntr >= MIN_TRAIN_OBS and nte >= MIN_TEST_OBS:
            return True
    return False


# ════════════════════════════════════════════════════════════════════════════════
# Per-horizon sweep with checkpointing
# ════════════════════════════════════════════════════════════════════════════════

def _sweep_horizon(H_run, tenors, df_raw, out_dir, verbose=False):
    """Walk-forward sweep for one horizon across all tenors, with resume support.

    Mirrors the per-horizon checkpoint structure of the HTBoost notebook:
    each completed (security, horizon) pair is flushed to disk immediately
    so the run is resumable if interrupted.
    """
    wf_csv_p   = _wf_csv(H_run, out_dir)
    pred_csv_p = _pred_csv(H_run, out_dir)
    imp_csv_p  = _imp_csv(H_run, out_dir)
    imp_pkl_p  = _imp_pkl(H_run, out_dir)
    tun_csv_p  = _tuning_csv(H_run, out_dir)

    done      = _done_secs(wf_csv_p)
    all_imps  = []
    if os.path.exists(imp_pkl_p):
        try:
            with open(imp_pkl_p, 'rb') as f:
                all_imps = pickle.load(f)
        except Exception:
            pass
    if done:
        print(f'  [H={H_run}] resume: {len(done)} security(ies) already on disk — skipping them')

    failed = []
    for sec in tenors:
        if sec in done:
            continue
        t0 = time.time()
        try:
            rows, imp_recs, tuning_rows, preds = _run_security_xgb(
                sec, df_raw, H_run, verbose=verbose)
        except Exception as e:
            print(f'  [H={H_run}] {sec}: FAILED  {repr(e)[:80]}')
            failed.append((H_run, sec))
            continue

        # Durably append metrics rows
        if rows:
            _append_csv(pd.DataFrame(rows), wf_csv_p)

        # Durably append predictions
        if preds:
            _append_csv(pd.concat(preds, ignore_index=True), pred_csv_p)

        # Durably append tuning log
        if tuning_rows:
            _append_csv(pd.DataFrame(tuning_rows), tun_csv_p)

        # Update importance pickle + CSV (long format: feature / relevance)
        if imp_recs:
            all_imps.extend(imp_recs)
            with open(imp_pkl_p, 'wb') as f:
                pickle.dump(all_imps, f)
                f.flush()
                os.fsync(f.fileno())
            imp_long = pd.DataFrame([
                {'security': r['security'], 'horizon': r['horizon'],
                 'fold': r['fold'], 'feature': feat, 'relevance': val}
                for r in all_imps
                for feat, val in (r['imp'] or {}).items()
            ])
            imp_long.to_csv(imp_csv_p, index=False)

        oos_rows = [r for r in rows if r.get('sample') == 'oos']
        da_str   = ', '.join(f'{r["dir_acc"]:.3f}' for r in oos_rows)
        print(f'  [H={H_run}] {sec:<10s}  folds={len(oos_rows)}  '
              f'OOS_DA=[{da_str}]  ({time.time()-t0:.1f}s)')

    if failed:
        print(f'  [H={H_run}] {len(failed)} failure(s): {failed}')

    return failed


# ════════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='XGBoost benchmark — v5 per-security walk-forward, Norwegian swap rates')
    parser.add_argument('--smoke',   action='store_true',
                        help=f'Smoke test: {SMOKE_SEC} × H={SMOKE_H} only '
                             f'(writes to results/xgb_v5_nor/_smoke/)')
    parser.add_argument('--tenor',   type=str,  default=None,
                        help='Restrict sweep to a single tenor, e.g. NOR_10Y')
    parser.add_argument('--horizon', type=int,  default=None,
                        help='Restrict sweep to a single horizon, e.g. 21')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-fold diagnostics')
    args = parser.parse_args()

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = SMOKE_DIR if args.smoke else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    print(f'Output directory: {out_dir}')

    # ── Load and augment data ─────────────────────────────────────────────────
    df_raw = load_and_augment_data()

    # ── Universe (same rule as HTBoost: NOR swaps with ≥ UNIVERSE_MIN_OBS obs) ─
    import re as _re
    _SWAP_PAT = _re.compile(r'^[A-Z]+_\d+[WMY]$')
    swap_cols = sorted(c for c in df_raw.columns if _SWAP_PAT.match(c))
    obs       = df_raw[swap_cols].notna().sum()
    universe  = sorted(obs[obs >= UNIVERSE_MIN_OBS].index.tolist())
    universe  = [s for s in universe if s.rsplit('_', 1)[0] == 'NOR']
    print(f'\nNorwegian swap universe ({len(universe)} securities): {universe}')

    assert SMOKE_SEC in universe, f'{SMOKE_SEC} not in NOR universe — check data'

    # ── Smoke test mode ───────────────────────────────────────────────────────
    if args.smoke:
        print(f'\n[SMOKE TEST] {SMOKE_SEC} × H={SMOKE_H}')
        print('─' * 60)
        t0 = time.time()
        rows, imp_recs, tuning_rows, preds = _run_security_xgb(
            SMOKE_SEC, df_raw, SMOKE_H, verbose=True)
        elapsed = time.time() - t0

        if not rows:
            print('[SMOKE] No rows produced — check data and folds.')
            return

        # Save smoke outputs
        df_rows = pd.DataFrame(rows)
        df_rows.to_csv(_wf_csv(SMOKE_H, out_dir), index=False)
        if preds:
            pd.concat(preds, ignore_index=True).to_csv(_pred_csv(SMOKE_H, out_dir), index=False)
        if tuning_rows:
            pd.DataFrame(tuning_rows).to_csv(_tuning_csv(SMOKE_H, out_dir), index=False)
        if imp_recs:
            imp_long = pd.DataFrame([
                {'security': r['security'], 'horizon': r['horizon'],
                 'fold': r['fold'], 'feature': feat, 'relevance': val}
                for r in imp_recs
                for feat, val in (r['imp'] or {}).items()
            ])
            imp_long.to_csv(_imp_csv(SMOKE_H, out_dir), index=False)

        # Print smoke summary
        oos = df_rows[df_rows['sample'] == 'oos']
        tr_rows = df_rows[df_rows['sample'] == 'train']
        print(f'\n{"─"*60}')
        print(f'[SMOKE RESULT]  {SMOKE_SEC}  H={SMOKE_H}  elapsed={elapsed:.1f}s')
        print(f'  Folds:    {len(oos)} OOS folds  ({len(tr_rows)} train rows)')
        print(f'  Columns:  {list(df_rows.columns)[:8]} ...')
        print(f'\n  OOS summary:')
        for _, r in oos.iterrows():
            print(f'    {r["fold"]:12s}  n={r["n_obs"]:4.0f}  '
                  f'DirAcc={r["dir_acc"]:.3f}  CT-R²={r["ct_r2_oos"]:+.4f}  '
                  f'n_trees={r["xgb_n_trees"]:.0f}  depth={r["xgb_max_depth"]:.0f}  '
                  f'lr={r["xgb_lr"]:.3f}')

        # Compare column set to HTBoost output for format validation.
        # Try MACHINE_ID first (same machine produced both runs), then glob for any
        # machine's file so the check works regardless of which machine ran HTBoost.
        import glob as _glob
        _v5_nor = os.path.join(_ROOT, 'results', 'v5_nor')
        htb_ref = os.path.join(_v5_nor, f'v5_wf_H{SMOKE_H}__{MACHINE_ID}.csv')
        if not os.path.exists(htb_ref):
            _candidates = sorted(_glob.glob(
                os.path.join(_v5_nor, f'v5_wf_H{SMOKE_H}__*.csv')))
            htb_ref = _candidates[0] if _candidates else None
        if htb_ref and os.path.exists(htb_ref):
            htb_cols = set(pd.read_csv(htb_ref, nrows=0).columns)
            xgb_cols = set(df_rows.columns)
            shared   = htb_cols & xgb_cols
            htb_only = htb_cols - xgb_cols
            xgb_only = xgb_cols - htb_cols
            print(f'\n  Format comparison vs HTBoost (H={SMOKE_H}):')
            print(f'    Reference file  : {os.path.basename(htb_ref)}')
            print(f'    Shared columns  : {len(shared)} (SHARED_COLS + pca_k + xm_pca_evr)')
            print(f'    HTBoost-only    : {sorted(htb_only)}  (htb_ntrees/htb_depth expected)')
            print(f'    XGBoost-only    : {sorted(xgb_only)}  (xgb_n_trees/xgb_max_depth/xgb_lr)')
        else:
            print(f'  [format check skipped — no HTBoost v5_wf_H{SMOKE_H}__*.csv found '
                  f'in {_v5_nor}]')

        print(f'\nSmoke outputs written to: {out_dir}')
        return

    # ── Full sweep (or single tenor / horizon) ────────────────────────────────

    # Resolve tenor list
    if args.tenor:
        assert args.tenor in universe, f'{args.tenor} not in NOR universe'
        sweep_tenors = [args.tenor]
    else:
        sweep_tenors = list(universe)

    # Resolve horizon list (gate long horizons on data length, same as HTBoost)
    if args.horizon:
        horizons = [args.horizon]
    else:
        horizons = list(H_GRID)
        for h in H_GRID_LONG:
            if _horizon_supported(df_raw, h):
                horizons.append(h)
                print(f'  long horizon H={h}: data supports it → included')
            else:
                print(f'  long horizon H={h}: insufficient data per fold → SKIPPED')
        horizons = sorted(set(horizons))

    print(f'\nSweep: {len(sweep_tenors)} tenor(s) × {len(horizons)} horizon(s)')
    print(f'  Tenors  : {sweep_tenors}')
    print(f'  Horizons: {horizons}')
    print(f'  Grid    : {sum(len(v) for v in XGB_PARAM_GRID.values())} values across '
          f'{len(XGB_PARAM_GRID)} params = '
          f'{len(list(itertools.product(*XGB_PARAM_GRID.values())))} combinations/fold')
    print()

    t0_sweep = time.time()
    all_failed = []
    for H_run in horizons:
        print(f'H={H_run} {"─"*50}')
        failed = _sweep_horizon(H_run, sweep_tenors, df_raw, out_dir,
                                verbose=args.verbose)
        all_failed.extend(failed)

    # ── Assemble final metrics CSV + apply MTC ─────────────────────────────────
    wf_parts = [pd.read_csv(_wf_csv(h, out_dir))
                for h in horizons if os.path.exists(_wf_csv(h, out_dir))]
    if wf_parts:
        df_all = finalize_long_csv(pd.concat(wf_parts, ignore_index=True))

        # Walk-forward MTC family (same family definition as HTBoost)
        N_wf, b_wf, h_wf = apply_mtc_family(
            df_all, ['walk_forward'],
            'walk_forward:{horizon×tenor×regime}',
            model_kind=MODEL_KIND)

        # Ensure SHARED_COLS order matches HTBoost for seamless merging
        final_csv = os.path.join(out_dir, f'xgb_metrics_long__{MACHINE_ID}.csv')
        df_all.to_csv(final_csv, index=False)

        elapsed = (time.time() - t0_sweep) / 60
        print(f'\nSweep complete in {elapsed:.1f} min')
        print(f'  rows={len(df_all)}  cols={df_all.shape[1]}')
        print(f'  Walk-forward MTC: N={N_wf}  Bonferroni={b_wf}  BH-FDR={h_wf}')
        print(f'  Saved: {final_csv}')

    if all_failed:
        print(f'\n{len(all_failed)} failure(s): {all_failed}')


if __name__ == '__main__':
    main()
