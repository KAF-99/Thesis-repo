"""Opt-in auto-push of machine-stamped result CSVs (Prompt 7).

`push_results(paths, message)` is a COMPLETE no-op unless ``config.AUTO_PUSH`` is True
(`THESIS_AUTO_PUSH=1`). When enabled, it stages ONLY the given results/ CSV/MANIFEST paths,
commits, `pull --rebase`, and pushes — wrapped so that ANY git failure prints a warning and
returns without raising. A failed push therefore never crashes the sweep: the run keeps
going and the next unit (or a manual `git pull --rebase && git push`) recovers it.

Safety: it refuses to stage anything that is not under ``results/`` and a ``.csv`` or
``MANIFEST.md``, so it can never stage licensed data, ``.pkl`` importance files, or
``_pilot/`` scratch. It stages explicitly by path — never ``git add -A``.

Because every result file is machine-stamped (``…__{MACHINE_ID}.csv``), two machines never
edit the same file, so the ``pull --rebase`` before push is conflict-free on ``results/``.
"""
import os
import subprocess

from src import config


def _repo_root():
    # this module lives at <root>/src/utils/gitpush.py
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _is_allowed(path, root):
    """True iff `path` is under <root>/results/ AND is a .csv or MANIFEST.md."""
    ap = os.path.abspath(path)
    results_root = os.path.join(root, "results") + os.sep
    if not ap.startswith(results_root):
        return False
    base = os.path.basename(ap)
    return base.endswith(".csv") or base == "MANIFEST.md"


def _run(root, *args):
    """Run a git command, capturing output. Returns the CompletedProcess (returncode kept)."""
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)


def push_results(paths, message):
    """Stage the given results/ CSV(s), commit, pull --rebase, and push.

    No-op unless ``config.AUTO_PUSH``. Never raises — on any git failure it warns and
    returns, leaving the run going.
    """
    if not config.AUTO_PUSH:
        return

    root = _repo_root()
    paths = [paths] if isinstance(paths, (str, os.PathLike)) else list(paths)

    # ── Safety gate: only results/*.csv (or MANIFEST.md). Refuse everything else. ──────
    allowed, rejected = [], []
    for p in paths:
        (allowed if _is_allowed(p, root) else rejected).append(str(p))
    if rejected:
        print(f"[auto-push] REFUSED non-results/non-csv path(s): {rejected} "
              f"— nothing staged, no push.")
        return
    allowed = [p for p in allowed if os.path.exists(p)]   # stage only what exists
    if not allowed:
        print("[auto-push] no existing results CSV among the given paths — skipping.")
        return

    try:
        # Stage explicitly by path (never `git add -A`).
        r = _run(root, "add", "--", *allowed)
        if r.returncode != 0:
            print(f"[auto-push] git add failed: {r.stderr.strip()[:160]} — continuing run.")
            return
        # Nothing actually staged (file unchanged) → nothing to commit, quietly return.
        if _run(root, "diff", "--cached", "--quiet").returncode == 0:
            print(f"[auto-push] {allowed} already up to date — nothing to commit.")
            return

        r = _run(root, "commit", "-m", message)
        if r.returncode != 0:
            print(f"[auto-push] git commit failed: {r.stderr.strip()[:160]} — continuing run.")
            return

        # Machine-stamped files ⇒ conflict-free on results/; rebase keeps history linear.
        r = _run(root, "pull", "--rebase")
        if r.returncode != 0:
            _run(root, "rebase", "--abort")   # leave a clean tree; local commit is kept
            print(f"[auto-push] pull --rebase failed: {r.stderr.strip()[:160]} — aborted "
                  f"rebase, local commit kept. Recover with `git pull --rebase && git push`.")
            return

        r = _run(root, "push")
        if r.returncode != 0:
            print(f"[auto-push] git push failed: {r.stderr.strip()[:160]} — local commit kept; "
                  f"the next unit or a manual `git push` will send it.")
            return

        print(f"[auto-push] pushed {len(allowed)} result file(s): {message}")
    except Exception as e:                       # defensive: never crash the run
        print(f"[auto-push] unexpected error: {repr(e)[:160]} — continuing run.")
        return
