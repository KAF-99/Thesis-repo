#!/usr/bin/env python
"""Headless REPRO_HANDSHAKE runner for the gradient-boosting notebooks.

Loads a GB notebook, truncates it to the REPRO_HANDSHAKE cell (inclusive) so NO heavy
sweep cells run, executes it with nbclient, and tees the captured FINGERPRINT + PILOT
line to results/<OUT_DIR>/_pilot/handshake_<machine>.log — so a backgrounded or
timed-out run never loses its fingerprint (Part A prints before the Part B fit, so even
an interrupted run keeps the FINGERPRINT block).

Usage:
    python scripts/run_handshake.py notebooks/model_htboost_v5_clean.ipynb
    python scripts/run_handshake.py notebooks/model_htboost_pooled_v5.ipynb

Environment:
    THESIS_DATA_PATH   must point at your Bloomberg data directory.
    NORWAY_LIVE_FETCH  leave UNSET for the offline cache-first build (recommended).
    HANDSHAKE_KERNEL   optional Jupyter kernel name (default: the notebook's kernelspec).
    HANDSHAKE_TIMEOUT  optional per-cell timeout in seconds (default: 3600).
"""
import os
import re
import socket
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def _machine_id() -> str:
    raw = os.environ.get("THESIS_MACHINE_ID") or socket.gethostname()
    return re.sub(r"[^A-Za-z0-9_-]", "-", raw) or "unknown"


def _repo_root(start: Path) -> Path:
    """Walk up until a directory contains src/ (the repo root); fall back to start's parent."""
    p = start
    while p != p.parent:
        if (p / "src").is_dir():
            return p
        p = p.parent
    return start.parent


def _stream_text(cell) -> str:
    parts = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append(o.get("text", ""))
        elif o.get("output_type") == "error":
            parts.append("\n!!! ERROR: %s: %s\n" % (o.get("ename", ""), o.get("evalue", "")))
            parts.append("\n".join(o.get("traceback", [])))
    return "".join(parts)


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/run_handshake.py <path-to-GB-notebook.ipynb>")
    nb_path = Path(sys.argv[1]).resolve()
    if not nb_path.exists():
        sys.exit(f"notebook not found: {nb_path}")

    nb = nbformat.read(nb_path, as_version=4)

    # Truncate to the REPRO_HANDSHAKE cell inclusive — guarantees no sweep cells run.
    cut = next((i for i, c in enumerate(nb.cells)
                if c.cell_type == "code"
                and "REPRO_HANDSHAKE" in c.source and "FINGERPRINT" in c.source), None)
    if cut is None:
        sys.exit(f"no REPRO_HANDSHAKE cell found in {nb_path.name}")
    nb.cells = nb.cells[:cut + 1]

    repo_root = _repo_root(nb_path.parent)

    # Tee destination, derived from the notebook's own OUT_DIR.
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    m = re.search(r"OUT_DIR\s*=\s*['\"]([^'\"]+)['\"]", code)
    out_dir = m.group(1) if m else "results/_handshake"
    log_dir = repo_root / out_dir / "_pilot"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"handshake_{nb_path.stem}_{_machine_id()}.log"

    timeout = int(os.environ.get("HANDSHAKE_TIMEOUT", "3600"))
    kernel = os.environ.get("HANDSHAKE_KERNEL")  # None -> use the notebook's kernelspec
    client_kwargs = {"timeout": timeout}
    if kernel:
        client_kwargs["kernel_name"] = kernel
    client = NotebookClient(nb, **client_kwargs)

    print(f"[run_handshake] {nb_path.name} -> truncated to handshake cell {cut} "
          f"(running {len(nb.cells)} cells); cwd={repo_root}")
    status = "completed"
    try:
        client.execute(cwd=str(repo_root))
    except Exception as e:                       # timeout / kernel error: keep partial output
        status = f"interrupted ({type(e).__name__}: {str(e)[:120]})"
    finally:
        text = _stream_text(nb.cells[-1])
        log_path.write_text(text)
        sys.stdout.write(text)
        if not text.rstrip().endswith(("PILOT OK", ")")) and "PILOT OK" not in text:
            sys.stdout.write("\n[run_handshake] note: no 'PILOT OK' captured — see status below.\n")
        print(f"\n[run_handshake] status: {status}")
        print(f"[run_handshake] fingerprint teed to: {log_path}")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
