"""
E2b worker: run ONE held-out predictive cell and save its score.

Cell = (weight model, direction, degree-correction, fold method, fold index, seed).
Fold = leak-free edge-removed held-out weight prediction (Gate A-1 outcome). Folds are built
from a FIXED seed, so every model is scored on the SAME held-out edges (paired comparison);
only the SBM inference seed varies.

Writes one small JSON per cell -> results are cached; a re-run skips finished cells.
Run as a child of batch_e2b.py, with cwd = src/.
"""
import argparse
import json
import time
import numpy as np
import config
import data
import xval

OUT = config.WORK_DIR / "e2b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model", required=True, choices=["lognormal", "gaussian", "poisson", "geometric"])
    ap.add_argument("--method", required=True, choices=["random", "stratified"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--undirected", action="store_true")
    ap.add_argument("--no-dc", action="store_true")
    a = ap.parse_args()

    directed, deg_corr = not a.undirected, not a.no_dc
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{a.run_name}.json"
    if out_path.exists():                       # cached: never redo finished work
        print(f"[e2b] {a.run_name} already done, skipping"); return

    pre, post, w = data.load_edges(threshold=a.threshold, directed=directed)
    folds = xval.make_folds(w, a.test_frac, a.n_folds, a.method)
    train_mask, test_idx = folds[a.fold]

    t0 = time.time()
    r = xval.run_fold_spec(pre, post, w, directed, a.model, a.threshold,
                           train_mask, test_idx, deg_corr=deg_corr, seed=a.seed)
    r.update(model=a.model, method=a.method, fold=a.fold, seed=a.seed,
             directed=directed, deg_corr=deg_corr, test_frac=a.test_frac,
             threshold=a.threshold, elapsed_s=round(time.time() - t0, 1))
    out_path.write_text(json.dumps(r, indent=2))
    print(json.dumps(r))


if __name__ == "__main__":
    main()
