"""
E2b scheduler — held-out predictive selection (RQ-B, the load-bearing experiment).

Grid: 4 weight models x {directed, undirected} x {DC, non-DC} x 2 fold methods x 3 folds
      x 3 seeds = 288 cells, at 5% held-out edges.

Every cell writes its own small JSON (worker_e2b.py), so finished cells are SKIPPED on
re-run -- interrupting and restarting is free. Per-cell timeout + retry with a fresh seed.

Run from src/:   python batch_e2b.py            (add --smoke for a tiny dry run)
Then report with: python report_e2b.py
"""
import argparse
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import monitor

os.environ["OMP_NUM_THREADS"] = "1"

MODELS = ["lognormal", "gaussian", "poisson", "geometric"]
DIRECTIONS = [True, False]          # directed, undirected
DC = [True, False]
METHODS = ["random", "stratified"]
FOLDS = [0, 1, 2]
SEEDS = [0, 1, 2]
TEST_FRAC = 0.05

N_CONCURRENT = 60
HARD_TIMEOUT = 12 * 3600
MAX_RETRIES = 2

OUT = config.WORK_DIR / "e2b"
MANIFEST = config.WORK_DIR / "e2b_manifest.json"


def cell_name(model, directed, dc, method, fold, seed):
    return (f"{model}_{'dir' if directed else 'und'}_{'dc' if dc else 'ndc'}"
            f"_{method}_f{fold}_s{seed}")


def run_cell(model, directed, dc, method, fold, seed, smoke=False):
    name = cell_name(model, directed, dc, method, fold, seed)
    if (OUT / f"{name}.json").exists():
        return dict(cell=name, status="CACHED")
    base = [config.ENV_PYTHON, "worker_e2b.py", "--model", model, "--method", method,
            "--fold", str(fold), "--test-frac", str(TEST_FRAC)]
    if not directed:
        base.append("--undirected")
    if not dc:
        base.append("--no-dc")
    attempts = []
    for k in range(MAX_RETRIES + 1):
        s = seed if k == 0 else seed + 1000 * k
        run_name = name if k == 0 else f"{name}_retry{k}"
        argv = base + ["--seed", str(s), "--run-name", run_name]
        res = monitor.run_supervised(argv, hard_timeout=120 if smoke else HARD_TIMEOUT, grace=60)
        ok = res["returncode"] == 0 and (OUT / f"{run_name}.json").exists()
        attempts.append(dict(seed=s, ok=ok, **res))
        if ok:
            return dict(cell=name, status="OK", used_seed=s, attempts=attempts)
    return dict(cell=name, status="FAILED", attempts=attempts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    grid = list(itertools.product(MODELS, DIRECTIONS, DC, METHODS, FOLDS, SEEDS))
    if a.smoke:
        grid = list(itertools.product(MODELS[:2], [True], [True], METHODS[:1], [0], [0]))
    OUT.mkdir(parents=True, exist_ok=True)
    done_already = sum(1 for c in grid if (OUT / f"{cell_name(*c)}.json").exists())
    print(f"[e2b] {len(grid)} cells ({done_already} already cached), "
          f"{N_CONCURRENT} concurrent, {TEST_FRAC:.0%} held out", flush=True)

    t0, results = time.time(), {}
    with ThreadPoolExecutor(max_workers=4 if a.smoke else N_CONCURRENT) as ex:
        futs = {ex.submit(run_cell, *c, a.smoke): c for c in grid}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result(); results[r["cell"]] = r
            MANIFEST.write_text(json.dumps(
                {"done": i, "total": len(grid), "elapsed_s": round(time.time() - t0, 1),
                 "results": results}, indent=2))
            print(f"[e2b] {i}/{len(grid)}  {r['cell']} -> {r['status']}  "
                  f"({(time.time()-t0)/60:.0f}m)", flush=True)
    n_ok = sum(1 for r in results.values() if r["status"] in ("OK", "CACHED"))
    print(f"[e2b] DONE {n_ok}/{len(grid)} in {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
