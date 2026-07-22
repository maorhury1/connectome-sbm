"""
Full-factorial batch scheduler.

Grid: 5 models x 2 thresholds x 2 directions x 2 (DC / non-DC) x 5 seeds = 200 fits.
Each fit runs as a supervised child process (killed if it hangs on a wall-clock cap). If a
fit is killed or crashes, it is retried with a fresh seed, up to MAX_RETRIES times. All raw
outputs are saved per run by worker.py; this scheduler writes a live manifest of statuses.

Concurrency: N_CONCURRENT fits at once, each pinned to 1 core (OMP_NUM_THREADS=1), leaving
cores free. Run from src/:  python batch.py   (add --smoke for a fast dry run).
"""
import argparse
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import config
import monitor

os.environ["OMP_NUM_THREADS"] = "1"          # 1 core per fit -> clean core accounting

MODELS = ["lognormal", "poisson", "gaussian", "geometric", "exponential"]
THRESHOLDS = [1, 5]
DIRECTIONS = [True, False]                    # directed, undirected
DC = [True, False]
SEEDS = [0, 1, 2, 3, 4]

N_CONCURRENT = 60                             # of ~80 cores; leaves headroom (cores + RAM)
HARD_TIMEOUT = 12 * 3600                      # kill a fit after 12h (real fits ~5-8h)
MAX_RETRIES = 2                               # extra fresh seeds if a fit is killed/crashes

RESULTS = config.WORK_DIR / "results"
MANIFEST = config.WORK_DIR / "batch_manifest.json"


def run_cell(model, threshold, directed, dc, seed, smoke=False):
    dtag = "dir" if directed else "und"
    ctag = "dc" if dc else "ndc"
    base = f"{model}_t{threshold}_{dtag}_{ctag}_s{seed}"
    common = [config.ENV_PYTHON, "worker.py", "--model", model, "--threshold", str(threshold)]
    if not directed:
        common.append("--undirected")
    if not dc:
        common.append("--no-dc")
    if smoke:
        common.append("--test")
    attempts = []
    for k in range(MAX_RETRIES + 1):
        s = seed if k == 0 else seed + 1000 * k          # fresh seed on retry
        run_name = base if k == 0 else f"{base}_retry{k}"
        argv = common + ["--seed", str(s), "--run-name", run_name]
        to = 120 if smoke else HARD_TIMEOUT
        res = monitor.run_supervised(argv, hard_timeout=to, grace=60)
        ok = (res["returncode"] == 0) and (RESULTS / run_name / "summary.json").exists()
        attempts.append(dict(seed=s, run_name=run_name, ok=ok, **res))
        if ok:
            return dict(cell=base, status="OK", used_seed=s, run_name=run_name, attempts=attempts)
    return dict(cell=base, status="FAILED", attempts=attempts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny --test fits to validate orchestration")
    a = ap.parse_args()
    grid = (list(itertools.product(MODELS[:2], THRESHOLDS[:1], [True], [True], SEEDS[:2]))
            if a.smoke else
            list(itertools.product(MODELS, THRESHOLDS, DIRECTIONS, DC, SEEDS)))
    n_workers = 4 if a.smoke else N_CONCURRENT
    print(f"[batch] {len(grid)} cells, {n_workers} concurrent, "
          f"cap {120 if a.smoke else HARD_TIMEOUT}s, smoke={a.smoke}", flush=True)

    t0 = time.time()
    results = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_cell, *c, a.smoke): c for c in grid}
        done = 0
        for fut in as_completed(futs):
            r = fut.result(); results[r["cell"]] = r; done += 1
            MANIFEST.write_text(json.dumps(
                {"done": done, "total": len(grid), "elapsed_s": round(time.time() - t0, 1),
                 "results": results}, indent=2))
            print(f"[batch] {done}/{len(grid)}  {r['cell']} -> {r['status']}", flush=True)
    n_ok = sum(1 for r in results.values() if r["status"] == "OK")
    print(f"[batch] DONE {n_ok}/{len(grid)} ok in {round((time.time()-t0)/3600,2)}h", flush=True)


if __name__ == "__main__":
    main()
