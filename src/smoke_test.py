"""Smoke tests for the kept pieces: graph-tool's minimizer finds blocks; watchdog kills a
hang; the CV scorer is finite. Run from src/."""
import subprocess, json, sys
import numpy as np
import config, monitor, xval

ENV = config.ENV_PYTHON
OK = True
def check(name, cond):
    global OK; OK = OK and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

print("== scorer: finite log-pmf across families ==")
w = np.array([5, 6, 7, 10, 15, 22, 30], float)
for m in ["lognormal", "gaussian", "poisson", "geometric"]:
    p = xval._fit_params(m, w, 5)
    vals = [xval._log_trunc_pmf(m, k, p, 5) for k in [5, 7, 15, 30]]
    check(f"{m}", all(np.isfinite(vals)) and all(v <= 0 for v in vals))

print("== graph-tool minimizer finds >1 block on a tiny graph ==")
r = subprocess.run([ENV, "worker.py", "--test", "--model", "lognormal"],
                   capture_output=True, text=True)
info = json.loads(r.stdout.strip().splitlines()[-1]) if (r.returncode == 0 and r.stdout.strip()) else {}
check("minimizer runs and finds >1 block", info.get("n_blocks", 0) > 1)

print("== watchdog kills a hung process ==")
res = monitor.run_supervised(["sleep", "1000"], hard_timeout=3, grace=2)
check("hung 'sleep' killed within budget", res["killed"] and res["elapsed_s"] < 10)

print("\nRESULT:", "ALL PASS" if OK else "FAILURES ABOVE")
sys.exit(0 if OK else 1)
