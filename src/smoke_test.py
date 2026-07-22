"""Smoke tests for the four reviewer fixes (tiny graph / no connectome). Run from src/."""
import subprocess, sys, json, time
import numpy as np
import config, monitor, xval

ENV = config.ENV_PYTHON
OK = True
def check(name, cond):
    global OK; OK = OK and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

print("== #4 scorer: common truncated-discrete support, comparable across families ==")
w_train = np.array([1,1,2,2,3,5,8,13,21,34], float)
for model in ["lognormal","gaussian","poisson","geometric"]:
    p = xval._fit_params(model, w_train, 1)
    vals = [xval._log_trunc_pmf(model, k, p, threshold=1) for k in [1,2,5,20]]
    check(f"{model}: finite log-pmf on integer support", all(np.isfinite(vals)) and all(v<=0 for v in vals))

print("== #1 watchdog: kills a hung process on wall-clock ==")
r = monitor.run_supervised(["sleep","1000"], hard_timeout=3, grace=2)
check("hung 'sleep' killed within budget", r["killed"] and r["elapsed_s"] < 10)

print("== #1/#2/#5 supervised nested fit -> heartbeat + checkpoint + status ==")
run="smoke_nested"
r = monitor.run_supervised(
    [ENV,"worker.py","--run-name",run,"--model","lognormal","--nested","--test",
     "--heartbeat-s","0.5","--checkpoint-s","1","--max-seconds","60"], hard_timeout=120)
d = config.RUNS_DIR/run
check("worker finished under supervision (rc=0)", r["returncode"]==0 and not r["killed"])
check("STATUS == CONVERGED", (d/"STATUS").exists() and (d/"STATUS").read_text().strip()=="CONVERGED")
check("heartbeat progress.jsonl written", (d/"progress.jsonl").exists() and (d/"progress.jsonl").read_text().strip()!="")
check("atomic checkpoint state.pkl written", (d/"state.pkl").exists())
ck = monitor.load_checkpoint(run)
check("checkpoint has model_config + per-level blocks", ck is not None and "model_config" in ck and isinstance(ck["blocks"], list))

print("== #5 resume: reload the checkpoint and continue without error ==")
r2 = subprocess.run([ENV,"worker.py","--run-name",run,"--model","lognormal","--nested","--test","--resume","--max-seconds","30"],
                    capture_output=True, text=True)
check("resume run exits 0", r2.returncode==0)
check("resume printed a fit info line", r2.stdout.strip().endswith("}"))

print("\nRESULT:", "ALL PASS" if OK else "FAILURES ABOVE")
sys.exit(0 if OK else 1)
