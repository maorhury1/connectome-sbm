# connectome-sbm

Clean, minimal reimplementation for the study *"When Compression Agrees with Biology"*
(see [PLAN.md](PLAN.md) for the full, approved research plan and checkpoint gates).

Scope: degree-corrected SBM inference on the FlyWire connectome with different edge-weight
likelihoods, plus label-free model selection and evaluation. No morphology.

## Environment
graph-tool **3.0** (parallel inference) lives on the machine's **local disk** because the
shared home volume is chronically full:

```
conda create -p /var/tmp/gt3 -c conda-forge "graph-tool>=3" pandas scikit-learn pyarrow tqdm -y
```

Run everything with `/var/tmp/gt3/bin/python`. Exact versions are in `env/`.
Note: this env is local to the machine and may not survive a reset; rebuild with the line above.

## Layout
```
PLAN.md            approved plan (gates CP-1..CP-7)
RESULTS.md         live results, appended at each checkpoint
src/
  config.py        paths + constants (heavy outputs -> /var/tmp local disk)
  data.py          load FlyWire edges + label hierarchy (labels are evaluation-only)
  graph.py         build graph-tool graph; raw-weight + log-weight edge props; checksum
  monitor.py       heartbeat / wall-clock timeout / atomic checkpoint / status board
  sbm.py           DC-SBM fit as a MONITORED, interruptible multilevel-MCMC loop
  xval.py          leak-free held-out predictive scoring (Gate A-1)
env/               frozen environment record
```

## Monitoring (anti-stuck)
Fits are driven one sweep at a time via `multilevel_mcmc_sweep` (parallel), so every run is
observable and interruptible. Watch progress:

```
tail -f /var/tmp/csbm_work/runs/<run_name>/progress.jsonl
/var/tmp/gt3/bin/python src/monitor.py          # status board across all runs
```

Each run has a hard wall-clock cap; on breach it checkpoints atomically and is marked
`TIMED_OUT` (never treated as converged).

## Status
Gate A (env + this core code) — at the **code-review checkpoint**. A-1/A-2 feasibility tests
have **not** been run yet; nothing runs on the full brain until the code is approved.
