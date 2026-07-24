# Artifacts — computed results (organized snapshot)

Backed-up copy of everything under `/var/tmp/csbm_work` (machine-local, unbacked). Heavy
`.npz` live here so a `/var/tmp` wipe costs nothing. All fits are the **>=5-synapse, flat,
graph-tool 2.98, directed/undirected** regime (the working canonical — see `PLAN.md`
clarifications). Snapshot date: 2026-07-24.

## Layout

```
artifacts/
├── sbm_fits/            E2 sweep — 200 SBM fits (5 models x {dir,und} x {dc,ndc} x 5 seeds)
│   └── <model>_t<thr>_<dir|und>_<dc|ndc>_s<seed>/
│         partition.npz    node_ids + block label per neuron   (the partition)
│         blockmat.npz     block x block: ecount, wsum, wsq     (block-pair weight stats)
│         summary.json     config, mdl_entropy, n_blocks, sizes, elapsed
│   (t1 == t5: the codex source is pre-floored at >=5, so the two are byte-identical)
├── E2b/                 held-out predictive selection (RQ-B) — PARTIAL until the run finishes
│   ├── fold_scores/       one JSON per cell: logscore_per_edge (nats/edge), n_blocks, ...
│   └── e2b_manifest.json  scheduler status
├── tables/
│   ├── eval_scores.csv       per-fit biology scores (V/homog/compl/AMI/ARI x 4 levels, MDL)
│   ├── eval_stability.csv    per-config cross-seed AMI
│   └── batch_manifest.json   E2 sweep scheduler status
└── gateA_misc/          old Gate-A graphs / runs / logs (profiling, smoke tests)
```

## Key result tables (human-readable) are in `../RESULTS.md`
CP-3 (biology-agreement sweep, mean±std), the Jacobian audit, and the E2b table are appended
there by `eval_sweep.py`, `jacobian_audit.py`, `report_e2b.py`.

## Reproduce / re-read (all cached, instant)
```
cd ../src
python eval_sweep.py      # CP-3 table from tables/  -> RESULTS.md   (seconds, from cache)
python report_e2b.py      # E2b table from E2b/       -> RESULTS.md   (after run completes)
```

## Notes
- `E2b/` here is a **partial** snapshot (the run was still going); re-sync after it finishes.
- `sbm_fits/` is 1.1 GB of binary `.npz`; it is git-ignored to keep the repo/GitHub pushable —
  the files are safe in this folder regardless. Use git-lfs or a data release if they must go
  to GitHub.
