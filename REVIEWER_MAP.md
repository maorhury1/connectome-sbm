# Repository & filesystem map (for an external reviewer)

You have direct server access. This document says where everything is. Nothing here asks you to
refit anything — every fit is already on disk and every number in `RESULTS.md` traces to a file
or a script listed below.

Companion document: **`PREPAPER.md`** — the claims we make, and which script produced each.

---

## 0. Quick orientation

| you want to | go to |
|---|---|
| the claims, and what backs each one | `PREPAPER.md` |
| the raw numbers behind every table | `RESULTS.md` |
| the pre-registered design and its amendments | `PLAN.md` |
| the code | `src/` (43 scripts, one purpose each — see §2) |
| the fitted SBMs | `/var/tmp/csbm_work/results/` (NOT in git — 1.1 GB) |
| the held-out prediction results | `/home/gamir/maorhury/csbm_g11/e2b/` (282 JSONs, NOT in git) |
| derived analysis outputs (CSV) | `/var/tmp/csbm_work/scratch/` and `/var/tmp/csbm_work/results/` |

**Environment.** Use `/home/gamir/maorhury/miniconda3/envs/env_C/bin/python`. It has graph-tool
2.98. graph-tool 3.x does NOT work on this machine (needs glibc 2.38, system has 2.35) — see
`src/test_gt35.py` for the exact failures that forced the pin. Run scripts from inside `src/`
so `import config, data` resolve.

---

## 1. What is in git and what is not

The repo is deliberately thin: code, documents, and small tables only.

| | location | size | in git |
|---|---|---|---|
| code + docs | this repo | ~700 KB | yes |
| E2b artifacts (STALE — see warning below) | `artifacts/E2b/` | small | yes |
| **fitted flat SBMs** (partition + blockmat + summary, 200 runs) | `/var/tmp/csbm_work/results/` | 1.1 GB | no |
| **fitted nested SBMs** | `/var/tmp/csbm_work/nested_results/` | 1.8 GB | no |
| **held-out (E2b) per-cell results, 282 JSONs** | `/home/gamir/maorhury/csbm_g11/e2b/` | 1.2 MB | no |
| derived CSVs from every analysis | `/var/tmp/csbm_work/scratch/`, `/var/tmp/csbm_work/results/` | 3.6 MB | no |
| downloaded reference data (hex coords, annotations) | `/var/tmp/csbm_work/scratch/` | 159 MB | no |
| raw FlyWire data | `/home/gamir/maorhury/Projects/Research_Project_2026/Data/` | 21 GB | no |

> **WARNING — stale artifact.** `artifacts/E2b/e2b_manifest.json` is dated 2026-07-25 and
> predates four rounds of corrections to the held-out scorer. **It contradicts the current
> results and should be ignored.** The authoritative held-out results are the 282 JSONs in
> `/home/gamir/maorhury/csbm_g11/e2b/`.

---

## 2. Code (`src/`), grouped by role

### Core infrastructure
| file | role |
|---|---|
| `config.py` | all paths and constants in one place. `WORK_DIR` (heavy outputs), `SCRATCH_DIR` (derived intermediates), `DATA_DIR` (raw FlyWire). Overridable by `CSBM_WORK` / `CSBM_SCRATCH`. |
| `data.py` | loads the connectome and the label hierarchy. **Labels are evaluation-only and never enter a fit.** |
| `graph.py` | builds the graph-tool graph, raw-weight and log-weight edge properties |
| `sbm.py` | the fit itself — graph-tool's own minimiser. We do not hand-roll inference. |
| `metrics.py` | partition vs biology scoring (the ONLY place labels are used) |
| `monitor.py` | watchdog: hard wall-clock cap per fit, SIGTERM then SIGKILL |
| `worker.py` / `batch.py` | run one fit / schedule the full factorial |

### Held-out prediction (the load-bearing experiment)
| file | role |
|---|---|
| `xval.py` | **read this one carefully.** Leak-free fold construction and the truncated, rounded predictive score. Six defects were found and fixed here across four review rounds; the docstrings record each. |
| `worker_e2b.py` / `batch_e2b.py` | one held-out cell / the 288-cell grid |
| `report_e2b.py` | aggregation |
| `verify_scorer.py` | regression test: truncated MLE recovers known parameters |

### Model-selection criteria (several superseded — see PREPAPER §5)
| file | status |
|---|---|
| `jacobian_audit.py` | live — makes lognormal-vs-Gaussian MDL comparison legitimate |
| `peixoto_evidence.py` | live — Bayesian evidence, arXiv:1708.01432 Eq. 109 |
| `bic.py` | **superseded** — a hand-built criterion, later shown invalid |
| `integer_weight_models.py`, `bundle_description.py`, `topology_description.py`, `validate_candidate_mdl.py` | CPWDL implementation + its validation gate |

### Analyses behind the claims (all added 2026-08-05)
| file | question |
|---|---|
| `spatial_vs_identity.py` | do blocks group by eye position or cell identity? (`--coords hex` / `--coords centroid`) |
| `t1_mixture_test.py` | is lognormality of node strength a mixture artefact? |
| `t4_closure_test.py` | do Piazza et al.'s closure identities hold within groups? |
| `t5_regime_split.py` | does their law differ between map-organised and identity-organised neurons? |
| `t6_edge_level.py` | their procedure, run on EDGE weights |
| `t7_which_structure.py` | which biological labelling does each weight model recover? |
| `t7b_hemilineage_threshold.py` | is the hemilineage tie a small-class artefact? |
| `t8_symmetry_labelfree.py` | do blocks merge or split the two hemispheres? (uses only `side`) |
| `t9_nested_hierarchy.py` | does SBM hierarchy level track biological level? |
| `t10_block_conditioned.py` | what law holds CONDITIONAL on a block pair? (the mechanism) |
| `t11_conditional_mi.py` | factorization of identity x position, or only refinement? |
| `t12_mirror_symmetry.py` | does the recovered map respect the mirror symmetry of the optic lobes? |
| `t13_recovery_by_size.py` | recovery vs cell-type size |
| `t14_spectral_geometry.py` | block-graph spectral axes (**run but NOT used** — see PREPAPER §6) |

### Probes and dead ends (kept for the record)
`probe_a1.py`, `nested_probe.py`, `smoke_test.py`, `test_gt35.py`, `test298.py`, `rp_test.py`,
`export_edges.py`, `eval_sweep.py`, `nested_sweep.py`, `report_nested.py`

---

## 3. Data

**Raw FlyWire, codex v783** — `/home/gamir/maorhury/Projects/Research_Project_2026/Data/network/codex_2025/`

| file | contents |
|---|---|
| `connections_princeton.csv.gz` | **one row per (pre, post, NEUROPIL)** — see the trap below |
| `classification.csv.gz` | super_class / class / sub_class / side |
| `consolidated_cell_types.csv.gz` | primary_type (~8,600 types) |
| `fafb_v783_princeton_synapse_table.csv.gz` | 2.7 GB, individual synapses with coordinates |

> **TRAP, worth your attention.** `connections_princeton.csv.gz` has one row per
> (pre, post, neuropil). Individual rows go down to `syn_count = 1`, so it *looks* unthresholded.
> It is not: summing over neuropils gives 3,732,460 edges with a **minimum weight of 5**, because
> the codex only reports pairs reaching 5 synapses in total. Treating rows as edges invents
> ~1.6M sub-5 edges that do not exist. We made exactly this error once (it reversed a result);
> `t6_edge_level.py` now asserts against it. If you re-derive anything from this file, sum over
> neuropils first.

**Morphology** — `../tree_morphology/simplyfied_neural_structures.ftr` (42M skeleton nodes;
per-neuron `cable_length`, and x/y/z used for centroids).

**Reference data downloaded from public sources** — `/var/tmp/csbm_work/scratch/`

| file | source | why it is trustworthy |
|---|---|---|
| `column_assignment.csv.gz` | `storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz` (Matsliah et al. 2024) | 45,528 neurons with hex (p,q) + column_id; **100% root-id overlap with our graph**, verified |
| `flywire_ann.tsv` | `flyconnectome/flywire_annotations` Supplemental file 1 | hemilineage (2 schemes), neurotransmitter |
| `conn.parquet`, `centroids.parquet`, `rho.parquet` | derived locally | cached intermediates |

---

## 4. Fitted models

**Flat sweep** — `/var/tmp/csbm_work/results/{model}_{t1|t5}_{dir|und}_{dc|ndc}_s{0..4}/`

5 weight models x 2 thresholds x directed/undirected x degree-corrected/not x 5 seeds = 200 runs.
Each directory holds `partition.npz` (`node_ids`, `blocks`), `blockmat.npz` (block-by-block
`ecount`, `wsum`, `wsq`), `summary.json` (MDL, block counts, runtime).

> **The `t1` runs are byte-identical to their `t5` twins** and carry no independent information —
> a >=1 graph cannot be built from this table (see the trap in §3). `PLAN.md` records this.
> **The canonical setting throughout is `t5_dir_dc`.**

**Nested sweep** — `/var/tmp/csbm_work/nested_results/{model}_dc_dir_s{seed}_*`

> **KNOWN BUG.** The saved `level_*` arrays are **corrupt** — block counts contradict the JSON,
> the levels are not nested, and AMI(level_0, true level 0) ~ 0 in 17 of 18 runs.
> `t9_nested_hierarchy.py` rebuilds every level from the raw `bs_*` tree and verifies nesting.
> **Do not read `level_*` directly.** Coverage is also incomplete: no lognormal, and poisson lost
> 2 of 5 seeds.

**Held-out (E2b)** — `/home/gamir/maorhury/csbm_g11/e2b/` — 282 of 288 cells.
The 6 failures are all `gaussian / undirected / random / fold 2`, every seed, both DC settings,
deterministic across retries. This is discussed in `docs_heldout_prediction.md`; it is a
non-random dropout and it favours gaussian.

---

## 5. Derived outputs

`/var/tmp/csbm_work/scratch/`: `t1_mixture.csv`, `t1_per_group.csv`, `t4_closure.csv`,
`t5_regime.csv`, `t6_edge_ks.csv`, `t7_which_structure.csv`, `t7b_hemilineage.csv`,
`t9_nested_hierarchy_{rows,summary}.csv`, `spatial_vs_identity_raw_hex.csv`

`/var/tmp/csbm_work/results/`: `t11_conditional_mi.csv`, `t13_recovery_by_size*.csv`,
`t12_mirror_symmetry.json`, `t14_spectral.csv`

`/var/tmp/csbm_work/`: `eval_scores.csv`, `eval_stability.csv`, `peixoto_evidence.csv`,
`bic_scores.csv` (the last is from a superseded criterion)

---

## 6. Documents

| file | contents |
|---|---|
| `PREPAPER.md` | **start here** — the story, every claim, and the script behind each |
| `RESULTS.md` | all numbers, chronological, with nulls and caveats |
| `PLAN.md` | the pre-registered design. Contains 6 dated `CLARIFICATION` blocks amending the original where reality intervened; the superseded text is kept deliberately. |
| `docs_heldout_prediction.md` | the held-out experiment written up for external review, with four rounds of corrections folded in |
| `docs_peixoto_evidence.md` | the Bayesian evidence implementation |
| `docs_description_length.md` | **superseded** — kept for the record |
| `FINDINGS_SUMMARY.md` | an earlier plain-language summary; `PREPAPER.md` replaces it |

---

## 7. Things we would most like checked

1. **`src/xval.py`** — the held-out scorer. It carries the headline claim and it is where all six
   known defects were found. Particularly: the truncation normaliser is `1 - F(4.5)` for
   continuous families (integers are rounded *before* the w>=5 threshold applies), and fitting
   maximises the same truncated likelihood that is scored.
2. **Every null.** Almost every measure here is inflated by block count, and the models differ
   ~8x in it (lognormal ~191 blocks, poisson ~1470). Each analysis carries a permutation null;
   the nulls are what make the comparisons legitimate, so they are the right place to attack.
3. **The unit of uncertainty in the held-out test.** Seeds are re-inferences of the same split,
   not independent replicates. There are 6 independent splits, and they share ~90% of their
   training edges, so all spreads are descriptive and no significance is claimed.
4. **Circularity.** FlyWire cell types were themselves partly defined from connectivity. Every
   cell-type comparison inherits this. The label-free results (`t8`, `t11`, `t12`) are the ones
   that do not.
