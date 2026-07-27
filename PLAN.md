# Research Plan — *When Compression Agrees with Biology*
### Unsupervised model selection recovers cell identity, symmetry, and retinotopy in the fly connectome

*Status: locked (approved). Nothing executes until explicit go. Gate A is the first hard stop.*

> **⚠ LIVE CLARIFICATIONS (2026-07-24).** The original plan text below is preserved verbatim; inline `⚠ CLARIFICATION` blocks record where execution diverged from it after contact with the data/tooling. The headline revisions: (1) **canonical graph is >=5, flat, graph-tool 2.98** — the plan's `>=1 / nested / gt-3.0` proved unbuildable (codex pre-floored at 5), unreliable (sub-5 = FlyWire false-positive floor), or crashing; >=1 is downgraded to an optional caveated robustness check (see 3.2, 3.4). (2) MDL is comparable only within a weight family; the lognormal-vs-Gaussian **Jacobian audit is done** and *reverses* the naive verdict (gaussian lower corrected MDL). (3) E2b (held-out prediction) is running and is now the deciding criterion for the title (§6).

---

## 0. Operating principles

**0.1 Checkpoint-driven, never end-to-end.** Each experiment is a separate, gated invocation. After each, we **stop, write to `RESULTS.md`, summarize, and wait for explicit approval**. Hard stops are numbered **Gate A** and **CP-2 … CP-7**; none is crossed without sign-off.

**0.2 No run hangs silently.** Every long fit is instrumented (§4) and time-bounded. Stalls are detected, killed, and reported — never left running.

**0.3 Calendar-boxed.** Per-run wall-clock caps are not enough: **each checkpoint has a maximum number of days**, and the **final week is reserved for analysis and writing**. A checkpoint that overruns its day-budget triggers its predeclared fallback rather than open-ended debugging.

## 1. Thesis & contribution
The statistically simplest description of the connectome — a degree-corrected SBM with a heavy-tailed weight likelihood — is chosen by label-free model selection (description length / held-out predictive log score), and the same model, **fit without labels and subsequently evaluated against** biology, best matches the cell-type taxonomy and exposes two geometric factors: **bilateral symmetry** and **retinotopy**. The contribution is a representation-geometry claim: the inductive bias in the weight likelihood determines the geometry and granularity of the recovered representation.

## 2. Research questions (priority order)
1. **RQ-A** audit + canonical comparison: is the weight likelihood the best *predictive* approximation, and does the likelihood choice change the partition?
2. **RQ-B** selection agreement: do label-free criteria (MDL and held-out predictive log score, on matched support) select the model that best matches biology?
3. **RQ-C** symmetry: does the model recover left/right homolog structure it was never given (out-of-model validation against label circularity)?
4. **RQ-D** granularity: which biological level (superclass/class/subclass/type) do the blocks best represent, and where is recovery limited by sample size vs genuine connectivity similarity?
5. **RQ-E** retinotopy (if time): do visual-type fragments factorize into identity × retinal position, with cross-type transfer?

## 3. Data, preprocessing, methods

**3.1 Data versioning (frozen and recorded).** Exact FlyWire snapshot, annotation release, root-ID reconciliation, inclusion criteria, download checksums, and the explicit treatment of untyped neurons. Report neuron/edge counts.

**3.2 Preprocessing.** Canonical: directed, threshold >=1, no self-loops, nested DC-SBM. Sensitivity (best model + nearest competitor only): >=5, undirected, flat. **Block-count reconciliation is required early** — explain the pilot ~360 vs the FlyWire preprint's larger count (threshold / directedness / nested-vs-flat / node set) before interpreting granularity.

> **⚠ CLARIFICATION (2026-07-24, after data + methods audit — original text kept intentionally). The "canonical" setting above is revised; here is what actually holds and why:**
> - **Threshold >=1 is downgraded from canonical to (at most) an optional, caveated robustness check; >=5 is the working and defensible canonical.** Reasons: (a) the codex `connections_princeton` table is *pre-floored at >=5 synapses* (min summed weight = 5), so a real >=1 graph cannot be built from it — every batch cell labelled `t1` is byte-identical to its `t5` twin. (b) A true >=1 graph exists only in the raw `synapses.ftr` (14.2M pairs vs 2.46M at >=5; 82.7% of pairs are <5), which may be a *different FlyWire snapshot* than codex v783 (pilot >=5 build = 131k nodes vs codex 139k), so blending its edges with codex labels/coordinates risks root-id mismatch — unverified. (c) FlyWire floors at >=5 *deliberately*: sub-5 links are considered unreliable (automated-detection false positives), so >=1 is "more data but noisier," and for held-out **weight prediction** (E2b) scoring a possibly-false edge = scoring noise, contaminating the criterion. Net: >=5 is version-consistent with our labels and the retinotopic (p,q) coordinates and matches the whole-brain literature; treat it as canonical, and if >=1 is run at all it is an explicitly-labelled sensitivity check, never the headline.
> - **Nested was never achieved; flat is canonical (fair-fallback 4.7).** graph-tool 3.0/3.1 segfault / return NaN on our weighted nested fits; we run graph-tool **2.98, flat** (see 3.4 clarification). Per 4.7, every likelihood is therefore compared under flat inference.

**3.3 Labels (withheld from fitting; evaluation-only).** `superclass`, `class`, `subclass`, `primary_type`, `side`, `nt_type`. Framed as "labels withheld during fitting," never "independent ground truth" (FlyWire typing itself used connectivity).

> **⚠ CLARIFICATION (2026-07-25) — version/environment audit for NESTED fits.** Tested whether a
> newer graph-tool fixes the nested-weighted failure. Findings (environment, not a scientific result):
> - **Availability:** conda-forge has graph-tool **3.0, 3.1, 3.5** — 3.2/3.3/3.4 do not exist.
> - **Hard blocker on this server:** 3.1 and 3.5 binaries require **glibc 2.38**; the machine is
>   Ubuntu 22.04 with **glibc 2.35**, so they cannot run (conda installs them anyway — the package
>   omits its glibc floor, so the failure surfaces as an ImportError at runtime). 3.0 and 2.98
>   (glibc <=2.32) are the only versions that run here. No container tooling
>   (docker/podman/apptainer/singularity) is available as a workaround.
> - **3.5 verified on macOS (arm64), SYNTHETIC data only** (800-node planted-block graph with
>   lognormal weights >=5; the real connectome never left the server): **nested works — 16/16
>   nested cells OK, including nested + lognormal + DC**, recovering the planted 8 blocks.
> - **The failure mode inverts between versions:** on 2.98 flat works and nested does not; on 3.5
>   nested works but **flat weighted fits are unusable** (poisson-flat returns NaN; several
>   lognormal/gaussian flat cells return numerically absurd entropies ~-1e176), and flat
>   over-fragments (hundreds of blocks where nested finds the planted 8).
> - **Consequence:** the nested experiments are feasible, but only on graph-tool 3.1+, i.e. a
>   **Ubuntu 24.04+ host**. Until such a machine is available, flat remains canonical under the
>   fair-fallback rule (4.7). Before committing a full nested grid, re-verify at ~20k nodes — the
>   3.0 segfault only appeared at full-brain scale.

**3.4 Inference.** graph-tool **3.0**, frozen exactly (build/commit, thread count, environment recorded). Degree-corrected `BlockState`; weight covariates via `WeightedBlockState`.

> **⚠ CLARIFICATION (2026-07-24).** We use graph-tool **2.98**, not 3.0: 3.0/3.1 segfault on nested weighted fits and return NaN entropy on flat weighted fits for this data; 2.98 gives finite entropy and stable partitions. API is the 2.x form — degree-corrected `gt.BlockState` with `recs=[prop], rec_types=[...]` (not `WeightedBlockState`). Inference is **flat** (see 3.2 clarification).

**3.5 Likelihoods & comparability (precise).**
- Clean primary pair: **lognormal vs Gaussian** (real-normal on log vs raw).
- Clean discrete pair: **Poisson vs geometric**.
- **Cross-family predictive comparison:** common integer support via `P(W=k)=F(k+0.5)-F(k-0.5)`, with **threshold-specific truncation** — condition on `W>=1` for the >=1 graph, `W>=5` for the >=5 graph.
  > **⚠ CLARIFICATION (2026-07-24).** In practice only the **`W>=5`** branch is used: our graph is the >=5 codex graph (see 3.2). `xval.py` truncates at 5 (`logsf(threshold-1)`, fits by truncated MLE). The `W>=1` branch stays defined for the optional >=1 robustness check only.
- **MDL comparability is separate from predictive common support:** definitive for lognormal-vs-Gaussian (after the Jacobian audit) and within Poisson-vs-geometric; **four-way MDL is exploratory** unless a full common coding scheme is built. "Statistically simplest" is narrowed to whatever pair MDL actually establishes.

**3.6 Selection criteria (label-free).** Description length (with Jacobian + the caveats above) and **held-out predictive log score** (§5, E2b). Report predictive advantage in **nats/bits per held-out edge**.

**3.7 Agreement metrics & predefined effect sizes.** V/homogeneity/completeness, AMI, ARI; per-level matching; symmetry enrichment; retinotopy statistics. Predeclared meaningful effects: minimum biological-score improvement **relative to cross-seed variation**; symmetry **enrichment over its matched null**; predictive advantage in nats/edge.

## 4. Engineering & run-monitoring (built around prior stuck-run failures)

**4.1 Heartbeat on the real path.** Every fit writes `<run>.progress.jsonl` every N seconds: `{iter, entropy, n_blocks, delta_entropy, accepted_moves, elapsed_s, seed, phase}`. A `status` command shows all runs: `RUNNING / CONVERGED / TIMED_OUT / FAILED`. **Critically, monitoring is implemented on the exact production fit loop** — the nested fit is built from `mcmc_equilibrate` with callbacks (or a level-by-level agglomerative loop that yields control between levels), **not** the opaque high-level minimizer, so progress is actually observable on the run that hung before.

**4.2 Convergence (multi-signal).** Not entropy alone (a flat trace can be a local optimum): track **entropy delta + accepted-move rate + cross-seed partition stability** (AMI between seeds). Early-stop on the combination.

**4.3 Timeouts & graceful termination.** Per-run wall-clock cap; on breach, send a **graceful termination signal first**, let it checkpoint, then hard-kill. Marked `TIMED_OUT`. **Timed-out partial fits are never aggregated as if converged.**

**4.4 Checkpoint & resume (robust).** Periodic state saves written **atomically (temp file + rename)**, storing **graph checksum, partition/hierarchy, fitted parameters, and RNG state** (not just a pickle). Resume validates the graph checksum before continuing.

**4.5 Multi-seed, fault-isolated, resource-safe.** K seeds as separate processes; a hung seed does not block others; **CPU/RAM oversubscription is prevented** when several multi-core seeds run at once (thread budgeting). Aggregate only converged seeds + report per-seed status and pairwise agreement.

**4.6 Subgraph smoke test before every full-brain fit**, same settings, predeclared subgraph.

**4.7 Nested-SBM fallback (predeclared, fair).** If the nested lognormal cannot converge and stay observable within budget across seeds, **flat DC-SBM becomes canonical — and then every likelihood is compared under flat inference** (fair fallback), with nested reported as exploratory.

## 5. Experiment sequence with checkpoints

**Gate A — environment, minimal core code, compute profiling, and two pass/fail feasibility criteria.**
Gate A has an explicit build step and **two** review checkpoints (code, then results). Sequence:

- **A.0 (env).** Create the frozen `gt` (graph-tool 3.0) env; verify a tiny SBM imports and runs.
- **A.1-code (minimal core modules).** Write only the code the feasibility tests need — `data` (load), `graph` (build gt graph), `monitor` (heartbeat / timeout / atomic-checkpoint), `sbm` (fit loop built on the **real production path**, not the opaque high-level wrapper), `xval` (masking fold). No experiment-specific scripts yet. Minimal and reviewable.
- **-> CODE-REVIEW CHECKPOINT.** Present the core modules for review **before anything runs on the full brain**. These are the highest-risk, load-bearing pieces (the monitoring loop and the leak-free masking), small enough to read line by line. Approve before running A-1/A-2.

Then, on the **exact production path**:

- **A-1 (weight-CV leak-freedom).** Implement and test a miniature end-to-end **weight-masking fold**: hide weights on visible edges and verify held-out weights affect **neither partitions, fitted parameters, prior hyperparameters, transforms/truncation constants, nor preprocessing** (all estimated from training data only). The **partition itself must be inferred with held-out weights genuinely unobserved** — masked *before* fitting — not merely excluded from scoring. Predeclared outcomes:
  1. leak-free weight-CV demonstrated (partition + parameters + hyperparameters all training-only; held-out weights enter solely at final scoring) -> use it;
  2. only leak-free **joint edge prediction** works (masked node-pairs fully unobserved during inference — adjacency and weight both hidden; predefined non-edge sampling + importance weighting) -> use edge prediction as the primary predictive criterion;
  3. neither can be done leak-free -> **abandon weight cross-validation entirely**, rely on the lognormal-vs-Gaussian MDL (after Jacobian audit) + the held-out audit (E1), and scope "statistically simplest" accordingly.
  **Never report a leaked result.**
- **A-2 (nested observability/interruptibility).** Profile one full-brain nested DC-SBM fit and one flat fit for time/memory/convergence, and confirm heartbeats + graceful interruption work **on the nested production path**. Pass/fail: if the nested path is not observable/interruptible within budget, **flat becomes canonical** (fair-fallback rule 4.7).

-> **Gate A RESULTS CHECKPOINT: report profiling + the A-1/A-2 outcomes and the resulting canonical setup. Approve before any experiment.** (Gate A thus has two stops: the code-review checkpoint above, then this results checkpoint.)

**E1 — Likelihood audit (RQ-A), held-out (not in-sample).**
Held-out predictive log scores; randomized-quantile/PIT residuals for discrete models; tail calibration; edge- and block-pair-weighted summaries; minimum sample-size per block-pair. Language: *"best predictive approximation among tested likelihoods,"* not "well-specified" (integer counts cannot literally be generated by a continuous lognormal).
*Prior in-sample evidence (optimistic, to be replaced by held-out): conditional-lognormal KS ~ 0.071, partition-independent.*
-> **CP-2: results + verdict. Kill-criterion: if the held-out audit removes the effect, stop.**

**E2 — Canonical comparison + sensitivity (RQ-A).**
Canonical condition, all likelihoods, K=8-10 seeds; sensitivity variants (>=5 / undirected / flat) ~3 seeds; DC ablation on best + nearest competitor only. Report #blocks, V/homog/compl, AMI/ARI, DL, cross-seed agreement, runtime.
-> **CP-3: comparison table + stability.**

> **⚠ CLARIFICATION (2026-07-24). What was actually run for E2, and how it maps here:** a full factorial (5 likelihoods x {dir,und} x {DC,ndc} x 5 seeds = 200 fits), all on the **>=5, flat** graph — i.e. per the 3.2 clarification this is the *canonical* condition now, and "sensitivity variants (>=5/und/flat)" collapses into it rather than being a separate axis. This over-scopes the plan (full DC ablation across *all* likelihoods, not just best + competitor) and was run before E1/CP-2. CP-3 done and cached (`eval_sweep.py` -> `RESULTS.md`); MDL shown with §3.5 limits and, for lognormal-vs-Gaussian, the completed Jacobian audit (`jacobian_audit.py`). Threshold (t1=t5) is therefore not a real axis in this run.

**E2b — Compression / predictive-selection agreement (RQ-B, load-bearing).**
Does the label-free winner equal the biological winner? Held-out likelihood = **weight prediction** (mask weights, refit within fold) primary; **joint edge prediction** (mask node pairs incl. non-edges, predefined sampling + importance weights, refit) secondary — subject to the Gate A-1 outcome. Folds restricted to canonical + top-two likelihoods, 3 holdouts (millions of test edges -> high precision at few folds). MDL reported with its per-pair comparability limits (§3.5).
-> **CP-4: agreement result + title decision** (§6).

> **⚠ CLARIFICATION (2026-07-24). What is running:** all **4 weight models** (not just top-two) x {dir,und} x {DC,ndc} x {random, stratified folds} x 3 disjoint folds x 3 seeds = **288 cells**, 5% held out, on the **>=5** graph. The A-1 outcome forced the *edge-removed weight-prediction* fold (adjacency+weight both hidden; leak-free), so "mask weights only" is not what runs — but the removed-adjacency confound is common to all models, so relative selection stays fair. **"Joint edge prediction" (the secondary, non-edge variant) is deferred, not done.** Folds are shared across models (paired); per-cell JSONs are cached so restarts resume (`batch_e2b.py`/`worker_e2b.py`, report via `report_e2b.py`). Early partial read: prediction favours **lognormal**, gaussian predicts catastrophically despite winning corrected MDL.

**E3 — Symmetry / cross-hemisphere (RQ-C), fully specified.**
Predeclared: hemisphere assignment per neuron; treatment of **midline neurons and contralateral edges**; **infer left and right hemispheres separately**; alignment **features** (block-level connectivity signatures only — **no homolog/type/side labels; those are evaluation-only**); whether block counts may differ (yes); **matching algorithm** (Hungarian or optimal transport, predeclared); **negative pairs + degree/class-matched permutation null**; **primary symmetry metric** (homolog co-assignment enrichment over null). Cheaper joint-graph version reported only as "orthogonal evidence."
-> **CP-5: symmetry results. Failure language: failed recovery does *not* prove circularity — it means the concern is unresolved or the test underpowered.**

**E4 — Biological hierarchy + fair baselines (RQ-D).**
Match each SBM level to superclass/class/subclass/type; recovery vs type-size with a-priori bins {2,4,8,16,...}; honest framing (small types = weak statistical support under a regularized global model). **Baselines labeled honestly: NTAC at the known #types and Leiden/Louvain tuned to match K are oracle-resolution comparisons, not unsupervised model selection; resolution is never tuned using V-measure.**
-> **CP-6: hierarchy + baselines.**

**E5 — Retinotopy / optic factorization (RQ-E, if time).**
Blocks vs identity x position: conditional MI given type; **categorical spatial assortativity / join-counts** (never Moran's I on numeric block IDs); predict **axial hex coordinates (q,r)** with **hex-grid distance error**; **cross-type transfer** (learn spatial mapping on some visual types, test on held-out types) — required for a genuine factorization claim (within-type prediction alone is only "retinotopic refinement"); within-type permutation null; optional spectral embedding of the visual block-graph as a map; compare across likelihoods.
-> **CP-7: retinotopy results + decision on inclusion.**

## 6. Naming rule
- MDL (label-free) selects the heavy-tail model **and** it best matches biology -> **"When Compression Agrees with Biology."**
- If only the held-out predictive score agrees while cross-family MDL stays incomparable -> **"Unsupervised Predictive Selection Agrees with Biology."**
- "Statistically simplest" is scoped to whichever pair MDL actually establishes.

## 7. Deliverables
Minimal, reviewable repo (`connectome-sbm/`): small modules (`data`, `graph`, `sbm`, `metrics`, `monitor`, `xval`) + one gated script per experiment; frozen `environment.yml`; `PLAN.md`; live `RESULTS.md` appended at each checkpoint; a `status` board. No morphology code.

## 8. Gates & kill-criteria (summary)
- **Gate A-1:** leak-free weight-CV demonstrated (partition + all parameters training-only), else leak-free edge-prediction, else abandon predictive CV. Never a leaked result.
- **Gate A-2:** nested production path observable/interruptible, else flat canonical (fair-fallback: all likelihoods flat).
- **CP-2:** held-out audit removes the effect -> stop.
- **CP-4:** selection disagrees with biology -> retitle/reframe (still legitimate).
- **CP-5:** symmetry fails -> concern unresolved / test underpowered (not proof of circularity).
- **Calendar:** any checkpoint over its day-budget -> predeclared fallback, protect the final writing week.
