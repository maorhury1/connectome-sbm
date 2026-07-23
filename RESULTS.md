# RESULTS (live)

Appended at each checkpoint. See PLAN.md for the gates.

## Gate A — environment, core code, feasibility

### A.0 environment
- graph-tool **3.0** (commit 1a3c1a29), Python 3.14.6, **80 OpenMP threads**, on local disk `/var/tmp/gt3`
  (shared home volume is full; see README). numpy 2.5.1, scipy 1.18.0, pandas 3.0.3,
  scikit-learn 1.9.0, pyarrow 25.0.0.

### A.1 code + reviewer P1 fixes (approved: #1,#2,#4,#5)
- Core modules committed; monitored/interruptible `multilevel_mcmc_sweep` loop.
- #1 external subprocess watchdog (SIGTERM->SIGKILL) — kills a hung sweep the in-loop timeout can't.
- #2 robust nested block extraction (gt3 `get_bs()` mixes PropertyArray / VertexPropertyMap).
- #4 common truncated-discrete integer support in the scorer — nats/edge comparable across families.
- #5 resumable checkpoints (partition-continue, not bit-identical; gt3 exposes no RNG get/set).
- All four verified on a tiny graph (`smoke_test.py`, all pass).

### A-1 feasibility probe (`probe_a1.py`) — decides the CV fold (#3)
On a weight-defined 2-block graph (adjacency carries no signal; only weights do):
| test | result |
|---|---|
| weighted SBM recovers weight-defined groups (sanity) | ARI 1.00 |
| leave held-out edges in, corrupt their weights -> partition | **ARI 0.34 (leaks)** |
| remove held-out edges, corrupt their weights -> partition | ARI 1.00 (leak-free) |
| `LatentMaskBlockState` | models missing *adjacency*, not covariates — N/A |

**Verdict:** graph-tool 3.0 has **no leak-free per-edge weight mask** — held-out weights leak
into the partition whenever weights matter. Fold = **edge-removed held-out weight prediction**
(leak-free; the removed-adjacency confound is common to all weight models, so relative
weight-model selection in E2b stays fair). This is what `xval.py` implements.

### Scorer correctness (two further issues raised on xval.py; both fixed)
- Truncation-inconsistent fitting: params now fit by **truncated MLE** (consistent with the
  truncated scoring). Verified recovery on synthetic truncated data (e.g. lognormal mu 2.01 vs
  true 2.0 where the naive fit gives biased 2.22); geometric closed-form, Poisson 1-D,
  lognormal/Gaussian 2-D Nelder-Mead with moment init + fallback.
- Unstable tail probabilities: discretized pmf now computed on the stable CDF/survival side via
  a log-difference; deep-tail example goes from the artificial -691 floor to the true -108.
- `verify_scorer.py` (all pass) + `smoke_test.py` (all pass).

### Status
All 5 reviewer P1s resolved (4 fixed + #3 decided by A-1); scorer truncation + tail-stability
also fixed. Remaining Gate A item: **A-2**
(profile one full-brain nested fit for time/memory + confirm observability/interruptibility on
the real path) — the first connectome run. Not yet run.


## CP-3 — Overnight sweep evaluation (aggregated, mean±std over 5 seeds)

*Scope: >=5-synapse, FLAT fits (the plan's **sensitivity** regime, not canonical >=1/nested). MDL comparable only within a weight family. stability = mean pairwise AMI between seeds (30k-node subsample). Ranked by V(primary_type).*

| model | dir | dc | blocks | V:super_class | V:class | V:sub_class | V:primary_type | homog | compl | AMI | stability | MDL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lognormal | dir | dc | 191±33 | 0.355±0.004 | 0.400±0.005 | 0.714±0.010 | 0.773±0.003 | 0.666±0.006 | 0.920±0.007 | 0.729±0.003 | 0.847 | 2.53e+07 |
| poisson | dir | ndc | 1599±135 | 0.259±0.003 | 0.295±0.004 | 0.572±0.007 | 0.751±0.003 | 0.756±0.008 | 0.746±0.007 | 0.659±0.005 | 0.712 | 3.78e+07 |
| lognormal | und | dc | 220±23 | 0.335±0.002 | 0.382±0.003 | 0.661±0.015 | 0.731±0.010 | 0.639±0.010 | 0.855±0.012 | 0.677±0.011 | 0.849 | 2.18e+07 |
| gaussian | dir | ndc | 646±76 | 0.296±0.003 | 0.322±0.005 | 0.586±0.006 | 0.727±0.004 | 0.704±0.010 | 0.752±0.007 | 0.645±0.005 | 0.796 | 3.38e+07 |
| poisson | dir | dc | 1470±232 | 0.246±0.002 | 0.285±0.004 | 0.529±0.007 | 0.718±0.006 | 0.730±0.013 | 0.707±0.007 | 0.611±0.007 | 0.679 | 3.76e+07 |
| poisson | und | ndc | 2326±327 | 0.234±0.006 | 0.268±0.006 | 0.502±0.013 | 0.704±0.007 | 0.732±0.010 | 0.679±0.013 | 0.578±0.015 | 0.619 | 3.46e+07 |
| exponential | dir | ndc | 694±55 | 0.283±0.003 | 0.306±0.004 | 0.524±0.008 | 0.680±0.007 | 0.666±0.009 | 0.695±0.006 | 0.579±0.008 | 0.764 | 3.41e+07 |
| geometric | dir | ndc | 682±25 | 0.285±0.002 | 0.308±0.002 | 0.521±0.005 | 0.678±0.005 | 0.663±0.006 | 0.694±0.004 | 0.578±0.006 | 0.766 | 3.43e+07 |
| gaussian | und | ndc | 835±37 | 0.275±0.001 | 0.301±0.002 | 0.509±0.003 | 0.665±0.002 | 0.661±0.003 | 0.669±0.003 | 0.552±0.003 | 0.739 | 3.02e+07 |
| poisson | und | dc | 2368±260 | 0.217±0.004 | 0.253±0.005 | 0.445±0.007 | 0.663±0.007 | 0.700±0.012 | 0.631±0.007 | 0.510±0.009 | 0.580 | 3.46e+07 |
| gaussian | dir | dc | 561±88 | 0.281±0.004 | 0.306±0.006 | 0.517±0.006 | 0.661±0.005 | 0.640±0.011 | 0.683±0.007 | 0.559±0.006 | 0.754 | 3.34e+07 |
| gaussian | und | dc | 727±43 | 0.264±0.001 | 0.289±0.002 | 0.438±0.003 | 0.600±0.004 | 0.595±0.005 | 0.605±0.004 | 0.468±0.005 | 0.721 | 2.98e+07 |
| geometric | und | ndc | 788±78 | 0.260±0.002 | 0.282±0.003 | 0.416±0.002 | 0.584±0.006 | 0.581±0.010 | 0.587±0.003 | 0.444±0.004 | 0.721 | 3.02e+07 |
| exponential | und | ndc | 774±141 | 0.260±0.003 | 0.282±0.004 | 0.416±0.003 | 0.583±0.010 | 0.579±0.018 | 0.587±0.002 | 0.445±0.005 | 0.723 | 3.01e+07 |
| exponential | dir | dc | 578±55 | 0.253±0.001 | 0.270±0.002 | 0.408±0.005 | 0.569±0.007 | 0.557±0.010 | 0.582±0.004 | 0.437±0.006 | 0.754 | 3.34e+07 |
| geometric | dir | dc | 563±26 | 0.252±0.001 | 0.269±0.001 | 0.408±0.001 | 0.569±0.002 | 0.555±0.004 | 0.583±0.002 | 0.437±0.002 | 0.759 | 3.35e+07 |
| lognormal | und | ndc | 11547±6277 | 0.127±0.115 | 0.151±0.122 | 0.313±0.221 | 0.567±0.127 | 0.644±0.040 | 0.522±0.205 | 0.188±0.312 | 0.014 | 2.99e+07 |
| exponential | und | dc | 689±76 | 0.240±0.002 | 0.260±0.002 | 0.354±0.005 | 0.529±0.007 | 0.526±0.011 | 0.532±0.003 | 0.374±0.004 | 0.748 | 2.94e+07 |
| geometric | und | dc | 686±58 | 0.239±0.002 | 0.258±0.002 | 0.349±0.003 | 0.525±0.004 | 0.523±0.007 | 0.528±0.001 | 0.369±0.001 | 0.750 | 2.95e+07 |
| lognormal | dir | ndc | 12157±971 | 0.072±0.003 | 0.096±0.002 | 0.204±0.005 | 0.498±0.006 | 0.603±0.011 | 0.424±0.004 | 0.054±0.001 | 0.004 | 3.65e+07 |
