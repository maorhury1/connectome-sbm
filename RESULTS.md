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

| model | dir | dc | blocks | V:super_class | V:class | V:sub_class | V:primary_type | homog | compl | AMI | stability | MDL(M nats) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lognormal | dir | dc | 191±33 | 0.355±0.004 | 0.400±0.005 | 0.714±0.010 | 0.773±0.003 | 0.666±0.006 | 0.920±0.007 | 0.729±0.003 | 0.847 | 25.25±0.08 |
| poisson | dir | ndc | 1599±135 | 0.259±0.003 | 0.295±0.004 | 0.572±0.007 | 0.751±0.003 | 0.756±0.008 | 0.746±0.007 | 0.659±0.005 | 0.712 | 37.84±0.09 |
| lognormal | und | dc | 220±23 | 0.335±0.002 | 0.382±0.003 | 0.661±0.015 | 0.731±0.010 | 0.639±0.010 | 0.855±0.012 | 0.677±0.011 | 0.849 | 21.81±0.17 |
| gaussian | dir | ndc | 646±76 | 0.296±0.003 | 0.322±0.005 | 0.586±0.006 | 0.727±0.004 | 0.704±0.010 | 0.752±0.007 | 0.645±0.005 | 0.796 | 33.77±0.09 |
| poisson | dir | dc | 1470±232 | 0.246±0.002 | 0.285±0.004 | 0.529±0.007 | 0.718±0.006 | 0.730±0.013 | 0.707±0.007 | 0.611±0.007 | 0.679 | 37.63±0.12 |
| poisson | und | ndc | 2326±327 | 0.234±0.006 | 0.268±0.006 | 0.502±0.013 | 0.704±0.007 | 0.732±0.010 | 0.679±0.013 | 0.578±0.015 | 0.619 | 34.56±0.20 |
| exponential | dir | ndc | 694±55 | 0.283±0.003 | 0.306±0.004 | 0.524±0.008 | 0.680±0.007 | 0.666±0.009 | 0.695±0.006 | 0.579±0.008 | 0.764 | 34.13±0.04 |
| geometric | dir | ndc | 682±25 | 0.285±0.002 | 0.308±0.002 | 0.521±0.005 | 0.678±0.005 | 0.663±0.006 | 0.694±0.004 | 0.578±0.006 | 0.766 | 34.29±0.06 |
| gaussian | und | ndc | 835±37 | 0.275±0.001 | 0.301±0.002 | 0.509±0.003 | 0.665±0.002 | 0.661±0.003 | 0.669±0.003 | 0.552±0.003 | 0.739 | 30.17±0.04 |
| poisson | und | dc | 2368±260 | 0.217±0.004 | 0.253±0.005 | 0.445±0.007 | 0.663±0.007 | 0.700±0.012 | 0.631±0.007 | 0.510±0.009 | 0.580 | 34.63±0.15 |
| gaussian | dir | dc | 561±88 | 0.281±0.004 | 0.306±0.006 | 0.517±0.006 | 0.661±0.005 | 0.640±0.011 | 0.683±0.007 | 0.559±0.006 | 0.754 | 33.40±0.05 |
| gaussian | und | dc | 727±43 | 0.264±0.001 | 0.289±0.002 | 0.438±0.003 | 0.600±0.004 | 0.595±0.005 | 0.605±0.004 | 0.468±0.005 | 0.721 | 29.81±0.03 |
| geometric | und | ndc | 788±78 | 0.260±0.002 | 0.282±0.003 | 0.416±0.002 | 0.584±0.006 | 0.581±0.010 | 0.587±0.003 | 0.444±0.004 | 0.721 | 30.20±0.05 |
| exponential | und | ndc | 774±141 | 0.260±0.003 | 0.282±0.004 | 0.416±0.003 | 0.583±0.010 | 0.579±0.018 | 0.587±0.002 | 0.445±0.005 | 0.723 | 30.08±0.04 |
| exponential | dir | dc | 578±55 | 0.253±0.001 | 0.270±0.002 | 0.408±0.005 | 0.569±0.007 | 0.557±0.010 | 0.582±0.004 | 0.437±0.006 | 0.754 | 33.36±0.03 |
| geometric | dir | dc | 563±26 | 0.252±0.001 | 0.269±0.001 | 0.408±0.001 | 0.569±0.002 | 0.555±0.004 | 0.583±0.002 | 0.437±0.002 | 0.759 | 33.51±0.02 |
| lognormal | und | ndc | 11547±6277 | 0.127±0.115 | 0.151±0.122 | 0.313±0.221 | 0.567±0.127 | 0.644±0.040 | 0.522±0.205 | 0.188±0.312 | 0.014 | 29.90±4.42 |
| exponential | und | dc | 689±76 | 0.240±0.002 | 0.260±0.002 | 0.354±0.005 | 0.529±0.007 | 0.526±0.011 | 0.532±0.003 | 0.374±0.004 | 0.748 | 29.41±0.02 |
| geometric | und | dc | 686±58 | 0.239±0.002 | 0.258±0.002 | 0.349±0.003 | 0.525±0.004 | 0.523±0.007 | 0.528±0.001 | 0.369±0.001 | 0.750 | 29.55±0.01 |
| lognormal | dir | ndc | 12157±971 | 0.072±0.003 | 0.096±0.002 | 0.204±0.005 | 0.498±0.006 | 0.603±0.011 | 0.424±0.004 | 0.054±0.001 | 0.004 | 36.54±0.02 |


## Jacobian audit — lognormal vs Gaussian MDL

*Correction: `DL_on_w = DL_on_log_w + sum_e log(w_e)`. Audit = the corrected gap must be invariant to rescaling the weights.*

- Audit spread across scales x1..x100: **150.358 nats** -> **PASS**


| model | dir | dc | MDL raw (M) | **MDL corrected (M)** | V type |
|---|---|---|---|---|---|
| gaussian | dir | dc | 33.40 | **33.40±0.05** | 0.661 |
| gaussian | dir | ndc | 33.77 | **33.77±0.09** | 0.727 |
| gaussian | und | dc | 29.81 | **29.81±0.03** | 0.600 |
| gaussian | und | ndc | 30.17 | **30.17±0.04** | 0.665 |
| lognormal | dir | dc | 25.25 | **33.81±0.08** | 0.773 |
| lognormal | dir | ndc | 36.54 | **45.10±0.02** | 0.498 |
| lognormal | und | dc | 21.81 | **29.83±0.17** | 0.731 |
| lognormal | und | ndc | 29.90 | **37.92±4.42** | 0.567 |

- **dir/dc:** MDL picks **gaussian** (33.81 vs 33.40 M); biology (V) picks **lognormal** -> **DISAGREE**

- **dir/ndc:** MDL picks **gaussian** (45.10 vs 33.77 M); biology (V) picks **gaussian** -> **AGREE**

- **und/dc:** MDL picks **gaussian** (29.83 vs 29.81 M); biology (V) picks **lognormal** -> **DISAGREE**

- **und/ndc:** MDL picks **gaussian** (37.92 vs 30.17 M); biology (V) picks **gaussian** -> **AGREE**


## CP-4 — E2b held-out predictive selection

*Leak-free edge-removed weight prediction, 5% held out, 3 disjoint folds x 3 seeds, folds shared across models (paired). nats/edge: HIGHER = better prediction. V(type) from CP-3.*

| model | dir | dc | fold method | nats/edge | sd | cells | blocks | V type |
|---|---|---|---|---|---|---|---|---|
| geometric | dir | ndc | stratified | **-2.9188** | 0.0042 | 9 | 692 | 0.678 |
| geometric | dir | ndc | random | **-2.9191** | 0.0012 | 9 | 686 | 0.678 |
| geometric | dir | dc | random | **-3.0128** | 0.0063 | 9 | 537 | 0.569 |
| geometric | dir | dc | stratified | **-3.0133** | 0.0055 | 9 | 529 | 0.569 |
| lognormal | dir | ndc | stratified | **-3.0713** | 0.1794 | 9 | 9938 | 0.498 |
| geometric | und | ndc | stratified | **-3.0920** | 0.0034 | 9 | 821 | 0.584 |
| geometric | und | ndc | random | **-3.0986** | 0.0040 | 9 | 733 | 0.584 |
| lognormal | dir | dc | random | **-3.1276** | 0.4708 | 9 | 169 | 0.773 |
| lognormal | dir | ndc | random | **-3.1400** | 0.3034 | 9 | 11274 | 0.498 |
| geometric | und | dc | stratified | **-3.1475** | 0.0016 | 9 | 674 | 0.525 |
| geometric | und | dc | random | **-3.1534** | 0.0064 | 9 | 619 | 0.525 |
| lognormal | und | ndc | random | **-3.1548** | 0.0023 | 9 | 14470 | 0.567 |
| lognormal | und | dc | random | **-3.2494** | 0.3498 | 9 | 215 | 0.731 |
| lognormal | und | ndc | stratified | **-3.6171** | 1.3961 | 9 | 11868 | 0.567 |
| lognormal | dir | dc | stratified | **-3.6803** | 1.0018 | 9 | 193 | 0.773 |
| poisson | dir | ndc | random | **-5.8166** | 0.0312 | 9 | 1741 | 0.751 |
| poisson | dir | ndc | stratified | **-5.9163** | 0.0573 | 9 | 1657 | 0.751 |
| poisson | dir | dc | random | **-5.9536** | 0.0373 | 9 | 1612 | 0.718 |
| poisson | dir | dc | stratified | **-6.0315** | 0.0530 | 9 | 1546 | 0.718 |
| lognormal | und | dc | stratified | **-6.3701** | 6.3882 | 9 | 209 | 0.731 |
| poisson | und | ndc | stratified | **-6.9480** | 0.0834 | 9 | 2178 | 0.704 |
| poisson | und | ndc | random | **-7.0017** | 0.0472 | 9 | 2269 | 0.704 |
| poisson | und | dc | random | **-7.3058** | 0.0943 | 9 | 2180 | 0.663 |
| poisson | und | dc | stratified | **-7.3331** | 0.0980 | 9 | 2383 | 0.663 |
| gaussian | und | dc | random | **-105.2344** | 162.1757 | 9 | 691 | 0.600 |
| gaussian | dir | ndc | random | **-123.0690** | 65.5535 | 9 | 654 | 0.727 |
| gaussian | dir | dc | random | **-216.0759** | 282.8237 | 9 | 592 | 0.661 |
| gaussian | dir | ndc | stratified | **-356.0840** | 792.8550 | 9 | 674 | 0.727 |
| gaussian | und | ndc | random | **-391.2103** | 395.9513 | 9 | 763 | 0.665 |
| gaussian | dir | dc | stratified | **-721.8517** | 1911.1501 | 9 | 619 | 0.661 |
| gaussian | und | ndc | stratified | **-966.3700** | 1535.4571 | 9 | 772 | 0.665 |
| gaussian | und | dc | stratified | **-2736.3313** | 7763.1871 | 9 | 685 | 0.600 |

- **dir/dc/random:** prediction picks **geometric**, biology picks **lognormal** -> **DISAGREE**

- **dir/dc/stratified:** prediction picks **geometric**, biology picks **lognormal** -> **DISAGREE**

- **dir/ndc/random:** prediction picks **geometric**, biology picks **poisson** -> **DISAGREE**

- **dir/ndc/stratified:** prediction picks **geometric**, biology picks **poisson** -> **DISAGREE**

- **und/dc/random:** prediction picks **geometric**, biology picks **lognormal** -> **DISAGREE**

- **und/dc/stratified:** prediction picks **geometric**, biology picks **lognormal** -> **DISAGREE**

- **und/ndc/random:** prediction picks **geometric**, biology picks **poisson** -> **DISAGREE**

- **und/ndc/stratified:** prediction picks **geometric**, biology picks **poisson** -> **DISAGREE**
