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


## Discussion — held-out selection vs biology, and reconciliation with the lognormal architecture (2026-07-25)

**Validity of the held-out test (E2b).** Methodologically sound and genuinely label-free
(leak-free edge-removed weight prediction; Gate A-1). Its result is a real finding, not to be
explained away: across all 8 settings the best predictor of held-out edge weights is
**geometric** (-2.92 to -3.15), narrowly beating lognormal (-3.13 in the stable dir·dc);
poisson far behind (-5.9); gaussian collapses (-100 to -2700, huge variance). So held-out
prediction **disagrees** with biology (lognormal best, V=0.773) in every setting.

**This contradicts the original hypothesis** that a held-out criterion would reveal lognormal
as the best model independently of annotation. It does not — it reveals geometric. That stands.

**Reconciliation with the Piazza multiplicative-lognormal architecture.**
- The paper's lognormality is node-level: strength `S_i ~ rho_i L_i` and degree `k_i` are
  lognormal (multiplicative CLT / Galton-Watson branching). It propagates to edges: if
  `s_ij ~ (rho_i L_i)(rho_j L_j)`, a product of lognormals is lognormal, so edge weights inherit
  lognormality. This is why the lognormal-weight SBM best recovers biology (V 0.77) — it matches
  the true multiplicative structure. (Corrects an earlier over-strong "node-only" framing.)
- The paper's own label-free selection is KS on the node `S`/`k` distributions, which selects
  lognormal — i.e. at the architecturally-relevant level, selection AGREES with biology.

**Why geometric can be best at prediction yet worst at V — decoupled, not contradictory:**
- Held-out weight prediction is dominated by **curve-shape fit to the marginal integer-count
  distribution** (a marginal property), largely independent of whether blocks are biological.
  Geometric's shape hugs the truncated small counts best -> best prediction. Poisson makes MORE
  blocks than geometric yet predicts WORSE -> curve-fit, not partition, drives the score.
- V is dominated by whether the **partition aligns with cell types** (a structural property).
  Geometric's partition aligns poorly (0.57); lognormal's aligns well (0.77).
- So "best count-fitter" != "best biology-organizer"; a model can fit the histogram well while
  grouping neurons non-biologically. This decoupling IS the E2b finding.

**Open, UNTESTED hypothesis (do not use it to dismiss the result).** Geometric's marginal edge
over lognormal may be a scoring artifact: native-discrete geometric vs a continuous lognormal
discretized via `P(k)=F(k+0.5)-F(k-0.5)`, on small truncated (>=5) integer counts (median 8).

**Decisive test (proposed, NOT yet run):**
1. Calibration / synthetic recovery: generate edge weights from a KNOWN truncated integer-rounded
   lognormal (matched moments, same block structure), run the exact E2b scorer.
   - recovers lognormal -> scorer fair -> geometric's real-data win is genuine.
   - geometric wins on lognormal-generated data -> discretization artifact confirmed.
   Mirror with geometric-generated data.
2. Direct empirical KS fit of the real truncated edge-weight histogram (lognormal vs geometric).
- Likely resolution: node strengths lognormal (multiplicative rho·L) while individual edge COUNTS
  carry Poisson/geometric-style count-noise around lognormal means -> both hold, no contradiction.

**Honest stance.** The held-out test is good; its geometric result stands and must be reported
openly (no metric-shopping). Whether it undermines the thesis depends on (a) the calibration
test above and (b) whether the node-strength/degree level is the right selection criterion for
this paper. Currently unresolved — flagged, not decided.


## Next steps — PENDING (to settle E2b before deciding the narrative)

Two small tests, both cheap, set aside for now:
1. **Calibration / synthetic recovery** — generate edge weights from a KNOWN truncated,
   integer-rounded lognormal (matched moments, same block structure) and run the exact E2b
   scorer. Recovers lognormal -> scorer fair, geometric's real-data win is genuine; geometric
   wins on lognormal-generated data -> the win is a discretization artifact. Mirror with a
   geometric generator.
2. **Node-level distributional selection (the paper's own criterion)** — fit lognormal vs
   poisson/exponential/geometric to the node **strength `S_i`** and **degree `k_i`** histograms
   and select by KS + held-out likelihood. Tests selection at the architecturally-relevant level.

Together these decide whether, for this paper, unsupervised selection **agrees** with biology
(node level) or **disagrees** (edge level) — to be reported honestly, both ways. Neither run yet.


## Node-level distributional selection (Piazza-style KS) + why Poisson's V is misleading (2026-07-28)

**Test.** For each node: strength `S` (total synapses) and degree `k` (distinct partners), on the
canonical >=5 directed graph (138,584 nodes). Each candidate distribution fitted to those two
histograms; ranked by KS (lower = better). No SBM involved. This is the paper's own criterion.

| rank | strength S | KS | degree k | KS |
|---|---|---|---|---|
| 1 | **lognormal** | **0.037** | **lognormal** | **0.054** |
| 2 | weibull | 0.104 | weibull | 0.078 |
| 3 | gamma | 0.135 | geometric | 0.087 |
| 4 | geometric | 0.159 | exponential | 0.090 |
| 5 | exponential | 0.159 | gamma | 0.091 |
| 6 | normal | 0.354 | normal | 0.313 |
| 7 | power-law | 0.414 | power-law | 0.367 |
| 8 | neg-binomial | 0.528 | neg-binomial | 0.393 |
| 9 | poisson | 0.739 | poisson | 0.576 |

- **Lognormal wins decisively on both** (2.8x better than runner-up on S), replicating Piazza et al.
  on our data, including lognormal >> power-law and >> Poisson. Overdispersion is extreme
  (var/mean = 5105 for S, 218 for k).
- **The extra candidates do not threaten lognormal** (weibull 2-3x worse; gamma / neg-binomial
  unremarkable) -> no justification to add them to the SBM sweep. Cheap negative result.
- Caveat: KS is bulk-dominated and these are single-sample fits without CIs; the ranking is
  robust but KS alone is not proof about tail behaviour.

**Interpretation — why Poisson is LAST here but near-TOP on V-measure.** Different objects:
KS asks whether *one global* Poisson describes the degree histogram (it cannot: Poisson forces
var = mean, data has var/mean = 218). The SBM never fits one global Poisson — it fits a
*separate* Poisson per block-pair, i.e. a mixture of ~1600 of them, which can represent a
heavy-tailed graph. Crucially, that rigidity is *why* it scores well on V: unable to absorb
within-block variance, the model must **split until each block-pair is homogeneous**.

| | blocks | homogeneity | completeness | V |
|---|---|---|---|---|
| poisson dir ndc | 1599 | **0.756** | 0.746 | 0.751 |
| lognormal dir dc | 191 | 0.666 | **0.920** | 0.773 |

Poisson wins homogeneity (many small pure blocks), lognormal wins completeness (types kept
intact); they reach similar V by opposite routes. **So Poisson's V is a resolution artefact, not
evidence of a Poisson connectome** — its misfit causes over-splitting, and over-splitting
flatters a metric scored against ~8,000 fine-grained cell types. Same decoupling as geometric
(best held-out prediction, worst V). Methodological consequence: **V-measure conflates "correct
model" with "convenient resolution,"** which is why the node-level KS test above is the cleaner
criterion for the architecture claim.


## Biological agreement under chance-adjusted metrics; what the blocks actually are (2026-07-28)

**1. lognormal·dir·dc ranks 1st of 20 on EVERY biological metric** — AMI, ARI, and V at all four
label levels (super_class, class, sub_class, primary_type).

**2. The margin WIDENS under chance-adjustment — this answers the "coarseness artefact" objection.**
V is untrustworthy here because the models sit at very different resolutions (191 vs 1599 blocks):
more blocks inflate homogeneity for free, fewer blocks inflate completeness for free. AMI/ARI
subtract the chance baseline computed at each partition's own resolution, so neither splitting nor
merging earns credit.

| | blocks | homog | compl | V | AMI | ARI |
|---|---|---|---|---|---|---|
| lognormal dir dc | 191 | 0.666 | **0.920** | 0.773 | **0.729** | **0.556** |
| poisson dir ndc | 1599 | **0.756** | 0.746 | 0.751 | 0.659 | 0.300 |

On V lognormal leads by 3%; on **ARI by 85%**. Poisson's homogeneity advantage *evaporates* once
chance-corrected — i.e. most of that purity was expected from having 1599 blocks. ARI shows the
widest gap because it is pair-based and punishes fragmentation: every cell type split across
blocks turns all its same-type pairs into errors. So lognormal's biological win is NOT a
coarseness artefact.

**3. But the four granularity wins are NOT independent.** super_class > class > sub_class >
primary_type are nested aggregations, so a partition respecting primary_type boundaries
automatically respects the coarser ones. "Wins at every granularity" is largely ONE result
propagating upward, not four confirmations.

**4. What the blocks are: a COARSENING of the taxonomy.** For lognormal·dir·dc (165 blocks on
137,767 labelled neurons, 8,767 primary types):

| types with >= N neurons | n | % >=90% inside ONE block | median concentration |
|---|---|---|---|
| >= 1 | 8767 | 72.0% | 1.00 |
| >= 5 | 2159 | 53.9% | 0.96 |
| >= 20 | 434 | 61.5% | 0.97 |
| >= 50 | 236 | 59.3% | 0.96 |

~75 primary types per block; ~3.06 blocks per type (>=20 neurons). So the model **rarely splits a
cell type and routinely merges several** — connectivity defines *super-sets* of cell types
("type families"), not sub-divisions of them. Caveat: only ~60% of types (>=5 neurons) are >=90%
in one block, so this is a tendency, not a rule.

**5. This reframes the earlier retinotopy null.** lognormal barely splits ANY type brain-wide, so
"lognormal did not recover retinotopy" is weak evidence about retinotopy and strong evidence about
**lognormal's resolution**. Poisson, which over-splits (1599 blocks), DID produce clean DV/AP
eye-axis splits across ~20 optic types. Same graph, same biology, different granularity ->
different answer.

**6. Status of the poisson retinotopy controls.**
- CONTROLLED: degree (DC vs non-DC -> axes become *type-specific*, DV for Mi10/Dm2/Sm07, AP for
  Tm9/Tm5a, which rules out one global degree gradient); and the rim/edge effect (DV/AP is
  distinct from the eye-boundary split).
- NOT controlled: resolution. But a random-subdivision null is a **strawman** — it destroys
  spatial structure by construction and would score 0.50 regardless — and in a retinotopic system
  "connectivity-defined blocks align with eye position" IS the phenomenon, not an artefact.
- The one meaningful open test is **matched-resolution lognormal** (force ~1599 blocks, re-run the
  DV/AP scan): a disambiguation, not a null, informative either way — if the axes appear, the claim
  strengthens to "retinotopy is recoverable given adequate resolution"; if not, the weight model
  itself is doing the work. UNTESTED, and it is not established that pinning the block count
  converges.


## OPEN ISSUE — resolution must be controlled in future comparisons (2026-07-28)

Models are being compared at wildly different granularities (lognormal·dir·dc ~191 blocks vs
poisson·dir·dc ~1470), so **model and resolution are confounded** in every cross-model claim.
This affects at least:
- the **DV/AP retinotopy** result (poisson splits ~19/20 optic types along the eye axes,
  lognormal ~1/16) — currently unclear whether the weight model or the granularity causes it;
- the **biology metrics** (V is directly resolution-biased: more blocks inflate homogeneity,
  fewer inflate completeness; AMI/ARI are chance-adjusted and therefore the fairer comparison).

**Requirement going forward:** either compare at MATCHED block count, or report a
chance-adjusted metric (AMI/ARI), or state explicitly that the comparison is confounded.

**Mechanism is available and verified** (dense 3k-node subgraph, graph-tool 2.98):
`minimize_blockmodel_dl(..., multilevel_mcmc_args=dict(B_min=X, B_max=Y))` is honoured and does
not stall — it was in fact *faster* than unconstrained.

| model | range requested | blocks returned | MDL |
|---|---|---|---|
| lognormal | unconstrained | 76 | 532,863 |
| lognormal | 150-250 | 152 | 555,596 |
| lognormal | 1200-1700 | 1213 | 756,947 |
| poisson | unconstrained | 788 | 1,071,594 |
| poisson | 150-250 | 247 | 1,266,532 |
| poisson | 1200-1700 | 1228 | 1,085,578 |

Note the natural 10x resolution gap (lognormal 76 vs poisson 788) reproduces on the subgraph,
mirroring full-brain 191 vs 1470 — the confound is systematic, not incidental. Forcing costs
description length (lognormal +42% to reach 1213 blocks), so constrained fits are deliberately
suboptimal compressions; that is acceptable when the question is biology-alignment at matched
resolution, but such MDLs must NOT be compared against unconstrained ones.

**DESIGN ON HOLD (not run):** 2x2 separating model from resolution, all directed + DC —
lognormal natural (191) / forced fine (1200-1700); poisson forced coarse (150-250) / natural
(1470); outcome = number of optic types with strong DV/AP alignment. Note the coarse-lognormal
DV/AP figure currently on record came from the OLD pilot partition, so that cell needs
re-scanning on the batch partition for the 2x2 to be internally consistent.

---

## 2026-08-05 — Which structure does each weight model recover?

All runs below: threshold >=5, directed, degree-corrected, 5 seeds, per hemisphere.
Scripts: `src/spatial_vs_identity.py`, `src/t7_which_structure.py`,
`src/t7b_hemilineage_threshold.py`, `src/t8_symmetry_labelfree.py`.

### Retinotopy, on PUBLISHED coordinates (not skeleton centroids)

FlyWire column assignment (Matsliah et al. 2024) downloaded from
`storage.googleapis.com/flywire-data/codex/data/fafb/783/column_assignment.csv.gz`
— 45,528 neurons, **100% root-id overlap with our graph** (version-consistent).

`spatial_vs_identity.py --coords hex`, both ratios divided by a permutation null that preserves
block sizes exactly, so neither can be faked by block count:

| model | blocks | spatial (1=chance) | identity (1=chance) |
|---|---|---|---|
| geometric | 112 | **0.297** | **0.273** |
| gaussian | 103 | 0.427 | 0.143 |
| poisson | 222 | 0.505 | 0.152 |
| lognormal | 40 | **0.999** | 0.170 |

Geometric blocks are the tightest in space AND the most type-mixed: several cell types at one
eye position, i.e. columns. Lognormal is exactly at chance on position.

**Matched-resolution control:** gaussian 103 blocks vs geometric 112 — same resolution, yet
geometric is far more spatial (0.297 vs 0.427). The effect is the weight distribution, not B.

### Every labeling vs every model (AMI, chance-corrected)

| labeling | lognormal | gaussian | geometric | exponential | poisson | winner |
|---|---|---|---|---|---|---|
| cell type | **0.736** | 0.594 | 0.462 | 0.462 | 0.662 | lognormal |
| neuropil | **0.541** | 0.468 | 0.448 | 0.448 | 0.395 | lognormal |
| neurotransmitter | **0.188** | 0.134 | 0.102 | 0.102 | 0.132 | lognormal |
| hemilineage (ito-lee) | 0.502 | 0.514 | 0.509 | 0.508 | 0.380 | tie |
| hemilineage (hartenstein) | 0.481 | 0.495 | 0.493 | 0.492 | 0.355 | tie |
| retinotopic column | **-0.076** | 0.153 | 0.318 | **0.319** | 0.047 | geometric/exponential |

Lognormal is **below chance** on retinotopy (-0.076) — not merely worse, anti-correlated.
Geometric and exponential agree to 3 decimals throughout (consistency check: same distribution,
discrete vs continuous).

**No third structure exists in this data.** Neuropil / neurotransmitter / hemilineage all go to
lognormal because none of them VARIES WITHIN A CELL TYPE — they are coarsenings of cell type,
not independent axes. Retinotopy is the only labeling that varies within a type (every Mi1 sits
at a different column), which is exactly why it dissociates. Hemilineage stays a 4-way tie at
every minimum-class-size threshold (1..100, margins 0.001-0.006); restricted to the 10 largest
lineages lognormal wins by 0.023 in BOTH annotation schemes, i.e. it is identity-like too.

### Bilateral symmetry, label-free (only `side`, no cell types)

13.1% of edges cross the midline, so splitting the brain by hemisphere is cheap but not free.

| model | blocks | AMI(block, side) | neurons in two-sided blocks | minority share |
|---|---|---|---|---|
| **lognormal** | 191 | **0.053** | **80%** | 0.38 |
| poisson | 1470 | 0.141 | 20% | 0.09 |
| gaussian | 561 | 0.164 | 17% | 0.09 |
| exponential | 578 | 0.164 | 14% | 0.08 |
| geometric | 563 | 0.165 | 14% | 0.08 |

Lognormal merges homologous groups across the midline; every other model splits down it.
**This is not a third leg** — homologs are the same cell type, so this is the cell-type result
established WITHOUT the annotation, which answers the circularity objection.

Not run: separate per-hemisphere fits (PLAN RQ-C / E3). Not needed for the comparative claim
above (all models saw identical data); would upgrade it to independent replication.

---

## 2026-08-05 — Held-out prediction, final (E2b complete)

282/288 cells scored. Full method + caveats: `docs_heldout_prediction.md` (written for external
review; four rounds of corrections folded in).

Canonical (directed + DC), **unit = the 6 unique held-out splits**, seeds averaged within a
split first (seeds are re-inferences of the same split, not replicates):

| family | nats/held-out edge | spread across splits |
|---|---|---|
| **lognormal** | **-2.849** | 0.003 |
| gaussian | -2.932 | 0.005 |
| geometric | -3.015 | 0.004 |
| poisson | -6.134 | 0.073 |

Paired on identical held-out edges: lognormal beats gaussian by +0.083 and geometric by +0.166,
in 6/6 splits. SD/SEM are DESCRIPTIVE — the six splits resample one graph and any two training
sets share ~90% of edges, so no significance is implied.

Lognormal wins the two DEGREE-CORRECTED combinations only; **gaussian wins both non-DC ones**
(dir+ndc -2.889, und+ndc -3.044). Undirected rows are computed on the 5 splits every family
completed.

**Fitted weight parameters** (`d x locally-fitted pairs + d`; fallback pairs share ONE global
fit, so counting occupied pairs overstates by up to 14x):

| family | blocks | occupied pairs | locally fitted | fitted params |
|---|---|---|---|---|
| **lognormal** | 182 | 11,799 | 5,905 | **11,811** |
| geometric | 533 | 48,968 | 23,694 | 23,695 |
| gaussian | 610 | 53,315 | 18,992 | 37,985 |
| poisson | 1,531 | 590,963 | 41,421 | 41,422 |

Lognormal predicts best while fitting 2-3.5x fewer weight parameters.

**6 failed cells are non-random**: all gaussian / undirected / random / fold 2, every seed, both
DC settings, deterministic across retries. Those two gaussian numbers are optimistic.

## 2026-08-05 — Piazza et al. (bioRxiv 2025.02.27.640551) re-analysis

Scripts: `src/t1_mixture_test.py`, `src/t4_closure_test.py`, `src/t5_regime_split.py`,
`src/t6_edge_level.py`.

**T6 (edge level).** Their procedure — MLE fit, ranked by KS — run on edge weights instead of
node strength/degree. Lognormal ranks **1st in every regime** (all edges 0.155; map 0.139;
non-optic 0.146; optic-rest 0.165); geometric is near-worst (0.259-0.393). Rescaling collapse
0.172 on the log scale vs 0.249 raw. So their criterion and our held-out criterion AGREE on edge
weights — two independent justifications for lognormal.
DATA BUG FOUND: `connections_princeton` has ONE ROW PER (pre, post, NEUROPIL). Rows go down to
1 synapse; EDGES do not. Summing over neuropils gives 3,732,460 edges, all w>=5, confirming
PLAN's >=5 floor. The first T6 run treated 5.34M rows as edges and gave the opposite answer.

**T1 (is lognormality a mixture effect?).** At MATCHED n — each group vs a random pooled
subsample of identical size, so both carry the same KS noise floor — conditioning makes the fit
WORSE: strength S by type 0.109 vs 0.069 pooled (group worse in 80% of 236 types); by block
0.100 vs 0.055 (83%); degree k by type 0.123 vs 0.071 (88%); by block 0.111 vs 0.058 (80%).
Pooled data is closer to lognormal than homogeneous groups are. Their Galton-Watson derivation
acts per neuron and should survive conditioning; it does not.
Note: the naive KS RATIO suggests the opposite (21.8 pooled -> 2.2 by type) but it grows like
sqrt(n) for fixed shape deviation — corr(log n, ratio) = +0.5..+0.8. Only matched-n is valid.

**T4 (do their closure identities hold within groups?).** rho measured INDEPENDENTLY from
synapse positions (102.7M synapses scanned, 10um radius, their SI 3.2) for 59,010 neurons —
defining rho := S/L would make the identities vacuous.

| level | corr(log S, log rho + log L) | sigma predicted / observed |
|---|---|---|
| pooled | 0.904 | 1.014 |
| by cell type (330) | 0.897 | 1.023 |
| by SBM block (152) | 0.905 | 1.015 |

Their PHYSICAL law (Eq. 1 and the Eq. 4 variance closure) holds within every group. Eq. 3 as
literally written cannot be tested — S ~ rho*L is a proportionality and L is in nm, so the
`mu_err ~ 12.58` offset is the unit constant (near-identical at all three levels).

**Net:** their physical constraint is solid and local; the lognormal SHAPE is an aggregate
property. T5 (splitting neurons into map / optic-rest / non-optic regimes) found no two-law
story — lognormal wins strength and degree in every regime.

**OPEN — no validated mechanism** for WHY a lognormal weight model recovers cell type. The one
untested candidate: the families differ only in how spread scales with the mean (Poisson locks
var=mean, geometric locks the ratio, gaussian wants constant absolute spread, lognormal leaves
the ratio free), so measuring spread-vs-mean across cell-type-pair bundles would discriminate.
Nothing in the results above depends on it.
