# Pre-paper: claims, evidence, and provenance

Target venue: NeurReps (workshop). Status: results complete, paper unwritten.

This document states every claim we intend to make, the script that produced its evidence, the
control that makes it legitimate, and the limitation that bounds it. Claims we considered and
dropped are in §6; things we deliberately do NOT claim are in §5.

Companion: **`REVIEWER_MAP.md`** (where everything lives), **`RESULTS.md`** (all numbers).

---

## 1. The question

To group neurons by connectivity with a weighted stochastic block model, you must assume
*something* about how connection strengths are distributed. That assumption is normally treated
as a technicality — a line in a config.

We ask whether it is one. Holding the graph, the preprocessing, and the SBM family fixed, and
varying only the edge-weight likelihood, we ask which biological structure the inferred
partition recovers.

**Headline.** The weight distribution decides *which* biological structure the model recovers.
A lognormal likelihood recovers cell-type identity and is at or below chance on retinotopic
position. A geometric likelihood recovers retinotopic position — including its factorization
from identity and its mirror symmetry — and is markedly worse on identity. This is not a
resolution effect; it survives matching the models at equal block count.

Setting throughout: FlyWire codex v783, w>=5, directed, degree-corrected, 5 seeds, per
hemisphere. `t5_dir_dc` in the file naming.

---

## 2. Claim 1 — lognormal recovers cell-type identity

### 2.1 It predicts unseen weights best, with the fewest parameters
`src/xval.py`, `src/worker_e2b.py`, `src/batch_e2b.py` -> `docs_heldout_prediction.md`

5% of edges are **removed from the graph** before fitting (not masked afterwards), so the model
never sees them in adjacency or in weight. Every family is scored as a probability of the
observed integer on a common support: `[F(k+0.5) - F(k-0.5)] / (1 - F(4.5))` for continuous
families, own pmf over `P(K>=5)` for discrete. Fitting maximises the same truncated likelihood
that is scored.

| family | nats / held-out edge | fitted weight parameters |
|---|---|---|
| **lognormal** | **-2.849** | **11,811** |
| gaussian | -2.932 | 37,985 |
| geometric | -3.015 | 23,695 |
| poisson | -6.134 | 41,422 |

Paired on identical held-out edges: lognormal beats gaussian by +0.083 and geometric by +0.166,
in **6/6 splits**.

- *Control*: every family is scored on the same held-out edge set; the comparison is paired.
- *Unit*: the 6 unique splits, not the 18 seed-cells — seeds are re-inferences of one split.
  Spreads are **descriptive**; the six splits share ~90% of their training edges, so no
  significance is claimed.
- *Limitation*: lognormal wins the two **degree-corrected** settings only. Gaussian wins both
  non-DC settings. 6 cells failed, all gaussian/undirected/fold-2, a non-random dropout that
  favours gaussian.

### 2.2 It agrees best with the annotation
`src/t7_which_structure.py`, `src/t7b_hemilineage_threshold.py`

AMI (chance-corrected, so extra blocks do not help): cell type **0.736** vs gaussian 0.594,
poisson 0.662, geometric 0.462. Also first on neuropil (0.541), neurotransmitter (0.188), and —
restricted to the 10 largest lineages — hemilineage (+0.023 over second place, in **both**
independent annotation schemes).

- *Limitation*: neuropil, neurotransmitter and hemilineage do **not vary within a cell type**.
  They are coarsenings of it, not independent axes. We report them as such, not as extra wins.
- *Limitation*: FlyWire cell types were partly defined from connectivity. **This claim is
  partly circular and must be stated as such.** §2.3 is the version that is not.

### 2.3 The same result without the annotation
`src/t8_symmetry_labelfree.py` — uses only `side`, never cell type.

13.1% of edges cross the midline, so splitting the brain by hemisphere is cheap but not free.
Lognormal declines to: AMI(block, side) **0.053**, with **80%** of neurons in blocks containing
both hemispheres. Every other model splits down the midline (14-20% two-sided).

Homologous neurons are the same cell type, so this is §2.2 established **without touching the
annotation** — it answers the circularity objection directly. It is not an independent third
finding and we will not present it as one.

### 2.4 Scope: it holds for well-populated types only
`src/t13_recovery_by_size.py` — bins declared in advance, null = 20 block-label shuffles.

Null-corrected ratio (obs/null) of per-type Dice against the best-matching block:

| type size (per hemisphere) | lognormal | gaussian | geometric | poisson |
|---|---|---|---|---|
| 2-3 | 2.4 | 2.2 | 2.4 | 2.6 |
| 8-15 | 4.2 | 4.7 | 4.3 | 3.2 |
| **32+** | **19.6** | 11.3 | 9.2 | 13.4 |

**Below 8 neurons per hemisphere all five models are statistically indistinguishable.**
Lognormal is the *worst* model below 16 and only dominates from 32+.

The failure mode for small types is **merging, not fragmentation**: recall stays 0.85-0.91 while
purity is 0.01-0.05 — small types sit intact inside a much larger block. The claim to make is
that the cell-type signal is real and model-dependent for types with >=32 neurons, not that a
~190-block model recovers 8,600 types.

---

## 3. Claim 2 — geometric recovers retinotopic position

### 3.1 Its blocks are columns
`src/spatial_vs_identity.py --coords hex`

Published FlyWire column coordinates (Matsliah et al. 2024), 100% root-id match with our graph.
Both statistics divided by a permutation null that **preserves block sizes exactly**, so neither
can be produced by block count.

| model | blocks | spatial (1 = chance) | identity (1 = chance) |
|---|---|---|---|
| geometric | 112 | **0.297** | **0.273** |
| gaussian | 103 | 0.427 | 0.143 |
| lognormal | 40 | **0.999** | 0.170 |

Geometric's blocks are simultaneously the tightest in space and the most type-*mixed* — several
cell types at one eye position, i.e. columns. Lognormal is exactly at chance on position, and by
AMI it is **below** chance (-0.076, `t7_which_structure.py`).

- *Control that matters most*: **gaussian 103 blocks vs geometric 112** — matched resolution, and
  geometric is still far more spatial. The effect is the weight distribution, not B.

### 3.2 It factorizes identity x position, not merely refines type
`src/t11_conditional_mi.py`

Conditional MI I(block ; position | cell type), null = 50 **within-type** block shuffles.

| model | excess CMI | as fraction of H(position \| type) | types with excess > 0.05 |
|---|---|---|---|
| geometric | 1.543 | **0.474** | **31/31** |
| gaussian | 1.303 | 0.400 | 31/31 |
| lognormal | **0.009** | 0.003 | 1/31 left, 0/31 right |

**Within a single cell type**, geometric's blocks still resolve ~47% of the remaining positional
entropy, 3.4x above its own null, in every type and both hemispheres. This is the difference
between "retinotopic refinement" and a genuine factorization claim.

Enabling fact: `I(type ; position) = 0.006 nats` — every columnar type tiles the whole eye, so
identity carries essentially no positional information and conditioning costs the spatial models
nothing.

- *Trap we hit, worth checking*: raw `column_id` is unique per (hemisphere, type, column), which
  forces `I(B;C|T) = H(B|T)` for **any** partition and makes the null identical. The hex lattice
  must be pooled into super-columns. Ordering is invariant at d = 2/4/6/8/12.

### 3.3 The recovered map respects the mirror symmetry of the visual system
`src/t12_mirror_symmetry.py`

The two optic lobes are mirror images, so a genuine map should correspond across hemispheres
under a reflection. Block centroids in (p,q), Hungarian matching, 8 candidate transforms, null =
within-hemisphere block-id shuffles.

| model | winning transform | residual | null | obs/null |
|---|---|---|---|---|
| geometric | identity, 4/5 seeds | **1.750** | 8.985 | 0.195 |
| lognormal | none consistent | 1.105 | 1.243 | **0.89 (chance)** |

The four spatial models sit 60-130 sd below their nulls, in every seed and every draw.

- *Convention established independently of the SBM*: regressing 3D centroids on (p,q) per
  hemisphere shows the published file is already in a **mirrored, eye-centric frame** (implied
  right->left map ||A - I|| = 0.117 vs ||A - S|| = 1.97). So identity is the anatomically correct
  correspondence, and its winning is expected rather than trivial.
- *Limitation*: the four DV-preserving transforms score 1.75-2.0 and the four DV<->AP-swapping
  ones 5.1-5.4, but most of that gap is the **shape** of the (p,q) cloud, which is elongated
  along p=q. Within the shape-preserving four, identity leads only modestly.

---

## 4. Claim 3 — why: the dispersion ratio must be free

`src/t10_block_conditioned.py`

Everything above is about partitions. This is about the likelihood itself, and it is the audit
the original brief asked for first: the relevant object is not the pooled weight histogram but
**the weight distribution conditional on a block pair**.

All families scored as distributions on integers >=5 on identical footing (same truncated
discretised likelihood, MLE on it, KS against that step CDF), so the usual discrete-vs-continuous
KS objection does not apply. % of bundles **rejected** at KS > 1.36/sqrt(n):

| family | rejected (unweighted / size-weighted) |
|---|---|
| lognormal | **10.1 / 4.3** |
| gamma | 25.1 / 12.1 |
| geometric = exponential | 45.6 / 32.9 |
| poisson | 76.8 / 78.6 |

**The mechanism.** A truncated geometric locks `var = (m-4)(m-5)` exactly. Observed
`var_obs / var_geom` at n>=1000 has median 1.165 and a 5-95% range of [0.454, 3.901] — an 8.6x
spread, with only 26.4% anywhere near 1. Per-bundle fitted sigma implies CV from 0.41 to 4.04.
**The dispersion ratio genuinely varies, and it is not sampling noise.**

That is why Poisson (variance locked to the mean) and geometric (ratio locked to 1) fail
*structurally* rather than by bad luck — and it predicts Poisson's observed behaviour: it
fragments into ~1,470 blocks because splitting is its only way to manufacture dispersion it
cannot express.

**Free dispersion is necessary but not sufficient.** Gaussian has a completely free ratio and
still loses decisively (6/6 on held-out, 0.594 vs 0.736 on cell type); gamma is rejected 2.5x
more often. The log scale does real work on top of the free ratio.

- *Scope*: Weibull fits bundles at least as well as lognormal (rejected 4.5%), but it is not
  available as a graph-tool weight likelihood and was only fit **post-hoc** to bundles from
  another model's partition — it never produced a partition. The claim is therefore
  **"best-fitting among the weight models the SBM can use"**, with the falsifiable prediction
  that a gamma or Weibull weight SBM would land in the same region as lognormal.
- *Do not use the log(var) ~ log(mean) slope as a discriminator*: it is unstable across the
  weight floor (3.94 all bundles, 2.58 at mean>=10, 1.49 at mean>=20) and straddles the geometric
  null. The locked-ratio test above is the sound version.

---

## 5. What we do NOT claim

- **Not** that lognormal is the true or unique weight law (see §4 scope).
- **Not** that lognormal recovers cell types *in general* — only for types with >=32 neurons per
  hemisphere (§2.4).
- **Not** that the cell-type agreement is free of circularity (§2.2); §2.3, §3.2 and §3.3 are the
  label-free results.
- **Not** anything about edge *existence*. Every predictive statement is conditional on an edge
  already known to exist with w>=5.
- **Not** that compression and prediction agree. Description length ranks the families
  differently — exponential/geometric ahead of lognormal. We report the disagreement rather
  than selecting the criterion that flatters the result.
- **Not** a mechanistic account of why lognormal specifically recovers *cell types*. §4 explains
  why the losers lose; it does not explain why the winner's groups coincide with the taxonomy.
- **Not** a third biological axis. Neuropil, neurotransmitter and hemilineage do not vary within
  a cell type, so they cannot dissociate. Retinotopy is the only orthogonal structure this
  connectome offers.

---

## 6. Considered and dropped (with reasons)

| what | why dropped |
|---|---|
| Hand-built BIC (`src/bic.py`) | four correct objections from an external reader; the adjacency term does not cancel, the penalty should use `log m_rs`, `N*H(pi)` is a plug-in, the directed count is K^2. Superseded. |
| `docs_description_length.md` | superseded by `docs_peixoto_evidence.md` |
| A "two regimes, two laws" story (`t5_regime_split.py`) | tested and **false**: lognormal wins strength and degree in the map regime too. |
| "Poisson over-splits specifically things with maps" | tested across all models: geometric does it *more* (7.1x vs 6.1x) with a third the blocks. The selectivity tracks resolution, not Poisson. |
| Spectral block-graph geometry (`t14_spectral_geometry.py`) | run, not used. It re-confirms §2.3 and §3.1 by another route, but Poisson leads the retinotopy axis (block count is not controlled there), which would cost a paragraph to defend for no gain. |
| Nested level vs biological level (`t9_nested_hierarchy.py`) | **real but not evidence for the thesis** — all four models walk primary_type -> sub_class -> class identically, so it is about resolution, not the likelihood. Discussion material. Also: no lognormal nested run exists. |
| KS-within-subgroups on strength | circular: the SBM groups by connectivity and strength is a connectivity property. |

---

## 7. Relation to Piazza et al. (bioRxiv 2025.02.27.640551)

Their claim is that connectome node degree and strength are lognormal, derived from a
per-neuron multiplicative growth process. Ours is one level down, about edge weights within
block pairs. Three checks:

- `src/t6_edge_level.py` — their procedure run at the **edge** level ranks lognormal **1st in
  every regime** (all edges 0.155; map 0.139; non-optic 0.146). So their criterion and our
  held-out criterion agree on edge weights: two independent justifications.
- `src/t4_closure_test.py` — their **physical** law survives conditioning. With rho measured
  independently from 102.7M synapse positions (defining rho := S/L would make the identities
  vacuous), `corr(log S, log rho + log L)` = 0.904 pooled, 0.897 by cell type, 0.905 by block;
  the Eq. 4 variance closure holds at 1.01-1.02 at every level.
- `src/t1_mixture_test.py` — their **lognormal shape** does not. At matched n (each group against
  a random pooled subsample of the same size, so both carry the same KS noise floor),
  conditioning makes the fit *worse* in 80-88% of groups. Pooled data is closer to lognormal
  than homogeneous groups are, which is a mixture signature and is in tension with a per-neuron
  derivation.

  *Method note*: the naive KS **ratio** suggests the opposite (21.8 pooled -> 2.2 by type). It is
  an artefact — for a fixed shape deviation the ratio grows like sqrt(n)
  (corr(log n, ratio) = +0.5..+0.8). Only the matched-n comparison is valid.

We do not claim their result is wrong. We claim their physical constraint is local and their
distributional shape is aggregate.

---

## 8. Known weaknesses, stated plainly

1. **Circularity** in every cell-type comparison (§2.2). Mitigated, not eliminated, by §2.3.
2. **Six non-random held-out failures**, all gaussian/undirected/fold-2, favouring gaussian.
3. **The `t1` half of the flat sweep is void** — a >=1 graph cannot be built from the codex table,
   so those 100 runs duplicate their `t5` twins.
4. **Nested `level_*` arrays are corrupt** (`t9` rebuilds from `bs_*`); nested coverage excludes
   lognormal entirely.
5. **`rec_params` was never varied.** It is the prior over each block-pair's weight parameters
   and it is *not symmetric across families* — two-parameter families (lognormal, gaussian) vs
   one-parameter (Poisson, geometric). It is the one remaining untested knob that could move the
   description-length comparison.
6. **No baselines.** Leiden/Louvain at matched K and NTAC at the known type count were planned
   (`PLAN.md` E4) and never run. We have not shown an SBM is *needed*.
7. **Single connectome.** No replication on a second dataset.
