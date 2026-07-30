# Findings so far — connectome SBM project

*Status document for discussion, 2026-07-30. Not a final write-up. Solid findings and open
problems are marked separately.*

---

## The question

We fit a Stochastic Block Model (SBM) to the fly connectome (FlyWire v783: 138,584 neurons,
3.7M connections, ≥5 synapses). The SBM groups neurons by who they connect to, **without ever
seeing biological labels**.

We can choose different assumptions about how connection *strengths* are distributed —
lognormal, Gaussian, Poisson, geometric, exponential. Two questions follow:

1. Which assumption best describes the connectome, judged **without labels**?
2. Does that same model best recover **biology** (cell types, spatial maps)?

---

## Solid findings

### 1. Node-level statistics: lognormal wins clearly

Fitting distributions directly to each neuron's total synapses and number of partners
(no SBM involved), ranked by goodness of fit (lower = better):

| rank | node strength | | node degree | |
|---|---|---|---|---|
| 1 | **lognormal** | **0.037** | **lognormal** | **0.054** |
| 2 | Weibull | 0.104 | Weibull | 0.078 |
| 3 | gamma | 0.135 | geometric | 0.087 |
| … | | | | |
| 7 | power-law | 0.414 | power-law | 0.367 |
| 9 | Poisson | 0.739 | Poisson | 0.576 |

- Lognormal is **~3× better** than the runner-up.
- This **replicates Piazza et al. on our own data**, including their key claim that the
  connectome is lognormal rather than scale-free (power-law).
- We also tested extra candidates (Weibull, gamma, negative binomial) — none competes, so
  there is no reason to expand the model set.

### 2. Lognormal SBM matches biology best — on every measure

Comparing the SBM's groups against FlyWire's cell-type labels, across 20 model variants:

- **lognormal (directed, degree-corrected) ranks 1st of 20 on every biological measure** —
  at all four label levels (super-class, class, sub-class, cell type).
- Importantly, its lead **grows** on the measures that correct for chance:
  - simple score: lognormal 0.773 vs Poisson 0.751 (only 3% better)
  - chance-corrected: lognormal **0.556** vs Poisson **0.300** (85% better)
- This rules out the obvious objection that lognormal only looks good because it makes fewer,
  bigger groups. Coarseness earns nothing once you correct for chance.

### 3. What the groups actually are: families of cell types

- A typical cell type sits **97% inside a single group**.
- Each group contains **~75 different cell types**.
- So the SBM does **not** cut cell types apart — it **bundles related types together**.
- Connectivity therefore defines something *coarser* than the cell-type catalogue: "families"
  of types that wire alike.

### 4. Splitting happens only where biology has a spatial map

Looking at which cell types the SBM subdivides:

- **Every** subdivided type is in a topographically organised system: vision (optic lobe),
  hearing (Johnston's organ), smell (olfactory receptors), navigation (central complex).
- **Kenyon cells — the textbook example of random, map-free wiring — are never subdivided**
  (each subtype stays ≥95% in one group).
- Brain-wide, **no** non-visual type passes our splitting threshold.

This is a clean positive/negative control pair: the method subdivides exactly where a spatial
map exists, and leaves random wiring alone.

### 5. Retinotopy: the eye's map is recovered — at fine resolution

Using FlyWire's published eye-coordinate map (each optic neuron's position in the eye):

- With a **fine** model (Poisson, ~1,470 groups), **19 of 20** optic cell types are split along
  a real eye axis (up–down or front–back), with 82–99% accuracy.
- With a **coarse** model (lognormal, ~190 groups), almost none are.
- We controlled for a trivial explanation (connection counts): after correction the axes become
  **type-specific** — some types split up–down, others front–back — which a single global
  artefact could not produce.

### 6. Hierarchies now work (new)

The nested/hierarchical SBM previously never completed. It now runs: **80/80 fits, 72 clean**.
Example hierarchy (Poisson): **4795 → 1320 → 681 → 416 → 118 → 44 → 19** groups.

This matters because a single hierarchical fit spans *both* scales at once — the cell-type-family
scale and the retinotopic scale.

---

## The central tension (honest, unresolved)

Different criteria pick different models:

| criterion | picks | typical #groups |
|---|---|---|
| matches cell types | **lognormal** | ~190 |
| node-level statistics | **lognormal** | — |
| description length (compression) | exponential / geometric | ~570 |
| recovers the eye map | Poisson | ~1,470 |

The one variable that orders all of them is **resolution**: the weight assumption controls how
finely the SBM divides the brain, and different biological structures live at different scales —
cell-type families at ~190 groups, the eye's map at ~1,470. Compression optimises something in
between and matches neither.

We cannot yet say whether this is a fact about the weight models or purely about resolution.

---

## Bottom line for discussion

- **Strong and stable:** lognormal is the best description of the connectome's node statistics
  and the best match to cell types, by every measure including chance-corrected ones.
- **Strong and clean:** the method subdivides cell types only in systems with real spatial maps,
  never in randomly-wired ones, and recovers the eye's axes when resolution allows.
- **Open:** whether label-free model selection agrees with biology. Current evidence says it
  does not, but model and resolution are still confounded.
