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
| hide-and-check prediction | *being re-run* (see below) | ~570 |
| recovers the eye map | Poisson | ~1,470 |

Two readings, and we cannot yet separate them:

- **Reading A:** the weight assumption controls the *resolution* of the grouping, and different
  biological structures live at different resolutions. Label-free criteria optimise compression
  and land between them.
- **Reading B:** the label-free criteria were measured with buggy code (see below), and once
  corrected they may agree with biology after all.

---

## Correction in progress (important)

An external reviewer found real defects in our hide-and-check prediction test. All were
verified in the code and **all penalised the 2-parameter models (lognormal, Gaussian)
specifically** — i.e. in exactly the direction of the result we had reported:

1. models were **fitted** one way and **scored** another;
2. parameters were fitted from as few as 5 data points, and failed fits were silently accepted;
3. for undirected graphs, each group-pair was **split in two** by a key-ordering bug;
4. (found by our own test) a core function was not vectorised, so a fix silently did nothing.

All four are fixed and pushed. The full test (288 runs) is re-running.

**Early result, 76 of 288 runs done:**

- lognormal (directed, degree-corrected) now scores **−2.849** (very stable, std 0.007).
- The previous best of *any* model, under buggy code, was **−2.92**.
- Lognormal's own previous score in this setting was −3.13 to −3.68, with huge variability.

So the corrections moved lognormal up substantially — as predicted if the bugs were the cause.
**Provisional only:** geometric and Poisson have not yet re-run, so the comparison is not final.

---

## Method work (defensive, for reviewers)

- **Description length is not comparable across model families** as reported by the standard
  software — continuous and discrete models are measured in different units, and log-transformed
  models need a correction term. We audited this: the correction is ~8.6M units and **reverses**
  the naive answer.
- We adopted **Peixoto's published framework** for the comparison rather than inventing one, and
  documented precisely where our version departs from the literature.
- We built a common-scale compression criterion (CPWDL) and **validated it before use**:
  simulated data from each of the five models, then checked the criterion recovers the correct
  one. It does — **5/5 on the real group-size distribution**.

---

## What we would still want

1. **Finish the corrected hide-and-check test** (~1 day) — decides whether the label-free
   criteria agree with biology once the bugs are gone.
2. **Separate model from resolution.** Fit lognormal and Poisson at the *same* number of groups
   and re-test the eye map. Answers whether retinotopy needs Poisson specifically, or just needs
   finer resolution.
3. **Use the hierarchies** now that they run — a single nested fit may recover cell types at one
   level and the eye map at another, which would resolve the tension directly.

---

## Bottom line for discussion

- **Strong and stable:** lognormal is the best description of the connectome's node statistics
  and the best match to cell types, by every measure including chance-corrected ones.
- **Strong and clean:** the method subdivides cell types only in systems with real spatial maps,
  never in randomly-wired ones, and recovers the eye's axes when resolution allows.
- **Open:** whether label-free model selection agrees with biology. The result that said "no"
  was measured with code that was biased against lognormal; the corrected re-run is underway and
  currently pointing the other way.
