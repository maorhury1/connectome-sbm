# All findings so far

*Fly connectome (FlyWire v783): 138,584 neurons, 3.7M connections of ≥5 synapses.
We group neurons by who they connect to, without ever showing the method any biological labels,
then ask what it recovered.*

---

## A. Which distribution describes the connectome?

- **Lognormal fits the neuron-level data best, by a wide margin.** Fitting distributions directly
  to each neuron's total synapses and number of partners (no grouping involved), lognormal is
  ~3× better than anything else.
- **It beats the power-law decisively** (0.037 vs 0.414 — lower is better). The connectome is not
  "scale-free". This replicates Piazza et al. on our own data.
- **Poisson is the worst of all** (0.739). The data are far too spread out for it.
- **We tested extra candidates** — Weibull, gamma, negative binomial — and none competes. So
  there is no reason to widen the model set.
- **The connections are extremely uneven**: variance is 5,105× the mean for synapse totals. This
  is why simple models fail.

## B. What do the groups the method finds actually correspond to?

- **Lognormal's groups match biological cell types better than any other model** — 1st out of 20
  variants, on every measure, at all four levels of the biological hierarchy.
- **This is not just because it makes fewer groups.** On measures that correct for chance, its
  lead *grows*: 0.556 vs Poisson's 0.300 (85% better), where the naive measure showed only 3%.
- **The groups are families of cell types, not subdivisions of them.** A typical cell type sits
  **97% inside a single group**, and each group holds **~75 different types**.
- So connectivity defines something **coarser** than the cell-type catalogue: bundles of types
  that wire alike.

## C. Where does the method split a cell type in two?

- **Only in systems that have a spatial map.** Every subdivided type belongs to vision, hearing
  (Johnston's organ), smell (olfactory receptors), or the navigation centre.
- **Kenyon cells are never subdivided.** These are the textbook case of random, map-free wiring —
  each subtype stays ≥95% in one group. The method correctly leaves them alone.
- **Brain-wide, no non-visual cell type passes our splitting threshold at all.**
- Together this is a clean control pair: it splits exactly where a map exists, and not otherwise.

## D. The eye map (retinotopy)

- **With a fine-grained model (Poisson, ~1,470 groups), 19 of 20 optic cell types split along a
  real eye axis** — up–down or front–back — at 82–99% accuracy, using FlyWire's own published
  eye-coordinate map.
- **With a coarse model (lognormal, ~190 groups), almost none do.**
- **It is not an artefact of connection counts.** After correcting for that, the axes become
  *type-specific* — some types split up–down, others front–back — which a single global artefact
  could not produce.
- **Most splits under the coarse model are just the edge of the eye.** Neurons at the eye's rim
  have fewer neighbours to wire with, so they separate out. This is real but uninteresting — it
  is the boundary, not a functional map.
- **We checked it is not the "acute zone"** (the eye's high-acuity patch): the split is a ring
  around the centre, while the acute zone is a one-sided patch.
- **One type, Tm32, splits along the up–down axis in *both* eyes** (0.86 and 0.77) — the only
  clean case under the coarse model.
- **The map is weakly present in the wiring itself.** Using only connectivity, we can predict a
  neuron's position in the eye ~28% better than chance (after a strict spatial control). Real,
  but weak.
- **Published eye coordinates exist only for the right eye.** Mirroring them to the left is
  noticeably less accurate (27 µm vs 7 µm), so left-eye results are approximate.

## E. Choosing a model without using biological labels

- **The software's built-in score cannot be compared across model families.** Continuous and
  discrete models are measured in different units, and log-transformed models need a correction.
- **That correction is large and it reverses the naive answer**: 8.56M units, enough to move
  lognormal from apparent 1st to 4th.
- **Under the correct published framework, compression picks exponential**, with geometric close
  behind — not lognormal.
- **Compression and biology disagree.** Geometric and exponential, which compress best, are the
  *worst* match to biology at every level.
- **Resolution is the variable that orders everything:**

  | what it recovers | model | groups |
  |---|---|---|
  | families of cell types | lognormal | ~190 |
  | (best compression) | exponential/geometric | ~570 |
  | the eye's map | Poisson | ~1,470 |

  Biology lives at two separated scales; compression lands between them and matches neither.
- **Caveat:** each model produces its own number of groups, so "wrong model" and "wrong
  resolution" are not yet separated.

## F. Practical findings about the method itself

- **Hierarchical (nested) grouping now works** — 80 fits, 72 clean. Example hierarchy:
  4,795 → 1,320 → 681 → 416 → 118 → 44 → 19 groups. A single hierarchical fit therefore spans
  *both* biological scales at once.
- **Except with lognormal**, which never converges on the full brain (a week without finishing).
  It works on smaller pieces, so this is a scale limit, not a bug.
- **The "≥1 synapse" dataset does not exist** in the standard release — it is pre-filtered at ≥5.
  Sub-5 connections are considered unreliable by the data producers, so ≥5 is the honest setting.
- **Newer versions of the grouping software cannot run on the lab server** (they require a newer
  Linux). This is why hierarchies had to be run the way they were.

---

## Bottom line

- **Lognormal is the best description of the connectome's neuron-level statistics, and the best
  match to biological cell types — including under chance-corrected measures.**
- **The method subdivides cell types only where biology has a spatial map, and never in
  randomly-wired neurons.**
- **The eye's map is recoverable from wiring alone, but only at fine resolution.**
- **Label-free compression does not select the model that matches biology.** It prefers a
  resolution between the two scales where biology actually lives.
