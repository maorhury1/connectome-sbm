# The description-length criterion used here, and how it differs from published ICL

*Written 2026-07-28. Purpose: state exactly which parts of our model-selection criterion come
from the literature and which are our own adaptation, so the distinction is never blurred in
the paper.*

---

## 1. Why we needed a criterion at all

We fit the same connectome with five weight likelihoods (lognormal, gaussian, poisson,
geometric, exponential) and want to ask, **without using biological labels**, which one is best.

graph-tool reports its own MDL (`state.entropy()`), but that number is **not comparable across
weight families**:

- discrete families (poisson, geometric) encode **probabilities**; continuous families
  (normal, exponential) encode **densities**, which are per-unit-of-x and can exceed 1 — so the
  two are in different units;
- lognormal encodes `log w` while gaussian encodes `w` — a change of variable worth a Jacobian
  term (`Σ log w` ≈ 8.56M nats here, which reversed the naive lognormal-vs-gaussian verdict);
- each `rec_type` is priced with its own internal priors.

MDL therefore stays valid **within** a family and on the same graph, and we report it that way.
For a five-way comparison we need one criterion we compute ourselves under a single fixed rule.

---

## 2. The published criterion: ICL for SBM

**Origin:** Biernacki, Celeux & Govaert (2000) for mixture models; adapted to SBM by
**Daudin, Picard & Robin (2008)**. We could not obtain a machine-readable copy of Daudin;
the formula below is as stated in **Côme & Latouche, arXiv:1303.2962**, which quotes it.

ICL is built on the *integrated complete-data log-likelihood*, which factorises as

```
log p(X, Z | K)  =  log p(X | Z, K)  +  log p(Z | K)
                     ^^^^^^^^^^^^^       ^^^^^^^^^^^
                     data given          the partition
                     the partition       itself
```

and, after Laplace/Stirling approximation, for a directed graph without self-loops:

```
ICL(Z, K)  ≈  log p(X, Z | K)  −  (1/2)·[K(K+1)/2]·log[N(N−1)]  −  ((K−1)/2)·log N
                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^
                                   connectivity parameters          mixture proportions
```

Higher ICL = better. Two things to note, because they matter below:

- **`X` is the adjacency matrix.** Daudin's SBM models *edge existence* (Bernoulli) over all
  `N(N−1)` possible node pairs. The observation count is **dyads**, not edges.
- **BIC/AIC are explicitly rejected** in this literature: *"Standard criteria such as AIC or
  BIC cannot be used because they rely on the SBM observed data log likelihood which is not
  tractable in practice"* (Côme & Latouche). ICL sidesteps this by conditioning on the hard
  partition `Z`, which is exactly our situation — we have a fitted partition in hand.

---

## 3. What we actually compute

```
DL  =  −loglik  +  (k_dist/2)·log(n_edges)  +  N·H(π)
        ^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^
        weights     distribution parameters     partition
```

Lower DL = better (it is `−ICL` in sign convention). Term by term:

| term | meaning | how computed |
|---|---|---|
| `−loglik` | nats to transmit every edge weight | for each edge, look up its block-pair's fitted distribution and score `log P(W = w)`; summed over 3.7M edges |
| `(k_dist/2)·log(n_edges)` | nats for the fitted rules | `k_dist` = occupied block-pairs × params per family (2 for lognormal/gaussian, 1 for poisson/geometric/exponential) |
| `N·H(π)` | nats to say which block each neuron is in | `N` = 138,584 neurons; `H = −Σ πₖ log πₖ` from block sizes |

**Common support (essential).** Every family is scored as a *truncated discrete pmf on the same
integers*: continuous densities are converted to probabilities on the unit bin around each
integer, `P(W=k) = F(k+½) − F(k−½)`, and all families are conditioned on `W ≥ 5` (weights below
threshold are unobservable). Without this the continuous and discrete numbers are in different
units and the comparison is meaningless. Parameters are fitted by **truncated MLE**, consistent
with the truncated scoring.

---

## 4. The changes we made, and why

**Change 1 — we model edge WEIGHTS, not edge EXISTENCE. (substantive)**
Daudin's ICL scores the adjacency; our question is which *weight* likelihood is best, and the
adjacency is byte-identical across all five models, so it contributes a constant that cancels
from every comparison. Scoring it would add noise, not information.
*Consequence: our observations are the 3.7M edges, not the ~1.9×10¹⁰ dyads.*

**Change 2 — penalty scale `log(n_edges)` instead of `log[N(N−1)]`.**
Mechanical consequence of Change 1: the BIC-style penalty is `(k/2)·log(#observations)`, and our
observations are edges (`log ≈ 15.1`), not dyads (`log ≈ 23.7`).

**Change 3 — parameter count = occupied block-pairs × params-per-family.**
Daudin counts all `K(K+1)/2` block-pairs with one Bernoulli parameter each. We have (a) weight
families with one *or two* parameters, and (b) many block-pairs that are empty and cost nothing.
Counting only *occupied* bundles follows the weighted-SBM formulation of **Aicher, Jacobs &
Clauset (arXiv:1305.5782)**: *"The block structure R defines a partition on the edges into R
disjoint bundles, one for each pair of blocks… each bundle has its own set of distribution
parameters."* (Note: that paper contains **no** BIC/ICL — it selects models by Bayes factors.
We cite it only for the bundle-parameter structure.)

**Change 4 — plug-in partition cost `N·H(π)`; we omit `(K−1)/2·log N`.**
`N·H(π)` is the exact codelength of transmitting the block labels: an outcome of probability `p`
costs `−log p` nats (Shannon), so labels cost `Σᵢ −log π_{zᵢ} = N·H(π)`. Daudin instead uses the
*marginalised* `log p(Z|K)` plus a separate term for the mixture proportions. We use the plug-in
form because we already have a hard partition and are not integrating over `α`. This is a
**simplification, not a derivation** — the omitted term is `O(K log N)`, negligible against
`N·H(π)` (~10⁶ nats) at our sizes, but it is an approximation.

---

## 5. Honest status

- **Structure:** ICL's (`−log p(X|Z) − log p(Z) + parameter penalty`). Published.
- **Instantiation for weights:** **ours**. No paper we read applies ICL to *weight* likelihoods
  on existing edges, nor uses it to rank *different weight families* rather than to select `K`.
- **Mathematically sound?** Yes, given the common discretized support: same data, same units,
  same rule for every model. The obstacle to cross-family comparison was units, and
  discretization removes it.
- **Known approximations:** (i) BIC-type penalties assume regular models with independent
  observations, whereas edges within a bundle share parameters and the partition is latent;
  (ii) the omitted `(K−1)/2·log N`; (iii) each model brings its **own partition**, so a DL gap
  reflects the whole model (weights *and* its resolution), not the weight likelihood in
  isolation — see the resolution-confound note in `RESULTS.md`.

**A superseded earlier version** charged the partition as `(K−1)` *parameters* instead of
`N·H(π)` *nats*. That undercharged a 1,682-block model by roughly three orders of magnitude
(1,681 vs ~10⁶ nats) and systematically flattered high-resolution models. It was replaced before
any result was reported.

---

## 6. References

- Biernacki, Celeux & Govaert (2000) — ICL for mixture models.
- Daudin, Picard & Robin (2008) — ICL adapted to SBM *(formula read via Côme & Latouche)*.
- Côme & Latouche, arXiv:1303.2962 — states Daudin's ICL; source of the formula quoted here.
- Aicher, Jacobs & Clauset, arXiv:1305.5782 — weighted SBM, bundle-parameter structure.
- Wang & Bickel, arXiv:1502.02069 — penalty-order theory for SBM; gives an asymptotic *order*
  condition with a free constant, **not** a plug-in formula, so it is not used here.
- Peixoto — graph-tool's MDL; exact family-specific priors, valid within a family.
