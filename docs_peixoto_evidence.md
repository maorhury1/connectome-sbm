# Model selection with Peixoto's evidence framework

*Written 2026-07-28. Supersedes the hand-rolled criterion in `docs_description_length.md` for
the purpose of comparing weight models. That document is kept for the record, and Sec. 6 below
states exactly why it was replaced.*

---

## 1. The problem

We fit the same connectome with five weight likelihoods — lognormal, gaussian, exponential
(continuous) and poisson, geometric (discrete) — and want to know, **without using biological
labels**, which describes the data best.

The obstacle: graph-tool's `state.entropy()` is a valid description length, but two things stop
us reading it straight off a table.

1. **Transformed weights.** For "lognormal" we hand graph-tool `y = ln w`, so its entropy is a
   density in *y*-space, not *w*-space. Comparing that to a model scored on raw `w` compares
   two different variables.
2. **Different measures.** Continuous families produce **densities** (per unit of *x*, can
   exceed 1); discrete families produce **probabilities**. Their logs are not on the same scale.

## 2. The published solution

**Peixoto, "Nonparametric weighted stochastic block models", arXiv:1708.01432.** He gives a
Bayesian formulation of weighted SBMs with edge covariates, and — the part we need — an
*unsupervised model-selection criterion* for choosing among weight models, based on **posterior
odds of the joint evidence** rather than any information criterion.

For two weight transformations `y = f(x)` and `z = g(x)` with models `M_y`, `M_z`, his Eq. 109
gives (with equal model priors):

```
          P(A, y(x) | A, {b}_1, M_y) · P({b}_1) · Π_ij f'(x_ij)^A_ij
Λ  =  ─────────────────────────────────────────────────────────────────
          P(A, z(x) | A, {b}_2, M_z) · P({b}_2) · Π_ij g'(x_ij)^A_ij
```

Three things make this the right tool:

- **The evidence is already integrated over parameters**, so model complexity is charged
  automatically — *"the Bayesian criterion above takes into account the complexity of the model,
  and will point towards a more complicated one only if the statistical evidence in the data
  supports it."* No BIC/ICL-style penalty has to be invented.
- **The derivative terms are exactly the Jacobian** of the weight transformation. For untransformed
  or discrete weights, *"one simply omits the derivative terms."*
- **He validates it on our exact comparison**: exponential on raw `x` versus normal on `ln x`
  (i.e. lognormal), on human-brain connectivity, obtaining `ln Λ ≈ 4458` in favour of lognormal
  (his Table I / Fig. 11).

Crucially, `−log[P(A, x | {b}, M) · P({b})]` **is** what graph-tool's `state.entropy()` returns:
adjacency, partition prior, and weight evidence with the model's own priors, all included. So a
valid comparison is a difference of *corrected entropies* — nothing needs re-deriving, and the
adjacency term is genuinely part of each model's score rather than assumed to cancel.

## 3. What we compute

```
S_peixoto  =  S_graphtool  +  Jacobian                        ← Eq. 109, the headline
S_common   =  S_peixoto    +  bin correction + truncation     ← common measurement model
```

**Jacobian.** With `y = ln x`, `|dy/dx| = 1/x`, so converting a *y*-space score to *x*-space adds

```
Σ_edges log(w)      (= 8,560,387 nats directed; 8,023,999 undirected)
```

Applied to **lognormal only**; zero for gaussian, exponential, poisson, geometric.

**Common measurement model.** Synapse counts are integers with a hard floor at `w ≥ 5`, so every
family must be a probability model over *those* integers:

- *Bin correction* — a density becomes a probability on the unit bin around each integer,
  `P(W=w) = F(w+½) − F(w−½)`. For unit spacing this is `≈ p(w)·1`, i.e. ~0 to first order; we
  compute the **exact** difference per bundle and report it rather than assuming it away.
- *Truncation* — no family is conditioned on `w ≥ 5`, so each wastes probability on impossible
  values. The correction `Σ_edges log P_bundle(W ≥ 5)` is computed per block-pair and reported.

Both are reported as **separate columns**, so the headline (Peixoto's own criterion) is visible
alongside how much the measurement-model terms move it. Measured sizes: bin ≈ +0.02M nats
(lognormal), +0.31M (gaussian), 0 (discrete); truncation ≈ −0.26M to −0.69M — small relative to
the between-model gaps, but not assumed to be.

## 4. Scope and honest limits

- **Each model brings its own partition.** That is intended here — Eq. 109 compares *whole
  models*, `P(A, x, {b})`, so a gap reflects weights *and* the structure they induce. It is
  therefore not an isolated statement about the weight likelihood alone (see the
  resolution-confound note in `RESULTS.md`).
- **Directed vs undirected are different datasets** (undirected collapses reciprocal pairs:
  3,422,370 vs 3,732,460 edges). Never compare across that split.
- **The bin/truncation corrections are ours**, not Peixoto's; he does not treat a hard
  observation floor. They are reported separately for exactly that reason.
- **graph-tool 2.98** is used throughout; entropies come from the saved `summary.json` of the
  200-fit sweep, so no refitting is involved.

## 5. Reproduce

```bash
cd src
python peixoto_evidence.py --refresh      # scores all 100 t5 fits, caches to WORK_DIR
python peixoto_evidence.py                # instant re-read from cache
```

## 6. Why this replaces the earlier criterion

`docs_description_length.md` described a hand-built criterion (`−loglik + (k/2)log n + N·H(π)`).
Reviewer feedback identified four defects, all of which are correct and all of which vanish here:

| defect in the hand-built criterion | status under Peixoto |
|---|---|
| Adjacency term dropped as "cancelling" — but `P(A\|Z)` changes with the partition, and each model has its own | included: entropy scores `P(A, x, {b})` |
| Penalty used `log(n_edges)` for every bundle; each bundle only sees `m_rs` weights, so it should involve `log m_rs` | no ad-hoc penalty at all — parameters are integrated out |
| `N·H(π)` is a plug-in likelihood, not a complete partition code | uses graph-tool's actual partition prior `P({b})` |
| For directed SBMs the parameter count is `K²`, not `K(K+1)/2` | no parameter counting needed |

The fifth and decisive point: **Peixoto already solved this**, including transformations and the
Jacobian, and demonstrated it on lognormal-vs-exponential brain data. Building a substitute was
unnecessary.

## 7. References

- Peixoto, arXiv:1708.01432 — *Nonparametric weighted stochastic block models* (Eq. 109; Sec. IV G;
  Table I). The framework used here.
- Aicher, Jacobs & Clauset, arXiv:1305.5782 — weighted SBM, bundle structure (no BIC/ICL; uses
  Bayes factors).
- Daudin, Picard & Robin (2008) via Côme & Latouche, arXiv:1303.2962 — ICL for SBM; models edge
  *existence* over dyads, hence not directly applicable to weight-family selection.
- Wang & Bickel, arXiv:1502.02069 — penalty-order theory; gives an asymptotic order with a free
  constant, not a plug-in formula.
