# Held-out prediction of unseen integer weights

## Claim

On this graph, conditional on an observed edge with w ≥ 5 and with degree correction enabled,
the **lognormal** weight model predicts unseen integer weights better than Gaussian, geometric
or Poisson. The margin is modest and consistent across all six tested splits, and lognormal
achieves it while fitting the fewest block-pair weight parameters.

Scope: this concerns *predicting the weight of an edge already known to exist*. It says nothing
about edge existence, Bayesian evidence, raw description length, or which distribution is
"true".

## What we did

**Data.** FlyWire connectome, synaptic counts thresholded at w ≥ 5. Weights are integers.

**Held-out construction (leak-free).** 5% of edges are *removed from the graph* before any
fitting — not masked afterwards. The SBM never sees them, in adjacency or in weight. Splits
are built from a fixed seed, so every family is scored on the **identical** held-out edge set;
only the SBM inference seed varies.

**Fitting.** Degree-corrected weighted SBM on the remaining 95%. Block count B is chosen by the
model itself (graph-tool's MDL-based agglomerative + MCMC), not set by us. Then, per block-pair,
that family's weight parameters are fit on *training* weights only.

**Scoring — common measurement model.** Every family is evaluated as a probability of the
observed *integer*, on the same support:

- Poisson / geometric: their own pmf, `P(K = k)`, divided by `P(K ≥ 5)`
- lognormal / Gaussian: probability of producing the rounded integer,
  `[F(k + 0.5) − F(k − 0.5)] / [1 − F(4.5)]`

The continuous normaliser is **`1 − F(4.5)`, not `1 − F(5)`** — integers are rounded *before*
the w ≥ 5 threshold applies, so 4.5 is the correct continuous cut point. (Verified numerically
against the implementation.) We sum `log P` over held-out weights and report the mean in **nats
per held-out edge**, higher = better. Held-out weights are only ever scored, never fit.

**Fitting and scoring use the same likelihood.** Parameters are obtained by maximising the
*same* truncated, rounded likelihood that is later scored — not by fitting an ordinary
untruncated distribution and applying truncation afterward. Verified numerically for all four
families against an independent brute-force maximiser (agreement to 5+ decimal places in
log-likelihood). Geometric uses a closed form, `p̂ = 1/(mean − t + 1)`; this is not an
approximation — conditioning a geometric at a threshold returns a shifted geometric, so the
sample mean is sufficient and the closed form is the exact truncated MLE (matched the numeric
optimum to 6 decimals).

**Grid.** 4 families × {directed, undirected} × {degree-corrected, not} × {random, stratified}
split construction × 3 splits × 3 inference seeds = 288 cells; **282 scored, 6 failed** (see
"Failed cells").

**Units of variation.** Seeds are repeated *inferences on the same split*, so they are averaged
within a split first, leaving **6 splits**. These six are not independent replicates: they
resample one graph, and any two training sets share ~90% of their edges. All SD/SEM figures
below are therefore **descriptive spread across resampling units**, not standard errors of an
independent sample, and no significance test is implied. (Within-split seed spread, ~0.007 nats,
is in fact larger than between-split spread, ~0.003.)

## Results

Canonical setting (directed, degree-corrected), seeds collapsed, 6 splits:

| family | nats / held-out edge | spread across splits (descriptive) |
|---|---|---|
| **lognormal** | **−2.849** | 0.003 |
| Gaussian | −2.932 | 0.005 |
| geometric | −3.015 | 0.004 |
| Poisson | −6.134 | 0.073 |

Paired on identical held-out edges, unit = split:

| contrast | mean Δ (nats/edge) | splits favouring first |
|---|---|---|
| lognormal − Gaussian | +0.083 | 6 / 6 |
| lognormal − geometric | +0.166 | 6 / 6 |
| Gaussian − geometric | +0.083 | 6 / 6 |

### Where lognormal wins, and where it does not

Winner within each of the four direction × degree-correction combinations:

All families compared on the **same** splits within each combination (the undirected rows use
the 5 splits every family completed, since Gaussian failed one — see "Failed cells"):

| combination | splits | winner | lognormal | Gaussian | geometric | Poisson |
|---|---|---|---|---|---|---|
| directed, DC | 6 | **lognormal** | **−2.849** | −2.932 | −3.015 | −6.134 |
| undirected, DC | 5 | **lognormal** | **−2.983** | −3.078 | −3.143 | −7.311 |
| directed, non-DC | 6 | Gaussian | −3.027 | **−2.889** | −2.924 | −6.016 |
| undirected, non-DC | 5 | Gaussian | −3.139 | **−3.044** | −3.090 | −7.016 |

Lognormal wins **the two degree-corrected combinations**, not all four. Without degree
correction, Gaussian wins. We report this rather than restricting to the canonical setting.

Dropping the incomplete split changes the undirected means by at most 0.003 nats and **changes
no winner** (unpaired vs paired: lognormal −2.984 → −2.983, geometric −3.145 → −3.143 in
und+DC; lognormal −3.142 → −3.139 in und+non-DC). So the Gaussian failure does not by itself
explain the undirected non-DC result — though the underlying concern stands, since the dropped
split is one Gaussian could not be fit on at all.

## Parsimony

Families differ in parameters per bundle (lognormal/Gaussian d = 2; geometric/Poisson d = 1),
so B alone is not comparable. The right count is of **actually fitted** parameters: block-pairs
below the local-fit threshold do not carry their own parameters — they all share the *single*
global fit. So the fitted count is `d × (locally fitted pairs) + d`, **not** `d × (occupied
pairs)`:

| family | blocks | occupied pairs | locally fitted pairs | d | **fitted weight parameters** (mean; min–max) | × lognormal |
|---|---|---|---|---|---|---|
| **lognormal** | 182 | 11,799 | 5,905 | 2 | **11,811** (8,960–23,086) | 1.0 |
| geometric | 533 | 48,968 | 23,694 | 1 | 23,695 (19,929–25,690) | 2.0 |
| Gaussian | 610 | 53,315 | 18,992 | 2 | 37,985 (30,646–42,914) | 3.2 |
| Poisson | 1,531 | 590,963 | 41,421 | 1 | 41,422 (38,789–44,192) | 3.5 |

Every figure above is a **mean over the 18 cells** of that family, so it is not an integer. The
fitted-parameter column averages the per-cell integer counts (`d × local + d`, exact within each
cell); computing it instead from the rounded mean pair count gives 11,812 / 23,695 / 37,986 /
41,422 — the ±1 differences are purely round-then-multiply vs multiply-then-average. Note also
the min–max ranges: lognormal's fitted count varies more than 2× across cells, so these means
should not be read as fixed model sizes.

Lognormal uses the fewest **fitted block-pair weight parameters** even while paying two per
bundle — roughly 2× fewer than geometric and 3× fewer than Gaussian or Poisson.

Two qualifications. First, this is a statement about the **weight component only**; it is
**not** a claim about total SBM description length, which additionally includes the partition,
degree sequence and block-pair edge counts, and which we did not compare here. Second, counting
by *occupied* pairs (23,598 / 48,968 / 106,630 / 590,963) badly overstates the spread — it
implies Poisson carries 25× lognormal's parameters when the fitted ratio is 3.5×. The occupied-
pair count reflects how finely each model partitions the graph, not how many weight parameters
it estimates.

## Sparse and empty block-pairs

A block-pair is fit locally only if it has at least `10 × d` training weights **and** ≥ 3
distinct values. Otherwise — including block-pairs with no training edges at all, and
held-out edges touching a node absent from the training graph — the edge is scored under a
**single global fit** of that family to all training weights. Non-converged local fits are
rejected and also fall back to the global fit. If the *global* fit fails to converge, the cell
raises and is not scored, rather than returning a number.

Note the local-fit threshold is family-dependent: **20** weights for lognormal/Gaussian, **10**
for Poisson/geometric. This `10 × d` rule is a **pre-specified heuristic**, fixed before these
runs and not tuned against the results; it is not derived from any optimality criterion. It is
asymmetric — the 2-parameter families face the higher bar. We did not test sensitivity to it.

Canonical setting (directed, DC), mean over cells:

| family | min weights for local fit | occupied pairs | locally fit | fell back to global | % fallback | rejected fits |
|---|---|---|---|---|---|---|
| lognormal | 20 | 11,799 | 5,905 | 5,895 | 48.8% | 8 |
| geometric | 10 | 48,968 | 23,694 | 25,274 | 51.5% | 0 |
| Gaussian | 20 | 53,315 | 18,992 | 34,324 | 64.2% | 1,336 |
| Poisson | 10 | 590,963 | 41,421 | 549,543 | **93.0%** | 0 |

**This qualifies the Poisson result.** Poisson splits into ~1,500 blocks and 591k occupied
pairs, and **93% of those occupied pairs** are too sparse to fit locally and share the single
global fit. We do **not** know what fraction of Poisson's held-out *predictions* used the
global fit — that is edge-weighted, not pair-weighted, and was not recorded (see below). Its
poor score should be read as "this configuration predicts badly", not as a clean statement
about the Poisson likelihood itself. It also means Poisson does not estimate 591k weight
parameters — it estimates ~41k (see Parsimony), so its over-splitting shows up as extreme
*fragmentation of the partition* rather than as an inflated weight-parameter count.

**Known gap:** these fallback rates are per **block-pair**, not per **held-out edge**. Because
large bundles carry most edges, the fraction of *scored edges* using the global fallback is
certainly far below 93%, but we did not record it. Quantifying it requires a re-run.

## Failed cells

6 of 288 cells produced no score. All six are the same family and split:
`gaussian / undirected / random / fold 2`, across all 3 inference seeds and both degree-correction
settings. Retries with fresh inference seeds on the same fold failed identically, so the failure
is fold-deterministic, not flaky. Failures are non-scores, never zeros or imputed values.

**This is not neutral.** The absent split is absent *because Gaussian could not be fit on it* —
a non-random dropout, in exactly the two combinations where Gaussian wins. All undirected
comparisons above are therefore restricted to the 5 splits every family completed, so no mean
is computed on a different split set from the one it is compared against. Doing so shifts the
undirected means by ≤ 0.003 nats and changes no winner. Even so, those two Gaussian numbers
should be treated as optimistic until the failure cause is resolved, since a split a model
cannot be fit on is plausibly a split it would have scored poorly on.

## Reading

- The advantage is small in size (0.083 nats ≈ 9% more probability per held-out weight) but
  went the same way in **all six tested splits**.
- The parsimony gap points the same way: lognormal predicts best while fitting 2–3.5× fewer
  weight parameters, so it is not buying its predictive advantage with extra capacity.
- Gaps between lognormal, Gaussian and geometric are evenly spaced (0.083, 0.083), so we do
  **not** claim a "two parameters beat one" mechanism.
- That Poisson over-splits *because* it lacks a free dispersion parameter is a **plausible
  hypothesis, not something these data establish** — and see the fallback caveat above.

## What this does not show

Description length on the observed data ranks the families differently — exponential and
geometric come out ahead of lognormal there. The predictive and compression criteria disagree,
and we are not claiming they should agree, nor that one supersedes the other.
