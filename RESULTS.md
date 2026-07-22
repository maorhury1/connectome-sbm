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
