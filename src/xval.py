"""
Leak-free held-out predictive scoring of weight likelihoods.

SCORER (fixed, correct now): every family is scored as a TRUNCATED DISCRETE pmf on the same
integer support -- log P(W=k | W>=threshold) -- so nats/edge are comparable across
lognormal / Gaussian / Poisson / geometric. The lognormal and Gaussian continuous densities
are discretized via  P(W=k)=F(k+0.5)-F(k-0.5), conditioned on W>=threshold. Without this the
cross-family numbers are on different measures and meaningless (reviewer P1 #4).

FOLD (RESOLVED by the Gate A-1 probe, probe_a1.py): graph-tool 3.0 has NO per-edge covariate
mask, and leaving a held-out weight in the property map leaks into the partition whenever
weights matter (probe: ARI 0.34 between fits under true vs corrupted held-out weights, on a
weight-defined graph). So "pure weight prediction" (keep adjacency, mask only the weight) is
not achievable leak-free. The fold therefore REMOVES held-out edges before fitting (leak-free:
probe ARI 1.0) and scores their weights from TRAIN-only block-pair params. This is
"edge-removed held-out weight prediction": it also hides test-edge adjacency, but that confound
is COMMON to every weight model, so the RELATIVE weight-model comparison (E2b) stays fair. It
targets weight prediction -- what distinguishes the likelihoods -- not edge existence.
"""
import numpy as np
from scipy.stats import norm, poisson, geom
from scipy.optimize import minimize, minimize_scalar
import graph as G
import sbm

MIN_LOCAL_PER_PARAM = 10   # weights required per free parameter for a LOCAL bundle fit


def split_edges(m, test_frac, seed):
    """Split the ORIGINAL edge arrays (pre/post/weight index order) into train/test."""
    rng = np.random.default_rng(seed)
    test_idx = np.sort(rng.choice(m, size=int(round(test_frac * m)), replace=False))
    train_mask = np.ones(m, dtype=bool)
    train_mask[test_idx] = False
    return train_mask, test_idx


FOLD_SEED = 12345           # fixed: every model/seed sees the SAME folds (paired comparison)


def make_folds(weight, test_frac, n_folds, method, seed=FOLD_SEED):
    """n_folds DISJOINT held-out sets over the edge arrays.

    method='random'     -- uniform over edges.
    method='stratified' -- equal share drawn from each weight decile, so every fold covers
                           the heavy tail evenly (same estimand, lower fold-to-fold variance).
    Folds are built from a FIXED seed so all weight models are scored on identical held-out
    edges; the SBM inference seed is a separate argument.
    """
    m = len(weight)
    rng = np.random.default_rng(seed)
    if method == "random":
        perm = rng.permutation(m)
        size = int(round(test_frac * m))
        tests = [np.sort(perm[i * size:(i + 1) * size]) for i in range(n_folds)]
    elif method == "stratified":
        edges = np.quantile(weight, np.linspace(0, 1, 11))[1:-1]
        bins = np.searchsorted(edges, weight, side="right")
        tests = [[] for _ in range(n_folds)]
        for b in np.unique(bins):
            idx = np.nonzero(bins == b)[0]
            idx = idx[rng.permutation(len(idx))]
            per = int(round(test_frac * len(idx)))
            for f in range(n_folds):
                tests[f].extend(idx[f * per:(f + 1) * per])
        tests = [np.sort(np.asarray(t, dtype=np.int64)) for t in tests]
    else:
        raise ValueError(f"unknown fold method: {method}")
    folds = []
    for t in tests:
        tm = np.ones(m, dtype=bool)
        tm[t] = False
        folds.append((tm, t))
    return folds


def run_fold_spec(pre, post, weight, directed, model, threshold, train_mask, test_idx,
                  deg_corr=True, seed=0, nested=False):
    """One leak-free fold with a PRE-BUILT split (so folds are shared across models)."""
    g_tr, node_ids, _ = train_graph(pre, post, weight, train_mask, directed)
    assert g_tr.num_edges() == int(train_mask.sum()), "train graph leaked held-out edges"
    state, info = sbm.fit(g_tr, model, nested=nested, deg_corr=deg_corr, seed=seed)
    score, _, diag = predictive_logscore(state, node_ids, pre, post, weight,
                                         test_idx, train_mask, model, threshold)
    return {"logscore_per_edge": score, "n_test": int(len(test_idx)),
            "n_blocks": info["n_blocks"], "n_train_edges": int(train_mask.sum()), **diag}


def train_graph(pre, post, weight, train_mask, directed):
    """Graph from TRAINING edges only (held-out edges physically absent -> leak-free partition)."""
    return G.build_graph(pre[train_mask], post[train_mask], weight[train_mask], directed=directed)


# ---- common-support scorer ---------------------------------------------------
# Weights are observed only for W>=threshold, so parameters must be fit by TRUNCATED MLE
# (maximizing the same truncated likelihood used for scoring); fitting on the truncated sample
# as if untruncated biases each family differently and corrupts the comparison. Tail
# probabilities are computed on the numerically stable side (CDF vs survival) with a
# log-difference, so deep-tail weights get their true tiny score instead of an artificial floor.
def _fit_params(model, w_train, threshold):
    """MLE of the SAME truncated-integer pmf used for scoring.

    FIX 1 (fit/score mismatch): previously lognormal/gaussian were fitted on the continuous
    DENSITY (`norm.logpdf`) but scored on unit-bin PROBABILITIES (`logcdf` differences). Fitting
    and scoring under different rules biases the 2-parameter families specifically. Both now
    maximise `_log_trunc_pmf`, i.e. the exact quantity that is scored.

    Returns (params, ok). `ok=False` means the optimiser did not converge; the caller must then
    fall back to a pooled/global fit and RECORD it, rather than silently accepting moments
    (FIX 2).
    """
    x = np.asarray(w_train, dtype=float)
    if model == "geometric":            # closed form, no optimiser needed
        pv = 1.0 / max(x.mean() - threshold + 1.0, 1.0)
        return {"p": float(min(max(pv, 1e-6), 1 - 1e-6))}, True
    if model == "poisson":              # 1-D bounded MLE of the truncated pmf
        def nll_p(lam):
            lam = max(lam, 1e-9)
            return -(poisson.logpmf(x, lam).sum() - len(x) * poisson.logsf(threshold - 1, lam))
        r = minimize_scalar(nll_p, bounds=(1e-3, max(x.max(), 3 * x.mean()) + 1.0),
                            method="bounded")
        return {"lam": float(max(r.x, 1e-9))}, bool(r.success)

    # lognormal / gaussian: 2-D MLE of the DISCRETIZED truncated pmf
    y = np.log(x) if model == "lognormal" else x
    mu0, s0 = float(y.mean()), float(max(y.std(ddof=0), 1e-3))

    def nll(th):
        par = {"mu": float(th[0]), "sigma": float(np.exp(np.clip(th[1], -30, 30)))}
        if not np.isfinite(par["sigma"]) or par["sigma"] <= 0:
            return 1e12
        v = _log_trunc_pmf(model, x, par, threshold)     # the scored quantity
        s = -float(np.sum(v))
        return s if np.isfinite(s) else 1e12

    try:
        r = minimize(nll, [mu0, np.log(s0)], method="Nelder-Mead",
                     options=dict(maxiter=1200, xatol=1e-5, fatol=1e-7))
        mu, s = float(r.x[0]), float(np.exp(r.x[1]))
        f_end, f_start = nll(r.x), nll([mu0, np.log(s0)])
        ok = (np.isfinite(mu) and np.isfinite(s) and s > 1e-6 and np.isfinite(f_end)
              and f_end <= f_start + 1e-6)          # converged OR at least not worse than init
        if not ok:
            return {"mu": mu0, "sigma": max(s0, 1e-6)}, False
        return {"mu": mu, "sigma": max(s, 1e-6)}, True
    except Exception:
        return {"mu": mu0, "sigma": max(s0, 1e-6)}, False


def _logdiff(a, b):
    """log(exp(a) - exp(b)) for a >= b, numerically stable."""
    return a + np.log(-np.expm1(np.minimum(b - a, -1e-300)))


def _log_trunc_pmf(model, k, p, threshold):
    """log P(W=k) - log P(W>=threshold): common integer support, stable in both tails.

    VECTORISED: k may be a scalar or an array. (It was scalar-only, which silently broke the
    fitting objective -- the exception was swallowed and moment estimates returned instead.)
    """
    k = np.asarray(k, dtype=float)
    scalar = (k.ndim == 0)
    k = np.atleast_1d(k)
    if model in ("lognormal", "gaussian"):
        mu, s = p["mu"], p["sigma"]
        if model == "lognormal":
            lo, hi, mid = np.log(k - 0.5), np.log(k + 0.5), np.log(k)
            c = np.log(threshold - 0.5)
        else:
            lo, hi, mid, c = k - 0.5, k + 0.5, k, threshold - 0.5
        zlo, zhi, zmid = (lo - mu) / s, (hi - mu) / s, (mid - mu) / s
        log_pk = np.empty(k.shape, float)
        low = zmid <= 0                              # lower tail: CDF side is stable
        if np.any(low):
            log_pk[low] = _logdiff(norm.logcdf(zhi[low]), norm.logcdf(zlo[low]))
        if np.any(~low):                             # upper tail: sf(lo)-sf(hi)
            log_pk[~low] = _logdiff(norm.logsf(zlo[~low]), norm.logsf(zhi[~low]))
        log_tail = norm.logsf((c - mu) / s)
    elif model == "poisson":
        log_pk = poisson.logpmf(k, p["lam"]); log_tail = poisson.logsf(threshold - 1, p["lam"])
    elif model == "geometric":
        log_pk = geom.logpmf(k, p["p"]); log_tail = geom.logsf(threshold - 1, p["p"])
    else:
        raise ValueError(model)
    out = log_pk - log_tail
    return float(out[0]) if scalar else out


def predictive_logscore(state, node_ids, pre, post, weight, test_idx, train_mask,
                        model, threshold):
    """Mean truncated-discrete predictive log-score (nats/edge) on held-out edges.
    Block assignment + per-block-pair params come from TRAIN only; held-out weights are
    scored, never fitted."""
    from collections import defaultdict
    blk = sbm.blocks_by_neuron(state, node_ids)
    w = weight.astype(float)

    pair_w = defaultdict(list)
    for i in np.nonzero(train_mask)[0]:
        a, b = blk.get(int(pre[i])), blk.get(int(post[i]))
        if a is not None and b is not None:
            pair_w[(a, b)].append(w[i])

    # FIX 2 (under-determined bundles + silent optimiser failures).
    # Previously ANY bundle with >=5 weights got its own 2-parameter fit and the optimiser's
    # output was accepted without checking convergence, so a handful of catastrophic fits
    # dragged the mean down -- and only for the 2-parameter families (lognormal, gaussian),
    # which is precisely the direction of the reported result.
    # Now: a bundle is fitted locally only with enough data to identify the parameters, and a
    # non-converged fit is REJECTED and falls back to the global fit, with counts recorded.
    d_params = 2 if model in ("lognormal", "gaussian") else 1
    min_local = MIN_LOCAL_PER_PARAM * d_params           # 10 per parameter
    global_p, global_ok = _fit_params(model, w[train_mask], threshold)

    pair_p, n_local, n_rejected, n_small = {}, 0, 0, 0
    for k, v in pair_w.items():
        if len(v) < min_local or len(np.unique(v)) < 3:
            n_small += 1
            continue
        par, ok = _fit_params(model, v, threshold)
        if not ok:
            n_rejected += 1
            continue
        pair_p[k] = par
        n_local += 1

    scores = []
    for i in test_idx:
        a, b = blk.get(int(pre[i])), blk.get(int(post[i]))
        scores.append(_log_trunc_pmf(model, int(weight[i]), pair_p.get((a, b), global_p), threshold))
    scores = np.array(scores)
    diag = dict(n_local_bundles=n_local, n_rejected_fits=n_rejected,
                n_small_bundles=n_small, global_fit_ok=bool(global_ok),
                min_local_required=int(min_local))
    return float(scores.mean()), scores, diag


def run_fold(pre, post, weight, directed, model, threshold, test_frac=0.1, seed=0,
             nested=False):
    """One leak-free edge-removed held-out weight-prediction fold, end-to-end (the fold
    resolved by the A-1 probe; valid for relative weight-model comparison in E2b)."""
    train_mask, test_idx = split_edges(len(weight), test_frac, seed)
    g_tr, node_ids, _ = train_graph(pre, post, weight, train_mask, directed)
    assert g_tr.num_edges() == int(train_mask.sum()), "train graph leaked held-out edges"
    state, info = sbm.fit(g_tr, model, nested=nested, seed=seed)
    score, _, diag = predictive_logscore(state, node_ids, pre, post, weight,
                                         test_idx, train_mask, model, threshold)
    return {"model": model, "seed": seed, "test_frac": test_frac, "n_test": len(test_idx),
            "logscore_per_edge": score, "n_blocks": info["n_blocks"], **diag}
