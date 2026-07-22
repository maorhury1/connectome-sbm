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


def split_edges(m, test_frac, seed):
    """Split the ORIGINAL edge arrays (pre/post/weight index order) into train/test."""
    rng = np.random.default_rng(seed)
    test_idx = np.sort(rng.choice(m, size=int(round(test_frac * m)), replace=False))
    train_mask = np.ones(m, dtype=bool)
    train_mask[test_idx] = False
    return train_mask, test_idx


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
    x = np.asarray(w_train, dtype=float)
    if model == "geometric":            # k>=1 geometric truncated at k>=t -> closed form
        p = 1.0 / max(x.mean() - threshold + 1.0, 1.0)
        return {"p": float(min(max(p, 1e-6), 1 - 1e-6))}
    if model == "poisson":              # 1-D bounded truncated MLE
        def nll(lam):
            lam = max(lam, 1e-9)
            return -(poisson.logpmf(x, lam).sum() - len(x) * poisson.logsf(threshold - 1, lam))
        r = minimize_scalar(nll, bounds=(1e-3, max(x.max(), 3 * x.mean()) + 1.0), method="bounded")
        return {"lam": float(max(r.x, 1e-9))}
    # lognormal / gaussian: 2-D truncated-normal MLE, init at moments, log-sigma keeps sigma>0
    y = np.log(x) if model == "lognormal" else x
    c = np.log(threshold - 0.5) if model == "lognormal" else (threshold - 0.5)
    mu0, s0 = float(y.mean()), float(max(y.std(ddof=0), 1e-3))
    def nll(th):
        mu, s = th[0], np.exp(th[1])
        return -(np.sum(norm.logpdf(y, mu, s)) - len(y) * norm.logsf((c - mu) / s))
    try:
        r = minimize(nll, [mu0, np.log(s0)], method="Nelder-Mead",
                     options=dict(maxiter=300, xatol=1e-3, fatol=1e-2))
        mu, s = float(r.x[0]), float(np.exp(r.x[1]))
        if not (np.isfinite(mu) and np.isfinite(s) and s > 1e-6):
            mu, s = mu0, s0                          # fall back to moments if the optimizer misbehaves
    except Exception:
        mu, s = mu0, s0
    return {"mu": mu, "sigma": max(s, 1e-6)}


def _logdiff(a, b):
    """log(exp(a) - exp(b)) for a >= b, numerically stable."""
    return a + np.log(-np.expm1(b - a))


def _log_trunc_pmf(model, k, p, threshold):
    """log P(W=k) - log P(W>=threshold): common integer support, stable in both tails."""
    if model in ("lognormal", "gaussian"):
        mu, s = p["mu"], p["sigma"]
        if model == "lognormal":
            lo, hi, mid, c = np.log(k - 0.5), np.log(k + 0.5), np.log(k), np.log(threshold - 0.5)
        else:
            lo, hi, mid, c = k - 0.5, k + 0.5, float(k), threshold - 0.5
        zlo, zhi, zmid = (lo - mu) / s, (hi - mu) / s, (mid - mu) / s
        if zmid <= 0:                                # lower tail: CDF side is stable
            log_pk = _logdiff(norm.logcdf(zhi), norm.logcdf(zlo))
        else:                                        # upper tail: F(hi)-F(lo) = sf(lo)-sf(hi)
            log_pk = _logdiff(norm.logsf(zlo), norm.logsf(zhi))
        log_tail = norm.logsf((c - mu) / s)
    elif model == "poisson":
        log_pk = poisson.logpmf(k, p["lam"]); log_tail = poisson.logsf(threshold - 1, p["lam"])
    elif model == "geometric":
        log_pk = geom.logpmf(k, p["p"]); log_tail = geom.logsf(threshold - 1, p["p"])
    else:
        raise ValueError(model)
    return float(log_pk - log_tail)


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
    global_p = _fit_params(model, w[train_mask], threshold)
    pair_p = {k: _fit_params(model, v, threshold) for k, v in pair_w.items() if len(v) >= 5}

    scores = []
    for i in test_idx:
        a, b = blk.get(int(pre[i])), blk.get(int(post[i]))
        scores.append(_log_trunc_pmf(model, int(weight[i]), pair_p.get((a, b), global_p), threshold))
    scores = np.array(scores)
    return float(scores.mean()), scores


def run_fold(pre, post, weight, directed, model, threshold, test_frac=0.1, seed=0,
             nested=False):
    """One leak-free edge-removed held-out weight-prediction fold, end-to-end (the fold
    resolved by the A-1 probe; valid for relative weight-model comparison in E2b)."""
    train_mask, test_idx = split_edges(len(weight), test_frac, seed)
    g_tr, node_ids, _ = train_graph(pre, post, weight, train_mask, directed)
    assert g_tr.num_edges() == int(train_mask.sum()), "train graph leaked held-out edges"
    state, info = sbm.fit(g_tr, model, nested=nested, seed=seed)
    score, _ = predictive_logscore(state, node_ids, pre, post, weight,
                                   test_idx, train_mask, model, threshold)
    return {"model": model, "seed": seed, "test_frac": test_frac, "n_test": len(test_idx),
            "logscore_per_edge": score, "n_blocks": info["n_blocks"]}
